// Speed Demon 2026, Track C (Join).
//
// regions.csv becomes a byte array, region id -> country index, where country
// indices follow the codes' byte order (0 is _UNMATCHED). records.csv is
// memory-mapped and cut into one range per thread at record starts: in this
// data a record starts exactly where an LF is followed by a digit or a BOM,
// because names and notes never contain digits.
//
// Each thread classifies its range 64 bytes at a time, simdjson style: a
// carry-less multiply turns the quote bits into an "inside quotes" mask, and
// the commas and LFs outside it are the field and record boundaries. Every
// record has exactly four such commas and one LF. A quoted field with k quote
// bytes (two delimiters and the doubled ones) unescapes to raw - 1 - k / 2
// bytes, and quote counts come from running popcounts.
//
// Distinct names: names are interned in one lock-free table shared by all
// threads (the raw field identifies the value, except that "" and the empty
// field are both the empty name), and each (country, name id) pair sets one
// bit of a shared bitset. The answer is a popcount per country.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;

constexpr u32 COUNTRIES = 256;      // index 0 is _UNMATCHED
constexpr u32 ROW_BITS = 104 << 10; // name ids per bitset row (pool: 100,000)

template <class T> T* zeroed(size_t n) {
    void* p = mmap(nullptr, n * sizeof(T), PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) _exit(1);
    return static_cast<T*>(p);
}

// Maps a file read-only, with a zero page after it so 64-byte loads are safe.
const char* map_file(const char* path, size_t& size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) _exit(1);
    size = lseek(fd, 0, SEEK_END);
    auto* p = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(p, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED)
        _exit(1);
    close(fd);
    return p;
}

inline u64 load8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

inline u64 eq64(__m256i lo, __m256i hi, char c) {
    __m256i k = _mm256_set1_epi8(c);
    return u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, k))) |
           u64(u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, k)))) << 32;
}

// The value of the 1 to 8 digits in [s, e), branch-free (SWAR).
inline u64 digits(const char* s, const char* e) {
    u64 v = load8(e - 8) & 0x0F0F0F0F0F0F0F0F & ~u64(0) << 8 * (8 - (e - s));
    v = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FF;
    v = (v * 100 + (v >> 16)) & 0x0000FFFF0000FFFF;
    return (v * 10000 + (v >> 32)) & 0xFFFFFFFF;
}

inline u64 number(const char* s, const char* e) { // any length
    return e - s <= 8 ? digits(s, e) : number(s, e - 8) * 100000000 + digits(e - 8, e);
}

// Raw name fields -> dense ids, shared by all threads without locks. A slot is
// one 32-byte line: the name's first 24 bytes and a meta word (ready bit, hash
// tag, id, length). An insert claims the meta word, writes the bytes, then
// sets the ready bit; a reader that finds a claimed slot waits for that bit.
// Longer names keep their remaining bytes in an arena.
class NameTable {
  public:
    static constexpr u32 SLOTS = 1 << 17, INLINE = 24;
    static constexpr u64 READY = u64(1) << 63;

    // Stage 1: the name's hash; also starts loading its first slot.
    u64 hash(const char* s, u32 len) const {
        u64 a = load8(s), b = load8(s + len - 8), c = load8(s + len / 2 - 4);
        if (len < 8) a &= (u64(1) << 8 * len) - 1, b = c = 0;
        u64 h = _mm_crc32_u64(_mm_crc32_u64(_mm_crc32_u64(len, a), b), c) * 0x9E3779B97F4A7C15;
        _mm_prefetch(reinterpret_cast<const char*>(&slots_[h >> 17 & (SLOTS - 1)]), _MM_HINT_T0);
        return h;
    }

    // Stage 2: the name's id, inserting it if it is new.
    u32 id(const char* s, u32 len, u64 h) {
        const u64 key = (h >> 33 << 32) | len; // tag in bits 32..62, length in 0..7
        __m256i want = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
        const u32 need = _bzhi_u32(~0u, std::min(len, INLINE));
        for (u32 i = h >> 17;; ++i) {
            Slot& slot = slots_[i & (SLOTS - 1)];
            u64 m = slot.meta.load(std::memory_order_acquire);
            if (m == 0 && slot.meta.compare_exchange_strong(m, key, std::memory_order_acquire))
                return insert(slot, key, s, len);
            if ((m & ~READY & ~(u64(0xFFFFFF) << 8)) != key) continue;
            while (!(m & READY)) m = slot.meta.load(std::memory_order_acquire);
            __m256i have = _mm256_load_si256(reinterpret_cast<const __m256i*>(&slot));
            u32 eq = _mm256_movemask_epi8(_mm256_cmpeq_epi8(want, have));
            u32 id = m >> 8 & 0xFFFFFF;
            if ((~eq & need) == 0 &&
                (len <= INLINE || memcmp(arena_ + tail_[id], s + INLINE, len - INLINE) == 0))
                return id;
        }
    }

  private:
    struct alignas(32) Slot {
        char head[24];
        std::atomic<u64> meta; // 0 empty; else READY? | tag << 32 | id << 8 | len
    };

    u32 insert(Slot& slot, u64 key, const char* s, u32 len) {
        u32 id = next_id_.fetch_add(1, std::memory_order_relaxed);
        if (id >= MAX_IDS || len > 255) _exit(1);
        memcpy(slot.head, s, std::min(len, INLINE));
        if (len > INLINE) {
            u64 off = arena_used_.fetch_add(len - INLINE, std::memory_order_relaxed);
            if (off + len > ARENA) _exit(1);
            memcpy(arena_ + off, s + INLINE, len - INLINE);
            tail_[id] = off;
        }
        slot.meta.store(READY | key | u64(id) << 8, std::memory_order_release);
        return id;
    }

    static constexpr u64 ARENA = 1 << 20, MAX_IDS = 1 << 20; // virtual sizes
    Slot* slots_ = zeroed<Slot>(SLOTS);
    char* arena_ = zeroed<char>(ARENA);
    u32* tail_ = zeroed<u32>(MAX_IDS);
    std::atomic<u64> arena_used_{0};
    std::atomic<u32> next_id_{1}; // id 0 is the empty name
};

struct Totals {
    u64 count = 0, id_sum = 0, name_bytes = 0, note_bytes = 0, distinct = 0;
    i64 amount_sum = 0, amount_min = INT64_MAX, amount_max = INT64_MIN;
};

struct Shared {
    const u8* country_of;
    size_t nregions;
    NameTable names;
    std::atomic<u64>* seen = zeroed<std::atomic<u64>>(COUNTRIES * ROW_BITS / 64);
    std::vector<std::vector<u64>> overflow; // (country, id) beyond ROW_BITS
};

// Aggregates the records in [p, end), which starts at a record start.
void parse_records(const char* p, const char* end, Shared& sh, Totals* tot,
                   std::vector<u64>& overflow) {
    u64 inside = 0;          // 1 when the previous block ended inside quotes
    u64 quotes = 0;          // quotes before the current block
    u32 field = 0;           // structural characters seen in this record
    const char* at[5];       // their positions: 4 commas, then the LF
    u64 q_at[3];             // quote counts at the first three commas
    const char* rec = p;     // the current record's first byte
    // Name lookups and bitset updates trail the parse by LAG records each, so
    // the cache lines they need have been prefetched by the time they run.
    constexpr u32 LAG = 16;
    struct Name { const char* s; u32 len, country; u64 hash; } names[LAG];
    u64 marks[LAG]; // bit index in the bitset, or ~0 for none
    u32 n = 0;
    auto mark = [&](u64 bit) {
        if (bit == ~u64(0)) return;
        std::atomic<u64>& w = sh.seen[bit / 64];
        if (!(w.load(std::memory_order_relaxed) >> bit % 64 & 1))
            w.fetch_or(u64(1) << bit % 64, std::memory_order_relaxed);
    };
    auto lookup = [&](const Name& x) -> u64 { // the bit to set, prefetched
        u32 nid = x.len ? sh.names.id(x.s, x.len, x.hash) : 0;
        if (nid >= ROW_BITS) {
            overflow.push_back(u64(x.country) << 32 | nid);
            return ~u64(0);
        }
        u64 bit = u64(x.country) * ROW_BITS + nid;
        _mm_prefetch(reinterpret_cast<const char*>(&sh.seen[bit / 64]), _MM_HINT_T0);
        return bit;
    };
    for (const char* blk = p; blk < end; blk += 64) {
        __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
        __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 32));
        u64 q = eq64(lo, hi, '"');
        u64 in = _mm_cvtsi128_si64(_mm_clmulepi64_si128(_mm_set_epi64x(0, q), _mm_set1_epi8(-1), 0)) ^ inside;
        inside = u64(i64(in) >> 63);
        u64 structural = (eq64(lo, hi, ',') | eq64(lo, hi, '\n')) & ~in;
        if (blk + 64 > end) structural &= _bzhi_u64(~u64(0), end - blk);
        for (; structural; structural = _blsr_u64(structural)) {
            u32 off = __builtin_ctzll(structural);
            const char* s = blk + off;
            if (field < 3) q_at[field] = quotes + __builtin_popcountll(q & _bzhi_u64(~u64(0), off));
            at[field++] = s;
            if (field < 5) continue;
            // A whole record: id , name , note , region , amount (CR) LF.
            field = 0;
            const char* id_start = rec + (u8(rec[0]) == 0xEF ? 3 : 0); // BOM
            u64 id = digits(id_start, at[0]); // below 4 * 10^7, spec 2.2
            u32 name_raw = at[1] - at[0] - 1, note_raw = at[2] - at[1] - 1;
            u32 nq = q_at[1] - q_at[0], tq = q_at[2] - q_at[1];
            u32 name_len = nq ? name_raw - 1 - nq / 2 : name_raw;
            u32 note_len = tq ? note_raw - 1 - tq / 2 : note_raw;
            u64 region = digits(at[2] + 1, at[3]); // at most 9,999,999
            const char* a = at[3] + 1;
            const char* ae = at[4] - (at[4][-1] == '\r');
            bool neg = *a == '-';
            i64 cents = i64(number(a + neg, ae - 3) * 100 + digits(ae - 2, ae));
            if (neg) cents = -cents;
            u32 c = region < sh.nregions ? sh.country_of[region] : 0;
            Totals& t = tot[c];
            ++t.count;
            t.id_sum += id;
            t.amount_sum += cents;
            t.amount_min = std::min(t.amount_min, cents);
            t.amount_max = std::max(t.amount_max, cents);
            t.name_bytes += name_len;
            t.note_bytes += note_len;
            Name& slot = names[n % LAG];
            u64& pending = marks[n % LAG];
            if (n >= LAG) mark(pending);
            pending = n >= LAG ? lookup(slot) : ~u64(0);
            u32 raw = name_len ? name_raw : 0; // "" is the empty name too
            slot = {at[0] + 1, raw, c, raw ? sh.names.hash(at[0] + 1, raw) : 0};
            ++n;
            rec = s + 1;
        }
        quotes += __builtin_popcountll(q);
    }
    // Drain the pipeline: lookups still queued, then their bits.
    for (u32 i = n >= LAG ? n - LAG : 0; i < n; ++i) {
        mark(marks[i % LAG]);
        marks[i % LAG] = lookup(names[i % LAG]);
    }
    for (u32 i = 0; i < std::min(n, LAG); ++i) mark(marks[i]);
}

// The next record start at or after p: just past an LF, at a digit or a BOM.
size_t record_start(const char* d, size_t p, size_t size) {
    for (; p < size; ++p)
        if (d[p - 1] == '\n' && (u8(d[p]) - '0' < 10 || u8(d[p]) == 0xEF)) return p;
    return size;
}

inline u32 code_of(const char* s) { return (s[0] - 'A') << 10 | (s[1] - 'A') << 5 | (s[2] - 'A'); }

void put_money(char*& o, i64 v) {
    if (v < 0) *o++ = '-', v = -v;
    o += sprintf(o, "%lld.%02lld", (long long)(v / 100), (long long)(v % 100));
}

int main() {
    const int nthreads = std::clamp<int>(std::thread::hardware_concurrency(), 1, 12);
    size_t rsize, size;
    const char* regions = map_file("/data/regions.csv", rsize);
    const char* records = map_file("/data/records.csv", size);

    // Region ids are ascending, so the last line holds the largest.
    const char* last = static_cast<const char*>(memrchr(regions, '\n', rsize - 1)) + 1;
    Shared sh;
    sh.nregions = number(last, strchr(last, ',')) + 1;
    u8* country_of = zeroed<u8>(sh.nregions);
    sh.country_of = country_of;
    sh.overflow.resize(nthreads);

    std::vector<size_t> rcut(nthreads + 1, rsize), cut(nthreads + 1, size);
    rcut[0] = strchr(regions, '\n') - regions + 1;
    cut[0] = record_start(records, 1, size); // skip the header
    for (int t = 1; t < nthreads; ++t) {
        rcut[t] = std::max(rcut[t - 1], rsize * t / nthreads);
        while (rcut[t] < rsize && regions[rcut[t] - 1] != '\n') ++rcut[t];
        cut[t] = record_start(records, std::max(cut[t - 1], size * t / nthreads), size);
    }

    // Country codes present -> index in byte order, numbered between phases.
    std::vector<std::atomic<u8>> present(1 << 15);
    std::vector<u8> index_of(1 << 15);
    std::vector<u32> codes;
    std::barrier sync(nthreads, [&]() noexcept {
        if (!codes.empty()) return;
        for (u32 k = 0; k < present.size(); ++k)
            if (present[k]) codes.push_back(k), index_of[k] = codes.size();
        if (codes.size() >= COUNTRIES) _exit(1);
    });
    std::vector<std::vector<Totals>> totals(nthreads, std::vector<Totals>(COUNTRIES));

    auto work = [&](int t) {
        const char* from = regions + rcut[t];
        const char* to = regions + rcut[t + 1];
        for (const char* p = from; p < to; p = strchr(p, '\n') + 1)
            present[code_of(strchr(p, ',') + 1)].store(1, std::memory_order_relaxed);
        sync.arrive_and_wait();
        for (const char* p = from; p < to; p = strchr(p, '\n') + 1) {
            const char* comma = strchr(p, ',');
            country_of[number(p, comma)] = index_of[code_of(comma + 1)];
        }
        sync.arrive_and_wait();
        parse_records(records + cut[t], records + cut[t + 1], sh, totals[t].data(), sh.overflow[t]);
        sync.arrive_and_wait();
        for (u32 c = t; c < COUNTRIES; c += nthreads) {
            u64 n = 0;
            for (u32 w = c * ROW_BITS / 64; w < (c + 1) * ROW_BITS / 64; ++w)
                n += __builtin_popcountll(sh.seen[w].load(std::memory_order_relaxed));
            totals[t][c].distinct = n;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();

    // Pairs whose name id did not fit a bitset row (none in practice).
    std::vector<u64> extra;
    for (auto& v : sh.overflow) extra.insert(extra.end(), v.begin(), v.end());
    std::sort(extra.begin(), extra.end());
    extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
    for (u64 x : extra) ++totals[0][x >> 32].distinct;

    char out[256 * 160], *o = out;
    for (u32 i = 1; i <= codes.size() + 1; ++i) {
        u32 c = i % (codes.size() + 1); // countries first, _UNMATCHED last
        Totals sum;
        for (auto& per_thread : totals) {
            const Totals& t = per_thread[c];
            sum.count += t.count, sum.id_sum += t.id_sum, sum.distinct += t.distinct;
            sum.name_bytes += t.name_bytes, sum.note_bytes += t.note_bytes;
            sum.amount_sum += t.amount_sum;
            sum.amount_min = std::min(sum.amount_min, t.amount_min);
            sum.amount_max = std::max(sum.amount_max, t.amount_max);
        }
        if (sum.count == 0) continue;
        if (c == 0) {
            o += sprintf(o, "_UNMATCHED");
        } else {
            u32 k = codes[c - 1];
            o += sprintf(o, "%c%c%c", 'A' + (k >> 10), 'A' + (k >> 5 & 31), 'A' + (k & 31));
        }
        o += sprintf(o, " %llu %llu", (unsigned long long)sum.count, (unsigned long long)sum.id_sum);
        for (i64 v : {sum.amount_sum, sum.amount_min, sum.amount_max}) *o++ = ' ', put_money(o, v);
        o += sprintf(o, " %llu %llu %llu\n", (unsigned long long)sum.distinct,
                     (unsigned long long)sum.name_bytes, (unsigned long long)sum.note_bytes);
    }
    int fd = open("/out/aggregates.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || write(fd, out, o - out) != o - out) return 1;
    close(fd);
    _exit(0);
}
