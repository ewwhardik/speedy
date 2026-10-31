#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <malloc.h>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;

template <class T>
T* zeroed(size_t n) {
    void* p = mmap(nullptr, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) _exit(1);
    return static_cast<T*>(p);
}

const char* map_file(const char* path, size_t& size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) _exit(1);
    size = lseek(fd, 0, SEEK_END);
    auto* p = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(p, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) _exit(1);
    close(fd);
    return p;
}

inline u64 load8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

inline u32 match32(const char* p, char c) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(c)));
}

inline u64 match64(const char* p, char c) { return match32(p, c) | u64(match32(p + 32, c)) << 32; }

inline u64 digits(const char* s, const char* e) {
    u64 v = load8(e - 8) & 0x0F0F0F0F0F0F0F0F & ~u64(0) << 8 * (8 - (e - s));
    v = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FF;
    v = (v * 100 + (v >> 16)) & 0x0000FFFF0000FFFF;
    return (v * 10000 + (v >> 32)) & 0xFFFFFFFF;
}

class Structure {
public:
    struct Separator {
        const char* at;
        u64 quotes;
    };

    Structure(const char* p, const char* file_end) : block_(p - 64), end_(file_end) {}

    Separator next() {
        while (bits_ == 0) {
            block_ += 64;
            if (block_ >= end_) return {end_, before_ + __builtin_popcountll(q_)};
            before_ += __builtin_popcountll(q_);
            __m256i v0, v1;
            if (__builtin_expect(block_ + 64 <= end_, 1)) {
                v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_));
                v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_ + 32));
            } else {
                alignas(32) char tail[64] = {};
                size_t rem = end_ > block_ ? size_t(end_ - block_) : 0;
                memcpy(tail, block_, rem);
                v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(tail));
                v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(tail + 32));
            }
            u64 q0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('"')));
            u64 q1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8('"')));
            q_ = q0 | (q1 << 32);
            __m128i x = _mm_clmulepi64_si128(_mm_cvtsi64_si128(q_), _mm_set1_epi8(-1), 0);
            u64 inside = u64(_mm_cvtsi128_si64(x)) ^ carry_;
            carry_ = u64(i64(inside) >> 63);
            u64 s0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8(','))) |
                     _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\n')));
            u64 s1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8(','))) |
                     _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\n')));
            bits_ = (s0 | (s1 << 32)) & ~inside;
        }
        u32 i = __builtin_ctzll(bits_);
        bits_ &= bits_ - 1;
        return {block_ + i, before_ + __builtin_popcountll(q_ & ((u64(1) << i) - 1))};
    }

private:
    const char* block_;
    const char* end_;
    u64 bits_ = 0, carry_ = 0, q_ = 0, before_ = 0;
};

class NameTable {
public:
    static constexpr u32 SLOTS = 1 << 17, IDS = 1 << 17, HEAD = 24;

    struct Probe {
        u32 slot;
        u16 tag;
    };

    Probe probe(const char* s, u32 len) const {
        u64 h = len;
        for (u32 i = 0; i < len; i += 8) {
            u64 w = load8(s + i);
            if (len - i < 8) w &= (u64(1) << 8 * (len - i)) - 1;
            h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        }
        Probe p{u32(h >> 20) & (SLOTS - 1), u16(1 + (h >> 48) % (BUSY - 1))};
        for (u16 t; (t = tags_[p.slot].load(std::memory_order_relaxed)) && t != p.tag && t != BUSY;)
            p.slot = (p.slot + 1) & (SLOTS - 1);
        _mm_prefetch(reinterpret_cast<const char*>(&slots_[p.slot]), _MM_HINT_T0);
        return p;
    }

    u32 id(const char* s, u32 len, Probe p) {
        for (u32 i = p.slot;; i = (i + 1) & (SLOTS - 1)) {
            u16 t = tags_[i].load(std::memory_order_acquire);
            if (t == 0 && tags_[i].compare_exchange_strong(t, BUSY, std::memory_order_acquire)) {
                Slot& slot = slots_[i];
                slot.id = next_id_++;
                slot.len = len;
                if (slot.id >= IDS) _exit(1);
                memcpy(slot.head, s, std::min(len, HEAD));
                if (len > HEAD) {
                    overflow_[i] = used_.fetch_add(len - HEAD);
                    memcpy(arena_ + overflow_[i], s + HEAD, len - HEAD);
                }
                tags_[i].store(p.tag, std::memory_order_release);
                return slot.id;
            }
            while (t == BUSY) t = tags_[i].load(std::memory_order_acquire);
            const Slot& slot = slots_[i];
            if (t == p.tag && slot.len == len && same_head(slot, s, len) &&
                (len <= HEAD || memcmp(arena_ + overflow_[i], s + HEAD, len - HEAD) == 0))
                return slot.id;
        }
    }

private:
    static constexpr u16 BUSY = 0xFFFF;
    struct alignas(32) Slot {
        u32 id, len;
        char head[HEAD];
    };

    static bool same_head(const Slot& slot, const char* s, u32 len) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&slot));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s - 8));
        u32 equal = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
        u32 want = ((u64(1) << std::min(len, HEAD)) - 1) << 8;
        return (equal & want) == want;
    }
    std::atomic<u16>* tags_ = zeroed<std::atomic<u16>>(SLOTS);
    Slot* slots_ = zeroed<Slot>(SLOTS);
    u32* overflow_ = zeroed<u32>(SLOTS);
    char* arena_ = zeroed<char>(8 << 20);
    std::atomic<u32> used_{0}, next_id_{0};
};

struct Totals {
    u64 count = 0, id_sum = 0, name_bytes = 0, note_bytes = 0, distinct = 0;
    i64 amount_sum = 0, amount_min = INT64_MAX, amount_max = INT64_MIN;
};

struct NamesSeen {
    static constexpr u32 WORDS = NameTable::IDS / 64;
    std::atomic<u64>* bits = zeroed<std::atomic<u64>>(256 * WORDS);

    std::atomic<u64>& word(u32 c, u32 name) { return bits[c * WORDS + name / 64]; }

    void prefetch(u32 c, u32 name) { _mm_prefetch(reinterpret_cast<const char*>(&word(c, name)), _MM_HINT_T0); }

    void mark(u32 c, u32 name) {
        u64 bit = u64(1) << (name % 64);
        if (!(word(c, name).load(std::memory_order_relaxed) & bit))
            word(c, name).fetch_or(bit, std::memory_order_relaxed);
    }

    u64 count(u32 c) const {
        u64 n = 0;
        for (u32 w = 0; w < WORDS; ++w) n += __builtin_popcountll(bits[c * WORDS + w].load(std::memory_order_relaxed));
        return n;
    }
};

inline u32 text_length(const char* s, const char* e, u32 q) { return (e - s) - (q ? q / 2 + 1 : 0); }

void parse_records(const char* p, const char* end, const char* file_end, const u8* country_of,
                   size_t regions, NameTable& names, NamesSeen& seen, Totals* totals) {
    constexpr u64 LAG = 16;
    struct Pending {
        const char* name;
        u32 len, country;
        NameTable::Probe probe;
    } pending[LAG];
    struct Mark {
        u32 country, id;
    } marks[LAG];
    u64 n = 0, last = UINT64_MAX;
    auto advance = [&] {
        u32 k = n % LAG;
        if (n >= 2 * LAG && n - 2 * LAG < last) seen.mark(marks[k].country, marks[k].id);
        if (n >= LAG && n - LAG < last) {
            const Pending& old = pending[k];
            marks[k] = {old.country, names.id(old.name, old.len, old.probe)};
            seen.prefetch(marks[k].country, marks[k].id);
        }
    };

    Structure sep(p, file_end);
    while (p < end) {
        auto [f1, q1] = sep.next();
        auto [f2, q2] = sep.next();
        auto [f3, q3] = sep.next();
        const char* f4 = sep.next().at;
        const char* eol = sep.next().at;
        if (u8(*p) == 0xEF) p += 3;
        u64 id = digits(p, f1);
        u64 region = digits(f3 + 1, f4);
        const char* amount = f4 + 1;
        const char* amount_end = eol - (eol[-1] == '\r');
        bool negative = *amount == '-';
        i64 cents = digits(amount + negative, amount_end - 3) * 100 + digits(amount_end - 2, amount_end);
        if (negative) cents = -cents;

        u32 c = region < regions ? country_of[region] : 0;
        Totals& t = totals[c];
        ++t.count;
        t.id_sum += id;
        t.amount_sum += cents;
        t.amount_min = std::min(t.amount_min, cents);
        t.amount_max = std::max(t.amount_max, cents);
        u32 name_quotes = q2 - q1;
        t.name_bytes += text_length(f1 + 1, f2, name_quotes);
        t.note_bytes += text_length(f2 + 1, f3, q3 - q2);
        bool quoted = name_quotes != 0;
        advance();
        Pending& next = pending[n++ % LAG];
        next = {f1 + 1 + quoted, u32(f2 - f1 - 1 - 2 * quoted), c, {}};
        next.probe = names.probe(next.name, next.len);
        p = eol + 1;
    }
    for (last = n; n < last + 2 * LAG; ++n) advance();
}

inline u32 code(const char* p) { return (p[0] - 'A') << 10 | (p[1] - 'A') << 5 | (p[2] - 'A'); }

std::vector<size_t> split(const char* data, size_t from, size_t size, int n, std::string_view delim) {
    std::vector<size_t> cut(n + 1, size);
    cut[0] = from;
    for (int t = 1; t < n; ++t) {
        size_t c = std::max(cut[t - 1], from + (size - from) * t / n);
        auto* hit = static_cast<const char*>(memmem(data + c, size - c, delim.data(), delim.size()));
        cut[t] = hit ? hit - data + delim.size() : size;
    }
    return cut;
}

void put_money(std::string& out, i64 cents) {
    u64 a = cents < 0 ? -u64(cents) : u64(cents);
    out += cents < 0 ? " -" : " ";
    out += std::to_string(a / 100);
    out += {'.', char('0' + a % 100 / 10), char('0' + a % 10)};
}

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    const int nthreads = std::clamp<int>(std::thread::hardware_concurrency(), 1, 12);
    size_t rsize, size;
    const char* regions = map_file("/data/regions.csv", rsize);
    const char* records = map_file("/data/records.csv", size);

    const char* last = static_cast<const char*>(memrchr(regions, '\n', rsize - 1)) + 1;
    const size_t nregions = atoll(last) + 1;
    u8* country_of = zeroed<u8>(nregions);
    std::vector<size_t> rcut = split(regions, strchr(regions, '\n') - regions + 1, rsize, nthreads, "\n");
    std::vector<std::atomic<u8>> present(1 << 15);
    std::array<u8, 1 << 15> index_of{};
    std::vector<u32> codes;
    auto number_countries = [&]() noexcept {
        if (!codes.empty()) return;
        for (u32 k = 0; k < present.size(); ++k)
            if (present[k]) codes.push_back(k), index_of[k] = codes.size();
        if (codes.size() > 255) _exit(1);
    };
    std::barrier sync(nthreads, number_countries);

    size_t start = size ? static_cast<const char*>(memchr(records, '\n', size)) - records + 1 : 0;
    std::vector<size_t> cut = split(records, start, size, nthreads, "\r\n");
    std::vector<std::array<Totals, 256>> totals(nthreads);
    NameTable names;
    NamesSeen seen;

    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t)
        pool.emplace_back([&, t] {
            const char* from = regions + rcut[t];
            const char* to = regions + rcut[t + 1];
            for (const char* p = from; p < to; p = strchr(p, '\n') + 1)
                present[code(strchr(p, ',') + 1)].store(1, std::memory_order_relaxed);
            sync.arrive_and_wait();
            for (const char* p = from; p < to; p = strchr(p, '\n') + 1) {
                const char* comma = strchr(p, ',');
                country_of[digits(p, comma)] = index_of[code(comma + 1)];
            }
            sync.arrive_and_wait();
            parse_records(records + cut[t], records + cut[t + 1], records + size, country_of,
                          nregions, names, seen, totals[t].data());
            sync.arrive_and_wait();
            for (u32 c = t; c <= codes.size(); c += nthreads) totals[t][c].distinct = seen.count(c);
        });
    for (auto& th : pool) th.join();

    std::string out;
    for (u32 i = 1; i <= codes.size() + 1; ++i) {
        u32 c = i % (codes.size() + 1);
        Totals sum;
        for (const auto& per_thread : totals) {
            const Totals& t = per_thread[c];
            sum.count += t.count;
            sum.id_sum += t.id_sum;
            sum.name_bytes += t.name_bytes;
            sum.note_bytes += t.note_bytes;
            sum.distinct += t.distinct;
            sum.amount_sum += t.amount_sum;
            sum.amount_min = std::min(sum.amount_min, t.amount_min);
            sum.amount_max = std::max(sum.amount_max, t.amount_max);
        }
        if (sum.count == 0) continue;
        if (c == 0) {
            out += "_UNMATCHED";
        } else {
            u32 k = codes[c - 1];
            out += {char('A' + (k >> 10)), char('A' + (k >> 5 & 31)), char('A' + (k & 31))};
        }
        for (u64 v : {sum.count, sum.id_sum}) out += ' ' + std::to_string(v);
        for (i64 v : {sum.amount_sum, sum.amount_min, sum.amount_max}) put_money(out, v);
        for (u64 v : {sum.distinct, sum.name_bytes, sum.note_bytes}) out += ' ' + std::to_string(v);
        out += '\n';
    }
    FILE* f = fopen("/out/aggregates.txt", "wb");
    if (!f) _exit(1);
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    _exit(0);
}
