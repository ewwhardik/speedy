// Speed Demon 2026, Track C (Join)
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <mutex>
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

template <class T>
void populate_share(T* p, size_t n, int t, int nthreads) {
    const size_t pages = (n * sizeof(T) + 4095) / 4096, from = pages * t / nthreads, to = pages * (t + 1) / nthreads;
    if (to > from) madvise(reinterpret_cast<char*>(p) + from * 4096, (to - from) * 4096, MADV_POPULATE_WRITE);
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

inline u64 fold(u64 v, size_t n) {
    v &= 0x0F0F0F0F0F0F0F0F & ~u64(0) << 8 * (8 - n);
    v = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FF;
    v = (v * 100 + (v >> 16)) & 0x0000FFFF0000FFFF;
    return (v * 10000 + (v >> 32)) & 0xFFFFFFFF;
}

inline u64 digits(const char* s, const char* e) { return fold(load8(e - 8), e - s); }

inline u64 money(const char* s, const char* e) {
    u64 w = load8(e - 8);
    return fold((w & 0xFFFFFFFFFF) << 8 | (w & 0xFFFF000000000000), e - s - 1);
}

inline const char* comma_after(const char* p) {
    u64 x = load8(p) ^ 0x2C2C2C2C2C2C2C2C;
    return p + __builtin_ctzll((x - 0x0101010101010101) & ~x & 0x8080808080808080) / 8;
}

[[gnu::always_inline]] inline const char* find_comma(const char* p) {
    u32 m = match32(p, ',');
    if (m) return p + __builtin_ctz(m);
    p += 32;
    for (;; p += 64)
        if (u64 m2 = match32(p, ',') | u64(match32(p + 32, ',')) << 32) return p + __builtin_ctzll(m2);
}

struct Text {
    const char* at;
    u32 raw, length;
};

[[gnu::always_inline]] inline Text text(const char*& p) {
    const char* s = p;
    if (__builtin_expect(*s != '"', 1)) {
        const char* e = find_comma(s);
        p = e + 1;
        return {s, u32(e - s), u32(e - s)};
    }
    u32 pairs = 0;
    const char* q = s + 1;
    for (u64 m = match32(q, '"') | u64(match32(q + 32, '"')) << 32;;) {
        u32 i = m ? __builtin_ctzll(m) : 64;
        if (i >= 63) {
            q += i;
            m = match32(q, '"') | u64(match32(q + 32, '"')) << 32;
            continue;
        }
        if (!(m >> (i + 1) & 1)) {
            q += i;
            break;
        }
        ++pairs;
        m &= m - 1;
        m &= m - 1;
    }
    p = q + 2;
    return {s + 1, u32(q - s - 1), u32(q - s - 1 - pairs)};
}

class NameTable {
public:
    static constexpr u32 BITS = 17, SLOTS = 1 << BITS;

    static u64 key(const char* s, u32 len) {
        static constexpr char KEEP[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        __m128i head = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
        head = _mm_and_si128(head, _mm_loadu_si128(reinterpret_cast<const __m128i*>(KEEP + 16 - std::min(len, 16u))));
        u64 h = (u64(_mm_cvtsi128_si64(head)) * 0x9E3779B97F4A7C15ull ^ len) + u64(_mm_extract_epi64(head, 1)) * 0xBF58476D1CE4E5B9ull;
        for (u32 i = 16; i < len; i += 8) {
            u64 w = load8(s + i);
            if (len - i < 8) w &= (u64(1) << 8 * (len - i)) - 1;
            h = (h ^ w) * 0x94D049BB133111EBull;
        }
        h = (h ^ h >> 31) * 0xD6E8FEB86659FD93ull;
        h ^= h >> 32;
        return h + (h == 0);
    }

    void prefetch(u64 k) const { _mm_prefetch(reinterpret_cast<const char*>(&slots_[k >> (64 - BITS)]), _MM_HINT_T0); }

    u32 id(u64 k) {
        for (u32 i = k >> (64 - BITS) & ~3u;; i = (i + 4) % SLOTS) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&slots_[i]));
            u32 hit = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v, _mm256_set1_epi64x(k))));
            u32 free = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v, _mm256_setzero_si256())));
            if (hit) return i + __builtin_ctz(hit);
            for (; free; free &= free - 1) {
                u32 j = i + __builtin_ctz(free);
                u64 s = 0;
                if (slots_[j].compare_exchange_strong(s, k) || s == k) return j;
            }
        }
    }

    void populate(int t, int nthreads) { populate_share(slots_, SLOTS, t, nthreads); }

private:
    std::atomic<u64>* slots_ = zeroed<std::atomic<u64>>(SLOTS);
};

struct alignas(64) Totals {
    u64 count = 0, id_sum = 0, name_bytes = 0, note_bytes = 0, distinct = 0;
    i64 amount_sum = 0, amount_min = INT64_MAX, amount_max = INT64_MIN;
};

struct NamesSeen {
    static constexpr u32 WORDS = NameTable::SLOTS / 64;
    std::atomic<u64>* bits = zeroed<std::atomic<u64>>(256 * WORDS);

    std::atomic<u64>& word(u32 c, u32 name) { return bits[c * WORDS + name / 64]; }

    void prefetch(u32 c, u32 name) { __builtin_prefetch(&word(c, name), 1); }

    bool mark(u32 c, u32 name) {
        u64 bit = u64(1) << name % 64;
        return !(word(c, name).fetch_or(bit, std::memory_order_relaxed) & bit);
    }
};

void parse_records(const char* p, const char* end, const char* file_end, const u8* country_of,
                   size_t regions, NameTable& names, NamesSeen& seen, Totals* totals) {
    constexpr u64 LAG = 16;
    struct Pending {
        u64 name;
        u32 region, id, name_bytes, note_bytes;
        i64 cents;
    } pending[LAG];
    struct Mark {
        u32 country, id;
    } marks[LAG];
    u64 n = 0, last = UINT64_MAX;
    auto advance = [&]() __attribute__((always_inline)) {
        u32 k = n % LAG;
        if (n >= 2 * LAG && n - 2 * LAG < last) totals[marks[k].country].distinct += seen.mark(marks[k].country, marks[k].id);
        if (n >= LAG && n - LAG < last) {
            const Pending& old = pending[k];
            u32 c = old.region < regions ? country_of[old.region] : 0;
            Totals& t = totals[c];
            ++t.count;
            t.id_sum += old.id;
            t.amount_sum += old.cents;
            t.amount_min = std::min(t.amount_min, old.cents);
            t.amount_max = std::max(t.amount_max, old.cents);
            t.name_bytes += old.name_bytes;
            t.note_bytes += old.note_bytes;
            marks[k] = {c, names.id(old.name)};
            seen.prefetch(c, marks[k].id);
        }
    };

    while (p < end) {
        _mm_prefetch(p + 2048, _MM_HINT_T0);
        if (u8(*p) == 0xEF) p += 3;
        const char* f1 = comma_after(p);
        u64 id = digits(p, f1);
        p = f1 + 1;
        Text name = text(p), note = text(p);
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        u32 lf = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n')));
        u32 comma = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(',')));
        const char* f4 = p + __builtin_ctz(comma);
        u64 region = digits(p, f4);
        const char* amount = f4 + 1;
        const char* eol = lf ? p + __builtin_ctz(lf) : file_end;
        const char* amount_end = eol - (eol[-1] == '\r');
        bool negative = *amount == '-';
        i64 cents = money(amount + negative, amount_end);
        if (negative) cents = -cents;

        u32 r = region < regions ? u32(region) : u32(regions);
        _mm_prefetch(reinterpret_cast<const char*>(country_of + r), _MM_HINT_T0);
        advance();
        u64 key = NameTable::key(name.at, name.raw);
        pending[n++ % LAG] = {key, r, u32(id), name.length, note.length, cents};
        names.prefetch(key);
        p = eol + 1;
    }
    for (last = n; n < last + 2 * LAG; ++n) advance();
}

inline u32 code(const char* p) { return (p[0] - 'A') << 10 | (p[1] - 'A') << 5 | (p[2] - 'A'); }

inline std::pair<u64, u32> region_line(const char*& p) {
    u64 x = load8(p) ^ 0x2C2C2C2C2C2C2C2C;
    const char* comma = p + __builtin_ctzll((x - 0x0101010101010101) & ~x & 0x8080808080808080) / 8;
    std::pair<u64, u32> line{digits(p, comma), code(comma + 1)};
    for (p = comma + 4; *p++ != '\n';) {}
    return line;
}

struct Phase {
    std::atomic<int> claimed{0}, done{0};
    const int pieces;

    template <class F>
    void run(F&& f) {
        for (int k; (k = claimed++) < pieces;)
            if (f(k), ++done == pieces) done.notify_all();
        for (int d; (d = done) < pieces;) done.wait(d);
    }
};

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
    const int nthreads = 12;
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
    std::once_flag numbered;
    auto number_countries = [&] {
        for (u32 k = 0; k < present.size(); ++k)
            if (present[k]) codes.push_back(k), index_of[k] = codes.size();
        if (codes.size() > 255) _exit(1);
    };

    size_t start = size ? static_cast<const char*>(memchr(records, '\n', size)) - records + 1 : 0;
    const int pieces = 8 * nthreads;
    std::vector<size_t> cut = split(records, start, size, pieces, "\r\n");
    std::vector<std::array<Totals, 256>> totals(nthreads);
    NameTable names;
    NamesSeen seen;

    Phase flag{.pieces = nthreads}, fill{.pieces = nthreads}, parse{.pieces = pieces};
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t)
        pool.emplace_back([&, t] {
            flag.run([&](int k) {
                populate_share(country_of, nregions, k, nthreads);
                names.populate(k, nthreads);
                for (const char* p = regions + rcut[k]; p < regions + rcut[k + 1];)
                    present[region_line(p).second].store(1, std::memory_order_relaxed);
            });
            std::call_once(numbered, number_countries);
            fill.run([&](int k) {
                populate_share(seen.bits, (codes.size() + 1) * NamesSeen::WORDS, k, nthreads);
                for (const char* p = regions + rcut[k]; p < regions + rcut[k + 1];) {
                    auto [id, c] = region_line(p);
                    country_of[id] = index_of[c];
                }
            });
            parse.run([&](int k) {
                uintptr_t from = uintptr_t(records + cut[k]) & ~uintptr_t(4095);
                madvise(reinterpret_cast<void*>(from), uintptr_t(records + cut[k + 1]) - from, MADV_POPULATE_READ);
                parse_records(records + cut[k], records + cut[k + 1], records + size, country_of, nregions, names,
                              seen, totals[t].data());
                uintptr_t to = uintptr_t(records + cut[k + 1]) & ~uintptr_t(4095);
                if (to > from) madvise(reinterpret_cast<void*>(from), to - from, MADV_DONTNEED);
            });
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
