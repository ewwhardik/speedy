#include <algorithm>
#include <atomic>
#include <barrier>
#include <charconv>
#include <cstdint>
#include <cstdio>
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
#include <utility>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int HOURS = 72;
constexpr u32 KEY_BITS = 12, ID_MASK = (1u << 11) - 1, ERR_BIT = 1u << 11, BIG = 1u << 20;

inline u64 load8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

inline u32 match32(const char* p, char c) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(c)));
}

const char* find(const char* s, char c) {
    for (;; s += 32)
        if (u32 m = match32(s, c)) return s + __builtin_ctz(m);
}

inline u32 parse_digits(const char* end, u32 len) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(end - 16));
    v = _mm_subs_epu8(v, _mm_set1_epi8('0'));
    __m128i lane = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    v = _mm_and_si128(v, _mm_cmpgt_epi8(lane, _mm_set1_epi8(char(15 - len))));
    v = _mm_maddubs_epi16(v, _mm_set1_epi16(0x010A));
    v = _mm_madd_epi16(v, _mm_set1_epi32(0x00010064));
    v = _mm_packus_epi32(v, v);
    v = _mm_madd_epi16(v, _mm_set1_epi32(0x00012710));
    u64 high = u32(_mm_cvtsi128_si32(v)), low = u32(_mm_extract_epi32(v, 1));
    return u32(high * 100000000 + low);
}

class EndpointTable {
public:
    static constexpr u32 SLOTS = 1 << 12, MAX = SLOTS / 2;
    std::vector<std::string_view> names;

    u32 intern(const char* s, u32 len) {
        u64 a = load8(s), b = len >= 8 ? load8(s + len - 8) : 0;
        if (len < 8) a &= (u64(1) << 8 * len) - 1;
        for (u32 i = _mm_crc32_u64(_mm_crc32_u64(len, a), b);; ++i) {
            u32& slot = slots_[i % SLOTS];
            if (slot == 0) {
                if (names.size() == MAX) _exit(1);
                names.emplace_back(s, len);
                return (slot = names.size()) - 1;
            }
            std::string_view n = names[slot - 1];
            if (n.size() == len && memcmp(n.data(), s, len) == 0) return slot - 1;
        }
    }

private:
    std::vector<u32> slots_ = std::vector<u32>(SLOTS);
};

struct Block {
    static constexpr u32 CAP = (16384 - 16) / 4;
    Block* next;
    u32 n = 0;
    u32 rec[CAP];
};

struct Worker {
    EndpointTable endpoints;
    Block* head[HOURS] = {};
    std::vector<std::pair<u32, u32>> big[HOURS];
    char* region = nullptr;
    size_t used = 0;

    void parse(const char* p, const char* end) {
        const size_t reserve = ((end - p) / 40 / Block::CAP + 1 + HOURS) * sizeof(Block);
        region = static_cast<char*>(mmap(nullptr, reserve, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        while (p < end) {
            int day_idx = (p[6] == '0') ? (p[9] - '0') : (p[9] - '0' + 1);
            int hour = (p[11] - '0') * 10 + (p[12] - '0');
            int h = day_idx * 24 + hour - 18;

            const char* m = p + 28;
            u64 sp = match32(m, ' ') | u64(match32(m + 32, ' ')) << 32;
            u64 nl = match32(m, '\n') | u64(match32(m + 32, '\n')) << 32;
            const char* ep = m + __builtin_ctzll(sp) + 1;
            const char* ee = nl ? m + __builtin_ctzll(sp & (sp - 1)) : find(ep, ' ');
            const char* eol = nl ? m + __builtin_ctzll(nl) : find(ee + 5, '\n');
            const u32 key = endpoints.intern(ep, ee - ep) | (ee[1] == '5' ? ERR_BIT : 0);
            const u32 lat = parse_digits(eol, eol - ee - 5);
            p = eol + 1;
            if (lat >= BIG) {
                big[h].push_back({lat, key});
                continue;
            }
            Block*& b = head[h];
            if (!b || b->n == Block::CAP) {
                auto* fresh = reinterpret_cast<Block*>(region + used);
                used += sizeof(Block);
                fresh->next = b;
                b = fresh;
            }
            b->rec[b->n++] = lat << KEY_BITS | key;
        }
    }
};

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    mallopt(M_MMAP_THRESHOLD, 1 << 16);
    const int nthreads = 12;
    int fd = open("/data/requests.log", O_RDONLY);
    FILE* out = fopen("/out/summary.txt", "wb");
    if (fd < 0 || !out) _exit(1);

    size_t size = lseek(fd, 0, SEEK_END);
    auto* data = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(data, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) _exit(1);

    char labels[HOURS][22];
    for (int h = 0; h < HOURS; ++h) {
        int day = (h + 18) / 24;
        sprintf(labels[h], "2026-%02d-%02dT%02d:00:00Z ", day < 2 ? 10 : 11,
                day < 2 ? 30 + day : day - 1, (h + 18) % 24);
    }

    std::vector<size_t> cut(nthreads + 1, size);
    cut[0] = 0;
    for (int t = 1; t < nthreads; ++t) {
        cut[t] = std::max(cut[t - 1], size * t / nthreads);
        while (cut[t] < size && data[cut[t] - 1] != '\n') ++cut[t];
    }
    std::vector<Worker> workers(nthreads);
    EndpointTable global;
    std::vector<std::vector<u32>> rank(nthreads);
    std::vector<std::string_view> names;
    u32 ne = 0;

    std::barrier sync(nthreads, [&]() noexcept {
        for (int t = 0; t < nthreads; ++t)
            for (std::string_view n : workers[t].endpoints.names)
                rank[t].push_back(global.intern(n.data(), n.size()));
        names = global.names;
        std::sort(names.begin(), names.end());
        std::vector<u32> rank_of(names.size());
        for (u32 i = 0; i < names.size(); ++i)
            rank_of[global.intern(names[i].data(), names[i].size())] = i;
        for (auto& r : rank)
            for (u32& id : r) id = rank_of[id];
        ne = names.size();
    });

    std::vector<std::string> text(HOURS);
    for (int h = 0; h < HOURS; ++h) text[h].reserve(131072);
    std::atomic<int> next_hour{0};

    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            workers[t].parse(data + cut[t], data + cut[t + 1]);
            sync.arrive_and_wait();

            std::vector<u32> count, errors, cursor, lat;
            for (int h; (h = next_hour.fetch_add(1, std::memory_order_relaxed)) < HOURS;) {
                count.assign(ne + 1, 0);
                errors.assign(ne, 0);
                for (int wt = 0; wt < nthreads; ++wt) {
                    auto tally = [&](u32 key) {
                        u32 e = rank[wt][key & ID_MASK];
                        ++count[e + 1];
                        errors[e] += (key & ERR_BIT) != 0;
                    };
                    for (Block* b = workers[wt].head[h]; b; b = b->next)
                        for (u32 i = 0; i < b->n; ++i) tally(b->rec[i]);
                    for (auto [l, key] : workers[wt].big[h]) tally(key);
                }
                for (u32 e = 0; e < ne; ++e) count[e + 1] += count[e];
                cursor.assign(count.begin(), count.end() - 1);
                lat.resize(count[ne]);
                for (int wt = 0; wt < nthreads; ++wt) {
                    for (Block* b = workers[wt].head[h]; b; madvise(std::exchange(b, b->next), sizeof(Block), MADV_DONTNEED))
                        for (u32 i = 0; i < b->n; ++i)
                            lat[cursor[rank[wt][b->rec[i] & ID_MASK]]++] = b->rec[i] >> KEY_BITS;
                    for (auto [l, key] : workers[wt].big[h])
                        lat[cursor[rank[wt][key & ID_MASK]]++] = l;
                }

                char line[256];
                const char* label = labels[h];
                for (u32 e = 0; e < ne; ++e) {
                    u32* a = lat.data() + count[e];
                    u32 n = count[e + 1] - count[e];
                    if (n == 0) continue;
                    u32 v[6] = {n, 0, 0, 0, errors[e], u32(u64(errors[e]) * 1000000 / n)};
                    if (n <= 256) {
                        std::sort(a, a + n);
                        v[1] = a[(50u * n + 99) / 100 - 1];
                        v[2] = a[(95u * n + 99) / 100 - 1];
                        v[3] = a[(99u * n + 99) / 100 - 1];
                    } else {
                        u32 lo = 0, k = 1;
                        for (u32 p : {50u, 95u, 99u}) {
                            u32 r = (p * n + 99) / 100 - 1;
                            std::nth_element(a + lo, a + r, a + n);
                            v[k++] = a[lo = r];
                        }
                    }
                    char* o = std::copy(names[e].begin(), names[e].end(), std::copy_n(label, 21, line));
                    for (u32 x : v) {
                        *o++ = ' ';
                        o = std::to_chars(o, o + 10, x).ptr;
                    }
                    *o++ = '\n';
                    text[h].append(line, o);
                }
            }
        });
    }
    for (auto& th : pool) th.join();

    for (int h = 0; h < HOURS; ++h)
        fwrite(text[h].data(), 1, text[h].size(), out);
    fclose(out);
    _exit(0);
}
