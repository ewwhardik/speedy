// Speed Demon 2026, Track A (Ingest).
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <string>
#include <string_view>
#include <malloc.h>
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

    struct alignas(8) Slot {
        u32 hash = 0;
        u16 id_plus_1 = 0;
        u16 len = 0;
    };

    u32 intern(const char* s, u32 len) {
        u64 a = load8(s), b = len >= 8 ? load8(s + len - 8) : 0;
        if (len < 8) a &= (u64(1) << (8 * len)) - 1;
        u32 h = _mm_crc32_u64(_mm_crc32_u64(len, a), b);
        for (u32 i = h;; ++i) {
            Slot& slot = slots_[i & (SLOTS - 1)];
            if (slot.id_plus_1 == 0) {
                if (names.size() == MAX) _exit(1);
                u32 id = names.size();
                names.emplace_back(s, len);
                slot.hash = h;
                slot.len = len;
                slot.id_plus_1 = id + 1;
                return id;
            }
            if (slot.hash == h && slot.len == len) {
                std::string_view name = names[slot.id_plus_1 - 1];
                if (__builtin_expect(memcmp(name.data(), s, len) == 0, 1))
                    return slot.id_plus_1 - 1;
            }
        }
    }

private:
    std::array<Slot, SLOTS> slots_{};
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
        u64 last_ts = 0;
        int h = 0;
        Block* cur_b = nullptr;
        while (p < end) {
            u64 ts = load8(p + 8);
            if (__builtin_expect(ts != last_ts, 0)) {
                last_ts = ts;
                int day = (p[9] - '0') + (p[6] - '0');
                int hour = p[11] * 10 + p[12] - 528;
                h = day * 24 + hour - 18;
                cur_b = head[h];
            }

            const char* m = p + 28;
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + 32));
            u64 sp = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8(' '))) |
                     (u64(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8(' ')))) << 32);
            u64 nl = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\n'))) |
                     (u64(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\n')))) << 32);
            const char* ep = m + __builtin_ctzll(sp) + 1;
            const char* ee = nl ? m + __builtin_ctzll(sp & (sp - 1)) : find(ep, ' ');
            const char* eol = nl ? m + __builtin_ctzll(nl) : find(ee + 5, '\n');
            const u32 key = endpoints.intern(ep, ee - ep) | (ee[1] == '5' ? ERR_BIT : 0);
            const u32 lat = parse_digits(eol, eol - (ee + 5));
            p = eol + 1;
            if (__builtin_expect(lat >= BIG, 0)) {
                big[h].push_back({lat, key});
                continue;
            }
            if (__builtin_expect(!cur_b || cur_b->n == Block::CAP, 0)) {
                auto* fresh = reinterpret_cast<Block*>(region + used);
                used += sizeof(Block);
                fresh->next = cur_b;
                head[h] = cur_b = fresh;
            }
            cur_b->rec[cur_b->n++] = lat << KEY_BITS | key;
        }
    }
};

void hour_label(int h, char out[22]) {
    int day = (h + 18) / 24;
    sprintf(out, "2026-%02d-%02dT%02d:00:00Z ", day < 2 ? 10 : 11,
            day < 2 ? 30 + day : day - 1, (h + 18) % 24);
}

template <class F>
void run_threads(int n, F&& body) {
    std::vector<std::thread> pool;
    for (int t = 0; t < n; ++t) pool.emplace_back(body, t);
    for (auto& th : pool) th.join();
}

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    mallopt(M_MMAP_THRESHOLD, 1 << 16);
    const int nthreads = std::clamp<int>(std::thread::hardware_concurrency(), 1, 12);
    int fd = open("/data/requests.log", O_RDONLY);
    FILE* out = fopen("/out/summary.txt", "wb");
    if (fd < 0 || !out) return 1;

    size_t size = lseek(fd, 0, SEEK_END);
    auto* data = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(data, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) return 1;

    std::vector<size_t> cut(nthreads + 1, size);
    cut[0] = 0;
    for (int t = 1; t < nthreads; ++t) {
        cut[t] = std::max(cut[t - 1], size * t / nthreads);
        while (cut[t] < size && data[cut[t] - 1] != '\n') ++cut[t];
    }
    std::vector<Worker> workers(nthreads);
    run_threads(nthreads, [&](int t) { workers[t].parse(data + cut[t], data + cut[t + 1]); });

    EndpointTable global;
    std::vector<std::vector<u32>> rank(nthreads);
    for (int t = 0; t < nthreads; ++t)
        for (std::string_view n : workers[t].endpoints.names)
            rank[t].push_back(global.intern(n.data(), n.size()));
    std::vector<std::string_view> names = global.names;
    std::sort(names.begin(), names.end());
    std::vector<u32> rank_of(names.size());
    for (u32 i = 0; i < names.size(); ++i)
        rank_of[global.intern(names[i].data(), names[i].size())] = i;
    for (auto& r : rank)
        for (u32& id : r) id = rank_of[id];

    std::vector<std::string> text(HOURS);
    for (int h = 0; h < HOURS; ++h) text[h].reserve(131072);
    std::vector<std::atomic<bool>> done(HOURS);
    std::thread writer([&] {
        for (int h = 0; h < HOURS; ++h) {
            done[h].wait(false);
            fwrite(text[h].data(), 1, text[h].size(), out);
            std::string().swap(text[h]);
        }
    });

    const u32 ne = names.size();
    std::atomic<int> next{0};
    run_threads(nthreads, [&](int) {
        std::vector<u32> count, errors, cursor, lat;
        for (int h; (h = next++) < HOURS;) {
            count.assign(ne + 1, 0);
            errors.assign(ne, 0);
            for (int i = 0; i < nthreads; ++i) {
                auto tally = [&](u32 key) {
                    u32 e = rank[i][key & ID_MASK];
                        ++count[e + 1];
                        errors[e] += (key & ERR_BIT) != 0;
                    };
                    for (Block* b = workers[i].head[h]; b; b = b->next)
                        for (u32 j = 0; j < b->n; ++j) tally(b->rec[j]);
                    for (auto [l, key] : workers[i].big[h]) tally(key);
                }
                for (u32 e = 0; e < ne; ++e) count[e + 1] += count[e];
                cursor.assign(count.begin(), count.end() - 1);
                lat.resize(count[ne]);
                for (int i = 0; i < nthreads; ++i) {
                    for (Block* b = workers[i].head[h]; b; madvise(std::exchange(b, b->next), sizeof(Block), MADV_DONTNEED))
                        for (u32 j = 0; j < b->n; ++j)
                            lat[cursor[rank[i][b->rec[j] & ID_MASK]]++] = b->rec[j] >> KEY_BITS;
                    for (auto [l, key] : workers[i].big[h]) lat[cursor[rank[i][key & ID_MASK]]++] = l;
                }

                char label[22], line[256];
                hour_label(h, label);
                for (u32 e = 0; e < ne; ++e) {
                    u32* a = lat.data() + count[e];
                    u32 n = count[e + 1] - count[e];
                    if (n == 0) continue;
                    u32 v[6] = {n, 0, 0, 0, errors[e], u32(u64(errors[e]) * 1000000 / n)};
                    if (n <= 128) {
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
                done[h] = true;
                done[h].notify_one();
            }
        });
    writer.join();
    fclose(out);
    return 0;
}
