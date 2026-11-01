// Speed Demon 2026, Track A (Ingest).
//
// Phase 1 (parallel): the log is memory-mapped and cut into line-aligned
// ranges, one per thread. Every field sits at a predictable place: the
// timestamp is always 27 bytes and the 72-hour span is fixed, so the hour
// index comes straight from four digits. Endpoints are interned in a private
// open-addressing table (checked with full SIMD compares, never trusted on the
// hash alone). Each line becomes one 32-bit record, latency << 12 | local
// endpoint id << 1 | 5xx flag, appended to that thread's block list for its
// hour. Latencies of 2^20 us or more (rare) go to a side list.
//
// Phase 2 (parallel, one hour at a time): the private tables are merged and
// sorted once, so an endpoint's global id is its rank in byte order. Each hour
// counting-sorts its ~43k latencies by endpoint rank into L2-sized scratch,
// finds p50/p95/p99 per group with a radix select (small groups are sorted),
// and formats its lines into its own buffer. A writer thread writes the hours
// in order as they complete.
#include <algorithm>
#include <atomic>
#include <utility>
#include <sys/prctl.h>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr int HOURS = 72;          // 2026-10-30T18:00Z to 2026-11-02T17:59:59Z
constexpr int MAX_ENDPOINTS = 2048; // the catalogue has 1,842
constexpr u32 BIG = 1u << 20;      // latencies at or above this skip the packing

inline u64 load8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

inline __m256i load32(const char* p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

// Bit i is set when byte i of the 64 at p is a space, LF or other control
// byte. The input is ASCII, so a signed compare against 0x21 finds them all.
inline u64 delims64(const char* p) {
    __m256i k = _mm256_set1_epi8(0x21);
    u32 lo = _mm256_movemask_epi8(_mm256_cmpgt_epi8(k, load32(p)));
    u32 hi = _mm256_movemask_epi8(_mm256_cmpgt_epi8(k, load32(p + 32)));
    return lo | u64(hi) << 32;
}

// The first delimiter at or after s.
inline const char* next_delim(const char* s) {
    for (;; s += 64)
        if (u64 m = delims64(s)) return s + __builtin_ctzll(m);
}

// Up to 8 ASCII digits ending at e, with no branches (SWAR: each step merges
// neighbouring digit groups, most significant first in little-endian order).
inline u32 digits8(const char* e, u32 len) {
    u64 v = load8(e - 8) & 0x0F0F0F0F0F0F0F0F & ~u64(0) << 8 * (8 - len);
    v = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FF;
    v = (v * 100 + (v >> 16)) & 0x0000FFFF0000FFFF;
    return u32((v * 10000 + (v >> 32)) & 0xFFFFFFFF);
}

inline u32 parse_latency(const char* e, u32 len) {
    if (__builtin_expect(len <= 8, 1)) return digits8(e, len);
    return digits8(e - 8, len - 8) * 100000000u + digits8(e, 8);
}

// Endpoint names live in 128-byte zero-padded slots so they can be compared
// and copied 32 bytes at a time.
struct Name {
    alignas(32) char bytes[128];
    u32 len;
};

// True when the len bytes at s equal name n (len is already known to match).
inline bool same(const char* s, const Name& n, u32 len) {
    for (u32 i = 0; i < len; i += 32) {
        u32 eq = _mm256_movemask_epi8(
            _mm256_cmpeq_epi8(load32(s + i), load32(n.bytes + i)));
        u32 need = len - i >= 32 ? ~0u : _bzhi_u32(~0u, len - i);
        if (~eq & need) return false;
    }
    return true;
}

// A private endpoint table: open addressing on a CRC of length, head and tail.
// Each slot keeps the name's first 32 bytes, so most lookups are one compare.
struct EndpointTable {
    static constexpr u32 SLOTS = 4096;
    struct alignas(64) Slot {
        char head[32];
        u32 len, id; // len 0 marks an empty slot
    };
    std::vector<Slot> slots = std::vector<Slot>(SLOTS);
    std::vector<Name> names;

    EndpointTable() { names.reserve(MAX_ENDPOINTS); }

    u32 intern(const char* s, u32 len) {
        __m256i first = load32(s);
        u64 tail = len >= 8 ? load8(s + len - 8) : 0;
        u32 i = _mm_crc32_u64(_mm_crc32_u32(len, _mm256_extract_epi32(first, 0)), tail);
        for (;; ++i) {
            Slot& slot = slots[i & (SLOTS - 1)];
            if (slot.len == 0) return insert(slot, s, len);
            u32 eq = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first, load32(slot.head)));
            u32 need = len >= 32 ? ~0u : _bzhi_u32(~0u, len);
            if (slot.len == len && (~eq & need) == 0 &&
                (len <= 32 || same(s, names[slot.id], len)))
                return slot.id;
        }
    }

    u32 insert(Slot& slot, const char* s, u32 len) {
        if (names.size() == MAX_ENDPOINTS || len > 125) _exit(1);
        Name& n = names.emplace_back();
        memset(n.bytes, 0, sizeof n.bytes);
        memcpy(n.bytes, s, len);
        n.len = len;
        memcpy(slot.head, n.bytes, 32);
        slot.len = len;
        slot.id = names.size() - 1;
        return slot.id;
    }
};

// Records of one hour, from every thread, in 16 KB blocks claimed from one
// contiguous region per hour: an hour is then read and freed in one sweep.
struct Block {
    static constexpr u32 CAP = (16384 - 8) / 4;
    u32 n, worker;
    u32 rec[CAP];
};

struct HourRegion {
    char* base;
    std::atomic<size_t> used{0};
};
HourRegion hour_region[HOURS];

struct Worker {
    u32 id;
    EndpointTable endpoints;
    Block* cur[HOURS] = {};
    std::vector<std::pair<u32, u32>> big[HOURS]; // (latency, key) pairs
    std::vector<u32> rank;                       // local id -> global rank

    Block* fresh_block(int h) {
        HourRegion& r = hour_region[h];
        auto* b = reinterpret_cast<Block*>(r.base + r.used.fetch_add(sizeof(Block)));
        b->n = 0;
        b->worker = id;
        return cur[h] = b;
    }

    // Parses the whole lines in [p, end).
    void parse(const char* p, const char* end) {
        u64 last_prefix = ~u64(0);
        int h = 0;
        Block* b = nullptr;
        while (p < end) {
            // "YYYY-MM-DDTHH": the month, day and hour digits at offset 6.
            u64 prefix = load8(p + 6);
            if (__builtin_expect(prefix != last_prefix, 0)) {
                int day = (p[8] - '0') * 10 + p[9] - '0';
                int hour = (p[11] - '0') * 10 + p[12] - '0';
                h = (p[6] == '0' ? day - 30 : day + 1) * 24 + hour - 18;
                last_prefix = prefix;
                b = cur[h];
            }
            // The method (3 to 7 letters) starts at 28; the endpoint after it.
            // The fields after the timestamp: method, endpoint, status and
            // latency, ended by three spaces and the LF.
            const char* m = p + 28;
            u64 d = delims64(m);
            const char* ep = m + __builtin_ctzll(d) + 1; // the method is <= 7
            d = _blsr_u64(d);
            const char* ee = d ? m + __builtin_ctzll(d) : next_delim(m + 64);
            d = _blsr_u64(_blsr_u64(d)); // skip the space after the status
            const char* eol = d ? m + __builtin_ctzll(d) : next_delim(ee + 5);
            u32 key = endpoints.intern(ep, ee - ep) << 1 | (ee[1] == '5');
            u32 lat = parse_latency(eol, eol - ee - 5);
            p = eol + 1;
            if (__builtin_expect(lat >= BIG, 0)) {
                big[h].push_back({lat, key});
                continue;
            }
            if (__builtin_expect(!b || b->n == Block::CAP, 0)) b = fresh_block(h);
            b->rec[b->n++] = lat << 12 | key;
        }
    }
};

// "00" to "99", for writing two digits at a time.
constexpr auto PAIRS = [] {
    struct { char c[200]; } t{};
    for (int i = 0; i < 100; ++i) t.c[2 * i] = '0' + i / 10, t.c[2 * i + 1] = '0' + i % 10;
    return t;
}();

// Writes v in decimal at o and returns the end.
inline char* put(char* o, u32 v) {
    int n = 1;
    for (u32 p = 10; n < 10 && v >= p; p *= 10) ++n;
    char* e = o + n;
    for (; v >= 100; v /= 100) memcpy(e -= 2, PAIRS.c + 2 * (v % 100), 2);
    if (v >= 10) memcpy(e - 2, PAIRS.c + 2 * v, 2);
    else e[-1] = '0' + v;
    return o + n;
}

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    const int nthreads = std::clamp<int>(std::thread::hardware_concurrency(), 1, 12);
    int fd = open("/data/requests.log", O_RDONLY);
    int out = open("/out/summary.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || out < 0) return 1;
    size_t size = lseek(fd, 0, SEEK_END);
    // Map one spare page past the end so 64-byte loads never fault.
    auto* data = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(data, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED)
        return 1;

    std::vector<size_t> cut(nthreads + 1, size);
    cut[0] = 0;
    for (int t = 1; t < nthreads; ++t) {
        cut[t] = std::max(cut[t - 1], size * t / nthreads);
        while (cut[t] < size && data[cut[t] - 1] != '\n') ++cut[t];
    }

    // Lines are at least 40 bytes, which bounds the blocks one hour can need.
    const size_t hour_bytes = (size / 40 / Block::CAP + nthreads + 1) * sizeof(Block);
    char* records = static_cast<char*>(mmap(nullptr, hour_bytes * HOURS, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (records == MAP_FAILED) return 1;
    for (int h = 0; h < HOURS; ++h) hour_region[h].base = records + h * hour_bytes;
    std::vector<Worker> workers(nthreads);
    for (int t = 0; t < nthreads; ++t) workers[t].id = t;
    std::vector<Name> names; // every endpoint, in byte order
    // Runs once, between the phases: global ranks for every local id.
    auto merge = [&]() noexcept {
        EndpointTable global;
        for (auto& w : workers)
            for (const Name& n : w.endpoints.names) global.intern(n.bytes, n.len);
        names = global.names;
        std::sort(names.begin(), names.end(), [](const Name& x, const Name& y) {
            return std::string_view(x.bytes, x.len) < std::string_view(y.bytes, y.len);
        });
        std::vector<u32> rank_of(names.size());
        for (u32 i = 0; i < names.size(); ++i)
            rank_of[global.intern(names[i].bytes, names[i].len)] = i;
        for (auto& w : workers)
            for (const Name& n : w.endpoints.names)
                w.rank.push_back(rank_of[global.intern(n.bytes, n.len)]);
    };
    std::barrier sync(nthreads, merge);

    std::vector<char*> text(HOURS);
    std::vector<size_t> text_len(HOURS);
    // Every hour formats into its own slice of one lazily-touched region; a
    // slice is handed back to the kernel once it has been written.
    const size_t slice = ((MAX_ENDPOINTS + 1) * 224 + 4095) & ~size_t(4095);
    char* text_region = static_cast<char*>(mmap(nullptr, slice * HOURS, PROT_READ | PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    std::vector<std::atomic<bool>> done(HOURS);
    std::atomic<int> next_hour{0};

    auto work = [&](int t) {
        workers[t].parse(data + cut[t], data + cut[t + 1]);
        sync.arrive_and_wait();

        const u32 ne = names.size();
        std::vector<u32> start(ne + 1), errors(ne), keys, tmp;
        std::vector<std::pair<u32, u32>> big; // (rank, latency), sorted
        for (int h; (h = next_hour++) < HOURS;) {
            // Gather the hour's records as rank << 20 | latency, counting all
            // three radix digits (11 + 9 bits of latency, 11 of rank) at once.
            u32 c0[2048] = {}, c1[512] = {}, c2[2049] = {};
            std::fill(errors.begin(), errors.end(), 0);
            keys.clear();
            big.clear();
            const HourRegion& region = hour_region[h];
            for (size_t off = 0; off < region.used; off += sizeof(Block)) {
                const Block* b = reinterpret_cast<const Block*>(region.base + off);
                const Worker& w = workers[b->worker];
                for (u32 i = 0; i < b->n; ++i) {
                    u32 r = b->rec[i], e = w.rank[r >> 1 & 2047];
                    u32 k = e << 20 | r >> 12;
                    keys.push_back(k);
                    ++c0[k & 2047], ++c1[k >> 11 & 511], ++c2[e];
                    errors[e] += r & 1;
                }
            }
            madvise(region.base, region.used, MADV_DONTNEED); // done with them
            for (auto& w : workers)
                for (auto [l, key] : w.big[h]) {
                    big.push_back({w.rank[key >> 1], l});
                    errors[w.rank[key >> 1]] += key & 1;
                }
            std::sort(big.begin(), big.end());
            // Three stable counting passes sort keys by (rank, latency).
            const u32 n = keys.size();
            tmp.resize(n);
            auto pass = [n](u32* from, u32* to, u32* count, u32 buckets, int shift, u32 mask) {
                for (u32 b = 0, sum = 0; b < buckets; ++b) sum += std::exchange(count[b], sum);
                for (u32 i = 0; i < n; ++i) to[count[from[i] >> shift & mask]++] = from[i];
            };
            pass(keys.data(), tmp.data(), c0, 2048, 0, 2047);
            pass(tmp.data(), keys.data(), c1, 512, 11, 511);
            for (u32 e = 0, sum = 0; e < ne; ++e) start[e] = sum, sum += c2[e];
            start[ne] = n;
            pass(keys.data(), tmp.data(), c2, ne, 20, 2047);
            // Group e: tmp[start[e] .. start[e + 1]) ascending, then its big ones.

            int day = (h + 18) / 24, hour = (h + 18) % 24;
            char label[32] = "2026-10-30T18:00:00Z ";
            label[6] = day < 2 ? '0' : '1';
            label[8] = '0' + (day < 2 ? 3 : 0);
            label[9] = '0' + (day < 2 ? day : day - 1);
            label[11] = '0' + hour / 10;
            label[12] = '0' + hour % 10;

            // A line is at most 21 + 125 + 6 * 11 bytes.
            char* buf = text_region + h * slice;
            char* o = buf;
            const auto* bg = big.data();
            for (u32 e = 0; e < ne; ++e) {
                const u32 packed = start[e + 1] - start[e];
                const auto* bend = bg;
                while (bend != big.data() + big.size() && bend->first == e) ++bend;
                const u32 n = packed + (bend - bg);
                if (n == 0) continue;
                u32 pct[3];
                for (int k = 0; k < 3; ++k) {
                    u32 r = (u32(k == 0 ? 50 : k == 1 ? 95 : 99) * n + 99) / 100 - 1;
                    pct[k] = r < packed ? tmp[start[e] + r] & (BIG - 1) : bg[r - packed].second;
                }
                bg = bend;
                memcpy(o, label, 32);
                o += 21;
                for (u32 i = 0; i < names[e].len; i += 32) memcpy(o + i, names[e].bytes + i, 32);
                o += names[e].len;
                for (u32 v : {n, pct[0], pct[1], pct[2], errors[e],
                              u32(u64(errors[e]) * 1000000 / n)}) {
                    *o++ = ' ';
                    o = put(o, v);
                }
                *o++ = '\n';
            }
            text[h] = buf;
            text_len[h] = o - buf;
            done[h].store(true, std::memory_order_release);
            done[h].notify_one();
        }
    };

    std::thread writer([&] {
        for (int h = 0; h < HOURS; ++h) {
            done[h].wait(false, std::memory_order_acquire);
            for (size_t off = 0; off < text_len[h];) {
                ssize_t w = write(out, text[h] + off, text_len[h] - off);
                if (w <= 0) _exit(1);
                off += w;
            }
            madvise(text[h], slice, MADV_DONTNEED);
        }
    });
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();
    writer.join();
    close(out);
    _exit(0);
}
