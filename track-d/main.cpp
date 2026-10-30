#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <malloc.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
constexpr u64 INF = ~u64(0);

const char* map_file(const char* path, size_t& size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); _exit(1); }
    size = lseek(fd, 0, SEEK_END);
    auto* p = static_cast<char*>(mmap(nullptr, size + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (size && mmap(p, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) {
        perror("mmap");
        _exit(1);
    }
    close(fd);
    return p;
}

inline u32 number(const char*& p) {
    u32 v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return v;
}

struct Edge {
    u32 v, w;
};

struct RadixHeap {
    using Item = std::pair<u64, u32>;
    std::array<std::vector<Item>, 65> buckets;
    u64 last = 0;
    size_t size = 0;

    void clear() {
        for (auto& b : buckets) b.clear();
        last = 0;
        size = 0;
    }
    bool empty() const { return size == 0; }
    void push(u64 key, u32 node) {
        buckets[std::bit_width(key ^ last)].emplace_back(key, node);
        ++size;
    }
    const Item& top() {
        if (buckets[0].empty()) {
            unsigned i = 1;
            while (buckets[i].empty()) ++i;
            last = INF;
            for (auto [key, node] : buckets[i]) last = std::min(last, key);
            for (auto item : buckets[i]) buckets[std::bit_width(item.first ^ last)].push_back(item);
            buckets[i].clear();
        }
        return buckets[0].back();
    }
    Item pop() {
        Item item = top();
        buckets[0].pop_back();
        --size;
        return item;
    }
};

class PagedDist {
public:
    static constexpr u32 PAGE = 512;
    explicit PagedDist(u32 n) : table_((n + PAGE - 1) / PAGE, nullptr) {}
    ~PagedDist() { for (auto* p : owned_) delete[] p; }

    inline u64 get(u32 v) const {
        u64* p = table_[v / PAGE];
        return p ? p[v % PAGE] : INF;
    }

    inline void set(u32 v, u64 d) {
        u64*& p = table_[v / PAGE];
        if (!p) {
            if (spare_.empty()) owned_.push_back(new u64[PAGE]), spare_.push_back(owned_.back());
            p = spare_.back();
            spare_.pop_back();
            std::fill(p, p + PAGE, INF);
            used_.push_back(v / PAGE);
        }
        p[v % PAGE] = d;
    }

    void reset() {
        for (u32 i : used_) {
            spare_.push_back(table_[i]);
            table_[i] = nullptr;
        }
        used_.clear();
    }

private:
    std::vector<u64*> table_, spare_, owned_;
    std::vector<u32> used_;
};

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
    size_t gsize, qsize;
    const char* graph_data = map_file("/data/graph.txt", gsize);
    const char* qdata = map_file("/data/queries.txt", qsize);

    std::vector<std::pair<u32, u32>> queries;
    for (const char* p = qdata; p < qdata + qsize;) {
        u32 s = number(p); ++p;
        u32 t = number(p);
        while (p < qdata + qsize && *p != '\n') ++p;
        if (p < qdata + qsize) ++p;
        queries.emplace_back(s, t);
    }

    const char* p = graph_data;
    const u32 n = number(p); ++p;
    number(p); // m
    while (p < graph_data + gsize && *p != '\n') ++p;
    if (p < graph_data + gsize) ++p;

    const char* edge_start = p;
    size_t edge_len = gsize - (edge_start - graph_data);

    std::vector<size_t> cut(nthreads + 1, edge_len);
    cut[0] = 0;
    for (int t = 1; t < nthreads; ++t) {
        size_t c = edge_len * t / nthreads;
        while (c < edge_len && edge_start[c - 1] != '\n') ++c;
        cut[t] = c;
    }

    std::vector<std::vector<u32>> f_count(nthreads), b_count(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        f_count[t].assign(n, 0);
        b_count[t].assign(n, 0);
    }

    run_threads(nthreads, [&](int t) {
        const char* q = edge_start + cut[t];
        const char* q_end = edge_start + cut[t + 1];
        auto& fc = f_count[t];
        auto& bc = b_count[t];
        while (q < q_end) {
            u32 u = number(q); ++q;
            u32 v = number(q); ++q;
            number(q); // w
            while (q < q_end && *q != '\n') ++q;
            if (q < q_end) ++q;
            if (u != v && u < n && v < n) {
                ++fc[u];
                ++bc[v];
            }
        }
    });

    std::vector<u32> fwd_head(n + 1), bwd_head(n + 1);
    std::vector<std::vector<u32>> f_cur(nthreads), b_cur(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        f_cur[t].resize(n);
        b_cur[t].resize(n);
    }

    u32 f_total = 0, b_total = 0;
    for (u32 u = 0; u < n; ++u) {
        fwd_head[u] = f_total;
        bwd_head[u] = b_total;
        for (int t = 0; t < nthreads; ++t) {
            f_cur[t][u] = f_total;
            f_total += f_count[t][u];
            b_cur[t][u] = b_total;
            b_total += b_count[t][u];
        }
    }
    fwd_head[n] = f_total;
    bwd_head[n] = b_total;

    std::vector<Edge> fwd_edges(f_total), bwd_edges(b_total);

    run_threads(nthreads, [&](int t) {
        const char* q = edge_start + cut[t];
        const char* q_end = edge_start + cut[t + 1];
        auto& fc = f_cur[t];
        auto& bc = b_cur[t];
        while (q < q_end) {
            u32 u = number(q); ++q;
            u32 v = number(q); ++q;
            u32 w = number(q);
            while (q < q_end && *q != '\n') ++q;
            if (q < q_end) ++q;
            if (u != v && u < n && v < n) {
                fwd_edges[fc[u]++] = {v, w};
                bwd_edges[bc[v]++] = {u, w};
            }
        }
    });

    std::vector<int64_t> answers(queries.size());
    std::atomic<size_t> next_q{0};

    run_threads(nthreads, [&](int) {
        RadixHeap heap_f, heap_b;
        PagedDist dist_f(n), dist_b(n);

        for (size_t q; (q = next_q.fetch_add(1, std::memory_order_relaxed)) < queries.size();) {
            auto [s, t] = queries[q];
            if (s == t) { answers[q] = 0; continue; }
            if (s >= n || t >= n || fwd_head[s] == fwd_head[s + 1] || bwd_head[t] == bwd_head[t + 1]) {
                answers[q] = -1;
                continue;
            }

            heap_f.clear(); heap_b.clear();
            dist_f.reset(); dist_b.reset();

            dist_f.set(s, 0); heap_f.push(0, s);
            dist_b.set(t, 0); heap_b.push(0, t);
            u64 best = INF;

            while (!heap_f.empty() && !heap_b.empty()) {
                while (!heap_f.empty() && heap_f.top().first != dist_f.get(heap_f.top().second)) heap_f.pop();
                while (!heap_b.empty() && heap_b.top().first != dist_b.get(heap_b.top().second)) heap_b.pop();
                if (heap_f.empty() || heap_b.empty()) break;

                if (heap_f.top().first + heap_b.top().first >= best) break;

                int dir = heap_f.top().first <= heap_b.top().first ? 0 : 1;
                if (dir == 0) {
                    auto [du, u] = heap_f.pop();
                    u64 db_u = dist_b.get(u);
                    if (db_u != INF) best = std::min(best, du + db_u);

                    for (u32 e = fwd_head[u]; e < fwd_head[u + 1]; ++e) {
                        u32 v = fwd_edges[e].v;
                        u64 nd = du + fwd_edges[e].w;
                        if (nd < dist_f.get(v)) {
                            dist_f.set(v, nd);
                            heap_f.push(nd, v);
                            u64 db_v = dist_b.get(v);
                            if (db_v != INF) best = std::min(best, nd + db_v);
                        }
                    }
                } else {
                    auto [du, u] = heap_b.pop();
                    u64 df_u = dist_f.get(u);
                    if (df_u != INF) best = std::min(best, du + df_u);

                    for (u32 e = bwd_head[u]; e < bwd_head[u + 1]; ++e) {
                        u32 v = bwd_edges[e].v;
                        u64 nd = du + bwd_edges[e].w;
                        if (nd < dist_b.get(v)) {
                            dist_b.set(v, nd);
                            heap_b.push(nd, v);
                            u64 df_v = dist_f.get(v);
                            if (df_v != INF) best = std::min(best, nd + df_v);
                        }
                    }
                }
            }
            answers[q] = best == INF ? -1 : static_cast<int64_t>(best);
        }
    });

    std::string out;
    out.reserve(queries.size() * 12);
    for (int64_t a : answers) out += std::to_string(a) + '\n';

    FILE* f = fopen("/out/distances.txt", "wb");
    if (!f) { perror("/out/distances.txt"); return 1; }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return 0;
}
