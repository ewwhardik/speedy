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
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

struct Edge { u32 node, weight; };

struct MappedFile {
    const char* data = nullptr;
    size_t size = 0;
    explicit MappedFile(const char* path) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); _exit(1); }
        struct stat st {};
        if (fstat(fd, &st) != 0) { close(fd); _exit(1); }
        size = static_cast<size_t>(st.st_size);
        if (size) {
            void* p = mmap(nullptr, size + 4096, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p == MAP_FAILED) { perror("mmap"); _exit(1); }
            data = static_cast<const char*>(p);
        }
        close(fd);
    }
    ~MappedFile() {
        if (size && data) munmap(const_cast<char*>(data), size + 4096);
    }
    const char* end() const { return data + size; }
};

inline u64 integer(const char*& p, const char* end) {
    while (p < end && (*p < '0' || *p > '9')) ++p;
    u64 v = 0;
    while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return v;
}

struct Graph {
    std::vector<u32> offset[2];
    std::vector<Edge> edges[2];
    u32 n = 0;
    explicit Graph(const char* path) {
        MappedFile file(path);
        const char* p = file.data;
        const char* end = file.end();
        n = integer(p, end);
        u32 m = integer(p, end);
        const char* start = p;
        for (auto& off : offset) off.resize(size_t(n) + 1);
        for (u32 i = 0; i < m; ++i) {
            u32 u = integer(p, end);
            u32 v = integer(p, end);
            integer(p, end);
            if (u != v && u < n && v < n) {
                ++offset[0][u + 1];
                ++offset[1][v + 1];
            }
        }
        for (int dir = 0; dir < 2; ++dir) {
            std::partial_sum(offset[dir].begin(), offset[dir].end(), offset[dir].begin());
            edges[dir].resize(offset[dir][n]);
        }
        auto cur_f = offset[0], cur_b = offset[1];
        p = start;
        for (u32 i = 0; i < m; ++i) {
            u32 u = integer(p, end);
            u32 v = integer(p, end);
            u32 w = integer(p, end);
            if (u != v && u < n && v < n) {
                edges[0][cur_f[u]++] = {v, w};
                edges[1][cur_b[v]++] = {u, w};
            }
        }
    }
};

struct RadixHeap {
    using Item = std::pair<u64, u32>;
    std::array<std::vector<Item>, 65> buckets;
    u64 last = 0;
    size_t size = 0;
    void clear() { for (auto& b : buckets) b.clear(); last = 0; size = 0; }
    bool empty() const { return size == 0; }
    void push(u64 key, u32 node) {
        buckets[std::bit_width(key ^ last)].emplace_back(key, node);
        ++size;
    }
    const Item& top() {
        if (buckets[0].empty()) {
            unsigned i = 1;
            while (buckets[i].empty()) ++i;
            last = UINT64_MAX;
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

struct Search {
    std::vector<u64> distance[2];
    std::vector<u32> stamp[2];
    RadixHeap heap[2];
    static constexpr u64 INF = UINT64_MAX / 2;
    explicit Search(u32 n) {
        for (int d = 0; d < 2; ++d) {
            distance[d].resize(n);
            stamp[d].resize(n);
        }
    }
    int64_t run(const Graph& graph, u32 s, u32 t, u32 epoch) {
        if (s == t) return 0;
        if (s >= graph.n || t >= graph.n ||
            graph.offset[0][s] == graph.offset[0][s + 1] ||
            graph.offset[1][t] == graph.offset[1][t + 1])
            return -1;
        heap[0].clear(); heap[1].clear();
        distance[0][s] = distance[1][t] = 0;
        stamp[0][s] = stamp[1][t] = epoch;
        heap[0].push(0, s); heap[1].push(0, t);
        u64 best = INF;
        for (;;) {
            for (int d = 0; d < 2; ++d)
                while (!heap[d].empty() && heap[d].top().first != distance[d][heap[d].top().second])
                    heap[d].pop();
            if (heap[0].empty() || heap[1].empty()) break;
            if (heap[0].top().first + heap[1].top().first >= best) break;
            int dir = heap[0].top().first <= heap[1].top().first ? 0 : 1;
            auto [du, u] = heap[dir].pop();
            if (stamp[1 - dir][u] == epoch) best = std::min(best, du + distance[1 - dir][u]);
            for (u32 i = graph.offset[dir][u]; i < graph.offset[dir][u + 1]; ++i) {
                auto [v, w] = graph.edges[dir][i];
                u64 nd = du + w;
                if (stamp[dir][v] != epoch || nd < distance[dir][v]) {
                    stamp[dir][v] = epoch;
                    distance[dir][v] = nd;
                    heap[dir].push(nd, v);
                    if (stamp[1 - dir][v] == epoch) best = std::min(best, nd + distance[1 - dir][v]);
                }
            }
        }
        return best == INF ? -1 : static_cast<int64_t>(best);
    }
};

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    mallopt(M_MMAP_THRESHOLD, 1 << 16);
    Graph graph("/data/graph.txt");
    std::vector<std::pair<u32, u32>> queries;
    {
        MappedFile qfile("/data/queries.txt");
        const char* p = qfile.data;
        const char* end = qfile.end();
        while (p < end) {
            while (p < end && (*p < '0' || *p > '9')) ++p;
            if (p >= end) break;
            u32 s = integer(p, end);
            u32 t = integer(p, end);
            queries.emplace_back(s, t);
        }
    }
    std::vector<int64_t> answers(queries.size());
    std::atomic<size_t> next{0};
    const unsigned workers = std::clamp<unsigned>(std::thread::hardware_concurrency(), 1, 6);
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < workers; ++i) pool.emplace_back([&] {
        Search search(graph.n);
        for (size_t q; (q = next.fetch_add(1, std::memory_order_relaxed)) < queries.size();) {
            answers[q] = search.run(graph, queries[q].first, queries[q].second, static_cast<u32>(q + 1));
        }
    });
    for (auto& th : pool) th.join();

    std::string out;
    out.reserve(queries.size() * 12);
    char buf[32];
    for (int64_t a : answers) {
        if (a < 0) {
            out += "-1\n";
        } else {
            char* b = buf + 31;
            *b = '\n';
            u64 x = static_cast<u64>(a);
            do { *--b = '0' + (x % 10); x /= 10; } while (x);
            out.append(b, (buf + 32) - b);
        }
    }
    FILE* f = fopen("/out/distances.txt", "wb");
    if (!f) { perror("/out/distances.txt"); return 1; }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return 0;
}