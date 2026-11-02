#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <fcntl.h>
#include <numeric>
#include <queue>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
// Above any path length (all are below 2^53), so INF + d cannot wrap.
constexpr u64 INF = u64(1) << 62;
// Edge index -> weight too large for its packed field, sorted by index.
using Escapes = std::vector<std::pair<size_t, u64>>;
using Bits = std::vector<u64>;
bool test(const Bits &b, u32 v) { return b[v >> 6] >> (v & 63) & 1; }
void set(Bits &b, u32 v) { b[v >> 6] |= u64(1) << (v & 63); }
unsigned id_bits(u64 n) { return n < 2 ? 1 : 64 - __builtin_clzll(n - 1); }
u32 cut(size_t n, unsigned t, unsigned threads) { return n * t / threads; }

// Uninitialized storage, committed page by page as it is written. Huge pages
// are refused, as partly used buffers would round up to them.
template <class T> T *alloc(size_t n) {
  size_t bytes = n * sizeof(T) + 64;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
  void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
  madvise(p, bytes, MADV_NOHUGEPAGE);
  return static_cast<T *>(p);
}
template <class T> void release(T *p, size_t n) { munmap(p, n * sizeof(T)); }

// Storage for n values, made of the written pages of the buffers {start,
// bytes written}, which are moved rather than faulted in afresh.
using Parts = std::vector<std::pair<void *, size_t>>;
template <class T> T *splice(const Parts &parts, size_t n) {
  size_t page = 4096, want = (n * sizeof(T) + 64 + page - 1) / page * page;
  char *p = alloc<char>(want), *at = p;
  for (auto [part, used] : parts) {
    size_t move =
        std::min<size_t>((used + page - 1) / page * page, p + want - at);
    mremap(part, move, move, MREMAP_MAYMOVE | MREMAP_FIXED, at), at += move;
  }
  return (T *)p;
}

template <class F> void parallel(unsigned threads, F f) {
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; ++t)
    pool.emplace_back(f, t);
  f(0);
  for (auto &t : pool)
    t.join();
}

// ---------------------------------------------------------------- parsing

// Bit i is set when p[i] is a separator (space or LF), the only bytes below
// '0' in the input.
u64 separators(const char *p) {
  u64 m = 0;
  for (int k = 0; k < 64; k += 16) {
    __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + k));
    u32 bits = _mm_movemask_epi8(_mm_cmplt_epi8(x, _mm_set1_epi8('0')));
    m |= u64(bits) << k;
  }
  return m;
}

// Value of the digits [a, b). Up to eight digits are converted at once from
// the 8 bytes ending at b, the bytes before a masked off.
u32 decimal(const char *a, const char *b, const char *file) {
  u32 len = b - a, v = 0;
  if (len > 8 || b - file < 8) {
    for (; a < b; ++a)
      v = v * 10 + (*a - '0');
    return v;
  }
  u64 x, m = 0x000000FF000000FF;
  memcpy(&x, b - 8, 8);
  x &= (~u64(0) << (64 - 8 * len)) & 0x0F0F0F0F0F0F0F0F;
  x = x * 10 + (x >> 8); // adjacent digit pairs
  x = (x & m) * 0x000F424000000064 + ((x >> 16) & m) * 0x0000271000000001;
  return x >> 32;
}

// Calls emit(u, v, w) for each line in [begin, end) of the file [lo, hi).
template <class Emit>
void parse(const char *lo, const char *hi, const char *begin, const char *end,
           Emit emit) {
  const char *field = begin;
  u32 val[3], k = 0;
  char pad[64];
  for (const char *q = begin; q < end; q += 64) {
    // The file's last block is scanned from a copy padded with digits.
    if (hi - q < 64)
      memset(pad, '0', 64), memcpy(pad, q, hi - q);
    u64 m = separators(hi - q < 64 ? pad : q);
    if (end - q < 64)
      m &= (u64(1) << (end - q)) - 1;
    for (; m; m &= m - 1) {
      const char *sep = q + __builtin_ctzll(m);
      val[k] = decimal(field, sep, lo), field = sep + 1;
      if (++k == 3)
        emit(val[0], val[1], val[2]), k = 0;
    }
  }
}

u32 scan(const char *&p) {
  u32 v = 0;
  while (*p >= '0')
    v = v * 10 + (*p++ - '0');
  return ++p, v;
}

// Maps a file read-only; its pages stay in the page cache.
const char *map(const std::string &path, size_t &size) {
  int fd = open(path.c_str(), O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st))
    exit(2);
  size = st.st_size;
  void *p = mmap(nullptr, size + 1, PROT_READ, MAP_PRIVATE, fd, 0);
  return static_cast<const char *>(p);
}

// ---------------------------------------------------------------- graphs

// Empty raw slot, and the first slot of a vertex with its edges spilled.
constexpr u32 NONE = ~0u, SPILL = ~1u;

// Compressed rows of packed edges, head << wbits | weight. A weight that does
// not fit leaves the field all ones and sits in the escape table.
struct Graph {
  u32 n, m, wbits, mask, *off, *adj;
  Escapes esc;
  // Allocates the rows unless given storage.
  Graph(u32 n, u32 m, u32 *off = nullptr, u32 *adj = nullptr)
      : n(n), m(m), wbits(32 - id_bits(n + 1)), mask((1u << wbits) - 1),
        off(off ? off : alloc<u32>(n + 1)), adj(adj ? adj : alloc<u32>(m)) {}
  u32 head(u32 e) const { return e >> wbits; }
  u64 weight(size_t i) const {
    if ((adj[i] & mask) != mask)
      return adj[i] & mask;
    return std::lower_bound(esc.begin(), esc.end(), std::pair(i, 0ul))->second;
  }
  void put(size_t i, u32 v, u64 w, Escapes &fresh) {
    if (w >= mask)
      fresh.push_back({i, w});
    adj[i] = v << wbits | std::min<u64>(w, mask);
  }
  void merge(std::vector<Escapes> &parts) {
    for (Escapes &e : parts)
      esc.insert(esc.end(), e.begin(), e.end());
    std::sort(esc.begin(), esc.end());
  }
};

// An edge binned by its tail's bucket: the tail's low bits with the head, and
// 16 weight bits. Heavier weights keep all ones and go to a side table.
struct __attribute__((packed)) Entry {
  u32 key;
  uint16_t w;
};

// One worker's edges grouped by their tail's bucket of ids, in chunks. The
// workers share a pool laid out by group of buckets, so a group's entries are
// released together once its rows are built, and the bins shrink while the
// graph grows. Each worker owns `cap` entries (whole pages) of each group.
struct alignas(64) Bins {
  static constexpr u32 CHUNK = 256, GROUP = 16;
  u32 bits, shift, buckets, cap;
  Entry *pool;
  std::vector<u32> fill, used;
  std::vector<std::vector<u32>> chunks;
  std::vector<std::pair<u32, u64>> heavy; // pool index -> weight
  Bins(u32 n, u32 cap, unsigned t, unsigned workers)
      : bits(id_bits(n)), shift(std::max(6u, std::min(11u, 32 - bits))),
        buckets(n >> shift), cap(cap), fill(buckets + 1), chunks(buckets + 1) {
    for (u32 g = 0; g <= buckets / GROUP; ++g)
      used.push_back((g * workers + t) * cap);
  }
  void add(u32 u, u32 v, u64 w) {
    u32 b = u >> shift;
    if (fill[b] % CHUNK == 0)
      chunks[b].push_back(fill[b] = used[b / GROUP]), used[b / GROUP] += CHUNK;
    if (w >= 0xFFFF)
      heavy.push_back({fill[b], w});
    u32 key = (u & ((1u << shift) - 1)) << bits | v;
    pool[fill[b]++] = {key, uint16_t(std::min<u64>(w, 0xFFFF))};
  }
  template <class F> void each(u32 b, F f) const {
    for (u32 c : chunks[b])
      for (u32 i = c; i < std::min(c + CHUNK, fill[b]); ++i)
        f(pool[i], i);
  }
};

// Parses the edges into per-worker bins, then builds the raw graph's rows one
// bucket of tails at a time in cache, releasing a group's bins once its rows
// exist. A row is four slots, or for more edges a pointer to spilled ones.
// Marks the vertices that can be bypassed: out-degree <= 2 and no query
// endpoint.
Graph *read_graph(const char *lo, const char *hi, const Bits &terminal,
                  Bits &bypass, unsigned threads) {
  const char *p = lo;
  u32 n = scan(p), m = scan(p);
  // Worker k parses from the first line start at or after its share's start.
  auto line = [&](unsigned k) {
    const char *x = p + (hi - p) * k / threads;
    return x == p || x == hi ? x : (const char *)memchr(x, '\n', hi - x) + 1;
  };
  size_t most = 0; // edges any one worker may receive
  for (unsigned t = 0; t < threads; ++t)
    most = std::max<size_t>(most, (line(t + 1) - line(t)) / 6);
  u32 cap = (most / 2048 + 9) * 2048;
  std::vector<Bins> bins;
  for (unsigned t = 0; t < threads; ++t)
    bins.emplace_back(n, cap, t, threads);
  const Bins &shape = bins[0];
  u32 bits = shape.bits, shift = shape.shift, buckets = shape.buckets;
  Entry *pool = alloc<Entry>(size_t(buckets / Bins::GROUP + 1) * threads * cap);
  parallel(threads, [&](unsigned t) {
    Bins &bin = bins[t];
    bin.pool = pool;
    parse(lo, hi, line(t), line(t + 1),
          [&](u32 u, u32 v, u32 w) { bin.add(u, v, w); });
    std::sort(bin.heavy.begin(), bin.heavy.end());
  });
  // Four slots per vertex, then the spill area. The map and the contracted
  // graph reuse this storage.
  u32 *rows = alloc<u32>(size_t(4) * n + m);
  Graph *g = new Graph(n, m, rows, rows);
  std::atomic<u32> spilled{0};
  std::vector<std::atomic<u32>> left(buckets / Bins::GROUP + 1);
  for (u32 b = 0; b <= buckets; ++b)
    ++left[b / Bins::GROUP]; // buckets not yet built
  std::vector<Escapes> escapes(threads);
  std::atomic<u32> next{0};
  parallel(threads, [&](unsigned t) {
    std::vector<u32> at(1u << shift);
    for (u32 b; (b = next++) <= buckets;) {
      u32 first = b << shift, size = std::min(n - first, 1u << shift);
      std::fill(at.begin(), at.end(), 0);
      for (const Bins &bin : bins)
        bin.each(b, [&](Entry e, u32) { ++at[e.key >> bits]; });
      u32 extra = 0;
      for (u32 x = 0; x < size; ++x)
        extra += at[x] > 4 ? at[x] : 0;
      u32 pos = 4 * n + spilled.fetch_add(extra);
      for (u32 x = 0; x < size; ++x) {
        u32 v = first + x, *row = rows + 4 * size_t(v);
        if (at[x] <= 2 && !test(terminal, v))
          set(bypass, v); // buckets are word aligned
        std::fill(row, row + 4, NONE);
        if (at[x] > 4)
          row[0] = SPILL, row[1] = pos, row[2] = at[x], at[x] = pos,
          pos += row[2];
        else
          at[x] = 4 * v;
      }
      for (const Bins &bin : bins)
        bin.each(b, [&](Entry e, u32 i) {
          u64 w = e.w;
          if (w == 0xFFFF)
            w = std::lower_bound(bin.heavy.begin(), bin.heavy.end(),
                                 std::pair(i, u64(0)))
                    ->second;
          g->put(at[e.key >> bits]++, e.key & ((1u << bits) - 1), w,
                 escapes[t]);
        });
      if (--left[b / Bins::GROUP] == 0)
        release(pool + size_t(b / Bins::GROUP) * threads * cap, threads * cap);
    }
  });
  g->merge(escapes);
  return g;
}

// ---------------------------------------------------------------- ordering

// The edge indices [first, end) of raw vertex u.
std::pair<u32, u32> row(const Graph &raw, u32 u) {
  const u32 *r = raw.adj + 4 * size_t(u);
  if (r[0] == SPILL)
    return {r[1], r[1] + r[2]};
  return {4 * u, 4 * u + (r[0] != NONE) + (r[1] != NONE) + (r[2] != NONE) +
                     (r[3] != NONE)};
}

// One worker's share of the search order: vertices in discovery order, the
// end of each one's out-edges, and those edges with their raw heads.
struct alignas(64) Region {
  u32 *order, *ends, *out, len = 0, edges = 0;
  Escapes esc; // keyed by position in out
};

// Grows breadth-first regions over the raw graph. Expanding a vertex follows
// each out-edge through bypassable vertices (one way on, never back) to the
// first survivor, so degree-2 contraction, locality ordering and gathering
// share one pass. The walks of a batch advance one hop per round, so their
// cache misses overlap.
struct Grower {
  static constexpr u32 BATCH = 8;
  struct Walk {
    u32 pos, root, prev, at, steps;
    u64 w;
  };
  const Graph &raw;
  const Bits &bypass;
  Bits &claimed;
  Region &reg;
  std::vector<Walk> walks;

  // The relaxed OR linearizes a claim. The initial load avoids locked
  // operations for vertices that are already claimed.
  void claim(u32 z) {
    u64 bit = u64(1) << (z & 63), *word = &claimed[z >> 6];
    u64 seen = __atomic_load_n(word, __ATOMIC_RELAXED);
    if (seen & bit)
      return;
    if (!(__atomic_fetch_or(word, bit, __ATOMIC_RELAXED) & bit))
      reg.order[reg.len++] = z;
  }
  void put(u32 pos, u32 z, u64 w) {
    if (w >= raw.mask)
      reg.esc.push_back({pos, w});
    reg.out[pos] = z << raw.wbits | std::min<u64>(w, raw.mask);
  }
  // Advances a walk by one hop; returns false once its edge is final. A
  // branch ends the walk at a survivor; a dead end or a way back to the root
  // leaves a useless self-loop.
  bool step(Walk &w) {
    auto [i, end] = row(raw, w.at);
    u32 ahead = 0, next = w.at, which = i;
    for (; i < end; ++i) {
      u32 h = raw.head(raw.adj[i]);
      if (h != w.prev && h != w.at)
        ++ahead, next = h, which = i;
    }
    if (ahead > 1) {
      next = w.at;
    } else if (ahead == 0 || next == w.root) {
      next = w.root, w.w = 0;
    } else {
      w.w += raw.weight(which), w.prev = w.at, w.at = next;
      if (test(bypass, next) && ++w.steps < 64)
        return __builtin_prefetch(&raw.adj[4 * size_t(next)]), true;
    }
    if (next != w.root)
      claim(next);
    put(w.pos, next, w.w);
    return false;
  }
  void expand(u32 u, size_t &walking) {
    for (auto [i, end] = row(raw, u); i < end; ++i) {
      u32 y = raw.head(raw.adj[i]), pos = reg.edges;
      if (y == u)
        continue;
      ++reg.edges;
      if (!test(bypass, y)) {
        claim(y), put(pos, y, raw.weight(i));
        continue;
      }
      if (walks.size() == walking)
        walks.resize(2 * walking + 64);
      walks[walking++] = {pos, u, u, y, 0, raw.weight(i)};
      __builtin_prefetch(&raw.adj[4 * size_t(y)]);
    }
  }
  // Grows a region from z unless z is already claimed. The rows of the next
  // two batches are prefetched.
  void grow(u32 z) {
    u32 lo = reg.len, hi;
    for (claim(z); lo < reg.len; lo = hi) {
      hi = std::min(reg.len, lo + BATCH);
      for (u32 h = hi; h < std::min(reg.len, hi + 2 * BATCH); ++h)
        __builtin_prefetch(&raw.adj[4 * size_t(reg.order[h])]);
      size_t walking = 0;
      for (u32 h = lo; h < hi; ++h)
        expand(reg.order[h], walking), reg.ends[h] = reg.edges;
      while (walking) {
        size_t kept = 0;
        for (size_t i = 0; i < walking; ++i)
          __builtin_prefetch(&raw.adj[4 * size_t(walks[i].at)]);
        for (size_t i = 0; i < walking; ++i)
          if (step(walks[i]))
            walks[kept++] = walks[i];
        walking = kept;
      }
    }
  }
};

// The reverse of a graph whose edges mostly stay inside slices of ids:
// slice t's worker counts the in-edges of its slice as the edges are made,
// then places them. An edge into another slice is mailed to its worker.
struct Reverse {
  Graph *r;
  std::vector<u32> cuts; // slice t is [cuts[t], cuts[t + 1])
  std::vector<std::vector<std::pair<u32, u32>>> mail; // (tail, edge), to x from
  Reverse(u32 n, u32 m, const std::vector<u32> &cuts, u32 *off)
      : r(new Graph(n, m, off, off)), cuts(cuts),
        mail(cuts.size() * cuts.size()) {}
  // Worker t made edge i from u to h.
  void count(unsigned t, u32 u, u32 h, u32 i) {
    if (h - cuts[t] < cuts[t + 1] - cuts[t])
      ++r->off[h];
    else
      mail[(std::upper_bound(cuts.begin(), cuts.end(), h) - cuts.begin() - 1) *
               cuts.size() +
           t]
          .push_back({u, i});
  }
  void place(const Graph &g, unsigned t, Escapes &esc) {
    unsigned width = cuts.size();
    u32 lo = cuts[t], hi = cuts[t + 1], pos = g.off[lo];
    // The slices before t start with their own out-edges, adjusted by mail.
    for (unsigned k = 0; k < width * width; ++k)
      pos += ((k / width < t) - (k % width < t)) * u32(mail[k].size());
    for (unsigned k = t * width; k < (t + 1) * width; ++k)
      for (auto [u, i] : mail[k])
        ++r->off[g.head(g.adj[i])];
    for (u32 v = lo; v < hi; ++v)
      r->off[v] = pos += r->off[v]; // the end of v's row, for now
    auto put = [&](u32 u, u32 i) {
      r->put(--r->off[g.head(g.adj[i])], u, g.weight(i), esc);
    };
    for (u32 u = lo; u < hi; ++u)
      for (u32 i = g.off[u]; i < g.off[u + 1]; ++i)
        if (g.head(g.adj[i]) - lo < hi - lo)
          put(u, i);
    for (unsigned k = t * width; k < (t + 1) * width; ++k)
      for (auto [u, i] : mail[k])
        put(u, i);
  }
};

// Grows the regions from the survivors of each worker's id slice, then
// concatenates them into the forward graph and renames the query endpoints.
std::pair<Graph *, Graph *> contract(Graph *raw, const Bits &bypass,
                                     unsigned threads, std::vector<u32> &ends) {
  u32 n = raw->n, m = raw->m;
  Bits claimed(bypass.size());
  std::vector<Region> regions(threads);
  parallel(threads, [&](unsigned t) {
    Region &reg = regions[t];
    reg.order = alloc<u32>(n), reg.ends = alloc<u32>(n),
    reg.out = alloc<u32>(m);
    Grower grower{*raw, bypass, claimed, reg};
    for (u32 x = cut(n, t, threads); x < cut(n, t + 1, threads); ++x)
      if (!test(bypass, x))
        grower.grow(x);
  });
  u32 wbits = raw->wbits, mask = raw->mask;
  // The raw rows' pages become the map and the forward graph.
  u32 *where = raw->adj, *spare = where + n;
  delete raw;
  // Region t's vertices and edges follow those of the regions before it.
  std::vector<u32> base(threads + 1, 0), edges(threads + 1, 0);
  for (unsigned t = 0; t < threads; ++t) {
    base[t + 1] = base[t] + regions[t].len;
    edges[t + 1] = edges[t] + regions[t].edges;
  }
  // Work is split into equal slices of new ids, as regions can differ widely
  // in size. Calls f(region, first index, end index) on slice k.
  u32 total = base[threads];
  std::vector<u32> cuts;
  for (unsigned k = 0; k <= threads; ++k)
    cuts.push_back(cut(total, k, threads));
  auto each = [&](unsigned k, auto f) {
    for (unsigned t = 0; t < threads; ++t)
      if (std::max(cuts[k], base[t]) < std::min(cuts[k + 1], base[t + 1]))
        f(t, std::max(cuts[k], base[t]) - base[t],
          std::min(cuts[k + 1], base[t + 1]) - base[t]);
  };
  // Raw id -> new id. A vertex claimed twice is named by the entry that lands
  // last, the same for every head.
  parallel(threads, [&](unsigned k) {
    each(k, [&](unsigned t, u32 lo, u32 hi) {
      for (u32 i = lo; i < hi; ++i)
        __atomic_store_n(&where[regions[t].order[i]], base[t] + i,
                         __ATOMIC_RELAXED);
    });
  });
  // The pages of the regions' orders and edges become the reverse graph.
  Parts orders, outs;
  for (Region &reg : regions) {
    std::sort(reg.esc.begin(), reg.esc.end());
    orders.push_back({reg.order, reg.len * 4ul});
    outs.push_back({reg.out, reg.edges * 4ul});
  }
  Graph *g = new Graph(total, edges[threads], spare + edges[threads], spare);
  Reverse rev(g->n, g->m, cuts, splice<u32>(orders, total + 1));
  std::vector<Escapes> escapes(threads);
  parallel(threads, [&](unsigned k) {
    std::fill(rev.r->off + cuts[k], rev.r->off + cuts[k + 1], 0);
    each(k, [&](unsigned t, u32 lo, u32 hi) {
      Region &reg = regions[t];
      u32 pos = lo ? reg.ends[lo - 1] : 0, shift = edges[t];
      auto esc = std::lower_bound(reg.esc.begin(), reg.esc.end(),
                                  std::pair(size_t(pos), u64(0)));
      for (u32 i = lo; i < hi; ++i) {
        g->off[base[t] + i] = shift + pos;
        for (; pos < reg.ends[i]; ++pos) {
          __builtin_prefetch(
              &where[reg.out[std::min(pos + 24, reg.edges - 1)] >> wbits]);
          u32 e = reg.out[pos], w = e & mask, h = where[e >> wbits];
          u64 weight = w == mask ? esc++->second : w;
          g->put(shift + pos, h, weight, escapes[k]);
          rev.count(k, base[t] + i, h, shift + pos);
        }
      }
    });
  });
  for (Region &reg : regions)
    release(reg.ends, n);
  rev.r->adj = splice<u32>(outs, g->m);
  g->off[g->n] = edges[threads];
  g->merge(escapes);
  std::vector<Escapes> back(threads);
  parallel(threads, [&](unsigned t) { rev.place(*g, t, back[t]); });
  rev.r->off[g->n] = g->m;
  rev.r->merge(back);
  for (u32 &v : ends)
    v = where[v];
  munmap(where, n * 4ul / 4096 * 4096);
  return {g, rev.r};
}

// ---------------------------------------------------------------- queries

// Monotone queue: a ring of buckets 2^shift wide; larger jumps wait in a heap.
// push always writes the entry but counts it only when `take`, so relaxations
// need no branch.
struct Queue {
  static constexpr u32 B = 2048;
  struct Bucket {
    u32 *data = nullptr, size = 0, cap = 0;
  } bucket[B];
  u64 occupied[B / 64] = {}, cur = 0, near = 0;
  // Claims are globally unique, so legal vertex ids fit in 25 bits,
  // leaving seven bits for the key remainder.
  static constexpr u32 shift = 7, node_bits = 25;
  using Far = std::pair<u64, u32>;
  std::priority_queue<Far, std::vector<Far>, std::greater<Far>> far;
  explicit Queue(u32) {}
  void clear() {
    for (u32 w = 0; w < B / 64; occupied[w++] = 0)
      for (u64 bits = occupied[w]; bits; bits &= bits - 1)
        bucket[w * 64 + __builtin_ctzll(bits)].size = 0;
    far = {}, cur = near = 0;
  }
  void push(u64 d, u32 v, bool take = true) {
    if ((d >> shift) - cur >= B) {
      if (take)
        far.push({d, v});
      return;
    }
    u32 b = (d >> shift) % B;
    Bucket &k = bucket[b];
    if (k.size == k.cap)
      k.data = (u32 *)realloc(k.data, (k.cap = std::max(16u, 2 * k.cap)) * 4);
    k.data[k.size] = v | u32(d & ((1u << shift) - 1)) << node_bits;
    k.size += take, near += take;
    occupied[b / 64] |= u64(take) << b % 64;
  }
  // Moves cur to the first nonempty bucket and returns its least possible
  // key; INF when the queue is empty.
  u64 lower_bound() {
    if (near) {
      u32 b = cur % B, w = b / 64;
      u64 bits = occupied[w] & (~u64(0) << b % 64);
      while (!bits)
        bits = occupied[w = (w + 1) % (B / 64)];
      cur += (w * 64 + __builtin_ctzll(bits) - b) % B;
    } else if (!far.empty()) {
      cur = far.top().first >> shift;
    } else {
      return INF;
    }
    for (; !far.empty() && (far.top().first >> shift) - cur < B; far.pop())
      push(far.top().first, far.top().second);
    return cur << shift;
  }
};

// Labels in pages of 128, materialized on first write and recycled by the
// next search, so memory follows the searches rather than the graph. Pages
// are handed out in order: only the deepest search's pages are ever touched.
// A blank label holds NONE: INF, or all ones when L is narrower.
template <class L> struct Labels {
  static constexpr L NONE = std::min<u64>(INF, L(-1));
  std::vector<L *> dir;
  std::vector<u32> used;
  L *store, blank[128];
  explicit Labels(u32 n) : dir(n / 128 + 1, blank), store(alloc<L>(n + 128)) {
    std::fill(blank, blank + 128, NONE);
  }
  L *at(u32 v) const { return dir[v / 128] + v % 128; }
  u64 get(u32 v) const { return *at(v); }
  // Any finite distance improves a blank label, so writes go to real pages.
  L *slot(u32 v) {
    if (dir[v / 128] == blank) {
      dir[v / 128] = std::fill_n(store + used.size() * 128, 128, NONE) - 128;
      used.push_back(v / 128);
    }
    return at(v);
  }
  void reset() {
    for (u32 page : used)
      dir[page] = blank;
    used.clear();
  }
};

// One direction with saturating labels of type L. A missing label reads
// NONE, so a meeting below NONE always represents a real directed path.
// Nonnegative weights ensure saturation cannot hide a path below NONE.
template <class L> struct Side {
  const Graph &g;
  Labels<L> dist;
  Queue queue;
  explicit Side(const Graph &g) : g(g), dist(g.n), queue(g.n) {}
  void start(u32 s) {
    dist.reset(), queue.clear();
    *dist.slot(s) = 0, queue.push(0, s);
  }
  // Settles the current bucket in FIFO order, prefetching rows and labels of
  // the entries ahead; returns the number of vertices settled. Relaxations are
  // branch-free: nd + other(v) is a real path length either way.
  template <unsigned W = 0> size_t drain_bits(const Side &other, u64 &best) {
    Queue::Bucket &bk = queue.bucket[queue.cur % Queue::B];
    u32 mask = (1u << queue.node_bits) - 1, settled = 0;
    for (u32 k = 0; k < bk.size; ++k) {
      u32 ahead = bk.data[std::min(k + 8, bk.size - 1)] & mask;
      __builtin_prefetch(&g.off[ahead]), __builtin_prefetch(dist.at(ahead));
      ahead = bk.data[std::min(k + 4, bk.size - 1)] & mask;
      __builtin_prefetch(&g.adj[g.off[ahead]]);
      u32 u = bk.data[k] & mask;
      u64 d = queue.cur << queue.shift | bk.data[k] >> queue.node_bits;
      if (dist.get(u) != d)
        continue;
      ++settled;
      for (u32 i = g.off[u]; i < g.off[u + 1]; ++i) {
        u32 v = g.adj[i] >> (W ? W : g.wbits);
        L *x = dist.slot(v);
        u32 mask = W ? (1u << W) - 1 : g.mask;
        u64 w = g.adj[i] & mask;
        u64 nd = d + (w == mask ? g.weight(i) : w), old = *x;
        *x = L(std::min(nd, old));
        queue.push(nd, v, nd < old);
        best = std::min(best, nd + other.dist.get(v));
      }
    }
    queue.near -= bk.size, bk.size = 0;
    queue.occupied[queue.cur % Queue::B / 64] &= ~(u64(1) << queue.cur % 64);
    return settled;
  }
  size_t drain(const Side &other, u64 &best) {
    return g.wbits == 11 ? drain_bits<11>(other, best)
                         : drain_bits<>(other, best);
  }
};

// An exact forward probe settles up to 256 vertices; then bidirectional
// Dijkstra balances settled counts until the two queue fronts sum to at least
// the best meeting. Returns the distance, INF when there is no path, or NONE
// or more when the distance may not fit L.
template <class L> struct Worker {
  Side<L> fwd, bwd;
  u64 solve(u32 s, u32 t) {
    if (s == t)
      return 0;
    if (fwd.g.off[s] == fwd.g.off[s + 1] || bwd.g.off[t] == bwd.g.off[t + 1])
      return INF;
    bwd.dist.reset(), fwd.start(s);
    u64 best = INF, fs = 0, bs = 0;
    while (fs < 256) {
      u64 f = fwd.queue.lower_bound(), known = fwd.dist.get(t);
      if (known < f || f == INF) // t is settled, or the forward side ran out
        return known < Labels<L>::NONE ? known : INF;
      fs += fwd.drain(bwd, best);
    }
    bwd.start(t);
    best = fwd.dist.get(t);
    for (;;) {
      u64 f = fwd.queue.lower_bound(), b = bwd.queue.lower_bound();
      if (f == INF || b == INF) // one side has seen all it can reach
        return best < Labels<L>::NONE ? best : INF;
      if (f + b >= best)
        return best;
      if (fs <= bs)
        fs += fwd.drain(bwd, best);
      else
        bs += bwd.drain(fwd, best);
    }
  }
};

int main() {
  const char *env_data = getenv("SD_DATA"), *env_out = getenv("SD_OUT");
  std::string data = env_data ? env_data : "/data";
  std::string out = env_out ? env_out : "/out";
  unsigned threads = std::min(12u, std::thread::hardware_concurrency());
  if (getenv("SD_THREADS"))
    threads = atoi(getenv("SD_THREADS"));
  threads = std::clamp(threads, 1u, 12u);
  size_t qsize, gsize;
  const char *q = map(data + "/queries.txt", qsize), *qend = q + qsize;
  std::vector<u32> ends; // s0 t0 s1 t1 ...
  while (q < qend)
    ends.push_back(scan(q));
  const char *graph = map(data + "/graph.txt", gsize), *p = graph;
  Bits terminal(scan(p) / 64 + 1), bypass(terminal.size());
  for (u32 v : ends)
    set(terminal, v);
  Graph *raw = read_graph(graph, graph + gsize, terminal, bypass, threads);
  auto [fwd, rev] = contract(raw, bypass, threads, ends);
  size_t queries = ends.size() / 2;
  std::vector<int64_t> answers(queries);
  std::vector<u32> by_source(queries);
  std::iota(by_source.begin(), by_source.end(), 0);
  std::sort(by_source.begin(), by_source.end(),
            [&](u32 a, u32 b) { return ends[2 * a] < ends[2 * b]; });
  std::atomic<size_t> next{0};
  parallel(std::min<size_t>(threads, queries + 1), [&](unsigned) {
    // Paths below 65535 are exact with 16-bit labels. Saturated or
    // unreachable results are rerun with exact 64-bit labels.
    Worker<uint16_t> narrow{Side<uint16_t>(*fwd), Side<uint16_t>(*rev)};
    Worker<u64> *wide = nullptr;
    for (size_t k; (k = next++) < queries;) {
      u32 i = by_source[k], s = ends[2 * i], t = ends[2 * i + 1];
      u64 d = narrow.solve(s, t);
      if (d >= Labels<uint16_t>::NONE) {
        if (!wide)
          wide = new Worker<u64>{Side<u64>(*fwd), Side<u64>(*rev)};
        d = wide->solve(s, t);
      }
      answers[i] = d < INF ? int64_t(d) : -1;
    }
  });
  std::string text;
  for (int64_t x : answers)
    text += std::to_string(x) + '\n';
  int fd = open((out + "/distances.txt").c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                0644);
  if (fd < 0 || write(fd, text.data(), text.size()) != ssize_t(text.size()))
    exit(2);
  exit(0); // leaves the large buffers to the kernel
}
