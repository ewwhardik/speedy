#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <fcntl.h>
#include <memory>
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
// Real distances stay below 2^53, so INF plus any distance cannot wrap.
static constexpr u64 INF = u64(1) << 62;
using Escapes =
    std::vector<std::pair<u32, u64>>; // slot -> weight too large to pack
using Bits = std::vector<u64>;
static bool test(const Bits &bits, u32 v) {
  return bits[v >> 6] >> (v & 63) & 1;
}
[[noreturn]] static void fail(const char *what) { perror(what), std::exit(2); }
static void set(Bits &bits, u32 v) { bits[v >> 6] |= u64(1) << (v & 63); }

// Uninitialized, lazily faulted storage, so memory is committed only where it
// is written.
template <class T> struct Array {
  T *p = nullptr;
  size_t bytes = 0;
  Array() = default;
  explicit Array(size_t n, bool sparse = false) : bytes(n * sizeof(T) + 64) {
    void *q = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q == MAP_FAILED)
      fail("mmap");
    p = static_cast<T *>(q);
    // Partly used arrays would round up to whole huge pages.
    if (sparse)
      madvise(q, bytes, MADV_NOHUGEPAGE);
  }
  Array(Array &&o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
  Array &operator=(Array &&o) noexcept {
    release();
    std::swap(p, o.p);
    bytes = o.bytes;
    return *this;
  }
  ~Array() { release(); }
  void release() {
    if (p)
      munmap(p, bytes);
    p = nullptr;
  }
  T &operator[](size_t i) { return p[i]; }
  const T &operator[](size_t i) const { return p[i]; }
};

template <class F> static void parallel(unsigned threads, F &&f) {
  std::vector<std::thread> pool;
  for (unsigned i = 1; i < threads; ++i)
    pool.emplace_back(f, i);
  f(0);
  for (auto &t : pool)
    t.join();
}

static unsigned id_bits(u64 n) {
  unsigned bits = 1;
  while (bits < 32 && (u64(1) << bits) < n)
    ++bits;
  return bits;
}

static void merge(Escapes &into, std::vector<Escapes> &parts) {
  for (auto &e : parts)
    into.insert(into.end(), e.begin(), e.end());
  std::sort(into.begin(), into.end());
}

// The slice [n * t / threads, n * (t + 1) / threads) of worker t.
static u32 cut(size_t n, unsigned t, unsigned threads) {
  return u32(n * t / threads);
}

// ---------------------------------------------------------------- parsing

// Bit i is set when p[i] is a separator (space or LF), the only bytes below
// '0'.
static u64 separators(const char *p) {
  u64 m = 0;
  for (int k = 0; k < 4; ++k) {
    __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16 * k));
    u64 bits = u32(_mm_movemask_epi8(_mm_cmplt_epi8(x, _mm_set1_epi8('0'))));
    m |= bits << (16 * k);
  }
  return m;
}

// Value of the digits [a, b). Short numbers are read as the 8 bytes ending at
// b, leading bytes masked off.
static u32 decimal(const char *a, const char *b, const char *lo) {
  size_t len = size_t(b - a);
  if (len > 8 || b - lo < 8) {
    u32 v = 0;
    for (; a < b; ++a)
      v = v * 10 + u32(*a - '0');
    return v;
  }
  u64 x, m = 0x000000FF000000FFull;
  memcpy(&x, b - 8, 8);
  x &= (~u64(0) << (64 - 8 * len)) & 0x0F0F0F0F0F0F0F0Full;
  x = x * 10 + (x >> 8);
  return u32(((x & m) * 0x000F424000000064ull +
              ((x >> 16) & m) * 0x0000271000000001ull) >>
             32);
}

// Calls emit(u, v, w) for every line in [begin, end); both are line starts
// within the file [lo, hi).
template <class Emit>
static void parse_lines(const char *lo, const char *hi, const char *begin,
                        const char *end, Emit &&emit) {
  const char *field = begin;
  u32 vals[3], k = 0;
  alignas(16) char tail[64];
  for (const char *q = begin; q < end; q += 64) {
    u64 m;
    if (hi - q >= 64) {
      m = separators(q);
    } else { // the last block of the file: scan a padded copy
      memset(tail, '0', sizeof tail);
      memcpy(tail, q, size_t(hi - q));
      m = separators(tail);
    }
    if (end - q < 64)
      m &= (u64(1) << (end - q)) - 1;
    for (; m; m &= m - 1) {
      const char *sep = q + __builtin_ctzll(m);
      vals[k] = decimal(field, sep, lo);
      field = sep + 1;
      if (++k == 3) {
        emit(vals[0], vals[1], vals[2]);
        k = 0;
      }
    }
  }
}

static u32 scan(const char *&p) {
  u32 v = 0;
  while (*p >= '0')
    v = v * 10 + u32(*p++ - '0');
  ++p;
  return v;
}

// A bin entry packs the tail's low bits with the head and keeps 16 weight bits,
// so it takes 6 bytes; heavier weights store the all-ones marker and live in
// the binner's sorted side table.
struct __attribute__((packed)) Entry {
  u32 key;
  uint16_t w;
};
using Run = std::pair<const Entry *, u32>;

// One thread's edges partitioned by u >> shift into short chunk runs, so each
// bucket can be built in cache.
class alignas(64) Binner { // aligned for the same reason as Region
  static constexpr u32 CHUNK = 256;
  Array<Entry> pool_;
  size_t used_ = 0;
  std::vector<Entry *> fill_, limit_;
  std::vector<std::vector<Run>> runs_;
  std::vector<std::pair<const Entry *, u32>> heavy_;
  unsigned shift_, bits_;
  void close(size_t b) {
    if (limit_[b] && fill_[b] != limit_[b] - CHUNK)
      runs_[b].push_back(
          {limit_[b] - CHUNK, u32(fill_[b] - limit_[b] + CHUNK)});
  }

public:
  Binner(unsigned shift, unsigned bits, size_t buckets, size_t edges)
      : pool_(edges + (buckets + 1) * CHUNK, true), fill_(buckets),
        limit_(buckets), runs_(buckets), shift_(shift), bits_(bits) {}
  void add(u32 u, u32 v, u32 w) {
    size_t b = u >> shift_;
    if (fill_[b] == limit_[b]) {
      close(b);
      fill_[b] = pool_.p + used_;
      limit_[b] = fill_[b] + CHUNK;
      used_ += CHUNK;
    }
    if (w >= 0xFFFF)
      heavy_.push_back({fill_[b], w});
    *fill_[b]++ = {(u & ((u32(1) << shift_) - 1)) << bits_ | v,
                   uint16_t(std::min(w, 0xFFFFu))};
  }
  void finish() {
    for (size_t b = 0; b < runs_.size(); ++b) {
      close(b);
      limit_[b] = nullptr;
    }
    std::sort(heavy_.begin(), heavy_.end());
  }
  u32 weight(const Entry &e) const {
    if (e.w != 0xFFFF)
      return e.w;
    return std::lower_bound(heavy_.begin(), heavy_.end(),
                            std::make_pair(&e, 0u))
        ->second;
  }
  template <class F> void each(size_t b, F &&f) const {
    for (const Run &r : runs_[b])
      for (u32 i = 0; i < r.second; ++i)
        f(r.first[i]);
  }
};

// ---------------------------------------------------------------- graphs

// Adjacency in 4-slot rows of packed edges (head in the high bits, weight in
// the low bits), so visiting a vertex costs one cache miss. A vertex with more
// than four edges keeps three inline and a spill code in the fourth slot that
// locates a counted list after the rows.
struct Rows {
  static constexpr u32 EMPTY = ~0u;
  Array<u32> slot;
  Escapes escapes;
  u32 n, wbits, mask, spill;
  alignas(64) std::atomic<size_t> tail{
      0}; // list slots handed out so far, in chunks
  struct Cursor {
    size_t at = 0, end = 0;
  };
  Rows(u32 count, size_t extra, bool sparse = false)
      : slot(size_t(count) * 4 + extra, sparse), n(count) {
    wbits = 32 - id_bits(u64(n) + 2);
    while (wbits > 1 && (u64(n + 1) << wbits) + extra >= EMPTY)
      --wbits; // spill codes stay below EMPTY
    mask = (u32(1) << wbits) - 1;
    spill = (n + 1) << wbits;
  }
  u32 head(u32 e) const { return e >> wbits; }
  u64 weight(u32 e, size_t i) const {
    if ((e & mask) != mask)
      return e & mask;
    return std::lower_bound(escapes.begin(), escapes.end(),
                            std::make_pair(u32(i), u64(0)))
        ->second;
  }
  void put(size_t i, u32 v, u64 w, Escapes &fresh) {
    if (w >= mask)
      fresh.push_back({u32(i), w});
    slot[i] = v << wbits | u32(std::min<u64>(w, mask));
  }
  // Empties u's row and, for more than four edges, reserves its list from the
  // caller's chunk.
  void open(u32 u, u32 degree, Cursor &c) {
    u32 *r = &slot[size_t(u) * 4];
    r[0] = r[1] = r[2] = r[3] = EMPTY;
    if (degree <= 4)
      return;
    if (c.end - c.at < degree - 2) {
      size_t size = std::max<size_t>(1024, degree - 2);
      c.at = tail.fetch_add(size, std::memory_order_relaxed),
      c.end = c.at + size;
    }
    r[3] = spill + u32(c.at);
    slot[size_t(n) * 4 + c.at] = degree - 3;
    c.at += degree - 2;
  }
  size_t list(u32 code) const { return size_t(n) * 4 + (code - spill); }
  // Slot of u's k-th edge once the row is open; the fourth slot holds a spill
  // code only for long rows.
  size_t at(u32 u, u32 k) const {
    u32 code = slot[size_t(u) * 4 + 3];
    return k < 3 || code < spill || code == EMPTY ? size_t(u) * 4 + k
                                                  : list(code) + k - 2;
  }
  template <class F> void each(u32 u, F &&f) const {
    const u32 *r = &slot[size_t(u) * 4];
    for (unsigned k = 0; k < 4; ++k) {
      if (r[k] < spill) {
        f(r[k], size_t(u) * 4 + k);
        continue;
      }
      if (r[k] == EMPTY)
        return;
      for (size_t i = list(r[k]) + 1, e = i + slot[i - 1]; i < e; ++i)
        f(slot[i], i);
      return;
    }
  }
};

// Builds the raw rows one tail bucket at a time in cache and marks bypassable
// vertices: out-degree <= 2 and not a query endpoint.
static void fill_rows(const std::vector<Binner> &bins, unsigned shift,
                      const Bits &terminal, Rows &raw, Bits &bypass,
                      unsigned threads) {
  size_t buckets = (size_t(raw.n) + (size_t(1) << shift) - 1) >> shift;
  std::vector<Escapes> escapes(threads);
  std::atomic<size_t> next{0};
  const unsigned bits = id_bits(raw.n);
  parallel(threads, [&](unsigned t) {
    std::vector<u32> degree(size_t(1) << shift), placed(size_t(1) << shift);
    Rows::Cursor cursor;
    for (size_t b; (b = next.fetch_add(1)) < buckets;) {
      u32 lo = u32(b << shift),
          hi = u32(std::min<size_t>(raw.n, lo + (size_t(1) << shift)));
      std::fill(degree.begin(), degree.end(), 0);
      std::fill(placed.begin(), placed.end(), 0);
      for (const Binner &bin : bins)
        bin.each(b, [&](const Entry &e) { ++degree[e.key >> bits]; });
      for (u32 u = lo; u < hi; ++u) {
        raw.open(u, degree[u - lo], cursor);
        if (degree[u - lo] <= 2 && !test(terminal, u))
          set(bypass, u); // buckets are word aligned
      }
      for (const Binner &bin : bins)
        bin.each(b, [&](const Entry &e) {
          u32 x = e.key >> bits, v = e.key & ((u32(1) << bits) - 1);
          raw.put(raw.at(lo + x, placed[x]++), v, bin.weight(e), escapes[t]);
        });
    }
  });
  merge(raw.escapes, escapes);
}

// ---------------------------------------------------------------- fused
// contraction, ordering and gathering

// One worker's share of the search order: vertices in discovery order and their
// surviving out-edges. Aligned: each worker updates its region's counters
// constantly, so regions must not share cache lines.
struct alignas(64) Region {
  Array<u32> order, ends,
      out;         // out holds raw heads with weights packed as in the raw rows
  Escapes escapes; // keyed by position in out
  u32 len = 0, edges = 0;
};

// Concurrent BFS regions over the raw graph. Expanding a vertex follows each
// out-edge through bypassable vertices (one way on, not back) to the first
// survivor, so contraction, ordering and gathering share one pass. Walks of a
// batch advance one hop per round, so their cache misses overlap.
class Grower {
  static constexpr u32 BATCH = 32;
  struct Walk {
    u32 pos, root, prev, at, steps;
    u64 weight;
  };
  const Rows &raw;
  const Bits &bypass;
  Bits &claimed;
  Bits mine_;
  Region &reg;
  std::vector<Walk> walks_;
  u64 sink_ = 0;

  // A private bitmap answers most checks, since a region mostly meets its own
  // vertices. Claims race without locked instructions: a vertex claimed twice
  // leaves a copy that no forward edge reaches, as heads resolve through a
  // single map entry, so it is never on a shortest path.
  void claim(u32 z) {
    u64 bit = u64(1) << (z & 63), *word = &claimed[z >> 6];
    if (mine_[z >> 6] & bit)
      return;
    mine_[z >> 6] |= bit;
    u64 seen = __atomic_load_n(word, __ATOMIC_RELAXED);
    bool fresh = !(seen & bit);
    // Shared lines are written only by claims.
    __atomic_store_n(fresh ? word : &sink_, seen | bit, __ATOMIC_RELAXED);
    reg.order[reg.len] = z;
    reg.len += fresh;
  }
  void put(u32 pos, u32 z, u64 w) {
    if (w >= raw.mask)
      reg.escapes.push_back({pos, w});
    reg.out[pos] = z << raw.wbits | u32(std::min<u64>(w, raw.mask));
  }
  // Advances a walk one hop; returns false once its edge is final.
  bool step(Walk &w) {
    const u32 *r = &raw.slot[size_t(w.at) * 4]; // at most two edges
    u32 ahead = 0, next = w.at, which = 0;
    for (u32 k = 0; k < 2 && r[k] != Rows::EMPTY; ++k) {
      u32 h = raw.head(r[k]);
      if (h != w.prev && h != w.at)
        ++ahead, next = h, which = k;
    }
    // A branch ends the walk at a surviving vertex; a useless walk becomes a
    // self-loop.
    if (ahead > 1)
      next = w.at;
    else if (ahead == 0 || next == w.root)
      next = w.root, w.weight = 0;
    else {
      w.weight += raw.weight(r[which], size_t(w.at) * 4 + which);
      w.prev = w.at, w.at = next;
      if (test(bypass, next) && ++w.steps < 64) {
        __builtin_prefetch(&raw.slot[size_t(next) * 4]);
        return true;
      }
    }
    if (next != w.root)
      claim(next);
    put(w.pos, next, w.weight);
    return false;
  }
  void gather(u32 u, size_t &walking) {
    raw.each(u, [&](u32 e, size_t slot) {
      u32 y = raw.head(e), pos = reg.edges;
      if (y == u)
        return;
      ++reg.edges;
      u64 w = raw.weight(e, slot);
      if (!test(bypass, y))
        return claim(y), put(pos, y, w);
      if (walks_.size() == walking)
        walks_.resize(2 * walking + 64);
      walks_[walking++] = {pos, u, u, y, 0, w};
      __builtin_prefetch(&raw.slot[size_t(y) * 4]);
    });
  }

public:
  Grower(const Rows &r, const Bits &b, Bits &c, Region &g)
      : raw(r), bypass(b), claimed(c), mine_(c.size()), reg(g) {}

  // Grows a region from z unless z is already claimed.
  void grow(u32 z) {
    u32 lo = reg.len;
    claim(z);
    for (u32 hi; lo < reg.len; lo = hi) {
      hi = std::min(reg.len, lo + BATCH);
      for (u32 h = lo + 2 * BATCH; h < std::min(reg.len, hi + 2 * BATCH); ++h)
        __builtin_prefetch(&raw.slot[size_t(reg.order[h]) * 4]);
      size_t walking = 0;
      for (u32 h = lo; h < hi; ++h) {
        gather(reg.order[h], walking);
        reg.ends[h] = reg.edges;
      }
      while (walking) {
        size_t kept = 0;
        for (size_t i = 0; i < walking; ++i)
          if (step(walks_[i]))
            walks_[kept++] = walks_[i];
        walking = kept;
      }
    }
  }
};

struct Graph {
  std::unique_ptr<Rows> fwd, rev;
  size_t edges = 0;
};

struct Query {
  u32 from, to;
};

// Grows the regions, concatenates them into the forward rows and renames the
// query endpoints.
static Graph build_forward(Rows &raw, const Bits &bypass, size_t m,
                           unsigned threads, std::vector<Query> &queries) {
  u32 n = raw.n;
  unsigned lbits = id_bits(u64(n) + 1);
  Bits claimed((size_t(n) + 63) / 64);
  std::vector<Region> regions(threads);
  parallel(threads, [&](unsigned t) {
    Region &reg = regions[t];
    reg.order = Array<u32>(n, true);
    reg.ends = Array<u32>(n, true);
    reg.out = Array<u32>(m, true);
    Grower grower(raw, bypass, claimed, reg);
    // Seeds are the unclaimed survivors of this worker's raw-id slice, so idle
    // workers start new regions.
    for (u32 x = cut(n, t, threads); x < cut(n, t + 1, threads); ++x)
      if (!test(bypass, x))
        grower.grow(x);
  });
  raw.slot.release();
  Array<u32> where(n); // raw id -> region << lbits | position
  parallel(threads, [&](unsigned t) {
    // A vertex listed twice is named by whichever entry lands last,
    // consistently for every head.
    for (u32 i = 0; i < regions[t].len; ++i)
      __atomic_store_n(&where[regions[t].order[i]], t << lbits | i,
                       __ATOMIC_RELAXED);
  });
  std::vector<u32> base(threads + 1, 0);
  size_t edges = 0;
  for (unsigned t = 0; t < threads; ++t)
    base[t + 1] = base[t] + regions[t].len, edges += regions[t].edges;
  auto locate = [&](u32 r) {
    return base[r >> lbits] + (r & ((u32(1) << lbits) - 1));
  };
  Graph g;
  g.edges = edges;
  g.fwd = std::make_unique<Rows>(base[threads], edges + 1024 * threads);
  Rows &fwd = *g.fwd;
  std::vector<Escapes> escapes(threads);
  parallel(threads, [&](unsigned t) {
    Region &reg = regions[t];
    std::sort(reg.escapes.begin(), reg.escapes.end());
    auto esc = reg.escapes.begin();
    Rows::Cursor cursor;
    for (u32 i = 0, pos = 0; i < reg.len; ++i) {
      u32 id = base[t] + i;
      fwd.open(id, reg.ends[i] - pos, cursor);
      for (u32 k = 0; pos < reg.ends[i]; ++k, ++pos) {
        u32 e = reg.out[pos];
        u64 w = (e & raw.mask) == raw.mask ? (esc++)->second : e & raw.mask;
        fwd.put(fwd.at(id, k), locate(where[raw.head(e)]), w, escapes[t]);
      }
    }
  });
  merge(fwd.escapes, escapes);
  for (Query &q : queries)
    q = {locate(where[q.from]), locate(where[q.to])};
  return g;
}

// Transposes the forward rows. Each worker owns an id slice; in-edges from
// other slices go through inboxes.
static void add_reverse(Graph &g, unsigned threads) {
  const Rows &fwd = *g.fwd;
  u32 n = fwd.n;
  g.rev = std::make_unique<Rows>(n, g.edges + 1024 * threads);
  Rows &rev = *g.rev;
  struct Remote {
    u32 head, tail;
    u64 w;
  };
  std::vector<std::vector<std::vector<Remote>>> box(
      threads, std::vector<std::vector<Remote>>(threads));
  Array<u32> degree(n), placed(n);
  std::vector<Escapes> escapes(threads);
  std::atomic<unsigned> arrived{0};
  std::vector<u32> cuts(threads + 1);
  for (unsigned t = 0; t <= threads; ++t)
    cuts[t] = cut(n, t, threads);
  parallel(threads, [&](unsigned t) {
    u32 lo = cuts[t], hi = cuts[t + 1];
    memset(&degree[lo], 0, size_t(hi - lo) * 4);
    memset(&placed[lo], 0, size_t(hi - lo) * 4);
    for (u32 u = lo; u < hi; ++u)
      fwd.each(u, [&](u32 e, size_t i) {
        u32 v = fwd.head(e);
        if (v >= lo && v < hi)
          ++degree[v];
        else
          box[t]
             [std::upper_bound(cuts.begin(), cuts.end(), v) - cuts.begin() - 1]
                 .push_back({v, u, fwd.weight(e, i)});
      });
    for (arrived.fetch_add(1); arrived < threads;)
      std::this_thread::yield();
    for (unsigned s = 0; s < threads; ++s)
      for (const Remote &r : box[s][t])
        ++degree[r.head];
    Rows::Cursor cursor;
    for (u32 v = lo; v < hi; ++v)
      rev.open(v, degree[v], cursor);
    auto add = [&](u32 v, u32 u, u64 w) {
      u32 k = placed[v]++;
      rev.put(degree[v] <= 4 || k < 3
                  ? size_t(v) * 4 + k
                  : rev.list(rev.slot[size_t(v) * 4 + 3]) + k - 2,
              u, w, escapes[t]);
    };
    for (u32 u = lo; u < hi; ++u)
      fwd.each(u, [&](u32 e, size_t i) {
        u32 v = fwd.head(e);
        if (v >= lo && v < hi)
          add(v, u, fwd.weight(e, i));
      });
    for (unsigned s = 0; s < threads; ++s)
      for (const Remote &r : box[s][t])
        add(r.head, r.tail, r.w);
  });
  merge(rev.escapes, escapes);
}

// Maps a whole file read-only; the mapping lives until exit and its pages
// belong to the page cache.
static const char *map_file(const std::string &path, size_t &size) {
  int fd = open(path.c_str(), O_RDONLY);
  struct stat st {};
  if (fd < 0 || fstat(fd, &st) != 0)
    fail(path.c_str());
  size = size_t(st.st_size);
  void *q = size ? mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
  if (q == MAP_FAILED)
    fail("mmap");
  return static_cast<const char *>(q);
}

static std::vector<Query> read_queries(const std::string &path) {
  size_t size;
  const char *p = map_file(path, size);
  std::vector<Query> result;
  for (const char *end = p + size; p < end;) {
    u32 s = scan(p);
    result.push_back({s, scan(p)});
  }
  return result;
}

static Graph read_graph(const std::string &path, std::vector<Query> &queries,
                        unsigned threads) {
  size_t size;
  const char *lo = map_file(path, size), *hi = lo + size, *p = lo;
  u32 n = scan(p);
  size_t m = scan(p);
  unsigned shift = std::max(
      6u, std::min(11u, 32 - id_bits(n))); // bucket key and head share 32 bits
  size_t buckets = (size_t(n) + (size_t(1) << shift) - 1) >> shift;
  std::vector<const char *> starts(threads + 1, hi);
  starts[0] = p;
  for (unsigned i = 1; i < threads; ++i) {
    const char *start = p + size_t(hi - p) * i / threads;
    const char *q =
        static_cast<const char *>(memchr(start, '\n', size_t(hi - start)));
    starts[i] = q ? q + 1 : hi;
  }
  Bits terminal((size_t(n) + 63) / 64), bypass(terminal.size());
  for (const Query &q : queries)
    set(terminal, q.from), set(terminal, q.to);
  Rows raw(n, m + 1024 * threads,
           true); // huge pages here would raise the peak, which this phase sets
  {
    std::vector<Binner> bins;
    for (unsigned t = 0; t < threads; ++t)
      bins.emplace_back(shift, id_bits(n), buckets,
                        size_t(starts[t + 1] - starts[t]) / 6 + 1);
    parallel(threads, [&](unsigned t) {
      parse_lines(lo, hi, starts[t], starts[t + 1],
                  [&](u32 u, u32 v, u32 w) { bins[t].add(u, v, w); });
      bins[t].finish();
    });
    fill_rows(bins, shift, terminal, raw, bypass, threads);
  }
  Graph g = build_forward(raw, bypass, m, threads, queries);
  return g;
}

// ---------------------------------------------------------------- queries

// Monotone queue: width 2^shift buckets in a ring; larger jumps wait in a
// binary heap. push writes unconditionally and counts the entry only when
// `take`, so relaxations need no branch.
struct Queue {
  static constexpr unsigned B = 2048, SHIFT_MAX = 7;
  struct Bucket {
    u32 *data = nullptr;
    u32 size = 0, cap = 0;
  };
  using Far = std::pair<u64, u32>;
  std::array<Bucket, B> bucket;
  std::array<u64, B / 64> occupied{};
  std::priority_queue<Far, std::vector<Far>, std::greater<Far>> far;
  u64 cur = 0;
  size_t near = 0;
  unsigned shift, node_bits;
  explicit Queue(u32 n)
      : shift(std::min(SHIFT_MAX, 32 - id_bits(n))), node_bits(32 - shift) {}
  ~Queue() {
    for (Bucket &k : bucket)
      free(k.data);
  }
  void clear() {
    for (size_t w = 0; w < occupied.size(); ++w)
      for (u64 bits = occupied[w]; bits; bits &= bits - 1)
        bucket[w * 64 + __builtin_ctzll(bits)].size = 0;
    occupied.fill(0);
    far = {};
    cur = near = 0;
  }
  void push(u64 d, u32 v, bool take = true) {
    if ((d >> shift) - cur >= B) {
      if (take)
        far.push({d, v});
      return;
    }
    unsigned b = (d >> shift) & (B - 1);
    Bucket &k = bucket[b];
    if (k.size == k.cap) {
      k.cap = std::max(16u, k.cap * 2);
      k.data = static_cast<u32 *>(realloc(k.data, k.cap * sizeof(u32)));
      if (!k.data)
        std::exit(2);
    }
    k.data[k.size] = v | u32(d & ((u64(1) << shift) - 1)) << node_bits;
    k.size += take;
    occupied[b >> 6] |= u64(take) << (b & 63);
    near += take;
  }
  // Moves cur to the first nonempty bucket and returns its smallest possible
  // key; INF if empty.
  u64 lower_bound() {
    if (near) {
      unsigned b = cur & (B - 1), w = b >> 6;
      u64 bits = occupied[w] & (~u64(0) << (b & 63));
      while (!bits)
        bits = occupied[w = (w + 1) & (B / 64 - 1)];
      cur += (((w << 6) | __builtin_ctzll(bits)) - b) & (B - 1);
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

// Distances in fixed pages materialized on first write and recycled per query,
// so memory tracks the search.
class Distances {
  static constexpr unsigned SHIFT = 7, SIZE = 1u << SHIFT;
  std::vector<u64 *> dir_;
  std::vector<u32> used_;
  Array<u64> store_; // pages are handed out in order, so only the deepest
                     // search's pages are touched
  std::array<u64, SIZE> blank_;

public:
  explicit Distances(u32 n)
      : dir_((n >> SHIFT) + 1), store_(dir_.size() * SIZE, true) {
    blank_.fill(INF);
    std::fill(dir_.begin(), dir_.end(), blank_.data());
  }
  const u64 *at(u32 v) const { return dir_[v >> SHIFT] + (v & (SIZE - 1)); }
  u64 get(u32 v) const { return *at(v); }
  // A writable label for v; a page is materialized first, as any finite
  // distance improves a blank one.
  u64 *slot(u32 v) {
    u64 *&p = dir_[v >> SHIFT];
    if (p == blank_.data()) {
      p = &store_[used_.size() * SIZE];
      std::fill(p, p + SIZE, INF);
      used_.push_back(v >> SHIFT);
    }
    return p + (v & (SIZE - 1));
  }
  void reset() {
    for (u32 page : used_)
      dir_[page] = blank_.data();
    used_.clear();
  }
};

struct Side {
  static constexpr u32 LOOK = 4;
  const Rows &g;
  Distances dist;
  Queue queue;
  explicit Side(const Rows &c) : g(c), dist(c.n), queue(c.n) {}
  void start(u32 s) {
    dist.reset();
    queue.clear();
    *dist.slot(s) = 0;
    queue.push(0, s);
  }
  // Settles the current bucket, FIFO so entries are prefetched before
  // expansion, and returns how many vertices it expanded. Relaxations are
  // branch-free: nd + other(v) is a real path length either way.
  size_t drain(const Side &other, u64 &best) {
    Queue::Bucket &bk = queue.bucket[queue.cur & (Queue::B - 1)];
    const u64 base = queue.cur << queue.shift;
    const u32 mask = (u32(1) << queue.node_bits) - 1;
    size_t count = 0;
    for (u32 k = 0; k < bk.size; ++k) {
      if (k + LOOK < bk.size) {
        u32 ahead = bk.data[k + LOOK] & mask;
        __builtin_prefetch(&g.slot[size_t(ahead) * 4], 0, 1);
        __builtin_prefetch(dist.at(ahead), 0, 1);
      }
      u32 u = bk.data[k] & mask;
      u64 d = base | bk.data[k] >> queue.node_bits;
      if (dist.get(u) != d)
        continue;
      ++count;
      g.each(u, [&](u32 e, size_t i) {
        u32 v = g.head(e);
        u64 nd = d + g.weight(e, i), *x = dist.slot(v), old = *x;
        bool take = nd < old;
        *x = take ? nd : old;
        queue.push(nd, v, take);
        best = std::min(best, nd + other.dist.get(v));
      });
    }
    queue.near -= bk.size;
    bk.size = 0;
    unsigned b = queue.cur & (Queue::B - 1);
    queue.occupied[b >> 6] &= ~(u64(1) << (b & 63));
    return count;
  }
};

// Exact forward probe, then bidirectional Dijkstra balancing settled vertices;
// stops once f + b >= best.
struct Worker {
  static constexpr size_t PROBE = 256;
  Side fwd, bwd;
  explicit Worker(const Graph &g) : fwd(*g.fwd), bwd(*g.rev) {}
  int64_t solve(u32 s, u32 t) {
    if (s == t)
      return 0;
    if (fwd.g.slot[size_t(s) * 4] == Rows::EMPTY ||
        bwd.g.slot[size_t(t) * 4] == Rows::EMPTY)
      return -1;
    bwd.dist.reset();
    fwd.start(s);
    u64 best = INF;
    size_t fs = 0, bs = 0;
    while (fs < PROBE) {
      u64 f = fwd.queue.lower_bound(), known = fwd.dist.get(t);
      if (known < f)
        return int64_t(known);
      if (f == INF)
        return -1;
      fs += fwd.drain(bwd, best);
    }
    bwd.start(t);
    best = std::min(best, fwd.dist.get(t));
    for (;;) {
      u64 f = fwd.queue.lower_bound(), b = bwd.queue.lower_bound();
      if (f == INF || b == INF || f + b >= best)
        break;
      if (fs <= bs)
        fs += fwd.drain(bwd, best);
      else
        bs += bwd.drain(fwd, best);
    }
    return best == INF ? -1 : int64_t(best);
  }
};

int main() {
  const char *env_data = getenv("SD_DATA");
  const char *env_out = getenv("SD_OUT");
  std::string data = env_data ? env_data : "/data",
              out = env_out ? env_out : "/out";
  unsigned threads =
      std::min(12u, std::max(1u, std::thread::hardware_concurrency()));
  if (getenv("SD_THREADS"))
    threads = std::max(1, std::min(64, atoi(getenv("SD_THREADS"))));
  std::vector<Query> queries = read_queries(data + "/queries.txt");
  Graph g = read_graph(data + "/graph.txt", queries, threads);
  add_reverse(g, threads);
  std::vector<int64_t> answers(queries.size(), -1);
  std::vector<u32> order(queries.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](u32 a, u32 b) { return queries[a].from < queries[b].from; });
  std::atomic<size_t> next{0};
  parallel(std::min<size_t>(threads, std::max<size_t>(1, queries.size())),
           [&](unsigned) {
             Worker worker(g);
             for (size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) <
                            queries.size();)
               answers[order[k]] =
                   worker.solve(queries[order[k]].from, queries[order[k]].to);
           });
  std::string text;
  for (int64_t x : answers)
    text += std::to_string(x) + '\n';
  FILE *f = fopen((out + "/distances.txt").c_str(), "wb");
  if (!f || fwrite(text.data(), 1, text.size(), f) != text.size() || fclose(f))
    fail("distances");
  std::exit(0); // skips destructors of the large arrays; still flushes profiles
                // for PGO builds
}
