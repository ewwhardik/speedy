// Speed Demon 2026, Track D (Traverse): exact shortest-path distances for each
// query in a directed graph with non-negative uint32 weights. See specs/d-traverse.md.
//
// The graph is read by all threads in two passes into forward and backward CSRs.
// Self-loops are dropped. Queries run in parallel, each an exact bidirectional
// Dijkstra search with frontier balancing and 4-ary min-heaps that terminates when
// the sum of the two frontier minima reaches the best known connection.
#define _GNU_SOURCE
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUFSZ (1 << 16)
#define INF32 UINT32_MAX

typedef struct { uint32_t v, w; } Edge;
typedef struct { uint64_t d; uint32_t u; } Item;

typedef struct {
  long beg, end;
  char *buf;
  uint32_t *d32[2], *touched[2];
  uint64_t *d64[2];
  Item *heap[2];
  size_t hcap[2], tcap[2];
} Th;

static int fd, nth, pass, next_q;
static Th *th;
static uint32_t n, nq, *f_off, *b_off, *f_cur, *b_cur, *qs, *qt;
static Edge *f_edges, *b_edges;
static int64_t *ans;

static void die(const char *m) {
  fprintf(stderr, "%s\n", m);
  exit(1);
}

static void edge_lines(const char *p, const char *lim) {
  while (p < lim) {
    uint32_t u = 0, v = 0;
    uint64_t w = 0;
    for (; *p != ' '; p++) u = u * 10 + (*p - '0');
    for (p++; *p != ' '; p++) v = v * 10 + (*p - '0');
    for (p++; *p != '\n'; p++) w = w * 10 + (*p - '0');
    p++;
    if (u == v || u >= n || v >= n) continue;
    if (pass == 1) {
      __atomic_fetch_add(&f_off[u + 1], 1, __ATOMIC_RELAXED);
      __atomic_fetch_add(&b_off[v + 1], 1, __ATOMIC_RELAXED);
    } else {
      f_edges[__atomic_fetch_add(&f_cur[u], 1, __ATOMIC_RELAXED)] = (Edge){v, (uint32_t)w};
      b_edges[__atomic_fetch_add(&b_cur[v], 1, __ATOMIC_RELAXED)] = (Edge){u, (uint32_t)w};
    }
  }
}

static void *scan(void *arg) {
  Th *t = arg;
  size_t have = 0;
  for (long o = t->beg; o < t->end;) {
    long want = t->end - o < (long)(BUFSZ - have) ? t->end - o : (long)(BUFSZ - have);
    ssize_t k = pread(fd, t->buf + have, want, o);
    if (k <= 0) die("read failed");
    o += k;
    have += k;
    char *last = memrchr(t->buf, '\n', have);
    if (!last) continue;
    edge_lines(t->buf, last + 1);
    have -= last + 1 - t->buf;
    memmove(t->buf, last + 1, have);
  }
  return NULL;
}

static inline void push(Th *t, int dir, size_t *hn, uint64_t d, uint32_t u) {
  if (*hn == t->hcap[dir]) {
    t->hcap[dir] *= 2;
    t->heap[dir] = realloc(t->heap[dir], t->hcap[dir] * sizeof(Item));
  }
  size_t i = (*hn)++;
  while (i) {
    size_t p = (i - 1) >> 2;
    if (t->heap[dir][p].d <= d) break;
    t->heap[dir][i] = t->heap[dir][p];
    i = p;
  }
  t->heap[dir][i] = (Item){d, u};
}

static inline Item pop(Th *t, int dir, size_t *hn) {
  Item top = t->heap[dir][0];
  size_t n_items = --*hn;
  if (!n_items) return top;
  Item last = t->heap[dir][n_items];
  size_t i = 0;
  while (1) {
    size_t c = (i << 2) + 1;
    if (c >= n_items) break;
    size_t min_c = c;
    uint64_t min_d = t->heap[dir][c].d;
    if (c + 1 < n_items && t->heap[dir][c + 1].d < min_d) { min_c = c + 1; min_d = t->heap[dir][c + 1].d; }
    if (c + 2 < n_items && t->heap[dir][c + 2].d < min_d) { min_c = c + 2; min_d = t->heap[dir][c + 2].d; }
    if (c + 3 < n_items && t->heap[dir][c + 3].d < min_d) { min_c = c + 3; min_d = t->heap[dir][c + 3].d; }
    if (last.d <= min_d) break;
    t->heap[dir][i] = t->heap[dir][min_c];
    i = min_c;
  }
  t->heap[dir][i] = last;
  return top;
}

static int64_t search(Th *t, uint32_t s, uint32_t tg, int wide) {
  size_t tn[2] = {0, 0}, hn[2] = {0, 0};
  int64_t res = -1;
#define DIST(dir, x) (wide ? t->d64[dir][x] : (uint64_t)t->d32[dir][x])
#define SET(dir, x, val) (wide ? (void)(t->d64[dir][x] = (val)) : (void)(t->d32[dir][x] = (uint32_t)(val)))
#define INF (wide ? UINT64_MAX : (uint64_t)INF32)

  SET(0, s, 0);
  t->touched[0][tn[0]++] = s;
  push(t, 0, &hn[0], 0, s);

  SET(1, tg, 0);
  t->touched[1][tn[1]++] = tg;
  push(t, 1, &hn[1], 0, tg);

  uint64_t best = UINT64_MAX;

  while (1) {
    while (hn[0] && t->heap[0][0].d > DIST(0, t->heap[0][0].u)) pop(t, 0, &hn[0]);
    while (hn[1] && t->heap[1][0].d > DIST(1, t->heap[1][0].u)) pop(t, 1, &hn[1]);

    if (!hn[0] || !hn[1]) break;

    uint64_t min0 = t->heap[0][0].d, min1 = t->heap[1][0].d;
    if (min0 + min1 >= best) break;

    int dir = min0 <= min1 ? 0 : 1;
    Item it = pop(t, dir, &hn[dir]);

    uint64_t od_u = DIST(1 - dir, it.u);
    if (od_u != INF) {
      uint64_t cand = it.d + od_u;
      if (cand < best) best = cand;
    }

    uint32_t *start_off = dir == 0 ? f_off : b_off;
    Edge *ed = dir == 0 ? f_edges : b_edges;

    for (uint32_t e = start_off[it.u]; e < start_off[it.u + 1]; e++) {
      uint32_t v = ed[e].v;
      uint64_t nd = it.d + ed[e].w;
      if (!wide && nd >= INF32) {
        res = -2;
        goto cleanup;
      }
      if (nd < DIST(dir, v)) {
        if (DIST(dir, v) == INF) {
          if (tn[dir] == t->tcap[dir])
            t->touched[dir] = realloc(t->touched[dir], (t->tcap[dir] *= 2) * 4);
          t->touched[dir][tn[dir]++] = v;
        }
        SET(dir, v, nd);
        push(t, dir, &hn[dir], nd, v);
        uint64_t od_v = DIST(1 - dir, v);
        if (od_v != INF) {
          uint64_t cand = nd + od_v;
          if (cand < best) best = cand;
        }
      }
    }
  }

  res = (best == UINT64_MAX ? -1 : (int64_t)best);

cleanup:
  for (size_t i = 0; i < tn[0]; i++) SET(0, t->touched[0][i], INF);
  for (size_t i = 0; i < tn[1]; i++) SET(1, t->touched[1][i], INF);
  return res;
#undef DIST
#undef SET
#undef INF
}

static void *answer(void *arg) {
  Th *t = arg;
  t->d32[0] = malloc(n * 4ull);
  t->d32[1] = malloc(n * 4ull);
  memset(t->d32[0], 0xff, n * 4ull);
  memset(t->d32[1], 0xff, n * 4ull);
  t->tcap[0] = t->tcap[1] = 1 << 16;
  t->touched[0] = malloc(t->tcap[0] * 4);
  t->touched[1] = malloc(t->tcap[1] * 4);
  t->hcap[0] = t->hcap[1] = 1 << 16;
  t->heap[0] = malloc(t->hcap[0] * sizeof(Item));
  t->heap[1] = malloc(t->hcap[1] * sizeof(Item));

  for (uint32_t q; (q = __atomic_fetch_add(&next_q, 1, __ATOMIC_RELAXED)) < nq;) {
    uint32_t s = qs[q], tg = qt[q];
    if (s == tg)
      ans[q] = 0;
    else if (f_off[s] == f_off[s + 1] || b_off[tg] == b_off[tg + 1])
      ans[q] = -1;
    else if ((ans[q] = search(t, s, tg, 0)) == -2) {
      if (!t->d64[0]) {
        t->d64[0] = malloc(n * 8ull);
        t->d64[1] = malloc(n * 8ull);
        memset(t->d64[0], 0xff, n * 8ull);
        memset(t->d64[1], 0xff, n * 8ull);
      }
      ans[q] = search(t, s, tg, 1);
    }
  }
  free(t->d32[0]); free(t->d32[1]);
  free(t->touched[0]); free(t->touched[1]);
  free(t->heap[0]); free(t->heap[1]);
  if (t->d64[0]) { free(t->d64[0]); free(t->d64[1]); }
  return NULL;
}

static void run_threads(void *(*f)(void *), int count) {
  pthread_t id[64];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 64 * 1024);
  for (int i = 0; i < count; i++)
    if (pthread_create(&id[i], &attr, f, &th[i])) die("pthread_create failed");
  pthread_attr_destroy(&attr);
  for (int i = 0; i < count; i++) pthread_join(id[i], NULL);
}

int main(void) {
  mallopt(M_ARENA_MAX, 1);
  FILE *qf = fopen("/data/queries.txt", "r");
  if (!qf) die("cannot open /data/queries.txt");
  size_t qcap = 1024;
  qs = malloc(qcap * 4), qt = malloc(qcap * 4);
  for (uint32_t s, t; fscanf(qf, "%u %u", &s, &t) == 2; nq++) {
    if (nq == qcap) {
      qcap *= 2;
      qs = realloc(qs, qcap * 4), qt = realloc(qt, qcap * 4);
    }
    qs[nq] = s, qt[nq] = t;
  }
  fclose(qf);

  struct stat st;
  fd = open("/data/graph.txt", O_RDONLY);
  if (fd < 0 || fstat(fd, &st)) die("cannot open /data/graph.txt");
  long size = st.st_size, body;
  {
    char c[64];
    ssize_t k = pread(fd, c, sizeof c - 1, 0);
    c[k > 0 ? k : 0] = 0;
    n = strtoul(c, NULL, 10);
    char *nl = strchr(c, '\n');
    if (!nl) die("bad graph.txt header");
    body = nl - c + 1;
  }
  for (uint32_t i = 0; i < nq; i++)
    if (qs[i] >= n || qt[i] >= n) die("query node out of range");

  f_off = calloc(n + 2, 4);
  b_off = calloc(n + 2, 4);

  cpu_set_t cpus;
  nth = sched_getaffinity(0, sizeof cpus, &cpus) ? 1 : CPU_COUNT(&cpus);
  nth = nth < 1 ? 1 : nth > 64 ? 64 : nth;
  th = calloc(nth, sizeof(Th));
  th[0].beg = body;
  for (int i = 1; i < nth; i++) {
    long b = body + (size - body) * i / nth;
    char c[64];
    ssize_t k = pread(fd, c, sizeof c, b - 1);
    char *nl = k > 0 ? memchr(c, '\n', k) : NULL;
    b = nl ? b + (nl - c) : size;
    th[i].beg = b < th[i - 1].beg ? th[i - 1].beg : b;
    th[i - 1].end = th[i].beg;
  }
  th[nth - 1].end = size;
  for (int i = 0; i < nth; i++) th[i].buf = malloc(BUFSZ);

  pass = 1;
  run_threads(scan, nth);

  for (uint32_t u = 0; u < n; u++) {
    f_off[u + 1] += f_off[u];
    b_off[u + 1] += b_off[u];
  }
  f_edges = malloc((f_off[n] + 1ull) * sizeof(Edge));
  b_edges = malloc((b_off[n] + 1ull) * sizeof(Edge));
  f_cur = malloc((n + 1) * 4ull);
  b_cur = malloc((n + 1) * 4ull);
  memcpy(f_cur, f_off, (n + 1) * 4ull);
  memcpy(b_cur, b_off, (n + 1) * 4ull);

  pass = 2;
  run_threads(scan, nth);
  free(f_cur);
  free(b_cur);
  close(fd);
  for (int i = 0; i < nth; i++) free(th[i].buf);

  malloc_trim(0);

  ans = malloc((nq + 1) * sizeof *ans);
  int q_threads = nth > 6 ? 6 : nth;
  run_threads(answer, q_threads);

  FILE *out = fopen("/out/distances.txt", "w");
  if (!out) die("cannot create /out/distances.txt");
  char outbuf[65536];
  setvbuf(out, outbuf, _IOFBF, sizeof outbuf);
  for (uint32_t i = 0; i < nq; i++)
    fprintf(out, "%lld\n", (long long)ans[i]);
  if (fclose(out)) die("write failed");
  return 0;
}
