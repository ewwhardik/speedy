// Speed Demon 2026, Track D (Traverse): exact shortest-path distances for each
// query in a directed graph with non-negative uint32 weights. See specs/d-traverse.md.
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
  Item *items;
  uint32_t n, cap;
} Bucket;

typedef struct {
  Bucket b[65];
  uint64_t last;
  uint32_t size;
} RadixHeap;

typedef struct {
  long beg, end;
  char *buf;
  uint32_t *d32, *touched;
  uint64_t *d64;
  RadixHeap rh;
  size_t tcap;
} Th;

static int fd, nth, pass, next_q;
static Th *th;
static uint32_t n, nq, *off, *cur, *qs, *qt;
static uint8_t *has_in;
static Edge *edges;
static int64_t *ans;

static void die(const char *m) {
  fprintf(stderr, "%s\n", m);
  exit(1);
}

static inline void rh_push(RadixHeap *h, uint64_t key, uint32_t val) {
  uint64_t diff = key ^ h->last;
  int bit = diff ? 64 - __builtin_clzll(diff) : 0;
  Bucket *b = &h->b[bit];
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 16;
    b->items = realloc(b->items, b->cap * sizeof(Item));
  }
  b->items[b->n++] = (Item){key, val};
  h->size++;
}

static inline Item rh_pop(RadixHeap *h) {
  if (!h->b[0].n) {
    int i = 1;
    while (!h->b[i].n) i++;
    uint64_t min_k = UINT64_MAX;
    Bucket *src = &h->b[i];
    for (uint32_t j = 0; j < src->n; j++)
      if (src->items[j].d < min_k) min_k = src->items[j].d;
    h->last = min_k;
    for (uint32_t j = 0; j < src->n; j++) {
      uint64_t k = src->items[j].d;
      uint32_t v = src->items[j].u;
      uint64_t diff = k ^ min_k;
      int bit = diff ? 64 - __builtin_clzll(diff) : 0;
      Bucket *dst = &h->b[bit];
      if (dst->n == dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 16;
        dst->items = realloc(dst->items, dst->cap * sizeof(Item));
      }
      dst->items[dst->n++] = (Item){k, v};
    }
    src->n = 0;
  }
  h->size--;
  return h->b[0].items[--h->b[0].n];
}

static inline void rh_clear(RadixHeap *h) {
  for (int i = 0; i < 65; i++) h->b[i].n = 0;
  h->last = 0;
  h->size = 0;
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
      __atomic_fetch_add(&off[u + 1], 1, __ATOMIC_RELAXED);
      __atomic_store_n(&has_in[v], 1, __ATOMIC_RELAXED);
    } else {
      edges[__atomic_fetch_add(&cur[u], 1, __ATOMIC_RELAXED)] = (Edge){v, (uint32_t)w};
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

static int64_t search(Th *t, uint32_t s, uint32_t tg, int wide) {
  rh_clear(&t->rh);
  size_t nt = 0;
  int64_t res = -1;
#define DIST(x) (wide ? t->d64[x] : t->d32[x])
#define SET(x, val) (wide ? (void)(t->d64[x] = (val)) : (void)(t->d32[x] = (uint32_t)(val)))
  SET(s, 0);
  t->touched[nt++] = s;
  rh_push(&t->rh, 0, s);
  while (t->rh.size) {
    Item it = rh_pop(&t->rh);
    if (it.d > DIST(it.u)) continue;
    if (it.u == tg) {
      res = it.d;
      break;
    }
    for (uint32_t e = off[it.u]; e < off[it.u + 1]; e++) {
      uint32_t v = edges[e].v;
      uint64_t nd = it.d + edges[e].w, dv = DIST(v);
      if (!wide && nd >= INF32) {
        res = -2;
        break;
      }
      if (nd < dv) {
        if (dv == (wide ? UINT64_MAX : INF32)) {
          if (nt == t->tcap)
            t->touched = realloc(t->touched, (t->tcap *= 2) * 4);
          t->touched[nt++] = v;
        }
        SET(v, nd);
        rh_push(&t->rh, nd, v);
      }
    }
  }
  for (size_t i = 0; i < nt; i++)
    SET(t->touched[i], wide ? UINT64_MAX : INF32);
  return res;
}

static void *answer(void *arg) {
  Th *t = arg;
  t->d32 = malloc(n * 4ull);
  memset(t->d32, 0xff, n * 4ull);
  t->tcap = 1 << 16;
  t->touched = malloc(t->tcap * 4);
  memset(&t->rh, 0, sizeof(RadixHeap));

  for (uint32_t q; (q = __atomic_fetch_add(&next_q, 1, __ATOMIC_RELAXED)) < nq;) {
    uint32_t s = qs[q], tg = qt[q];
    if (s == tg)
      ans[q] = 0;
    else if (!has_in[tg] || off[s] == off[s + 1])
      ans[q] = -1;
    else if ((ans[q] = search(t, s, tg, 0)) == -2) {
      if (!t->d64) {
        t->d64 = malloc(n * 8ull);
        memset(t->d64, 0xff, n * 8ull);
      }
      ans[q] = search(t, s, tg, 1);
    }
  }
  free(t->d32);
  free(t->touched);
  if (t->d64) free(t->d64);
  for (int i = 0; i < 65; i++)
    if (t->rh.b[i].items) free(t->rh.b[i].items);
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

  off = calloc(n + 2, 4);
  has_in = calloc(n + 1, 1);

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

  for (uint32_t u = 0; u < n; u++)
    off[u + 1] += off[u];
  edges = malloc((off[n] + 1ull) * sizeof(Edge));
  cur = malloc((n + 1) * 4ull);
  memcpy(cur, off, (n + 1) * 4ull);

  pass = 2;
  run_threads(scan, nth);
  free(cur);
  close(fd);
  for (int i = 0; i < nth; i++) free(th[i].buf);

  // Renumber the nodes in breadth-first order for cache locality
  uint32_t *order = malloc(n * 4ull), *perm = malloc(n * 4ull), tail = 0;
  memset(perm, 0xff, n * 4ull);
  for (uint32_t r = 0; r < n; r++) {
    if (perm[r] != INF32) continue;
    perm[r] = tail, order[tail++] = r;
    for (uint32_t h = tail - 1; h < tail; h++)
      for (uint32_t e = off[order[h]]; e < off[order[h] + 1]; e++)
        if (perm[edges[e].v] == INF32)
          perm[edges[e].v] = tail, order[tail++] = edges[e].v;
  }
  uint32_t *off2 = malloc((n + 2) * 4ull);
  Edge *edges2 = malloc((off[n] + 1ull) * sizeof(Edge));
  off2[0] = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t u = order[i], k = off2[i];
    for (uint32_t e = off[u]; e < off[u + 1]; e++)
      edges2[k++] = (Edge){perm[edges[e].v], edges[e].w};
    off2[i + 1] = k;
  }
  uint8_t *has_in2 = calloc(n + 1, 1);
  for (uint32_t i = 0; i < n; i++)
    has_in2[i] = has_in[order[i]];
  free(edges); free(off); free(has_in); free(order);
  edges = edges2; off = off2; has_in = has_in2;
  for (uint32_t i = 0; i < nq; i++) {
    qs[i] = perm[qs[i]];
    qt[i] = perm[qt[i]];
  }
  free(perm);

  malloc_trim(0);

  ans = malloc((nq + 1) * sizeof *ans);
  int q_threads = nth > 4 ? 4 : nth;
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
