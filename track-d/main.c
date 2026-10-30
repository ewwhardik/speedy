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
  long beg, end;
  char *buf;
  uint32_t *d32, *touched;
  uint64_t *d64;
  Item *heap;
  size_t hcap, tcap;
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

static inline void push(Th *t, size_t *hn, uint64_t d, uint32_t u) {
  if (*hn == t->hcap) {
    t->hcap *= 2;
    t->heap = realloc(t->heap, t->hcap * sizeof(Item));
  }
  size_t i = (*hn)++;
  while (i) {
    size_t p = (i - 1) >> 2;
    if (t->heap[p].d <= d) break;
    t->heap[i] = t->heap[p];
    i = p;
  }
  t->heap[i] = (Item){d, u};
}

static inline Item pop(Th *t, size_t *hn) {
  Item top = t->heap[0];
  size_t n_items = --*hn;
  if (!n_items) return top;
  Item last = t->heap[n_items];
  size_t i = 0;
  while (1) {
    size_t c = (i << 2) + 1;
    if (c >= n_items) break;
    size_t min_c = c;
    uint64_t min_d = t->heap[c].d;
    if (c + 1 < n_items && t->heap[c + 1].d < min_d) { min_c = c + 1; min_d = t->heap[c + 1].d; }
    if (c + 2 < n_items && t->heap[c + 2].d < min_d) { min_c = c + 2; min_d = t->heap[c + 2].d; }
    if (c + 3 < n_items && t->heap[c + 3].d < min_d) { min_c = c + 3; min_d = t->heap[c + 3].d; }
    if (last.d <= min_d) break;
    t->heap[i] = t->heap[min_c];
    i = min_c;
  }
  t->heap[i] = last;
  return top;
}

static int64_t search(Th *t, uint32_t s, uint32_t tg, int wide) {
  size_t hn = 0, nt = 0;
  int64_t res = -1;
#define DIST(x) (wide ? t->d64[x] : t->d32[x])
#define SET(x, val) (wide ? (void)(t->d64[x] = (val)) : (void)(t->d32[x] = (uint32_t)(val)))
  SET(s, 0);
  t->touched[nt++] = s;
  push(t, &hn, 0, s);
  while (hn) {
    Item it = pop(t, &hn);
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
        hn = 0;
        break;
      }
      if (nd < dv) {
        if (dv == (wide ? UINT64_MAX : INF32)) {
          if (nt == t->tcap)
            t->touched = realloc(t->touched, (t->tcap *= 2) * 4);
          t->touched[nt++] = v;
        }
        SET(v, nd);
        push(t, &hn, nd, v);
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
  t->hcap = 1 << 16;
  t->heap = malloc(t->hcap * sizeof(Item));
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
  free(t->heap);
  if (t->d64) free(t->d64);
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
