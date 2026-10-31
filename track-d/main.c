// Speed Demon 2026, Track D (Traverse): exact shortest-path distances for each
// query in a directed graph with non-negative uint32 weights. See specs/d-traverse.md.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUFSZ (1 << 16)
#define INF32 UINT32_MAX
#define ESC 0xFFFF
#define RING 4096

typedef struct {
  uint32_t v, w;
} Edge;
typedef struct {
  uint64_t d;
  uint32_t u;
} Item;
typedef struct {
  uint32_t *v, n, cap;
} Bucket;
typedef struct {
  long beg, end;
  char *buf;
  uint32_t *d32, *touched;
  uint64_t *d64;
  Item *heap;
  size_t hcap, tcap;
  Bucket ring[RING];
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
    for (; *p != ' '; p++)
      u = u * 10 + (*p - '0');
    for (p++; *p != ' '; p++)
      v = v * 10 + (*p - '0');
    for (p++; *p != '\n'; p++)
      w = w * 10 + (*p - '0');
    p++;
    if (u == v || u >= n || v >= n)
      continue;
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
    if (k <= 0)
      die("read failed");
    o += k;
    have += k;
    char *last = memrchr(t->buf, '\n', have);
    if (!last)
      continue;
    edge_lines(t->buf, last + 1);
    have -= last + 1 - t->buf;
    memmove(t->buf, last + 1, have);
  }
  return NULL;
}

static void push(Th *t, size_t *hn, uint64_t d, uint32_t u) {
  if (*hn == t->hcap) {
    t->hcap *= 2;
    t->heap = realloc(t->heap, t->hcap * sizeof(Item));
  }
  size_t i = (*hn)++;
  for (; i && t->heap[(i - 1) / 2].d > d; i = (i - 1) / 2)
    t->heap[i] = t->heap[(i - 1) / 2];
  t->heap[i] = (Item){d, u};
}

static Item pop(Th *t, size_t *hn) {
  Item top = t->heap[0], last = t->heap[--*hn];
  size_t i = 0;
  for (size_t c; (c = 2 * i + 1) < *hn; i = c) {
    if (c + 1 < *hn && t->heap[c + 1].d < t->heap[c].d)
      c++;
    if (t->heap[c].d >= last.d)
      break;
    t->heap[i] = t->heap[c];
  }
  if (*hn)
    t->heap[i] = last;
  return top;
}

static inline void edge_at(uint32_t u, uint32_t e, uint32_t *v, uint32_t *w) {
  (void)u;
  *v = edges[e].v;
  *w = edges[e].w;
}

static void *dist_array(size_t bytes) {
  void *a = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (a == MAP_FAILED)
    die("out of memory");
  return a;
}

static int64_t search(Th *t, uint32_t s, uint32_t tg) {
  size_t hn = 0, nt = 0;
  int64_t res = -1;
  if (!t->d64)
    t->d64 = dist_array(n * 8ull);
  t->d64[s] = ~0ull;
  t->touched[nt++] = s;
  push(t, &hn, 0, s);
  while (hn) {
    Item it = pop(t, &hn);
    if (it.d > ~t->d64[it.u])
      continue;
    if (it.u == tg) {
      res = it.d;
      break;
    }
    for (uint32_t e = off[it.u]; e < off[it.u + 1]; e++) {
      uint32_t v, w;
      edge_at(it.u, e, &v, &w);
      uint64_t nd = it.d + w;
      if (nd < ~t->d64[v]) {
        if (!t->d64[v]) {
          if (nt == t->tcap)
            t->touched = realloc(t->touched, (t->tcap *= 2) * 4);
          t->touched[nt++] = v;
        }
        t->d64[v] = ~nd;
        push(t, &hn, nd, v);
      }
    }
  }
  for (size_t i = 0; i < nt; i++)
    t->d64[t->touched[i]] = 0;
  return res;
}

static inline void bpush(Bucket *b, uint32_t v) {
  if (b->n == b->cap)
    b->v = realloc(b->v, (b->cap = b->cap ? b->cap * 2 : 64) * 4ull);
  b->v[b->n++] = v;
}

static int64_t dial(Th *t, uint32_t s, uint32_t tg) {
  size_t live = 1, nt = 0;
  int64_t res = -1;
  t->d32[s] = 0;
  t->touched[nt++] = s;
  bpush(&t->ring[0], s);
  for (uint32_t d = 0; live && res == -1; d++) {
    Bucket *b = &t->ring[d & (RING - 1)];
    for (uint32_t k = 0; k < b->n && res == -1; k++) {
      uint32_t u = b->v[k];
      live--;
      if (t->d32[u] != d)
        continue;
      if (u == tg)
        res = d;
      for (uint32_t e = off[u]; e < off[u + 1] && res == -1; e++) {
        uint32_t v, w;
        edge_at(u, e, &v, &w);
        uint64_t nd = (uint64_t)d + w;
        if (w >= RING || nd >= INF32)
          res = -2;
        else if (nd < t->d32[v]) {
          if (t->d32[v] == INF32) {
            if (nt == t->tcap)
              t->touched = realloc(t->touched, (t->tcap *= 2) * 4);
            t->touched[nt++] = v;
          }
          t->d32[v] = nd;
          bpush(&t->ring[nd & (RING - 1)], v);
          live++;
        }
      }
    }
    b->n = 0;
  }
  for (int i = 0; i < RING; i++)
    t->ring[i].n = 0;
  for (size_t i = 0; i < nt; i++)
    t->d32[t->touched[i]] = INF32;
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
    else if ((ans[q] = dial(t, s, tg)) == -2)
      ans[q] = search(t, s, tg);
  }
  return NULL;
}

static void run_threads(void *(*f)(void *)) {
  pthread_t id[64];
  for (int i = 0; i < nth; i++)
    if (pthread_create(&id[i], NULL, f, &th[i]))
      die("pthread_create failed");
  for (int i = 0; i < nth; i++)
    pthread_join(id[i], NULL);
}

int main(void) {
  FILE *qf = fopen("/data/queries.txt", "r");
  if (!qf)
    die("cannot open /data/queries.txt");
  size_t qcap = 1024;
  qs = malloc(qcap * 4);
  qt = malloc(qcap * 4);
  for (uint32_t s, t; fscanf(qf, "%u %u", &s, &t) == 2; nq++) {
    if (nq == qcap) {
      qcap *= 2;
      qs = realloc(qs, qcap * 4);
      qt = realloc(qt, qcap * 4);
    }
    qs[nq] = s;
    qt[nq] = t;
  }
  fclose(qf);

  struct stat st;
  fd = open("/data/graph.txt", O_RDONLY);
  if (fd < 0 || fstat(fd, &st))
    die("cannot open /data/graph.txt");
  long size = st.st_size, body;
  {
    char c[64];
    ssize_t k = pread(fd, c, sizeof c - 1, 0);
    c[k > 0 ? k : 0] = 0;
    n = strtoul(c, NULL, 10);
    char *nl = strchr(c, '\n');
    if (!nl)
      die("bad graph.txt header");
    body = nl - c + 1;
  }
  for (uint32_t i = 0; i < nq; i++)
    if (qs[i] >= n || qt[i] >= n)
      die("query node out of range");
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
  for (int i = 0; i < nth; i++)
    th[i].buf = malloc(BUFSZ);
  pass = 1;
  run_threads(scan);
  for (uint32_t u = 0; u < n; u++)
    off[u + 1] += off[u];
  edges = malloc((off[n] + 1ull) * sizeof(Edge));
  cur = malloc((n + 1) * 4ull);
  memcpy(cur, off, (n + 1) * 4ull);
  pass = 2;
  run_threads(scan);
  free(cur);
  for (int i = 0; i < nth; i++)
    free(th[i].buf);

  ans = malloc((nq + 1) * sizeof *ans);
  run_threads(answer);
  FILE *out = fopen("/out/distances.txt", "w");
  if (!out)
    die("cannot create /out/distances.txt");
  for (uint32_t i = 0; i < nq; i++)
    fprintf(out, "%lld\n", (long long)ans[i]);
  if (fclose(out))
    die("write failed");
  return 0;
}
