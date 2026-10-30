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

typedef struct {
  uint32_t v, w;
} Edge;

typedef struct {
  uint64_t d;
  uint32_t u;
} Item;

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

static inline void heap_push(Item *h, size_t *hn, uint64_t d, uint32_t u) {
  size_t i = (*hn)++;
  while (i > 0) {
    size_t p = (i - 1) >> 2;
    if (h[p].d <= d) break;
    h[i] = h[p];
    i = p;
  }
  h[i] = (Item){d, u};
}

static inline Item heap_pop(Item *h, size_t *hn) {
  Item top = h[0];
  Item last = h[--*hn];
  size_t i = 0, n = *hn;
  while (1) {
    size_t c = (i << 2) + 1;
    if (c >= n) break;
    size_t best = c;
    uint64_t min_d = h[c].d;
    if (c + 1 < n && h[c + 1].d < min_d) { min_d = h[c + 1].d; best = c + 1; }
    if (c + 2 < n && h[c + 2].d < min_d) { min_d = h[c + 2].d; best = c + 2; }
    if (c + 3 < n && h[c + 3].d < min_d) { min_d = h[c + 3].d; best = c + 3; }
    if (min_d >= last.d) break;
    h[i] = h[best];
    i = best;
  }
  if (n) h[i] = last;
  return top;
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
  size_t hn = 0, nt = 0;
  int64_t res = -1;
#define DIST(x) (wide ? t->d64[x] : t->d32[x])
#define SET(x, val) (wide ? (void)(t->d64[x] = (val)) : (void)(t->d32[x] = (uint32_t)(val)))
  SET(s, 0);
  t->touched[nt++] = s;
  heap_push(t->heap, &hn, 0, s);
  while (hn) {
    Item it = heap_pop(t->heap, &hn);
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
        if (hn == t->hcap)
          t->heap = realloc(t->heap, (t->hcap *= 2) * sizeof(Item));
        heap_push(t->heap, &hn, nd, v);
      }
    }
  }
  for (size_t i = 0; i < nt; i++)
    SET(t->touched[i], wide ? UINT64_MAX : INF32);
  return res;
}

static void *answer(void *arg) {
  Th *t = arg;
  for (int q; (q = __atomic_fetch_add(&next_q, 1, __ATOMIC_RELAXED)) < (int)nq;) {
    uint32_t s = qs[q], tg = qt[q];
    if (s == tg) { ans[q] = 0; continue; }
    if (!has_in[tg] || off[s] == off[s + 1]) { ans[q] = -1; continue; }
    int64_t d = search(t, s, tg, 0);
    if (d == -2) {
      if (!t->d64) {
        t->d64 = malloc(n * 8ull);
        memset(t->d64, 0xff, n * 8ull);
      }
      d = search(t, s, tg, 1);
    }
    ans[q] = d;
  }
  return NULL;
}

static void run_threads(void *(*f)(void *), int count) {
  pthread_t id[64];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 128 * 1024);
  for (int i = 0; i < count; i++)
    if (pthread_create(&id[i], &attr, f, &th[i])) die("pthread_create failed");
  pthread_attr_destroy(&attr);
  for (int i = 0; i < count; i++) pthread_join(id[i], NULL);
}

int main(void) {
  mallopt(M_ARENA_MAX, 1);
  fd = open("/data/graph.txt", O_RDONLY);
  if (fd < 0) die("cannot open /data/graph.txt");
  uint64_t m = 0;
  long body = 0;
  {
    char hdr[64];
    ssize_t k = pread(fd, hdr, sizeof hdr, 0);
    const char *p = hdr;
    for (; *p != ' '; p++) n = n * 10 + (*p - '0');
    for (p++; *p != '\n'; p++) m = m * 10 + (*p - '0');
    body = p + 1 - hdr;
  }
  struct stat st;
  if (fstat(fd, &st)) die("cannot stat /data/graph.txt");
  long size = st.st_size;

  cpu_set_t cpus;
  nth = sched_getaffinity(0, sizeof cpus, &cpus) ? 1 : CPU_COUNT(&cpus);
  nth = nth < 1 ? 1 : nth > 64 ? 64 : nth;
  th = calloc(nth, sizeof(Th));

  th[0].beg = body;
  for (int i = 1; i < nth; i++) {
    long b = body + (size - body) * i / nth;
    if (b > body) {
      char c[256];
      ssize_t k = pread(fd, c, sizeof c, b - 1);
      char *nl = k > 0 ? memchr(c, '\n', k) : NULL;
      b = nl ? b + (nl - c) : size;
    }
    th[i].beg = b < th[i - 1].beg ? th[i - 1].beg : b;
    th[i - 1].end = th[i].beg;
  }
  th[nth - 1].end = size;
  for (int i = 0; i < nth; i++) th[i].buf = malloc(BUFSZ);

  off = calloc(n + 2, 4);
  has_in = calloc(n + 1, 1);
  pass = 1;
  run_threads(scan, nth);

  for (uint32_t i = 1; i <= n; i++) off[i] += off[i - 1];
  cur = malloc((n + 1) * 4);
  memcpy(cur, off, (n + 1) * 4);
  edges = malloc((off[n] + 1ull) * sizeof(Edge));
  pass = 2;
  run_threads(scan, nth);
  free(cur);
  close(fd);
  for (int i = 0; i < nth; i++) free(th[i].buf);

  int qf = open("/data/queries.txt", O_RDONLY);
  if (qf < 0 || fstat(qf, &st)) die("cannot open /data/queries.txt");
  char *qb = malloc(st.st_size + 1);
  if (read(qf, qb, st.st_size) != st.st_size) die("read queries failed");
  close(qf);
  size_t qcap = 1024;
  qs = malloc(qcap * 4);
  qt = malloc(qcap * 4);
  for (const char *p = qb, *end = qb + st.st_size; p < end;) {
    uint32_t s = 0, tg = 0;
    for (; *p != ' '; p++) s = s * 10 + (*p - '0');
    for (p++; *p != '\n'; p++) tg = tg * 10 + (*p - '0');
    p++;
    if (nq == qcap) {
      qcap *= 2;
      qs = realloc(qs, qcap * 4);
      qt = realloc(qt, qcap * 4);
    }
    qs[nq] = s;
    qt[nq++] = tg;
  }
  free(qb);

  // Renumber nodes in breadth-first order for cache locality
  uint32_t *order = malloc(n * 4ull), *perm = malloc(n * 4ull), tail = 0;
  memset(perm, 0xff, n * 4ull);
  for (uint32_t r = 0; r < n; r++) {
    if (perm[r] != INF32) continue;
    perm[r] = tail; order[tail++] = r;
    for (uint32_t h = tail - 1; h < tail; h++)
      for (uint32_t e = off[order[h]]; e < off[order[h] + 1]; e++)
        if (perm[edges[e].v] == INF32) {
          perm[edges[e].v] = tail;
          order[tail++] = edges[e].v;
        }
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
  for (uint32_t i = 0; i < n; i++) has_in2[i] = has_in[order[i]];
  free(edges); free(off); free(has_in); free(order);
  edges = edges2; off = off2; has_in = has_in2;
  for (uint32_t i = 0; i < nq; i++) {
    qs[i] = perm[qs[i]];
    qt[i] = perm[qt[i]];
  }
  free(perm);

  malloc_trim(0);

  int q_threads = nth > 8 ? 8 : nth;
  for (int i = 0; i < q_threads; i++) {
    th[i].d32 = malloc(n * 4ull);
    memset(th[i].d32, 0xff, n * 4ull);
    th[i].tcap = 65536;
    th[i].touched = malloc(th[i].tcap * 4);
    th[i].hcap = 65536;
    th[i].heap = malloc(th[i].hcap * sizeof(Item));
  }

  ans = malloc((nq + 1) * sizeof *ans);
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
