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
#define INF64 (UINT64_MAX / 2)

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
  uint64_t *dist_f, *dist_b;
  uint32_t *touched_f, *touched_b;
  Item *heap_f, *heap_b;
  size_t hcap_f, hcap_b, tcap_f, tcap_b;
} Th;

static int fd, nth, pass, next_q;
static Th *th;
static uint32_t n, nq, *off_f, *off_b, *cur_f, *cur_b, *qs, *qt;
static Edge *edges_f, *edges_b;
static int64_t *ans;

static void die(const char *m) {
  fprintf(stderr, "%s\n", m);
  exit(1);
}

static inline void heap_push(Item **h, size_t *hn, size_t *hcap, uint64_t d, uint32_t u) {
  if (*hn == *hcap) {
    *hcap = (*hcap) ? (*hcap) * 2 : 1024;
    *h = realloc(*h, (*hcap) * sizeof(Item));
  }
  size_t i = (*hn)++;
  while (i > 0) {
    size_t p = (i - 1) >> 2;
    if ((*h)[p].d <= d) break;
    (*h)[i] = (*h)[p];
    i = p;
  }
  (*h)[i] = (Item){d, u};
}

static inline Item heap_pop(Item *h, size_t *hn) {
  Item top = h[0];
  Item last = h[--*hn];
  size_t i = 0, sz = *hn;
  while (1) {
    size_t c = (i << 2) + 1;
    if (c >= sz) break;
    size_t best = c;
    uint64_t min_d = h[c].d;
    if (c + 1 < sz && h[c + 1].d < min_d) { min_d = h[c + 1].d; best = c + 1; }
    if (c + 2 < sz && h[c + 2].d < min_d) { min_d = h[c + 2].d; best = c + 2; }
    if (c + 3 < sz && h[c + 3].d < min_d) { min_d = h[c + 3].d; best = c + 3; }
    if (min_d >= last.d) break;
    h[i] = h[best];
    i = best;
  }
  if (sz) h[i] = last;
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
      __atomic_fetch_add(&off_f[u + 1], 1, __ATOMIC_RELAXED);
      __atomic_fetch_add(&off_b[v + 1], 1, __ATOMIC_RELAXED);
    } else {
      edges_f[__atomic_fetch_add(&cur_f[u], 1, __ATOMIC_RELAXED)] = (Edge){v, (uint32_t)w};
      edges_b[__atomic_fetch_add(&cur_b[v], 1, __ATOMIC_RELAXED)] = (Edge){u, (uint32_t)w};
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

static int64_t bi_search(Th *t, uint32_t s, uint32_t tg) {
  size_t hn_f = 0, hn_b = 0, nt_f = 0, nt_b = 0;
  t->dist_f[s] = 0;
  t->touched_f[nt_f++] = s;
  heap_push(&t->heap_f, &hn_f, &t->hcap_f, 0, s);

  t->dist_b[tg] = 0;
  t->touched_b[nt_b++] = tg;
  heap_push(&t->heap_b, &hn_b, &t->hcap_b, 0, tg);

  uint64_t best = INF64;
  while (hn_f && hn_b) {
    while (hn_f && t->heap_f[0].d > t->dist_f[t->heap_f[0].u]) heap_pop(t->heap_f, &hn_f);
    while (hn_b && t->heap_b[0].d > t->dist_b[t->heap_b[0].u]) heap_pop(t->heap_b, &hn_b);
    if (!hn_f || !hn_b) break;
    if (t->heap_f[0].d + t->heap_b[0].d >= best) break;

    if (t->heap_f[0].d <= t->heap_b[0].d) {
      Item it = heap_pop(t->heap_f, &hn_f);
      uint32_t u = it.u;
      uint64_t du = it.d;
      if (t->dist_b[u] != INF64) {
        uint64_t sum = du + t->dist_b[u];
        if (sum < best) best = sum;
      }
      for (uint32_t e = off_f[u]; e < off_f[u + 1]; e++) {
        uint32_t v = edges_f[e].v;
        uint64_t nd = du + edges_f[e].w;
        if (nd < t->dist_f[v]) {
          if (t->dist_f[v] == INF64) {
            if (nt_f == t->tcap_f) t->touched_f = realloc(t->touched_f, (t->tcap_f *= 2) * 4);
            t->touched_f[nt_f++] = v;
          }
          t->dist_f[v] = nd;
          if (t->dist_b[v] != INF64) {
            uint64_t sum = nd + t->dist_b[v];
            if (sum < best) best = sum;
          }
          heap_push(&t->heap_f, &hn_f, &t->hcap_f, nd, v);
        }
      }
    } else {
      Item it = heap_pop(t->heap_b, &hn_b);
      uint32_t u = it.u;
      uint64_t du = it.d;
      if (t->dist_f[u] != INF64) {
        uint64_t sum = du + t->dist_f[u];
        if (sum < best) best = sum;
      }
      for (uint32_t e = off_b[u]; e < off_b[u + 1]; e++) {
        uint32_t v = edges_b[e].v;
        uint64_t nd = du + edges_b[e].w;
        if (nd < t->dist_b[v]) {
          if (t->dist_b[v] == INF64) {
            if (nt_b == t->tcap_b) t->touched_b = realloc(t->touched_b, (t->tcap_b *= 2) * 4);
            t->touched_b[nt_b++] = v;
          }
          t->dist_b[v] = nd;
          if (t->dist_f[v] != INF64) {
            uint64_t sum = nd + t->dist_f[v];
            if (sum < best) best = sum;
          }
          heap_push(&t->heap_b, &hn_b, &t->hcap_b, nd, v);
        }
      }
    }
  }

  for (size_t i = 0; i < nt_f; i++) t->dist_f[t->touched_f[i]] = INF64;
  for (size_t i = 0; i < nt_b; i++) t->dist_b[t->touched_b[i]] = INF64;
  return best == INF64 ? -1 : (int64_t)best;
}

static void *answer(void *arg) {
  Th *t = arg;
  for (int q; (q = __atomic_fetch_add(&next_q, 1, __ATOMIC_RELAXED)) < (int)nq;) {
    uint32_t s = qs[q], tg = qt[q];
    if (s == tg) { ans[q] = 0; continue; }
    if (off_f[s] == off_f[s + 1] || off_b[tg] == off_b[tg + 1]) { ans[q] = -1; continue; }
    ans[q] = bi_search(t, s, tg);
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

  off_f = calloc(n + 2, 4);
  off_b = calloc(n + 2, 4);
  pass = 1;
  run_threads(scan, nth);

  for (uint32_t i = 1; i <= n; i++) {
    off_f[i] += off_f[i - 1];
    off_b[i] += off_b[i - 1];
  }
  cur_f = malloc((n + 1) * 4);
  cur_b = malloc((n + 1) * 4);
  memcpy(cur_f, off_f, (n + 1) * 4);
  memcpy(cur_b, off_b, (n + 1) * 4);

  edges_f = malloc((off_f[n] + 1ull) * sizeof(Edge));
  edges_b = malloc((off_b[n] + 1ull) * sizeof(Edge));
  pass = 2;
  run_threads(scan, nth);
  free(cur_f);
  free(cur_b);
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

  malloc_trim(0);

  int q_threads = nth > 2 ? 2 : nth;
  for (int i = 0; i < q_threads; i++) {
    th[i].dist_f = malloc(n * 8ull);
    th[i].dist_b = malloc(n * 8ull);
    for (uint32_t j = 0; j < n; j++) {
      th[i].dist_f[j] = INF64;
      th[i].dist_b[j] = INF64;
    }
    th[i].tcap_f = th[i].tcap_b = 4096;
    th[i].touched_f = malloc(th[i].tcap_f * 4);
    th[i].touched_b = malloc(th[i].tcap_b * 4);
    th[i].hcap_f = th[i].hcap_b = 4096;
    th[i].heap_f = malloc(th[i].hcap_f * sizeof(Item));
    th[i].heap_b = malloc(th[i].hcap_b * sizeof(Item));
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
