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
#define MAPN 1000001
#define MAXG 256
#define IDBITS 24
#define IDMASK ((1u << IDBITS) - 1)
#define INL 24
#define MAXNAMES 105000

typedef struct {
  uint64_t count, id_sum, name_bytes, note_bytes;
  int64_t sum, min, max;
} Agg;

typedef struct {
  uint32_t len, off;
  uint64_t inl[INL / 8];
  uint64_t groups[4];
} Name;

typedef struct {
  uint64_t h, w0;
  uint32_t id, len;
} NameCache;

typedef struct {
  long beg, end;
  char *buf, *tmp;
  uint64_t distinct[MAXG];
  Agg agg[MAXG];
  NameCache cache[1024];
} Th;

static int fd, nth, ngroups, overflow;
static Th *th;
static uint8_t *region;
static char codes[MAXG][12];
static uint64_t *slots;
static uint32_t cap, nnames;
static Name *names;
static char *arena;
static uint64_t arena_used;

static void die(const char *m) {
  fprintf(stderr, "%s\n", m);
  exit(1);
}

static uint64_t hash_bytes(const char *s, uint32_t n) {
  uint64_t h = n * 0x9E3779B97F4A7C15ull, w;
  for (uint32_t i = 0; i < n; i += 8) {
    memcpy(&w, s + i, 8);
    if (n - i < 8) w &= (1ull << (8 * (n - i))) - 1;
    h = (h ^ w) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
  }
  return h | 1ull << 63;
}

static uint32_t name_id(Th *t, const char *s, uint32_t n, uint64_t h, uint64_t w0) {
  uint32_t cidx = h & 1023;
  if (t->cache[cidx].h == h && t->cache[cidx].len == n && t->cache[cidx].w0 == w0) {
    uint32_t cid = t->cache[cidx].id;
    if (n <= INL || memcmp(arena + names[cid].off, s + INL, n - INL) == 0)
      return cid;
  }
  uint64_t key = h & ~(uint64_t)IDMASK, w[INL / 8] = {w0, 0, 0};
  if (n > 8) memcpy(w + 1, s + 8, (n < INL ? n : INL) - 8);
  for (uint32_t i = h & (cap - 1);; i = (i + 1) & (cap - 1)) {
    uint64_t v = __atomic_load_n(&slots[i], __ATOMIC_ACQUIRE);
    if (v == 0) {
      if (__atomic_load_n(&nnames, __ATOMIC_RELAXED) >= MAXNAMES - 1000 || __atomic_load_n(&nnames, __ATOMIC_RELAXED) >= cap / 4 * 3) {
        overflow = 1;
        return 0;
      }
      if (__atomic_compare_exchange_n(&slots[i], &v, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE)) {
        uint32_t id = __atomic_fetch_add(&nnames, 1, __ATOMIC_RELAXED);
        Name *e = &names[id];
        *e = (Name){n, 0, {w[0], w[1], w[2]}, {0}};
        if (n > INL) {
          e->off = __atomic_fetch_add(&arena_used, n - INL, __ATOMIC_RELAXED);
          memcpy(arena + e->off, s + INL, n - INL);
        }
        __atomic_store_n(&slots[i], key | id, __ATOMIC_RELEASE);
        t->cache[cidx] = (NameCache){h, w0, id, n};
        return id;
      }
    }
    while (v == 1) v = __atomic_load_n(&slots[i], __ATOMIC_ACQUIRE);
    if ((v & ~(uint64_t)IDMASK) == key) {
      uint32_t id = v & IDMASK;
      Name *e = &names[id];
      if (e->len == n && e->inl[0] == w[0] && e->inl[1] == w[1] && e->inl[2] == w[2] &&
          (n <= INL || memcmp(arena + e->off, s + INL, n - INL) == 0)) {
        t->cache[cidx] = (NameCache){h, w0, id, n};
        return id;
      }
    }
  }
}

static int field(const char **pp, const char *end, const char **s, uint32_t *len, uint32_t *esc) {
  const char *p = *pp;
  *esc = 0;
  if (p < end && *p == '"') {
    const char *b = ++p;
    for (;;) {
      const char *q = memchr(p, '"', end - p);
      if (!q || q + 1 >= end) return 0;
      if (q[1] != '"') {
        *s = b;
        *len = q - b - *esc;
        *pp = q + 1;
        return 1;
      }
      ++*esc;
      p = q + 2;
    }
  }
  const char *q = p;
  while (q < end && *q != ',' && *q != '\r' && *q != '\n') q++;
  if (q >= end) return 0;
  *s = p;
  *len = q - p;
  *pp = q;
  return 1;
}

static const char *record(Th *t, const char *p, const char *end, int eof) {
  if (end - p >= 3 && (uint8_t)p[0] == 0xEF && (uint8_t)p[1] == 0xBB && (uint8_t)p[2] == 0xBF)
    p += 3;
  uint64_t id = 0, rid = 0;
  for (; p < end && *p != ','; p++) id = id * 10 + (*p - '0');
  const char *ns, *xs;
  uint32_t nlen, xlen, nesc, xesc;
  if (p >= end || (++p, !field(&p, end, &ns, &nlen, &nesc)) || p >= end || *p != ',') return NULL;
  if ((++p, !field(&p, end, &xs, &xlen, &xesc)) || p >= end || *p != ',') return NULL;
  for (p++; p < end && *p != ','; p++) rid = rid * 10 + (*p - '0');
  if (p >= end) return NULL;
  int neg = *++p == '-';
  int64_t cents = 0;
  for (p += neg; p < end && *p != '.'; p++) cents = cents * 10 + (*p - '0');
  if (end - p < 3 + !eof) return NULL;
  cents = cents * 100 + (p[1] - '0') * 10 + (p[2] - '0');
  cents = neg ? -cents : cents;
  p += 3;
  if (p < end && *p == '\r') p++;
  if (p < end && *p == '\n') p++;
  else if (!eof) return NULL;

  const char *name = ns;
  if (nesc) {
    char *o = t->tmp;
    for (const char *q = ns; o < t->tmp + nlen; q++) {
      *o++ = *q;
      q += *q == '"';
    }
    name = t->tmp;
  }
  uint32_t g = rid < MAPN ? region[rid] : 0;
  Agg *a = &t->agg[g];
  a->count++;
  a->id_sum += id;
  a->sum += cents;
  a->min = cents < a->min ? cents : a->min;
  a->max = cents > a->max ? cents : a->max;
  a->name_bytes += nlen;
  a->note_bytes += xlen;

  uint64_t w0 = 0;
  memcpy(&w0, name, nlen < 8 ? nlen : 8);
  uint64_t h = hash_bytes(name, nlen);
  uint32_t nid = name_id(t, name, nlen, h, w0);
  uint64_t *w = &names[nid].groups[g / 64], bit = 1ull << (g % 64);
  if (!(__atomic_load_n(w, __ATOMIC_RELAXED) & bit))
    __atomic_fetch_or(w, bit, __ATOMIC_RELAXED);
  return p;
}

static void *count_names(void *arg) {
  Th *t = arg;
  int k = t - th;
  for (uint32_t i = k; i < nnames; i += nth)
    for (int w = 0; w < 4; w++)
      for (uint64_t x = names[i].groups[w]; x; x &= x - 1)
        t->distinct[w * 64 + __builtin_ctzll(x)]++;
  return NULL;
}

static void *parse(void *arg) {
  Th *t = arg;
  for (int g = 0; g < ngroups; g++)
    t->agg[g] = (Agg){0, 0, 0, 0, 0, INT64_MAX, INT64_MIN};
  size_t have = 0;
  for (long off = t->beg;;) {
    long want = t->end - off < (long)(BUFSZ - have) ? t->end - off : (long)(BUFSZ - have);
    if (want > 0) {
      ssize_t n = pread(fd, t->buf + have, want, off);
      if (n <= 0) die("read failed");
      posix_fadvise(fd, off, n, POSIX_FADV_DONTNEED);
      off += n;
      have += n;
    }
    int eof = off >= t->end;
    const char *p = t->buf, *end = t->buf + have, *q;
    while (p < end && (q = record(t, p, end, eof))) p = q;
    if (p < end && (eof || (p == t->buf && have == BUFSZ))) die("malformed records.csv");
    have = end - p;
    memmove(t->buf, p, have);
    if (eof) return NULL;
  }
}

static void run_threads(void *(*f)(void *)) {
  pthread_t id[64];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 128 * 1024);
  for (int i = 0; i < nth; i++)
    if (pthread_create(&id[i], &attr, f, &th[i])) die("pthread_create failed");
  pthread_attr_destroy(&attr);
  for (int i = 0; i < nth; i++) pthread_join(id[i], NULL);
}

static char *money(char *o, int64_t c) {
  uint64_t a = c < 0 ? -(uint64_t)c : (uint64_t)c;
  return o + sprintf(o, "%s%llu.%02llu", c < 0 ? "-" : "", (unsigned long long)(a / 100),
                     (unsigned long long)(a % 100));
}

static int cmp_group(const void *x, const void *y) {
  return strcmp(codes[*(const int *)x], codes[*(const int *)y]);
}

int main(void) {
  mallopt(M_ARENA_MAX, 1);
  int rf = open("/data/regions.csv", O_RDONLY);
  if (rf < 0) die("cannot open /data/regions.csv");
  struct stat st;
  if (fstat(rf, &st)) die("cannot stat /data/regions.csv");
  char *rb = malloc(st.st_size + 1);
  if (read(rf, rb, st.st_size) != st.st_size) die("read regions failed");
  close(rf);

  region = calloc(MAPN, sizeof *region);
  int16_t *by_code = malloc(26 * 26 * 26 * sizeof *by_code);
  memset(by_code, 0xff, 26 * 26 * 26 * sizeof *by_code);
  ngroups = 1;
  strcpy(codes[0], "_UNMATCHED");
  char *p = memchr(rb, '\n', st.st_size), *rend = rb + st.st_size;
  for (p = p ? p + 1 : rend; p < rend;) {
    uint64_t rid = 0;
    for (; *p != ','; p++) rid = rid * 10 + (*p - '0');
    int code = ((p[1] - 'A') * 26 + (p[2] - 'A')) * 26 + (p[3] - 'A');
    if (by_code[code] < 0) {
      if (ngroups == MAXG) die("too many countries");
      memcpy(codes[ngroups], p + 1, 3);
      by_code[code] = ngroups++;
    }
    if (rid < MAPN) region[rid] = by_code[code];
    p += 5;
  }
  free(rb);
  free(by_code);

  fd = open("/data/records.csv", O_RDONLY);
  if (fd < 0 || fstat(fd, &st)) die("cannot open /data/records.csv");
  long size = st.st_size, body = 0;
  {
    char c[64];
    ssize_t n = pread(fd, c, sizeof c, 0);
    char *nl = n > 0 ? memchr(c, '\n', n) : NULL;
    body = nl ? nl - c + 1 : size;
  }
  cpu_set_t cpus;
  nth = sched_getaffinity(0, sizeof cpus, &cpus) ? 1 : CPU_COUNT(&cpus);
  nth = nth < 1 ? 1 : nth > 64 ? 64 : nth;
  th = calloc(nth, sizeof(Th));

  th[0].beg = body;
  for (int i = 1; i < nth; i++) {
    long pos = body + (size - body) * i / nth;
    for (;;) {
      unsigned char c[4096];
      ssize_t n = pread(fd, c, sizeof c, pos), k = 0;
      if (n <= 1) { pos = size; break; }
      while (k + 1 < n && !(c[k] == '\n' && ((c[k + 1] >= '0' && c[k + 1] <= '9') || c[k + 1] == 0xEF))) k++;
      pos += k + 1;
      if (k + 1 < n) break;
      pos--;
    }
    th[i].beg = pos < th[i - 1].beg ? th[i - 1].beg : pos;
  }
  for (int i = 0; i < nth; i++) {
    th[i].end = i + 1 < nth ? th[i + 1].beg : size;
    th[i].buf = malloc(BUFSZ + 64);
    th[i].tmp = malloc(BUFSZ + 64);
  }

  if (size > UINT32_MAX) die("records.csv over 4 GB");
  arena = malloc(size + 64);
  for (cap = 1 << 18;; cap *= 4) {
    if (cap > 1u << IDBITS) die("too many distinct names");
    slots = calloc(cap, 8);
    names = aligned_alloc(64, MAXNAMES * sizeof(Name));
    nnames = overflow = arena_used = 0;
    run_threads(parse);
    if (!overflow) break;
    free(slots); free(names);
  }
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  free(region);
  run_threads(count_names);
  free(names);
  free(slots);
  free(arena);

  int order[MAXG], n = 0;
  Agg tot[MAXG];
  uint64_t distinct[MAXG] = {0};
  for (int g = 0; g < ngroups; g++) {
    tot[g] = th[0].agg[g];
    distinct[g] = th[0].distinct[g];
    for (int i = 1; i < nth; i++) {
      distinct[g] += th[i].distinct[g];
      Agg *a = &th[i].agg[g];
      tot[g].count += a->count;
      tot[g].id_sum += a->id_sum;
      tot[g].sum += a->sum;
      tot[g].name_bytes += a->name_bytes;
      tot[g].note_bytes += a->note_bytes;
      tot[g].min = a->min < tot[g].min ? a->min : tot[g].min;
      tot[g].max = a->max > tot[g].max ? a->max : tot[g].max;
    }
    if (tot[g].count) order[n++] = g;
  }
  for (int i = 0; i < nth; i++) { free(th[i].buf); free(th[i].tmp); }
  free(th);

  qsort(order, n, sizeof *order, cmp_group);
  FILE *out = fopen("/out/aggregates.txt", "w");
  if (!out) die("cannot create /out/aggregates.txt");
  char line[256];
  for (int k = 0; k < n; k++) {
    Agg *a = &tot[order[k]];
    char *o = line;
    o += sprintf(o, "%s %llu %llu ", codes[order[k]], (unsigned long long)a->count,
                 (unsigned long long)a->id_sum);
    o = money(o, a->sum);
    *o++ = ' ';
    o = money(o, a->min);
    *o++ = ' ';
    o = money(o, a->max);
    sprintf(o, " %llu %llu %llu\n", (unsigned long long)distinct[order[k]],
            (unsigned long long)a->name_bytes, (unsigned long long)a->note_bytes);
    fputs(line, out);
  }
  if (fclose(out)) die("write failed");
  return 0;
}
