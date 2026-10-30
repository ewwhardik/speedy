// Speed Demon 2026, Track B (Index): BOOL (strict left-to-right AND/OR/NOT) and
// PHRASE queries over a one-document-per-line corpus. See specs/b-index.md.
//
// The queries are read first and their terms interned; only those terms are
// indexed. Threads scan disjoint line ranges of docs.txt and look each token
// up. A term starts as a sorted id list: its matches go to a per-thread stream
// that is later scattered into one posting array (sorted by construction,
// because each thread owns a contiguous run of ids). Once a term turns out to
// be frequent it is promoted to a shared bitmap and set directly.
// BOOL queries combine lists and bitmaps left to right. PHRASE candidates come
// from the rarest term and are confirmed by re-tokenizing the document; the
// few phrases whose terms are all frequent share one extra scan instead.
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUFSZ (1 << 18) // line token buffer slack
#define OUTSZ (1 << 16)
#define MAXT 8         // terms per query
#define NONE UINT32_MAX

typedef struct {
  uint64_t w[4]; // the term's bytes, zero padded; terms are at most 32 bytes
  uint32_t len;
} Term;
typedef struct {
  uint64_t w0; // first 8 bytes, so most compares never touch the Term
  uint32_t len, id1; // id1 = term id + 1, 0 when empty
} Slot;
typedef struct {
  uint8_t phrase, n;
  char op[MAXT];
  uint32_t t[MAXT];
  int32_t heavy; // index into the shared-scan phrases, or -1
} Query;
typedef struct {
  uint32_t count, m, ids[10];
  uint64_t sum;
} Result;
typedef struct {
  uint32_t stamp, cnt; // last doc + 1 that counted this term; then the write cursor
} Tc;
typedef struct {
  long beg, end;
  Tc *tc;
  uint32_t *la, *lb, *tokids, first, wlo, whi; // first doc id; its, the last's word
  uint64_t *bits;
} Th;
typedef struct {
  const uint32_t *l; // sorted ids l[0..n) when b is NULL
  const uint64_t *b; // otherwise a bitmap of W words
  size_t n;
} Set;

static Term *terms;
static Slot *slots;
static uint32_t nterms, tcap, smask;
static Query *qs;
static Result *res;
static uint32_t nq, N, W, *df, *pstart, *post, *dlen, *hq, lcap, nheavy;
static int32_t *head, *hnext; // shared-scan phrases by first term
static uint64_t *doff, **bm, *hres;
static int fd, nth, next_q;
static const char *docs_data;
static Th *th;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void die(const char *m) {
  fprintf(stderr, "%s\n", m);
  exit(1);
}

static uint32_t hash_words(const uint64_t *w, uint32_t len) {
  uint64_t h = len * 0x9E3779B97F4A7C15ull;
  for (uint32_t i = 0; i < (len + 7) / 8; i++)
    h = (h ^ w[i]) * 0xFF51AFD7ED558CCDull;
  return (uint32_t)(h >> 32);
}

// The id of the zero-padded token w, or NONE when it is not a query term.
static uint32_t lookup(const uint64_t *w, uint32_t len) {
  if (len > 32)
    return NONE;
  for (uint32_t i = hash_words(w, len) & smask;; i = (i + 1) & smask) {
    const Slot *s = &slots[i];
    if (!s->id1)
      return NONE;
    if (s->w0 == w[0] && s->len == len) {
      const uint64_t *tw = terms[s->id1 - 1].w;
      if (len <= 8 || (tw[1] == w[1] && tw[2] == w[2] && tw[3] == w[3]))
        return s->id1 - 1;
    }
  }
}

static void place(uint32_t id) {
  Term *t = &terms[id];
  uint32_t i = hash_words(t->w, t->len) & smask;
  while (slots[i].id1)
    i = (i + 1) & smask;
  slots[i] = (Slot){t->w[0], t->len, id + 1};
}

static uint32_t intern(const char *s, uint32_t len) {
  if (len > 32)
    die("query term longer than 32 bytes");
  if (nterms == tcap) { // grow first, so the table exists before the first lookup
    tcap = tcap ? tcap * 2 : 1024;
    terms = realloc(terms, tcap * sizeof(Term));
    free(slots);
    smask = tcap * 2 - 1;
    slots = calloc(smask + 1, sizeof(Slot));
    for (uint32_t i = 0; i < nterms; i++)
      place(i);
  }
  Term t = {{0}, len};
  memcpy(t.w, s, len);
  uint32_t id = lookup(t.w, len);
  if (id != NONE)
    return id;
  terms[nterms] = t;
  place(nterms);
  return nterms++;
}

// Bit i is set when p[i] is a token byte: A-Z, a-z or 0-9 (c | 0x20 folds A-Z
// onto a-z and leaves a-z and 0-9 alone).
static uint32_t token_mask(const uint8_t *p) {
  __m256i v = _mm256_loadu_si256((const __m256i *)p);
  __m256i a = _mm256_sub_epi8(_mm256_or_si256(v, _mm256_set1_epi8(0x20)), _mm256_set1_epi8('a'));
  __m256i d = _mm256_sub_epi8(v, _mm256_set1_epi8('0'));
  __m256i al = _mm256_cmpeq_epi8(_mm256_min_epu8(a, _mm256_set1_epi8(25)), a);
  __m256i dg = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
  return _mm256_movemask_epi8(_mm256_or_si256(al, dg));
}

// The term id of the token p[0..n), or NONE.
static uint32_t id_of(const uint8_t *p, uint32_t n) {
  uint64_t w[4];
  memcpy(w, p, 32);
  for (int i = 0; i < 4; i++) { // lowercase (every byte is a letter or a digit), zero pad
    int k = (int)n - 8 * i;
    w[i] = k >= 8 ? w[i] | 0x2020202020202020ull
           : k > 0 ? (w[i] | 0x2020202020202020ull) & ((1ull << (8 * k)) - 1)
                   : 0;
  }
  return lookup(w, n);
}

// The term ids (NONE for other tokens) of the text [s, e), 64 bytes at a time:
// a token starts where the mask rises and ends where it falls. Reads up to 64
// bytes past e, so every tokenized buffer has 64 bytes of slack.
static uint32_t tokenize(const uint8_t *s, const uint8_t *e, uint32_t *ids) {
  uint32_t n = 0;
  uint64_t in = 0; // 1 when a token runs into the next block
  const uint8_t *start = s;
  for (const uint8_t *p = s; p < e; p += 64) {
    uint64_t m = token_mask(p) | (uint64_t)token_mask(p + 32) << 32;
    if (e - p < 64)
      m &= (1ull << (e - p)) - 1;
    uint64_t prev = m << 1 | in;
    in = m >> 63;
    for (uint64_t ev = m ^ prev; ev; ev &= ev - 1) {
      const uint8_t *q = p + __builtin_ctzll(ev);
      if (m >> (q - p) & 1)
        start = q;
      else
        ids[n++] = id_of(start, q - start);
    }
  }
  if (in)
    ids[n++] = id_of(start, e - start);
  return n;
}

static int bit(const uint64_t *b, uint32_t d) { return b[d >> 6] >> (d & 63) & 1; }

// Only the words holding a thread's first and last document can be shared with
// a neighbouring thread; every other word of a shared bitmap is this thread's.
static void set_bit(Th *t, uint64_t *b, uint32_t d) {
  uint32_t w = d >> 6;
  if (w == t->wlo || w == t->whi)
    __atomic_fetch_or(&b[w], 1ull << (d & 63), __ATOMIC_RELAXED);
  else
    b[w] |= 1ull << (d & 63);
}

// Pass 1: count term document frequencies and record doc offsets.
static void count_lines(Th *t, const char *p, const char *lim) {
  while (p < lim) {
    uint32_t doc = 0;
    for (; *p != '\t'; p++)
      doc = doc * 10 + (*p - '0');
    const uint8_t *s = (const uint8_t *)++p;
    const char *e = memchr(p, '\n', lim - p);
    doff[doc] = p - docs_data;
    dlen[doc] = (const uint8_t *)e - s;
    uint32_t n;
    if (doc == N - 1) {
      char tmp[64 * 1024];
      size_t len = (const uint8_t *)e - s;
      memcpy(tmp, s, len);
      memset(tmp + len, 0, 64);
      n = tokenize((const uint8_t *)tmp, (const uint8_t *)tmp + len, t->tokids);
    } else {
      n = tokenize(s, (const uint8_t *)e, t->tokids);
    }
    for (uint32_t i = 0; i < n; i++) {
      uint32_t id = t->tokids[i];
      if (id == NONE || t->tc[id].stamp == doc + 1)
        continue;
      t->tc[id].stamp = doc + 1;
      t->tc[id].cnt++;
    }
    p = e + 1;
  }
}

// Pass 2: populate bitmaps and contiguous posting array.
static void fill_lines(Th *t, const char *p, const char *lim) {
  while (p < lim) {
    uint32_t doc = 0;
    for (; *p != '\t'; p++)
      doc = doc * 10 + (*p - '0');
    const uint8_t *s = (const uint8_t *)++p;
    const char *e = memchr(p, '\n', lim - p);
    uint32_t n;
    if (doc == N - 1) {
      char tmp[64 * 1024];
      size_t len = (const uint8_t *)e - s;
      memcpy(tmp, s, len);
      memset(tmp + len, 0, 64);
      n = tokenize((const uint8_t *)tmp, (const uint8_t *)tmp + len, t->tokids);
    } else {
      n = tokenize(s, (const uint8_t *)e, t->tokids);
    }
    for (uint32_t i = 0; i < n; i++) {
      uint32_t id = t->tokids[i];
      if (id == NONE || t->tc[id].stamp == doc + 1)
        continue;
      t->tc[id].stamp = doc + 1;
      if (bm[id])
        set_bit(t, bm[id], doc);
      else
        post[t->tc[id].cnt++] = doc;
    }
    p = e + 1;
  }
}

// The shared scan: every phrase whose terms are all frequent, at every position.
static void heavy_lines(Th *t, const char *p, const char *lim) {
  while (p < lim) {
    uint32_t doc = 0;
    for (; *p != '\t'; p++)
      doc = doc * 10 + (*p - '0');
    const uint8_t *s = (const uint8_t *)++p;
    const char *e = memchr(p, '\n', lim - p);
    uint32_t n;
    if (doc == N - 1) {
      char tmp[64 * 1024];
      size_t len = (const uint8_t *)e - s;
      memcpy(tmp, s, len);
      memset(tmp + len, 0, 64);
      n = tokenize((const uint8_t *)tmp, (const uint8_t *)tmp + len, t->tokids);
    } else {
      n = tokenize(s, (const uint8_t *)e, t->tokids);
    }
    for (uint32_t i = 0; i < n; i++)
      for (int32_t h = t->tokids[i] == NONE ? -1 : head[t->tokids[i]]; h >= 0; h = hnext[h]) {
        const Query *q = &qs[hq[h]];
        uint64_t *r = hres + (size_t)h * W;
        uint32_t k = 1;
        if (i + q->n > n || bit(r, doc))
          continue;
        while (k < q->n && t->tokids[i + k] == q->t[k])
          k++;
        if (k == q->n)
          set_bit(t, r, doc);
      }
    p = e + 1;
  }
}

static void *pass1_count(void *arg) {
  Th *t = arg;
  count_lines(t, docs_data + t->beg, docs_data + t->end);
  return NULL;
}

static void *pass2_fill(void *arg) {
  Th *t = arg;
  fill_lines(t, docs_data + t->beg, docs_data + t->end);
  return NULL;
}

static void *pass_heavy(void *arg) {
  Th *t = arg;
  heavy_lines(t, docs_data + t->beg, docs_data + t->end);
  return NULL;
}

// First index i >= lo with a[i] >= x, galloping from lo.
static size_t gallop(const uint32_t *a, size_t lo, size_t n, uint32_t x) {
  size_t step = 1, hi = lo;
  while (hi < n && a[hi] < x) {
    lo = hi + 1;
    hi += step;
    step <<= 1;
  }
  if (hi > n)
    hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (a[mid] < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

static int has(Set s, uint32_t d) {
  if (s.b)
    return bit(s.b, d);
  size_t i = gallop(s.l, 0, s.n, d);
  return i < s.n && s.l[i] == d;
}

static Set term_set(uint32_t t) {
  return bm[t] ? (Set){NULL, bm[t], df[t]} : (Set){post + pstart[t], NULL, df[t]};
}

// One left-to-right step: r op s, into this thread's buffers.
static Set apply(Th *t, Set r, char op, Set s) {
  uint32_t *o = r.l == t->la ? t->lb : t->la;
  uint64_t *bw = t->bits;
  size_t k = 0;
  if (!r.b && !s.b) {
    if (op == 'O' && r.n + s.n > lcap) { // too big for a list: switch to a bitmap
      memset(bw, 0, W * 8ull);
      for (size_t i = 0; i < r.n; i++)
        bw[r.l[i] >> 6] |= 1ull << (r.l[i] & 63);
      for (size_t i = 0; i < s.n; i++)
        bw[s.l[i] >> 6] |= 1ull << (s.l[i] & 63);
      return (Set){NULL, bw, 0};
    }
    if (op == 'O') {
      size_t i = 0, j = 0;
      while (i < r.n && j < s.n) {
        uint32_t x = r.l[i], y = s.l[j];
        o[k++] = x < y ? x : y;
        i += x <= y;
        j += y <= x;
      }
      while (i < r.n)
        o[k++] = r.l[i++];
      while (j < s.n)
        o[k++] = s.l[j++];
    } else if (op == 'A' && s.n < r.n) {
      for (size_t i = 0, j = 0; i < s.n && j < r.n; i++)
        if ((j = gallop(r.l, j, r.n, s.l[i])) < r.n && r.l[j] == s.l[i])
          o[k++] = s.l[i];
    } else {
      for (size_t i = 0, j = 0; i < r.n; i++) {
        j = gallop(s.l, j, s.n, r.l[i]);
        if ((j < s.n && s.l[j] == r.l[i]) == (op == 'A'))
          o[k++] = r.l[i];
      }
    }
    return (Set){o, NULL, k};
  }
  if (!r.b && op != 'O') { // list AND/NOT bitmap
    for (size_t i = 0; i < r.n; i++)
      if (bit(s.b, r.l[i]) == (op == 'A'))
        o[k++] = r.l[i];
    return (Set){o, NULL, k};
  }
  if (r.b && !s.b && op == 'A') { // bitmap AND list
    for (size_t i = 0; i < s.n; i++)
      if (bit(r.b, s.l[i]))
        o[k++] = s.l[i];
    return (Set){o, NULL, k};
  }
  if (!r.b) { // list OR bitmap: start from the bitmap
    memcpy(bw, s.b, W * 8ull);
    for (size_t i = 0; i < r.n; i++)
      bw[r.l[i] >> 6] |= 1ull << (r.l[i] & 63);
    return (Set){NULL, bw, 0};
  }
  if (r.b != bw)
    memcpy(bw, r.b, W * 8ull);
  if (!s.b)
    for (size_t i = 0; i < s.n; i++)
      if (op == 'O')
        bw[s.l[i] >> 6] |= 1ull << (s.l[i] & 63);
      else
        bw[s.l[i] >> 6] &= ~(1ull << (s.l[i] & 63));
  else
    for (uint32_t i = 0; i < W; i++)
      bw[i] = op == 'A' ? bw[i] & s.b[i] : op == 'O' ? bw[i] | s.b[i] : bw[i] & ~s.b[i];
  return (Set){NULL, bw, 0};
}

static void add(Result *o, uint32_t d) {
  if (o->count < 10)
    o->ids[o->m++] = d;
  o->count++;
  o->sum += d;
}

static void summarize(Set r, Result *o) {
  if (!r.b)
    for (size_t i = 0; i < r.n; i++)
      add(o, r.l[i]);
  else
    for (uint32_t w = 0; w < W; w++)
      for (uint64_t x = r.b[w]; x; x &= x - 1)
        add(o, w * 64 + __builtin_ctzll(x));
}

// True when document d has the query's terms at consecutive positions.
static int phrase_in(Th *t, const Query *q, uint32_t d) {
  const uint8_t *s = (const uint8_t *)(docs_data + doff[d]);
  uint32_t n;
  if (d == N - 1) {
    char tmp[64 * 1024];
    memcpy(tmp, s, dlen[d]);
    memset(tmp + dlen[d], 0, 64);
    n = tokenize((const uint8_t *)tmp, (const uint8_t *)tmp + dlen[d], t->tokids);
  } else {
    n = tokenize(s, s + dlen[d], t->tokids);
  }
  for (uint32_t i = 0; i + q->n <= n; i++) {
    uint32_t k = 0;
    while (k < q->n && t->tokids[i + k] == q->t[k])
      k++;
    if (k == q->n)
      return 1;
  }
  return 0;
}

static uint32_t rarest(const Query *q) {
  uint32_t f = 0;
  for (uint32_t i = 1; i < q->n; i++)
    if (df[q->t[i]] < df[q->t[f]])
      f = i;
  return f;
}

static void *answer(void *arg) {
  Th *t = arg;
  for (uint32_t qi; (qi = __atomic_fetch_add(&next_q, 16, __ATOMIC_RELAXED)) < nq;)
    for (uint32_t q_end = qi + 16 < nq ? qi + 16 : nq; qi < q_end; qi++) {
      const Query *q = &qs[qi];
      Result *out = &res[qi];
      if (!q->phrase) {
        Set r = term_set(q->t[0]);
        for (uint32_t i = 1; i < q->n; i++)
          r = apply(t, r, q->op[i], term_set(q->t[i]));
        summarize(r, out);
      } else {
        uint32_t f = rarest(q);
        Set c = term_set(q->t[f]);
        if (c.b) {
          for (uint32_t w = 0; w < W; w++) {
            uint64_t m = c.b[w];
            if (!m)
              continue;
            for (uint32_t i = 0; i < q->n; i++)
              if (i != f && bm[q->t[i]]) {
                m &= bm[q->t[i]][w];
                if (!m)
                  break;
              }
            while (m) {
              uint32_t d = w * 64 + __builtin_ctzll(m);
              m &= m - 1;
              uint32_t i = 0;
              while (i < q->n && (i == f || bm[q->t[i]] || has(term_set(q->t[i]), d)))
                i++;
              if (i == q->n && phrase_in(t, q, d))
                add(out, d);
            }
          }
        } else {
          for (uint32_t j = 0; j < c.n; j++) {
            uint32_t d = c.l[j];
            uint32_t i = 0;
            while (i < q->n && (i == f || has(term_set(q->t[i]), d)))
              i++;
            if (i == q->n && phrase_in(t, q, d))
              add(out, d);
          }
        }
      }
    }
  return NULL;
}

static void run_threads(void *(*f)(void *)) {
  pthread_t id[64];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 64 * 1024);
  for (int i = 0; i < nth; i++)
    if (pthread_create(&id[i], &attr, f, &th[i]))
      die("pthread_create failed");
  pthread_attr_destroy(&attr);
  for (int i = 0; i < nth; i++)
    pthread_join(id[i], NULL);
}

static char *num(char *o, uint64_t v) {
  char tmp[20];
  int i = 0;
  do
    tmp[i++] = '0' + v % 10;
  while (v /= 10);
  while (i)
    *o++ = tmp[--i];
  return o;
}

static char *slurp(const char *path, long *size) {
  int f = open(path, O_RDONLY);
  struct stat st;
  if (f < 0 || fstat(f, &st))
    die("cannot open input");
  char *b = malloc(st.st_size + 1);
  for (long got = 0; got < st.st_size;) {
    ssize_t n = pread(f, b + got, st.st_size - got, got);
    if (n <= 0)
      die("read failed");
    got += n;
  }
  close(f);
  *size = st.st_size;
  return b;
}

int main(void) {
  mallopt(M_ARENA_MAX, 1);
  // Queries: BOOL t1 op t2 ... or PHRASE "t1 t2 ...".
  long qsize;
  char *qb = slurp("/data/queries.txt", &qsize);
  for (long i = 0; i < qsize; i++)
    nq += qb[i] == '\n';
  qs = calloc(nq + 1, sizeof(Query));
  res = calloc(nq + 1, sizeof(Result));
  char *p = qb;
  for (uint32_t i = 0; i < nq; i++) {
    Query *q = &qs[i];
    char *e = memchr(p, '\n', qb + qsize - p);
    q->phrase = *p == 'P';
    q->heavy = -1;
    p += q->phrase ? 8 : 5; // past `PHRASE "` or `BOOL `
    char *stop = q->phrase ? e - 1 : e;
    for (int k = 0; p < stop; k++) {
      char *w = p;
      while (p < stop && *p != ' ')
        p++;
      if (q->phrase || k % 2 == 0) {
        if (q->n == MAXT)
          die("too many terms in a query");
        q->t[q->n++] = intern(w, p - w);
      } else {
        q->op[q->n] = *w; // the operator applied with the next term
      }
      p++;
    }
    p = e + 1;
  }

  struct stat st;
  fd = open("/data/docs.txt", O_RDONLY);
  if (fd < 0 || fstat(fd, &st))
    die("cannot open /data/docs.txt");
  long size = st.st_size;
  docs_data = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (docs_data == MAP_FAILED)
    die("mmap failed");

  if (size > 0) {
    const char *last_nl = memrchr(docs_data, '\n', size > 0 ? size - 1 : 0);
    const char *last_line = last_nl ? last_nl + 1 : docs_data;
    N = strtoul(last_line, NULL, 10) + 1;
  }
  W = (N + 63) / 64;
  doff = malloc((N + 1) * sizeof *doff);
  dlen = malloc((N + 1) * sizeof *dlen);
  bm = calloc(nterms + 1, sizeof *bm);

  cpu_set_t cpus; // the CPUs this container may use, not the host's count
  nth = sched_getaffinity(0, sizeof cpus, &cpus) ? 1 : CPU_COUNT(&cpus);
  nth = nth < 1 ? 1 : nth > 64 ? 64 : nth;
  th = calloc(nth, sizeof(Th));
  for (int i = 1; i < nth; i++) {
    long b = size * i / nth;
    while (b < size && docs_data[b - 1] != '\n')
      b++;
    th[i].beg = b;
    th[i - 1].end = b;
  }
  th[0].beg = 0;
  th[nth - 1].end = size;
  for (int i = nth - 1; i >= 0; i--) { // each range's first doc id, from its first line
    uint32_t next = i + 1 < nth ? th[i + 1].first : N;
    th[i].first = th[i].beg < size ? strtoul(docs_data + th[i].beg, NULL, 10) : N;
    th[i].wlo = th[i].first >> 6;
    th[i].whi = (next ? next - 1 : 0) >> 6;
    th[i].tokids = malloc((BUFSZ / 2 + 2) * 4ull);
    th[i].tc = calloc(nterms + 1, sizeof(Tc));
  }
  run_threads(pass1_count);

  for (uint32_t t = 0; t < nterms; t++) {
    uint32_t sum = 0;
    for (int i = 0; i < nth; i++)
      sum += th[i].tc[t].cnt;
    if (sum > N / 32)
      bm[t] = calloc(W, 8);
  }

  // Posting slices for the list terms, each thread's cursor after earlier threads'.
  df = calloc(nterms + 1, 4);
  pstart = calloc(nterms + 1, 4);
  uint32_t total = 0, maxlist = 0;
  for (uint32_t t = 0; t < nterms; t++) {
    pstart[t] = total;
    if (!bm[t]) {
      for (int i = 0; i < nth; i++) {
        uint32_t c = th[i].tc[t].cnt;
        th[i].tc[t].cnt = total;
        total += c;
      }
      df[t] = total - pstart[t];
      maxlist = df[t] > maxlist ? df[t] : maxlist;
    }
  }
  post = malloc((size_t)total * 4 + 4);

  // Clear stamp for Pass 2
  for (int i = 0; i < nth; i++)
    for (uint32_t t = 0; t < nterms; t++)
      th[i].tc[t].stamp = 0;

  run_threads(pass2_fill);

  for (uint32_t t = 0; t < nterms; t++) {
    if (bm[t])
      for (uint32_t w = 0; w < W; w++)
        df[t] += __builtin_popcountll(bm[t][w]);
  }

  lcap = (maxlist > N / 32 ? maxlist : N / 32) + 1;
  for (int i = 0; i < nth; i++) {
    free(th[i].tc);
    th[i].la = malloc(lcap * 4ull);
    th[i].lb = malloc(lcap * 4ull);
    th[i].bits = malloc(W * 8ull + 8);
  }

  malloc_trim(0);
  run_threads(answer);
  munmap((void *)docs_data, size);
  close(fd);

  int out = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out < 0)
    die("cannot create /out/results.txt");
  char *ob = malloc(OUTSZ), *o = ob;
  for (uint32_t i = 0; i < nq; i++) {
    Result *r = &res[i];
    o = num(o, r->count);
    *o++ = ' ';
    o = num(o, r->sum);
    for (uint32_t k = 0; k < r->m; k++) {
      *o++ = ' ';
      o = num(o, r->ids[k]);
    }
    *o++ = '\n';
    if (o - ob > OUTSZ - 256) {
      if (write(out, ob, o - ob) != o - ob)
        die("write failed");
      o = ob;
    }
  }
  if (write(out, ob, o - ob) != o - ob || close(out))
    die("write failed");
  close(fd);
  return 0;
}
