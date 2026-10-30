#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <malloc.h>
#include <immintrin.h>
#include <memory>
#include <span>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int MAX_TERM = 32;
constexpr u32 NONE = 0xFFFFFFFFu;

struct Lower {
    uint8_t t[256] = {};
    Lower() {
        for (int c = '0'; c <= '9'; ++c) t[c] = c;
        for (int c = 'a'; c <= 'z'; ++c) t[c] = c;
        for (int c = 'A'; c <= 'Z'; ++c) t[c] = c + 32;
    }
};
const Lower LOWER;

struct File {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

File map_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); _exit(1); }
    File f;
    struct stat st;
    fstat(fd, &st);
    f.size = st.st_size;
    void* p = mmap(nullptr, f.size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); _exit(1); }
    f.data = static_cast<const uint8_t*>(p);
    close(fd);
    return f;
}

inline u64 token_bytes(const uint8_t* p) {
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    __m256i c0 = _mm256_set1_epi8('0'), c9 = _mm256_set1_epi8('9');
    __m256i ca = _mm256_set1_epi8('a'), cz = _mm256_set1_epi8('z');
    __m256i cA = _mm256_set1_epi8('A'), cZ = _mm256_set1_epi8('Z');
    auto is_tok = [&](__m256i v) {
        __m256i d = _mm256_and_si256(_mm256_cmpgt_epi8(v, _mm256_sub_epi8(c0, _mm256_set1_epi8(1))),
                                     _mm256_cmpgt_epi8(_mm256_add_epi8(c9, _mm256_set1_epi8(1)), v));
        __m256i l = _mm256_and_si256(_mm256_cmpgt_epi8(v, _mm256_sub_epi8(ca, _mm256_set1_epi8(1))),
                                     _mm256_cmpgt_epi8(_mm256_add_epi8(cz, _mm256_set1_epi8(1)), v));
        __m256i u = _mm256_and_si256(_mm256_cmpgt_epi8(v, _mm256_sub_epi8(cA, _mm256_set1_epi8(1))),
                                     _mm256_cmpgt_epi8(_mm256_add_epi8(cZ, _mm256_set1_epi8(1)), v));
        return _mm256_or_si256(d, _mm256_or_si256(l, u));
    };
    u32 m0 = _mm256_movemask_epi8(is_tok(v0));
    u32 m1 = _mm256_movemask_epi8(is_tok(v1));
    return u64(m0) | (u64(m1) << 32);
}

struct Key {
    uint8_t len = 0;
    char text[MAX_TERM] = {};
    bool operator==(const Key& o) const {
        return len == o.len && memcmp(text, o.text, len) == 0;
    }
};

inline Key make_key(const uint8_t* p, size_t len) {
    Key k;
    k.len = static_cast<uint8_t>(len);
    for (size_t i = 0; i < len; ++i) k.text[i] = static_cast<char>(LOWER.t[p[i]]);
    return k;
}

inline u64 hash_key(const Key& k) {
    u64 h = 0x100;
    for (size_t i = 0; i < k.len; ++i) h = (h ^ static_cast<uint8_t>(k.text[i])) * 0x100000001b3ull;
    return h;
}

class TermTable {
public:
    u32 insert(const Key& k) {
        if (keys_.size() * 2 >= cap_) grow();
        const u64 h = hash_key(k);
        size_t b = h & (cap_ - 1);
        while (table_[b] != NONE) {
            if (keys_[table_[b]] == k) return table_[b];
            b = (b + 1) & (cap_ - 1);
        }
        const u32 id = static_cast<u32>(keys_.size());
        keys_.push_back(k);
        hashes_.push_back(h);
        table_[b] = id;
        return id;
    }

    u32 find(const Key& k, u64 h) const {
        if (cap_ == 0) return NONE;
        size_t b = h & (cap_ - 1);
        while (table_[b] != NONE) {
            if (keys_[table_[b]] == k) return table_[b];
            b = (b + 1) & (cap_ - 1);
        }
        return NONE;
    }

    void prefetch(u64 h) const {
        _mm_prefetch(reinterpret_cast<const char*>(&table_[h & (cap_ - 1)]), _MM_HINT_T0);
    }

    void renumber(const std::vector<u32>& new_id) {
        std::vector<Key> nk(keys_.size());
        std::vector<u64> nh(keys_.size());
        for (size_t i = 0; i < keys_.size(); ++i) {
            nk[new_id[i]] = keys_[i];
            nh[new_id[i]] = hashes_[i];
        }
        keys_ = std::move(nk);
        hashes_ = std::move(nh);
        std::fill(table_.begin(), table_.end(), NONE);
        for (size_t i = 0; i < keys_.size(); ++i) {
            size_t b = hashes_[i] & (cap_ - 1);
            while (table_[b] != NONE) b = (b + 1) & (cap_ - 1);
            table_[b] = static_cast<u32>(i);
        }
    }

    size_t size() const { return keys_.size(); }
    const Key& key(u32 id) const { return keys_[id]; }

private:
    void grow() {
        const size_t ncap = cap_ ? cap_ * 2 : 1024;
        std::vector<u32> nt(ncap, NONE);
        for (size_t i = 0; i < keys_.size(); ++i) {
            size_t b = hashes_[i] & (ncap - 1);
            while (nt[b] != NONE) b = (b + 1) & (ncap - 1);
            nt[b] = static_cast<u32>(i);
        }
        table_ = std::move(nt);
        cap_ = ncap;
    }

    std::vector<Key> keys_;
    std::vector<u64> hashes_;
    std::vector<u32> table_;
    size_t cap_ = 0;
};

template <class F>
inline void for_each_token(const uint8_t* q, const uint8_t* e, const TermTable& terms, F&& f) {
    constexpr size_t LAG = 8;
    Key keys[LAG];
    u64 hashes[LAG];
    size_t n = 0;
    u64 m = 0;
    for (bool fresh = true; q < e;) {
        if (fresh) {
            m = token_bytes(q);
            fresh = false;
        }
        if (m == 0) { q += 64; fresh = true; continue; }
        const uint8_t* s = q + __builtin_ctzll(m);
        const u64 after = ~m & (~0ull << __builtin_ctzll(m));
        if (after == 0 && s != q) { q = s; fresh = true; continue; }
        const uint8_t* t = after ? q + __builtin_ctzll(after) : s + 64;
        while (t < e && LOWER.t[*t]) ++t;
        m &= after ? ~0ull << __builtin_ctzll(after) : 0;
        if (t > q + 64) { q = t; fresh = true; }
        const size_t len = t - s;
        if (len <= MAX_TERM) {
            keys[n % LAG] = make_key(s, len);
            hashes[n % LAG] = hash_key(keys[n % LAG]);
            terms.prefetch(hashes[n % LAG]);
            if (n >= LAG) f(terms.find(keys[(n - LAG) % LAG], hashes[(n - LAG) % LAG]));
            ++n;
        }
    }
    for (size_t i = n >= LAG ? n - LAG : 0; i < n; ++i)
        f(terms.find(keys[i % LAG], hashes[i % LAG]));
}

struct Query {
    bool phrase;
    std::span<const u32> terms;
    std::span<const char> ops;
};

struct Queries {
    std::vector<u32> terms;
    std::vector<char> ops;
    std::vector<uint32_t> first;
    std::vector<uint8_t> phrase;

    size_t size() const { return phrase.size(); }
    Query operator[](size_t i) const {
        const u32 b = first[i], e = first[i + 1];
        return Query{phrase[i] != 0, std::span<const u32>(&terms[b], e - b),
                     std::span<const char>(&ops[b], e - b)};
    }
};

Queries parse_queries(const File& f, TermTable& terms) {
    Queries qs;
    const uint8_t* p = f.data;
    const uint8_t* end = f.data + f.size;
    while (p < end) {
        qs.first.push_back(static_cast<u32>(qs.terms.size()));
        const uint8_t* nl = static_cast<const uint8_t*>(memchr(p, '\n', end - p));
        if (!nl) nl = end;
        const bool phrase = p[0] == 'P';
        const uint8_t* s = p + (phrase ? 8 : 5);
        const uint8_t* e = phrase ? nl - 1 : nl;
        for (int word = 0; s < e; ++word) {
            const uint8_t* sp = static_cast<const uint8_t*>(memchr(s, ' ', e - s));
            if (!sp) sp = e;
            if (!phrase && word % 2 == 1) {
                qs.ops.back() = static_cast<char>(*s);
            } else {
                qs.terms.push_back(terms.insert(make_key(s, sp - s)));
                qs.ops.push_back(0);
            }
            s = sp < e ? sp + 1 : e;
        }
        qs.phrase.push_back(phrase ? 1 : 0);
        p = nl + 1;
    }
    qs.first.push_back(static_cast<u32>(qs.terms.size()));
    return qs;
}

struct Index {
    u32 docs = 0;
    size_t words = 0;
    std::vector<u32> dense;
    std::vector<u64> bits;
    std::vector<u32> df;
    int threads = 0;
    std::vector<u32> first;
    std::vector<u32> start;
    std::vector<std::vector<u32>> seg_end;
    std::unique_ptr<uint16_t[]> post;
    std::vector<u32> text;
};

u32 parse_id(const uint8_t* p) {
    u32 v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return v;
}

Index build_index(const File& f, TermTable& terms, std::vector<u32>& query_terms, int nthreads) {
    Index ix;
    const u32 nterms = terms.size();
    const uint8_t* base = f.data;
    const size_t size = f.size;
    if (size) {
        size_t last = size - 1;
        while (last > 0 && base[last - 1] != '\n') --last;
        ix.docs = parse_id(base + last) + 1;
    }
    ix.threads = nthreads;
    ix.first.resize(nthreads);
    ix.text.resize(2 * size_t(ix.docs));
    ix.words = (ix.docs + 63) / 64;
    ix.dense.assign(nterms, NONE);

    std::vector<size_t> cut(nthreads + 1);
    for (int t = 0; t <= nthreads; ++t) {
        size_t c = size * t / nthreads;
        while (c > 0 && c < size && base[c - 1] != '\n') ++c;
        cut[t] = c;
    }

    std::vector<std::vector<u32>> count(nthreads);
    std::barrier sync(nthreads, [&]() noexcept {
        std::vector<u64> total_df(nterms);
        std::vector<u32> order(nterms), new_id(nterms);
        for (u32 id = 0; id < nterms; ++id) {
            for (int t = 0; t < nthreads; ++t) total_df[id] += count[t][id];
            order[id] = id;
        }
        std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) { return total_df[a] > total_df[b]; });
        u32 ndense = 0;
        for (u32 rank = 0; rank < nterms; ++rank) {
            new_id[order[rank]] = rank;
            if (total_df[order[rank]] * 16 > ix.docs) ix.dense[rank] = ndense++;
        }
        terms.renumber(new_id);
        for (u32& id : query_terms) id = new_id[id];
        ix.bits.assign(ndense * ix.words, 0);

        for (int t = 0; t < nthreads; ++t) {
            std::vector<u32> remapped(nterms);
            for (u32 old_id = 0; old_id < nterms; ++old_id)
                remapped[new_id[old_id]] = count[t][old_id];
            count[t] = std::move(remapped);
        }

        ix.start.resize(nterms + 1);
        u32 total_sparse = 0;
        for (u32 rank = 0; rank < nterms; ++rank) {
            ix.start[rank] = total_sparse;
            if (ix.dense[rank] == NONE) {
                for (int t = 0; t < nthreads; ++t) {
                    u32 c = count[t][rank];
                    count[t][rank] = total_sparse;
                    total_sparse += c;
                }
            } else {
                for (int t = 0; t < nthreads; ++t) count[t][rank] = 0;
            }
        }
        ix.start[nterms] = total_sparse;
        ix.post.reset(new uint16_t[total_sparse]);
    });

    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            const uint8_t* begin = base + cut[t];
            const uint8_t* end = base + cut[t + 1];
            const u32 first = ix.first[t] = begin < end ? parse_id(begin) : 0;
            auto& cnt = count[t];
            cnt.assign(nterms, 0);
            std::vector<u32> last_doc(nterms, NONE);

            u32 doc = first;
            for (const uint8_t* p = begin; p < end; ++doc) {
                const uint8_t* nl = static_cast<const uint8_t*>(memchr(p, '\n', end - p));
                if (!nl) nl = end;
                const uint8_t* tx = static_cast<const uint8_t*>(memchr(p, '\t', nl - p));
                if (!tx) tx = p; else ++tx;
                ix.text[2 * size_t(doc)] = tx - base;
                ix.text[2 * size_t(doc) + 1] = nl - base;
                for_each_token(tx, nl, terms, [&](u32 id) {
                    if (id != NONE && last_doc[id] != doc) last_doc[id] = doc, ++cnt[id];
                });
                p = nl + 1;
            }

            sync.arrive_and_wait();

            std::fill(last_doc.begin(), last_doc.end(), NONE);
            doc = first;
            for (const uint8_t* p = begin; p < end; ++doc) {
                const uint8_t* nl = base + ix.text[2 * size_t(doc) + 1];
                const uint8_t* tx = base + ix.text[2 * size_t(doc)];
                for_each_token(tx, nl, terms, [&](u32 id) {
                    if (id != NONE && last_doc[id] != doc) {
                        last_doc[id] = doc;
                        if (u32 d = ix.dense[id]; d != NONE) {
                            std::atomic_ref<u64>(ix.bits[d * ix.words + doc / 64])
                                .fetch_or(1ull << (doc % 64), std::memory_order_relaxed);
                        } else {
                            ix.post[cnt[id]++] = static_cast<uint16_t>(doc - first);
                        }
                    }
                });
                p = nl + 1;
            }
        });
    }
    for (auto& th : pool) th.join();

    madvise(const_cast<uint8_t*>(base), size, MADV_DONTNEED);
    malloc_trim(0);
    ix.seg_end = std::move(count);

    ix.df.resize(nterms);
    for (u32 id = 0; id < nterms; ++id) {
        ix.df[id] = ix.start[id + 1] - ix.start[id];
        if (ix.dense[id] != NONE)
            for (size_t w = 0; w < ix.words; ++w)
                ix.df[id] += __builtin_popcountll(ix.bits[ix.dense[id] * ix.words + w]);
    }
    return ix;
}

struct Postings {
    const u32* p;
    size_t n;
    const u64* bits;
    size_t words;
};

inline bool has(const u64* bits, u32 doc) { return bits[doc / 64] >> (doc % 64) & 1; }

template <class F>
void for_each_bit(const u64* bits, size_t words, F&& f) {
    for (size_t w = 0; w < words; ++w)
        for (u64 m = bits[w]; m; m &= m - 1) f(u32(w * 64 + __builtin_ctzll(m)));
}

inline u64 position_sum(u64 m) {
    u64 s = 0;
    while (m) { s += __builtin_ctzll(m); m &= m - 1; }
    return s;
}

void op_and(const std::vector<u32>& a, const Postings& b, std::vector<u32>& out) {
    out.clear();
    if (b.bits) {
        for (u32 x : a) if (has(b.bits, x)) out.push_back(x);
        return;
    }
    size_t ia = 0, ib = 0;
    while (ia < a.size() && ib < b.n) {
        u32 va = a[ia], vb = b.p[ib];
        if (va == vb) { out.push_back(va); ++ia; ++ib; }
        else if (va < vb) ++ia;
        else ++ib;
    }
}

void op_or(const std::vector<u32>& a, const Postings& b, std::vector<u32>& out) {
    out.clear();
    if (b.bits) {
        out.reserve(a.size() + b.n);
        auto it = a.begin();
        for_each_bit(b.bits, b.words, [&](u32 doc) {
            while (it != a.end() && *it < doc) out.push_back(*it++);
            if (it != a.end() && *it == doc) ++it;
            out.push_back(doc);
        });
        out.insert(out.end(), it, a.end());
        return;
    }
    size_t ia = 0, ib = 0;
    out.reserve(a.size() + b.n);
    while (ia < a.size() && ib < b.n) {
        u32 va = a[ia], vb = b.p[ib];
        if (va == vb) { out.push_back(va); ++ia; ++ib; }
        else if (va < vb) { out.push_back(va); ++ia; }
        else { out.push_back(vb); ++ib; }
    }
    out.insert(out.end(), a.begin() + ia, a.end());
    out.insert(out.end(), b.p + ib, b.p + b.n);
}

void op_not(const std::vector<u32>& a, const Postings& b, std::vector<u32>& out) {
    out.clear();
    if (b.bits) {
        for (u32 x : a) if (!has(b.bits, x)) out.push_back(x);
        return;
    }
    size_t ia = 0, ib = 0;
    while (ia < a.size() && ib < b.n) {
        u32 va = a[ia], vb = b.p[ib];
        if (va == vb) { ++ia; ++ib; }
        else if (va < vb) { out.push_back(va); ++ia; }
        else ++ib;
    }
    out.insert(out.end(), a.begin() + ia, a.end());
}

class Solver {
public:
    Solver(const Index& ix, const TermTable& terms, const File& docs)
        : ix_(ix), terms_(terms), docs_(docs) {}

    int answer(const Query& q, char* out) {
        if (q.phrase) phrase(q);
        else boolean(q);
        return format(out);
    }

private:
    Postings postings(u32 id) {
        if (u32 d = ix_.dense[id]; d != NONE) return {nullptr, ix_.df[id], &ix_.bits[d * ix_.words], ix_.words};
        list_.resize(ix_.df[id]);
        u32 i = ix_.start[id], n = 0;
        for (int t = 0; t < ix_.threads; ++t)
            for (u32 e = ix_.seg_end[t][id]; i < e; ++i) list_[n++] = ix_.first[t] + ix_.post[i];
        return {list_.data(), n, nullptr, ix_.words};
    }

    void boolean(const Query& q) {
        Postings first = postings(q.terms[0]);
        dense_ = first.bits != nullptr;
        if (dense_) bm_.assign(first.bits, first.bits + first.words);
        else cur_.assign(first.p, first.p + first.n);
        for (size_t i = 0; i < q.ops.size(); ++i) {
            Postings b = postings(q.terms[i + 1]);
            if (dense_) apply_dense(q.ops[i], b);
            else if (b.bits && q.ops[i] == 'O') {
                bm_.assign(b.bits, b.bits + b.words);
                for (u32 x : cur_) bm_[x / 64] |= 1ull << (x % 64);
                dense_ = true;
            } else {
                switch (q.ops[i]) {
                    case 'A': op_and(cur_, b, tmp_); break;
                    case 'O': op_or(cur_, b, tmp_); break;
                    default: op_not(cur_, b, tmp_); break;
                }
                cur_.swap(tmp_);
            }
        }
    }

    void apply_dense(char op, const Postings& b) {
        if (b.bits) {
            for (size_t w = 0; w < ix_.words; ++w) {
                switch (op) {
                    case 'A': bm_[w] &= b.bits[w]; break;
                    case 'O': bm_[w] |= b.bits[w]; break;
                    default: bm_[w] &= ~b.bits[w]; break;
                }
            }
            return;
        }
        if (op == 'A') {
            dense_ = false;
            cur_.clear();
            cur_.reserve(b.n);
            for (size_t i = 0; i < b.n; ++i) if (has(bm_.data(), b.p[i])) cur_.push_back(b.p[i]);
        } else if (op == 'O') {
            for (size_t i = 0; i < b.n; ++i) bm_[b.p[i] / 64] |= 1ull << (b.p[i] % 64);
        } else {
            for (size_t i = 0; i < b.n; ++i) bm_[b.p[i] / 64] &= ~(1ull << (b.p[i] % 64));
        }
    }

    void phrase(const Query& q) {
        dense_ = false;
        if (q.terms.empty()) { cur_.clear(); return; }
        u32 rarest = 0;
        for (size_t i = 1; i < q.terms.size(); ++i)
            if (ix_.df[q.terms[i]] < ix_.df[q.terms[rarest]]) rarest = static_cast<u32>(i);
        Postings first = postings(q.terms[rarest]);
        if (first.bits) {
            cur_.clear();
            for_each_bit(first.bits, first.words, [&](u32 doc) { cur_.push_back(doc); });
        } else {
            cur_.assign(first.p, first.p + first.n);
        }
        for (size_t i = 0; i < q.terms.size(); ++i) {
            if (i == rarest || cur_.empty()) continue;
            op_and(cur_, postings(q.terms[i]), tmp_);
            cur_.swap(tmp_);
        }
        if (cur_.empty()) return;
        size_t n = 0;
        for (u32 doc : cur_) {
            const uint8_t* p = docs_.data + ix_.text[2 * size_t(doc)];
            const uint8_t* end = docs_.data + ix_.text[2 * size_t(doc) + 1];
            size_t match = 0;
            for_each_token(p, end, terms_, [&](u32 id) {
                if (match == q.terms.size()) return;
                match = (id == q.terms[match]) ? match + 1 : (id == q.terms[0] ? 1 : 0);
            });
            if (match == q.terms.size()) cur_[n++] = doc;
        }
        cur_.resize(n);
    }

    static inline char* put_u64(char* p, u64 v) {
        char tmp[24];
        int n = 0;
        do { tmp[n++] = '0' + (v % 10); v /= 10; } while (v);
        while (n) *p++ = tmp[--n];
        return p;
    }

    int format(char* out) {
        u64 count = 0, sum = 0;
        u32 first10[10];
        int nfirst = 0;
        auto observe = [&](u32 doc) {
            ++count;
            sum += doc;
            if (nfirst < 10) first10[nfirst++] = doc;
        };
        if (dense_) {
            for (size_t w = 0; w < ix_.words; ++w) {
                u64 m = bm_[w];
                if (!m) continue;
                count += __builtin_popcountll(m);
                sum += u64(w * 64) * __builtin_popcountll(m) + position_sum(m);
                while (m && nfirst < 10) {
                    first10[nfirst++] = u32(w * 64 + __builtin_ctzll(m));
                    m &= m - 1;
                }
            }
        } else {
            for (u32 doc : cur_) observe(doc);
        }
        char* p = out;
        p = put_u64(p, count);
        *p++ = ' ';
        p = put_u64(p, sum);
        for (int i = 0; i < nfirst; ++i) {
            *p++ = ' ';
            p = put_u64(p, first10[i]);
        }
        *p++ = '\n';
        return static_cast<int>(p - out);
    }

    const Index& ix_;
    const TermTable& terms_;
    const File& docs_;
    bool dense_ = false;
    std::vector<u64> bm_;
    std::vector<u32> cur_, tmp_, list_;
};

}  // namespace

int main() {
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    const int nthreads = 12;

    File queries_file = map_file("/data/queries.txt");
    TermTable terms;
    Queries queries = parse_queries(queries_file, terms);
    std::vector<u32> query_terms = queries.terms;

    File docs_file = map_file("/data/docs.txt");
    Index ix = build_index(docs_file, terms, query_terms, nthreads);
    {
        size_t b = 0;
        for (size_t i = 0; i < queries.size(); ++i) {
            size_t n = queries.first[i + 1] - queries.first[i];
            for (size_t k = 0; k < n; ++k) queries.terms[b + k] = query_terms[b + k];
            b += n;
        }
    }

    const size_t nq = queries.size();
    std::vector<std::string> chunks(nthreads);
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            Solver solver(ix, terms, docs_file);
            std::string& out = chunks[t];
            const size_t q_start = t * nq / nthreads;
            const size_t q_end = (t + 1) * nq / nthreads;
            out.reserve((q_end - q_start) * 48);
            char line[256];
            for (size_t k = q_start; k < q_end; ++k) {
                int len = solver.answer(queries[k], line);
                out.append(line, len);
            }
        });
    }
    for (auto& th : pool) th.join();

    int out_fd = open("/out/results.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (out_fd < 0) { perror("open out"); _exit(1); }
    for (int t = 0; t < nthreads; ++t) {
        const char* p = chunks[t].data();
        size_t rem = chunks[t].size();
        while (rem) {
            ssize_t w = write(out_fd, p, rem);
            if (w <= 0) { perror("write"); _exit(1); }
            p += w;
            rem -= w;
        }
    }
    close(out_fd);
    _exit(0);
}
