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
    File f;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); _exit(1); }
    struct stat st;
    fstat(fd, &st);
    f.size = st.st_size;
    void* p = mmap(nullptr, f.size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); _exit(1); }
    f.data = static_cast<const uint8_t*>(p);
    close(fd);
    return f;
}

struct alignas(32) Key {
    u64 w[4];
};

inline u64 hash_key(const Key& k) {
    u64 h = (k.w[0] + k.w[2] * 0x94D049BB133111EBull) * 0x9E3779B97F4A7C15ull ^
            (k.w[1] + k.w[3] * 0xD6E8FEB86659FD93ull) * 0xBF58476D1CE4E5B9ull;
    return h ^ h >> 29;
}

inline bool same(const Key& a, const Key& b) {
    return ((a.w[0] ^ b.w[0]) | (a.w[1] ^ b.w[1]) | (a.w[2] ^ b.w[2]) | (a.w[3] ^ b.w[3])) == 0;
}

class TermTable {
public:
    TermTable() { rehash(1 << 16); }
    u32 size() const { return keys_.size(); }
    const uint8_t* bytes(u32 id) const { return reinterpret_cast<const uint8_t*>(keys_[id].w); }
    size_t length(u32 id) const { return lens_[id]; }

    u32 insert(const Key& k) {
        if ((keys_.size() + 1) * 2 > slots_.size()) rehash(slots_.size() * 2);
        u64 h = hash_key(k);
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            u64 s = slots_[i];
            if (s == 0) {
                keys_.push_back(k);
                lens_.push_back(strnlen(reinterpret_cast<const char*>(k.w), MAX_TERM));
                slots_[i] = (h & 0xFFFFFFFF00000000ull) | keys_.size();
                return keys_.size() - 1;
            }
            if ((s >> 32) == (h >> 32) && same(keys_[(u32)s - 1], k)) return (u32)s - 1;
        }
    }

    void renumber(const std::vector<u32>& new_id) {
        std::vector<Key> keys(keys_.size());
        std::vector<uint8_t> lens(lens_.size());
        for (u32 i = 0; i < keys_.size(); ++i) keys[new_id[i]] = keys_[i], lens[new_id[i]] = lens_[i];
        keys_.swap(keys);
        lens_.swap(lens);
        rehash(slots_.size());
    }

    void prefetch(u64 h) const { _mm_prefetch(reinterpret_cast<const char*>(&slots_[h & mask_]), _MM_HINT_T0); }

    u32 find(const Key& k, u64 h) const {
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            u64 s = slots_[i];
            if (s == 0) return NONE;
            if ((s >> 32) == (h >> 32) && same(keys_[(u32)s - 1], k)) return (u32)s - 1;
        }
    }

private:
    void rehash(size_t cap) {
        slots_.assign(cap, 0);
        mask_ = cap - 1;
        for (u32 id = 0; id < keys_.size(); ++id) {
            u64 h = hash_key(keys_[id]);
            size_t i = h & mask_;
            while (slots_[i]) i = (i + 1) & mask_;
            slots_[i] = (h & 0xFFFFFFFF00000000ull) | (id + 1);
        }
    }

    std::vector<u64> slots_;
    std::vector<Key> keys_;
    std::vector<uint8_t> lens_;
    size_t mask_ = 0;
};

inline u64 token_bytes(const uint8_t* p) {
    auto half = [](const uint8_t* q) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        __m256i letter = _mm256_sub_epi8(_mm256_or_si256(v, _mm256_set1_epi8(0x20)), _mm256_set1_epi8('a'));
        __m256i digit = _mm256_sub_epi8(v, _mm256_set1_epi8('0'));
        __m256i zero = _mm256_setzero_si256();
        __m256i in = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(letter, _mm256_set1_epi8(25)), zero),
                                     _mm256_cmpeq_epi8(_mm256_subs_epu8(digit, _mm256_set1_epi8(9)), zero));
        return u32(_mm256_movemask_epi8(in));
    };
    return half(p) | u64(half(p + 32)) << 32;
}

template <class F>
inline void for_each_token(const uint8_t* q, const uint8_t* e, const TermTable& terms, F&& emit) {
    constexpr size_t LAG = 8;
    Key keys[LAG];
    u64 hashes[LAG];
    size_t n = 0;
    u64 m = 0;
    for (bool fresh = true; q < e;) {
        if (fresh) {
            m = token_bytes(q);
            if (e - q < 64) m &= (1ull << (e - q)) - 1;
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
        if (len > MAX_TERM) continue;
        Key k;
        for (size_t i = 0; i < 4; ++i) {
            size_t b = len > 8 * i ? std::min<size_t>(len - 8 * i, 8) : 0;
            u64 w;
            memcpy(&w, s + 8 * i, 8);
            k.w[i] = (w | 0x2020202020202020ull) & (b == 8 ? ~0ull : (1ull << (8 * b)) - 1);
        }
        const u64 h = hash_key(k);
        terms.prefetch(h);
        const size_t slot = n++ % LAG;
        if (n > LAG) emit(terms.find(keys[slot], hashes[slot]));
        keys[slot] = k;
        hashes[slot] = h;
    }
    for (size_t i = n > LAG ? n - LAG : 0; i < n; ++i) emit(terms.find(keys[i % LAG], hashes[i % LAG]));
}

struct Query {
    bool phrase;
    std::span<const u32> terms;
    std::span<const char> ops;
};

struct Queries {
    std::vector<u32> terms, first{0};
    std::vector<char> ops;
    std::vector<bool> phrase;

    size_t size() const { return phrase.size(); }
    Query operator[](size_t i) const {
        size_t b = first[i], n = first[i + 1] - b;
        return {phrase[i], {terms.data() + b, n}, {ops.data() + b, n - 1}};
    }
};

Key make_key(const uint8_t* p, size_t len) {
    Key k{{0, 0, 0, 0}};
    memcpy(k.w, p, std::min<size_t>(len, MAX_TERM));
    return k;
}

Queries parse_queries(const File& f, TermTable& terms) {
    Queries qs;
    const uint8_t* p = f.data;
    const uint8_t* end = f.data + f.size;
    while (p < end) {
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
            s = sp + 1;
        }
        qs.phrase.push_back(phrase);
        qs.first.push_back(qs.terms.size());
        p = nl + 1;
    }
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

class BlockList {
public:
    struct Block {
        Block* next;
        size_t cap;
        size_t n;
        uint8_t data[];
    };

    BlockList() = default;
    BlockList(const BlockList&) = delete;
    BlockList& operator=(const BlockList&) = delete;
    BlockList(BlockList&& o) noexcept : head_(o.head_), tail_(o.tail_), cur_(o.cur_) {
        o.head_ = o.tail_ = o.cur_ = nullptr;
    }
    BlockList& operator=(BlockList&& o) noexcept {
        if (this != &o) {
            this->~BlockList();
            head_ = o.head_; tail_ = o.tail_; cur_ = o.cur_;
            o.head_ = o.tail_ = o.cur_ = nullptr;
        }
        return *this;
    }
    ~BlockList() {
        for (Block* b = head_; b;) {
            Block* next = b->next;
            munmap(b, sizeof(Block) + 3 * b->cap + 1);
            b = next;
        }
    }

    void push(u32 x) {
        if (!cur_ || cur_->n == cur_->cap) {
            size_t next_cap = cur_ ? std::min<size_t>(cur_->cap * 4, 8192) : 256;
            size_t bytes = sizeof(Block) + 3 * next_cap + 1;
            auto* b = static_cast<Block*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            b->next = nullptr;
            b->cap = next_cap;
            b->n = 0;
            if (!head_) head_ = b;
            else tail_->next = b;
            tail_ = b;
            cur_ = b;
        }
        memcpy(cur_->data + 3 * cur_->n++, &x, 4);
    }

    template <class F>
    void drain(F&& f) {
        for (Block* b = head_; b;) {
            for (size_t i = 0; i < b->n; ++i) {
                u32 x;
                memcpy(&x, b->data + 3 * i, 4);
                f(x & 0xFFFFFF);
            }
            Block* next = b->next;
            munmap(b, sizeof(Block) + 3 * b->cap + 1);
            b = next;
        }
        head_ = tail_ = cur_ = nullptr;
    }

private:
    Block* head_ = nullptr;
    Block* tail_ = nullptr;
    Block* cur_ = nullptr;
};

constexpr u32 BUCKET_BITS = 9, DOC_BITS = 15;

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
    const u32 nbuckets = (nterms >> BUCKET_BITS) + 1;

    std::vector<size_t> cut(nthreads + 1);
    for (int t = 0; t <= nthreads; ++t) {
        size_t c = size * t / nthreads;
        while (c > 0 && c < size && base[c - 1] != '\n') ++c;
        cut[t] = c;
    }
    std::vector<std::vector<BlockList>> seen(nthreads);
    std::vector<std::vector<u32>> count(nthreads);
    std::vector<u32> sampled(nthreads);
    std::barrier sync(nthreads, [&]() noexcept {
        u64 docs = 0;
        for (u32 s : sampled) docs += s;
        std::vector<u64> df(nterms);
        std::vector<u32> order(nterms), new_id(nterms);
        for (u32 id = 0; id < nterms; ++id) {
            for (int t = 0; t < nthreads; ++t) df[id] += count[t][id];
            order[id] = id;
        }
        std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) { return df[a] > df[b]; });
        u32 ndense = 0;
        for (u32 rank = 0; rank < nterms; ++rank) {
            new_id[order[rank]] = rank;
            if (df[order[rank]] * 16 > docs) ix.dense[rank] = ndense++;
        }
        terms.renumber(new_id);
        for (u32& id : query_terms) id = new_id[id];
        ix.bits.assign(ndense * ix.words, 0);
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

            auto walk = [&](const uint8_t* stop, auto&& term, auto&& done) {
                u32 doc = first;
                for (const uint8_t* p = begin; p < stop; ++doc) {
                    const uint8_t* nl = static_cast<const uint8_t*>(memchr(p, '\n', end - p));
                    const uint8_t* tx = static_cast<const uint8_t*>(memchr(p, '\t', nl - p)) + 1;
                    ix.text[2 * size_t(doc)] = tx - base;
                    ix.text[2 * size_t(doc) + 1] = nl - base;
                    for_each_token(tx, nl, terms, [&](u32 id) {
                        if (id != NONE && last_doc[id] != doc) last_doc[id] = doc, term(id, doc);
                    });
                    done();
                    p = nl + 1;
                }
                return doc - first;
            };

            sampled[t] = walk(begin + (end - begin) / 32, [&](u32 id, u32) { ++cnt[id]; }, [] {});
            sync.arrive_and_wait();
            std::fill(cnt.begin(), cnt.end(), 0);
            std::fill(last_doc.begin(), last_doc.end(), NONE);
            auto& out = seen[t];
            out.resize(nbuckets);
            const u32 docs = walk(end, [&](u32 id, u32 doc) {
                if (u32 d = ix.dense[id]; d != NONE) {
                    std::atomic_ref<u64>(ix.bits[d * ix.words + doc / 64])
                        .fetch_or(1ull << (doc % 64), std::memory_order_relaxed);
                } else {
                    out[id >> BUCKET_BITS].push((id & ((1u << BUCKET_BITS) - 1)) << DOC_BITS | (doc - first));
                    ++cnt[id];
                }
            }, [] {});
            if (docs >> DOC_BITS) { fputs("too many documents per thread\n", stderr); _exit(1); }
        });
    }
    for (auto& th : pool) th.join();
    pool.clear();

    ix.start.resize(nterms + 1);
    u32 total = 0;
    for (u32 id = 0; id < nterms; ++id) {
        ix.start[id] = total;
        for (int t = 0; t < nthreads; ++t) {
            u32 c = count[t][id];
            count[t][id] = total;
            total += c;
        }
    }
    ix.start[nterms] = total;
    ix.post.reset(new uint16_t[total]);
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            for (u32 b = 0; b < nbuckets; ++b)
                seen[t][b].drain([&](u32 x) {
                    u32 id = b << BUCKET_BITS | x >> DOC_BITS;
                    ix.post[count[t][id]++] = x & ((1u << DOC_BITS) - 1);
                });
        });
    }
    for (auto& th : pool) th.join();
    seen.clear();
    seen.shrink_to_fit();
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

void assign(std::vector<u32>& out, Postings b) {
    out.clear();
    if (b.bits) for_each_bit(b.bits, b.words, [&](u32 x) { out.push_back(x); });
    else out.assign(b.p, b.p + b.n);
}

inline size_t gallop(const u32* a, size_t lo, size_t n, u32 x) {
    size_t step = 1, hi = lo;
    while (hi < n && a[hi] < x) lo = hi + 1, hi += step, step <<= 1;
    return std::lower_bound(a + lo, a + std::min(hi, n), x) - a;
}

void op_and(const std::vector<u32>& a, Postings b, std::vector<u32>& out) {
    out.clear();
    if (b.bits) {
        for (u32 x : a)
            if (has(b.bits, x)) out.push_back(x);
    } else if (a.size() * 16 < b.n) {
        size_t j = 0;
        for (u32 x : a) {
            j = gallop(b.p, j, b.n, x);
            if (j == b.n) break;
            if (b.p[j] == x) out.push_back(x);
        }
    } else if (b.n * 16 < a.size()) {
        size_t i = 0;
        for (size_t k = 0; k < b.n; ++k) {
            i = gallop(a.data(), i, a.size(), b.p[k]);
            if (i == a.size()) break;
            if (a[i] == b.p[k]) out.push_back(b.p[k]);
        }
    } else {
        std::set_intersection(a.begin(), a.end(), b.p, b.p + b.n, std::back_inserter(out));
    }
}

void op_not(const std::vector<u32>& a, Postings b, std::vector<u32>& out) {
    out.clear();
    if (b.bits) {
        for (u32 x : a)
            if (!has(b.bits, x)) out.push_back(x);
    } else if (a.size() * 16 < b.n) {
        size_t j = 0;
        for (u32 x : a) {
            j = gallop(b.p, j, b.n, x);
            if (j == b.n || b.p[j] != x) out.push_back(x);
        }
    } else {
        std::set_difference(a.begin(), a.end(), b.p, b.p + b.n, std::back_inserter(out));
    }
}

void op_or(const std::vector<u32>& a, Postings b, std::vector<u32>& out) {
    out.clear();
    std::set_union(a.begin(), a.end(), b.p, b.p + b.n, std::back_inserter(out));
}

inline u64 position_sum(u64 m) {
    static constexpr u64 SEL[6] = {0xAAAAAAAAAAAAAAAAull, 0xCCCCCCCCCCCCCCCCull, 0xF0F0F0F0F0F0F0F0ull,
                                   0xFF00FF00FF00FF00ull, 0xFFFF0000FFFF0000ull, 0xFFFFFFFF00000000ull};
    u64 s = 0;
    for (int k = 0; k < 6; ++k) s += u64(__builtin_popcountll(m & SEL[k])) << k;
    return s;
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

    void apply_dense(char op, Postings b) {
        if (b.bits) {
            for (size_t w = 0; w < ix_.words; ++w) {
                u64 y = b.bits[w];
                switch (op) {
                    case 'A': bm_[w] &= y; break;
                    case 'O': bm_[w] |= y; break;
                    default: bm_[w] &= ~y; break;
                }
            }
        } else if (op == 'O') {
            for (size_t i = 0; i < b.n; ++i) bm_[b.p[i] / 64] |= 1ull << (b.p[i] % 64);
        } else {
            cur_.clear();
            for_each_bit(bm_.data(), ix_.words, [&](u32 x) { cur_.push_back(x); });
            dense_ = false;
            if (op == 'A') op_and(cur_, b, tmp_);
            else op_not(cur_, b, tmp_);
            cur_.swap(tmp_);
        }
    }

    void phrase(const Query& q) {
        cur_.clear();
        u32 rarest = 0;
        for (size_t i = 1; i < q.terms.size(); ++i)
            if (ix_.df[q.terms[i]] < ix_.df[q.terms[rarest]]) rarest = i;
        assign(cur_, postings(q.terms[rarest]));
        for (size_t i = 0; i < q.terms.size(); ++i) {
            if (i == rarest) continue;
            op_and(cur_, postings(q.terms[i]), tmp_);
            cur_.swap(tmp_);
            if (cur_.empty()) return;
        }
        size_t n = 0;
        for (u32 doc : cur_) {
            const uint8_t* b = docs_.data + ix_.text[2 * size_t(doc)];
            const uint8_t* e = docs_.data + ix_.text[2 * size_t(doc) + 1];
            size_t k = 0;
            for_each_token(b, e, terms_, [&](u32 id) {
                if (k == q.terms.size()) return;
                k = (id == q.terms[k]) ? k + 1 : (id == q.terms[0] ? 1 : 0);
            });
            if (k == q.terms.size()) cur_[n++] = doc;
        }
        cur_.resize(n);
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
        p += sprintf(p, "%lu %lu", count, sum);
        for (int i = 0; i < nfirst; ++i) p += sprintf(p, " %u", first10[i]);
        *p++ = '\n';
        return p - out;
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

    std::vector<std::string> chunks(nthreads);
    std::atomic<size_t> next_query{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            Solver solver(ix, terms, docs_file);
            std::string& out = chunks[t];
            out.reserve(queries.size() / nthreads * 48);
            char line[256];
            for (size_t i; (i = next_query.fetch_add(64, std::memory_order_relaxed)) < queries.size();) {
                size_t end = std::min(i + 64, queries.size());
                for (size_t k = i; k < end; ++k) {
                    int len = solver.answer(queries[k], line);
                    out.append(line, len);
                }
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
