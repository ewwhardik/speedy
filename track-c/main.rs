// Speed Demon 2026, Track C (Join). See specs/c-join.md.
//
// records.csv is memory-mapped and cut into one range per thread. A range
// starts at the first LF followed by a digit or a BOM: notes hold only
// lowercase words, spaces, commas and quotes, so that pair is always a record
// start. Each record is parsed in place, with word-at-a-time (SWAR) number
// parsing and an AVX2 search for the closing '"' of each quoted field; a
// field's byte count is its raw length minus one per "" pair. A name is known
// by a 64-bit hash of its raw quoted bytes (escaping is one-to-one, and both
// empty forms are empty), interned in a shared lock-free table that hands out
// dense ids, and marked in a per-country bitset. Table and bitset lookups are
// cache misses, so they trail the parser through a small prefetch ring.
// Everything else is summed per thread and merged at the end.
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

const SLOT_BITS: u32 = 18; // up to 2^18 distinct names; the generator uses 100,000
const WORDS: usize = 1 << (SLOT_BITS - 6);
const RING: usize = 16;
type Group = [i64; 7]; // count, id_sum, amount_sum, min, max, name_bytes, note_bytes
const EMPTY: Group = [0, 0, 0, i64::MAX, i64::MIN, 0, 0];

fn map(path: &str) -> &'static [u8] {
    let f = std::fs::File::open(path).expect(path);
    #[cfg(unix)]
    if let len @ 1.. = f.metadata().unwrap().len() as usize {
        extern "C" {
            fn mmap(a: *mut u8, l: usize, p: i32, f: i32, fd: i32, o: i64) -> *const u8;
        }
        use std::os::fd::AsRawFd;
        return unsafe {
            std::slice::from_raw_parts(mmap(0 as _, len, 1, 2 | 0x8000, f.as_raw_fd(), 0), len)
        };
    }
    Box::leak(std::fs::read(path).unwrap().into_boxed_slice())
}

// The len (1 to 8) decimal digits at p.
unsafe fn digits(p: *const u8, len: usize) -> i64 {
    let v = (p as *const u64)
        .read_unaligned()
        .wrapping_sub(0x3030_3030_3030_3030)
        << (64 - 8 * len);
    let v = (v.wrapping_mul(10) + (v >> 8)) & 0x00FF_00FF_00FF_00FF; // leading zeros in front
    let v = (v.wrapping_mul(100) + (v >> 16)) & 0x0000_FFFF_0000_FFFF;
    ((v.wrapping_mul(10000) + (v >> 32)) & 0xFFFF_FFFF) as i64
}

// Bit masks of the '"', ',' and LF bytes among the 64 at p.
unsafe fn masks(p: *const u8) -> [u64; 3] {
    let (lo, hi) = (
        _mm256_loadu_si256(p as _),
        _mm256_loadu_si256(p.add(32) as _),
    );
    [b'"', b',', b'\n'].map(|c| {
        let c = _mm256_set1_epi8(c as i8);
        let m = |v| _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, c)) as u32 as u64;
        m(lo) | m(hi) << 32
    })
}

// Bit i is the xor of bits 0..=i: set inside quotes, from each opening quote.
fn prefix_xor(x: u64) -> u64 {
    unsafe {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128(
            _mm_set_epi64x(0, x as i64),
            _mm_set1_epi8(-1),
            0,
        )) as u64
    }
}

// A hash of the len bytes at p, reading whole words past them.
unsafe fn hash(p: *const u8, len: usize) -> u64 {
    let word = |i: usize| {
        let keep = len.saturating_sub(8 * i).min(8);
        (p.add(8 * i) as *const u64).read_unaligned()
            & if keep == 8 {
                u64::MAX
            } else {
                (1 << 8 * keep) - 1
            }
    };
    let mut h = (word(0) ^ len as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(23)
        ^ word(1).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(41)
        ^ word(2).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    for i in 3..len.div_ceil(8) {
        h = (h ^ word(i))
            .wrapping_mul(0xFF51_AFD7_ED55_8CCD)
            .rotate_left(29); // rare: over 24 bytes
    }
    let h = (h ^ h >> 32).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ h >> 29
}

struct Names {
    slots: Vec<AtomicU64>, // (hash tag << SLOT_BITS) | id, 0 = empty
    next: AtomicU32,
}

impl Names {
    fn slot(&self, h: u64) -> &AtomicU64 {
        &self.slots[h as usize & ((1 << SLOT_BITS) - 1)]
    }

    fn id(&self, h: u64) -> usize {
        let (tag, mask) = ((h >> SLOT_BITS) | 1, (1 << SLOT_BITS) - 1);
        let mut i = h as usize & mask;
        loop {
            let v = self.slots[i].load(Relaxed);
            if v >> SLOT_BITS == tag {
                return v as usize & mask;
            }
            if v == 0 {
                let id = self.next.fetch_add(1, Relaxed) as u64;
                match self.slots[i].compare_exchange(0, tag << SLOT_BITS | id, Relaxed, Relaxed) {
                    Ok(_) => return id as usize,
                    Err(_) => continue, // lost the race: look at what won
                }
            }
            i = (i + 1) & mask;
        }
    }
}

fn zeroed_atomics(n: usize) -> Vec<AtomicU64> {
    // vec![0; n] comes from calloc, so pages nothing reaches are never touched.
    let mut v = std::mem::ManuallyDrop::new(vec![0u64; n]);
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut AtomicU64, n, n) }
}

fn prefetch<T>(p: &T) {
    unsafe { _mm_prefetch(p as *const T as *const i8, _MM_HINT_T0) }
}

// Aggregates the records starting in s[start..end]. Records are found by
// their separators: commas and LFs outside quotes, five per record, located
// 64 bytes at a time with a carry-less multiply for the in-quote mask. Each
// separator also gets the number of quotes before it, so a field's unescaped
// length is its span minus half its quotes, minus one for the delimiters.
// Needs 64 readable bytes after the last record.
fn chunk(
    s: &[u8],
    start: usize,
    end: usize,
    region: &[u8],
    names: &Names,
    seen: &[AtomicU64],
) -> Vec<Group> {
    let other = (seen.len() / WORDS - 1) as u8;
    let mut g = vec![EMPTY; other as usize + 1];
    // Name lookups for record k happen at k + RING/2, bit sets at k + RING.
    let (mut ring, mut k) = ([(0usize, 0u64); RING], 0);
    let mark = |ring: &mut [(usize, u64); RING], k: usize| {
        let (c, h) = ring[k % RING];
        let (w, bit) = (&seen[c * WORDS + h as usize / 64], 1 << (h % 64));
        if w.load(Relaxed) & bit == 0 {
            w.fetch_or(bit, Relaxed);
        }
    };
    let at = |i: usize| unsafe { s.as_ptr().add(i) };
    let text = |sep: &[usize], quotes: &[u32]| match sep[1] - sep[0] - 1 {
        0 => 0,
        span => (span - (quotes[1] - quotes[0]) as usize / 2 - 1) as i64,
    };
    let (mut b, mut rs, mut nf, mut qbase, mut carry) = (start, start, 0, 0, 0);
    let (mut sep, mut quotes) = ([0usize; 5], [0u32; 5]);
    'blocks: while rs < end {
        let [q, comma, lf] = unsafe { masks(at(b)) };
        let inq = prefix_xor(q) ^ carry;
        carry = ((inq as i64) >> 63) as u64;
        let mut bits = (comma | lf) & !inq;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            bits &= bits - 1;
            (sep[nf], quotes[nf]) = (
                b + bit as usize,
                qbase + (q & ((1 << bit) - 1)).count_ones(),
            );
            nf += 1;
            if nf < 5 {
                continue;
            }
            nf = 0;
            unsafe {
                let first = rs + 3 * (s[rs] == 0xEF) as usize; // after a byte order mark
                let id = digits(at(first), sep[0] - first);
                let (name_len, note_len) =
                    (text(&sep[0..], &quotes[0..]), text(&sep[1..], &quotes[1..]));
                let r = digits(at(sep[2] + 1), sep[3] - sep[2] - 1) as usize;
                let neg = s[sep[3] + 1] == b'-';
                let (from, to) = (
                    sep[3] + 1 + neg as usize,
                    sep[4] - (s[sep[4] - 1] == b'\r') as usize,
                );
                let cents = digits(at(from), to - from - 3) * 100 + digits(at(to - 2), 2);
                let cents = if neg { -cents } else { cents };
                let c = *region.get(r).unwrap_or(&other) as usize;
                let a = &mut g[c];
                a[0] += 1;
                a[1] += id;
                a[2] += cents;
                a[3] = a[3].min(cents);
                a[4] = a[4].max(cents);
                a[5] += name_len;
                a[6] += note_len;
                let h = hash(at(sep[0] + 2), (sep[1] - sep[0]).saturating_sub(3));
                prefetch(names.slot(h));
                if k >= RING {
                    mark(&mut ring, k);
                }
                ring[k % RING] = (c, h);
                let half = k.wrapping_sub(RING / 2) % RING;
                if k >= RING / 2 {
                    let (c, h) = ring[half];
                    ring[half].1 = names.id(h) as u64;
                    prefetch(&seen[c * WORDS + ring[half].1 as usize / 64]);
                }
                k += 1;
            }
            rs = sep[4] + 1;
            if rs >= end {
                break 'blocks;
            }
        }
        qbase += q.count_ones();
        b += 64;
    }
    for j in k.saturating_sub(RING)..k {
        if j + RING / 2 >= k {
            ring[j % RING].1 = names.id(ring[j % RING].1) as u64;
        }
        mark(&mut ring, j);
    }
    g
}

fn main() {
    #[cfg(unix)]
    unsafe {
        extern "C" {
            fn prctl(option: i32, arg: u64, ...) -> i32;
        }
        prctl(41, 1); // PR_SET_THP_DISABLE: huge pages would make every touched bitset page 2 MB
    }
    let dir = std::env::args().nth(1).unwrap_or("/data".into());
    let out = std::env::args().nth(2).unwrap_or("/out".into());
    let regions = std::fs::read(format!("{dir}/regions.csv")).unwrap();
    let rows: Vec<(usize, &[u8])> = (regions
        .split(|&c| c == b'\n')
        .skip(1)
        .filter(|l| !l.is_empty()))
    .map(|l| {
        let comma = l.iter().position(|&c| c == b',').unwrap();
        (
            std::str::from_utf8(&l[..comma]).unwrap().parse().unwrap(),
            &l[comma + 1..],
        )
    })
    .collect();
    let mut codes: Vec<&[u8]> = rows.iter().map(|r| r.1).collect();
    codes.sort();
    codes.dedup();
    assert!(codes.len() < 255);
    let mut region = vec![codes.len() as u8; rows.iter().map(|r| r.0 + 1).max().unwrap_or(0)];
    for (id, code) in &rows {
        region[*id] = codes.binary_search(code).unwrap() as u8;
    }

    let s = map(&format!("{dir}/records.csv"));
    let names = Names {
        slots: zeroed_atomics(1 << SLOT_BITS),
        next: AtomicU32::new(0),
    };
    let seen = zeroed_atomics((codes.len() + 1) * WORDS);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Start of the record at or after offset o.
    let sync = |o: usize| match o {
        0 => s
            .iter()
            .position(|&c| c == b'\n')
            .map_or(s.len(), |p| p + 1),
        _ => (o..s.len())
            .find(|&i| s[i - 1] == b'\n' && (s[i].is_ascii_digit() || s[i] == 0xEF))
            .unwrap_or(s.len()),
    };
    // The last few hundred bytes are parsed from a padded copy.
    let tail = sync(s.len().saturating_sub(512)).max(sync(0));
    let cuts: Vec<usize> = (0..threads)
        .map(|t| sync(tail * t / threads).min(tail))
        .chain([tail])
        .collect();
    let mut padded = s[tail..].to_vec();
    padded.resize(padded.len() + 64, 0);
    let mut total = chunk(&padded, 0, s.len() - tail, &region, &names, &seen);
    std::thread::scope(|sc| {
        let (region, names, seen, cuts) = (&region, &names, &seen, &cuts);
        let work: Vec<_> = (0..threads)
            .map(|t| sc.spawn(move || chunk(s, cuts[t], cuts[t + 1], region, names, seen)))
            .collect();
        for w in work {
            for (a, b) in total.iter_mut().zip(w.join().unwrap()) {
                for k in [0, 1, 2, 5, 6] {
                    a[k] += b[k];
                }
                a[3] = a[3].min(b[3]);
                a[4] = a[4].max(b[4]);
            }
        }
    });

    let money = |c: i64| {
        format!(
            "{}{}.{:02}",
            if c < 0 { "-" } else { "" },
            c.abs() / 100,
            c.abs() % 100
        )
    };
    let mut o = String::new();
    for (c, a) in total.iter().enumerate().filter(|(_, a)| a[0] > 0) {
        let key = codes
            .get(c)
            .map_or("_UNMATCHED", |k| std::str::from_utf8(k).unwrap());
        let distinct: u32 = seen[c * WORDS..(c + 1) * WORDS]
            .iter()
            .map(|w| w.load(Relaxed).count_ones())
            .sum();
        o += &format!(
            "{key} {} {} {} {} {} {distinct} {} {}\n",
            a[0],
            a[1],
            money(a[2]),
            money(a[3]),
            money(a[4]),
            a[5],
            a[6]
        );
    }
    std::fs::write(format!("{out}/aggregates.txt"), o).unwrap();
}
