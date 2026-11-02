//! # Track B: Index
//!
//! This program answers 50,000 queries over a corpus of about 250,000 documents, one per line
//! of `docs.txt`, and writes one line per query: the number of matching documents, the sum of
//! their ids and the ten smallest ids (spec §4). A query is a boolean expression evaluated
//! strictly left to right, or an exact phrase:
//!
//! ```text
//! BOOL estate OR surety AND bail
//! PHRASE "writ of habeas corpus"
//! ```
//!
//! ## The pipeline
//!
//! 1. **Queries first.** We read `queries.txt` before the corpus (spec §3 allows it) and give
//!    every distinct query term a small id ([`TermTable`]). A corpus token that no query uses
//!    is looked up once and then forgotten.
//! 2. **Triggers.** Tokenizing the first 1/[`SAMPLE_SHARE`] of the corpus estimates how many
//!    documents contain each term ([`sample_hits`]). Every query then gets *triggers*, terms
//!    such that each matching document contains at least one of them ([`Plan`]): a phrase's
//!    rarest term, or a BOOL query's [`bool_cover`]. Triggers decide how much work is done,
//!    never the answer. A BOOL query whose cover holds a common term would be checked on too
//!    many documents, so it is evaluated on whole sets instead.
//! 3. **One pass on every thread.** Each thread scans a chunk of whole lines ([`scan`]), and
//!    [`walk`] finds the tokens 64 bytes at a time from AVX2 bit masks. At the end of each
//!    document, every query its triggers point at is checked against that document alone: a
//!    phrase at each position of its trigger term, a BOOL query by one lookup in a truth table
//!    of which of its terms this document contains ([`Check`]). Set operations act on each
//!    document separately, so this gives the same answer as combining whole sets (spec §3).
//!    For the queries on sets, the pass records which documents of the current [`Window`]
//!    hold their terms, a bitset per common term and an id list per rare one, and evaluates
//!    the queries at the window's end.
//! 4. **Merge.** Per query, the chunks' counts and id sums add up, and the smallest ids come
//!    from the earliest chunks ([`format_range`]).
//!
//! Memory stays at a few arrays per thread: per query a count, an id sum and the first
//! matches, and per kept term one window's documents. Numbers: PERF.md.

// The AVX2 intrinsics of `classify` and `Check::matches`, and the prefetch of `prefetch`.
use std::{arch::asm, arch::x86_64::*, fs::File, io::Write, os::fd::AsRawFd, path::Path, thread};
// The atomics through which every thread interns its query terms ([`TermTable::intern`]).
use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::{OnceLock, atomic::AtomicU32, atomic::AtomicU64, atomic::Ordering::Relaxed};

/// The first `1 / SAMPLE_SHARE` of the corpus is tokenized up front to estimate how rare each
/// query term is.
const SAMPLE_SHARE: usize = 64;
/// An output line lists at most this many ids (spec §4).
const SHOWN_IDS: usize = 10;
/// The low bits of a [`Tally`] cell hold the sum of the ids matched in a chunk, which
/// [`answer_all`] keeps below `2^SUM_BITS`; the high bits hold their count.
const SUM_BITS: u32 = 40;
/// A term is common when at least one sampled document in this many holds it: as a list of
/// 16-bit offsets ([`Window`]), it would then take at least a third of the memory of its bitset.
const DENSE_SHARE: usize = 48;
/// The queries on sets are evaluated every `64 * WINDOW` documents, on bitsets of `WINDOW`
/// words ([`Window`]). Its id lists hold 16-bit offsets, so it spans at most 2^16 documents.
const WINDOW: usize = 64;
const _: () = assert!(64 * WINDOW <= 1 << 16);

unsafe extern "C" {
    /// libc's `mmap(2)`, used with `PROT_READ` (1) and `MAP_PRIVATE` (2) in [`map_file`]. The
    /// standard library already links libc, so declaring the functions costs no dependency.
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    /// glibc's `mallopt(3)`, used once, with `M_MMAP_THRESHOLD` (-3).
    fn mallopt(param: i32, value: i32) -> i32;
    /// libc's `madvise(2)`, with `MADV_DONTNEED` (4), `MADV_HUGEPAGE` (14), `MADV_NOHUGEPAGE` (15)
    /// and `MADV_COLLAPSE` (25).
    fn madvise(addr: *const u8, len: usize, advice: i32) -> i32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // glibc raises its mmap threshold to the largest block freed so far, and blocks below it
    // then stay resident after they are freed. Fixed, every block of 128 KiB or more goes back
    // to the kernel when freed, such as each thread's first matches once they are grouped.
    // SAFETY: no other thread exists yet, and mallopt takes no pointers.
    unsafe { mallopt(-3, 128 << 10) }; // M_MMAP_THRESHOLD
    let mut args = std::env::args().skip(1);
    let data_dir = args.next().unwrap_or_else(|| "/data".into());
    let out = Path::new(&args.next().unwrap_or_else(|| "/out".into())).join("results.txt");
    let file = |name| map_file(&Path::new(&data_dir).join(name));
    let parts = answer_all(file("queries.txt")?, file("docs.txt")?)?;
    let mut results = File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    (parts.iter()).try_for_each(|part| results.write_all(part))?;
    // Exit without running destructors: freeing the output's pieces would only add to the timed
    // wall. `File` keeps no user-space buffer, so every byte is already written.
    std::process::exit(0)
}

/// Maps the whole file read-only. The page cache is pre-warmed, so this copies nothing and
/// charges almost nothing to our memory cgroup, unlike reading it into the heap.
fn map_file(path: &Path) -> Result<&'static [u8], String> {
    let err = |e| format!("{}: {e}", path.display());
    let file = File::open(path).map_err(err)?;
    let len = file.metadata().map_err(err)?.len() as usize;
    // SAFETY: a new read-only private mapping of an open fd; the kernel picks the address. An
    // empty file maps one byte past its end, never read, as mmap rejects a zero length.
    let ptr = unsafe { mmap(std::ptr::null_mut(), len.max(1), 1, 2, file.as_raw_fd(), 0) };
    if ptr.addr() == usize::MAX {
        return Err(err(std::io::Error::last_os_error()));
    }
    // SAFETY: the mapping holds `len` readable bytes and is never unmapped, so it outlives every
    // borrow. /data is mounted read-only, so the bytes cannot change under us.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Returns the output in pieces, one per thread, to be written in order.
fn answer_all(query_text: &[u8], docs: &'static [u8]) -> Result<Vec<Vec<u8>>, String> {
    if docs.last().is_some_and(|&b| b != b'\n') {
        return Err("docs.txt: the last line has no LF (spec §2.1)".into());
    }
    let threads = thread::available_parallelism().map_or(1, usize::from);
    let (mut queries, terms) = parse_queries(query_text, threads)?;
    // The plan is never freed: the process exits once the output is written ([`main`]), and the
    // exit unmaps its blocks faster than freeing did (PERF.md, step 38); the chunks, below.
    let plan = std::mem::ManuallyDrop::new(Plan::new(&mut queries, terms, docs, threads));
    let bounds = line_bounds(docs, threads);
    // Line `i` holds id `i` (spec §2.1), so the id at a chunk's first line numbers all of its
    // lines, and the id after the last line is the number of documents.
    let firsts = (bounds.iter().map(|&at| id_at(docs, at))).collect::<Result<Vec<_>, _>>()?;
    // A query's id sum over a chunk stays below the square of the document count.
    // shortcut: 2^20 or more documents stop here; the spec's full scale has 250,000. Widen the
    // cells to u128 if a corpus that large must run.
    if (firsts.last()).is_some_and(|&n| u64::from(n).pow(2) >> SUM_BITS != 0) {
        return Err("docs.txt: too many documents".into());
    }
    let chunks = std::mem::ManuallyDrop::new(parallel(threads, &|c| {
        let (text, ids) = (&docs[bounds[c]..bounds[c + 1]], firsts[c]..firsts[c + 1]);
        scan(text, ids, &queries, &plan)
    }));
    let range = |t| queries.len() * t / threads..queries.len() * (t + 1) / threads;
    let parts = parallel(threads, &|t| format_range(range(t), &chunks));
    // The chunks are freed on a thread of their own while [`main`] writes the output, and the
    // exit unmaps what is left: on one thread, freeing them took as long as the exit then saved.
    thread::spawn(move || drop(std::mem::ManuallyDrop::into_inner(chunks)));
    Ok(parts)
}

/// Runs `work(i)` for each `i` in `0..n`, each on a thread of its own, and returns the results
/// in order.
fn parallel<T: Send>(n: usize, work: &(impl Fn(usize) -> T + Sync)) -> Vec<T> {
    thread::scope(|s| {
        let threads: Vec<_> = (0..n).map(|i| s.spawn(move || work(i))).collect();
        let join = |t: thread::ScopedJoinHandle<T>| t.join().expect("a panic aborts the process");
        threads.into_iter().map(join).collect()
    })
}

/// Splits `docs` at `parts - 1` line starts, so every line lands in exactly one chunk.
fn line_bounds(docs: &[u8], parts: usize) -> Vec<usize> {
    let ends = (1..=parts).map(|i| line_end(docs, docs.len() * i / parts));
    std::iter::once(0).chain(ends).collect()
}

/// Where the line that holds byte `at` ends, past its LF. From `docs.len()`, which no line
/// holds, it finds no LF and returns `docs.len()`.
fn line_end(docs: &[u8], at: usize) -> usize {
    let newline = docs[at..].iter().position(|&b| b == b'\n');
    newline.map_or(docs.len(), |p| at + p + 1)
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    And,
    Or,
    Not,
}

/// A query term with the operator that joins it to the terms before it ([`Queries::ops`]).
type Step = (Op, u32);

/// The queries of `queries.txt` in file order, their terms in flat arrays. Duplicate lines stay
/// separate queries, since each one gets its own output line. While a thread's share of the lines
/// is parsed, each term is still its key: `Queries<Key>`.
#[derive(Default)]
struct Queries<T = u32> {
    /// Per query: a phrase rather than a BOOL query.
    phrase: Vec<bool>,
    /// Query `q`'s terms are `terms[at[q]..at[q + 1]]`, in query order.
    at: Vec<u32>,
    /// Term ids, or their keys while parsed.
    terms: Vec<T>,
    /// `ops[i]` joins `terms[i]` to the result of the query's terms before it (spec §3). A
    /// query's first term joins the empty result with OR, and so does every phrase term.
    ops: Vec<Op>,
}

impl Queries {
    fn len(&self) -> usize {
        self.phrase.len()
    }

    /// Query `q`'s terms.
    fn terms(&self, q: usize) -> &[u32] {
        &self.terms[self.at[q] as usize..self.at[q + 1] as usize]
    }

    /// Query `q`'s terms, each with the operator that joins it. The zip ends with the terms.
    fn steps(&self, q: usize) -> impl Iterator<Item = Step> {
        (self.ops[self.at[q] as usize..].iter().copied()).zip(self.terms(q).iter().copied())
    }
}

/// Parses `queries.txt` (grammar in spec §2.3) and gives each distinct term its id. Every other
/// thread parses a share of the lines and then interns its terms ([`TermTable::intern`]), while
/// this one counts the terms and sets up the table they go into.
fn parse_queries(text: &[u8], threads: usize) -> Result<(Queries, TermTable), String> {
    let parsers = threads.max(2) - 1;
    let (bounds, table) = (line_bounds(text, parsers), OnceLock::new());
    let next = AtomicU32::new(1);
    let parts = thread::scope(|s| {
        let parse = |i: usize| {
            let (text, table, next) = (&text[bounds[i]..bounds[i + 1]], &table, &next);
            s.spawn(move || {
                let part = parse_lines(text)?;
                let ids = TermTable::intern(table.wait(), &part.terms, next);
                // The keys go back to the system here, on each thread, rather than on this one.
                Ok::<_, String>((part.phrase, part.at, part.ops, ids))
            })
        };
        let parts: Vec<_> = (0..parsers).map(parse).collect();
        // A term follows every space but those before an operator, the only capitals after one. A
        // line that does not parse stops the parse, so the table holds every term interned.
        let pairs = text.iter().zip(&text[1.min(text.len())..]);
        let terms = pairs.map(|(&a, b)| u32::from(a == b' ' && !b.is_ascii_uppercase()));
        let _ = table.set(TermTable::new(terms.sum::<u32>() as usize, parsers));
        let join = |t: thread::ScopedJoinHandle<_>| t.join().expect("a panic aborts the process");
        parts.into_iter().map(join).collect::<Result<Vec<_>, _>>()
    })?;
    let mut table = table.into_inner().expect("set above");
    table.lens.truncate(next.into_inner() as usize);
    let mut queries = Queries::default();
    queries.at.push(0);
    for (phrase, at, ops, terms) in parts {
        // A part's `at` counts from its own first term.
        let base = queries.terms.len() as u32;
        (queries.at).extend(at[1..].iter().map(|&at| base + at));
        queries.phrase.extend(phrase);
        queries.ops.extend(ops);
        queries.terms.extend(terms);
    }
    Ok((queries, table))
}

/// Parses whole lines of `queries.txt`.
fn parse_lines(text: &[u8]) -> Result<Queries<Key>, String> {
    let mut lines = Queries::default();
    lines.at.push(0);
    for line in text.split_inclusive(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let malformed = || format!("queries.txt: not a query: {}", line.escape_ascii());
        parse_line(line, &mut lines).ok_or_else(malformed)?;
        lines.at.push(lines.terms.len() as u32);
    }
    Ok(lines)
}

/// The BOOL operators by name (spec §2.3).
const OPS: [(&[u8], Op); 3] = [(b"AND", Op::And), (b"OR", Op::Or), (b"NOT", Op::Not)];

/// Appends the query on `line` to `lines`, or returns `None` if the line is not one.
fn parse_line(line: &[u8], lines: &mut Queries<Key>) -> Option<()> {
    // Rejecting anything but `[a-z0-9]{1,32}` keeps a term from matching a token that differs
    // from it only in case, which the case-folding key would otherwise allow, and keeps every
    // term within one `Key`. An empty term's key is all zeros, an empty slot's mark.
    let mut push = |op, term: &[u8]| {
        let lower = term.iter().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9'));
        lines.terms.push(key(term).filter(|k| lower && k[0] != 0)?);
        lines.ops.push(op);
        Some(())
    };
    let phrase = (line.strip_prefix(b"PHRASE \"")).and_then(|r| r.strip_suffix(b"\""));
    if let Some(words) = phrase {
        (words.split(|&b| b == b' ')).try_for_each(|term| push(Op::Or, term))?;
    } else {
        // Terms sit at even positions and operators at odd ones, so a term spelled `and` is an
        // ordinary word and `AND` there is malformed.
        let mut words = line.strip_prefix(b"BOOL ")?.split(|&b| b == b' ');
        push(Op::Or, words.next()?)?;
        while let Some(op) = words.next() {
            let (_, op) = OPS.iter().find(|(name, _)| *name == op)?;
            push(*op, words.next()?)?;
        }
    }
    lines.phrase.push(phrase.is_some());
    Some(())
}

/// How the corpus pass keeps a term's documents for the queries evaluated on whole sets: in row
/// `k` of the [`Window`]'s bitsets, one bit per document, as `k`; in list `j` of its id lists,
/// as `LIST | j`; or not at all, as `NO_STORE`. A `u32`, half the cache space of an enum.
type Store = u32;
const LIST: Store = 1 << 31;
const NO_STORE: Store = Store::MAX;

/// What the corpus pass needs to know about each query term.
struct Plan {
    /// The query terms, each with the [`Slot`] flags that say what the pass does with it.
    terms: TermTable,
    /// Term `t` triggers the phrases `phrases[phrase_at[t]..phrase_at[t + 1]]`, each as (query
    /// index, position of the term in the phrase, position and id of another of its terms).
    phrase_at: Vec<u32>,
    phrases: Vec<(u32, u32, u32, u32)>,
    /// Term `t` triggers the per-document checks `checks[check_at[t]..check_at[t + 1]]`, which
    /// sit on huge pages of their own ([`huge`]), as the pass reads them at random.
    check_at: Vec<u32>,
    checks: &'static mut [Check],
    /// Per term: how its documents are kept.
    store: Vec<Store>,
    /// The number of bitset rows, counting row 0, which no term keeps, and the number of id lists.
    sizes: [usize; 2],
    /// The queries evaluated on sets ([`Window::flush`]) rather than per document, in file order.
    sets: Vec<u32>,
}

impl Plan {
    /// Picks every query's triggers and every term's [`Store`].
    ///
    /// A phrase match holds each of its terms at a fixed offset from the match's start, so
    /// checking the spans around one term's occurrences finds all of them, and the rarest
    /// term costs the fewest checks. A BOOL query whose cover holds a common term would be
    /// evaluated on a large share of the documents, so it runs on sets instead: its common
    /// terms keep a bitset and the others an id list ([`DENSE_SHARE`]). A term absent from
    /// the corpus never fires and keeps an empty list, so its queries see it as the empty
    /// set, with no special case.
    fn new(queries: &mut Queries, terms: TermTable, docs: &'static [u8], threads: usize) -> Self {
        let mut terms = terms;
        let (hits, sampled) = sample_hits(docs, &terms, threads);
        // A bitset takes a bit per document and a list 16 bits per document holding the term.
        let common = |t: u32| hits[t as usize] as usize * DENSE_SHARE >= sampled.max(1);
        // The common terms take the ids from 1 up, those of the bitset rows, so that a doc's rows
        // are the first words of its term bits ([`Window::group`]); the others follow them. The
        // slots take theirs as they are marked, below, and only a long term's key is read.
        let (mut old, rest): (Vec<u32>, Vec<_>) =
            (0..terms.len() as u32).partition(|&t| t == 0 || common(t));
        let rows = old.len();
        old.extend(rest);
        let mut id = vec![0; old.len()];
        (old.iter().enumerate()).for_each(|(k, &t)| id[t as usize] = k as u32);
        let hits: Vec<_> = old.iter().map(|&t| hits[t as usize]).collect();
        terms.lens = old.iter().map(|&t| terms.lens[t as usize]).collect();
        let long = (1..old.len()).filter(|&k| terms.lens[k] > 12);
        let long: Vec<_> = long.map(|k| (k, terms.keys[old[k] as usize])).collect();
        for (k, key) in long {
            terms.keys[k] = key;
        }
        queries.terms.iter_mut().for_each(|t| *t = id[*t as usize]);
        // Fewer sample hits cost less. Among equals (often zero hits), longer terms tend to be
        // rarer, so they cost a little less: 32 (`MAX_TERM`) less their length.
        let cost = |t: u32| 64 * hits[t as usize] + 32 - u32::from(terms.lens[t as usize]);
        // Renumbered, the common terms are those below `rows` (0 is no query's term).
        let common = |t: u32| (t as usize) < rows;
        // Each thread plans a share of the queries: its (trigger term, phrase) and (trigger term,
        // check) pairs and its queries on sets, each in query order.
        let parts = parallel(threads, &|p| {
            let (mut phrases, mut checks, mut sets, mut cover) = (vec![], vec![], vec![], vec![]);
            for i in queries.len() * p / threads..queries.len() * (p + 1) / threads {
                if queries.phrase[i] {
                    let phrase = queries.terms(i);
                    let (offset, &term) = (phrase.iter().enumerate())
                        .min_by_key(|&(_, &t)| cost(t))
                        .expect("the parser rejects a phrase without terms");
                    // The term that [`scan`] tests first: the first, or the second if the trigger
                    // is the first and the phrase has one.
                    let other = usize::from(offset == 0).min(phrase.len() - 1);
                    phrases.push((term, (i as u32, offset as u32, other as u32, phrase[other])));
                    continue;
                }
                bool_cover(queries.steps(i), cost, &mut cover);
                let check = Check::new(i as u32, queries.steps(i), terms.len() as u32);
                if let Some(check) = check.filter(|_| !cover.iter().any(|&t| common(t))) {
                    for (k, &term) in cover.iter().enumerate() {
                        // A document holding an earlier cover term was checked from that term.
                        checks.push((term, check.unless(&cover[..k])));
                    }
                    continue;
                }
                sets.push(i as u32);
            }
            (phrases, (checks, sets))
        });
        let (phrases, (checks, sets)): (Vec<_>, (Vec<_>, Vec<_>)) = parts.into_iter().unzip();
        let n = terms.len();
        // The checks are grouped on a thread of their own, and the term table settles on another,
        // while this one groups the phrases, flags the terms, numbers the stores and then marks
        // the table, which takes as long.
        thread::scope(|s| {
            let grouping = s.spawn(|| {
                // While the plan's tail keeps three threads busy, two thirds of the threads read
                // a share of the corpus each, a byte per 64 KiB, the span that a fault on a
                // cached file page maps, so that the pass takes no faults on it (PERF.md, steps
                // 60 and 64).
                let helpers = threads * 2 / 3;
                for part in docs.chunks(docs.len() / helpers.max(1) + 1).take(helpers) {
                    thread::spawn(move || part.iter().step_by(1 << 16).fold(0, |x, &b| x ^ b));
                }
                // SAFETY: an all-zero check is a valid one.
                let grouped = unsafe { huge(checks.iter().map(Vec::len).sum()) };
                (group_into(n, &checks, grouped), grouped)
            });
            let settling = s.spawn(|| terms.settle(|t| hits[id[t.id() as usize] as usize]));
            let (phrase_at, phrases) = group(n, &phrases);
            // A term triggers phrases when its group holds any, and checks when it keys one.
            let phrase = |at: &[u32]| u32::from(at[0] < at[1]) * Slot::PHRASE;
            let mut flags: Vec<_> = phrase_at.windows(2).map(phrase).collect();
            (checks.iter().flatten()).for_each(|&(t, _)| flags[t as usize] |= Slot::BOOL);
            // Each term's store, and how many bitset rows and id lists there are so far.
            let (mut store, mut sizes) = (vec![NO_STORE; n], [rows, 0]);
            let sets = sets.concat();
            for &t in sets.iter().flat_map(|&q| queries.terms(q as usize)) {
                if store[t as usize] == NO_STORE {
                    // A common term keeps the bitset row of its id, any other the next id list.
                    let kind = usize::from(!common(t));
                    store[t as usize] = [t, sizes[1] as u32 + LIST][kind];
                    sizes[1] += kind;
                    flags[t as usize] |= [0, Slot::LISTED][kind];
                }
            }
            // Each term, old id `t`, takes its new id and that id's flags. An empty slot, id 0,
            // takes 0: a test for empty slots, which lie at random, would mispredict often.
            let flags: Vec<_> = id.iter().map(|&t| t << 5 | flags[t as usize]).collect();
            let terms = settling.join().expect("a panic aborts the process");
            (terms.slots.iter_mut())
                .for_each(|slot| slot.flags = flags[slot.id() as usize] | slot.flags & Slot::MORE);
            let (check_at, checks) = grouping.join().expect("a panic aborts the process");
            Self {
                terms,
                phrase_at,
                phrases,
                check_at,
                checks,
                store,
                sizes,
                sets,
            }
        })
    }

    /// The phrases that term `t` triggers, as (query index, position of `t` in the phrase,
    /// position and id of another of its terms).
    fn phrases(&self, t: u32) -> &[(u32, u32, u32, u32)] {
        &self.phrases[self.phrase_at[t as usize] as usize..self.phrase_at[t as usize + 1] as usize]
    }
}

/// Fills `cover` with terms such that every document matching the BOOL query `steps` holds one.
///
/// Read left to right, the first term covers `S(t1)`. `AND t` leaves a subset of both sides,
/// so the cheaper of the running cover and `t` alone will do. `OR t` adds `S(t)`, so `t`
/// joins the cover. `NOT t` only removes documents, so the cover stands. For
/// `estate OR surety AND bail` that is `{estate, surety}` or `{bail}`, whichever costs less.
fn bool_cover(steps: impl Iterator<Item = Step>, cost: impl Fn(u32) -> u32, cover: &mut Vec<u32>) {
    cover.clear();
    for (op, term) in steps {
        if op == Op::And && cost(term) < cover.iter().map(|&t| cost(t)).sum() {
            cover.clear();
        } else if op != Op::Or {
            continue;
        }
        cover.push(term);
    }
    cover.sort_unstable();
    cover.dedup();
}

/// Groups the items of `parts`, taken in order, by key, keeping their order within a key: key
/// `k`'s are `grouped[at[k]..at[k + 1]]`. A counting sort, so it allocates no list per key.
fn group<T: Copy + Default>(keys: usize, parts: &[Vec<(u32, T)>]) -> (Vec<u32>, Vec<T>) {
    let mut grouped = vec![T::default(); parts.iter().map(Vec::len).sum()];
    (group_into(keys, parts, &mut grouped), grouped)
}

/// [`group`] into `grouped`, which holds as many items as `parts`; returns `at`.
fn group_into<T: Copy>(keys: usize, parts: &[Vec<(u32, T)>], grouped: &mut [T]) -> Vec<u32> {
    // Written here, as in [`Tally::new`], so the counts' loads find no shared zero page.
    let mut at = [0].repeat(keys + 1);
    for &(k, _) in parts.iter().flatten() {
        at[k as usize] += 1;
    }
    for k in 1..=keys {
        at[k] += at[k - 1];
    }
    // Each key's end moves back to its start as its items go in, last first.
    for &(k, item) in parts.iter().flatten().rev() {
        at[k as usize] -= 1;
        grouped[at[k as usize] as usize] = item;
    }
    at
}

/// A BOOL query compiled for one document at a time: its result as a truth table over which
/// of its distinct terms the document holds, so that checking it takes no branch.
#[derive(Clone, Copy, Default)]
struct Check {
    /// Bit `x` is the query's result on a document that holds `terms[j]` exactly for the bits
    /// `j` set in `x`.
    table: u64,
    q: u32,
    /// The query's distinct terms, padded with a term id that no document holds.
    terms: [u32; CHECK_TERMS],
}

/// The most distinct terms a [`Check`] takes; spec §2.3 allows 6 terms per BOOL query.
const CHECK_TERMS: usize = 6;

impl Check {
    /// Compiles query `q`, or returns `None` when it has too many distinct terms. `pad` is a
    /// term id that no document holds.
    ///
    /// On a single document the set operations become boolean ones, applied strictly left to
    /// right (spec §3): `AND` is `&&`, `OR` is `||` and `NOT` is `&& !`.
    ///
    /// Bit `x` of `HAS[j]` is set when input `x` holds `terms[j]`, so applying the operators
    /// to those masks evaluates the query on all 64 inputs at once.
    fn new(q: u32, steps: impl Iterator<Item = Step>, pad: u32) -> Option<Self> {
        let (mut terms, mut n, mut table) = ([pad; CHECK_TERMS], 0, 0);
        for (op, t) in steps {
            let j = terms[..n].iter().position(|&u| u == t).unwrap_or(n);
            *terms.get_mut(j)? = t;
            n = n.max(j + 1);
            table = match op {
                Op::And => table & HAS[j],
                Op::Or => table | HAS[j],
                Op::Not => table & !HAS[j],
            };
        }
        Some(Self { table, q, terms })
    }

    /// This check, matching only documents that hold none of `skip`.
    fn unless(mut self, skip: &[u32]) -> Self {
        for (j, _) in (self.terms.iter().enumerate()).filter(|(_, t)| skip.contains(t)) {
            self.table &= !HAS[j];
        }
        self
    }

    /// Whether a document matches, given `seen`, where bit `t` is set when it holds term `t`.
    /// One AVX2 gather loads the 32-bit words that hold the six terms' bits, and a shift moves
    /// each bit to the top of its lane, where `movemask` collects them as the table's index.
    #[target_feature(enable = "avx2")]
    fn matches(&self, seen: &[u64]) -> bool {
        let lanes = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
        // SAFETY: the mask reads the six lanes of `terms` only, and zeroes the other two.
        let terms = unsafe { _mm256_maskload_epi32(self.terms.as_ptr().cast(), lanes) };
        let at = _mm256_srli_epi32::<5>(terms);
        // SAFETY: word `t / 32` lies inside `seen`, which `scan` sizes past every id and the pad.
        let words = unsafe { _mm256_i32gather_epi32::<4>(seen.as_ptr().cast(), at) };
        let top = _mm256_sllv_epi32(words, _mm256_andnot_si256(terms, _mm256_set1_epi32(31)));
        // The two zeroed lanes hold term 0's bit; the mask drops them.
        self.table >> (_mm256_movemask_ps(_mm256_castsi256_ps(top)) & 63) & 1 != 0
    }
}

/// Counts, for each term, the documents in the first `1 / SAMPLE_SHARE` of the corpus that
/// contain it, and returns those counts with the number of documents sampled: a cheap estimate
/// of document frequency. Each of `threads` threads counts a part of the sample.
fn sample_hits(docs: &[u8], terms: &TermTable, threads: usize) -> (Vec<u32>, usize) {
    let sample = &docs[..line_end(docs, docs.len() / SAMPLE_SHARE)];
    let bounds = line_bounds(sample, threads);
    // Each thread adds the sum so far into its part as soon as it is counted, and the part
    // becomes the sum: freed one by one on a single thread after the last part, the parts took
    // longer than the sums. A part's pages are written already, where a new zeroed vector would
    // fault twice on each page, once for the load and again for the store. The old sum is freed
    // once the lock is released, as `part` outlives the guard.
    let sum = std::sync::Mutex::new((Vec::new(), 0));
    parallel(threads, &|i| {
        let (mut part, docs) = count_hits(&sample[bounds[i]..bounds[i + 1]], terms);
        let mut sum = sum.lock().expect("a panic aborts the process");
        std::mem::swap(&mut sum.0, &mut part);
        sum.0.iter_mut().zip(&part).for_each(|(h, p)| *h += p);
        sum.1 += docs;
    });
    sum.into_inner().expect("a panic aborts the process")
}

/// [`sample_hits`] on one part of the sample.
fn count_hits(sample: &[u8], terms: &TermTable) -> (Vec<u32>, usize) {
    // Per term, 2^16 times the number of documents that hold it, plus the number of the last of
    // them, so that a document counts once, in one access per token. The low half starts at
    // 0xFFFF, no document's number but the 65,536th. Written when allocated, as in
    // [`Tally::new`], so that no page faults twice.
    // shortcut: a part of 2^16 documents or more wraps the numbers and counts, which only skews
    // the estimates; the most documents `answer_all` takes, 2^20, sample about 2^14. Widen the
    // cells to u64 if parts grow that large.
    let (mut cells, mut doc) = ([0xFFFF].repeat(terms.len()), 0);
    // A line's tokens are probed at its end, as in [`scan`].
    walk(sample, terms, |probes| {
        let low = doc as u32 & 0xFFFF;
        for (key, at) in probes {
            // A token that is no query term counts at id 0, which no query reads.
            // SAFETY: `at` is the key's home slot, from [`TermTable::prefetch`].
            let cell = &mut cells[unsafe { terms.find_from(key, *at, 0) }.id() as usize];
            *cell = (*cell & !0xFFFF | low) + (u32::from(*cell & 0xFFFF != low) << 16);
        }
        doc += 1;
    });
    (cells.into_iter().map(|cell| cell >> 16).collect(), doc)
}

/// The id of the line that starts at `at`, or the number of documents when `at` is the end of
/// `docs`: the last line's id plus one. Line `i` holds id `i` (spec §2.1), so the ids of the
/// lines after it follow by counting. An empty `docs.txt` stops here with an error: its lines
/// end in LF (spec §2.1), so it holds at least one.
fn id_at(docs: &[u8], at: usize) -> Result<u32, String> {
    if at == docs.len() && at > 0 {
        let lf = docs[..at - 1].iter().rposition(|&b| b == b'\n');
        return Ok(id_at(docs, lf.map_or(0, |lf| lf + 1))? + 1);
    }
    let digits = docs[at..].iter().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || docs.get(at + digits) != Some(&b'\t') {
        return Err(format!("docs.txt: no id and TAB at byte {at}"));
    }
    Ok((docs[at..at + digits].iter()).fold(0, |id, &digit| id * 10 + u32::from(digit - b'0')))
}

/// Calls `on_line_end(probes)` at every line end in `chunk`, in order, where `probes` holds each
/// token of the line's document as its key and its home slot in `terms`, whose load is under
/// way by then ([`TermTable::prefetch`]). `chunk` holds whole lines, the last one ending in LF.
///
/// No loop here looks at single bytes. [`classify`] turns 64 bytes at a time into two masks:
/// `token`, with bit `i` set when byte `i` is a letter or digit, and `lf`. A token starts where
/// a token byte follows a separator and ends at the separator after its last byte. With `prev`
/// the `token` mask shifted by one byte (the previous block's last bit shifted in), that is:
///
/// ```text
/// bytes    T h e   s u r e t y ,   b a i l
/// token    1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1
/// starts   1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0    token & !prev
/// ends     0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0    !token & prev
/// ```
///
/// `trailing_zeros` walks the set bits. Starts and ends alternate, so each end closes the
/// oldest open start, and a token still open at a block's end carries into the next block.
/// A line starts with its id, digits that look like a token, so a line's first token is left
/// out: document text only begins after the TAB.
fn walk(chunk: &[u8], terms: &TermTable, mut on_line_end: impl FnMut(&[(Key, *const Slot)])) {
    let (mut padded, tail) = ([0; 64], chunk.as_chunks::<64>().1);
    padded[..tail.len()].copy_from_slice(tail);
    // Before the first block: block -1, which holds nothing.
    let at = (0usize.wrapping_sub(64), 0, 0, 0, 0, 0);
    let (mut tokens, mut probes) = (Tokens { chunk, padded, at }, Vec::new());
    while let Some(n) = tokens.line(terms, &mut probes) {
        on_line_end(probes.get(1..n).unwrap_or_default());
    }
}

/// Where [`walk`] is in its chunk.
struct Tokens<'a> {
    chunk: &'a [u8],
    /// The last block, short of 64 bytes, padded with zero bytes, which are separators.
    padded: [u8; 64],
    /// The current block's offset in `chunk`; its token starts, token ends and LFs not taken
    /// yet; the start of the token that its next end closes, as an offset from the block's, below
    /// 0, wrapped, for a token that began in an earlier block; and 1 when its last byte is a token
    /// byte.
    at: (usize, u64, u64, u64, usize, u64),
}

impl Tokens<'_> {
    /// Writes the next line's tokens to `probes`, each as its key and home slot, and returns
    /// their count, or `None` past the last line. A call per line, never inlined: inlined into
    /// [`scan`]'s loop, the token loop kept 4 of its values on the stack, as the line end's held
    /// the registers (PERF.md, step 42).
    #[inline(never)]
    fn line(&mut self, terms: &TermTable, probes: &mut Vec<(Key, *const Slot)>) -> Option<usize> {
        let (chunk, mut n) = (self.chunk, 0);
        let (mut base, mut starts, mut ends, mut lfs, mut next, mut prev_bit) = self.at;
        loop {
            // Room for 32 more tokens, the most that one block ends, so that the slice's address
            // and `n` stay in registers: a `Vec` pushed per token was reloaded and stored after
            // each push (PERF.md, step 41).
            probes.resize(probes.len().max(n + 32), (Key::default(), std::ptr::null()));
            let room = &mut probes[..];
            // A token never spans an LF, so the ends up to the block's next LF, the LF's own bit
            // included, close the line's tokens. Without an LF, `lfs ^ (lfs - 1)` takes all of
            // them. So the token loop tests no LF (PERF.md, step 41).
            let upto = lfs ^ lfs.wrapping_sub(1);
            let mut line = ends & upto;
            ends &= !upto;
            while line != 0 {
                // The token's end and start as offsets from `base`, which takes no add for either.
                let end = pop(&mut line);
                let start = std::mem::replace(&mut next, starts.trailing_zeros() as usize);
                starts &= starts.wrapping_sub(1);
                // A token past `MAX_TERM` bytes takes a key of 0xFF bytes, which no term has.
                let key = key_in(chunk, base, start, end).unwrap_or([!0; MAX_TERM / 8]);
                // SAFETY: `n` has grown by at most one per end since `room` held 32 entries more
                // than `n`, and those ends lie in one block, which holds at most 32, as each
                // follows a token byte.
                *unsafe { room.get_unchecked_mut(n) } = (key, terms.prefetch(&key));
                n += 1;
            }
            if lfs != 0 {
                lfs &= lfs - 1;
                break;
            }
            base = base.wrapping_add(64);
            // Past the end of `chunk`, which ends in LF, no line is left.
            let block = chunk.get(base..)?.first_chunk().unwrap_or(&self.padded);
            // SAFETY: the binary targets alderlake (.cargo/config.toml), which always has AVX2.
            let (token, lf) = unsafe { classify(block) };
            let prev = token << 1 | prev_bit;
            (starts, ends, lfs) = (token & !prev, !token & prev, lf);
            // The start of the token that the block's next end closes: the one open since an
            // earlier block, if any, which `next` holds, else the block's first start. It is
            // picked once per block, by a conditional move, as whether a token runs on follows no
            // pattern, so that the loop above takes no test on it. An open token's start moves 64
            // back, as the offsets count from the new block.
            let (first, carry) = (starts.trailing_zeros() as usize, prev_bit != 0);
            next = std::hint::select_unpredictable(carry, next.wrapping_sub(64), first);
            starts =
                std::hint::select_unpredictable(carry, starts, starts & starts.wrapping_sub(1));
            prev_bit = token >> 63;
        }
        self.at = (base, starts, ends, lfs, next, prev_bit);
        Some(n)
    }
}

/// Transposes the 64 × 64 bit matrix `a`: bit `c` of `a[r]` moves to bit `r` of `a[c]`. Each step
/// swaps the off-diagonal blocks of every `2j × 2j` block, `j` from 32 down to 1.
fn transpose(a: &mut [u64; 64]) {
    for j in [32, 16, 8, 4, 2, 1] {
        // The low `j` bits of every `2j`; the two runs of `j` words side by side vectorize.
        let m = u64::MAX / ((1 << j) + 1);
        for pair in a.chunks_exact_mut(2 * j) {
            let (low, high) = pair.split_at_mut(j);
            for (x, y) in low.iter_mut().zip(high) {
                let t = (*x >> j ^ *y) & m;
                (*x, *y) = (*x ^ t << j, *y ^ t);
            }
        }
    }
}

/// Clears the lowest set bit of `bits`, which is not 0, and returns its position.
fn pop(bits: &mut u64) -> usize {
    std::mem::replace(bits, *bits & (*bits - 1)).trailing_zeros() as usize
}

/// Returns `(token, lf)` for a 64-byte block: bit `i` of `token` is set when `block[i]` is a
/// token byte, an ASCII letter or digit (spec §2.2), and bit `i` of `lf` when it is an LF.
///
/// Each half of the block is one AVX2 register. A byte is a letter when `(b | 0x20) - b'a'`
/// is at most 25 as an unsigned byte, and a digit when `b - b'0'` is at most 9, so every byte
/// from 0x80 up, TAB and LF included, is a separator. `_mm256_movemask_epi8` then packs the
/// top bit of each byte's 0x00 or 0xFF result into one bit.
#[target_feature(enable = "avx2")]
fn classify(block: &[u8; 64]) -> (u64, u64) {
    let (mut token, mut lf) = (0, 0);
    for offset in [0, 32] {
        // SAFETY: the 32-byte unaligned load lies inside the 64-byte block.
        let bytes = unsafe { _mm256_loadu_si256(block.as_ptr().add(offset).cast()) };
        let lower = _mm256_or_si256(bytes, _mm256_set1_epi8(0x20));
        let letter = _mm256_sub_epi8(lower, _mm256_set1_epi8(b'a'.cast_signed()));
        let letter = _mm256_cmpeq_epi8(_mm256_min_epu8(letter, _mm256_set1_epi8(25)), letter);
        let digit = _mm256_sub_epi8(bytes, _mm256_set1_epi8(b'0'.cast_signed()));
        let digit = _mm256_cmpeq_epi8(_mm256_min_epu8(digit, _mm256_set1_epi8(9)), digit);
        let newline = _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(b'\n'.cast_signed()));
        let is_token = _mm256_movemask_epi8(_mm256_or_si256(letter, digit)).cast_unsigned();
        token |= u64::from(is_token) << offset;
        lf |= u64::from(_mm256_movemask_epi8(newline).cast_unsigned()) << offset;
    }
    (token, lf)
}

/// One chunk's answers: how many of its documents match each query, the sum of their ids, and
/// the first `SHOWN_IDS` of them.
struct Tally {
    /// Per query, a cell: its matches as `count << SUM_BITS | id sum`, one load for both.
    queries: Vec<u64>,
    /// Per check ([`Plan::checks`]), a cell as above, next to its neighbours in the check list.
    checks: Vec<u64>,
    /// Per query, how many first matches its checks recorded: a byte, which stays in cache where
    /// the query's cell would not.
    shown: Vec<u8>,
    /// (query index, doc id) of each query's first matches in the chunk, in doc order.
    first: Vec<(u32, u32)>,
}

impl Tally {
    fn new(queries: usize, checks: usize) -> Self {
        // Written here, each page of cells faults in once, as a page of its own. Zeroed by
        // calloc, as `vec!` does, the pass's first load from a page maps the shared zero page,
        // and the store after that load replaces it, which interrupts every core to flush its
        // TLB.
        Self {
            queries: [0].repeat(queries),
            checks: [0].repeat(checks),
            shown: vec![0; queries],
            first: Vec::new(),
        }
    }

    /// Records that `doc` matches query `q`. Docs arrive in id order, so the first matches a
    /// chunk records are its smallest ids.
    fn add(&mut self, q: u32, doc: u32) {
        let cell = &mut self.queries[q as usize];
        *cell += 1 << SUM_BITS | u64::from(doc);
        if *cell >> SUM_BITS <= SHOWN_IDS as u64 {
            self.first.push((q, doc));
        }
    }

    /// Records which of the checks `todo`, indices into `checks`, `doc` matches, given `seen`,
    /// with no branch on a check's result: it follows no pattern, and it comes at the end of a
    /// chain of loads. The queries of the checks that `doc` is one of the first matches of
    /// overwrite the front of `todo`, and only those then go on to the first matches. A
    /// method, so the compiler knows that the stores to the cells and to `todo` leave `self`
    /// alone.
    ///
    /// # Safety
    ///
    /// Every index in `todo` is below `checks.len()`.
    unsafe fn add_checks(&mut self, checks: &[Check], todo: &mut [u32], seen: &[u64], doc: u32) {
        // A cell per check ([`Tally::new`]), so a check's index is its cell's.
        let cells = &mut self.checks[..checks.len()];
        let (add, mut firsts) = (1 << SUM_BITS | u64::from(doc), 0);
        for i in 0..todo.len() {
            let c = todo[i] as usize;
            // SAFETY: `c < checks.len()`, as the caller ensures, the length of `cells` too.
            let (check, cell) = unsafe { (checks.get_unchecked(c), cells.get_unchecked_mut(c)) };
            // SAFETY: the binary targets alderlake (.cargo/config.toml), which has AVX2.
            let hit = unsafe { check.matches(seen) };
            // A conditional move: the compiler makes a product with `hit` a branch.
            *cell += std::hint::select_unpredictable(hit, add, 0);
            // SAFETY: `firsts <= i < todo.len()`, as `firsts` moves on at most once per check.
            unsafe { *todo.get_unchecked_mut(firsts) = check.q };
            firsts += usize::from(hit && *cell >> SUM_BITS <= SHOWN_IDS as u64);
        }
        for &q in &todo[..firsts] {
            if usize::from(self.shown[q as usize]) < SHOWN_IDS {
                self.shown[q as usize] += 1;
                self.first.push((q, doc));
            }
        }
    }

    /// Records the documents `start + i` for each bit `i` set in `bits`, as [`Tally::add`] would
    /// one by one. So it does until the query has its first matches; the rest are counted and
    /// summed four words at a time ([`window_cell`]).
    fn add_bits(&mut self, q: u32, start: u32, mut bits: [u64; WINDOW]) {
        let mut i = 0;
        while i < WINDOW && self.queries[q as usize] >> SUM_BITS < SHOWN_IDS as u64 {
            if bits[i] != 0 {
                self.add(q, start + 64 * i as u32 + pop(&mut bits[i]) as u32);
            }
            i += usize::from(bits[i] == 0);
        }
        // SAFETY: the binary targets alderlake (.cargo/config.toml), which has AVX2.
        self.queries[q as usize] += unsafe { window_cell(start, &bits) };
    }

    /// Takes the first matches out, grouped by query and each in doc order: query `q`'s are
    /// `ids[at[q]..at[q + 1]]`. Each scanning thread groups its own, instead of one big sort
    /// after the pass.
    fn take_first(&mut self) -> (Vec<u32>, Vec<u32>) {
        group(self.queries.len(), &[std::mem::take(&mut self.first)])
    }
}

/// What the corpus pass keeps from one chunk of whole lines.
struct Chunk {
    tally: Tally,
    /// Each query's first matches in the chunk, smallest first: query `q`'s are
    /// `ids[at[q]..at[q + 1]]`.
    at: Vec<u32>,
    ids: Vec<u32>,
}

/// The documents of one window of `64 * WINDOW` that contain the terms of the queries on sets
/// ([`Store`]). Only one window is kept at a time, so these sets take the same memory however
/// long the chunk is.
struct Window {
    /// The window's first document.
    start: u32,
    /// The bitset rows: bit `i` of row `k` is set when document `start + i` holds row `k`'s term,
    /// whose id is `k` ([`Plan::new`]). A row is kept per id below the next multiple of 64; those
    /// that are no row's are never read.
    bits: Vec<[u64; WINDOW]>,
    /// The first words of the term bits of the current 64 documents, which hold their rows' bits:
    /// word `j` of document `d` is `group[j][d]`, so a doc's rows take a store per word instead
    /// of a store per row and a load of its term's store.
    group: Vec<[u64; 64]>,
    /// The id lists' entries in the order the documents add them, list `j` holding document
    /// `start + i` as `j << 5 | i % 32`: a push at the end of one log, not into one of thousands
    /// of lists whose ends lie at random, each a cache miss (PERF.md, step 57). `j` is below the
    /// term count, below 2^27 ([`TermTable::intern`]).
    log: Vec<u32>,
    /// Where the log's entries of documents `start + 32 * b` to `start + 32 * b + 31` end, `b`
    /// past the last document 0.
    ends: [u32; 2 * WINDOW],
    /// The id lists, the log sorted by list in [`Window::flush`]: list `j` is
    /// `ids[at[j]..at[j + 1]]`, the window's documents that contain its term, ascending, as
    /// offsets from `start`, which take half the memory of ids.
    at: Vec<u32>,
    ids: Vec<u16>,
}

impl Window {
    /// Moves the current 64 documents' bits into word `g` of the rows, and zeroes them for the
    /// next 64: [`transpose`] turns the docs' words `j` into those of rows `64 * j` and up.
    fn spill(&mut self, g: usize) {
        for (j, block) in self.group.iter_mut().enumerate() {
            transpose(block);
            let (rows, words) = (&mut self.bits[64 * j..], std::mem::replace(block, [0; 64]));
            (rows.iter_mut().zip(words)).for_each(|(row, word)| row[g] = word);
        }
    }

    /// The window's documents that contain a term kept as `store`, as a bitset. A list is
    /// spread into `scratch` first.
    fn row<'a>(&'a self, store: Store, scratch: &'a mut [u64; WINDOW]) -> &'a [u64; WINDOW] {
        if store & LIST == 0 {
            return &self.bits[store as usize];
        }
        *scratch = [0; WINDOW];
        let j = (store ^ LIST) as usize;
        for &i in &self.ids[self.at[j] as usize..self.at[j + 1] as usize] {
            scratch[usize::from(i / 64)] |= 1 << (i % 64);
        }
        scratch
    }

    /// Evaluates every query on sets over the window, strictly left to right (spec §3), adds
    /// the matches to `tally`, and moves on to the next window.
    ///
    /// On bitsets, AND, OR and NOT are the word-wise `&`, `|` and `& !`, 64 documents per
    /// operation.
    fn flush(&mut self, queries: &Queries, plan: &Plan, tally: &mut Tally) {
        // A counting sort: list `j`'s count goes to `at[j + 2]`, the sums then make `at[j + 1]`
        // its start, and each entry moves it on, to its end and the start of list `j + 1`.
        self.at.fill(0);
        (self.log.iter()).for_each(|&e| self.at[(e >> 5) as usize + 2] += 1);
        (1..self.at.len()).for_each(|j| self.at[j] += self.at[j - 1]);
        self.ids.resize(self.log.len(), 0);
        let mut from = 0;
        for (b, end) in self.ends.iter_mut().enumerate() {
            let to = std::mem::take(end).max(from);
            for &e in &self.log[from as usize..to as usize] {
                let end = &mut self.at[(e >> 5) as usize + 1];
                self.ids[*end as usize] = (32 * b) as u16 | (e % 32) as u16;
                *end += 1;
            }
            from = to;
        }
        let mut scratch = [0; WINDOW];
        for &q in &plan.sets {
            let mut acc = [0u64; WINDOW];
            for (op, term) in queries.steps(q as usize) {
                let row = self.row(plan.store[term as usize], &mut scratch);
                let pairs = acc.iter_mut().zip(row);
                match op {
                    Op::And => pairs.for_each(|(a, &w)| *a &= w),
                    Op::Or => pairs.for_each(|(a, &w)| *a |= w),
                    Op::Not => pairs.for_each(|(a, &w)| *a &= !w),
                }
            }
            tally.add_bits(q, self.start, acc);
        }
        self.bits.fill([0; WINDOW]);
        self.log.clear();
        self.start += 64 * WINDOW as u32;
    }
}

/// Runs the corpus pass over `text`, whole lines holding the documents `ids`.
///
/// A token costs one probe of the term table, whose [`Slot`] says what to do with the term,
/// and otherwise only touches the current document's state: which terms it contains (one bit
/// each, small enough to stay in L1), and the [`Window`] of the terms that keep their
/// documents. At each line end, the phrases triggered in the document are checked at each
/// position of their trigger term, and the per-document checks of the BOOL terms it contains
/// are run ([`Check`]).
fn scan(text: &[u8], ids: std::ops::Range<u32>, queries: &Queries, plan: &Plan) -> Chunk {
    let mut sets = Window {
        start: ids.start,
        bits: vec![[0; WINDOW]; plan.sizes[0].next_multiple_of(64)],
        group: vec![[0; 64]; plan.sizes[0].div_ceil(64)],
        log: Vec::new(),
        ends: [0; 2 * WINDOW],
        at: vec![0; plan.sizes[1] + 2],
        ids: Vec::new(),
    };
    let mut tally = Tally::new(queries.len(), plan.checks.len());
    // Bit `t` is set while the current doc contains term `t`, bit 0 while it holds a token that
    // is no query term. One more bit, never set, pads the checks ([`Check`]).
    let mut seen = vec![0u64; (plan.terms.len() + 1).div_ceil(64)];
    // The token loop below reads the words of `seen` without bounds checks.
    assert!(64 * seen.len() > plan.terms.len());
    // The checks of the current doc's BOOL triggers, from the start: each term triggers once per
    // doc, and no two terms share a check, so they are at most all the checks, plus 4, as a
    // trigger's four writes below may run past its own.
    let mut todo = vec![0; plan.checks.len() + 4];
    // `found`: the phrases the current doc matched, so none counts twice. `kept`: for the current
    // doc's `n` tokens, four runs of `n` entries: each token's term id (0 for the rest), then the
    // doc's distinct terms kept in lists and BOOL triggers, and the positions of its phrase
    // triggers, up to the cursors `lists`, `bools` and `phrases`. Keeping a term
    // takes a store and an add, not a branch, and one slice without bounds checks leaves the loop
    // registers to spare.
    let (mut found, mut kept, mut doc) = (Vec::new(), Vec::new(), ids.start);
    // The current doc's tokens are probed only at the line end: by then every slot load is long
    // under way ([`TermTable::prefetch`]), instead of each stalling in turn.
    walk(text, &plan.terms, |probes| {
        let n = probes.len();
        kept.resize(kept.len().max(4 * n + 32), 0);
        let (kept, table, seen) = (&mut kept[..4 * n + 32], &plan.terms, &mut seen[..]);
        // A zero the optimizer cannot see, held in a register for the token loop's selects: with
        // a constant 0, it set a register to 0 before each of two selects per token.
        let mut zero = 0u64;
        // SAFETY: the asm block is empty.
        unsafe { asm!("/* {} */", inout(reg) zero, options(pure, nomem)) };
        // No branch on what a token is, a query term or not, new in the doc or not, kept in a
        // list, a BOOL or a phrase trigger: none of it follows a pattern. A phrase trigger pushed
        // to a `Vec` behind a branch held 1 M mispredicts (PERF.md, step 47).
        for (k, (key, at)) in probes.iter().enumerate() {
            // SAFETY: `at` is the key's home slot, from [`TermTable::prefetch`].
            let slot = unsafe { table.find_from(key, *at, zero) };
            let (t, bit) = (slot.id() as usize / 64, 1 << (slot.id() % 64));
            // SAFETY: a slot's id is 0 or a term's, below `plan.terms.len()`, so its word `t` lies
            // below `seen.len()`, as asserted where `seen` is sized.
            let word = unsafe { seen.get_unchecked_mut(t) };
            // The term's flags, but `LISTED` and `BOOL` only if the doc did not hold it yet: a term
            // is kept and triggers once per doc.
            let once = slot.flags & !(Slot::LISTED | Slot::BOOL);
            let flags = std::hint::select_unpredictable(*word & bit == 0, slot.flags, once);
            *word |= bit;
            // SAFETY: `k < n`, below the length of `kept`.
            unsafe { *kept.get_unchecked_mut(k) = flags };
        }
        let [lists, bools, phrases] = pack(kept, n);
        let terms = &kept[..n];
        // As locals, their addresses stay in registers across the stores to `todo`.
        let (check_at, checks, cells) = (&plan.check_at[..], &plan.checks[..], &tally.checks[..]);
        let mut triggered = 0;
        for &t in &kept[2 * n + 16..bools] {
            // Term `t` triggers the checks `checks[check_at[t]..check_at[t + 1]]`.
            // SAFETY: a trigger is a term id, below `plan.terms.len()`, and `check_at` holds one
            // entry more.
            let [start, end] = unsafe { *check_at.as_ptr().add(t as usize).cast::<[u32; 2]>() };
            // Start loading its checks and their cells now, to use below.
            prefetch(checks.as_ptr().wrapping_add(start as usize));
            prefetch(cells.as_ptr().wrapping_add(start as usize));
            // Four indices written at once, the cursor then moved on by the range's length: a
            // loop per trigger would mispredict its exit, as the lengths follow no pattern.
            // SAFETY: the triggers' ranges are disjoint, so `triggered + end - start` is at most
            // the check count, and `todo` holds 4 entries more.
            unsafe {
                let at = todo.as_mut_ptr().add(triggered);
                *at.cast::<[u32; 4]>() = [0, 1, 2, 3].map(|i| start + i);
                (4..end - start).for_each(|i| *at.add(i as usize) = start + i);
            }
            triggered += (end - start) as usize;
        }
        // As locals, their addresses stay in registers across the stores below.
        let (i, store) = ((doc - sets.start) as usize, &plan.store[..]);
        // A doc's rows are the bits of its terms 1 to `plan.sizes[0] - 1`.
        (sets.group.iter_mut().zip(&*seen)).for_each(|(block, &word)| block[i % 64] = word);
        if i % 64 == 63 {
            sets.spill(i / 64);
        }
        let entry = |&t: &u32| (store[t as usize] ^ LIST) << 5 | (i % 32) as u32;
        sets.log.extend(kept[n + 8..lists].iter().map(entry));
        sets.ends[i / 32] = sets.log.len() as u32;
        for at in kept[3 * n + 24..phrases].iter().map(|&k| k as usize) {
            for &(q, offset, other, term) in plan.phrases(terms[at]) {
                // `terms` holds this doc only, so a match never spans two docs (spec §3).
                let start = at.checked_sub(offset as usize);
                // 98% of the candidates match no phrase, and one more of its terms rules out most
                // of them before its terms are loaded: 2.5% get past it (PERF.md, step 43).
                if start.and_then(|s| terms.get(s + other as usize)) != Some(&term) {
                    continue;
                }
                let phrase = queries.terms(q as usize);
                let span = start.and_then(|s| terms.get(s..s + phrase.len()));
                // Compared inline, with no branch per term: phrases are short, and a call to
                // `memcmp` costs more than the compare.
                let differ = |s: &[u32]| s.iter().zip(phrase).fold(0, |d, (a, b)| d | (a ^ b));
                if span.is_some_and(|span| differ(span) == 0) && !found.contains(&q) {
                    found.push(q);
                    tally.add(q, doc);
                }
            }
        }
        // SAFETY: each index in `todo[..triggered]` lies in a term's range of checks, below the
        // check count.
        unsafe { tally.add_checks(checks, &mut todo[..triggered], seen, doc) };
        // Every set bit is one of the doc's terms, so zeroing their words clears `seen`: a store
        // with no load, so tokens that share a word, as all misses do, do not wait on each other.
        for &t in terms {
            // SAFETY: `terms` holds term ids and 0, all below `64 * seen.len()`.
            *unsafe { seen.get_unchecked_mut(t as usize / 64) } = 0;
        }
        found.clear();
        doc += 1;
        if doc - sets.start == 64 * WINDOW as u32 {
            sets.flush(queries, plan, &mut tally);
        }
    });
    // The chunk's pages leave our page tables now, each thread its own, instead of one by one
    // in the exit after the output is written.
    let offset = text.as_ptr().addr() % 4096;
    // SAFETY: madvise only drops pages. Those of the docs mapping, private and never written,
    // fault the same bytes back in if read again, as a neighbouring chunk's first or last page
    // may be; outside a mapping, madvise fails and changes nothing.
    unsafe { madvise(text.as_ptr().wrapping_sub(offset), offset + text.len(), 4) }; // MADV_DONTNEED
    // The last documents' rows: their 64 is not full. The words of no doc are still zero.
    sets.spill((doc - sets.start) as usize / 64);
    sets.flush(queries, plan, &mut tally);
    // The window's bitsets go back to the kernel before the first matches are grouped, when
    // memory peaks.
    drop(sets);
    // The checks' cells hold their queries' whole counts and sums.
    for (check, cell) in plan.checks.iter().zip(std::mem::take(&mut tally.checks)) {
        tally.queries[check.q as usize] += cell;
    }
    let (at, ids) = tally.take_first();
    Chunk { tally, at, ids }
}

/// Formats the answers to queries `range`, in order (spec §4).
fn format_range(range: std::ops::Range<usize>, chunks: &[Chunk]) -> Vec<u8> {
    #[inline(always)]
    fn put_number(out: &mut Vec<u8>, mut value: u64) {
        let from = out.len();
        while out.len() == from || value > 0 {
            out.push(b'0' + (value % 10) as u8);
            value /= 10;
        }
        out[from..].reverse();
        out.push(b' ');
    }

    let mut out = Vec::new();
    for q in range {
        let cell: u64 = chunks.iter().map(|c| c.tally.queries[q]).sum();
        let (count, sum) = (cell >> SUM_BITS, cell & ((1 << SUM_BITS) - 1));
        let shown = (chunks.iter()).flat_map(|c| &c.ids[c.at[q] as usize..c.at[q + 1] as usize]);
        let ids = shown.take(SHOWN_IDS).map(|&id| u64::from(id));
        for n in [count, sum].into_iter().chain(ids) {
            put_number(&mut out, n);
        }
        *out.last_mut().expect("a line holds two numbers") = b'\n';
    }
    out
}

/// Mask `k` keeps the bits whose position has bit `k` set.
const HAS: [u64; 6] = [
    0xAAAA_AAAA_AAAA_AAAA,
    0xCCCC_CCCC_CCCC_CCCC,
    0xF0F0_F0F0_F0F0_F0F0,
    0xFF00_FF00_FF00_FF00,
    0xFFFF_0000_FFFF_0000,
    0xFFFF_FFFF_0000_0000,
];

/// The cell ([`Tally::queries`]) of the documents `start + i` for each bit `i` set in `bits`.
/// Four words at a time, a lookup per nibble gives each byte's bit count and the sum of its bits'
/// positions in it, `sad` adds those per word, and a multiply-add weighs each byte's count by its
/// offset in the word, instead of six masked popcounts per word (PERF.md, step 46).
#[target_feature(enable = "avx2")]
fn window_cell(start: u32, bits: &[u64; WINDOW]) -> u64 {
    let table = |lo: i64, hi: i64| _mm256_setr_epi64x(lo, hi, lo, hi);
    let count = table(0x0302_0201_0201_0100, 0x0403_0302_0302_0201);
    let low = table(0x0303_0202_0101_0000, 0x0606_0505_0404_0303);
    // A high nibble's bits sit 4 places further on.
    let high = _mm256_add_epi8(low, _mm256_slli_epi16::<2>(count));
    let offsets = _mm256_set1_epi64x(0x3830_2820_1810_0800);
    let [mut n, mut sum] = [_mm256_setzero_si256(); 2];
    let mut first = _mm256_setr_epi64x(0, 64, 128, 192);
    let pick = |t, x| _mm256_shuffle_epi8(t, _mm256_and_si256(x, _mm256_set1_epi8(15)));
    for words in bits.as_chunks::<4>().0 {
        // SAFETY: the 32-byte load reads the four words.
        let v = unsafe { _mm256_loadu_si256(words.as_ptr().cast()) };
        let hi = _mm256_srli_epi16::<4>(v);
        let c = _mm256_add_epi8(pick(count, v), pick(count, hi));
        let p = _mm256_add_epi8(pick(low, v), pick(high, hi));
        let words_n = _mm256_sad_epu8(c, _mm256_setzero_si256());
        n = _mm256_add_epi64(n, words_n);
        // Per word: its bits' places in their bytes, its bytes' counts weighed by their offsets
        // in two 32-bit halves, and its offset in the window times its count.
        let weighed = _mm256_madd_epi16(_mm256_maddubs_epi16(c, offsets), _mm256_set1_epi16(1));
        let places = _mm256_add_epi64(_mm256_sad_epu8(p, _mm256_setzero_si256()), weighed);
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(words_n, first));
        sum = _mm256_add_epi64(sum, places);
        first = _mm256_add_epi64(first, _mm256_set1_epi64x(256));
    }
    // SAFETY: an `__m256i` is four `u64`s.
    let [n, sum] = [n, sum].map(|v| unsafe { std::mem::transmute::<__m256i, [u64; 4]>(v) });
    let [n, sum]: [u64; 2] = [n, sum].map(|lanes| lanes.iter().sum());
    // The low halves add up to less than 2^32: a window's positions do.
    n << SUM_BITS | (u64::from(start) * n + (sum & 0xFFFF_FFFF) + (sum >> 32))
}

/// Turns the [`Slot::flags`] of a doc's `n` tokens, `kept[..n]`, into their term ids, and packs
/// the doc's terms kept in lists, its BOOL triggers and the positions of its phrase triggers into
/// three runs, from `kept[n + 8]`, `kept[2 * n + 16]` and `kept[3 * n + 24]` to the ends returned:
/// 8 tokens at a time, without a branch. In the token loop, a store and an add per run and token
/// took 8 more instructions per token. A store writes 8 entries, so a run's room is 8 more.
fn pack(kept: &mut [u32], n: usize) -> [usize; 3] {
    // Entry `m` holds, from its low byte up, the positions of the bits set in `m`: its lowest
    // bit's, then those of `m` without that bit.
    const ORDER: [u64; 256] = {
        let (mut table, mut m) = ([0; 256], 1);
        while m < 256 {
            (table[m], m) = (table[m & (m - 1)] << 8 | m.trailing_zeros() as u64, m + 1);
        }
        table
    };
    let mut runs = [n + 8, 2 * n + 16, 3 * n + 24];
    // SAFETY: the binary targets alderlake (.cargo/config.toml), which has AVX2.
    let mut at = unsafe { _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7) };
    // The last 8 tokens' load reads up to 7 entries past `n`: zero, they join no run.
    kept[n..n + 8].fill(0);
    let p = kept[..4 * n + 32].as_mut_ptr();
    for k in (0..n).step_by(8) {
        // SAFETY: AVX2, as above. `kept` holds `4 * n + 32` entries; `k + 8 <= n + 8`, and each
        // run's cursor stays 8 below the next run's start, as it moves on by at most 8 per 8.
        unsafe {
            let flags = _mm256_loadu_si256(p.add(k).cast());
            let ids = _mm256_srli_epi32::<5>(flags);
            _mm256_storeu_si256(p.add(k).cast(), ids);
            let lanes = [(Slot::LISTED, ids), (Slot::BOOL, ids), (Slot::PHRASE, at)];
            for (run, (bit, from)) in runs.iter_mut().zip(lanes) {
                // The flag moved to the top bit of each lane, which `movemask` takes.
                let shift = _mm256_set1_epi32(bit.leading_zeros().cast_signed());
                let top = _mm256_castsi256_ps(_mm256_sllv_epi32(flags, shift));
                let m = _mm256_movemask_ps(top).cast_unsigned() as usize;
                let order = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ORDER.as_ptr().add(m).cast()));
                _mm256_storeu_si256(p.add(*run).cast(), _mm256_permutevar8x32_epi32(from, order));
                *run += m.count_ones() as usize;
            }
            at = _mm256_add_epi32(at, _mm256_set1_epi32(8));
        }
    }
    runs
}

/// Starts loading the cache line that holds `item` into L1. A pointer, so that callers need no
/// bounds check: a prefetch of any address is harmless.
fn prefetch<T>(item: *const T) {
    // SAFETY: a prefetch never faults or changes memory.
    unsafe { _mm_prefetch::<_MM_HINT_T0>(item.cast()) };
}

/// A token's bytes, ASCII-lowercased and zero-padded to `MAX_TERM`, as little-endian words.
/// A token never holds a zero byte, so two keys are equal exactly when the tokens are.
type Key = [u64; MAX_TERM / 8];
/// The longest query term (spec §2.3), and so the longest token worth looking up.
const MAX_TERM: usize = 32;

/// The key of a token or query term, or `None` past `MAX_TERM` bytes: no query term is longer.
fn key(token: &[u8]) -> Option<Key> {
    let mut bytes = [0; MAX_TERM];
    bytes.get_mut(..token.len())?.copy_from_slice(token);
    Some(fold(&bytes, token.len().wrapping_neg()))
}

/// The key of the token at `chunk[base + start..base + end]`, read as one 32-byte load from
/// `chunk` and masked to the token's length, or `None` past `MAX_TERM` bytes. `start` and `end`
/// are offsets from `base`, `start` below 0, wrapped, for a token that began in an earlier
/// block. Copying the token into a zeroed buffer first would call `memcpy` and then stall the
/// load that reads the buffer back, once per token.
fn key_in(chunk: &[u8], base: usize, start: usize, end: usize) -> Option<Key> {
    // One test for the start and the 32 bytes from it, against a bound that stays in a register
    // within a block, signed, as `start` may be below 0: `chunk.get(start..start + MAX_TERM)`
    // also tested whether the sum wraps around.
    let room = chunk.len().saturating_sub(MAX_TERM - 1).wrapping_sub(base);
    if start.cast_signed() < room.cast_signed() {
        let block = chunk.as_ptr().wrapping_add(base);
        // SAFETY: `base + start + MAX_TERM <= chunk.len()`, as just tested.
        let window = unsafe { &*block.wrapping_add(start).cast::<[u8; MAX_TERM]>() };
        // Minus the length, so that the mask's address is one add. Past `MAX_TERM` bytes, on a
        // cold branch: else each token set a register to the key of 0xFF bytes, just in case.
        let neg = start.wrapping_sub(end);
        if neg < MAX_TERM.wrapping_neg() {
            std::hint::cold_path();
            return None;
        }
        return Some(fold(window, neg));
    }
    key(&chunk[base.wrapping_add(start)..base + end]) // within 32 bytes of the chunk's end
}

/// Lowercases the first `len` bytes of `window`, all letters and digits, and zeroes the rest,
/// given `neg`, `len` negated, wrapped.
fn fold(window: &[u8; MAX_TERM], neg: usize) -> Key {
    // Flattened, `ONES` is 32 bytes of 0xFF then 32 zeros, so `len` bytes of 0xFF, then zeros,
    // start at `MAX_TERM - len`. On one 64-byte cache line, that 32-byte load never splits
    // across two; unaligned, the table sat wherever the rest of the constants left it, and a
    // split load per token cost 0.7 ms of a run (PERF.md, step 19).
    #[repr(align(64))]
    struct Line([[u8; MAX_TERM]; 2]);
    const ONES: Line = Line([[0xFF; MAX_TERM], [0; MAX_TERM]]);
    let mask: &[u8; MAX_TERM] = ONES.0.as_flattened()[MAX_TERM.wrapping_add(neg)..]
        .first_chunk()
        .expect("len <= MAX_TERM");
    std::array::from_fn(|i| {
        let word = |bytes: &[u8; MAX_TERM]| {
            u64::from_le_bytes(*bytes[8 * i..].first_chunk().expect("8 of 32 bytes"))
        };
        // A lowercase letter and a digit have bit 5 set, so setting it lowercases a letter and
        // leaves a digit as it is; the mask then zeroes the bytes past the token.
        (word(window) | 0x2020_2020_2020_2020) & word(mask)
    })
}

/// One slot of a [`TermTable`]: a term's first 12 bytes, its id and what the corpus pass does
/// with it, 16 bytes that one load brings in. In this order: [`TermTable::intern`] takes `tail` and
/// `flags` as one word.
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    /// `key[0]`, the term's first 8 bytes; 0 marks an empty slot, as a term's first byte is
    /// never zero.
    head: u64,
    /// [`Slot::tail`] of the term's key: its bytes 8 to 11, and whether it runs past 12 bytes.
    tail: u32,
    /// The term id times 32, plus the [`Plan`]'s flags for the term and [`Slot::MORE`].
    flags: u32,
}

impl Slot {
    /// The term triggers phrase checks.
    const PHRASE: u32 = 1;
    /// The term triggers per-document BOOL checks.
    const BOOL: u32 = 2;
    /// The corpus pass keeps the term's documents in an id list ([`Store`]).
    const LISTED: u32 = 8;
    /// A term whose home slot this is lies further on or runs past 12 bytes, so a probe from
    /// here must go on with the full compare ([`TermTable::find_from`]).
    const MORE: u32 = 16;

    /// The term id.
    fn id(self) -> u32 {
        self.flags >> 5
    }

    /// The `tail` of a slot holding `key`: the key's bytes 8 to 11, plus, in the top bit, which no
    /// letter or digit has, whether the key runs past 12 bytes. Each of a key's bytes has bit 5
    /// set ([`fold`]), so it does exactly when bit 5 of its byte 12, bit 37 of `key[1]`, is set,
    /// which one shift moves to the top bit. With that bit, a key that shares a term's first 12
    /// bytes matches its `head` and `tail` only when both or neither run past them.
    fn tail(key: &Key) -> u32 {
        key[1] as u32 | (key[1] >> 6) as u32 & 1 << 31
    }
}

/// `len` zeroed items alone on transparent huge pages, never freed: for an array read at random,
/// whose reads on 4 KiB pages would mostly walk the page tables first. If the first writes
/// found no free huge page, `MADV_COLLAPSE` asks again, now compacting memory if it must,
/// whatever the sysfs THP settings say; a kernel before 6.1 or without THP refuses it,
/// harmlessly.
///
/// # Safety
///
/// An all-zero `T` is a valid `T`.
unsafe fn huge<T>(len: usize) -> &'static mut [T] {
    // Whole huge pages, as a huge page only maps a whole 2 MiB of the block; and one at least,
    // as `alloc` takes no empty layout.
    let size = (size_of::<T>() * len).max(1).next_multiple_of(2 << 20);
    let layout = std::alloc::Layout::from_size_align(size, 2 << 20).expect("fits");
    // SAFETY: the size is nonzero.
    let block = unsafe { std::alloc::alloc(layout) };
    assert!(!block.is_null(), "out of memory");
    // SAFETY: `block` holds `layout.size()` bytes, aligned for any `T`, and is never freed; an
    // all-zero `T` is valid, as the caller ensures. Neither advice changes any data.
    unsafe {
        madvise(block, layout.size(), 14); // MADV_HUGEPAGE
        block.write_bytes(0, layout.size());
        madvise(block, layout.size(), 25); // MADV_COLLAPSE
        std::slice::from_raw_parts_mut(block.cast(), len)
    }
}

/// Open-addressing map from query term to its [`Slot`]. A probe compares a slot's `head` and
/// `tail` with the token's first 12 bytes; only a term longer than 12 bytes, rare among tokens,
/// then loads the rest of its key. So most tokens cost one memory access here, whether or not
/// they are query terms. It holds 3 slots per 2 term occurrences or more, so it is at most two
/// thirds full whatever share of the occurrences are distinct terms.
struct TermTable {
    /// The slot count is a power of two.
    slots: &'static mut [Slot],
    /// Term id → its length in bytes. Id 0 is no term: it is the id of an empty slot, all zero,
    /// which a probe for a token that is no query term returns.
    lens: Vec<u8>,
    /// Term id → its key, for the terms longer than 12 bytes, the only keys a probe reads. The
    /// others stay zero, so that a page that holds no long term's key is never touched.
    keys: Vec<Key>,
}

impl TermTable {
    /// An empty table for the terms of `occurrences` term occurrences: 3 slots per 2, plus one,
    /// rounded up to a power of two. A probe for a key that is no term stops at an empty slot,
    /// and there is always one. For the spec's 150,000 or so occurrences, of which about half
    /// are distinct terms, that is 262,144 slots, at most 29% full.
    ///
    /// The slots sit on huge pages of their own ([`huge`]), as the pass reads them at random.
    /// `threads` intern into it at once, each taking blocks of 64 ids ([`TermTable::intern`]).
    fn new(occurrences: usize, threads: usize) -> Self {
        // Id 0, an id per distinct term, and the unused rest of a block per thread.
        let ids = occurrences + 64 * threads + 1;
        // A term id times 32 fits in `Slot::flags`.
        assert!(ids < 1 << 27, "queries.txt: too many terms");
        // 2 slots or more, so that the shift in [`TermTable::home`] stays below 64.
        let len = (occurrences * 3 / 2 + 1).next_power_of_two().max(2);
        let (lens, keys) = (vec![0; ids], vec![[0; MAX_TERM / 8]; ids]);
        // Only the few long terms' keys are written (13 on seed 1), each a page of its own, which
        // under THP `always` might instead fault in a whole huge page.
        let page = (keys.as_ptr().cast::<u8>()).wrapping_sub(keys.as_ptr().addr() % 4096);
        // SAFETY: advice that changes no data, on the pages of the block that holds `keys`.
        unsafe { madvise(page, size_of_val(&keys[..]), 15) }; // MADV_NOHUGEPAGE
        // SAFETY: an all-zero slot is a valid one, empty.
        let slots = unsafe { huge(len) };
        Self { slots, lens, keys }
    }

    fn len(&self) -> usize {
        self.lens.len()
    }

    /// The slot of the query term whose key this is, given `at = self.home(key)`, or else a slot
    /// with the flags of an empty one, 0, whose id is no term's. Inlined, as both callers' token
    /// loops run it; not inlined, the pass ran 22% more instructions (PERF.md, step 45).
    ///
    /// # Safety
    ///
    /// `at` points to a slot of `slots`, as [`TermTable::prefetch`] returns, and `zero` is 0.
    #[inline(always)]
    unsafe fn find_from(&self, key: &Key, at: *const Slot, zero: u64) -> Slot {
        // SAFETY: `at` points to a slot, as the caller ensures.
        let slot = unsafe { *at };
        // A token whose home slot holds another term is no query term unless the slot has
        // `MORE`, which an empty slot never has: 29% of the tokens that are no term found their
        // home slot taken, and each of them mispredicted the probe's first test. The key's whole
        // second word against the `tail`, with no branch on the key's length: a key or a term
        // past 12 bytes never matches here, and the term's home slot has `MORE`. The probe goes on
        // when `on` is not 0: one select, where `found || !more` took two more instructions.
        let diff = (slot.head ^ key[0]) | (u64::from(slot.tail) ^ key[1]);
        let mut on = std::hint::select_unpredictable(slot.flags & Slot::MORE != 0, diff, zero);
        // SAFETY: the asm block is empty; it keeps the test one branch, as in `position`.
        unsafe { asm!("/* {} */", inout(reg) on, options(pure, nomem)) };
        if on != 0 {
            // SAFETY: `at` points into `slots`, as the caller ensures.
            return unsafe { self.find_past(key, at) };
        }
        let flags = std::hint::select_unpredictable(diff == 0, slot.flags, zero as u32);
        Slot { flags, ..slot }
    }

    /// [`TermTable::find_from`] from the home slot `at` on, with the full compare, out of line,
    /// so that the token loops keep no register for the table's address and size.
    ///
    /// # Safety
    ///
    /// `at` points to a slot of `slots`.
    #[cold]
    #[inline(never)]
    unsafe fn find_past(&self, key: &Key, at: *const Slot) -> Slot {
        // SAFETY: `at` points into `slots`, as the caller ensures.
        let home = unsafe { at.offset_from_unsigned(self.slots.as_ptr()) };
        // SAFETY: `home` is below the slot count, as is the index `position` returns.
        unsafe { *self.slots.get_unchecked(self.position(key, home)) }
    }

    /// Where the probe for `key` starts: below the slot count, a power of two of 2 or more, as
    /// the hash is shifted down to the count's bits.
    ///
    /// The hash multiplies the exclusive or of the key's first two words, one multiply rather
    /// than one per word. Slots come from its top bits, which depend on both words. A key of 16
    /// bytes or less hashes as it would on all four words, which are zero past its end; longer
    /// tokens are rare, and those whose first 16 bytes agree share a home slot, told apart by the
    /// probe.
    fn home(&self, key: &Key) -> usize {
        let hash = (key[0] ^ key[1]).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (hash >> (64 - self.slots.len().trailing_zeros())) as usize
    }

    /// Moves into each home slot that has `MORE`, of the terms of 12 bytes or less whose home it
    /// is, the one with the most sample `hits`: a probe that finds its term at the home slot
    /// stops there ([`TermTable::find_from`]), and 44% fewer go on ([`TermTable::find_past`]).
    /// Terms with the same home lie in one run of slots from it, as does a term in a home slot
    /// not its own, which can move on to any later slot of the run, so these two trade places.
    fn settle(self, hits: impl Fn(Slot) -> u32) -> Self {
        let mask = self.slots.len() - 1;
        for home in 0..=mask {
            let mut at = (home + 1) & mask;
            while self.slots[home].flags & Slot::MORE != 0 && self.slots[at].head != 0 {
                let (here, there) = (self.slots[home], self.slots[at]);
                if self.owns(there, home) && (!self.owns(here, home) || hits(there) > hits(here)) {
                    // `MORE` stays with the slot.
                    let more = (here.flags ^ there.flags) & Slot::MORE;
                    self.slots.swap(home, at);
                    self.slots[home].flags ^= more;
                    self.slots[at].flags ^= more;
                }
                at = (at + 1) & mask;
            }
        }
        self
    }

    /// Whether `slot` holds a term of 12 bytes or less whose home slot is `home`: only those
    /// stop at the home ([`TermTable::find_from`]).
    fn owns(&self, slot: Slot, home: usize) -> bool {
        slot.tail >> 31 == 0 && self.home(&[slot.head, slot.tail.into(), 0, 0]) == home
    }

    /// Returns [`TermTable::home`] and starts loading that slot into L1, so that the load
    /// overlaps other work until [`TermTable::find_from`] needs it.
    fn prefetch(&self, key: &Key) -> *const Slot {
        let at = self.slots.as_ptr().wrapping_add(self.home(key));
        prefetch(at);
        at
    }

    /// The ids of `keys`, while other threads intern theirs into the table at once. A new term
    /// claims the empty slot where its probe ends by a compare-and-swap of its `head`, writes the
    /// rest of the slot, its length and, past 12 bytes, its key, and then publishes its id in the
    /// slot's `flags`, which a probe that finds the same `head` waits for. Its id comes from a
    /// block of 64 that the thread takes from `next`, as one id at a time would move that
    /// counter's cache line between the cores at every new term. Ids thus follow no file
    /// order, and the unused rest of each thread's last block is no term's: a length of 0, never
    /// in a query or a slot. The home slot of `keys[i + 16]` starts loading meanwhile, so that its
    /// probe seldom waits.
    fn intern(&self, keys: &[Key], next: &AtomicU32) -> Vec<u32> {
        let (mask, mut ids, slots) = (self.slots.len() - 1, 0..0, self.slots.as_ptr().cast_mut());
        let (lens, long) = (self.lens.as_ptr().cast_mut(), self.keys.as_ptr().cast_mut());
        // A slot as two words, its `head` and its `tail` with its `flags` above ([`Slot`]). While
        // the threads intern, they read and write the slots only through these atomics, and a
        // term's length and key are written before its id is published (Release) and read after
        // it is seen (Acquire).
        // SAFETY: `at` is below the slot count, and a slot is 16 bytes aligned to 8.
        let atomic = |at: usize| unsafe { &*slots.add(at).cast::<[AtomicU64; 2]>() };
        let id = |i: usize| {
            keys.get(i + 16).map(|ahead| self.prefetch(ahead));
            let (key, home) = (&keys[i], self.home(&keys[i]));
            let mut at = home;
            let id = loop {
                let [head, rest] = atomic(at);
                let mut found = head.load(Relaxed);
                if found == 0 {
                    found = head
                        .compare_exchange(0, key[0], Relaxed, Relaxed)
                        .err()
                        .unwrap_or(0);
                }
                if found == 0 {
                    if ids.is_empty() {
                        let start = next.fetch_add(64, Relaxed);
                        ids = start..start + 64;
                    }
                    let id = ids.next().expect("a block holds 64 ids");
                    // A key's length is its count of nonzero bytes.
                    let len = key.iter().map(|w| 8 - w.leading_zeros() as u8 / 8).sum();
                    // SAFETY: as above; `id` is below the id count ([`TermTable::new`]), the
                    // length of `lens` and `keys`.
                    unsafe {
                        *lens.add(id as usize) = len;
                        if key[1] >> 32 != 0 {
                            *long.add(id as usize) = *key;
                        }
                    }
                    rest.fetch_or(u64::from(Slot::tail(key)) | u64::from(id) << 37, Release);
                    break id;
                }
                let mut seen = rest.load(Acquire);
                while found == key[0] && seen >> 37 == 0 {
                    std::hint::spin_loop();
                    seen = rest.load(Acquire);
                }
                // The slot from its two words: `head`, then `tail` with `flags` above it.
                let (head, tail, flags) = (found, seen as u32, (seen >> 32) as u32);
                if self.holds(&Slot { head, tail, flags }, key) {
                    break (seen >> 37) as u32;
                }
                at = (at + 1) & mask;
            };
            let more = &atomic(home)[1];
            if (at != home || key[1] >> 32 != 0) && more.load(Relaxed) & 1 << 36 == 0 {
                more.fetch_or(u64::from(Slot::MORE) << 32, Relaxed);
            }
            id
        };
        (0..keys.len()).map(id).collect()
    }

    /// Whether `slot` holds `key`. No `&&`: whether a token is a query term follows no pattern,
    /// so a probe takes one branch, mostly taken, ending at the term or an empty slot; long keys
    /// take one more, on which a short key's `tail` is its word's low half, with no long bit.
    fn holds(&self, slot: &Slot, key: &Key) -> bool {
        if key[1] & 1 << 37 != 0 {
            let found = (slot.head ^ key[0]) | u64::from(slot.tail ^ Slot::tail(key)) == 0;
            // SAFETY: `keys` holds an entry per term occurrence and one more, so past every id.
            return found && unsafe { self.keys.get_unchecked(slot.id() as usize) } == key;
        }
        (slot.head ^ key[0]) | u64::from(slot.tail ^ key[1] as u32) == 0
    }

    /// The slot holding `key`, or the empty slot where it belongs, probing linearly from `at`.
    ///
    /// # Safety
    ///
    /// `at` is below the slot count, as [`TermTable::home`] is.
    unsafe fn position(&self, key: &Key, mut at: usize) -> usize {
        let mask = self.slots.len() - 1;
        loop {
            // SAFETY: `at` starts below the slot count, as the caller ensures, and wraps at it.
            let slot = unsafe { self.slots.get_unchecked(at) };
            let found = self.holds(slot, key);
            // Hidden from the optimizer, the test stays one branch: LLVM would split it in two,
            // the first of which mispredicts on every token that is no query term.
            let mut stop = u8::from(slot.head == 0) | u8::from(found);
            // SAFETY: the asm block is empty; it only keeps the optimizer from seeing `stop`.
            unsafe { asm!("/* {} */", inout(reg_byte) stop, options(pure, nomem)) };
            if stop != 0 {
                return at;
            }
            at = (at + 1) & mask;
        }
    }
}
