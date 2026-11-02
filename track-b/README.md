# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Dependency-free Rust (`D = 1`), one file: [`src/main.rs`](src/main.rs). The header comment there is the full design note; the decisions that matter most:

1. **Answer these 50,000 queries, not "build an index".** The queries are readable from the start, so they are parsed first and every term goes into one table of 2^15 one-cache-line buckets (7 tags and 8 actions each, on a huge page). Scanning the corpus, a token costs one hash and one bucket probe; its *action* says what to do with it: nothing (not a query term), set a bit, append to a list, or also start a phrase check. A token that no query uses is dropped right there.
2. **Two representations, chosen from a sample.** Twelve sample claims estimate each term's document frequency. Terms common enough get a per-phase bitmap; the rest get document lists, written first as rows that need no shared counters and then compacted into 16-bit postings relative to the phase's first document.
3. **No positions stored.** A phrase is verified during the scan, while the document's token hashes are at hand: when its rarest word (the *anchor*, picked from the sample) appears, the neighbouring hashes are compared with the phrase, and a match is recorded as a list entry for a pseudo-term that stands for the whole phrase. A phrase query is then a one-term lookup, and no document is read twice.
4. **Phases bound memory.** The corpus is processed in phases of 50 MiB. After each phase its rows become postings, the queries are answered on that phase's postings and bitmaps, and each query keeps a running tally (count, id sum, ten smallest ids). No full index ever exists, which keeps peak memory near 43 MB against the reference's 2.2 GB. The answers of phase k - 1 run on whichever threads finish scanning phase k first.
5. **BOOL strictly left to right,** on whichever of list or bitmap suits each step; bitmap results are tallied with nibble lookups (`pshufb`) and `psadbw`, 256 bits at a time.

## Correctness

Tokens are compared by a 64-bit hash (47 bits kept in the table), a deliberate shortcut documented in the source. Besides the board (seed 1), every change is checked on seeds 2, 3 and 7 at full scale against the organisers' reference, on Linux, with the exact Dockerfile build (see PERF.md, "How we measure").

## Dependencies

None. Standard library only (`D = 1`).

## With another 24 hours

- The profile is now flat: tokenizing, hashing and probing are about 70% of CPU, spread over a dozen lines with no single hotspot. The next step is fewer probes, not faster ones: for example, skipping the probe for tokens whose length and first bytes no query term shares.
- Profile on the scoring CPU itself; our two-core lab cannot show how twelve hyperthreads share the bucket table.
