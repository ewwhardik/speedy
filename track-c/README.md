# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Dependency-free Rust (`D = 1`), one file: [`src/main.rs`](src/main.rs). Its header comment walks through the design; in short:

1. **Parallel parse without scanning from the start.** The spec gives a safe resync point: inside a note an LF is only ever followed by a lowercase word or a quote, so an LF followed by a digit or a BOM always starts a record. Each of the twelve workers claims 1 MiB of the mapped file and starts at its first such boundary.
2. **Structure with AVX2 and a carry-less multiply.** Per 64 bytes, AVX2 finds quotes, commas and LFs, and `pclmulqdq` by all-ones turns the quotes into an inside-quotes mask. `""` toggles twice, so escapes need no handling. Every record then has exactly five structural bytes, whose positions go into one array.
3. **Branch-free fields.** 23% of records hold an escape, so nothing branches on field shapes. Ids, amounts and region ids are gathered as digit words and decoded four records at a time with `maddubs`/`madd`. Unescaped name and note lengths come from counting quote bytes (a quoted field with q quotes holds (q - 2) / 2 escapes), using a prefix count of quotes per 64-byte step.
4. **Distinct names in a bit matrix.** A name is a 64-bit AES hash; its top 15 bits pick a cache-line bucket of eight slots that map it to a dense number, claimed with one compare-and-swap. Each record sets its name's bit in its country's row of a shared bit matrix (256 groups x 2^17 names), and row popcounts give the distinct counts.
5. **Hide the misses.** The name table, region table and bit matrix are each a random access per record. Each batch of about 190 records goes through separate passes: one prefetches the bucket and region lines, the next reads them and prefetches the bit-matrix line for writing, and the last does the locked `or`s. That split alone was worth 30% (see PERF.md).
6. **Profile-guided build.** The Dockerfile builds with a committed `pgo.profdata` trained on the seed-1 data in the allowlisted `halloumee/speed-demon-rust` image (same rustc 1.99.1 / LLVM 23.1.1 as the build image). The target is pinned to `alderlake` (the i5-13500's ISA) rather than `native`, so the profile matches the scoring build wherever it was trained.

## Correctness

Besides the board (seed 1), every change is checked on seeds 2, 3 and 7 at full scale against the organisers' reference, on Linux, with the exact Dockerfile build (see PERF.md, "How we measure").

## Dependencies

None. Standard library only (`D = 1`).

## With another 24 hours

- The locked `or` per record is about 10% of CPU. Routing (group, name) pairs to an owner thread per group would make those writes plain stores, but buffering the pairs costs more memory than the time it saves under the scoring weights; a bounded ring per thread pair might get both.
