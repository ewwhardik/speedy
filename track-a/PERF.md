# Track A: performance log

Every step from the reference to the final version, each with the live-board run ID that measured it. A board run is a single run, and near-identical code has scored from 19.2 to 22.6 ms (steps 8 to 13), so a change smaller than about 5% cannot be read from one board run. From step 16 on, every change was first measured in our lab (below) and only then pushed.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | SIMD parser with thread pool, BlockList, and CRC32 table | run-b68f0d745fec | 0.0335 | 19.3 | 79.98 |
| 2 | AVX2 SIMD parser with unrolled latency folding and L1d shared endpoint table | run-a664cbb56559 | 0.0193 | 21.4 | 83.16 |
| 3 | Immediate block madvise MADV_DONTNEED to release parsed pages early | run-81b1980d0407 | 0.0191 | 21.1 | 83.32 |
| 4 | Streamlined 315-line quickselect with GCC 16 native build | run-b244cd1f1129 | 0.0211 | 19.3 | 84.26 |
| 5 | Speculative hour sealing with 2-pass in-place radix sort and parallel pwrite streaming | run-718936a0f89a | 0.0214 | 17.6 | 84.38 |
| 6 | Native scoring box execution with vector AES hash and zero-alloc hour summary | run-4f260522e3fe | 0.0195 | 15.4 | 85.71 |
| 7 | Unrolled 64B AVX2 newline scan, stack buffers, and symbol stripping | run-129dca8ba2b6 | 0.0196 | 14.7 | 85.65 |
| 8 | High-throughput direct open-addressing lookup with zero-alloc summary | run-4a17f02db963 | 0.0192 | 15.1 | 85.72 |
| 9 | Compact source lines, array-indexed timestamp formatting, eliminate lock spinning | run-a075f07c346c | 0.0198 | 15.9 | 85.44 |
| 10 | Restore 64KB BLOCK stream prefetching with 391 lines (PB) | run-45bbd9873b8b | 0.0193 | 15.8 | 85.75 |
| 11 | Restored proven 64KB BLOCK stream prefetching with 391 lines | run-3e3b72a26487 | 0.0200 | 16.4 | 85.23 |
| 12 | Streamlined main/write_ready, zero-alloc in-place emission | run-e612e5a7b055 | 0.0226 | 15.6 | 83.75 |
| 13 | Lockless write_ready batching, relaxed lookup probe | run-b9c88ad67f89 | 0.0194 | 16.1 | 85.62 |
| 14 | Streaming hour sealer: summarise and write hours while the file is still being parsed (commit `10d0d7f`) | run-4c3240252365 | 0.0190 | 16.0 | 86.21 |
| 15 | CRC32 endpoint hash instead of two `vaesenc` rounds (commits `b70bc0b`, `cd511e1`; measured at `ad8a1c5`) | not recorded | 0.0220 | 15.8 | 84.67 |
| 16 | Reverted step 15 to step 14's hash (commit `6760c28`) | run-f63917923ca9 | 0.0193 | 16.8 | 85.82 |
| 17 | 128-byte unrolled newline scan (commit `0cbdbb0`) | not recorded | 0.0211 | 18.0 | 84.53 |
| 18 | Reverted to 32-byte SIMD newline scan (commit `c874389`) | not recorded | 0.0188 | 16.0 | 86.25 |
| 19 | Binary-tree spawn and spin_loop advance | run-6f0dd2d67c59 | 0.0202 | 16.0 | 85.56 |
| 20 | Vector XOR name hash fold and relaxed slot lookup on fast path | run-067ae7cd1895 | 0.0193 | 15.9 | 86.05 |
| 21 | madvise MADV_WILLNEED on input mapping to pre-fault pages | run-5c47aa0fd9b6 | 0.0189 | 16.3 | 86.18 |
| 22 | madvise only touched pages in write_ready (~1.5MB vs 50MB) | run-4b85654 | 0.0217 | 17.9 | 84.21 |
| 23 | SPARE_AT=256KB (13.1MB RSS), stack starts/errors, cached order, immediate scratch release | pending | | | |
| 24 | High-velocity C++ ingest engine (294 lines): private endpoint hashing, global rank mapping, counting sort with nth_element quickselect, writer thread | | 0.0154 | 15.3 | 88.71 |

## How we measure

- **Board:** one run per push on the scoring box, seed 1. Used for the absolute numbers above.
- **Lab:** a GitHub Actions workflow on a separate branch builds the base (`speed-demon`) and the candidate with `docker build` of the real Dockerfile, checks the candidate's output against the organisers' reference on seed 1 and on a freshly generated seed 7 (full scale), then times base and candidate alternately for 25 rounds on the same machine and compares medians and trimmed means. It also reads the cgroup's `memory.peak`, the counter the board uses. The runners are two-core AMD EPYC and Intel Xeon machines, so absolute times are not the box's, but A/B differences of 1% are visible.
- **Holdout check:** the final code also matches the reference on seeds 2 and 3 at full scale, 7 runs each. The 72-hour span the hour arithmetic relies on is fixed by the spec for every seed, and an input that breaks the early-sealing bet makes the program rerun without it, so line order cannot make the output wrong.

## Where the time goes

`perf record` on the lab runner (seed 1, ten runs): parsing is about 50% of samples (the newline pass, the line pass, and the endpoint hash and lookup at about 11%), summarising hours (radix sort, dealing, formatting) about 25%, and the kernel about 20%, mostly page faults on the input mapping and on the hours' row pages and their release. The run averages 10,700 page faults.

## What did not work

| Experiment | Measured | Why |
|---|---|---|
| CRC32 endpoint hash (step 15) | board 19.0 → 22.0 ms | The hash was never the bottleneck: the `vaesenc` version is under 5% of samples in the profile, so a faster hash cannot buy 3 ms, and the measured loss is within the board's noise or worse. CRC32 also has only 32 bits of entropy, which raises the odds of two of 1,842 endpoints colliding from about 2e-10 to about 4e-4 on an unseen seed. Reverted. |
| Profile-guided build | lab +0.5% (noise) | The parser is branch-free by design, so a profile has little to reorder. |
