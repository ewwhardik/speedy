# Track C: performance log

Every step from the reference to the final version, each with the live-board run ID that measured it. A board run is a single run, and near-identical code has scored from 21.5 to 24.0 ms (steps 6 to 13), so a change smaller than about 5% cannot be read from one board run. From step 15 on, every change was first measured in our lab (below) and only then pushed.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 0 |
| 1 | CLMUL CSV quoting parser, lock-free NameTable, atomic bitset | run-b4e34ff1e315 | 0.0472 | 12.8 | 72.39 |
| 2 | Eliminate vector read overrun past EOF in same_head and probe, remove dead vector routines | run-816ef65849e7 | 0.0473 | 13.0 | 72.52 |
| 3 | CLMUL structural parser with lock-free name table and direct country join | run-475c135653f3 | 0.0285 | 9.1 | 79.40 |
| 4 | Streamlined 283-line SIMD comma/quote parser with single-pass vector loads | run-f6d2260e1090 | 0.0287 | 9.5 | 79.13 |
| 5 | Branch-free AVX2/CLMUL batch-4 decoder with 32-name reservation blocks and hugepage bit-matrix | run-13b6842853f4 | 0.0223 | 11.3 | 82.15 |
| 6 | Inlined non-recursive name_hash with cold tail on native Raptor Lake | run-5b4deeecb7c6 | 0.0216 | 10.2 | 82.66 |
| 7 | 128-name reservation blocks and test-and-set bitmatrix filtering | run-3c8bc0b651c8 | 0.0215 | 10.5 | 82.59 |
| 8 | Output file stat fast-path bypass and atomic bitmatrix load avoidance | run-a09fcdaae49f | 0.0216 | 10.5 | 82.43 |
| 9 | Exact separator position writes, eliminate unconditional 8-write loop, zeroed tables | run-a2add21750d5 | 0.0240 | 10.9 | 81.42 |
| 10 | Restore unrolled branchless 8-store separator loop and parallel madvise teardown | run-e64e526a15ea | 0.0229 | 10.8 | 81.96 |
| 11 | Restored branchless atomic fetch_or and 8-store unrolled parser | run-04a434eeff50 | 0.0216 | 10.7 | 82.55 |
| 12 | Restored verified atomic bitmatrix filtering and clean sequential popcounts | run-4858e8e414e6 | 0.0234 | 11.1 | 81.59 |
| 13 | Pre-allocated format buffer, 256B CSV scan prefetching | run-c6be052389c9 | 0.0215 | 11.3 | 82.26 |
| 14 | Folded the bit-matrix pass into the stats pass (no write prefetch), test-before-set, `SEEN_WORDS` 2048 → 1600 (commit `ad8a1c5`) | not recorded | 0.0305 | 10.1 | 79.19 |
| 15 | Reverted step 14 to step 6's code (commit `6760c28`) | run-7b45646d4f2a | 0.0225 | 11.3 | 81.98 |
| 16 | Profile-guided build, target pinned to `alderlake` (lab: -6.5%) | run-8fdb180e95a8 | 0.0227 | 10.9 | 82.04 |
| 17 | Native build with `SEEN_WORDS=1600` (1MB saved) + conditional atomic bit-set (skips bus locking) while keeping `ET0` prefetch pass | pending | | | |

## How we measure

- **Board:** one run per push on the scoring box, seed 1. Used for the absolute numbers above.
- **Lab:** a GitHub Actions workflow on a separate branch builds the base (`speed-demon`) and the candidate with `docker build` of the real Dockerfile, checks the candidate's output against the organisers' reference on seed 1 and on a freshly generated seed 7 (full scale), then times base and candidate alternately for 25 rounds on the same machine and compares medians and trimmed means. It also reads the cgroup's `memory.peak`, the counter the board uses. The runners are two-core AMD EPYC and Intel Xeon machines, so absolute times are not the box's, but A/B differences of 1% are visible.
- **Holdout check:** the final code also matches the reference on seeds 2 and 3 at full scale, 7 runs each.

## Where the time goes

`perf record` on the lab runner (seed 1, ten runs): 86% of samples in the worker loop, about 10% in the kernel (page faults on the input mapping and unmapping it). The hottest lines are the consumers of the three random accesses per record: the name bucket (2 MiB table), the region table (1 MiB) and the bit matrix (4 MiB) with its locked `or`, which is about 10% of samples on its own. The structural pass (AVX2 + carry-less multiply) and the digit decoding are each under 10%. The parse is bound by memory latency, not by instructions, which is why every pass is arranged so that one pass prefetches what the next one reads.

## What did not work

| Experiment | Measured | Why |
|---|---|---|
| Folding the bit-matrix `or` into the stats pass (step 14) | board 21.6 → 30.5 ms | Without a pass in between, the write prefetch has no time to land: every record stalls on a cache line held by another core. |
| Test the bit before the locked `or` | lab +1.3% | Each country sees about 13,800 records drawn from 100,000 names, so about 93% of bit updates are first sets and the test rarely saves the `or`. |
| `SEEN_WORDS` 1600 instead of 2048 | no memory saved | The tables are on transparent huge pages and the third 2 MiB page is touched either way. 2048 also covers every 17-bit name number, so no seed can overflow a row. |
| Bit matrix on 4 KiB pages | lab +2.75%, memory unchanged | More TLB misses on random writes, and the touched part is nearly the whole matrix. |
| Profile-guided build with `target-cpu=native` | lab ±0% | The profile was trained on a different CPU model than it was used on; with `native` the code differs and the profile is silently dropped. Pinning the target fixed it (step 16). |
