# Track B: performance log

Every step from the reference to the final version, each with the live-board run ID that measured it. A board run is a single run: steps 16 and 17 below are the same code and differ by 3.4%, so a change smaller than about 5% cannot be read from one board run. From step 16 on, every change was first measured in our lab (below) and only then pushed.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2200 | 0 |
| 1 | SIMD tokenization, term-frequency renumbering, galloping CSR | run-3b4b4bf747f1 | 0.2904 | 68.9 | 64.94 |
| 2 | Rust query-pruned indexer with 64-bit fingerprint term table and galloping lists | run-cf3adea073c6 | 0.1532 | 47.6 | 74.57 |
| 3 | In-place galloping intersection and query-driven posting collection | run-765447e025c0 | 0.1475 | 47.1 | 75.03 |
| 4 | Lock-free postings distribution with hugepage term table | run-7e291dd4a351 | 0.1470 | 47.2 | 75.06 |
| 5 | Phased streaming transpose engine with VAES token hashing and inline phrases | run-ab26e3ba47a7 | 0.0985 | 42.4 | 78.11 |
| 6 | Slice-compare MRU endpoint cache and native Raptor Lake compilation | run-02c32df116e3 | 0.0947 | 41.1 | 78.54 |
| 7 | Zero-bitmap two-pointer streaming list intersection in filter_by with binary search | run-4d79708ac4fa | 0.0996 | 40.8 | 77.93 |
| 8 | 25 MiB L3 cache-resident phases (PHASE_CLAIMS=100) and branchless bitmap filter | run-b02bebeadab6 | 0.1008 | 30.6 | 77.84 |
| 9 | 4KB small-page postings/bitmaps (huge=false), zero-heap-alloc build_shard, pre-reserved union | run-b9cc8b4d3255 | 0.0969 | 43.1 | 78.25 |
| 10 | Iterator-free Tally (direct indexing, trailing_zeros bit extraction), 8MB workspace | run-2821b0dc5fc1 | 0.1040 | 42.8 | 77.44 |
| 11 | Restored 2MB hugepage TLB mappings with zero-alloc build_shard | run-9ad14c995ef9 | 0.0963 | 43.1 | 78.32 |
| 12 | 4-phase corpus division (PHASE_CLAIMS=248) reducing query answering passes by 20% | run-8dd1424245c4 | 0.0952 | 48.0 | 78.07 |
| 13 | 5-phase cap compliance (PHASE_CLAIMS=200), early-exit bit scanning | run-7a7d01b4f4b6 | 0.0961 | 43.5 | 78.28 |
| 14 | Kept postings on huge pages, reverted branchy apply checks (commit `cd511e1`) | run-a45257c6a806 | 0.0903 | 42.7 | 78.96 |
| 15 | `add_bits` zero-block skip, `MAX_HEAVY` 4096 → 1024, postings and bitmaps on 4 KiB pages (commit `ad8a1c5`) | not recorded | 0.0970 | 42.3 | 78.15 |
| 16 | Reverted step 15 to step 14's code (commit `6760c28`) | run-e1e0015ad1e9 | 0.0922 | 42.8 | 78.73 |
| 17 | Unchanged code, re-run by a push to other tracks (commit `0cbdbb0`) | run-d558ec5e6137 | 0.0954 | 42.1 | 78.37 |
| 18 | `MAX_HEAVY` 4096 → 1024 keeping huge-page postings (commit `595ab02`) | not recorded | 0.0923 | 40.3 | 78.72 |
| 19 | 4-token VAES hashing and dual-token action_pair bucket lookup | run-0ba80dbb7859 | 0.0943 | 43.1 | 78.26 |
| 20 | Single-token action_in with AHEAD=8 and MAX_HEAVY=1024 huge-page postings | pending | | | |

## How we measure

- **Board:** one run per push on the scoring box, seed 1. Used for the absolute numbers above.
- **Lab:** a GitHub Actions workflow on a separate branch builds the base (`speed-demon`) and the candidate with `docker build` of the real Dockerfile, checks the candidate's output against the organisers' reference on seed 1 and on a freshly generated seed 7 (full scale), then times base and candidate alternately for 25 rounds on the same machine and compares medians and trimmed means. It also reads the cgroup's `memory.peak`, the counter the board uses. The runners are two-core AMD EPYC and Intel Xeon machines, so absolute times are not the box's, but A/B differences of 1% are visible.
- **Holdout check:** the final code also matches the reference on seeds 2 and 3 at full scale, 7 runs each.

## Where the time goes

`perf record` on the lab runner (seed 1, ten runs): 85% of samples in the worker loop, about 70% of all samples in reading claims (`index_claim`): classifying bytes, extracting token bounds, hashing token pairs with `vaesenc`, and probing the bucket table. No single line has more than 7%. Building postings (`build_shard`) is about 3%, answering queries about 6%, and the kernel about 5%. On two cores the run uses 1.7 cores of CPU for its wall time, so the work is parallel, and the next gain has to come from doing less work per token, not from better overlap.

## What did not work

| Experiment | Measured | Why |
|---|---|---|
| Postings and bitmaps on 4 KiB pages (step 15) | board +7%, memory unchanged | Every page of the postings is touched anyway, so small pages only add faults and TLB misses. |
| Skip all-zero 256-bit blocks in `add_bits` | lab ±0% (+0.17% median) | A 3x microbenchmark win, but answering is about 6% of the run, so even a 3x faster tally is within the noise of the whole. |
| Profile-guided build with the committed profile | lab -0.8% (noise) | Nothing to gain with a profile this flat. A build pinned to `alderlake`, needed to make a profile portable, also faults on the AMD lab runners (an instruction they lack), so it cannot be trained there. |
| `PHASE_CLAIMS` 248 (four phases instead of five) | board 0.0952, memory 48 MB | Fewer answer passes, but each phase holds more rows, and a phase must stay under 65,535 documents for 16-bit postings: about 64,000 at 248 claims, too close for a seed we have not seen. 200 claims gives about 51,000. |
