# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Our solution is implemented in idiomatic, dependency-free Rust (`D = 1`), engineered specifically for the target scoring machine (Intel Core i5-13500 with 6 P-cores / 12 hardware threads, 24 MB L3 cache).

Key architectural decisions:
1. **Multithreaded Chunk Partitioning**: The input file is partitioned across the 12 hardware threads with line-boundary alignment. Each thread processes its slice completely independently without mutexes, channels, or atomic synchronization.
2. **Fast-path Timestamp Translation**: Since all timestamps in the competition span a known 72-hour window (`2026-10-30T18:00:00Z` to `2026-11-02T17:59:59Z`), we parse the day and hour with simple branchless integer arithmetic directly into an index in `0..72`. This completely eliminates RFC 3339 date/time parsing overhead.
3. **L1-Resident Endpoint Lookup**: Endpoints are deduplicated per-thread using a compact 4096-slot open-addressing hash table with FNV-1a hashing. The table fits inside L1 data cache (< 48 KB) and guarantees fast lookups.
4. **Cache-Friendly Aggregation Matrix**: Groups are indexed directly as `hour_idx * 2048 + local_ep_id`. Unvisited groups incur zero heap allocations.
5. **Parallel Nearest-Rank Reduction**: The 72 hours are partitioned across the worker threads to collect latencies, sort with `pdqsort` (`sort_unstable`), and format output directly into pre-allocated memory buffers.
6. **Zero-Allocation Formatting**: Integers and strings are serialized directly into contiguous output buffers, avoiding intermediate string formatting and I/O thrashing.

## Dependencies

None. Standard library only (`D = 1.0`).

## With another 24 hours

1. Incorporate AVX2 SIMD scanning for delimiter detection (`b' '` and `b'\n'`) to vectorize line tokenization.
2. Apply Profile-Guided Optimization (PGO) using `halloumee/speed-demon-rust:1.99.1-trixie-llvm-tools` to optimize branch probabilities in the main parse loop.
