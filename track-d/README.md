# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Our solution is implemented in zero-dependency Rust (`D = 1`), engineered for extreme-scale graph parsing and high-throughput bidirectional shortest-path queries:

1. **Parallel Lock-Free CSR Construction**: The 5-million-edge graph is parsed across 12 hardware threads. Edge counts per node are accumulated using atomic offsets, followed by parallel placement into forward and reverse Compressed Sparse Row (CSR) arrays.
2. **Compact 8-Byte Memory-Aligned Edges**: Edges are laid out as packed `Edge { target: u32, weight: u32 }` structures in contiguous memory for optimal L2/L3 cache-line utilization.
3. **Bidirectional Dijkstra with 4-ary MinHeap**:
   - A 4-ary heap halves tree depth compared to standard binary heaps, significantly reducing branch mispredictions and improving cache locality during sift-down operations.
   - Searches expand simultaneously from source and target until meeting with the tight stopping criterion $\min(Q_f) + \min(Q_b) \ge \mu$.
4. **Epoch-Stamped Distance Array**: Distance and visited states use a monotonically incremented `epoch` counter. Re-initializing between queries costs $O(1)$ time with **zero memory reallocations**.
5. **Instant Degree-Zero Pruning**: Queries starting from nodes with 0 out-edges or targeting nodes with 0 in-edges immediately return `-1` without entering the heap.
6. **Parallel Query Distribution**: The 2,000 queries are distributed evenly across worker threads and answered concurrently into partitioned result buffers.

## Dependencies

None. Standard library only (`D = 1.0`).

## With another 24 hours

1. Incorporate SIMD-based edge parsing using AVX2 to parse three decimal integers per line simultaneously.
2. Precompute a two-level landmark / ALT heuristic to direct Dijkstra search frontiers towards targets.
