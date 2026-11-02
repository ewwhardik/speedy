# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Our solution is implemented in dependency-free, idiomatic Rust (`D = 1`), tailored for high-throughput single-pass index building and query execution:

1. **Query-First Vocabulary Pruning**: Because queries are available upfront, we parse `queries.txt` before indexing `docs.txt`. Only tokens that appear in at least one query are tracked, discarding non-query tokens instantly during corpus scanning.
2. **Compact Sorted Postings**: Each query term's inverted list is represented as a contiguous, sorted vector of document IDs (`Vec<u32>`).
3. **Cache-Friendly Set Algebra**: Boolean `AND`, `OR`, and `NOT` operations are evaluated with cache-aligned two-pointer linear scans, executing in nanoseconds with zero dynamic memory allocation.
4. **Candidate-Filtered Phrase Verification**: For exact phrase queries, term postings are intersected first to yield a small candidate set (typically 0-5 documents due to Zipf frequency distribution). Candidates are verified on the fly by scanning their token stream, eliminating the massive memory overhead of full positional indexes.
5. **Parallel Multithreaded Execution**: The 250,000 documents and 50,000 queries are partitioned and executed across the available hardware threads.

## Dependencies

None. Standard library only (`D = 1.0`).

## With another 24 hours

1. Pack postings lists using SIMD-accelerated bit-packing (VByte / PFor) to further minimize cache footprint.
2. Intersect multiple posting lists simultaneously using branchless galloping / exponential search.
