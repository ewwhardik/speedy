# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

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
| 10 | 4MB workspace buffer restriction to fit in Raptor Lake L3 cache | pending | - | - | - |
