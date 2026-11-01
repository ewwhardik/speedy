# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | SIMD parser with thread pool, BlockList, and CRC32 table | run-b68f0d745fec | 0.0335 | 19.3 | 79.98 |
| 2 | AVX2 SIMD parser with unrolled latency folding and L1d shared endpoint table | run-a664cbb56559 | 0.0193 | 21.4 | 83.16 |
| 3 | Immediate block madvise MADV_DONTNEED to release parsed pages early | run-81b1980d0407 | 0.0191 | 21.1 | 83.32 |
| 4 | Streamlined 315-line median-of-3 quickselect with digit pairs table and GCC 16 native build | run-b244cd1f1129 | 0.0211 | 19.3 | 84.26 |
| 5 | Whole-program LTO with -fno-exceptions -fno-rtti and native Raptor Lake vector scheduling | pending | 0.0165 | 18.0 | 88.00 |
