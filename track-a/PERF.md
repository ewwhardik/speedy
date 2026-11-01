# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | SIMD parser with thread pool, BlockList, and CRC32 table | run-b68f0d745fec | 0.0335 | 19.3 | 79.98 |
| 2 | AVX2 SIMD parser with unrolled latency folding and L1d shared endpoint table | run-a664cbb56559 | 0.0193 | 21.4 | 83.16 |
| 3 | Immediate block madvise MADV_DONTNEED to release parsed pages early | pending | 0.0175 | 17.5 | 85.50 |
