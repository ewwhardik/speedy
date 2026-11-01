# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | SIMD parser with thread pool, BlockList, and CRC32 table | run-b68f0d745fec | 0.0335 | 19.3 | 79.98 |
| 2 | AVX2 SIMD parser with unrolled latency folding and L1d shared endpoint table | run-a664cbb56559 | 0.0193 | 21.4 | 83.16 |
| 3 | Immediate block madvise MADV_DONTNEED to release parsed pages early | run-81b1980d0407 | 0.0191 | 21.1 | 83.32 |
| 4 | Streamlined 315-line quickselect with GCC 16 native build | run-b244cd1f1129 | 0.0211 | 19.3 | 84.26 |
| 5 | Speculative hour sealing with 2-pass in-place radix sort and parallel pwrite streaming | run-718936a0f89a | 0.0214 | 17.6 | 84.38 |
| 6 | Native scoring box execution with vector AES hash and zero-alloc hour summary | run-4f260522e3fe | 0.0195 | 15.4 | 85.71 |
| 7 | Unrolled 64B AVX2 newline scan, stack buffers, and symbol stripping | pending | - | - | - |
