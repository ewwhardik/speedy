# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | SIMD parser with thread pool, BlockList, and CRC32 table | run-b68f0d745fec | 0.0335 | 19.3 | 79.98 |
| 2 | Sub-microsecond user-space atomic spin barrier and persistent thread pool | pending | 0.0210 | 19.3 | 85.80 |
| 3 | L1d-resident 16KB endpoint hash table, direct bitmask indexing, eliminated CRC32 loop | pending | 0.0210 | 18.5 | 86.50 |
