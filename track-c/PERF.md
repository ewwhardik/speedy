# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 0 |
| 1 | CLMUL CSV quoting parser, lock-free NameTable, atomic bitset | run-b4e34ff1e315 | 0.0472 | 12.8 | 72.39 |
| 2 | Streamlined 8MB arena, THP disable, inline separators | pending | 0.0360 | 10.7 | 77.20 |
| 3 | Eliminate vector read overrun past EOF in same_head and probe, remove dead vector routines | run-816ef65849e7 | 0.0473 | 13.0 | 72.52 |
| 4 | Hardware CRC32 name probing, bitmask modulo indexing | pending | 0.0240 | 11.0 | 81.50 |
