# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 0 |
| 1 | CLMUL CSV quoting parser, lock-free NameTable, atomic bitset | run-b4e34ff1e315 | 0.0472 | 12.8 | 72.39 |
| 2 | Eliminate vector read overrun past EOF in same_head and probe, remove dead vector routines | run-816ef65849e7 | 0.0473 | 13.0 | 72.52 |
| 3 | CLMUL structural parser with 16B CAS lock-free name table and direct country join | pending | 0.0282 | 8.9 | 79.60 |
