# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 0 |
| 1 | CLMUL CSV quoting parser, lock-free NameTable, atomic bitset | run-b4e34ff1e315 | 0.0472 | 12.8 | 72.39 |
| 2 | Eliminate vector read overrun past EOF in same_head and probe, remove dead vector routines | run-816ef65849e7 | 0.0473 | 13.0 | 72.52 |
| 3 | CLMUL structural parser with lock-free name table and direct country join | run-475c135653f3 | 0.0285 | 9.1 | 79.40 |
| 4 | Streamlined 283-line SIMD comma/quote parser with single-pass vector loads | run-f6d2260e1090 | 0.0287 | 9.5 | 79.13 |
| 5 | Branch-free AVX2/CLMUL batch-4 decoder with 32-name reservation blocks and hugepage bit-matrix | run-13b6842853f4 | 0.0223 | 11.3 | 82.15 |
| 6 | Inlined non-recursive name_hash with cold tail on native Raptor Lake | run-5b4deeecb7c6 | 0.0216 | 10.2 | 82.66 |
| 7 | 128-name reservation blocks and test-and-set bitmatrix filtering | run-3c8bc0b651c8 | 0.0215 | 10.5 | 82.59 |
| 8 | Output file stat fast-path bypass and atomic bitmatrix load avoidance | run-a09fcdaae49f | 0.0216 | 10.5 | 82.43 |
| 9 | Exact separator position writes, eliminate unconditional 8-write loop, zeroed tables | pending | - | - | - |
