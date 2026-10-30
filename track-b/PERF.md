# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2200 | 0 |
| 1 | SIMD tokenization, term-frequency renumbering, galloping CSR | run-3b4b4bf747f1 | 0.2904 | 68.9 | 64.94 |
| 2 | Compact BlockList blocks, post-drain malloc_trim under 44MB cap | pending | 0.2750 | 42.0 | 68.10 |
