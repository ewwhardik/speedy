# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2200 | 0 |
| 1 | SIMD tokenization, term-frequency renumbering, galloping CSR | run-3b4b4bf747f1 | 0.2904 | 68.9 | 64.94 |
| 2 | Rust query-pruned indexer with 64-bit fingerprint term table and galloping lists | run-cf3adea073c6 | 0.1532 | 47.6 | 74.57 |
