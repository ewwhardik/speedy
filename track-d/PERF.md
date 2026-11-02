# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Bidirectional traversal engine with compact CSR | run-d25a28d627ff | 0.3246 | 153.7 | 58.73 |
| 2 | Bidirectional Dijkstra with cache-aligned CSR frontier and touched-only reset | run-03c0df59540c | 0.1126 | 73.0 | 71.67 |
| 3 | Native architecture build flags (-march=native) targeting scoring box | run-4301346a6ea4 | 0.1145 | 67.9 | 71.80 |
| 4 | Immediate munmap of 92MB graph.txt and queries.txt, reserved output string buffer | run-f3db523516c1 | 0.1080 | 67.6 | 72.42 |
| 5 | Contraction hierarchies with dual-width labels (16-bit narrow fast path, 64-bit wide fallback) and page-splicing buffer reuse | | 0.0863 | 55.0 | 76.03 |
