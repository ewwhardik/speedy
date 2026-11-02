# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Our solution is implemented in zero-dependency Rust (`D = 1`), engineered for hostile CSV edge-case handling with maximum throughput:

1. **Quote-Parity Boundary Detection**: The 200 MB CSV file is chunked across the 12 hardware threads. To guarantee boundary correctness despite embedded newlines inside quoted fields, chunk splitting counts unescaped quote parity.
2. **Direct-Indexed Region Table**: `regions.csv` is parsed into a flat direct lookup array mapping `region_id -> country_code`, providing $O(1)$ zero-cost joins without hash map collisions.
3. **Zero-Copy In-Place Scanning**: Fields (`id`, `name`, `note`, `region_id`, `amount`) are parsed directly from memory. Unescaped byte counts for notes and names are computed without string allocations.
4. **Collision-Free 128-Bit Name Hashing**: Names are hashed into 128-bit keys using a dual FNV/Wyhash mixer. For each of the 240 countries, distinct names are counted using parallel cache-local sort & dedup.
5. **Exact Integer Money Arithmetic**: Amounts are parsed into integer cents without floating-point arithmetic.
6. **Parallel Country Aggregation**: Thread-local aggregates are reduced in parallel and formatted directly into a pre-allocated output buffer.

## Dependencies

None. Standard library only (`D = 1.0`).

## With another 24 hours

1. Utilize AVX2 256-bit SIMD registers to scan for quote characters (`"`) and commas (`,`) 32 bytes per cycle.
2. Compile with Profile-Guided Optimization (PGO) using seed 1 profile data.
