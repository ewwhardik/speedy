# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Dependency-free Rust (`D = 1`), one file: [`src/main.rs`](src/main.rs). Its header comment walks through the design; in short:

1. **Map, claim, parse.** The log is `mmap`ed read-only with a spare zero page behind it, so wide loads may run past the end. Twelve workers (started by `main`, which then works itself) claim 1 MiB at a time. Within a claim, each 64 KiB block is scanned for newlines first (AVX2, 128 bytes per iteration, one unconditional store per 32-byte step) and then every line is read from both ends with no data-dependent branches: the hour comes from the fixed-position `DDTHH` digits (the 72-hour span is fixed by the spec for every seed), the latency is decoded back from the newline with three multiply-shift steps, and the endpoint is whatever sits between.
2. **Endpoints as ids.** An endpoint is a 64-bit hash (two `vaesenc` rounds over its first and last 32 bytes) looked up in one shared open-addressing table of 8,192 slots that fills in the first moments of the run and is read-only after. With 1,842 endpoints, a collision on the 53 compared bits has odds of about 2e-10.
3. **4-byte rows per hour.** Each line becomes `id + 1 | error bit | 20-bit latency` in its hour's area of one lazily backed reservation; the 1% of latencies that need more than 20 bits escape into a second row.
4. **Seal and summarise while parsing.** The input interleaves eight time-ordered streams, so once the parsed prefix of the file holds a line two hours later, an hour is sealed: its rows are radix sorted by latency (two 10-bit passes) in place, dealt out by endpoint so every endpoint's latencies come out sorted, and the percentiles are single array reads. If that bet ever fails on some input, the whole run repeats with no early sealing, so the output is exact for any line order.
5. **Write in parallel.** Each hour's text is written over its own rows; once all earlier hours have their text, its file offset is known and it goes out with `pwrite`, after which its pages are returned with `MADV_DONTNEED`. Only a few hours of rows are alive at once, which keeps peak memory near 16 MB.

## Correctness

Besides the board (seed 1), every change is checked on seeds 2, 3 and 7 at full scale against the organisers' reference, on Linux, with the exact Dockerfile build (see PERF.md, "How we measure").

## Dependencies

None. Standard library only (`D = 1`).

## With another 24 hours

- Profile on the scoring CPU itself: our lab runners are AMD EPYC with two cores, so effects of twelve hyperthreads sharing six cores are invisible there.
- Try `MADV_POPULATE_READ` per claim instead of demand faults on the input mapping, which is about a sixth of the CPU time in the kernel profile.
