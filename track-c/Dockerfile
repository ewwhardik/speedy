# Build one optimised binary for the scoring CPU with plain rustc (no crates), then ship it alone.
FROM rust:1-slim-trixie@sha256:1a939748a7cbab172de91a014e3804a97af1394a8c844faa67fce9401c4e40fb AS build
WORKDIR /src
COPY main.rs .
RUN rustc --edition 2021 -C opt-level=3 -C target-cpu=native -C panic=abort -o /join main.rs

FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /join /app/join
ENTRYPOINT ["/app/join"]
