FROM rust:1.93 AS builder

WORKDIR /build

COPY static static
COPY src src
COPY Cargo.toml .
COPY Cargo.lock .

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    protobuf-compiler \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cargo build --release && \
    mkdir -p /build/onnxruntime-lib && \
    { find /build/target/release/build -type f -name 'libonnxruntime*.so*' -exec cp -v {} /build/onnxruntime-lib/ \; || true; }

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libstdc++6 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 parakeet \
    && useradd -u 1000 -g 1000 -m -d /home/parakeet parakeet

WORKDIR /home/parakeet

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

COPY --from=builder /build/target/release/parakeet-server /usr/local/bin/
COPY --from=builder /build/onnxruntime-lib /opt/onnxruntime/lib
COPY --from=builder /build/static static

RUN chown -R parakeet:parakeet /home/parakeet

USER parakeet

EXPOSE 8000

ENTRYPOINT ["parakeet-server"]
