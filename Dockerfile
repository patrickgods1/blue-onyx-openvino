# Blue Onyx OpenVINO: Linux x86_64 image with Intel GPU support via /dev/dri.
# Build:  docker build -t blue-onyx-openvino .
# Run:    docker run --rm -p 32168:32168 --device /dev/dri:/dev/dri \
#           -v $PWD/models:/app/models -v $PWD/cache:/app/cache -v $PWD/config:/app/config blue-onyx-openvino

FROM rust:1-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release --locked

# openvino/ubuntu24_runtime ships the OpenVINO runtime, Intel compute runtime (OpenCL) and sets
# INTEL_OPENVINO_DIR, which openvino-finder honours.
FROM openvino/ubuntu24_runtime:2026.4.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/blue-onyx-openvino /app/blue-onyx-openvino
COPY --from=builder /src/target/release/blue-onyx-openvino-benchmark /app/blue-onyx-openvino-benchmark
RUN mkdir -p /app/models /app/cache /app/config && chown -R openvino:openvino /app
USER openvino
ENV RUST_LOG=info
EXPOSE 32168
ENTRYPOINT ["/app/blue-onyx-openvino", "--config", "/app/config/blue_onyx_openvino_config.json", "--cache-dir", "/app/cache"]
