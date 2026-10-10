# Blue Onyx Prism: Linux x86_64 image with Intel GPU support via /dev/dri.
# Build:  docker build -t blue-onyx-prism .
# Run:    docker run --rm -p 32168:32168 --device /dev/dri:/dev/dri \
#           -v $PWD/models:/app/models -v $PWD/cache:/app/cache -v $PWD/config:/app/config \
#           --group-add $(getent group render | cut -d: -f3) blue-onyx-prism

FROM rust:1-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release --locked

# Builder glibc 2.36 (bookworm) <= runtime glibc 2.39 (ubuntu24), so the binary runs unchanged.
# Keep the tag equal to OPENVINO_VERSION in src/setup_openvino.rs.
# openvino/ubuntu24_runtime ships the OpenVINO runtime, Intel compute runtime (OpenCL) and sets
# INTEL_OPENVINO_DIR, which openvino-finder honours.
FROM openvino/ubuntu24_runtime:2026.4.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/blue-onyx-prism /app/blue-onyx-prism
COPY --from=builder /src/target/release/blue-onyx-prism-benchmark /app/blue-onyx-prism-benchmark
# ONNX Runtime (CPU flavor, pinned in src/setup_onnxruntime.rs) so `ort:cpu` works; OpenVINO comes
# from the base image. NVIDIA/CUDA is not supported by this image.
RUN /app/blue-onyx-prism setup-onnxruntime --flavor cpu --dir /app/onnxruntime
RUN mkdir -p /app/models /app/cache /app/config && chown -R openvino:openvino /app
USER openvino
EXPOSE 32168
ENTRYPOINT ["/app/blue-onyx-prism", "--config", "/app/config/blue_onyx_prism_config.json", "--cache-dir", "/app/cache"]
