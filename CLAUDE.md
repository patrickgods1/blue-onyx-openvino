# Blue Onyx OpenVINO

Cross-platform object-detection HTTP service for Blue Iris (CodeProject.AI-compatible API),
written in Rust on **native OpenVINO** via the `openvino` crate (runtime-linking). Modeled on
[blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT) but without ONNX Runtime, with
multi-model support and Intel GPU inference on Windows/Linux (CPU on macOS arm64).

Primary target: Windows 11, Intel i5-8500 + UHD 630 iGPU. Must also build/run on Linux x86_64 and macOS arm64.

## Build & run

```powershell
cargo build --release
cargo run -- setup-openvino                 # downloads OpenVINO runtime libs into ./openvino (next to exe)
cargo run -- download-models --name IPcam-general
cargo run -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
cargo test                                   # unit tests need no OpenVINO
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Python export env (YOLO26 -> OpenVINO IR): `.venv` (Python 3.11, `ultralytics`, `openvino`),
created with `uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-export.txt`;
run `.venv/Scripts/python.exe scripts/export_yolo26.py` (Windows) or `.venv/bin/python scripts/export_yolo26.py`.

OpenVINO libs are discovered through `openvino-finder`: `OPENVINO_INSTALL_DIR` (set at runtime by
`backend/libs.rs` to `<exe_dir>/openvino` when present) or the OS library path. Pinned version lives in
one constant in `src/setup_openvino.rs`.

## Architecture (see the plan in the repo history / README for detail)

- `src/server.rs` axum HTTP (one tokio current-thread runtime). Blue Iris calls
  `POST /v1/vision/detection` (default model) and `POST /v1/vision/custom/{model}` (named model),
  `POST /v1/vision/custom/list`. Responses use `api.rs` structs (camelCase; `Prediction` snake_case).
- `src/registry.rs` loads N models from config sequentially on one `Core`; each model gets its own
  worker thread (`worker.rs`) and bounded crossbeam channel. `startup.rs` tracks Initializing/Ready/Failed
  so HTTP serves immediately while models compile.
- `src/backend/` wraps OpenVINO (`Core`, `CompiledModel`, `InferRequest`): device selection with
  GPU->CPU fallback, properties (CACHE_DIR, LATENCY hint, f16 on GPU), tensor IO.
- `src/model/` model families: `yolo26` (end-to-end `[1,300,6]`), `yolo5` (`[1,N,5+C]` + NMS),
  `yolo8` (`[1,4+C,8400]` + NMS), `rtdetr` (`images` + i64 `orig_target_sizes`; `labels/boxes/scores`).
  Preprocess = letterbox (YOLO) or stretch (RT-DETR) to 640x640 RGB f32 0..1, CHW.
- `src/config.rs` JSON config next to the exe; CLI overrides only non-default values and writes back.
- Windows service in `src/bin/blue_onyx_openvino_service.rs` (cfg windows). systemd/launchd/Docker in `deploy/`.

## Conventions

- Edition 2024, `anyhow` for app errors, `tracing` for logs. No `build.rs`. No `unsafe` outside `backend/`.
- Everything portable unless it must be `#[cfg(windows)]` (service, event log, DXGI).
- Copy ideas and API shapes from the blue-onyx reference freely (MIT, attributed in LICENSE); do not
  copy ONNX Runtime code.
- Model weights, `models/`, `cache/`, `openvino/` and `*_config*.json` are git-ignored. Never commit weights
  (YOLO26 is AGPL-3.0).
- Keep the Blue Iris response shape byte-compatible with CodeProject.AI: `success, message, error,
  predictions[{x_min,y_min,x_max,y_max,confidence,label}], count, command, moduleId, executionProvider,
  canUseGPU, inferenceMs, processMs, analysisRoundTripMs`.
- Tests: pure post-processing tests in `tests/postprocess.rs` with synthetic tensors; HTTP integration
  test skips when `models/` is absent.

## Working style for agents

- The main agent orchestrates; implementation work is delegated to subagents (Sonnet for well-specified
  porting/boilerplate, Opus for inference/backend/concurrency code). Parallel agents own disjoint files
  and build against the shared interfaces in `src/lib.rs`, `src/api.rs`, `src/config.rs`, `src/model/mod.rs`.
- Every subagent must finish with `cargo build` and `cargo test` passing for its files (cargo's target
  lock serializes concurrent builds; that is fine).
- Commit at the end of each phase with a message ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
