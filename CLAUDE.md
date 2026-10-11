# Blue Onyx Prism

Cross-platform object-detection HTTP service for Blue Iris (CodeProject.AI-compatible API),
written in Rust on **native OpenVINO** via the `openvino` crate (runtime-linking), with
**ONNX Runtime** (`ort` crate, load-dynamic) as a second runtime for NVIDIA (CUDA/TensorRT),
DirectML and CoreML. Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), with
multi-model support and `auto` device selection by detected hardware (see `docs/PLAN.md`, phase 6).

Primary target: Windows 11, Intel i5-8500 + UHD 630 iGPU. Must also build/run on Linux x86_64 and macOS arm64.

## Build & run

```powershell
cargo build --release
cargo run -- setup-openvino                 # downloads OpenVINO runtime libs into ./openvino (next to exe)
cargo run -- setup-onnxruntime [--flavor auto|cpu|cuda|directml] [--dir <d>]   # ORT libs into ./onnxruntime
cargo run -- list-devices                    # runnable device options and the `auto` pick
cargo run -- download-models --name IPcam-general
cargo run -- fetch [--for-config] [--resource <id>] [--all-for-platform] [--allow-large]   # pre-download what the config needs
cargo run -- list-resources [--check-urls]   # installed / needed / available; --check-urls checks every pinned URL (weekly CI)
cargo run -- fetch --resource model:yolo26s --allow-large   # on-demand YOLO26 export (pinned uv + Python env in <data root>/tools)
cargo run -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
cargo run -- --model models/IPcam-general.onnx --family yolo5 --device ort:cpu
cargo run -- benchmark --all-devices [--apply] [--report r.html] [--dataset coco-cctv|bmd45-cctv|exdark-night|dir:<p>] [--max-images 20]   # grade models x devices; benchmark.json next to the config
cargo run -- benchmark --list-datasets       # built-in image sets (assets/bench/*.json, resource bench:<id>)
cargo build --no-default-features            # OpenVINO-only build (drops the `onnxruntime` feature / `ort`)
cargo test                                   # unit tests need no OpenVINO
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

On-demand YOLO26 export (`src/resources/export.rs`, the only code that runs downloaded programs; user-triggered
only, never by `auto_download`): `tool:uv` + `scripts/yolo26-export/locks/*.txt` (hashed; regenerate with
`scripts/yolo26-export/gen_locks.sh`) + the embedded `scripts/export_yolo26.py`; everything under `<data root>/tools/`.

Python export env (YOLO26 -> OpenVINO IR, by hand): `.venv` (Python 3.11, `ultralytics`, `openvino`),
created with `uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-export.txt`;
run `.venv/Scripts/python.exe scripts/export_yolo26.py` (Windows) or `.venv/bin/python scripts/export_yolo26.py`.

OpenVINO libs are discovered through `openvino-finder`: `OPENVINO_INSTALL_DIR` (Windows; on Unix
`backend/libs.rs` loads `openvino_c` by path) pointing at `<download root>/openvino` when present, or the OS library path.

On-demand resources: pins, URLs, SHA-256 and sizes of everything downloadable (OpenVINO, ONNX Runtime
flavors, DirectML, `nvidia-cuda-libs`, models) live in `src/resources/catalog.rs`; change them only in
reviewed commits. The service downloads what the config + hardware need at startup (`auto_download`, default
true; `allow_large_downloads`, default false, gates > 500 MB; `download_dir`, default the exe dir).

ONNX Runtime libs: `onnxruntime_dir` in the config, else `ORT_DYLIB_PATH`, else `onnxruntime/<flavor>/` named by
`onnxruntime/active.txt`, else the old flat `onnxruntime/`, else any flavor dir. `ONNXRUNTIME_VERSION` 1.24.4 and
`DIRECTML_VERSION` are in the catalog. The `ort` crate's api-level must stay <= the pinned ORT lib (1.24), because
the DirectML NuGet stops there. A process loads one ORT flavor only. `onnxruntime/cuda-libs` (opt-in NVIDIA
wheels, ~1.9 GB) is preloaded by path before the CUDA provider loads.

Device spec (`device` in config/model entry, `--device`): `auto` (default), `openvino:gpu[.N]`,
`openvino:cpu`, `openvino:npu`, `ort:cuda[:N]`, `ort:tensorrt[:N]`, `ort:directml[:N]`, `ort:coreml`,
`ort:cpu`; legacy `GPU`/`GPU.N`/`CPU` mean `openvino:*`. `auto` ranks: NVIDIA+CUDA, Intel GPU (OpenVINO),
AMD/other on Windows (DirectML), macOS arm64 (CoreML), CPU (`openvino:cpu`, else `ort:cpu`). TensorRT and
NPU are never auto. RT-DETR runs on CoreML only with every input dim pinned (a dynamic batch dim aborts MPSGraph in ORT 1.24.4). ORT needs `.onnx` files.

## Architecture (see the plan in the repo history / README for detail)

- `src/server.rs` axum HTTP (one tokio current-thread runtime). Blue Iris calls
  `POST /v1/vision/detection` (default model) and `POST /v1/vision/custom/{model}` (named model),
  `POST /v1/vision/custom/list`. Responses use `api.rs` structs (camelCase; `Prediction` snake_case).
- `src/registry.rs` loads N models from config sequentially on one `Core`; each model gets its own
  worker thread (`worker.rs`) and bounded crossbeam channel. `startup.rs` tracks Initializing/Ready/Failed
  so HTTP serves immediately while models compile.
- `src/backend/` runtimes behind a `Backend` enum (`OpenVino | Ort`): OpenVINO wrapper (`Core`,
  `CompiledModel`, `InferRequest`, properties CACHE_DIR/LATENCY/f16 on GPU, tensor IO); `spec.rs` parses
  device specs; `detect.rs` finds GPUs (DXGI / sysfs / Apple); `select.rs` builds the ranked, pure
  `DeviceOption` list and the `auto` pick; `plan.rs` turns it into per-model load candidates (compile +
  warm-up, falling down the list, CPU last); `ort.rs` is the ONNX Runtime backend (all `ort` code lives here).
  `onnx_rewrite.rs`: `ort:coreml` loads a cached copy without HardSigmoid/HardSwish/opset<13 Split (protobuf wire-level, `cache/coreml/<stem>-<sha16>-v<N>.onnx`, `BOP_COREML_REWRITE=0` off).
  Also `GET /v1/devices` and the config-page device dropdown.
- `src/model/` model families: `yolo26` (end-to-end `[1,300,6]`), `yolo5` (`[1,N,5+C]` + NMS),
  `yolo8` (`[1,4+C,8400]` + NMS), `detr` (D-FINE: `pixel_values` -> `logits/pred_boxes`), `rfdetr` (same, ImageNet-normalized, COCO-91 head), `rtdetr` (`images` + i64 `orig_target_sizes`; `labels/boxes/scores`).
  Preprocess = letterbox (YOLO) or stretch (RT-DETR) to 640x640 RGB f32 0..1, CHW.
- `src/resources/` on-demand resources: `catalog` (pins), `resolve` (pure needs from config + hardware +
  installed), `manager` (one download thread, Range resume, SHA-256, staging + atomic rename, `.installed.json`,
  `.downloads.lock`, backoff), `extract` (whitelist-only, nothing executed), `provision` (startup: models wait
  with progress; a new runtime starts a new registry generation), `status` (`/v1/resources` + UI actions),
  `commands` (`fetch`, `list-resources`), `export` (YOLO26 export: stages uv -> Python -> packages -> weights ->
  export -> install, fake-able `CommandRunner`/`Fetcher`, one at a time, cancellable).
- `src/config.rs` JSON config next to the exe; CLI overrides only non-default values and writes back.
  The server holds it in `config_store.rs` (revision + change log, detects edits on disk); every UI writer goes
  through `save_config`, and forms 3-way merge against the values they were rendered from (`config_merge.rs`, 409 on conflict).
- Windows service in `src/bin/blue_onyx_prism_service.rs` (cfg windows). systemd/launchd/Docker in `deploy/`.

## Conventions

- Edition 2024, `anyhow` for app errors, `tracing` for logs. No `build.rs`. No `unsafe` outside `backend/`.
- Everything portable unless it must be `#[cfg(windows)]` (service, event log, DXGI).
- Copy ideas and API shapes from the blue-onyx reference freely (MIT, attributed in LICENSE),
  including its ONNX Runtime usage. Keep all `ort` code in `src/backend/ort.rs`.
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
