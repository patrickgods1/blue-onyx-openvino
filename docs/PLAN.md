# Plan: Blue Onyx OpenVINO — cross-platform Blue Iris object detection service on native OpenVINO (Rust)

## Context

You want a Blue Iris AI server like [blue-onyx](https://github.com/xnorpx/blue-onyx) that runs on Windows 11 with an Intel i5-8500 + UHD 630 iGPU, using OpenVINO for inference, with a model good at people/animals/cars in CCTV footage (YOLO26 or similar). It must also build and run on Linux x86_64 and macOS arm64, and live in a public GitHub repo.

**Decision: start a new Rust project using native OpenVINO (the Intel `openvino` crate), not a fork.** Reasons found during research:

- blue-onyx reaches OpenVINO only through ONNX Runtime's OpenVINO execution provider, and only on Linux. Its `build.rs` compiles ONNX Runtime 1.29 from source (and `sed`-patches the OpenVINO provider). Extending that to Windows means building ORT+OpenVINO EP with MSVC yourself, and a community report on HD 630 found the EP path ~2x slower than native OpenVINO.
- Native OpenVINO reads ONNX and IR directly, has a compiled-model cache, FP16 on the iGPU, and the Rust bindings (`openvino` 0.11.0, Intel-maintained, pregenerated bindings so no libclang) can load the shared libraries at runtime, so the service ships as one exe + a lib folder on every OS.
- blue-onyx is MIT, so its API shapes, service pattern, HTML pages, model catalog and release workflow can be copied. What we gain over it: multi-model in one instance with real `/v1/vision/custom/{model}` support, YOLO26 end-to-end models, and the Intel GPU on Windows.

**Decisions you made:** new Rust service; detection + custom-model endpoints; ship YOLO26 (n/s/m), RT-DETRv2 and MikeLud's YOLOv5 ipcam models; full feature parity with blue-onyx extras; cross-platform (Windows, Linux, macOS arm64); public GitHub repo "Blue Onyx OpenVINO".

**Verified environment facts (this dev box):** i5-8600K (6C/6T), 16 GB, UHD 630 driver 31.0.101.2140 with OpenCL runtime present, plus a GTX 1070 Ti (ignored). No Rust, no VS Build Tools, no cmake; Python is Anaconda 3.8 (too old for ultralytics/openvino wheels). Git 2.45 present with identity set; `gh` 2.52 installed but **not logged in**. The target i5-8500 box is the same GPU generation, so GPU testing here is representative.

## Naming and repository

- GitHub repo slug `blue-onyx-openvino` (display/description "Blue Onyx OpenVINO"), public, MIT with blue-onyx attribution in LICENSE/README. Crate `blue-onyx-openvino`, lib `blue_onyx_openvino`, bins `blue-onyx-openvino`, `blue-onyx-openvino-service` (Windows only), `blue-onyx-openvino-benchmark`, `test-blue-onyx-openvino`. Windows service name `BlueOnyxOpenVINOService`; config `blue_onyx_openvino_config.json` (+ `_service.json`). Nothing collides with upstream, so both can coexist on one machine.
- Repo creation (phase 0): `git init -b main` in this folder, `.gitignore` (`target/`, `models/`, `cache/`, `openvino/` lib dir, `*.log`, `*_config*.json`, `.venv/`), initial commit of the skeleton, then **you run `gh auth login`** (interactive, browser), then `gh repo create blue-onyx-openvino --public --description "Blue Onyx OpenVINO" --source . --push`. Commits happen at the end of each phase.

## Cross-platform support

| | Windows x86_64 | Linux x86_64 | macOS arm64 |
|---|---|---|---|
| OpenVINO CPU | yes | yes | yes (Apple silicon CPU plugin; archive is CPU-only) |
| OpenVINO GPU | Intel iGPU/dGPU (driver provides OpenCL) | Intel GPU via `/dev/dri` + intel-compute-runtime (same as blue-onyx's Docker) | no GPU plugin; CPU only |
| OpenVINO libs | `openvino_c.dll` + plugins from the Windows `.zip` | `libopenvino_c.so` from the Linux `.tgz` (or apt/pip) | `libopenvino_c.dylib` from the macOS `_arm64.tgz` |
| Run as service | Windows service bin | `deploy/blue-onyx-openvino.service` systemd unit + `Dockerfile` (runtime image with `--device /dev/dri`) | `deploy/com.blueonyx.openvino.plist` launchd |
| Logging sink | event log (`tracing-layer-win-eventlog`, cfg windows) + file | stdout/journald + file | stdout + file |

- **Library discovery:** `openvino-finder` searches `OPENVINO_BUILD_DIR`, `OPENVINO_INSTALL_DIR`, `INTEL_OPENVINO_DIR`, then the OS library-path variable entries (`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`), then `/opt/intel/openvino*` defaults. It never checks the exe dir and never pip site-packages. So `backend/dll.rs` becomes `backend/libs.rs`: before `Core::new()`, if `<exe_dir>/openvino/` exists (or config `openvino_dir`), set `OPENVINO_INSTALL_DIR` to it at runtime (`std::env::set_var`, read by the finder at lookup time on all OSes; on macOS `DYLD_LIBRARY_PATH` is read only at process launch, which is why the env var route is used). The shipped folder keeps the archive's `runtime/{bin|lib}/<arch>/Release` layout so the finder's known subdirectories match, and `plugins.xml` sits next to the C library.
- **Fetching libs is a Rust subcommand, not a script:** `blue-onyx-openvino setup-openvino [--version 2026.4.0] [--dest <exe_dir>/openvino]` downloads the right archive for the OS/arch from `https://storage.openvinotoolkit.org/repositories/openvino/packages/<ver>/{windows,linux,macos}/...` (zip on Windows, tgz elsewhere; verify exact filenames before pinning), extracts into an isolated temp dir, copies only the runtime libs (`openvino`, `openvino_c`, `openvino_intel_cpu_plugin`, `openvino_intel_gpu_plugin` where present, `openvino_ir_frontend`, `openvino_onnx_frontend`, `plugins.xml`, `tbb12`/`libtbb`), and prints what it did. Deps: `reqwest` (stream), `zip`, `tar` + `flate2`. CI uses the same subcommand. Users with a system OpenVINO (apt, brew, pip) can instead point `openvino_dir` at it.
- **macOS risk:** openvino-rs CI skips macOS over an `@rpath` issue in the dynamic-link path; the `runtime-linking` feature dlopens by absolute path, which sidesteps it, but if plugin dylibs fail to load, mitigation is `install_name_tool -add_rpath` in `setup-openvino` or a documented `DYLD_LIBRARY_PATH` launch wrapper. CPU-only there regardless.
- **Linux GPU:** `device: "GPU"` works when `/dev/dri/renderD*` is accessible and `intel-opencl-icd` is installed; `Dockerfile` based on `openvino/ubuntu24_runtime:<ver>` (same as blue-onyx) copies the binary in. Systemd unit runs as a user in the `render` group.
- **Code gating:** `#[cfg(windows)]` only for the service bin, event log layer and the DXGI adapter listing; everything else is portable. `system_info.rs` reports CPU name via `raw-cpuid` on x86 and `sysctl`/`/proc/cpuinfo` fallback on arm64/Linux.
- **CI (`.github/workflows/ci.yml`):** matrix `windows-latest`, `ubuntu-latest`, `macos-latest` (arm64): `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` (unit tests need no OpenVINO), then `cargo run -- setup-openvino` and a smoke test that starts the server on CPU with the smallest downloadable model (`ipcam-bird`) and posts an image. **Release (`release.yml`, copied in shape from upstream):** on tag `x.y.z`, build the three targets, bundle binaries + the `openvino/` lib folder + `templates`-free exe (templates are compiled in) + scripts/deploy files into `.zip` (Windows, plus NSIS installer via `cargo-packager` later) and `.tar.gz` (Linux, macOS), with SHA-256 files, as a draft release.

## Project layout

Single crate, edition 2024, lib + bins.

```
Cargo.toml  rust-toolchain.toml  LICENSE  README.md  .gitignore  Dockerfile
.github/workflows/{ci,release}.yml
deploy/blue-onyx-openvino.service   deploy/com.blueonyx.openvino.plist   deploy/docker-compose.yml
src/lib.rs            module list, VERSION
src/cli.rs            clap Cli + subcommands (run, setup-openvino, download-models, list-models); merge rules copied from upstream (CLI overrides file only when non-default; written back)
src/config.rs         Config + ModelConfig (JSON next to exe; service variant *_service.json)
src/api.rs            serde structs copied from upstream api.rs (camelCase response, snake_case Prediction)
src/server.rs         axum Router + handlers + AppState
src/metrics.rs        atomics per model; Prometheus text rendering
src/registry.rs       ModelRegistry: loads N models, one worker thread + bounded channel each, name lookup
src/worker.rs         worker loop: recv -> preprocess -> infer -> postprocess -> reply
src/startup.rs        ModelState {Initializing, Ready, Failed} shared with HTTP (serve immediately while compiling)
src/backend/mod.rs    OvBackend: Core/CompiledModel/InferRequest, tensor IO, output copy-out
src/backend/device.rs device string, properties, GPU->CPU fallback, FULL_DEVICE_NAME reporting
src/backend/libs.rs   OPENVINO_INSTALL_DIR/PATH setup from exe dir, preflight check, per-OS lib names
src/setup_openvino.rs download/extract/copy OpenVINO runtime per OS/arch
src/model/mod.rs      ModelFamily enum {yolo26, yolo5, yolo8, rtdetr, auto}, Detection, common filter/clamp
src/model/preprocess.rs  decode -> letterbox or stretch -> CHW f32 (fast_image_resize)
src/model/nms.rs      class-aware greedy NMS
src/model/classes.rs  `NAMES:` yaml loader (same format as upstream/HF yamls)
src/model/{yolo26,yolo5,yolo8,rtdetr}.rs  per-family postprocess
src/image.rs          JPEG decode (zune-jpeg; `image` fallback), draw boxes, save annotated
src/download.rs       hf-hub download of xnorpx/rt-detr2-onnx and xnorpx/blue-onyx-yolo5 catalogs
src/system_info.rs    CPU name, OpenVINO available devices/versions
src/update.rs         GitHub release check for /v1/status/updateavailable
src/bin/blue_onyx_openvino.rs            server + subcommands
src/bin/blue_onyx_openvino_service.rs    windows-service wrapper (cfg windows; stub main elsewhere)
src/bin/blue_onyx_openvino_benchmark.rs  compile/warmup/p50/p95 latency, --device GPU|CPU, --compare-cpu
src/bin/test_blue_onyx_openvino.rs       posts an image N times, prints JSON
templates/{welcome,stats,config,test}.html, prometheus.txt   (askama; structure copied from upstream)
assets/favicon.ico, assets/style.css, assets/<permissive>.ttf  (include_bytes)
scripts/export_yolo26.py       Ultralytics -> OpenVINO IR + NAMES yaml
scripts/convert_onnx_to_ir.py  `ovc` fallback for ONNX models the frontend rejects
scripts/install_service.ps1, scripts/uninstall_service.ps1   (Windows)
scripts/install_systemd.sh, scripts/install_launchd.sh       (Linux, macOS)
tests/postprocess.rs, tests/integration_http.rs
```

**Key deps:** `openvino = { version = "0.11", features = ["runtime-linking"] }` (confirm the feature forwards to `openvino-sys`; else add `openvino-sys` with it), `axum` (http1,json,multipart,tokio), `tokio` (rt,macros,sync,time,signal,fs), `crossbeam-channel`, `askama`, `clap` (derive), `serde`/`serde_json`/`serde_yaml`, `tracing` + `tracing-subscriber` (env-filter) + `tracing-appender`, `anyhow`, `fast_image_resize`, `zune-jpeg`, `jpeg-encoder`, `image` (png,jpeg), `imageproc`, `ab_glyph`, `hf-hub` (rustls-tls,tokio), `indicatif`, `reqwest` (rustls-tls,json,multipart,stream), `zip`, `tar`, `flate2`, `num_cpus`; x86 only: `raw-cpuid`; Windows-only: `windows-service`, `tracing-layer-win-eventlog`, `windows` (DXGI). Release profile: lto, codegen-units=1. No `build.rs`.

## Multi-model design

Config example:
```json
{ "port": 32168, "request_timeout_secs": 15, "worker_queue_size": 0,
  "device": "GPU", "gpu_index": 0, "force_cpu": false, "cache_dir": "cache", "openvino_dir": null,
  "confidence_threshold": 0.5, "nms_iou": 0.5, "object_filter": [],
  "log_level": "info", "log_path": null, "save_image_path": null, "save_ref_image": false,
  "intra_threads": 0, "models_dir": "models", "default_model": "yolo26s",
  "models": [
    { "name": "yolo26s", "path": "models/yolo26s.xml", "family": "yolo26", "classes": "models/yolo26s.yaml" },
    { "name": "ipcam-general", "path": "models/IPcam-general.onnx", "family": "yolo5", "device": "CPU" },
    { "name": "rt-detrv2-s", "path": "models/rt-detrv2-s.onnx", "family": "rtdetr", "lazy": true } ] }
```
`ModelConfig { name (default: file stem), path, family, classes (default `<stem>.yaml`), device override, confidence_threshold override, object_filter override, lazy }`. `--model <path> --family <f>` with no `models` array = single-entry list. Relative paths resolve against the exe dir.

- One `Core` (in the registry, `Mutex`) with properties set once; models compiled **sequentially** on a loader thread, then each `CompiledModel` (it is `Send`) moves into its own worker thread which creates the `InferRequest`, runs one warmup, measures it, sizes its bounded channel (`request_timeout / inference_ms`, clamp 1..64 unless `worker_queue_size > 0`), and flips `Ready`. `lazy` models compile on first request.
- `WorkItem = (VisionDetectionRequest, oneshot::Sender<VisionDetectionResponse>, Instant)` as upstream.
- Routing: `/v1/vision/detection` -> `default_model`; `/v1/vision/custom/{model}` -> case-insensitive lookup by name with extension stripped (Blue Iris sends the file stem); unknown -> HTTP 200 `success:false, error:"Unknown model"`. `custom/list` -> all configured names, `moduleId "ObjectDetectionOpenVINO"`, `inferenceDevice` from the default model.
- Memory (iGPU shares RAM): roughly YOLO26n ~100 MB, YOLO26s ~200 MB, RT-DETRv2-s ~300 MB, each YOLOv5 ipcam ~80 MB resident. 3-5 models fine on 16 GB; `lazy` and per-model `device: "CPU"` are the knobs.

## Inference abstraction

`OvBackend::load(core, &ModelConfig, &DeviceSelection)` = `read_model_from_file(path, "")` (ONNX or IR; empty weights path is the documented no-weights form) -> introspect inputs/outputs by index (`Model::get_input_by_index` + `Node::get_name/get_element_type/get_partial_shape`; there is no lookup-by-name on `Model`) -> `reshape` dynamic dims to static -> `compile_model` with fallback -> `create_infer_request` in the worker. `infer()` fills a reused f32 input `Tensor` via `get_data_mut::<f32>()`, sets extra named tensors, calls `request.infer()`, and copies every output into owned `Vec`s keyed by name (slices borrow the request).

| Family | Input | Preprocess | Output | Postprocess |
|---|---|---|---|---|
| yolo26 (IR exported `nms=False`) | `images` f32 [1,3,640,640] RGB 0..1 | letterbox (scale `s=min(640/w,640/h)`, gray 114 pad, centered) | `[1,300,6]` = x1,y1,x2,y2,conf,cls in input px | conf >= thr; undo letterbox `(x-pad)/s`; no NMS |
| yolo5 (xnorpx ipcam ONNX) | `images` [1,3,640,640] | letterbox | `[1,N,5+C]` cx,cy,w,h,obj,cls... | conf = obj*max_cls; xyxy; per-class NMS IoU 0.5; undo letterbox |
| yolo8 (YOLOv8/11 raw, bonus) | `images` [1,3,640,640] | letterbox | `[1,4+C,8400]` channel-major | transpose read; score = max cls; NMS; undo letterbox |
| rtdetr (xnorpx rt-detrv2 ONNX) | `images` [1,3,640,640] + `orig_target_sizes` **i64** [1,2] | plain stretch, 0..1, no mean/std | `labels` i64, `boxes` f32 xyxy, `scores` f32 (match by name) | score >= thr; no NMS; scale by resize factors |

For rtdetr, copy upstream exactly: pass `orig_target_sizes = [640, 640]` (so [w,h] vs [h,w] is moot) and scale boxes by `orig/640` per axis. `family: "auto"` picks by output rank/dims at load (`[.,300,6]`->yolo26, last dim `5+C`->yolo5, `[.,4+C,8400]`->yolo8, 3 named outputs->rtdetr).

Common post-step: `min_confidence` from the request overrides config when > 0; class filter (case-insensitive); clamp to image; round to `usize` -> `Prediction { x_min, y_min, x_max, y_max, confidence, label }`.

## OpenVINO specifics

- Device: `force_cpu` -> CPU; else `GPU` (or `DeviceType::from("GPU.N")`, which maps to `Other` and round-trips) with CPU fallback. Preflight `core.available_devices()`; if GPU absent (always on macOS), log and use CPU without trying. On GPU compile error: `warn!`, retry CPU, report `executionProvider = "OpenVINO CPU (fallback)"`. Otherwise `"OpenVINO GPU (Intel(R) UHD Graphics 630)"` from the `FULL_DEVICE_NAME` property; `canUseGPU` = GPU listed.
- Properties: `CacheDir = <exe_dir>/cache` on GPU and CPU (cuts the 20-60 s GPU JIT to seconds on restart); `HintPerformanceMode = "LATENCY"`; GPU `HintInferencePrecision = "f16"` (config `gpu_precision: "f32"` escape hatch); CPU `InferenceNumThreads = intra_threads` when > 0 (recommend 4 of 6 cores), `HintNumRequests = "1"`.
- `Core::new()` loads the library (runtime-linking) and rejects OpenVINO < 2025.1; pin `setup-openvino` to a version openvino-rs CI tests (2026.1.0 or 2026.4.0). The MSVC runtime redistributable must be present on Windows targets.

## Model acquisition

- `blue-onyx-openvino download-models [--all|--yolo5|--rtdetr|--name X] [--dir models]`: hf-hub catalogs copied from upstream (`xnorpx/rt-detr2-onnx`: rt-detrv2-{s,ms,m,l,x}.{onnx,yaml}; `xnorpx/blue-onyx-yolo5`: delivery, IPcam-animal, ipcam-bird, IPcam-combined, IPcam-dark, IPcam-general, package .{onnx,yaml}); `--add-to-config` appends entries with family inferred. `list-models` shows catalog + local presence.
- `scripts/export_yolo26.py` (Python 3.11 venv with `ultralytics openvino`): for n/s/m, `YOLO("yolo26X.pt").export(format="openvino", imgsz=640, nms=False, dynamic=False, quantize=16)` (current docs say `quantize` replaces `half`/`int8`; fall back to `half=True` on older ultralytics; `--int8` -> `quantize=8, data=coco128.yaml`). Copies `.xml/.bin` to `models/`, converts `metadata.yaml` names to a `NAMES:` yaml, asserts the output shape ends in `[300, 6]`, prints the config snippet. YOLO26 is AGPL-3.0: never commit weights; users export locally.
- `models/`, `cache/` and `openvino/` live next to the exe; all git-ignored.

## HTTP layer

| Method | Path | Notes |
|---|---|---|
| GET | `/` | welcome: model table (state/device), links |
| POST | `/v1/vision/detection` | default model |
| POST, GET | `/v1/vision/custom/list` | all configured model names |
| POST | `/v1/vision/custom/{model}` | named model |
| GET | `/v1/status/updateavailable` | GitHub release check |
| GET | `/stats` | per-model rows, auto-refresh |
| GET | `/prometheus` | `blue_onyx_openvino_*{model=...}` |
| GET, POST | `/test` | upload form + annotated result + JSON, model selector |
| GET, POST | `/config` | edit JSON fields, write file |
| POST | `/config/restart` | trigger restart token |
| POST | `/config/loglevel` | `tracing_subscriber::reload` handle |
| GET | `/favicon.ico`, `/static/style.css` | include_bytes |

Handler flow: multipart (`image`, optional `min_confidence`; 32 MB body limit) -> resolve model -> if not Ready return 200 `success:false,"Model initializing"` -> `sender.is_full()` ? count drop -> send -> `timeout(30s, rx)` -> JSON with `inferenceMs/processMs/analysisRoundTripMs`. Worker discards items older than `request_timeout`. Metrics per model: requests, dropped, total/min/max of inference, processing, round-trip (atomics).

## Services and deployment

- **Windows:** `blue_onyx_openvino_service.rs`: `define_windows_service!`, `OWN_PROCESS`, controls Interrogate/Stop/Shutdown + user codes 130 (stop) / 131 (restart), `StartPending` wait hint 600 s, current-thread tokio runtime, outer loop reloads the service config, rebuilds registry, retries after 5 s on failure; `libs::prepare_environment()` before the runtime; event log source + optional rolling file log. `install_service.ps1` (admin): event log source, `ServicesPipeTimeout = 600000`, `sc.exe create BlueOnyxOpenVINOService ... start= auto obj= LocalSystem type= own`, `sc.exe failure` auto-restart, firewall rule, start. `uninstall_service.ps1` reverses it. If the GPU plugin fails only under LocalSystem, `-Account` runs it as the user.
- **Linux:** the main binary handles SIGTERM; `deploy/blue-onyx-openvino.service` (`Restart=on-failure`, `SupplementaryGroups=render video`, `WorkingDirectory=` install dir); `Dockerfile` + `deploy/docker-compose.yml` with `/dev/dri` passthrough and a volume for `models/`, `cache/`, config.
- **macOS:** `deploy/com.blueonyx.openvino.plist` (KeepAlive, WorkingDirectory) installed by `scripts/install_launchd.sh`.

## Implementation phases (commit at the end of each)

0. **Prereqs + repo:** install rustup (stable MSVC), VS 2022 Build Tools "Desktop development with C++"; `conda create -n yolo python=3.11` + `pip install ultralytics openvino`; `git init -b main`, `.gitignore`, skeleton `Cargo.toml`/README/LICENSE, first commit; you run `gh auth login`; `gh repo create blue-onyx-openvino --public --description "Blue Onyx OpenVINO" --source . --push`.
1. **Single model on CPU + `/v1/vision/detection` + `setup-openvino`** — cli/config/api, backend (CPU path), libs.rs, setup_openvino.rs, model/{preprocess,nms,classes,yolo5}, worker/startup/registry (single entry), server (detection + `/`), download CLI. Milestone: `cargo run -- setup-openvino` then `cargo run -- --model models/IPcam-general.onnx --family yolo5 --force-cpu` returns `person` boxes for a test JPEG. `tests/postprocess.rs` for yolo5. CI workflow added (all three OSes build + unit tests).
2. **GPU + cache + fallback + remaining families** — device.rs full, yolo26/yolo8/rtdetr, `export_yolo26.py`, reshape of dynamic dims. Milestone: yolo26s on `GPU` logs the UHD 630 name, second start compiles in < 5 s, `--force-cpu` works, RT-DETRv2-s loads (or via `ovc`).
3. **Multi-model + custom endpoints** — registry with N workers, `models[]` config, `custom/list`, `custom/{model}`, per-model metrics. Milestone: yolo26s (GPU) + ipcam-general (CPU) served concurrently.
4. **UI parity** — templates, stats, prometheus, test page with drawn boxes, config page + restart, loglevel, update check, annotated-image saving.
5. **Services, deploy files, benchmark, test client, release workflow, README** — Windows service + scripts, systemd/launchd/Docker files, benchmark, test client, `release.yml`, Blue Iris setup docs; CI smoke test on all OSes.

## Verification

- **Unit:** synthetic tensors per family (yolo26 letterbox undo on 1920x1080: `s=1/3, pad_y=140`; yolo5 NMS keeps one of two IoU-0.8 same-class boxes and both when classes differ; yolo8 channel-major transpose; rtdetr output matching regardless of order); NMS IoU math; `NAMES:` parse; filter case-insensitivity. Run on all three OSes in CI without OpenVINO.
- **Integration (local + CI smoke):** `setup-openvino`, `download-models --name ipcam-bird`, start server on CPU, then
  `Invoke-RestMethod http://127.0.0.1:32168/v1/vision/detection -Method Post -Form @{ image = Get-Item .\test.jpg; min_confidence='0.4' }` (curl `-F image=@test.jpg` on Linux/macOS) -> `success:true`, `inferenceMs > 0`; same via `/v1/vision/custom/<name>`; `custom/list` contains the names. `tests/integration_http.rs` automates this (skipped when `models/` is absent).
- **GPU confirmation (Windows dev box, later the i5-8500 box; Linux if available):** startup log shows `available devices: [CPU, GPU]` and the UHD 630 full name; `/stats` shows `OpenVINO GPU`; Task Manager GPU 0 "Compute" graph rises during the benchmark; `cache/` gets a blob and restart is fast.
- **Benchmark expectations (UHD 630 FP16 / i5-8500 CPU):** yolo26n ~25-45 ms GPU, ~40-60 ms CPU; yolo26s ~60-100 ms GPU; IPcam-general ~30-50 ms GPU; rt-detrv2-s ~90-150 ms GPU. Flag if GPU is slower than CPU (fallback or FP32 path). The dev box's GTX 1070 Ti is irrelevant; only `GPU`/`GPU.0` = Intel is used.
- **Blue Iris:** Settings -> AI -> CodeProject.AI server `127.0.0.1:32168`; camera -> Alerts -> AI: default detection (hits `/v1/vision/detection`) or custom models `yolo26s,ipcam-general` (hits `/v1/vision/custom/<name>`); Trigger now; alert shows the AI label; `/stats` counters increment per model; `dropped` stays 0 under multi-camera load.
- **Cross-platform:** CI green on windows/ubuntu/macos runners; release artifacts for all three.

## Risks and mitigations

- **Gen9.5 GPU on OpenVINO 2026.x:** docs list "Intel UHD Graphics" and the GPU plugin README says Gen8+, but it is unverified on 2026.4. If `available_devices()` lacks GPU, try the 2025.1/2025.2 archive (still passes the crate's >= 2025.1 check); CPU fallback keeps the service working either way.
- **macOS dylib loading:** openvino-rs skips macOS CI (rpath); runtime-linking by absolute path should work, else `install_name_tool -add_rpath` in `setup-openvino` or a `DYLD_LIBRARY_PATH` wrapper. CPU only on macOS.
- **Archive filenames/URLs per OS:** verify on storage.openvinotoolkit.org before pinning in `setup-openvino`; keep the version in one constant.
- **openvino-rs gaps:** no name lookup on `Model` (iterate by index); `Tensor::get_data<T>` is unchecked (assert element type first); if core-level `set_property` with an empty device fails, set per device.
- **First compile time:** StartPending 600 s, HTTP served during init, CacheDir, sequential loading, `lazy`.
- **RT-DETRv2 ONNX ops on the ONNX frontend / int64 input on GPU:** fall back to `ovc` conversion, then to CPU for that model.
- **FP16 accuracy:** `gpu_precision: "f32"` per model; benchmark `--compare-cpu` diffs confidences.
- **Export shape drift in Ultralytics:** export script validates `[.,300,6]`; `family: auto` routes a raw `[1,84,8400]` export to yolo8.
- **Licenses:** YOLO26 AGPL-3.0 (weights exported locally, not committed); RT-DETRv2 Apache-2.0; check MikeLud model license before bundling; code MIT with blue-onyx attribution.
- **gh not logged in:** phase 0 pauses for your interactive `gh auth login`; everything else in phase 0 proceeds before that.
