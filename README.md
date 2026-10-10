# Blue Onyx Prism

Blue Iris / CodeProject.AI compatible object detection service, written in Rust on native
[OpenVINO](https://github.com/openvinotoolkit/openvino). Runs on Intel integrated and discrete GPUs
(Windows, Linux) and on CPU everywhere (Windows x86_64, Linux x86_64, macOS arm64). ONNX Runtime
(NVIDIA CUDA/TensorRT, DirectML, CoreML) with automatic device selection by detected hardware is
in progress; see [docs/PLAN.md](docs/PLAN.md), phases 6 and 7.

Formerly **Blue Onyx OpenVINO**. Existing `blue_onyx_openvino_config*.json` files are renamed to
the new names on first start, and `scripts/install_service.ps1` removes the old
`BlueOnyxOpenVINOService`. Prometheus metrics are now prefixed `blue_onyx_prism_`.

Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), adding multi-model serving
(`/v1/vision/custom/{model}`), YOLO26 end-to-end models and Intel GPU inference on Windows.

## Features

- Drop-in CodeProject.AI / DeepStack compatible API for Blue Iris (`/v1/vision/detection`, `/v1/vision/custom/{model}`, `/v1/vision/custom/list`).
- Native OpenVINO inference: Intel integrated/discrete GPU on Windows and Linux, CPU everywhere, automatic GPU -> CPU fallback.
- Several models served at once, each with its own worker and metrics; per-model device, threshold, class filter and lazy loading.
- Model families: YOLO26 (end-to-end), YOLOv5 (the blue-onyx IPcam models), YOLOv8/11, RT-DETRv2. ONNX or OpenVINO IR.
- Compiled-model cache (fast restarts), HTTP served while models compile, web UI (stats, test page, config editor), Prometheus metrics.
- Runs as a Windows service, systemd unit, launchd daemon or Docker container.

## Supported platforms

| Platform | Inference | Notes |
|---|---|---|
| Windows 11 x86_64 | Intel GPU (iGPU/Arc) + CPU | Primary target; Windows service. Needs the MSVC runtime redistributable. |
| Linux x86_64 | Intel GPU (`/dev/dri`) + CPU | Install `intel-opencl-icd`; user in the `render` group. systemd and Docker files provided. |
| macOS arm64 | CPU only | launchd daemon. No GPU plugin exists for Apple silicon. |

## Quick start

Download the archive for your OS from the [releases page](https://github.com/patrickgods1/blue-onyx-prism/releases)
(`blue-onyx-prism-<version>-<os>-<arch>.zip|tar.gz`, with a `.sha256` file). It already contains the
OpenVINO runtime in `openvino/`, the binaries and the helper scripts. Or build from source (see
[Developer setup](#developer-setup)) and run `blue-onyx-prism setup-openvino`.

Windows (PowerShell):

```powershell
Expand-Archive blue-onyx-prism-*-windows-x86_64.zip .; cd blue-onyx-prism-*-windows-x86_64
.\blue-onyx-prism.exe download-models --name IPcam-general --add-to-config
.\blue-onyx-prism.exe            # GPU by default; add --force-cpu to stay on the CPU
```

Linux / macOS:

```sh
tar xzf blue-onyx-prism-*-linux-x86_64.tar.gz && cd blue-onyx-prism-*-linux-x86_64
./blue-onyx-prism download-models --name IPcam-general --add-to-config
./blue-onyx-prism                 # macOS: CPU is used automatically
```

Check it: open `http://127.0.0.1:32168/`, or `./test-blue-onyx-prism` (see [Test client](#test-client)).
`--model <path> --family yolo5|yolo26|yolo8|rtdetr|auto` runs a single model without editing the config.
`blue-onyx-prism --help` lists every option; `list-models` shows the downloadable catalog.

The first GPU start compiles each model (20-60 s on an iGPU) and stores the result in `cache/`;
later starts take seconds. The HTTP server is up immediately and answers `Model initializing` until
the model is ready.

## Configuration

`blue_onyx_prism_config.json` next to the executable (`--config <file>` to use another). A CLI
flag overrides the file and the merged result is written back. Relative paths in the file resolve
against the executable directory. The Windows service reads `blue_onyx_prism_config_service.json`
(created with debug logging on first start).

| Field | Default | Meaning |
|---|---|---|
| `port` | `32168` | HTTP listen port (binds `0.0.0.0`) |
| `request_timeout_secs` | `15` | Max time a request may wait in the queue plus processing |
| `worker_queue_size` | `0` | Per-model queue length; 0 = auto from timeout and measured inference time |
| `device` | `"GPU"` | `GPU`, `GPU.N` or `CPU`; falls back to CPU when no GPU is available |
| `gpu_index` | `0` | GPU to use when several are present |
| `force_cpu` | `false` | Always use the CPU |
| `cache_dir` | `"cache"` | Compiled-model cache; empty disables it |
| `openvino_dir` | `null` | OpenVINO runtime dir; default `<exe_dir>/openvino`, else the system install |
| `confidence_threshold` | `0.5` | Default minimum confidence (a request's `min_confidence` > 0 overrides it) |
| `nms_iou` | `0.5` | IoU threshold for NMS-based families |
| `object_filter` | `[]` | Only report these labels (case-insensitive); empty = all |
| `log_level` | `"info"` | `trace`, `debug`, `info`, `warn`, `error` |
| `log_path` | `null` | Directory for daily rolling log files (default: stdout) |
| `save_image_path` | `null` | Save annotated detections here |
| `save_ref_image` | `false` | Also save the unannotated image |
| `intra_threads` | `0` | CPU inference threads (0 = OpenVINO default; try 4 of 6 cores) |
| `models_dir` | `"models"` | Where `download-models` puts files |
| `default_model` | `null` | Model serving `/v1/vision/detection`; default = first enabled entry |
| `models` | `[]` | Model list, see [Multiple models](#multiple-models) |

Model entry fields: `name`, `path` (`.onnx` or IR `.xml`), `family` (`auto`, `yolo26`, `yolo5`, `yolo8`,
`rtdetr`), `classes` (YAML with `NAMES:`; default `<stem>.yaml`, then COCO-80), `device`,
`confidence_threshold`, `object_filter`, `lazy`, `gpu_precision`, `enabled` (default `true`; disabled
models stay in the config but are not loaded or served).

## Blue Iris setup

1. Settings -> AI: enable the AI server, choose CodeProject.AI Server, address `127.0.0.1` (or the
   host running this service), port `32168`. (Older versions: the "Use AI server on IP/port" fields.)
2. Per camera: Camera settings -> Alerts -> Artificial Intelligence. Leave the custom models field
   empty to use the default object detection (`/v1/vision/detection`, served by `default_model`), or
   enter the configured model names comma separated, e.g. `ipcam-general,yolo26s`
   (each is served via `/v1/vision/custom/<name>`; matching is case-insensitive and ignores a file extension).
3. Click "Trigger now" on the camera (or use the AI test in Blue Iris); the alert shows the label.
4. Check `http://127.0.0.1:32168/stats`: per-model request counters rise and `dropped` should stay 0.
   If it does not, lower the number of models on the GPU or move one to `"device": "CPU"`.

## Developer setup

### Prerequisites

| | Windows | Linux (x86_64 / aarch64) | macOS (Apple silicon) |
|---|---|---|---|
| Rust | `winget install Rustlang.Rustup` (stable, MSVC) | `curl https://sh.rustup.rs -sSf \| sh` | `curl https://sh.rustup.rs -sSf \| sh` |
| C/C++ toolchain | Visual Studio 2022 Build Tools with the "Desktop development with C++" workload (`winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`) | `sudo apt install build-essential pkg-config` | `xcode-select --install` |
| Intel GPU (optional) | Intel graphics driver (OpenCL comes with it) | `sudo apt install intel-opencl-icd` and access to `/dev/dri` (`render` group) | not available (CPU only) |
| Python 3.10-3.12 (only for exporting YOLO26) | `winget install astral-sh.uv` then `uv venv --python 3.11 .venv` | `uv venv --python 3.11 .venv` or `python3 -m venv .venv` | same |

The pinned Rust channel is in `rust-toolchain.toml`; `rustup` picks it up automatically. No `build.rs`,
no system OpenVINO install is needed to build: the `openvino` crate loads the runtime at run time.

### Build and run

```sh
git clone https://github.com/patrickgods1/blue-onyx-prism.git
cd blue-onyx-prism
cargo build --release

# 1. Fetch the OpenVINO runtime (~100 MB) into ./target/release/openvino (next to the exe).
cargo run --release -- setup-openvino

# 2. Get a model. Either download a ready-made ONNX model from Hugging Face ...
cargo run --release -- download-models --name IPcam-general
# ... or export YOLO26 to OpenVINO IR (needs the Python env, see below).

# 3. Run the server (CPU) and test it.
cargo run --release -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
curl -F image=@test.jpg -F min_confidence=0.4 http://127.0.0.1:32168/v1/vision/detection
```

Drop `--force-cpu` to use the Intel GPU (`--device GPU`, default). The first GPU start compiles the
model (20-60 s on an iGPU); later starts load from `cache/`.

Development loop:

```sh
cargo test                                  # unit + synthetic post-processing tests, no OpenVINO needed
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

### Exporting YOLO26 models

```sh
uv venv --python 3.11 .venv
uv pip install -r scripts/requirements-export.txt         # or .venv/bin/pip install -r ...
.venv/Scripts/python.exe scripts/export_yolo26.py          # Windows
.venv/bin/python scripts/export_yolo26.py                  # Linux / macOS
```

This writes `models/yolo26{n,s,m}.{xml,bin,yaml}` and prints the `"models"` snippet for the config
file. YOLO26 weights are AGPL-3.0 and are never committed; `models/`, `cache/` and `openvino/` are
git-ignored.

### Converting ONNX models to OpenVINO IR

ONNX models (the YOLOv5 ipcam and RT-DETRv2 catalogs) load directly. If one fails on the ONNX
frontend, or to ship FP16 weights, convert it with the same venv:

```sh
.venv/bin/python scripts/convert_onnx_to_ir.py models/rt-detrv2-s.onnx   # -> models/rt-detrv2-s.{xml,bin}
```

### Repository layout

- `src/` Rust crate (see CLAUDE.md for the module map)
- `scripts/` service install scripts (Windows PowerShell, systemd, launchd) and the YOLO26 export script
- `deploy/` systemd unit, launchd plist, docker-compose; `Dockerfile` at the root
- `docs/PLAN.md` implementation plan and phase checklist
- `.github/workflows/ci.yml` build, lint, test and a CPU smoke test on Windows, Linux and macOS; `release.yml` builds the per-OS archives and a draft release
- `.claude/` Claude Code project configuration (agent definitions)

## Multiple models

List any number of models in `models[]` of the config file:

```json
{
  "default_model": "yolo26s",
  "models": [
    { "name": "yolo26s", "path": "models/yolo26s.xml", "family": "yolo26" },
    { "path": "models/IPcam-general.onnx", "family": "yolo5", "device": "CPU",
      "confidence_threshold": 0.4, "object_filter": ["person", "car"] },
    { "path": "models/rt-detrv2-s.onnx", "family": "rtdetr", "lazy": true, "gpu_precision": "f32" },
    { "path": "models/ipcam-animal.onnx", "family": "yolo5", "enabled": false }
  ]
}
```

- Names: `name`, else the file stem. Matching is case-insensitive and a `.onnx`/`.xml` suffix is ignored,
  so `/v1/vision/custom/IPcam-General.onnx` reaches `IPcam-general`.
- Only models with `enabled: true` (the default) are compiled and served. Pick them on the Config
  page: the **Models** card has a *Load* checkbox and a *Default* radio per model; **Save and restart**
  applies the selection. Disabled models stay listed so switching is a click plus a restart.
- `default_model` (or `--default-model <name>`) serves `/v1/vision/detection`; default is the first
  enabled entry (also used when `default_model` names a disabled model).
- `GET`/`POST /v1/vision/custom/list` returns the loaded (enabled) model names.
- An unknown model name returns HTTP 200 with `success: false` and an error message.
- Per-model overrides: `device` (`GPU`, `GPU.1`, `CPU`), `confidence_threshold`, `object_filter`,
  `lazy` (compile on first request instead of at startup) and `gpu_precision` (`f16` default, `f32`).
- `blue-onyx-prism download-models --name IPcam-general --add-to-config` downloads and appends the
  model (family and classes file filled in) to the config, skipping names or paths already present.
  New entries are added disabled when the config already has an enabled model (otherwise the first
  one is enabled), so `download-models --all --add-to-config` does not load every model.
- `--model <path>` always enables that model.

Each model gets its own worker thread and compiled copy in memory. Expect a few hundred MB per model
on the iGPU (shared system RAM); use `lazy` for rarely used models and `device: "CPU"` to keep the GPU
free for the primary model.

## Web UI

Open `http://<host>:32168/` in a browser:

| Page | What it does |
|---|---|
| `/` | Models (state, device, requests, queue), OpenVINO version and devices, uptime, API usage; shows a hint when a newer release exists |
| `/stats` | Per-model state, device, CPU fallback, requests, dropped, queue, inference/process/round-trip avg/min/max; refreshes every 5 s. JSON at `/stats.json` |
| `/test` | Upload an image, pick a model and `min_confidence`; shows the annotated image and the JSON response (same code path and metrics as the API) |
| `/config` | Choose which models load and the default model (Models card, with **Save and restart**), edit the main settings and the `models` list (JSON) and save them to the config file; **Restart server** reloads the file, recompiles the enabled models and rebinds the port without restarting the process. The log level applies immediately (`POST /config/loglevel` with `level=debug`, form or query) |
| `/prometheus` | Prometheus metrics (`blue_onyx_prism_*{model="..."}`) |

`log_path` changes need a full process restart. The UI has no authentication: do not expose the port
beyond your LAN.

## Running as a service

**Windows** (elevated PowerShell, from the extracted archive):

```powershell
.\scripts\install_service.ps1        # service BlueOnyxPrismService, auto start, auto restart, firewall rule
.\scripts\uninstall_service.ps1
```

The script raises `ServicesPipeTimeout` to 600000 ms (applies after a reboot) so the first start can
compile models. If GPU access fails under LocalSystem, install with `-Account DOMAIN\user`.
Logs go to the Application event log (source `BlueOnyxPrism`) and, with `log_path`, to files.

**Linux** (systemd):

```sh
sudo cp -r blue-onyx-prism-<version>-linux-x86_64 /opt/blue-onyx-prism
sudo /opt/blue-onyx-prism/scripts/install_systemd.sh      # creates user blueonyx (render, video groups)
journalctl -u blue-onyx-prism -f
```

**macOS** (launchd, CPU only):

```sh
sudo cp -r blue-onyx-prism-<version>-macos-aarch64 /usr/local/blue-onyx-prism
sudo /usr/local/blue-onyx-prism/scripts/install_launchd.sh   # KeepAlive; log in blue-onyx-prism.log
```

**Docker** (Linux, Intel GPU through `/dev/dri`; image based on `openvino/ubuntu24_runtime`):

```sh
cd deploy && mkdir -p models cache config
RENDER_GID=$(getent group render | cut -d: -f3) docker compose up -d --build
```

Put models in `deploy/models` and a `blue_onyx_prism_config.json` in `deploy/config`.

## HTTP API

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/vision/detection` | multipart `image`, optional `min_confidence`; default model |
| POST | `/v1/vision/custom/{model}` | same, named model |
| POST, GET | `/v1/vision/custom/list` | configured model names |
| GET | `/v1/status/updateavailable` | GitHub release check |
| GET | `/`, `/stats`, `/stats.json`, `/prometheus` | UI and metrics |
| GET, POST | `/test`, `/config` | test page, config editor |
| POST | `/config/restart`, `/config/loglevel` | reload config, change log level |

The detection response is byte-compatible with CodeProject.AI: `success, message, error, predictions
[{x_min, y_min, x_max, y_max, confidence, label}], count, command, moduleId, executionProvider,
canUseGPU, inferenceMs, processMs, analysisRoundTripMs`.

## Benchmark

```sh
blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --repeat 50 --warmup 5
blue-onyx-prism-benchmark --model models/yolo26s.xml --family yolo26 --device GPU --compare-cpu --json
```

Flags: `--model`, `--family`, `--device`, `--force-cpu`, `--image`, `--repeat`, `--warmup`,
`--compare-cpu` (also runs on CPU and diffs detections), `--cache-dir` (`""` disables), `--threads`,
`--classes`, `--min-confidence`, `--config` (enabled models from the config when no `--model`), `--json`, `-v`.
If GPU latency is not clearly below CPU latency, the GPU was probably not used (check the log) or
the FP32 path is active.

## Test client

```sh
test-blue-onyx-prism                                   # embedded sample image -> /v1/vision/detection
test-blue-onyx-prism --model ipcam-general --image cam.jpg --min-confidence 0.4
test-blue-onyx-prism --repeat 50 --parallel 4          # latency summary (client, inferenceMs, processMs)
test-blue-onyx-prism --list                            # /v1/vision/custom/list
test-blue-onyx-prism --save out.jpg                    # annotated copy of the image
```

Also `--url` (default `http://127.0.0.1:32168`) and `--interval-ms`. Exits non-zero if any
response has `success: false` or a non-200 status.

## Troubleshooting

- **GPU not used / `executionProvider` says CPU:** the log prints the available devices. On Windows
  install the current Intel graphics driver; on Linux install `intel-opencl-icd` and make sure the service
  user is in the `render` group (`ls -l /dev/dri`). If the GPU still fails the service falls back to
  CPU and reports `OpenVINO CPU (fallback)`. Under Windows services try `-Account` (see above).
- **First start is slow:** GPU kernels are compiled once (20-60 s per model) and cached in `cache/`;
  keep that directory. Windows service starts allow up to 10 minutes.
- **macOS:** only the CPU plugin exists for Apple silicon; the GPU option is ignored.
- **`Unable to find the openvino_c library`:** run `blue-onyx-prism setup-openvino` or set
  `openvino_dir` / `OPENVINO_INSTALL_DIR`. On Windows install the Visual C++ redistributable.
- **Windows error 1053 / `ServicesPipeTimeout`:** the service did not report in time; reboot once after
  installing (the timeout registry value applies at boot).
- **Blue Iris shows `Unknown model`:** the name in the custom models field must match a configured model name.

## License

MIT. Portions derived from [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT, Marcus Asteborg).
See `LICENSE`. YOLO26 weights are AGPL-3.0 and are not distributed with this project: export them
locally (see above). The blue-onyx IPcam models and RT-DETRv2 models are downloaded from their own
Hugging Face repositories under their own licenses.
