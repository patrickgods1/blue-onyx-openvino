# Blue Onyx Prism

Blue Iris / CodeProject.AI compatible object detection service, written in Rust on native
[OpenVINO](https://github.com/openvinotoolkit/openvino) and [ONNX Runtime](https://onnxruntime.ai/).
Runs on Intel integrated and discrete GPUs (OpenVINO), NVIDIA GPUs (CUDA/TensorRT), any DirectX 12 GPU on
Windows (DirectML), Apple silicon (CoreML) and on CPU everywhere (Windows x86_64, Linux x86_64, macOS arm64).
The default device `auto` picks the best option for the detected hardware; see
[Hardware and device selection](#hardware-and-device-selection) and [docs/PLAN.md](docs/PLAN.md).

Formerly **Blue Onyx OpenVINO**. Existing `blue_onyx_openvino_config*.json` files are renamed to
the new names on first start, and `scripts/install_service.ps1` removes the old
`BlueOnyxOpenVINOService`. Prometheus metrics are now prefixed `blue_onyx_prism_`.

Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), adding multi-model serving
(`/v1/vision/custom/{model}`), YOLO26 end-to-end models and Intel GPU inference on Windows.

## Features

- Drop-in CodeProject.AI / DeepStack compatible API for Blue Iris (`/v1/vision/detection`, `/v1/vision/custom/{model}`, `/v1/vision/custom/list`).
- Native OpenVINO inference: Intel integrated/discrete GPU on Windows and Linux, CPU everywhere, automatic GPU -> CPU fallback.
- ONNX Runtime as a second runtime (NVIDIA CUDA/TensorRT, DirectML, CoreML, CPU) with `auto` device selection by detected hardware.
- Several models served at once, each with its own worker and metrics; per-model device, threshold, class filter and lazy loading.
- Model families: YOLO26 (end-to-end), YOLOv5 (the blue-onyx IPcam models), YOLOv8/11, RT-DETRv2. ONNX or OpenVINO IR.
- Compiled-model cache (fast restarts), HTTP served while models compile, web UI (stats, test page, config editor), Prometheus metrics.
- Runs as a Windows service, systemd unit, launchd daemon or Docker container.

## Supported platforms

| Platform | Inference | Notes |
|---|---|---|
| Windows 11 x86_64 | Intel GPU (iGPU/Arc) + CPU | Primary target; Windows service. Needs the MSVC runtime redistributable. |
| Linux x86_64 | Intel GPU (`/dev/dri`) + CPU | Install `intel-opencl-icd`; user in the `render` group. systemd and Docker files provided. |
| macOS arm64 | CPU, CoreML | launchd daemon. No OpenVINO GPU plugin exists for Apple silicon; CoreML runs through ONNX Runtime. |

NVIDIA (CUDA/TensorRT) and DirectML devices use ONNX Runtime; see the matrix below.

## Hardware and device selection

`"device": "auto"` (the default) chooses per model, tries the next candidate if compile or warm-up
fails, and always ends on the CPU. Ranking: NVIDIA GPU with CUDA, Intel GPU (OpenVINO), AMD or other GPU
on Windows (DirectML), Apple silicon (CoreML), CPU (`openvino:cpu`, else `ort:cpu`).

| OS | GPU | `auto` picks | Setup | Notes |
|---|---|---|---|---|
| Windows | Intel iGPU/Arc | `openvino:gpu` | `setup-openvino` | Primary target. `ort:directml` is selectable. |
| Windows | NVIDIA | `ort:cuda` | `setup-onnxruntime --flavor cuda` | CUDA 12 and cuDNN 9 are installed by you (not bundled). Without them `auto` skips CUDA. |
| Windows | AMD / other DX12 | `ort:directml` | `setup-onnxruntime --flavor directml` | |
| Linux | Intel | `openvino:gpu` | `setup-openvino` | `intel-opencl-icd`, `render` group. |
| Linux | NVIDIA | `ort:cuda` | `setup-onnxruntime --flavor cuda` | CUDA 12 + cuDNN 9 user-installed. |
| Linux | AMD | `openvino:cpu` | `setup-openvino` | No ROCm/MIGraphX in the stock ONNX Runtime packages: CPU. |
| macOS arm64 | Apple GPU | `ort:coreml` | `setup-onnxruntime` | RT-DETRv2 runs on CoreML (~1.6-1.9x `openvino:cpu` on an M1) because its batch dimension is pinned to 1; an RT-DETR export with an unnamed dynamic dimension is refused there (MPSGraph would abort) and uses the CPU. YOLO26 needs the `.onnx` export. |
| any | none | `openvino:cpu` | `setup-openvino` | `ort:cpu` if OpenVINO is missing. |

TensorRT (`ort:tensorrt`, first engine build takes minutes) and the NPU (`openvino:npu`) can be
selected but are never chosen by `auto`. ONNX Runtime needs `.onnx` models; an IR `.xml` model uses a
sibling `<stem>.onnx` if present, otherwise its ONNX Runtime options are skipped.

Device specs: `auto`, `openvino:gpu[.N]`, `openvino:cpu`, `openvino:npu`, `ort:cuda[:N]`,
`ort:tensorrt[:N]`, `ort:directml[:N]`, `ort:coreml`, `ort:cpu`. The old `GPU`, `GPU.N` and `CPU` still
work and mean `openvino:*`. Set it globally (`device`), per model (`models[].device`) or with `--device <spec>`.

```sh
blue-onyx-prism list-devices         # detected GPUs, runnable options with reasons, and the auto pick
curl http://127.0.0.1:32168/v1/devices   # the same as JSON
```

The Config page has a device dropdown ("Auto - currently: ...", then the runnable options; options that
cannot run are greyed out with the reason).

Measured on an Apple M1 (`inferenceMs`):

| Model | `openvino:cpu` | `ort:cpu` | `ort:coreml` |
|---|---|---|---|
| IPcam-general | 37 | 69 | 104 |
| rt-detrv2-s | 101 | 235 | not run (excluded) |

On Apple silicon `auto` currently picks CoreML, which measured slower than OpenVINO CPU for these models
(CoreML splits the graph into many pieces). To use the faster option set `"device": "openvino:cpu"`.

## First run

A fresh install can be just the binary and a config that names a model. On start the service works
out what the config and the hardware need, downloads only what is missing in the background, and
serves HTTP right away (models answer `Model initializing` with the download progress, e.g.
`downloading OpenVINO runtime 42% (18/44 MB)`, until they are ready):

```json
{ "device": "auto", "models": [ { "path": "models/IPcam-general.onnx" } ] }
```

- **What gets downloaded:** the model files (if the name matches the catalog, `list-models`), the
  runtime of the best device for this machine (`auto`: CUDA on NVIDIA, OpenVINO GPU on Intel,
  DirectML on other Windows GPUs, CoreML on Apple silicon, else CPU) and the runtime of its CPU
  fallback. On a Mac that is the model (29 MB), ONNX Runtime CoreML (31 MB) and OpenVINO (44 MB).
- **When:** models load on a CPU option as soon as one is installed, then the server restarts its
  registry (in the same process, a few seconds) to switch to the better runtime when that arrives.
  A second ONNX Runtime flavor (a process can load only one) is downloaded next to the first and
  used after the process restarts.
- **Where:** next to the executable (`openvino/`, `onnxruntime/<flavor>/` with `active.txt` naming
  the active flavor, `onnxruntime/cuda-libs/`, `models/`), or under `download_dir`. Every file is
  pinned by URL, size and SHA-256 in the binary (`src/resources/catalog.rs`), downloads resume after
  an interruption, and only whitelisted libraries are extracted; nothing downloaded is executed.

| Setting | Default | Meaning |
|---|---|---|
| `auto_download` | `true` | Download what the config needs at startup. `false` (air-gapped, Docker): only report it; affected models are `Failed` with the command to run |
| `allow_large_downloads` | `false` | Allow downloads over 500 MB, in practice the NVIDIA CUDA libraries |
| `download_dir` | `null` | Root for runtimes and relative model paths (default: the executable's directory) |

Commands (the same code the service uses):

```sh
blue-onyx-prism list-resources          # what is installed, needed by the config, or available, with sizes
blue-onyx-prism fetch --for-config      # download everything the config needs now (default without flags)
blue-onyx-prism fetch --resource model:ipcam-bird --resource onnxruntime-cpu
blue-onyx-prism fetch --all-for-platform  # everything for this OS/arch (large ones need --allow-large)
```

- **Offline / air-gapped:** set `"auto_download": false`, run `fetch --for-config` on a machine with
  network (same OS/arch and config), then copy the whole directory. `list-resources` on the target
  shows anything still missing.
- **Web UI:** the Config page has a *Resources* card, grouped into Runtimes, GPU libraries, Models
  by family (YOLOv5, RT-DETRv2, D-FINE, RF-DETR) and *Local models (not in config)* (`.onnx` / `.xml`
  files in `models_dir` that no config entry uses, e.g. YOLO26 exports), with state, size,
  Download / Remove, a live progress bar and *Add to config*. `GET /v1/resources` returns the same as
  JSON (each row has a `group`; local files are in `localModels`).
  Device options that need a download show "will download ... (size)" in the Device dropdown;
  choosing one and saving starts the download and switches when it is installed.
- **NVIDIA CUDA libraries (opt-in):** `ort:cuda` needs CUDA 12 and cuDNN 9. If they are not
  installed, the `nvidia-cuda-libs` resource (about 1.9 GB on Windows, 1.7 GB on Linux: cudart,
  cuBLAS, cuDNN, cuFFT, cuRAND, nvJitLink, NVRTC from NVIDIA's PyPI wheels) provides them. It is
  only downloaded with `allow_large_downloads`, `fetch --allow-large` or
  `fetch --resource nvidia-cuda-libs`, and the libraries are covered by the
  [NVIDIA Software License Agreement](https://docs.nvidia.com/cuda/eula/) (CUDA, cuDNN
  redistributables). TensorRT is never downloaded.
- **YOLO26** weights are AGPL-3.0 and cannot be redistributed: the service exports them on
  request (Config page, *YOLO26 (exported locally, AGPL-3.0)*, or
  `fetch --resource model:yolo26s`; see [Exporting YOLO26 models](#exporting-yolo26-models)). A
  config entry pointing at a YOLO26 file that is not exported yet shows "needs export"; it is never
  exported automatically.

## Quick start

Download the archive for your OS from the [releases page](https://github.com/patrickgods1/blue-onyx-prism/releases)
(`blue-onyx-prism-<version>-<os>-<arch>.zip|tar.gz`, with a `.sha256` file). It already contains the
OpenVINO runtime in `openvino/`, the default ONNX Runtime in `onnxruntime/` (CPU on Linux, CoreML on
macOS, DirectML on Windows), the binaries and the helper scripts. Or build from source (see
[Developer setup](#developer-setup)) and run `blue-onyx-prism setup-openvino`. For NVIDIA, DirectML or
CoreML devices run `blue-onyx-prism setup-onnxruntime` (`--flavor auto|cpu|cuda|directml`, default `auto`;
installs into `onnxruntime/` next to the executable, `--dir` to change).

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
| `device` | `"auto"` | Device spec (`auto`, `openvino:gpu`, `ort:cuda`, ..., or legacy `GPU`/`CPU`); see [Hardware and device selection](#hardware-and-device-selection). Falls back to CPU when the device cannot run |
| `gpu_index` | `0` | GPU to use when several are present |
| `force_cpu` | `false` | Always use the CPU |
| `cache_dir` | `"cache"` | Compiled-model cache; empty disables it |
| `openvino_dir` | `null` | OpenVINO runtime dir; default `<exe_dir>/openvino`, else the system install |
| `onnxruntime_dir` | `null` | ONNX Runtime dir (user-managed, never replaced); lookup order: this field, `ORT_DYLIB_PATH`, `onnxruntime/<flavor>/` named by `onnxruntime/active.txt`, `onnxruntime/` itself (older installs), any `onnxruntime/<flavor>/` |
| `auto_download` | `true` | Download missing runtimes and models at startup; see [First run](#first-run) |
| `allow_large_downloads` | `false` | Allow downloads over 500 MB (NVIDIA CUDA libraries) |
| `download_dir` | `null` | Root for downloaded runtimes and relative model paths (default: the executable's directory) |
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
| Python 3.10-3.12 (only for exporting YOLO26 by hand; the Config page export brings its own) | `winget install astral-sh.uv` then `uv venv --python 3.11 .venv` | `uv venv --python 3.11 .venv` or `python3 -m venv .venv` | same |

The pinned Rust channel is in `rust-toolchain.toml`; `rustup` picks it up automatically. No `build.rs`,
no system OpenVINO install is needed to build: the `openvino` crate loads the runtime at run time.

### Build and run

```sh
git clone https://github.com/patrickgods1/blue-onyx-prism.git
cd blue-onyx-prism
cargo build --release

# 1. Fetch the OpenVINO runtime (~100 MB) into ./target/release/openvino (next to the exe).
cargo run --release -- setup-openvino
# 1b. Optional, for NVIDIA / DirectML / CoreML: ONNX Runtime into ./target/release/onnxruntime.
cargo run --release -- setup-onnxruntime        # --flavor auto|cpu|cuda|directml
cargo run --release -- list-devices             # what can run here and what `auto` picks

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
cargo build --no-default-features           # OpenVINO-only build without ONNX Runtime
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

### Exporting YOLO26 models

**On demand (no repo or Python needed).** On the Config page, *Resources*, section
*YOLO26 (exported locally, AGPL-3.0)*: click *Export* next to yolo26n/s/m/l/x. The confirmation
names the download and the license. The first export sets up a private toolchain (about 350 MB
download and 1.2 GB on disk on macOS arm64, up to ~470 MB / 1.6 GB on Linux x86_64): the pinned
[uv](https://github.com/astral-sh/uv) release (SHA-256 checked), Python 3.11 installed by uv, and
the hashed package locks in `scripts/yolo26-export/locks/` (Ultralytics, PyTorch CPU, ONNX;
installed with `--require-hashes --only-binary :all:`). Then it downloads Ultralytics' official
`yolo26<size>.pt` (SHA-256 pinned), runs the embedded `scripts/export_yolo26.py` with Ultralytics'
network access and auto-install off, and moves `yolo26<size>.onnx` + `.yaml` into `models_dir`. The
row shows the stage (1/6 ... 6/6) and a progress bar, the output goes to the Logs page, and a running
export can be cancelled; one export runs at a time. *and add to config* (checked by default) appends
the model to `models` when it is done (enabled when no other model is; restart to load it). Later
exports reuse the toolchain (only the weights are downloaded; yolo26n took 55 s from scratch on an
M1, yolo26s 5 s afterwards).

Headless: `blue-onyx-prism fetch --resource model:yolo26s --allow-large` (`--allow-large`, or
`allow_large_downloads`, is only needed while the toolchain is not installed).

Where things live: everything is under `<download_dir or exe dir>/tools/`: `uv/`, `python/`,
`yolo26-env/` (the virtual environment), `yolo26-weights/`, `yolo26-work/`, `yolo26-config/`
(Ultralytics settings). *Remove export toolchain* on the `tool:uv` row (or deleting `tools/`)
frees it; exported models stay. Removing a YOLO26 row deletes its `.onnx` / `.yaml`.

This is the one feature that runs downloaded code (everything else is data loaded by whitelist); it
only runs when you click Export or name the resource on the command line. Supported where PyTorch
2.14 has CPU wheels: Windows x86_64, Linux x86_64 / aarch64, macOS arm64 (14+).

**By hand** (also makes OpenVINO IR):

```sh
uv venv --python 3.11 .venv
uv pip install -r scripts/requirements-export.txt         # or .venv/bin/pip install -r ...
.venv/Scripts/python.exe scripts/export_yolo26.py          # Windows
.venv/bin/python scripts/export_yolo26.py                  # Linux / macOS
```

This writes `yolo26{n,s,m}.{xml,bin,onnx,yaml}` and prints the `"models"` snippet for the config
file. The output directory is `--out-dir` (default `target/release/models` when it exists, i.e. next
to the release exe, else `models/`); point it at the `models` directory next to the exe you run (the
Config page names it). The files then appear under *Local models (not in config)* with an
*Add to config* button (the `.onnx` is preferred: it also runs on ONNX Runtime, incl. CoreML). YOLO26 weights are AGPL-3.0 and are never committed; `models/`, `cache/` and `openvino/` are
git-ignored.

### Converting ONNX models to OpenVINO IR

ONNX models (the YOLOv5 ipcam and RT-DETRv2 catalogs) load directly. If one fails on the ONNX
frontend, or to ship FP16 weights, convert it with the same venv:

```sh
.venv/bin/python scripts/convert_onnx_to_ir.py models/rt-detrv2-s.onnx   # -> models/rt-detrv2-s.{xml,bin}
```

To change the pinned export packages, edit `scripts/requirements-yolo26-export.txt` and regenerate
the locks with `scripts/yolo26-export/gen_locks.sh <path to the pinned uv>`.

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
- Per-model overrides: `device` (any device spec, see below), `confidence_threshold`, `object_filter`,
  `lazy` (compile on first request instead of at startup) and `gpu_precision` (`f16` default, `f32`).
- `blue-onyx-prism download-models --name IPcam-general --add-to-config` downloads and appends the
  model (family and classes file filled in) to the config, skipping names or paths already present.
  New entries are added disabled when the config already has an enabled model (otherwise the first
  one is enabled), so `download-models --all --add-to-config` does not load every model.
- `--model <path>` always enables that model.

Each model gets its own worker thread and compiled copy in memory. Expect a few hundred MB per model
on the iGPU (shared system RAM); use `lazy` for rarely used models and `device: "CPU"` to keep the GPU
free for the primary model.

### Per-model devices and benchmarking

Every model can run on its own device: `"device"` in a model entry (any spec of `list-devices`:
`openvino:cpu`, `openvino:gpu.1`, `ort:coreml`, `ort:cuda`, ...) overrides the global `device`, so
OpenVINO and ONNX Runtime models are served side by side (one ONNX Runtime flavor per process).
*Force CPU* overrides every per-model device. On the Config page the Models card has a **Device**
select per model ("Global (...)" = follow the global device), shows the execution provider in use
and links the latest benchmark recommendation.

Which device suits a model differs per model and machine (on an M1, IPcam-general runs fastest on
`openvino:cpu`, D-FINE on `ort:coreml`). The benchmark measures it: each model on each runnable device
over a set of images, graded for accuracy and speed, with a detection check against the CPU:

```sh
blue-onyx-prism benchmark --all-devices                     # config models x runnable devices x configured datasets
blue-onyx-prism benchmark --all-devices --apply             # ...and write each model's recommended device
blue-onyx-prism benchmark --all-devices --report report.html  # standalone HTML (or .md) report
blue-onyx-prism benchmark --list-datasets
```

Or open **Benchmark** in the web UI: choose datasets, models, devices and settings, **Start**
(runs in the background, one run at a time, cancelable, live progress and ETA), then **Use <device>**
per row or **Apply recommended devices to all models** (offers a restart). Benchmarking while serving
competes for the CPU/GPU with live requests. Results go to `benchmark.json` next to the config file;
the CLI and the UI read and write the same file.

- **Datasets** (`benchmark.datasets`, CLI `--dataset`): the built-in, pinned and SHA-256-checked sets
  `coco-cctv` (COCO val2017 CCTV-relevant subset, 180 images), `bmd45-cctv` (real 1080p CCTV,
  vehicles, 44 images) and `exdark-night` (low light, 160 images), downloaded on demand to
  `<data root>/bench/<id>/` (resource `bench:<id>`, also on the Config page's Resources card);
  `sample` (the embedded test image, offline); and folders (`dir:<path>` or
  `{"dir": "...", "gt": "...", "name": "..."}`, CLI `--images <dir> [--gt <file|dir>]`) with optional
  ground truth as COCO instances JSON or YOLO txt labels and tags in `manifest.json`. Without ground
  truth a dataset is scored against a reference model (the most accurate installed one, or
  `benchmark.reference_model`); such scores are marked *relative*.
- **Accuracy**: AP@0.5, AP@[.5:.95] (COCO 101-point), precision/recall/F1 at the model's confidence
  threshold, recall by object size, by dataset and by tag (day/night, complexity, resolution, crowd,
  small objects). Scored classes: person, bicycle, car, motorcycle, bus, truck, dog, cat, bird, horse
  as far as the model has them (IPcam `vehicle` = car/truck/bus); AP over all of a COCO model's classes
  is reported too.
- **Frame-level ROC AUC** (what Blue Iris decides: alert on this frame or not). Box-level ROC is
  undefined for detection, so each frame is one sample per scored class: positive when it has a
  non-ignored object of the class, negative when it has none (frames whose only objects of the class
  are ignore regions, and classes a dataset does not annotate, are left out); its score is the top
  confidence of the class in the frame (0 without a detection). AUC is exact (Mann-Whitney U, ties
  count half), per class, macro over classes (the headline) and micro (pooled), with the ROC curve and
  frame TPR / FPR (alert rate on frames with an object / false-alert rate on frames without) at the
  configured and best thresholds. It uses the 0.05+ predictions: lower scores count as 0, which can
  only understate the AUC slightly.
- **Grades**: speed from the full-request p50 (A < 50 ms, B < 100, C < 200, D < 400, else F); accuracy
  by `benchmark.accuracy_metric` / `--accuracy-metric`: `ap50` (default; AP@0.5 blended 20% with
  small-object recall, A >= 0.70, B >= 0.60, C >= 0.50, D >= 0.40) or `roc_auc` (macro frame ROC AUC,
  A >= 0.95, B >= 0.90, C >= 0.80, D >= 0.70; 0.5 is chance); overall = weighted grade points
  (`benchmark.weights`, default 60% accuracy / 40% speed). The metric also drives the ranking and the
  device recommendation's accuracy tie (0.015 AP50, 0.01 AUC); results kept from earlier runs are
  re-graded with the current metric when a run is saved.
- **Recommendation**: the best-graded device whose detections agree with the CPU reference device
  (devices that disagree, e.g. a broken FP16 path, are never recommended); within 5% of the best p50
  the configured device is kept. The page and the report also rank the models ("best model for this
  machine").
- **Confidence threshold**: the same low-threshold (0.05) predictions are scored at every confidence
  threshold, without extra inference: TP/FP/FN, precision, recall, F1, F2 and false positives per
  image, overall, per dataset, per tag (day, night, small objects) and per class. The best threshold
  per model is exact (precision and recall only change at the predictions' confidence values, so
  every distinct confidence is evaluated; filtering 0.05-threshold output at a higher threshold equals
  running at it, NMS included). Among thresholds within 0.002 of the best score the middle of the
  widest range is taken, rounded down to 0.01 (at least 0.05). The objective is
  `benchmark.threshold_objective` / `--threshold-objective`: `f1` (default), `f2` (favor recall: fewer
  missed people), `precision:<p>` (highest recall with precision >= p, e.g. `precision:0.9` for fewer
  false alerts), `recall:<r>` (highest precision with recall >= r), `youden` (highest frame-level
  TPR - FPR, Youden's J, classes pooled) or `fpr:<x>` (highest frame alert rate with at most x of the
  frames without an object alerting, e.g. `fpr:0.05`); ties go to the higher threshold. The
  recommendation text adds P/R (and frame TPR / FPR) at the best threshold.
  The server threshold is what a model reports: Blue Iris may send its own `min_confidence` per
  request (when it does, it replaces the server threshold for that request), and each camera's
  minimum confidence filters the returned objects again on top.

```sh
blue-onyx-prism benchmark --all-devices --apply-threshold    # also write each model's best confidence_threshold
blue-onyx-prism benchmark --apply --apply-threshold --threshold-objective f2
blue-onyx-prism benchmark --threshold-search --tag night --threshold-objective recall:0.9   # stored predictions, no inference
blue-onyx-prism benchmark --threshold-search --dataset exdark-night --class person --iou 0.6 --json
blue-onyx-prism benchmark --all-devices --accuracy-metric roc_auc --threshold-objective youden
blue-onyx-prism benchmark --threshold-search --threshold-objective fpr:0.05   # frame ROC from stored predictions
```

The predictions behind the search are stored in `benchmark-preds.json` next to `benchmark.json`
(`[class, confidence, x_min, y_min, x_max, y_max]` per prediction and image, per model and device). The
Benchmark page's **Confidence search** card runs the same search (models, device, datasets, tags,
classes, objective with target, IoU) in milliseconds and shows P/R/F1(/F2) vs threshold with hover
values, the configured and best thresholds marked, the frame ROC curve (with ROC AUC per class and frame
TPR / FPR at the configured and best thresholds), a summary and a 0.05-step table, **Apply** /
**Apply all** (writes `confidence_threshold`, same restart flow) and CSV copy/download.

The `benchmark` config section holds the defaults (**Save as default** on the page writes it):

```json
"benchmark": {
  "datasets": ["coco-cctv", "bmd45-cctv", "exdark-night"],
  "max_images_per_dataset": 0, "devices": [], "models": [],
  "warmup": 3, "repeat_per_image": 1, "reference_model": null,
  "weights": {"accuracy": 0.6, "speed": 0.4}, "auto_download_datasets": true,
  "threshold_objective": "f1", "accuracy_metric": "ap50"
}
```

## Web UI

Open `http://<host>:32168/` in a browser:

| Page | What it does |
|---|---|
| `/` | Models (state badge with download progress or failure reason, device, requests, queue), the execution providers *in use*, the `auto` pick, runtimes and GPUs, uptime, API usage; warns when *Force CPU* overrides the device; shows a hint when a newer release exists |
| `/stats` | Per-model state, runtime, device, CPU fallback, requests, dropped, queue, inference/process/round-trip avg/min/max; updates in place every 5 s from `/stats.json` |
| `/test` | Pick, drop or paste an image, choose a model and `min_confidence`; the page posts to `/v1/vision/custom/{model}` and draws the boxes client-side, with a detections table and the raw JSON (same code path and metrics as the API; without JavaScript the form posts to `/test` and the server draws the image) |
| `/config` | Models card (which models load, the default model, **Save and restart**), Resources, server/inference/logging settings with "applies now" / "needs restart" tags and the `models` list as JSON under *Advanced*; **Restart server** reloads the file, recompiles the enabled models and rebinds the port without restarting the process (the page waits for the server and reloads) |
| `/benchmark` | Benchmark models x devices x datasets in the background (progress, ETA, cancel); grade cards, model ranking, sortable per-device tables (fastest, recommended and configured marked), breakdowns by dataset/tag/class/size/resolution, per-image drill-down with ground truth and predictions drawn; **Use <device>** / **Apply recommended to all** write per-model devices; best confidence threshold per model with a P/R/F1 vs threshold chart, **Apply threshold** / **Apply all thresholds**; **Confidence search** over the stored predictions (filters, objective, chart with hover values, tables, CSV) |
| `/logs` | The last 2,000 log events of this process, live (polls `/logs.json`), with level filter, search, pause/follow, Copy and Download .txt, and the server log level (applies immediately; `POST /config/loglevel` with `level=debug`, form or query) |
| `/prometheus` | Prometheus metrics (`blue_onyx_prism_*{model="..."}`), linked in the footer with the JSON endpoints |

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

**macOS** (launchd):

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
| GET | `/logs`, `/logs.json?after=<seq>&level=<min>` | log page; recent log events newer than `after` at `level` (`trace`..`error`) or more severe, with `last` as the next cursor |
| GET, POST | `/test`, `/config` | test page, config editor |
| POST | `/config/restart`, `/config/loglevel` | reload config, change log level |
| GET | `/v1/devices` | device options (runnable, downloadable with size, unavailable with reason) and the `auto` pick |
| GET | `/v1/resources` | downloadable resources for this platform: state (installed, downloading with %, queued, needed, optional, available, failed with error and retry), size, what they provide, which models wait for them |
| POST | `/v1/resources/download` | form `id` (e.g. `model:ipcam-bird`); large resources also need `confirm_large=1` unless `allow_large_downloads` |
| POST | `/v1/resources/remove` | form `id`; refused (409) while downloading, while the runtime is loaded ("restart required") or while an enabled model uses the files |
| GET | `/v1/benchmark` | benchmark state, progress (ETA), partial results of a running benchmark, the saved results (`benchmark.json` without per-image details), model ranking, per-model devices |
| POST | `/v1/benchmark` | start (202; 409 while one runs); form fields, all optional with the `benchmark` config as default: `model`, `device`, `dataset` (repeated), `max_images`, `repeat`, `warmup`, `reference_model`, `accuracy_weight` |
| POST | `/v1/benchmark/cancel`, `/v1/benchmark/settings` | cancel; save the form fields as the `benchmark` config defaults |
| POST | `/v1/benchmark/apply` | `model` + `device` (`""`/`global` clears it), or `all=1` for every saved recommendation; `restart=1` restarts |
| POST | `/v1/benchmark/apply-threshold` | `model` + `threshold` (pairs, repeatable), or `all=1` for every saved best threshold; writes `confidence_threshold`; `restart=1` restarts |
| GET, POST | `/v1/benchmark/threshold-search` | GET: searchable models/devices/datasets/tags/classes; POST: `model` (repeated or `all`), `device` (default recommended), `dataset`, `tag`, `class` (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`), `iou` (0.5): best threshold, P/R/F1/F2/FP per image at best vs configured, exact curve, 0.05 grid, per dataset/class, CSV; 400 when nothing is stored (run a benchmark first) |
| GET | `/v1/benchmark/images?model=&device=`, `/v1/benchmark/image?set=&file=` | per-image drill-down data; an image of the saved results |
| POST | `/v1/resources/add-to-config` | form `id` of a downloaded model, or `local:<file>` for a model file in `models_dir` (family `auto`): append it to `models` |

The detection response is byte-compatible with CodeProject.AI: `success, message, error, predictions
[{x_min, y_min, x_max, y_max, confidence, label}], count, command, moduleId, executionProvider,
canUseGPU, inferenceMs, processMs, analysisRoundTripMs`.

## Benchmark

```sh
blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --repeat 50 --warmup 5
blue-onyx-prism-benchmark --model models/yolo26s.xml --family yolo26 --device GPU --compare-cpu --json
blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --all-devices
```

`blue-onyx-prism-benchmark` takes the same flags as `blue-onyx-prism benchmark` (see
[Per-model devices and benchmarking](#per-model-devices-and-benchmarking)). `--device` takes any device
spec and `--all-devices` benchmarks every runnable option for each model (grades, agreement,
recommendation, `benchmark.json`). Flags: `--model`, `--family`, `--device`, `--all-devices`, `--apply`,
`--apply-threshold`, `--threshold-objective`, `--threshold-search` (with `--tag`, `--class`, `--search-model`,
`--search-device`, `--iou`),
`--report <file.html|.md>`, `--no-save`, `--force-cpu`, `--image` (one file), `--dataset` (repeatable),
`--images <dir>`, `--gt`, `--max-images`, `--list-datasets`, `--reference-model`, `--accuracy-weight`,
`--repeat` (timed runs per image; 100 for a single image), `--warmup`, `--compare-cpu` (also runs on CPU
and diffs detections), `--cache-dir` (`""` disables), `--threads`, `--classes`, `--min-confidence`,
`--config` (enabled models from the config when no `--model`), `--json`, `-v`.
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
- **macOS:** only the CPU plugin exists for OpenVINO on Apple silicon. `auto` uses CoreML through ONNX Runtime
  (run `setup-onnxruntime`); set `"device": "openvino:cpu"` if that is slower for your models.
- **`ort:*` device shows as unavailable:** run `list-devices` for the reason. Usually `setup-onnxruntime` has not
  been run, or (NVIDIA) CUDA 12 / cuDNN 9 is not installed.
- **`Unable to find the openvino_c library`:** run `blue-onyx-prism setup-openvino` or set
  `openvino_dir` / `OPENVINO_INSTALL_DIR`. On Windows install the Visual C++ redistributable.
- **Windows error 1053 / `ServicesPipeTimeout`:** the service did not report in time; reboot once after
  installing (the timeout registry value applies at boot).
- **Blue Iris shows `Unknown model`:** the name in the custom models field must match a configured model name.

## License

MIT. Portions derived from [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT, Marcus Asteborg).
See `LICENSE`. YOLO26 weights and the Ultralytics exporter are AGPL-3.0 and are not distributed
with this project: the on-demand export downloads the official weights from Ultralytics' GitHub
release on your request and exports them on your machine (see above). The blue-onyx IPcam models and RT-DETRv2 models are downloaded from their own
Hugging Face repositories under their own licenses.
