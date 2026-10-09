# Blue Onyx OpenVINO

Blue Iris / CodeProject.AI compatible object detection service, written in Rust on native
[OpenVINO](https://github.com/openvinotoolkit/openvino). Runs on Intel integrated and discrete GPUs
(Windows, Linux) and on CPU everywhere (Windows x86_64, Linux x86_64, macOS arm64).

Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT) without ONNX Runtime, adding
multi-model serving (`/v1/vision/custom/{model}`), YOLO26 end-to-end models and Intel GPU inference
on Windows.

Status: under construction. The implementation plan is in [docs/PLAN.md](docs/PLAN.md); the module
map and conventions are in [CLAUDE.md](CLAUDE.md).

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
git clone https://github.com/patrickgods1/blue-onyx-openvino.git
cd blue-onyx-openvino
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

### Repository layout

- `src/` Rust crate (see CLAUDE.md for the module map)
- `scripts/` service install scripts (Windows PowerShell, systemd, launchd) and the YOLO26 export script
- `deploy/` systemd unit, launchd plist, docker-compose; `Dockerfile` at the root
- `docs/PLAN.md` implementation plan and phase checklist
- `.github/workflows/ci.yml` build, lint, test and a CPU smoke test on Windows, Linux and macOS
- `.claude/` Claude Code project configuration (agent definitions)

## Blue Iris configuration

Settings -> AI: enable "Use AI server", address `127.0.0.1`, port `32168`. Per camera -> Alerts -> AI:
either default object detection (served by the default model via `/v1/vision/detection`) or custom
models by name (served via `/v1/vision/custom/{name}`). Details follow once phase 3 lands.

## License

MIT. Portions derived from blue-onyx (MIT, Marcus Asteborg). See `LICENSE`.
