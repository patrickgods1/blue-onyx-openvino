#!/bin/sh
# Regenerate the hashed package locks of the on-demand YOLO26 export environment from
# scripts/requirements-yolo26-export.txt. The service embeds the locks (src/resources/export.rs)
# and installs them with `uv pip install --require-hashes --no-deps --only-binary :all:`, so every
# wheel is checked against these hashes.
#
#   scripts/yolo26-export/gen_locks.sh <path to uv> [exclude-newer timestamp]
#
# Use the pinned uv (catalog `tool:uv`, src/resources/catalog.rs). The timestamp freezes the
# transitive dependencies; keep it at least two weeks in the past and update the pins in
# requirements-yolo26-export.txt in the same commit.
#
# - locks/macos-arm64.txt: macOS arm64 (macOS 14+, the oldest PyTorch 2.14 supports), from PyPI.
# - locks/torch-cpu-index.txt: Linux x86_64 / aarch64 and Windows x86_64, with torch/torchvision
#   CPU builds from the PyTorch CPU index (`--torch-backend cpu`); the three resolutions must be
#   identical (checked below). macOS x86_64 has no PyTorch 2.14 wheels and is not supported.
set -eu
UV="$1"
EXCLUDE_NEWER="${2:-2026-09-26T00:00:00Z}"
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$DIR/../requirements-yolo26-export.txt"
OUT="$DIR/locks"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

compile() { # <python-platform> <output> [extra args]
  t="$1"; o="$2"; shift 2
  MACOSX_DEPLOYMENT_TARGET=14.0 UV_NO_CONFIG=1 UV_CACHE_DIR="$TMP/cache" "$UV" pip compile "$IN" \
    --python-version 3.11 --python-platform "$t" --generate-hashes --only-binary :all: \
    --exclude-newer "$EXCLUDE_NEWER" --no-header --no-annotate -q -o "$o" "$@"
}

compile aarch64-apple-darwin "$OUT/macos-arm64.txt"
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-pc-windows-msvc; do
  compile "$t" "$TMP/$t.txt" --torch-backend cpu
done
cmp "$TMP/x86_64-unknown-linux-gnu.txt" "$TMP/aarch64-unknown-linux-gnu.txt"
cmp "$TMP/x86_64-unknown-linux-gnu.txt" "$TMP/x86_64-pc-windows-msvc.txt"
cp "$TMP/x86_64-unknown-linux-gnu.txt" "$OUT/torch-cpu-index.txt"
for f in "$OUT"/*.txt; do echo "$f: $(grep -c '==' "$f") packages"; done
