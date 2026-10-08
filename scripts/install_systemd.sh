#!/usr/bin/env bash
# Install Blue Onyx OpenVINO as a systemd service on Linux.
# Usage: sudo scripts/install_systemd.sh [/opt/blue-onyx-openvino]
set -euo pipefail

INSTALL_DIR="${1:-/opt/blue-onyx-openvino}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$HERE/deploy/blue-onyx-openvino.service"

if [ ! -x "$INSTALL_DIR/blue-onyx-openvino" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi

id -u blueonyx >/dev/null 2>&1 || useradd --system --home "$INSTALL_DIR" --shell /usr/sbin/nologin blueonyx
usermod -aG render,video blueonyx 2>/dev/null || true
chown -R blueonyx:blueonyx "$INSTALL_DIR"

sed "s#/opt/blue-onyx-openvino#$INSTALL_DIR#g" "$UNIT_SRC" > /etc/systemd/system/blue-onyx-openvino.service
systemctl daemon-reload
systemctl enable --now blue-onyx-openvino
systemctl --no-pager status blue-onyx-openvino | head -20
echo "Logs: journalctl -u blue-onyx-openvino -f"
