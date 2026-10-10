#!/usr/bin/env bash
# Install Blue Onyx Prism as a systemd service on Linux.
# Usage: sudo scripts/install_systemd.sh [/opt/blue-onyx-prism]
set -euo pipefail

INSTALL_DIR="${1:-/opt/blue-onyx-prism}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$HERE/deploy/blue-onyx-prism.service"

if [ ! -x "$INSTALL_DIR/blue-onyx-prism" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi

id -u blueonyx >/dev/null 2>&1 || useradd --system --home "$INSTALL_DIR" --shell /usr/sbin/nologin blueonyx
usermod -aG render,video blueonyx 2>/dev/null || true
chown -R blueonyx:blueonyx "$INSTALL_DIR"

# Pre-rename unit (Blue Onyx OpenVINO): stop and remove it so the two don't fight over the port.
if [ -f /etc/systemd/system/blue-onyx-openvino.service ]; then
  systemctl disable --now blue-onyx-openvino 2>/dev/null || true
  rm -f /etc/systemd/system/blue-onyx-openvino.service
fi

sed "s#/opt/blue-onyx-prism#$INSTALL_DIR#g" "$UNIT_SRC" > /etc/systemd/system/blue-onyx-prism.service
systemctl daemon-reload
systemctl enable --now blue-onyx-prism
systemctl --no-pager status blue-onyx-prism | head -20
echo "Logs: journalctl -u blue-onyx-prism -f"
