#!/usr/bin/env bash
# Install Blue Onyx OpenVINO as a launchd daemon on macOS (CPU inference only on Apple silicon).
# Usage: sudo scripts/install_launchd.sh [/usr/local/blue-onyx-openvino]
set -euo pipefail

INSTALL_DIR="${1:-/usr/local/blue-onyx-openvino}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$HERE/deploy/com.blueonyx.openvino.plist"
PLIST_DST="/Library/LaunchDaemons/com.blueonyx.openvino.plist"

if [ ! -x "$INSTALL_DIR/blue-onyx-openvino" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi

sed "s#/usr/local/blue-onyx-openvino#$INSTALL_DIR#g" "$PLIST_SRC" > "$PLIST_DST"
chown root:wheel "$PLIST_DST"
chmod 644 "$PLIST_DST"
launchctl bootout system "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap system "$PLIST_DST"
launchctl print system/com.blueonyx.openvino | head -20
echo "Logs: $INSTALL_DIR/blue-onyx-openvino.log"
