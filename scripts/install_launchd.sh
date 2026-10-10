#!/usr/bin/env bash
# Install Blue Onyx Prism as a launchd daemon on macOS (CPU inference only on Apple silicon).
# Usage: sudo scripts/install_launchd.sh [/usr/local/blue-onyx-prism]
set -euo pipefail

INSTALL_DIR="${1:-/usr/local/blue-onyx-prism}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$HERE/deploy/com.blueonyx.prism.plist"
PLIST_DST="/Library/LaunchDaemons/com.blueonyx.prism.plist"

if [ ! -x "$INSTALL_DIR/blue-onyx-prism" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi

# Pre-rename daemon (Blue Onyx OpenVINO): unload and remove it so the two don't fight over the port.
OLD_PLIST="/Library/LaunchDaemons/com.blueonyx.openvino.plist"
if [ -f "$OLD_PLIST" ]; then
  launchctl bootout system "$OLD_PLIST" 2>/dev/null || true
  rm -f "$OLD_PLIST"
fi

sed "s#/usr/local/blue-onyx-prism#$INSTALL_DIR#g" "$PLIST_SRC" > "$PLIST_DST"
chown root:wheel "$PLIST_DST"
chmod 644 "$PLIST_DST"
launchctl bootout system "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap system "$PLIST_DST"
launchctl print system/com.blueonyx.prism | head -20
echo "Logs: $INSTALL_DIR/blue-onyx-prism.log"
