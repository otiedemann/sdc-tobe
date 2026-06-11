#!/usr/bin/env bash
# Wrapper for direct_takeoff_test.py: stops the FC service so it releases the
# Anafi link, runs the direct-to-drone takeoff probe, then ALWAYS restarts the
# service afterwards (even on error or Ctrl-C).
#
# Usage (on the FC):
#     tools/direct_takeoff_test.sh
#     DRONE_IP=192.168.42.1 HOVER_S=4 tools/direct_takeoff_test.sh
#
# Needs sudo for systemctl (sdc-fc has a sudoers drop-in for the service user).
set -euo pipefail

SERVICE="${SERVICE:-sdc-fc.service}"
# Resolve paths relative to this script so it works from any cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PROBE="$HERE/direct_takeoff_test.py"

# Prefer the Olympe venv; fall back to system python3.
VENV_PY="$REPO/controller_unified/.venv/bin/python3"
if [[ -x "$VENV_PY" ]]; then
    PY="$VENV_PY"
else
    PY="$(command -v python3)"
    echo "[wrap] venv python not found at $VENV_PY — falling back to $PY" >&2
fi

restart_service() {
    echo "[wrap] restarting $SERVICE ..."
    sudo systemctl start "$SERVICE" || \
        echo "[wrap] WARNING: failed to restart $SERVICE — start it manually!" >&2
}
# Guarantee the FC comes back no matter how we exit.
trap restart_service EXIT

echo "[wrap] stopping $SERVICE to release the drone link ..."
sudo systemctl stop "$SERVICE"

echo "[wrap] running direct takeoff probe ($PY) ..."
"$PY" "$PROBE" "$@"
