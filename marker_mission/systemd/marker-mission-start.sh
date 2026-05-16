#!/usr/bin/env bash
# Entry point for marker-mission.service. Launches the combined Flask
# app (unified backend routes + mission UI on one port, drone control
# in-process via drone_core). Picks the venv python if present,
# otherwise falls back to system python — same convention as
# sdc-fc-start.sh so a host with only one venv runs both units.

set -u
set -o pipefail

REPO="${SDC_FC_REPO:-/home/sdc/sdc-tobe}"
cd "$REPO"

if [[ -x "$REPO/.venv/bin/python3" ]]; then
    PY="$REPO/.venv/bin/python3"
else
    PY=/usr/bin/python3
fi

echo "[marker-mission] starting combined app with $PY"
# -u for unbuffered stdout/stderr so journalctl -f shows output live.
# -m runs the package entry point (marker_mission/app.py:main).
exec "$PY" -u -m marker_mission.app
