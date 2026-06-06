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

# Kill any stale process still holding port 8080 from a previous run
# (e.g. a zombie child that survived SIGINT when systemd restarted us).
# Without this, the new Flask server can't bind and video never appears.
PORT=8080
ZOMBIE_PIDS=$(ss -tlnpH "sport = :${PORT}" 2>/dev/null \
              | grep -oP 'pid=\K[0-9]+' | sort -u)
if [[ -n "$ZOMBIE_PIDS" ]]; then
    for ZPID in $ZOMBIE_PIDS; do
        echo "[marker-mission] port ${PORT} still held by PID ${ZPID} — killing"
        kill -9 "$ZPID" 2>/dev/null || true
    done
    sleep 1
fi

echo "[marker-mission] starting combined app with $PY"
# -u for unbuffered stdout/stderr so journalctl -f shows output live.
# -m runs the package entry point (marker_mission/app.py:main).
exec "$PY" -u -m marker_mission.app
