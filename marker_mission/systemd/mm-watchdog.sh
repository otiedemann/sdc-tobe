#!/usr/bin/env bash
# Watchdog wrapper for marker_mission.
# If the drone fails to take off (state_not_ready / takeoff_failed) within
# TAKEOFF_TIMEOUT seconds after a takeoff attempt, the service is hard-killed
# and restarted automatically. Maximum RESTART_MAX restarts before giving up.

set -euo pipefail

REPO="${SDC_FC_REPO:-/home/sdc/sdc-tobe}"
PY="${REPO}/.venv/bin/python3"
LOG="/tmp/mm.log"
TAKEOFF_TIMEOUT=15   # seconds to wait after takeoff before declaring stuck
RESTART_MAX=5
restart_count=0

start_service() {
    # Hard-kill any leftover process
    pkill -9 -f 'marker_mission.app' 2>/dev/null || true
    sleep 2
    # Start fresh, redirect to log
    "$PY" -u -m marker_mission.app > "$LOG" 2>&1 &
    SERVICE_PID=$!
    echo "[watchdog] started pid=$SERVICE_PID (attempt $((restart_count+1))/$RESTART_MAX)"
}

SERVICE_PID=0
start_service

while true; do
    sleep 2

    # If process died on its own, restart it
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
        echo "[watchdog] process $SERVICE_PID exited — restarting"
        restart_count=$((restart_count+1))
        if [ "$restart_count" -ge "$RESTART_MAX" ]; then
            echo "[watchdog] max restarts reached, giving up"
            exit 1
        fi
        start_service
        continue
    fi

    # Scan log for takeoff failure after a takeoff attempt
    if grep -q 'phase init -> takeoff' "$LOG" 2>/dev/null; then
        if grep -q 'state_not_ready\|takeoff_failed\|takeoff API error' "$LOG" 2>/dev/null; then
            echo "[watchdog] takeoff failure detected — hard-killing and restarting"
            kill -9 "$SERVICE_PID" 2>/dev/null || true
            restart_count=$((restart_count+1))
            if [ "$restart_count" -ge "$RESTART_MAX" ]; then
                echo "[watchdog] max restarts reached, giving up"
                exit 1
            fi
            sleep 3
            start_service
        fi
    fi
done
