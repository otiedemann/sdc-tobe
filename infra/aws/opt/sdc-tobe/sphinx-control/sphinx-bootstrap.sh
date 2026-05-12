#!/bin/bash
# Bring the sphinx sim to a healthy "flyable" state, from any starting
# state (cold boot, half-wedged env, dead drone, wedged FC, whatever).
#
# Drives the system toward the goal:
#     env (UE4 ↔ gzserver via sphinx daemon on :8383)
#  → drone subprocess (firmwared netns, ARSDK on 10.202.0.1)
#  → flight controller (unified_api_server, Olympe.connect → telemetry.connected=true)
#
# Each step is a healthcheck → repair, not a blind start. Repair means
# tear down the current attempt forcefully (DELETE API + kill orphan
# processes + wait for them to actually exit) then start fresh and wait
# for the next state to become healthy.
#
# Idempotent: a no-op when everything is already healthy. Single source
# of truth for "get this box back to a working sim" — used by the Recover
# button in sphinx-control and by sphinx-bootstrap.service at boot.
set -u
log() { echo "[bootstrap $(date -Iseconds)] $*"; }

FC_VENV=/opt/sdc-tobe/.venv
DRONE_IP=10.202.0.1
SPC_API=http://localhost:8090
FC_API=http://localhost:8080
SPHINX_RPC_PORT=8383

# ─── helpers ────────────────────────────────────────────────────

env_healthy() {
    # sphinx daemon's JSON-RPC on :8383 only binds once UE4 has
    # connected to gz and the world is fully loaded. That's also the
    # condition Olympe needs to talk to the drone. Without it nothing
    # downstream works, so we use this as THE single env signal.
    ss -lnt sport = :"$SPHINX_RPC_PORT" 2>/dev/null | grep -q ":$SPHINX_RPC_PORT"
}

drone_pingable() {
    ping -c 1 -W 2 "$DRONE_IP" >/dev/null 2>&1
}

olympe_reachable() {
    "$FC_VENV/bin/python" - <<PY 2>/dev/null
import sys, olympe, logging
logging.basicConfig(level=logging.CRITICAL)
d = olympe.Drone("$DRONE_IP")
ok = d.connect(timeout=4)
try: d.disconnect()
except Exception: pass
sys.exit(0 if ok else 1)
PY
}

fc_connected() {
    local con
    con=$(curl -s --max-time 2 "$FC_API/api/telemetry" 2>/dev/null | \
        python3 -c "import json,sys
try: print(json.load(sys.stdin).get('connected'))
except Exception: pass" 2>/dev/null)
    [ "$con" = "True" ]
}

# Force every sim-related process gone. Used after an API teardown to
# make sure the next start gets a clean slate (sphinx-control's
# DELETE returns before the underlying processes actually exit).
kill_orphans() {
    pkill -9 -f "sphinx /opt/parrot-sphinx" 2>/dev/null || true
    pkill -9 gzserver 2>/dev/null || true
    pkill -9 -f UnrealApp 2>/dev/null || true
    pkill -9 -f "parrot-ue4-" 2>/dev/null || true
}

wait_for() {
    local label="$1"; local fn="$2"; local timeout_s="$3"
    local start_ts; start_ts=$(date +%s)
    while ! "$fn"; do
        if [ $(( $(date +%s) - start_ts )) -ge "$timeout_s" ]; then
            log "TIMEOUT after ${timeout_s}s waiting for: $label"
            return 1
        fi
        sleep 2
    done
    log "ok: $label after $(( $(date +%s) - start_ts ))s"
    return 0
}

# ─── stage 0: sphinx-control is up ──────────────────────────────

log "starting recovery / bootstrap"
spc_up() { curl -sf --max-time 2 "$SPC_API/api/environment" >/dev/null 2>&1; }
if ! wait_for "sphinx-control API on :8090" spc_up 60; then
    log "abort: sphinx-control not answering"; exit 1
fi

# ─── stage 1: env (UE4 + sphinx daemon) ─────────────────────────

if env_healthy; then
    log "stage 1 skip: env healthy (:$SPHINX_RPC_PORT bound)"
else
    log "stage 1: env unhealthy — tear down and start fresh"
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc"          >/dev/null
    curl -s --max-time 5 -X DELETE "$SPC_API/api/environment" >/dev/null
    sleep 4
    kill_orphans
    sleep 3
    curl -sf --max-time 10 -X POST "$SPC_API/api/environment" \
        -H "Content-Type: application/json" \
        -d '{"world_name":"sdc_arena_test"}' >/dev/null \
        || { log "abort: env POST failed"; exit 1; }
    # UE4 takes 30-60s on a warm box, longer on a cold AMI launch.
    if ! wait_for "env healthy (:$SPHINX_RPC_PORT)" env_healthy 180; then
        log "abort: env never became healthy"; exit 1
    fi
fi

# ─── stage 2: drone reachable on 10.202.0.1 ─────────────────────

if drone_pingable && olympe_reachable; then
    log "stage 2 skip: drone reachable on $DRONE_IP"
else
    log "stage 2: drone unreachable — respawn"
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc" >/dev/null
    # Stop existing drone(s) via sphinx-control's bulk endpoint
    curl -s --max-time 5 -X POST "$SPC_API/api/drones/stop-all" >/dev/null 2>&1
    sleep 4
    curl -sf --max-time 10 -X POST "$SPC_API/api/drones" \
        -H "Content-Type: application/json" \
        -d '{"drone_profile":"anafi-4k"}' >/dev/null \
        || { log "abort: drone POST failed"; exit 1; }
    if ! wait_for "drone pingable on $DRONE_IP" drone_pingable 90; then
        log "abort: drone never came up"; exit 1
    fi
    if ! wait_for "drone olympe-reachable" olympe_reachable 90; then
        log "warn: drone not olympe-reachable — continuing, FC has retry"
    fi
fi

# ─── stage 3: FC connected ──────────────────────────────────────

if fc_connected; then
    log "stage 3 skip: FC connected"
else
    log "stage 3: FC not connected — restart"
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc" >/dev/null
    sleep 4
    # sphinx-control's port-availability check sometimes falsely flags
    # 8080 as in-use right after DELETE. Retry POST a few times.
    attempt=0; fc_pid=""
    while [ "$attempt" -lt 6 ] && [ -z "$fc_pid" ]; do
        attempt=$((attempt + 1))
        sleep $((attempt * 4))
        resp=$(curl -s --max-time 10 -X POST "$SPC_API/api/fc" \
                -H "Content-Type: application/json" -d "{}")
        if echo "$resp" | grep -q '"pid"'; then
            fc_pid=$(echo "$resp" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("pid"))')
            log "FC POST ok attempt $attempt pid=$fc_pid"
        else
            log "FC POST attempt $attempt failed: $(echo "$resp" | head -c 140)"
        fi
    done
    [ -z "$fc_pid" ] && { log "abort: FC never started after $attempt attempts"; exit 1; }
    if ! wait_for "FC telemetry.connected=true" fc_connected 60; then
        log "warn: FC started but not connected after 60s — leaving as-is"
        exit 1
    fi
fi

log "recovery done — env+drone+FC all healthy"
