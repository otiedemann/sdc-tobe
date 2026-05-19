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
SPC_API=http://localhost:9090
FC_API=http://localhost:8080
SPHINX_RPC_PORT=8383

# ─── helpers ────────────────────────────────────────────────────

env_healthy() {
    # An /api/environment record exists AND sphinx-control's own
    # reconcile says its UE4 process is still alive. We don't probe
    # :8383 here because the sphinx daemon (gzserver) is spawned only
    # when a drone is added, not by env-start alone. UE4 alone is fine.
    curl -s --max-time 2 "$SPC_API/api/environment" 2>/dev/null | \
        python3 -c "import json,sys
try: d = json.load(sys.stdin)
except Exception: sys.exit(1)
sys.exit(0 if d and d.get('extras',{}).get('alive') else 1)" 2>/dev/null
}

rpc_bound() {
    # The sphinx daemon (gzserver Sphinx plugin) binds :8383 once it's
    # up. Combined with ping 10.202.0.1 + an Olympe-test connect, this
    # is what makes "drone is ready" definitive.
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
    # Test 1: telemetry says connected
    local con
    con=$(curl -s --max-time 2 "$FC_API/api/telemetry" 2>/dev/null | \
        python3 -c "import json,sys
try: print(json.load(sys.stdin).get('connected'))
except Exception: pass" 2>/dev/null)
    [ "$con" = "True" ] || return 1

    # Test 2: NOT in the stale-connected wedge state. Even one
    # "Too many ping failures" line in the recent FC log means ARSDK
    # keepalive is gone — telemetry's connected flag will linger as
    # True for a while, but every command (takeoff, RC) silently fails.
    # We catch the wedge before declaring the FC healthy.
    local fc_log
    fc_log=$(ls -t /opt/sdc-tobe/sphinx-control/logs/fc-*/fc.log 2>/dev/null | head -1)
    if [ -n "$fc_log" ] && [ -f "$fc_log" ]; then
        if tail -n 200 "$fc_log" 2>/dev/null | grep -q 'Too many ping failures'; then
            return 1
        fi
    fi
    return 0
}

# Force every sim-related process gone. Used after an API teardown to
# make sure the next start gets a clean slate (sphinx-control's
# DELETE returns before the underlying processes actually exit).
# All these processes run as user 'ubuntu' so pkill needs no sudo.
kill_orphans() {
    pkill -9 -f "sphinx /opt/parrot-sphinx" 2>/dev/null || true
    pkill -9 gzserver 2>/dev/null || true
    pkill -9 -f UnrealApp 2>/dev/null || true
    pkill -9 -f "parrot-ue4-" 2>/dev/null || true
}

# Nuclear option for sphinx-control's phantom DB state: when DELETE
# doesn't clear an env entry, POST returns 409 "already running"
# forever. Restart sphinx-control so its reconcile_on_startup() rebuilds
# DB state from actually-running processes.
restart_sphinx_control() {
    log "  RESTART sphinx-control to clear phantom DB state"
    sudo -n systemctl restart sphinx-control \
        || { log "  ERROR: can't restart sphinx-control (needs sudoers)"; return 1; }
    wait_for "sphinx-control back online" spc_up 30
}

# Heavier nuclear option for when the drone spawn dies with "Exit is
# pending..." right after firmware load. That pattern means firmwared
# itself has stale loop-mount state under /tmp/fd_mount_points/, and
# the only known fix is restarting firmwared. Has to stop sphinx-control
# first or sphinx-control's reconcile thrashes the new state on the way
# in.
restart_firmwared_full() {
    log "  HARD RESET: stop sphinx-control + firmwared restart"
    sudo -n systemctl stop sphinx-control marker-mission c2-controller 2>/dev/null
    kill_orphans
    sleep 3
    sudo -n systemctl restart firmwared \
        || { log "  ERROR: can't restart firmwared (needs sudoers)"; return 1; }
    sleep 3
    sudo -n systemctl start sphinx-control
    wait_for "sphinx-control back online" spc_up 30 || return 1
    sudo -n systemctl start c2-controller marker-mission 2>/dev/null || true
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

# spc_up has to be defined before the HARD block below — restart_firmwared_full
# calls wait_for spc_up internally, and bash function lookup is at call time
# from the current scope. Earlier versions defined spc_up only here (just
# above the normal wait_for), and the HARD branch failed with
# "spc_up: command not found".
spc_up() { curl -sf --max-time 2 "$SPC_API/api/environment" >/dev/null 2>&1; }

# Operator can request a HARD recover via the dashboard. Two triggers,
# whichever exists at start wins (then we consume both):
#   /tmp/sphinx-bootstrap-hard.flag   — created by /api/recover?hard=1
#   SPHINX_BOOTSTRAP_HARD=1           — env override (manual / debug)
#
# Hard mode bypasses the "drone reachable → skip" healthchecks that
# can't see emergency state, and always restarts firmwared up front.
# This is the only known way to clear an Olympe FlyingState=emergency
# without rebooting the host (firmwared owns the netns + loop-mounts;
# until it restarts the simulated drone is "connected but wedged").
HARD_FLAG=/tmp/sphinx-bootstrap-hard.flag
if [ -f "$HARD_FLAG" ] || [ "${SPHINX_BOOTSTRAP_HARD:-0}" = "1" ]; then
    rm -f "$HARD_FLAG"
    log "HARD mode requested — restarting firmwared before normal bootstrap"
    if ! restart_firmwared_full; then
        log "abort: hard reset failed"; exit 1
    fi
fi

if ! wait_for "sphinx-control API on :9090" spc_up 60; then
    log "abort: sphinx-control not answering"; exit 1
fi

# ─── stage 1: env (UE4 + sphinx daemon) ─────────────────────────

if env_healthy; then
    log "stage 1 skip: env healthy (UE4 alive)"
else
    log "stage 1: env unhealthy — tear down and start fresh"
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc"          >/dev/null
    curl -s --max-time 5 -X DELETE "$SPC_API/api/environment" >/dev/null
    sleep 4
    kill_orphans
    sleep 3

    # Try the clean path first: POST a fresh env. POST is synchronous
    # in sphinx-control — it blocks until UE4 has launched, which can
    # take 30-90s on a cold box. Long timeout. If sphinx-control's
    # internal state has a phantom env that DELETE didn't clear, we
    # get 409 quickly; fall back to restarting sphinx-control itself
    # which rebuilds state via reconcile_on_startup().
    start_env() {
        curl -s --max-time 120 -X POST "$SPC_API/api/environment" \
            -H "Content-Type: application/json" \
            -d '{"world_name":"sdc_arena_test"}'
    }
    resp=$(start_env)
    if ! echo "$resp" | grep -q '"env_id"'; then
        log "  initial POST failed: $(echo "$resp" | head -c 200)"
        if echo "$resp" | grep -qi "already running\|singleton\|already exists"; then
            restart_sphinx_control || { log "abort: can't recover sphinx-control"; exit 1; }
            kill_orphans
            sleep 4
            resp=$(start_env)
        fi
    fi
    if ! echo "$resp" | grep -q '"env_id"'; then
        log "abort: env POST failed after fallback: $(echo "$resp" | head -c 200)"; exit 1
    fi

    # UE4 boot is ~30-60s on a warm box, longer on a cold AMI launch.
    if ! wait_for "env healthy (UE4 alive)" env_healthy 180; then
        log "abort: env never became healthy"; exit 1
    fi
fi

# ─── stage 2: drone reachable on 10.202.0.1 ─────────────────────

spawn_and_wait_drone() {
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc" >/dev/null
    curl -s --max-time 5 -X POST "$SPC_API/api/drones/stop-all" >/dev/null 2>&1
    sleep 4
    # POST /api/drones is synchronous on sphinx-control — blocks until
    # the sphinx wrapper has spawned + firmwared has loaded the .ext2
    # image (can take 15-45 s on a cold box).
    curl -sf --max-time 60 -X POST "$SPC_API/api/drones" \
        -H "Content-Type: application/json" \
        -d '{"drone_profile":"anafi-4k"}' >/dev/null \
        || return 1
    wait_for "sphinx RPC :$SPHINX_RPC_PORT bound" rpc_bound 60 || return 1
    wait_for "drone pingable on $DRONE_IP" drone_pingable 60 || return 1
    wait_for "drone olympe-reachable" olympe_reachable 90 || \
        log "warn: drone not olympe-reachable — continuing, FC has its own retry"
    return 0
}

if rpc_bound && drone_pingable && olympe_reachable; then
    log "stage 2 skip: drone reachable on $DRONE_IP"
else
    log "stage 2: drone unreachable — respawn"
    if ! spawn_and_wait_drone; then
        log "stage 2: drone spawn failed — HARD RESET (firmwared has stale state)"
        restart_firmwared_full || { log "abort: hard reset failed"; exit 1; }
        # After hard reset everything is gone. Restart from stage 1
        # so we re-create env first.
        if ! wait_for "env healthy after hard reset" env_healthy 30; then
            # env was killed by sphinx-control stop above — re-POST it
            curl -s --max-time 120 -X POST "$SPC_API/api/environment" \
                -H "Content-Type: application/json" \
                -d '{"world_name":"sdc_arena_test"}' >/dev/null
            wait_for "env healthy after re-POST" env_healthy 180 \
                || { log "abort: env never came back after hard reset"; exit 1; }
        fi
        spawn_and_wait_drone \
            || { log "abort: drone still won't spawn after firmwared reset"; exit 1; }
    fi
fi

# ─── stage 3: FC connected ──────────────────────────────────────
#
# SPHINX_BOOTSTRAP_SKIP_FC=1 disables this stage entirely. Set on the
# sphinx-bootstrap.service unit so boot + Recover only bring up env +
# drone — the operator then starts whichever FC they want manually
# (controller_unified's unified_api_server.py, marker_mission.app, or
# something else). Stage 1 + 2 still run because the sim isn't useful
# without UE4 and a spawned drone.

if [ "${SPHINX_BOOTSTRAP_SKIP_FC:-0}" = "1" ]; then
    log "stage 3 skip: SPHINX_BOOTSTRAP_SKIP_FC=1 (FC must be started manually)"
    log "recovery done — env+drone healthy, FC intentionally not started"
    exit 0
fi

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
