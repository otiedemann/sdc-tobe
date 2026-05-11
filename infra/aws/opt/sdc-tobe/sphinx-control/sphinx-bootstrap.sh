#!/bin/bash
# Auto-bootstrap sphinx sim after boot:
#  1. wait for sphinx-control HTTP API on :8090
#  2. start environment (sdc_arena_test) if none running
#  3. spawn anafi-4k drone if none alive
#  4. wait for the drone to be Olympe-reachable on 10.202.0.1
#  5. start flight controller
#  6. verify FC actually connected; restart if stuck disconnected
#
# Idempotent: re-running while everything is already up is a no-op.
set -u
log() { echo "[bootstrap $(date -Iseconds)] $*"; }

FC_VENV=/opt/sdc-tobe/.venv
DRONE_IP=10.202.0.1

# Re-usable: try Olympe.connect with a short timeout. Exit 0 if reachable.
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

# 1. wait up to 60s for sphinx-control to answer
for i in $(seq 1 30); do
    if curl -sf http://localhost:8090/api/environment >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
if ! curl -sf http://localhost:8090/api/environment >/dev/null 2>&1; then
    log "sphinx-control API never came up on :8090 — abort"; exit 1
fi
log "sphinx-control reachable"

# 2. environment
ENV=$(curl -s http://localhost:8090/api/environment)
if [ "$ENV" = "null" ] || [ -z "$ENV" ]; then
    log "starting environment sdc_arena_test"
    curl -sf -X POST http://localhost:8090/api/environment \
      -H "Content-Type: application/json" \
      -d "{\"world_name\":\"sdc_arena_test\"}" >/dev/null \
      || { log "env start failed"; exit 1; }
    sleep 20
else
    log "environment already running"
fi

# 3. drone subprocess
HAS_LIVE=$(curl -s http://localhost:8090/api/drones | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(any(x.get(\"extras\",{}).get(\"alive\") for x in d))" 2>/dev/null)
if [ "$HAS_LIVE" != "True" ]; then
    log "spawning anafi-4k drone"
    curl -sf -X POST http://localhost:8090/api/drones \
      -H "Content-Type: application/json" \
      -d "{\"drone_profile\":\"anafi-4k\"}" >/dev/null \
      || { log "drone spawn failed"; exit 1; }
    # poll up to 60s for the drone subprocess to report alive
    for i in $(seq 1 30); do
        sleep 2
        HAS_LIVE=$(curl -s http://localhost:8090/api/drones | \
          python3 -c "import json,sys; d=json.load(sys.stdin); print(any(x.get(\"extras\",{}).get(\"alive\") for x in d))" 2>/dev/null)
        [ "$HAS_LIVE" = "True" ] && break
    done
    if [ "$HAS_LIVE" = "True" ]; then
        log "drone subprocess alive after $((i*2))s"
    else
        log "drone never reported alive — continuing anyway"
    fi
else
    log "drone subprocess already alive"
fi

# 4. wait for the drone to be Olympe-reachable. "Subprocess alive" only
# means the gz/firmware process exists; the netns ARSDK endpoint takes
# a further few seconds to start announcing. Starting FC before that is
# the #1 cause of the "connected:False forever" flake.
log "waiting for olympe-reachable on $DRONE_IP (up to 90s)..."
for i in $(seq 1 18); do
    if olympe_reachable; then
        log "drone is olympe-reachable after $((i*5))s"
        break
    fi
    sleep 5
done
if ! olympe_reachable; then
    log "drone NOT olympe-reachable after 90s — will start FC anyway, but expect connect retries"
fi

# 5. flight controller
FC=$(curl -s http://localhost:8090/api/fc)
if [ "$FC" = "null" ] || [ -z "$FC" ]; then
    log "starting flight controller"
    curl -sf -X POST http://localhost:8090/api/fc \
      -H "Content-Type: application/json" -d "{}" >/dev/null \
      || { log "FC start failed"; exit 1; }
    sleep 8
else
    log "FC already running"
fi

# 6. verify FC actually connected to the drone; if stuck, restart it
# once. This catches the case where Olympe's first attempt raced the
# drone arsdk announce window above and got wedged.
log "verifying FC reached connected:true..."
CONNECTED=False
for i in $(seq 1 12); do
    sleep 5
    CONNECTED=$(curl -s http://localhost:8080/api/telemetry 2>/dev/null | \
      python3 -c "import json,sys; print(json.load(sys.stdin).get(\"connected\"))" 2>/dev/null)
    [ "$CONNECTED" = "True" ] && break
done
if [ "$CONNECTED" = "True" ]; then
    log "FC connected after $((i*5))s — bootstrap done"
    exit 0
fi

log "FC stayed disconnected for 60s — restarting once"
curl -s -X DELETE http://localhost:8090/api/fc >/dev/null
sleep 5
# sphinx-control's port-in-use bug after rapid DELETE+POST: give it room
sleep 5
curl -sf -X POST http://localhost:8090/api/fc \
  -H "Content-Type: application/json" -d "{}" >/dev/null \
  || { log "FC restart POST failed — leaving alone"; exit 1; }
sleep 10
CONNECTED=$(curl -s http://localhost:8080/api/telemetry 2>/dev/null | \
  python3 -c "import json,sys; print(json.load(sys.stdin).get(\"connected\"))" 2>/dev/null)
log "after restart: connected=$CONNECTED"
log "bootstrap done"
