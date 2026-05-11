#!/bin/bash
# Auto-bootstrap sphinx sim after boot:
#  1. wait for sphinx-control HTTP API on :8090
#  2. start environment (sdc_arena_test) if none running
#  3. spawn anafi-4k drone if none alive
#  4. start flight controller
#
# Idempotent: re-running while everything is already up is a no-op.
set -u
log() { echo "[bootstrap $(date -Iseconds)] $*"; }

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

# 3. drone
HAS_LIVE=$(curl -s http://localhost:8090/api/drones | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(any(x.get(\"extras\",{}).get(\"alive\") for x in d))" 2>/dev/null)
if [ "$HAS_LIVE" != "True" ]; then
    log "spawning anafi-4k drone"
    curl -sf -X POST http://localhost:8090/api/drones \
      -H "Content-Type: application/json" \
      -d "{\"drone_profile\":\"anafi-4k\"}" >/dev/null \
      || { log "drone spawn failed"; exit 1; }
    # poll up to 60s for the drone to report alive
    for i in $(seq 1 30); do
        sleep 2
        HAS_LIVE=$(curl -s http://localhost:8090/api/drones | \
          python3 -c "import json,sys; d=json.load(sys.stdin); print(any(x.get(\"extras\",{}).get(\"alive\") for x in d))" 2>/dev/null)
        [ "$HAS_LIVE" = "True" ] && break
    done
    [ "$HAS_LIVE" = "True" ] && log "drone alive after $((i*2))s" || log "drone never came alive — continuing anyway"
else
    log "drone already alive"
fi

# 4. flight controller
FC=$(curl -s http://localhost:8090/api/fc)
if [ "$FC" = "null" ] || [ -z "$FC" ]; then
    log "starting flight controller"
    curl -sf -X POST http://localhost:8090/api/fc \
      -H "Content-Type: application/json" -d "{}" >/dev/null \
      || { log "FC start failed"; exit 1; }
    sleep 5
else
    log "FC already running"
fi

log "bootstrap done"
