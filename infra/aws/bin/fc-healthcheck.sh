#!/bin/bash
# Runs every 60s under fc-healthcheck.timer. If the FC reports
# connected=false for THREE CONSECUTIVE checks (~3 minutes), restart it
# via sphinx-control. The state file under /run is wiped at every boot
# so we never carry stale miss-counts across reboots.
#
# Why not act on a single miss? Olympe can blip during e.g. an apt
# transaction on the box; we don't want to thrash the FC on noise.
# Three misses ≈ 3 minutes of true disconnection — long enough that
# any in-flight session is already broken anyway.

set -u
STATE=/run/fc-healthcheck.misses
THRESHOLD=3                   # consecutive misses before action
FC_API=http://localhost:8080
SPC_API=http://localhost:8090

log() { logger -t fc-healthcheck "$*"; echo "$*"; }

# Sanity: bootstrap must have set up an FC at least once
FC=$(curl -s --max-time 2 "$SPC_API/api/fc" 2>/dev/null)
if [ -z "$FC" ] || [ "$FC" = "null" ]; then
    log "no FC registered with sphinx-control — nothing to heal"
    : > "$STATE"
    exit 0
fi

# Read connected flag
CON=$(curl -s --max-time 2 "$FC_API/api/telemetry" 2>/dev/null | \
  python3 -c "import json,sys
try: print(json.load(sys.stdin).get('connected'))
except Exception: print('error')" 2>/dev/null)

if [ "$CON" = "True" ]; then
    # healthy — clear miss counter
    [ -f "$STATE" ] && rm -f "$STATE"
    exit 0
fi

# Increment miss counter
MISSES=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$MISSES" > "$STATE"
log "FC connected=$CON  (miss $MISSES/$THRESHOLD)"

if [ "$MISSES" -lt "$THRESHOLD" ]; then
    exit 0
fi

log "FC stuck disconnected for $MISSES checks — restarting via sphinx-control"
curl -s --max-time 5 -X DELETE "$SPC_API/api/fc" >/dev/null
sleep 6
# Workaround for sphinx-control's port-in-use bug after rapid DELETE+POST
RESP=$(curl -s --max-time 10 -X POST "$SPC_API/api/fc" \
       -H "Content-Type: application/json" -d "{}")
echo "$RESP" | grep -q '"pid"' \
    && log "FC restart ok: $(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("pid"))')" \
    || log "FC restart failed: $RESP"

# Clear counter so we don't immediately restart again
: > "$STATE"
