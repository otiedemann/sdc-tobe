#!/bin/bash
# Runs every 60 s under fc-healthcheck.timer. Detects TWO failure modes
# and restarts the FC through sphinx-control if either persists:
#
#   A) "obvious" disconnect — telemetry.connected=false for ≥3 consecutive
#      checks (~3 min of true disconnection).
#
#   B) "stuck-connected wedge" — telemetry says connected=true, but the
#      FC log shows the unmistakable Olympe-wedged pattern (ARSDK ping
#      failures, "Unable to stop piloting interface", or piling-up
#      "connection retries failed"). This is the failure mode we hit
#      after a flaky cold boot: state cache stays warm so connected=true,
#      but the command channel is dead and every takeoff returns
#      "takeoff_failed" silently.
#
# Either way, never restart more than once every COOLDOWN_S seconds to
# avoid thrashing. State lives under /run/ (tmpfs) so misses are wiped
# at boot — bootstrap.sh has its own first-boot verify+restart and
# shouldn't be double-restarted by this timer.

set -u

MISSES_STATE=/run/fc-healthcheck.misses
RESTART_STATE=/run/fc-healthcheck.last-restart
THRESHOLD=3                              # consecutive disconnected before A) acts
WEDGE_PATTERN_THRESHOLD=3                # wedge-line hits in WEDGE_TAIL_LINES before B) acts
WEDGE_TAIL_LINES=80                      # how many recent log lines to inspect
COOLDOWN_S=300                           # don't restart more than once / 5 min
FC_API=http://localhost:8080
SPC_API=http://localhost:8090

log() { logger -t fc-healthcheck "$*"; echo "$*"; }

# ── helpers ────────────────────────────────────────────────────────

within_cooldown() {
    local last_ts now_ts
    [ -f "$RESTART_STATE" ] || return 1
    last_ts=$(cat "$RESTART_STATE" 2>/dev/null || echo 0)
    now_ts=$(date +%s)
    [ $((now_ts - last_ts)) -lt "$COOLDOWN_S" ]
}

restart_fc() {
    local reason="$1"
    log "restarting FC via sphinx-control — $reason"
    curl -s --max-time 5 -X DELETE "$SPC_API/api/fc" >/dev/null
    # Mark restart time NOW so even if POST fails we still respect cooldown
    date +%s > "$RESTART_STATE"
    : > "$MISSES_STATE"

    # sphinx-control has a known bug where right after DELETE its port-
    # availability check sometimes incorrectly returns "port 8080
    # already in use" even though `fuser 8080/tcp` shows nothing. The
    # stale state clears after ~10-30 s. Retry POST several times with
    # increasing backoff. We deliberately do NOT restart sphinx-control
    # as a workaround — its reconcile on startup tears down the live
    # env+drone subprocesses too, which is much worse than a temporarily
    # unrecoverable FC.
    local attempt resp pid
    for attempt in 1 2 3 4 5 6; do
        sleep $((attempt * 6))   # 6,12,18,24,30,36s — total ~2 min before giving up
        resp=$(curl -s --max-time 10 -X POST "$SPC_API/api/fc" \
               -H "Content-Type: application/json" -d "{}")
        if echo "$resp" | grep -q '"pid"'; then
            pid=$(echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("pid"))')
            log "FC restart ok (pid=$pid) on attempt $attempt"
            return 0
        fi
        log "  POST attempt $attempt failed: $(echo "$resp" | head -c 160)"
    done
    log "FC restart FAILED after $attempt attempts — manual intervention or wait for next timer fire (cooldown $COOLDOWN_S s)"
    return 1
}

# ── precondition: a registered FC exists ──────────────────────────

FC=$(curl -s --max-time 2 "$SPC_API/api/fc" 2>/dev/null)
if [ -z "$FC" ] || [ "$FC" = "null" ]; then
    log "no FC registered with sphinx-control — nothing to heal"
    : > "$MISSES_STATE"
    exit 0
fi

# ── read connection state ─────────────────────────────────────────

CON=$(curl -s --max-time 2 "$FC_API/api/telemetry" 2>/dev/null | \
  python3 -c "import json,sys
try: print(json.load(sys.stdin).get('connected'))
except Exception: print('error')" 2>/dev/null)

# ── failure mode A: connected=false ───────────────────────────────

if [ "$CON" != "True" ]; then
    MISSES=$(( $(cat "$MISSES_STATE" 2>/dev/null || echo 0) + 1 ))
    echo "$MISSES" > "$MISSES_STATE"
    log "connected=$CON  (miss $MISSES/$THRESHOLD)"
    if [ "$MISSES" -ge "$THRESHOLD" ]; then
        if within_cooldown; then
            log "would restart but still in cooldown — waiting"
        else
            restart_fc "$MISSES consecutive disconnected misses"
        fi
    fi
    exit 0
fi

# CON is True — clear miss counter
[ -f "$MISSES_STATE" ] && rm -f "$MISSES_STATE"

# ── failure mode B: connected=true but Olympe wedged ──────────────
# Look for the smoking-gun log lines in the recent FC log. We use the
# log file directly (not journald) because the FC is spawned by
# sphinx-control as a subprocess and writes to a per-FC log dir under
# /opt/sdc-tobe/sphinx-control/logs/fc-<id>/fc.log.

FC_LOG=$(ls -t /opt/sdc-tobe/sphinx-control/logs/fc-*/fc.log 2>/dev/null | head -1)
if [ -z "$FC_LOG" ] || [ ! -f "$FC_LOG" ]; then
    exit 0   # no log to inspect; nothing more to do
fi

# Patterns. Order matters only for readability; all are equally fatal.
#   "Too many ping failures"               — arsdk-level keepalive lost
#   "Unable to stop piloting interface"    — Olympe piloting state machine dead
#   "connection retries failed"            — discovery never completed cleanly
WEDGE_HITS=$(tail -n "$WEDGE_TAIL_LINES" "$FC_LOG" 2>/dev/null | \
  grep -cE 'Too many ping failures|Unable to stop piloting interface|_do_connect - .* connection retries failed')

if [ "${WEDGE_HITS:-0}" -ge "$WEDGE_PATTERN_THRESHOLD" ]; then
    log "wedge: $WEDGE_HITS bad lines in last $WEDGE_TAIL_LINES of $FC_LOG (connected=true but command channel likely dead)"
    if within_cooldown; then
        log "would restart but still in cooldown — waiting"
    else
        restart_fc "stuck-connected wedge: $WEDGE_HITS bad lines"
    fi
fi
