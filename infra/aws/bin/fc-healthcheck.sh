#!/bin/bash
# Runs every 60 s under fc-healthcheck.timer. Detects TWO failure modes
# and restarts the FC through sphinx-control if either persists:
#
#   A) "obvious" disconnect — telemetry.connected=false for ≥3 consecutive
#      checks (~3 min of true disconnection).
#
#   B) "stuck-connected wedge" — telemetry says connected=true, but
#      the FC log shows "Too many ping failures" — ARSDK keepalive lost.
#      That's the smoking-gun of "state cache stays warm so
#      connected=true, but the command channel is actually dead".
#      Earlier versions of this script ALSO tripped on
#      "Unable to stop piloting interface", but that pattern fires
#      during normal land/takeoff cycles when Olympe tries piloting-
#      cleanup twice on a now-landed drone — we got a false positive
#      restart on 2026-05-12 that killed a healthy FC. Drop that
#      signal; require ping-failures explicitly.
#
# Either way, never restart more than once every COOLDOWN_S seconds to
# avoid thrashing. State lives under /run/ (tmpfs) so misses are wiped
# at boot — bootstrap.sh has its own first-boot verify+restart and
# shouldn't be double-restarted by this timer.

set -u

MISSES_STATE=/run/fc-healthcheck.misses
RESTART_STATE=/run/fc-healthcheck.last-restart
THRESHOLD=3                              # consecutive disconnected before A) acts
WEDGE_PATTERN_THRESHOLD=1                # "Too many ping failures" lines required before B) acts
WEDGE_TAIL_LINES=200                     # how many recent log lines to inspect
COOLDOWN_S=300                           # don't restart more than once / 5 min
FC_API=http://localhost:8080
SPC_API=http://localhost:9090

log() { logger -t fc-healthcheck "$*"; echo "$*"; }

# ── helpers ────────────────────────────────────────────────────────

within_cooldown() {
    local last_ts now_ts
    [ -f "$RESTART_STATE" ] || return 1
    last_ts=$(cat "$RESTART_STATE" 2>/dev/null || echo 0)
    now_ts=$(date +%s)
    [ $((now_ts - last_ts)) -lt "$COOLDOWN_S" ]
}

BOOTSTRAP=/opt/sdc-tobe/sphinx-control/sphinx-bootstrap.sh

restart_fc() {
    local reason="$1"
    log "delegating recovery to sphinx-bootstrap.sh — $reason"
    # Mark restart time NOW so even if bootstrap fails we still respect
    # cooldown and don't relaunch every minute.
    date +%s > "$RESTART_STATE"
    : > "$MISSES_STATE"

    # bootstrap.sh is the single source of truth for "get sim healthy":
    #   * stage 1 (env) — only restarts if UE4 missing per API
    #   * stage 2 (drone) — respawns drone if unreachable, with
    #                       restart_firmwared_full() escalation if
    #                       that fails too
    #   * stage 3 (FC) — DELETE+POST with retries; if connected=false
    #                    persists OR the FC log shows "Too many ping
    #                    failures", we land here.
    # On a healthy box it's a ~4 s no-op; on a wedged FC, ~15 s; on a
    # wedged firmwared, ~90 s. All bounded.
    if [ ! -x "$BOOTSTRAP" ]; then
        log "ERROR: $BOOTSTRAP not found / not executable"
        return 1
    fi
    if "$BOOTSTRAP"; then
        log "bootstrap reports healthy"
        return 0
    else
        log "bootstrap exited non-zero — sim probably needs hands"
        return 1
    fi
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

# Only "Too many ping failures" is a definitive wedge signal — it means
# arsdk lost keepalive with the drone, so the command channel is dead
# even if the state cache hasn't noticed yet. The other patterns we
# tried (Unable to stop piloting interface, connection retries failed)
# fire during normal operation and produced false positives.
WEDGE_HITS=$(tail -n "$WEDGE_TAIL_LINES" "$FC_LOG" 2>/dev/null | \
  grep -cE 'Too many ping failures')

if [ "${WEDGE_HITS:-0}" -ge "$WEDGE_PATTERN_THRESHOLD" ]; then
    log "wedge: $WEDGE_HITS bad lines in last $WEDGE_TAIL_LINES of $FC_LOG (connected=true but command channel likely dead)"
    if within_cooldown; then
        log "would restart but still in cooldown — waiting"
    else
        restart_fc "stuck-connected wedge: $WEDGE_HITS bad lines"
    fi
fi
