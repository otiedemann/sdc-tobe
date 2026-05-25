#!/usr/bin/env bash
#
# run_sim_stack.sh — spin up the whole local SDC26 simulation stack with
# one command, then tear it all down on Ctrl-C.
#
# It launches (all Python, no hardware, no real Olympe/Sphinx):
#   1. marker_mission_sim   — the FAKE flight controllers ("fake Olympe"):
#                             N simulated drones, each exposing the
#                             marker_mission FC HTTP API on its own port,
#                             plus a Three.js 3D arena view.
#   2. marker_mission_c2     — the C2 server (talks to the sim drones).
#   3. marker_mission_c2.strategy — the SDC26 strategy dashboard.
#
# Usage:
#   ./run_sim_stack.sh                 # sim + C2 + ONE strategy (red)
#   ./run_sim_stack.sh --match         # + a SECOND strategy (blue) for a
#                                      #   full red-vs-blue match
#   ./run_sim_stack.sh --no-strategy   # just sim + C2 (drive via curl/UI)
#
# Env overrides:
#   PY=...          python to use (default: ./.venv/bin/python, else python3)
#   SIM_CONFIG=...  sim config (default marker_mission_sim/sim_config.example.json)
#   C2_CONFIG=...   C2 config  (default marker_mission_c2/config.dev.json)
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

SIM_CONFIG="${SIM_CONFIG:-marker_mission_sim/sim_config.example.json}"
C2_CONFIG="${C2_CONFIG:-marker_mission_c2/config.dev.json}"

MATCH=0
NO_STRATEGY=0
for a in "$@"; do
  case "$a" in
    --match)       MATCH=1 ;;
    --no-strategy) NO_STRATEGY=1 ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $a (try --help)"; exit 2 ;;
  esac
done

LOGDIR="${LOGDIR:-/tmp/sim_stack}"
mkdir -p "$LOGDIR"

# Ports the stack uses (sim UI + 4 FC ports from the example config, C2,
# and up to two strategy dashboards). Pre-killed below + on cleanup.
PORTS=(8090 8091 8092 9100 9101 9102 9103 9104)

PIDS=()

free_ports() {
  for p in "${PORTS[@]}"; do
    local pid
    pid="$(lsof -nP -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
  done
}

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[run_sim_stack] stopping…"
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  free_ports
  echo "[run_sim_stack] done."
}
trap cleanup INT TERM EXIT

start() { # start <name> <cmd...>
  local name="$1"; shift
  "$@" >"$LOGDIR/$name.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  printf '  %-14s pid=%-6s log=%s\n' "$name" "$pid" "$LOGDIR/$name.log"
}

seed_settings() { # seed_settings <file> <team> — only if absent
  local f="$1" team="$2"
  [ -f "$f" ] && return 0
  cat >"$f" <<JSON
{"markers": {"our_team": "$team", "active_slots": [1,2,3,4,5,6]}, "drones": {}}
JSON
}

echo "[run_sim_stack] python: $PY"
echo "[run_sim_stack] freeing stale ports: ${PORTS[*]}"
free_ports
sleep 1

echo "[run_sim_stack] launching:"
start sim "$PY" -m marker_mission_sim --config "$SIM_CONFIG"
sleep 2
start c2  "$PY" -m marker_mission_c2 --config "$C2_CONFIG"
sleep 2

if [ "$NO_STRATEGY" = 0 ]; then
  RED_SETTINGS="$LOGDIR/strat_red.json"
  seed_settings "$RED_SETTINGS" red
  start strategy_red "$PY" -m marker_mission_c2.strategy \
        --c2 http://127.0.0.1:8090 --port 8091 --settings "$RED_SETTINGS"
  if [ "$MATCH" = 1 ]; then
    BLUE_SETTINGS="$LOGDIR/strat_blue.json"
    seed_settings "$BLUE_SETTINGS" blue
    start strategy_blue "$PY" -m marker_mission_c2.strategy \
          --c2 http://127.0.0.1:8090 --port 8092 --settings "$BLUE_SETTINGS"
  fi
fi

cat <<EOF

[run_sim_stack] up — open these:
  3D arena view     http://127.0.0.1:9100
  C2 server         http://127.0.0.1:8090
EOF
[ "$NO_STRATEGY" = 0 ] && echo "  Strategy (red)    http://127.0.0.1:8091"
[ "$MATCH" = 1 ]       && echo "  Strategy (blue)   http://127.0.0.1:8092"
cat <<EOF

In each strategy dashboard: drones auto-appear (red1/red2/blue1/blue2);
set each drone's team + role, Arm, then drive MANUAL or AUTO.
Per-drone video shows in the C2 overview (Show video) and the feeds are
also at http://127.0.0.1:910X/video.mjpg.

Logs: $LOGDIR/   ·   Ctrl-C to stop everything.
EOF

# Wait until interrupted; if any child dies, keep the rest up (operator decides).
wait
