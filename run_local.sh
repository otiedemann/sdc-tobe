#!/usr/bin/env bash
#
# run_local.sh — bring up the ENTIRE SDC26 stack locally (e.g. on a Mac),
# with one command, and tear it all down on Ctrl-C.
#
# Architecture note
# -----------------
# The real flight controller (`marker_mission`) is Olympe-based and only
# runs on Linux against a real/Sphinx drone, so it CANNOT run on macOS.
# Locally, the simulator stands in for it: `marker_mission_sim` serves the
# exact same marker_mission FC HTTP API for N simulated drones (plus a 3D
# arena view). So "the whole infrastructure" locally is:
#
#   1. marker_mission_sim        -> the (simulated) marker_mission FCs:
#                                   one FC API per drone + a 3D arena view.
#   2. marker_mission_c2         -> the C2 server (polls/commands the FCs).
#   3. marker_mission_c2.strategy-> the SDC26 strategy dashboard.
#
# Usage:
#   ./run_local.sh                 # sim + C2 + BOTH strategies (red+blue), 4 drones
#   ./run_local.sh --drones 6      # spawn 6 drones (3 red + 3 blue), available in C2
#   ./run_local.sh --red 3 --blue 2  # explicit per-team counts
#   ./run_local.sh --no-blue       # only spawn the RED strategy
#   ./run_local.sh --match         # (back-compat alias: blue is now on by default)
#   ./run_local.sh --no-strategy   # just sim + C2 (drive via the 3D UI / curl)
#   ./run_local.sh --open          # also open the control hub in a browser
#
# --drones / --red / --blue regenerate matching sim + C2 configs so the
# chosen number of drones flows through to the sim, C2, strategy and hub.
# Without them the committed example config (4 drones) is used.
#
# Env overrides:
#   PY=...          python to use (default ./.venv/bin/python, else python3)
#   SIM_CONFIG=...  default marker_mission_sim/sim_config.example.json
#   C2_CONFIG=...   default marker_mission_c2/config.dev.json
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

SIM_CONFIG="${SIM_CONFIG:-marker_mission_sim/sim_config.example.json}"
C2_CONFIG="${C2_CONFIG:-marker_mission_c2/config.dev.json}"

MATCH=1; NO_STRATEGY=0; OPEN=0   # blue strategy on by default (team-vs-team play)
DRONES=""; RED=""; BLUE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --match)        MATCH=1 ;;                # back-compat: blue is on by default
    --no-blue)      MATCH=0 ;;                # opt out of the blue strategy
    --no-strategy)  NO_STRATEGY=1 ;;
    --open)         OPEN=1 ;;
    --drones)       shift; DRONES="${1:-}" ;;
    --drones=*)     DRONES="${1#*=}" ;;
    --red)          shift; RED="${1:-}" ;;
    --red=*)        RED="${1#*=}" ;;
    --blue)         shift; BLUE="${1:-}" ;;
    --blue=*)       BLUE="${1#*=}" ;;
    -h|--help)      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

# Resolve drone counts: explicit --red/--blue win; else split --drones
# evenly (red gets the extra one); else 0/0 => use the example config as-is.
GEN=0
if [ -n "$RED" ] || [ -n "$BLUE" ]; then
  RED="${RED:-0}"; BLUE="${BLUE:-0}"; GEN=1
elif [ -n "$DRONES" ]; then
  RED=$(( (DRONES + 1) / 2 )); BLUE=$(( DRONES / 2 )); GEN=1
fi

LOGDIR="${LOGDIR:-/tmp/sdc_local}"
mkdir -p "$LOGDIR"

# If a drone count was requested, regenerate matching sim + C2 configs so
# the chosen number of drones is available everywhere (sim/C2/strategy/hub).
if [ "$GEN" = 1 ]; then
  SIM_CONFIG="$LOGDIR/sim_config.json"
  C2_CONFIG="$LOGDIR/c2_config.json"
  echo "[run_local] generating configs for ${RED} red + ${BLUE} blue drone(s)"
  "$PY" -m marker_mission_sim.gen_config --red "$RED" --blue "$BLUE" \
        --out-sim "$SIM_CONFIG" --out-c2 "$C2_CONFIG" \
    || { echo "  ! config generation failed"; exit 1; }
fi

# Ports: control ports + a generous FC port range (covers up to 20 drones).
PORTS=(8090 8091 8092 9100)
for _p in $(seq 9101 9120); do PORTS+=("$_p"); done
PIDS=()

free_ports() {
  for p in "${PORTS[@]}"; do
    local pid; pid="$(lsof -nP -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
  done
}

cleanup() {
  trap - INT TERM EXIT
  echo; echo "[run_local] stopping…"
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1; free_ports
  echo "[run_local] done."
}
trap cleanup INT TERM EXIT

start() { # start <name> <cmd...>
  local name="$1"; shift
  "$@" >"$LOGDIR/$name.log" 2>&1 &
  local pid=$!; PIDS+=("$pid")
  printf '  %-14s pid=%-6s log=%s\n' "$name" "$pid" "$LOGDIR/$name.log"
}

wait_http() { # wait_http <url> <label> [timeout_s]
  local url="$1" label="$2" timeout="${3:-20}" i=0
  while [ "$i" -lt "$timeout" ]; do
    if curl -s -o /dev/null --max-time 1 "$url"; then return 0; fi
    sleep 1; i=$((i+1))
  done
  echo "  ! WARN: $label not responding at $url after ${timeout}s (see $LOGDIR/)"
  return 1
}

seed_settings() { # seed_settings <file> <team> — only if absent
  local f="$1" team="$2"
  [ -f "$f" ] && return 0
  printf '{"markers": {"our_team": "%s", "active_slots": [1,2,3,4,5,6]}, "drones": {}}\n' "$team" >"$f"
}

# ---- pre-flight ----------------------------------------------------------
echo "[run_local] python: $PY"
"$PY" -c "import flask, httpx, yaml, PIL" 2>/dev/null || {
  echo "  ! missing deps. Install with:"
  echo "      $PY -m pip install -r marker_mission_c2/requirements.txt -r marker_mission_sim/requirements.txt 2>/dev/null || \\"
  echo "      $PY -m pip install flask httpx pyyaml pillow"
  exit 1
}
if ! "$PY" -c "import olympe" 2>/dev/null; then
  echo "  i Olympe not present (expected on macOS) — using the SIMULATOR as the"
  echo "    marker_mission flight controllers. No real/Sphinx drone needed."
fi

echo "[run_local] freeing stale ports: ${PORTS[*]}"
free_ports; sleep 1

# ---- launch --------------------------------------------------------------
echo "[run_local] launching:"
start sim "$PY" -m marker_mission_sim --config "$SIM_CONFIG"
wait_http "http://127.0.0.1:9100/api/world" "sim 3D view" 25
wait_http "http://127.0.0.1:9101/api/state" "sim FC red1" 10

start c2  "$PY" -m marker_mission_c2 --config "$C2_CONFIG"
wait_http "http://127.0.0.1:8090/api/c2/overview" "C2 server" 20

if [ "$NO_STRATEGY" = 0 ]; then
  RED_SETTINGS="$LOGDIR/strat_red.json"; seed_settings "$RED_SETTINGS" red
  start strategy_red "$PY" -m marker_mission_c2.strategy \
        --c2 http://127.0.0.1:8090 --port 8091 --settings "$RED_SETTINGS"
  wait_http "http://127.0.0.1:8091/api/state" "strategy (red)" 20
  if [ "$MATCH" = 1 ]; then
    BLUE_SETTINGS="$LOGDIR/strat_blue.json"; seed_settings "$BLUE_SETTINGS" blue
    start strategy_blue "$PY" -m marker_mission_c2.strategy \
          --c2 http://127.0.0.1:8090 --port 8092 --settings "$BLUE_SETTINGS"
    wait_http "http://127.0.0.1:8092/api/state" "strategy (blue)" 20
  fi
fi

# ---- summary -------------------------------------------------------------
N_FC="$(curl -s --max-time 2 http://127.0.0.1:8090/api/c2/overview \
        | "$PY" -c 'import json,sys;print(len(json.load(sys.stdin)))' 2>/dev/null || echo '?')"
if [[ "$N_FC" =~ ^[0-9]+$ ]] && [ "$N_FC" -gt 0 ]; then
  VIDEO_RANGE="http://127.0.0.1:9101..$((9100 + N_FC))/video.mjpg"
else
  VIDEO_RANGE="http://127.0.0.1:9101+/video.mjpg"
fi
URL_3D="http://127.0.0.1:9100"
URL_C2="http://127.0.0.1:8090"
URL_RED="http://127.0.0.1:8091"
URL_BLUE="http://127.0.0.1:8092"

URL_HUB="http://127.0.0.1:9100/hub"

cat <<EOF

[run_local] UP — the whole local stack is running ($N_FC simulated FC drones).

  ➜ CONTROL HUB (all links): $URL_HUB

  3D arena view (sim) : $URL_3D
  C2 server           : $URL_C2
EOF
[ "$NO_STRATEGY" = 0 ] && echo "  Strategy (red)      : $URL_RED"
[ "$MATCH" = 1 ]       && echo "  Strategy (blue)     : $URL_BLUE"
cat <<EOF
  Per-drone video     : $VIDEO_RANGE  (also in C2 overview / the hub)

The hub page links to everything + shows live status and per-drone video.
In each strategy dashboard the sim drones auto-appear: set each drone's
team + role, Arm, then drive MANUAL or AUTO.

Logs: $LOGDIR/   ·   Ctrl-C to stop everything.
EOF

if [ "$OPEN" = 1 ] && command -v open >/dev/null 2>&1; then
  open "$URL_HUB" 2>/dev/null || true   # the hub links to all the rest
fi

wait
