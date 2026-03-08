#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${C2_HOST:-127.0.0.1}"
PORT="${C2_PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"
DRONE_COUNT="${DRONE_COUNT:-4}"

cd "$ROOT_DIR"

echo "[c2] Ensuring C2 server is running on ${BASE_URL} ..."
if ! curl -fsS "${BASE_URL}/api/state" >/dev/null 2>&1; then
  echo "[c2] C2 not reachable; starting uvicorn..."
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
  fi
  nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >/tmp/c2_uvicorn.log 2>&1 &
  sleep 2
fi

curl -fsS "${BASE_URL}/api/state" >/dev/null

echo "[c2] Switching mode to simulator (${DRONE_COUNT} drones) ..."
curl -fsS -X POST "${BASE_URL}/api/system" \
  -H "Content-Type: application/json" \
  -d "{\"mode\":\"simulator\",\"drone_count\":${DRONE_COUNT}}" >/dev/null

# validate simulator UI and per-drone simulated feed endpoints
curl -fsS "${BASE_URL}/simviewer/web3d/" >/dev/null
for i in $(seq 1 "$DRONE_COUNT"); do
  d="team-drone-${i}"
  if ! curl -fsS "${BASE_URL}/api/sim/video/${d}" | grep -q "SIM FEED"; then
    echo "[c2] ERROR: simulator feed not ready for ${d}"
    exit 1
  fi
done

echo "[c2] OK: simulator connected and feeds available for ${DRONE_COUNT} drones"
echo "[c2] Open ${BASE_URL}/ and keep mode=simulator"
