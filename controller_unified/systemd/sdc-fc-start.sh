#!/usr/bin/env bash
# Entry point for sdc-fc.service. Picks the venv python if present,
# otherwise falls back to system python. Kept as a separate script so
# the logic (and any future env bootstrapping) lives in one place.

set -u
set -o pipefail

REPO="${SDC_FC_REPO:-/home/sdc/sdc-tobe}"
cd "$REPO/controller_unified"

if [[ -x "$REPO/.venv/bin/python3" ]]; then
    PY="$REPO/.venv/bin/python3"
else
    PY=/usr/bin/python3
fi

echo "[sdc-fc] starting unified_api_server.py with $PY"
# -u for unbuffered stdout/stderr so journalctl -f shows output live.
exec "$PY" -u unified_api_server.py
