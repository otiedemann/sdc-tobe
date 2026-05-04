#!/usr/bin/env bash
# One-shot installer for Sphinx Control on the Linux Sphinx host.
#
# Idempotent: safe to re-run after pulling new code. Does NOT install
# the systemd unit by default — pass --systemd to wire that up too.
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"
INSTALL_SYSTEMD=0
ENABLE_SYSTEMD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --systemd)        INSTALL_SYSTEMD=1; shift ;;
    --enable)         INSTALL_SYSTEMD=1; ENABLE_SYSTEMD=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: ./install.sh [--systemd] [--enable]

  --systemd  copy systemd unit to /etc/systemd/system/ (needs sudo)
  --enable   --systemd + systemctl enable --now sphinx-control
EOF
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "==> Creating virtualenv ($VENV)…"
$PYTHON -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f config.yaml ]]; then
  echo "==> Seeding config.yaml from config.example.yaml"
  cp config.example.yaml config.yaml
fi

echo "==> Compile-check"
python -m compileall -q .

if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
  echo "==> Installing systemd unit (sudo required)"
  sudo install -m 0644 systemd/sphinx-control.service /etc/systemd/system/sphinx-control.service
  sudo systemctl daemon-reload
  if [[ "$ENABLE_SYSTEMD" == "1" ]]; then
    sudo systemctl enable --now sphinx-control
    sleep 1
    sudo systemctl status sphinx-control --no-pager | head -20
  else
    echo "    Now: systemctl enable --now sphinx-control"
  fi
fi

echo
echo "Sphinx Control installed."
echo "Run manually:"
echo "  source $VENV/bin/activate"
echo "  uvicorn server:app --host 0.0.0.0 --port 8090"
echo
echo "Then open http://<host>:8090/"
