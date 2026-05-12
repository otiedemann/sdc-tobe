#!/bin/bash
# Apply this directory's host config to the running AWS instance.
#
# Idempotent: copies files into /etc and /opt, daemon-reloads, enables
# units. Safe to re-run after editing anything in infra/aws/.
#
# Run from anywhere as a sudo-capable user on the AWS box, e.g.:
#   cd /opt/sdc-tobe/infra/aws && sudo ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "[install] copying /etc files..."
sudo cp -a etc/. /etc/

echo "[install] copying /opt files..."
sudo cp -a opt/. /opt/
sudo chmod +x /opt/sdc-tobe/sphinx-control/sphinx-bootstrap.sh

echo "[install] installing repo-local bin scripts under /opt/sdc-tobe/infra/aws/bin..."
# Skip if the source and destination are already the same file (happens
# when this script runs from /opt/sdc-tobe/infra/aws on a deployed box).
SRC=$(realpath bin/fc-healthcheck.sh)
DST=/opt/sdc-tobe/infra/aws/bin/fc-healthcheck.sh
if [ "$SRC" != "$DST" ]; then
    sudo install -d -m 755 /opt/sdc-tobe/infra/aws/bin
    sudo install -m 755 bin/fc-healthcheck.sh "$DST"
fi

echo "[install] ensuring marker_mission venv exists at /opt/sdc-tobe/.venv-marker..."
# marker-mission.service expects this venv. It's intentionally separate
# from the FC's /opt/sdc-tobe/.venv so matplotlib + the marker_mission
# deps don't have to coexist with the Olympe SDK.
if [ ! -x /opt/sdc-tobe/.venv-marker/bin/python3 ]; then
    sudo -u ubuntu python3 -m venv /opt/sdc-tobe/.venv-marker
    sudo -u ubuntu /opt/sdc-tobe/.venv-marker/bin/pip install --upgrade pip wheel --quiet
fi
if [ -f /opt/sdc-tobe/marker_mission/requirements.txt ]; then
    sudo -u ubuntu /opt/sdc-tobe/.venv-marker/bin/pip install \
        -r /opt/sdc-tobe/marker_mission/requirements.txt --quiet
fi

echo "[install] reloading systemd + cron..."
sudo systemctl daemon-reload
sudo systemctl reload cron 2>/dev/null || sudo systemctl restart cron

echo "[install] enabling units..."
# Order matters less than that they're all enabled — systemd resolves
# After=/Wants= ordering at activation time.
for u in \
  xvfb.service \
  sphinx-bootstrap.service \
  marker-mission.service \
  c2-controller.service \
  auto-shutdown.timer \
  fc-healthcheck.timer
do
  sudo systemctl enable "$u" 2>&1 | grep -v "already enabled" || true
done

echo "[install] tz check (target: Europe/Berlin so the 21:00 cron is local)..."
TZ_NOW=$(timedatectl | awk '/Time zone/ {print $3}')
if [ "$TZ_NOW" != "Europe/Berlin" ]; then
    echo "  current TZ is $TZ_NOW — switching to Europe/Berlin"
    sudo timedatectl set-timezone Europe/Berlin
fi

echo
echo "[install] done. Apply takes effect now for new triggers."
echo "  - 21:00 nightly cron is now armed in $(timedatectl | awk '/Time zone/ {print $3}')"
echo "  - auto-shutdown.timer:"
systemctl list-timers auto-shutdown.timer --no-pager | tail -2
echo
echo "  to actually restart the live services now, e.g. after editing them:"
echo "    sudo systemctl restart sphinx-control marker-mission c2-controller"
