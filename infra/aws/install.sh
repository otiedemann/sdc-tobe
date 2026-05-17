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
# NOTE: `cp -a` preserves the *caller's* uid/gid on the source tree,
# which (when the repo is cloned as ubuntu) silently chowns /etc and
# every subdir we touch to ubuntu:ubuntu. sudo then refuses to run
# because /etc/sudoers.d is "owned by uid 1000, should be 0".
# Use --preserve=mode (not =all) and force ownership back to root.
sudo cp -r --preserve=mode etc/. /etc/
# Re-assert root ownership on every dir/file we just placed under /etc.
# Walk the source tree so we don't accidentally chown unrelated /etc
# entries that happen to share a name.
while IFS= read -r -d '' rel; do
    sudo chown root:root "/etc/$rel"
done < <(cd etc && find . -mindepth 1 -print0)

# sudoers drop-ins MUST be 0440 (root:root) or sudo silently ignores
# them. Force-fix mode here regardless of what cp picked up.
if [ -d /etc/sudoers.d ]; then
    sudo chmod 0440 /etc/sudoers.d/sphinx-bootstrap 2>/dev/null || true
    sudo chown root:root /etc/sudoers.d/sphinx-bootstrap 2>/dev/null || true
    # validate — visudo -cf returns non-zero on syntax error
    sudo visudo -cf /etc/sudoers.d/sphinx-bootstrap >/dev/null \
        && echo "  sudoers drop-in syntax ok" \
        || echo "  WARNING: sudoers drop-in has a syntax error!"
fi

echo "[install] running ldconfig for any new /etc/ld.so.conf.d entries..."
# Picks up etc/ld.so.conf.d/parrot-sphinx.conf so pysphinx can find
# libtelemetry.so & friends after a fresh image. Without this, the
# FC's sim-video reader fails to import pysphinx and the video feed
# silently stays dark.
sudo ldconfig

echo "[install] copying /opt files..."
# /opt/sdc-tobe IS the git checkout (cloned as ubuntu) — keep it that
# way so `git pull` keeps working. cp -a preserves ubuntu ownership,
# which matches dst here. (Only /etc above is sensitive.)
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

echo "[install] ensuring marker_mission_c2 venv exists at /opt/sdc-tobe/.venv-c2..."
# marker-mission-c2.service (the fleet dashboard) and the optional
# marker_mission_c2.strategy.app (operator UI on :8091) share this venv.
# Deps are flask + httpx — separate from .venv so a future C2 dep bump
# doesn't drag the FC's Olympe runtime into pip-install dependency hell.
if [ ! -x /opt/sdc-tobe/.venv-c2/bin/python3 ]; then
    sudo -u ubuntu python3 -m venv /opt/sdc-tobe/.venv-c2
    sudo -u ubuntu /opt/sdc-tobe/.venv-c2/bin/pip install --upgrade pip wheel --quiet
fi
if [ -f /opt/sdc-tobe/marker_mission_c2/requirements.txt ]; then
    sudo -u ubuntu /opt/sdc-tobe/.venv-c2/bin/pip install \
        -r /opt/sdc-tobe/marker_mission_c2/requirements.txt --quiet
fi

echo "[install] ensuring marker_mission_c2/config.json exists..."
if [ ! -f /opt/sdc-tobe/marker_mission_c2/config.json ]; then
    sudo -u ubuntu cp /opt/sdc-tobe/marker_mission_c2/config.example.json \
                      /opt/sdc-tobe/marker_mission_c2/config.json
    echo "  copied from config.example.json — edit fcs[] for this host"
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
  drone-detector-ui.service \
  marker-mission-c2.service \
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
