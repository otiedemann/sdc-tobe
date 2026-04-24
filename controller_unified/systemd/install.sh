#!/usr/bin/env bash
# Idempotent installer for the flight-controller systemd units.
# Run on a fresh flight-controller box, or after pulling updated unit files:
#
#   cd /home/sdc/sdc-tobe/controller_unified/systemd
#   sudo ./install.sh
#
# After install the service starts on every boot, pulls the latest revision
# from origin/main (stashing any local changes), and restarts on crash.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "re-execing under sudo…"
    exec sudo --preserve-env=USER "$0" "$@"
fi

# Make sure the scripts themselves are executable in the installed tree.
chmod +x "$HERE/sdc-fc-update.sh" "$HERE/sdc-fc-start.sh"

install -m 0644 "$HERE/sdc-fc-update.service" /etc/systemd/system/sdc-fc-update.service
install -m 0644 "$HERE/sdc-fc.service"        /etc/systemd/system/sdc-fc.service

systemctl daemon-reload
systemctl enable sdc-fc-update.service sdc-fc.service

echo
echo "installed + enabled. one-shot start now:"
echo "    systemctl start sdc-fc.service"
echo
echo "status / logs:"
echo "    systemctl status sdc-fc.service"
echo "    journalctl -u sdc-fc.service -f"
echo "    journalctl -u sdc-fc-update.service -b"
