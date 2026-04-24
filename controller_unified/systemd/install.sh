#!/usr/bin/env bash
# Idempotent installer for the flight-controller systemd unit.
# Run on a fresh flight-controller box, or after pulling updated unit
# files:
#
#   cd /home/sdc/sdc-tobe/controller_unified/systemd
#   sudo ./install.sh
#
# After install the service starts on every boot, pulls the latest
# revision from origin/main on every (re)start — stashing any local
# changes — and restarts on crash.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "re-execing under sudo…"
    exec sudo --preserve-env=USER "$0" "$@"
fi

# Make sure the scripts themselves are executable in the installed tree.
chmod +x "$HERE/sdc-fc-update.sh" "$HERE/sdc-fc-start.sh"

# Clean up the previous two-unit layout if it was installed earlier.
if [[ -f /etc/systemd/system/sdc-fc-update.service ]]; then
    systemctl disable sdc-fc-update.service 2>/dev/null || true
    rm -f /etc/systemd/system/sdc-fc-update.service
    echo "removed legacy sdc-fc-update.service"
fi

install -m 0644 "$HERE/sdc-fc.service" /etc/systemd/system/sdc-fc.service

systemctl daemon-reload
systemctl enable sdc-fc.service

# Passwordless sudo for the sdc user, scoped to FC-management commands.
# Validate with visudo -c BEFORE installing — a bad sudoers drop-in can
# lock the box out of sudo entirely.
SUDOERS_SRC="$HERE/sudoers-sdc-fc"
SUDOERS_DST=/etc/sudoers.d/sdc-fc
if [[ -f "$SUDOERS_SRC" ]]; then
    if visudo -c -f "$SUDOERS_SRC" >/dev/null; then
        install -m 0440 -o root -g root "$SUDOERS_SRC" "$SUDOERS_DST"
        echo "installed $SUDOERS_DST (scoped NOPASSWD for sdc user)"
    else
        echo "WARN: $SUDOERS_SRC failed visudo validation — skipping sudoers install"
    fi
fi

echo
echo "installed + enabled. start now without rebooting:"
echo "    systemctl start sdc-fc.service"
echo
echo "status / logs:"
echo "    systemctl status sdc-fc.service"
echo "    journalctl -u sdc-fc.service -f"
