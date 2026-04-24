# Flight-controller systemd units

Two units run the fleet flight controllers (`/home/sdc/sdc-tobe/controller_unified/unified_api_server.py`) on Ubuntu 24.04 hosts:

| Unit | Type | Purpose |
|------|------|---------|
| `sdc-fc-update.service` | `oneshot` | On boot: stash local changes, force-reset to `origin/main`, refresh venv deps |
| `sdc-fc.service`        | `simple` + `Restart=always` | Runs `unified_api_server.py`; watchdog-restarts on crash |

`sdc-fc.service` depends on (`After=` + `Wants=`) `sdc-fc-update.service`, so an update always runs first at boot — but a failing update never blocks the flight controller from starting (the update script exits 0 when origin is unreachable).

## Install (per flight-controller box)

```bash
cd /home/sdc/sdc-tobe/controller_unified/systemd
sudo ./install.sh
sudo systemctl start sdc-fc.service    # first-time start without reboot
```

The installer is idempotent — re-run it whenever the unit files change.

## Operate

```bash
systemctl status sdc-fc.service            # running / crashed / restart count
journalctl -u sdc-fc.service -f            # live log
journalctl -u sdc-fc-update.service -b     # boot-time git-sync log
sudo systemctl restart sdc-fc.service      # manual restart
sudo systemctl reset-failed sdc-fc         # after StartLimitBurst tripped
```

## Override environment

Example: pin `flightctrl4` to a different drone IP:

```bash
sudo systemctl edit sdc-fc.service
# ── drop-in opens in $EDITOR ──
[Service]
Environment=DRONE_TYPE=anafi
Environment=DRONE_IP=192.168.42.1
```

## What the update script does

On every boot, `sdc-fc-update.sh`:

1. Probes `origin` with a 15 s timeout — if unreachable, keeps the current revision and the flight controller still starts.
2. Saves a recovery ref at `refs/sdc-fc/pre-pull/<timestamp>` pointing at the old HEAD.
3. `git stash push --include-untracked` if the tree is dirty (stash message `sdc-fc-update auto-stash <iso8601>`).
4. `git fetch --prune origin` + `git reset --hard origin/main`.
5. `pip install -r controller_unified/requirements.txt` into `~/.venv` if that venv exists.

Recover a discarded local commit:

```bash
cd /home/sdc/sdc-tobe
git for-each-ref refs/sdc-fc/pre-pull/    # find the pre-pull HEAD
git stash list                            # or find the auto-stash
```

## Watchdog behavior

- `Restart=always`, `RestartSec=3s` — any exit (clean or crash) restarts the service after 3 s.
- `StartLimitBurst=5` over `StartLimitIntervalSec=120` — if it crash-loops 5× within 2 min, systemd gives up; clear with `systemctl reset-failed sdc-fc`.
- `KillSignal=SIGINT` on stop so the server's SIGINT handler has a chance to land the drone; SIGKILL after `TimeoutStopSec=15`.
