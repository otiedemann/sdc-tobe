# Flight-controller systemd unit

Single unit `sdc-fc.service` runs the fleet flight controllers (`/home/sdc/sdc-tobe/controller_unified/unified_api_server.py`) on Ubuntu 24.04.

- **`ExecStartPre=`** runs `sdc-fc-update.sh` on every start: stash local changes, `git reset --hard origin/main`, refresh venv deps. This means the code is refreshed on boot **and** on every `systemctl restart` / watchdog restart — no need to manually pull on each box.
- **`ExecStart=`** runs `sdc-fc-start.sh` which launches `unified_api_server.py` via `.venv/bin/python3` when present, else system `/usr/bin/python3`.
- **`Restart=always`** with 3 s backoff acts as the crash watchdog. Rate-limited to 5 restarts per 300 s; trip clears with `systemctl reset-failed sdc-fc`.

If origin is unreachable, the update script exits 0 and the server still starts on whatever revision is on disk.

## Install (per flight-controller box)

```bash
cd /home/sdc/sdc-tobe/controller_unified/systemd
sudo ./install.sh
sudo systemctl start sdc-fc.service    # first-time start without reboot
```

The installer is idempotent — re-run after editing unit files. It also cleans up the legacy two-unit layout (`sdc-fc-update.service`) if an older install ever put one in place.

`install.sh` drops a scoped [`/etc/sudoers.d/sdc-fc`](sudoers-sdc-fc) so the `sdc` user can run the FC-management `systemctl` commands and re-run `install.sh` without a password. Everything else still prompts. Swap to a blanket `NOPASSWD: ALL` by editing the drop-in (there's a commented alternative in the same file) and re-running the installer — but be aware it means a compromised FC server can escalate to root.

## Operate

```bash
systemctl status sdc-fc.service            # running / crashed / restart count
journalctl -u sdc-fc.service -f            # live log (includes ExecStartPre output)
journalctl -u sdc-fc.service -b            # everything since last boot
sudo systemctl restart sdc-fc.service      # force re-pull + restart
sudo systemctl reset-failed sdc-fc         # after StartLimitBurst tripped
```

## Override environment

Example: pin `flightctrl4` to a specific drone IP:

```bash
sudo systemctl edit sdc-fc.service
# ── drop-in opens in $EDITOR ──
[Service]
Environment=DRONE_TYPE=anafi
Environment=DRONE_IP=192.168.42.1
```

Override the branch for a single box (e.g. staging a feature branch on one flight controller):

```bash
sudo systemctl edit sdc-fc.service
[Service]
Environment=SDC_FC_BRANCH=feature/xyz
```

## What the update script does

On every (re)start, `sdc-fc-update.sh`:

1. Probes `origin` with a 15 s timeout — if unreachable, keeps the current revision and the flight controller still starts.
2. Saves a recovery ref at `refs/sdc-fc/pre-pull/<timestamp>` pointing at the old HEAD.
3. `git stash push --include-untracked` if the tree is dirty (stash message `sdc-fc-update auto-stash <iso8601>`).
4. `git fetch --prune origin` + `git reset --hard origin/$SDC_FC_BRANCH`.
5. `pip install -r controller_unified/requirements.txt` into `~/.venv` if that venv exists.

Recover a discarded local commit:

```bash
cd /home/sdc/sdc-tobe
git for-each-ref refs/sdc-fc/pre-pull/    # find the pre-pull HEAD
git stash list                            # or find the auto-stash
```

## Watchdog behavior

- `Restart=always`, `RestartSec=3s` — any exit (clean or crash) restarts the service after 3 s. Because `ExecStartPre=` re-runs on each restart, the process always comes back on the latest `origin/main`.
- `StartLimitBurst=5` over `StartLimitIntervalSec=300` — if it crash-loops 5× within 5 min, systemd gives up; clear with `systemctl reset-failed sdc-fc`.
- `KillSignal=SIGINT` on stop so the server's SIGINT handler has a chance to land the drone; SIGKILL after `TimeoutStopSec=15`.
- `TimeoutStartSec=120s` caps the entire ExecStartPre + ExecStart window.
