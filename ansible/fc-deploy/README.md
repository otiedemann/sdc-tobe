# fc-deploy — flight controller systemd unit rollout

Ansible playbook for installing **and switching between** the two
mutually-exclusive flight-controller systemd units on a flightctrl
host:

| Unit                      | What it runs                                              | Default port |
| ------------------------- | --------------------------------------------------------- | -----------: |
| `sdc-fc.service`          | `controller_unified/unified_api_server.py` (HTTP only)    |       `5050` |
| `marker-mission.service`  | `python -m marker_mission.app` (combined Flask app)       |       `8080` |

Both units carry `Conflicts=` for the other in their `[Unit]` section,
so even if both were enabled, systemd would stop one before starting
the other. The playbook makes the choice explicit by `enable`-ing
exactly one and disabling the other.

## Prerequisites

- Target host has `/home/sdc/sdc-tobe` cloned and the `sdc` user can
  pull `origin/main` without a passphrase prompt (BatchMode-safe SSH
  keys).
- Target host has the project's Python venv at `/home/sdc/sdc-tobe/.venv/`
  with Olympe + cv2 installed. The start scripts auto-fall-back to
  system Python if the venv is missing, but Olympe will not work
  without it.
- Ansible 2.10+ on the control machine (no extra collections required).
- `--ask-become-pass` for root steps unless the `sdc` user has
  passwordless sudo.

## Quick reference

```bash
# Switch flightctrl2 to the in-process combined app
ansible-playbook -i inventory.ini deploy.yml \
    -e "target=flightctrl2 mode=marker-mission" --ask-become-pass

# Revert flightctrl2 to the legacy HTTP-only flight controller
ansible-playbook -i inventory.ini deploy.yml \
    -e "target=flightctrl2 mode=fc" --ask-become-pass

# Stage unit files without starting anything (e.g. before maintenance)
ansible-playbook -i inventory.ini deploy.yml \
    -e "target=flightctrl2 mode=none" --ask-become-pass
```

## What the playbook does, step by step

1. **Pull latest** — `git fetch && git reset --hard origin/main` in
   `/home/sdc/sdc-tobe` so the unit files installed in step 2 match
   what's in the repo HEAD.
2. **Install unit files** — copy `controller_unified/systemd/sdc-fc.service`
   and `marker_mission/systemd/marker-mission.service` to
   `/etc/systemd/system/`. `daemon-reload` runs only if either file
   actually changed.
3. **Switch active unit** — disables + stops the unselected unit, then
   enables + (re)starts the selected one. `restarted` (not `started`)
   so a unit-file content change always takes effect.
4. **Health probe** — `wait_for` the listening port, then GET
   `/api/telemetry` over loopback. Fails the playbook if either step
   times out, so a broken deploy doesn't silently appear "green".

## Why two units instead of one

`unified_api_server.py` predates the in-process integration. Some
deployments (CI hosts, dev boxes, AWS sphinx-sim) want the legacy
HTTP-only mode so external tools talk to it the same way they always
did. flightctrl hosts in production want the combined app — same
URLs externally, but marker_mission's hot path (telemetry, RC,
takeoff/land/emergency, video frames) bypasses HTTP entirely via
`drone_core`.

Mutual exclusion via `Conflicts=` keeps the choice clean: the host
either is the "old" FC (port 5050, HTTP) or the "new" one (port 8080,
combined). It is never both. An operator can flip between them with a
single `ansible-playbook` call and ~5 s of downtime.

## Inventory

Edit `inventory.ini` to match your fleet. The defaults assume
`flightctrl1`, `flightctrl2`, `flightctrl3` resolve via DNS / hosts
file and the `sdc` user has SSH access.
