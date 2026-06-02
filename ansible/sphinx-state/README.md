# sphinx-state — backup & restore the simulator host

Captures the runtime state of `sphinx.otconsulting.de` (the SDC26 sim
host) into a tarball, then re-applies it on a different machine that
already has `parrot-sphinx` + the `parrot-ue4-*` world packages
installed.

## What's captured

| Path on source | Why |
|---|---|
| `sdc-tobe/sphinx-control/config.yaml` | World registry + AMS args + drone spawn pose |
| `sdc-tobe/sphinx-control/sphinx_control.db` | Persisted env / drone state |
| `sdc-tobe/sphinx-control/logs/` | Per-env UE4 + sphinx logs |
| `sdc-tobe/controller_unified/arena_config.json` | 16-marker arena layout |
| `sdc-tobe/controller_unified/position_config.json` | Positioning runtime config |
| `sdc-tobe/controller/position_presets.json` | All saved position presets |
| `sdc-tobe/controller/drones_config.json` | Drone fleet definitions |
| `sdc-tobe/tools/sphinx-arena/out/` | **Built** arena assets (FBXs, YAML) — saves ~10 min Blender rebuild |
| `sdc-tobe/aruco-position/` | Per-camera `.npz` calibration files |
| `~/.parrot-sphinx/unreal_sphinx/settings.json` | UE4 HMI panel state, sky preset |
| `~/.parrot-sphinx/telemetrylogs/` | Newest 10 flight telemetry logs |
| Git revision + uncommitted diff | Replayed onto target's checkout |
| `dpkg -l 'parrot-*'` | Parity check for restore target |

What's **not** captured: the simulator itself (`parrot-sphinx`,
`parrot-ue4-empty`, `parrot-ue4-sphx-tests`, …). Those are system
packages — install them on the target via apt before restoring.

## Prerequisites on control machine

```bash
pip install ansible
# or:  brew install ansible
```

## Backup

Captures `sphinx.otconsulting.de` to `./backups/`:

```bash
cd ansible/sphinx-state
ansible-playbook -i inventory.ini backup.yml --ask-pass
```

If you must script it without the interactive prompt, pass the SSH password
via the `SSHPASS` environment variable — never hard-code it in a command or in
this repo:
```bash
export SSHPASS='<sdc-fleet-ssh-password>'   # set in your shell; do not commit
sshpass -e ansible-playbook -i inventory.ini backup.yml --ask-become-pass
```

Output:
```
backups/
  sphinx-state-20260505-2200.tar.gz
  sphinx-state-20260505-2200.manifest.txt
  sphinx-state-20260505-2200.apt-packages.txt
```

The manifest file shows the git rev, telemetry log list, and parrot
package versions captured.

## Restore

1. **Install parrot-sphinx + worlds** on the target host first (the
   restore playbook fails fast if they're missing).

2. **Add the target host** to `inventory.ini` under `[target]`.

3. **Run restore**, pointing at the archive:

```bash
ansible-playbook -i inventory.ini restore.yml --ask-pass \
    -e "target_host=newsim.example.com archive=sphinx-state-20260505-2200.tar.gz"
```

Restore performs:

- Sanity check: `parrot-sphinx` installed? `parrot-ue4-*` worlds present?
- Clones/pulls `sdc-tobe` if it's not already there
- Snapshots existing state on target → `/tmp/sphinx-pre-restore-<ts>.tar.gz`
  (in case you want to revert)
- Extracts the backup archive at `/` (paths are absolute)
- Re-applies any uncommitted `.patch` from the source repo
- Restarts `sphinx-control` on port 8090 (skip via `-e restart_service=false`)
- Probes `http://localhost:8090/api/system` to verify it came up

## Common overrides

```bash
# Different repo path on source
-e "remote_repo=/srv/sdc-tobe"

# Skip sphinx-control restart on restore (manual control)
-e "restart_service=false"

# Skip cloning the repo (already exists on target, just merge state)
-e "skip_repo_clone=true"

# Different expected branch
-e "expected_branch=develop"
```

## Files

```
ansible/sphinx-state/
├── README.md           ← this file
├── inventory.ini       ← source / target hosts
├── vars.yml            ← state_paths, telemetry_keep, etc.
├── backup.yml          ← capture playbook
├── restore.yml         ← apply playbook
└── backups/            ← *.tar.gz output (gitignored)
```
