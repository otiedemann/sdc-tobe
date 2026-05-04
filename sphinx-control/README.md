# Sphinx Control

A FastAPI web service that manages a fleet of Parrot Sphinx simulator
instances on one Linux host. Spawn, stop, restart, and inspect up to
10 simulated drones from a single dashboard, accessible from anywhere
via Tailscale.

This service exists because **Sphinx 2.x runs one drone per process**
([Parrot's official position](https://forum.developer.parrot.com/t/can-you-explain-why-sphinx-2-does-not-support-multiple-drones-give-some-hints-to-overcome-this-limitation/21672)).
Running a swarm means running multiple Sphinx instances side-by-side,
each with its own port (or its own network namespace) so external
clients can address them individually.

## What you get

- **Dashboard** at `http://<host>:8090/`: list active drones, spawn
  new ones, restart/stop/delete individual or all drones, view who's
  connected to each.
- **REST API** at `/api/*` for scripting (see "API reference" below).
- **Per-drone "virtual IPs"** in two flavors:
  - `ports` mode: every drone shares the host IP with a unique port
    (`9081, 9082, ...`). No root needed. Tailscale exposes the host;
    clients use `host:port` per drone.
  - `netns` mode: every drone gets its own IP in a private subnet
    (`10.202.0.11, 10.202.0.12, ...`) via Linux network namespaces.
    Needs `sudo`. Tailscale advertises the subnet so external nodes
    route directly to drone IPs.
- **Tailscale awareness**: the dashboard surfaces the host's tailnet
  IP and any advertised routes (read-only — never modifies your
  Tailscale config).
- **Dry-run mode** for development on machines without Sphinx
  installed (auto-detected when `/usr/bin/sphinx` is missing).

## Requirements

| Component | Where | Why |
|---|---|---|
| Ubuntu 22.04 | Sphinx host | Sphinx 2.x is Linux-only |
| `parrot-sphinx`, `parrot-ue-sdk` | Sphinx host | the simulator |
| Python 3.10+ | anywhere | the management service |
| `tailscale` | Sphinx host (optional) | external access |
| GPU | Sphinx host | each Sphinx instance renders one UE4 view; multi-drone is GPU-bound |

## Install (Linux Sphinx host)

```bash
# Once: install Sphinx + Tailscale
sudo apt install parrot-sphinx parrot-ue-sdk
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up   # follow the auth flow

# Pull the repo
git clone https://github.com/otiedemann/sdc-tobe.git
cd sdc-tobe/sphinx-control

# Install (creates .venv, installs deps, compile-checks, optional systemd)
./install.sh                    # local-only run
./install.sh --systemd          # also drop the unit file
./install.sh --enable           # ...and enable+start it
```

By default the service listens on `0.0.0.0:8090`. Open the dashboard
at `http://<host>:8090/`.

## Quick start (development on macOS)

```bash
cd sphinx-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
uvicorn server:app --host 127.0.0.1 --port 8090 --reload
```

Sphinx is auto-detected as missing → **dry-run mode kicks in** and
each "drone" becomes a `sleep` placeholder. The dashboard is fully
functional; you can spawn 10 drones, restart, stop, watch the table
update. Useful for iterating on the UI before flying anything.

## Configuration

`config.example.yaml` is the source of truth. Copy to `config.yaml`
and edit. Key knobs:

```yaml
host: "0.0.0.0"
port: 8090
state_db: "./sphinx_control.db"
max_drones: 10            # cap on simultaneous instances
network:
  mode: "ports"           # or "netns"
  base_port: 9080         # ports mode: drone N → port 9080+N
  subnet: "10.202.0.0/24" # netns mode subnet
  bridge_name: "sphinx-br0"
  ip_offset: 10           # drone N → 10.202.0.(10+N)
sphinx:
  binary: "/usr/bin/sphinx"
  drone_descriptor_dir: "/opt/parrot-sphinx/usr/share/sphinx/drones"
  firmware_url: "https://firmware.parrot.com/Versions/anafi2/pc/%23latest/images/anafi2-pc.ext2.zip"
ue4_apps: ...             # registry of available UE4 worlds
drone_profiles: ...       # registry of drone descriptors
dry_run: false            # auto-true if sphinx binary missing
tailscale:
  enabled: true
  binary: "/usr/bin/tailscale"
```

Environment override: `SPHINX_CONTROL_CONFIG=/path/to/config.yaml`.

## Watching the simulator (Sunshine, RDP, VNC, NoMachine, …)

The `parrot-ue4-*` renderer is a regular X11/Wayland window — to be
visible in the operator's GNOME desktop (and therefore captured by
desktop-streaming tools like Sunshine + Moonlight), the UE4 subprocess
needs `DISPLAY` / `WAYLAND_DISPLAY` / `XAUTHORITY` pointing at the
active session.

When `sphinx-control` runs as a systemd service it has none of those
env vars in its parent process. To bridge that, the launcher
discovers the active graphical session via `loginctl` at startup and
splices the relevant env into every UE4 spawn. The dashboard's
session badge shows what was discovered (e.g. `session: sdc@wayland:wayland-0`).

```yaml
# config.yaml
gnome_session_attach: "auto"   # default
# or explicit:
gnome_session_attach: "x11::0"
gnome_session_attach: "wayland:wayland-0"
# or disable (UE4 inherits parent env unchanged):
gnome_session_attach: "off"
```

If multiple users are logged in, `SPHINX_CONTROL_SESSION_USER=foo`
forces auto-detect to pick `foo`'s session.

### Sunshine setup (quick reference)

For drone operators streaming the desktop to a Mac/iPad:

```bash
# On the Sphinx host:
sudo dpkg -i sunshine.deb           # from github.com/LizardByte/Sunshine
sunshine                            # web config at https://localhost:47990

# In sunshine config: enable the desktop application; the GNOME
# session is captured automatically. UE4 windows opened by
# sphinx-control will appear there.
```

From the Mac/iPad, install Moonlight, point it at the Sphinx host's
Tailscale IP, and you'll see the live UE4 view streamed at GPU-
encode speed. This is the path the SDC team uses for "I want to
watch the sim from the couch" — much smoother than X11 forwarding or
VNC because it's hardware-encoded H.264/HEVC.

The session-attach feature is what makes this work: without it, UE4
would launch into systemd's empty environment, fail to find a
display, and either crash or render to an offscreen buffer that
Sunshine can't see.

### Troubleshooting

If the session badge shows "none detected":

```bash
# 1. Confirm a graphical session exists at all:
loginctl list-sessions

# 2. Check that Active=yes and Type∈{x11,wayland} for one of them:
loginctl show-session <id>

# 3. Force a specific user (multi-user host):
sudo systemctl edit sphinx-control
# Add to override: Environment=SPHINX_CONTROL_SESSION_USER=sdc
sudo systemctl restart sphinx-control
```

If you're running `uvicorn` from a terminal (not systemd) and the
session is still not detected, you're already inside it — the
launcher's auto-detect would attach to your session. Just spawn a
drone and confirm the UE4 window appears on screen.

## Tailscale: making the per-drone endpoints externally reachable

Two paths, depending on which `network.mode` you chose:

### Option A — ports mode (recommended starting point)

Every drone is at `<host>:<base_port + N>`. Tailscale just needs to be
running on the host:

```bash
sudo tailscale up
# pin your tailnet IP (Tailscale assigns one):
tailscale ip -4   # → e.g. 100.85.42.7
```

From any other tailnet node:

```bash
# Drone 1 endpoint:
curl http://100.85.42.7:9081/...
# Drone 2:
curl http://100.85.42.7:9082/...
# Management dashboard:
open http://100.85.42.7:8090/
```

That's it — Tailscale already routes traffic to the host; the ports
disambiguate drones.

### Option B — netns mode (real per-drone IPs)

Each drone has its own IP. To make those IPs reachable from other
tailnet nodes, advertise the subnet:

```bash
# On the Sphinx host:
sudo tailscale up --advertise-routes=10.202.0.0/24

# Then in the Tailscale admin console (or via tailscale CLI),
# APPROVE the route. Routes don't take effect until approved.

# On any other tailnet node that wants to reach drones:
sudo tailscale up --accept-routes
```

After that, any tailnet node can reach `10.202.0.11, .12, ...` as if
they were normal LAN IPs. Your existing `controller_unified` running
on a separate machine simply opens connections to those IPs.

**Caveats for netns mode:**
- The management service needs `sudo` (passwordless) to create
  namespaces and veth interfaces. The provided `install.sh --enable`
  does NOT configure this — you'd need to add a sudoers rule like:
  ```
  sdc ALL=(root) NOPASSWD: /usr/sbin/ip
  ```
- The `parrot-ue4-*` binaries inside a netns may not have GPU access
  on every distro. If UE crashes inside the namespace but works
  outside, this is why; switching back to ports mode is the easy
  workaround.

## API reference

```
GET    /                           dashboard (HTML)
GET    /healthz                    {"ok": true}

GET    /api/drones                 list all (running + stopped) drones
POST   /api/drones                 spawn — body: {drone_profile, world_name, instance_id?, firmware_url?}
DELETE /api/drones/{id}            stop and remove from state
POST   /api/drones/{id}/stop       stop only (row stays for restart)
POST   /api/drones/{id}/restart    stop + respawn with same params
POST   /api/drones/restart-all     restart every running drone
POST   /api/drones/stop-all        stop every running drone
GET    /api/drones/{id}/connections   ss-based listeners + established peers

GET    /api/worlds                 registered UE4 apps
GET    /api/profiles               registered drone descriptors
GET    /api/system                 host + Tailscale + Sphinx info
```

Example: spawn a drone via curl

```bash
curl -X POST http://localhost:8090/api/drones \
     -H 'Content-Type: application/json' \
     -d '{"drone_profile":"anafi","world_name":"empty"}'
```

## How it actually works

```
              ┌────────────────────────────────┐
              │  sphinx-control (this service) │
              │  uvicorn FastAPI :8090         │
              └────────┬───────────────────────┘
                       │
        spawns subprocesses + tracks PIDs
                       │
   ┌───────────────────┼───────────────────────┐
   ▼                   ▼                       ▼
┌─────────────┐ ┌─────────────┐    …     ┌─────────────┐
│ sphinx (1)  │ │ sphinx (2)  │          │ sphinx (10) │
│ + parrot-   │ │ + parrot-   │          │ + parrot-   │
│   ue4-empty │ │   ue4-empty │          │   ue4-empty │
│ port 9081   │ │ port 9082   │          │ port 9090   │
└─────────────┘ └─────────────┘          └─────────────┘
```

State is persisted to SQLite (`sphinx_control.db`), so restarting the
management service doesn't lose track of what's running. On startup
the service reconciles: any rows whose PIDs are dead get marked
`stopped` so the dashboard isn't lying.

The launcher is intentionally tolerant of:
- Missing Sphinx binary → auto-dry-run.
- Missing CAP_NET_ADMIN in netns mode → falls back to ports mode and
  surfaces the reason as a banner.
- Stale rows from previous runs → reconciled at startup.

## Limits and caveats

- **One drone per Sphinx process** is a hard upper limit from Parrot's
  current architecture. Running 10 simultaneous instances on one box is
  GPU-limited; each UE4 renderer wants its own GPU slice. 4-6 is
  realistic on a single mid-range GPU; 10 may need either a beefy GPU
  or running multiple of these services across multiple hosts.
- **Sphinx CLI flags vary by version.** `launcher.py:_sphinx_argv` is
  a best-effort construction. Verify with `sphinx --help` on your
  installed version and adjust the flags (especially `--instance` and
  port-binding) if needed.
- **Multiple Sphinx instances on one host is not officially
  supported by Parrot.** Forum-confirmed workarounds exist (separate
  netns + per-instance ports) but you may hit edge cases. When you
  do, dry-run mode is your friend for verifying the management UI is
  doing the right thing before debugging Sphinx itself.

## Logs

Per-drone subprocess logs land at:

```
sphinx-control/logs/<drone_id>/sphinx.log
sphinx-control/logs/<drone_id>/ue4.log
```

The management service itself logs to stdout (and to journald when
run via systemd). Bump verbosity with `SPHINX_CONTROL_LOG_LEVEL=DEBUG`.

## Files

```
sphinx-control/
├── server.py                     FastAPI app, all HTTP endpoints
├── launcher.py                   Sphinx process supervisor
├── network.py                    PortsMode, NetnsMode, ss helpers
├── state.py                      SQLite drone records
├── worlds.py                     UE4 + drone-profile registry
├── templates/index.html          dashboard UI
├── static/{app.js,app.css}       frontend (vanilla JS, no build)
├── systemd/sphinx-control.service  unit file
├── install.sh                    venv + (optionally) systemd
├── requirements.txt              fastapi/uvicorn/pydantic/yaml/httpx
├── config.example.yaml           configuration template
└── README.md                     this file
```
