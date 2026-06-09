# sdc-tobe

Code and infrastructure for the Stuttgart Drone Challenge "ToBeDefined"
team. Fly a real or simulated Parrot Anafi from a browser, with arena
positioning, mission scripting, video, and recording.

This README is the operator's map for the **AWS sphinx deployment**:
what's running, on which ports, how to reach each component, and how
to recover from common failures. For component-internal details see the
README inside each subfolder.

---

## TL;DR — what's where

The AWS instance `sphinx3.otconsulting.de` (Tailnet IP `100.105.250.85`)
runs the simulator and three web UIs. All four HTTP services start
automatically at boot via systemd.

| Port | Service                              | What it does                                                     |
|-----:|--------------------------------------|------------------------------------------------------------------|
| 8070 | **C2 / remote web controller**       | Operator UI — manual flight, video, telemetry, recording          |
| 8080 | **FC** or **marker-mission**         | Mutually exclusive: either unified_api_server.py (HTTP-only) OR `marker_mission.app` (combined: HTTP API + mission UI in one process) |
| 8090 | **sphinx-control**                   | Sim dashboard — start/stop env, drones, FC                       |
| 9091 | **drone-detector-ui**                | Collision-detector dashboard — live video + YOLO overlay + evasion config |

Open these in your browser over Tailscale:

- `http://sphinx3.otconsulting.de:8070/` — **start here for manual flight**
- `http://sphinx3.otconsulting.de:8090/` — sim manager / "is the drone up?"
- `http://sphinx3.otconsulting.de:8080/` — FC HTTP API (always present)
  - …and `:8080/mission` when marker-mission mode is active — scripted ArUco missions

---

## System architecture

```
                                    AWS g4dn.xlarge (Tesla T4)
                                    ┌────────────────────────────────────┐
   your browser                     │                                    │
   over Tailscale  ─────────────────▶  C2 :8070  remote_web_controller   │
                                    │     │                              │
                                    │     ▼  (HTTP proxy)                │
                                    │  FC :8080  unified_api_server.py   │
                                    │     │   Olympe + pdraw + pysphinx  │
                                    │     ▼  (UDP/TCP to 10.202.0.1)     │
                                    │  sphinx ─ gzserver + UE4 (NVENC)   │
                                    │     ▲   (managed by ↓)             │
                                    │     │                              │
                                    │  sphinx-control :8090              │
                                    │     uses sphinx-bootstrap.service  │
                                    │     to spawn env+drone+FC at boot  │
                                    │                                    │
                                    │  marker-mission :8080 (combined)   │
                                    │     in-process drone control +     │
                                    │     mission UI (/mission)          │
                                    └────────────────────────────────────┘
```

Boot order (enforced by systemd):

```
firmwared.service ── xvfb.service ──┐
                                    │
sphinx-control.service ◀────────────┘
        │
        ▼
sphinx-bootstrap.service          (oneshot — ~60 s)
        │   1. POST /api/environment   sdc_arena_test
        │   2. POST /api/drones        anafi-4k
        │   3. POST /api/fc            (hardcoded 10.202.0.1)
        ▼
marker-mission.service + c2-controller.service
```

`xvfb.service` is misnamed: it actually runs **Xorg with the NVIDIA
driver** on `:99` (not Xvfb). UE4 needs a GPU-backed X server for its
SceneCaptureComponents — Xvfb's software GLX produced black vertical
camera frames that prevented takeoff. See `docker/AWS-NATIVE.md` for
the full investigation if you ever need to rebuild this host.

---

## Interface reference

### `:8070` — Remote web controller (C2)

Operator UI for manual flight. Built on Flask, single-page HTML.
Source: `controller/remote_web_controller.py`.

**What you can do here**:
- Connect to a drone, take off / land, fly with WASD or arrow keys
- See live video — three modes:
  - **Way 1: MJPEG** — per-frame JPEG, ~8 Mbps, decoded in browser. Works in every browser but slow on weak links and CPU-heavy.
  - **Way 2: UDP forward** — H.264 over UDP to your C2 box, decoded there, re-encoded to MJPEG for the browser. Useful when the FC↔C2 path is internal but the C2↔browser path is fast.
  - **Way 3: H.264 (NVENC)** — H.264 over MPEG-TS straight from the GPU, ~700 kbps, played in `<video>` via hardware decode. Safari plays it native; Chrome/Firefox auto-load `mpegts.js` from CDN. **About 11× less bandwidth than MJPEG.** Sim-only.
- Live telemetry: altitude, battery, GPS, attitude, position-tracker output
- Camera zoom (1.0× – 3.0× digital)
- Record video to disk, browse + download recordings
- Mission preset library + flight-log viewer

**Override port at launch** (in case of conflict):

```bash
C2_HTTP_PORT=8071 PI_BASE=http://localhost:8080 python3 controller/remote_web_controller.py
```

### `:8080` — Flight controller (FC)

The actual drone-control HTTP API. Used by the C2 + by the mission
planner. Source: `controller_unified/unified_api_server.py`.

Selected endpoints (full list in the source):

| Method | Path                          | Purpose                                          |
|--------|-------------------------------|--------------------------------------------------|
| POST   | `/api/takeoff`                | Take off                                         |
| POST   | `/api/land`                   | Land                                             |
| POST   | `/api/rc`                     | RC stick input (roll, pitch, yaw, throttle)      |
| GET    | `/api/telemetry`              | Snapshot of pose, battery, alerts                |
| GET    | `/api/state`                  | Full Olympe state dump                           |
| POST   | `/api/video/start`            | Start MJPEG producer (mode = mjpeg or forward)   |
| GET    | `/api/video`                  | MJPEG `image/x-mixed-replace` stream             |
| POST   | `/api/video/h264/start`       | Ensure H.264 producer is up                      |
| GET    | `/api/video/h264`             | **H.264/MPEG-TS live stream** (NVENC)            |
| GET    | `/api/video/status`           | `{mode, has_frame, frames_decoded, ...}`         |
| POST   | `/api/video/record/start`     | Start writing video to disk                      |
| GET    | `/api/video/recordings`       | List recordings                                  |

### `:8090` — sphinx-control

Sim manager. Manages a single Sphinx simulator instance, the UE4
environment, one or more sim drones, and the FC process. Source:
`sphinx-control/server.py`.

**The Anafi IP is hardcoded to `10.202.0.1`** — the UI no longer offers
a selector, and the `/api/fc` handler ignores any `anafi_ip` field a
caller might send. This server only ever flies the simulated Anafi.

Selected endpoints:

| Method | Path                       | Purpose                                    |
|--------|----------------------------|--------------------------------------------|
| POST   | `/api/environment`         | Start a UE4 world (e.g. `sdc_arena_test`)  |
| DELETE | `/api/environment`         | Stop the world                             |
| POST   | `/api/drones`              | Spawn a sim drone (`{drone_profile: "anafi-4k"}`) |
| GET    | `/api/drones`              | List drones with `extras.alive`            |
| POST   | `/api/fc`                  | Start the FC subprocess                    |
| DELETE | `/api/fc`                  | Stop the FC                                |
| POST   | `/api/fc/restart`          | Restart the FC                             |
| GET    | `/api/system`              | Host info: GPU, X session, disk, etc.      |

`sphinx-bootstrap.service` calls these in order at boot to bring the
sim up automatically — see `sphinx-control/sphinx-bootstrap.sh`.

### `:8080/mission` — marker-mission (combined app)

Mission planner UI. Source: `marker_mission/` in the main branch.
Runs as `python -m marker_mission.app`, which boots one Flask app
serving both the unified drone REST API AND the mission UI on the
same port (8080 by default). Drone control inside marker_mission is
in-process via `drone_core` — no HTTP round-trip from the mission
controller to the FC.

The legacy two-process setup (FC HTTP on :8080 + mission on :9090)
is retired; use the Ansible playbook in `ansible/fc-deploy/` to
switch a host between FC-only mode and combined-app mode (mutually
exclusive — both units `Conflicts=` each other in their `[Unit]`
section so they can never run together).

**Mission script DSL** (one command per line, `#` for comments):

| Command | Args | Description |
|---|---|---|
| `TAKEOFF` | — | Lift off |
| `APPROACH` | `[marker-id] [dist]` | Fly to a marker, stop at `dist` m |
| `HOOVER` | `[seconds]` | Station-keep |
| `AWAIT` | `marker-id timeout-s` | Hold until marker visible (with timeout) |
| `WAIT_AND_ATTACK` | `marker-id [dist] [dist_tol_m] [yaw_tol_deg]` | APPROACH the box (acquired via either the target face or its sibling face, offset ±10 by default), hold at standoff while the wrong face is shown, attack the instant `marker-id` becomes visible. No timeout. |
| `PAUSE` | `seconds` | Unconditional IDLE |
| `REPEAT` | — | Last step only: loop back to the first non-`TAKEOFF` step without landing. Converts a one-shot mission into a continuous patrol. |
| `LAND` | — | Land |
| `HEIGHT` | `[height]` | Climb/descend to arena height (m) |
| `TO` | `x y [z] [yaw\|auto]` | Drive to arena-frame coordinate |
| `DANCE` | `[seconds] [mode]` | mode ∈ `wobble \| spin \| random` |
| `LR_RC` | `rc [seconds]` | Raw strafe stick (+right, -left), `rc` ∈ `[-100,+100]`, default 1 s |
| `FB_RC` | `rc [seconds]` | Raw forward/back stick |
| `UD_RC` | `rc [seconds]` | Raw up/down stick |
| `YAW_RC` | `rc [seconds]` | Raw yaw stick (+cw, -ccw) |
| `LR_IMU` | `meters` | **Closed-loop** strafe by exact distance (+right, -left), `\|m\|` ∈ `[0.01,5.0]` |
| `FB_IMU` | `meters` | Closed-loop forward/back move |
| `UD_IMU` | `meters` | Closed-loop up/down move |
| `YAW_IMU` | `deg` | Closed-loop rotation, `deg` ∈ `[-180,+180]` (+cw, -ccw) |
| `RC` | `lr fb ud yaw [seconds]` | All four sticks at once |
| `SCOUT` | — | Slow 360° yaw spin (CW, stick=15) — "look around" |

Seconds may be fractional (`1.5`, `0.25`). Full docs at
`marker_mission/docs/MISSION_REFERENCE.md`.

---

## Operating cheatsheet

### SSH in

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@sphinx3.otconsulting.de
```

Tailscale-routed via the `sphinx-aws` node; no AWS Security Group rules
required for SSH.

### See what's running

```bash
systemctl is-active firmwared xvfb sphinx-control sphinx-bootstrap \
                    marker-mission c2-controller
sudo ss -lntp | awk '/:8070|:8080|:8090/ {print $4}'
```

### Tail logs

```bash
sudo journalctl -u sphinx-control -f      # sim manager
sudo journalctl -u sphinx-bootstrap       # one-shot boot sequence
sudo journalctl -u marker-mission -f
sudo journalctl -u c2-controller -f
ls -lt /opt/sdc-tobe/sphinx-control/logs/  # per-instance FC + drone + env logs
```

### Restart a single service

```bash
sudo systemctl restart sphinx-control
sudo systemctl restart marker-mission
sudo systemctl restart c2-controller
```

### Re-bring the sim up by hand (skip waiting for systemd)

```bash
curl -s -X DELETE http://localhost:8090/api/fc
curl -s -X DELETE http://localhost:8090/api/environment
sudo systemctl restart firmwared sphinx-control
sudo systemctl start sphinx-bootstrap   # re-runs the same script
```

### Pull a code update + restart

```bash
cd /opt/sdc-tobe && git pull
sudo systemctl restart sphinx-control marker-mission c2-controller
# FC restarts via sphinx-control's POST /api/fc
```

### Perf debug

```bash
# from this box
/opt/sdc-tobe/.venv/bin/python /opt/sdc-tobe/tools/perf_probe.py \
  --client-url http://localhost:8080/api/video

# from your laptop, measuring across the network
python3 tools/perf_probe.py \
  --skip-shm --skip-jpeg \
  --fc-url http://sphinx3.otconsulting.de:8080 \
  --client-url http://sphinx3.otconsulting.de:8080/api/video \
  --ping sphinx3.otconsulting.de
```

The probe measures GPU util, UE4 render rate, FC encode rate, network
delivery, and per-stage bitrate. Output ends with a `how to read this`
section that names the bottleneck.

---

## Repo layout

```
controller_unified/        flight controller (FC)
  unified_api_server.py    the :8080 service, Olympe + pdraw + pysphinx
controller/                operator UI (C2)
  remote_web_controller.py the :8070 service
sphinx-control/            sim manager
  server.py                the :8090 service
  launcher.py              drone/env/FC subprocess management
  sphinx-bootstrap.sh      systemd oneshot bring-up
marker_mission/            mission DSL + planner UI
  app.py                   combined entry point (FC + UI on :8080)
  mission.py               the mission state machine + CLI
  docs/                    DSL reference
tools/
  perf_probe.py            end-to-end perf measurement
  sphinx-arena/            ArUco arena generation (FBX + arena.yml)
aruco-position/            ArUco marker PDF generators
  generate_markers_50cm.py 50x50 cm 4x4_1000 markers
docker/
  AWS.md                   Docker deployment (historic)
  AWS-NATIVE.md            current native install on AWS — read this if rebuilding
```

---

## Common gotchas

- **`sphinx3.otconsulting.de` is a Tailnet hostname** (resolves to `100.x.x.x`). It only works on devices joined to your Tailscale tailnet. From outside the tailnet, use the EC2 public IP — and that requires the AWS Security Group to allow inbound on the relevant port (8070, 8080, 8090).
- **GitHub auth on the box uses SSH**, not HTTPS. The deploy key for `sdc-tobe` lives at `~/.ssh/github_ed25519`. Use `git pull` and `git push` over `git@github.com:otiedemann/sdc-tobe.git`.
- **`libjsonrpccpp-{common,server,client}0` are `apt-mark manual`** — if those packages get auto-removed (as happened once after a Docker purge), UE4 crashes during init because gzserver's `libsphinx_fwman.so` can't bind port 8383.
- **EC2 Spot instances can be reclaimed** without warning. If `start-instances` fails with "no Spot capacity", relaunch the EBS root on an On-Demand instance via the AMI route — see `docker/AWS.md`.
- **The host runs Xorg with NVIDIA on `:99`**, not Xvfb. Xvfb has software GLX which doesn't produce frames for the vertical camera SceneCaptureComponent → takeoff is blocked by `sensorState=0`. If you find Xvfb running here, the host was rebuilt from a stale image and needs the NVIDIA-Xorg setup re-applied.
- **Never `olympe.Drone("10.202.0.1").connect()` while the FC is running.** ARSDK on the Anafi is single-controller per drone — your probe evicts the FC's session, the FC's keepalive immediately fails (`Too many ping failures`), and the wedge propagates. If you need to inspect drone state, hit the FC's HTTP API instead (`curl :8080/api/state | jq`) or stop the FC first with `curl -X DELETE :8090/api/fc`. Same trap from any second Olympe client — including a second instance of `sphinx-cli` if it grows an Olympe consumer.

---

## Where to go next

- Want to add a new FC endpoint? Look at `controller_unified/unified_api_server.py` and proxy it through `controller/remote_web_controller.py` if the C2 UI needs it.
- Want to add a new sim world? Drop a `parrot-ue4-<world>` package in `/opt/`, register it in `sphinx-control/config.yaml` under `ue4_apps`.
- Want a new mission DSL command? Extend `marker_mission/mission_script.py`'s parser + `marker_mission/controller.py` step executor, document in `marker_mission/docs/MISSION_REFERENCE.md`.
