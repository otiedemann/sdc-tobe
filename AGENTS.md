# AGENTS Guide (`sdc-tobe`)
This repo is a multi-module Python workspace for Swarm Drone Challenge 2026.
Use this file as the default guide for agentic coding tools.

## Scope and layout
- Active modules:
  - `c2_interface/` (FastAPI C2 backend + web UI)
  - `controller/` (Flask C2 / remote web controller — "the C2"; also
    contains the original Tello flight-controller API, now historical)
  - `controller_unified/` (Flask unified flight-controller API for
    Anafi + Tello. Runs on an **x86 Linux PC** — the flight controller
    is NOT a Raspberry Pi in the current SDC26 fleet, despite legacy
    "Pi" naming in some files.)
  - `aruco-position/` (ArUco detection, calibration, streaming)
  - `sim_swarm/` and `sim_swarm_c2/` (simulation + viewers)
  - `drone-sim/` (HTTP/UDP simulator)
- `old/` is legacy; avoid edits unless requested.
- No root build system (`Makefile`, `pyproject.toml`, `package.json`).

## Environment setup
- Use per-module virtualenvs.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```
- Install dependencies per module:
```bash
pip install -r c2_interface/requirements.txt
pip install -r aruco-position/requirements.txt
pip install -r controller_unified/requirements.txt
```
- Only use `controller_unified/requirements-pi.txt` on legacy ARM hosts
  (Raspberry Pi, Tello-only). The current flight controllers are x86
  Linux — use the main `requirements.txt`.

## Build/run commands
No compile step exists; run scripts/services directly.

### C2 interface
```bash
cd c2_interface
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```
```bash
cd c2_interface
bash ensure_c2_sim_connected.sh
```

### Controller modules
```bash
cd controller
python3 tello_pi_api_server.py
python3 tello_remote_web_controller.py
```
```bash
cd controller_unified
python3 unified_pi_api_server.py
```

### ArUco module
```bash
cd aruco-position
python3 calibrate_camera.py
python3 stream.py
python3 aruco_positioning.py
```

### Simulators
```bash
python3 sim_swarm/run.py --scenario baseline
python3 sim_swarm/run.py --scenario degraded_link
python3 sim_swarm/run.py --scenario dropout
```
```bash
bash sim_swarm/serve_viewer.sh
bash sim_swarm_c2/serve_viewer.sh
```
```bash
cd drone-sim
python3 sim_server.py --config sim_config.example.json --udp-ip 127.0.0.1 --udp-port 5005
```

## Lint/static checks
- No committed repo-level lint/type config was found.
- Minimum syntax gate:
```bash
python -m compileall c2_interface controller controller_unified aruco-position sim_swarm drone-sim
```
- Optional (if installed):
```bash
ruff check c2_interface controller controller_unified aruco-position sim_swarm drone-sim
```
- Do not add new lint tooling/config unless explicitly requested.

## Test commands (including single test)
- No standardized automated test suite is currently committed.
- If `pytest` tests exist in your branch:
```bash
python -m pytest -q
python -m pytest path/to/test_file.py -q
python -m pytest path/to/test_file.py::test_name -q
```
- If `unittest` tests are used:
```bash
python -m unittest
python -m unittest path.to.module.TestClass.test_method
```
- Existing single-run legacy scripts:
```bash
python old/controller/test_connect.py
python old/controller/test_takeoff.py
python old/controller/test_sequence.py
```
- Scenario-focused single-run validation:
```bash
python3 sim_swarm/run.py --scenario baseline
```

## Code style guidelines
Follow local style in the touched module; this codebase mixes modern typed services and legacy scripts.

### Imports
- Group imports: standard library, third-party, local.
- In `c2_interface/app`, use relative imports (`from .models import ...`).
- In script-style modules (`sim_swarm/*.py`), existing direct local imports are acceptable.

### Formatting
- 4-space indentation.
- Keep lines readable (around 100 chars when practical).
- Prefer trailing commas in multiline literals/calls for stable diffs.

### Types and data modeling
- Add type hints for new/modified functions unless file is clearly legacy script code.
- Prefer `list[str]`, `dict[str, int]`, and `X | None` syntax.
- Use `pydantic.BaseModel` for C2 API payload/state models.
- Use `dataclass` for simulation domain structs where already established.

### Naming
- `snake_case`: functions, vars, files, route handlers.
- `PascalCase`: classes, dataclasses, Pydantic models.
- `UPPER_CASE`: module constants and env-backed defaults.
- Keep JSON/model fields snake_case to match existing APIs.

### Error handling and resilience
- FastAPI endpoints: validate early and raise `HTTPException` with clear `detail`.
- Convert lookup errors (`KeyError`) to HTTP 404/400 at API boundaries.
- Hardware/network I/O: catch operational exceptions and degrade safely.
- Async code: avoid blocking operations; use `asyncio` patterns.
- Threaded code: protect shared mutable state with locks.

### State, config, logging
- In C2 flows, update drone state via `StateStore` (avoid ad hoc shared mutation).
- Keep mission transitions explicit (`IDLE -> MOVING -> SEARCHING -> ...`).
- Keep connect/disconnect and simulator start/stop behavior idempotent.
- Read runtime settings from env vars with explicit defaults.
- Keep constants near module top and keep logs useful (avoid noisy per-tick output).

### Git hygiene
- Do not commit local envs, caches, logs, local Wi-Fi configs, or capture artifacts.
- Respect `.gitignore` entries (including `S10e/`, `S22/`, `aruco-position/Tello/`).

## Cursor/Copilot rules
No repository rule files were found:
- `.cursorrules` not present
- `.cursor/rules/` not present
- `.github/copilot-instructions.md` not present
If added later, treat them as higher-priority instructions and update this file.

## Agent workflow defaults
- Read the module README and nearby code before editing.
- Keep changes minimal and module-scoped unless broader refactor is requested.
- After behavior changes, run at least one relevant command from above.
- When adding tests, include a documented single-test invocation in your notes.

## Position tracker invariants (`controller_unified/ctrl_position.py`)
The FC positioner is a long-lived process whose internal state can leak
between flights. Three guards keep the published position trustworthy:

1. **Takeoff resets full tracker state.** `/api/takeoff` calls
   `HeadlessAruCoPositioning.reset_tracker_state()` before commanding the
   drone up. This clears `state_pos`, `state_vel`, `last_valid_pose`,
   the per-axis Kalman filters, the direction filter, the recovery
   candidate, and the per-target EMAs. Without this, a previous flight's
   poisoned `state_pos` (e.g. Z = −1866 m from a bad single-marker fix)
   carries into the new takeoff and the EMA needs ~5 s to drag the
   published position back to reality. **Do not bypass.**
2. **Pose-jump gate (`max_pose_jump_m`, default 3.0 m).** While tracking
   is continuous (`now − last_valid_ts ≤ POSE_HOLD_SEC`), a fresh fix
   that disagrees with the predicted state by more than this is
   rejected and we coast on the prediction. Set 0 to disable for
   debugging.
3. **Re-acquisition consensus (`RECOVERY_*` constants).** After a stale
   gap longer than `POSE_HOLD_SEC`, the next fresh fix becomes a
   *candidate* and we keep coasting; only when a second fresh fix lands
   within `RECOVERY_DT_S` and `RECOVERY_DIST_M` of the candidate is it
   confirmed and accepted (with the measurement partially blended toward
   the predicted state via `RECOVERY_ALPHA_SCALE` so the position
   doesn't snap).

When changing the position tracker, keep these three guards in place
unless field-log analysis shows a clearly better alternative. Live-tune
via `/api/position/config` (knob: `max_pose_jump_m`); recovery and
takeoff-reset behaviour are intentionally not exposed as runtime knobs.
