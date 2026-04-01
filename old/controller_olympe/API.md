# Olympe Anafi Control API (Pi Server)

This documents all endpoints exposed by `controller_olympe/olympe_pi_api_server.py` for controlling a Parrot Anafi drone via the Olympe SDK, including ArUco positioning and arena configuration.

## Base URL

`http://<PI_IP>:8080`

Use JSON for POST requests: `Content-Type: application/json`

---

## Runtime Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANAFI_IP` | `192.168.42.1` | Drone IP address |
| `TELEMETRY_HZ` | `2.0` | Telemetry polling rate in Hz |
| `KEY_STALE_S` | `1.0` | Key auto-release timeout (seconds) |
| `SAFE_TAKEOFF_S` | `3.0` | Post-takeoff hover hold time (seconds) |
| `SAFE_TAKEOFF_DEFAULT` | `0` | Enable safe takeoff by default (`1`/`true`) |
| `MAX_ALTITUDE_M` | `2.0` | Hard altitude ceiling (meters) — overridden by `flight_config.json` if present |
| `MAX_VERTICAL_SPEED` | `0.5` | Max vertical speed (m/s) — overridden by `flight_config.json` |
| `MAX_TILT` | `15` | Max tilt angle (degrees) — overridden by `flight_config.json` |
| `API_COMMAND_LOG` | `1` | Enable command logging (`1`/`true`) |
| `VIDEO_MODE` | `off` | Auto-start video on launch: `off`, `mjpeg`, `forward` |

---

## Persistent Config Files

All files live alongside `olympe_pi_api_server.py`:

| File | Purpose |
|------|---------|
| `flight_config.json` | Flight limits (altitude, speed, tilt, yaw) — saved by `POST /api/settings` |
| `position_config.json` | ArUco positioning settings (enabled, profile, FOV, latency) |
| `position_calib.npz` | Camera calibration data (uploaded via `POST /api/position/calibration`) |
| `arena_config.json` | Arena dimensions and ArUco marker layout |

---

## Endpoints

### Health Check

#### `GET /`

Returns server status.

**Response:** `{"ok": true, "service": "olympe_pi_api_server", "connected": true}`

---

### Key-Based Control

Keys are translated to PCMD frames at 20 Hz.

| Key | Action |
|-----|--------|
| W / S | Forward / Backward (pitch) |
| A / D | Left / Right (roll) |
| R / F | Up / Down (gaz) |
| Q / E | Rotate CCW / CW (yaw) |
| T | Takeoff |
| L | Land |
| Space / X | Zero all axes (stop) |

#### `POST /api/key_down`
Register a key press. Keepalive must arrive before `KEY_STALE_S` expires.

**Body:** `{"key": "w"}` · **Response:** `{"ok": true}`

#### `POST /api/key_up`
Release a key.

**Body:** `{"key": "w"}` · **Response:** `{"ok": true}`

---

### Flight Actions

#### `POST /api/takeoff`
Take off. Returns 409 during cooldown.

**Response:** `{"ok": true, "flying": true, "safe_takeoff": false}`

#### `POST /api/land`
Land the drone. Retries up to 3 times.

**Response:** `{"ok": true, "flying": false}`

#### `POST /api/flip`
Execute a flip. Requires battery ≥ 50% and flying.

**Body:** `{"dir": "f"}` — `l`, `r`, `f`, `b` · **Response:** `{"ok": true, "dir": "f"}`

#### `POST /api/emergency`
Motor cutoff — falls immediately.

**Response:** `{"ok": true}`

#### `POST /api/recover`
Reconnect to drone, clear all keys and overrides, re-apply flight limits.

**Response:** `{"ok": true, "message": "recovered"}`

---

### Movement Commands

#### `POST /api/move`
Move in a cardinal direction by distance in cm (Olympe `moveBy`).

**Body:**
```json
{"dir": "forward", "cm": 50}
```
- `dir`: `up`, `down`, `left`, `right`, `forward`, `back`
- `cm`: 20–2000

**Response:** `{"ok": true, "dir": "forward", "cm": 50}`

#### `POST /api/rotate`
Rotate by degrees.

**Body:** `{"dir": "cw", "deg": 90}` — `dir`: `cw` or `ccw`, `deg`: 1–360

**Response:** `{"ok": true, "dir": "cw", "deg": 90}`

#### `POST /api/go`
Move to relative XYZ position in cm (Tello convention: Z+ = up).

**Body:** `{"x": 100, "y": 0, "z": 50}`

**Response:** `{"ok": true, "x": 100, "y": 0, "z": 50}`

#### `POST /api/rc`
RC override: set PCMD values directly for a duration.

**Body:**
```json
{"lr": 0, "fb": 50, "ud": 0, "yaw": 0, "duration_ms": 500}
```
- `lr`, `fb`, `ud`, `yaw`: -100 to 100
- `duration_ms`: 50–2000

**Response:** `{"ok": true, "rc": {...}, "duration_ms": 500}`

---

### GPS Navigation

#### `POST /api/moveto`
Move to absolute GPS coordinates. Requires GPS fix.

**Body:** `{"lat": 48.8566, "lon": 2.3522, "alt": 5.0, "heading": 0}`

**Response:** `{"ok": true, "lat": 48.8566, "lon": 2.3522, "alt": 5.0}`

#### `POST /api/rth`
Start or cancel Return-To-Home.

**Body:** `{"action": "start"}` or `{"action": "cancel"}`

**Response:** `{"ok": true, "action": "start"}`

---

### Camera

#### `POST /api/camera/photo`
Take a single photo. · **Response:** `{"ok": true}`

#### `POST /api/camera/record/start`
Start video recording. · **Response:** `{"ok": true, "recording": true}`

#### `POST /api/camera/record/stop`
Stop video recording. · **Response:** `{"ok": true, "recording": false}`

#### `POST /api/gimbal`
Set gimbal tilt and pan.

**Body:** `{"tilt": -45, "pan": 0}` — tilt: -90..+30°, pan: -180..+180°

**Response:** `{"ok": true, "tilt": -45, "pan": 0}`

---

### Telemetry

#### `GET /api/telemetry`
All telemetry as JSON. No-cache headers set.

**Response:**
```json
{
  "battery": 85,
  "height_cm": 150.0,
  "pitch": 1.23, "roll": -0.45, "yaw": 90.1,
  "vgx": 12.3, "vgy": -5.0, "vgz": 0.0, "speed": 13.3,
  "gps_lat": 48.8566, "gps_lon": 2.3522, "gps_alt": 45.2,
  "gimbal_pitch": -45.0, "gimbal_roll": 0.0, "gimbal_yaw": 0.0,
  "flying": true, "connected": true,
  "state_age_s": 0.31, "state_fresh": true,
  "updated_at": 1711360000.0
}
```

| Field | Unit | Source |
|-------|------|--------|
| `battery` | % | `BatteryStateChanged` |
| `height_cm` | cm | `AltitudeChanged` (m→cm) |
| `pitch`, `roll`, `yaw` | degrees | `AttitudeChanged` (rad→deg) |
| `vgx`, `vgy`, `vgz` | cm/s | `SpeedChanged` (m/s→cm/s) |
| `speed` | cm/s | √(vgx²+vgy²+vgz²) |
| `gps_lat`, `gps_lon` | decimal deg | `GpsLocationChanged` |
| `gps_alt` | meters | `GpsLocationChanged` |
| `gimbal_pitch/roll/yaw` | degrees | Gimbal attitude |
| `state_age_s` | seconds | Time since last telemetry |
| `state_fresh` | boolean | True if `state_age_s` ≤ 2.0 |

#### `GET /api/telemetry/stream`
SSE stream — pushes telemetry JSON every ~500ms.

```
data: {"battery": 85, "height_cm": 150, ...}
```

---

### Flight Settings

#### `GET /api/settings`
Returns current flight limits and feature availability.

**Response:**
```json
{
  "max_altitude_m": 2.0,
  "max_vertical_speed": 0.5,
  "max_tilt": 15,
  "max_yaw_speed": 150,
  "geofence_available": true,
  "camera_available": true,
  "gimbal_available": true,
  "gps_available": true,
  "rth_available": true,
  "moveto_available": true,
  "video_mjpeg_available": true
}
```

#### `POST /api/settings`
Update flight limits on the drone. All fields optional. **Persists to `flight_config.json`** — values survive server restart.

**Body:**
```json
{
  "max_altitude_m": 3.0,
  "max_vertical_speed": 1.0,
  "max_tilt": 20,
  "max_yaw_speed": 150,
  "geofence_distance": 50,
  "geofence_enabled": true
}
```

| Setting | Range | Unit |
|---------|-------|------|
| `max_altitude_m` | 0.5–150 | meters |
| `max_vertical_speed` | 0.1–4.0 | m/s |
| `max_tilt` | 1–35 | degrees |
| `max_yaw_speed` | 1–200 | deg/s |
| `geofence_distance` | 10–4000 | meters |
| `geofence_enabled` | bool | — |

**Response:** `{"ok": true, "max_altitude_m": 3.0, ...}`

---

### Safety

#### `GET /api/safety/takeoff`
Get safe takeoff status.

**Response:** `{"enabled": false, "hold_s": 3.0}`

#### `POST /api/safety/takeoff`
Toggle safe takeoff mode.

**Body:** `{"enabled": true}` · **Response:** `{"ok": true, "enabled": true, "hold_s": 3.0}`

---

### Video Streaming

#### `GET /api/video`
MJPEG stream (Way 1 — on-Pi decode). Returns `multipart/x-mixed-replace`.
Must start with `POST /api/video/start` first.

#### `POST /api/video/start`
Start video streaming.

**Body:** `{"mode": "mjpeg"}` or `{"mode": "forward", "target_host": "192.168.1.5", "target_port": 55004}`

**Response:** `{"ok": true, "mode": "mjpeg", "stream_url": "http://PI_IP:8080/api/video"}`

#### `POST /api/video/stop`
Stop video streaming. · **Response:** `{"ok": true, "mode": "off"}`

#### `GET /api/video/status`
Video mode status.

**Response:** `{"mode": "mjpeg", "has_frame": true, "frame_count": 412}`

---

### Logging

#### `GET /api/logging/commands`
Command log status. · **Response:** `{"enabled": true, "path": "api_command_log.jsonl"}`

#### `GET /api/logging/commands/download`
Download command log as JSONL.

#### `POST /api/logging/commands/clear`
Clear command log.

#### `GET /api/logging/telemetry`
Telemetry log status.

#### `POST /api/logging/telemetry`
Configure telemetry logging.

**Body:** `{"enabled": true, "path": "/custom/path.jsonl"}`

#### `GET /api/logging/telemetry/download`
Download telemetry log as JSONL.

#### `POST /api/logging/telemetry/clear`
Clear telemetry log.

---

### ArUco Positioning

The positioning subsystem runs in a background thread and processes frames from the live video stream — **no second Olympe connection is needed**. Enable it via `POST /api/position/config`.

#### `GET /api/position`
Current position snapshot.

**Response:**
```json
{
  "enabled": true,
  "pos": [1.23, 4.56, 0.12],
  "dir": [0.0, 1.0, 0.0],
  "vel": [0.1, -0.05, 0.0],
  "ref_markers": [5, 7, 9],
  "marker_weights": {"5": 0.87, "7": 0.72, "9": 0.91},
  "stale": false,
  "fps": 14.2,
  "ts": 1711360000.0,
  "latency_ms": 200.0,
  "proc_active": true
}
```

| Field | Description |
|-------|-------------|
| `pos` | `[X, Y, Z]` in arena coords (m). X: ±arena_half_width, Y: 0–depth, Z: height |
| `dir` | Normalised forward-direction unit vector `[dx, dy, dz]` |
| `vel` | EMA-smoothed arena-frame velocity `[vx, vy, vz]` m/s |
| `ref_markers` | ArUco IDs used in current pose estimate |
| `marker_weights` | Per-marker quality score (area × distance × squareness × reprojection) |
| `stale` | True when using cached pose during temporary marker loss (≤ 0.8 s) |
| `proc_active` | True once the processor has been initialised (after first frame) |

#### `GET /api/position/events`
SSE stream — emits one event per processed frame (~camera FPS).

```
data: {"ts": 1711360000.1, "pos": [1.23, 4.56, 0.12], "vel": [...], "fps": 14.2, ...}
```

Additional fields: `battery_pct`, `altitude_cm`, `yaw_deg`, `flying_state`.

#### `GET /api/position/video`
MJPEG stream of ArUco-annotated frames. Returns `multipart/x-mixed-replace`.
Only produces frames when positioning is enabled AND video streaming is active.

#### `GET /api/position/config`
Read positioning settings.

**Response:**
```json
{
  "enabled": false,
  "detect_profile": "balanced",
  "fov_deg": 69.0,
  "latency_ms": 200.0,
  "has_calibration": false,
  "has_module": true
}
```

#### `POST /api/position/config`
Update positioning settings. **Persists to `position_config.json`**.

**Body** (all fields optional):
```json
{
  "enabled": true,
  "detect_profile": "sensitive",
  "fov_deg": 69.0,
  "latency_ms": 250.0
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `enabled` | bool | Start/stop the processing thread |
| `detect_profile` | `balanced` / `sensitive` / `strict` | ArUco detector tuning |
| `fov_deg` | 40–120 | Camera horizontal FOV (used for default intrinsics when no calibration file) |
| `latency_ms` | 0–2000 | Dead-reckoning latency compensation applied on the browser side |

**Response:** `{"ok": true}`

#### `POST /api/position/calibration`
Upload camera calibration as NPZ file (`camera_matrix`, `dist_coeffs` fields).
**Persists to `position_calib.npz`** and takes effect immediately.

**Body:** `multipart/form-data` with field `file` = `.npz` file

**Response:** `{"ok": true, "shape": [3, 3]}`

#### `GET /api/position/calibration`
Download saved calibration NPZ.

---

### Arena Configuration

Defines physical arena dimensions and the world-space positions of all ArUco reference markers. Used by the positioning processor to convert camera observations into arena coordinates.
**Persists to `arena_config.json`** — survives server restart.

#### `GET /api/arena/config`
Returns the full arena configuration.

**Response:**
```json
{
  "arena": {
    "width_m": 20.0,
    "depth_m": 10.0,
    "height_min_m": -1.0,
    "height_max_m": 1.0
  },
  "marker_size_m": 0.5,
  "markers": {
    "0":  {"pos": [0.0, 0.0, 0.0],    "wall": "front"},
    "1":  {"pos": [10.0, 6.667, -1.0], "wall": "right"},
    "5":  {"pos": [6.0, 0.0, -1.0],   "wall": "front"},
    "17": {"pos": [-6.0, 10.0, -1.0], "wall": "back"}
  }
}
```

**Arena coordinate system:**
- X axis: left (−) ↔ right (+), total `width_m` (default ±10 m)
- Y axis: back wall (0) → front wall (`depth_m`, default 10 m)
- Z axis: floor (≈ `height_min_m`) ↔ ceiling (≈ `height_max_m`)

**Wall values:** `front`, `back`, `left`, `right`

#### `POST /api/arena/config`
Update arena configuration. Omit any top-level key to leave it unchanged.
Triggers immediate positioning processor reinit.

**Body** (all top-level keys optional):
```json
{
  "arena": {
    "width_m": 16.0,
    "depth_m": 8.0,
    "height_min_m": -0.5,
    "height_max_m": 1.5
  },
  "marker_size_m": 0.4,
  "markers": {
    "0": {"pos": [0.0, 0.0, 0.0], "wall": "front"},
    "5": {"pos": [4.0, 0.0, -0.5], "wall": "front"}
  }
}
```

**Response:** `{"ok": true, "marker_count": 24, "marker_size_m": 0.4, "arena": {...}}`

#### `POST /api/arena/config/reset`
Reset to built-in defaults (SDC arena layout: 20 × 10 m, 24 markers).

**Response:** `{"ok": true, "reset": true}`

---

## Safety Features

### Altitude Fence
- `MaxAltitude` set on drone connect (from `flight_config.json` or `MAX_ALTITUDE_M` env var)
- RC loop also zeros upward gaz when `height_cm >= max_altitude_m * 100`

### Speed Limits
Applied via Olympe SDK on connect and on `POST /api/settings`:
- Max vertical speed: 0.5 m/s default (`MaxVerticalSpeed`)
- Max tilt: 15° default (`MaxTilt`)
- Max yaw speed: 150°/s default (`MaxRotationSpeed`)

### Persistent Flight Limits
Flight limits set via `POST /api/settings` are saved to `flight_config.json` and reloaded on every server start, so your custom limits survive restarts without environment variables.

### Stale Key Timeout
Keys auto-release after `KEY_STALE_S` seconds without keepalive.

### Auto-Reconnect
Background loop retries connection every 3 seconds when disconnected. Flight limits are re-applied on every successful reconnect.

### Watchdog Auto-Land
If no request arrives for `REMOTE_TIMEOUT_S` seconds while flying, the drone lands automatically.

---

## Architecture

```
Remote Web Controller (port 8090)
        │
        │  /proxy/*  →  /api/*
        ▼
  Olympe Pi API Server (port 8080)
        │
        ├── Olympe SDK ─── Parrot Anafi (192.168.42.1)
        │     PCMD @ 20 Hz, commands, telemetry polling
        │
        └── ArUco Positioning (positioning_loop thread)
              │  reads from video frame queue (no extra Olympe connection)
              │  HeadlessAruCoPositioning (pi_position.py)
              └── SSE  →  /api/position/events
                  MJPEG →  /api/position/video
```

### Background Threads

| Thread | Purpose |
|--------|---------|
| `reconnect_loop` | Monitors Olympe connection, retries every 3 s, re-applies flight limits |
| `telemetry_loop` | Polls battery, attitude, speed, altitude, GPS, gimbal at `TELEMETRY_HZ` |
| `rc_loop` | Sends PCMD at 20 Hz from web key state + RC override; enforces altitude fence |
| `watchdog_loop` | Auto-lands if no remote heartbeat for `REMOTE_TIMEOUT_S` seconds |
| `positioning_loop` | Reads BGR frames from video queue, runs ArUco detection, publishes SSE |

### Config Files Summary

| File | Written by | Read by |
|------|-----------|---------|
| `flight_config.json` | `POST /api/settings` | `main()` on startup |
| `position_config.json` | `POST /api/position/config` | `_PositioningState.__init__` |
| `position_calib.npz` | `POST /api/position/calibration` | `positioning_loop` (on init) |
| `arena_config.json` | `POST /api/arena/config` | `positioning_loop` (on processor reinit) |
