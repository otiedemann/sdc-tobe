# Olympe Anafi Control API (Pi Server)

This documents the API exposed by `controller_olympe/olympe_pi_api_server.py` for controlling a Parrot Anafi drone via the Olympe SDK.

## Base URL

`http://<PI_IP>:8080`

Use JSON for POST requests:

`Content-Type: application/json`

---

## Runtime Configuration (Environment Variables)

All env vars are optional.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANAFI_IP` | `192.168.42.1` | Drone IP address |
| `TELEMETRY_HZ` | `2.0` | Telemetry polling rate in Hz |
| `KEY_STALE_S` | `1.0` | Key auto-release timeout (seconds) |
| `SAFE_TAKEOFF_S` | `3.0` | Post-takeoff hover hold time (seconds) |
| `SAFE_TAKEOFF_DEFAULT` | `0` | Enable safe takeoff by default (`1`/`true`) |
| `MAX_ALTITUDE_M` | `2.0` | Hard altitude ceiling (meters) |
| `MAX_VERTICAL_SPEED` | `0.5` | Max vertical speed (m/s) |
| `MAX_TILT` | `15` | Max tilt angle (degrees), limits horizontal speed |
| `API_COMMAND_LOG` | `1` | Enable command logging (`1`/`true`) |
| `API_COMMAND_LOG_PATH` | `controller_olympe/api_command_log.jsonl` | Command log file path |

---

## Endpoints

### Health Check

#### `GET /`

Returns server status.

**Response:**
```json
{"ok": true, "service": "olympe_pi_api_server", "connected": true}
```

---

### Key-Based Control

Keys are translated to PCMD frames at 20 Hz. The same WASD/RFEQ mapping as the Tello server.

| Key | Action |
|-----|--------|
| W / S | Forward / Backward (pitch) |
| A / D | Left / Right (roll) |
| R / F | Up / Down (gaz) |
| Q / E | Rotate CCW / CW (yaw) |
| T | Takeoff |
| L | Land |
| Space / X | Zero all axes (emergency stop) |

#### `POST /api/key_down`

Register a key press. Must send keepalive every < `KEY_STALE_S` seconds.

**Body:** `{"key": "w"}`
**Response:** `{"ok": true}`

#### `POST /api/key_up`

Release a key.

**Body:** `{"key": "w"}`
**Response:** `{"ok": true}`

---

### Flight Actions

#### `POST /api/takeoff`

Take off. Blocked during takeoff cooldown.

**Response:** `{"ok": true, "flying": true, "safe_takeoff": false}`

#### `POST /api/land`

Land the drone. Retries up to 3 times.

**Response:** `{"ok": true, "flying": false}`

#### `POST /api/flip`

Execute a flip animation. Requires battery >= 50% and flying state.

**Body:** `{"dir": "f"}` — Accepts: `l`, `r`, `f`, `b`, `left`, `right`, `front`, `back`

**Response:** `{"ok": true, "dir": "f"}`

**Error codes:** `409` if blocked (battery, not flying, cooldown).

#### `POST /api/emergency`

Emergency motor cutoff. Falls immediately.

**Response:** `{"ok": true}`

---

### Movement Commands

#### `POST /api/move`

Move in a cardinal direction by distance in cm. Translates to Olympe `moveBy`.

**Body:**
```json
{"dir": "forward", "cm": 50}
```

- `dir`: `up`, `down`, `left`, `right`, `forward`, `back`
- `cm`: 20–500 (default: 20)

**Response:** `{"ok": true, "dir": "forward", "cm": 50}`

#### `POST /api/rotate`

Rotate by degrees. Translates to Olympe `moveBy` with dPsi.

**Body:**
```json
{"dir": "cw", "deg": 90}
```

- `dir`: `cw` (clockwise), `ccw` (counter-clockwise)
- `deg`: 1–360 (default: 45)

**Response:** `{"ok": true, "dir": "cw", "deg": 90}`

#### `POST /api/go`

Move to a relative XYZ position in cm. Tello convention: Z positive = up.

**Body:**
```json
{"x": 100, "y": 0, "z": 50}
```

- `x`: forward(+)/back(-) in cm
- `y`: right(+)/left(-) in cm
- `z`: up(+)/down(-) in cm

**Response:** `{"ok": true, "x": 100, "y": 0, "z": 50}`

#### `POST /api/rc`

RC override: set PCMD values directly for a duration.

**Body:**
```json
{"lr": 0, "fb": 50, "ud": 0, "yaw": 0, "duration_ms": 500}
```

- `lr`, `fb`, `ud`, `yaw`: -100 to 100
- `duration_ms`: 50–2000 (default: 250)

**Response:** `{"ok": true, "rc": {"lr": 0, "fb": 50, "ud": 0, "yaw": 0}, "duration_ms": 500}`

---

### GPS Navigation

#### `POST /api/moveto`

Move to absolute GPS coordinates. Requires GPS fix.

**Body:**
```json
{"lat": 48.8566, "lon": 2.3522, "alt": 5.0, "heading": 0}
```

- `lat`, `lon`: required (decimal degrees)
- `alt`: altitude in meters (default: 2.0)
- `heading`: heading in degrees (default: 0)

**Response:** `{"ok": true, "lat": 48.8566, "lon": 2.3522, "alt": 5.0}`

#### `POST /api/rth`

Start or cancel Return-To-Home.

**Body:** `{"action": "start"}` or `{"action": "cancel"}`

**Response:** `{"ok": true, "action": "start"}`

---

### Camera

#### `POST /api/camera/photo`

Take a single photo.

**Response:** `{"ok": true}`

#### `POST /api/camera/record/start`

Start video recording.

**Response:** `{"ok": true, "recording": true}`

#### `POST /api/camera/record/stop`

Stop video recording.

**Response:** `{"ok": true, "recording": false}`

#### `POST /api/gimbal`

Set gimbal tilt and pan in degrees.

**Body:**
```json
{"tilt": -45, "pan": 0}
```

- `tilt`: -90 (straight down) to +30 (slightly up) degrees
- `pan`: -180 to +180 degrees (default: 0)

**Response:** `{"ok": true, "tilt": -45, "pan": 0}`

---

### Telemetry

#### `GET /api/telemetry`

Returns all telemetry data. No-cache headers set.

**Response:**
```json
{
  "battery": 85,
  "height_cm": 150.0,
  "pitch": 1.23,
  "roll": -0.45,
  "yaw": 90.1,
  "vgx": 12.3,
  "vgy": -5.0,
  "vgz": 0.0,
  "speed": 13.3,
  "gps_lat": 48.8566,
  "gps_lon": 2.3522,
  "gps_alt": 45.2,
  "gimbal_pitch": -45.0,
  "gimbal_roll": 0.0,
  "gimbal_yaw": 0.0,
  "flying": true,
  "connected": true,
  "state_age_s": 0.312,
  "state_fresh": true,
  "updated_at": 1711360000.0
}
```

**Fields:**
| Field | Unit | Source |
|-------|------|--------|
| `battery` | % | `BatteryStateChanged` |
| `height_cm` | cm | `AltitudeChanged` (m→cm) |
| `pitch`, `roll`, `yaw` | degrees | `AttitudeChanged` (rad→deg) |
| `vgx`, `vgy`, `vgz` | cm/s | `SpeedChanged` (m/s→cm/s) |
| `speed` | cm/s | Aggregate √(vgx²+vgy²+vgz²) |
| `gps_lat`, `gps_lon` | decimal degrees | `GpsLocationChanged` |
| `gps_alt` | meters | `GpsLocationChanged` |
| `gimbal_pitch/roll/yaw` | degrees | Gimbal attitude |
| `flying` | boolean | Derived from `FlyingStateChanged` |
| `connected` | boolean | Connection state |
| `state_age_s` | seconds | Time since last telemetry update |
| `state_fresh` | boolean | True if `state_age_s` ≤ 2.0 |

#### `GET /api/telemetry/stream`

Server-Sent Events (SSE) stream. Pushes telemetry JSON every 400ms.

```
data: {"battery": 85, "height_cm": 150, ...}

data: {"battery": 85, "height_cm": 148, ...}
```

---

### Flight Settings

#### `GET /api/settings`

Returns current flight limit settings and feature availability.

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
  "moveto_available": true
}
```

#### `POST /api/settings`

Update flight limits. All fields optional — only provided fields are changed.

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

**Response:** `{"ok": true, "max_altitude_m": 3.0, "max_tilt": 20, ...}`

---

### Safety

#### `GET /api/safety/takeoff`

Get safe takeoff status.

**Response:** `{"enabled": false, "hold_s": 3.0}`

#### `POST /api/safety/takeoff`

Toggle safe takeoff (holds hover for `hold_s` seconds post-takeoff).

**Body:** `{"enabled": true}`

**Response:** `{"ok": true, "enabled": true, "hold_s": 3.0}`

---

### Recovery

#### `POST /api/recover`

Disconnect and reconnect to the drone. Clears all key presses, RC overrides, and resets flying state. Re-applies flight limits.

**Response:** `{"ok": true, "message": "recovered"}`

---

### Logging

#### `GET /api/logging/commands`

Command log status.

#### `GET /api/logging/commands/download`

Download command log as JSONL.

#### `POST /api/logging/commands/clear`

Clear command log file.

#### `GET /api/logging/telemetry`

Telemetry log status.

#### `POST /api/logging/telemetry`

Configure telemetry logging.

**Body:** `{"enabled": true, "path": "/custom/path.jsonl"}`

#### `GET /api/logging/telemetry/download`

Download telemetry log as JSONL.

#### `POST /api/logging/telemetry/clear`

Clear telemetry log file.

---

## Safety Features

### Altitude Fence
- **Hardware limit**: `MaxAltitude` set on drone connect (default 2m via `MAX_ALTITUDE_M`)
- **Software fence**: RC loop zeros upward gaz when `height_cm >= MAX_ALTITUDE_M * 100`
- Configurable via env var or `POST /api/settings`

### Speed Limits
- **Max vertical speed**: 0.5 m/s default (via `MaxVerticalSpeed`)
- **Max tilt**: 15° default (limits horizontal speed via `MaxTilt`)
- **Max yaw speed**: 150°/s default (via `MaxRotationSpeed`)

### Takeoff Cooldown
- After takeoff, a discrete command window blocks RC input for `SAFE_TAKEOFF_S` seconds
- Prevents accidental flip commands immediately after takeoff

### Stale Key Timeout
- Keys auto-release after `KEY_STALE_S` seconds without keepalive
- Prevents runaway movement if the controller disconnects

### Auto-Reconnect
- Background loop retries connection every 3 seconds when disconnected
- Flight limits are re-applied on every reconnect

---

## Architecture

```
Remote Web Controller (port 8090)
        │
        │  /proxy/*  →  /api/*
        ▼
  Olympe Pi API Server (port 8080)
        │
        │  Olympe SDK (PCMD @ 20Hz, commands, telemetry)
        ▼
    Parrot Anafi Drone (192.168.42.1)
```

### Background Threads
1. **Telemetry loop** – Polls battery, attitude, speed, altitude, GPS, gimbal at `TELEMETRY_HZ`
2. **Reconnect loop** – Monitors connection, retries every 3s
3. **RC/PCMD loop** – Sends piloting commands at 20 Hz from key state + RC override
