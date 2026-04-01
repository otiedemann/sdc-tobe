# Unified Pi API Server — API Reference

Auto-detects Tello or Anafi drone by IP.
Default port: **8080**

---

## Connection & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Connection state, drone type, battery |
| GET | `/api/telemetry` | Full telemetry snapshot |
| GET | `/api/telemetry/stream` | SSE stream of telemetry at ~2.5 Hz |

---

## Flight Control

| Method | Endpoint | Body / Params | Description |
|--------|----------|---------------|-------------|
| POST | `/api/takeoff` | — | Take off |
| POST | `/api/land` | — | Land |
| POST | `/api/emergency` | — | Cut motors immediately |
| POST | `/api/move` | `{direction, cm}` | Discrete move (forward/back/left/right/up/down) |
| POST | `/api/rotate` | `{direction, degrees}` | Rotate cw/ccw |
| POST | `/api/flip` | `{direction}` | Flip (Tello only) |
| POST | `/api/go_xyz` | `{x, y, z, speed}` | Go to relative XYZ (cm) |
| POST | `/api/moveto` | `{lat, lon, alt, heading}` | GPS move-to (Anafi only) |
| POST | `/api/rth` | `{action}` | Return-to-home: `start`/`cancel` (Anafi only) |

---

## RC / Keyboard Control

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/api/rc` | `{lr, fb, ud, yaw}` | Raw RC (-100…100 per axis) |
| POST | `/api/keys/down` | `{key}` | Key pressed |
| POST | `/api/keys/up` | `{key}` | Key released |
| POST | `/api/keys/heartbeat` | `{keys[]}` | Keep keys alive |
| POST | `/api/rc/override` | `{lr,fb,ud,yaw,duration_ms}` | Timed RC override |

---

## Video Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/video/start` | Start stream. Body: `{mode: "mjpeg"}` or `{mode:"forward", target_host, target_port}` |
| POST | `/api/video/stop` | Stop video |
| GET | `/api/video` | MJPEG stream (when started in mjpeg mode) |
| GET | `/api/video/status` | Current video mode |

---

## Camera & Gimbal (Anafi only)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/camera/photo` | Take a photo |
| POST | `/api/camera/record/start` | Start video recording |
| POST | `/api/camera/record/stop` | Stop video recording |
| POST | `/api/gimbal` | Body: `{tilt, pan}` — set gimbal angles |

---

## Settings / Flight Limits (Anafi)

Limits are applied to the drone via Olympe and **persisted** to `flight_config.json` so they survive restarts.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings` | Get current limits |
| POST | `/api/settings` | Set limits (see fields below) |

**POST `/api/settings` fields:**

```json
{
  "max_altitude_m":      2.0,
  "max_vertical_speed":  0.5,
  "max_tilt":           15.0,
  "max_yaw_speed":      150,
  "geofence_distance":  50,
  "geofence_enabled":   true
}
```

---

## Safety

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/safety/takeoff` | Get safe-takeoff state (`enabled`, `hold_s`) |
| POST | `/api/safety/takeoff` | Body: `{enabled: bool}` |

---

## Logging

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/logging/commands` | Command log status |
| GET | `/api/logging/commands/download` | Download command log (NDJSON) |
| POST | `/api/logging/commands/clear` | Clear command log |
| GET | `/api/logging/telemetry` | Telemetry log status |
| POST | `/api/logging/telemetry` | Body: `{enabled: bool, path: str}` |
| GET | `/api/logging/telemetry/download` | Download telemetry log (NDJSON) |
| POST | `/api/logging/telemetry/clear` | Clear telemetry log |

---

## ArUco Positioning (Anafi only)

Requires `HeadlessAruCoPositioning` from `aruco-position/control-unit/pi_position.py`.
Frames are tapped directly from the existing Olympe video callback — **no second SDK connection**.
Config is persisted to `position_config.json`; calibration to `position_calib.npz`.

### Position State

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/position` | Current position snapshot |
| GET | `/api/position/events` | SSE stream of position updates |
| GET | `/api/position/video` | MJPEG stream with ArUco annotations |

**Position snapshot fields:**

```json
{
  "x": 3.14, "y": 1.5, "z": -1.0,
  "dx": 0.0, "dy": 1.0,
  "vx": 0.0, "vy": 0.0, "vz": 0.0,
  "markers": {"0": 0.92, "3": 0.45},
  "ref_markers": [0, 3],
  "stale": false,
  "ts": 1712345678.123
}
```

### Positioning Config

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/position/config` | Get config + calibration status |
| POST | `/api/position/config` | Update config (persisted) |

**POST `/api/position/config` fields:**

```json
{
  "enabled":         false,
  "fps":             5,
  "detect_profile":  "default",
  "latency_comp_s":  0.05
}
```

### Calibration

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/position/calibration` | Upload `.npz` file (`camera_matrix`, `dist_coeffs`) |
| GET | `/api/position/calibration` | Download current calibration `.npz` |

---

## Arena Configuration

Stored in `arena_config.json` alongside the server. Defines marker layout used by the positioning subsystem.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/arena/config` | Get current arena config |
| POST | `/api/arena/config` | Update arena config (persisted) |
| POST | `/api/arena/config/reset` | Reset to defaults |

**Arena config fields:**

```json
{
  "arena_width_m":  10.0,
  "arena_height_m": 10.0,
  "marker_size_m":  0.5,
  "markers": [
    {"id": 0, "x": 0.0, "y": 0.0, "z": 0.0, "wall": "north"},
    {"id": 1, "x": 5.0, "y": 0.0, "z": 0.0, "wall": "south"}
  ]
}
```

**Wall values:** `north`, `south`, `east`, `west`, `floor`, `ceiling`

---

## Config Files

All config files are stored alongside the server script:

| File | Purpose |
|------|---------|
| `flight_config.json` | Persisted flight limits (altitude, speed, tilt, yaw) |
| `position_config.json` | Persisted positioning settings (enabled, fps, profile) |
| `position_calib.npz` | Camera calibration (NumPy: `camera_matrix`, `dist_coeffs`) |
| `arena_config.json` | Arena dimensions and marker layout |
| `tello_wifi_config.json` | Tello Wi-Fi credentials |
| `api_command_log.jsonl` | Command log (NDJSON) |
| `telemetry_log.jsonl` | Telemetry log (NDJSON) |

---

## Capabilities

GET `/api/status` includes a `capabilities` object listing supported features per drone type.

| Capability | Tello | Anafi |
|-----------|-------|-------|
| flip | ✓ | — |
| gimbal | — | ✓ |
| gps | — | ✓ |
| rth | — | ✓ |
| moveto | — | ✓ |
| geofence | — | ✓ |
| positioning | — | ✓ |
| video_mjpeg | — | ✓ |
| video_forward | ✓ | ✓ |
