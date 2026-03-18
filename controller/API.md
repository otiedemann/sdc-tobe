# Tello Control API (Pi Server)

This documents the API exposed by `controller/tello_pi_api_server.py` and the key runtime environment variables.

## Base URL

`http://<PI_IP>:8080`

Use JSON for POST requests:

`Content-Type: application/json`

---

## Runtime configuration (Environment Variables)

All env vars are optional.

- `TELEMETRY_HZ` (float, default: `2.0`)
  - Telemetry polling/logging loop rate in Hz.
  - Example: `TELEMETRY_HZ=5` for 5 Hz.
- `KEY_STALE_S` (float, default: `1.0`)
  - Stale-key timeout (deadman release). If key-down keepalive stops for longer than this timeout, key is auto-released server-side.

### Fixed in code (not env-configurable currently)
- `TELLO_HOST = "192.168.10.1"`
- `HTTP_HOST = "0.0.0.0"`
- `HTTP_PORT = 8080`
- `RC_HZ = 20`

---

## 1) Health

### `GET /`
Service check.

**Response**
```json
{ "ok": true, "service": "tello_pi_api_server" }
```

---

## 2) Key-style Control (Keyboard compatible)

### `POST /api/key_down`
Press/hold a key in the controller state machine.

**Body**
```json
{ "key": "w" }
```

### `POST /api/key_up`
Release a key in the controller state machine.

**Body**
```json
{ "key": "w" }
```

### Supported keys
- Movement: `w a s d r f q e`
- Actions: `t` (takeoff), `l` (land), `x` / `space` (stop)

### Reliability behavior
- Remote UI sends held-key keepalive updates.
- Server uses stale-key timeout (`KEY_STALE_S`) to auto-release stuck keys.

---

## 3) High-level Flight Actions

### `POST /api/takeoff`
Take off if not already flying.

**Success**
```json
{ "ok": true, "flying": true }
```

### `POST /api/land`
Land if currently flying.

**Success**
```json
{ "ok": true, "flying": false }
```

### `POST /api/flip`
Execute a flip maneuver.

**Body**
```json
{ "dir": "l" }
```

Allowed `dir`: `l`, `r`, `f`, `b`

**Success**
```json
{ "ok": true, "dir": "l" }
```

### Flip guardrails
Flip can return `409` with explicit safety errors:
- `flip_blocked_takeoff_cooldown`
- `flip_requires_flying`
- `flip_requires_battery_50_plus`
- `flip_requires_low_horizontal_speed`

---

## 4) Direct RC Override

### `POST /api/rc`
Applies a temporary RC command override.

**Body**
```json
{
  "lr": 0,
  "fb": 40,
  "ud": 0,
  "yaw": 0,
  "duration_ms": 300
}
```

### Notes
- `lr`, `fb`, `ud`, `yaw` are clamped to `[-100..100]`
- `duration_ms` is clamped to `[50..2000]`
- After timeout, control returns to key-driven behavior.

**Success**
```json
{
  "ok": true,
  "rc": { "lr": 0, "fb": 40, "ud": 0, "yaw": 0 },
  "duration_ms": 300
}
```

---

## 5) Recovery

### `POST /api/recover`
Triggers crash/disconnect recovery flow (neutral RC, try land, reconnect, restore control).

**Response**
```json
{ "ok": true, "message": "recovered" }
```

---

## 6) Telemetry

### `GET /api/telemetry`
Returns latest telemetry snapshot.

**Typical response**
```json
{
  "battery": 78,
  "temperature": 63,
  "height_cm": 42,
  "tof_cm": 45,
  "barometer_cm": 1023,
  "flight_time_s": 38,
  "wifi_snr": -56,
  "pitch": 0,
  "roll": 1,
  "yaw": 90,
  "vgx": 0,
  "vgy": 0,
  "vgz": 0,
  "agx": -2,
  "agy": 1,
  "agz": 1000,
  "flying": true,
  "connected": true,
  "updated_at": 1773750000.123
}
```

### `GET /api/telemetry/stream`
Server-Sent Events (SSE) stream with continuous telemetry updates.

**Event format**
```text
data: { ...telemetry json... }

```

---

## 7) Telemetry Logging

### `GET /api/logging/telemetry`
Read telemetry logging config.

**Response**
```json
{ "enabled": false, "path": "/.../telemetry_log.jsonl" }
```

### `POST /api/logging/telemetry`
Update telemetry logging config.

**Body example**
```json
{ "enabled": true, "path": "/tmp/telemetry_log.jsonl" }
```

### `GET /api/logging/telemetry/download`
Download current telemetry log file (`jsonl`/`ndjson`).

### `POST /api/logging/telemetry/clear`
Truncate/clear telemetry log file.

**Response**
```json
{ "ok": true, "cleared": true, "path": "/.../telemetry_log.jsonl" }
```

---

## 8) Quick `curl` Examples

```bash
# takeoff
curl -X POST http://<PI_IP>:8080/api/takeoff

# land
curl -X POST http://<PI_IP>:8080/api/land

# flip right
curl -X POST http://<PI_IP>:8080/api/flip \
  -H "Content-Type: application/json" \
  -d '{"dir":"r"}'

# short forward movement (300ms)
curl -X POST http://<PI_IP>:8080/api/rc \
  -H "Content-Type: application/json" \
  -d '{"lr":0,"fb":40,"ud":0,"yaw":0,"duration_ms":300}'

# telemetry snapshot
curl http://<PI_IP>:8080/api/telemetry

# enable telemetry logging
curl -X POST http://<PI_IP>:8080/api/logging/telemetry \
  -H "Content-Type: application/json" \
  -d '{"enabled":true}'

# clear telemetry log
curl -X POST http://<PI_IP>:8080/api/logging/telemetry/clear

# recover
curl -X POST http://<PI_IP>:8080/api/recover
```
