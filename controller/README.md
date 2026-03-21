# Controller Quickstart

## 1) Setup Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install djitellopy flask requests
```

## 2) Wi‑Fi

- Power on Tello
- Connect host (Pi) to Tello SSID (or use configured `tello_wifi_config.json` auto-connect loop)

## 3) Run Pi API server

```bash
python3 tello_pi_api_server.py
```

Server listens on `http://0.0.0.0:8080`.

## 4) Optional runtime env vars

```bash
TELEMETRY_HZ=2.0 KEY_STALE_S=1.0 python3 tello_pi_api_server.py
```

- `TELEMETRY_HZ` controls telemetry/log write loop rate (default `2.0`)
- `KEY_STALE_S` controls stale key auto-release timeout (default `1.0s`)

## 5) Remote web controller

```bash
python3 tello_remote_web_controller.py
```

Default UI: `http://0.0.0.0:8090` (proxies to Pi API base URL).

---

## Full API + telemetry/logging docs

See: [`API.md`](./API.md)
