# Drone Simulation (SDC26)

Simulator for Tello-like control API (`API.md`) + UDP output compatible with `pi_position.py` consumers.

## What it does

- Exposes HTTP API on `:8080` with core endpoints:
    - `POST /api/key_down`, `POST /api/key_up`
    - `POST /api/takeoff`, `POST /api/land`
    - `POST /api/rc`, `POST /api/recover`, `POST /api/flip`
    - `GET /api/telemetry`
- Emits UDP JSON stream (`cam`, `dir`, `targets`, `debug`) to dashboard/relay (default `127.0.0.1:5005`).

## Config format

See `sim_config.example.json`.

- `arena`: movement bounds for drone camera position
- `drone_start`: initial position/yaw/flying state
- `targets`: static map of target IDs -> `[x,y,z]`

## Run

```bash
cd /home/openclaw/.openclaw/workspace/sdc26/drone-sim
python3 sim_server.py --config sim_config_sdc26.json --udp-ip 127.0.0.1 --udp-port 5005
```

Optional flags:

- `--http-host 0.0.0.0`
- `--http-port 8080`
- `--sim-hz 30`
- `--key-stale-s 1.0`

## Example control calls

```bash
curl -X POST http://127.0.0.1:8080/api/takeoff
curl -X POST http://127.0.0.1:8080/api/key_down -H 'Content-Type: application/json' -d '{"key":"w"}'
sleep 0.6
curl -X POST http://127.0.0.1:8080/api/key_up -H 'Content-Type: application/json' -d '{"key":"w"}'
curl -X POST http://127.0.0.1:8080/api/land
```

## UDP payload shape

```json
{
  "cam": [
    0.0,
    1.2,
    1.2
  ],
  "dir": [
    1.0,
    0.0,
    0.0
  ],
  "targets": {
    "60": [
      2.0,
      3.0,
      -1.5
    ],
    "70": [
      4.0,
      7.0,
      -1.5
    ]
  },
  "debug": false
}
```
