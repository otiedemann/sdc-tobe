# Swarm Simulation MVP

Leichtgewichtige 2D-Simulationsumgebung (ohne externe Abhängigkeiten) für:
- Missionslogik
- Schwarmkoordination
- Kommunikationsstörungen
- Safety-Regeln
- Logging & Metrics

## Start

```bash
python3 sim_swarm/run.py --scenario baseline
python3 sim_swarm/run.py --scenario degraded_link
python3 sim_swarm/run.py --scenario dropout
```

## Szenarien
- `baseline`: 2 Drones, 1 Ziel, gute Funkverbindung
- `degraded_link`: höhere Latenz + Paketverlust
- `dropout`: eine Drone fällt während der Mission aus

## Metriken
- mission_success
- completion_time_s
- near_collisions
- link_loss_events
- avg_battery_pct
- events_logged

## Struktur
- `run.py` Einstiegspunkt
- `sim.py` Simulationsschleife
- `models.py` Welt/Drone-Zustände
- `mission.py` Rollen/Missionslogik
- `comms.py` Latenz/Verlust-Modell
- `scenarios.py` Konfigurationen

## Web GUI Viewer

1. Szenarien erzeugen:
```bash
python3 sim_swarm/run.py --scenario baseline
python3 sim_swarm/run.py --scenario degraded_link
python3 sim_swarm/run.py --scenario dropout
```

2. Webserver starten:
```bash
./sim_swarm/serve_viewer.sh
```

3. Browser öffnen:
- http://localhost:8765/web/

Dort kannst du Timeline + Drone-States abspielen.

## 3D Arena Simulator (Web)

A simple 3D arena with 2–5 drones and two control modes:
- Manual (keyboard control of selected drone)
- Auto Swarm (basic cohesion + separation + target/home behavior)

Start web server:
```bash
./sim_swarm/serve_viewer.sh
```
Open:
- http://localhost:8765/web3d/

> If official SDC arena dimensions differ, set exact X/Y/Z values in the UI.

---

## API-compatible simulator server (5 drone endpoints)

`sim_api_server.py` provides a Tello-API-compatible backend for controller testing without real drones.

It starts **5 endpoints on separate ports** (default):
- Drone 1 → `http://localhost:8081`
- Drone 2 → `http://localhost:8082`
- Drone 3 → `http://localhost:8083`
- Drone 4 → `http://localhost:8084`
- Drone 5 → `http://localhost:8085`

### Start

```bash
python3 sim_swarm/sim_api_server.py
```

Optional:
```bash
python3 sim_swarm/sim_api_server.py --host 0.0.0.0 --base-port 8081 --drones 5
```

### API compatibility

Implemented endpoints mirror the Pi API shape, including:
- `GET /`
- `POST /api/key_down`
- `POST /api/key_up`
- `POST /api/takeoff`
- `POST /api/land`
- `POST /api/flip`
- `POST /api/rc`
- `POST /api/recover`
- `GET /api/telemetry`
- `GET /api/telemetry/stream`
- `GET/POST /api/logging/telemetry`
- `GET /api/logging/telemetry/download`
- `POST /api/logging/telemetry/clear`
- `GET/POST /api/safety/takeoff` (compatibility no-op)

### Notes
- Each port controls one simulated drone instance.
- No real drone SDK calls are made.
- Intended as drop-in target for your existing remote controller/testing scripts.
