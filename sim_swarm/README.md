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
