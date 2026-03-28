# SDC26 Strategic C2 Server

Autonomous drone fleet command & control for the SDC26 competition. Controls drones via the API, receives ArUco positioning data, and executes game strategies (simultaneous capture, team return bonus, instant win).

## Quick Start

```bash
pip install -r c2_strategy/requirements.txt

# From the project root:
cd /path/to/sdc-tobe
python -m c2_strategy
```

Open **http://localhost:9090** in your browser.

## Prerequisites

- Python 3.11+
- A running drone API server:
  - **Simulator**: `sim_swarm_API/sim_api_server.py` on port 8080
  - **Tello**: `tello_pi_api_server.py` per drone
  - **Anafi**: `olympe_pi_api_server.py` per drone
- (Optional) ArUco positioning: `aruco-position/c2-server/relay.py` on port 8000

## Configuration

All configuration is done through the **web dashboard** at http://localhost:9090.

### Team Selection

In the **Configuration** section, select your team color (red/blue). This determines which home zone is yours and which boxes count as enemy targets.

### Drone Fleet

In the **Configuration → Drone Fleet** section:

1. Click **+ Add Drone** for each drone
2. Fill in:
   - **ID** — unique name (e.g. `drone-1`)
   - **API URL** — the drone's API server (e.g. `http://localhost:8080`)
   - **Type** — `Simulator`, `Tello`, or `Anafi`
   - **Sim ID** — for simulator drones only, which sim drone to control (e.g. `B1`, `B2`, `B3`)
3. Click **Save Fleet Config**

### Target Boxes

Targets are discovered automatically via ArUco detection. To add them manually (e.g. for testing):

1. In the **Target Boxes** section, enter the box ID (ArUco marker number), X/Y position, and owner
2. Click **Add Target**

### Arena Dimensions

Default arena is 20m × 10m with standard home zones (Red: 0–7m, Blue: 13–20m). Adjustable in the Configuration section.

## Dashboard Features

### Arena Map
Top-down canvas showing drone positions (colored triangles), target boxes (colored squares), home zones, and assignment lines. Updates in real-time via WebSocket.

### Drone Cards
Per-drone status showing role, phase, position, altitude, heading, battery, and assigned target. Each card has Takeoff/Land/E-Stop buttons.

### Strategy Control
- **Start/Stop Engine** — enable/disable autonomous strategy execution
- **Phase Select** — manually override the game phase
- **Fleet Controls** — takeoff all, land all, emergency stop all

### Captures Log
Live feed of capture events with point totals.

## Strategy Phases

| Phase | Description |
|-------|-------------|
| **Setup** | Wait for all drones to connect, then auto-takeoff |
| **Scouting** | Fly into enemy zone to discover target boxes via ArUco |
| **Simultaneous Attack** | Assign drones to targets, fly there, capture simultaneously (10pt bonus) |
| **Team Return** | All drones return to home zone for special capture bonus (5pt per capture) |
| **Defend** | Orbit patrol near captured boxes |
| **Instant Win** | Rush remaining boxes when controlling 5+ (all 6 for 5s = win) |

The engine cycles: **Scouting → Attack → Return → repeat**, escalating to Instant Win when possible.

## Scoring (SDC26 v4.0)

| Action | Points |
|--------|--------|
| Normal capture (hover 2s over enemy box) | 1 |
| Special capture (team return after captures) | 5 per capture |
| Simultaneous capture (2+ boxes at once) | 10 bonus |
| Instant win (all 6 boxes for 5s) | Immediate win |

## API Endpoints

### Status
- `GET /api/state` — full game state snapshot
- `GET /api/scoring` — score summary
- `GET /api/locator` — ArUco positioning data

### Strategy Control
- `POST /api/engine/start` — start autonomous strategy
- `POST /api/engine/stop` — stop strategy engine
- `POST /api/phase` — `{"phase": "scouting"}` — set game phase
- `POST /api/assign` — `{"drone_id": "drone-1", "box_id": "30"}` — assign drone to target

### Drone Commands
- `POST /api/drone/{id}/takeoff`
- `POST /api/drone/{id}/land`
- `POST /api/drone/{id}/emergency`
- `POST /api/drone/command` — `{"drone_id": "...", "action": "goto", "params": {"x": 10, "y": 5}}`
- `POST /api/fleet/takeoff` / `land` / `emergency`

### Configuration
- `GET /api/config` — current config
- `POST /api/config` — update arena settings
- `POST /api/config/drones` — replace drone fleet config
- `POST /api/config/team` — `{"team": "blue"}`

### Targets
- `POST /api/targets` — `{"box_id": "30", "marker_id": 30, "x": 15, "y": 5, "owner": "red"}`

### WebSocket
- `ws://localhost:9090/ws` — real-time state updates + command channel

## ArUco Integration

The server listens on **UDP port 5005** for position packets from `pi_position.py` nodes. Packet format:

```json
{
  "cam": [x, y, z],
  "dir": [dx, dy, dz],
  "targets": {"30": [x, y, z]},
  "ref_markers": [0, 5, 6],
  "stale": false
}
```

Coordinates are automatically transformed from pi_position world coords (centered origin, x: -10..10) to arena coords (x: 0..20) via `arena_x = world_x + 10`.

## Architecture

```
c2_strategy/
├── strategy_server.py   # FastAPI server (port 9090)
├── strategy.py          # Game state machine & phase logic
├── drone_client.py      # Async HTTP fleet client
├── aruco_locator.py     # ArUco position receiver (UDP/WS)
├── scoring.py           # Capture timers & score tracking
├── arena_config.py      # Arena geometry & marker config
├── models.py            # Pydantic data models
├── requirements.txt
└── web/
    ├── index.html       # Dashboard
    ├── app.js           # WebSocket client & arena renderer
    └── styles.css
```
