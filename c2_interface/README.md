# SDC Main Command & Control Interface (Python, Web)

This module provides a modular Python server for swarm command-and-control with a web GUI.

## Current MVP capabilities
- 20m x 10m arena rendered as 1m x 1m clickable grid (200 cells)
- 3 arena zones shown in UI:
  - Red home area (left)
  - Neutral area (middle)
  - Blue home area (right)
- Team control model:
  - One team only
  - Selectable number of drones: 2..5
- System mode selector:
  - `live` (Tello connect + live JPEG feed endpoints)
  - `simulator` (auto-starts `sim_swarm/serve_viewer.sh`, embeds web3d simulator, C2->sim command bridge via postMessage)
- Per-drone controls:
  - Click-to-send drone to a selected cell
  - "Target in sight" trigger
- Mission behavior:
  - Drone goes to commanded cell and enters SEARCHING
  - On target sighting, drone hovers 5s
  - Then automatically returns to home cell
- Targets editor:
  - Set up to 6 target cells directly on grid
  - Per-target color (red/blue)
  - Coordinate list shown in UI
- WebSocket live state updates
- Video panels per drone (live URL or simulator placeholder feed endpoint)

## Architecture
- `app/main.py` - FastAPI app + API + websocket
- `app/arena.py` - arena grid + zone mapping
- `app/models.py` - typed command/state models
- `app/state.py` - in-memory state store + listeners
- `app/mission.py` - mission state machine
- `app/aruco_nav.py` - Aruco localization integration boundary
- `app/drone_adapter.py` - real/sim drone command integration boundary
- `web/` - browser UI

## Run
```bash
cd c2_interface
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Open: `http://localhost:8080`

## Live mode (Tello) quick setup
1. Configure up to 5 drones in `c2_interface/live_drones.json`.
2. Start C2 UI, select mode `live`, click **Connect Live Drones**.
3. UI cards use `/api/live/video/{drone_id}` for current frames.

### Multi WiFi stick note (important)
For 2..5 real Tellos with same default IP (`192.168.10.1`), each WiFi stick should be isolated (network namespaces / dedicated routing) so each drone session is bound to its own adapter path. The current MVP contains connection/video scripts and manager, but final robust multi-NIC binding + command deconfliction should be added before competition flight.

## Integration notes (next)
1. Replace `ArucoNavigator.localize()` with your simulator/live Aruco pipeline.
2. Wire `DroneAdapter` movement commands to Aruco-corrected grid navigation for live drones.
3. Upgrade live feed to MJPEG/WebRTC streamer per drone process.
4. Add target detector callback to call `/api/aruco/target-detected/{id}` automatically when target ArUco marker is detected.

## Safety notes
- Keep hard geofence checks in adapter layer before movement execution.
- Add collision prevention (reservation map + deconfliction) before real-flight use.
- Maintain kill-switch and return-home override outside C2 UI.
