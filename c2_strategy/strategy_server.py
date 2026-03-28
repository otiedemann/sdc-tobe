"""
FastAPI server for the SDC26 Strategic C2 system.

Runs on port 9090. Provides:
- REST API for configuration, manual overrides, and status
- WebSocket hub for real-time state broadcasting to the dashboard
- Background tasks: strategy engine loop, telemetry polling, state broadcast
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .arena_config import ArenaConfig, default_arena_config, load_arena_config, save_arena_config
from .aruco_locator import ArucoLocator
from .drone_client import FleetClient
from .models import (
    ArenaConfigUpdate,
    AssignTargetCommand,
    DroneConfigItem,
    ManualOverrideCommand,
    SetPhaseCommand,
)
from .scoring import ScoreTracker
from .strategy import StrategyEngine

log = logging.getLogger("strategy_server")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------

config: ArenaConfig = default_arena_config()
fleet: FleetClient = FleetClient()
locator: ArucoLocator = ArucoLocator()
scorer: ScoreTracker = ScoreTracker(config=config)
engine: StrategyEngine = StrategyEngine(config, fleet, locator, scorer)

ws_clients: set[WebSocket] = set()
broadcast_task: Optional[asyncio.Task] = None
aruco_task: Optional[asyncio.Task] = None

WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, fleet, locator, scorer, engine, broadcast_task, aruco_task

    # Load config
    config = load_arena_config()
    fleet = FleetClient()
    locator = ArucoLocator()
    scorer = ScoreTracker(config=config)
    engine = StrategyEngine(config, fleet, locator, scorer)

    # Initialize fleet from config
    engine.init_fleet_from_config()

    # Start ArUco UDP listener
    aruco_task = asyncio.ensure_future(locator.start_udp_listener())

    # Start state broadcast loop
    broadcast_task = asyncio.ensure_future(_broadcast_loop())

    log.info(f"C2 Strategy server started with {len(config.drone_configs)} drones")
    yield

    # Cleanup
    engine.stop()
    if broadcast_task:
        broadcast_task.cancel()
    if aruco_task:
        aruco_task.cancel()
    await locator.stop()
    await fleet.close_all()


app = FastAPI(title="SDC26 Strategy C2", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static files (web dashboard)
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "SDC26 Strategy C2 Server", "status": "running"}


# ---------------------------------------------------------------------------
# WebSocket: real-time state
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        # Send initial state
        await ws.send_json(engine.get_state_snapshot())
        # Keep alive and receive commands
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                await _handle_ws_command(msg, ws)
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


async def _handle_ws_command(msg: dict, ws: WebSocket) -> None:
    """Handle commands from dashboard WebSocket."""
    cmd = msg.get("cmd")

    if cmd == "set_phase":
        engine.set_phase(msg.get("phase", ""))
    elif cmd == "start_engine":
        engine.start()
    elif cmd == "stop_engine":
        engine.stop()
    elif cmd == "takeoff":
        drone_id = msg.get("drone_id")
        if drone_id:
            result = await engine.manual_command(drone_id, "takeoff", {})
            await ws.send_json({"cmd_result": result})
    elif cmd == "land":
        drone_id = msg.get("drone_id")
        if drone_id:
            result = await engine.manual_command(drone_id, "land", {})
            await ws.send_json({"cmd_result": result})
    elif cmd == "emergency":
        drone_id = msg.get("drone_id")
        if drone_id:
            result = await engine.manual_command(drone_id, "emergency", {})
            await ws.send_json({"cmd_result": result})
    elif cmd == "emergency_all":
        for did in engine.state.drones:
            await engine.manual_command(did, "emergency", {})
    elif cmd == "takeoff_all":
        for did in engine.state.drones:
            await engine.manual_command(did, "takeoff", {})
    elif cmd == "land_all":
        for did in engine.state.drones:
            await engine.manual_command(did, "land", {})
    elif cmd == "goto":
        drone_id = msg.get("drone_id")
        if drone_id:
            result = await engine.manual_command(
                drone_id, "goto",
                {"x": msg.get("x", 0), "y": msg.get("y", 0), "z": msg.get("z", 1.5)},
            )
            await ws.send_json({"cmd_result": result})
    elif cmd == "assign_target":
        engine.assign_target(msg.get("drone_id", ""), msg.get("box_id", ""))


async def _broadcast_loop() -> None:
    """Broadcast game state to all connected WebSocket clients."""
    while True:
        try:
            if ws_clients:
                snapshot = engine.get_state_snapshot()
                dead: list[WebSocket] = []
                for ws in ws_clients:
                    try:
                        await ws.send_json(snapshot)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    ws_clients.discard(ws)
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# REST API: Status
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_state():
    return engine.get_state_snapshot()


@app.get("/api/scoring")
async def get_scoring():
    return scorer.summary()


@app.get("/api/locator")
async def get_locator():
    return locator.summary()


# ---------------------------------------------------------------------------
# REST API: Strategy control
# ---------------------------------------------------------------------------

@app.post("/api/phase")
async def set_phase(cmd: SetPhaseCommand):
    ok = engine.set_phase(cmd.phase)
    return {"ok": ok, "phase": engine.state.phase.value}


@app.post("/api/engine/start")
async def start_engine():
    engine.start()
    return {"ok": True, "running": True}


@app.post("/api/engine/stop")
async def stop_engine():
    engine.stop()
    return {"ok": True, "running": False}


@app.post("/api/assign")
async def assign_target(cmd: AssignTargetCommand):
    ok = engine.assign_target(cmd.drone_id, cmd.box_id)
    return {"ok": ok}


# ---------------------------------------------------------------------------
# REST API: Manual drone control
# ---------------------------------------------------------------------------

@app.post("/api/drone/command")
async def drone_command(cmd: ManualOverrideCommand):
    result = await engine.manual_command(cmd.drone_id, cmd.action, cmd.params)
    return result


@app.post("/api/drone/{drone_id}/takeoff")
async def drone_takeoff(drone_id: str):
    return await engine.manual_command(drone_id, "takeoff", {})


@app.post("/api/drone/{drone_id}/land")
async def drone_land(drone_id: str):
    return await engine.manual_command(drone_id, "land", {})


@app.post("/api/drone/{drone_id}/emergency")
async def drone_emergency(drone_id: str):
    return await engine.manual_command(drone_id, "emergency", {})


@app.post("/api/fleet/takeoff")
async def fleet_takeoff():
    results = {}
    for did in engine.state.drones:
        results[did] = await engine.manual_command(did, "takeoff", {})
    return results


@app.post("/api/fleet/land")
async def fleet_land():
    results = {}
    for did in engine.state.drones:
        results[did] = await engine.manual_command(did, "land", {})
    return results


@app.post("/api/fleet/emergency")
async def fleet_emergency():
    results = {}
    for did in engine.state.drones:
        results[did] = await engine.manual_command(did, "emergency", {})
    return results


# ---------------------------------------------------------------------------
# REST API: Arena configuration
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    return config.model_dump()


@app.post("/api/config")
async def update_config(update: ArenaConfigUpdate):
    global config, scorer
    data = config.model_dump()
    for k, v in update.model_dump(exclude_none=True).items():
        data[k] = v
    config = ArenaConfig(**data)
    scorer = ScoreTracker(config=config)
    engine.config = config
    engine.scorer = scorer
    save_arena_config(config)
    return {"ok": True}


@app.post("/api/config/drones")
async def update_drones(drones: list[DroneConfigItem]):
    global config
    config = config.model_copy(update={"drone_configs": drones})
    save_arena_config(config)

    # Rebuild fleet
    await fleet.close_all()
    fleet.drones.clear()
    fleet._locks.clear()
    engine.state.drones.clear()
    engine.init_fleet_from_config()

    return {"ok": True, "drone_count": len(drones)}


@app.post("/api/config/team")
async def set_team(data: dict):
    global config
    team = data.get("team", "blue")
    if team not in ("red", "blue"):
        return {"ok": False, "error": "team must be red or blue"}
    config = config.model_copy(update={"our_team": team})
    engine.state.our_team = team
    save_arena_config(config)
    return {"ok": True, "team": team}


# ---------------------------------------------------------------------------
# REST API: Target boxes (manual registration)
# ---------------------------------------------------------------------------

@app.post("/api/targets")
async def add_target(data: dict):
    """Manually register a target box."""
    box_id = data.get("box_id", "")
    marker_id = data.get("marker_id")
    x = data.get("x", 10.0)
    y = data.get("y", 5.0)
    owner = data.get("owner", "red")

    if not box_id:
        return {"ok": False, "error": "box_id required"}

    locator.update_target_from_sim(
        box_id, marker_id or 0, float(x), float(y)
    )
    box = locator.target_boxes.get(box_id)
    if box:
        box.owner = owner
        engine.state.target_boxes[box_id] = box

    return {"ok": True, "box_id": box_id}


# ---------------------------------------------------------------------------
# Run with: uvicorn c2_strategy.strategy_server:app --port 9090
# ---------------------------------------------------------------------------
