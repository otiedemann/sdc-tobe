import asyncio
import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .arena import ArenaGrid
from .aruco_nav import ArucoNavigator
from .drone_adapter import DroneAdapter
from .live_tello import LiveDroneConfig, LiveTelloManager
from .mission import MissionController
from .models import (
    DroneState,
    GotoCommand,
    GridPoint,
    RegisterDroneCommand,
    SetTargetsCommand,
    SystemConfigCommand,
    TargetCell,
    TargetSpottedCommand,
)
from .state import StateStore

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
SIM_DIR = BASE_DIR.parent / "sim_swarm"
LIVE_CFG = BASE_DIR / "live_drones.json"

app = FastAPI(title="SDC Command & Control")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

arena = ArenaGrid()
store = StateStore()
mission = MissionController(store=store, adapter=DroneAdapter(), aruco=ArucoNavigator())

system_mode = "live"  # live | simulator
team_drone_count = 2
targets: list[TargetCell] = []
sim_server: subprocess.Popen | None = None
live_manager = LiveTelloManager()


class WsHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        dead: list[WebSocket] = []
        for c in self.clients:
            try:
                await c.send_text(json.dumps(payload))
            except Exception:
                dead.append(c)
        for d in dead:
            self.disconnect(d)


hub = WsHub()


def simulator_url() -> str | None:
    return "http://localhost:8765/web3d/" if system_mode == "simulator" else None


def load_live_configs(count: int) -> list[LiveDroneConfig]:
    if not LIVE_CFG.exists():
        return [LiveDroneConfig(drone_id=f"team-drone-{i+1}") for i in range(count)]
    try:
        raw = json.loads(LIVE_CFG.read_text())
    except Exception:
        raw = {}
    items = raw.get("drones", []) if isinstance(raw, dict) else []
    cfgs: list[LiveDroneConfig] = []
    for i in range(min(count, len(items))):
        it = items[i]
        cfgs.append(LiveDroneConfig(drone_id=it.get("drone_id", f"team-drone-{i+1}"), host=it.get("host", "192.168.10.1")))
    while len(cfgs) < count:
        idx = len(cfgs)
        cfgs.append(LiveDroneConfig(drone_id=f"team-drone-{idx+1}"))
    return cfgs


async def ensure_team_drones(count: int) -> None:
    # One team only, selectable 2..5 drones.
    all_existing = await store.all_drones()
    keep_ids = {f"team-drone-{i+1}" for i in range(count)}

    # keep + create required drones
    for i in range(count):
        drone_id = f"team-drone-{i+1}"
        home = GridPoint(x=1, y=min(1 + i * 2, arena.height - 1))
        existing = next((d for d in all_existing if d.drone_id == drone_id), None)
        if system_mode == "simulator":
            video_url = f"/api/sim/video/{drone_id}"
        elif system_mode == "live":
            video_url = f"/api/live/video/{drone_id}"
        else:
            video_url = None
        if existing:
            await store.update_drone(drone_id, home=home, video_url=video_url)
        else:
            await store.upsert_drone(
                DroneState(drone_id=drone_id, position=home, home=home, video_url=video_url)
            )

    # remove extras by marking offline (keep historical visibility in-memory)
    for d in all_existing:
        if d.drone_id not in keep_ids:
            await store.update_drone(d.drone_id, status="offline")


def start_simulator() -> None:
    global sim_server
    if sim_server and sim_server.poll() is None:
        return
    if not (SIM_DIR / "serve_viewer.sh").exists():
        return
    sim_server = subprocess.Popen(
        ["bash", str(SIM_DIR / "serve_viewer.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(SIM_DIR),
    )


def stop_simulator() -> None:
    global sim_server
    if sim_server and sim_server.poll() is None:
        sim_server.terminate()
    sim_server = None


async def publish_state() -> None:
    drones = [d.model_dump() for d in await store.all_drones()]
    await hub.broadcast(
        {
            "type": "state",
            "drones": drones,
            "system": {"mode": system_mode, "drone_count": team_drone_count, "simulator_url": simulator_url()},
            "targets": [t.model_dump() for t in targets],
        }
    )


store.subscribe(lambda: asyncio.create_task(publish_state()))


@app.on_event("startup")
async def startup() -> None:
    await ensure_team_drones(team_drone_count)
    if system_mode == "live":
        live_manager.set_drones(load_live_configs(team_drone_count))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
async def get_state() -> dict:
    drones = [d.model_dump() for d in await store.all_drones()]
    return {
        "arena": {"width": arena.width, "height": arena.height, "cells": [c.model_dump() for c in arena.to_cells()]},
        "drones": drones,
        "system": {"mode": system_mode, "drone_count": team_drone_count, "simulator_url": simulator_url()},
        "targets": [t.model_dump() for t in targets],
    }


@app.post("/api/system")
async def set_system(cfg: SystemConfigCommand) -> dict:
    global system_mode, team_drone_count
    if cfg.mode not in {"live", "simulator"}:
        raise HTTPException(status_code=400, detail="mode must be live or simulator")
    if cfg.drone_count < 2 or cfg.drone_count > 5:
        raise HTTPException(status_code=400, detail="drone_count must be 2..5")

    system_mode = cfg.mode
    team_drone_count = cfg.drone_count
    if system_mode == "simulator":
        start_simulator()
        live_manager.disconnect_all()
    else:
        stop_simulator()
        live_manager.set_drones(load_live_configs(team_drone_count))

    await ensure_team_drones(team_drone_count)
    await publish_state()
    return {"ok": True}


@app.post("/api/targets")
async def set_targets(cmd: SetTargetsCommand) -> dict:
    global targets
    if len(cmd.targets) > 6:
        raise HTTPException(status_code=400, detail="max 6 targets")
    for t in cmd.targets:
        if not arena.in_bounds(t.x, t.y):
            raise HTTPException(status_code=400, detail=f"target out of bounds: ({t.x},{t.y})")
        if t.color not in {"red", "blue"}:
            raise HTTPException(status_code=400, detail="target color must be red|blue")
    targets = cmd.targets[:]
    await publish_state()
    return {"ok": True}


@app.post("/api/aruco/target-detected/{drone_id}")
async def aruco_target_detected(drone_id: str, cmd: TargetSpottedCommand) -> dict:
    # Automatic callback entrypoint from Aruco/vision pipeline.
    # marker-id matching logic can be added here once mapping is fixed.
    try:
        await store.get_drone(drone_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drone not found") from exc
    await mission.mark_target_spotted(drone_id)
    return {"ok": True, "source": "aruco"}


@app.post("/api/drones/register")
async def register_drone(cmd: RegisterDroneCommand) -> dict:
    if not arena.in_bounds(cmd.home_x, cmd.home_y):
        raise HTTPException(status_code=400, detail="Home position out of bounds")

    drone = DroneState(
        drone_id=cmd.drone_id,
        position=GridPoint(x=cmd.home_x, y=cmd.home_y),
        home=GridPoint(x=cmd.home_x, y=cmd.home_y),
        video_url=cmd.video_url,
    )
    await store.upsert_drone(drone)
    return {"ok": True, "drone": drone.model_dump()}


@app.post("/api/drones/{drone_id}/goto")
async def goto_cell(drone_id: str, cmd: GotoCommand) -> dict:
    if not arena.in_bounds(cmd.x, cmd.y):
        raise HTTPException(status_code=400, detail="Target cell out of bounds")
    try:
        await store.get_drone(drone_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drone not found") from exc

    await mission.send_drone_to(drone_id, GridPoint(x=cmd.x, y=cmd.y))
    return {"ok": True}


@app.post("/api/drones/{drone_id}/target-spotted")
async def target_spotted(drone_id: str, _cmd: TargetSpottedCommand) -> dict:
    try:
        await store.get_drone(drone_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drone not found") from exc

    await mission.mark_target_spotted(drone_id)
    return {"ok": True}


@app.post("/api/live/connect")
async def connect_live() -> dict:
    live_manager.set_drones(load_live_configs(team_drone_count))
    result = live_manager.connect_all()
    await publish_state()
    return {"ok": True, "result": result}


@app.get("/api/live/video/{drone_id}")
async def live_video(drone_id: str) -> Response:
    jpeg = live_manager.latest_jpeg(drone_id)
    if not jpeg:
        svg = f"""
        <svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'>
          <rect width='100%' height='100%' fill='#000'/>
          <text x='24' y='40' fill='#ffaa00' font-size='24'>LIVE FEED: {drone_id}</text>
          <text x='24' y='80' fill='#ffffff' font-size='18'>No frame yet. Click 'Connect Live Drones'.</text>
        </svg>
        """.strip()
        return Response(content=svg, media_type="image/svg+xml")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/sim/video/{drone_id}")
async def sim_video(drone_id: str) -> Response:
    # lightweight fake per-drone feed until simulator stream bridge is wired.
    try:
        d = await store.get_drone(drone_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drone not found") from exc

    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'>
      <rect width='100%' height='100%' fill='#000'/>
      <text x='24' y='40' fill='#00ff88' font-size='24'>SIM FEED: {drone_id}</text>
      <text x='24' y='80' fill='#66ccff' font-size='20'>status: {d.status}</text>
      <text x='24' y='110' fill='#ffffff' font-size='20'>pos: ({d.position.x}, {d.position.y})</text>
      <text x='24' y='140' fill='#ffffff' font-size='18'>mode: {system_mode}</text>
      <circle cx='{40 + d.position.x * 20}' cy='{200 + d.position.y * 10}' r='8' fill='#22c55e'/>
    </svg>
    """.strip()
    return Response(content=svg, media_type="image/svg+xml")


@app.on_event("shutdown")
def shutdown() -> None:
    stop_simulator()
    live_manager.disconnect_all()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        await publish_state()
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(ws)
