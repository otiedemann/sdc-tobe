#!/usr/bin/env python3
"""
Simulator API compatibility server for swarm challenge testing.

Goal:
- Provide a Tello-like HTTP API compatible with controller/tello_pi_api_server.py clients.
- Run 5 independent API endpoints on separate ports (one per simulated drone).
- Route commands to simulated drones (no real drone SDK calls).

Default ports:
- Drone 1 -> 8081
- Drone 2 -> 8082
- Drone 3 -> 8083
- Drone 4 -> 8084
- Drone 5 -> 8085
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server

RC_HZ = 20.0
TELEMETRY_HZ = 5.0
GRAVITY = -0.10  # gentle sink when flying with no vertical command


@dataclass
class DroneSimState:
    drone_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    battery: float = 100.0
    flying: bool = False
    connected: bool = True
    flight_time_s: float = 0.0
    temperature: float = 38.0

    # command state
    pressed: Set[str] = field(default_factory=set)
    rc_override: tuple[int, int, int, int] | None = None
    rc_override_until: float = 0.0

    # telemetry log
    telemetry_log_enabled: bool = False
    telemetry_log_path: Path = field(default_factory=lambda: Path("telemetry_log.jsonl"))

    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.time)


class DroneServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int):
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def axis(pos: bool, neg: bool) -> int:
    return (1 if pos else 0) + (-1 if neg else 0)


def build_app(state: DroneSimState) -> Flask:
    app = Flask(f"sim_api_drone_{state.drone_id}")

    def _snapshot() -> Dict:
        with state.lock:
            return {
                "drone_id": state.drone_id,
                "battery": int(max(0.0, min(100.0, state.battery))),
                "temperature": int(state.temperature),
                "height_cm": int(state.z * 100),
                "tof_cm": int(state.z * 100),
                "barometer_cm": int(1000 + state.z * 100),
                "flight_time_s": int(state.flight_time_s),
                "pitch": 0,
                "roll": 0,
                "yaw": int(state.yaw),
                "vgx": 0,
                "vgy": 0,
                "vgz": 0,
                "agx": 0,
                "agy": 0,
                "agz": 1000,
                "flying": state.flying,
                "connected": state.connected,
                "updated_at": time.time(),
                "sim_pos": {"x": round(state.x, 3), "y": round(state.y, 3), "z": round(state.z, 3)},
            }

    def _append_telemetry(payload: Dict):
        with state.lock:
            enabled = state.telemetry_log_enabled
            p = state.telemetry_log_path
        if not enabled:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @app.get("/")
    def root():
        return jsonify(ok=True, service="sim_api_server", drone_id=state.drone_id)

    @app.post("/api/key_down")
    def api_key_down():
        data = request.get_json(silent=True) or {}
        k = str(data.get("key", "")).lower().strip()
        if k:
            with state.lock:
                state.pressed.add(k)
        return jsonify(ok=True)

    @app.post("/api/key_up")
    def api_key_up():
        data = request.get_json(silent=True) or {}
        k = str(data.get("key", "")).lower().strip()
        if k:
            with state.lock:
                state.pressed.discard(k)
        return jsonify(ok=True)

    @app.post("/api/takeoff")
    def api_takeoff():
        with state.lock:
            if not state.flying:
                state.flying = True
                state.z = max(state.z, 0.8)
        return jsonify(ok=True, flying=True)

    @app.post("/api/land")
    def api_land():
        with state.lock:
            state.flying = False
            state.z = 0.0
        return jsonify(ok=True, flying=False)

    @app.post("/api/flip")
    def api_flip():
        data = request.get_json(silent=True) or {}
        direction = str(data.get("dir", "")).lower()
        if direction not in {"l", "r", "f", "b"}:
            return jsonify(ok=False, error="dir must be one of l|r|f|b"), 400
        with state.lock:
            if not state.flying:
                return jsonify(ok=False, error="flip_requires_flying"), 409
            if state.battery < 20:
                return jsonify(ok=False, error="flip_requires_battery_20_plus", battery=int(state.battery)), 409
            # visual-only quick yaw/position nudge to indicate flip execution in sim telemetry
            if direction == "l":
                state.yaw -= 45
            elif direction == "r":
                state.yaw += 45
            elif direction == "f":
                state.x += 0.2
            elif direction == "b":
                state.x -= 0.2
        return jsonify(ok=True, dir=direction)

    @app.post("/api/rc")
    def api_rc():
        data = request.get_json(silent=True) or {}
        lr = clamp(int(data.get("lr", 0)), -100, 100)
        fb = clamp(int(data.get("fb", 0)), -100, 100)
        ud = clamp(int(data.get("ud", 0)), -100, 100)
        yaw = clamp(int(data.get("yaw", 0)), -100, 100)
        dur_ms = clamp(int(data.get("duration_ms", 250)), 50, 2000)
        with state.lock:
            state.rc_override = (lr, fb, ud, yaw)
            state.rc_override_until = time.time() + (dur_ms / 1000.0)
        return jsonify(ok=True, rc={"lr": lr, "fb": fb, "ud": ud, "yaw": yaw}, duration_ms=dur_ms)

    @app.post("/api/recover")
    def api_recover():
        with state.lock:
            state.pressed.clear()
            state.rc_override = None
            state.rc_override_until = 0.0
            state.flying = False
            state.z = 0.0
            state.connected = True
        return jsonify(ok=True, message="recovered")

    @app.get("/api/telemetry")
    def api_telemetry():
        return jsonify(_snapshot())

    @app.get("/api/telemetry/stream")
    def api_telemetry_stream():
        def gen():
            while True:
                payload = _snapshot()
                _append_telemetry(payload)
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1.0 / TELEMETRY_HZ)

        return Response(gen(), mimetype="text/event-stream")

    @app.get("/api/logging/telemetry")
    def api_log_status():
        with state.lock:
            return jsonify(enabled=state.telemetry_log_enabled, path=str(state.telemetry_log_path))

    @app.post("/api/logging/telemetry")
    def api_log_set():
        data = request.get_json(silent=True) or {}
        with state.lock:
            if isinstance(data.get("enabled"), bool):
                state.telemetry_log_enabled = data["enabled"]
            if isinstance(data.get("path"), str) and data["path"].strip():
                state.telemetry_log_path = Path(data["path"].strip())
            return jsonify(enabled=state.telemetry_log_enabled, path=str(state.telemetry_log_path))

    @app.get("/api/logging/telemetry/download")
    def api_log_download():
        with state.lock:
            p = state.telemetry_log_path
        if not p.exists():
            return jsonify(ok=False, error="telemetry log file not found", path=str(p)), 404
        return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")

    @app.post("/api/logging/telemetry/clear")
    def api_log_clear():
        with state.lock:
            p = state.telemetry_log_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))

    @app.get("/api/safety/takeoff")
    def api_safe_takeoff_get():
        # Compatibility endpoint; no-op in simulator backend.
        return jsonify(enabled=False, hold_s=0.0)

    @app.post("/api/safety/takeoff")
    def api_safe_takeoff_set():
        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled", False))
        return jsonify(ok=True, enabled=enabled, hold_s=0.0)

    return app


def drone_update_loop(state: DroneSimState, stop_evt: threading.Event):
    period = 1.0 / RC_HZ
    while not stop_evt.is_set():
        t0 = time.time()
        with state.lock:
            now = time.time()
            lr = axis("d" in state.pressed, "a" in state.pressed) * 50
            fb = axis("w" in state.pressed, "s" in state.pressed) * 50
            ud = axis("r" in state.pressed, "f" in state.pressed) * 50
            yw = axis("e" in state.pressed, "q" in state.pressed) * 50

            if "x" in state.pressed or "space" in state.pressed:
                lr = fb = ud = yw = 0

            if state.rc_override is not None and now < state.rc_override_until:
                lr, fb, ud, yw = state.rc_override
            elif state.rc_override is not None and now >= state.rc_override_until:
                state.rc_override = None

            if state.flying:
                dt = period
                state.x += (fb / 100.0) * dt * 1.6
                state.y += (lr / 100.0) * dt * 1.6
                state.z += (ud / 100.0) * dt * 1.2
                if ud == 0:
                    state.z += GRAVITY * dt
                state.z = max(0.0, min(6.0, state.z))
                state.yaw = (state.yaw + (yw / 100.0) * dt * 120.0) % 360.0
                state.flight_time_s += dt
                state.battery = max(0.0, state.battery - dt * 0.08)
                state.temperature = min(85.0, state.temperature + dt * 0.015)
                if state.z <= 0.01 and ud <= 0:
                    state.z = 0.0
            else:
                state.z = 0.0
                state.temperature = max(30.0, state.temperature - period * 0.01)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def main():
    p = argparse.ArgumentParser(description="Run 5-port simulator API compatibility server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--base-port", type=int, default=8081)
    p.add_argument("--drones", type=int, default=5)
    args = p.parse_args()

    stop_evt = threading.Event()
    server_threads: list[DroneServerThread] = []

    print("[sim_api_server] starting compatibility endpoints:")
    for i in range(1, args.drones + 1):
        state = DroneSimState(drone_id=i, telemetry_log_path=Path(f"sim_drone_{i}_telemetry.jsonl"))
        app = build_app(state)
        port = args.base_port + (i - 1)
        st = DroneServerThread(app, args.host, port)
        ut = threading.Thread(target=drone_update_loop, args=(state, stop_evt), daemon=True)
        st.start()
        ut.start()
        server_threads.append(st)
        print(f"  drone {i}: http://{args.host}:{port}")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[sim_api_server] stopping...")
    finally:
        stop_evt.set()
        for s in server_threads:
            s.shutdown()


if __name__ == "__main__":
    main()
