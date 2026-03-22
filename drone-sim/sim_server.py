#!/usr/bin/env python3
import argparse
import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8080
DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 5005
DEFAULT_SIM_HZ = 30.0
DEFAULT_KEY_STALE_S = 1.0

# Simple physics / motion constants
MAX_SPEED_MPS = 2.0  # when rc=100
MAX_YAW_DPS = 120.0  # when yaw=100
KEY_VEL_MPS = 0.8  # key-hold translational speed
KEY_YAW_DPS = 60.0  # key-hold yaw speed
TAKEOFF_Z = 1.2


@dataclass
class RcOverride:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0
    until_ts: float = 0.0


@dataclass
class SimState:
    # World frame (same style as dashboard/pi_position)
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    flying: bool = False
    connected: bool = True

    battery: int = 100
    temperature: int = 35
    flight_time_s: float = 0.0

    key_down: Dict[str, float] = field(default_factory=dict)  # key -> last_keepalive_ts
    rc_override: RcOverride = field(default_factory=RcOverride)
    lock: threading.Lock = field(default_factory=threading.Lock)


class DroneSimulator:
    def __init__(self, cfg: dict, udp_ip: str, udp_port: int, sim_hz: float, key_stale_s: float):
        arena = cfg.get("arena", {})
        start = cfg.get("drone_start", {})
        targets = cfg.get("targets", {})

        self.bounds = {
            "x": tuple(arena.get("x", [-10.0, 10.0])),
            "y": tuple(arena.get("y", [0.0, 10.0])),
            "z": tuple(arena.get("z", [-3.0, 3.0])),
        }

        self.targets = {
            str(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in targets.items()
        }

        self.state = SimState(
            x=float(start.get("x", 0.0)),
            y=float(start.get("y", 0.0)),
            z=float(start.get("z", 0.0)),
            yaw_deg=float(start.get("yaw_deg", 0.0)),
            flying=bool(start.get("flying", False)),
        )

        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sim_hz = max(1.0, sim_hz)
        self.key_stale_s = max(0.1, key_stale_s)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.sock.close()

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _bound_position(self, st: SimState):
        st.x = self._clamp(st.x, self.bounds["x"][0], self.bounds["x"][1])
        st.y = self._clamp(st.y, self.bounds["y"][0], self.bounds["y"][1])
        st.z = self._clamp(st.z, self.bounds["z"][0], self.bounds["z"][1])

    def _active_keys(self, now: float, st: SimState):
        stale = [k for k, ts in st.key_down.items() if (now - ts) > self.key_stale_s]
        for k in stale:
            st.key_down.pop(k, None)

    def _key_to_rc(self, st: SimState) -> Tuple[int, int, int, int]:
        # Match API docs: movement w a s d r f q e
        # w/s: forward/back
        # a/d: left/right
        # r/f: up/down
        # q/e: yaw left/right
        lr = 0
        fb = 0
        ud = 0
        yaw = 0

        keys = st.key_down
        if "a" in keys:
            lr -= 40
        if "d" in keys:
            lr += 40
        if "w" in keys:
            fb += 40
        if "s" in keys:
            fb -= 40
        if "r" in keys:
            ud += 40
        if "f" in keys:
            ud -= 40
        if "q" in keys:
            yaw -= 40
        if "e" in keys:
            yaw += 40

        # stop keys
        if "x" in keys or "space" in keys:
            lr = fb = ud = yaw = 0

        return lr, fb, ud, yaw

    def _apply_motion(self, st: SimState, dt: float, now: float):
        if not st.flying:
            return

        if now <= st.rc_override.until_ts:
            lr, fb, ud, yaw = st.rc_override.lr, st.rc_override.fb, st.rc_override.ud, st.rc_override.yaw
        else:
            lr, fb, ud, yaw = self._key_to_rc(st)

        # yaw
        yaw_rate = (yaw / 100.0) * MAX_YAW_DPS
        st.yaw_deg = (st.yaw_deg + yaw_rate * dt) % 360.0

        # body-frame velocities -> world
        v_right = (lr / 100.0) * MAX_SPEED_MPS
        v_fwd = (fb / 100.0) * MAX_SPEED_MPS
        v_up = (ud / 100.0) * MAX_SPEED_MPS

        yr = math.radians(st.yaw_deg)
        fwd_x = math.cos(yr)
        fwd_y = math.sin(yr)
        right_x = math.cos(yr + math.pi / 2.0)
        right_y = math.sin(yr + math.pi / 2.0)

        st.x += (fwd_x * v_fwd + right_x * v_right) * dt
        st.y += (fwd_y * v_fwd + right_y * v_right) * dt
        st.z += v_up * dt
        self._bound_position(st)

    def _payload(self, st: SimState) -> dict:
        yr = math.radians(st.yaw_deg)
        dir_vec = [math.cos(yr), math.sin(yr), 0.0]
        return {
            "cam": [st.x, st.y, st.z],
            "dir": dir_vec,
            "targets": self.targets,
            "debug": False,
        }

    def telemetry(self) -> dict:
        with self.state.lock:
            st = self.state
            return {
                "battery": st.battery,
                "temperature": st.temperature,
                "height_cm": int(max(0.0, st.z) * 100.0),
                "tof_cm": int(max(0.0, st.z) * 100.0),
                "barometer_cm": int((1000.0 + st.z * 100.0)),
                "flight_time_s": int(st.flight_time_s),
                "wifi_snr": -45,
                "pitch": 0,
                "roll": 0,
                "yaw": int(st.yaw_deg),
                "vgx": 0,
                "vgy": 0,
                "vgz": 0,
                "agx": 0,
                "agy": 0,
                "agz": 1000,
                "flying": st.flying,
                "connected": st.connected,
                "updated_at": time.time(),
            }

    def _loop(self):
        tick = 1.0 / self.sim_hz
        last = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            dt = max(0.0, now - last)
            last = now

            with self.state.lock:
                st = self.state
                self._active_keys(now, st)
                self._apply_motion(st, dt, now)
                if st.flying:
                    st.flight_time_s += dt
                payload = self._payload(st)

            try:
                self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.udp_ip, self.udp_port))
            except OSError:
                pass

            elapsed = time.time() - now
            time.sleep(max(0.0, tick - elapsed))


class ApiHandler(BaseHTTPRequestHandler):
    simulator: DroneSimulator = None  # injected

    def _send_json(self, code: int, data: dict):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            return self._send_json(HTTPStatus.OK, {"ok": True, "service": "tello_pi_api_server_sim"})
        if self.path == "/api/telemetry":
            return self._send_json(HTTPStatus.OK, self.simulator.telemetry())
        return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_POST(self):
        try:
            data = self._read_json()
        except Exception:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_json"})

        sim = self.simulator
        p = self.path

        with sim.state.lock:
            st = sim.state
            now = time.time()

            if p == "/api/key_down":
                key = str(data.get("key", "")).strip().lower()
                if not key:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_key"})
                st.key_down[key] = now
                if key == "t":
                    st.flying = True
                    if st.z < TAKEOFF_Z:
                        st.z = TAKEOFF_Z
                elif key == "l":
                    st.flying = False
                    st.z = 0.0
                elif key in ("x", "space"):
                    st.rc_override = RcOverride()
                return self._send_json(HTTPStatus.OK, {"ok": True, "key": key})

            if p == "/api/key_up":
                key = str(data.get("key", "")).strip().lower()
                if not key:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_key"})
                st.key_down.pop(key, None)
                return self._send_json(HTTPStatus.OK, {"ok": True, "key": key})

            if p == "/api/takeoff":
                st.flying = True
                if st.z < TAKEOFF_Z:
                    st.z = TAKEOFF_Z
                return self._send_json(HTTPStatus.OK, {"ok": True, "flying": True})

            if p == "/api/land":
                st.flying = False
                st.z = 0.0
                return self._send_json(HTTPStatus.OK, {"ok": True, "flying": False})

            if p == "/api/rc":
                lr = int(data.get("lr", 0))
                fb = int(data.get("fb", 0))
                ud = int(data.get("ud", 0))
                yaw = int(data.get("yaw", 0))
                duration_ms = int(data.get("duration_ms", 300))

                def clamp_i(v, lo, hi):
                    return max(lo, min(hi, int(v)))

                lr = clamp_i(lr, -100, 100)
                fb = clamp_i(fb, -100, 100)
                ud = clamp_i(ud, -100, 100)
                yaw = clamp_i(yaw, -100, 100)
                duration_ms = clamp_i(duration_ms, 50, 2000)

                st.rc_override = RcOverride(lr=lr, fb=fb, ud=ud, yaw=yaw, until_ts=now + duration_ms / 1000.0)
                return self._send_json(HTTPStatus.OK, {
                    "ok": True,
                    "rc": {"lr": lr, "fb": fb, "ud": ud, "yaw": yaw},
                    "duration_ms": duration_ms
                })

            if p == "/api/recover":
                st.rc_override = RcOverride()
                st.key_down.clear()
                st.connected = True
                return self._send_json(HTTPStatus.OK, {"ok": True, "message": "recovered"})

            if p == "/api/flip":
                d = str(data.get("dir", "")).strip().lower()
                if d not in {"l", "r", "f", "b"}:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_flip_dir"})
                if not st.flying:
                    return self._send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "flip_requires_flying"})
                return self._send_json(HTTPStatus.OK, {"ok": True, "dir": d})

            # Optional compatibility stubs
            if p in {"/api/speed", "/api/move", "/api/rotate", "/api/go", "/api/curve", "/api/stream", "/api/sdk",
                     "/api/emergency"}:
                return self._send_json(HTTPStatus.OK, {"ok": True, "simulated": True, "path": p})

        return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def log_message(self, fmt, *args):
        # quiet default HTTP logs
        return


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def main():
    parser = argparse.ArgumentParser(description="SDC26 drone simulator (Tello-like API + UDP pose stream)")
    parser.add_argument("--config", required=True, help="Path to simulation JSON config")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST)
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument("--udp-ip", default=DEFAULT_UDP_IP)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--sim-hz", type=float, default=DEFAULT_SIM_HZ)
    parser.add_argument("--key-stale-s", type=float, default=DEFAULT_KEY_STALE_S)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    sim = DroneSimulator(
        cfg=cfg,
        udp_ip=args.udp_ip,
        udp_port=args.udp_port,
        sim_hz=args.sim_hz,
        key_stale_s=args.key_stale_s,
    )
    sim.start()

    ApiHandler.simulator = sim
    server = ThreadingHTTPServer((args.http_host, args.http_port), ApiHandler)

    print(f"✅ Drone sim API listening on http://{args.http_host}:{args.http_port}")
    print(f"📡 UDP pose stream -> {args.udp_ip}:{args.udp_port} @ {args.sim_hz:.1f}Hz")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        sim.stop()


if __name__ == "__main__":
    main()
