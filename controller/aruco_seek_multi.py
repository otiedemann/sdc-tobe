"""
ArUco Seek — multi-drone observer/controller module.

Spun out of tools/aruco_seek_web.py so the remote_web_controller can run
one independent observer per drone (each with its own api_base, params,
target, trackers, thread, and OBSERVE/LIVE mode).

The PD math and pipeline structure mirror tools/aruco_seek_web.py exactly
— this module just swaps the single-drone module globals (_api_base, etc.)
for per-instance HTTP helpers so N observers can run in parallel without
fighting over any shared state.

Re-uses VideoMarkerTracker + PositionListener from tools/aruco_seek.py
(both already take api_base in their constructor — no globals there).
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests


# ─── Mission trace logger ───────────────────────────────────────────────────

class MissionTraceLogger:
    """Append-only JSONL logger for mission state. Each line is a single
    JSON object. Used by ScanAllMarkersMission to record every FSM tick,
    phase transition, claim/release, and scan event, so post-flight
    debugging can reconstruct exactly what each drone thought was happening.
    """

    def __init__(self, path: Path, mission_name: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", buffering=1)     # line-buffered
        self.mission_name = mission_name
        self._closed = False
        self.write("trace_open", {
            "mission": mission_name,
            "started_at_wall": time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        })

    def write(self, event: str, payload: Optional[dict] = None):
        if self._closed:
            return
        rec = {"t": round(time.time(), 3), "event": event}
        if payload:
            rec.update(payload)
        try:
            with self._lock:
                self._fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.write(json.dumps({
                    "t": round(time.time(), 3), "event": "trace_close"
                }) + "\n")
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass


# Default on-disk location for mission traces
MISSION_LOG_DIR = Path(__file__).resolve().parent / "logs"

# Pull in the stateless parts of aruco_seek (trackers + pure helpers)
_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
import aruco_seek as ASEEK  # noqa: E402
from aruco_seek import (  # noqa: E402
    PositionListener,
    VideoMarkerTracker,
    marker_skew,
    pixel_distance_estimate,
    wrap_pi,
)


# ─── Tunable parameters (mirrors HOVER_* in aruco_seek.py) ──────────────────


@dataclass
class HoverParams:
    # Camera filter / deadbands
    ema_alpha: float       = ASEEK.HOVER_EMA_ALPHA
    deadband_x: float      = ASEEK.HOVER_DEADBAND_X
    deadband_y: float      = ASEEK.HOVER_DEADBAND_Y
    deadband_skew: float   = ASEEK.HOVER_DEADBAND_SKEW
    deadband_dist_m: float = ASEEK.HOVER_DEADBAND_DIST_M
    # P gains
    yaw_p: float           = ASEEK.HOVER_YAW_P
    skew_p: float          = ASEEK.HOVER_SKEW_P
    alt_p: float           = ASEEK.HOVER_ALT_P
    dist_p: float          = ASEEK.HOVER_DIST_P
    # D gains (IMU damping)
    d_lr: float            = ASEEK.HOVER_D_LR
    d_fb: float            = ASEEK.HOVER_D_FB
    d_ud: float            = ASEEK.HOVER_D_UD
    d_yaw: float           = ASEEK.HOVER_D_YAW
    # Output clamps
    yaw_max: int           = ASEEK.HOVER_YAW_MAX
    lr_max: int            = ASEEK.HOVER_LR_MAX
    ud_max: int            = ASEEK.HOVER_UD_MAX
    fb_max: int            = ASEEK.HOVER_FB_MAX
    fb_back_max: int       = ASEEK.HOVER_FB_BACK_MAX
    rc_min: int            = ASEEK.HOVER_RC_MIN
    # Mission target
    hover_distance_m: float = ASEEK.HOVER_DISTANCE_M
    # Camera HFOV — drawing-only
    cam_hfov_deg: float    = 69.0
    # Physical marker size in meters — drives distance estimation.
    # Distance scale auto-adjusts via (marker_size_m / MARKER_SIZE_CALIB_M) ** CALIB_B.
    marker_size_m: float   = ASEEK.MARKER_SIZE_M


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def _clamp_rc(v: int) -> int:
    return max(-100, min(100, int(v)))


# ─── Per-drone observer ─────────────────────────────────────────────────────


class DroneObserver:
    """
    Background pipeline for ONE drone: camera + IMU + position SSE + PD math.

    Fully self-contained — talks to the drone API via its own HTTP session,
    never touches aruco_seek module globals, so many can run in parallel.
    """

    # RC_TICK_MS is the duration_ms we send with each /api/rc packet. The
    # server's rc_loop zeros rc_override the moment it expires, so if this
    # is too close to our tick-interval the drone sees brief gaps of
    # zero-stick between our packets → stutters forward in visible steps.
    #
    # Keep this comfortably longer than the worst-case tick-to-tick time
    # (HTTP roundtrip + ArUco detection + PD math ≈ 20-150ms at 10Hz tick).
    # Each subsequent tick supersedes the previous override, so a long
    # duration here is "expiration as safety ceiling", not actual setpoint
    # lifetime. If the observer thread dies, the server auto-zeros the
    # stick after this many milliseconds.
    RC_TICK_MS = 400

    def __init__(self, drone_id: str, api_base: str, name: str = "",
                 session: Optional[requests.Session] = None,
                 allow_live: bool = True):
        self.drone_id = drone_id
        self.api_base = api_base.rstrip("/")
        self.name = name or drone_id
        self._session = session or requests.Session()
        self._lock = threading.RLock()

        self.params = HoverParams()
        self.target_id: Optional[int] = None    # None = auto-pick largest
        self._chosen_id: Optional[int] = None

        self.mode: str = "observe"       # "observe" | "live"
        self.allow_live = bool(allow_live)
        # ── Waypoint navigation (world-frame goto) ──────────────────
        # When set, the observer flies toward a world-frame XYZ setpoint
        # via P-control on arena-frame position, with independent yaw so
        # the camera keeps facing a user-specified point (typically the
        # arena centre to maximise marker visibility). Overrides the
        # marker-tracking PD loop when active.
        self._waypoint_xyz: Optional[tuple] = None     # (x, y, z) in arena frame
        self._waypoint_face_xy: Optional[tuple] = None  # (x, y) to aim camera at
        # ── Arena safety bounds (world frame, metres) ────────────────
        # Drone must never leave this box. Default per SDC26 ruleset
        # (20 × 10.8 × 6 m field). Configurable via set_arena_bounds().
        # safety_margin_m is how close to a wall we start braking.
        self._arena_bounds = {
            "x_min": -10.0, "x_max": 10.0,
            "y_min":   0.0, "y_max": 10.8,
            "z_min":   0.3, "z_max":  5.0,   # some ceiling headroom
        }
        self._safety_margin_m = 1.0
        self._last_guard_info = {}   # diagnostics for state snapshot
        # Search RC — commands the drone sends when in LIVE mode but not
        # actively tracking a marker (e.g. mission SEARCH maneuver).
        # Tuple: (lr, fb, ud, yaw) in RC units -100..100. Mission controls this.
        self._search_rc: tuple = (0, 0, 0, 0)
        # IMPORTANT: include mode + allow_live in the initial state so the
        # client's /proxy/aruco/state poll can enable the LIVE button BEFORE
        # the observer is started. Without these keys the UI stays in
        # "LIVE disabled" mode until _tick() fires for the first time.
        self.latest: dict = {
            "running": False,
            "drone_id": drone_id,
            "mode": self.mode,
            "allow_live": self.allow_live,
            "params": asdict(self.params),
        }
        self._last_send_t: float = 0.0

        self._vid: Optional[VideoMarkerTracker] = None
        self._pos: Optional[PositionListener] = None
        self._thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False

        # EMA / yaw-rate state
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._fs: Optional[float] = None
        self._fd: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._prev_yaw_t: float = 0.0
        self._yaw_rate: float = 0.0

    # ── Per-drone HTTP helpers (replaces aruco_seek module globals) ──
    def _api_post(self, path: str, body: Optional[dict] = None, timeout: float = 12.0) -> dict:
        try:
            r = self._session.post(f"{self.api_base}{path}", json=body or {}, timeout=timeout)
            if r.headers.get("Content-Type", "").startswith("application/json"):
                return r.json()
            return {"ok": r.ok, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _api_get(self, path: str, timeout: float = 2.0) -> dict:
        try:
            r = self._session.get(f"{self.api_base}{path}", timeout=timeout)
            if r.headers.get("Content-Type", "").startswith("application/json"):
                return r.json()
            return {"ok": r.ok, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_waypoint(self, xyz, face_xy=None):
        """Fly to a world-frame (x, y, z) setpoint. If face_xy is given,
        the drone yaws to aim its camera at that world-XY point (typically
        the arena centre so more markers stay in view for triangulation).
        Pass None to cancel waypoint nav (drone returns to normal
        marker-tracking PD)."""
        with self._lock:
            self._waypoint_xyz = (
                (float(xyz[0]), float(xyz[1]), float(xyz[2])) if xyz is not None else None
            )
            self._waypoint_face_xy = (
                (float(face_xy[0]), float(face_xy[1])) if face_xy is not None else None
            )

    def clear_waypoint(self):
        self.set_waypoint(None, None)

    def _compute_waypoint_rc(self):
        """Return (lr, fb, ud, yaw, dist_xy, dz) for the current waypoint,
        or None if waypoint nav isn't configured or position is unknown.
        All values are RC% (-100..+100)."""
        with self._lock:
            wp = self._waypoint_xyz
            face = self._waypoint_face_xy
            snap = dict(self.latest)
        if wp is None:
            return None
        pos = snap.get("pos") or snap.get("cam")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None
        cx = float(pos[0])
        cy = float(pos[1])
        cz = float(pos[2]) if len(pos) >= 3 else (snap.get("altitude_m") or 1.5)

        tx, ty, tz = wp
        dx = tx - cx     # world
        dy = ty - cy
        dz = tz - cz
        dist_xy = (dx * dx + dy * dy) ** 0.5

        yaw_deg = float(snap.get("yaw", 0.0) or 0.0)
        yaw_rad = math.radians(yaw_deg)
        s = math.sin(yaw_rad)
        c = math.cos(yaw_rad)

        # Body-frame RC from world-frame desired motion. Anafi yaw
        # convention: yaw=0 → nose points world +y. Body-fwd in world is
        # (sin(y), cos(y)); body-right is (cos(y), -sin(y)). Solving
        # world_move = lr · body_right + fb · body_fwd gives:
        fb_err_m = dx * s + dy * c           # metres along body-fwd
        lr_err_m = dx * c - dy * s           # metres along body-right

        # P gains (RC% per metre). Conservative so the drone doesn't
        # overshoot in the often-noisy arena-frame pose.
        P_XY = 25
        P_Z  = 35
        P_YAW = 2.0   # RC% per degree

        fb_rc = int(round(fb_err_m * P_XY))
        lr_rc = int(round(lr_err_m * P_XY))
        ud_rc = int(round(dz * P_Z))

        # Face-point yaw control. Work out the world bearing from the
        # drone to the face_point (typically arena centre), express it
        # as a yaw angle (0° = world +y, CW positive to match Anafi),
        # then P-control the yaw error.
        yaw_rc = 0
        if face is not None:
            fx, fy = face
            dfx = fx - cx
            dfy = fy - cy
            if (dfx * dfx + dfy * dfy) > 1e-4:
                desired_yaw = math.degrees(math.atan2(dfx, dfy))
                yaw_err = ((desired_yaw - yaw_deg + 180.0) % 360.0) - 180.0
                yaw_rc = int(round(yaw_err * P_YAW))

        # Clamps — safer than full-stick on waypoints
        fb_rc  = max(-40, min(40, fb_rc))
        lr_rc  = max(-40, min(40, lr_rc))
        ud_rc  = max(-30, min(30, ud_rc))
        yaw_rc = max(-50, min(50, yaw_rc))
        return lr_rc, fb_rc, ud_rc, yaw_rc, dist_xy, dz

    def set_arena_bounds(self, bounds: dict, safety_margin_m: float = None):
        """Override the arena safety box. `bounds` keys: x_min, x_max,
        y_min, y_max, z_min, z_max (all metres, world frame)."""
        with self._lock:
            self._arena_bounds.update({k: float(v) for k, v in bounds.items()
                                       if k in self._arena_bounds})
            if safety_margin_m is not None:
                self._safety_margin_m = float(safety_margin_m)

    # Latency compensation: when checking "will this RC push the drone
    # past the margin?", we look LOOKAHEAD_S seconds into the future
    # using the current IMU velocity. Accounts for the drone's response
    # lag + our network RTT — roughly 200-400 ms from RC command to
    # physical response.
    GUARD_LOOKAHEAD_S = 0.35

    def _apply_boundary_guard(self, lr: int, fb: int, ud: int, yaw_rc: int):
        """Return (lr, fb, ud, yaw_rc) clamped so the drone won't be
        pushed past the arena margin. Latency-aware: the check uses a
        PREDICTED position based on current IMU velocity, not just the
        instantaneous position, so the guard reacts before the drone
        has already overshot.

        Body-frame → world-frame conversion uses the drone's current yaw:
            world_dx = cos(yaw)·lr + sin(yaw)·fb
            world_dy = -sin(yaw)·lr + cos(yaw)·fb
        (Anafi yaw=0 → nose points +y, positive yaw rotates CW.)
        """
        with self._lock:
            snap = self.latest
            bounds = dict(self._arena_bounds)
            margin = self._safety_margin_m
        pos = snap.get("pos") or snap.get("cam")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            self._last_guard_info = {"active": False, "reason": "no-pos"}
            return lr, fb, ud, yaw_rc
        px = float(pos[0])
        py = float(pos[1])
        pz = float(pos[2]) if len(pos) >= 3 else 1.0

        yaw_deg = snap.get("yaw", 0.0) or 0.0
        yaw_rad = math.radians(float(yaw_deg))
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

        # ── Latency-aware prediction ────────────────────────────────
        # IMU velocity from Anafi telemetry is in BODY FRAME, cm/s.
        # Convert to world frame (m/s) for prediction.
        vbx = float(snap.get("vx_cms", 0) or 0) * 0.01   # body fwd
        vby = float(snap.get("vy_cms", 0) or 0) * 0.01   # body right
        vbz = float(snap.get("vz_cms", 0) or 0) * 0.01   # body down (Anafi)
        # Same body→world rotation as RC (fwd=body+y, right=body+x).
        vwx =  cy * vby + sy * vbx
        vwy = -sy * vby + cy * vbx
        vwz = -vbz   # world z-up, body z-down
        dt = self.GUARD_LOOKAHEAD_S
        px_pred = px + vwx * dt
        py_pred = py + vwy * dt
        pz_pred = pz + vwz * dt
        # Use the PREDICTED position for clearance calculations so the
        # guard reacts to where we'll be, not where we are.
        px_eff, py_eff, pz_eff = px_pred, py_pred, pz_pred

        # Body-frame RC → world-frame intended velocity direction
        wdx =  cy * lr + sy * fb
        wdy = -sy * lr + cy * fb

        # How deep inside each wall's margin are we, T seconds from now?
        # Positive = still outside the danger zone, negative = already
        # past the margin (accounting for current velocity + RTT).
        clearance = {
            "x_max": bounds["x_max"] - px_eff - margin,
            "x_min": (px_eff - bounds["x_min"]) - margin,
            "y_max": bounds["y_max"] - py_eff - margin,
            "y_min": (py_eff - bounds["y_min"]) - margin,
            "z_max": bounds["z_max"] - pz_eff - margin,
            "z_min": (pz_eff - bounds["z_min"]) - margin,
        }

        # Retreat-RC magnitude. Modest so we don't overshoot in the opposite
        # direction. The PD loop's normal authority is higher so it can
        # override this when heading back toward center is the correct thing.
        RETREAT_RC = 15
        guard_info = {"active": False, "actions": [], "clearance": {
            k: round(v, 2) for k, v in clearance.items()
        }, "pos": [round(px, 2), round(py, 2), round(pz, 2)],
           "pos_pred": [round(px_eff, 2), round(py_eff, 2), round(pz_eff, 2)],
           "vel_world": [round(vwx, 2), round(vwy, 2), round(vwz, 2)],
           "lookahead_s": self.GUARD_LOOKAHEAD_S}

        # X axis (world)
        if clearance["x_max"] <= 0 and wdx >= 0:
            wdx = -abs(wdx) * 0.0 - RETREAT_RC    # force negative (retreat)
            guard_info["actions"].append("retreat-x_max")
        elif clearance["x_max"] < margin and wdx > 0:
            wdx = min(wdx, int(RETREAT_RC * clearance["x_max"] / margin))
            guard_info["actions"].append("clamp-x_max")
        if clearance["x_min"] <= 0 and wdx <= 0:
            wdx = RETREAT_RC
            guard_info["actions"].append("retreat-x_min")
        elif clearance["x_min"] < margin and wdx < 0:
            wdx = max(wdx, -int(RETREAT_RC * clearance["x_min"] / margin))
            guard_info["actions"].append("clamp-x_min")

        # Y axis (world)
        if clearance["y_max"] <= 0 and wdy >= 0:
            wdy = -RETREAT_RC
            guard_info["actions"].append("retreat-y_max")
        elif clearance["y_max"] < margin and wdy > 0:
            wdy = min(wdy, int(RETREAT_RC * clearance["y_max"] / margin))
            guard_info["actions"].append("clamp-y_max")
        if clearance["y_min"] <= 0 and wdy <= 0:
            wdy = RETREAT_RC
            guard_info["actions"].append("retreat-y_min")
        elif clearance["y_min"] < margin and wdy < 0:
            wdy = max(wdy, -int(RETREAT_RC * clearance["y_min"] / margin))
            guard_info["actions"].append("clamp-y_min")

        # Convert world-frame intent back to body frame
        new_lr =  cy * wdx - sy * wdy
        new_fb =  sy * wdx + cy * wdy

        # Altitude (already world-frame)
        new_ud = ud
        if clearance["z_max"] <= 0 and ud >= 0:
            new_ud = -RETREAT_RC
            guard_info["actions"].append("retreat-z_max")
        elif clearance["z_max"] < margin and ud > 0:
            new_ud = min(ud, int(RETREAT_RC * clearance["z_max"] / margin))
            guard_info["actions"].append("clamp-z_max")
        if clearance["z_min"] <= 0 and new_ud <= 0:
            new_ud = RETREAT_RC
            guard_info["actions"].append("retreat-z_min")
        elif clearance["z_min"] < margin and new_ud < 0:
            new_ud = max(new_ud, -int(RETREAT_RC * clearance["z_min"] / margin))
            guard_info["actions"].append("clamp-z_min")

        if guard_info["actions"]:
            guard_info["active"] = True
        self._last_guard_info = guard_info
        return (int(round(new_lr)), int(round(new_fb)),
                int(round(new_ud)), int(yaw_rc))

    def _send_rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> dict:
        # Boundary guard clamps/reverses RC to keep the drone inside the
        # arena safety box. If we're moving into a wall we already passed
        # the margin for, the drone is actively pushed back to center.
        lr, fb, ud, yaw = self._apply_boundary_guard(lr, fb, ud, yaw)
        return self._api_post("/api/rc", {
            "lr": _clamp_rc(lr), "fb": _clamp_rc(fb),
            "ud": _clamp_rc(ud), "yaw": _clamp_rc(yaw),
            "duration_ms": self.RC_TICK_MS,
        })

    def _send_rc_stop(self) -> dict:
        return self._send_rc(0, 0, 0, 0)

    # ── Configuration ──
    def set_api_base(self, api_base: str):
        """Re-point this observer at a different API URL. Stops any running threads first."""
        with self._lock:
            was_running = self._running
        if was_running:
            self.stop()
        self.api_base = api_base.rstrip("/")

    def get_params(self) -> dict:
        with self._lock:
            return asdict(self.params)

    def update_params(self, updates: dict) -> dict:
        applied = {}
        with self._lock:
            for k, v in updates.items():
                if not hasattr(self.params, k):
                    continue
                cur = getattr(self.params, k)
                try:
                    new_val = int(round(float(v))) if (isinstance(cur, int) and not isinstance(cur, bool)) else float(v)
                    setattr(self.params, k, new_val)
                    applied[k] = new_val
                except Exception:
                    pass
        return applied

    def set_target(self, mid: Optional[int]):
        with self._lock:
            self.target_id = mid
            self._chosen_id = None
            self._fx = self._fy = self._fs = self._fd = None

    def set_search_rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0):
        """Set the RC commands the drone sends while in LIVE mode with no
        tracked marker. Used by the mission's SEARCH phase to rotate or
        move the drone while hunting for the next target. Zero tuple
        disables the search maneuver (drone just hovers in place)."""
        with self._lock:
            self._search_rc = (_clamp_rc(lr), _clamp_rc(fb),
                               _clamp_rc(ud), _clamp_rc(yaw))

    def set_mode(self, mode: str) -> str:
        mode = (mode or "").lower()
        if mode not in ("observe", "live"):
            return self.mode
        if mode == "live" and not self.allow_live:
            return self.mode
        with self._lock:
            prev = self.mode
            self.mode = mode
        if prev == "live" and mode == "observe":
            try:
                self._send_rc_stop()
            except Exception:
                pass
        return mode

    # ── Direct manual commands ──
    def cmd_takeoff(self) -> dict:
        return self._api_post("/api/takeoff")

    def cmd_land(self) -> dict:
        return self._api_post("/api/land")

    def cmd_emergency(self) -> dict:
        return self._api_post("/api/emergency")

    def cmd_rc_stop(self) -> dict:
        return self._send_rc_stop()

    # ── State snapshot ──
    def get_state(self) -> dict:
        with self._lock:
            return dict(self.latest)

    # ── Lifecycle ──
    def start(self):
        if self._running:
            return
        try:
            self._api_post("/api/video/start", {"mode": "mjpeg"})
            self._api_post("/api/position/config", {"enabled": True})
        except Exception:
            pass
        self._vid = VideoMarkerTracker(self.api_base)
        self._pos = PositionListener(self.api_base)
        self._vid.start()
        self._pos.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name=f"obs-{self.drone_id}")
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True,
                                            name=f"obs-hb-{self.drone_id}")
        self._thread.start()
        self._hb_thread.start()

    def stop(self):
        if self.mode == "live":
            try:
                self._send_rc_stop()
            except Exception:
                pass
        self._running = False
        if self._vid:
            self._vid.stop()
            self._vid = None
        if self._pos:
            self._pos.stop()
            self._pos = None
        with self._lock:
            self.latest = {
                "running": False,
                "drone_id": self.drone_id,
                "mode": self.mode,
                "allow_live": self.allow_live,
            }

    def _hb_loop(self):
        while self._running:
            try:
                self._api_get("/api/heartbeat", timeout=0.5)
            except Exception:
                pass
            time.sleep(0.4)

    def _run(self):
        time.sleep(1.0)  # let trackers warm up
        TICK_PERIOD_S = 0.05   # 20 Hz — matches the server's PCMD loop rate
                               # so every server tick has a fresh setpoint.
                               # Combined with RC_TICK_MS=400, the drone
                               # never experiences a zero-stick gap.
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                with self._lock:
                    self.latest["error"] = repr(e)
            # Sleep for the REMAINDER of the tick period so slow ticks
            # don't pile up (if _tick takes 80 ms, we sleep 20 ms, not
            # a fixed 50 ms which would drift out to ~130 ms/tick).
            elapsed = time.time() - t0
            time.sleep(max(0.005, TICK_PERIOD_S - elapsed))

    def _tick(self):
        if self._vid is None:
            return
        with self._lock:
            p = HoverParams(**asdict(self.params))
            target_id = self.target_id

        snapshot: dict = {
            "running": True,
            "drone_id": self.drone_id,
            "mode": self.mode,
            "allow_live": self.allow_live,
            "frame_w": self._vid.frame_size[0],
            "frame_h": self._vid.frame_size[1],
            "vid_active": self._vid.is_active,
            "vid_age": round(self._vid.age, 2),
        }

        vid_all = self._vid.get_all()
        snapshot["visible_ids"] = sorted(vid_all.keys())

        tel = self._api_get("/api/telemetry")
        vx_cms = float(tel.get("vgx") or 0.0)
        vy_cms = float(tel.get("vgy") or 0.0)
        vz_cms = float(tel.get("vgz") or 0.0)
        pitch  = float(tel.get("pitch") or 0.0)
        roll   = float(tel.get("roll")  or 0.0)
        yaw_d  = float(tel.get("yaw")   or 0.0)
        height_m = (tel.get("height_cm") or 0) / 100.0
        bat = tel.get("battery", None)

        # Yaw rate (filtered)
        a = p.ema_alpha
        t_now = time.time()
        if self._prev_yaw is not None:
            dt = t_now - self._prev_yaw_t
            if 0.01 < dt < 1.0:
                d_yaw = wrap_pi(math.radians(yaw_d - self._prev_yaw))
                inst_rate = math.degrees(d_yaw) / dt
                self._yaw_rate = (1 - a) * self._yaw_rate + a * inst_rate
        self._prev_yaw = yaw_d
        self._prev_yaw_t = t_now

        snapshot.update({
            "vx_cms": vx_cms, "vy_cms": vy_cms, "vz_cms": vz_cms,
            "pitch": pitch, "roll": roll, "yaw": yaw_d,
            "yaw_rate_dps": round(self._yaw_rate, 1),
            "altitude_m": round(height_m, 2),
            "battery": bat,
        })
        # Surface the most recent boundary-guard decision so UI + trace
        # can visualise when the safety net is actively braking/retreating.
        if self._last_guard_info:
            snapshot["guard"] = dict(self._last_guard_info)

        # ── Waypoint navigation (overrides marker PD when active) ──
        # If a mission has set a world-frame waypoint, fly to it now.
        # Position feedback comes from the ArUco pipeline (snap["pos"]).
        # We also keep the camera aimed at face_point (typically arena
        # centre) via independent yaw control so more markers stay in
        # view for triangulation — decoupled from the body-forward axis.
        wp_rc = self._compute_waypoint_rc()
        if wp_rc is not None:
            lr, fb, ud, yaw_rc, dist_xy, dz = wp_rc
            if self.mode == "live":
                try:
                    self._send_rc(lr, fb, ud, yaw_rc)
                    snapshot["rc_sent_at"] = round(t_now, 2)
                except Exception as e:
                    snapshot["rc_send_error"] = str(e)[:80]
            with self._lock:
                wp  = self._waypoint_xyz
                face = self._waypoint_face_xy
            snapshot.update({
                "waypoint": list(wp) if wp else None,
                "waypoint_face": list(face) if face else None,
                "waypoint_dist_xy": round(dist_xy, 2),
                "waypoint_dz": round(dz, 2),
                "rc_lr": lr, "rc_fb": fb, "rc_ud": ud, "rc_yaw": yaw_rc,
                "marker_id": None,
                "status_msg": f"→ waypoint ({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f}) "
                              f"d={dist_xy:.2f}m dz={dz:+.2f}m",
            })
            self._chosen_id = None
            with self._lock:
                self.latest = snapshot
            return

        if not vid_all:
            snapshot["marker_id"] = None
            snapshot["status_msg"] = "no markers visible"
            # Safety: if we were actively tracking in live mode and just lost
            # the marker, push a single RC-stop so the drone doesn't drift.
            if self.mode == "live" and self._chosen_id is not None:
                try:
                    self._send_rc_stop()
                    snapshot["rc_sent_at"] = round(t_now, 2)
                except Exception:
                    pass
                self._chosen_id = None
            # Apply search RC (mission-driven rotation/translation) if set
            with self._lock:
                src = self._search_rc
            if self.mode == "live" and any(v != 0 for v in src):
                try:
                    self._send_rc(*src)
                    snapshot["rc_sent_at"] = round(t_now, 2)
                    snapshot["search_rc"] = list(src)
                    snapshot["status_msg"] = (
                        f"searching — rc(lr={src[0]}, fb={src[1]}, "
                        f"ud={src[2]}, yaw={src[3]})"
                    )
                except Exception:
                    pass
            with self._lock:
                self.latest = snapshot
            return

        if target_id is not None and target_id in vid_all:
            chosen = target_id
        elif target_id is None:
            # If the mission is actively driving a search maneuver, DO NOT
            # auto-pick the largest visible marker — that would lock the
            # observer into PD control on an already-scanned or
            # unwanted marker and override the mission's rotation/move RC.
            # The scenario: mission scans marker N, transitions to SEARCH/
            # rotate, sets search_rc=(0,0,0,yaw). Without this guard, the
            # observer sees marker N still in frame, picks it as 'chosen',
            # and holds position on it — the drone never rotates.
            with self._lock:
                src = self._search_rc
            if self.mode == "live" and any(v != 0 for v in src):
                try:
                    self._send_rc(*src)
                    snapshot["rc_sent_at"] = round(t_now, 2)
                    snapshot["search_rc"] = list(src)
                    snapshot["marker_id"] = None
                    snapshot["status_msg"] = (
                        f"searching (overriding PD on visible markers) — "
                        f"rc(lr={src[0]}, fb={src[1]}, ud={src[2]}, yaw={src[3]})"
                    )
                except Exception:
                    pass
                self._chosen_id = None
                with self._lock:
                    self.latest = snapshot
                return
            chosen = max(vid_all.items(), key=lambda kv: kv[1].get("px_size", 0))[0]
        else:
            snapshot["marker_id"] = None
            snapshot["status_msg"] = f"target marker {target_id} not visible"
            # Apply search RC even when an irrelevant marker IS visible
            with self._lock:
                src = self._search_rc
            if self.mode == "live" and any(v != 0 for v in src):
                try:
                    self._send_rc(*src)
                    snapshot["rc_sent_at"] = round(t_now, 2)
                    snapshot["search_rc"] = list(src)
                    snapshot["status_msg"] = (
                        f"searching (target {target_id} not in sight) — "
                        f"rc(lr={src[0]}, fb={src[1]}, ud={src[2]}, yaw={src[3]})"
                    )
                except Exception:
                    pass
            with self._lock:
                self.latest = snapshot
            return

        if chosen != self._chosen_id:
            self._chosen_id = chosen
            self._fx = self._fy = self._fs = self._fd = None

        info = vid_all[chosen]
        cx = float(info["center"][0])
        cy = float(info["center"][1])
        px_size = float(info["px_size"])
        left_len  = float(info.get("left_len", 0))
        right_len = float(info.get("right_len", 0))
        fw, fh = self._vid.frame_size

        # Camera measurements
        raw_err_x = (cx - fw / 2.0) / (fw / 2.0)
        raw_err_y = (cy - fh / 2.0) / (fh / 2.0)
        raw_dist  = pixel_distance_estimate(px_size, marker_size_m=p.marker_size_m)
        raw_skew  = marker_skew(left_len, right_len) if (left_len + right_len) > 0 else 0.0

        # EMA
        self._fx = raw_err_x if self._fx is None else (1 - a) * self._fx + a * raw_err_x
        self._fy = raw_err_y if self._fy is None else (1 - a) * self._fy + a * raw_err_y
        self._fs = raw_skew  if self._fs is None else (1 - a) * self._fs + a * raw_skew
        self._fd = raw_dist  if self._fd is None else (1 - a) * self._fd + a * raw_dist

        err_x, err_y, skew, est_dist = self._fx, self._fy, self._fs, self._fd

        def db(e, dz):
            if abs(e) < dz:
                return 0.0
            return e - math.copysign(dz, e)

        err_x_eff   = db(err_x, p.deadband_x)
        err_y_eff   = db(err_y, p.deadband_y)
        skew_eff    = db(skew,  p.deadband_skew)
        dist_err    = est_dist - p.hover_distance_m
        dist_err_ef = db(dist_err, p.deadband_dist_m)

        # PD control
        yaw_p_term = err_x_eff    * p.yaw_p
        yaw_d_term = -p.d_yaw * self._yaw_rate
        lr_p_term  = skew_eff     * p.skew_p
        lr_d_term  = -p.d_lr  * vy_cms
        ud_p_term  = -err_y_eff   * p.alt_p
        ud_d_term  = p.d_ud   * vz_cms
        fb_p_term  = dist_err_ef  * p.dist_p
        fb_d_term  = -p.d_fb  * vx_cms

        yaw_rc = _clamp_int(yaw_p_term + yaw_d_term, -p.yaw_max, p.yaw_max)
        lr     = _clamp_int(lr_p_term  + lr_d_term,  -p.lr_max,  p.lr_max)
        ud     = _clamp_int(ud_p_term  + ud_d_term,  -p.ud_max,  p.ud_max)
        fb     = _clamp_int(fb_p_term  + fb_d_term,  -p.fb_back_max, p.fb_max)
        if abs(yaw_rc) < p.rc_min: yaw_rc = 0
        if abs(lr)     < p.rc_min: lr     = 0
        if abs(ud)     < p.rc_min: ud     = 0
        if abs(fb)     < p.rc_min: fb     = 0

        snapshot.update({
            "marker_id": chosen,
            "px_size": round(px_size, 1),
            "center_x": round(cx, 1),
            "center_y": round(cy, 1),
            "left_len": round(left_len, 1),
            "right_len": round(right_len, 1),
            "distance_m": round(est_dist, 3),
            "raw_distance_m": round(raw_dist, 3),
            "err_x": round(err_x, 3),
            "err_y": round(err_y, 3),
            "skew":  round(skew, 3),
            "dist_err": round(dist_err, 3),
            "rc_yaw": yaw_rc, "rc_yaw_p": round(yaw_p_term, 2), "rc_yaw_d": round(yaw_d_term, 2),
            "rc_lr":  lr,     "rc_lr_p":  round(lr_p_term,  2), "rc_lr_d":  round(lr_d_term,  2),
            "rc_ud":  ud,     "rc_ud_p":  round(ud_p_term,  2), "rc_ud_d":  round(ud_d_term,  2),
            "rc_fb":  fb,     "rc_fb_p":  round(fb_p_term,  2), "rc_fb_d":  round(fb_d_term,  2),
            "status_msg": f"tracking {chosen}",
        })

        # LIVE: actually send RC (throttled to ~5 Hz)
        if self.mode == "live":
            if t_now - self._last_send_t >= 0.18:
                try:
                    self._send_rc(lr, fb, ud, yaw_rc)
                    snapshot["rc_sent_at"] = round(t_now, 2)
                except Exception as e:
                    snapshot["rc_send_error"] = repr(e)
                self._last_send_t = t_now

        with self._lock:
            self.latest = snapshot


# ─── Manager for a fleet ────────────────────────────────────────────────────


class ObserverFleet:
    """Maps drone_id → DroneObserver. Thread-safe lookup + reconfigure."""

    def __init__(self, session: Optional[requests.Session] = None,
                 allow_live: bool = True):
        self._session = session or requests.Session()
        self._allow_live = bool(allow_live)
        self._lock = threading.RLock()
        self._obs: dict[str, DroneObserver] = {}

    @property
    def allow_live(self) -> bool:
        return self._allow_live

    def configure(self, drones: dict):
        """Create/update observers to match the passed drone fleet.

        `drones` is the DRONES dict from remote_web_controller:
            { "1": {"name": "...", "base": "http://...", ...}, ... }
        """
        with self._lock:
            seen = set()
            for did, info in drones.items():
                did = str(did)
                seen.add(did)
                api_base = info.get("base", "")
                name = info.get("name", did)
                if did in self._obs:
                    obs = self._obs[did]
                    # If the URL changed, re-point it (stops the observer)
                    if obs.api_base != api_base.rstrip("/"):
                        obs.set_api_base(api_base)
                    obs.name = name
                else:
                    self._obs[did] = DroneObserver(
                        drone_id=did,
                        api_base=api_base,
                        name=name,
                        session=self._session,
                        allow_live=self._allow_live,
                    )
            # Stop & drop any observer whose drone has been removed
            for did in list(self._obs.keys()):
                if did not in seen:
                    try:
                        self._obs[did].stop()
                    except Exception:
                        pass
                    del self._obs[did]

    def get(self, drone_id: str) -> Optional[DroneObserver]:
        with self._lock:
            return self._obs.get(str(drone_id))

    def all_states(self) -> dict:
        with self._lock:
            return {did: obs.get_state() for did, obs in self._obs.items()}

    def stop_all(self):
        with self._lock:
            for obs in self._obs.values():
                try:
                    obs.stop()
                except Exception:
                    pass


# ─── Special mission: Scan-All-Markers ──────────────────────────────────────
#
# Coordinates multiple DroneObservers to find and scan every marker in a target
# set exactly once. Phases per drone:
#
#     IDLE      (not assigned to mission yet)
#     SEARCH    (observer running, scanning, hovering in place if no free marker)
#     APPROACH  (claimed a marker — observer flying to hover_distance_m)
#     HOVER     (within approach_tolerance_m of target — counting down hover secs)
#     DONE      (all target markers scanned)
#     ERROR     (unrecoverable)
#
# Collision avoidance: a shared "claimed" dict maps drone_id → marker_id.
# Two drones cannot claim the same marker. If a drone's only visible candidate
# is already claimed, it waits (RC stays zero via observer's own logic when no
# target is set — the observer simply doesn't send motion commands).


class ScanAllMarkersMission:
    """
    One instance = one running mission. Spins up a manager thread that ticks
    each participating drone's FSM at 2 Hz. Tolerates observers being stopped
    mid-mission (the FSM will recover on next tick).
    """

    TICK_S = 0.5
    # Search-maneuver tuning (used when a drone is in SEARCH but no target
    # marker is visible). The observer's _tick runs at ~5 Hz; we push a fresh
    # search RC each mission tick and the observer forwards it each frame.
    SEARCH_IDLE_BEFORE_MOVE_S = 2.0   # hover this long before starting search RC
    SEARCH_ROTATE_YAW_RC = 40         # RC yaw magnitude during rotation (≈ 60°/s).
                                      # Raised from 25 for faster scan cycles.
                                      # If drone can't detect markers mid-rotation
                                      # (motion blur) drop back to 25–30.
    SEARCH_ROTATION_TIME_S = 7.0      # one full 360° at the above rate
                                      # (7s × 60°/s ≈ 420°, some margin for deceleration)
    SEARCH_CENTER_XY = (0.0, 5.4)     # arena center (SDC field 20×10.8)
    SEARCH_CENTER_TOL_M = 0.8         # "close enough to centre" threshold
    SEARCH_CENTER_SPEED_RC = 8        # slow translate speed toward centre
    SEARCH_MAX_CYCLES = 3             # rotate+centre attempts before error

    def __init__(
        self,
        fleet: "ObserverFleet",
        drone_ids: list[str],
        target_markers: list[int],
        hover_seconds: float = 1.5,
        approach_tolerance_m: float = 0.30,
        approach_skew_tol: float = 0.12,
        approach_err_x_tol: float = 0.15,
        auto_takeoff: bool = False,
    ):
        self.fleet = fleet
        self.drone_ids = [str(d) for d in drone_ids]
        self.target_markers = [int(m) for m in target_markers]
        self.hover_seconds = float(hover_seconds)
        self.approach_tolerance_m = float(approach_tolerance_m)
        # Maximum perspective skew to accept before declaring APPROACH done.
        # 0.08 ≈ 6° off the marker normal. Below 0.04 we're inside the skew
        # deadband so the drone stops correcting anyway.
        self.approach_skew_tol = float(approach_skew_tol)
        # Maximum horizontal pixel error (normalised) before declaring the
        # drone is centred on the marker.
        self.approach_err_x_tol = float(approach_err_x_tol)
        self.auto_takeoff = bool(auto_takeoff)

        self._lock = threading.RLock()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self.scanned: set[int] = set()
        self.claimed: dict[str, int] = {}   # drone_id → marker_id
        self.drones: dict[str, dict] = {}   # drone_id → per-drone FSM state
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.error: Optional[str] = None
        # Trace logger — created on start(), closed on stop().
        self._trace: Optional[MissionTraceLogger] = None
        self.trace_path: Optional[str] = None

    # ── Lifecycle ──
    def start(self):
        with self._lock:
            if self._active:
                return False
            # Validate all drones exist
            for did in self.drone_ids:
                if self.fleet.get(did) is None:
                    self.error = f"unknown drone id {did}"
                    return False
            # Validate LIVE is allowed
            if not self.fleet.allow_live:
                self.error = "mission requires LIVE mode (REMOTE_NO_LIVE is set)"
                return False
            self._active = True
            self.started_at = time.time()
            self.ended_at = None
            self.error = None
            self.scanned.clear()
            self.claimed.clear()
            # Open mission trace log
            ts_tag = time.strftime("%Y%m%d_%H%M%S")
            log_path = MISSION_LOG_DIR / f"mission_scan_all_{ts_tag}.jsonl"
            try:
                self._trace = MissionTraceLogger(log_path, "scan_all_markers")
                self.trace_path = str(log_path)
                self._trace.write("mission_start", {
                    "drone_ids": list(self.drone_ids),
                    "target_markers": list(self.target_markers),
                    "hover_seconds": self.hover_seconds,
                    "approach_tolerance_m": self.approach_tolerance_m,
                    "approach_err_x_tol": self.approach_err_x_tol,
                    "approach_skew_tol": self.approach_skew_tol,
                    "auto_takeoff": self.auto_takeoff,
                })
                print(f"[MISSION] Trace logging to {log_path}")
            except Exception as e:
                print(f"[MISSION] Trace log init failed: {e}")
                self._trace = None
            self.drones = {
                did: {
                    "phase": "SEARCH",
                    "target": None,
                    "hover_start": None,
                    "last_marker_id": None,
                    "note": "",
                    # Search maneuver state
                    "search_sub": "idle",     # idle | rotate | center | error
                    "search_sub_start": None, # wall-clock when sub-phase began
                    "search_cycles": 0,       # how many rotate+center cycles done
                }
                for did in self.drone_ids
            }
            # Start each observer, flip to LIVE
            failed_live = []
            takeoff_errs = []
            for did in self.drone_ids:
                obs = self.fleet.get(did)
                assert obs is not None
                obs.set_target(None)          # auto-pick until we claim one
                obs.start()
                resulting_mode = obs.set_mode("live")
                if resulting_mode != "live":
                    failed_live.append(did)
                if self.auto_takeoff:
                    try:
                        r = obs.cmd_takeoff()
                        if isinstance(r, dict) and r.get("ok") is False:
                            takeoff_errs.append(f"{did}:{r.get('error') or r.get('status_code') or 'unknown'}")
                    except Exception as e:
                        takeoff_errs.append(f"{did}:{e}")
            if failed_live:
                # Roll back: stop observers we just started
                for did in self.drone_ids:
                    obs = self.fleet.get(did)
                    if obs is not None:
                        try: obs.set_mode("observe")
                        except Exception: pass
                self._active = False
                self.error = (f"could not switch {','.join(failed_live)} to LIVE — "
                              "check REMOTE_NO_LIVE env var, observer allow_live, "
                              "or that the drone API is reachable")
                return False
            if takeoff_errs:
                # Takeoff failures don't abort the mission — the mission can still
                # run if the user takes off manually — but surface the errors
                # so they appear in the UI as a warning.
                self.error = "takeoff issues: " + "; ".join(takeoff_errs)
            self._thread = threading.Thread(target=self._run, daemon=True,
                                             name="mission-scan-all")
            self._thread.start()
            return True

    def stop(self, land: bool = False):
        """Stop the mission. If land=True, send /api/land to each drone."""
        with self._lock:
            was_active = self._active
            self._active = False
            self.ended_at = time.time()
            if self._trace:
                self._trace.write("mission_stop", {
                    "land": bool(land),
                    "scanned": sorted(self.scanned),
                    "scanned_count": len(self.scanned),
                    "target_count": len(self.target_markers),
                    "claimed": dict(self.claimed),
                })
        # Out of the lock — these can block on HTTP
        if was_active:
            for did in self.drone_ids:
                obs = self.fleet.get(did)
                if obs is None:
                    continue
                try:
                    obs.set_search_rc(0, 0, 0, 0)   # cancel any active search maneuver
                except Exception:
                    pass
                try:
                    obs.set_mode("observe")     # this also pushes RC-stop
                except Exception:
                    pass
                if land:
                    try:
                        obs.set_mode("live")    # land must be in live (the RPC allows it always)
                        obs.cmd_land()
                        obs.set_mode("observe")
                    except Exception:
                        pass
        # Close the trace file last so all the per-drone shutdown events
        # above have a chance to flush first.
        if self._trace:
            try:
                self._trace.close()
            except Exception:
                pass
            self._trace = None

    # ── Status snapshot ──
    def get_status(self) -> dict:
        with self._lock:
            remaining = [m for m in self.target_markers if m not in self.scanned]
            return {
                "active": self._active,
                "drone_ids": list(self.drone_ids),
                "target_markers": list(self.target_markers),
                "hover_seconds": self.hover_seconds,
                "approach_tolerance_m": self.approach_tolerance_m,
                "scanned": sorted(self.scanned),
                "remaining": remaining,
                "progress": f"{len(self.scanned)}/{len(self.target_markers)}",
                "claimed": dict(self.claimed),
                "drones": {did: dict(state) for did, state in self.drones.items()},
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "error": self.error,
                "trace_path": self.trace_path,
            }

    # ── FSM tick loop ──
    def _run(self):
        try:
            while True:
                with self._lock:
                    active = self._active
                if not active:
                    break
                self._tick()
                time.sleep(self.TICK_S)
        except Exception as e:
            with self._lock:
                self.error = repr(e)
                self._active = False
                self.ended_at = time.time()

    def _tick(self):
        # Are we done?
        with self._lock:
            all_done = all(m in self.scanned for m in self.target_markers)
            if all_done:
                for did in self.drone_ids:
                    self.drones[did]["phase"] = "DONE"
                    self.drones[did]["note"] = "all markers scanned"
                self._active = False
                self.ended_at = time.time()
                return

        for did in self.drone_ids:
            obs = self.fleet.get(did)
            if obs is None:
                with self._lock:
                    self.drones[did]["phase"] = "ERROR"
                    self.drones[did]["note"] = "observer missing"
                continue
            self._tick_drone(did, obs)

    def _tick_drone(self, did: str, obs: "DroneObserver"):
        now = time.time()
        snap = obs.get_state()
        running = snap.get("running")
        visible = snap.get("visible_ids") or []
        marker_id = snap.get("marker_id")
        distance_m = snap.get("distance_m")

        with self._lock:
            state = self.drones[did]
            phase_before = state["phase"]
            phase = phase_before
            target = state["target"]

            # Trace every tick — this is what you hand back to diagnose
            # "drone hovered but never transitioned". Includes everything
            # the FSM decision path reads.
            if self._trace:
                self._trace.write("tick", {
                    "drone": did,
                    "phase": phase,
                    "target": target,
                    "marker_id": marker_id,
                    "distance_m": distance_m,
                    "err_x": snap.get("err_x"),
                    "err_y": snap.get("err_y"),
                    "skew":  snap.get("skew"),
                    "visible": list(visible),
                    "running": bool(running),
                    "mode": snap.get("mode"),
                    "hover_start": state.get("hover_start"),
                    "hover_elapsed": (now - state["hover_start"])
                        if state.get("hover_start") else None,
                    "hover_target_s": self.hover_seconds,
                    "search_sub": state.get("search_sub"),
                    "search_sub_elapsed": (now - state["search_sub_start"])
                        if state.get("search_sub_start") else None,
                    "search_cycles": state.get("search_cycles", 0),
                    "scanned": sorted(self.scanned),
                    "claimed": dict(self.claimed),
                    "note": state.get("note", ""),
                    "pos": snap.get("pos") or snap.get("cam"),
                    "yaw":  snap.get("yaw"),
                    "battery": snap.get("battery"),
                    "altitude_m": snap.get("altitude_m"),
                    "guard": snap.get("guard"),
                })

            # Observer not running? Try to recover
            if not running:
                try:
                    obs.start()
                    obs.set_mode("live")
                except Exception as e:
                    state["note"] = f"restart failed: {e}"
                    state["phase"] = "ERROR"
                    if self._trace:
                        self._trace.write("phase_change", {
                            "drone": did, "from": phase_before,
                            "to": "ERROR", "reason": f"observer restart failed: {e}",
                        })
                return

            if phase == "SEARCH":
                available = [
                    m for m in visible
                    if m in self.target_markers
                    and m not in self.scanned
                    and m not in self.claimed.values()
                ]
                if available:
                    # Prefer the marker already centered in view (marker_id is the
                    # observer's auto-pick: largest on screen). If that's available
                    # and in our target set, use it; otherwise pick the lowest id.
                    if marker_id in available:
                        chosen = marker_id
                    else:
                        chosen = min(available)
                    self.claimed[did] = chosen
                    obs.set_target(chosen)
                    obs.set_search_rc(0, 0, 0, 0)   # stop the search maneuver
                    state["phase"] = "APPROACH"
                    state["target"] = chosen
                    state["hover_start"] = None
                    state["search_sub"] = "idle"
                    state["search_sub_start"] = None
                    state["note"] = f"approaching marker {chosen}"
                    if self._trace:
                        self._trace.write("marker_claimed", {
                            "drone": did, "marker": chosen, "visible": list(visible),
                        })
                        self._trace.write("phase_change", {
                            "drone": did, "from": "SEARCH", "to": "APPROACH",
                            "target": chosen,
                        })
                    return

                # No free target marker visible — run the search maneuver.
                # Observer's set_target(None) keeps it in "pick-largest" mode;
                # we drive the search RC through it. A step-function state
                # machine: idle → rotate → center → rotate → center → error
                obs.set_target(None)

                sub = state.get("search_sub", "idle")
                sub_start = state.get("search_sub_start")
                idle_since = state.get("hover_start") or now

                if sub == "idle":
                    # Quick grace period — maybe a marker pops in within 2s.
                    if state.get("hover_start") is None:
                        state["hover_start"] = now
                        state["note"] = "searching — waiting for markers"
                    elif (now - state["hover_start"]) >= self.SEARCH_IDLE_BEFORE_MOVE_S:
                        state["search_sub"] = "rotate"
                        state["search_sub_start"] = now
                        state["note"] = "rotating to scan 360°"
                    obs.set_search_rc(0, 0, 0, 0)
                elif sub == "rotate":
                    # Slow yaw rotation for a full 360°
                    if sub_start is None:
                        state["search_sub_start"] = now
                        sub_start = now
                    obs.set_search_rc(0, 0, 0, self.SEARCH_ROTATE_YAW_RC)
                    elapsed = now - sub_start
                    state["note"] = (
                        f"rotating {elapsed:.1f}/{self.SEARCH_ROTATION_TIME_S:.0f}s — "
                        f"no unclaimed targets visible"
                    )
                    if elapsed >= self.SEARCH_ROTATION_TIME_S:
                        # Didn't find anything in 360° → move to arena centre
                        state["search_sub"] = "center"
                        state["search_sub_start"] = now
                        state["search_cycles"] = state.get("search_cycles", 0) + 1
                        state["note"] = "moving to arena centre to widen search"
                elif sub == "center":
                    # Compute translation toward the arena centre.
                    # NOTE: observer provides drone position in ARENA frame
                    # via _pos (SSE from /api/position/events). Use it if
                    # available; otherwise fall back to a blind slow-forward.
                    pos = snap.get("pos") or snap.get("cam") or None
                    arena_cx, arena_cy = self.SEARCH_CENTER_XY
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        dx = arena_cx - float(pos[0])
                        dy = arena_cy - float(pos[1])
                        dist = (dx * dx + dy * dy) ** 0.5
                    else:
                        dx = dy = 0.0
                        dist = 0.0  # no position → skip centering
                    if dist < self.SEARCH_CENTER_TOL_M or pos is None:
                        # At (or near) centre, OR no position known — back to rotate
                        obs.set_search_rc(0, 0, 0, 0)
                        state["search_sub"] = "rotate"
                        state["search_sub_start"] = now
                        state["note"] = (
                            f"at arena centre, rotating (cycle "
                            f"{state.get('search_cycles', 1)})"
                        )
                        if state.get("search_cycles", 0) >= self.SEARCH_MAX_CYCLES:
                            # Give up — drone can't find any target marker
                            state["search_sub"] = "error"
                            state["note"] = (
                                f"exhausted {self.SEARCH_MAX_CYCLES} search cycles"
                                " — marker(s) unreachable"
                            )
                    else:
                        # Translate toward centre in WORLD frame. lr=east, fb=forward
                        # For the Anafi with yaw=0 ≈ arena +y, fb maps to +y and
                        # lr maps to +x. This is a rough approximation — with
                        # yaw ≠ 0 the drone's body axes differ, but the search
                        # maneuver doesn't need high precision (we just need to
                        # move away from the wall and eventually see markers).
                        scale = min(1.0, dist / 2.0)   # ease out near centre
                        rc_fb = int(self.SEARCH_CENTER_SPEED_RC * scale * (dy / max(dist, 0.01)))
                        rc_lr = int(self.SEARCH_CENTER_SPEED_RC * scale * (dx / max(dist, 0.01)))
                        obs.set_search_rc(rc_lr, rc_fb, 0, 0)
                        state["note"] = (
                            f"moving to centre — dist={dist:.2f}m "
                            f"(dx={dx:+.2f}, dy={dy:+.2f})"
                        )
                elif sub == "error":
                    # Stay parked
                    obs.set_search_rc(0, 0, 0, 0)
                return

            if phase == "APPROACH":
                if target is None:
                    state.update(phase="SEARCH", search_sub="idle",
                                 search_sub_start=None, hover_start=None)
                    return
                if marker_id != target:
                    # Lost sight of the target marker — HARD BRAKE + STOP-AND-SPIN
                    # 1. Release the claim
                    # 2. Zero the observer's target → no PD on any visible marker
                    # 3. Push a single rc_stop so momentum is cancelled
                    # 4. Immediately start the rotate sub-phase so the drone
                    #    begins searching from its current position
                    self.claimed.pop(did, None)
                    obs.set_target(None)
                    try: obs.cmd_rc_stop()
                    except Exception: pass
                    obs.set_search_rc(0, 0, 0, self.SEARCH_ROTATE_YAW_RC)
                    state.update(phase="SEARCH", target=None, hover_start=None,
                                 search_sub="rotate", search_sub_start=now,
                                 note=f"lost sight of {target} — brake + rotate to re-acquire")
                    if self._trace:
                        self._trace.write("marker_lost", {
                            "drone": did, "lost": target,
                            "visible_now": list(visible),
                            "phase_was": "APPROACH",
                        })
                    return
                if distance_m is None:
                    return
                target_dist = obs.params.hover_distance_m
                err_x_mag = abs(snap.get("err_x", 1))
                skew_mag  = abs(snap.get("skew", 1))
                # APPROACH → HOVER: the ONLY hard requirement is distance within
                # ±approach_tolerance_m. Horizontal alignment (err_x) and
                # perpendicularity (skew) are nice-to-have indicators shown in
                # the status line, but they don't block the transition — the
                # observer's PD loop keeps correcting them once HOVER starts,
                # and a hard err_x/skew gate kept drones stuck "approaching"
                # forever when the scene had marginal conditions.
                dist_ok = abs(distance_m - target_dist) <= self.approach_tolerance_m
                skew_ok = skew_mag <= self.approach_skew_tol
                # Only commit to HOVER when the drone is both at the right
                # distance AND roughly head-on. The observer's PD loop is
                # now aggressive enough (dist_p=15, fb_back_max=30) to
                # actively back up and strafe into the marker's normal,
                # so this gate is reachable in reasonable time.
                if dist_ok and skew_ok:
                    state["phase"] = "HOVER"
                    state["hover_start"] = time.time()
                    state["note"] = (f"hovering on {target} "
                                     f"(dist={distance_m:.2f}m, skew={skew_mag:.2f})")
                    if self._trace:
                        self._trace.write("phase_change", {
                            "drone": did, "from": "APPROACH", "to": "HOVER",
                            "target": target,
                            "distance_m": round(distance_m, 3),
                            "target_dist_m": target_dist,
                            "err_x": round(err_x_mag, 3),
                            "skew": round(skew_mag, 3),
                        })
                else:
                    # Narrate the specific blocker so the operator can tune if
                    # we get stuck. "not perpendicular" means the drone needs
                    # to strafe / back-up to set up a head-on approach.
                    blockers = []
                    if not dist_ok:
                        blockers.append(
                            f"dist {distance_m:.2f}→{target_dist:.2f}m "
                            f"(tol ±{self.approach_tolerance_m:.2f})")
                    if not skew_ok:
                        blockers.append(
                            f"|skew|={skew_mag:.2f}>{self.approach_skew_tol} "
                            "(not perpendicular — strafing)")
                    state["note"] = (f"approaching {target}: "
                                     + "; ".join(blockers))
                return

            if phase == "HOVER":
                if target is None:
                    state.update(phase="SEARCH", search_sub="idle",
                                 search_sub_start=None, hover_start=None)
                    return
                if marker_id != target:
                    # Lost sight mid-hover — brake immediately and start
                    # rotating to reacquire a target marker.
                    self.claimed.pop(did, None)
                    obs.set_target(None)
                    try: obs.cmd_rc_stop()
                    except Exception: pass
                    obs.set_search_rc(0, 0, 0, self.SEARCH_ROTATE_YAW_RC)
                    state.update(phase="SEARCH", target=None, hover_start=None,
                                 search_sub="rotate", search_sub_start=now,
                                 note=f"lost sight of {target} during hover — brake + rotate")
                    if self._trace:
                        self._trace.write("marker_lost", {
                            "drone": did, "lost": target,
                            "visible_now": list(visible),
                            "phase_was": "HOVER",
                        })
                    return
                elapsed = time.time() - (state["hover_start"] or time.time())
                if elapsed >= self.hover_seconds:
                    # Scanned! Record the success, then CONTINUOUS-FLIGHT:
                    # if another unclaimed target is already in view, skip
                    # the rotate sub-phase entirely and go straight to
                    # APPROACH on the next marker. No stop, no rotation,
                    # seamless flow from one marker to the next.
                    self.scanned.add(target)
                    self.claimed.pop(did, None)

                    next_available = [
                        m for m in visible
                        if m in self.target_markers
                        and m not in self.scanned
                        and m not in self.claimed.values()
                        and m != target          # never re-chain the just-scanned one
                    ]

                    if self._trace:
                        self._trace.write("marker_scanned", {
                            "drone": did, "marker": target,
                            "hover_elapsed": round(elapsed, 3),
                            "scanned_count": len(self.scanned),
                            "remaining": sorted(set(self.target_markers) - self.scanned),
                            "next_available": next_available,
                        })

                    if next_available:
                        # Prefer the marker currently centred in view if it's in
                        # our candidate set; otherwise lowest id.
                        if marker_id in next_available:
                            chosen = marker_id
                        else:
                            chosen = min(next_available)
                        self.claimed[did] = chosen
                        obs.set_target(chosen)
                        obs.set_search_rc(0, 0, 0, 0)  # cancel any pending search RC
                        state.update(phase="APPROACH", target=chosen,
                                     hover_start=None,
                                     search_sub="idle", search_sub_start=None,
                                     search_cycles=0,
                                     note=f"scanned {target}, chaining → {chosen}")
                        if self._trace:
                            self._trace.write("marker_claimed", {
                                "drone": did, "marker": chosen,
                                "visible": list(visible),
                                "chained_from": target,
                            })
                            self._trace.write("phase_change", {
                                "drone": did, "from": "HOVER", "to": "APPROACH",
                                "target": chosen, "reason": "chained_from_hover",
                            })
                    else:
                        # No next target visible → start rotating to find one.
                        obs.set_target(None)
                        state.update(phase="SEARCH", target=None, hover_start=None,
                                     search_sub="rotate", search_sub_start=now,
                                     search_cycles=0,
                                     note=f"scanned marker {target} — rotating to find next")
                        obs.set_search_rc(0, 0, 0, self.SEARCH_ROTATE_YAW_RC)
                        if self._trace:
                            self._trace.write("phase_change", {
                                "drone": did, "from": "HOVER", "to": "SEARCH",
                                "sub": "rotate", "reason": "scan_complete",
                            })
                else:
                    remaining = self.hover_seconds - elapsed
                    state["note"] = f"hovering on {target}: {remaining:.1f}s left"
                return


# ═══════════════════════════════════════════════════════════════════════════
#  CaptureAllTargetsMission — SDC26 capture-the-box mission
# ═══════════════════════════════════════════════════════════════════════════
#
# Per SDC26 rules (v9): 6 boxes on the field, 3 per team zone. To capture
# an enemy box, a drone hovers over it for ≥ 2 s without a defending drone
# present. The capturing drone then must return to the team's home area for
# the point to count.
#
# This mission flies the swarm one-by-one over the configured targets, holds
# 1.5 m above each for 4 s (conservative vs the 2-s rule), then returns to
# the home zone. Camera is kept pointed at the arena centre throughout so
# the position tracker has as many markers as possible in view for
# triangulation.
#
# FSM per drone:
#   IDLE → NAV_TO_TARGET → HOVER → NAV_TO_HOME → DONE
#
# Shared claim map prevents two drones from attacking the same box.

class CaptureAllTargetsMission:
    """Fly each selected drone to a sequence of target boxes, hover 1.5 m
    above each for `hover_seconds` seconds, then return home.

    target_boxes is a list of dicts like:
        [{"id": 1, "x": -5.0, "y": 2.0}, ...]
    """

    TICK_S = 0.5

    def __init__(
        self,
        fleet: "ObserverFleet",
        drone_ids: list[str],
        target_boxes: list[dict],
        home_xy: tuple[float, float] = (0.0, 1.5),
        arena_face_xy: tuple[float, float] = (0.0, 5.4),
        hover_above_m: float = 1.5,
        hover_seconds: float = 4.0,
        nav_tol_xy_m: float = 0.3,
        nav_tol_z_m: float = 0.3,
        auto_takeoff: bool = False,
    ):
        self.fleet = fleet
        self.drone_ids = [str(d) for d in drone_ids]
        # Normalise and index boxes
        self.target_boxes = []
        for i, b in enumerate(target_boxes or []):
            try:
                self.target_boxes.append({
                    "idx": i,
                    "id": b.get("id", i + 1),
                    "x":  float(b["x"]),
                    "y":  float(b["y"]),
                    "home_team": b.get("home_team"),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self.home_xy = (float(home_xy[0]), float(home_xy[1]))
        self.arena_face_xy = (float(arena_face_xy[0]), float(arena_face_xy[1]))
        self.hover_above_m = float(hover_above_m)
        self.hover_seconds = float(hover_seconds)
        self.nav_tol_xy_m = float(nav_tol_xy_m)
        self.nav_tol_z_m  = float(nav_tol_z_m)
        self.auto_takeoff = bool(auto_takeoff)

        self._lock = threading.RLock()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self.captured: set[int] = set()       # idx of boxes hovered over
        self.claimed: dict[str, int] = {}     # drone_id → box idx
        self.drones: dict[str, dict] = {}
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.error: Optional[str] = None
        self._trace: Optional[MissionTraceLogger] = None
        self.trace_path: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────
    def start(self):
        with self._lock:
            if self._active:
                return False
            if not self.target_boxes:
                self.error = "no target_boxes configured"
                return False
            for did in self.drone_ids:
                if self.fleet.get(did) is None:
                    self.error = f"unknown drone id {did}"
                    return False
            if not self.fleet.allow_live:
                self.error = "mission requires LIVE mode (REMOTE_NO_LIVE is set)"
                return False
            self._active = True
            self.started_at = time.time()
            self.ended_at = None
            self.error = None
            self.captured.clear()
            self.claimed.clear()
            self.drones = {
                did: {
                    "phase": "IDLE",
                    "target_idx": None,
                    "hover_start": None,
                    "note": "",
                    "boxes_done": 0,
                }
                for did in self.drone_ids
            }
            ts_tag = time.strftime("%Y%m%d_%H%M%S")
            log_path = MISSION_LOG_DIR / f"mission_capture_{ts_tag}.jsonl"
            try:
                self._trace = MissionTraceLogger(log_path, "capture_all_targets")
                self.trace_path = str(log_path)
                self._trace.write("mission_start", {
                    "drone_ids": list(self.drone_ids),
                    "target_boxes": list(self.target_boxes),
                    "home_xy": self.home_xy,
                    "arena_face_xy": self.arena_face_xy,
                    "hover_above_m": self.hover_above_m,
                    "hover_seconds": self.hover_seconds,
                    "auto_takeoff": self.auto_takeoff,
                })
                print(f"[MISSION] Capture-targets trace → {log_path}")
            except Exception as e:
                print(f"[MISSION] Trace init failed: {e}")
                self._trace = None

            # Start every observer in LIVE, always face arena centre
            failed = []
            for did in self.drone_ids:
                obs = self.fleet.get(did)
                obs.set_target(None)
                obs.start()
                if obs.set_mode("live") != "live":
                    failed.append(did)
                # Aim camera at arena centre from the start
                obs.set_waypoint(None, self.arena_face_xy)
                if self.auto_takeoff:
                    try: obs.cmd_takeoff()
                    except Exception: pass
            if failed:
                for did in self.drone_ids:
                    o = self.fleet.get(did)
                    if o: o.set_mode("observe")
                self._active = False
                self.error = f"could not switch {','.join(failed)} to LIVE"
                return False
            self._thread = threading.Thread(target=self._run, daemon=True,
                                             name="mission-capture")
            self._thread.start()
            return True

    def stop(self, land: bool = False):
        with self._lock:
            was_active = self._active
            self._active = False
            self.ended_at = time.time()
            if self._trace:
                self._trace.write("mission_stop", {
                    "land": bool(land),
                    "captured_count": len(self.captured),
                    "target_count": len(self.target_boxes),
                })
        if was_active:
            for did in self.drone_ids:
                obs = self.fleet.get(did)
                if obs is None:
                    continue
                try:
                    obs.clear_waypoint()
                    obs.set_search_rc(0, 0, 0, 0)
                    obs.set_mode("observe")
                except Exception:
                    pass
                if land:
                    try:
                        obs.set_mode("live")
                        obs.cmd_land()
                        obs.set_mode("observe")
                    except Exception:
                        pass
        if self._trace:
            try: self._trace.close()
            except Exception: pass
            self._trace = None

    # ── Status snapshot ──
    def get_status(self) -> dict:
        with self._lock:
            remaining = [b for b in self.target_boxes
                         if b["idx"] not in self.captured]
            return {
                "active": self._active,
                "kind": "capture_all_targets",
                "drone_ids": list(self.drone_ids),
                "target_boxes": list(self.target_boxes),
                "home_xy": list(self.home_xy),
                "arena_face_xy": list(self.arena_face_xy),
                "hover_above_m": self.hover_above_m,
                "hover_seconds": self.hover_seconds,
                "captured": sorted(self.captured),
                "remaining": [b["id"] for b in remaining],
                "progress": f"{len(self.captured)}/{len(self.target_boxes)}",
                "claimed": dict(self.claimed),
                "drones": {did: dict(state) for did, state in self.drones.items()},
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "error": self.error,
                "trace_path": self.trace_path,
            }

    # ── FSM loop ──
    def _run(self):
        while True:
            with self._lock:
                if not self._active:
                    break
                all_done = all(
                    d["phase"] == "DONE" for d in self.drones.values()
                )
            if all_done:
                with self._lock:
                    self._active = False
                    self.ended_at = time.time()
                if self._trace:
                    self._trace.write("mission_end_auto", {"reason": "all_done"})
                    try: self._trace.close()
                    except Exception: pass
                    self._trace = None
                # Clear waypoints on all drones
                for did in self.drone_ids:
                    obs = self.fleet.get(did)
                    if obs: obs.clear_waypoint()
                break
            for did in self.drone_ids:
                obs = self.fleet.get(did)
                if obs is None:
                    continue
                self._tick_drone(did, obs)
            time.sleep(self.TICK_S)

    def _pick_next_box(self, did: str, current_pos) -> Optional[int]:
        """Pick the closest unclaimed, unvisited box for this drone."""
        cands = [b for b in self.target_boxes
                 if b["idx"] not in self.captured
                 and b["idx"] not in self.claimed.values()]
        if not cands:
            return None
        if current_pos is None:
            return cands[0]["idx"]
        cx, cy = float(current_pos[0]), float(current_pos[1])
        cands.sort(key=lambda b: (b["x"] - cx) ** 2 + (b["y"] - cy) ** 2)
        return cands[0]["idx"]

    def _tick_drone(self, did: str, obs: "DroneObserver"):
        now = time.time()
        snap = obs.get_state()
        running = snap.get("running")
        pos = snap.get("pos") or snap.get("cam")
        if not running:
            try:
                obs.start()
                obs.set_mode("live")
            except Exception:
                pass
            return

        with self._lock:
            state = self.drones[did]
            phase = state["phase"]

            # Select next target if IDLE
            if phase == "IDLE":
                idx = self._pick_next_box(did, pos)
                if idx is None:
                    # Nothing left to capture → go home
                    state.update(phase="NAV_TO_HOME",
                                 target_idx=None,
                                 note="all boxes captured, returning home")
                    if self._trace:
                        self._trace.write("phase_change", {
                            "drone": did, "from": "IDLE", "to": "NAV_TO_HOME",
                            "reason": "no_unclaimed_boxes",
                        })
                    return
                self.claimed[did] = idx
                box = self.target_boxes[idx]
                state.update(phase="NAV_TO_TARGET", target_idx=idx,
                             hover_start=None,
                             note=f"flying to box {box['id']} @ "
                                  f"({box['x']:.1f},{box['y']:.1f})")
                # Set waypoint: box xy, hover_above_m altitude, camera at arena centre
                obs.set_waypoint((box["x"], box["y"], self.hover_above_m),
                                 self.arena_face_xy)
                if self._trace:
                    self._trace.write("box_claimed", {
                        "drone": did, "box_idx": idx, "box_id": box["id"],
                        "box_xy": [box["x"], box["y"]],
                    })
                    self._trace.write("phase_change", {
                        "drone": did, "from": "IDLE", "to": "NAV_TO_TARGET",
                        "target": box["id"],
                    })
                return

            if phase == "NAV_TO_TARGET":
                idx = state["target_idx"]
                if idx is None or idx not in range(len(self.target_boxes)):
                    state["phase"] = "IDLE"
                    return
                box = self.target_boxes[idx]
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    state["note"] = f"→ box {box['id']}: no position fix"
                    return
                dx = box["x"] - float(pos[0])
                dy = box["y"] - float(pos[1])
                dist_xy = (dx * dx + dy * dy) ** 0.5
                cur_z = float(pos[2]) if len(pos) >= 3 else (snap.get("altitude_m") or 0)
                dz = self.hover_above_m - cur_z
                if dist_xy <= self.nav_tol_xy_m and abs(dz) <= self.nav_tol_z_m:
                    # Arrived — start the capture hover
                    state.update(phase="HOVER", hover_start=now,
                                 note=f"over box {box['id']} — capturing")
                    if self._trace:
                        self._trace.write("phase_change", {
                            "drone": did, "from": "NAV_TO_TARGET", "to": "HOVER",
                            "target": box["id"],
                            "dist_xy": round(dist_xy, 2),
                            "dz": round(dz, 2),
                        })
                else:
                    state["note"] = (
                        f"→ box {box['id']} d={dist_xy:.2f}m "
                        f"dz={dz:+.2f}m"
                    )
                return

            if phase == "HOVER":
                idx = state["target_idx"]
                if idx is None:
                    state["phase"] = "NAV_TO_HOME"
                    return
                box = self.target_boxes[idx]
                elapsed = now - (state["hover_start"] or now)
                if elapsed >= self.hover_seconds:
                    # Captured!
                    self.captured.add(idx)
                    self.claimed.pop(did, None)
                    state["boxes_done"] += 1
                    # Decide next action: another box or home?
                    nxt = self._pick_next_box(did, pos)
                    if nxt is not None:
                        self.claimed[did] = nxt
                        nbox = self.target_boxes[nxt]
                        state.update(phase="NAV_TO_TARGET", target_idx=nxt,
                                     hover_start=None,
                                     note=f"captured {box['id']}, → box {nbox['id']}")
                        obs.set_waypoint(
                            (nbox["x"], nbox["y"], self.hover_above_m),
                            self.arena_face_xy)
                        if self._trace:
                            self._trace.write("box_captured", {
                                "drone": did, "box_idx": idx, "box_id": box["id"],
                                "hover_elapsed": round(elapsed, 2),
                            })
                            self._trace.write("box_claimed", {
                                "drone": did, "box_idx": nxt, "box_id": nbox["id"],
                            })
                    else:
                        # No more boxes → return home
                        obs.set_waypoint((self.home_xy[0], self.home_xy[1],
                                          self.hover_above_m),
                                         self.arena_face_xy)
                        state.update(phase="NAV_TO_HOME",
                                     target_idx=None, hover_start=None,
                                     note=f"captured {box['id']} — returning home")
                        if self._trace:
                            self._trace.write("box_captured", {
                                "drone": did, "box_idx": idx, "box_id": box["id"],
                                "hover_elapsed": round(elapsed, 2),
                            })
                            self._trace.write("phase_change", {
                                "drone": did, "from": "HOVER", "to": "NAV_TO_HOME",
                                "reason": "all_done",
                            })
                else:
                    remaining = self.hover_seconds - elapsed
                    state["note"] = f"capturing box {box['id']}: {remaining:.1f}s left"
                return

            if phase == "NAV_TO_HOME":
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    state["note"] = "returning home (no pos fix)"
                    return
                dx = self.home_xy[0] - float(pos[0])
                dy = self.home_xy[1] - float(pos[1])
                dist_xy = (dx * dx + dy * dy) ** 0.5
                if dist_xy <= self.nav_tol_xy_m:
                    obs.clear_waypoint()
                    state.update(phase="DONE",
                                 note=f"arrived home ({state['boxes_done']} boxes)")
                    if self._trace:
                        self._trace.write("phase_change", {
                            "drone": did, "from": "NAV_TO_HOME", "to": "DONE",
                            "boxes_done": state["boxes_done"],
                        })
                else:
                    state["note"] = f"→ home d={dist_xy:.2f}m"
                return

            # DONE — do nothing


class MissionManager:
    """Single-slot holder for the currently-running special mission.

    Only one mission runs at a time. Replacing an active mission stops the old."""

    def __init__(self, fleet: "ObserverFleet"):
        self.fleet = fleet
        self._lock = threading.RLock()
        self._current: Optional[ScanAllMarkersMission] = None

    @property
    def current(self) -> Optional[ScanAllMarkersMission]:
        return self._current

    def start_scan_all(self, drone_ids: list[str], target_markers: list[int],
                       hover_seconds: float = 1.5,
                       approach_tolerance_m: float = 0.30,
                       approach_skew_tol: float = 0.12,
                       approach_err_x_tol: float = 0.15,
                       auto_takeoff: bool = False) -> tuple[bool, str]:
        with self._lock:
            if self._current is not None and self._current._active:
                return False, "a mission is already running — stop it first"
            m = ScanAllMarkersMission(
                self.fleet, drone_ids, target_markers,
                hover_seconds=hover_seconds,
                approach_tolerance_m=approach_tolerance_m,
                approach_skew_tol=approach_skew_tol,
                approach_err_x_tol=approach_err_x_tol,
                auto_takeoff=auto_takeoff,
            )
            ok = m.start()
            if not ok:
                return False, m.error or "failed to start"
            self._current = m
            return True, "mission started"

    def start_capture_all_targets(
        self, drone_ids: list[str], target_boxes: list[dict],
        home_xy: tuple = (0.0, 1.5),
        arena_face_xy: tuple = (0.0, 5.4),
        hover_above_m: float = 1.5,
        hover_seconds: float = 4.0,
        nav_tol_xy_m: float = 0.3,
        auto_takeoff: bool = False,
    ) -> tuple[bool, str]:
        """Launch the capture-all-targets mission (SDC26 box-capture scoring)."""
        with self._lock:
            if self._current is not None and self._current._active:
                return False, "a mission is already running — stop it first"
            m = CaptureAllTargetsMission(
                self.fleet, drone_ids, target_boxes,
                home_xy=home_xy,
                arena_face_xy=arena_face_xy,
                hover_above_m=hover_above_m,
                hover_seconds=hover_seconds,
                nav_tol_xy_m=nav_tol_xy_m,
                auto_takeoff=auto_takeoff,
            )
            ok = m.start()
            if not ok:
                return False, m.error or "failed to start"
            self._current = m
            return True, "mission started"

    def stop(self, land: bool = False) -> bool:
        with self._lock:
            m = self._current
        if m is None:
            return False
        m.stop(land=land)
        return True

    def status(self) -> dict:
        with self._lock:
            m = self._current
        if m is None:
            return {"active": False, "has_mission": False}
        out = m.get_status()
        out["has_mission"] = True
        return out
