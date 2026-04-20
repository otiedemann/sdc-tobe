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

import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

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

    RC_TICK_MS = 100

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

    def _send_rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> dict:
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
        while self._running:
            try:
                self._tick()
            except Exception as e:
                with self._lock:
                    self.latest["error"] = repr(e)
            time.sleep(0.1)  # 10 Hz update

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
    SEARCH_ROTATE_YAW_RC = 25         # RC yaw magnitude during rotation (≈ 37°/s)
    SEARCH_ROTATION_TIME_S = 12.0     # one full 360° at the above rate
    SEARCH_CENTER_XY = (0.0, 5.4)     # arena center (SDC field 20×10.8)
    SEARCH_CENTER_TOL_M = 0.8         # "close enough to centre" threshold
    SEARCH_CENTER_SPEED_RC = 8        # slow translate speed toward centre
    SEARCH_MAX_CYCLES = 3             # rotate+centre attempts before error

    def __init__(
        self,
        fleet: "ObserverFleet",
        drone_ids: list[str],
        target_markers: list[int],
        hover_seconds: float = 3.0,
        approach_tolerance_m: float = 0.35,
        auto_takeoff: bool = False,
    ):
        self.fleet = fleet
        self.drone_ids = [str(d) for d in drone_ids]
        self.target_markers = [int(m) for m in target_markers]
        self.hover_seconds = float(hover_seconds)
        self.approach_tolerance_m = float(approach_tolerance_m)
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
        snap = obs.get_state()
        running = snap.get("running")
        visible = snap.get("visible_ids") or []
        marker_id = snap.get("marker_id")
        distance_m = snap.get("distance_m")

        with self._lock:
            state = self.drones[did]
            phase = state["phase"]
            target = state["target"]

            # Observer not running? Try to recover
            if not running:
                try:
                    obs.start()
                    obs.set_mode("live")
                except Exception as e:
                    state["note"] = f"restart failed: {e}"
                    state["phase"] = "ERROR"
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
                    # Lost sight — release claim, go back to searching
                    self.claimed.pop(did, None)
                    obs.set_target(None)
                    obs.set_search_rc(0, 0, 0, 0)
                    state.update(phase="SEARCH", target=None, hover_start=None,
                                 search_sub="idle", search_sub_start=None,
                                 note=f"lost sight of {target}, re-searching")
                    return
                if distance_m is None:
                    return
                target_dist = obs.params.hover_distance_m
                if abs(distance_m - target_dist) <= self.approach_tolerance_m \
                        and abs(snap.get("err_x", 1)) < 0.15:
                    state["phase"] = "HOVER"
                    state["hover_start"] = time.time()
                    state["note"] = f"hovering on {target}"
                else:
                    state["note"] = f"approaching {target}: {distance_m:.2f} m → {target_dist:.2f} m"
                return

            if phase == "HOVER":
                if target is None:
                    state.update(phase="SEARCH", search_sub="idle",
                                 search_sub_start=None, hover_start=None)
                    return
                if marker_id != target:
                    self.claimed.pop(did, None)
                    obs.set_target(None)
                    obs.set_search_rc(0, 0, 0, 0)
                    state.update(phase="SEARCH", target=None, hover_start=None,
                                 search_sub="idle", search_sub_start=None,
                                 note=f"lost sight of {target} during hover, re-searching")
                    return
                elapsed = time.time() - (state["hover_start"] or time.time())
                if elapsed >= self.hover_seconds:
                    # Scanned!
                    self.scanned.add(target)
                    self.claimed.pop(did, None)
                    obs.set_target(None)
                    obs.set_search_rc(0, 0, 0, 0)
                    state.update(phase="SEARCH", target=None, hover_start=None,
                                 search_sub="idle", search_sub_start=None,
                                 search_cycles=0,
                                 note=f"scanned marker {target}")
                else:
                    remaining = self.hover_seconds - elapsed
                    state["note"] = f"hovering on {target}: {remaining:.1f}s left"
                return


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
                       hover_seconds: float = 3.0,
                       approach_tolerance_m: float = 0.35,
                       auto_takeoff: bool = False) -> tuple[bool, str]:
        with self._lock:
            if self._current is not None and self._current._active:
                return False, "a mission is already running — stop it first"
            m = ScanAllMarkersMission(
                self.fleet, drone_ids, target_markers,
                hover_seconds=hover_seconds,
                approach_tolerance_m=approach_tolerance_m,
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
