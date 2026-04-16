#!/usr/bin/env python3
"""
ArUco Seek — Live Tuning Web Console
=====================================

Tiny Flask web server that runs the HOVER pipeline of aruco_seek.py
and exposes a live tuning UI in the browser.

Features
--------
  * MJPEG video feed (proxied from the drone API server)
  * Top-down canvas drawing of the drone's relative position to the marker
  * Live numerical readouts: distance, IMU velocities, attitude, pixel
    error, perspective skew, and the RC commands the controller computes
  * Sliders / number-inputs for every HOVER_* tuning constant — applied
    live, no observer restart needed
  * Start / Stop button for the observer pipeline
  * Two modes:
      OBSERVE — never sends takeoff / land / RC; safe for hand-held tuning
      LIVE    — actually sends RC commands so the drone hovers in front
                of the marker. ENABLED BY DEFAULT; pass --no-live for safe
                hand-held tuning only.

Usage
-----
    # Default — LIVE toggle available in the UI:
    python3 tools/aruco_seek_web.py --api http://flightctrl1:8080

    # Safe hand-held tuning — LIVE disabled:
    python3 tools/aruco_seek_web.py --api http://flightctrl1:8080 --no-live

    open http://localhost:8095
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from flask import Flask, Response, jsonify, request

# Reuse all the heavy lifting from aruco_seek.py
sys.path.insert(0, str(Path(__file__).parent))
import aruco_seek as ASEEK  # noqa: E402
from aruco_seek import (  # noqa: E402
    CALIB_A,
    CALIB_B,
    MARKER_SIZE_M,
    PositionListener,
    VideoMarkerTracker,
    api_get,
    api_post,
    marker_skew,
    pixel_distance_estimate,
    send_rc,
    send_rc_stop,
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
    # Camera HFOV — used ONLY by the top-down drawing, not by the controller
    cam_hfov_deg: float    = 69.0


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


# ─── Observer ───────────────────────────────────────────────────────────────


class Observer:
    """Background pipeline: camera + IMU + position SSE + HOVER PD math."""

    def __init__(self, api_base: str, allow_live: bool = False):
        self.api_base = api_base.rstrip("/")
        self._lock = threading.RLock()
        self.params = HoverParams()
        self.target_id: Optional[int] = None  # None = auto-pick largest visible
        self._chosen_id: Optional[int] = None
        self.latest: dict = {"running": False}
        # Mode: "observe" = compute only, NEVER send RC.
        #       "live"    = compute AND send RC (drone WILL move).
        # Always starts in observe — switching to live is an explicit action.
        self.mode: str = "observe"
        # Safety gate: live mode is impossible unless the launcher passed
        # --allow-live on the command line.
        self.allow_live: bool = bool(allow_live)
        self._last_send_t: float = 0.0

        self._vid: Optional[VideoMarkerTracker] = None
        self._pos: Optional[PositionListener] = None
        self._thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False

        # Filter / yaw-rate state (reset whenever target changes)
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._fs: Optional[float] = None
        self._fd: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._prev_yaw_t: float = 0.0
        self._yaw_rate: float = 0.0

    # ── Param management ──
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
                    if isinstance(cur, int) and not isinstance(cur, bool):
                        new_val = int(round(float(v)))
                    else:
                        new_val = float(v)
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

    def set_mode(self, mode: str) -> str:
        """Switch between 'observe' and 'live'. Returns the actual mode set."""
        mode = (mode or "").lower()
        if mode not in ("observe", "live"):
            return self.mode
        # Hard safety gate — live mode is impossible without --allow-live
        if mode == "live" and not self.allow_live:
            return self.mode
        with self._lock:
            prev = self.mode
            self.mode = mode
        # When leaving live mode, send a single zero-RC to stop motion
        if prev == "live" and mode == "observe":
            try:
                send_rc_stop()
            except Exception:
                pass
        return mode

    # ── Direct manual commands (only meaningful when the drone is connected) ──
    def cmd_takeoff(self) -> dict:
        return api_post("/api/takeoff")

    def cmd_land(self) -> dict:
        return api_post("/api/land")

    def cmd_emergency(self) -> dict:
        return api_post("/api/emergency")

    def cmd_rc_stop(self) -> dict:
        return send_rc_stop() or {"ok": True}

    # ── State snapshot ──
    def get_state(self) -> dict:
        with self._lock:
            return dict(self.latest)

    # ── Lifecycle ──
    def start(self):
        if self._running:
            return
        # Configure aruco_seek module globals
        ASEEK._api_base = self.api_base
        ASEEK._abort.clear()
        # Bring the drone-side pipeline up
        try:
            api_post("/api/video/start", {"mode": "mjpeg"})
            api_post("/api/position/config", {"enabled": True})
        except Exception:
            pass
        self._vid = VideoMarkerTracker(self.api_base)
        self._pos = PositionListener(self.api_base)
        self._vid.start()
        self._pos.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="observer")
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True, name="observer-hb")
        self._thread.start()
        self._hb_thread.start()

    def stop(self):
        # If we were sending RC, zero it before tearing down trackers
        if self.mode == "live":
            try:
                send_rc_stop()
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
                "mode": self.mode,
                "allow_live": self.allow_live,
            }

    def _hb_loop(self):
        while self._running:
            try:
                api_get("/api/heartbeat", timeout=0.5)
            except Exception:
                pass
            time.sleep(0.4)

    # ── Main loop ──
    def _run(self):
        # Wait briefly for first frame
        time.sleep(1.0)
        while self._running:
            try:
                self._tick()
            except Exception as e:
                with self._lock:
                    self.latest["error"] = repr(e)
            time.sleep(0.1)  # 10 Hz update rate

    def _tick(self):
        if self._vid is None:
            return
        with self._lock:
            p = HoverParams(**asdict(self.params))  # snapshot
            target_id = self.target_id

        snapshot: dict = {
            "running": True,
            "mode": self.mode,
            "allow_live": self.allow_live,
            "frame_w": self._vid.frame_size[0],
            "frame_h": self._vid.frame_size[1],
            "vid_active": self._vid.is_active,
            "vid_age": round(self._vid.age, 2),
        }

        vid_all = self._vid.get_all()
        snapshot["visible_ids"] = sorted(vid_all.keys())

        # Always poll IMU even when no marker
        tel = api_get("/api/telemetry")
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
            with self._lock:
                self.latest = snapshot
            return

        # Pick target marker
        if target_id is not None and target_id in vid_all:
            chosen = target_id
        elif target_id is None:
            chosen = max(vid_all.items(), key=lambda kv: kv[1].get("px_size", 0))[0]
        else:
            # User wants a specific ID but it's not visible
            snapshot["marker_id"] = None
            snapshot["status_msg"] = f"target marker {target_id} not visible"
            with self._lock:
                self.latest = snapshot
            return

        if chosen != self._chosen_id:
            self._chosen_id = chosen
            self._fx = self._fy = self._fs = self._fd = None  # reset filters

        info = vid_all[chosen]
        cx = float(info["center"][0])
        cy = float(info["center"][1])
        px_size = float(info["px_size"])
        left_len  = float(info.get("left_len", 0))
        right_len = float(info.get("right_len", 0))
        fw, fh = self._vid.frame_size

        # ── Camera measurements ──
        raw_err_x = (cx - fw / 2.0) / (fw / 2.0)
        raw_err_y = (cy - fh / 2.0) / (fh / 2.0)
        raw_dist  = pixel_distance_estimate(px_size)
        raw_skew  = marker_skew(left_len, right_len) if (left_len + right_len) > 0 else 0.0

        # EMA filters
        self._fx = raw_err_x if self._fx is None else (1 - a) * self._fx + a * raw_err_x
        self._fy = raw_err_y if self._fy is None else (1 - a) * self._fy + a * raw_err_y
        self._fs = raw_skew  if self._fs is None else (1 - a) * self._fs + a * raw_skew
        self._fd = raw_dist  if self._fd is None else (1 - a) * self._fd + a * raw_dist

        err_x, err_y, skew, est_dist = self._fx, self._fy, self._fs, self._fd

        # Deadbands
        def db(e, dz):
            if abs(e) < dz:
                return 0.0
            return e - math.copysign(dz, e)

        err_x_eff   = db(err_x, p.deadband_x)
        err_y_eff   = db(err_y, p.deadband_y)
        skew_eff    = db(skew,  p.deadband_skew)
        dist_err    = est_dist - p.hover_distance_m
        dist_err_ef = db(dist_err, p.deadband_dist_m)

        # PD math
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
            # Computed RC commands and their P/D breakdown.
            # In observe mode they are NEVER sent; in live mode they ARE sent.
            "rc_yaw": yaw_rc, "rc_yaw_p": round(yaw_p_term, 2), "rc_yaw_d": round(yaw_d_term, 2),
            "rc_lr":  lr,     "rc_lr_p":  round(lr_p_term,  2), "rc_lr_d":  round(lr_d_term,  2),
            "rc_ud":  ud,     "rc_ud_p":  round(ud_p_term,  2), "rc_ud_d":  round(ud_d_term,  2),
            "rc_fb":  fb,     "rc_fb_p":  round(fb_p_term,  2), "rc_fb_d":  round(fb_d_term,  2),
            "mode": self.mode,
            "status_msg": f"tracking {chosen}",
        })

        # ── LIVE: actually send RC commands ──
        # Throttle to ~5 Hz max to avoid overwhelming the API server.
        if self.mode == "live":
            now_t = time.time()
            if now_t - self._last_send_t >= 0.18:  # ~5.5 Hz
                try:
                    send_rc(lr, fb, ud, yaw_rc)
                    snapshot["rc_sent_at"] = round(now_t, 2)
                except Exception as e:
                    snapshot["rc_send_error"] = repr(e)
                self._last_send_t = now_t

        with self._lock:
            self.latest = snapshot


# ─── Flask app ───────────────────────────────────────────────────────────────


app = Flask(__name__)
observer: Optional[Observer] = None


HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ArUco Seek — Tuning Console</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 12px; }
  h1 { font-size: 18px; margin: 0 0 12px 0; color: #38bdf8; }
  h2 { font-size: 12px; margin: 14px 0 6px 0; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
  .grid { display: grid; grid-template-columns: minmax(420px, 1fr) 1fr; gap: 14px; align-items: start; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; }
  .row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; }
  .row label { width: 180px; color: #cbd5e1; flex-shrink: 0; }
  .row input[type=range] { flex: 1; min-width: 100px; }
  .row input[type=number] { width: 70px; background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 3px; padding: 2px 4px; font-size: 11px; }
  .stat { font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; line-height: 1.6; }
  .stat .k { color: #94a3b8; display: inline-block; min-width: 90px; }
  .stat .v { color: #e2e8f0; }
  .stat .pd { color: #64748b; font-size: 10px; }
  .stat b { color: #38bdf8; font-weight: 600; }
  button { background: #1e3a5f; border: 1px solid #3b82f6; color: #e2e8f0; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: inherit; }
  button.danger { background: #7f1d1d; border-color: #ef4444; }
  button.success { background: #065f46; border-color: #10b981; }
  button:hover { filter: brightness(1.2); }
  img.video { width: 100%; max-width: 640px; border-radius: 4px; background: #0f172a; min-height: 200px; }
  canvas { background: #0b1220; border-radius: 4px; display: block; margin: 0 auto; }
  .top-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  .top-bar .warn { margin-left: auto; color: #fbbf24; font-size: 11px; }
  #status_dot { font-size: 14px; }
  .pgroup { border-top: 1px dashed #334155; margin-top: 8px; padding-top: 6px; }
  .pgroup-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  /* Mode toggle */
  .mode-seg { display: inline-flex; border: 1px solid #334155; border-radius: 4px; overflow: hidden; }
  .mode-seg button { background: transparent; border: 0; border-radius: 0; padding: 6px 14px; font-weight: 600; color: #94a3b8; }
  .mode-seg button.active.observe { background: #065f46; color: #ecfdf5; }
  .mode-seg button.active.live    { background: #b91c1c; color: #fee2e2; }
  .mode-seg button:disabled { opacity: 0.45; cursor: not-allowed; }
  /* Manual command buttons (only visible in live mode) */
  .manual-bar { display: none; gap: 6px; align-items: center; margin-left: 8px; padding: 4px 6px; background: #450a0a; border: 1px solid #ef4444; border-radius: 4px; }
  .manual-bar.show { display: inline-flex; }
  .manual-bar button { font-size: 11px; padding: 4px 8px; }
  /* LIVE banner — unmissable when active */
  #live_banner { display: none; background: #b91c1c; color: #fee2e2; padding: 8px 12px; border-radius: 4px; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 10px; box-shadow: 0 0 0 2px #fbbf24 inset; text-align: center; }
  #live_banner.show { display: block; animation: pulse 1.6s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 2px #fbbf24 inset; } 50% { box-shadow: 0 0 0 4px #fbbf24 inset; } }
  body.live-mode { box-shadow: 0 0 0 4px #b91c1c inset; }
  .rc-sent { color: #f87171; font-weight: 600; }
</style>
</head><body>

<h1>ArUco Seek — Live Tuning Console
  <span id="status_dot" style="color:#94a3b8;">○</span>
  <span id="status_text" style="font-size:12px;color:#94a3b8;">stopped</span>
</h1>

<div id="live_banner">⚠ LIVE MODE — DRONE WILL MOVE — RC commands are being sent</div>

<div class="top-bar">
  <button id="btn_start" class="success">▶ Start</button>
  <button id="btn_stop" class="danger">■ Stop</button>

  <span style="margin-left:14px;color:#94a3b8;font-size:11px;">Mode:</span>
  <span class="mode-seg">
    <button id="btn_mode_observe" class="active observe" data-mode="observe">OBSERVE</button>
    <button id="btn_mode_live"    data-mode="live">LIVE</button>
  </span>
  <span id="mode_gate_note" style="font-size:11px;color:#fbbf24;display:none;">live disabled — restart with --allow-live</span>

  <span id="manual_bar" class="manual-bar">
    <button id="btn_takeoff" class="success">⬆ Takeoff</button>
    <button id="btn_land">⬇ Land</button>
    <button id="btn_rc_stop">RC Stop</button>
    <button id="btn_emergency" class="danger">⛔ EMERGENCY</button>
  </span>

  <span style="margin-left:14px;color:#94a3b8;font-size:11px;">Target marker:</span>
  <input id="target_marker" type="number" placeholder="auto" min="0" style="width:70px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;padding:4px;border-radius:4px;font-size:12px;" />
  <button id="btn_target">Lock</button>
  <button id="btn_target_auto">Auto</button>
  <button id="btn_reload_params" style="margin-left:8px;">↻ Reload params</button>
</div>

<div class="grid">

  <div>
    <div class="panel">
      <h2>Video Feed</h2>
      <img id="video" class="video" alt="(start observer to load)" />
    </div>

    <div class="panel" style="margin-top:12px;">
      <h2>Top-Down View — drone ↔ marker</h2>
      <canvas id="topdown" width="420" height="380"></canvas>
    </div>

    <div class="panel" style="margin-top:12px;">
      <h2>Live Readout</h2>
      <div id="readout" class="stat">—</div>
    </div>
  </div>

  <div class="panel">
    <h2>Live Tuning Parameters</h2>
    <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">All changes apply immediately to the running observer.</div>
    <div id="params"></div>
  </div>

</div>

<script>
// ── Slider definitions: [key, label, min, max, step, group] ──
const PGROUPS = ['Mission target', 'Camera filter / deadbands', 'P gains (camera)', 'D gains (IMU damping)', 'Output clamps', 'Drawing'];
const SLIDERS = [
  ['hover_distance_m', 'Hover distance (m)',           0.5, 4.0, 0.05, 0],

  ['ema_alpha',        'EMA α (smoothing, lower = smoother)', 0.05, 0.95, 0.05, 1],
  ['deadband_x',       'Deadband err_x',                0.00, 0.30, 0.01, 1],
  ['deadband_y',       'Deadband err_y',                0.00, 0.30, 0.01, 1],
  ['deadband_skew',    'Deadband skew',                 0.00, 0.30, 0.01, 1],
  ['deadband_dist_m',  'Deadband distance (m)',         0.00, 1.00, 0.05, 1],

  ['yaw_p',            'P · yaw      (per err_x)',      0,   50,    1,    2],
  ['skew_p',           'P · lateral  (per skew)',       0,   50,    1,    2],
  ['alt_p',            'P · altitude (per err_y)',      0,  100,    1,    2],
  ['dist_p',           'P · distance (per dist err)',   0,   30,    0.5,  2],

  ['d_yaw',            'D · yaw      (per °/s)',        0,    2,    0.05, 3],
  ['d_lr',             'D · lateral  (per cm/s vgy)',   0,    2,    0.05, 3],
  ['d_ud',             'D · vertical (per cm/s vgz)',   0,    2,    0.05, 3],
  ['d_fb',             'D · fwd/back (per cm/s vgx)',   0,    2,    0.05, 3],

  ['yaw_max',          'Clamp · yaw RC max',            0,   50,    1,    4],
  ['lr_max',           'Clamp · lateral RC max',        0,   30,    1,    4],
  ['ud_max',           'Clamp · vertical RC max',       0,   50,    1,    4],
  ['fb_max',           'Clamp · forward RC max',        0,   30,    1,    4],
  ['fb_back_max',      'Clamp · backward RC max',       0,   30,    1,    4],
  ['rc_min',           'RC dead-floor (|rc| <)',        0,   10,    1,    4],

  ['cam_hfov_deg',     'Camera HFOV (drawing only)',    30,  110,    1,    5],
];

let params = {};

async function loadParams() {
  const r = await fetch('/params');
  params = await r.json();
  renderParams();
}
function renderParams() {
  const cont = document.getElementById('params');
  cont.innerHTML = '';
  let curGroup = -1;
  SLIDERS.forEach(([k, label, min, max, step, group]) => {
    if (group !== curGroup) {
      curGroup = group;
      const hdr = document.createElement('div');
      hdr.className = 'pgroup';
      hdr.innerHTML = '<div class="pgroup-label">' + PGROUPS[group] + '</div>';
      cont.appendChild(hdr);
    }
    const v = params[k] ?? 0;
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `
      <label title="${k}">${label}</label>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${v}" data-k="${k}" />
      <input type="number" min="${min}" max="${max}" step="${step}" value="${v}" data-k="${k}" />
    `;
    cont.appendChild(row);
  });
  cont.querySelectorAll('input').forEach(el => {
    el.addEventListener('input', () => {
      const k = el.dataset.k;
      const v = parseFloat(el.value);
      if (isNaN(v)) return;
      params[k] = v;
      cont.querySelectorAll(`input[data-k="${k}"]`).forEach(s => { if (s !== el) s.value = v; });
    });
    el.addEventListener('change', () => {
      const k = el.dataset.k;
      const v = parseFloat(el.value);
      if (!isNaN(v)) saveParam(k, v);
    });
  });
}
async function saveParam(k, v) {
  await fetch('/params', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({[k]: v})});
}

// ── Polling ──
let lastMode = 'observe';
let allowLive = false;

function applyModeUI(mode, allow) {
  // Cache for buttons that fire before next poll
  lastMode = mode || 'observe';
  if (typeof allow === 'boolean') allowLive = allow;

  // Mode segmented control
  const bObs  = document.getElementById('btn_mode_observe');
  const bLive = document.getElementById('btn_mode_live');
  bObs.classList.toggle('active', lastMode === 'observe');
  bLive.classList.toggle('active', lastMode === 'live');
  bLive.classList.toggle('live',   lastMode === 'live');
  bLive.disabled = !allowLive;
  document.getElementById('mode_gate_note').style.display = allowLive ? 'none' : 'inline';

  // Banner + body border + manual bar visibility
  const live = (lastMode === 'live');
  document.getElementById('live_banner').classList.toggle('show', live);
  document.body.classList.toggle('live-mode', live);
  document.getElementById('manual_bar').classList.toggle('show', live);
}

async function poll() {
  try {
    const r = await fetch('/state');
    const s = await r.json();
    const dot = document.getElementById('status_dot');
    const txt = document.getElementById('status_text');
    if (s.running) { dot.textContent = '●'; dot.style.color = '#22c55e'; txt.textContent = s.status_msg || 'running'; }
    else           { dot.textContent = '○'; dot.style.color = '#94a3b8'; txt.textContent = 'stopped'; }
    applyModeUI(s.mode || 'observe', s.allow_live);
    renderReadout(s);
    drawTopDown(s);
  } catch (e) {}
}
function f(x, n) { if (x === undefined || x === null) return '—'; return Number(x).toFixed(n); }
function renderReadout(s) {
  if (!s.running) {
    document.getElementById('readout').innerHTML = '<span style="color:#64748b;">stopped — press Start</span>';
    return;
  }
  const html = `
<b>Marker</b><br>
<span class="k">ID:</span><span class="v">${s.marker_id ?? '—'}</span>
<span class="k">visible:</span><span class="v">${(s.visible_ids||[]).join(', ') || '—'}</span><br>
<span class="k">distance:</span><span class="v">${f(s.distance_m,2)} m</span>
  <span class="pd">(raw ${f(s.raw_distance_m,2)} m, target ${f(params.hover_distance_m,2)} m)</span><br>
<span class="k">px size:</span><span class="v">${f(s.px_size,0)}</span>
<span class="k">center:</span><span class="v">(${f(s.center_x,0)}, ${f(s.center_y,0)}) / ${s.frame_w}×${s.frame_h}</span><br>
<span class="k">err_x:</span><span class="v">${f(s.err_x,3)}</span>
<span class="k">err_y:</span><span class="v">${f(s.err_y,3)}</span>
<span class="k">skew:</span><span class="v">${f(s.skew,3)}</span><br>

<br><b>IMU</b><br>
<span class="k">v_body:</span><span class="v">vgx ${f(s.vx_cms,0)}, vgy ${f(s.vy_cms,0)}, vgz ${f(s.vz_cms,0)} cm/s</span><br>
<span class="k">yaw rate:</span><span class="v">${f(s.yaw_rate_dps,0)} °/s</span><br>
<span class="k">attitude:</span><span class="v">pitch ${f(s.pitch,1)}°, roll ${f(s.roll,1)}°, yaw ${f(s.yaw,1)}°</span><br>
<span class="k">altitude:</span><span class="v">${f(s.altitude_m,2)} m</span>
<span class="k">battery:</span><span class="v">${s.battery ?? '—'} %</span><br>

${s.mode === 'live'
  ? `<br><b class="rc-sent">RC — SENT to drone</b> <span class="pd">(P = camera, D = IMU damping)</span>`
  : `<br><b>RC — would-send</b> <span class="pd">(observe — not sent)</span>`}<br>
<span class="k">lr:</span><span class="v">${s.rc_lr ?? '—'}</span>
  <span class="pd">(P=${f(s.rc_lr_p,2)}  D=${f(s.rc_lr_d,2)})</span><br>
<span class="k">fb:</span><span class="v">${s.rc_fb ?? '—'}</span>
  <span class="pd">(P=${f(s.rc_fb_p,2)}  D=${f(s.rc_fb_d,2)})</span><br>
<span class="k">ud:</span><span class="v">${s.rc_ud ?? '—'}</span>
  <span class="pd">(P=${f(s.rc_ud_p,2)}  D=${f(s.rc_ud_d,2)})</span><br>
<span class="k">yaw:</span><span class="v">${s.rc_yaw ?? '—'}</span>
  <span class="pd">(P=${f(s.rc_yaw_p,2)}  D=${f(s.rc_yaw_d,2)})</span><br>
${s.rc_sent_at ? `<span class="k">last sent:</span><span class="v rc-sent">${(Date.now()/1000 - s.rc_sent_at).toFixed(1)} s ago</span><br>` : ''}
${s.rc_send_error ? `<span class="k">send err:</span><span class="v" style="color:#fca5a5;">${s.rc_send_error}</span><br>` : ''}
`;
  document.getElementById('readout').innerHTML = html;
}

function drawTopDown(s) {
  const c = document.getElementById('topdown');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#0b1220';
  ctx.fillRect(0,0,W,H);

  // Coordinate system
  const cx = W / 2;
  const marker_y = 36;
  const target = params.hover_distance_m || 1.5;
  const maxDist = Math.max(target * 2.2, 3.0);
  const ppm = (H - 80) / maxDist;          // pixels per metre

  // ── Distance grid (concentric arcs) ──
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let d = 0.5; d <= maxDist; d += 0.5) {
    ctx.beginPath();
    ctx.arc(cx, marker_y, d * ppm, 0, Math.PI, false);
    ctx.stroke();
  }
  // Distance labels
  ctx.fillStyle = '#334155';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  for (let d = 1; d <= maxDist; d += 1) {
    ctx.fillText(d + 'm', cx + 4, marker_y + d * ppm - 2);
  }

  // ── Target hover arc (cyan) ──
  ctx.strokeStyle = '#0ea5e9';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(cx, marker_y, target * ppm, 0, Math.PI, false);
  ctx.stroke();
  ctx.setLineDash([]);

  // ── Marker (green bar at top, with normal arrow) ──
  const mw = 60;
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(cx - mw, marker_y); ctx.lineTo(cx + mw, marker_y);
  ctx.stroke();
  ctx.strokeStyle = '#22c55ec0';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, marker_y); ctx.lineTo(cx, marker_y + 24);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - 5, marker_y + 18); ctx.lineTo(cx, marker_y + 24); ctx.lineTo(cx + 5, marker_y + 18);
  ctx.stroke();
  ctx.fillStyle = '#22c55e';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('marker ' + (s.marker_id ?? '?'), cx + mw + 6, marker_y + 4);

  if (!s.running || s.distance_m == null) {
    ctx.fillStyle = '#475569';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(s.running ? 'no marker visible' : 'stopped', W/2, H/2);
    return;
  }

  // ── Drone position ──
  // skew > 0 → drone is LEFT of marker normal → drone_x < cx
  const dist = s.distance_m;
  const lateral_m = -(s.skew || 0) * dist;     // metres, +ve right
  const drone_x = cx + lateral_m * ppm;
  const drone_y = marker_y + dist * ppm;

  // Drone yaw offset (from "facing the marker"):
  //   err_x > 0 → marker appears RIGHT of camera centre → drone is yawed LEFT.
  const hfov = (params.cam_hfov_deg || 69) * Math.PI / 180;
  const yaw_off_rad = Math.atan((s.err_x || 0) * Math.tan(hfov / 2));
  // Direction in canvas frame from drone → marker
  const dxv = cx - drone_x;
  const dyv = marker_y - drone_y;
  const aimAng = Math.atan2(dyv, dxv);
  // Drone's nose direction = aim direction rotated CCW by yaw_off_rad
  const droneAng = aimAng - yaw_off_rad;

  // Line of sight (drone → marker centre)
  ctx.strokeStyle = '#fbbf2470';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(drone_x, drone_y); ctx.lineTo(cx, marker_y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Camera FOV cone (the actual frustum the drone sees)
  ctx.strokeStyle = '#38bdf833';
  ctx.fillStyle   = '#38bdf815';
  const fovLen = ppm * Math.max(2.5, dist + 1.0);
  ctx.save();
  ctx.translate(drone_x, drone_y);
  ctx.rotate(droneAng);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(fovLen * Math.cos(-hfov/2), fovLen * Math.sin(-hfov/2));
  ctx.lineTo(fovLen * Math.cos( hfov/2), fovLen * Math.sin( hfov/2));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // Drone (triangle, nose along droneAng = local +x)
  ctx.save();
  ctx.translate(drone_x, drone_y);
  ctx.rotate(droneAng);
  ctx.fillStyle = '#fbbf24';
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(15, 0);
  ctx.lineTo(-10, -10);
  ctx.lineTo(-10, 10);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // velocity vector (body-frame vgx fwd, vgy right)
  const vx = s.vx_cms || 0, vy = s.vy_cms || 0;
  const vmag = Math.hypot(vx, vy);
  if (vmag > 2) {
    const vscale = 0.7;  // canvas px per cm/s
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(vx * vscale, vy * vscale);  // body frame: +x fwd, +y right
    ctx.stroke();
  }
  ctx.restore();

  // Numerical labels
  ctx.fillStyle = '#fbbf24';
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(dist.toFixed(2) + ' m', (drone_x + cx) / 2 + 14, (drone_y + marker_y) / 2 + 4);

  ctx.fillStyle = '#cbd5e1';
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('lateral: ' + lateral_m.toFixed(2) + ' m', 8, H - 38);
  ctx.fillText('yaw off: ' + (yaw_off_rad * 180 / Math.PI).toFixed(1) + '°', 8, H - 22);
  ctx.fillText('dist err: ' + (dist - target).toFixed(2) + ' m', 8, H - 6);
  ctx.textAlign = 'right';
  ctx.fillText('target ' + target.toFixed(2) + ' m', W - 8, H - 38);
  ctx.fillText('skew '   + (s.skew  || 0).toFixed(3),   W - 8, H - 22);
  ctx.fillText('err_x '  + (s.err_x || 0).toFixed(3),   W - 8, H - 6);
}

// ── Buttons ──
document.getElementById('btn_start').onclick = async () => {
  await fetch('/start', {method:'POST'});
  document.getElementById('video').src = '/video.mjpg?t=' + Date.now();
};
document.getElementById('btn_stop').onclick = async () => {
  await fetch('/stop', {method:'POST'});
  document.getElementById('video').src = '';
};
document.getElementById('btn_target').onclick = async () => {
  const v = document.getElementById('target_marker').value;
  await fetch('/target', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({id: v ? parseInt(v) : null})});
};
document.getElementById('btn_target_auto').onclick = async () => {
  document.getElementById('target_marker').value = '';
  await fetch('/target', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({id: null})});
};
document.getElementById('btn_reload_params').onclick = loadParams;

// ── Mode toggle ──
async function setMode(mode) {
  const r = await fetch('/mode', {method:'POST', headers:{'Content-Type':'application/json'},
                                  body: JSON.stringify({mode})});
  const j = await r.json();
  if (!j.ok && j.error) alert('Mode change refused: ' + j.error);
  applyModeUI(j.mode || mode, allowLive);
}
document.getElementById('btn_mode_observe').onclick = () => setMode('observe');
document.getElementById('btn_mode_live').onclick = () => {
  if (!allowLive) {
    alert('LIVE mode is disabled. Restart with --allow-live to enable.');
    return;
  }
  const ok = confirm(
    '⚠ SWITCH TO LIVE MODE?\n\n' +
    'The drone WILL receive RC commands as soon as a marker is tracked.\n' +
    'Make sure the area is clear and you are ready to take over.\n\n' +
    'Click OK to enable LIVE mode.'
  );
  if (ok) setMode('live');
};

// ── Manual commands (live mode only) ──
async function postCmd(path, confirmMsg) {
  if (confirmMsg && !confirm(confirmMsg)) return;
  const r = await fetch(path, {method:'POST'});
  const j = await r.json();
  if (!j.ok && j.error) alert(path + ' refused: ' + j.error);
}
document.getElementById('btn_takeoff').onclick   = () => postCmd('/takeoff',   'Send TAKEOFF to the drone?');
document.getElementById('btn_land').onclick      = () => postCmd('/land',      'Send LAND to the drone?');
document.getElementById('btn_rc_stop').onclick   = () => postCmd('/rc_stop',   null);
document.getElementById('btn_emergency').onclick = () => postCmd('/emergency', '⛔ EMERGENCY STOP — cut motors immediately. Confirm?');

loadParams();
setInterval(poll, 200);
poll();  // initial paint so mode UI is correct before observer starts
</script>
</body></html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/state")
def state():
    if observer is None:
        return jsonify({"running": False, "status_msg": "observer not initialised"})
    return jsonify(observer.get_state())


@app.route("/params", methods=["GET"])
def get_params():
    if observer is None:
        return jsonify(asdict(HoverParams()))
    return jsonify(observer.get_params())


@app.route("/params", methods=["POST"])
def post_params():
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"})
    data = request.get_json(force=True, silent=True) or {}
    applied = observer.update_params(data)
    return jsonify({"ok": True, "applied": applied})


@app.route("/start", methods=["POST"])
def start():
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"})
    observer.start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    if observer is None:
        return jsonify({"ok": False})
    observer.stop()
    return jsonify({"ok": True})


@app.route("/target", methods=["POST"])
def target():
    if observer is None:
        return jsonify({"ok": False})
    data = request.get_json(force=True, silent=True) or {}
    mid = data.get("id")
    if mid is not None:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            mid = None
    observer.set_target(mid)
    return jsonify({"ok": True, "target": mid})


# ─── LIVE-mode routes ────────────────────────────────────────────────────────
# These actually move the drone. /mode is gated by Observer.allow_live, and
# the manual command routes only proxy if mode is currently "live".


@app.route("/mode", methods=["POST"])
def post_mode():
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"})
    data = request.get_json(force=True, silent=True) or {}
    requested = (data.get("mode") or "").lower()
    if requested == "live" and not observer.allow_live:
        return jsonify({
            "ok": False,
            "mode": observer.mode,
            "error": "live mode disabled — restart with --allow-live to enable",
        }), 403
    actual = observer.set_mode(requested)
    return jsonify({"ok": actual == requested, "mode": actual})


def _require_live():
    """Refuse manual commands unless we're explicitly in live mode."""
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"}), 503
    if observer.mode != "live":
        return jsonify({
            "ok": False,
            "error": f"refused — current mode is '{observer.mode}', switch to 'live' first",
        }), 409
    return None


@app.route("/takeoff", methods=["POST"])
def post_takeoff():
    err = _require_live()
    if err is not None:
        return err
    return jsonify({"ok": True, "result": observer.cmd_takeoff()})


@app.route("/land", methods=["POST"])
def post_land():
    err = _require_live()
    if err is not None:
        return err
    return jsonify({"ok": True, "result": observer.cmd_land()})


@app.route("/emergency", methods=["POST"])
def post_emergency():
    # Emergency is the one command we DO allow even outside live mode —
    # if the drone is already flying, the user must always be able to kill it.
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"}), 503
    return jsonify({"ok": True, "result": observer.cmd_emergency()})


@app.route("/rc_stop", methods=["POST"])
def post_rc_stop():
    # Same exception as /emergency — always allowed.
    if observer is None:
        return jsonify({"ok": False, "error": "observer not initialised"}), 503
    return jsonify({"ok": True, "result": observer.cmd_rc_stop()})


@app.route("/video.mjpg")
def video_proxy():
    """Pass-through MJPEG stream from the drone API server to the browser."""
    if observer is None:
        return Response(b"", status=503)
    upstream_url = f"{observer.api_base}/api/position/video"
    try:
        upstream = requests.get(upstream_url, stream=True, timeout=10)
    except Exception as e:
        return Response(f"upstream error: {e}".encode(), status=502, mimetype="text/plain")

    content_type = upstream.headers.get(
        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
    )

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        except (GeneratorExit, requests.exceptions.RequestException):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    return Response(generate(), content_type=content_type)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    global observer
    p = argparse.ArgumentParser(
        description="ArUco Seek — live tuning web console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default — LIVE mode is available (but starts in OBSERVE until toggled):
    %(prog)s --api http://flightctrl1:8080

    # Safe hand-held tuning (LIVE disabled):
    %(prog)s --api http://flightctrl1:8080 --no-live

    %(prog)s --api http://localhost:8080            # against a sim API server

Then open:  http://localhost:8095
        """,
    )
    p.add_argument("--api", default=ASEEK.DEFAULT_API_BASE,
                   help=f"Drone API base URL (default: {ASEEK.DEFAULT_API_BASE})")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=8095, help="Bind port (default 8095)")
    p.add_argument("--auto-start", action="store_true",
                   help="Start the observer automatically on launch")
    p.add_argument("--allow-live", dest="allow_live", action="store_true", default=True,
                   help="Enable LIVE mode in the UI (drone WILL move when toggled). [default]")
    p.add_argument("--no-live", dest="allow_live", action="store_false",
                   help="Disable LIVE mode — only OBSERVE mode is selectable. "
                        "Use this for safe hand-held tuning.")
    args = p.parse_args()

    observer = Observer(args.api, allow_live=args.allow_live)
    if args.auto_start:
        observer.start()

    print(f"\n  ArUco Seek tuning console")
    print(f"  Drone API:  {args.api}")
    print(f"  Web UI:     http://{args.host}:{args.port}")
    if args.allow_live:
        print(f"  ⚠ LIVE mode AVAILABLE — toggle in UI to start sending RC commands.")
    else:
        print(f"  OBSERVE-only (--no-live) — no takeoff / land / RC commands will be sent.")
    print()

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
