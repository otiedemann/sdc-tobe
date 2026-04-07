#!/usr/bin/env python3
"""
Smoke-test ArUco video triangulation and visualize it in a browser.

Usage:
  python3 controller_unified/aruco_video_smoke_web.py \
    --video controller_unified/recordings/rec_raw_1775134672.mp4
"""

import argparse
import base64
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_headless_aruco():
    pi_mod_dir = _repo_root() / "aruco-position" / "control-unit"
    if str(pi_mod_dir) not in sys.path:
        sys.path.insert(0, str(pi_mod_dir))
    from pi_position import HeadlessAruCoPositioning  # pylint: disable=import-error

    return HeadlessAruCoPositioning


def _default_camera_matrix(frame_w: int, frame_h: int) -> np.ndarray:
    f = frame_w * 0.9
    return np.array(
        [[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def _load_calibration(path: Optional[Path], frame_w: int, frame_h: int):
    if path and path.exists():
        try:
            npz = np.load(str(path))
            return np.array(npz["camera_matrix"], dtype=np.float64), np.array(
                npz["dist_coeffs"], dtype=np.float64
            )
        except Exception as exc:
            print(f"[SMOKE] Failed to load calibration '{path}': {exc}")
    return _default_camera_matrix(frame_w, frame_h), np.zeros((5, 1), dtype=np.float64)


def _arena_default() -> Dict[str, Any]:
    return {
        "arena_width_m": 20.0,
        "arena_height_m": 10.0,
        "arena_origin_x": -10.0,
        "arena_origin_y": 0.0,
        "marker_size_m": 0.18,
        "markers": [],
    }


def _load_arena_config(path: Path) -> Dict[str, Any]:
    cfg = _arena_default()
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                cfg.update(raw)
        except Exception as exc:
            print(f"[SMOKE] Failed to load arena config '{path}': {exc}")
    return cfg


def _arena_z_bounds(markers: List[Dict[str, Any]]):
    if not markers:
        return -1.0, 5.0
    z_vals = [float(m.get("z", 0.0)) for m in markers]
    z_min = min(z_vals) - 0.5
    z_max = max(z_vals) + 0.5
    if z_max - z_min < 1.0:
        z_max = z_min + 1.0
    return z_min, z_max


def _apply_arena_to_processor(processor: Any, arena_cfg: Dict[str, Any]):
    markers = arena_cfg.get("markers", [])
    mk_size = float(arena_cfg.get("marker_size_m", 0.18))
    if markers:
        marker_positions = {}
        marker_wall_types = {}
        for m in markers:
            mid = int(m["id"])
            marker_positions[mid] = np.array(
                [float(m.get("x", 0.0)), float(m.get("y", 0.0)), float(m.get("z", 0.0))]
            )
            marker_wall_types[mid] = str(m.get("wall", "front"))
        processor.marker_positions = marker_positions
        processor.marker_wall_type = marker_wall_types
    processor.marker_size = mk_size
    half = mk_size / 2.0
    processor.MARKER_3D_POINTS = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )


def _json_safe(val: Any) -> Any:
    if val is None or isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, np.floating):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return str(val)


def _finite_vec3(vec: Any) -> Optional[List[float]]:
    try:
        arr = np.asarray(vec, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    xyz = arr[:3]
    if not np.isfinite(xyz).all():
        return None
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _encode_frame_jpeg_b64(frame: np.ndarray, width: int = 640, quality: int = 55) -> Optional[str]:
    try:
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return None
        width = max(64, int(width))
        if w > width:
            scale = width / float(w)
            frame = cv2.resize(frame, (width, max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(max(25, min(95, quality)))])
        if not ok:
            return None
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return None


def _reset_playback_state(state: Dict[str, Any]):
    state["frame_idx"] = 0
    state["video_time_s"] = 0.0
    state["pose_count"] = 0
    state["stale_count"] = 0
    state["latest_pose"] = None
    state["trajectory"] = []
    state["video_frame_jpeg"] = None


def _build_app(state: Dict[str, Any], state_lock: threading.Lock, control_cv: threading.Condition) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AruCo Video Smoke Test</title>
  <style>
    body { margin: 0; background: #0f172a; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
    #hud {
      position: fixed; left: 12px; top: 12px;
      background: rgba(2,6,23,0.86); border: 1px solid rgba(148,163,184,0.25);
      padding: 10px 12px; border-radius: 10px; z-index: 20; min-width: 350px;
      max-width: 620px;
    }
    #status { font-size: 13px; line-height: 1.4; white-space: pre-wrap; }
    #err { margin-top: 6px; color: #fda4af; font-size: 12px; white-space: pre-wrap; }
    #controls { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .btn {
      border: 1px solid #334155; background: #0b1220; color: #e2e8f0;
      border-radius: 8px; padding: 4px 8px; cursor: pointer; font-size: 12px;
    }
    .btn.active { border-color: #38bdf8; color: #7dd3fc; }
    #videoWrap {
      position: fixed; right: 12px; top: 12px; width: min(42vw, 560px);
      border: 1px solid rgba(148,163,184,0.35); border-radius: 10px; overflow: hidden;
      background: rgba(2,6,23,0.9); z-index: 20; display: none;
    }
    #videoFrame { display: block; width: 100%; height: auto; }
    #videoLabel { padding: 6px 8px; font-size: 12px; color: #93c5fd; border-top: 1px solid rgba(148,163,184,0.2); }
    #view { width: 100vw; height: 100vh; display: block; }
    @media (max-width: 900px) {
      #hud { max-width: calc(100vw - 24px); min-width: unset; }
      #videoWrap { width: calc(100vw - 24px); top: auto; bottom: 12px; }
    }
  </style>
</head>
<body>
  <div id="hud">
    <div style="font-weight:700; margin-bottom:6px;">AruCo Triangulation Smoke Test</div>
    <div id="status">waiting for data...</div>
    <div id="controls">
      <button class="btn" id="btnPlay">Play</button>
      <button class="btn" id="btnPause">Pause</button>
      <button class="btn" id="btnStep">Step +1</button>
      <button class="btn" id="btnRestart">Restart</button>
      <span style="margin-left:6px; font-size:12px; color:#94a3b8;">Speed:</span>
      <button class="btn speedBtn" data-speed="0.25">0.25x</button>
      <button class="btn speedBtn" data-speed="0.5">0.5x</button>
      <button class="btn speedBtn" data-speed="1">1x</button>
      <button class="btn speedBtn" data-speed="2">2x</button>
      <button class="btn speedBtn" data-speed="4">4x</button>
      <label style="margin-left:8px; font-size:12px;">
        <input type="checkbox" id="chkVideo" /> Show Video
      </label>
    </div>
    <div id="err"></div>
  </div>
  <div id="videoWrap">
    <img id="videoFrame" alt="video frame" />
    <div id="videoLabel">Synchronized video playback</div>
  </div>
  <canvas id="view"></canvas>

  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const errEl = document.getElementById("err");
    const videoWrap = document.getElementById("videoWrap");
    const videoFrame = document.getElementById("videoFrame");
    const chkVideo = document.getElementById("chkVideo");
    const speedButtons = Array.from(document.querySelectorAll(".speedBtn"));

    let latest = null;
    let arenaBuilt = false;

    const cam = {
      yaw: 0.8,
      pitch: -0.45,
      dist: 30.0,
      target: {x: 0, y: 5, z: 2}
    };

    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      cam.yaw += dx * 0.005;
      cam.pitch += dy * 0.004;
      cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch));
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const f = Math.exp(e.deltaY * 0.001);
      cam.dist *= f;
      cam.dist = Math.max(5.0, Math.min(150.0, cam.dist));
    }, { passive: false });

    async function postControl(payload) {
      try {
        await fetch("/api/control", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload || {})
        });
      } catch (_) {}
    }

    document.getElementById("btnPlay").addEventListener("click", () => postControl({action: "play"}));
    document.getElementById("btnPause").addEventListener("click", () => postControl({action: "pause"}));
    document.getElementById("btnRestart").addEventListener("click", () => postControl({action: "restart"}));
    document.getElementById("btnStep").addEventListener("click", () => postControl({action: "step", frames: 1}));
    chkVideo.addEventListener("change", () => postControl({show_video: chkVideo.checked}));
    speedButtons.forEach((b) => {
      b.addEventListener("click", () => {
        const s = Number(b.dataset.speed || "1");
        postControl({speed: s});
      });
    });

    function v3(x, y, z) { return {x, y, z}; }
    function add(a, b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
    function sub(a, b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
    function scale(a, s) { return v3(a.x * s, a.y * s, a.z * s); }
    function dot(a, b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    function cross(a, b) { return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
    function norm(a) { return Math.sqrt(dot(a, a)) || 1e-9; }
    function normalize(a) { const n = norm(a); return v3(a.x/n, a.y/n, a.z/n); }
    function headingFromDir(dirArr) {
      if (!dirArr || dirArr.length < 2) return null;
      const x = Number(dirArr[0]);
      const y = Number(dirArr[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      const d = Math.atan2(x, y) * 180.0 / Math.PI;
      return ((d % 360) + 360) % 360;
    }

    function worldPoint(posArr) {
      return v3(Number(posArr[0]), Number(posArr[1]), Number(posArr[2]));
    }

    function cameraPosition() {
      const cp = Math.cos(cam.pitch);
      const sp = Math.sin(cam.pitch);
      const cy = Math.cos(cam.yaw);
      const sy = Math.sin(cam.yaw);
      return v3(
        cam.target.x + cam.dist * cp * cy,
        cam.target.y + cam.dist * cp * sy,
        cam.target.z + cam.dist * sp
      );
    }

    function project(p) {
      const cpos = cameraPosition();
      const forward = normalize(sub(cam.target, cpos));
      let right = cross(forward, v3(0, 0, 1));
      if (norm(right) < 1e-6) right = v3(1, 0, 0);
      right = normalize(right);
      const up = normalize(cross(right, forward));
      const rel = sub(p, cpos);
      const xCam = dot(rel, right);
      const yCam = dot(rel, up);
      const zCam = dot(rel, forward);
      if (zCam <= 0.05) return null;
      const focal = 0.95 * Math.min(canvas.width, canvas.height);
      const sx = canvas.width * 0.5 + focal * (xCam / zCam);
      const sy = canvas.height * 0.5 - focal * (yCam / zCam);
      return {x: sx, y: sy, z: zCam};
    }

    function drawPoint(p, color, r) {
      const q = project(p);
      if (!q) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(q.x, q.y, r, 0, 2 * Math.PI);
      ctx.fill();
    }

    function drawLine(a, b, color, w=1) {
      const pa = project(a);
      const pb = project(b);
      if (!pa || !pb) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    function drawLabel(p, text, color="#f8fafc") {
      const q = project(p);
      if (!q) return;
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(2,6,23,0.9)";
      ctx.strokeText(text, q.x + 7, q.y - 8);
      ctx.fillStyle = color;
      ctx.fillText(text, q.x + 7, q.y - 8);
    }

    function buildArena(arena) {
      const width = Number(arena.arena_width_m || 20.0);
      const depth = Number(arena.arena_height_m || 10.0);
      const ox = Number(arena.arena_origin_x || 0.0);
      const oy = Number(arena.arena_origin_y || 0.0);
      const zmin = Number(arena.z_min || -1.0);
      const zmax = Number(arena.z_max || 5.0);
      const cx = ox + width / 2.0;
      const cy = oy + depth / 2.0;
      const cz = (zmin + zmax) / 2.0;
      cam.target = {x: cx, y: cy, z: cz};
      arenaBuilt = true;
    }

    function drawScene() {
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!latest || !latest.arena) return;

      const arena = latest.arena;
      const width = Number(arena.arena_width_m || 20.0);
      const depth = Number(arena.arena_height_m || 10.0);
      const ox = Number(arena.arena_origin_x || 0.0);
      const oy = Number(arena.arena_origin_y || 0.0);
      const zmin = Number(arena.z_min || -1.0);
      const zmax = Number(arena.z_max || 5.0);
      const x0 = ox, x1 = ox + width;
      const y0 = oy, y1 = oy + depth;

      const c000 = v3(x0, y0, zmin), c100 = v3(x1, y0, zmin), c110 = v3(x1, y1, zmin), c010 = v3(x0, y1, zmin);
      const c001 = v3(x0, y0, zmax), c101 = v3(x1, y0, zmax), c111 = v3(x1, y1, zmax), c011 = v3(x0, y1, zmax);
      const edges = [
        [c000,c100],[c100,c110],[c110,c010],[c010,c000],
        [c001,c101],[c101,c111],[c111,c011],[c011,c001],
        [c000,c001],[c100,c101],[c110,c111],[c010,c011],
      ];
      for (const [a,b] of edges) drawLine(a,b,"#94a3b8",1.4);

      for (let i = 0; i <= 10; i++) {
        const t = i / 10;
        const gx = x0 + (x1 - x0) * t;
        const gy = y0 + (y1 - y0) * t;
        drawLine(v3(gx, y0, zmin), v3(gx, y1, zmin), "#1e293b", 1);
        drawLine(v3(x0, gy, zmin), v3(x1, gy, zmin), "#1e293b", 1);
      }

      for (const m of (arena.markers || [])) {
        const mp = v3(Number(m.x || 0), Number(m.y || 0), Number(m.z || 0));
        drawPoint(mp, "#f59e0b", 3.5);
        drawLabel(mp, `ID ${m.id}`, "#fbbf24");
      }

      const traj = latest.trajectory || [];
      for (let i = 1; i < traj.length; i++) {
        drawLine(worldPoint(traj[i-1].pos), worldPoint(traj[i].pos), "#34d399", 1.6);
      }

      const pose = latest.pose;
      if (pose && pose.pos) {
        const p = worldPoint(pose.pos);
        drawPoint(p, "#22d3ee", 6);

        // Draw facing direction arrow if pose direction is available.
        if (pose.dir && pose.dir.length >= 3) {
          const dirRaw = worldPoint(pose.dir);
          const mag = norm(dirRaw);
          if (mag > 1e-6) {
            const d = scale(dirRaw, 1.0 / mag);
            const arrowLen = 1.2;
            const tip = add(p, scale(d, arrowLen));
            drawLine(p, tip, "#f97316", 2.4);

            let refUp = v3(0, 0, 1);
            if (Math.abs(dot(d, refUp)) > 0.93) refUp = v3(0, 1, 0);
            const side = normalize(cross(d, refUp));
            const back = scale(d, -0.28);
            const wing = scale(side, 0.14);
            drawLine(tip, add(add(tip, back), wing), "#f97316", 2.0);
            drawLine(tip, add(add(tip, back), scale(wing, -1.0)), "#f97316", 2.0);
          }
        }
      }
    }

    function updateButtons(ctrl) {
      speedButtons.forEach((b) => {
        const v = Number(b.dataset.speed || "1");
        b.classList.toggle("active", Math.abs(v - Number(ctrl.speed || 1)) < 1e-6);
      });
    }

    async function pullState() {
      try {
        const res = await fetch("/api/state");
        const data = await res.json();
        if (!data.ok) return;
        latest = data;

        if (!arenaBuilt) buildArena(data.arena);

        const pose = data.pose || null;
        const pb = data.playback || {};
        const ctrl = data.controls || {};
        const headingDeg = pose && pose.dir ? headingFromDir(pose.dir) : null;
        const txt = [
          `Video: ${pb.video_name || "-"}`,
          `Frame: ${pb.frame_idx || 0} / ${pb.frame_count || "?"}`,
          `Video time: ${(pb.video_time_s || 0).toFixed(2)} s`,
          `Pose frames: ${pb.pose_count || 0} | stale: ${pb.stale_count || 0}`,
          `Seen markers: ${(pose && pose.seen_count) ? pose.seen_count : 0}`,
          pose && pose.pos ? `Pos: x=${pose.pos[0].toFixed(3)} y=${pose.pos[1].toFixed(3)} z=${pose.pos[2].toFixed(3)}` : "Pos: -",
          headingDeg !== null ? `Heading: ${headingDeg.toFixed(1)}°` : "Heading: -",
          `Playback: ${ctrl.paused ? "paused" : "playing"} @ ${(Number(ctrl.speed || 1)).toFixed(2)}x`,
          `Status: ${pb.done ? "done" : "running"}`
        ].join("\\n");
        statusEl.textContent = txt;
        errEl.textContent = data.error ? ("Worker error: " + data.error) : "";

        chkVideo.checked = !!ctrl.show_video;
        videoWrap.style.display = chkVideo.checked ? "block" : "none";
        if (chkVideo.checked && data.video_frame_jpeg) {
          videoFrame.src = "data:image/jpeg;base64," + data.video_frame_jpeg;
        }

        updateButtons(ctrl);
      } catch (err) {
        statusEl.textContent = "API error: " + err;
      }
    }

    function animate() {
      requestAnimationFrame(animate);
      drawScene();
    }
    animate();
    pullState();
    setInterval(pullState, 120);
  </script>
</body>
</html>
        """

    @app.get("/api/state")
    def api_state():
        with state_lock:
            arena = dict(state["arena"])
            trajectory = list(state["trajectory"])
            latest_pose = dict(state["latest_pose"]) if state["latest_pose"] else None
            playback = {
                "video_name": state["video_name"],
                "frame_idx": state["frame_idx"],
                "frame_count": state["frame_count"],
                "video_time_s": state["video_time_s"],
                "pose_count": state["pose_count"],
                "stale_count": state["stale_count"],
                "done": state["done"],
            }
            controls = {
                "paused": bool(state.get("paused", False)),
                "speed": float(state.get("speed", 1.0)),
                "show_video": bool(state.get("show_video", False)),
                "step_frames": int(state.get("step_frames", 0)),
            }
            frame_b64 = state.get("video_frame_jpeg")
            err = state.get("error")

        if len(trajectory) > 1200:
            step = max(1, len(trajectory) // 1200)
            trajectory = trajectory[::step]

        payload = {
            "ok": True,
            "arena": arena,
            "pose": latest_pose,
            "trajectory": trajectory,
            "playback": playback,
            "controls": controls,
            "video_frame_jpeg": frame_b64 if controls["show_video"] else None,
            "error": err,
        }
        return jsonify(_json_safe(payload))

    @app.post("/api/control")
    def api_control():
        data = request.get_json(silent=True) or {}
        with control_cv:
            action = str(data.get("action", "")).strip().lower()

            if "speed" in data:
                try:
                    state["speed"] = max(0.05, min(8.0, float(data["speed"])))
                except Exception:
                    pass

            if "show_video" in data:
                state["show_video"] = bool(data["show_video"])
                if not state["show_video"]:
                    state["video_frame_jpeg"] = None

            if action == "play":
                if state.get("done", False):
                    state["restart_requested"] = True
                    state["done"] = False
                state["paused"] = False

            elif action == "pause":
                state["paused"] = True

            elif action == "toggle":
                state["paused"] = not bool(state.get("paused", False))

            elif action == "restart":
                state["restart_requested"] = True
                state["done"] = False
                state["paused"] = False
                state["step_frames"] = 0

            elif action == "step":
                try:
                    n = max(1, min(500, int(data.get("frames", 1))))
                except Exception:
                    n = 1
                if state.get("done", False):
                    state["restart_requested"] = True
                    state["done"] = False
                state["paused"] = True
                state["step_frames"] = int(state.get("step_frames", 0)) + n

            control_cv.notify_all()
            controls = {
                "paused": bool(state.get("paused", False)),
                "speed": float(state.get("speed", 1.0)),
                "show_video": bool(state.get("show_video", False)),
                "step_frames": int(state.get("step_frames", 0)),
                "done": bool(state.get("done", False)),
                "restart_requested": bool(state.get("restart_requested", False)),
            }

        return jsonify(ok=True, controls=_json_safe(controls))

    return app


def _triangulation_worker(args, state: Dict[str, Any], state_lock: threading.Lock, control_cv: threading.Condition):
    try:
        HeadlessAruCoPositioning = _load_headless_aruco()

        while True:
            cap = cv2.VideoCapture(str(args.video))
            if not cap.isOpened():
                with control_cv:
                    state["done"] = True
                    state["error"] = f"Could not open video: {args.video}"
                    control_cv.notify_all()
                print(f"[SMOKE] Could not open video: {args.video}")
                return

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-6:
                fps = float(args.fallback_fps)
            fps = max(1.0, fps)

            with control_cv:
                state["frame_count"] = frame_count
                state["video_name"] = Path(args.video).name
                _reset_playback_state(state)
                state["done"] = False
                state["error"] = None
                state["restart_requested"] = False
                control_cv.notify_all()

            processor = None
            frame_idx = 0
            restart_now = False

            while True:
                with control_cv:
                    while (
                        state.get("paused", False)
                        and int(state.get("step_frames", 0)) <= 0
                        and not state.get("restart_requested", False)
                    ):
                        control_cv.wait(timeout=0.25)

                    if state.get("restart_requested", False):
                        state["restart_requested"] = False
                        restart_now = True
                        break

                    speed = max(0.05, float(state.get("speed", 1.0)))
                    show_video = bool(state.get("show_video", False))
                    step_mode = int(state.get("step_frames", 0)) > 0
                    if step_mode:
                        state["step_frames"] = int(state.get("step_frames", 0)) - 1

                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                if processor is None:
                    cam_mat, dist = _load_calibration(args.calib, w, h)
                    arena_cfg = state["arena"]
                    processor = HeadlessAruCoPositioning(
                        cam_mat,
                        dist,
                        detect_profile=args.detect_profile,
                        marker_size=float(arena_cfg.get("marker_size_m", 0.18)),
                    )
                    _apply_arena_to_processor(processor, arena_cfg)
                    print(
                        f"[SMOKE] Processor ready: {w}x{h}, profile={args.detect_profile}, marker={arena_cfg.get('marker_size_m', 0.18)}m"
                    )

                video_ts = frame_idx / fps
                now_ts = video_ts + args.latency_comp_s
                result = processor.process_frame(
                    frame,
                    frame_ts=video_ts,
                    latency_s=args.latency_comp_s,
                    now_ts=now_ts,
                )
                if result is None:
                    result = {"cam": None, "stale": True, "seen_markers": [], "seen_count": 0, "ref_markers": []}

                frame_b64 = None
                if show_video:
                    frame_b64 = _encode_frame_jpeg_b64(frame, width=args.video_width, quality=args.video_jpeg_quality)

                with control_cv:
                    state["frame_idx"] = frame_idx
                    state["video_time_s"] = video_ts
                    if bool(result.get("stale")):
                        state["stale_count"] += 1

                    cam_f = _finite_vec3(result.get("cam")) if result.get("cam") is not None else None
                    if cam_f is not None:
                        state["pose_count"] += 1
                        dir_f = _finite_vec3(result.get("dir"))
                        state["latest_pose"] = {
                            "pos": cam_f,
                            "dir": dir_f,
                            "stale": bool(result.get("stale", False)),
                            "seen_markers": [int(m) for m in result.get("seen_markers", [])],
                            "seen_count": int(result.get("seen_count", 0)),
                            "ref_markers": [int(m) for m in result.get("ref_markers", [])],
                            "marker_weights": _json_safe(result.get("marker_weights", {})),
                            "video_time_s": video_ts,
                        }
                        state["trajectory"].append({"t": video_ts, "pos": cam_f})
                        if len(state["trajectory"]) > args.max_trail_points:
                            drop_n = len(state["trajectory"]) - args.max_trail_points
                            del state["trajectory"][:drop_n]

                    if show_video:
                        state["video_frame_jpeg"] = frame_b64
                    else:
                        state["video_frame_jpeg"] = None

                frame_idx += 1

                frame_dt = 1.0 / fps
                if not step_mode:
                    time.sleep(max(0.0, frame_dt / speed))

            cap.release()

            if restart_now:
                continue

            with control_cv:
                state["done"] = True
                state["paused"] = True
                control_cv.notify_all()
                while not state.get("restart_requested", False):
                    control_cv.wait(timeout=0.3)
                state["restart_requested"] = False
                state["paused"] = False
                state["step_frames"] = 0

    except Exception as exc:
        with control_cv:
            state["done"] = True
            state["error"] = repr(exc)
            control_cv.notify_all()
        print(f"[SMOKE] Worker error: {exc!r}")


def _parse_args():
    p = argparse.ArgumentParser(description="AruCo video smoke test with 3D web visualization")
    p.add_argument("--video", required=True, help="Input video path (mp4/mov/...)")
    p.add_argument(
        "--arena-config",
        default=str(Path(__file__).with_name("arena_config.json")),
        help="Arena config JSON path",
    )
    p.add_argument("--calib", default="", help="Optional camera calibration .npz file")
    p.add_argument("--detect-profile", default="balanced", choices=["sensitive", "balanced", "strict"])
    p.add_argument("--latency-comp-s", type=float, default=0.05, help="Stream latency compensation in seconds")
    p.add_argument("--fallback-fps", type=float, default=10.0, help="Used if video FPS metadata is missing")
    p.add_argument("--max-trail-points", type=int, default=4000)
    p.add_argument("--video-width", type=int, default=640, help="Preview frame width when video panel is enabled")
    p.add_argument("--video-jpeg-quality", type=int, default=55, help="Preview JPEG quality (25..95)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    return p.parse_args()


def main():
    args = _parse_args()
    args.video = str(Path(args.video).expanduser().resolve())
    args.arena_config = str(Path(args.arena_config).expanduser().resolve())
    args.calib = Path(args.calib).expanduser().resolve() if args.calib else None

    if not Path(args.video).exists():
        raise SystemExit(f"Video not found: {args.video}")

    arena_cfg = _load_arena_config(Path(args.arena_config))
    z_min, z_max = _arena_z_bounds(arena_cfg.get("markers", []))
    arena_cfg["z_min"] = z_min
    arena_cfg["z_max"] = z_max

    state_lock = threading.Lock()
    control_cv = threading.Condition(state_lock)
    state: Dict[str, Any] = {
        "arena": arena_cfg,
        "video_name": Path(args.video).name,
        "frame_idx": 0,
        "frame_count": 0,
        "video_time_s": 0.0,
        "pose_count": 0,
        "stale_count": 0,
        "latest_pose": None,
        "trajectory": [],
        "video_frame_jpeg": None,
        "done": False,
        "error": None,
        "paused": False,
        "speed": 1.0,
        "show_video": False,
        "step_frames": 0,
        "restart_requested": False,
    }

    worker = threading.Thread(
        target=_triangulation_worker,
        args=(args, state, state_lock, control_cv),
        daemon=True,
        name="aruco-smoke-worker",
    )
    worker.start()

    app = _build_app(state, state_lock, control_cv)
    print(f"[SMOKE] Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
