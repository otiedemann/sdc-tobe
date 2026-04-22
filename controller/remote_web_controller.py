import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

# ── ArUco Seek (multi-drone observer / LIVE controller) ────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from aruco_seek_multi import (  # noqa: E402
    HoverParams as AsHoverParams,
    MissionManager,
    ObserverFleet,
)

# ── Connection-pooled HTTP session ───────────────────────────────────────────
# Reuses TCP connections (keep-alive) instead of opening a new one per request.
# Dramatically reduces per-request latency on LAN (~30-50 ms saved per call).
_http_session = requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})
adapter = requests.adapters.HTTPAdapter(
    pool_connections=8,    # one per drone + spare
    pool_maxsize=16,       # concurrent requests per host
    max_retries=0,         # fail fast — don't retry on control commands
)
_http_session.mount("http://", adapter)
_http_session.mount("https://", adapter)

# Thread pool for parallel heartbeats / fan-out requests
_heartbeat_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hb")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Runs on remote PC. Proxies to Pi API server.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8090
TIMEOUT_CMD = float(os.getenv("PI_TIMEOUT_CMD", "12"))
TIMEOUT_STATUS = float(os.getenv("PI_TIMEOUT_STATUS", "0.5"))
VIDEO_UDP_FORWARD_PORT = int(os.getenv("VIDEO_UDP_FORWARD_PORT", "55004"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_JPEG_QUALITY = int(os.getenv("VIDEO_JPEG_QUALITY", "70"))

# Drone fleet config file — stored alongside this script
DRONES_CONFIG_PATH = Path(__file__).parent / "drones_config.json"

DEFAULT_DRONES = {
    "1": {"name": "Anafi 1", "type": "anafi", "base": "http://flightctrl1:8080"},
    "2": {"name": "Anafi 2", "type": "anafi", "base": "http://flightctrl2:8080"},
    "3": {"name": "Anafi 3", "type": "anafi", "base": "http://flightctrl3:8080"},
    "4": {"name": "Anafi 4", "type": "anafi", "base": "http://flightctrl4:8080"},
}

def load_drones_config() -> dict:
    if DRONES_CONFIG_PATH.exists():
        try:
            with open(DRONES_CONFIG_PATH) as f:
                cfg = json.load(f)
            # Validate structure
            for did, info in cfg.items():
                if not all(k in info for k in ("name", "type", "base")):
                    print(f"[CONFIG] Invalid drone entry {did}, using defaults")
                    return dict(DEFAULT_DRONES)
            print(f"[CONFIG] Loaded {len(cfg)} drones from {DRONES_CONFIG_PATH}")
            return cfg
        except Exception as e:
            print(f"[CONFIG] Error loading {DRONES_CONFIG_PATH}: {e}, using defaults")
    else:
        save_drones_config(DEFAULT_DRONES)
        print(f"[CONFIG] Created default config at {DRONES_CONFIG_PATH}")
    return dict(DEFAULT_DRONES)

def save_drones_config(drones: dict):
    with open(DRONES_CONFIG_PATH, "w") as f:
        json.dump(drones, f, indent=2)

DRONES = load_drones_config()
active_drone_id = "1"
# PI_API_BASE env var overrides the base URL from drones_config.json
_env_base = os.getenv("PI_API_BASE")
if _env_base:
    PI_BASE = _env_base.rstrip("/")
    DRONES[active_drone_id]["base"] = PI_BASE
else:
    PI_BASE = DRONES[active_drone_id]["base"]

app = Flask(__name__)

# ArUco Seek fleet — one observer per configured drone. LIVE mode is on by
# default (matches tools/aruco_seek_web.py default); disable with REMOTE_NO_LIVE=1.
_aruco_allow_live = os.getenv("REMOTE_NO_LIVE", "0") not in {"1", "true", "True"}
aruco_fleet = ObserverFleet(session=_http_session, allow_live=_aruco_allow_live)
aruco_fleet.configure(DRONES)
mission_manager = MissionManager(aruco_fleet)

command_log_enabled = os.getenv("REMOTE_COMMAND_LOG", "0") in {"1", "true", "True"}
command_log_path = Path(os.getenv("REMOTE_COMMAND_LOG_PATH", "remote_command_log.jsonl"))
command_log_last: dict[str, float] = {}

HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Drone Remote Controller</title>
  <!-- Three.js for optional 3D arena view. Loaded up-front so the 3D
       checkbox handler can init the scene the first time it's ticked.
       Uses ES module imports via importmap, which works in Chrome/Safari
       desktop. Falls back gracefully — if the module fails to load, the
       3D checkbox will log an error and the 2D view still works. -->
  <script type=\"importmap\">
  {
    \"imports\": {
      \"three\": \"https://unpkg.com/three@0.161.0/build/three.module.js\",
      \"three/addons/\": \"https://unpkg.com/three@0.161.0/examples/jsm/\"
    }
  }
  </script>
  <style>
    body { background:#0f172a; color:#e2e8f0; font-family:Arial,sans-serif; margin:0; padding:16px; }
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
    .panel { background:#111827; border:1px solid #334155; border-radius:8px; padding:12px; }
    .grid { display:grid; grid-template-columns:repeat(3,70px); gap:8px; }
    button { height:52px; border-radius:8px; border:1px solid #475569; background:#1e293b; color:#e2e8f0; font-weight:700; cursor:pointer; }
    button:active, .active { background:#0ea5e9; color:#001018; }
    .small { color:#94a3b8; font-size:12px; }
    .status-wrap { margin-top:8px; display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
    .meter { min-width:220px; }
    .meter-label { font-size:12px; color:#94a3b8; margin-bottom:4px; }
    .meter-track { height:12px; border-radius:999px; background:#1f2937; border:1px solid #334155; overflow:hidden; }
    .meter-fill { height:100%; width:0%; background:#22c55e; transition:width .2s ease, background .2s ease; }
    .adv { margin-top:8px; border-top:1px solid #334155; padding-top:8px; }
    .adv-grid { display:grid; grid-template-columns:repeat(3,minmax(100px,1fr)); gap:8px; }
    .adv input { height:36px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 8px; }
    .drone-bar { display:flex; gap:8px; margin-bottom:12px; }
    .drone-btn { height:44px; padding:0 18px; border-radius:8px; border:2px solid #475569; background:#1e293b; color:#e2e8f0; font-weight:700; cursor:pointer; font-size:14px; transition:all .15s; }
    .drone-btn.selected { background:#0ea5e9; color:#001018; border-color:#0ea5e9; }
    .drone-btn:hover:not(.selected) { border-color:#94a3b8; }
    .drone-type { font-size:10px; font-weight:400; opacity:.7; display:block; line-height:1; }
    .video-panel { margin-top:12px; }
    .video-panel img { max-width:100%; border-radius:8px; background:#000; }
    .video-controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
    .video-controls button { height:36px; font-size:12px; padding:0 12px; }
    .video-controls select, .video-controls input { height:36px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 8px; }
    .video-status { font-size:11px; color:#94a3b8; margin-top:4px; }
    .video-url { font-size:11px; color:#38bdf8; word-break:break-all; }
    .pos-panel { min-width:260px; }
    .pos-coords { font-family:monospace; font-size:14px; letter-spacing:0.05em; margin:8px 0; }
    .pos-x { color:#38bdf8; } .pos-y { color:#4ade80; } .pos-z { color:#fb923c; }
    .pos-stale { color:#f59e0b; font-size:11px; }
    .arena-canvas { display:block; border-radius:6px; background:#0f172a; border:1px solid #334155; }
    .pos-cfg { margin-top:8px; border-top:1px solid #334155; padding-top:8px; }
    .pos-cfg label { font-size:12px; color:#94a3b8; }
    .pos-cfg input, .pos-cfg select { height:32px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 6px; }
    .pos-cfg button { height:32px; font-size:12px; padding:0 10px; }
    /* ── ArUco Seek panel ─────────────────────────────────────────── */
    #aruco_panel { margin-top:16px; padding:12px; background:#0b1220; border:1px solid #334155; border-radius:8px; }
    #aruco_panel h3 { margin:0 0 10px 0; color:#38bdf8; font-size:15px; }
    #aruco_panel .arc-grid { display:grid; grid-template-columns:minmax(360px,1fr) minmax(420px,1fr); gap:12px; align-items:flex-start; }
    #aruco_panel canvas.arc-topdown { background:#0b1220; border:1px solid #1e293b; border-radius:4px; display:block; width:100%; max-width:420px; height:auto; }
    #aruco_panel img.arc-video { width:100%; max-width:480px; background:#0f172a; border-radius:4px; min-height:180px; }
    #aruco_panel .arc-readout { font-family:'SF Mono','Menlo',monospace; font-size:11px; line-height:1.55; color:#cbd5e1; }
    #aruco_panel .arc-readout .k { color:#94a3b8; display:inline-block; min-width:80px; }
    #aruco_panel .arc-readout .pd { color:#64748b; font-size:10px; }
    #aruco_panel .arc-readout b { color:#38bdf8; font-weight:600; }
    #aruco_panel .arc-params { max-height:360px; overflow-y:auto; padding-right:6px; }
    #aruco_panel .arc-row { display:flex; align-items:center; gap:6px; margin-bottom:3px; font-size:11px; }
    #aruco_panel .arc-row label { width:150px; color:#cbd5e1; flex-shrink:0; font-size:10px; }
    #aruco_panel .arc-row input[type=range] { flex:1; min-width:80px; }
    #aruco_panel .arc-row input[type=number] { width:60px; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:3px; padding:2px 4px; font-size:10px; }
    #aruco_panel .arc-pgroup { border-top:1px dashed #334155; margin-top:6px; padding-top:4px; }
    #aruco_panel .arc-pgroup-label { font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px; }
    #aruco_panel .mode-seg { display:inline-flex; border:1px solid #334155; border-radius:4px; overflow:hidden; vertical-align:middle; }
    #aruco_panel .mode-seg button { background:transparent; border:0; border-radius:0; padding:5px 12px; font-weight:600; color:#94a3b8; font-size:11px; }
    #aruco_panel .mode-seg button.active.observe { background:#065f46; color:#ecfdf5; }
    #aruco_panel .mode-seg button.active.live    { background:#b91c1c; color:#fee2e2; }
    #aruco_panel .mode-seg button:disabled { opacity:0.45; cursor:not-allowed; }
    #aruco_panel .arc-manual { display:none; gap:6px; align-items:center; padding:4px 6px; background:#450a0a; border:1px solid #ef4444; border-radius:4px; }
    #aruco_panel .arc-manual.show { display:inline-flex; }
    #aruco_panel .arc-manual button { font-size:11px; padding:4px 8px; }
    #arc_live_banner { display:none; background:#b91c1c; color:#fee2e2; padding:6px 10px; border-radius:4px; font-weight:700; letter-spacing:0.04em; margin-bottom:8px; box-shadow:0 0 0 2px #fbbf24 inset; text-align:center; font-size:12px; }
    #arc_live_banner.show { display:block; animation:arcpulse 1.6s ease-in-out infinite; }
    @keyframes arcpulse { 0%,100% { box-shadow:0 0 0 2px #fbbf24 inset; } 50% { box-shadow:0 0 0 4px #fbbf24 inset; } }
    body.arc-live-mode { box-shadow:0 0 0 4px #b91c1c inset; }
    .arc-rc-sent { color:#f87171; font-weight:600; }
    /* ── Special Missions panel ──────────────────────────────────── */
    #missions_panel { margin-top:16px; padding:12px; background:#0b1220; border:1px solid #334155; border-radius:8px; }
    #missions_panel h3 { margin:0 0 10px 0; color:#a78bfa; font-size:15px; }
    #missions_panel .mis-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:12px; margin-bottom:6px; }
    #missions_panel select, #missions_panel input { background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:4px; padding:4px 6px; font-size:12px; }
    #missions_panel .mis-status { font-family:monospace; font-size:11px; color:#cbd5e1; background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:8px; max-height:200px; overflow-y:auto; white-space:pre-wrap; }
    #missions_panel .mis-drone-line { padding:2px 0; border-bottom:1px solid #1e293b; }
    #missions_panel .mis-drone-line b { color:#38bdf8; }
    #missions_panel .mis-badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:600; margin-left:6px; }
    #missions_panel .mis-badge.idle { background:#1e293b; color:#94a3b8; }
    #missions_panel .mis-badge.scan { background:#1e3a5f; color:#60a5fa; }
    #missions_panel .mis-badge.approach { background:#164e63; color:#22d3ee; }
    #missions_panel .mis-badge.hover { background:#065f46; color:#4ade80; }
    #missions_panel .mis-badge.wait { background:#451a03; color:#fbbf24; }
    #missions_panel .mis-badge.done { background:#14532d; color:#86efac; }
    #missions_panel .mis-badge.error { background:#7f1d1d; color:#fca5a5; }
    /* ── Takeoff error banner ───────────────────────────────────── */
    #takeoff_err { display:none; background:#7f1d1d; border:1px solid #ef4444; color:#fee2e2;
                   padding:10px 14px; border-radius:8px; margin-top:10px; font-size:13px; line-height:1.5; }
    #takeoff_err.show { display:block; animation:takeoffpulse 1.2s ease-in-out 2; }
    #takeoff_err .hdr { font-weight:700; letter-spacing:0.02em; margin-bottom:4px; font-size:14px; }
    #takeoff_err .reason { font-family:monospace; font-size:12px; color:#fecaca; margin-bottom:6px; word-break:break-word; }
    #takeoff_err .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:6px; }
    #takeoff_err .actions button { height:32px; padding:0 12px; font-size:12px; font-weight:600; }
    #takeoff_err .actions .magneto { background:#1e3a5f; border-color:#3b82f6; color:#dbeafe; }
    #takeoff_err .actions .dismiss { background:#374151; border-color:#6b7280; }
    @keyframes takeoffpulse { 0%,100% { box-shadow:0 0 0 2px #fbbf24 inset; } 50% { box-shadow:0 0 0 4px #fbbf24 inset; } }
    /* ── Magnetometer recalibration wizard ──────────────────────── */
    #mag_modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.78); z-index:1100;
                 justify-content:center; align-items:center; }
    #mag_modal.show { display:flex; }
    #mag_modal .card { background:#0b1220; border:1px solid #334155; border-radius:10px;
                       padding:18px; width:min(720px,95vw); max-height:92vh; overflow-y:auto;
                       box-shadow:0 10px 40px rgba(0,0,0,0.5); }
    #mag_modal h3 { margin:0 0 4px 0; color:#60a5fa; font-size:18px; }
    #mag_modal .sub { color:#94a3b8; font-size:12px; margin-bottom:12px; }
    #mag_modal .steps { display:grid; grid-template-columns:1fr; gap:6px; margin-bottom:12px; }
    .mag-step { display:flex; align-items:center; gap:10px; padding:8px 10px; background:#0f172a;
                border:1px solid #1e293b; border-radius:6px; transition:all .2s ease; }
    .mag-step.active { border-color:#3b82f6; background:#0f1e33; box-shadow:0 0 0 1px #3b82f6 inset; }
    .mag-step.ok { border-color:#10b981; background:#052e1b; }
    .mag-step.fail { border-color:#ef4444; background:#2a0a0a; }
    .mag-step .num { flex:0 0 26px; height:26px; border-radius:50%; background:#1e293b;
                     color:#94a3b8; font-weight:700; font-size:13px; display:flex;
                     align-items:center; justify-content:center; border:1px solid #334155; }
    .mag-step.active .num { background:#3b82f6; color:#eff6ff; border-color:#3b82f6; }
    .mag-step.ok .num { background:#10b981; color:#052e1b; border-color:#10b981; }
    .mag-step.fail .num { background:#ef4444; color:#450a0a; border-color:#ef4444; }
    .mag-step .title { flex:1; font-size:13px; color:#e2e8f0; }
    .mag-step .info { font-size:11px; color:#94a3b8; font-family:monospace; }
    #mag_axes { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:10px 0; }
    .mag-axis { background:#0f172a; border:1px solid #1e293b; border-radius:6px; padding:10px;
                text-align:center; transition:all .2s ease; }
    .mag-axis.active { border-color:#fbbf24; box-shadow:0 0 0 1px #fbbf24 inset;
                       animation:magpulse 1s ease-in-out infinite; }
    .mag-axis.ok { border-color:#10b981; background:#052e1b; }
    .mag-axis .ax-name { font-weight:700; font-size:13px; color:#e2e8f0; letter-spacing:0.04em; }
    .mag-axis .ax-hint { font-size:10px; color:#94a3b8; margin-top:2px; }
    .mag-axis .ax-state { font-size:18px; margin-top:4px; }
    @keyframes magpulse { 0%,100% { background:#0f172a; } 50% { background:#1a2740; } }
    #mag_instructions { background:#0f172a; border:1px solid #334155; border-radius:6px;
                        padding:10px; margin:10px 0; font-size:12px; color:#cbd5e1; line-height:1.55; }
    #mag_instructions b { color:#fbbf24; }
    #mag_log { max-height:140px; overflow-y:auto; background:#0f172a; border:1px solid #1e293b;
               border-radius:6px; padding:8px; font-family:monospace; font-size:11px;
               color:#94a3b8; white-space:pre-wrap; margin-bottom:10px; }
    #mag_result { display:none; padding:10px; border-radius:6px; margin-bottom:10px;
                  font-weight:600; font-size:13px; }
    #mag_result.ok { display:block; background:#052e1b; border:1px solid #10b981; color:#86efac; }
    #mag_result.fail { display:block; background:#2a0a0a; border:1px solid #ef4444; color:#fca5a5; }
    #mag_modal .btnrow { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    #mag_modal .btnrow button { height:36px; padding:0 14px; font-size:12px; font-weight:600; }
    #mag_start_btn { background:#065f46; border-color:#10b981; }
    #mag_retry_btn { background:#1e3a5f; border-color:#3b82f6; display:none; }
    #mag_close_btn { background:#374151; border-color:#6b7280; }
  </style>
</head>
<body>
  <h2>Drone Remote Controller</h2>
  <div style=\"display:flex;align-items:center;gap:8px;\">
    <div class=\"drone-bar\" id=\"drone_bar\" style=\"flex:1;\"></div>
    <button id=\"land_all_btn\" style=\"padding:6px 14px;font-size:13px;font-weight:700;background:#7f1d1d;border-color:#ef4444;color:#fee2e2;letter-spacing:0.4px;\" title=\"Land every drone in the fleet safely. Keyboard shortcut: 0 (zero)\">&#11088; LAND ALL (0)</button>
    <button id=\"edit_drones_btn\" style=\"padding:4px 12px;font-size:12px;background:#1e3a5f;border-color:#3b82f6;\" title=\"Edit drone fleet config\">Config</button>
  </div>
  <div id=\"drone_config_modal\" style=\"display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:1000;justify-content:center;align-items:center;\">
    <div style=\"background:#1e293b;border:1px solid #334155;border-radius:8px;padding:20px;max-width:700px;width:90%;max-height:80vh;overflow-y:auto;\">
      <h3 style=\"margin:0 0 12px 0;color:#e2e8f0;\">Drone Fleet Configuration</h3>
      <div id=\"drone_config_fields\"></div>
      <div style=\"margin-top:12px;display:flex;gap:8px;\">
        <button id=\"drone_config_add\" style=\"background:#065f46;border-color:#10b981;padding:6px 16px;\">Add Drone</button>
        <button id=\"drone_config_save\" style=\"background:#1e3a5f;border-color:#3b82f6;padding:6px 16px;\">Save</button>
        <button id=\"drone_config_cancel\" style=\"background:#374151;border-color:#6b7280;padding:6px 16px;\">Cancel</button>
        <span id=\"drone_config_status\" class=\"small\" style=\"color:#94a3b8;align-self:center;\"></span>
      </div>
    </div>
  </div>
  <div id=\"mag_modal\" role=\"dialog\" aria-labelledby=\"mag_title\">
    <div class=\"card\">
      <h3 id=\"mag_title\">Magnetometer Recalibration</h3>
      <div class=\"sub\">
        Anafi requires a figure-8 dance around each axis whenever it is moved
        between locations or power-cycled. Keep the drone on a flat,
        non-metallic surface; no cables or phones nearby.
      </div>
      <div class=\"steps\" id=\"mag_steps\">
        <div class=\"mag-step\" data-step=\"heartbeat\">
          <div class=\"num\">1</div>
          <div class=\"title\">Pre-check — drone connected &amp; on the ground</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"pre_status\">
          <div class=\"num\">2</div>
          <div class=\"title\">Read current magnetometer status</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"start\">
          <div class=\"num\">3</div>
          <div class=\"title\">Start calibration command</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"poll\">
          <div class=\"num\">4</div>
          <div class=\"title\">Figure-8 around each axis until all axes confirm</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
      </div>
      <div id=\"mag_instructions\">
        <b>How to perform the dance:</b> hold the drone firmly and rotate it
        slowly (~2 s per full turn) around each body axis in turn — roll (X),
        pitch (Y), yaw (Z) — tracing a smooth figure-8. Each axis panel below
        lights green once its bit flips.
      </div>
      <div id=\"mag_axes\">
        <div class=\"mag-axis\" data-axis=\"x\">
          <div class=\"ax-name\">X (roll)</div>
          <div class=\"ax-hint\">tilt left/right repeatedly</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
        <div class=\"mag-axis\" data-axis=\"y\">
          <div class=\"ax-name\">Y (pitch)</div>
          <div class=\"ax-hint\">tilt nose up/down</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
        <div class=\"mag-axis\" data-axis=\"z\">
          <div class=\"ax-name\">Z (yaw)</div>
          <div class=\"ax-hint\">rotate around vertical</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
      </div>
      <div id=\"mag_result\"></div>
      <div id=\"mag_log\">ready.</div>
      <div class=\"btnrow\">
        <button id=\"mag_start_btn\">Start Recalibration</button>
        <button id=\"mag_retry_btn\">Retry</button>
        <button id=\"mag_close_btn\">Close</button>
      </div>
    </div>
  </div>
  <div class=\"small\">Active: <span id=\"pi\"></span></div>
  <div class=\"small\">API status: <span id=\"api_status\">checking...</span></div>
  <div class=\"small\">Drone telemetry status: <span id=\"drone_status\">checking...</span></div>
  <div class=\"row\" style=\"margin-top:10px;\">
    <div class=\"panel\">
      <div class=\"grid\" id=\"grid\">
        <button data-k=\"q\">Q</button><button data-k=\"w\">W</button><button data-k=\"e\">E</button>
        <button data-k=\"a\">A</button><button data-k=\"x\">STOP</button><button data-k=\"d\">D</button>
        <button data-k=\"r\">R</button><button data-k=\"s\">S</button><button data-k=\"f\">F</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"takeoff\">Takeoff (T)</button>
        <button id=\"land\">Land (L)</button>
        <button id=\"recover\">Recover</button>
        <button id=\"safe_takeoff\">Safe Takeoff: OFF</button>
      </div>
      <div id=\"takeoff_err\" role=\"alert\">
        <div class=\"hdr\">&#9888; Cannot take off</div>
        <div class=\"reason\" id=\"takeoff_err_reason\">—</div>
        <div class=\"small\" id=\"takeoff_err_hint\" style=\"color:#fecaca;\"></div>
        <div class=\"actions\">
          <button class=\"magneto\" id=\"takeoff_err_mag\" style=\"display:none;\">Recalibrate Magnetometer</button>
          <button class=\"dismiss\" id=\"takeoff_err_dismiss\">Dismiss</button>
        </div>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"toggle_log\">Enable Telemetry Log</button>
        <button id=\"download_log\">Download Telemetry Log</button>
        <button id=\"clear_log\">Clear Telemetry Log</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"toggle_cmd_log\">Command Logging: OFF</button>
        <button id=\"download_cmd_log\">Download Command Log</button>
        <button id=\"clear_cmd_log\">Clear Command Log</button>
      </div>
      <div class=\"adv\">
        <div class=\"small\" style=\"margin-bottom:6px;\">Advanced SDK controls</div>
        <div class=\"adv-grid\">
          <button id=\"rotate_cw\">Rotate CW 45°</button>
          <button id=\"rotate_ccw\">Rotate CCW 45°</button>
          <button id=\"move_up\">Up 30cm</button>
          <button id=\"move_down\">Down 30cm</button>
          <button id=\"move_fwd\">Forward 30cm</button>
          <button id=\"move_back\">Back 30cm</button>
          <button id=\"move_left\">Left 30cm</button>
          <button id=\"move_right\">Right 30cm</button>
          <button id=\"stream_on\">Stream ON</button>
          <button id=\"stream_off\">Stream OFF</button>
          <button id=\"set_speed\">Set Speed</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px;\">
          <input id=\"speed_val\" type=\"number\" min=\"10\" max=\"100\" value=\"30\" placeholder=\"speed 10..100\" />
          <input id=\"sdk_cmd\" type=\"text\" placeholder=\"raw sdk cmd (e.g. speed? or battery?)\" style=\"min-width:320px;flex:1;\" />
          <button id=\"sdk_send\">Send SDK Command</button>
        </div>
      </div>
      <div class=\"adv\" id=\"anafi_panel\">
        <div class=\"small\" style=\"margin-bottom:6px;\">Anafi / Olympe controls</div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Gimbal tilt</span>
          <input id=\"gimbal_tilt\" type=\"range\" min=\"-90\" max=\"30\" value=\"0\" style=\"flex:1;\" />
          <span id=\"gimbal_tilt_val\" class=\"small\" style=\"min-width:40px;\">0°</span>
          <button id=\"gimbal_set\">Set</button>
          <button id=\"gimbal_down\">Down (-90)</button>
          <button id=\"gimbal_fwd\">Forward (0)</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center; padding-top:8px; border-top:1px solid #1e293b;\">
          <span class=\"small\" style=\"min-width:80px;\">Magnetometer</span>
          <span id=\"mag_status\" class=\"small\" style=\"color:#94a3b8;font-family:monospace;flex:1;\">—</span>
          <button id=\"mag_open\" style=\"background:#1e3a5f;border-color:#3b82f6;height:32px;padding:0 12px;font-size:12px;\" title=\"Walk through the figure-8 recalibration wizard\">Recalibrate Magnetometer</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Max altitude (m)</span>
          <input id=\"set_alt\" type=\"number\" min=\"0.5\" max=\"150\" step=\"0.5\" value=\"5\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max vert spd</span>
          <input id=\"set_vspd\" type=\"number\" min=\"0.1\" max=\"4\" step=\"0.1\" value=\"0.5\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max tilt (°)</span>
          <input id=\"set_tilt\" type=\"number\" min=\"1\" max=\"35\" step=\"1\" value=\"15\" style=\"width:70px;\" />
          <button id=\"apply_settings\">Apply Settings</button>
        </div>
        <!-- ── Environment (indoor/outdoor) ─────────────────────── -->
        <div class=\"row\" style=\"margin-top:10px; align-items:center; padding-top:8px; border-top:1px solid #1e293b;\">
          <span class=\"small\" style=\"min-width:80px;\">Environment</span>
          <select id=\"env_mode\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;\">
            <option value=\"indoor\">Indoor (GPS-less, relaxed checks)</option>
            <option value=\"outdoor\">Outdoor (GPS-required, default)</option>
          </select>
          <button id=\"env_apply\" style=\"background:#1e3a5f;border-color:#3b82f6;\">Apply</button>
          <span id=\"env_status\" class=\"small\" style=\"color:#94a3b8;\">—</span>
        </div>
        <!-- ── Wi-Fi band + channel ─────────────────────────────── -->
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Wi-Fi band</span>
          <select id=\"wifi_band\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;\">
            <option value=\"5_GHz\">5 GHz (recommended)</option>
            <option value=\"2_4_GHz\">2.4 GHz</option>
          </select>
          <span class=\"small\" style=\"min-width:56px;\">Channel</span>
          <select id=\"wifi_channel\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;min-width:130px;\">
            <option value=\"auto\">Auto (drone picks)</option>
            <!-- 5 GHz non-DFS -->
            <option value=\"36\">36 (5 GHz UNII-1)</option>
            <option value=\"40\">40 (5 GHz UNII-1)</option>
            <option value=\"44\">44 (5 GHz UNII-1)</option>
            <option value=\"48\">48 (5 GHz UNII-1)</option>
            <option value=\"149\">149 (5 GHz UNII-3)</option>
            <option value=\"153\">153 (5 GHz UNII-3)</option>
            <option value=\"157\">157 (5 GHz UNII-3)</option>
            <option value=\"161\">161 (5 GHz UNII-3)</option>
            <option value=\"165\">165 (5 GHz UNII-3)</option>
            <!-- 2.4 GHz -->
            <option value=\"1\">1 (2.4 GHz)</option>
            <option value=\"6\">6 (2.4 GHz)</option>
            <option value=\"11\">11 (2.4 GHz)</option>
          </select>
          <button id=\"wifi_apply\" style=\"background:#1e3a5f;border-color:#3b82f6;\" title=\"Apply band+channel. Drone on ground only. Wi-Fi link drops briefly while re-associating.\">Apply Wi-Fi</button>
          <button id=\"wifi_scan\" style=\"background:#374151;border-color:#6b7280;\" title=\"Scan the selected band for in-use channels\">Scan</button>
          <span id=\"wifi_status\" class=\"small\" style=\"color:#94a3b8;margin-left:6px;\">—</span>
        </div>
      </div>
      <div class=\"adv\" id=\"mission_panel\">
        <div class=\"small\" style=\"margin-bottom:6px;\"><b>Mission Planner</b> — enter one command per line</div>
        <textarea id=\"mission_cmds\" rows=\"8\" style=\"width:100%;font-family:monospace;font-size:12px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;resize:vertical;\" placeholder=\"Examples:\n100 forward\n90 cw\n50 back\n45 ccw\n80 up\n60 down\n30 left\n40 right\nwait 2\nhover 3\nland\ntakeoff\"></textarea>
        <div style=\"margin-top:6px;display:flex;gap:8px;align-items:center;\">
          <button id=\"mission_run\" style=\"background:#065f46;border-color:#10b981;\">Run Mission</button>
          <button id=\"mission_stop\" style=\"background:#7f1d1d;border-color:#dc2626;display:none;\">Abort Mission</button>
          <span id=\"mission_status\" class=\"small\" style=\"color:#94a3b8;\">idle</span>
        </div>
        <div id=\"mission_log\" class=\"small\" style=\"margin-top:6px;max-height:120px;overflow-y:auto;white-space:pre-wrap;color:#94a3b8;\"></div>
      </div>
      <div class=\"small\" style=\"margin-top:8px;\">Keyboard in browser: W/A/S/D R/F Q/E, T, L, Space stop</div>
    </div>

    <div style=\"display:flex; flex-direction:column; gap:12px;\">
      <div class=\"panel\" id=\"video_panel\">
        <div><b>Video Stream</b></div>
        <div class=\"video-controls\">
          <select id=\"video_mode\">
            <option value=\"off\">Off</option>
            <option value=\"mjpeg\">Way 1: MJPEG (decoded on Pi)</option>
            <option value=\"forward\">Way 2: UDP Forward (decoded on C2)</option>
          </select>
          <button id=\"video_toggle\">Start Video</button>
        </div>
        <div class=\"video-status\" id=\"video_status\">Mode: off</div>
        <div class=\"video-url\" id=\"video_url\" style=\"display:none;\"></div>
        <div id=\"video_container\" style=\"margin-top:8px;display:none;\">
          <img id=\"video_img\" src=\"\" alt=\"video stream\" style=\"width:480px;height:auto;\" />
        </div>
      </div>
      <div class=\"panel\">
        <div><b>Telemetry</b></div>
        <div class=\"status-wrap\">
          <div class=\"meter\">
            <div class=\"meter-label\">Battery SoC: <span id=\"battery_val\">-</span></div>
            <div class=\"meter-track\"><div id=\"battery_bar\" class=\"meter-fill\"></div></div>
          </div>
        </div>
        <div id=\"compass_wrap\" style=\"display:flex;gap:12px;align-items:center;margin-top:10px;padding:8px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <canvas id=\"compass_canvas\" width=\"96\" height=\"96\" style=\"flex:0 0 96px;display:block;\"></canvas>
          <div class=\"small\" style=\"flex:1;line-height:1.55;\">
            <div>Heading: <b style=\"color:#e2e8f0;\"><span id=\"compass_abs\">--</span>°</b> <span style=\"color:#64748b;font-size:11px;\">(mag)</span></div>
            <div>Takeoff ref: <span id=\"compass_ref\" style=\"color:#94a3b8;\">—</span></div>
            <div>Relative: <b style=\"color:#3b82f6;\"><span id=\"compass_rel\">--</span>°</b></div>
            <button id=\"compass_reset\" title=\"Re-capture takeoff heading from current yaw\" style=\"margin-top:4px;height:22px;font-size:11px;padding:0 8px;background:#1e3a5f;border-color:#3b82f6;\">Reset ref</button>
          </div>
        </div>
        <div id=\"telemetry\" class=\"small\" style=\"white-space:pre-wrap; margin-top:10px;\">loading...</div>
        <button id=\"graphs_toggle\" onclick=\"window.toggleGraphs && window.toggleGraphs()\" style=\"margin-top:8px;height:32px;font-size:12px;padding:0 14px;background:#1e3a5f;border-color:#3b82f6;\">Show Graphs</button>
      </div>
    </div>
  </div>

  <div id=\"graphs_panel\" style=\"display:none;margin-top:12px;padding:0 16px 16px;\">
    <div style=\"display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:8px;\" id=\"graphs_container\"></div>
  </div>

  <!-- ── ArUco Seek — hover in front of a marker (per active drone) ── -->
  <div id=\"aruco_panel\">
    <h3>ArUco Seek &mdash; hover in front of marker
      <span id=\"arc_status\" class=\"small\" style=\"margin-left:8px;color:#94a3b8;font-weight:400;\">stopped</span>
      <span id=\"arc_drone_label\" class=\"small\" style=\"margin-left:8px;color:#64748b;font-weight:400;\"></span>
    </h3>

    <div id=\"arc_live_banner\">&#9888; LIVE MODE &mdash; DRONE WILL MOVE &mdash; RC commands are being sent</div>

    <div class=\"mis-row\" style=\"margin-bottom:10px;\">
      <button id=\"arc_start\" style=\"background:#065f46;border-color:#10b981;\">&#9654; Start</button>
      <button id=\"arc_stop\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop</button>
      <span style=\"margin-left:8px;color:#94a3b8;font-size:11px;\">Mode:</span>
      <span class=\"mode-seg\">
        <button id=\"arc_mode_observe\" class=\"active observe\" data-mode=\"observe\">OBSERVE</button>
        <button id=\"arc_mode_live\" data-mode=\"live\">LIVE</button>
      </span>
      <span id=\"arc_mode_gate\" style=\"font-size:11px;color:#fbbf24;display:none;\">LIVE disabled (REMOTE_NO_LIVE=1)</span>
      <span id=\"arc_mode_err\" style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#7f1d1d;color:#fecaca;border:1px solid #ef4444;border-radius:4px;font-weight:600;\"></span>
      <span id=\"arc_manual\" class=\"arc-manual\">
        <button id=\"arc_takeoff\" style=\"background:#065f46;border-color:#10b981;\">&uarr; Takeoff</button>
        <button id=\"arc_land\">&darr; Land</button>
        <button id=\"arc_rc_stop\">RC Stop</button>
        <button id=\"arc_emergency\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9940; EMERGENCY</button>
      </span>
      <span style=\"margin-left:12px;color:#94a3b8;font-size:11px;\">Target marker:</span>
      <input id=\"arc_target_input\" type=\"number\" min=\"0\" placeholder=\"auto\" style=\"width:70px;\" />
      <button id=\"arc_target_lock\">Lock</button>
      <button id=\"arc_target_auto\">Auto</button>
    </div>

    <div class=\"arc-grid\">
      <div>
        <div class=\"small\" style=\"color:#94a3b8;margin-bottom:4px;\">Video feed (drone camera)</div>
        <img id=\"arc_video\" class=\"arc-video\" alt=\"(press Start to load video)\" />
        <div class=\"small\" style=\"color:#94a3b8;margin:8px 0 4px 0;\">Top-down &mdash; drone &harr; marker</div>
        <canvas id=\"arc_topdown\" class=\"arc-topdown\" width=\"420\" height=\"380\"></canvas>
        <div id=\"arc_readout\" class=\"arc-readout\" style=\"margin-top:8px;\">&mdash;</div>
      </div>
      <div>
        <div class=\"small\" style=\"color:#94a3b8;margin-bottom:4px;\">Live tuning parameters &mdash; applied immediately</div>
        <div id=\"arc_params\" class=\"arc-params\"></div>
        <div style=\"margin-top:6px;\"><button id=\"arc_reload\" style=\"font-size:11px;\">&#x21bb; Reload params from server</button></div>
      </div>
    </div>
  </div>

  <!-- ── Special Missions — coordinated multi-drone flights ───────── -->
  <div id=\"missions_panel\">
    <h3>Special Missions
      <span id=\"mis_title_status\" class=\"small\" style=\"margin-left:8px;color:#94a3b8;font-weight:400;\">idle</span>
    </h3>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Mission:</label>
      <select id=\"mis_type\">
        <option value=\"scan_all\">Scan all ArUco markers (sequential, collision-aware)</option>
        <option value=\"capture_targets\">Capture enemy targets (SDC26 — box-capture, camera on arena centre)</option>
      </select>
    </div>

    <!-- Capture-Targets specific inputs — hidden unless that mission is selected. -->
    <div id=\"mis_capture_rows\" style=\"display:none;\">
      <div class=\"mis-row\">
        <label style=\"color:#94a3b8;\" title=\"One JSON object per target box. Use the Blue team's 3 enemy boxes (the red-home boxes) for standard play.\">Target boxes (JSON):</label>
        <textarea id=\"mis_boxes_json\" rows=\"3\" style=\"width:100%;max-width:520px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;font-family:monospace;font-size:11px;\">[
  {\"id\": 1, \"x\": -5.0, \"y\": 2.0},
  {\"id\": 2, \"x\":  0.0, \"y\": 2.0},
  {\"id\": 3, \"x\":  5.0, \"y\": 2.0}
]</textarea>
      </div>
      <div class=\"mis-row\">
        <label style=\"color:#94a3b8;\" title=\"World-frame XY the drone returns to after all captures. Typically your team's home-zone centre.\">Home XY:</label>
        <input id=\"mis_home_x\" type=\"number\" step=\"0.1\" value=\"0.0\" style=\"width:70px;\" />
        <input id=\"mis_home_y\" type=\"number\" step=\"0.1\" value=\"9.0\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"World-frame XY the drone's camera aims at while moving (typically arena centre so many markers stay in view for triangulation).\">Face XY:</label>
        <input id=\"mis_face_x\" type=\"number\" step=\"0.1\" value=\"0.0\" style=\"width:70px;\" />
        <input id=\"mis_face_y\" type=\"number\" step=\"0.1\" value=\"5.4\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Altitude above the floor for the capture hover.\">Altitude (m):</label>
        <input id=\"mis_alt\" type=\"number\" step=\"0.1\" min=\"0.5\" value=\"1.5\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Hover duration above each box. Must be ≥ the 2s capture-hold from SDC26 rules.\">Hover s:</label>
        <input id=\"mis_cap_hover_s\" type=\"number\" step=\"0.5\" min=\"2.0\" value=\"4.0\" style=\"width:70px;\" />
      </div>
    </div>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Drones:</label>
      <span id=\"mis_drones\" style=\"display:flex;gap:10px;flex-wrap:wrap;\"></span>
    </div>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Target markers:</label>
      <input id=\"mis_markers\" type=\"text\" value=\"1-12\" placeholder=\"e.g. 1-12 or 1,2,3,7\" style=\"width:220px;\" />
      <label style=\"color:#94a3b8;margin-left:12px;\">Hover s:</label>
      <input id=\"mis_hover_s\" type=\"number\" min=\"0.5\" step=\"0.5\" value=\"1.5\" style=\"width:70px;\" title=\"How long the drone hovers in front of each marker before moving on. 1.5s scans cleanly; raise if the score validation needs longer.\" />
      <label style=\"color:#94a3b8;margin-left:12px;\">Approach tol (m):</label>
      <input id=\"mis_tol_m\" type=\"number\" min=\"0.1\" step=\"0.05\" value=\"0.30\" style=\"width:70px;\" title=\"Drone transitions from APPROACH to HOVER once within this many metres of the hover distance AND sufficiently perpendicular (see Skew tol).\" />
      <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Max skew before declaring the drone perpendicular. 0.08 ≈ 6° off the marker normal. Lower values force the drone to align straight-on before hovering.\">Skew tol:</label>
      <input id=\"mis_skew_tol\" type=\"number\" min=\"0.02\" max=\"0.50\" step=\"0.01\" value=\"0.12\" style=\"width:70px;\" title=\"Max perspective skew to accept for HOVER. 0.12 ≈ 9° off the marker normal. Lower = stricter perpendicular (longer to converge).\" />
      <label style=\"color:#94a3b8;margin-left:12px;display:flex;align-items:center;gap:4px;\">
        <input id=\"mis_auto_takeoff\" type=\"checkbox\" />
        auto-takeoff
      </label>
    </div>

    <div class=\"mis-row\" style=\"margin-top:8px;\">
      <button id=\"mis_start\" style=\"background:#065f46;border-color:#10b981;\">&#9654; Start mission</button>
      <button id=\"mis_stop\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop</button>
      <button id=\"mis_stop_land\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop + Land</button>
      <a id=\"mis_trace_download\" href=\"/proxy/missions/trace\" download style=\"margin-left:10px;color:#93c5fd;text-decoration:underline;font-size:12px;\" title=\"Download the JSONL trace log of the current (or most recent) mission\">⤓ Download trace</a>
      <span id=\"mis_err\" style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#7f1d1d;color:#fecaca;border:1px solid #ef4444;border-radius:4px;font-weight:600;\"></span>
      <span id=\"mis_ok\"  style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#064e3b;color:#a7f3d0;border:1px solid #22c55e;border-radius:4px;font-weight:600;\"></span>
      <span id=\"mis_progress\" style=\"margin-left:12px;color:#38bdf8;font-weight:600;\">—</span>
    </div>

    <div class=\"mis-row\" style=\"margin-top:6px;\">
      <label style=\"color:#94a3b8;\">Scanned:</label>
      <span id=\"mis_scanned\" style=\"color:#4ade80;font-family:monospace;\">—</span>
      <label style=\"color:#94a3b8;margin-left:12px;\">Remaining:</label>
      <span id=\"mis_remaining\" style=\"color:#fbbf24;font-family:monospace;\">—</span>
    </div>

    <div class=\"small\" style=\"color:#94a3b8;margin:10px 0 4px 0;\">Per-drone status</div>
    <div id=\"mis_status\" class=\"mis-status\">idle — no mission running</div>
  </div>

  <script>
  // ── Live Telemetry Graphs (standalone, runs independently) ─────────
  (function(){
    const WINDOW_S = 10;
    const SAMPLE_HZ = 20;
    const CANVAS_W = 340, CANVAS_H = 130;
    const GROUPS = [
      {title:'Altitude (cm)',     keys:['height_cm','tof_cm','barometer_cm'], colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Attitude (deg)',    keys:['pitch','roll','yaw'],                colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Velocity (cm/s)',   keys:['vgx','vgy','vgz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Acceleration',      keys:['agx','agy','agz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Speed',             keys:['speed'],                              colors:['#22d3ee']},
      {title:'Battery (%)',       keys:['battery'],                            colors:['#34d399']},
      {title:'Temperature (C)',   keys:['temperature'],                        colors:['#fb923c']},
      {title:'Position (m)',      keys:['pos_x','pos_y','pos_z'],              colors:['#22d3ee','#f472b6','#a78bfa']},
    ];
    const graphs = [];
    let visible = false, rafId = null, sampleTimer = null;

    function init() {
      const c = document.getElementById('graphs_container');
      if (!c || graphs.length) return;
      GROUPS.forEach(g => {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:6px;';
        const hdr = document.createElement('div');
        hdr.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;';
        let html = '<b style=\"color:#e2e8f0;\">' + g.title + '</b>';
        g.keys.forEach((k,i) => { html += '<span style=\"color:'+g.colors[i]+';\">'+k+'</span>'; });
        hdr.innerHTML = html;
        wrap.appendChild(hdr);
        const cv = document.createElement('canvas');
        cv.width = CANVAS_W; cv.height = CANVAS_H;
        cv.style.cssText = 'width:100%;height:auto;display:block;background:#020617;border-radius:4px;';
        wrap.appendChild(cv);
        c.appendChild(wrap);
        graphs.push({keys: g.keys, colors: g.colors, samples: [], canvas: cv, ctx: cv.getContext('2d')});
      });
      console.log('[graphs] init done,', graphs.length, 'graphs');
    }

    function sample() {
      const ts = performance.now();
      const t = (typeof window.lastTelemetry === 'object' && window.lastTelemetry) ? Object.assign({}, window.lastTelemetry) : {};
      if (Array.isArray(window._lastPos)) { t.pos_x = window._lastPos[0]; t.pos_y = window._lastPos[1]; t.pos_z = window._lastPos[2]; }
      graphs.forEach(g => {
        const vals = {}; let any = false;
        g.keys.forEach(k => { const v = t[k]; if (v != null && !isNaN(v)) { vals[k] = Number(v); any = true; } else { vals[k] = null; } });
        if (any) g.samples.push({t: ts, vals});
        const cutoff = ts - WINDOW_S * 1000;
        while (g.samples.length > 0 && g.samples[0].t < cutoff) g.samples.shift();
      });
    }

    function draw() {
      if (!visible) { rafId = null; return; }
      const now = performance.now();
      graphs.forEach(g => {
        const ctx = g.ctx, W = g.canvas.width, H = g.canvas.height;
        ctx.fillStyle = '#020617'; ctx.fillRect(0,0,W,H);
        if (g.samples.length < 2) {
          ctx.fillStyle = '#475569'; ctx.font = '11px sans-serif';
          ctx.textAlign = 'center'; ctx.fillText('waiting for data...', W/2, H/2); ctx.textAlign = 'left';
          return;
        }
        const tMin = now - WINDOW_S*1000, tMax = now;
        let yMin = Infinity, yMax = -Infinity;
        g.samples.forEach(s => g.keys.forEach(k => { if (s.vals[k]!=null) { yMin=Math.min(yMin,s.vals[k]); yMax=Math.max(yMax,s.vals[k]); } }));
        if (!isFinite(yMin)) return;
        if (yMin === yMax) { yMin -= 1; yMax += 1; }
        const pad = (yMax-yMin)*0.1 || 1; yMin -= pad; yMax += pad;
        ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
        for (let i=0;i<=4;i++) { const y=(i/4)*H; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
        ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
        for (let i=0;i<=4;i++) { const v = yMin + ((4-i)/4)*(yMax-yMin); ctx.fillText(v.toFixed(1), 2, (i/4)*H + 9); }
        g.keys.forEach((k, ki) => {
          ctx.strokeStyle = g.colors[ki]; ctx.lineWidth = 1.5; ctx.beginPath();
          let started = false;
          g.samples.forEach(s => {
            if (s.vals[k] == null) { started = false; return; }
            const x = ((s.t - tMin) / (tMax - tMin)) * W;
            const y = H - ((s.vals[k] - yMin) / (yMax - yMin)) * H;
            if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
          });
          ctx.stroke();
          const last = g.samples[g.samples.length - 1];
          if (last && last.vals[k] != null) {
            ctx.fillStyle = g.colors[ki]; ctx.font = '10px monospace'; ctx.textAlign = 'right';
            ctx.fillText(last.vals[k].toFixed(1), W - 2, 10 + ki * 11); ctx.textAlign = 'left';
          }
        });
      });
      rafId = requestAnimationFrame(draw);
    }

    window.toggleGraphs = function() {
      visible = !visible;
      const btn = document.getElementById('graphs_toggle');
      const panel = document.getElementById('graphs_panel');
      if (btn) btn.textContent = visible ? 'Hide Graphs' : 'Show Graphs';
      if (panel) panel.style.display = visible ? 'block' : 'none';
      console.log('[graphs] toggle ->', visible);
      if (visible) {
        init();
        if (!sampleTimer) sampleTimer = setInterval(sample, 1000 / SAMPLE_HZ);
        if (!rafId) rafId = requestAnimationFrame(draw);
      } else {
        if (sampleTimer) { clearInterval(sampleTimer); sampleTimer = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      }
    };
    console.log('[graphs] toggleGraphs registered');
  })();
  </script>

  <!-- ── ArUco Seek client JS ─────────────────────────────────────── -->
  <script>
  (function(){
    const PGROUPS = ['Mission target','Camera filter / deadbands','P gains (camera)','D gains (IMU damping)','Output clamps','Drawing'];
    const SLIDERS = [
      ['hover_distance_m','Hover distance (m)',          0.5, 4.0, 0.05, 0],
      ['fb_max',          'Approach speed (fwd RC %)',   0,100, 1, 0],
      ['fb_back_max',     'Retreat speed (back RC %)',   0,100, 1, 0],
      ['dist_p',          'Approach aggressiveness (P · distance)', 0, 60, 0.5, 0],
      ['ema_alpha',       'EMA α (smoothing)',           0.05,0.95, 0.05, 1],
      ['deadband_x',      'Deadband err_x',              0.00,0.30, 0.01, 1],
      ['deadband_y',      'Deadband err_y',              0.00,0.30, 0.01, 1],
      ['deadband_skew',   'Deadband skew',               0.00,0.30, 0.01, 1],
      ['deadband_dist_m', 'Deadband distance (m)',       0.00,1.00, 0.05, 1],
      ['yaw_p',           'P · yaw     (per err_x)',     0, 50, 1, 2],
      ['skew_p',          'P · lateral (per skew)',      0, 50, 1, 2],
      ['alt_p',           'P · altitude (per err_y)',    0,100, 1, 2],
      ['d_yaw',           'D · yaw     (°/s)',           0,  2, 0.05, 3],
      ['d_lr',            'D · lateral (cm/s vgy)',      0,  2, 0.05, 3],
      ['d_ud',            'D · vertical(cm/s vgz)',      0,  2, 0.05, 3],
      ['d_fb',            'D · fwd/back(cm/s vgx)',      0,  2, 0.05, 3],
      ['yaw_max',         'Clamp · yaw max',             0, 80, 1, 4],
      ['lr_max',          'Clamp · lateral max',         0,100, 1, 4],
      ['ud_max',          'Clamp · vertical max',        0,100, 1, 4],
      ['rc_min',          'RC dead-floor',               0, 10, 1, 4],
      ['cam_hfov_deg',    'Cam HFOV (drawing only)',    30,110, 1, 5],
      ['marker_size_m',   'Marker physical size (m)',    0.05, 2.0, 0.01, 5],
    ];
    let arcParams = {};
    let arcAllowLive = false;
    let arcMode = 'observe';

    // Which drone id is the ArUco panel currently tracking? Mirrors the
    // main UI's active drone, picked up from /proxy/drones on each poll.
    let arcActiveId = null;

    async function arcLoadParams() {
      try {
        const r = await fetch('/proxy/aruco/params');
        arcParams = await r.json();
        arcRenderParams();
      } catch {}
    }
    function arcRenderParams() {
      const cont = document.getElementById('arc_params');
      cont.innerHTML = '';
      let curGroup = -1;
      SLIDERS.forEach(([k,label,mn,mx,st,grp]) => {
        if (grp !== curGroup) {
          curGroup = grp;
          const h = document.createElement('div');
          h.className = 'arc-pgroup';
          h.innerHTML = '<div class=\"arc-pgroup-label\">' + PGROUPS[grp] + '</div>';
          cont.appendChild(h);
        }
        const v = arcParams[k] ?? 0;
        const r = document.createElement('div');
        r.className = 'arc-row';
        r.innerHTML =
          '<label title=\"'+k+'\">'+label+'</label>' +
          '<input type=\"range\" min=\"'+mn+'\" max=\"'+mx+'\" step=\"'+st+'\" value=\"'+v+'\" data-k=\"'+k+'\" />' +
          '<input type=\"number\" min=\"'+mn+'\" max=\"'+mx+'\" step=\"'+st+'\" value=\"'+v+'\" data-k=\"'+k+'\" />';
        cont.appendChild(r);
      });
      cont.querySelectorAll('input').forEach(el => {
        el.addEventListener('input', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (isNaN(v)) return;
          arcParams[k] = v;
          cont.querySelectorAll('input[data-k=\"'+k+'\"]').forEach(s => { if (s !== el) s.value = v; });
        });
        el.addEventListener('change', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (!isNaN(v)) fetch('/proxy/aruco/params', {method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify({[k]: v})});
        });
      });
    }

    function fmt(x, n) { return (x === undefined || x === null || isNaN(x)) ? '—' : Number(x).toFixed(n); }

    function arcApplyModeUI(mode, allow) {
      arcMode = mode || 'observe';
      if (typeof allow === 'boolean') arcAllowLive = allow;
      const bObs  = document.getElementById('arc_mode_observe');
      const bLive = document.getElementById('arc_mode_live');
      bObs.classList.toggle('active', arcMode === 'observe');
      bLive.classList.toggle('active', arcMode === 'live');
      bLive.classList.toggle('live',   arcMode === 'live');
      // Keep the button clickable — the server is the source of truth for
      // allow_live. We just visually hint with the "gate" badge if the
      // server has REMOTE_NO_LIVE set.
      bLive.disabled = false;
      bLive.style.opacity = arcAllowLive ? '1' : '0.7';
      document.getElementById('arc_mode_gate').style.display = arcAllowLive ? 'none' : 'inline';
      const live = (arcMode === 'live');
      document.getElementById('arc_live_banner').classList.toggle('show', live);
      document.body.classList.toggle('arc-live-mode', live);
      document.getElementById('arc_manual').classList.toggle('show', live);
    }

    function arcRenderReadout(s) {
      if (!s.running) {
        document.getElementById('arc_readout').innerHTML = '<span style=\"color:#64748b;\">stopped — press Start</span>';
        return;
      }
      const html =
        '<b>Marker</b><br>' +
        '<span class=\"k\">ID:</span>'+(s.marker_id ?? '—')+
        '&nbsp;&nbsp;<span class=\"k\">visible:</span>'+((s.visible_ids||[]).join(', ')||'—')+'<br>' +
        '<span class=\"k\">distance:</span>'+fmt(s.distance_m,2)+' m '+
          '<span class=\"pd\">(raw '+fmt(s.raw_distance_m,2)+', target '+fmt(arcParams.hover_distance_m,2)+')</span><br>' +
        '<span class=\"k\">err_x:</span>'+fmt(s.err_x,3)+
          '&nbsp;&nbsp;<span class=\"k\">err_y:</span>'+fmt(s.err_y,3)+
          '&nbsp;&nbsp;<span class=\"k\">skew:</span>'+fmt(s.skew,3)+'<br>' +
        '<br><b>IMU</b>  '+
        '<span class=\"pd\">vgx '+fmt(s.vx_cms,0)+', vgy '+fmt(s.vy_cms,0)+', vgz '+fmt(s.vz_cms,0)+' cm/s; '+
          'yaw '+fmt(s.yaw,1)+'° @ '+fmt(s.yaw_rate_dps,0)+'°/s; alt '+fmt(s.altitude_m,2)+' m</span><br>' +
        (s.mode === 'live'
          ? '<br><b class=\"arc-rc-sent\">RC — SENT to drone</b>'
          : '<br><b>RC — would-send</b> <span class=\"pd\">(observe — not sent)</span>') + '<br>' +
        '<span class=\"k\">lr:</span>'+(s.rc_lr ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_lr_p,1)+' D='+fmt(s.rc_lr_d,1)+')</span>' +
        '&nbsp; <span class=\"k\">fb:</span>'+(s.rc_fb ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_fb_p,1)+' D='+fmt(s.rc_fb_d,1)+')</span><br>' +
        '<span class=\"k\">ud:</span>'+(s.rc_ud ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_ud_p,1)+' D='+fmt(s.rc_ud_d,1)+')</span>' +
        '&nbsp; <span class=\"k\">yaw:</span>'+(s.rc_yaw ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_yaw_p,1)+' D='+fmt(s.rc_yaw_d,1)+')</span><br>' +
        (s.rc_sent_at ? '<span class=\"k\">last sent:</span><span class=\"arc-rc-sent\">'+((Date.now()/1000 - s.rc_sent_at).toFixed(1))+' s ago</span><br>' : '') +
        (s.rc_send_error ? '<span class=\"k\">send err:</span><span style=\"color:#fca5a5;\">'+s.rc_send_error+'</span><br>' : '') +
        // Arena safety guard banner — red when active, grey when idle.
        (s.guard && s.guard.active
          ? '<br><b style=\"color:#fca5a5;background:#7f1d1d;padding:2px 6px;border-radius:3px;\">⛔ SAFETY GUARD</b> '
              + '<span class=\"pd\">' + (s.guard.actions||[]).join(', ')
              + '  pos=('+ (s.guard.pos||[]).join(',') +')</span>'
          : '');
      document.getElementById('arc_readout').innerHTML = html;
    }

    function arcDrawTopDown(s) {
      const c = document.getElementById('arc_topdown');
      const ctx = c.getContext('2d');
      const W = c.width, H = c.height;
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
      const cx = W/2;
      const marker_y = 36;
      const target = arcParams.hover_distance_m || 1.5;
      const maxDist = Math.max(target*2.2, 3.0);
      const ppm = (H-80)/maxDist;
      ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
      for (let d=0.5; d<=maxDist; d+=0.5) { ctx.beginPath(); ctx.arc(cx, marker_y, d*ppm, 0, Math.PI, false); ctx.stroke(); }
      ctx.fillStyle = '#334155'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
      for (let d=1; d<=maxDist; d+=1) ctx.fillText(d+'m', cx+4, marker_y + d*ppm - 2);
      ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 1.5; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.arc(cx, marker_y, target*ppm, 0, Math.PI, false); ctx.stroke();
      ctx.setLineDash([]);
      // marker
      const mw = 60;
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(cx-mw, marker_y); ctx.lineTo(cx+mw, marker_y); ctx.stroke();
      ctx.fillStyle = '#22c55e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('marker '+(s.marker_id ?? '?'), cx+mw+6, marker_y+4);
      if (!s.running || s.distance_m == null) {
        ctx.fillStyle = '#475569'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(s.running ? 'no marker visible' : 'stopped', W/2, H/2);
        return;
      }
      const dist = s.distance_m;
      const lateral_m = -(s.skew || 0) * dist;
      const drone_x = cx + lateral_m*ppm;
      const drone_y = marker_y + dist*ppm;
      const hfov = (arcParams.cam_hfov_deg || 69) * Math.PI / 180;
      const yaw_off_rad = Math.atan((s.err_x || 0) * Math.tan(hfov/2));
      const dxv = cx - drone_x, dyv = marker_y - drone_y;
      const aimAng = Math.atan2(dyv, dxv);
      const droneAng = aimAng - yaw_off_rad;
      // LoS
      ctx.strokeStyle = '#fbbf2470'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(drone_x, drone_y); ctx.lineTo(cx, marker_y); ctx.stroke();
      ctx.setLineDash([]);
      // FOV
      ctx.strokeStyle = '#38bdf833'; ctx.fillStyle = '#38bdf815';
      const fovLen = ppm * Math.max(2.5, dist+1.0);
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.beginPath(); ctx.moveTo(0,0);
      ctx.lineTo(fovLen*Math.cos(-hfov/2), fovLen*Math.sin(-hfov/2));
      ctx.lineTo(fovLen*Math.cos( hfov/2), fovLen*Math.sin( hfov/2));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
      // drone
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.fillStyle = '#fbbf24'; ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(15,0); ctx.lineTo(-10,-10); ctx.lineTo(-10,10); ctx.closePath();
      ctx.fill(); ctx.stroke();
      const vx = s.vx_cms||0, vy = s.vy_cms||0;
      if (Math.hypot(vx,vy) > 2) {
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(vx*0.7, vy*0.7); ctx.stroke();
      }
      ctx.restore();
      ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center';
      ctx.fillText(dist.toFixed(2)+' m', (drone_x+cx)/2+14, (drone_y+marker_y)/2+4);
      ctx.fillStyle = '#cbd5e1'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
      ctx.fillText('lateral '+lateral_m.toFixed(2)+' m', 8, H-22);
      ctx.fillText('dist err '+(dist-target).toFixed(2)+' m', 8, H-6);
      ctx.textAlign = 'right';
      ctx.fillText('target '+target.toFixed(2)+' m', W-8, H-22);
      ctx.fillText('err_x '+(s.err_x||0).toFixed(3), W-8, H-6);
    }

    async function arcPoll() {
      try {
        const r = await fetch('/proxy/aruco/state');
        const s = await r.json();
        const st = document.getElementById('arc_status');
        if (s.running) { st.textContent = '● ' + (s.status_msg || 'running'); st.style.color = '#22c55e'; }
        else           { st.textContent = '○ stopped';                          st.style.color = '#94a3b8'; }
        const dl = document.getElementById('arc_drone_label');
        if (s.drone_id && arcActiveId !== s.drone_id) {
          arcActiveId = s.drone_id;
        }
        if (arcActiveId) dl.textContent = '[drone '+arcActiveId+']';
        arcApplyModeUI(s.mode || 'observe', s.allow_live);
        arcRenderReadout(s);
        arcDrawTopDown(s);
      } catch(e) {}
    }

    // Buttons
    document.getElementById('arc_start').onclick = async () => {
      await fetch('/proxy/aruco/start', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '/proxy/aruco/video.mjpg?t=' + Date.now();
      av.setAttribute('data-active', '1');
      // Reload params for the now-active drone
      arcLoadParams();
    };
    document.getElementById('arc_stop').onclick = async () => {
      await fetch('/proxy/aruco/stop', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '';
      av.removeAttribute('data-active');
    };
    document.getElementById('arc_target_lock').onclick = async () => {
      const v = document.getElementById('arc_target_input').value;
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: v ? parseInt(v) : null})});
    };
    document.getElementById('arc_target_auto').onclick = async () => {
      document.getElementById('arc_target_input').value = '';
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: null})});
    };
    document.getElementById('arc_reload').onclick = arcLoadParams;

    function arcShowModeErr(msg, kind) {
      const el = document.getElementById('arc_mode_err');
      if (!el) return;
      clearTimeout(arcShowModeErr._t);
      if (!msg) { el.style.display = 'none'; return; }
      // kind: 'warn' = amber, default = red
      const warn = (kind === 'warn') || msg.startsWith('⚠');
      el.textContent = warn ? msg : ('✗ ' + msg);
      el.style.background = warn ? '#78350f' : '#7f1d1d';
      el.style.color      = warn ? '#fde68a' : '#fecaca';
      el.style.borderColor = warn ? '#f59e0b' : '#ef4444';
      el.style.display = 'inline';
      // Auto-hide warnings in 3s, errors in 8s
      arcShowModeErr._t = setTimeout(() => { el.style.display = 'none'; }, warn ? 3200 : 8000);
    }
    async function arcSetMode(mode) {
      console.log('[arc] set-mode request:', mode);
      // Clear previous error
      document.getElementById('arc_mode_err').style.display = 'none';
      try {
        const r = await fetch('/proxy/aruco/mode', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({mode})});
        let j;
        try { j = await r.json(); } catch { j = {}; }
        console.log('[arc] set-mode response:', r.status, j);
        if (!r.ok || !j.ok) {
          const msg = (j.error || j.message || r.statusText || 'unknown') + ' (HTTP ' + r.status + ')';
          arcShowModeErr(msg);
          return;
        }
        arcApplyModeUI(j.mode || mode, arcAllowLive);
      } catch (err) {
        console.error('[arc] set-mode failed:', err);
        arcShowModeErr('network error: ' + err);
      }
    }
    document.getElementById('arc_mode_observe').onclick = () => arcSetMode('observe');
    // LIVE button: always post to server, let it decide. Client-side arcAllowLive
    // is now just a UI hint (disabled state) — if somehow wrong, the server
    // returns 403 with a clear error message.
    // LIVE button: double-click / re-click confirmation (no browser confirm()).
    // First click arms — button pulses + warning shows.
    // Second click within 3 seconds → switches mode.
    // Keeps us off native confirm() dialogs which some browsers auto-dismiss.
    const liveBtn = document.getElementById('arc_mode_live');
    let _arcArmedUntil = 0;
    if (liveBtn) {
      liveBtn.onclick = () => {
        const now = Date.now();
        console.log('[arc] LIVE clicked, arcAllowLive=', arcAllowLive, 'armed=', (now < _arcArmedUntil));
        if (now < _arcArmedUntil) {
          // Armed → actually switch
          _arcArmedUntil = 0;
          liveBtn.style.animation = '';
          liveBtn.textContent = 'LIVE';
          arcShowModeErr(''); // hides
          arcSetMode('live');
          return;
        }
        // First click — arm for 3s
        _arcArmedUntil = now + 3000;
        liveBtn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        liveBtn.textContent = 'LIVE — click again';
        arcShowModeErr('⚠ Click LIVE again within 3 s to confirm');
        setTimeout(() => {
          if (Date.now() >= _arcArmedUntil) {
            _arcArmedUntil = 0;
            liveBtn.style.animation = '';
            liveBtn.textContent = 'LIVE';
            arcShowModeErr('');
          }
        }, 3100);
      };
      console.log('[arc] LIVE button click handler attached');
    } else {
      console.error('[arc] could not find arc_mode_live button — handler NOT attached');
    }

    async function arcPostCmd(path, confirmMsg) {
      if (confirmMsg && !confirm(confirmMsg)) return;
      const r = await fetch(path, {method:'POST'});
      const j = await r.json();
      if (!j.ok && j.error) alert(path + ' refused: ' + j.error);
    }
    document.getElementById('arc_takeoff').onclick   = () => arcPostCmd('/proxy/aruco/takeoff',   'Send TAKEOFF to the active drone?');
    document.getElementById('arc_land').onclick      = () => arcPostCmd('/proxy/aruco/land',      'Send LAND to the active drone?');
    document.getElementById('arc_rc_stop').onclick   = () => arcPostCmd('/proxy/aruco/rc_stop',   null);
    document.getElementById('arc_emergency').onclick = () => arcPostCmd('/proxy/aruco/emergency', '⛔ EMERGENCY STOP — cut motors immediately. Confirm?');

    arcLoadParams();
    setInterval(arcPoll, 250);
    arcPoll();
    // Version marker — if this string doesn't appear in the DOM,
    // you're running stale JS (restart the Python server or hard-refresh).
    const BUILD = 'au-capture-targets-mission';
    console.log('[arc] init complete, build=' + BUILD);
    const ver = document.createElement('span');
    ver.id = 'arc_build_tag';
    ver.style.cssText = 'font-size:10px;color:#10b981;margin-left:8px;font-weight:700;';
    ver.textContent = 'build ' + BUILD;
    const hdr = document.querySelector('#aruco_panel h3');
    if (hdr) hdr.appendChild(ver);

    // Immediate visible click counter — proves the button event fires, even
    // if the subsequent fetch hangs or the server is wedged. Counts every
    // LIVE / OBSERVE / Takeoff / Land / RC-stop / Emergency press.
    let _arcClicks = 0;
    const clickTag = document.createElement('span');
    clickTag.id = 'arc_click_counter';
    clickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;';
    clickTag.textContent = 'clicks: 0';
    if (hdr) hdr.appendChild(clickTag);
    function arcBumpClicks(src) {
      _arcClicks += 1;
      clickTag.textContent = 'clicks: ' + _arcClicks + ' (' + src + ')';
      console.log('[arc] click #' + _arcClicks + ' from ' + src);
    }
    // Wire onto the existing buttons defensively. We use addEventListener so
    // we don't overwrite the onclick handlers that actually do the work.
    ['arc_mode_observe','arc_mode_live','arc_takeoff','arc_land','arc_rc_stop','arc_emergency','arc_start','arc_stop'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', () => arcBumpClicks(id));
    });
  })();
  </script>

  <!-- ── Special Missions client JS ───────────────────────────────── -->
  <script>
  (function(){
    let misDronesKnown = {};

    async function misLoadDrones() {
      try {
        const r = await fetch('/proxy/drones');
        const d = await r.json();
        misDronesKnown = d.drones || {};
        const cont = document.getElementById('mis_drones');
        const prev = {};
        cont.querySelectorAll('input[type=checkbox]').forEach(cb => { prev[cb.dataset.id] = cb.checked; });
        cont.innerHTML = '';
        const ids = Object.keys(misDronesKnown).sort();
        ids.forEach(id => {
          const info = misDronesKnown[id];
          const wrap = document.createElement('label');
          wrap.style.cssText = 'display:flex;gap:4px;align-items:center;font-size:12px;color:#e2e8f0;cursor:pointer;';
          const checked = (id in prev) ? prev[id] : true;
          wrap.innerHTML = '<input type=\"checkbox\" data-id=\"'+id+'\"'+(checked?' checked':'')+' /> '+info.name+' <span style=\"color:#64748b;\">#'+id+'</span>';
          cont.appendChild(wrap);
        });
      } catch {}
    }

    function misSelectedDroneIds() {
      return Array.from(document.querySelectorAll('#mis_drones input[type=checkbox]'))
        .filter(cb => cb.checked).map(cb => cb.dataset.id);
    }

    function misBadge(phase) {
      const low = (phase || 'idle').toLowerCase();
      let cls = 'idle';
      if (low === 'search') cls = 'scan';
      else if (low === 'approach') cls = 'approach';
      else if (low === 'hover') cls = 'hover';
      else if (low === 'done') cls = 'done';
      else if (low === 'error') cls = 'error';
      return '<span class=\"mis-badge '+cls+'\">'+phase+'</span>';
    }

    function misRenderStatus(st) {
      const title = document.getElementById('mis_title_status');
      const prog = document.getElementById('mis_progress');
      const panel = document.getElementById('mis_status');

      if (!st.has_mission) {
        title.textContent = 'idle';
        title.style.color = '#94a3b8';
        prog.textContent = '—';
        document.getElementById('mis_scanned').textContent = '—';
        document.getElementById('mis_remaining').textContent = '—';
        panel.textContent = 'idle — no mission running';
        return;
      }
      title.textContent = st.active ? '● running' : '○ stopped';
      title.style.color = st.active ? '#4ade80' : '#94a3b8';
      prog.textContent = 'Progress: ' + st.progress;
      document.getElementById('mis_scanned').textContent = (st.scanned || []).join(', ') || '—';
      document.getElementById('mis_remaining').textContent = (st.remaining || []).join(', ') || '—';

      let html = '';
      const drones = st.drones || {};
      Object.keys(drones).sort().forEach(did => {
        const d = drones[did];
        const name = (misDronesKnown[did] && misDronesKnown[did].name) || ('Drone '+did);
        const tgt = d.target != null ? 'target '+d.target : '—';
        html += '<div class=\"mis-drone-line\"><b>'+name+' #'+did+'</b> '+misBadge(d.phase)+
                ' <span style=\"color:#64748b;\">['+tgt+']</span> <span style=\"color:#cbd5e1;\">'+(d.note||'')+'</span></div>';
      });
      if (st.error) html += '<div style=\"color:#fca5a5;margin-top:6px;\">error: '+st.error+'</div>';
      panel.innerHTML = html || 'no drones assigned';
    }

    async function misPoll() {
      try {
        const r = await fetch('/proxy/missions/status');
        misRenderStatus(await r.json());
      } catch {}
    }

    // Click-to-arm pattern for Start Mission — no native confirm() dialog
    // (some browsers auto-dismiss rapid-fire dialogs, making the mission
    // never start with no visible feedback).
    let _misArmedUntil = 0;
    document.getElementById('mis_start').onclick = async () => {
      const drone_ids = misSelectedDroneIds();
      const misErr = document.getElementById('mis_err');
      const misOk  = document.getElementById('mis_ok');
      function misShowWarn(msg) {
        if (!misErr) return;
        misErr.textContent = msg;
        misErr.style.background = '#78350f';
        misErr.style.color = '#fde68a';
        misErr.style.borderColor = '#f59e0b';
        misErr.style.display = 'inline-block';
      }
      function misClearMsgs() {
        if (misErr) misErr.style.display = 'none';
        if (misOk)  misOk.style.display  = 'none';
      }
      if (!drone_ids.length) {
        misShowWarn('✗ Select at least one drone first');
        return;
      }
      const missionType = document.getElementById('mis_type').value;
      const auto_takeoff = document.getElementById('mis_auto_takeoff').checked;
      let endpoint, payload;
      if (missionType === 'capture_targets') {
        // Parse target-boxes JSON
        let boxes = [];
        try { boxes = JSON.parse(document.getElementById('mis_boxes_json').value); }
        catch (e) { misShowWarn('✗ target_boxes JSON invalid: ' + e.message); return; }
        if (!Array.isArray(boxes) || boxes.length === 0) {
          misShowWarn('✗ target_boxes must be a non-empty JSON array'); return;
        }
        const home_xy = [
          parseFloat(document.getElementById('mis_home_x').value) || 0,
          parseFloat(document.getElementById('mis_home_y').value) || 0,
        ];
        const face_xy = [
          parseFloat(document.getElementById('mis_face_x').value) || 0,
          parseFloat(document.getElementById('mis_face_y').value) || 0,
        ];
        const alt = parseFloat(document.getElementById('mis_alt').value) || 1.5;
        const hv  = parseFloat(document.getElementById('mis_cap_hover_s').value) || 4.0;
        endpoint = '/proxy/missions/capture_targets/start';
        payload = {
          drone_ids, target_boxes: boxes,
          home_xy, arena_face_xy: face_xy,
          hover_above_m: alt, hover_seconds: hv,
          auto_takeoff,
        };
      } else {
        // scan_all (default)
        const markers = document.getElementById('mis_markers').value;
        const hover_seconds = parseFloat(document.getElementById('mis_hover_s').value) || 3.0;
        const tol = parseFloat(document.getElementById('mis_tol_m').value) || 0.35;
        const skew_tol_el = document.getElementById('mis_skew_tol');
        const skew_tol = skew_tol_el ? (parseFloat(skew_tol_el.value) || 0.08) : 0.08;
        endpoint = '/proxy/missions/scan_all/start';
        payload = {
          drone_ids, target_markers: markers,
          hover_seconds, approach_tolerance_m: tol,
          approach_skew_tol: skew_tol,
          auto_takeoff,
        };
      }
      const btn = document.getElementById('mis_start');
      const now = Date.now();
      if (now >= _misArmedUntil) {
        // First click — arm for 3s, show summary inline
        _misArmedUntil = now + 3000;
        const origLabel = btn.textContent;
        btn._origLabel = origLabel;
        btn.textContent = '⚠ Click again to launch';
        btn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        const summary = 'Drones: ' + drone_ids.join(', ') +
                        '   •   Markers: ' + markers +
                        '   •   Hover ' + hover_seconds + 's' +
                        (auto_takeoff ? '   •   ⚠ AUTO-TAKEOFF' : '');
        misShowWarn('⚠ Starting mission — ' + summary + '. Click Start again within 3 s.');
        setTimeout(() => {
          if (Date.now() >= _misArmedUntil) {
            _misArmedUntil = 0;
            btn.textContent = btn._origLabel || '▶ Start mission';
            btn.style.animation = '';
            misClearMsgs();
          }
        }, 3100);
        return;
      }
      // Armed — proceed with launch
      _misArmedUntil = 0;
      btn.style.animation = '';
      btn.textContent = btn._origLabel || '▶ Start mission';
      misClearMsgs();
      const origLabel = btn.textContent;
      btn.disabled = true; btn.textContent = '… starting';
      let j = {}; let httpStatus = 0;
      try {
        const r = await fetch(endpoint, {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)});
        httpStatus = r.status;
        try { j = await r.json(); } catch { j = {}; }
      } catch (err) {
        j = { ok: false, error: 'network error: ' + err };
      } finally {
        btn.disabled = false; btn.textContent = origLabel;
      }
      console.log('[mission] start response:', httpStatus, j);
      if (!j.ok) {
        const msg = (j.error || j.message || 'unknown error') + (httpStatus ? ' (HTTP ' + httpStatus + ')' : '');
        if (misErr) {
          misErr.textContent = '✗ Mission start refused: ' + msg;
          misErr.style.display = 'inline-block';
        }
      } else {
        const msg = j.message || 'mission started';
        const warn = j.status && j.status.error ? ' — warning: ' + j.status.error : '';
        console.log('[mission] started:', msg, warn);
        if (misOk) {
          misOk.textContent = '✓ ' + msg + warn;
          misOk.style.display = 'inline-block';
        }
      }
      misPoll();
    };

    document.getElementById('mis_stop').onclick = async () => {
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:false})});
      misPoll();
    };

    document.getElementById('mis_stop_land').onclick = async () => {
      if (!confirm('Stop mission AND land all participating drones?')) return;
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:true})});
      misPoll();
    };

    misLoadDrones();
    // If the drone config changes, the drones list in the ArUco section
    // repopulates via /proxy/drones poll — mirror that for missions.
    setInterval(misLoadDrones, 5000);
    setInterval(misPoll, 500);
    misPoll();

    // Show/hide the capture-targets-specific rows based on mission type
    const misTypeSel = document.getElementById('mis_type');
    const misCaptureRows = document.getElementById('mis_capture_rows');
    const misScanRows = document.querySelector('#missions_panel .mis-row:nth-of-type(3)');
    function syncMissionUI() {
      const t = misTypeSel.value;
      if (misCaptureRows) misCaptureRows.style.display = (t === 'capture_targets') ? '' : 'none';
      if (misScanRows)    misScanRows.style.display    = (t === 'capture_targets') ? 'none' : '';
    }
    if (misTypeSel) { misTypeSel.addEventListener('change', syncMissionUI); syncMissionUI(); }

    // Mission panel click counter — same idea as the ArUco one. Proves the
    // Start / Stop buttons receive the click event, independent of what
    // the server does with the request afterwards.
    const misHdr = document.querySelector('#missions_panel h3');
    if (misHdr) {
      const misClickTag = document.createElement('span');
      misClickTag.id = 'mis_click_counter';
      misClickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;font-weight:400;';
      misClickTag.textContent = 'clicks: 0';
      misHdr.appendChild(misClickTag);
      let _misClicks = 0;
      ['mis_start','mis_stop','mis_stop_land'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => {
          _misClicks += 1;
          misClickTag.textContent = 'clicks: ' + _misClicks + ' (' + id + ')';
          console.log('[mission] click #' + _misClicks + ' from ' + id);
        });
      });
    }
  })();
  </script>

  <div class=\"row\" style=\"margin-top:12px;\">
    <div class=\"panel pos-panel\">
      <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;\">
        <b>Position Tracker</b>
        <label style=\"display:flex;align-items:center;gap:5px;font-size:12px;color:#94a3b8;cursor:pointer;\">
          <input type=\"checkbox\" id=\"pos_enabled\" style=\"accent-color:#0ea5e9;\" />
          Enable
        </label>
        <span id=\"pos_status_badge\" class=\"small\" style=\"color:#64748b;\">disabled</span>
      </div>
      <div style=\"display:flex;gap:8px;align-items:center;margin-bottom:6px;\">
        <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\">
          <input type=\"checkbox\" id=\"arena_show_3d\" style=\"accent-color:#0ea5e9;\" />
          3D view (Three.js)
        </label>
        <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\">
          <input type=\"checkbox\" id=\"arena_show_all_drones\" checked style=\"accent-color:#10b981;\" />
          Show all drones
        </label>
        <span class=\"small\" style=\"color:#64748b;margin-left:12px;\">Grid: 1 m</span>
      </div>
      <canvas id=\"arena_canvas\" class=\"arena-canvas\" width=\"960\" height=\"560\" style=\"max-width:100%;\"></canvas>
      <div id=\"arena3d_wrap\" style=\"display:none;margin-top:8px;position:relative;\">
        <div id=\"arena3d_container\" style=\"width:960px;max-width:100%;height:520px;background:#0f172a;border:1px solid #334155;border-radius:6px;\"></div>
        <div class=\"small\" style=\"color:#64748b;margin-top:4px;\">Drag to orbit · scroll to zoom · right-drag to pan</div>
      </div>
      <div class=\"pos-coords\" style=\"margin-top:6px;\">
        <span class=\"pos-x\">X: <span id=\"pos_x\">—</span></span>&nbsp;&nbsp;
        <span class=\"pos-y\">Y: <span id=\"pos_y\">—</span></span>&nbsp;&nbsp;
        <span class=\"pos-z\">Z: <span id=\"pos_z\">—</span></span>
      </div>
      <div class=\"small\" style=\"color:#94a3b8;\">
        Hdg: <span id=\"pos_hdg\">—</span>°&nbsp;&nbsp;
        Vel: <span id=\"pos_vel\">—</span>&nbsp;&nbsp;
        Refs: <span id=\"pos_refs\">—</span>&nbsp;&nbsp;
        FPS: <span id=\"pos_fps\">—</span>
        <span id=\"pos_stale\" class=\"pos-stale\" style=\"display:none;\"> ⚠ STALE</span>
      </div>
      <div class=\"small\" style=\"color:#94a3b8;margin-top:2px;\">
        Vx:&nbsp;<span id=\"tel_vx\">—</span>&nbsp;
        Vy:&nbsp;<span id=\"tel_vy\">—</span>&nbsp;
        Vz:&nbsp;<span id=\"tel_vz\">—</span>&nbsp;cm/s&nbsp;&nbsp;&nbsp;
        Ax:&nbsp;<span id=\"tel_ax\">—</span>&nbsp;
        Ay:&nbsp;<span id=\"tel_ay\">—</span>&nbsp;
        Az:&nbsp;<span id=\"tel_az\">—</span>&nbsp;cm/s²
      </div>
      <div class=\"pos-cfg\">
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;\">
          <label>Profile:
            <select id=\"pos_profile\" class=\"pos-cfg\">
              <option value=\"balanced\">Balanced</option>
              <option value=\"sensitive\">Sensitive</option>
              <option value=\"strict\">Strict</option>
            </select>
          </label>
          <label>FOV°:
            <input id=\"pos_fov\" type=\"number\" min=\"40\" max=\"120\" value=\"69\" style=\"width:60px;\" />
          </label>
          <label>Latency ms:
            <input id=\"pos_latency\" type=\"range\" min=\"0\" max=\"800\" value=\"200\" style=\"width:90px;vertical-align:middle;\" />
            <span id=\"pos_latency_val\" style=\"font-size:11px;color:#94a3b8;\">200</span>
          </label>
          <label title=\"0 = pure ArUco (vision only), 1 = pure IMU dead-reckoning. Higher = more IMU smoothing, less vision jitter.\" style=\"display:inline-flex;align-items:center;gap:4px;\">
            <span style=\"font-size:11px;color:#94a3b8;\">ArUco</span>
            <input id=\"pos_imu_weight\" type=\"range\" min=\"0\" max=\"100\" value=\"30\" style=\"width:110px;vertical-align:middle;accent-color:#06b6d4;\" />
            <span style=\"font-size:11px;color:#94a3b8;\">IMU</span>
            <span id=\"pos_imu_weight_val\" style=\"font-size:11px;color:#06b6d4;font-weight:bold;min-width:32px;\">30%</span>
          </label>
        </div>
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;padding:6px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <span class=\"small\" style=\"color:#64748b;min-width:70px;\">Filters:</span>
          <label style=\"display:flex;align-items:center;gap:4px;cursor:pointer;\">
            <input type=\"checkbox\" id=\"pos_kalman\" style=\"accent-color:#3b82f6;\" />
            <span class=\"small\" style=\"color:#e2e8f0;\">Kalman filter</span>
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Marker size (m):
            <input id=\"pos_marker_size\" type=\"number\" min=\"0.05\" max=\"2.0\" step=\"0.01\" value=\"0.5\" style=\"width:64px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Top-K:
            <input id=\"pos_top_k\" type=\"number\" min=\"0\" max=\"10\" step=\"1\" value=\"0\" style=\"width:50px;\" title=\"0 = auto (4)\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Outlier (m):
            <input id=\"pos_outlier\" type=\"number\" min=\"0.1\" max=\"20\" step=\"0.1\" value=\"2.5\" style=\"width:60px;\" />
          </label>
          <button id=\"pos_filters_apply\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;\">Apply filters</button>
          <button id=\"pos_filters_reset\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;background:#1e2a3a;\" title=\"Restore defaults: Kalman on, marker 0.5m, top-K auto, outlier 2.5m\">Defaults</button>
          <span id=\"pos_filters_status\" class=\"small\" style=\"color:#64748b;\"></span>
        </div>
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;\">
          <button id=\"pos_cfg_save\" class=\"pos-cfg\">Apply Config</button>
          <label style=\"cursor:pointer;\">
            <button id=\"pos_calib_btn\" class=\"pos-cfg\" onclick=\"document.getElementById('pos_calib_file').click()\">Upload Calibration (.npz)</button>
            <input type=\"file\" id=\"pos_calib_file\" accept=\".npz\" style=\"display:none;\" />
          </label>
          <span id=\"pos_calib_status\" class=\"small\" style=\"color:#94a3b8;\"></span>
        </div>
        <div style=\"margin-top:6px;display:flex;gap:8px;align-items:center;\">
          <button id=\"pos_video_toggle\" class=\"pos-cfg\">Show ArUco Video</button>
          <button id=\"rec_btn\" class=\"pos-cfg\" style=\"background:#1e3a2e;border-color:#22c55e;color:#22c55e;\">&#9679; Record</button>
          <label style=\"display:flex;align-items:center;gap:4px;font-size:12px;color:#94a3b8;cursor:pointer;\">
            <input type=\"checkbox\" id=\"rec_raw\" style=\"accent-color:#22c55e;\" /> Raw
          </label>
          <span id=\"rec_status\" class=\"small\" style=\"color:#64748b;\"></span>
          <a id=\"pos_tracker_link\" href=\"#\" target=\"_blank\" class=\"small\" style=\"color:#38bdf8;display:none;\">Open full arena tracker ↗</a>
        </div>
        <div id=\"pos_video_container\" style=\"display:none;margin-top:6px;\">
          <img id=\"pos_video_img\" src=\"\" style=\"max-width:100%;border-radius:6px;background:#000;\" />
        </div>
      </div>
    </div>
  </div>

  <div class=\"row\" style=\"margin-top:12px;\">
    <div class=\"panel\" style=\"min-width:340px;flex:1;\">
      <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;\">
        <b>Arena Configuration</b>
        <span class=\"small\" style=\"color:#94a3b8;\">marker layout &amp; physical dimensions</span>
        <button id=\"arena_cfg_toggle\" style=\"margin-left:auto;padding:2px 10px;font-size:11px;background:#1e293b;\">Show</button>
      </div>
      <div id=\"arena_cfg_body\" style=\"display:none;\">
        <div style=\"display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;\">
          <label class=\"small\">Arena width (m):
            <input id=\"ac_width\" type=\"number\" step=\"0.5\" value=\"20\" style=\"width:70px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Arena depth (m):
            <input id=\"ac_depth\" type=\"number\" step=\"0.5\" value=\"10\" style=\"width:70px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Height min (m):
            <input id=\"ac_hmin\" type=\"number\" step=\"0.1\" value=\"-1\" style=\"width:60px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Height max (m):
            <input id=\"ac_hmax\" type=\"number\" step=\"0.1\" value=\"1\" style=\"width:60px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Marker size (m):
            <input id=\"ac_msize\" type=\"number\" step=\"0.05\" value=\"0.5\" style=\"width:65px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
        </div>
        <div class=\"small\" style=\"margin-bottom:4px;color:#94a3b8;\">Marker positions (ID · X · Y · Z · Wall)</div>
        <div id=\"arena_marker_table\" style=\"max-height:280px;overflow-y:auto;font-size:11px;\"></div>
        <div style=\"margin-top:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;\">
          <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;\">ID:
            <input id=\"arena_new_marker_id\" type=\"number\" min=\"0\" step=\"1\" placeholder=\"auto\" style=\"width:60px;height:26px;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" title=\"Marker ID to add (leave blank for next free)\" />
          </label>
          <button id=\"arena_add_marker\" style=\"padding:4px 10px;font-size:11px;background:#065f46;border-color:#10b981;\">+ Add Marker</button>
          <button id=\"arena_save\" style=\"padding:4px 10px;font-size:11px;background:#1e3a5f;border-color:#3b82f6;\">Save Config</button>
          <button id=\"arena_reset\" style=\"padding:4px 10px;font-size:11px;background:#374151;border-color:#6b7280;\">Reset to Defaults</button>
          <span id=\"arena_cfg_status\" class=\"small\" style=\"color:#94a3b8;\"></span>
        </div>
      </div>
    </div>
  </div>
<script>
// --- Drone fleet selector ---
let drones = {};
let activeDroneId = null;

async function loadDrones() {
  try {
    const r = await fetch('/proxy/drones');
    const d = await r.json();
    drones = d.drones;
    activeDroneId = d.active;
    renderDroneBar();
    updatePiLabel();
  } catch {}
}

function renderDroneBar() {
  const bar = document.getElementById('drone_bar');
  bar.innerHTML = '';
  let slot = 1;
  for (const [id, info] of Object.entries(drones)) {
    const btn = document.createElement('button');
    btn.className = 'drone-btn' + (id === activeDroneId ? ' selected' : '');
    const slotBadge = (slot <= 5)
      ? `<span style="background:#1e3a5f;color:#93c5fd;padding:0 4px;margin-right:4px;border-radius:3px;font-size:10px;font-weight:700;">${slot}</span>`
      : '';
    btn.innerHTML = `${slotBadge}${info.name}<span class="drone-type">${info.type}</span>`;
    btn.title = (slot <= 5) ? `Hotkey: ${slot}` : '';
    btn.onclick = () => switchDrone(id);
    bar.appendChild(btn);
    slot += 1;
  }
  // Show/hide Anafi panel based on drone type
  const anafiPanel = document.getElementById('anafi_panel');
  if (anafiPanel) {
    const droneType = drones[activeDroneId]?.type || '';
    anafiPanel.style.display = droneType === 'anafi' ? '' : 'none';
  }
}

async function switchDrone(id) {
  if (id === activeDroneId) return;
  // Release all keys on the current drone before switching
  releaseAllKeys();
  try {
    await fetch('/proxy/switch', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:id})});
    activeDroneId = id;
    renderDroneBar();
    updatePiLabel();
    // Restart SSE streams for new drone
    startTelemetrySSE();
    if (document.getElementById('pos_enabled').checked) startPosEvents();
    refreshTelemetry();
    // ── Video feed must also switch to the new drone ───────────────
    // /proxy/video, /proxy/position/video and /proxy/aruco/video.mjpg
    // all proxy to the ACTIVE drone on the server. The browser's
    // existing <img> MJPEG connections are pinned to the OLD drone
    // (long-lived HTTP, established when src was set), so we have to
    // tear them down and re-establish. Be aggressive: always do this,
    // regardless of videoActive/src-empty flags.
    try {
      // 1) Guarantee Way-1 MJPEG is running on the new drone.
      //    Fire-and-forget; if it's already running, the server returns ok.
      fetch('/proxy/video/start', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode:'mjpeg'})}).catch(()=>{});

      // 2) Force every <img> to drop its current MJPEG connection.
      //    'about:blank' is more aggressive than an empty string; it
      //    actually tears down the existing HTTP socket.
      const elements = [
        { el: document.getElementById('video_img'),     src: '/proxy/video?' },
        { el: document.getElementById('pos_video_img'), src: '/proxy/position/video?' },
        { el: document.getElementById('arc_video'),     src: '/proxy/aruco/video.mjpg?t=' },
      ];
      elements.forEach(({el}) => { if (el) el.src = 'about:blank'; });

      // 3) After the browser has had time to close the old connections
      //    (browsers process src changes async — 300 ms is a safe
      //    upper bound), reconnect every feed with a fresh cache-buster.
      //    Main feed is always reconnected (even if user hadn't pressed
      //    Start Video on the old drone, the new drone should also
      //    show its feed). Position and ArUco feeds only reconnect
      //    if they had been displayed on the previous drone.
      const posVisible = document.getElementById('pos_video_img') &&
        document.getElementById('pos_video_img').parentElement &&
        document.getElementById('pos_video_img').parentElement.style.display !== 'none';
      const arcVisible = document.getElementById('arc_video') &&
        document.getElementById('arc_video').getAttribute('data-active') === '1';
      setTimeout(() => {
        const ts = Date.now();
        const mainImg = document.getElementById('video_img');
        if (mainImg) { mainImg.src = '/proxy/video?' + ts; videoActive = true; }
        if (posVisible) {
          document.getElementById('pos_video_img').src = '/proxy/position/video?' + ts;
        }
        if (arcVisible) {
          document.getElementById('arc_video').src = '/proxy/aruco/video.mjpg?t=' + ts;
        }
        console.log('[drone-switch] video reconnected → drone', id,
                    ' main=yes pos=', posVisible, ' arc=', arcVisible);
      }, 300);
    } catch (err) {
      console.warn('[drone-switch] video reconnect failed:', err);
    }
    // Refresh the environment + Wi-Fi status for the newly selected drone.
    // Both readouts are per-drone so they need to re-fetch after the switch.
    try {
      if (typeof envRefresh  === 'function') setTimeout(envRefresh,  400);
      if (typeof wifiRefresh === 'function') setTimeout(wifiRefresh, 400);
    } catch (e) {}
  } catch {}
}

function updatePiLabel() {
  const info = drones[activeDroneId];
  document.getElementById('pi').textContent = info ? `${info.name} (${info.type}) @ ${info.base}` : '-';
}

async function post(url, body){
  try {
    await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  } catch {}
}
function keyDown(k){ post('/proxy/key_down',{key:k}); }
function keyUp(k){ post('/proxy/key_up',{key:k}); }

const activeKeys = new Set();
function pressKey(k){
  if (!activeKeys.has(k)) {
    activeKeys.add(k);
    keyDown(k);
  }
}
function releaseKey(k){
  if (activeKeys.has(k)) {
    activeKeys.delete(k);
    keyUp(k);
  }
}
function releaseAllKeys(){
  Array.from(activeKeys).forEach(releaseKey);
}

// Batch all active key states into a single POST (instead of one POST per key)
setInterval(()=>{
  if (activeKeys.size === 0) return;
  const keys = Array.from(activeKeys);
  // Send as single batch request
  fetch('/proxy/key_batch', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({keys})}).catch(()=>{});
}, 100);

const holdButtons = document.querySelectorAll('button[data-k]');
holdButtons.forEach(btn=>{
  const k = btn.dataset.k;
  btn.addEventListener('pointerdown', e=>{ e.preventDefault(); btn.classList.add('active'); pressKey(k); });
  btn.addEventListener('pointerup',   e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointerleave',e=>{ btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointercancel',e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
});

// Takeoff + surface any server-side refusal (magneto / sensors / battery / …).
// /api/takeoff on Anafi returns a message like
//   "takeoff_failed (magneto=REQUIRED, axes=x0y0z0)"
// so we parse it, show it, and offer the magneto wizard if that's the cause.
async function tryTakeoff(){
  hideTakeoffError();
  let resp, body;
  try {
    resp = await fetch('/proxy/takeoff', {method:'POST',
                       headers:{'Content-Type':'application/json'}, body:'{}'});
    body = await resp.json().catch(()=>({ok:false, error:'non-JSON response'}));
  } catch (e) {
    showTakeoffError('network error contacting drone: ' + e, '');
    return;
  }
  if (!resp.ok || body.ok === false) {
    const err = (body && body.error) || ('HTTP ' + resp.status);
    let hint = '';
    if (/magneto/i.test(err))
      hint = 'Magnetometer needs figure-8 calibration before the drone will arm.';
    else if (/sensor/i.test(err))
      hint = 'A sensor check failed — verify GPS lock (outdoor mode), battery, and motor state.';
    else if (/battery|low/i.test(err))
      hint = 'Battery too low for takeoff.';
    else if (/motor/i.test(err))
      hint = 'Motor fault — inspect props and power-cycle the drone.';
    else if (/not.?ready|not.?connected/i.test(err))
      hint = 'Controller is not connected. Check Wi-Fi link and API server logs.';
    else if (/angle|tilt|level/i.test(err))
      hint = 'Drone is not level — place on a flat surface and retry.';
    showTakeoffError(err, hint);
  }
}

function showTakeoffError(reason, hint){
  const box = document.getElementById('takeoff_err');
  document.getElementById('takeoff_err_reason').textContent = reason;
  document.getElementById('takeoff_err_hint').textContent = hint || '';
  document.getElementById('takeoff_err_mag').style.display =
    /magneto/i.test(reason) ? '' : 'none';
  box.classList.add('show');
}
function hideTakeoffError(){
  document.getElementById('takeoff_err').classList.remove('show');
}

document.getElementById('takeoff').onclick = tryTakeoff;
document.getElementById('takeoff_err_dismiss').onclick = hideTakeoffError;
document.getElementById('takeoff_err_mag').onclick = ()=>{ hideTakeoffError(); openMagnetoWizard(); };
document.getElementById('land').onclick = ()=>post('/proxy/land',{});
document.getElementById('recover').onclick = ()=>post('/proxy/recover',{});

document.getElementById('safe_takeoff').onclick = async ()=>{
  try {
    const r = await fetch('/proxy/safety/takeoff');
    const s = await r.json();
    await post('/proxy/safety/takeoff', {enabled: !Boolean(s.enabled)});
    refreshSafeTakeoff();
  } catch {}
};

document.getElementById('toggle_log').onclick = async ()=>{
  try {
    const s = await fetch('/proxy/logging/telemetry');
    const cur = await s.json();
    const nextEnabled = !Boolean(cur.enabled);
    await post('/proxy/logging/telemetry',{enabled: nextEnabled});
    document.getElementById('toggle_log').textContent = nextEnabled ? 'Disable Telemetry Log' : 'Enable Telemetry Log';
  } catch {}
};

document.getElementById('download_log').onclick = ()=>{
  window.open('/proxy/logging/telemetry/download', '_blank');
};

document.getElementById('clear_log').onclick = async ()=>{
  try {
    await post('/proxy/logging/telemetry/clear', {});
  } catch {}
};

// (emergency killswitch button removed — was too easy to hit by accident)
document.getElementById('rotate_cw').onclick = ()=>post('/proxy/rotate',{dir:'cw',deg:45});
document.getElementById('rotate_ccw').onclick = ()=>post('/proxy/rotate',{dir:'ccw',deg:45});
document.getElementById('move_up').onclick = ()=>post('/proxy/move',{dir:'up',cm:30});
document.getElementById('move_down').onclick = ()=>post('/proxy/move',{dir:'down',cm:30});
document.getElementById('move_fwd').onclick = ()=>post('/proxy/move',{dir:'forward',cm:30});
document.getElementById('move_back').onclick = ()=>post('/proxy/move',{dir:'back',cm:30});
document.getElementById('move_left').onclick = ()=>post('/proxy/move',{dir:'left',cm:30});
document.getElementById('move_right').onclick = ()=>post('/proxy/move',{dir:'right',cm:30});
document.getElementById('stream_on').onclick = ()=>post('/proxy/stream',{action:'on'});
document.getElementById('stream_off').onclick = ()=>post('/proxy/stream',{action:'off'});
document.getElementById('set_speed').onclick = ()=>{
  const v = Number(document.getElementById('speed_val').value || 30);
  post('/proxy/speed',{speed:v});
};
document.getElementById('sdk_send').onclick = ()=>{
  const cmd = document.getElementById('sdk_cmd').value || '';
  if (cmd.trim()) post('/proxy/sdk',{command:cmd.trim()});
};

// Anafi / Olympe controls
const gimbalSlider = document.getElementById('gimbal_tilt');
const gimbalVal = document.getElementById('gimbal_tilt_val');
gimbalSlider.oninput = ()=>{ gimbalVal.textContent = gimbalSlider.value + '°'; };
document.getElementById('gimbal_set').onclick = ()=>post('/proxy/gimbal',{tilt:Number(gimbalSlider.value),pan:0});
document.getElementById('gimbal_down').onclick = ()=>{ gimbalSlider.value=-90; gimbalVal.textContent='-90°'; post('/proxy/gimbal',{tilt:-90,pan:0}); };
document.getElementById('gimbal_fwd').onclick = ()=>{ gimbalSlider.value=0; gimbalVal.textContent='0°'; post('/proxy/gimbal',{tilt:0,pan:0}); };

document.getElementById('apply_settings').onclick = ()=>{
  const alt = Number(document.getElementById('set_alt').value);
  const vs = Number(document.getElementById('set_vspd').value);
  const tilt = Number(document.getElementById('set_tilt').value);
  post('/proxy/settings', {max_altitude_m:alt, max_vertical_speed:vs, max_tilt:tilt});
};

// ── Environment (indoor / outdoor) ──────────────────────────────────
async function envRefresh() {
  const lbl = document.getElementById('env_status');
  if (!lbl) return;
  try {
    const r = await fetch('/proxy/environment', {cache:'no-store'});
    const d = await r.json();
    if (d.ok && d.status) {
      const cur = (d.status.environement || d.status.environment || '').toString();
      lbl.textContent = 'current: ' + (cur || 'unknown');
      lbl.style.color = cur.toLowerCase().includes('indoor') ? '#22c55e' : '#94a3b8';
      // Sync dropdown
      const sel = document.getElementById('env_mode');
      if (sel && cur) {
        const v = cur.toLowerCase().includes('indoor') ? 'indoor' : 'outdoor';
        if (sel.value !== v) sel.value = v;
      }
    } else {
      lbl.textContent = d.error || 'unavailable';
      lbl.style.color = '#f59e0b';
    }
  } catch (e) { lbl.textContent = 'error'; lbl.style.color = '#ef4444'; }
}
(function(){
  const btn = document.getElementById('env_apply');
  if (!btn) return;
  btn.onclick = async () => {
    const mode = document.getElementById('env_mode').value;
    const lbl = document.getElementById('env_status');
    lbl.textContent = '… applying';
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/environment', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode})});
      const d = await r.json();
      if (d.ok) {
        lbl.textContent = '✓ set to ' + mode;
        lbl.style.color = '#22c55e';
        setTimeout(envRefresh, 800);
      } else {
        lbl.textContent = '✗ ' + (d.error || 'unknown');
        lbl.style.color = '#ef4444';
      }
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
// Initial read + refresh when the user switches drones
setTimeout(envRefresh, 800);

// ── Wi-Fi band + channel ────────────────────────────────────────────
async function wifiRefresh() {
  const lbl = document.getElementById('wifi_status');
  if (!lbl) return;
  try {
    const r = await fetch('/proxy/wifi/status', {cache:'no-store'});
    const d = await r.json();
    if (d.ok && d.status && typeof d.status === 'object') {
      const band = (d.status.band || '').toString();
      const ch   = d.status.channel;
      const typ  = d.status.type || '';
      lbl.textContent = `band=${band}  ch=${ch}  (${typ})`;
      const is5 = band.toLowerCase().includes('5');
      lbl.style.color = is5 ? '#22c55e' : '#fbbf24';
      // Sync dropdowns
      const bandSel = document.getElementById('wifi_band');
      if (bandSel) bandSel.value = is5 ? '5_GHz' : '2_4_GHz';
      const chSel = document.getElementById('wifi_channel');
      if (chSel && ch != null) {
        // Prefer auto when type=auto_*
        if (typ.toLowerCase().includes('auto')) {
          chSel.value = 'auto';
        } else if (Array.from(chSel.options).some(o => o.value == String(ch))) {
          chSel.value = String(ch);
        }
      }
    } else {
      lbl.textContent = d.error || d.status || 'unavailable';
      lbl.style.color = '#f59e0b';
    }
  } catch(e) { lbl.textContent = 'error'; lbl.style.color = '#ef4444'; }
}
(function(){
  const apply = document.getElementById('wifi_apply');
  if (!apply) return;
  apply.onclick = async () => {
    const band = document.getElementById('wifi_band').value;
    const chStr = document.getElementById('wifi_channel').value;
    const lbl = document.getElementById('wifi_status');
    const auto = (chStr === 'auto');
    const body = auto
      ? {auto: true, band}
      : {auto: false, band, channel: Number(chStr)};
    if (!confirm('⚠ Wi-Fi change will disconnect the drone for ~5-10 s.\\n\\n' +
                 'Drone MUST be on the ground. Continue?')) return;
    lbl.textContent = '… applying  (wait ~10 s for reconnect)';
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/wifi/channel', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)});
      const d = await r.json();
      if (d.ok) {
        lbl.textContent = `✓ ${d.mode}: ${band}${auto ? '' : ' ch ' + body.channel} — reconnecting…`;
        lbl.style.color = '#22c55e';
        // Poll status a few times while the watchdog reconnects
        let n = 0;
        const poll = setInterval(() => {
          wifiRefresh();
          if (++n > 8) clearInterval(poll);
        }, 2000);
      } else {
        lbl.textContent = '✗ ' + (d.error || 'unknown');
        lbl.style.color = '#ef4444';
      }
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
(function(){
  const scan = document.getElementById('wifi_scan');
  if (!scan) return;
  scan.onclick = async () => {
    const band = document.getElementById('wifi_band').value;
    const lbl = document.getElementById('wifi_status');
    lbl.textContent = '… scanning ' + band;
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/wifi/scan', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({band, wait_s: 4})});
      const d = await r.json();
      if (!d.ok) {
        lbl.textContent = '✗ ' + (d.error || 'scan failed');
        lbl.style.color = '#ef4444';
        return;
      }
      const items = d.scanned_items || [];
      // Aggregate per channel
      const perCh = {};
      items.forEach(it => {
        const ch = it.channel;
        if (ch == null) return;
        if (!perCh[ch]) perCh[ch] = 0;
        perCh[ch] += 1;
      });
      const summary = Object.keys(perCh)
        .sort((a,b)=>Number(a)-Number(b))
        .map(c => `ch${c}:${perCh[c]}`).join(' ');
      lbl.textContent = summary ? `scan: ${summary}` : 'scan: no APs seen';
      lbl.style.color = '#22c55e';
      console.log('[wifi-scan]', d);
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
setTimeout(wifiRefresh, 1200);

document.getElementById('toggle_cmd_log').onclick = async ()=>{
  try {
    const r = await fetch('/proxy/logging/commands');
    const s = await r.json();
    await post('/proxy/logging/commands', {enabled: !Boolean(s.enabled)});
    refreshCommandLogStatus();
  } catch {}
};

document.getElementById('download_cmd_log').onclick = ()=>{
  window.open('/proxy/logging/commands/download', '_blank');
};

document.getElementById('clear_cmd_log').onclick = async ()=>{
  try {
    await post('/proxy/logging/commands/clear', {});
  } catch {}
};

const map = new Set(['w','a','s','d','q','e','r','f','t','l','x',' ']);
function _isTyping() {
  const t = document.activeElement?.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || document.activeElement?.isContentEditable;
}
window.addEventListener('keydown', (e)=>{
  if (_isTyping()) return;
  const k = e.key.toLowerCase();
  // ── Global panic-button: '0' lands EVERY drone in the fleet ────────
  // Using the zero key because it's not in the movement keymap (Q is
  // yaw-CCW, W/A/S/D are translation, R/F are altitude, so letters are
  // risky). '0' is far from the flight grid and easy to hit one-handed.
  if (k === '0') {
    e.preventDefault();
    if (window._landAllInFlight) return;    // debounce
    window._landAllInFlight = true;
    console.log('[LAND_ALL] 0 pressed — landing every drone');
    landAllDrones('0 hotkey').finally(() => { window._landAllInFlight = false; });
    return;
  }
  // ── Drone switch hotkey: digits 1-5 select the Nth drone in the bar ──
  // Order follows Object.entries(drones) insertion order, same as the
  // drone-bar buttons top→bottom. '1' = first drone, '2' = second, etc.
  // Accept both the top-row digit key and the numpad digit; e.key is '1'
  // in both cases so simple string comparison works, but also use e.code
  // as a fallback when an exotic layout remaps the character.
  const isDigit15 = /^[1-5]$/.test(k) ||
                    /^(Digit|Numpad)[1-5]$/.test(e.code || '');
  if (isDigit15) {
    e.preventDefault();
    const slot = parseInt(k, 10) || parseInt((e.code || '').slice(-1), 10);
    const idx  = slot - 1;
    const ids  = Object.keys(drones || {});
    console.log('[drone-switch] hotkey', slot, 'fleet=', ids,
                'active=', activeDroneId);
    if (ids.length === 0) {
      console.warn('[drone-switch] drones dict is empty — did loadDrones run?');
      return;
    }
    if (idx < ids.length) {
      const targetId = ids[idx];
      if (targetId !== activeDroneId) {
        switchDrone(targetId);
      } else {
        console.log('[drone-switch] already on', targetId);
      }
    } else {
      console.log('[drone-switch] no drone at slot', slot, '(have', ids.length, ')');
    }
    return;
  }
  if (map.has(k)) {
    e.preventDefault();
    pressKey(k === ' ' ? 'space' : k);
  }
});

// Button wiring for the top-of-page LAND ALL button
(function(){
  const b = document.getElementById('land_all_btn');
  if (!b) return;
  b.addEventListener('click', () => {
    if (window._landAllInFlight) return;
    window._landAllInFlight = true;
    console.log('[LAND_ALL] button clicked');
    landAllDrones('button').finally(() => { window._landAllInFlight = false; });
  });
})();

// Fleet-wide panic land — used by the 'q' hotkey and the big red
// LAND ALL button. Shows a banner with per-drone results.
async function landAllDrones(trigger) {
  // Visible flash so the operator knows the hotkey fired even before the
  // server responds — critical for a panic button.
  showLandAllBanner('⚠ LANDING ALL DRONES (' + trigger + ')…', '#78350f', '#fde68a');
  try {
    const r = await fetch('/proxy/land_all', {method:'POST'});
    const j = await r.json();
    const summary = (j.landed || 0) + '/' + (j.total || 0) +
                    ' acknowledged land' +
                    (j.mission_stopped ? ' (mission stopped)' : '');
    const color = (j.landed === j.total) ? '#064e3b' : '#7f1d1d';
    const txt   = (j.landed === j.total) ? '#a7f3d0' : '#fecaca';
    showLandAllBanner('✓ LAND ALL → ' + summary, color, txt, 6000);
    console.log('[LAND_ALL] result:', j);
  } catch (err) {
    showLandAllBanner('✗ LAND ALL failed: ' + err, '#7f1d1d', '#fecaca', 6000);
    console.error('[LAND_ALL]', err);
  }
}

function showLandAllBanner(msg, bg, fg, autohideMs) {
  let el = document.getElementById('land_all_banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'land_all_banner';
    el.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);' +
                       'z-index:9999;padding:10px 18px;border-radius:6px;' +
                       'font-weight:700;font-size:14px;box-shadow:0 4px 14px rgba(0,0,0,0.5);' +
                       'border:2px solid rgba(255,255,255,0.15);letter-spacing:0.4px;';
    document.body.appendChild(el);
  }
  el.style.background = bg;
  el.style.color = fg;
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(showLandAllBanner._t);
  if (autohideMs) {
    showLandAllBanner._t = setTimeout(() => { el.style.display = 'none'; }, autohideMs);
  }
}
window.addEventListener('keyup', (e)=>{
  if (_isTyping()) { releaseAllKeys(); return; }
  const k = e.key.toLowerCase();
  if (map.has(k)) {
    e.preventDefault();
    releaseKey(k === ' ' ? 'space' : k);
  }
});
window.addEventListener('blur', ()=>releaseAllKeys());
document.addEventListener('visibilitychange', ()=>{
  if (document.hidden) releaseAllKeys();
});

function meterColor(v){
  if (v >= 70) return '#22c55e';
  if (v >= 35) return '#f59e0b';
  return '#ef4444';
}
function setMeter(idBar, idVal, value, suffix=''){
  const bar = document.getElementById(idBar);
  const val = document.getElementById(idVal);
  if (value == null || Number.isNaN(Number(value))) {
    bar.style.width = '0%';
    bar.style.background = '#64748b';
    val.textContent = '-';
    return;
  }
  const v = Math.max(0, Math.min(100, Number(value)));
  bar.style.width = `${v}%`;
  bar.style.background = meterColor(v);
  val.textContent = `${Math.round(v)}${suffix}`;
}

let lastTelemetry = {};

async function refreshTelemetry(){
  const apiEl = document.getElementById('api_status');
  const droneEl = document.getElementById('drone_status');
  try {
    const r = await fetch('/proxy/telemetry', {cache:'no-store'});
    if (!r.ok) throw new Error('api_error');
    const t = await r.json();
    lastTelemetry = t;
    window.lastTelemetry = t;   // expose to standalone scripts (graphs)

    // Update telemetry speed/accel in position tracker section
    const fmt = v => (v != null && !isNaN(v)) ? Number(v).toFixed(1) : '—';
    document.getElementById('tel_vx').textContent = fmt(t.vgx);
    document.getElementById('tel_vy').textContent = fmt(t.vgy);
    document.getElementById('tel_vz').textContent = fmt(t.vgz);
    // Acceleration: only available on Tello; Anafi SDK does not expose it
    const hasAccel = t.agx != null;
    document.getElementById('tel_ax').textContent = hasAccel ? fmt(t.agx) : 'N/A';
    document.getElementById('tel_ay').textContent = hasAccel ? fmt(t.agy) : 'N/A';
    document.getElementById('tel_az').textContent = hasAccel ? fmt(t.agz) : 'N/A';

    apiEl.textContent = 'connected';
    apiEl.style.color = '#22c55e';

    // state_fresh may be absent from older API servers; infer liveness from real data
  const fresh = (t.state_fresh !== undefined) ? Boolean(t.state_fresh) : (t.battery != null || t.yaw != null);
  const live = Boolean(t.connected) && fresh;
    droneEl.textContent = live ? 'live' : 'no live telemetry';
    droneEl.style.color = live ? '#22c55e' : '#f59e0b';

    if (!live) {
      setMeter('battery_bar', 'battery_val', null);
      document.getElementById('telemetry').textContent =
        `no live drone telemetry\n` +
        `api reachable: yes\n` +
        `drone connected: ${t.connected}\n` +
        `state age: ${t.state_age_s ?? '-'} s`;
      return;
    }

    const battery = (typeof t.battery === 'number') ? t.battery : null;
    setMeter('battery_bar', 'battery_val', battery, '%');

    document.getElementById('telemetry').textContent =
      `battery: ${t.battery ?? '-'} %\n` +
      `temperature: ${t.temperature ?? '-'} °C\n` +
      `height: ${t.height_cm ?? '-'} cm\n` +
      `tof: ${t.tof_cm ?? '-'} cm\n` +
      `barometer: ${t.barometer_cm ?? '-'} cm\n` +
      `flight time: ${t.flight_time_s ?? '-'} s\n` +
      `speed: ${t.speed ?? '-'}\n` +
      `wifi snr: ${t.wifi_snr ?? '-'}\n` +
      `attitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\n` +
      `velocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\n` +
      `accel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\n` +
      `sdk version: ${t.sdk_version ?? '-'}\n` +
      `serial number: ${t.serial_number ?? '-'}\n` +
      `mission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}\n` +
      `gps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m\n` +
      `gimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}\n` +
      `state age: ${t.state_age_s ?? '-'} s\n` +
      `flying: ${t.flying}\n` +
      `connected: ${t.connected}`;
  } catch {
    apiEl.textContent = 'disconnected';
    apiEl.style.color = '#ef4444';
    const droneEl = document.getElementById('drone_status');
    droneEl.textContent = 'unknown';
    droneEl.style.color = '#ef4444';
    setMeter('battery_bar', 'battery_val', null);
    document.getElementById('telemetry').textContent = 'telemetry unavailable';
  }
}
async function refreshLogStatus(){
  try {
    const r = await fetch('/proxy/logging/telemetry');
    const s = await r.json();
    document.getElementById('toggle_log').textContent = s.enabled ? 'Disable Telemetry Log' : 'Enable Telemetry Log';
  } catch {}
}
async function refreshSafeTakeoff(){
  try {
    const r = await fetch('/proxy/safety/takeoff');
    const s = await r.json();
    document.getElementById('safe_takeoff').textContent = s.enabled ? 'Safe Takeoff: ON' : 'Safe Takeoff: OFF';
  } catch {}
}
async function refreshCommandLogStatus(){
  try {
    const r = await fetch('/proxy/logging/commands');
    const s = await r.json();
    document.getElementById('toggle_cmd_log').textContent = s.enabled ? 'Command Logging: ON' : 'Command Logging: OFF';
  } catch {}
}
// --- Video stream controls ---
const videoMode = document.getElementById('video_mode');
const videoToggle = document.getElementById('video_toggle');
const videoStatus = document.getElementById('video_status');
const videoUrl = document.getElementById('video_url');
const videoContainer = document.getElementById('video_container');
const videoImg = document.getElementById('video_img');
let videoActive = false;

// Shared video-start logic — used by the manual toggle AND by the
// auto-start path that fires as soon as the page loads. Both paths
// assume Way 1 (MJPEG) unless the user has manually picked forward.
async function startVideoStream(mode) {
  mode = mode || videoMode.value || 'mjpeg';
  if (mode === 'off') return false;
  try {
    const r = await fetch('/proxy/video/start', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode})});
    const d = await r.json();
    if (!d.ok) {
      videoStatus.textContent = 'Error: ' + (d.error || 'unknown');
      return false;
    }
    videoActive = true;
    videoToggle.textContent = 'Stop Video';
    videoStatus.textContent = 'Mode: ' + d.mode;
    videoContainer.style.display = '';
    if (d.mode === 'mjpeg') {
      videoImg.src = '/proxy/video?' + Date.now();
      videoUrl.style.display = '';
      videoUrl.innerHTML = 'Direct: <b>' + (d.stream_url || '') + '</b>';
    } else if (d.mode === 'forward') {
      videoImg.src = '/proxy/video/forward_stream?' + Date.now();
      videoUrl.style.display = '';
      videoUrl.innerHTML = 'UDP → C2 decode → MJPEG';
    }
    videoMode.value = d.mode;
    return true;
  } catch (e) {
    videoStatus.textContent = 'Error: ' + e;
    return false;
  }
}

videoToggle.onclick = async () => {
  if (videoActive) {
    await post('/proxy/video/stop', {});
    videoActive = false;
    videoToggle.textContent = 'Start Video';
    videoContainer.style.display = 'none';
    videoUrl.style.display = 'none';
    videoImg.src = '';
    videoStatus.textContent = 'Mode: off';
    return;
  }
  await startVideoStream(videoMode.value);
};

// Auto-start Way 1 (MJPEG) on page load. refreshVideoStatus() below will
// detect if the server reports an already-running stream and skip, so
// reloading the page won't restart the stream unnecessarily.
async function autoStartVideo() {
  try {
    const r = await fetch('/proxy/video/status', {cache:'no-store'});
    const d = await r.json();
    if (d && d.mode && d.mode !== 'off') {
      // Already running — adopt existing state without restarting
      videoActive = true;
      videoMode.value = d.mode;
      videoToggle.textContent = 'Stop Video';
      videoContainer.style.display = '';
      videoStatus.textContent = 'Mode: ' + d.mode + ' (existing)';
      if (d.mode === 'mjpeg') {
        videoImg.src = '/proxy/video?' + Date.now();
      } else if (d.mode === 'forward') {
        videoImg.src = '/proxy/video/forward_stream?' + Date.now();
      }
      console.log('[video] adopted existing stream:', d.mode);
      return;
    }
  } catch {}
  // Nothing running — start Way 1 (MJPEG, decoded on the flight controller).
  videoMode.value = 'mjpeg';
  console.log('[video] auto-starting MJPEG (Way 1)');
  const ok = await startVideoStream('mjpeg');
  if (!ok) console.warn('[video] auto-start failed, click Start Video manually');
}
// Fire a bit after page load so /proxy/heartbeat has time to succeed first
setTimeout(autoStartVideo, 300);

// Poll video status
async function refreshVideoStatus() {
  try {
    const r = await fetch('/proxy/video/status', {cache:'no-store'});
    const d = await r.json();
    if (!videoActive && d.mode !== 'off') {
      videoActive = true;
      videoToggle.textContent = 'Stop Video';
      videoMode.value = d.mode;
    }
    if (videoActive) {
      videoStatus.textContent = 'Mode: ' + d.mode + ' | has_frame: ' + (d.has_frame||false);
    }
  } catch {}
}
setInterval(refreshVideoStatus, 5000);

// Heartbeat — keeps the drone watchdog alive so it doesn't auto-land
async function sendHeartbeat(){
  try { await fetch('/proxy/heartbeat', {cache:'no-store'}); } catch {}
}
setInterval(sendHeartbeat, 500);
sendHeartbeat();

// ── Telemetry SSE (replaces polling for near-real-time updates) ──
let telEvtSource = null;
function startTelemetrySSE() {
  if (telEvtSource) { telEvtSource.close(); telEvtSource = null; }
  telEvtSource = new EventSource('/proxy/telemetry/stream');
  telEvtSource.onmessage = (e) => {
    try {
      const t = JSON.parse(e.data);
      _handleTelemetryData(t);
    } catch {}
  };
  telEvtSource.onerror = () => {
    telEvtSource.close(); telEvtSource = null;
    // Fast reconnect — 500ms instead of 3s
    setTimeout(startTelemetrySSE, 500);
  };
}
// Extract telemetry UI update into reusable function
function _handleTelemetryData(t) {
  lastTelemetry = t;
  window.lastTelemetry = t;   // expose to standalone scripts (graphs)
  const apiEl = document.getElementById('api_status');
  const droneEl = document.getElementById('drone_status');
  const fmt = v => (v != null && !isNaN(v)) ? Number(v).toFixed(1) : '\\u2014';
  document.getElementById('tel_vx').textContent = fmt(t.vgx);
  document.getElementById('tel_vy').textContent = fmt(t.vgy);
  document.getElementById('tel_vz').textContent = fmt(t.vgz);
  const hasAccel = t.agx != null;
  document.getElementById('tel_ax').textContent = hasAccel ? fmt(t.agx) : 'N/A';
  document.getElementById('tel_ay').textContent = hasAccel ? fmt(t.agy) : 'N/A';
  document.getElementById('tel_az').textContent = hasAccel ? fmt(t.agz) : 'N/A';
  apiEl.textContent = 'connected'; apiEl.style.color = '#22c55e';
  // state_fresh may be absent from older API servers; infer liveness from real data
  const fresh = (t.state_fresh !== undefined) ? Boolean(t.state_fresh) : (t.battery != null || t.yaw != null);
  const live = Boolean(t.connected) && fresh;
  droneEl.textContent = live ? 'live' : 'no live telemetry';
  droneEl.style.color = live ? '#22c55e' : '#f59e0b';
  if (!live) {
    setMeter('battery_bar', 'battery_val', null);
    document.getElementById('telemetry').textContent =
      `no live drone telemetry\\napi reachable: yes\\ndrone connected: ${t.connected}\\nstate age: ${t.state_age_s ?? '-'} s`;
    return;
  }
  const battery = (typeof t.battery === 'number') ? t.battery : null;
  setMeter('battery_bar', 'battery_val', battery, '%');
  document.getElementById('telemetry').textContent =
    `battery: ${t.battery ?? '-'} %\\ntemperature: ${t.temperature ?? '-'} °C\\nheight: ${t.height_cm ?? '-'} cm\\ntof: ${t.tof_cm ?? '-'} cm\\nbarometer: ${t.barometer_cm ?? '-'} cm\\nflight time: ${t.flight_time_s ?? '-'} s\\nspeed: ${t.speed ?? '-'}\\nwifi snr: ${t.wifi_snr ?? '-'}\\nattitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\\nvelocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\\naccel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\\nsdk version: ${t.sdk_version ?? '-'}\\nserial number: ${t.serial_number ?? '-'}\\nmission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}\\ngps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m\\ngimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}\\nstate age: ${t.state_age_s ?? '-'} s\\nflying: ${t.flying}\\nconnected: ${t.connected}`;

  // ── Compass + takeoff-heading tracking ──
  updateCompass(t);
}

// ── Compass widget: tracks magnetic yaw + takeoff-reference heading ──
// Anafi AttitudeChanged.yaw is NED yaw in degrees, range -180..+180
// (positive = nose turned clockwise when viewed from above). Magnetic-north
// referenced, NOT true north (no declination correction without GPS).
// Takeoff heading is captured the first time `flying` transitions false → true
// and on the Reset button. Stored per active drone id.
let _takeoffHeadingByDrone = {};
let _wasFlying = false;
let _lastActiveDroneId = null;

function normDeg(d) {
  // Normalize any degree to -180..+180
  while (d >  180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

function updateCompass(t) {
  const cv = document.getElementById('compass_canvas');
  if (!cv) return;

  // Detect active-drone change — clear stale state so ref doesn't bleed across drones
  const activeId = (window.currentDroneId || t.drone_id || 'default');
  if (activeId !== _lastActiveDroneId) {
    _lastActiveDroneId = activeId;
    _wasFlying = Boolean(t.flying);  // don't auto-capture just from switching
  }

  const yawDeg = (typeof t.yaw === 'number') ? t.yaw : null;
  const flying = Boolean(t.flying);

  // Capture takeoff heading on false→true transition of `flying`
  if (flying && !_wasFlying && yawDeg != null) {
    _takeoffHeadingByDrone[activeId] = yawDeg;
  }
  _wasFlying = flying;

  const takeoffRef = _takeoffHeadingByDrone[activeId];
  const relative = (yawDeg != null && takeoffRef != null)
    ? normDeg(yawDeg - takeoffRef) : null;

  // Numeric labels
  document.getElementById('compass_abs').textContent =
    yawDeg != null ? yawDeg.toFixed(0) : '--';
  document.getElementById('compass_ref').textContent =
    takeoffRef != null ? takeoffRef.toFixed(0) + '°' : 'not captured';
  document.getElementById('compass_rel').textContent =
    relative != null ? (relative >= 0 ? '+' : '') + relative.toFixed(0) : '--';

  // Draw the compass rose
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const cx = W / 2, cy = H / 2;
  const r = Math.min(W, H) / 2 - 6;

  ctx.clearRect(0, 0, W, H);

  // Outer ring
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Tick marks every 30°
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  for (let a = 0; a < 360; a += 30) {
    const rad = (a - 90) * Math.PI / 180;  // 0° = up = North
    const isCardinal = (a % 90 === 0);
    const inner = r - (isCardinal ? 8 : 4);
    ctx.beginPath();
    ctx.moveTo(cx + inner * Math.cos(rad), cy + inner * Math.sin(rad));
    ctx.lineTo(cx + (r - 1) * Math.cos(rad), cy + (r - 1) * Math.sin(rad));
    ctx.stroke();
  }

  // Cardinal labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('N', cx, cy - r + 7);
  ctx.fillText('S', cx, cy + r - 7);
  ctx.fillText('E', cx + r - 7, cy);
  ctx.fillText('W', cx - r + 7, cy);

  // Takeoff-heading marker on the rim (small gray triangle)
  if (takeoffRef != null) {
    const radRef = (takeoffRef - 90) * Math.PI / 180;
    const mx = cx + (r - 2) * Math.cos(radRef);
    const my = cy + (r - 2) * Math.sin(radRef);
    const backLen = 6;
    ctx.fillStyle = '#64748b';
    ctx.beginPath();
    ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
    ctx.fill();
    // Small "T" label next to it
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px sans-serif';
    const tx = cx + (r + 6) * Math.cos(radRef);
    const ty = cy + (r + 6) * Math.sin(radRef);
    ctx.fillText('T', tx, ty);
  }

  // Current heading needle — red/blue dual (red=front, blue=tail)
  if (yawDeg != null) {
    const rad = (yawDeg - 90) * Math.PI / 180;
    const nx = cx + (r - 10) * Math.cos(rad);
    const ny = cy + (r - 10) * Math.sin(rad);
    const tx = cx - (r - 18) * Math.cos(rad);
    const ty = cy - (r - 18) * Math.sin(rad);
    // Tail (blue)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    // Front (red arrow)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    // Arrowhead
    const ahLen = 7, ahAngle = 0.35;
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(nx, ny);
    ctx.lineTo(
      nx - ahLen * Math.cos(rad - ahAngle),
      ny - ahLen * Math.sin(rad - ahAngle));
    ctx.lineTo(
      nx - ahLen * Math.cos(rad + ahAngle),
      ny - ahLen * Math.sin(rad + ahAngle));
    ctx.closePath();
    ctx.fill();
  }

  // Center pivot
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
  ctx.fill();
}

// Manual Reset-ref button — re-capture takeoff heading from current yaw
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('compass_reset');
  if (btn) btn.addEventListener('click', () => {
    const t = lastTelemetry || {};
    const activeId = (activeDroneId || t.drone_id || 'default');
    if (typeof t.yaw === 'number') {
      _takeoffHeadingByDrone[activeId] = t.yaw;
      updateCompass(t);
    } else {
      alert('No yaw data available — drone not connected?');
    }
  });
});

startTelemetrySSE();
// Fallback poll — only triggers if SSE is down (reduced from 700ms to 2000ms since SSE handles real-time)
setInterval(()=>{ if (!telEvtSource) refreshTelemetry(); }, 2000);

setInterval(refreshLogStatus, 5000);
setInterval(refreshSafeTakeoff, 5000);
setInterval(refreshCommandLogStatus, 5000);
loadDrones();
refreshTelemetry();
refreshLogStatus();
refreshSafeTakeoff();
refreshCommandLogStatus();

// --- Magnetometer recalibration wizard ---
// Drives /proxy/magneto/recalibrate/stream (SSE) and updates the step list,
// per-axis panels, and result banner live as the server emits events.
const magModal = document.getElementById('mag_modal');
let magEventSource = null;

function magLog(line){
  const el = document.getElementById('mag_log');
  const t = new Date().toLocaleTimeString();
  el.textContent += `[${t}] ${line}\\n`;
  el.scrollTop = el.scrollHeight;
}

function magResetUI(){
  document.querySelectorAll('#mag_steps .mag-step').forEach(s=>{
    s.classList.remove('active','ok','fail');
    s.querySelector('[data-role="info"]').textContent = '';
  });
  document.querySelectorAll('#mag_axes .mag-axis').forEach(a=>{
    a.classList.remove('active','ok');
    a.querySelector('[data-role="state"]').innerHTML = '&#9675;';
  });
  const res = document.getElementById('mag_result');
  res.className = '';
  res.textContent = '';
  res.style.display = 'none';
  document.getElementById('mag_log').textContent = '';
  document.getElementById('mag_retry_btn').style.display = 'none';
}

function magSetStep(name, state, info){
  const el = document.querySelector(`#mag_steps .mag-step[data-step="${name}"]`);
  if (!el) return;
  document.querySelectorAll('#mag_steps .mag-step').forEach(s=>{
    if (s !== el && s.classList.contains('active')) s.classList.remove('active');
  });
  el.classList.remove('active','ok','fail');
  if (state) el.classList.add(state);
  if (info !== undefined) el.querySelector('[data-role="info"]').textContent = info;
}

function magSetAxes(axes){
  if (!axes) return;
  ['x','y','z'].forEach(ax=>{
    const el = document.querySelector(`#mag_axes .mag-axis[data-axis="${ax}"]`);
    if (!el) return;
    const stateEl = el.querySelector('[data-role="state"]');
    el.classList.remove('active','ok');
    if (axes[ax] === 1) {
      el.classList.add('ok');
      stateEl.innerHTML = '&#10004;';
    } else {
      if (axes[ax] === 0 || axes[ax] === null) {
        if (!axes.done && !axes.failed) el.classList.add('active');
      }
      stateEl.innerHTML = '&#9675;';
    }
  });
}

function magShowResult(ok, msg){
  const res = document.getElementById('mag_result');
  res.className = ok ? 'ok' : 'fail';
  res.textContent = msg || (ok ? 'Calibration complete.' : 'Calibration failed.');
  res.style.display = 'block';
  document.getElementById('mag_start_btn').disabled = false;
  document.getElementById('mag_retry_btn').style.display = ok ? 'none' : '';
}

function openMagnetoWizard(){
  magResetUI();
  magModal.classList.add('show');
  refreshMagStatus();
}

function closeMagnetoWizard(){
  magModal.classList.remove('show');
  if (magEventSource) { magEventSource.close(); magEventSource = null; }
}

async function refreshMagStatus(){
  try {
    const r = await fetch('/proxy/magneto');
    const j = await r.json();
    const txt = j.status || '(no report)';
    document.getElementById('mag_status').textContent = txt;
    document.getElementById('mag_status').style.color =
      j.required ? '#f87171' : (j.status ? '#86efac' : '#94a3b8');
    return j;
  } catch (e) {
    document.getElementById('mag_status').textContent = 'unreachable';
    document.getElementById('mag_status').style.color = '#f87171';
    return null;
  }
}

function runMagnetoWizard(){
  if (magEventSource) { magEventSource.close(); magEventSource = null; }
  magResetUI();
  document.getElementById('mag_start_btn').disabled = true;
  magLog('connecting to recalibration stream…');
  // Use SSE so every step flip shows up the instant the server observes it.
  // timeout_s=60 covers a patient operator; poll_s=1.0 is plenty.
  magEventSource = new EventSource('/proxy/magneto/recalibrate/stream?timeout_s=60&poll_s=1.0');

  magEventSource.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.kind === 'step') {
      const state = msg.ok ? 'ok' : (msg.step === 'poll' ? 'fail' : 'active');
      let info = '';
      if (msg.step === 'heartbeat')
        info = `connected=${msg.connected} flying=${msg.flying}`;
      else if (msg.step === 'pre_status')
        info = msg.status || '(not reported)';
      else if (msg.step === 'start')
        info = msg.message || msg.error || '';
      else if (msg.step === 'poll')
        info = msg.final_status || (msg.timed_out ? 'timed out' : '');
      magSetStep(msg.step, msg.ok ? 'ok' : 'fail', info);
      magLog(`step ${msg.step}: ${msg.ok ? 'OK' : 'FAIL'} ${info}`);
      if (!msg.ok && msg.step !== 'poll') {
        // Hard failure before the dance even started — stop the wizard.
        magShowResult(false,
          (msg.error ? 'Error: ' + msg.error : 'Step failed: ' + msg.step));
        if (magEventSource) { magEventSource.close(); magEventSource = null; }
      }
      if (msg.step === 'start' && msg.ok) magSetStep('poll', 'active');
    } else if (msg.kind === 'status') {
      magSetAxes(msg.axes);
      magLog(`status: ${msg.status || '(no report)'}`);
    } else if (msg.kind === 'final') {
      if (msg.post && msg.post.status) magSetAxes(
        (function(){ const s = (msg.post.status||'').toLowerCase().replace(/\\s+/g,'');
          const m = s.match(/axes=x(\\d)y(\\d)z(\\d)/);
          return { x: m?+m[1]:null, y: m?+m[2]:null, z: m?+m[3]:null,
                   done: s.indexOf('all-axes-ok')>=0,
                   failed: s.indexOf('failed')>=0 }; })()
      );
      magShowResult(msg.ok, msg.message || msg.error || '');
      magLog(msg.ok ? 'DONE.' : 'FINISHED (with errors).');
      refreshMagStatus();
      if (magEventSource) { magEventSource.close(); magEventSource = null; }
    }
  };

  magEventSource.onerror = () => {
    magLog('stream closed.');
    document.getElementById('mag_start_btn').disabled = false;
    if (magEventSource) { magEventSource.close(); magEventSource = null; }
  };
}

document.getElementById('mag_open').onclick = openMagnetoWizard;
document.getElementById('mag_close_btn').onclick = closeMagnetoWizard;
document.getElementById('mag_start_btn').onclick = runMagnetoWizard;
document.getElementById('mag_retry_btn').onclick = runMagnetoWizard;
magModal.addEventListener('click', (e)=>{ if (e.target === magModal) closeMagnetoWizard(); });

// Refresh magneto status in the Anafi panel every 10s (lightweight).
setInterval(refreshMagStatus, 10000);
refreshMagStatus();

// --- Mission Planner ---
let missionRunning = false;
let missionAbort = false;

function missionLog(msg) {
  const el = document.getElementById('mission_log');
  el.textContent += msg + '\\n';
  el.scrollTop = el.scrollHeight;
}

async function runMission() {
  if (missionRunning) return;
  const textarea = document.getElementById('mission_cmds');
  const lines = textarea.value.split('\\n').map(l=>l.trim()).filter(l=>l && !l.startsWith('#'));
  if (!lines.length) { missionLog('No commands to run.'); return; }

  missionRunning = true;
  missionAbort = false;
  document.getElementById('mission_run').style.display = 'none';
  document.getElementById('mission_stop').style.display = '';
  document.getElementById('mission_status').textContent = 'running...';
  document.getElementById('mission_status').style.color = '#22c55e';
  document.getElementById('mission_log').textContent = '';

  for (let i = 0; i < lines.length; i++) {
    if (missionAbort) { missionLog('ABORTED'); break; }
    const line = lines[i];
    document.getElementById('mission_status').textContent = `step ${i+1}/${lines.length}: ${line}`;
    missionLog(`> ${line}`);

    const parts = line.toLowerCase().split(/\\s+/);
    let ok = false;
    let result = '';

    try {
      if (parts[0] === 'takeoff') {
        const r = await post('/proxy/takeoff', {});
        result = 'takeoff sent';
        ok = true;
        await sleep(3000);
      } else if (parts[0] === 'land') {
        const r = await post('/proxy/land', {});
        result = 'land sent';
        ok = true;
        await sleep(3000);
      } else if (parts[0] === 'wait' || parts[0] === 'hover' || parts[0] === 'sleep') {
        const secs = parseFloat(parts[1]) || 2;
        result = `waiting ${secs}s`;
        ok = true;
        await sleep(secs * 1000);
      } else if (parts[0] === 'emergency') {
        await post('/proxy/emergency', {});
        result = 'emergency sent';
        ok = true;
        missionAbort = true;
      } else {
        // Parse as: <value> <direction> or <direction> <value>
        let val = parseFloat(parts[0]);
        let dir = parts[1];
        if (isNaN(val)) {
          dir = parts[0];
          val = parseFloat(parts[1]) || 30;
        }
        // Map directions
        const moveMap = {forward:1, fwd:1, back:1, backward:1, left:1, right:1, up:1, down:1};
        const rotateMap = {cw:1, ccw:1, clockwise:'cw', counterclockwise:'ccw', turn:'cw'};
        if (dir === 'backward') dir = 'back';
        if (dir === 'fwd') dir = 'forward';

        if (moveMap[dir]) {
          const r = await fetch('/proxy/move', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dir:dir, cm:Math.round(val)})});
          const d = await r.json();
          ok = d.ok;
          result = ok ? `moved ${dir} ${Math.round(val)}cm` : (d.error || 'failed');
          await sleep(1500);
        } else if (rotateMap[dir]) {
          const realDir = typeof rotateMap[dir] === 'string' ? rotateMap[dir] : dir;
          const r = await fetch('/proxy/rotate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dir:realDir, deg:Math.round(val)})});
          const d = await r.json();
          ok = d.ok;
          result = ok ? `rotated ${realDir} ${Math.round(val)}°` : (d.error || 'failed');
          await sleep(1500);
        } else {
          result = `unknown command: ${line}`;
        }
      }
    } catch(e) {
      result = `error: ${e.message}`;
    }
    missionLog(ok ? `  OK: ${result}` : `  FAIL: ${result}`);
  }

  missionRunning = false;
  missionAbort = false;
  document.getElementById('mission_run').style.display = '';
  document.getElementById('mission_stop').style.display = 'none';
  document.getElementById('mission_status').textContent = 'done';
  document.getElementById('mission_status').style.color = '#94a3b8';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

document.getElementById('mission_run').onclick = runMission;
document.getElementById('mission_stop').onclick = () => {
  missionAbort = true;
  missionLog('Abort requested...');
};

// --- Drone Config Editor ---
const modal = document.getElementById('drone_config_modal');
let editDrones = {};

function openDroneConfig() {
  editDrones = JSON.parse(JSON.stringify(drones));
  renderConfigFields();
  modal.style.display = 'flex';
}

function renderConfigFields() {
  const container = document.getElementById('drone_config_fields');
  container.innerHTML = '';
  const ids = Object.keys(editDrones).sort();
  ids.forEach(id => {
    const d = editDrones[id];
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;margin-bottom:8px;align-items:center;';
    row.innerHTML = `
      <span class="small" style="min-width:30px;color:#94a3b8;">#${id}</span>
      <input type="text" value="${d.name}" placeholder="Name" data-id="${id}" data-field="name"
        style="width:120px;padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;" />
      <select data-id="${id}" data-field="type"
        style="padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;">
        <option value="tello" ${d.type==='tello'?'selected':''}>Tello</option>
        <option value="anafi" ${d.type==='anafi'?'selected':''}>Anafi</option>
      </select>
      <input type="text" value="${d.base}" placeholder="http://IP:port" data-id="${id}" data-field="base"
        style="flex:1;min-width:200px;padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;font-family:monospace;" />
      <button onclick="deleteConfigDrone('${id}')" style="padding:2px 8px;background:#7f1d1d;border-color:#dc2626;font-size:11px;">X</button>
    `;
    container.appendChild(row);
  });
  // Bind change handlers
  container.querySelectorAll('input,select').forEach(el => {
    el.addEventListener('change', () => {
      const id = el.dataset.id, field = el.dataset.field;
      if (id && field && editDrones[id]) editDrones[id][field] = el.value;
    });
    el.addEventListener('input', () => {
      const id = el.dataset.id, field = el.dataset.field;
      if (id && field && editDrones[id]) editDrones[id][field] = el.value;
    });
  });
}

function deleteConfigDrone(id) {
  delete editDrones[id];
  renderConfigFields();
}

document.getElementById('drone_config_add').onclick = () => {
  const ids = Object.keys(editDrones).map(Number).filter(n=>!isNaN(n));
  const newId = String((ids.length ? Math.max(...ids) : 0) + 1);
  editDrones[newId] = {name: `Drone ${newId}`, type: 'tello', base: 'http://192.168.1.100:8080'};
  renderConfigFields();
};

document.getElementById('drone_config_save').onclick = async () => {
  // Sync fields from DOM
  document.querySelectorAll('#drone_config_fields input, #drone_config_fields select').forEach(el => {
    const id = el.dataset.id, field = el.dataset.field;
    if (id && field && editDrones[id]) editDrones[id][field] = el.value;
  });
  const st = document.getElementById('drone_config_status');
  try {
    const r = await fetch('/proxy/drones/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({drones: editDrones})
    });
    const d = await r.json();
    if (d.ok) {
      drones = d.drones;
      st.textContent = 'Saved!';
      st.style.color = '#22c55e';
      renderDroneBar();
      updatePiLabel();
      setTimeout(()=>{ modal.style.display='none'; st.textContent=''; }, 800);
    } else {
      st.textContent = d.error || 'Save failed';
      st.style.color = '#ef4444';
    }
  } catch(e) {
    st.textContent = 'Error: ' + e.message;
    st.style.color = '#ef4444';
  }
};

document.getElementById('drone_config_cancel').onclick = () => { modal.style.display = 'none'; };
document.getElementById('edit_drones_btn').onclick = openDroneConfig;
modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };

// ── Position Tracker ─────────────────────────────────────────────────────────
let posEvtSource = null;
let posVideoOn = false;

const arenaCanvas = document.getElementById('arena_canvas');
const arenaCtx = arenaCanvas.getContext('2d');

// World dimensions — updated when arena config loads
let arenaW = 20, arenaD = 10, arenaOX = -10, arenaOY = 0;
// View adds a margin around arena so out-of-bounds positions are still visible
const VIEW_MARGIN = 5;  // metres
let viewOX = -15, viewOY = -5, viewW = 30, viewD = 20;

function _updateView() {
  viewOX = arenaOX - VIEW_MARGIN;
  viewOY = arenaOY - VIEW_MARGIN;
  viewW  = arenaW + 2 * VIEW_MARGIN;
  viewD  = arenaD + 2 * VIEW_MARGIN;
}

const ARENA_PAD = 30;  // pixel padding for axis labels

function arenaToCanvas(ax, ay) {
  const W = arenaCanvas.width, H = arenaCanvas.height;
  const iw = W - 2 * ARENA_PAD, ih = H - 2 * ARENA_PAD;
  return [
    ARENA_PAD + (ax - viewOX) / viewW * iw,
    H - ARENA_PAD - (ay - viewOY) / viewD * ih,
  ];
}

const WALL_COLOR = { front:'#6366f1', back:'#a855f7', left:'#06b6d4', right:'#10b981' };
let _seenMarkers = new Set();   // IDs currently visible in the drone camera (as strings)
let _refMarkers  = new Set();   // IDs actually used as world-frame refs this frame

// Track last drawn position so arena config reload doesn't erase the dot
let _lastPos = null, _lastCompPos = null, _lastDir = null, _lastFrameRes = null;

function drawArena(pos, compPos, dir) {
  // Persist last known position so config reloads don't blank it
  if (pos !== undefined) { _lastPos = pos; _lastCompPos = compPos; _lastDir = dir; window._lastPos = pos; }
  const _pos = _lastPos, _compPos = _lastCompPos, _dir = _lastDir;

  const ctx = arenaCtx;
  const W = arenaCanvas.width, H = arenaCanvas.height;
  const PAD = ARENA_PAD;

  // Background
  ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, W, H);

  // Compute arena sub-rect in canvas pixels
  const [ax0, ay0] = arenaToCanvas(arenaOX, arenaOY + arenaD);
  const [ax1, ay1] = arenaToCanvas(arenaOX + arenaW, arenaOY);

  // Dim margin zone outside arena, brighter inside
  ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.fillRect(PAD, PAD, W - 2*PAD, H - 2*PAD);
  ctx.fillStyle = '#0f172a'; ctx.fillRect(ax0, ay0, ax1 - ax0, ay1 - ay0);

  // Grid lines — fine 1 m grid (subtle) + major 5 m grid (brighter)
  // so the user can read position to the metre at a glance.
  const minorStroke = '#17243b';
  const majorStroke = '#1e3a5f';
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(viewOX); gx <= viewOX + viewW + 0.01; gx += 1) {
    const [cx] = arenaToCanvas(gx, viewOY);
    ctx.strokeStyle = (gx % 5 === 0) ? majorStroke : minorStroke;
    ctx.beginPath(); ctx.moveTo(cx, PAD); ctx.lineTo(cx, H - PAD); ctx.stroke();
  }
  for (let gy = Math.ceil(viewOY); gy <= viewOY + viewD + 0.01; gy += 1) {
    const [, cy] = arenaToCanvas(viewOX, gy);
    ctx.strokeStyle = (gy % 5 === 0) ? majorStroke : minorStroke;
    ctx.beginPath(); ctx.moveTo(PAD, cy); ctx.lineTo(W - PAD, cy); ctx.stroke();
  }

  // Arena border (highlighted)
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 2;
  ctx.strokeRect(ax0, ay0, ax1 - ax0, ay1 - ay0);

  // Outer view border
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
  ctx.strokeRect(PAD, PAD, W - 2*PAD, H - 2*PAD);

  // Axis labels every 5 m (brighter inside arena range)
  ctx.font = '9px monospace'; ctx.textAlign = 'center';
  for (let gx = Math.ceil(viewOX / 5) * 5; gx <= viewOX + viewW + 0.01; gx += 5) {
    const [cx] = arenaToCanvas(gx, viewOY);
    ctx.fillStyle = (gx >= arenaOX && gx <= arenaOX + arenaW) ? '#64748b' : '#334155';
    ctx.fillText(gx, cx, H - 2);
  }
  for (let gy = Math.ceil(viewOY / 5) * 5; gy <= viewOY + viewD + 0.01; gy += 5) {
    const [, cy] = arenaToCanvas(viewOX, gy);
    ctx.textAlign = 'right';
    ctx.fillStyle = (gy >= arenaOY && gy <= arenaOY + arenaD) ? '#64748b' : '#334155';
    ctx.fillText(gy, PAD - 3, cy + 3);
  }

  // Wall labels inside arena
  const midAx = (ax0 + ax1) / 2, midAy = (ay0 + ay1) / 2;
  ctx.fillStyle = '#64748b'; ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
  ctx.fillText('BACK',  midAx, ay0 + 11);
  ctx.fillText('FRONT', midAx, ay1 - 4);
  ctx.save(); ctx.translate(ax0 + 10, midAy); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('LEFT', 0, 0); ctx.restore();
  ctx.save(); ctx.translate(ax1 - 10, midAy); ctx.rotate(Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('RIGHT', 0, 0); ctx.restore();

  // Arena markers — colored square + bold ID pill placed away from the wall
  // Currently-visible markers get a bright halo; markers used as refs get a
  // brighter border; unseen markers are dimmed to ~40% for visual separation.
  ctx.font = 'bold 11px monospace';
  ctx.textBaseline = 'middle';
  for (const [id, m] of Object.entries(arenaMarkers)) {
    if (!m.pos) continue;
    const [mx, my] = arenaToCanvas(m.pos[0], m.pos[1]);
    if (mx < PAD - 4 || mx > W - PAD + 4 || my < PAD - 4 || my > H - PAD + 4) continue;

    const isSeen = _seenMarkers.has(String(id));
    const isRef  = _refMarkers.has(String(id));
    const baseColor = WALL_COLOR[m.wall] || '#94a3b8';

    // Halo behind seen markers so they visibly pulse against the arena
    if (isSeen) {
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.fillStyle = (isRef ? 'rgba(34,197,94,0.35)' : 'rgba(251,191,36,0.35)');
      ctx.fill();
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.strokeStyle = (isRef ? '#22c55e' : '#fbbf24');
      ctx.lineWidth = 1.5; ctx.stroke();
    }

    // Square marker — always full-opacity so markers never disappear.
    // Seen markers get a brighter white border to make them pop.
    ctx.fillStyle = baseColor;
    ctx.fillRect(mx - 5, my - 5, 10, 10);
    ctx.strokeStyle = isSeen ? '#ffffff' : 'rgba(15,23,42,0.9)';
    ctx.lineWidth = isSeen ? 1.5 : 1;
    ctx.strokeRect(mx - 5.5, my - 5.5, 11, 11);

    // Position label pill on the side away from the wall (so ID doesn't collide with BACK/FRONT/LEFT/RIGHT)
    // front=y≈0 → label up, back=y≈max → down, left=x≈0 → right, right=x≈max → left
    const idText = String(id);
    const tw = ctx.measureText(idText).width;
    const pillW = tw + 8, pillH = 15;
    let labelX, labelY, anchor = 'center';
    const wall = (m.wall || '').toLowerCase();
    if (wall === 'front')      { labelX = mx; labelY = my - 13; }
    else if (wall === 'back')  { labelX = mx; labelY = my + 13; }
    else if (wall === 'left')  { labelX = mx + 9 + pillW / 2; labelY = my; }
    else if (wall === 'right') { labelX = mx - 9 - pillW / 2; labelY = my; }
    else                       { labelX = mx; labelY = my - 13; }

    // Dark background pill for legibility — always visible
    ctx.fillStyle = 'rgba(15,23,42,0.9)';
    ctx.fillRect(labelX - pillW / 2, labelY - pillH / 2, pillW, pillH);
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = isSeen ? 1.5 : 1;
    ctx.strokeRect(labelX - pillW / 2 + 0.5, labelY - pillH / 2 + 0.5, pillW - 1, pillH - 1);

    // ID text — white for seen markers, wall color for unseen
    ctx.fillStyle = isSeen ? '#ffffff' : baseColor;
    ctx.textAlign = 'center';
    ctx.fillText(idText, labelX, labelY + 0.5);
  }
  ctx.textBaseline = 'alphabetic';  // restore default so downstream drawing is unaffected

  // Debug overlay — large text, dark background box, drawn over everything
  const dbgLines = [
    `view: [${viewOX},${viewOX+viewW}] x [${viewOY},${viewOY+viewD}]`,
    `pos: ${_pos ? `${_pos[0].toFixed(2)}, ${_pos[1].toFixed(2)}, ${_pos[2].toFixed(2)}` : 'null'}`,
    `frame: ${_lastFrameRes || 'unknown'}`,
  ];
  ctx.font = 'bold 12px monospace';
  const dbgW = Math.max(...dbgLines.map(l => ctx.measureText(l).width)) + 10;
  ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(PAD, PAD, dbgW, dbgLines.length * 16 + 6);
  ctx.fillStyle = '#22d3ee'; ctx.textAlign = 'left';
  dbgLines.forEach((l, i) => ctx.fillText(l, PAD + 5, PAD + 15 + i * 16));

  // Drone positions
  if (!_pos) return;

  // Clamp canvas coords to inner area; returns [cx, cy, wasOutOfBounds, rawPx, rawPy]
  const M = PAD + 8;
  const cp = _compPos || _pos;
  const [rpx, rpy] = arenaToCanvas(cp[0], cp[1]);
  const cx2 = Math.max(M, Math.min(W - M, rpx));
  const cy2 = Math.max(M, Math.min(H - M, rpy));
  const outOfBounds = rpx !== cx2 || rpy !== cy2;

  const pxDbg = `px: raw(${rpx.toFixed(0)},${rpy.toFixed(0)}) clamped(${cx2.toFixed(0)},${cy2.toFixed(0)}) oob:${outOfBounds}`;
  console.log('[POS]', pxDbg);
  ctx.font = 'bold 12px monospace'; ctx.textAlign = 'left';
  const pdW = ctx.measureText(pxDbg).width + 10;
  ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(PAD, PAD + 56, pdW, 20);
  ctx.fillStyle = '#22d3ee'; ctx.fillText(pxDbg, PAD + 5, PAD + 70);

  const dotColor = outOfBounds ? '#ef4444' : '#f97316';

  // Outer glow ring (always drawn, big and visible)
  ctx.beginPath(); ctx.arc(cx2, cy2, 18, 0, Math.PI * 2);
  ctx.strokeStyle = outOfBounds ? 'rgba(239,68,68,0.5)' : 'rgba(249,115,22,0.4)';
  ctx.lineWidth = 4; ctx.stroke();

  // Solid filled dot
  ctx.beginPath(); ctx.arc(cx2, cy2, 10, 0, Math.PI * 2);
  ctx.fillStyle = dotColor; ctx.fill();
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();

  // Crosshair
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx2 - 20, cy2); ctx.lineTo(cx2 + 20, cy2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx2, cy2 - 20); ctx.lineTo(cx2, cy2 + 20); ctx.stroke();

  // Arrow toward true out-of-bounds position
  if (outOfBounds) {
    const ang = Math.atan2(rpy - cy2, rpx - cx2);
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 3; ctx.lineCap = 'round';
    const ax = cx2 + Math.cos(ang) * 24, ay = cy2 + Math.sin(ang) * 24;
    ctx.beginPath(); ctx.moveTo(cx2 + Math.cos(ang) * 12, cy2 + Math.sin(ang) * 12);
    ctx.lineTo(ax, ay); ctx.stroke();
    // arrowhead
    const perp = ang + Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - Math.cos(ang)*6 + Math.cos(perp)*4, ay - Math.sin(ang)*6 + Math.sin(perp)*4);
    ctx.lineTo(ax - Math.cos(ang)*6 - Math.cos(perp)*4, ay - Math.sin(ang)*6 - Math.sin(perp)*4);
    ctx.closePath(); ctx.fillStyle = '#ef4444'; ctx.fill();
    ctx.lineCap = 'butt';
  }

  // Heading arrow
  if (_dir) {
    const ang = Math.atan2(_dir[0], _dir[1]);
    ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx2, cy2);
    ctx.lineTo(cx2 + Math.sin(ang) * 28, cy2 - Math.cos(ang) * 28); ctx.stroke();
    ctx.lineCap = 'butt';
  }

  // Coordinate label — "OUT" prefix + background box for readability
  const label = (outOfBounds ? 'OUT ' : '') + `(${_pos[0].toFixed(1)},${_pos[1].toFixed(1)})`;
  ctx.font = 'bold 11px monospace';
  const lw = ctx.measureText(label).width;
  const lx = Math.min(cx2 + 14, W - lw - PAD - 4);
  const ly = cy2 - 6;
  ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(lx - 2, ly - 11, lw + 4, 14);
  ctx.fillStyle = dotColor; ctx.textAlign = 'left';
  ctx.fillText(label, lx, ly);

  // ── Multi-drone overlay (when checkbox enabled) ──────────────────
  // Draw every OTHER drone in the fleet in a distinct colour so the
  // C2 operator can see the whole swarm at once.
  const showAll = document.getElementById('arena_show_all_drones');
  if (showAll && showAll.checked && window._fleetObservers) {
    const FLEET_COLORS = ['#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399'];
    let idx = 0;
    for (const [did, st] of Object.entries(window._fleetObservers)) {
      if (did === (window.activeDroneId || activeDroneId)) { idx++; continue; }
      const p = st && (st.pos || st.cam);
      if (!p || p.length < 2) { idx++; continue; }
      const col = FLEET_COLORS[idx % FLEET_COLORS.length];
      const [rx, ry] = arenaToCanvas(p[0], p[1]);
      const cxN = Math.max(M, Math.min(W - M, rx));
      const cyN = Math.max(M, Math.min(H - M, ry));
      const oob = rx !== cxN || ry !== cyN;
      // Smaller dot for non-active drones
      ctx.beginPath(); ctx.arc(cxN, cyN, 7, 0, Math.PI*2);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();
      // heading tick
      const yawDeg = st.yaw;
      if (typeof yawDeg === 'number') {
        const a = yawDeg * Math.PI / 180;
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cxN, cyN);
        ctx.lineTo(cxN + Math.sin(a) * 16, cyN - Math.cos(a) * 16);
        ctx.stroke();
      }
      // label with drone name + position
      ctx.font = 'bold 10px monospace';
      const name = (window._fleetNames && window._fleetNames[did]) || did;
      const tag = `${name} (${p[0].toFixed(1)},${p[1].toFixed(1)})${oob ? ' OOB':''}`;
      const tw = ctx.measureText(tag).width;
      const tx = Math.min(cxN + 10, W - tw - PAD - 4);
      const ty = cyN - 9;
      ctx.fillStyle = 'rgba(0,0,0,0.65)'; ctx.fillRect(tx - 2, ty - 10, tw + 4, 13);
      ctx.fillStyle = col; ctx.textAlign = 'left';
      ctx.fillText(tag, tx, ty);
      idx++;
    }
  }
}

// ── Fleet-wide position poll (drives multi-drone arena view) ──
window._fleetObservers = {};
window._fleetNames = {};
async function fleetPoll() {
  try {
    const r = await fetch('/proxy/aruco/fleet', {cache:'no-store'});
    const d = await r.json();
    if (d && d.observers) {
      window._fleetObservers = d.observers;
      // Grab drone display names from the main drones dict if available
      if (typeof drones === 'object') {
        const names = {};
        for (const [did, info] of Object.entries(drones || {})) {
          names[did] = info?.name || did;
        }
        window._fleetNames = names;
      }
      // Feed positions into the Three.js scene if active
      if (window._arena3d && window._arena3d.updateDrones) {
        window._arena3d.updateDrones(d.observers);
      }
    }
  } catch {}
}
setInterval(fleetPoll, 500);
fleetPoll();

function updatePosUI(d) {
  const pos = d.pos;
  const vel = d.vel || [0, 0, 0];
  const lat = (d.latency_ms || 0) / 1000;
  const compPos = pos ? pos.map((v, i) => v + (vel[i] || 0) * lat) : null;

  document.getElementById('pos_x').textContent = pos ? pos[0].toFixed(2) : '\\u2014';
  document.getElementById('pos_y').textContent = pos ? pos[1].toFixed(2) : '\\u2014';
  document.getElementById('pos_z').textContent = pos ? pos[2].toFixed(2) : '\\u2014';

  const dir = d.dir;
  const hdg = dir ? ((Math.atan2(dir[0], dir[1]) * 180 / Math.PI + 360) % 360).toFixed(1) : '\\u2014';
  document.getElementById('pos_hdg').textContent = hdg;
  const spd = vel ? Math.sqrt((vel[0]||0)**2 + (vel[1]||0)**2).toFixed(2) : '\\u2014';
  document.getElementById('pos_vel').textContent = spd + ' m/s';
  document.getElementById('pos_refs').textContent = d.ref_markers ? d.ref_markers.length : '\\u2014';
  document.getElementById('pos_fps').textContent = d.fps != null ? d.fps : '\\u2014';
  document.getElementById('pos_stale').style.display = d.stale ? '' : 'none';

  const enabled = d.enabled !== false;
  const badge = document.getElementById('pos_status_badge');
  const dr = d.dead_reckoning;
  badge.textContent = !enabled ? 'disabled' : pos ? (dr ? 'IMU DR' : d.stale ? 'stale' : 'live') : 'no markers';
  badge.style.color = !enabled ? '#64748b' : (pos && !d.stale && !dr) ? '#22c55e' : dr ? '#06b6d4' : '#f59e0b';

  if (d.frame_w && d.frame_h) _lastFrameRes = `${d.frame_w}x${d.frame_h}`;
  // Stash visible + reference markers so drawArena can highlight them
  _seenMarkers = new Set((d.seen_markers || []).map(String));
  _refMarkers  = new Set((d.ref_markers  || []).map(String));
  if (pos) console.log('[POS] drawArena pos=', pos, 'compPos=', compPos, 'frame=', _lastFrameRes);
  drawArena(pos, compPos, dir);
}

function startPosEvents() {
  if (posEvtSource) { posEvtSource.close(); posEvtSource = null; }
  posEvtSource = new EventSource('/proxy/position/events');
  posEvtSource.onmessage = (e) => { try { updatePosUI(JSON.parse(e.data)); } catch(err) { console.error('POS SSE error:', err, e.data); } };
  posEvtSource.onerror = () => {
    posEvtSource.close(); posEvtSource = null;
    setTimeout(startPosEvents, 500);
  };
}

async function loadPosConfig() {
  try {
    const r = await fetch('/proxy/position/config');
    const d = await r.json();
    // Server returns {config:{...}, ...} but older endpoints are flat — handle both
    const c = d.config || d;
    document.getElementById('pos_enabled').checked = !!c.enabled;
    if (c.detect_profile) document.getElementById('pos_profile').value = c.detect_profile;
    if (c.fov_deg) document.getElementById('pos_fov').value = c.fov_deg;
    if (typeof c.imu_weight === 'number') {
      const pct = Math.round(c.imu_weight * 100);
      document.getElementById('pos_imu_weight').value = pct;
      document.getElementById('pos_imu_weight_val').textContent = pct + '%';
    }
    const latMs = (c.latency_ms != null) ? c.latency_ms
                  : (c.latency_comp_s != null ? Math.round(c.latency_comp_s * 1000) : null);
    if (latMs != null) {
      document.getElementById('pos_latency').value = latMs;
      document.getElementById('pos_latency_val').textContent = Math.round(latMs);
    }
    // ── Populate filter controls ──
    const kCb = document.getElementById('pos_kalman');
    if (kCb) kCb.checked = (c.enable_kalman_filter !== false);  // default ON if missing
    const mSz = document.getElementById('pos_marker_size');
    if (mSz && c.marker_size_m != null) mSz.value = Number(c.marker_size_m).toFixed(2);
    const tK = document.getElementById('pos_top_k');
    if (tK && c.top_k_markers != null) tK.value = c.top_k_markers;
    const out = document.getElementById('pos_outlier');
    if (out && c.outlier_reject_m != null) out.value = c.outlier_reject_m;
    const cs = document.getElementById('pos_calib_status');
    cs.textContent = d.has_calibration ? '\\u2713 calibration loaded' : 'no calibration';
    cs.style.color = d.has_calibration ? '#22c55e' : '#94a3b8';
    if (c.enabled) startPosEvents();
  } catch {}
}

// ── Filter controls — live-apply to running positioner ──
(function wireFilterControls(){
  const statusEl = () => document.getElementById('pos_filters_status');
  const flash = (msg, col) => {
    const e = statusEl(); if (!e) return;
    e.textContent = msg; e.style.color = col || '#64748b';
    setTimeout(() => { if (e.textContent === msg) e.textContent = ''; }, 2500);
  };
  const applyBtn = document.getElementById('pos_filters_apply');
  if (applyBtn) applyBtn.onclick = async () => {
    const payload = {
      enable_kalman_filter: document.getElementById('pos_kalman').checked,
      marker_size_m: parseFloat(document.getElementById('pos_marker_size').value),
      top_k_markers: parseInt(document.getElementById('pos_top_k').value, 10),
      outlier_reject_m: parseFloat(document.getElementById('pos_outlier').value),
    };
    try {
      const r = await fetch('/proxy/position/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.ok) flash('\\u2713 applied', '#22c55e');
      else flash('error: ' + (d.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
  const resetBtn = document.getElementById('pos_filters_reset');
  if (resetBtn) resetBtn.onclick = () => {
    document.getElementById('pos_kalman').checked = true;
    document.getElementById('pos_marker_size').value = '0.50';
    document.getElementById('pos_top_k').value = '0';
    document.getElementById('pos_outlier').value = '2.5';
    flash('defaults loaded — click Apply', '#94a3b8');
  };
})();

document.getElementById('pos_enabled').onchange = async function() {
  await post('/proxy/position/config', { enabled: this.checked });
  if (this.checked) startPosEvents();
  else { if (posEvtSource) { posEvtSource.close(); posEvtSource = null; } _lastPos = null; drawArena(); }
};

document.getElementById('pos_latency').oninput = function() {
  document.getElementById('pos_latency_val').textContent = this.value;
};

// IMU blend slider — debounced write-through so dragging doesn't spam
let _imuWeightTimer = null;
document.getElementById('pos_imu_weight').oninput = function() {
  document.getElementById('pos_imu_weight_val').textContent = this.value + '%';
  if (_imuWeightTimer) clearTimeout(_imuWeightTimer);
  const v = parseFloat(this.value) / 100;
  _imuWeightTimer = setTimeout(() => {
    post('/proxy/position/config', { imu_weight: v });
  }, 150);
};

document.getElementById('pos_cfg_save').onclick = async () => {
  const profile = document.getElementById('pos_profile').value;
  const fov = parseFloat(document.getElementById('pos_fov').value);
  const lat = parseFloat(document.getElementById('pos_latency').value);
  await post('/proxy/position/config', { detect_profile: profile, fov_deg: fov, latency_ms: lat });
  setTimeout(loadPosConfig, 400);
};

document.getElementById('pos_calib_file').onchange = async function() {
  if (!this.files.length) return;
  const fd = new FormData();
  fd.append('file', this.files[0]);
  const cs = document.getElementById('pos_calib_status');
  cs.textContent = 'uploading...'; cs.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/position/calibration', { method: 'POST', body: fd });
    const d = await r.json();
    cs.textContent = d.ok ? '\\u2713 calibration saved' : ('error: ' + d.error);
    cs.style.color = d.ok ? '#22c55e' : '#ef4444';
    if (d.ok) loadPosConfig();
  } catch { cs.textContent = 'upload error'; cs.style.color = '#ef4444'; }
  this.value = '';
};

document.getElementById('pos_video_toggle').onclick = () => {
  posVideoOn = !posVideoOn;
  document.getElementById('pos_video_toggle').textContent = posVideoOn ? 'Hide ArUco Video' : 'Show ArUco Video';
  const container = document.getElementById('pos_video_container');
  const img = document.getElementById('pos_video_img');
  if (posVideoOn) { img.src = '/proxy/position/video?' + Date.now(); container.style.display = ''; }
  else { img.src = ''; container.style.display = 'none'; }
};

let _recActive = false;
const recBtn = document.getElementById('rec_btn');
const recStatus = document.getElementById('rec_status');

async function refreshRecStatus() {
  try {
    const d = await (await fetch('/proxy/video/record/status')).json();
    _recActive = d.recording;
    recBtn.textContent = _recActive ? '\\u25a0 Stop Rec' : '\\u25cf Record';
    recBtn.style.borderColor = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.color = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.background = _recActive ? '#3b0f0f' : '#1e3a2e';
    document.getElementById('rec_raw').disabled = _recActive;
    if (_recActive) recStatus.textContent = `${d.frames} frames \u2022 ${d.raw ? 'raw' : 'ann'} \u2022 ${d.path.split('/').pop()}`;
    else recStatus.textContent = d.frames ? `saved ${d.frames} frames` : '';
  } catch {}
}

recBtn.onclick = async () => {
  if (_recActive) {
    const d = await (await fetch('/proxy/video/record/stop', {method:'POST'})).json();
    recStatus.textContent = d.ok ? `saved ${d.frames} frames: ${d.path.split('/').pop()}` : ('error: ' + d.error);
  } else {
    const raw = document.getElementById('rec_raw').checked;
    const d = await (await fetch('/proxy/video/record/start', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({raw})})).json();
    recStatus.textContent = d.ok ? `recording... ${d.path.split('/').pop()}` : ('error: ' + d.error);
  }
  refreshRecStatus();
};

refreshRecStatus();
setInterval(refreshRecStatus, 5000);

loadPosConfig();
// ── Arena Configuration ───────────────────────────────────────────────────────
let arenaMarkers = {};   // {id_str: {pos:[x,y,z], wall:'front'}}

loadArenaConfig();   // pre-load markers so canvas shows them before panel is opened
drawArena();

const WALLS = ['front','back','left','right'];

function renderMarkerTable() {
  const tbody = document.getElementById('arena_marker_table');
  const sorted = Object.keys(arenaMarkers).sort((a,b) => Number(a)-Number(b));
  tbody.innerHTML = '';
  const rowStyle = 'display:flex;gap:4px;align-items:center;margin-bottom:3px;';
  const iStyle = 'height:26px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;padding:0 4px;';
  sorted.forEach(id => {
    const m = arenaMarkers[id];
    const row = document.createElement('div');
    row.style.cssText = rowStyle;
    const wallOpts = WALLS.map(w => `<option value="${w}" ${m.wall===w?'selected':''}>${w}</option>`).join('');
    row.innerHTML = `
      <input type="number" step="1" min="0" value="${id}" data-oldid="${id}" data-rename="1" style="${iStyle}width:52px;text-align:right;" title="Marker ID (editable)" />
      <input type="number" step="0.001" value="${m.pos[0]}" data-id="${id}" data-f="0" style="${iStyle}width:58px;" title="X" />
      <input type="number" step="0.001" value="${m.pos[1]}" data-id="${id}" data-f="1" style="${iStyle}width:58px;" title="Y" />
      <input type="number" step="0.001" value="${m.pos[2]}" data-id="${id}" data-f="2" style="${iStyle}width:52px;" title="Z" />
      <select data-id="${id}" data-f="wall" style="${iStyle}width:70px;">${wallOpts}</select>
      <button data-del="${id}" style="padding:1px 6px;font-size:10px;background:#7f1d1d;border-color:#dc2626;">✕</button>
    `;
    tbody.appendChild(row);
  });
  // Bind change events
  tbody.querySelectorAll('input[data-f]').forEach(el => {
    el.addEventListener('change', () => {
      const id = el.dataset.id, f = parseInt(el.dataset.f);
      if (arenaMarkers[id]) arenaMarkers[id].pos[f] = parseFloat(el.value) || 0;
    });
  });
  tbody.querySelectorAll('select[data-f]').forEach(el => {
    el.addEventListener('change', () => {
      if (arenaMarkers[el.dataset.id]) arenaMarkers[el.dataset.id].wall = el.value;
    });
  });
  tbody.querySelectorAll('input[data-rename]').forEach(el => {
    el.addEventListener('change', () => {
      const oldId = el.dataset.oldid;
      const n = parseInt(el.value, 10);
      if (isNaN(n) || n < 0) { alert('Marker ID must be a non-negative integer.'); el.value = oldId; return; }
      const newId = String(n);
      if (newId === oldId) return;
      if (arenaMarkers[newId]) {
        alert('Marker ID ' + newId + ' already exists.');
        el.value = oldId;
        return;
      }
      arenaMarkers[newId] = arenaMarkers[oldId];
      delete arenaMarkers[oldId];
      renderMarkerTable();
    });
  });
  tbody.querySelectorAll('button[data-del]').forEach(btn => {
    btn.addEventListener('click', () => {
      delete arenaMarkers[btn.dataset.del];
      renderMarkerTable();
    });
  });
}

async function loadArenaConfig() {
  try {
    const r = await fetch('/proxy/arena/config');
    const d = await r.json();
    if (d.arena) {
      document.getElementById('ac_width').value = d.arena.width_m ?? 20;
      document.getElementById('ac_depth').value = d.arena.depth_m ?? 10;
      document.getElementById('ac_hmin').value  = d.arena.height_min_m ?? -1;
      document.getElementById('ac_hmax').value  = d.arena.height_max_m ?? 1;
    }
    if (d.marker_size_m != null) document.getElementById('ac_msize').value = d.marker_size_m;
    if (d.markers) { arenaMarkers = JSON.parse(JSON.stringify(d.markers)); renderMarkerTable(); }
    // Update arena canvas world dimensions
    if (d.arena) {
      arenaW = d.arena.width_m  || 20;
      arenaD = d.arena.depth_m  || 10;
      arenaOX = -(arenaW / 2);
      arenaOY = 0;
      _updateView();
      drawArena();  // redraw with last known position (no args = use saved state)
    }
  } catch {}
}

document.getElementById('arena_cfg_toggle').onclick = function() {
  const body = document.getElementById('arena_cfg_body');
  const hidden = body.style.display === 'none';
  body.style.display = hidden ? '' : 'none';
  this.textContent = hidden ? 'Hide' : 'Show';
  if (hidden) loadArenaConfig();
};

document.getElementById('arena_add_marker').onclick = () => {
  const input = document.getElementById('arena_new_marker_id');
  const typed = (input.value || '').trim();
  let newId;
  if (typed !== '') {
    const n = parseInt(typed, 10);
    if (isNaN(n) || n < 0) { alert('Marker ID must be a non-negative integer.'); return; }
    newId = String(n);
    if (arenaMarkers[newId]) {
      alert('Marker ID ' + newId + ' already exists. Pick a different number or clear the field to auto-assign.');
      return;
    }
  } else {
    const ids = Object.keys(arenaMarkers).map(Number).filter(n => !isNaN(n));
    newId = String(ids.length ? Math.max(...ids) + 1 : 1);
  }
  arenaMarkers[newId] = {pos: [0, 0, 0], wall: 'front'};
  // Auto-increment the input so repeated clicks add sequential IDs
  input.value = String(parseInt(newId, 10) + 1);
  renderMarkerTable();
  // Scroll to bottom
  const tb = document.getElementById('arena_marker_table');
  tb.scrollTop = tb.scrollHeight;
};

document.getElementById('arena_save').onclick = async () => {
  // Sync any un-committed inputs
  document.querySelectorAll('#arena_marker_table input[data-f]').forEach(el => {
    const id = el.dataset.id, f = parseInt(el.dataset.f);
    if (arenaMarkers[id]) arenaMarkers[id].pos[f] = parseFloat(el.value) || 0;
  });
  document.querySelectorAll('#arena_marker_table select[data-f]').forEach(el => {
    if (arenaMarkers[el.dataset.id]) arenaMarkers[el.dataset.id].wall = el.value;
  });
  const payload = {
    arena: {
      width_m: parseFloat(document.getElementById('ac_width').value),
      depth_m: parseFloat(document.getElementById('ac_depth').value),
      height_min_m: parseFloat(document.getElementById('ac_hmin').value),
      height_max_m: parseFloat(document.getElementById('ac_hmax').value),
    },
    marker_size_m: parseFloat(document.getElementById('ac_msize').value),
    markers: arenaMarkers,
  };
  const st = document.getElementById('arena_cfg_status');
  st.textContent = 'saving...'; st.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/arena/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    st.textContent = d.ok ? `\\u2713 saved (${d.marker_count} markers)` : ('error: ' + (d.error||'?'));
    st.style.color = d.ok ? '#22c55e' : '#ef4444';
  } catch(e) { st.textContent = 'error: ' + e.message; st.style.color = '#ef4444'; }
};

document.getElementById('arena_reset').onclick = async () => {
  if (!confirm('Reset arena config to built-in defaults?')) return;
  const st = document.getElementById('arena_cfg_status');
  st.textContent = 'resetting...'; st.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/arena/config/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    if (d.ok) { await loadArenaConfig(); st.textContent = '\\u2713 reset to defaults'; st.style.color = '#22c55e'; }
    else { st.textContent = 'error'; st.style.color = '#ef4444'; }
  } catch(e) { st.textContent = 'error: ' + e.message; st.style.color = '#ef4444'; }
};

// Dynamic arena dimensions (updated when config is loaded)
let ARENA_W_dyn = 20, ARENA_D_dyn = 10;

// ── Live Telemetry Graphs (DISABLED — moved to standalone script block earlier in HTML) ─
/* (function(){
  const WINDOW_S = 10;          // rolling window in seconds
  const SAMPLE_HZ = 20;         // sampling rate (polls lastTelemetry/_lastPos)
  const CANVAS_W = 340, CANVAS_H = 130;
  const GROUPS = [
    {title:'Altitude (cm)',     keys:['height_cm','tof_cm','barometer_cm'], colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Attitude (°)',      keys:['pitch','roll','yaw'],                colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Velocity (cm/s)',   keys:['vgx','vgy','vgz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Acceleration',      keys:['agx','agy','agz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Speed',             keys:['speed'],                              colors:['#22d3ee']},
    {title:'Battery (%)',       keys:['battery'],                            colors:['#34d399']},
    {title:'Temperature (°C)',  keys:['temperature'],                        colors:['#fb923c']},
    {title:'Position (m)',      keys:['pos_x','pos_y','pos_z'],              colors:['#22d3ee','#f472b6','#a78bfa']},
  ];
  const graphs = [];
  let graphsVisible = false;
  let rafId = null;
  let sampleTimer = null;

  function initGraphs() {
    const container = document.getElementById('graphs_container');
    if (!container) { console.error('[graphs] graphs_container not found'); return; }
    if (graphs.length) return;
    GROUPS.forEach(g => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:6px;';
      const hdr = document.createElement('div');
      hdr.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;';
      let html = '<b style="color:#e2e8f0;">' + g.title + '</b>';
      g.keys.forEach((k,i) => { html += '<span style="color:'+g.colors[i]+';">'+k+'</span>'; });
      hdr.innerHTML = html;
      wrap.appendChild(hdr);
      const canvas = document.createElement('canvas');
      canvas.width = CANVAS_W; canvas.height = CANVAS_H;
      canvas.style.cssText = 'width:100%;height:auto;display:block;background:#020617;border-radius:4px;';
      wrap.appendChild(canvas);
      container.appendChild(wrap);
      graphs.push({keys: g.keys, colors: g.colors, samples: [], canvas, ctx: canvas.getContext('2d')});
    });
    console.log('[graphs] initialized', graphs.length, 'graphs');
  }

  // Sample current telemetry + position into all graphs
  function takeSample() {
    const ts = performance.now();
    // Build a combined sample object from globals
    const t = (typeof lastTelemetry === 'object' && lastTelemetry) ? Object.assign({}, lastTelemetry) : {};
    if (typeof _lastPos !== 'undefined' && Array.isArray(_lastPos)) {
      t.pos_x = _lastPos[0]; t.pos_y = _lastPos[1]; t.pos_z = _lastPos[2];
    }
    graphs.forEach(g => {
      const vals = {};
      let hasAny = false;
      g.keys.forEach(k => {
        const v = t[k];
        if (v != null && !isNaN(v)) { vals[k] = Number(v); hasAny = true; }
        else { vals[k] = null; }
      });
      if (hasAny) g.samples.push({t: ts, vals});
      const cutoff = ts - WINDOW_S * 1000;
      while (g.samples.length > 0 && g.samples[0].t < cutoff) g.samples.shift();
    });
  }

  function drawAll() {
    if (!graphsVisible) { rafId = null; return; }
    const now = performance.now();
    graphs.forEach(g => {
      const ctx = g.ctx, W = g.canvas.width, H = g.canvas.height;
      ctx.fillStyle = '#020617'; ctx.fillRect(0,0,W,H);
      if (g.samples.length < 2) {
        ctx.fillStyle = '#475569'; ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('waiting for data...', W/2, H/2);
        ctx.textAlign = 'left';
        return;
      }
      const tMin = now - WINDOW_S*1000, tMax = now;
      let yMin = Infinity, yMax = -Infinity;
      g.samples.forEach(s => { g.keys.forEach(k => {
        if (s.vals[k]!=null) { yMin=Math.min(yMin,s.vals[k]); yMax=Math.max(yMax,s.vals[k]); }
      }); });
      if (!isFinite(yMin) || !isFinite(yMax)) return;
      if (yMin === yMax) { yMin -= 1; yMax += 1; }
      const pad = (yMax - yMin) * 0.1 || 1;
      yMin -= pad; yMax += pad;
      // grid
      ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
      for (let i=0;i<=4;i++) {
        const y = (i/4)*H;
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
      }
      // y labels
      ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
      for (let i=0;i<=4;i++) {
        const v = yMin + ((4-i)/4)*(yMax-yMin);
        ctx.fillText(v.toFixed(1), 2, (i/4)*H + 9);
      }
      // each series
      g.keys.forEach((k, ki) => {
        ctx.strokeStyle = g.colors[ki]; ctx.lineWidth = 1.5; ctx.beginPath();
        let started = false;
        g.samples.forEach(s => {
          if (s.vals[k] == null) { started = false; return; }
          const x = ((s.t - tMin) / (tMax - tMin)) * W;
          const y = H - ((s.vals[k] - yMin) / (yMax - yMin)) * H;
          if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
        });
        ctx.stroke();
        // last value label
        const last = g.samples[g.samples.length - 1];
        if (last && last.vals[k] != null) {
          ctx.fillStyle = g.colors[ki]; ctx.font = '10px monospace';
          ctx.textAlign = 'right';
          ctx.fillText(last.vals[k].toFixed(1), W - 2, 10 + ki * 11);
          ctx.textAlign = 'left';
        }
      });
    });
    rafId = requestAnimationFrame(drawAll);
  }

  function startGraphs() {
    initGraphs();
    if (!sampleTimer) sampleTimer = setInterval(takeSample, 1000 / SAMPLE_HZ);
    if (!rafId) rafId = requestAnimationFrame(drawAll);
  }
  function stopGraphs() {
    if (sampleTimer) { clearInterval(sampleTimer); sampleTimer = null; }
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }

  // Wire toggle button — wait for DOM if needed
  function wireToggle() {
    const btn = document.getElementById('graphs_toggle');
    const panel = document.getElementById('graphs_panel');
    if (!btn || !panel) {
      console.warn('[graphs] button or panel not found, retrying...');
      setTimeout(wireToggle, 100);
      return;
    }
    btn.addEventListener('click', () => {
      graphsVisible = !graphsVisible;
      btn.textContent = graphsVisible ? 'Hide Graphs' : 'Show Graphs';
      panel.style.display = graphsVisible ? 'block' : 'none';
      console.log('[graphs] toggled', graphsVisible);
      if (graphsVisible) startGraphs(); else stopGraphs();
    });
    console.log('[graphs] toggle wired');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireToggle);
  } else {
    wireToggle();
  }
})(); */
</script>

<!-- ── Optional 3D arena view via Three.js ─────────────────────────── -->
<script type=\"module\">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  let scene, camera, renderer, controls, droneMeshes = {}, markerMeshes = {}, rafId = 0;
  const ARENA_W = 20.0, ARENA_D = 10.8, ARENA_H = 6.0;
  const ARENA_OX = -10.0, ARENA_OY = 0.0;

  function init3D(container) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(15, 12, 15);

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.5, 5.4);
    controls.update();

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(5, 12, 8);
    scene.add(dir);

    // Arena floor — 1 m grid (matches the 2D overlay)
    const grid = new THREE.GridHelper(20, 20, 0x475569, 0x1e3a5f);
    grid.position.set(0, 0, ARENA_D / 2);
    scene.add(grid);
    // Depth-direction grid (10.8 m, we round to 11 cells)
    const grid2 = new THREE.GridHelper(12, 12, 0x475569, 0x1e3a5f);
    grid2.rotation.x = Math.PI / 2;
    grid2.position.set(0, 3, 0);
    grid2.material.opacity = 0.0;
    // keep subtle — 1 grid is enough; the floor grid is what matters

    // Arena box (wireframe)
    const boxGeom = new THREE.BoxGeometry(ARENA_W, ARENA_H, ARENA_D);
    const boxMat = new THREE.LineBasicMaterial({color: 0x3b82f6, transparent: true, opacity: 0.4});
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), boxMat);
    box.position.set(0, ARENA_H / 2, ARENA_D / 2);
    scene.add(box);

    // Arena origin marker (green corner cube)
    const originGeom = new THREE.BoxGeometry(0.3, 0.3, 0.3);
    const origin = new THREE.Mesh(originGeom, new THREE.MeshStandardMaterial({color: 0x10b981}));
    origin.position.set(0, 0.15, 0);
    scene.add(origin);

    // Fetch + plot arena markers (from the arena_config we already fetched)
    if (window.arenaMarkers && Object.keys(window.arenaMarkers).length) {
      for (const [id, m] of Object.entries(window.arenaMarkers)) {
        if (!m.pos) continue;
        const g = new THREE.BoxGeometry(0.5, 0.5, 0.05);
        const col = (m.wall === 'front') ? 0x6366f1
                 : (m.wall === 'back')  ? 0xa855f7
                 : (m.wall === 'left')  ? 0x06b6d4
                 : (m.wall === 'right') ? 0x10b981 : 0x94a3b8;
        const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: col}));
        mesh.position.set(m.pos[0], m.pos[2] || 2, m.pos[1]);
        scene.add(mesh);
        markerMeshes[id] = mesh;
      }
    }

    window.addEventListener('resize', () => {
      if (!renderer) return;
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    function loop() {
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(loop);
    }
    loop();
  }

  function updateDrones(observers) {
    if (!scene) return;
    const DRONE_COLORS = [0xf97316, 0x38bdf8, 0xa78bfa, 0xf472b6, 0xfbbf24];
    let idx = 0;
    const seen = new Set();
    for (const [did, st] of Object.entries(observers || {})) {
      seen.add(did);
      const p = st && (st.pos || st.cam);
      if (!p || p.length < 2) { idx++; continue; }
      let mesh = droneMeshes[did];
      if (!mesh) {
        const col = DRONE_COLORS[idx % DRONE_COLORS.length];
        // Drone = small sphere with a forward nose-cone
        const group = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 16, 12),
          new THREE.MeshStandardMaterial({color: col}));
        group.add(body);
        const nose = new THREE.Mesh(
          new THREE.ConeGeometry(0.08, 0.3, 8),
          new THREE.MeshStandardMaterial({color: 0xffffff}));
        nose.rotation.x = Math.PI / 2;
        nose.position.set(0, 0, 0.2);
        group.add(nose);
        // Label (using a sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 32;
        const c = canvas.getContext('2d');
        c.fillStyle = 'rgba(0,0,0,0.7)'; c.fillRect(0,0,128,32);
        c.fillStyle = '#fff'; c.font = 'bold 18px monospace';
        c.fillText(did, 8, 22);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true}));
        sprite.scale.set(0.9, 0.22, 1);
        sprite.position.set(0, 0.4, 0);
        group.add(sprite);
        scene.add(group);
        mesh = group;
        droneMeshes[did] = mesh;
      }
      mesh.position.set(p[0], p[2] || 1.5, p[1]);
      if (typeof st.yaw === 'number') mesh.rotation.y = -st.yaw * Math.PI / 180;
      idx++;
    }
    // Remove meshes for drones that disappeared
    for (const did of Object.keys(droneMeshes)) {
      if (!seen.has(did)) {
        scene.remove(droneMeshes[did]);
        delete droneMeshes[did];
      }
    }
  }

  function teardown3D() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
    if (renderer) {
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      renderer = null;
    }
    scene = null; camera = null; controls = null;
    droneMeshes = {}; markerMeshes = {};
  }

  // Expose for fleetPoll()
  window._arena3d = {updateDrones};

  // Toggle wiring
  const cb = document.getElementById('arena_show_3d');
  const wrap = document.getElementById('arena3d_wrap');
  const container = document.getElementById('arena3d_container');
  if (cb && wrap && container) {
    cb.addEventListener('change', () => {
      if (cb.checked) {
        wrap.style.display = '';
        if (!scene) {
          try { init3D(container); }
          catch (e) { console.error('[3D] init failed:', e); cb.checked = false; wrap.style.display = 'none'; }
        }
      } else {
        wrap.style.display = 'none';
        teardown3D();
      }
    });
  }
</script>
</body>
</html>
"""


def log_command(event: str, payload: dict | None = None):
    if not command_log_enabled:
        return
    try:
        # High-frequency keepalive events can starve command handling if logged each packet.
        now = time.time()
        key = event
        throttle_s = 0.0
        if event in {"key_down", "key_up"}:
            k = str((payload or {}).get("key", ""))
            key = f"{event}:{k}"
            throttle_s = 0.5
        last = command_log_last.get(key, 0.0)
        if throttle_s > 0 and (now - last) < throttle_s:
            return
        command_log_last[key] = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "ts": ts,
            "event": event,
            "payload": payload or {},
        }
        command_log_path.parent.mkdir(parents=True, exist_ok=True)
        with command_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[REMOTE CMD] {ts} {event} payload={payload or {}}")
    except Exception:
        # Logging must never break control flow.
        pass


def pi_post(path: str, body: dict | None = None, timeout: float | None = None):
    return _http_session.post(f"{PI_BASE}{path}", json=body or {}, timeout=TIMEOUT_CMD if timeout is None else timeout)


def pi_get(path: str, timeout: float | None = None):
    return _http_session.get(f"{PI_BASE}{path}", timeout=TIMEOUT_CMD if timeout is None else timeout)


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.get("/proxy/drones")
def proxy_drones():
    return jsonify(drones=DRONES, active=active_drone_id)


@app.post("/proxy/switch")
def proxy_switch():
    global active_drone_id, PI_BASE
    data = request.get_json(silent=True) or {}
    drone_id = str(data.get("id", ""))
    if drone_id not in DRONES:
        return jsonify(ok=False, error="unknown drone id"), 400
    active_drone_id = drone_id
    PI_BASE = DRONES[drone_id]["base"]
    log_command("switch_drone", {"id": drone_id, "name": DRONES[drone_id]["name"]})
    print(f"[REMOTE UI] Switched to {DRONES[drone_id]['name']} @ {PI_BASE}")
    return jsonify(ok=True, active=drone_id, name=DRONES[drone_id]["name"], base=PI_BASE)


@app.get("/proxy/drones/config")
def proxy_drones_config():
    """Return full drone config for editing."""
    return jsonify(drones=DRONES, config_path=str(DRONES_CONFIG_PATH))


@app.post("/proxy/drones/config")
def proxy_drones_config_save():
    """Save updated drone config. Expects {drones: {id: {name, type, base}, ...}}"""
    global DRONES, PI_BASE
    data = request.get_json(silent=True) or {}
    new_drones = data.get("drones")
    if not new_drones or not isinstance(new_drones, dict):
        return jsonify(ok=False, error="drones dict required"), 400
    for did, info in new_drones.items():
        if not all(k in info for k in ("name", "type", "base")):
            return jsonify(ok=False, error=f"drone {did} missing name/type/base"), 400
        if info["type"] not in ("tello", "anafi"):
            return jsonify(ok=False, error=f"drone {did} type must be tello or anafi"), 400
    DRONES.clear()
    DRONES.update(new_drones)
    if active_drone_id in DRONES:
        PI_BASE = DRONES[active_drone_id]["base"]
    save_drones_config(DRONES)
    # Keep ArUco fleet in sync with drone config
    try:
        aruco_fleet.configure(DRONES)
    except Exception as e:
        print(f"[ARUCO] fleet reconfigure failed: {e}")
    print(f"[CONFIG] Saved {len(DRONES)} drones to {DRONES_CONFIG_PATH}")
    return jsonify(ok=True, drones=DRONES)


@app.post("/proxy/key_down")
def proxy_key_down():
    data = request.get_json(silent=True) or {}
    log_command("key_down", data)
    r = pi_post("/api/key_down", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/key_up")
def proxy_key_up():
    data = request.get_json(silent=True) or {}
    log_command("key_up", data)
    r = pi_post("/api/key_up", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/key_batch")
def proxy_key_batch():
    """Batch key_down for all currently held keys in a single HTTP request."""
    data = request.get_json(silent=True) or {}
    keys = data.get("keys", [])
    last_r = None
    for k in keys:
        payload = {"key": k}
        log_command("key_down", payload)
        last_r = pi_post("/api/key_down", payload)
    if last_r is not None:
        return (last_r.text, last_r.status_code, {"Content-Type": last_r.headers.get("Content-Type", "application/json")})
    return jsonify(ok=True)


@app.post("/proxy/takeoff")
def proxy_takeoff():
    log_command("takeoff")
    r = pi_post("/api/takeoff")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/land")
def proxy_land():
    log_command("land")
    r = pi_post("/api/land")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/land_all")
def proxy_land_all():
    """Emergency panic-button: land every configured drone and halt any
    running mission. Used by the 'q' hotkey in the UI. Tolerates
    individual drones being unreachable — collects per-drone outcomes
    and always returns 200 so the client can render the summary."""
    log_command("land_all")

    # 1) Stop any running mission first so its LIVE-mode state machine
    #    doesn't immediately re-push RC commands that conflict with land.
    mission_stopped = False
    try:
        if mission_manager is not None and mission_manager.current is not None:
            mission_stopped = mission_manager.stop(land=False)
    except Exception as e:
        print(f"[LAND_ALL] mission stop failed: {e}")

    # 2) Send /api/land to every configured drone in parallel-ish
    #    (sequentially but with a short timeout per drone).
    results: dict[str, dict] = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            resp = _http_session.post(f"{base.rstrip('/')}/api/land",
                                      json={}, timeout=3.0)
            try:
                j = resp.json()
            except Exception:
                j = {"raw": resp.text[:120]}
            results[str(did)] = {
                "ok": bool(j.get("ok", resp.status_code == 200)),
                "status": resp.status_code,
                "msg": j.get("msg") or j.get("error") or "",
            }
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}

    # 3) Also put every ArUco observer back into OBSERVE mode and clear
    #    any lingering search RC so the drones don't fight the land.
    try:
        for did, obs in aruco_fleet._obs.items():
            try:
                obs.set_search_rc(0, 0, 0, 0)
                obs.set_mode("observe")
            except Exception:
                pass
    except Exception:
        pass

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[LAND_ALL] {ok_count}/{len(results)} drones acknowledged land "
          f"(mission_stopped={mission_stopped})")
    return jsonify(
        ok=True,
        landed=ok_count,
        total=len(results),
        mission_stopped=mission_stopped,
        results=results,
    )


@app.post("/proxy/flip")
def proxy_flip():
    data = request.get_json(silent=True) or {}
    log_command("flip", data)
    r = pi_post("/api/flip", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/recover")
def proxy_recover():
    log_command("recover")
    r = pi_post("/api/recover")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/emergency")
def proxy_emergency():
    log_command("emergency")
    r = pi_post("/api/emergency")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/speed")
def proxy_speed():
    data = request.get_json(silent=True) or {}
    log_command("speed", data)
    r = pi_post("/api/speed", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/move")
def proxy_move():
    data = request.get_json(silent=True) or {}
    log_command("move", data)
    r = pi_post("/api/move", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/rotate")
def proxy_rotate():
    data = request.get_json(silent=True) or {}
    log_command("rotate", data)
    r = pi_post("/api/rotate", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/go")
def proxy_go():
    data = request.get_json(silent=True) or {}
    log_command("go", data)
    r = pi_post("/api/go", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/curve")
def proxy_curve():
    data = request.get_json(silent=True) or {}
    log_command("curve", data)
    r = pi_post("/api/curve", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/stream")
def proxy_stream():
    data = request.get_json(silent=True) or {}
    log_command("stream", data)
    r = pi_post("/api/stream", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/sdk")
def proxy_sdk():
    data = request.get_json(silent=True) or {}
    log_command("sdk", data)
    r = pi_post("/api/sdk", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


# -- Anafi / Olympe proxy routes --

@app.post("/proxy/camera/photo")
def proxy_camera_photo():
    log_command("camera_photo")
    r = pi_post("/api/camera/photo")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/camera/record/start")
def proxy_camera_record_start():
    log_command("camera_record_start")
    r = pi_post("/api/camera/record/start")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/camera/record/stop")
def proxy_camera_record_stop():
    log_command("camera_record_stop")
    r = pi_post("/api/camera/record/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/gimbal")
def proxy_gimbal():
    data = request.get_json(silent=True) or {}
    log_command("gimbal", data)
    r = pi_post("/api/gimbal", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/rth")
def proxy_rth():
    data = request.get_json(silent=True) or {}
    log_command("rth", data)
    r = pi_post("/api/rth", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/moveto")
def proxy_moveto():
    data = request.get_json(silent=True) or {}
    log_command("moveto", data)
    r = pi_post("/api/moveto", data, timeout=65)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/settings")
def proxy_settings_get():
    try:
        r = pi_get("/api/settings", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/settings")
def proxy_settings_set():
    data = request.get_json(silent=True) or {}
    log_command("settings", data)
    r = pi_post("/api/settings", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/safety/takeoff")
def proxy_safe_takeoff_get():
    try:
        r = pi_get("/api/safety/takeoff", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/safety/takeoff")
def proxy_safe_takeoff_set():
    data = request.get_json(silent=True) or {}
    log_command("safe_takeoff_set", data)
    r = pi_post("/api/safety/takeoff", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/logging/commands")
def proxy_command_log_status():
    return jsonify(enabled=command_log_enabled, path=str(command_log_path))


@app.post("/proxy/logging/commands")
def proxy_command_log_config():
    global command_log_enabled, command_log_path
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    path = data.get("path")
    if enabled is not None and not isinstance(enabled, bool):
        return jsonify(ok=False, error="enabled must be boolean"), 400
    if path is not None and (not isinstance(path, str) or not path.strip()):
        return jsonify(ok=False, error="path must be non-empty string"), 400
    if isinstance(enabled, bool):
        command_log_enabled = enabled
    if isinstance(path, str) and path.strip():
        command_log_path = Path(path.strip())
    print(f"[REMOTE CMD] command logging {'enabled' if command_log_enabled else 'disabled'}")
    return jsonify(ok=True, enabled=command_log_enabled, path=str(command_log_path))


@app.get("/proxy/logging/commands/download")
def proxy_command_log_download():
    p = command_log_path
    if not p.exists():
        return jsonify(ok=False, error="command log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/proxy/logging/commands/clear")
def proxy_command_log_clear():
    p = command_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


@app.get("/proxy/logging/telemetry")
def proxy_log_status():
    try:
        r = pi_get("/api/logging/telemetry", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/logging/telemetry")
def proxy_log_config():
    data = request.get_json(silent=True) or {}
    log_command("telemetry_log_set", data)
    r = pi_post("/api/logging/telemetry", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/logging/telemetry/download")
def proxy_log_download():
    r = pi_get("/api/logging/telemetry/download")
    headers = {
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream"),
        "Content-Disposition": r.headers.get("Content-Disposition", "attachment; filename=telemetry_log.jsonl"),
    }
    return (r.content, r.status_code, headers)


@app.post("/proxy/logging/telemetry/clear")
def proxy_log_clear():
    log_command("telemetry_log_clear")
    r = pi_post("/api/logging/telemetry/clear")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/video")
def proxy_video_feed():
    """Proxy the MJPEG video stream from the Pi API server."""
    try:
        r = _http_session.get(f"{PI_BASE}/api/video", stream=True, timeout=30)
        return Response(
            r.iter_content(chunk_size=32768),
            mimetype=r.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"),
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/start")
def proxy_video_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "mjpeg")
    # For forward mode, auto-detect C2 IP (this machine) so the Pi sends UDP here
    if mode == "forward" and not data.get("target_host"):
        # Use the host part of request.host (what the browser connected to)
        c2_host = request.host.split(":")[0]
        if c2_host in ("127.0.0.1", "localhost"):
            # Try to get our real IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                c2_host = s.getsockname()[0]
                s.close()
            except Exception:
                pass
        data["target_host"] = c2_host
        data["target_port"] = data.get("target_port", VIDEO_UDP_FORWARD_PORT)
    log_command("video_start", data)
    # For forward mode, start the UDP receiver BEFORE telling the Pi to forward
    # so ffmpeg is already listening when packets arrive
    if mode == "forward":
        _start_udp_receiver()
        time.sleep(0.5)  # Give ffmpeg time to bind the UDP port
    r = pi_post("/api/video/start", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/video/stop")
def proxy_video_stop():
    log_command("video_stop")
    _stop_udp_receiver()
    r = pi_post("/api/video/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/video/status")
def proxy_video_status():
    try:
        r = pi_get("/api/video/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return jsonify(ok=False, error=str(e), mode="off"), 502


# ---------------------------------------------------------------------------
# UDP → MJPEG bridge for forward mode
# Receives raw H264 UDP from the Pi, decodes with cv2, serves as MJPEG
# ---------------------------------------------------------------------------
_udp_receiver_running = False
_udp_receiver_thread = None
_udp_last_frame_lock = threading.Lock()
_udp_last_jpeg: bytes = b""
_udp_has_frame = False


_ffmpeg_proc = None


def _start_udp_receiver():
    """Start ffmpeg to decode H264 UDP and produce raw frames, then encode to JPEG."""
    global _udp_receiver_running, _udp_receiver_thread, _udp_has_frame, _udp_last_jpeg
    if _udp_receiver_running:
        return
    _udp_receiver_running = True
    _udp_has_frame = False
    _udp_last_jpeg = b""
    _udp_receiver_thread = threading.Thread(target=_udp_receiver_loop, daemon=True, name="udp-video-recv")
    _udp_receiver_thread.start()


def _stop_udp_receiver():
    global _udp_receiver_running, _udp_has_frame, _udp_last_jpeg, _ffmpeg_proc
    _udp_receiver_running = False
    if _ffmpeg_proc is not None:
        try:
            _ffmpeg_proc.terminate()
            _ffmpeg_proc.wait(timeout=3)
        except Exception:
            try:
                _ffmpeg_proc.kill()
            except Exception:
                pass
        _ffmpeg_proc = None
    with _udp_last_frame_lock:
        _udp_last_jpeg = b""
        _udp_has_frame = False


def _udp_receiver_loop():
    """Use ffmpeg to decode H264 UDP stream → raw RGB frames → JPEG."""
    global _udp_last_jpeg, _udp_has_frame, _ffmpeg_proc, _udp_receiver_running
    import subprocess as sp
    import shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[C2-VIDEO] ffmpeg not found — install ffmpeg to decode UDP forward stream")
        _udp_receiver_running = False
        return

    width, height = 960, 720  # Tello default resolution
    frame_size = width * height * 3  # RGB24

    cmd = [
        ffmpeg_bin,
        "-y",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-probesize", "5000000",
        "-analyzeduration", "5000000",
        "-i", f"udp://0.0.0.0:{VIDEO_UDP_FORWARD_PORT}?overrun_nonfatal=1&fifo_size=50000000",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-vf", f"scale={width}:{height}",
        "pipe:1",
    ]
    print(f"[C2-VIDEO] Starting ffmpeg on port {VIDEO_UDP_FORWARD_PORT}...")
    try:
        _ffmpeg_proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=frame_size * 2)
    except Exception as e:
        print(f"[C2-VIDEO] ffmpeg failed to start: {e}")
        _udp_receiver_running = False
        return

    # Log stderr in a separate thread so we can see ffmpeg errors
    def _log_stderr():
        for line in _ffmpeg_proc.stderr:
            txt = line.decode(errors="replace").rstrip()
            if txt:
                print(f"[C2-FFMPEG] {txt}")
    threading.Thread(target=_log_stderr, daemon=True, name="ffmpeg-stderr").start()

    frame_count = 0
    buf = b""
    while _udp_receiver_running:
        try:
            to_read = frame_size - len(buf)
            chunk = _ffmpeg_proc.stdout.read(to_read)
            if not chunk:
                # ffmpeg exited — check if it's still running
                rc = _ffmpeg_proc.poll()
                if rc is not None:
                    print(f"[C2-VIDEO] ffmpeg exited with code {rc}")
                    break
                time.sleep(0.01)
                continue
            buf += chunk
            if len(buf) < frame_size:
                continue
            # We have a full frame (BGR24)
            frame_data = buf[:frame_size]
            buf = buf[frame_size:]
            if HAS_CV2:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                ok, jpg_buf = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
                if ok:
                    with _udp_last_frame_lock:
                        _udp_last_jpeg = jpg_buf.tobytes()
                        _udp_has_frame = True
                    frame_count += 1
                    if frame_count == 1:
                        print("[C2-VIDEO] First frame decoded successfully")
        except Exception as e:
            if _udp_receiver_running:
                print(f"[C2-VIDEO] Error: {e}")
            break

    _udp_receiver_running = False
    if _ffmpeg_proc:
        try:
            _ffmpeg_proc.terminate()
        except Exception:
            pass
    print(f"[C2-VIDEO] Receiver stopped ({frame_count} frames decoded)")


@app.get("/proxy/video/forward_stream")
def proxy_video_forward_stream():
    """Serve decoded UDP forward frames as MJPEG stream."""
    def gen():
        while _udp_receiver_running:
            with _udp_last_frame_lock:
                jpg = _udp_last_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1.0 / max(1, VIDEO_FPS))
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/proxy/heartbeat")
def proxy_heartbeat():
    """Send heartbeat to ALL drones in parallel to keep their watchdogs alive."""
    def _ping(did_info):
        did, info = did_info
        try:
            r = _http_session.get(f"{info['base']}/api/heartbeat", timeout=0.3)
            return did, r.status_code
        except Exception:
            return did, "timeout"

    futures = list(_heartbeat_pool.map(_ping, DRONES.items()))
    results = dict(futures)
    return jsonify(ok=True, drones=results)


@app.get("/proxy/telemetry")
def proxy_telemetry():
    try:
        r = pi_get("/api/telemetry", timeout=TIMEOUT_STATUS)
        headers = {
            "Content-Type": r.headers.get("Content-Type", "application/json"),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return (r.text, r.status_code, headers)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


# ── Positioning subsystem proxy ───────────────────────────────────────────────

@app.get("/proxy/position")
def proxy_position_get():
    try:
        r = pi_get("/api/position", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/position/events")
def proxy_position_events():
    """SSE proxy — streams ArUco position events from Pi to browser."""
    pi_url = PI_BASE + "/api/position/events"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/proxy/telemetry/stream")
def proxy_telemetry_stream():
    """SSE proxy — streams telemetry events from Pi to browser (replaces polling)."""
    pi_url = PI_BASE + "/api/telemetry/stream"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/proxy/position/video")
def proxy_position_video():
    """MJPEG proxy — streams ArUco-annotated frames from Pi."""
    pi_url = PI_BASE + "/api/position/video"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        yield chunk
        except Exception:
            pass

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.get("/proxy/position/config")
def proxy_position_config_get():
    try:
        r = pi_get("/api/position/config", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/position/config")
def proxy_position_config_set():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/position/config", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/position/calibration")
def proxy_position_calibration():
    """Proxy NPZ calibration file upload to Pi."""
    if "file" not in request.files:
        return jsonify(ok=False, error="no file"), 400
    f = request.files["file"]
    try:
        resp = _http_session.post(
            PI_BASE + "/api/position/calibration",
            files={"file": (f.filename, f.read(), "application/octet-stream")},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code,
                        content_type=resp.headers.get("Content-Type", "application/json"))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ── Arena configuration proxy ─────────────────────────────────────────────────

def _pi_arena_to_js(d: dict) -> dict:
    """Convert Pi API flat arena format → JS-expected nested format.

    Pi API returns:
      {arena_width_m, arena_height_m, marker_size_m,
       markers: [{id, x, y, z, wall, label?}, ...]}

    JS expects:
      {arena: {width_m, depth_m, height_min_m, height_max_m},
       marker_size_m,
       markers: {"0": {pos:[x,y,z], wall, label?}, ...}}
    """
    out: dict = {"ok": d.get("ok", True)}
    out["arena"] = {
        "width_m":      d.get("arena_width_m", 20.0),
        "depth_m":      d.get("arena_height_m", 10.0),
        "height_min_m": d.get("arena_height_min_m", -1.0),
        "height_max_m": d.get("arena_height_max_m",  1.0),
    }
    out["marker_size_m"] = d.get("marker_size_m", 0.5)
    markers_dict: dict = {}
    for m in (d.get("markers") or []):
        mid = str(m.get("id", "?"))
        entry: dict = {
            "pos": [
                float(m.get("x", 0)),
                float(m.get("y", 0)),
                float(m.get("z", 0)),
            ],
            "wall": m.get("wall", "front"),
        }
        if "label" in m:
            entry["label"] = m["label"]
        markers_dict[mid] = entry
    out["markers"] = markers_dict
    return out


def _js_arena_to_pi(data: dict) -> dict:
    """Convert JS POST format → Pi API format.

    JS sends:
      {arena: {width_m, depth_m, height_min_m, height_max_m},
       marker_size_m,
       markers: {"0": {pos:[x,y,z], wall, label?}, ...}}

    Pi API expects:
      {arena_width_m, arena_height_m, marker_size_m,
       markers: [{id, x, y, z, wall}, ...]}
    """
    out: dict = {}
    arena = data.get("arena") or {}
    if arena.get("width_m") is not None:
        out["arena_width_m"] = float(arena["width_m"])
    if arena.get("depth_m") is not None:
        out["arena_height_m"] = float(arena["depth_m"])
    if data.get("marker_size_m") is not None:
        out["marker_size_m"] = float(data["marker_size_m"])
    raw_markers = data.get("markers")
    if isinstance(raw_markers, dict):
        arr = []
        for mid_str, m in raw_markers.items():
            pos = m.get("pos") or [0, 0, 0]
            entry: dict = {
                "id":   int(mid_str),
                "x":    float(pos[0]) if len(pos) > 0 else 0.0,
                "y":    float(pos[1]) if len(pos) > 1 else 0.0,
                "z":    float(pos[2]) if len(pos) > 2 else 0.0,
                "wall": m.get("wall", "front"),
            }
            if "label" in m:
                entry["label"] = m["label"]
            arr.append(entry)
        arr.sort(key=lambda e: e["id"])
        out["markers"] = arr
    return out


@app.get("/proxy/arena/config")
def proxy_arena_config_get():
    try:
        r = pi_get("/api/arena/config", timeout=TIMEOUT_STATUS)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/arena/config")
def proxy_arena_config_set():
    data = request.get_json(silent=True) or {}
    try:
        pi_payload = _js_arena_to_pi(data)
        r = pi_post("/api/arena/config", pi_payload)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/arena/config/reset")
def proxy_arena_config_reset():
    try:
        r = pi_post("/api/arena/config/reset", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/video/record/status")
def proxy_rec_status():
    try:
        r = pi_get("/api/video/record/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/record/start")
def proxy_rec_start():
    try:
        data = request.get_json(silent=True) or {}
        r = pi_post("/api/video/record/start", data)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/record/stop")
def proxy_rec_stop():
    try:
        r = pi_post("/api/video/record/stop", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


# ─── ArUco Seek (multi-drone hover-in-front-of-marker) ─────────────────────


def _aruco_resolve(drone_id: str | None):
    """Return observer for the given id, or the active drone if omitted."""
    did = str(drone_id) if drone_id else active_drone_id
    obs = aruco_fleet.get(did)
    if obs is None:
        return None, did
    return obs, did


@app.get("/proxy/aruco/state")
def proxy_aruco_state():
    """Snapshot of ONE observer (query ?id=1, defaults to active drone)."""
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(running=False, drone_id=did, error="unknown drone"), 404
    return jsonify(obs.get_state())


@app.get("/proxy/aruco/fleet")
def proxy_aruco_fleet():
    """Snapshot of every observer in the fleet."""
    return jsonify(active=active_drone_id,
                   allow_live=aruco_fleet.allow_live,
                   observers=aruco_fleet.all_states())


@app.get("/proxy/aruco/params")
def proxy_aruco_params_get():
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(asdict_hover_defaults()), 200
    return jsonify(obs.get_params())


def asdict_hover_defaults():
    from dataclasses import asdict
    return asdict(AsHoverParams())


@app.post("/proxy/aruco/params")
def proxy_aruco_params_set():
    data = request.get_json(silent=True) or {}
    did = str(data.pop("id", "") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    applied = obs.update_params(data)
    return jsonify(ok=True, applied=applied)


@app.post("/proxy/aruco/start")
def proxy_aruco_start():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_start", {"id": did})
    obs.start()
    return jsonify(ok=True, drone_id=did)


@app.post("/proxy/aruco/stop")
def proxy_aruco_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_stop", {"id": did})
    obs.stop()
    return jsonify(ok=True, drone_id=did)


@app.post("/proxy/aruco/target")
def proxy_aruco_target():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    mid = data.get("marker")
    if mid is not None:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            mid = None
    obs.set_target(mid)
    log_command("aruco_target", {"id": did, "marker": mid})
    return jsonify(ok=True, drone_id=did, marker=mid)


@app.post("/proxy/aruco/mode")
def proxy_aruco_mode():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    requested = (data.get("mode") or "").lower()
    if requested == "live" and not obs.allow_live:
        return jsonify(ok=False, mode=obs.mode,
                       error="LIVE mode disabled on this server (REMOTE_NO_LIVE=1)"), 403
    actual = obs.set_mode(requested)
    log_command("aruco_mode", {"id": did, "mode": actual})
    return jsonify(ok=(actual == requested), drone_id=did, mode=actual)


def _aruco_require_live(obs):
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    if obs.mode != "live":
        return jsonify(ok=False,
                       error=f"refused — observer mode is '{obs.mode}', switch to 'live' first"), 409
    return None


@app.post("/proxy/aruco/takeoff")
def proxy_aruco_takeoff():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_takeoff", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_takeoff())


@app.post("/proxy/aruco/land")
def proxy_aruco_land():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_land", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_land())


@app.post("/proxy/aruco/emergency")
def proxy_aruco_emergency():
    # Allowed at any mode — killswitch must always be reachable
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_emergency", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_emergency())


@app.post("/proxy/aruco/rc_stop")
def proxy_aruco_rc_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_rc_stop", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_rc_stop())


@app.get("/proxy/aruco/video.mjpg")
def proxy_aruco_video():
    """Pass-through MJPEG for the observer's drone (separate from /proxy/video
    so the main UI's video stream and the ArUco Seek panel don't collide)."""
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return Response(b"", status=404)
    upstream_url = f"{obs.api_base}/api/position/video"
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


# ─── Special Missions (multi-drone coordinated flight) ────────────────────


def _parse_marker_list(raw) -> list[int]:
    """Parse '1-12' / '1,2,3,5-7' / [1,2,3] into a sorted unique int list."""
    if isinstance(raw, list):
        return sorted({int(x) for x in raw if str(x).strip()})
    if not isinstance(raw, str):
        return []
    out: set[int] = set()
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, _, b = tok.partition("-")
            try:
                a_i, b_i = int(a), int(b)
                if a_i > b_i:
                    a_i, b_i = b_i, a_i
                for v in range(a_i, b_i + 1):
                    out.add(v)
            except ValueError:
                continue
        else:
            try:
                out.add(int(tok))
            except ValueError:
                continue
    return sorted(out)


@app.get("/proxy/missions/status")
def proxy_missions_status():
    return jsonify(mission_manager.status())


@app.post("/proxy/missions/scan_all/start")
def proxy_missions_scan_all_start():
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_markers = _parse_marker_list(data.get("target_markers", "1-12"))
    if not target_markers:
        return jsonify(ok=False, error="target_markers must parse to at least one id"), 400
    hover_seconds = float(data.get("hover_seconds", 3.0))
    approach_tolerance_m = float(data.get("approach_tolerance_m", 0.30))
    approach_skew_tol   = float(data.get("approach_skew_tol", 0.12))
    approach_err_x_tol  = float(data.get("approach_err_x_tol", 0.15))
    auto_takeoff = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_scan_all(
        drone_ids=[str(d) for d in drone_ids],
        target_markers=target_markers,
        hover_seconds=hover_seconds,
        approach_tolerance_m=approach_tolerance_m,
        approach_skew_tol=approach_skew_tol,
        approach_err_x_tol=approach_err_x_tol,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_scan_all_start", {
        "drone_ids": drone_ids, "target_markers": target_markers,
        "hover_seconds": hover_seconds, "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@app.post("/proxy/missions/capture_targets/start")
def proxy_missions_capture_targets_start():
    """Launch the SDC26 capture-all-targets mission. Body:
    {
      "drone_ids":    ["1", "2", ...],
      "target_boxes": [{"id":1,"x":-5.0,"y":2.0}, ...],
      "home_xy":      [x, y],                    # team home coord
      "arena_face_xy":[x, y],                    # where the camera aims
      "hover_above_m": 1.5,
      "hover_seconds": 4.0,
      "auto_takeoff":  false
    }
    """
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_boxes = data.get("target_boxes") or []
    if not isinstance(target_boxes, list) or not target_boxes:
        return jsonify(ok=False, error="target_boxes (non-empty list) required"), 400
    home_xy = data.get("home_xy", [0.0, 1.5])
    arena_face_xy = data.get("arena_face_xy", [0.0, 5.4])
    hover_above_m = float(data.get("hover_above_m", 1.5))
    hover_seconds = float(data.get("hover_seconds", 4.0))
    nav_tol_xy_m  = float(data.get("nav_tol_xy_m", 0.3))
    auto_takeoff  = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_capture_all_targets(
        drone_ids=[str(d) for d in drone_ids],
        target_boxes=target_boxes,
        home_xy=tuple(home_xy),
        arena_face_xy=tuple(arena_face_xy),
        hover_above_m=hover_above_m,
        hover_seconds=hover_seconds,
        nav_tol_xy_m=nav_tol_xy_m,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_capture_targets_start", {
        "drone_ids": drone_ids,
        "target_boxes": target_boxes,
        "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@app.post("/proxy/missions/stop")
def proxy_missions_stop():
    data = request.get_json(silent=True) or {}
    land = bool(data.get("land", False))
    ok = mission_manager.stop(land=land)
    log_command("mission_stop", {"land": land, "ok": ok})
    return jsonify(ok=ok, status=mission_manager.status())


@app.get("/proxy/missions/trace")
def proxy_missions_trace():
    """Download the trace log for the most recent mission. The mission
    class writes a JSONL file per run; we return the current one (or the
    most recent if no mission is active)."""
    from pathlib import Path as _P
    import glob as _glob
    path = None
    try:
        cur = mission_manager.current
        if cur is not None and getattr(cur, "trace_path", None):
            path = cur.trace_path
    except Exception:
        pass
    if not path:
        # Fall back to newest file in the logs dir
        try:
            from aruco_seek_multi import MISSION_LOG_DIR
            files = sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl")))
            if files:
                path = files[-1]
        except Exception:
            pass
    if not path or not _P(path).exists():
        return jsonify(ok=False, error="no trace available yet"), 404
    return send_file(path, mimetype="application/x-ndjson",
                     as_attachment=True,
                     download_name=_P(path).name)


@app.get("/proxy/missions/traces")
def proxy_missions_traces():
    """List all mission trace files on disk with size and mtime."""
    import glob as _glob
    from aruco_seek_multi import MISSION_LOG_DIR
    files = []
    try:
        for f in sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl"))):
            p = Path(f)
            st = p.stat()
            files.append({
                "name": p.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(ok=True, files=files)


# ─── Environment & Wi-Fi control — pass-through to active drone ─────────────

@app.get("/proxy/environment")
def proxy_environment_get():
    try:
        r = _http_session.get(f"{PI_BASE}/api/environment", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/environment")
def proxy_environment_set():
    data = request.get_json(silent=True) or {}
    log_command("environment_set", data)
    try:
        r = _http_session.post(f"{PI_BASE}/api/environment", json=data, timeout=4)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/wifi/status")
def proxy_wifi_status():
    try:
        r = _http_session.get(f"{PI_BASE}/api/wifi/status", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/wifi/channel")
def proxy_wifi_channel():
    data = request.get_json(silent=True) or {}
    log_command("wifi_channel_set", data)
    try:
        r = _http_session.post(f"{PI_BASE}/api/wifi/channel", json=data, timeout=8)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/wifi/scan")
def proxy_wifi_scan():
    data = request.get_json(silent=True) or {}
    try:
        r = _http_session.post(f"{PI_BASE}/api/wifi/scan", json=data, timeout=10)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


# ── Magnetometer (Anafi) ────────────────────────────────────────────────────
# Parrot Anafi requires a figure-8 magnetometer calibration whenever the drone
# is moved between locations or re-powered. unified_api_server.py exposes the
# raw GET /api/magneto and POST /api/magneto/calibrate; here we expose a
# higher-level /proxy/magneto/recalibrate that drives the full cycle.

@app.get("/proxy/magneto")
def proxy_magneto_status():
    """Current magnetometer calibration status for the active drone."""
    try:
        r = pi_get("/api/magneto", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/magneto/calibrate")
def proxy_magneto_calibrate():
    """One-shot: send the StartMagnetoCalibration command and return.
    The caller is then responsible for the figure-8 dance and for polling
    /proxy/magneto until all axes report ok."""
    log_command("magneto_calibrate")
    try:
        r = pi_post("/api/magneto/calibrate", timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


def _parse_magneto_axes(status: str | None) -> dict:
    """Extract per-axis calibration bits from a status string like
    'REQUIRED, axes=x0y1z0, in-progress'. Returns {x, y, z, failed, done,
    in_progress, required} with sensible defaults when the drone hasn't
    reported yet."""
    s = (status or "").lower()
    out = {"x": None, "y": None, "z": None,
           "failed": False, "done": False,
           "in_progress": False, "required": None}
    compact = s.replace(" ", "")
    import re
    m = re.search(r"axes=x(\d)y(\d)z(\d)", compact)
    if m:
        out["x"] = int(m.group(1))
        out["y"] = int(m.group(2))
        out["z"] = int(m.group(3))
    out["failed"] = "failed" in s
    out["done"] = "all-axes-ok" in s or (
        out["x"] == 1 and out["y"] == 1 and out["z"] == 1
    )
    out["in_progress"] = "in-progress" in s
    if "not-required" in s:
        out["required"] = False
    elif "required" in s:
        out["required"] = True
    return out


def _magneto_cycle(timeout_s: float, poll_s: float):
    """Generator that drives one full magnetometer recalibration cycle.

    Yields dicts of shape:
      {"kind": "step", "step": <name>, "ok": bool, ...extra}
      {"kind": "status", "status": str, "axes": {x,y,z,...}}   # during polling
      {"kind": "final", "ok": bool, "pre": ..., "post": ..., ...}

    The caller is responsible for transport (blocking JSON or SSE)."""
    steps: list[dict] = []

    def step(name: str, ok: bool, **info):
        entry = {"kind": "step", "step": name, "ok": ok, **info}
        steps.append({k: v for k, v in entry.items() if k != "kind"})
        print(f"[MAGNETO] {'OK' if ok else 'FAIL'} {name} {info}")
        return entry

    # 1) Heartbeat
    try:
        hb = pi_get("/api/heartbeat", timeout=TIMEOUT_STATUS).json()
    except Exception as e:
        yield step("heartbeat", False, error=str(e))
        yield {"kind": "final", "ok": False, "fatal": "unreachable",
               "error": f"drone unreachable: {e}", "steps": steps}
        return
    connected = bool(hb.get("connected"))
    flying = bool(hb.get("flying"))
    yield step("heartbeat", connected and not flying,
               connected=connected, flying=flying,
               drone_type=hb.get("drone_type"))
    if not connected:
        yield {"kind": "final", "ok": False, "fatal": "not_connected",
               "error": "drone not connected", "steps": steps}
        return
    if flying:
        yield {"kind": "final", "ok": False, "fatal": "flying",
               "error": "refuse to recalibrate while flying — land first",
               "steps": steps}
        return

    # 2) Pre-calibration snapshot
    try:
        pre = pi_get("/api/magneto", timeout=TIMEOUT_STATUS).json()
    except Exception as e:
        pre = {"ok": False, "error": str(e)}
    yield step("pre_status", bool(pre.get("ok")),
               status=pre.get("status"), required=pre.get("required"))

    # 3) Trigger calibration
    try:
        r = pi_post("/api/magneto/calibrate", timeout=TIMEOUT_CMD)
        start = r.json() if r.headers.get("Content-Type",
                                          "").startswith("application/json") else {}
    except Exception as e:
        yield step("start", False, error=str(e))
        yield {"kind": "final", "ok": False, "fatal": "start_failed",
               "error": f"start failed: {e}", "steps": steps, "pre": pre}
        return
    started = bool(start.get("ok"))
    yield step("start", started, http_status=r.status_code,
               message=start.get("message"), error=start.get("error"))
    if not started:
        yield {"kind": "final", "ok": False, "fatal": "start_refused",
               "error": start.get("error", "calibration not started"),
               "steps": steps, "pre": pre, "start": start}
        return

    # 4) Poll status — emit per-axis progress as it flips.
    deadline = time.time() + timeout_s
    post = {"ok": False, "status": None, "required": None}
    finished = False
    failed = False
    last_axes = None
    while time.time() < deadline:
        try:
            post = pi_get("/api/magneto", timeout=TIMEOUT_STATUS).json()
        except Exception as e:
            post = {"ok": False, "error": str(e)}
        axes = _parse_magneto_axes(post.get("status"))
        if axes != last_axes:
            yield {"kind": "status", "status": post.get("status"),
                   "axes": axes, "required": post.get("required")}
            last_axes = axes
        if axes["failed"]:
            failed = True
            break
        if axes["done"]:
            finished = True
            break
        time.sleep(poll_s)

    timed_out = not (finished or failed)
    yield step("poll", finished and not failed,
               final_status=post.get("status"),
               required=post.get("required"), timed_out=timed_out)

    ok = finished and not failed
    if ok:
        msg = "magnetometer calibrated"
    elif failed:
        msg = "magnetometer calibration FAILED — retry the figure-8 dance"
    else:
        msg = ("magnetometer calibration timed out — perform the figure-8 "
               "dance around each axis and retry")
    yield {"kind": "final", "ok": ok, "pre": pre, "post": post,
           "steps": steps, "failed": failed, "timed_out": timed_out,
           "message": msg}


@app.post("/proxy/magneto/recalibrate")
def proxy_magneto_recalibrate():
    """Blocking orchestrator: runs the full recalibration cycle and returns a
    single summary JSON. See /proxy/magneto/recalibrate/stream for live
    per-step progress.

    Body (optional):
      { "timeout_s": 60, "poll_s": 1.0 }"""
    data = request.get_json(silent=True) or {}
    try:
        timeout_s = max(5.0, float(data.get("timeout_s", 60)))
        poll_s = max(0.25, float(data.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate", {"timeout_s": timeout_s, "poll_s": poll_s})

    final = {"ok": False, "error": "no final event"}
    for ev in _magneto_cycle(timeout_s, poll_s):
        if ev.get("kind") == "final":
            final = {k: v for k, v in ev.items() if k != "kind"}
    return jsonify(final), 200


@app.get("/proxy/magneto/recalibrate/stream")
def proxy_magneto_recalibrate_stream():
    """SSE stream of the recalibration cycle. Used by the wizard GUI to light
    up each step + per-axis indicator as it happens.

    Query params: timeout_s (default 60), poll_s (default 1.0)."""
    try:
        timeout_s = max(5.0, float(request.args.get("timeout_s", 60)))
        poll_s = max(0.25, float(request.args.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate_stream",
                {"timeout_s": timeout_s, "poll_s": poll_s})

    def generate():
        for ev in _magneto_cycle(timeout_s, poll_s):
            yield f"data: {json.dumps(ev)}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def main():
    print("[REMOTE UI] Starting server...")
    print(f"[REMOTE UI] URL: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[REMOTE UI] PI_API_BASE={PI_BASE}")
    print(f"[REMOTE UI] timeouts: cmd={TIMEOUT_CMD}s status={TIMEOUT_STATUS}s")
    print(f"[REMOTE UI] HTTP session pool: connections=8 maxsize=16 keep-alive=on")
    print(f"[REMOTE UI] Telemetry: SSE push (fallback poll at 2s)")
    print(f"[REMOTE UI] Heartbeat: parallel fan-out to {len(DRONES)} drones")
    print("[REMOTE UI] Ready (waiting for browser requests)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
