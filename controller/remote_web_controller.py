import json
import os
import socket
import threading
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

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
    "1": {"name": "Anafi 1", "type": "anafi", "base": "http://192.168.1.20:8080"},
    "2": {"name": "Tello 1", "type": "tello", "base": "http://192.168.1.100:8080"},
    "3": {"name": "Drone 3", "type": "tello", "base": "http://192.168.1.101:8080"},
    "4": {"name": "Drone 4", "type": "tello", "base": "http://192.168.1.102:8080"},
    "5": {"name": "Drone 5", "type": "tello", "base": "http://192.168.1.103:8080"},
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
PI_BASE = DRONES[active_drone_id]["base"]

app = Flask(__name__)
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
  </style>
</head>
<body>
  <h2>Drone Remote Controller</h2>
  <div style=\"display:flex;align-items:center;gap:8px;\">
    <div class=\"drone-bar\" id=\"drone_bar\" style=\"flex:1;\"></div>
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
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"flip_l\">Flip L</button>
        <button id=\"flip_r\">Flip R</button>
        <button id=\"flip_f\">Flip F</button>
        <button id=\"flip_b\">Flip B</button>
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
          <button id=\"emergency\" style=\"background:#7f1d1d;border-color:#dc2626;\">EMERGENCY<br><span style=\"font-size:9px;font-weight:400;opacity:.8;\">Killswitch - will shutdown drone immediately - no safe landing</span></button>
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
        <div class=\"adv-grid\">
          <button id=\"take_photo\">Take Photo</button>
          <button id=\"rec_start\">Record Start</button>
          <button id=\"rec_stop\">Record Stop</button>
          <button id=\"rth_start\">Return Home</button>
          <button id=\"rth_cancel\">Cancel RTH</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Gimbal tilt</span>
          <input id=\"gimbal_tilt\" type=\"range\" min=\"-90\" max=\"30\" value=\"0\" style=\"flex:1;\" />
          <span id=\"gimbal_tilt_val\" class=\"small\" style=\"min-width:40px;\">0°</span>
          <button id=\"gimbal_set\">Set</button>
          <button id=\"gimbal_down\">Down (-90)</button>
          <button id=\"gimbal_fwd\">Forward (0)</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Max altitude (m)</span>
          <input id=\"set_alt\" type=\"number\" min=\"0.5\" max=\"150\" step=\"0.5\" value=\"2\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max vert spd</span>
          <input id=\"set_vspd\" type=\"number\" min=\"0.1\" max=\"4\" step=\"0.1\" value=\"0.5\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max tilt (°)</span>
          <input id=\"set_tilt\" type=\"number\" min=\"1\" max=\"35\" step=\"1\" value=\"15\" style=\"width:70px;\" />
          <button id=\"apply_settings\">Apply Settings</button>
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
        <div id=\"telemetry\" class=\"small\" style=\"white-space:pre-wrap; margin-top:10px;\">loading...</div>
      </div>
    </div>
  </div>

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
      <canvas id=\"arena_canvas\" class=\"arena-canvas\" width=\"600\" height=\"400\"></canvas>
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
  for (const [id, info] of Object.entries(drones)) {
    const btn = document.createElement('button');
    btn.className = 'drone-btn' + (id === activeDroneId ? ' selected' : '');
    btn.innerHTML = `${info.name}<span class="drone-type">${info.type}</span>`;
    btn.onclick = () => switchDrone(id);
    bar.appendChild(btn);
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
    refreshTelemetry();
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

setInterval(()=>{
  activeKeys.forEach((k)=>keyDown(k));
}, 150);

const holdButtons = document.querySelectorAll('button[data-k]');
holdButtons.forEach(btn=>{
  const k = btn.dataset.k;
  btn.addEventListener('pointerdown', e=>{ e.preventDefault(); btn.classList.add('active'); pressKey(k); });
  btn.addEventListener('pointerup',   e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointerleave',e=>{ btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointercancel',e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
});

document.getElementById('takeoff').onclick = ()=>post('/proxy/takeoff',{});
document.getElementById('land').onclick = ()=>post('/proxy/land',{});
document.getElementById('recover').onclick = ()=>post('/proxy/recover',{});
document.getElementById('flip_l').onclick = ()=>post('/proxy/flip',{dir:'l'});
document.getElementById('flip_r').onclick = ()=>post('/proxy/flip',{dir:'r'});
document.getElementById('flip_f').onclick = ()=>post('/proxy/flip',{dir:'f'});
document.getElementById('flip_b').onclick = ()=>post('/proxy/flip',{dir:'b'});

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

document.getElementById('emergency').onclick = ()=>post('/proxy/emergency',{});
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
document.getElementById('take_photo').onclick = ()=>post('/proxy/camera/photo',{});
document.getElementById('rec_start').onclick = ()=>post('/proxy/camera/record/start',{});
document.getElementById('rec_stop').onclick = ()=>post('/proxy/camera/record/stop',{});
document.getElementById('rth_start').onclick = ()=>post('/proxy/rth',{action:'start'});
document.getElementById('rth_cancel').onclick = ()=>post('/proxy/rth',{action:'cancel'});

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
  if (map.has(k)) {
    e.preventDefault();
    pressKey(k === ' ' ? 'space' : k);
  }
});
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

    const live = Boolean(t.connected) && Boolean(t.state_fresh);
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
  const mode = videoMode.value;
  if (mode === 'off') return;
  const body = {mode};
  // For forward mode, target_host is auto-detected by the web controller server
  try {
    const r = await fetch('/proxy/video/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if (d.ok) {
      videoActive = true;
      videoToggle.textContent = 'Stop Video';
      videoStatus.textContent = 'Mode: ' + d.mode;
      // Both modes show video via MJPEG in the browser
      videoContainer.style.display = '';
      if (d.mode === 'mjpeg') {
        videoImg.src = '/proxy/video?' + Date.now();
        videoUrl.style.display = '';
        videoUrl.innerHTML = 'Direct: <b>' + (d.stream_url || '') + '</b>';
      } else if (d.mode === 'forward') {
        // Forward mode: C2 receives UDP, decodes, serves as MJPEG
        videoImg.src = '/proxy/video/forward_stream?' + Date.now();
        videoUrl.style.display = '';
        videoUrl.innerHTML = 'UDP → C2 decode → MJPEG';
      }
    } else {
      videoStatus.textContent = 'Error: ' + (d.error || 'unknown');
    }
  } catch(e) {
    videoStatus.textContent = 'Error: ' + e;
  }
};

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
setInterval(refreshVideoStatus, 3000);

// Heartbeat — keeps the drone watchdog alive so it doesn't auto-land
async function sendHeartbeat(){
  try { await fetch('/proxy/heartbeat', {cache:'no-store'}); } catch {}
}
setInterval(sendHeartbeat, 500);
sendHeartbeat();

setInterval(refreshTelemetry, 700);
setInterval(refreshLogStatus, 2000);
setInterval(refreshSafeTakeoff, 2000);
setInterval(refreshCommandLogStatus, 2000);
loadDrones();
refreshTelemetry();
refreshLogStatus();
refreshSafeTakeoff();
refreshCommandLogStatus();

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

// Track last drawn position so arena config reload doesn't erase the dot
let _lastPos = null, _lastCompPos = null, _lastDir = null, _lastFrameRes = null;

function drawArena(pos, compPos, dir) {
  // Persist last known position so config reloads don't blank it
  if (pos !== undefined) { _lastPos = pos; _lastCompPos = compPos; _lastDir = dir; }
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

  // Grid lines over full view (5 m spacing)
  ctx.strokeStyle = '#1e3a5f'; ctx.lineWidth = 1;
  for (let gx = Math.ceil(viewOX / 5) * 5; gx <= viewOX + viewW + 0.01; gx += 5) {
    const [cx] = arenaToCanvas(gx, viewOY);
    ctx.beginPath(); ctx.moveTo(cx, PAD); ctx.lineTo(cx, H - PAD); ctx.stroke();
  }
  for (let gy = Math.ceil(viewOY / 5) * 5; gy <= viewOY + viewD + 0.01; gy += 5) {
    const [, cy] = arenaToCanvas(viewOX, gy);
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

  // Arena markers
  for (const [id, m] of Object.entries(arenaMarkers)) {
    if (!m.pos) continue;
    const [mx, my] = arenaToCanvas(m.pos[0], m.pos[1]);
    if (mx < PAD - 4 || mx > W - PAD + 4 || my < PAD - 4 || my > H - PAD + 4) continue;
    ctx.fillStyle = WALL_COLOR[m.wall] || '#94a3b8';
    ctx.fillRect(mx - 4, my - 4, 8, 8);
    ctx.fillStyle = '#cbd5e1'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText(id, mx, my - 6);
  }

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
}

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
  badge.textContent = !enabled ? 'disabled' : pos ? (d.stale ? 'stale' : 'live') : 'no markers';
  badge.style.color = !enabled ? '#64748b' : (pos && !d.stale) ? '#22c55e' : '#f59e0b';

  if (d.frame_w && d.frame_h) _lastFrameRes = `${d.frame_w}x${d.frame_h}`;
  if (pos) console.log('[POS] drawArena pos=', pos, 'compPos=', compPos, 'frame=', _lastFrameRes);
  drawArena(pos, compPos, dir);
}

function startPosEvents() {
  if (posEvtSource) { posEvtSource.close(); posEvtSource = null; }
  posEvtSource = new EventSource('/proxy/position/events');
  posEvtSource.onmessage = (e) => { try { updatePosUI(JSON.parse(e.data)); } catch(err) { console.error('POS SSE error:', err, e.data); } };
  posEvtSource.onerror = () => {
    posEvtSource.close(); posEvtSource = null;
    setTimeout(startPosEvents, 3000);
  };
}

async function loadPosConfig() {
  try {
    const r = await fetch('/proxy/position/config');
    const d = await r.json();
    document.getElementById('pos_enabled').checked = !!d.enabled;
    if (d.detect_profile) document.getElementById('pos_profile').value = d.detect_profile;
    if (d.fov_deg) document.getElementById('pos_fov').value = d.fov_deg;
    if (d.latency_ms != null) {
      document.getElementById('pos_latency').value = d.latency_ms;
      document.getElementById('pos_latency_val').textContent = Math.round(d.latency_ms);
    }
    const cs = document.getElementById('pos_calib_status');
    cs.textContent = d.has_calibration ? '\\u2713 calibration loaded' : 'no calibration';
    cs.style.color = d.has_calibration ? '#22c55e' : '#94a3b8';
    if (d.enabled) startPosEvents();
  } catch {}
}

document.getElementById('pos_enabled').onchange = async function() {
  await post('/proxy/position/config', { enabled: this.checked });
  if (this.checked) startPosEvents();
  else { if (posEvtSource) { posEvtSource.close(); posEvtSource = null; } _lastPos = null; drawArena(); }
};

document.getElementById('pos_latency').oninput = function() {
  document.getElementById('pos_latency_val').textContent = this.value;
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
setInterval(refreshRecStatus, 3000);

loadPosConfig();
loadArenaConfig();   // pre-load markers so canvas shows them before panel is opened
drawArena();

// ── Arena Configuration ───────────────────────────────────────────────────────
let arenaMarkers = {};   // {id_str: {pos:[x,y,z], wall:'front'}}

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
      <span style="min-width:24px;text-align:right;color:#94a3b8;font-size:11px;">${id}</span>
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
  const ids = Object.keys(arenaMarkers).map(Number).filter(n=>!isNaN(n));
  const newId = String(ids.length ? Math.max(...ids) + 1 : 30);
  arenaMarkers[newId] = {pos: [0, 0, 0], wall: 'front'};
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
    return requests.post(f"{PI_BASE}{path}", json=body or {}, timeout=TIMEOUT_CMD if timeout is None else timeout)


def pi_get(path: str, timeout: float | None = None):
    return requests.get(f"{PI_BASE}{path}", timeout=TIMEOUT_CMD if timeout is None else timeout)


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
        r = requests.get(f"{PI_BASE}/api/video", stream=True, timeout=30)
        return Response(
            r.iter_content(chunk_size=8192),
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
    """Send heartbeat to ALL drones to keep their watchdogs alive."""
    results = {}
    for did, info in DRONES.items():
        base = info["base"]
        try:
            r = requests.get(f"{base}/api/heartbeat", timeout=0.3)
            results[did] = r.status_code
        except Exception:
            results[did] = "timeout"
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
            with requests.get(pi_url, stream=True, timeout=(3, 120)) as resp:
                for chunk in resp.iter_content(chunk_size=512):
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
            with requests.get(pi_url, stream=True, timeout=(3, 120)) as resp:
                for chunk in resp.iter_content(chunk_size=4096):
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
        resp = requests.post(
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


def main():
    print("[REMOTE UI] Starting server...")
    print(f"[REMOTE UI] URL: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[REMOTE UI] PI_API_BASE={PI_BASE}")
    print(f"[REMOTE UI] timeouts: cmd={TIMEOUT_CMD}s status={TIMEOUT_STATUS}s")
    print("[REMOTE UI] Ready (waiting for browser requests)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
