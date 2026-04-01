import json
import os
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

# Runs on remote PC. Proxies to Pi API server.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8090
TIMEOUT_CMD = float(os.getenv("PI_TIMEOUT_CMD", "12"))
TIMEOUT_STATUS = float(os.getenv("PI_TIMEOUT_STATUS", "0.5"))

# Drone fleet – id → {name, type, base URL}
DRONES = {
    "1": {"name": "Anafi 1", "type": "anafi", "base": "http://192.168.1.20:8080"},
    "2": {"name": "Tello 1", "type": "tello", "base": "http://192.168.1.100:8080"},
    "3": {"name": "Drone 3", "type": "tello", "base": "http://192.168.1.101:8080"},
    "4": {"name": "Drone 4", "type": "tello", "base": "http://192.168.1.102:8080"},
    "5": {"name": "Drone 5", "type": "tello", "base": "http://192.168.1.103:8080"},
}
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
  </style>
</head>
<body>
  <h2>Drone Remote Controller</h2>
  <div class=\"drone-bar\" id=\"drone_bar\"></div>
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
          <button id=\"emergency\">EMERGENCY</button>
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
      <div class=\"small\" style=\"margin-top:8px;\">Keyboard in browser: W/A/S/D R/F Q/E, T, L, Space stop</div>
    </div>

    <div class=\"panel\" style=\"min-width:320px; flex:1;\">
      <div style=\"display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px;\">
        <b>Mission</b>
        <label class=\"small\" style=\"display:flex; align-items:center; gap:6px;\">
          Step delay (ms):
          <input id=\"mission_delay\" type=\"number\" min=\"0\" max=\"30000\" step=\"100\" value=\"500\" style=\"width:80px;\" />
        </label>
        <button id=\"mission_run\" style=\"background:#166534; border-color:#16a34a; color:#dcfce7;\">&#9654; Run</button>
        <button id=\"mission_stop\" style=\"background:#7f1d1d; border-color:#b91c1c; color:#fee2e2;\" disabled>&#9632; Stop</button>
        <span id=\"mission_status\" class=\"small\" style=\"color:#94a3b8;\"></span>
      </div>
      <textarea id=\"mission_editor\" rows=\"12\" spellcheck=\"false\" style=\"width:100%; box-sizing:border-box; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:8px; font-family:monospace; font-size:13px; resize:vertical;\" placeholder=\"# Mission script&#10;# Commands: takeoff  land  forward <cm>  back <cm>&#10;#           left <cm>  right <cm>  up <cm>  down <cm>&#10;#           rotate cw <deg>  rotate ccw <deg>&#10;#           delay <ms>  (extra pause at this step)&#10;# Lines starting with # are comments&#10;&#10;takeoff&#10;forward 500&#10;rotate cw 90&#10;forward 500&#10;land\"></textarea>
      <div class=\"small\" style=\"margin-top:4px; color:#475569;\">Commands: takeoff &nbsp;|&nbsp; land &nbsp;|&nbsp; forward/back/left/right/up/down &lt;cm&gt; &nbsp;|&nbsp; rotate cw/ccw &lt;deg&gt; &nbsp;|&nbsp; delay &lt;ms&gt;</div>
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

async function refreshTelemetry(){
  const apiEl = document.getElementById('api_status');
  const droneEl = document.getElementById('drone_status');
  try {
    const r = await fetch('/proxy/telemetry', {cache:'no-store'});
    if (!r.ok) throw new Error('api_error');
    const t = await r.json();

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
setInterval(refreshTelemetry, 700);
setInterval(refreshLogStatus, 2000);
setInterval(refreshSafeTakeoff, 2000);
setInterval(refreshCommandLogStatus, 2000);
loadDrones();
refreshTelemetry();
refreshLogStatus();
refreshSafeTakeoff();
refreshCommandLogStatus();

// --- Mission runner ---
let missionRunning = false;
let missionAbort = false;

function parseMission(text) {
  const steps = [];
  for (const raw of text.split('\\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\\s+/);
    const cmd = parts[0].toLowerCase();
    if (cmd === 'takeoff') {
      steps.push({ type: 'takeoff' });
    } else if (cmd === 'land') {
      steps.push({ type: 'land' });
    } else if (['forward','back','left','right','up','down'].includes(cmd)) {
      const cm = parseInt(parts[1]);
      if (isNaN(cm)) throw new Error('Missing cm value for: ' + line);
      steps.push({ type: 'move', dir: cmd, cm });
    } else if (cmd === 'rotate') {
      const dir = (parts[1] || '').toLowerCase();
      if (!['cw','ccw'].includes(dir)) throw new Error('rotate needs cw or ccw: ' + line);
      const deg = parseInt(parts[2]);
      if (isNaN(deg)) throw new Error('Missing deg for: ' + line);
      steps.push({ type: 'rotate', dir, deg });
    } else if (cmd === 'delay') {
      const ms = parseInt(parts[1]);
      if (isNaN(ms)) throw new Error('Missing ms for: ' + line);
      steps.push({ type: 'delay', ms });
    } else {
      throw new Error('Unknown command: ' + cmd);
    }
  }
  return steps;
}

async function runStep(step) {
  if (step.type === 'takeoff')  return post('/proxy/takeoff', {});
  if (step.type === 'land')     return post('/proxy/land', {});
  if (step.type === 'move')     return post('/proxy/move', { dir: step.dir, cm: step.cm });
  if (step.type === 'rotate')   return post('/proxy/rotate', { dir: step.dir, deg: step.deg });
  if (step.type === 'delay')    return new Promise(r => setTimeout(r, step.ms));
}

async function runMission() {
  const text = document.getElementById('mission_editor').value;
  let steps;
  try {
    steps = parseMission(text);
  } catch (e) {
    document.getElementById('mission_status').textContent = 'Parse error: ' + e.message;
    document.getElementById('mission_status').style.color = '#f87171';
    return;
  }
  if (steps.length === 0) {
    document.getElementById('mission_status').textContent = 'No steps found.';
    return;
  }

  const stepDelayMs = parseInt(document.getElementById('mission_delay').value) || 0;
  missionRunning = true;
  missionAbort = false;
  document.getElementById('mission_run').disabled = true;
  document.getElementById('mission_stop').disabled = false;

  const statusEl = document.getElementById('mission_status');
  statusEl.style.color = '#94a3b8';

  for (let i = 0; i < steps.length; i++) {
    if (missionAbort) break;
    const step = steps[i];
    statusEl.textContent = `Step ${i + 1}/${steps.length}: ${JSON.stringify(step)}`;
    statusEl.style.color = '#60a5fa';
    await runStep(step);
    if (missionAbort) break;
    if (stepDelayMs > 0 && step.type !== 'delay') {
      await new Promise(r => setTimeout(r, stepDelayMs));
    }
  }

  missionRunning = false;
  document.getElementById('mission_run').disabled = false;
  document.getElementById('mission_stop').disabled = true;
  if (missionAbort) {
    statusEl.textContent = 'Mission stopped.';
    statusEl.style.color = '#fb923c';
  } else {
    statusEl.textContent = `Done (${steps.length} steps).`;
    statusEl.style.color = '#4ade80';
  }
}

document.getElementById('mission_run').onclick = () => { if (!missionRunning) runMission(); };
document.getElementById('mission_stop').onclick = () => { missionAbort = true; };
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


def main():
    print("[REMOTE UI] Starting server...")
    print(f"[REMOTE UI] URL: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[REMOTE UI] PI_API_BASE={PI_BASE}")
    print(f"[REMOTE UI] timeouts: cmd={TIMEOUT_CMD}s status={TIMEOUT_STATUS}s")
    print("[REMOTE UI] Ready (waiting for browser requests)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
