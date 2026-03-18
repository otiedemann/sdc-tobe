import os
import requests
from flask import Flask, Response, jsonify, request

# Runs on remote PC. Proxies to Pi API server.
PI_BASE = os.getenv("PI_API_BASE", "http://192.168.179.62:8080").rstrip("/")
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8090
TIMEOUT = 2.0

app = Flask(__name__)
last_api_status = None

HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Tello Remote Controller</title>
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
  </style>
</head>
<body>
  <h2>Tello Remote Web Controller</h2>
  <div class=\"small\">Pi API: <span id=\"pi\"></span></div>
  <div class=\"small\">API status: <span id=\"api_status\">checking...</span></div>
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
      <div class=\"small\" style=\"margin-top:8px;\">Keyboard in browser: W/A/S/D R/F Q/E, T, L, Space stop</div>
    </div>

    <div class=\"panel\">
      <div><b>Telemetry</b></div>
      <div class=\"status-wrap\">
        <div class=\"meter\">
          <div class=\"meter-label\">Battery SoC: <span id=\"battery_val\">-</span></div>
          <div class=\"meter-track\"><div id=\"battery_bar\" class=\"meter-fill\"></div></div>
        </div>
        <div class=\"meter\">
          <div class=\"meter-label\">Wi‑Fi SNR: <span id=\"wifi_val\">-</span></div>
          <div class=\"meter-track\"><div id=\"wifi_bar\" class=\"meter-fill\"></div></div>
        </div>
      </div>
      <div id=\"telemetry\" class=\"small\" style=\"white-space:pre-wrap; margin-top:10px;\">loading...</div>
    </div>
  </div>
<script>
document.getElementById('pi').textContent = location.origin + ' -> proxy -> Pi';

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

const map = new Set(['w','a','s','d','q','e','r','f','t','l','x',' ']);
window.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  if (map.has(k)) {
    e.preventDefault();
    pressKey(k === ' ' ? 'space' : k);
  }
});
window.addEventListener('keyup', (e)=>{
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
  try {
    const r = await fetch('/proxy/telemetry');
    if (!r.ok) throw new Error('api_error');
    const t = await r.json();
    apiEl.textContent = 'connected';
    apiEl.style.color = '#22c55e';

    const battery = (typeof t.battery === 'number') ? t.battery : null;
    const wifiRaw = (typeof t.wifi_snr === 'number') ? t.wifi_snr : null;
    const wifiPct = (wifiRaw == null) ? null : Math.max(0, Math.min(100, ((wifiRaw + 100) / 100) * 100));
    setMeter('battery_bar', 'battery_val', battery, '%');
    setMeter('wifi_bar', 'wifi_val', wifiPct, '%');

    document.getElementById('telemetry').textContent =
      `battery: ${t.battery ?? '-'} %\n` +
      `temperature: ${t.temperature ?? '-'} °C\n` +
      `height: ${t.height_cm ?? '-'} cm\n` +
      `tof: ${t.tof_cm ?? '-'} cm\n` +
      `barometer: ${t.barometer_cm ?? '-'} cm\n` +
      `flight time: ${t.flight_time_s ?? '-'} s\n` +
      `wifi snr: ${t.wifi_snr ?? '-'}\n` +
      `attitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\n` +
      `velocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\n` +
      `accel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\n` +
      `flying: ${t.flying}\n` +
      `connected: ${t.connected}`;
  } catch {
    apiEl.textContent = 'disconnected';
    apiEl.style.color = '#ef4444';
    setMeter('battery_bar', 'battery_val', null);
    setMeter('wifi_bar', 'wifi_val', null);
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
setInterval(refreshTelemetry, 700);
setInterval(refreshLogStatus, 2000);
refreshTelemetry();
refreshLogStatus();
</script>
</body>
</html>
"""


def pi_post(path: str, body: dict | None = None):
    return requests.post(f"{PI_BASE}{path}", json=body or {}, timeout=TIMEOUT)


def pi_get(path: str):
    return requests.get(f"{PI_BASE}{path}", timeout=TIMEOUT)


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.post("/proxy/key_down")
def proxy_key_down():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/key_down", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/key_up")
def proxy_key_up():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/key_up", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/takeoff")
def proxy_takeoff():
    r = pi_post("/api/takeoff")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/land")
def proxy_land():
    r = pi_post("/api/land")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/flip")
def proxy_flip():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/flip", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/recover")
def proxy_recover():
    r = pi_post("/api/recover")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/logging/telemetry")
def proxy_log_status():
    r = pi_get("/api/logging/telemetry")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/logging/telemetry")
def proxy_log_config():
    data = request.get_json(silent=True) or {}
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
    r = pi_post("/api/logging/telemetry/clear")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/telemetry")
def proxy_telemetry():
    global last_api_status
    try:
        r = pi_get("/api/telemetry")
        ok = r.status_code < 500
        if ok != last_api_status:
            print("[REMOTE UI] Connected to Pi API" if ok else "[REMOTE UI] Pi API error")
            last_api_status = ok
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        if last_api_status is not False:
            print("[REMOTE UI] Disconnected from Pi API")
            last_api_status = False
        return jsonify(ok=False, error=str(e)), 502


def main():
    print(f"Remote UI: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"PI_API_BASE={PI_BASE}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
