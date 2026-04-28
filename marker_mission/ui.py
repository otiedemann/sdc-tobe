"""
Operator UI: two screens served by Flask.

* ``/``         -- live camera view with marker overlay and HUD
* ``/charts``   -- live charts of yaw, distance, relative heading, telemetry

Endpoints:

* ``GET /api/state``         -- single JSON snapshot of the mission state
* ``GET /api/state/stream``  -- Server-Sent Events at ~5 Hz
* ``GET /video.mjpg``        -- annotated MJPEG stream (one client at a time
                                is best, but multiple are OK -- each gets
                                the latest frame, no buffering)

The UI is intentionally minimal -- vanilla HTML + small <script>. No
build step, no external CDN dependencies at runtime (Chart.js is small
and inlined into the page bundle if the operator wants to run fully
offline).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

from .controller import MissionState


# ---------------------------------------------------------------------------
# Frame holder -- thread-safe latest annotated frame
# ---------------------------------------------------------------------------

class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpg: Optional[bytes] = None
        self._ts: float = 0.0

    def set(self, frame_bgr: np.ndarray, jpeg_quality: int = 80) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._jpg = buf.tobytes()
            self._ts = time.monotonic()

    def get(self) -> tuple[Optional[bytes], float]:
        with self._lock:
            return self._jpg, self._ts


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PAGE_BASE_CSS = """
<style>
  :root {
    --bg: #111418; --fg: #e6e6e6; --panel: #1a1f25; --accent: #58c4ff;
    --good: #4ade80; --warn: #facc15; --bad: #f87171;
    font-family: -apple-system, "SF Pro Text", "Segoe UI", Roboto, system-ui, sans-serif;
  }
  body { margin: 0; background: var(--bg); color: var(--fg); }
  header { display: flex; align-items: center; gap: 1.5rem;
           padding: .6rem 1rem; background: var(--panel);
           border-bottom: 1px solid #2a3038; }
  header h1 { margin: 0; font-size: 1rem; font-weight: 600; }
  header nav a { margin-right: 1rem; color: var(--accent);
                  text-decoration: none; font-size: .9rem; }
  header nav a.active { color: var(--fg); border-bottom: 2px solid var(--accent); }
  main { padding: 1rem; }
  .pill { display: inline-block; padding: .15rem .5rem; border-radius: 999px;
          font-size: .8rem; background: #2a3038; }
  .pill.good { background: #103620; color: var(--good); }
  .pill.warn { background: #3a2e10; color: var(--warn); }
  .pill.bad  { background: #3a1010; color: var(--bad); }
  .grid { display: grid; gap: 1rem; }
  .card { background: var(--panel); border-radius: 8px; padding: 1rem; }
  .card h2 { margin: 0 0 .5rem 0; font-size: .85rem; font-weight: 500;
             color: #aab; text-transform: uppercase; letter-spacing: .03em; }
  .big { font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  td, th { padding: .25rem .5rem; text-align: left; }
  th { color: #aab; font-weight: 500; }
  tr + tr td, tr + tr th { border-top: 1px solid #2a3038; }
  canvas { background: #0c0f12; border-radius: 6px; max-width: 100%; height: auto; }
  img.video { display: block; max-width: 100%; height: auto;
              background: #000; border-radius: 6px; }
</style>
"""

_PAGE_HEADER = """
<header>
  <h1>Marker mission</h1>
  <nav>
    <a href="/" class="{{ 'active' if active=='video' else '' }}">Camera</a>
    <a href="/charts" class="{{ 'active' if active=='charts' else '' }}">Charts</a>
  </nav>
  <span style="margin-left:auto; font-size:.85rem; color:#aab;"
        id="phase">phase: …</span>
</header>
"""

# Reusable HTML fragments. The shared <script> below knows how to update
# whichever pieces are present on the page (status table, start button,
# chart canvases) so a page can include any subset of them.

_VIDEO_AND_STATUS_HTML = """
<section class="grid" style="grid-template-columns: minmax(0,2fr) minmax(0,1fr);">
  <div class="card">
    <h2>Annotated camera view</h2>
    <img class="video" src="/video.mjpg" alt="camera feed">
  </div>
  <div class="card">
    <h2>Mission control</h2>
    <div id="ctrl-row" style="display:flex; align-items:center; gap:.75rem; margin-bottom:.75rem;">
      <button id="btn-start" type="button"
              style="padding:.55rem 1rem; border:0; border-radius:6px;
                     background:var(--good); color:#072413; font-weight:600;
                     cursor:pointer; font-size:.95rem;">
        Start mission
      </button>
      <button id="btn-stop" type="button"
              style="padding:.55rem 1rem; border:0; border-radius:6px;
                     background:var(--bad); color:#240707; font-weight:600;
                     cursor:pointer; font-size:.95rem; display:none;">
        Stop &amp; land
      </button>
      <span id="ctrl-msg" style="font-size:.85rem; color:#aab;">—</span>
    </div>
    <h2>Mission status</h2>
    <table id="status">
      <tr><th>Phase</th><td id="s-phase">—</td></tr>
      <tr><th>Phase age</th><td id="s-pa">—</td></tr>
      <tr><th>Distance</th><td id="s-d">—</td></tr>
      <tr><th>Yaw to marker</th><td id="s-y">—</td></tr>
      <tr><th>Rel. heading</th><td id="s-h">—</td></tr>
      <tr><th>Target distance</th><td id="s-td">—</td></tr>
      <tr><th>Target heading</th><td id="s-th">—</td></tr>
      <tr><th>Marker last seen</th><td id="s-ms">—</td></tr>
      <tr><th>RC (lr,fb,ud,yaw)</th><td id="s-rc">—</td></tr>
      <tr><th>Battery</th><td id="s-bat">—</td></tr>
      <tr><th>Drone yaw</th><td id="s-dy">—</td></tr>
      <tr><th>Height</th><td id="s-ht">—</td></tr>
      <tr><th>Flying</th><td id="s-fl">—</td></tr>
      <tr><th>Note</th><td id="s-note">—</td></tr>
    </table>
  </div>
</section>
"""

_CHARTS_HTML = """
<section class="grid" style="grid-template-columns: 1fr 1fr; margin-top: 1rem;">
  <div class="card"><h2>Distance to marker (m)</h2><canvas id="c-d" width="700" height="200"></canvas></div>
  <div class="card"><h2>Yaw to marker (°)</h2><canvas id="c-y" width="700" height="200"></canvas></div>
  <div class="card"><h2>Relative heading (°)</h2><canvas id="c-h" width="700" height="200"></canvas></div>
  <div class="card"><h2>Drone telemetry (yaw / battery / height)</h2><canvas id="c-t" width="700" height="200"></canvas></div>
</section>
"""

# One shared script that updates whatever DOM elements happen to exist on
# the current page. Lets `/` and `/charts` share a single state-fetch loop
# instead of duplicating it.
_SHARED_SCRIPT = """
<script>
const HISTORY_S = {{ history_s }};
const $ = id => document.getElementById(id);
function fmt(v, unit, prec) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return v.toFixed(prec ?? 2) + (unit ? ' '+unit : '');
  return v;
}

// ---- Status panel (only runs if its DOM elements are present) -----------
const TERMINAL_PHASES = new Set(['done', 'abort']);
const STOPPABLE_PHASES = new Set(['takeoff','search','approach','orbit','hold','land']);
let stopRequested = false;
function setMissionButtons(phase) {
  const start = $('btn-start'); const stop = $('btn-stop'); const msg = $('ctrl-msg');
  if (!start || !stop) return;
  if (phase === 'init') {
    start.style.display = ''; stop.style.display = 'none';
    start.disabled = false; start.style.opacity = '1'; start.style.cursor = 'pointer';
    start.textContent = 'Start mission';
    msg.textContent = 'Ready. Press to arm + take off.';
    stopRequested = false;
  } else if (TERMINAL_PHASES.has(phase)) {
    start.style.display = 'none'; stop.style.display = 'none';
    msg.textContent = 'Mission ' + phase + '.';
  } else if (STOPPABLE_PHASES.has(phase)) {
    start.style.display = 'none'; stop.style.display = '';
    if (stopRequested) {
      stop.disabled = true; stop.style.opacity = '0.5'; stop.style.cursor = 'not-allowed';
      stop.textContent = 'Landing…';
      msg.textContent = 'Stop requested. Phase: ' + phase;
    } else {
      stop.disabled = false; stop.style.opacity = '1'; stop.style.cursor = 'pointer';
      stop.textContent = 'Stop & land';
      msg.textContent = 'Phase: ' + phase;
    }
  } else {
    start.style.display = 'none'; stop.style.display = 'none';
    msg.textContent = 'Phase: ' + phase;
  }
}
async function startMission() {
  const btn = $('btn-start'); const msg = $('ctrl-msg');
  if (!confirm('Start the mission now? The drone will take off.')) return;
  btn.disabled = true; msg.textContent = 'Starting…';
  try {
    const r = await fetch('/api/start', {method:'POST'});
    const j = await r.json();
    if (!j.ok) { msg.textContent = 'Could not start: ' + (j.error || 'unknown'); btn.disabled = false; }
  } catch (e) { msg.textContent = 'Request failed: ' + e; btn.disabled = false; }
}
async function stopMission() {
  const btn = $('btn-stop'); const msg = $('ctrl-msg');
  if (!confirm('Stop the mission and land now?')) return;
  stopRequested = true;
  btn.disabled = true; btn.style.opacity = '0.5'; btn.textContent = 'Landing…';
  msg.textContent = 'Stop requested -- drone is landing.';
  try {
    const r = await fetch('/api/stop', {method:'POST'});
    const j = await r.json();
    if (!j.ok) {
      msg.textContent = 'Could not stop: ' + (j.error || 'unknown');
      stopRequested = false;
    }
  } catch (e) {
    msg.textContent = 'Request failed: ' + e;
    stopRequested = false;
  }
}
function updateStatus(s) {
  if (!$('s-phase')) return;
  $('s-phase').innerHTML = '<span class="pill">'+s.phase+'</span>';
  $('s-pa').textContent = fmt(s.phase_age_s, 's', 1);
  $('s-d').textContent  = fmt(s.distance_m, 'm', 2);
  $('s-y').textContent  = fmt(s.yaw_to_marker_deg, '°', 1);
  $('s-h').textContent  = fmt(s.relative_heading_deg, '°', 1);
  $('s-td').textContent = fmt(s.target_distance_m, 'm', 2);
  $('s-th').textContent = fmt(s.target_relative_heading_deg, '°', 1);
  $('s-ms').textContent = fmt(s.marker_seen_age_s, 's ago', 2);
  const rc = s.rc || {};
  $('s-rc').textContent = `${rc.lr ?? 0}, ${rc.fb ?? 0}, ${rc.ud ?? 0}, ${rc.yaw ?? 0}`;
  const t = s.telemetry || {};
  $('s-bat').textContent = (t.battery !== undefined && t.battery !== null) ? (t.battery + '%') : '—';
  $('s-dy').textContent  = fmt(t.yaw, '°', 1);
  $('s-ht').textContent  = (t.height_cm !== undefined && t.height_cm !== null) ? (t.height_cm + ' cm') : '—';
  $('s-fl').textContent  = t.flying ? 'yes' : 'no';
  $('s-note').textContent = s.note || (s.abort_reason || '—');
  setMissionButtons(s.phase);
}
if ($('btn-start')) $('btn-start').addEventListener('click', startMission);
if ($('btn-stop'))  $('btn-stop').addEventListener('click',  stopMission);

// ---- Charts (only runs if at least one canvas is present) ---------------
const buf = { t: [], d: [], y: [], h: [], drone_yaw: [], battery: [], height: [] };
function pushSample(s) {
  const now = performance.now()/1000;
  buf.t.push(now);
  buf.d.push(s.distance_m);
  buf.y.push(s.yaw_to_marker_deg);
  buf.h.push(s.relative_heading_deg);
  const t = s.telemetry || {};
  buf.drone_yaw.push(t.yaw);
  buf.battery.push(t.battery);
  buf.height.push(t.height_cm);
  while (buf.t.length && (now - buf.t[0]) > HISTORY_S) {
    buf.t.shift(); buf.d.shift(); buf.y.shift(); buf.h.shift();
    buf.drone_yaw.shift(); buf.battery.shift(); buf.height.shift();
  }
}
function drawSeries(canvasId, series, opts) {
  const c = $(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = '#2a3038'; ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W-1, H-1);
  if (buf.t.length < 2) return;
  const t0 = buf.t[0], t1 = buf.t[buf.t.length-1];
  const span = Math.max(0.001, t1 - t0);
  let lo = +Infinity, hi = -Infinity;
  series.forEach(s => {
    s.data.forEach(v => {
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    });
  });
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi-lo) * 0.1; lo -= pad; hi += pad;
  if (opts && opts.target !== undefined && opts.target !== null) {
    const ty = H - ((opts.target - lo)/(hi-lo)) * H;
    ctx.strokeStyle = '#facc1555'; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (lo < 0 && hi > 0) {
    const zy = H - ((0 - lo)/(hi-lo)) * H;
    ctx.strokeStyle = '#aab4'; ctx.beginPath();
    ctx.moveTo(0, zy); ctx.lineTo(W, zy); ctx.stroke();
  }
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    for (let i=0;i<buf.t.length;i++) {
      const v = s.data[i];
      if (v === null || v === undefined || !Number.isFinite(v)) { started = false; continue; }
      const x = ((buf.t[i]-t0)/span) * W;
      const y = H - ((v - lo)/(hi-lo)) * H;
      if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
    }
    ctx.stroke();
    ctx.fillStyle = s.color; ctx.font = '11px ui-sans-serif';
    ctx.fillText(s.label, 6 + s.legendX*60, 14);
  });
  ctx.fillStyle = '#aab'; ctx.font = '11px ui-sans-serif';
  ctx.fillText(hi.toFixed(2), W-50, 12);
  ctx.fillText(lo.toFixed(2), W-50, H-2);
}
const HAS_CHARTS = !!$('c-d');
function updateCharts(s) {
  if (!HAS_CHARTS) return;
  pushSample(s);
  drawSeries('c-d', [{label:'distance', color:'#58c4ff', data:buf.d, legendX:0}], {target:s.target_distance_m});
  drawSeries('c-y', [{label:'yaw_to_marker', color:'#facc15', data:buf.y, legendX:0}], {target:0});
  drawSeries('c-h', [{label:'rel_heading', color:'#4ade80', data:buf.h, legendX:0}], {target:s.target_relative_heading_deg});
  drawSeries('c-t', [
    {label:'drone_yaw', color:'#f87171', data:buf.drone_yaw, legendX:0},
    {label:'battery%',  color:'#a78bfa', data:buf.battery,   legendX:2},
    {label:'h(cm)',     color:'#22d3ee', data:buf.height,    legendX:4},
  ]);
}

// ---- Single shared refresh loop -----------------------------------------
async function refresh() {
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const s = await r.json();
    $('phase').textContent = 'phase: '+s.phase;
    updateStatus(s);
    updateCharts(s);
  } catch (e) {}
  setTimeout(refresh, 250);
}
refresh();
</script>
"""

_PAGE_VIDEO = (_PAGE_BASE_CSS + _PAGE_HEADER
               + "<main>" + _VIDEO_AND_STATUS_HTML + _CHARTS_HTML + "</main>"
               + _SHARED_SCRIPT)

_PAGE_CHARTS = (_PAGE_BASE_CSS + _PAGE_HEADER
                + "<main>" + _CHARTS_HTML + "</main>"
                + _SHARED_SCRIPT)


# ---------------------------------------------------------------------------
# UI server
# ---------------------------------------------------------------------------

class UiServer:
    def __init__(self, state: MissionState, latest_frame: LatestFrame,
                 host: str = "0.0.0.0", port: int = 8080,
                 history_s: float = 60.0,
                 on_start: Optional[Callable[[], bool]] = None,
                 on_stop: Optional[Callable[[], bool]] = None):
        self.state = state
        self.frame = latest_frame
        self.host = host
        self.port = port
        self.history_s = history_s
        self.on_start = on_start
        self.on_stop = on_stop
        self.app = Flask(__name__)
        self._register_routes()
        self._thread: Optional[threading.Thread] = None

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        def index():
            return render_template_string(_PAGE_VIDEO, active="video",
                                           history_s=self.history_s)

        @app.get("/charts")
        def charts():
            return render_template_string(_PAGE_CHARTS, active="charts",
                                           history_s=self.history_s)

        @app.get("/api/state")
        def api_state():
            return jsonify(self.state.snapshot())

        @app.post("/api/start")
        def api_start():
            if self.on_start is None:
                return jsonify({"ok": False,
                                "error": "no start handler registered"}), 500
            try:
                started = bool(self.on_start())
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            if not started:
                return jsonify({"ok": False,
                                "error": "mission already started or not in INIT"}), 409
            return jsonify({"ok": True})

        @app.post("/api/stop")
        def api_stop():
            if self.on_stop is None:
                return jsonify({"ok": False,
                                "error": "no stop handler registered"}), 500
            phase = self.state.snapshot().get("phase")
            if phase in ("init", "done", "abort"):
                return jsonify({"ok": False,
                                "error": f"mission not running (phase={phase})"}), 409
            try:
                self.on_stop()
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            return jsonify({"ok": True})

        @app.get("/video.mjpg")
        def video():
            def gen():
                while True:
                    jpg, ts = self.frame.get()
                    if jpg:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg + b"\r\n")
                    time.sleep(0.05)
            return Response(gen(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        def run():
            # Disable Flask's request log spam
            import logging
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            self.app.run(host=self.host, port=self.port,
                         threaded=True, use_reloader=False)
        self._thread = threading.Thread(target=run, daemon=True, name="ui-server")
        self._thread.start()

    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"
