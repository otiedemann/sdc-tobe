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

import datetime as _dt
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from .config import MissionConfig, tuning_view
from .controller import MissionController, MissionState


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
    <a href="/replay" class="{{ 'active' if active=='replay' else '' }}">Replay</a>
    <a href="/tune" class="{{ 'active' if active=='tune' else '' }}">Tune</a>
    <a href="/calibrate" class="{{ 'active' if active=='calibrate' else '' }}">Calibrate</a>
  </nav>
  <span style="margin-left:auto; font-size:.85rem; color:#aab;"
        id="phase">{{ header_label or 'phase: …' }}</span>
</header>
"""

# Reusable HTML fragments. The shared <script> below knows how to update
# whichever pieces are present on the page (status table, start button,
# chart canvases) so a page can include any subset of them.

_VIDEO_AND_STATUS_HTML = """
<section class="grid" style="grid-template-columns: minmax(0,2fr) minmax(0,1fr);">
  <div class="card">
    <h2>{{ camera_heading }}</h2>
    <img class="video" src="{{ video_url }}" alt="camera feed">
  </div>
  <div class="card">
    {% if mode == 'replay' %}
    <h2>Replay control</h2>
    <div id="replay-row" style="display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem;">
      <button id="btn-rp-toggle" type="button"
              style="padding:.5rem .9rem; border:0; border-radius:6px;
                     background:var(--accent); color:#062633; font-weight:600;
                     cursor:pointer; font-size:.95rem; min-width:5rem;">
        ▶ Play
      </button>
      <input type="range" id="rp-seek" min="0" max="100" value="0" step="0.1"
             style="flex:1; accent-color: var(--accent);">
      <span id="rp-time"
            style="font-variant-numeric:tabular-nums; font-size:.85rem;
                   color:#aab; min-width:7rem; text-align:right;">
        0.0 / 0.0 s
      </span>
    </div>
    <div id="replay-step-row"
         style="display:flex; align-items:center; gap:.3rem; margin-bottom:.5rem;
                font-size:.85rem; color:#aab;">
      Step:
      <button class="rp-step" data-step="-1.0" type="button">−1s</button>
      <button class="rp-step" data-step="-0.5" type="button">−0.5s</button>
      <button class="rp-step" data-step="-0.1" type="button">−0.1s</button>
      <button class="rp-step" data-step="0.1"  type="button">+0.1s</button>
      <button class="rp-step" data-step="0.5"  type="button">+0.5s</button>
      <button class="rp-step" data-step="1.0"  type="button">+1s</button>
    </div>
    <div id="replay-speed-row"
         style="display:flex; align-items:center; gap:.4rem; margin-bottom:.75rem;
                font-size:.85rem; color:#aab;">
      Speed:
      <button class="rp-speed" data-speed="0.1"  type="button">0.1x</button>
      <button class="rp-speed" data-speed="0.25" type="button">0.25x</button>
      <button class="rp-speed" data-speed="0.5"  type="button">0.5x</button>
      <button class="rp-speed" data-speed="1"    type="button">1x</button>
      <button class="rp-speed" data-speed="2"    type="button">2x</button>
      <button class="rp-speed" data-speed="4"    type="button">4x</button>
      <span style="margin-left:auto;">
        <a href="/replay" style="color:var(--accent); text-decoration:none;">
          ← all flights
        </a>
      </span>
    </div>
    {% else %}
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
    {% endif %}
    <h2>{{ 'Replayed flight' if mode == 'replay' else 'Mission status' }}</h2>
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
  <div class="card"><h2>RC commands (lr / fb / ud / yaw)</h2><canvas id="c-rc" width="700" height="200"></canvas></div>
</section>
"""

# One shared script that updates whatever DOM elements happen to exist on
# the current page. Lets `/` and `/charts` share a single state-fetch loop
# instead of duplicating it.
_SHARED_SCRIPT = """
<script>
const HISTORY_S = {{ history_s }};
const STATE_URL = {{ state_url|tojson }};
const REPLAY_ID = {{ replay_id|tojson }};
// Drone connection state. Initial value comes from the server's flag
// at page-load time; refresh() updates it each tick from /api/state's
// drone_connected field, so the Start button enables / disables live.
let droneConnected = {{ 'true' if drone_connected else 'false' }};
const $ = id => document.getElementById(id);
function fmt(v, unit, prec) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return v.toFixed(prec ?? 2) + (unit ? ' '+unit : '');
  return v;
}

// ---- Status panel (only runs if its DOM elements are present) -----------
const TERMINAL_PHASES = new Set(['done', 'abort']);
const STOPPABLE_PHASES = new Set(['takeoff','search','align','approach','hold','land']);
let stopRequested = false;
function setMissionButtons(phase) {
  const start = $('btn-start'); const stop = $('btn-stop'); const msg = $('ctrl-msg');
  if (!start || !stop) return;
  if (!droneConnected) {
    // No drone wired in -- the camera page is essentially a stub for
    // navigation to the Replay tab. Surface that clearly instead of
    // letting the user click Start and get an opaque server error.
    start.style.display = ''; stop.style.display = 'none';
    start.disabled = true; start.style.opacity = '0.5'; start.style.cursor = 'not-allowed';
    start.textContent = 'Start mission';
    msg.textContent = 'No drone connected — open the Replay tab to view flights.';
    return;
  }
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
  btn.disabled = true; msg.textContent = 'Starting…';
  try {
    const r = await fetch('/api/start', {method:'POST'});
    const j = await r.json();
    if (!j.ok) { msg.textContent = 'Could not start: ' + (j.error || 'unknown'); btn.disabled = false; }
  } catch (e) { msg.textContent = 'Request failed: ' + e; btn.disabled = false; }
}
async function stopMission() {
  const btn = $('btn-stop'); const msg = $('ctrl-msg');
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

// ---- Replay controls (only present in replay pages) ---------------------
let rpSeekDragging = false;
async function rpToggle() {
  if (!REPLAY_ID) return;
  await fetch(`/api/replay/${encodeURIComponent(REPLAY_ID)}/toggle`,
              {method:'POST'});
}
async function rpSeek(t) {
  if (!REPLAY_ID) return;
  const u = new URL(`/api/replay/${encodeURIComponent(REPLAY_ID)}/seek`,
                    location.origin);
  u.searchParams.set('t', String(t));
  await fetch(u, {method:'POST'});
}
async function rpSpeed(rate) {
  if (!REPLAY_ID) return;
  const u = new URL(`/api/replay/${encodeURIComponent(REPLAY_ID)}/speed`,
                    location.origin);
  u.searchParams.set('rate', String(rate));
  await fetch(u, {method:'POST'});
}
async function refreshReplayStatus() {
  if (!REPLAY_ID) return;
  try {
    const r = await fetch(
      `/api/replay/${encodeURIComponent(REPLAY_ID)}/status`,
      {cache:'no-store'});
    const rs = await r.json();
    const btn = $('btn-rp-toggle');
    if (btn) btn.textContent = rs.paused ? '▶ Play' : '⏸ Pause';
    const seek = $('rp-seek');
    if (seek && !rpSeekDragging) {
      seek.max = String(rs.duration_s);
      seek.value = String(rs.playhead_s);
    }
    const tlabel = $('rp-time');
    if (tlabel) tlabel.textContent =
      `${rs.playhead_s.toFixed(1)} / ${rs.duration_s.toFixed(1)} s  `
      + `(${rs.speed.toFixed(2)}x)`;
  } catch (e) {}
}
if (REPLAY_ID) {
  if ($('btn-rp-toggle')) $('btn-rp-toggle').addEventListener('click', rpToggle);
  const seek = $('rp-seek');
  if (seek) {
    seek.addEventListener('mousedown',  () => { rpSeekDragging = true; });
    seek.addEventListener('touchstart', () => { rpSeekDragging = true; });
    seek.addEventListener('change', e => {
      rpSeekDragging = false;
      rpSeek(parseFloat(e.target.value));
    });
  }
  document.querySelectorAll('.rp-speed').forEach(b => {
    b.addEventListener('click', () => rpSpeed(parseFloat(b.dataset.speed)));
  });
  document.querySelectorAll('.rp-step').forEach(b => {
    b.addEventListener('click', async () => {
      const dt = parseFloat(b.dataset.step);
      const seek = $('rp-seek');
      const cur = seek ? parseFloat(seek.value || '0') : 0;
      const dur = seek ? parseFloat(seek.max || '0') : 0;
      const next = Math.max(0, Math.min(dur, cur + dt));
      await rpSeek(next);
    });
  });
}

// ---- Charts (only runs if at least one canvas is present) ---------------
const buf = { t: [], d: [], y: [], h: [], drone_yaw: [], battery: [], height: [],
              rc_lr: [], rc_fb: [], rc_ud: [], rc_yaw: [] };
let chartPlayheadT = null;     // replay only: marks current position on the x axis
let chartPrefilled = false;    // replay only: timeline already loaded

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
  const rc = s.rc || {};
  buf.rc_lr.push(rc.lr);
  buf.rc_fb.push(rc.fb);
  buf.rc_ud.push(rc.ud);
  buf.rc_yaw.push(rc.yaw);
  while (buf.t.length && (now - buf.t[0]) > HISTORY_S) {
    buf.t.shift(); buf.d.shift(); buf.y.shift(); buf.h.shift();
    buf.drone_yaw.shift(); buf.battery.shift(); buf.height.shift();
    buf.rc_lr.shift(); buf.rc_fb.shift(); buf.rc_ud.shift(); buf.rc_yaw.shift();
  }
}
async function prefillReplayCharts() {
  if (!REPLAY_ID || chartPrefilled) return;
  try {
    const r = await fetch(
      `/api/replay/${encodeURIComponent(REPLAY_ID)}/timeline`,
      {cache: 'no-store'});
    const tl = await r.json();
    if (!tl || !Array.isArray(tl.t)) return;
    buf.t = tl.t.slice();
    buf.d = tl.d.slice();
    buf.y = tl.y.slice();
    buf.h = tl.h.slice();
    buf.drone_yaw = tl.drone_yaw.slice();
    buf.battery = tl.battery.slice();
    buf.height = tl.height.slice();
    buf.rc_lr  = (tl.rc_lr  || new Array(tl.t.length).fill(null)).slice();
    buf.rc_fb  = (tl.rc_fb  || new Array(tl.t.length).fill(null)).slice();
    buf.rc_ud  = (tl.rc_ud  || new Array(tl.t.length).fill(null)).slice();
    buf.rc_yaw = (tl.rc_yaw || new Array(tl.t.length).fill(null)).slice();
    chartPrefilled = true;
  } catch (e) {}
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
  // Replay playhead marker: vertical line at current time + dot on
  // each series at the value sampled at that time.
  if (chartPlayheadT !== null && chartPlayheadT >= t0 && chartPlayheadT <= t1) {
    const px = ((chartPlayheadT - t0)/span) * W;
    ctx.strokeStyle = '#facc15cc'; ctx.lineWidth = 1; ctx.setLineDash([2,3]);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    ctx.setLineDash([]);
    let lo_idx = 0, hi_idx = buf.t.length - 1;
    while (lo_idx < hi_idx) {
      const mid = (lo_idx + hi_idx) >> 1;
      if (buf.t[mid] < chartPlayheadT) lo_idx = mid + 1; else hi_idx = mid;
    }
    series.forEach(s => {
      const v = s.data[lo_idx];
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        const py = H - ((v - lo)/(hi-lo)) * H;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#0e1116'; ctx.font = 'bold 10px ui-sans-serif';
        ctx.fillText(v.toFixed(1), px + 6, py - 6);
      }
    });
  }
  ctx.fillStyle = '#aab'; ctx.font = '11px ui-sans-serif';
  ctx.fillText(hi.toFixed(2), W-50, 12);
  ctx.fillText(lo.toFixed(2), W-50, H-2);
}
const HAS_CHARTS = !!$('c-d');
function updateCharts(s) {
  if (!HAS_CHARTS) return;
  // Live mode: append new samples. Replay mode: buf is pre-filled
  // once from /api/replay/<id>/timeline; we just track the playhead
  // position so drawSeries can mark it.
  if (REPLAY_ID) {
    chartPlayheadT = (s.uptime_s !== undefined && s.uptime_s !== null)
                     ? s.uptime_s : null;
  } else {
    pushSample(s);
  }
  drawSeries('c-d', [{label:'distance', color:'#58c4ff', data:buf.d, legendX:0}], {target:s.target_distance_m});
  drawSeries('c-y', [{label:'yaw_to_marker', color:'#facc15', data:buf.y, legendX:0}], {target:0});
  drawSeries('c-h', [{label:'rel_heading', color:'#4ade80', data:buf.h, legendX:0}], {target:s.target_relative_heading_deg});
  drawSeries('c-t', [
    {label:'drone_yaw', color:'#f87171', data:buf.drone_yaw, legendX:0},
    {label:'battery%',  color:'#a78bfa', data:buf.battery,   legendX:2},
    {label:'h(cm)',     color:'#22d3ee', data:buf.height,    legendX:4},
  ]);
  drawSeries('c-rc', [
    {label:'rc_lr',  color:'#fb7185', data:buf.rc_lr,  legendX:0},
    {label:'rc_fb',  color:'#34d399', data:buf.rc_fb,  legendX:1},
    {label:'rc_ud',  color:'#60a5fa', data:buf.rc_ud,  legendX:2},
    {label:'rc_yaw', color:'#fbbf24', data:buf.rc_yaw, legendX:3},
  ], {target:0});
}

// ---- Single shared refresh loop -----------------------------------------
async function refresh() {
  try {
    if (REPLAY_ID && HAS_CHARTS && !chartPrefilled) {
      await prefillReplayCharts();
    }
    const r = await fetch(STATE_URL, {cache:'no-store'});
    const s = await r.json();
    if (typeof s.drone_connected === 'boolean') {
      droneConnected = s.drone_connected;
    }
    $('phase').textContent = (REPLAY_ID ? 'replay phase: ' : 'phase: ') + s.phase;
    updateStatus(s);
    updateCharts(s);
    if (REPLAY_ID) await refreshReplayStatus();
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

# Flight list -- one card per recorded flight, with a "Replay" button.
_PAGE_FLIGHTS = _PAGE_BASE_CSS + _PAGE_HEADER + """
<main>
  <div class="card">
    <h2>Recorded flights</h2>
    {% if flights %}
    <table style="width:100%;">
      <tr>
        <th>Date</th><th>Serial</th><th>Duration</th>
        <th>Final phase</th><th>&nbsp;</th>
      </tr>
      {% for f in flights %}
      <tr>
        <td>{{ f.date }}</td>
        <td>{{ f.serial }}</td>
        <td>{{ '%.1f s' % f.duration_s if f.duration_s else '—' }}</td>
        <td><span class="pill">{{ f.final_phase or '—' }}</span></td>
        <td style="text-align:right;">
          <a href="/replay/{{ f.flight_id }}"
             style="padding:.35rem .8rem; border-radius:6px;
                    background:var(--accent); color:#062633;
                    font-weight:600; text-decoration:none;">
            Replay →
          </a>
        </td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p style="color:#aab;">
      No flights recorded yet. Run a mission to populate
      <code>{{ flights_dir }}</code>.
    </p>
    {% endif %}
  </div>
</main>
"""

# Camera-calibration page. Live preview on the left, capture-then-run
# workflow on the right. The /api/calibrate/* endpoints below drive
# this page; the operator never needs to drop to the CLI to recalibrate.
_PAGE_CALIBRATE = _PAGE_BASE_CSS + _PAGE_HEADER + """
<main class="grid" style="grid-template-columns: minmax(0,2fr) minmax(0,1fr);">
  <div class="card">
    <h2>Live camera</h2>
    <img class="video" src="/video.mjpg" alt="camera feed">
  </div>
  <div class="card">
    <h2>Camera calibration</h2>
    <p style="font-size:.9rem; line-height:1.45;">
      Print a checkerboard onto A4 / Letter, glue it to flat rigid
      cardboard, count the <em>inner corners</em> (one fewer than the
      square count along each edge), and <em>measure one square edge
      with a ruler</em> in millimetres -- printers rarely scale at
      exactly 100&nbsp;%. Common patterns: a 9&times;6 inner-corner board
      ships with OpenCV (<a target="_blank" rel="noopener"
        href="https://github.com/opencv/opencv/raw/4.x/doc/pattern.png"
        style="color:var(--accent);">pattern.png</a>); 10&times;7 fills A4
      slightly better and is the default below.
      Power the drone, hold it stationary (motors&nbsp;off!), and let the
      camera see the board from many angles, distances (~0.5-2 m) and
      tilts. Aim for 30+ distinct views over 30-60 s of capture.
    </p>
    <div style="display:grid; grid-template-columns: auto auto 1fr;
                gap:.45rem .8rem; align-items:center; margin-top:.75rem;
                font-size:.9rem;">
      <label for="cal-square">Square edge:</label>
      <div>
        <input id="cal-square" type="number" step="0.1" value="25.0"
               style="width:5rem; background:#0c0f12; color:var(--fg);
                      border:1px solid #2a3038; border-radius:4px;
                      padding:.25rem .35rem;"> mm
      </div>
      <div style="display:flex; align-items:center; gap:.4rem;
                  justify-content:flex-end; color:#aab;">
        Drone:
        <input id="cal-serial" type="text" placeholder="serial / label"
               style="width:11rem; background:#0c0f12; color:var(--fg);
                      border:1px solid #2a3038; border-radius:4px;
                      padding:.25rem .35rem; font-family:inherit;">
      </div>

      <label for="cal-cols">Inner corners:</label>
      <div style="display:flex; align-items:center; gap:.3rem;">
        <input id="cal-cols" type="number" min="3" max="20" step="1" value="10"
               style="width:4rem; background:#0c0f12; color:var(--fg);
                      border:1px solid #2a3038; border-radius:4px;
                      padding:.25rem .35rem;">
        &times;
        <input id="cal-rows" type="number" min="3" max="20" step="1" value="7"
               style="width:4rem; background:#0c0f12; color:var(--fg);
                      border:1px solid #2a3038; border-radius:4px;
                      padding:.25rem .35rem;">
      </div>
      <span style="color:#aab; text-align:right;">cols &times; rows</span>
    </div>
    <div style="display:flex; align-items:center; gap:.5rem; margin-top:.75rem;">
      <button id="cal-start" type="button"
              style="padding:.45rem .8rem; border:0; border-radius:6px;
                     background:var(--good); color:#072413; font-weight:600;
                     cursor:pointer; font-size:.9rem;">
        Start capture
      </button>
      <button id="cal-stop" type="button"
              style="padding:.45rem .8rem; border:0; border-radius:6px;
                     background:var(--warn); color:#3a2e10; font-weight:600;
                     cursor:pointer; font-size:.9rem; display:none;">
        Stop capture
      </button>
      <button id="cal-run" type="button"
              style="padding:.45rem .8rem; border:0; border-radius:6px;
                     background:var(--accent); color:#062633; font-weight:600;
                     cursor:pointer; font-size:.9rem;" disabled>
        Run calibration
      </button>
    </div>
    <div id="cal-msg" style="margin-top:.75rem; font-size:.85rem; color:#aab;
                              min-height:1.2em;">Idle.</div>
    <pre id="cal-result"
         style="margin-top:.5rem; font-size:.8rem; color:var(--fg);
                background:#0c0f12; border:1px solid #2a3038;
                border-radius:6px; padding:.6rem; overflow-x:auto;
                display:none; white-space:pre;"></pre>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
// Seed the serial input from live telemetry IF the operator hasn't
// typed anything yet. The unified API server doesn't currently expose
// a serial number, so most of the time the operator types in their
// own label here -- it becomes the suffix in the saved .npz filename.
let serialSeeded = false;
async function refreshSerial() {
  try {
    const inp = $('cal-serial');
    if (!inp || serialSeeded || inp.value) { serialSeeded = true; return; }
    const s = await (await fetch('/api/state', {cache:'no-store'})).json();
    const sn = (s.telemetry || {}).serial_number;
    if (sn) { inp.value = sn; serialSeeded = true; }
  } catch(e) {}
}
async function calStatus() {
  try {
    const r = await fetch('/api/calibrate/status', {cache:'no-store'});
    const s = await r.json();
    const start = $('cal-start'); const stop = $('cal-stop');
    const run = $('cal-run'); const msg = $('cal-msg'); const result = $('cal-result');
    let label = '';
    switch (s.state) {
      case 'idle':
        start.style.display = ''; stop.style.display = 'none';
        start.disabled = false; run.disabled = true;
        label = 'Idle. Press Start capture and let the camera see the checkerboard.';
        break;
      case 'capturing':
        start.style.display = 'none'; stop.style.display = '';
        run.disabled = true;
        label = `Capturing ... ${s.frames_captured} frames recorded `
              + `(min ${s.min_frames}). Move the checkerboard around.`;
        break;
      case 'captured':
        start.style.display = ''; stop.style.display = 'none';
        start.disabled = false;
        run.disabled = !$('cal-serial').value.trim();
        label = `Captured ${s.frames_captured} frames. `
              + (run.disabled
                 ? 'Type a drone serial / label, then press Run.'
                 : 'Press Run calibration when ready.');
        break;
      case 'running':
        start.style.display = ''; stop.style.display = 'none';
        start.disabled = true; run.disabled = true;
        label = 'Running calibration ... can take 5-30 s.';
        break;
      case 'done':
        start.style.display = ''; stop.style.display = 'none';
        start.disabled = false; run.disabled = false;
        label = `Calibration saved to ${s.saved_path}.`;
        if (s.calibration) {
          result.style.display = '';
          const c = s.calibration;
          const rms = c.rms_error == null ? 'n/a' : c.rms_error.toFixed(3);
          result.textContent =
            `serial:        ${c.serial}\n`
          + `resolution:    ${c.resolution}\n`
          + `image_size:    ${c.image_size[0]} x ${c.image_size[1]}\n`
          + `rms_error:     ${rms} px ${c.rms_error != null && c.rms_error < 0.5 ? '(good)' : c.rms_error != null && c.rms_error < 1.0 ? '(ok)' : '(consider re-running with more views)'}\n`
          + `fx, fy:        ${c.fx.toFixed(2)}, ${c.fy.toFixed(2)}\n`
          + `cx, cy:        ${c.cx.toFixed(2)}, ${c.cy.toFixed(2)}\n`
          + `calibrated_at: ${c.calibrated_at}`;
        }
        break;
      case 'failed':
        start.style.display = ''; stop.style.display = 'none';
        start.disabled = false; run.disabled = !s.video_path;
        label = `Failed: ${s.error || 'unknown error'}.`;
        break;
    }
    msg.textContent = label;
  } catch(e) {}
}
async function calStart() {
  await fetch('/api/calibrate/start', {method:'POST'});
  calStatus();
}
async function calStop() {
  await fetch('/api/calibrate/stop', {method:'POST'});
  calStatus();
}
async function calRun() {
  const sq_mm = parseFloat($('cal-square').value);
  const cols = parseInt($('cal-cols').value, 10);
  const rows = parseInt($('cal-rows').value, 10);
  if (!Number.isFinite(sq_mm) || sq_mm <= 0) {
    $('cal-msg').textContent = 'Square edge must be a positive number (mm).';
    return;
  }
  if (!Number.isInteger(cols) || !Number.isInteger(rows)
      || cols < 3 || rows < 3) {
    $('cal-msg').textContent = 'Inner-corner counts must be integers >= 3.';
    return;
  }
  const sq_m = sq_mm / 1000.0;
  const serial = $('cal-serial').value.trim();
  if (!serial) {
    $('cal-msg').textContent = 'Type a drone serial / label first.';
    return;
  }
  const u = new URL('/api/calibrate/run', location.origin);
  u.searchParams.set('serial', serial);
  u.searchParams.set('square_size_m', String(sq_m));
  u.searchParams.set('pattern', `${cols}x${rows}`);
  const r = await fetch(u, {method:'POST'});
  const j = await r.json();
  if (!j.ok) $('cal-msg').textContent = `Could not start: ${j.error || 'unknown'}`;
  calStatus();
}
$('cal-start').addEventListener('click', calStart);
$('cal-stop').addEventListener('click', calStop);
$('cal-run').addEventListener('click', calRun);
async function tick() {
  await refreshSerial();
  await calStatus();
  setTimeout(tick, 500);
}
tick();
</script>
"""


# ---------------------------------------------------------------------------
# Tuning page. Lists every UI-exposed config field grouped by purpose;
# numeric inputs are pre-filled with the current value and show the
# dataclass default next to each row. Apply pushes the new values to
# the running controller (PD gains, output clamps, smoother) without a
# restart. Save / Reload / Reset operate on the persisted JSON file.
# ---------------------------------------------------------------------------

_PAGE_TUNE = _PAGE_BASE_CSS + _PAGE_HEADER + """
<main class="grid" style="grid-template-columns: minmax(0, 1fr);">
  <div class="card">
    <h2>Snapshots</h2>
    <p style="font-size:.85rem; color:#aab; margin:.2rem 0 .8rem;">
      Save the entire current parameter set under a name and load it
      back later. Useful for switching between tuning presets ("indoor
      slow", "outdoor fast", etc.). Save also captures any unsaved
      edits in the form below.
    </p>
    <div style="display:flex; gap:.4rem; align-items:center; margin-bottom:.7rem;">
      <input id="snap-name" type="text" placeholder="snapshot name"
             style="flex:1; min-width:10rem; max-width:24rem;
                    background:#0e1116; color:#e6edf3;
                    border:1px solid #2a3038; border-radius:4px;
                    padding:.3rem .5rem; font-family:inherit;">
      <button type="button" id="btn-snap-save"
              style="padding:.4rem .8rem; border:0; border-radius:6px;
                     background:var(--good); color:#072413; font-weight:600;
                     cursor:pointer;">Save snapshot</button>
    </div>
    <table id="snap-table" style="width:100%; border-collapse:collapse;
                                  font-size:.85rem;">
      <thead>
        <tr style="text-align:left; color:#9ca3af;">
          <th style="padding:.3rem .4rem;">Name</th>
          <th style="padding:.3rem .4rem;">Saved</th>
          <th style="padding:.3rem .4rem; width:14rem;">Actions</th>
        </tr>
      </thead>
      <tbody id="snap-body">
        <tr><td colspan="3" style="color:#6b7280; padding:.4rem;">loading…</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Live PD / mission tuning</h2>
    <p style="font-size:.85rem; color:#aab; margin:.2rem 0 1rem;">
      Edits below are pushed to the running controller automatically
      (~250 ms after you stop typing, or immediately on blur / Enter).
      <b>Apply all</b> force-syncs every field at once. <b>Save</b>
      persists the current values to
      <code>~/.marker_mission/config.json</code>. <b>Reload</b>
      re-reads that file. <b>Reset</b> reverts every field to the
      dataclass default.
    </p>
    <div id="tune-status" style="font-size:.85rem; color:#aab; margin-bottom:.6rem;"></div>
    <form id="tune-form">
      <div id="tune-groups"></div>
      <div style="display:flex; gap:.5rem; margin-top:1rem; flex-wrap:wrap;">
        <button type="button" id="btn-apply"
                style="padding:.5rem .9rem; border:0; border-radius:6px;
                       background:var(--accent); color:#062633; font-weight:600;
                       cursor:pointer;" title="Force-sync every field at
                       once (each field is also auto-applied as you edit it).">Apply all</button>
        <button type="button" id="btn-save"
                style="padding:.5rem .9rem; border:0; border-radius:6px;
                       background:var(--good); color:#072413; font-weight:600;
                       cursor:pointer;">Save to disk</button>
        <button type="button" id="btn-reload"
                style="padding:.5rem .9rem; border:0; border-radius:6px;
                       background:#334155; color:#e6edf3; cursor:pointer;">
          Reload from disk</button>
        <button type="button" id="btn-reset"
                style="padding:.5rem .9rem; border:0; border-radius:6px;
                       background:#7f1d1d; color:#fee2e2; cursor:pointer;">
          Reset all to defaults</button>
      </div>
    </form>
  </div>
</main>

<style>
  .tune-group { margin-top: 1.2rem; }
  .tune-group h3 { margin: 0 0 .35rem; font-size: 1rem;
                   color: var(--accent); }
  .tune-row { display: grid; grid-template-columns: 18rem 8rem 6rem 9rem 1fr;
              gap: .5rem; align-items: center;
              padding: .25rem 0;
              border-bottom: 1px dashed #2a3038;
              font-size: .9rem; }
  .tune-label { color: #e6edf3; }
  .tune-input { background: #0e1116; color: #e6edf3;
                border: 1px solid #2a3038; border-radius: 4px;
                padding: .25rem .4rem;
                font-variant-numeric: tabular-nums;
                font-family: inherit; }
  .tune-input.dirty { border-color: var(--accent); }
  .tune-default { color: #6b7280; font-size: .8rem;
                  font-variant-numeric: tabular-nums; }
  .tune-unit { color: #9ca3af; font-size: .8rem; }
  .tune-resetbtn { background: transparent; color: #6b7280;
                   border: 1px solid #2a3038; border-radius: 4px;
                   font-size: .75rem; padding: .15rem .4rem;
                   cursor: pointer; }
  .tune-resetbtn:hover { color: #e6edf3; border-color: #4b5563; }
  .tune-info { color: #6b7280; cursor: help; margin-left: .25rem;
               font-size: .85rem; user-select: none; }
  .tune-info:hover, .tune-info:focus { color: var(--accent); outline: none; }
</style>

<script>
const $ = id => document.getElementById(id);
function setStatus(msg, ok) {
  const el = $('tune-status');
  if (!el) return;
  el.textContent = msg;
  el.style.color = ok ? '#4ade80' : '#f87171';
}
let TUNE_FIELDS = {};   // name -> {kind, default}
let TUNE_DIRTY = new Set();

function renderGroups(view) {
  TUNE_FIELDS = {};
  TUNE_DIRTY = new Set();
  const root = $('tune-groups');
  root.innerHTML = '';
  for (const g of view.groups) {
    const div = document.createElement('div');
    div.className = 'tune-group';
    const h = document.createElement('h3');
    h.textContent = g.name;
    div.appendChild(h);
    for (const it of g.items) {
      TUNE_FIELDS[it.name] = {kind: it.kind, default: it.default};
      const row = document.createElement('div');
      row.className = 'tune-row';
      const desc = it.desc || '';
      const descAttr = desc.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
      const infoHtml = desc
        ? `<span class="tune-info" title="${descAttr}" tabindex="0">ⓘ</span>`
        : '';
      row.title = desc;
      row.innerHTML = `
        <span class="tune-label">${it.label} ${infoHtml}
          <span style="color:#6b7280; font-size:.75rem;">(${it.name})</span></span>
        <input class="tune-input" type="number" data-name="${it.name}"
               step="${it.step}" value="${it.value}"
               title="${descAttr}">
        <span class="tune-unit">${it.unit || ''}</span>
        <span class="tune-default">default: ${it.default}</span>
        <button type="button" class="tune-resetbtn" data-name="${it.name}">reset</button>
      `;
      div.appendChild(row);
    }
    root.appendChild(div);
  }
  // Wire input change-tracking, per-field reset, and auto-apply
  document.querySelectorAll('.tune-input').forEach(inp => {
    inp.addEventListener('input', () => {
      const name = inp.dataset.name;
      const def = String(TUNE_FIELDS[name].default);
      if (String(inp.value) !== def) inp.classList.add('dirty');
      else inp.classList.remove('dirty');
      TUNE_DIRTY.add(name);
      scheduleAutoApply(name);
    });
    // Also commit immediately on blur / Enter (cancels any pending
    // debounce so the user gets fast feedback when leaving the field).
    inp.addEventListener('change', () => applyFieldNow(inp.dataset.name));
  });
  document.querySelectorAll('.tune-resetbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.name;
      const inp = document.querySelector(`.tune-input[data-name="${name}"]`);
      if (!inp) return;
      inp.value = TUNE_FIELDS[name].default;
      inp.classList.remove('dirty');
      TUNE_DIRTY.add(name);
      applyFieldNow(name);
    });
  });
}

// ---- Live auto-apply (debounced per-field) -------------------------------
const APPLY_DEBOUNCE_MS = 250;
const applyTimers = new Map();   // field name -> setTimeout id

function scheduleAutoApply(name) {
  if (applyTimers.has(name)) clearTimeout(applyTimers.get(name));
  applyTimers.set(name, setTimeout(() => {
    applyTimers.delete(name);
    applyFieldNow(name);
  }, APPLY_DEBOUNCE_MS));
}
async function applyFieldNow(name) {
  if (applyTimers.has(name)) {
    clearTimeout(applyTimers.get(name));
    applyTimers.delete(name);
  }
  const inp = document.querySelector(`.tune-input[data-name="${name}"]`);
  if (!inp) return;
  const body = {}; body[name] = inp.value;
  try {
    const r = await fetch('/api/tune/apply', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.ok) {
      setStatus('Apply ' + name + ' failed: '
                + JSON.stringify(j.errors || j.error), false);
      return;
    }
    setStatus(name + ' = ' + inp.value + ' applied to running controller.', true);
  } catch (e) {
    setStatus('Apply ' + name + ' failed: ' + e, false);
  }
}

function collectValues() {
  const out = {};
  document.querySelectorAll('.tune-input').forEach(inp => {
    const name = inp.dataset.name;
    out[name] = inp.value;
  });
  return out;
}

async function loadView() {
  try {
    const r = await fetch('/api/tune', {cache: 'no-store'});
    const view = await r.json();
    renderGroups(view);
    setStatus('Loaded current values.', true);
  } catch (e) {
    setStatus('Could not load tuning data: ' + e, false);
  }
}
async function applyValues() {
  try {
    const r = await fetch('/api/tune/apply', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(collectValues()),
    });
    const j = await r.json();
    if (!j.ok) {
      setStatus('Apply failed: ' + JSON.stringify(j.errors || j.error), false);
      return;
    }
    document.querySelectorAll('.tune-input.dirty')
            .forEach(i => i.classList.remove('dirty'));
    TUNE_DIRTY.clear();
    setStatus('Applied to running controller.', true);
  } catch (e) {
    setStatus('Apply failed: ' + e, false);
  }
}
async function saveValues() {
  try {
    await applyValues();
    const r = await fetch('/api/tune/save', {method: 'POST'});
    const j = await r.json();
    if (j.ok) setStatus('Saved to ' + (j.path || 'config.json') + '.', true);
    else setStatus('Save failed: ' + (j.error || 'unknown'), false);
  } catch (e) {
    setStatus('Save failed: ' + e, false);
  }
}
async function reloadValues() {
  try {
    const r = await fetch('/api/tune/reload', {method: 'POST'});
    if (!r.ok) { setStatus('Reload failed', false); return; }
    await loadView();
    setStatus('Reloaded from disk.', true);
  } catch (e) {
    setStatus('Reload failed: ' + e, false);
  }
}
async function resetAll() {
  if (!confirm('Reset every tunable parameter to its default?')) return;
  try {
    const r = await fetch('/api/tune/reset', {method: 'POST'});
    if (!r.ok) { setStatus('Reset failed', false); return; }
    await loadView();
    setStatus('All values reset to defaults (still un-saved).', true);
  } catch (e) {
    setStatus('Reset failed: ' + e, false);
  }
}

$('btn-apply').addEventListener('click', applyValues);
$('btn-save').addEventListener('click', saveValues);
$('btn-reload').addEventListener('click', reloadValues);
$('btn-reset').addEventListener('click', resetAll);

// ---- Snapshots ---------------------------------------------------------
function fmtMtime(t) {
  if (!t) return '';
  const d = new Date(t * 1000);
  return d.toLocaleString();
}
async function refreshSnapshots() {
  try {
    const r = await fetch('/api/tune/snapshots', {cache: 'no-store'});
    const j = await r.json();
    const tb = $('snap-body');
    tb.innerHTML = '';
    if (!j.snapshots || j.snapshots.length === 0) {
      tb.innerHTML = '<tr><td colspan="3" style="color:#6b7280; padding:.4rem;">No snapshots yet.</td></tr>';
      return;
    }
    for (const s of j.snapshots) {
      const tr = document.createElement('tr');
      tr.style.borderTop = '1px dashed #2a3038';
      const nameTd = document.createElement('td');
      nameTd.style.padding = '.3rem .4rem';
      nameTd.style.color = '#e6edf3';
      nameTd.textContent = s.name;
      const timeTd = document.createElement('td');
      timeTd.style.padding = '.3rem .4rem';
      timeTd.style.color = '#9ca3af';
      timeTd.textContent = fmtMtime(s.mtime);
      const actTd = document.createElement('td');
      actTd.style.padding = '.3rem .4rem';
      const loadBtn = document.createElement('button');
      loadBtn.textContent = 'Load';
      loadBtn.type = 'button';
      loadBtn.style.cssText = 'padding:.2rem .55rem; border:0; border-radius:4px; '
        + 'background:var(--accent); color:#062633; font-weight:600; cursor:pointer; '
        + 'margin-right:.4rem; font-size:.8rem;';
      loadBtn.addEventListener('click', () => loadSnap(s.name));
      const overBtn = document.createElement('button');
      overBtn.textContent = 'Overwrite';
      overBtn.type = 'button';
      overBtn.title = 'Save the current form values into this snapshot, replacing it.';
      overBtn.style.cssText = 'padding:.2rem .55rem; border:1px solid #2a3038; '
        + 'border-radius:4px; background:#1f2937; color:#e6edf3; cursor:pointer; '
        + 'margin-right:.4rem; font-size:.8rem;';
      overBtn.addEventListener('click', () => overwriteSnap(s.name));
      const delBtn = document.createElement('button');
      delBtn.textContent = 'Delete';
      delBtn.type = 'button';
      delBtn.style.cssText = 'padding:.2rem .55rem; border:1px solid #7f1d1d; '
        + 'border-radius:4px; background:transparent; color:#fca5a5; cursor:pointer; '
        + 'font-size:.8rem;';
      delBtn.addEventListener('click', () => deleteSnap(s.name));
      actTd.appendChild(loadBtn);
      actTd.appendChild(overBtn);
      actTd.appendChild(delBtn);
      tr.appendChild(nameTd);
      tr.appendChild(timeTd);
      tr.appendChild(actTd);
      tb.appendChild(tr);
    }
  } catch (e) {
    setStatus('Could not list snapshots: ' + e, false);
  }
}
async function saveSnap() {
  const name = $('snap-name').value.trim();
  if (!name) { setStatus('Type a snapshot name first.', false); return; }
  try {
    const r = await fetch('/api/tune/snapshots/' + encodeURIComponent(name), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(collectValues()),
    });
    const j = await r.json();
    if (!j.ok) { setStatus('Snapshot save failed: ' + (j.error || 'unknown'), false); return; }
    setStatus('Snapshot "' + name + '" saved.', true);
    $('snap-name').value = '';
    document.querySelectorAll('.tune-input.dirty').forEach(i => i.classList.remove('dirty'));
    await refreshSnapshots();
  } catch (e) {
    setStatus('Snapshot save failed: ' + e, false);
  }
}
async function overwriteSnap(name) {
  try {
    const r = await fetch('/api/tune/snapshots/' + encodeURIComponent(name), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(collectValues()),
    });
    const j = await r.json();
    if (!j.ok) { setStatus('Overwrite failed: ' + (j.error || 'unknown'), false); return; }
    setStatus('Snapshot "' + name + '" overwritten with current values.', true);
    document.querySelectorAll('.tune-input.dirty').forEach(i => i.classList.remove('dirty'));
    await refreshSnapshots();
  } catch (e) {
    setStatus('Overwrite failed: ' + e, false);
  }
}
async function loadSnap(name) {
  try {
    const r = await fetch('/api/tune/snapshots/' + encodeURIComponent(name) + '/load',
                          {method: 'POST'});
    const j = await r.json();
    if (!j.ok) { setStatus('Load failed: ' + (j.error || 'unknown'), false); return; }
    await loadView();
    setStatus('Snapshot "' + name + '" loaded into running controller.', true);
  } catch (e) {
    setStatus('Load failed: ' + e, false);
  }
}
async function deleteSnap(name) {
  if (!confirm('Delete snapshot "' + name + '"?')) return;
  try {
    const r = await fetch('/api/tune/snapshots/' + encodeURIComponent(name),
                          {method: 'DELETE'});
    const j = await r.json();
    if (!j.ok) { setStatus('Delete failed: ' + (j.error || 'unknown'), false); return; }
    setStatus('Snapshot "' + name + '" deleted.', true);
    await refreshSnapshots();
  } catch (e) {
    setStatus('Delete failed: ' + e, false);
  }
}
$('btn-snap-save').addEventListener('click', saveSnap);
$('snap-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); saveSnap(); }
});

loadView();
refreshSnapshots();
</script>
"""


# ---------------------------------------------------------------------------
# UI server
# ---------------------------------------------------------------------------

class UiServer:
    def __init__(self, state: MissionState, latest_frame: LatestFrame,
                 host: str = "0.0.0.0", port: int = 8080,
                 history_s: float = 60.0,
                 on_start: Optional[Callable[[], bool]] = None,
                 on_stop: Optional[Callable[[], bool]] = None,
                 flights_root: Optional[Path] = None,
                 drone_connected: bool = True,
                 calibration_capture=None,
                 cfg: Optional[MissionConfig] = None,
                 controller: Optional[MissionController] = None,
                 flight_dir_provider: Optional[Callable[[], Optional[Path]]] = None):
        self.state = state
        self.frame = latest_frame
        self.host = host
        self.port = port
        self.history_s = history_s
        self.on_start = on_start
        self.on_stop = on_stop
        # Live-tuning page references. cfg is mutated in-place; calling
        # controller.apply_config_changes() resyncs the running PD
        # state machine to pick up the new values.
        self.cfg = cfg
        self.controller = controller
        # Used to write parameter_changes.csv next to flight_log.csv.
        # mission.py rolls flight_dir between missions, so we receive
        # a callable rather than a fixed Path.
        self.flight_dir_provider = flight_dir_provider
        self.flights_root = Path(flights_root) if flights_root else None
        # Drives the /calibrate page. May be None (e.g. view mode); the
        # routes return 503 in that case.
        self.calibration = calibration_capture
        # Whether a real drone is wired in. False -> Start button is
        # disabled in the camera page (replay still works fully).
        self.drone_connected = bool(drone_connected)
        # Lazy-created replay sessions, keyed by flight_id (= dir name).
        # We hold one shared session per flight: pause/seek/speed are
        # shared across browser tabs, but that's fine for our scale.
        self._replays: dict = {}
        self._replays_lock = threading.Lock()
        self.app = Flask(__name__)
        self._register_routes()
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------- parameter-change log
    def _log_param_changes(self, before: dict, source: str) -> None:
        """Append one row per actually-changed field to
        ``<flight_dir>/parameter_changes.csv``. ``before`` is a
        ``{field: pre-change-value}`` snapshot; the post-change value
        is read from ``self.cfg`` at call time. Silently no-ops if
        there is no current flight_dir, no cfg, or nothing changed."""
        if self.cfg is None or self.flight_dir_provider is None:
            return
        try:
            fdir = self.flight_dir_provider()
        except Exception:
            return
        if not fdir:
            return
        fdir = Path(fdir)
        if not fdir.exists():
            return
        rows = []
        wt = _dt.datetime.now().isoformat(timespec="milliseconds")
        mt = f"{time.monotonic():.4f}"
        for k, old in before.items():
            new = getattr(self.cfg, k, None)
            if old != new:
                rows.append((wt, mt, source, k,
                             "" if old is None else old,
                             "" if new is None else new))
        if not rows:
            return
        p = fdir / "parameter_changes.csv"
        new_file = not p.exists()
        with p.open("a") as fp:
            if new_file:
                fp.write("wall_time,monotonic,source,field,old_value,new_value\n")
            for r in rows:
                fp.write(",".join(str(x) for x in r) + "\n")

    # --------------------------------------------------------- replay glue
    def _get_or_create_replay(self, flight_id: str):
        from .replay import FlightReplay  # local: avoid import cycle
        if self.flights_root is None:
            return None
        # Reject any flight_id that tries to escape flights_root.
        target = (self.flights_root / flight_id).resolve()
        try:
            target.relative_to(self.flights_root.resolve())
        except ValueError:
            return None
        if not target.is_dir():
            return None
        with self._replays_lock:
            existing = self._replays.get(flight_id)
            if existing is not None:
                return existing
            try:
                rp = FlightReplay(target)
            except Exception as e:
                print(f"[ui] failed to load replay {flight_id}: {e}")
                return None
            self._replays[flight_id] = rp
            return rp

    def _list_flights(self) -> list:
        if self.flights_root is None or not self.flights_root.is_dir():
            return []
        out = []
        for d in sorted(self.flights_root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            entry = {"flight_id": d.name, "date": d.name,
                     "serial": "?", "duration_s": None,
                     "final_phase": None}
            meta_path = d / "mission_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    entry["serial"] = (meta.get("calibration") or {}
                                       ).get("serial") or "?"
                    entry["final_phase"] = (meta.get("outcome") or {}
                                             ).get("final_phase")
                except Exception:
                    pass
            csv_path = d / "flight_log.csv"
            if csv_path.exists():
                try:
                    # Just read the first and last lines for duration.
                    with csv_path.open() as f:
                        lines = f.readlines()
                    if len(lines) >= 3:
                        first = lines[1].split(",")
                        last = lines[-1].split(",")
                        entry["duration_s"] = (
                            float(last[1]) - float(first[1]))
                except Exception:
                    pass
            out.append(entry)
        return out

    def _register_routes(self) -> None:
        app = self.app

        live_ctx = dict(
            active="video",
            history_s=self.history_s,
            mode="live",
            state_url="/api/state",
            video_url="/video.mjpg",
            replay_id=None,
            camera_heading="Annotated camera view",
            header_label="phase: …",
            drone_connected=self.drone_connected,
        )

        @app.get("/")
        def index():
            return render_template_string(_PAGE_VIDEO, **live_ctx)

        @app.get("/charts")
        def charts():
            return render_template_string(_PAGE_CHARTS,
                                           **{**live_ctx, "active": "charts"})

        @app.get("/api/state")
        def api_state():
            snap = self.state.snapshot()
            # Surface the live drone-connection flag so the JS can
            # enable / disable the Start button without a page reload.
            snap["drone_connected"] = self.drone_connected
            return jsonify(snap)

        # ---- Replay --------------------------------------------------
        @app.get("/replay")
        def replay_index():
            flights = self._list_flights()
            return render_template_string(
                _PAGE_FLIGHTS, active="replay",
                history_s=self.history_s,
                mode="live",
                state_url="/api/state",
                video_url="/video.mjpg",
                replay_id=None,
                header_label="replay browser",
                flights=flights,
                flights_dir=str(self.flights_root) if self.flights_root else "",
                drone_connected=self.drone_connected,
            )

        @app.get("/replay/<flight_id>")
        def replay_view(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return ("Flight not found or could not be loaded", 404)
            return render_template_string(
                _PAGE_VIDEO,
                active="replay",
                history_s=self.history_s,
                mode="replay",
                state_url=f"/api/replay/{flight_id}/state",
                video_url=f"/replay/{flight_id}/video.mjpg",
                replay_id=flight_id,
                camera_heading=f"Replay: {flight_id}",
                header_label=f"replay: {flight_id}",
                drone_connected=self.drone_connected,
            )

        @app.get("/api/flights")
        def api_flights():
            return jsonify(self._list_flights())

        @app.get("/api/replay/<flight_id>/state")
        def api_replay_state(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"error": "not found"}), 404
            return jsonify(rp.snapshot())

        @app.get("/api/replay/<flight_id>/status")
        def api_replay_status(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"error": "not found"}), 404
            return jsonify(rp.status)

        @app.get("/api/replay/<flight_id>/timeline")
        def api_replay_timeline(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"error": "not found"}), 404
            return jsonify(rp.timeline_arrays())

        @app.post("/api/replay/<flight_id>/play")
        def api_replay_play(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"ok": False}), 404
            rp.play(); return jsonify({"ok": True})

        @app.post("/api/replay/<flight_id>/pause")
        def api_replay_pause(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"ok": False}), 404
            rp.pause(); return jsonify({"ok": True})

        @app.post("/api/replay/<flight_id>/toggle")
        def api_replay_toggle(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"ok": False}), 404
            rp.toggle_play(); return jsonify({"ok": True})

        @app.post("/api/replay/<flight_id>/seek")
        def api_replay_seek(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"ok": False}), 404
            from flask import request
            try:
                t = float(request.args.get("t", "0"))
            except ValueError:
                return jsonify({"ok": False, "error": "bad t"}), 400
            rp.seek(t); return jsonify({"ok": True})

        @app.post("/api/replay/<flight_id>/speed")
        def api_replay_speed(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return jsonify({"ok": False}), 404
            from flask import request
            try:
                rate = float(request.args.get("rate", "1"))
            except ValueError:
                return jsonify({"ok": False, "error": "bad rate"}), 400
            rp.set_speed(rate); return jsonify({"ok": True})

        @app.get("/replay/<flight_id>/video.mjpg")
        def replay_video(flight_id):
            rp = self._get_or_create_replay(flight_id)
            if rp is None:
                return ("flight not found", 404)
            def gen():
                while True:
                    jpg, ts = rp.frame.get()
                    if jpg:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg + b"\r\n")
                    time.sleep(0.05)
            return Response(gen(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        # ---- Calibration ----------------------------------------------
        @app.get("/calibrate")
        def calibrate_page():
            return render_template_string(
                _PAGE_CALIBRATE, active="calibrate",
                history_s=self.history_s,
                mode="live",
                state_url="/api/state",
                video_url="/video.mjpg",
                replay_id=None,
                header_label="calibration",
                drone_connected=self.drone_connected,
            )

        # ---- Tuning page + API -----------------------------------------
        @app.get("/tune")
        def tune_page():
            return render_template_string(
                _PAGE_TUNE, active="tune",
                history_s=self.history_s,
                header_label="live tuning",
                replay_id=None,
                mode="live",
            )

        @app.get("/api/tune")
        def api_tune_get():
            if self.cfg is None:
                return jsonify({"error": "tuning not wired"}), 503
            return jsonify(tuning_view(self.cfg))

        @app.post("/api/tune/apply")
        def api_tune_apply():
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            data = request.get_json(silent=True) or {}
            before = {k: getattr(self.cfg, k, None)
                      for k in data.keys()
                      if k in self.cfg.__dataclass_fields__}
            errors = self.cfg.update_from_dict(data)
            if self.controller is not None:
                try:
                    self.controller.apply_config_changes()
                except Exception as e:
                    return jsonify({"ok": False,
                                    "error": f"controller resync failed: {e}"})
            self._log_param_changes(before, source="ui-apply")
            if errors:
                return jsonify({"ok": False, "errors": errors})
            return jsonify({"ok": True})

        @app.post("/api/tune/save")
        def api_tune_save():
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            try:
                p = self.cfg.save()
                return jsonify({"ok": True, "path": str(p)})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/tune/reload")
        def api_tune_reload():
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            try:
                fresh = MissionConfig.load()
                before = {k: getattr(self.cfg, k, None)
                          for k in self.cfg.__dataclass_fields__}
                # Mutate the existing cfg in place so anything holding a
                # reference to it (the controller) sees the new values.
                self.cfg.update_from_dict({
                    k: getattr(fresh, k)
                    for k in self.cfg.__dataclass_fields__
                })
                if self.controller is not None:
                    self.controller.apply_config_changes()
                self._log_param_changes(before, source="ui-reload")
                return jsonify({"ok": True})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/tune/reset")
        def api_tune_reset():
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            try:
                defaults = MissionConfig()
                before = {k: getattr(self.cfg, k, None)
                          for k in self.cfg.__dataclass_fields__}
                self.cfg.update_from_dict({
                    k: getattr(defaults, k)
                    for k in self.cfg.__dataclass_fields__
                })
                if self.controller is not None:
                    self.controller.apply_config_changes()
                self._log_param_changes(before, source="ui-reset")
                return jsonify({"ok": True})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        # ---- Snapshot storage. JSON files in SNAPSHOTS_DIR; the file
        # stem is the snapshot name. Same format as the persisted
        # config (cfg.save writes asdict(cfg) JSON), so a snapshot is
        # just a named copy of the live cfg at one moment.
        import re as _re
        _NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\. ]{0,63}$")

        def _snap_path(name: str) -> Optional[Path]:
            if not _NAME_RE.match(name):
                return None
            from .config import SNAPSHOTS_DIR
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            return SNAPSHOTS_DIR / f"{name}.json"

        @app.get("/api/tune/snapshots")
        def api_snap_list():
            from .config import SNAPSHOTS_DIR
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            items = []
            for p in SNAPSHOTS_DIR.glob("*.json"):
                try:
                    items.append({
                        "name":  p.stem,
                        "mtime": p.stat().st_mtime,
                        "size":  p.stat().st_size,
                    })
                except OSError:
                    continue
            items.sort(key=lambda x: -x["mtime"])
            return jsonify({"snapshots": items})

        @app.post("/api/tune/snapshots/<name>")
        def api_snap_save(name):
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            # Apply any pending edits in the request body first so
            # "Save snapshot" implicitly captures the current form
            # state, not whatever was last applied.
            data = request.get_json(silent=True)
            if isinstance(data, dict) and data:
                before = {k: getattr(self.cfg, k, None)
                          for k in data.keys()
                          if k in self.cfg.__dataclass_fields__}
                self.cfg.update_from_dict(data)
                if self.controller is not None:
                    try: self.controller.apply_config_changes()
                    except Exception: pass
                self._log_param_changes(before, source=f"ui-snapshot-save:{name}")
            p = _snap_path(name)
            if p is None:
                return jsonify({"ok": False,
                                "error": "invalid name (alphanum, _ - . space, "
                                         "max 64 chars, can't start with separator)"}), 400
            try:
                self.cfg.save(p)
                return jsonify({"ok": True, "name": name, "path": str(p)})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/tune/snapshots/<name>/load")
        def api_snap_load(name):
            if self.cfg is None:
                return jsonify({"ok": False, "error": "tuning not wired"}), 503
            p = _snap_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False, "error": "snapshot not found"}), 404
            try:
                fresh = MissionConfig.load(p)
                before = {k: getattr(self.cfg, k, None)
                          for k in self.cfg.__dataclass_fields__}
                self.cfg.update_from_dict({
                    k: getattr(fresh, k)
                    for k in self.cfg.__dataclass_fields__
                })
                if self.controller is not None:
                    self.controller.apply_config_changes()
                self._log_param_changes(before, source=f"ui-snapshot-load:{name}")
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.delete("/api/tune/snapshots/<name>")
        def api_snap_delete(name):
            p = _snap_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False, "error": "snapshot not found"}), 404
            try:
                p.unlink()
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.get("/api/calibrate/status")
        def api_calibrate_status():
            if self.calibration is None:
                return jsonify({"state": "unavailable",
                                "error": "calibration capture not configured"
                                }), 503
            return jsonify(self.calibration.status)

        @app.post("/api/calibrate/start")
        def api_calibrate_start():
            if self.calibration is None:
                return jsonify({"ok": False,
                                "error": "not configured"}), 503
            ok = self.calibration.start_capture()
            if not ok:
                return jsonify({"ok": False,
                                "error": "already capturing or running"}), 409
            return jsonify({"ok": True})

        @app.post("/api/calibrate/stop")
        def api_calibrate_stop():
            if self.calibration is None:
                return jsonify({"ok": False,
                                "error": "not configured"}), 503
            ok = self.calibration.stop_capture()
            if not ok:
                return jsonify({"ok": False,
                                "error": "not currently capturing"}), 409
            return jsonify({"ok": True})

        @app.post("/api/calibrate/run")
        def api_calibrate_run():
            if self.calibration is None:
                return jsonify({"ok": False,
                                "error": "not configured"}), 503
            from flask import request
            serial = request.args.get("serial", "").strip()
            try:
                square_size_m = float(request.args.get("square_size_m", "0.025"))
            except ValueError:
                return jsonify({"ok": False,
                                "error": "bad square_size_m"}), 400
            try:
                pat_str = request.args.get("pattern", "10x7")
                cols, rows = (int(x) for x in pat_str.lower().split("x"))
                if cols < 3 or rows < 3:
                    raise ValueError("dims < 3")
                pattern = (cols, rows)
            except ValueError:
                return jsonify({"ok": False,
                                "error": "bad pattern (want COLSxROWS, "
                                         "each >= 3)"}), 400
            ok, msg = self.calibration.run_calibration(
                serial=serial, square_size_m=square_size_m, pattern=pattern)
            if not ok:
                return jsonify({"ok": False, "error": msg}), 409
            return jsonify({"ok": True, "msg": msg})

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
