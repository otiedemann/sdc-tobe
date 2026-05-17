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
from flask import (Flask, Response, jsonify, render_template_string,
                    request, send_file)

from .config import CALIB_DIR, MissionConfig, tuning_view
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
  /* Hover-tooltip info icon, used wherever a control needs an
     explanation (originally /tune; now also /calibrate camera config). */
  .tune-info { color: #6b7280; cursor: help; margin-left: .25rem;
               font-size: .85rem; user-select: none; }
  .tune-info:hover, .tune-info:focus { color: var(--accent); outline: none; }
</style>
"""

_PAGE_HEADER = """
<header>
  <img src="/team_logo.png" alt="team logo"
       style="width:120px; height:120px; object-fit:contain;
              display:block; margin-right:.8rem;">
  <h1 style="line-height:1.15;">to be defined<br>mission control</h1>
  <nav>
    <a href="/mission" class="{{ 'active' if active=='video' else '' }}">Camera</a>
    <a href="/charts" class="{{ 'active' if active=='charts' else '' }}">Charts</a>
    <a href="/replay" class="{{ 'active' if active=='replay' else '' }}">Replay</a>
    <a href="/tune" class="{{ 'active' if active=='tune' else '' }}">Tune</a>
    <a href="/arena" class="{{ 'active' if active=='arena' else '' }}">Arena</a>
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
    <div style="display:flex; align-items:center; justify-content:space-between;
                gap:.5rem; margin-bottom:.5rem;">
      <h2 style="margin:0;">{{ camera_heading }}</h2>
      {% if mode != 'replay' %}
      <button id="btn-video-restart" type="button" title="Restart camera stream"
              style="padding:.3rem .65rem; border:1px solid #2a3038;
                     border-radius:5px; background:#0c0f12; color:var(--fg);
                     cursor:pointer; font-size:.8rem;">
        ↻ Restart camera
      </button>
      {% endif %}
    </div>
    <img class="video" src="{{ video_url }}" alt="camera feed">
    <span id="video-restart-msg" style="font-size:.75rem; color:#aab;
                                         display:block; min-height:1rem;
                                         margin-top:.25rem;"></span>
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
    <h2>Mission script</h2>
    <div id="script-runtime"
         style="background:#0c0f12; border:1px solid #2a3038;
                border-radius:6px; padding:.4rem; font-family:ui-monospace,
                SFMono-Regular, Menlo, Consolas, monospace;
                font-size:.85rem; line-height:1.35; margin-bottom:.6rem;
                color:#778;">No mission script recorded for this flight.</div>
    {% else %}
    <h2>Mission control</h2>
    <div id="script-row" style="margin-bottom:.5rem;">
      <textarea id="script-text" rows="8" spellcheck="false"
                placeholder="TAKEOFF&#10;APPROACH 4 1.0&#10;HOOVER 3&#10;LAND"
                style="width:100%; box-sizing:border-box;
                       background:#0c0f12; color:var(--fg);
                       border:1px solid #2a3038; border-radius:6px;
                       padding:.5rem; font-family:ui-monospace, SFMono-Regular,
                       Menlo, Consolas, monospace;
                       font-size:.85rem; line-height:1.35; resize:vertical;"
                ></textarea>
      <div id="script-runtime"
           style="display:none; background:#0c0f12; border:1px solid #2a3038;
                  border-radius:6px; padding:.4rem; font-family:ui-monospace,
                  SFMono-Regular, Menlo, Consolas, monospace;
                  font-size:.85rem; line-height:1.35;"></div>
      <div id="script-err" style="font-size:.8rem; color:var(--bad);
                                  min-height:1.1rem; margin-top:.25rem;"></div>
      <div id="script-save-status" style="font-size:.75rem; color:#778;"></div>
    </div>
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
    <details id="script-saves-card"
             style="margin-top:.5rem; border-top:1px solid #2a3038; padding-top:.5rem;">
      <summary style="cursor:pointer; font-size:.85rem; color:#aab;">
        Saved scripts
      </summary>
      <div style="display:flex; gap:.4rem; align-items:center; margin:.5rem 0;">
        <input id="script-save-name" type="text" placeholder="name"
               style="flex:1; padding:.35rem .5rem; background:#0c0f12;
                      color:var(--fg); border:1px solid #2a3038; border-radius:5px;
                      font-size:.85rem;">
        <button id="btn-script-save" type="button"
                style="padding:.35rem .7rem; border:0; border-radius:5px;
                       background:var(--accent); color:#062633; font-weight:600;
                       cursor:pointer; font-size:.85rem;">
          Save script
        </button>
      </div>
      <div id="script-default-banner" style="display:none;"></div>
      <div id="script-list" style="font-size:.85rem;"></div>
    </details>
    {% endif %}
    <h2>Position</h2>
    <div style="display:flex; gap:.6rem; align-items:center;
                margin-bottom:.6rem;">
      <canvas id="c-pos" width="240" height="240"
              style="background:#0c0f12; border-radius:6px;
                     flex:0 0 auto;"></canvas>
      <div id="c-pos-text" style="font-size:.8rem; color:#aab;
                                    font-variant-numeric:tabular-nums;
                                    line-height:1.5;">—</div>
    </div>
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
      <tr><th>Target height</th><td id="s-ht-tgt">—</td></tr>
      <tr><th>World position</th><td id="s-wp">—</td></tr>
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


# Compact right-side sidebar included on every non-Camera page so the
# operator never loses eyes-on telemetry / kill button when switching
# tabs. Uses the same DOM ids as the Camera page (s-phase, s-bat, etc.)
# so _SHARED_SCRIPT's updateStatus() / updateScriptRuntime() / button
# handlers light it up unchanged. The Camera page omits this fragment
# because its main right card already provides everything (plus the
# script editor + named-saves disclosure).
_COMPACT_SIDEBAR_HTML = """
<aside class="card" style="position:sticky; top:.6rem; align-self:start;">
  <h2>Live</h2>
  <img class="mini-video" src="/video.mjpg" alt="camera"
       style="width:100%; max-width:280px; display:block; border-radius:6px;
              background:#000; margin-bottom:.6rem;">

  <h2>Mission control</h2>
  <div id="ctrl-row" style="display:flex; align-items:center;
                            gap:.5rem; margin-bottom:.5rem;">
    <button id="btn-start" type="button"
            style="padding:.5rem .8rem; border:0; border-radius:6px;
                   background:var(--good); color:#072413; font-weight:600;
                   cursor:pointer; font-size:.9rem;">
      Start
    </button>
    <button id="btn-stop" type="button"
            style="padding:.5rem .8rem; border:0; border-radius:6px;
                   background:var(--bad); color:#240707; font-weight:600;
                   cursor:pointer; font-size:.9rem; display:none;">
      Stop &amp; land
    </button>
    <span id="ctrl-msg" style="font-size:.8rem; color:#aab;">—</span>
  </div>

  <div id="script-runtime"
       style="background:#0c0f12; border:1px solid #2a3038;
              border-radius:6px; padding:.4rem; font-family:ui-monospace,
              SFMono-Regular, Menlo, Consolas, monospace;
              font-size:.8rem; line-height:1.35; margin-bottom:.6rem;
              color:#778; max-height:180px; overflow-y:auto;
              display:none;">No script.</div>

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
    <tr><th>Target height</th><td id="s-ht-tgt">—</td></tr>
    <tr><th>World position</th><td id="s-wp">—</td></tr>
    <tr><th>Flying</th><td id="s-fl">—</td></tr>
    <tr><th>Note</th><td id="s-note">—</td></tr>
  </table>
</aside>
"""

# Open / close strings for non-Camera pages: 2-col grid with the
# existing page content on the left and _COMPACT_SIDEBAR_HTML on the
# right. Each non-Camera page template wraps its body between these.
_PAGE_GRID_OPEN = (
    '<main style="display:grid; gap:1rem; align-items:start;'
    ' grid-template-columns:minmax(0,1fr) 320px;">'
    '<div>'
)
_PAGE_GRID_CLOSE = '</div>' + _COMPACT_SIDEBAR_HTML + '</main>'


# Tiny script that runs FIRST on every page. Defines the killswitch
# keydown listener (binds the configured cfg.killswitch_key, fires
# /api/stop, shows a banner) and a Web Audio beep helper used by the
# phase-change notifier in _SHARED_SCRIPT below. Kept separate from
# _SHARED_SCRIPT so it can run on pages whose other JS isn't yet ready
# (e.g. the operator hits the killswitch the moment the page loads).
_COMMON_SCRIPT = """
<script>
const KILLSWITCH_KEY = {{ killswitch_key|tojson }};
let _audioCtx = null;
function _beep(freqHz, durMs) {
  try {
    _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (_audioCtx.state === 'suspended') _audioCtx.resume();
    const o = _audioCtx.createOscillator();
    const g = _audioCtx.createGain();
    o.frequency.value = freqHz; o.type = 'sine';
    o.connect(g); g.connect(_audioCtx.destination);
    const t0 = _audioCtx.currentTime;
    g.gain.setValueAtTime(0.18, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + durMs / 1000);
    o.start(t0); o.stop(t0 + durMs / 1000 + 0.05);
  } catch (e) {}
}
function _killswitchBanner() {
  const banner = document.createElement('div');
  banner.textContent = 'KILLSWITCH FIRED \\u2014 landing';
  banner.style.cssText = 'position:fixed; top:0; left:0; right:0; z-index:9999;'
                      + ' background:#f87171; color:#240707;'
                      + ' text-align:center; padding:.6rem;'
                      + ' font-weight:700; font-size:1rem;';
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 4000);
}
document.addEventListener('keydown', e => {
  if (!KILLSWITCH_KEY) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;     // browser shortcuts win
  if (String(e.key || '').toUpperCase()
      !== String(KILLSWITCH_KEY).toUpperCase()) return;
  // Don't fire when the operator is editing the killswitch_key tune
  // input itself -- otherwise every attempt to overwrite the current
  // key would trigger the killswitch instead of typing a new value.
  const t = e.target;
  if (t && t.classList && t.classList.contains('tune-input')
      && t.dataset && t.dataset.name === 'killswitch_key') return;
  // Fire the killswitch but DON'T preventDefault: the operator may
  // also need that character to land in an input (e.g. typing "0"
  // into a number field while killswitch_key="0"). Letting the
  // keystroke through means the killswitch is purely additive.
  _killswitchBanner();
  _beep(220, 350);                                    // distinct lower tone
  fetch('/api/stop', {method: 'POST'}).catch(() => {});
});
</script>
"""

# One shared script that updates whatever DOM elements happen to exist on
# the current page. Lets `/` and `/charts` share a single state-fetch loop
# instead of duplicating it. Now included on every page (not just / and
# /charts) so the universal sidebar's status table + Start/Stop + script
# runtime work everywhere.
_SHARED_SCRIPT = """
<script>
const HISTORY_S = {{ history_s }};
const STATE_URL = {{ state_url|tojson }};
const REPLAY_ID = {{ replay_id|tojson }};
// Drone connection state. Initial value comes from the server's flag
// at page-load time; refresh() updates it each tick from /api/state's
// drone_connected field, so the Start button enables / disables live.
let droneConnected = {{ 'true' if drone_connected else 'false' }};
// Server-rendered initial phase. Lets setMissionButtons() paint the
// correct Start vs Stop state on the very first frame after a refresh
// during a flight, instead of flashing "Start mission" until the first
// /api/state poll lands ~250ms later. Falls back to "init" on pages
// (replay browser, replay view) that don't supply it.
const INITIAL_PHASE = {{ (initial_phase if initial_phase is defined else 'init')|tojson }};
const $ = id => document.getElementById(id);

// Wall colour palette, identical to the /arena tab's WALL_COLORS.
// Used by the position view (drawPositionView) and the arena page;
// hoisted to shared scope so both surfaces stay in sync.
const WALL_COLORS = {front:'#facc15', right:'#4ade80',
                      back:'#58c4ff',  left:'#f87171'};

// ---- Default-pointer helpers (used by tune / arena / scripts / camera
// save tables) -----------------------------------------------------------
function appendDefaultBadge(td) {
  const badge = document.createElement('span');
  badge.textContent = 'default';
  badge.style.cssText = 'background:#16a34a; color:#062613; '
    + 'font-size:.7rem; font-weight:600; padding:.05rem .35rem; '
    + 'border-radius:3px; margin-left:.4rem; vertical-align:middle;';
  td.appendChild(badge);
}
function makeSetDefaultButton(isCurrentlyDefault, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = isCurrentlyDefault ? 'Clear default' : 'Set as default';
  b.title = isCurrentlyDefault
    ? 'Clear this row as the auto-loaded default.'
    : 'Make this the default that auto-loads at startup.';
  b.style.cssText = 'padding:.2rem .55rem; border:1px solid #2a3038; '
    + 'border-radius:4px; background:#1f2937; color:#e6edf3; '
    + 'cursor:pointer; margin-right:.4rem; font-size:.8rem;';
  b.addEventListener('click', onClick);
  return b;
}
async function clearDefault(listUrl, onDone) {
  // POST set-default with clear=true against any existing name in the
  // list works as a no-op when no rows match (server returns 404).
  // Easier path: hit the bare collection URL with a DELETE-style body.
  // Each subsystem's set-default endpoint already accepts {clear:true}
  // even for a name that no longer exists -- it just clears the pointer.
  // We discover the stale name from the list response.
  try {
    const r0 = await fetch(listUrl, {cache:'no-store'});
    const j0 = await r0.json();
    const stale = j0.default;
    if (!stale) { if (onDone) await onDone(); return; }
    // URL convention: <listUrl>/<name>/set-default
    await fetch(listUrl + '/' + encodeURIComponent(stale) + '/set-default',
                {method:'POST',
                 headers:{'Content-Type':'application/json'},
                 body: JSON.stringify({clear: true})});
  } catch (e) {}
  if (onDone) await onDone();
}
function renderDefaultMissingBanner(elementId, defaultName, exists, onClear) {
  let el = $(elementId);
  if (!el) return;
  if (defaultName && !exists) {
    el.style.cssText = 'padding:.4rem .55rem; margin-bottom:.4rem; '
      + 'border:1px solid #7f1d1d; border-radius:4px; '
      + 'background:rgba(127,29,29,0.18); color:#fca5a5; '
      + 'font-size:.8rem;';
    el.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = 'Default points at "' + defaultName
      + '" but that save no longer exists. ';
    el.appendChild(span);
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = 'Clear default';
    b.style.cssText = 'padding:.15rem .5rem; border:1px solid #7f1d1d; '
      + 'border-radius:4px; background:transparent; color:#fca5a5; '
      + 'cursor:pointer; font-size:.75rem;';
    b.addEventListener('click', onClear);
    el.appendChild(b);
  } else {
    el.style.display = 'none';
    el.innerHTML = '';
  }
}
function fmt(v, unit, prec) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return v.toFixed(prec ?? 2) + (unit ? ' '+unit : '');
  return v;
}

// ---- Status panel (only runs if its DOM elements are present) -----------
const TERMINAL_PHASES = new Set(['done', 'abort']);
const STOPPABLE_PHASES = new Set(['takeoff','search','align','height_align',
                                   'approach','hold','idle','height','dance',
                                   'land']);
let stopRequested = false;
function setScriptLocked(locked) {
  const ta = $('script-text');
  if (!ta) return;
  ta.readOnly = !!locked;
  ta.style.opacity = locked ? '0.55' : '1';
}
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
    setScriptLocked(true);
    return;
  }
  if (phase === 'init') {
    start.style.display = ''; stop.style.display = 'none';
    start.disabled = false; start.style.opacity = '1'; start.style.cursor = 'pointer';
    start.textContent = 'Start mission';
    msg.textContent = 'Ready. Press to arm + take off.';
    stopRequested = false;
    setScriptLocked(false);
  } else if (TERMINAL_PHASES.has(phase)) {
    start.style.display = 'none'; stop.style.display = 'none';
    msg.textContent = 'Mission ' + phase + '.';
    setScriptLocked(true);
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
    setScriptLocked(true);
  } else {
    start.style.display = 'none'; stop.style.display = 'none';
    msg.textContent = 'Phase: ' + phase;
    setScriptLocked(true);
  }
}
async function startMission() {
  const btn = $('btn-start'); const msg = $('ctrl-msg');
  const ta = $('script-text');
  // Flush any pending autosave by sending the current text in the
  // start request. Server parses + installs + triggers atomically;
  // a parse error returns ok=false with a line-numbered message.
  btn.disabled = true; msg.textContent = 'Starting…';
  setScriptError('');
  try {
    const body = ta ? JSON.stringify({script: ta.value}) : '{}';
    const r = await fetch('/api/start',
                          {method:'POST',
                           headers:{'Content-Type':'application/json'},
                           body: body});
    const j = await r.json();
    if (!j.ok) {
      msg.textContent = 'Could not start: ' + (j.error || 'unknown');
      btn.disabled = false;
      if (j.script_error) setScriptError(j.script_error);
    }
  } catch (e) { msg.textContent = 'Request failed: ' + e; btn.disabled = false; }
}

// ---- Mission-script textarea & saved-scripts list -----------------------
function setScriptError(s) {
  const el = $('script-err'); if (el) el.textContent = s || '';
}
function setScriptSaveStatus(s) {
  const el = $('script-save-status'); if (el) el.textContent = s || '';
}
let scriptSaveTimer = null;
let scriptLastSaved = null;
async function flushScriptDraft() {
  const ta = $('script-text'); if (!ta) return;
  const txt = ta.value;
  if (txt === scriptLastSaved) return;
  try {
    const r = await fetch('/api/mission/script',
                          {method:'POST',
                           headers:{'Content-Type':'application/json'},
                           body: JSON.stringify({text: txt})});
    if (r.ok) {
      scriptLastSaved = txt;
      setScriptSaveStatus('Saved.');
    } else {
      setScriptSaveStatus('Save failed: HTTP ' + r.status);
    }
  } catch (e) { setScriptSaveStatus('Save failed: ' + e); }
}
function scheduleScriptSave() {
  if (scriptSaveTimer) clearTimeout(scriptSaveTimer);
  scriptSaveTimer = setTimeout(flushScriptDraft, 250);
}
async function loadInitialScript() {
  const ta = $('script-text'); if (!ta) return;
  try {
    const r = await fetch('/api/mission/script', {cache:'no-store'});
    const j = await r.json();
    if (typeof j.text === 'string') {
      ta.value = j.text;
      scriptLastSaved = j.text;
    }
  } catch (e) {}
}
async function refreshScriptList() {
  const list = $('script-list'); if (!list) return;
  try {
    const r = await fetch('/api/mission/scripts', {cache:'no-store'});
    const j = await r.json();
    renderDefaultMissingBanner('script-default-banner', j.default,
      (j.scripts || []).some(x => x.name === j.default),
      () => clearDefault('/api/mission/scripts', refreshScriptList));
    if (!j.scripts || !j.scripts.length) {
      list.innerHTML = '<div style="color:#778;">No saved scripts.</div>';
      return;
    }
    const fmtTime = mt => new Date(mt*1000).toLocaleString();
    list.innerHTML = j.scripts.map(s => {
      const safe = s.name.replace(/"/g, '&quot;');
      const rowBg = s.is_default ? 'background:rgba(22,163,74,0.08);' : '';
      const badge = s.is_default
        ? '<span style="background:#16a34a; color:#062613; '
          + 'font-size:.7rem; font-weight:600; padding:.05rem .35rem; '
          + 'border-radius:3px; margin-left:.4rem;">default</span>' : '';
      const defLabel = s.is_default ? 'Clear default' : 'Set as default';
      return `
        <div style="display:flex; gap:.4rem; align-items:center;
                    padding:.25rem 0; border-top:1px solid #2a3038;${rowBg}">
          <span style="flex:1; font-family:ui-monospace, monospace;">${safe}${badge}</span>
          <span style="color:#778; font-size:.75rem;">${fmtTime(s.mtime)}</span>
          <button class="script-load" data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:var(--accent); color:#062633;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Load</button>
          <button class="script-overwrite" data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:#facc15; color:#241a07;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Overwrite</button>
          <button class="script-default" data-name="${safe}" data-isdef="${s.is_default ? '1' : '0'}"
                  style="padding:.2rem .5rem; border:1px solid #2a3038;
                         border-radius:4px; background:#1f2937; color:#e6edf3;
                         cursor:pointer; font-size:.75rem;">${defLabel}</button>
          <button class="script-delete" data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:var(--bad); color:#240707;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Delete</button>
        </div>`;
    }).join('');
    list.querySelectorAll('.script-load').forEach(b => {
      b.addEventListener('click', () => loadScript(b.dataset.name));
    });
    list.querySelectorAll('.script-overwrite').forEach(b => {
      b.addEventListener('click', () => saveScript(b.dataset.name, true));
    });
    list.querySelectorAll('.script-delete').forEach(b => {
      b.addEventListener('click', () => deleteScript(b.dataset.name));
    });
    list.querySelectorAll('.script-default').forEach(b => {
      b.addEventListener('click', async () => {
        const isDef = b.dataset.isdef === '1';
        await fetch('/api/mission/scripts/' + encodeURIComponent(b.dataset.name)
                    + '/set-default',
                    {method:'POST',
                     headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(isDef ? {clear: true} : {})});
        await refreshScriptList();
      });
    });
  } catch (e) {
    list.innerHTML = '<div style="color:var(--bad);">Failed to load list.</div>';
  }
}
async function saveScript(name, overwrite) {
  const ta = $('script-text'); if (!ta) return;
  if (!name) {
    setScriptSaveStatus('Enter a name first.'); return;
  }
  try {
    const r = await fetch(
      '/api/mission/scripts/' + encodeURIComponent(name),
      {method:'POST',
       headers:{'Content-Type':'application/json'},
       body: JSON.stringify({text: ta.value})});
    const j = await r.json();
    if (j.ok) {
      setScriptSaveStatus((overwrite ? 'Overwrote ' : 'Saved ') + name + '.');
      refreshScriptList();
    } else {
      setScriptSaveStatus('Save failed: ' + (j.error || 'unknown'));
    }
  } catch (e) { setScriptSaveStatus('Save failed: ' + e); }
}
async function loadScript(name) {
  const ta = $('script-text'); if (!ta) return;
  try {
    const r = await fetch(
      '/api/mission/scripts/' + encodeURIComponent(name) + '/load',
      {method:'POST'});
    const j = await r.json();
    if (j.ok && typeof j.text === 'string') {
      ta.value = j.text;
      setScriptError('');
      setScriptSaveStatus('Loaded ' + name + '.');
      // Persist as the active draft so a refresh keeps it.
      flushScriptDraft();
    } else {
      setScriptSaveStatus('Load failed: ' + (j.error || 'unknown'));
    }
  } catch (e) { setScriptSaveStatus('Load failed: ' + e); }
}
async function deleteScript(name) {
  try {
    const r = await fetch(
      '/api/mission/scripts/' + encodeURIComponent(name),
      {method:'DELETE'});
    const j = await r.json();
    if (j.ok) {
      setScriptSaveStatus('Deleted ' + name + '.');
      refreshScriptList();
    } else {
      setScriptSaveStatus('Delete failed: ' + (j.error || 'unknown'));
    }
  } catch (e) { setScriptSaveStatus('Delete failed: ' + e); }
}
if ($('script-text')) {
  $('script-text').addEventListener('input', () => {
    setScriptError('');
    scheduleScriptSave();
  });
  loadInitialScript();
  refreshScriptList();
}
if ($('btn-script-save')) {
  $('btn-script-save').addEventListener('click', () => {
    const n = $('script-save-name');
    saveScript(n ? n.value.trim() : '', false);
  });
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
function updateScriptRuntime(s) {
  const ta = $('script-text');           // present on the live page only
  const rt = $('script-runtime');
  if (!rt) return;
  const lines = Array.isArray(s.mission_script) ? s.mission_script : [];
  if (!lines.length) {
    if (ta) {
      // Live page: no script installed yet -> show the editor.
      rt.style.display = 'none';
      ta.style.display = '';
    } else {
      // Replay page: no script was recorded for this flight.
      rt.style.display = '';
      rt.innerHTML = '<div style="color:#778;">'
                   + 'No mission script recorded for this flight.</div>';
    }
    return;
  }
  const idx = (typeof s.mission_step_idx === 'number') ? s.mission_step_idx : -1;
  if (ta) ta.style.display = 'none';
  rt.style.display = '';
  rt.innerHTML = lines.map((line, i) => {
    let css, marker;
    if (i < idx) {
      css = 'color:#6b7280; padding:.2rem .55rem; '
          + 'border-left:3px solid transparent;';
      marker = '✓';   // check
    } else if (i === idx) {
      css = 'color:#e6e6e6; padding:.25rem .55rem; '
          + 'background:rgba(74, 222, 128, 0.18); '
          + 'border-left:3px solid var(--good); font-weight:600;';
      marker = '▶';   // play
    } else {
      css = 'color:#aab; padding:.2rem .55rem; '
          + 'border-left:3px solid transparent;';
      marker = '·';   // dot
    }
    const safe = line.replace(/&/g,'&amp;').replace(/</g,'&lt;')
                     .replace(/>/g,'&gt;');
    const num = String(i + 1).padStart(2, ' ');
    return `<div style="${css}">`
         + `<span style="display:inline-block; width:1.4em;">${marker}</span>`
         + `<span style="color:#778; margin-right:.5rem;">${num}.</span>`
         + `${safe}</div>`;
  }).join('');
}
// World-position staleness colour. Fresh fixes (< 1s) read green,
// 1-3s reads yellow, beyond 3s the operator is flying on a fix that
// could be far from reality so we go red. ``null`` (never seen yet)
// returns '' so the cell stays neutral.
function wpAgeColor(ageS) {
  if (typeof ageS !== 'number') return '';
  if (ageS < 1.0) return 'var(--good)';
  if (ageS < 3.0) return 'var(--warn)';
  return 'var(--bad)';
}
function updateStatus(s) {
  updateScriptRuntime(s);
  if (!$('s-phase')) return;
  $('s-phase').innerHTML = '<span class="pill">'+s.phase+'</span>';
  // Phase age and marker-last-seen are only meaningful while the
  // drone is airborne. On the ground (init / done / abort) they
  // either count up forever (phase age in INIT) or freeze on a stale
  // value (marker last seen), so blank them out.
  const inFlight = STOPPABLE_PHASES.has(s.phase);
  $('s-pa').textContent = inFlight ? fmt(s.phase_age_s, 's', 1) : '—';
  $('s-d').textContent  = fmt(s.distance_m, 'm', 2);
  $('s-y').textContent  = fmt(s.yaw_to_marker_deg, '°', 1);
  $('s-h').textContent  = fmt(s.relative_heading_deg, '°', 1);
  $('s-td').textContent = fmt(s.target_distance_m, 'm', 2);
  $('s-th').textContent = fmt(s.target_relative_heading_deg, '°', 1);
  $('s-ms').textContent = inFlight ? fmt(s.marker_seen_age_s, 's ago', 2) : '—';
  const rc = s.rc || {};
  $('s-rc').textContent = `${rc.lr ?? 0}, ${rc.fb ?? 0}, ${rc.ud ?? 0}, ${rc.yaw ?? 0}`;
  const t = s.telemetry || {};
  const batEl = $('s-bat');
  if (typeof t.battery === 'number') {
    batEl.textContent = t.battery + '%';
    batEl.style.color = t.battery >= 50 ? 'var(--good)'
                       : t.battery >= 25 ? 'var(--warn)'
                       : 'var(--bad)';
    batEl.style.fontWeight = '600';
  } else {
    batEl.textContent = '—';
    batEl.style.color = '';
    batEl.style.fontWeight = '';
  }
  $('s-dy').textContent  = fmt(t.yaw, '°', 1);
  $('s-ht').textContent  = (t.height_cm !== undefined && t.height_cm !== null) ? ((t.height_cm/100).toFixed(2) + ' m') : '—';
  if ($('s-ht-tgt')) {
    $('s-ht-tgt').textContent = (s.height_target_m !== undefined && s.height_target_m !== null)
      ? (s.height_target_m.toFixed(2) + ' m') : '—';
  }
  if ($('s-wp')) {
    const wpEl = $('s-wp');
    if (Array.isArray(s.world_position_m) && s.world_position_m.length === 3) {
      const [wx, wy, wz] = s.world_position_m;
      const n = (s.world_position_used_markers || []).length;
      const age = s.world_position_age_s;
      const ageTxt = (typeof age === 'number')
        ? ` · ${age.toFixed(1)}s ago` : '';
      wpEl.textContent = `(${wx.toFixed(2)}, ${wy.toFixed(2)}, ${wz.toFixed(2)}) m  · n=${n}${ageTxt}`;
      wpEl.style.color = wpAgeColor(age);
      wpEl.style.fontWeight = '600';
    } else {
      wpEl.textContent = '—';
      wpEl.style.color = '';
      wpEl.style.fontWeight = '';
    }
  }
  $('s-fl').textContent  = t.flying ? 'yes' : 'no';
  $('s-note').textContent = s.note || (s.abort_reason || '—');
  setMissionButtons(s.phase);
  drawRcWidgets(s);
  drawPositionView(s);
}
if ($('btn-start')) $('btn-start').addEventListener('click', startMission);
if ($('btn-stop'))  $('btn-stop').addEventListener('click',  stopMission);

async function restartVideo() {
  const btn = $('btn-video-restart'); const msg = $('video-restart-msg');
  if (!btn) return;
  const orig = btn.textContent;
  btn.disabled = true; btn.textContent = 'Restarting...';
  if (msg) { msg.textContent = ''; msg.style.color = '#aab'; }
  try {
    const r = await fetch('/api/video/restart', {method: 'POST'});
    const j = await r.json();
    if (msg) {
      if (j.ok) {
        msg.style.color = 'var(--good)';
        const fr = (j.reader_stats && j.reader_stats.frames_received) || 0;
        msg.textContent = 'Restarted (frames: ' + fr + ').';
      } else {
        msg.style.color = 'var(--bad)';
        msg.textContent = 'Restart errors: ' + (j.errors || []).join('; ');
      }
    }
    // Force the <img> to reload the stream URL too -- otherwise some
    // browsers keep displaying the cached frozen frame even after the
    // server-side reader has reconnected. Append a query param to bust
    // the request.
    const img = document.querySelector('img.video');
    if (img) {
      const base = (img.getAttribute('src') || '/video.mjpg').split('?')[0];
      img.src = base + '?t=' + Date.now();
    }
  } catch (e) {
    if (msg) {
      msg.style.color = 'var(--bad)';
      msg.textContent = 'Restart failed: ' + e;
    }
  } finally {
    btn.disabled = false; btn.textContent = orig;
  }
}
if ($('btn-video-restart')) {
  $('btn-video-restart').addEventListener('click', restartVideo);
}

// ---- Live RC widget (used by /tune; harmless on pages without it) -------
// Two 2D crosses (lr+fb on the left "throttle stick", yaw+ud on the
// right) plus a time-series history chart for the four channels.
// Auto-scales axes against an observed-max so it works without
// knowing the cfg's *_rc_max values.
let _rcAbsMax = 25;
function drawRc2D(canvasId, xVal, yVal, xLabel, yLabel) {
  const c = $(canvasId); if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#2a3038'; ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W-1, H-1);
  ctx.beginPath();
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2);
  ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H); ctx.stroke();
  ctx.fillStyle = '#778'; ctx.font = '10px ui-sans-serif';
  ctx.textAlign = 'center'; ctx.fillText(xLabel + ' →', W - 24, H/2 - 4);
  ctx.save(); ctx.translate(W/2 + 8, 14); ctx.fillText('↑ ' + yLabel, 0, 0);
  ctx.restore();
  // dot
  const xPx = W/2 + (xVal / _rcAbsMax) * (W/2 - 8);
  const yPx = H/2 - (yVal / _rcAbsMax) * (H/2 - 8);
  ctx.fillStyle = '#58c4ff';
  ctx.beginPath(); ctx.arc(xPx, yPx, 6, 0, 2*Math.PI); ctx.fill();
  ctx.strokeStyle = '#58c4ff66'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W/2, H/2); ctx.lineTo(xPx, yPx); ctx.stroke();
}
function drawRcWidgets(s) {
  // Skip when none of the RC canvases are present (i.e. not on /tune).
  const has2d = $('c-rc-lr-fb') || $('c-rc-yaw-ud');
  const hasHist = !!$('c-rc-history');
  if (!has2d && !hasHist) return;
  const rc = s.rc || {};
  const vals = [rc.lr || 0, rc.fb || 0, rc.ud || 0, rc.yaw || 0];
  const seen = Math.max(_rcAbsMax, ...vals.map(Math.abs));
  if (seen > _rcAbsMax) _rcAbsMax = Math.ceil(seen / 5) * 5;
  if (has2d) {
    drawRc2D('c-rc-lr-fb',  rc.lr || 0, rc.fb || 0, 'lr', 'fb');
    drawRc2D('c-rc-yaw-ud', rc.yaw || 0, rc.ud || 0, 'yaw', 'ud');
  }
  // Pages with HAS_CHARTS already pushed this tick inside updateCharts;
  // tune-style pages haven't, so push here so the history chart fills.
  if (hasHist && !HAS_CHARTS && !REPLAY_ID) pushSample(s);
  if (hasHist) {
    drawSeries('c-rc-history', [
      {label:'rc_lr',  color:'#fb7185', data:buf.rc_lr,  legendX:0},
      {label:'rc_fb',  color:'#34d399', data:buf.rc_fb,  legendX:1},
      {label:'rc_ud',  color:'#60a5fa', data:buf.rc_ud,  legendX:2},
      {label:'rc_yaw', color:'#fbbf24', data:buf.rc_yaw, legendX:3},
    ], {target: 0});
  }
}

// ---- Position view (used by /; harmless on pages without it) ------------
// Top-down arena diagram: 1 m grid, wall labels (front/back/left/right),
// origin tick, drone position dot + heading arrow.
//
// The arrow does NOT use tel.yaw because the Anafi's yaw=0 is its
// takeoff-time / compass reference, not "facing the front wall". For
// every flight the offset between tel.yaw and arena-frame yaw is
// arbitrary, so a tel.yaw-based arrow is meaningless without a
// per-flight calibration step.
//
// Instead the server publishes ``arena_yaw_deg`` (CW from arena +y /
// front wall) whenever ANY visible reference marker can resolve it;
// see vision_worker. We just read it here, gated by ``arena_yaw_age_s``
// so a stale value doesn't mislead the operator after the last marker
// leaves the frame. When the value is missing or stale we skip the
// arrow rather than draw a wrong one.
let _arenaCache = null;
let _arenaCacheTried = false;
async function _ensureArenaCache() {
  if (_arenaCacheTried) return;
  _arenaCacheTried = true;
  try {
    const r = await fetch('/api/arena/active', {cache:'no-store'});
    if (r.ok) _arenaCache = await r.json();
  } catch (e) {}
}
const ARENA_YAW_FRESH_S = 1.0;
function _droneYawArena(s) {
  if (typeof s.arena_yaw_deg !== 'number') return null;
  const age = s.arena_yaw_age_s;
  if (typeof age !== 'number' || age > ARENA_YAW_FRESH_S) return null;
  return s.arena_yaw_deg;
}
function drawPositionView(s) {
  const c = $('c-pos'); if (!c) return;
  _ensureArenaCache();
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#2a3038'; ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W-1, H-1);
  const margin = 16;
  const aw = (_arenaCache && _arenaCache.width_m) || 10;
  const ad = (_arenaCache && _arenaCache.depth_m) || 10;
  const pxPerM = Math.min((W - 2*margin) / aw, (H - 2*margin) / ad);
  const cx = W/2, cy = H/2;
  const ax = (xm) => cx + xm * pxPerM;
  const ay = (ym) => cy - ym * pxPerM;
  const xLeft = ax(-aw/2), xRight = ax(aw/2);
  const yTop = ay(ad/2), yBot = ay(-ad/2);
  // 1 m background grid clipped to the arena rect. Heavier line at
  // the centre lines (origin axes).
  const xMin = Math.ceil(-aw/2), xMax = Math.floor(aw/2);
  const yMin = Math.ceil(-ad/2), yMax = Math.floor(ad/2);
  ctx.strokeStyle = '#1d2229'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let xi = xMin; xi <= xMax; xi++) {
    if (xi === 0) continue;
    const xp = ax(xi);
    ctx.moveTo(xp, yTop); ctx.lineTo(xp, yBot);
  }
  for (let yi = yMin; yi <= yMax; yi++) {
    if (yi === 0) continue;
    const yp = ay(yi);
    ctx.moveTo(xLeft, yp); ctx.lineTo(xRight, yp);
  }
  ctx.stroke();
  // Origin axes (slightly brighter).
  ctx.strokeStyle = '#2a3038';
  ctx.beginPath();
  ctx.moveTo(cx, yTop); ctx.lineTo(cx, yBot);
  ctx.moveTo(xLeft, cy); ctx.lineTo(xRight, cy);
  ctx.stroke();
  // Arena rectangle.
  if (_arenaCache) {
    ctx.strokeStyle = '#aab'; ctx.lineWidth = 1.5;
    ctx.strokeRect(xLeft, yTop, aw * pxPerM, ad * pxPerM);
  }
  // Wall captions.
  ctx.fillStyle = '#778'; ctx.font = '9px ui-sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('front', cx, yTop - 4);
  ctx.fillText('back',  cx, yBot + 9);
  ctx.save(); ctx.translate(xLeft - 4, cy); ctx.rotate(-Math.PI/2);
  ctx.fillText('left', 0, 0); ctx.restore();
  ctx.save(); ctx.translate(xRight + 4, cy); ctx.rotate(Math.PI/2);
  ctx.fillText('right', 0, 0); ctx.restore();
  // Origin tick.
  ctx.fillStyle = '#facc15';
  ctx.beginPath(); ctx.arc(cx, cy, 2, 0, 2*Math.PI); ctx.fill();
  // Markers from the cached arena, same convention as the /arena
  // canvas + the in-video mini-map: filled wall-coloured dot per
  // marker (front=yellow, right=green, back=blue, left=red).
  // Co-located markers (same x/y, different z) are stacked along
  // the canvas vertical with high-z up. Currently-visible markers
  // get a white outline ring -- the operator can see at a glance
  // which references the position estimate is using.
  if (_arenaCache && Array.isArray(_arenaCache.markers)) {
    const seen = new Set((s.world_position_used_markers || [])
                          .concat(s.visible_marker_ids || [])
                          .map(v => parseInt(v, 10))
                          .filter(v => !Number.isNaN(v)));
    const STACK = 7;
    const groups = new Map();
    _arenaCache.markers.forEach((m, idx) => {
      const key = `${(+m.x).toFixed(2)},${(+m.y).toFixed(2)}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(idx);
    });
    const stackOff = new Map();
    groups.forEach(indices => {
      indices.sort((a, b) =>
        _arenaCache.markers[b].z - _arenaCache.markers[a].z);
      const n = indices.length;
      indices.forEach((idx, i) =>
        stackOff.set(idx, (i - (n - 1) / 2) * STACK));
    });
    _arenaCache.markers.forEach((m, idx) => {
      const px = ax(+m.x);
      const py = ay(+m.y) + (stackOff.get(idx) || 0);
      ctx.fillStyle = WALL_COLORS[m.wall] || '#fff';
      ctx.beginPath(); ctx.arc(px, py, 4, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = '#0e1116'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, 2*Math.PI); ctx.stroke();
      if (seen.has(parseInt(m.id, 10))) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py, 7, 0, 2*Math.PI); ctx.stroke();
      }
    });
  }
  // Drone position + heading arrow. Dot colour matches the
  // sidebar's wpAgeColor: fresh = green, stale = yellow/red, so
  // the operator can spot a frozen position even at a glance.
  const pos = s.world_position_m;
  const txt = $('c-pos-text');
  if (Array.isArray(pos) && pos.length === 3) {
    const dx = ax(pos[0]), dy = ay(pos[1]);
    const age = s.world_position_age_s;
    const dotCol = (typeof age !== 'number' || age < 1.0) ? '#4ade80'
                 : (age < 3.0) ? '#facc15'
                 : '#f87171';
    ctx.fillStyle = dotCol;
    ctx.beginPath(); ctx.arc(dx, dy, 5, 0, 2*Math.PI); ctx.fill();
    const yawArena = _droneYawArena(s);
    let yawTxt = '—';
    if (yawArena !== null) {
      const yawRad = yawArena * Math.PI / 180;
      const L = 18;
      const ex = dx + L * Math.sin(yawRad);
      const ey = dy - L * Math.cos(yawRad);     // canvas y flipped
      ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(ex, ey); ctx.stroke();
      const ang = Math.atan2(ey - dy, ex - dx);
      const HEAD = 5;
      ctx.fillStyle = '#4ade80';
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - HEAD * Math.cos(ang - Math.PI/6),
                 ey - HEAD * Math.sin(ang - Math.PI/6));
      ctx.lineTo(ex - HEAD * Math.cos(ang + Math.PI/6),
                 ey - HEAD * Math.sin(ang + Math.PI/6));
      ctx.closePath(); ctx.fill();
      const norm = ((yawArena % 360) + 540) % 360 - 180;
      yawTxt = norm.toFixed(1) + '°';
    }
    if (txt) {
      const n = (s.world_position_used_markers || []).length;
      const ageCol = wpAgeColor(age) || '#e6e6e6';
      const ageTxt = (typeof age === 'number')
        ? `${age.toFixed(1)} s ago` : 'no fix yet';
      txt.innerHTML = `x: <b style="color:#e6e6e6;">${pos[0].toFixed(2)}</b> m<br>`
                    + `y: <b style="color:#e6e6e6;">${pos[1].toFixed(2)}</b> m<br>`
                    + `z: <b style="color:#e6e6e6;">${pos[2].toFixed(2)}</b> m<br>`
                    + `yaw: <b style="color:#e6e6e6;">${yawTxt}</b><br>`
                    + `n markers: ${n}<br>`
                    + `age: <b style="color:${ageCol};">${ageTxt}</b>`;
    }
  } else {
    if (txt) txt.textContent = 'World position unknown.';
  }
}

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
// Track previous phase so we can beep when it changes. lastPhase is
// null on first iteration -- skipping the beep then avoids a chirp
// just because the operator opened the tab mid-flight.
let lastPhase = null;
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
    // Phase-change audible cue. Don't beep on the FIRST poll (lastPhase
    // null) so a freshly-opened tab doesn't chirp at us. Don't beep
    // during replay either -- the playback would cue every transition.
    if (!REPLAY_ID && lastPhase !== null && s.phase !== lastPhase
        && typeof _beep === 'function') {
      _beep(880, 150);
    }
    lastPhase = s.phase;
  } catch (e) {}
  setTimeout(refresh, 250);
}
// Paint the correct Start/Stop state from the server-rendered phase
// before the first /api/state lands. Otherwise mid-flight refreshes
// briefly show the green Start button (and re-arm the script editor)
// for the ~250ms it takes refresh() to land.
if (typeof setMissionButtons === 'function') {
  try { setMissionButtons(INITIAL_PHASE); } catch (e) {}
}
refresh();
</script>
"""

_PAGE_VIDEO = (_PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT
               + "<main>" + _VIDEO_AND_STATUS_HTML + _CHARTS_HTML + "</main>"
               + _SHARED_SCRIPT)

_PAGE_CHARTS = (_PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT
                + _PAGE_GRID_OPEN + _CHARTS_HTML + _PAGE_GRID_CLOSE
                + _SHARED_SCRIPT)

# Flight list -- one card per recorded flight, with a "Replay" button.
_PAGE_FLIGHTS = _PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT + _PAGE_GRID_OPEN + """
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
""" + _PAGE_GRID_CLOSE + _SHARED_SCRIPT

# Camera-calibration page. Live preview on the left, capture-then-run
# workflow on the right. The /api/calibrate/* endpoints below drive
# this page; the operator never needs to drop to the CLI to recalibrate.
_PAGE_CALIBRATE = _PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT + _PAGE_GRID_OPEN + """
<section class="grid" style="grid-template-columns: minmax(0,2fr) minmax(0,1fr);">
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
</section>
<section class="grid" style="grid-template-columns: 1fr; margin-top:1rem;">
  <div class="card">
    <h2>Camera config</h2>
    <p style="font-size:.85rem; color:#aab; margin:.2rem 0 .6rem;">
      Live values pulled from the Olympe camera (only the axes the
      firmware exposes show up). Type a valid enum string per Olympe
      docs and press <b>Apply</b>. <b>Save</b> stores the current
      form as a named config; <b>Set as default</b> on a row makes
      that config auto-load at startup.
    </p>
    <div id="camcfg-form" style="display:grid;
         grid-template-columns: minmax(11em, max-content) minmax(0, 1fr);
         gap:.35rem .9rem; align-items:center; font-size:.85rem;">
      <div style="grid-column: 1 / -1; color:#778;">loading…</div>
    </div>
    <div style="display:flex; gap:.4rem; align-items:center;
                margin-top:.8rem; flex-wrap:wrap;">
      <button id="camcfg-apply" type="button"
              style="padding:.4rem .8rem; border:0; border-radius:6px;
                     background:var(--good); color:#072413; font-weight:600;
                     cursor:pointer; font-size:.85rem;">Apply</button>
      <button id="camcfg-refresh" type="button"
              style="padding:.4rem .8rem; border:1px solid #2a3038;
                     border-radius:6px; background:#1f2937; color:#e6edf3;
                     cursor:pointer; font-size:.85rem;">Refresh</button>
      <input id="camcfg-name" type="text" placeholder="name to save as"
             style="margin-left:.6rem; padding:.4rem .55rem; flex:1;
                    min-width:10rem; background:#0c0f12; color:var(--fg);
                    border:1px solid #2a3038; border-radius:6px;
                    font-size:.85rem;">
      <button id="camcfg-save" type="button"
              style="padding:.4rem .8rem; border:0; border-radius:6px;
                     background:#facc15; color:#241a07; font-weight:600;
                     cursor:pointer; font-size:.85rem;">Save</button>
    </div>
    <div id="camcfg-msg" style="margin-top:.5rem; font-size:.8rem;
                                  color:#aab; min-height:1.2em;"></div>
    <div id="camcfg-default-banner" style="margin-top:.6rem;
                                            display:none;"></div>
    <div style="max-height:240px; overflow-y:auto; border:1px solid #2a3038;
                border-radius:6px; margin-top:.4rem;">
      <table id="camcfg-table" style="width:100%; border-collapse:collapse;
                                       font-size:.85rem;">
        <thead style="position:sticky; top:0; background:var(--panel);">
          <tr style="text-align:left; color:#9ca3af;">
            <th style="padding:.3rem .4rem;">Name</th>
            <th style="padding:.3rem .4rem;">Saved</th>
            <th style="padding:.3rem .4rem; width:22rem;">Actions</th>
          </tr>
        </thead>
        <tbody id="camcfg-body">
          <tr><td colspan="3" style="color:#6b7280; padding:.4rem;">
              loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
<script>
(() => {
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

// ---- Camera config form ------------------------------------------------
// Live values come from GET /api/camera/config. The form renders one
// row per axis-subfield using ``CAMCFG_SCHEMA`` for the control type
// (enum -> <select>, number -> numeric <input>, readonly -> static
// text), the dropdown options where applicable, and the explanatory
// description. Unknown keys (firmware fields the schema hasn't been
// taught about) fall back to a text input so the form never silently
// hides data.
const _ISO_VALUES = ['iso_50','iso_64','iso_80','iso_100','iso_125',
  'iso_160','iso_200','iso_250','iso_320','iso_400','iso_500','iso_640',
  'iso_800','iso_1000','iso_1200','iso_1600','iso_2000','iso_2500','iso_3200'];
const _SHUTTER_VALUES = ['shutter_1_over_10000','shutter_1_over_8000',
  'shutter_1_over_4000','shutter_1_over_3000','shutter_1_over_2000',
  'shutter_1_over_1000','shutter_1_over_500','shutter_1_over_250',
  'shutter_1_over_125','shutter_1_over_60','shutter_1_over_30',
  'shutter_1_over_15','shutter_1_over_8','shutter_1_over_4',
  'shutter_1_over_2','shutter_1','shutter_2','shutter_4','shutter_15',
  'shutter_30'];
const _EV_VALUES = ['ev_minus_3_00','ev_minus_2_67','ev_minus_2_33',
  'ev_minus_2_00','ev_minus_1_67','ev_minus_1_33','ev_minus_1_00',
  'ev_minus_0_67','ev_minus_0_33','ev_0_00','ev_plus_0_33',
  'ev_plus_0_67','ev_plus_1_00','ev_plus_1_33','ev_plus_1_67',
  'ev_plus_2_00','ev_plus_2_33','ev_plus_2_67','ev_plus_3_00'];
const _WB_TEMP = ['t_1500','t_1750','t_2000','t_2250','t_2500','t_2750',
  't_3000','t_3250','t_3500','t_3750','t_4000','t_4250','t_4500',
  't_4750','t_5000','t_5250','t_5500','t_5750','t_6000','t_6500',
  't_7000','t_7500','t_8000','t_8500','t_9000','t_9500','t_10000',
  't_10500','t_11000','t_11500','t_12000','t_12500','t_13000','t_13500',
  't_14000','t_14500','t_15000'];
const CAMCFG_SCHEMA = {
  'exposure.mode': {label: 'Exposure mode', type: 'enum',
    options: ['automatic','automatic_prefer_iso_sensitivity',
              'automatic_prefer_shutter_speed','manual_iso_sensitivity',
              'manual_shutter_speed','manual'],
    default: 'automatic',
    desc: 'How exposure is balanced. automatic picks both shutter and ISO. '
        + 'automatic_prefer_iso_sensitivity keeps ISO low (cleaner image, longer '
        + 'shutter); automatic_prefer_shutter_speed keeps shutter fast (less '
        + 'motion blur, higher ISO). manual_* pin one param; manual pins both.'},
  'exposure.shutter_speed': {label: 'Shutter speed', type: 'enum',
    options: _SHUTTER_VALUES, default: 'shutter_1_over_60',
    desc: 'Shutter time. Faster (e.g. shutter_1_over_500) freezes motion but '
        + 'darkens; slower (e.g. shutter_1_over_30) admits more light. '
        + 'Honoured only when the mode includes manual_shutter_speed or manual.'},
  'exposure.iso_sensitivity': {label: 'ISO sensitivity', type: 'enum',
    options: _ISO_VALUES, default: 'iso_100',
    desc: 'Sensor gain. Higher ISO (e.g. iso_3200) brightens dim scenes at the '
        + 'cost of noise. Honoured when mode includes manual_iso_sensitivity, '
        + 'manual, or automatic_prefer_iso_sensitivity.'},
  'exposure.max_iso_sensitivity': {label: 'Max auto ISO', type: 'enum',
    options: _ISO_VALUES, default: 'iso_3200',
    desc: 'Cap on the ISO that auto modes will use. Lower keeps images cleaner '
        + 'in dim light at the cost of slower shutters (more motion blur).'},
  'exposure.metering_mode': {label: 'Metering mode', type: 'enum',
    options: ['standard','center_top'], default: 'standard',
    desc: 'Where the light meter samples. standard averages the whole frame; '
        + 'center_top weights the top-centre (helps when the marker is high '
        + 'and a bright floor would otherwise pull the meter dark).'},
  'white_balance.mode': {label: 'White balance', type: 'enum',
    options: ['automatic','candle','sunset','incandescent','warm_white','halogen',
              'fluorescent','cool_white','horizon','daylight','flash','blue_sky',
              'shaded','custom'], default: 'automatic',
    desc: 'Colour-cast correction. automatic adapts to scene; presets (daylight, '
        + 'incandescent, etc.) lock to known light sources. custom uses the '
        + 'temperature below.'},
  'white_balance.temperature': {label: 'WB temperature', type: 'enum',
    options: _WB_TEMP, default: 't_5000',
    desc: 'Custom WB colour temperature in Kelvin (t_5000 = daylight, '
        + 't_3000 = incandescent). Honoured only when mode = custom.'},
  'hdr.value': {label: 'HDR', type: 'enum',
    options: ['inactive','active'], default: 'inactive',
    desc: 'High Dynamic Range. active boosts shadows and tames highlights but '
        + 'introduces motion artefacts on fast pans/flights. Leave inactive '
        + 'for marker tracking.'},
  'ev_compensation.value': {label: 'EV bias', type: 'enum',
    options: _EV_VALUES, default: 'ev_0_00',
    desc: 'Exposure compensation in stops on top of the chosen mode. Negative '
        + 'values (ev_minus_1_00) darken; positive values brighten. Useful '
        + 'when the marker reads washed-out (-) or murky (+).'},
  'antiflicker.mode': {label: 'Anti-flicker', type: 'enum',
    options: ['off','auto','mode_50hz','mode_60hz'], default: 'auto',
    desc: 'Filter for indoor lighting flicker (LED / fluorescent). mode_50hz '
        + 'for EU mains, mode_60hz for US mains; auto lets the camera detect; '
        + 'off disables.'},
  'video_stabilization.mode': {label: 'Stabilisation', type: 'enum',
    options: ['roll_pitch','pitch','none'], default: 'roll_pitch',
    desc: 'Gimbal-software stabilisation axes. roll_pitch (default) is the '
        + 'smoothest. pitch stabilises tilt only. none leaves the gimbal '
        + 'hard-locked (useful when calibrating intrinsics).'},
  'stream_mode.mode': {label: 'Stream mode', type: 'enum',
    options: ['low_latency','high_reliability','high_reliability_low_framerate'],
    default: 'low_latency',
    desc: 'Live-stream tuning. low_latency (default) minimises lag for marker '
        + 'tracking. The high_reliability_* options trade latency for fewer '
        + 'dropped frames on poor Wi-Fi.'},
  'recording.mode': {label: 'Recording mode', type: 'enum',
    options: ['standard','hyperlapse','slow_motion','high_framerate'],
    default: 'standard',
    desc: 'On-board recording mode. standard is normal video. hyperlapse uses '
        + 'the ratio below. slow_motion / high_framerate require a high-FPS '
        + 'framerate setting.'},
  'recording.resolution': {label: 'Recording resolution', type: 'enum',
    options: ['res_dci_4k','res_uhd_4k','res_2_7k','res_1080p','res_720p','res_480p'],
    default: 'res_1080p',
    desc: 'Resolution of the recorded clip. Higher resolutions limit max '
        + 'framerate. For marker work the live stream resolution is '
        + 'independent of this.'},
  'recording.framerate': {label: 'Recording framerate', type: 'enum',
    options: ['fps_24','fps_25','fps_30','fps_48','fps_50','fps_60','fps_96',
              'fps_100','fps_120','fps_240','fps_480'], default: 'fps_30',
    desc: 'Recorded framerate. Higher FPS supports slow-motion playback but '
        + 'caps achievable resolution.'},
  'recording.hyperlapse': {label: 'Hyperlapse ratio', type: 'enum',
    options: ['ratio_15','ratio_30','ratio_60','ratio_120'], default: 'ratio_15',
    desc: 'Time-lapse compression ratio when mode = hyperlapse. ratio_15 '
        + 'plays back 15x faster than real time.'},
  'recording.bitrate': {label: 'Recording bitrate', type: 'readonly',
    desc: 'Reported bitrate (bits/s) of the active recording mode. Read-only -- '
        + 'derived from resolution + framerate + mode by the firmware.'},
  'zoom.level': {label: 'Zoom level', type: 'number',
    min: 1.0, max: 3.0, step: 0.1, default: 1.0,
    desc: 'Digital zoom factor. 1.0 = no zoom (native FOV), 3.0 = maximum 3x '
        + 'digital zoom. Increasing zoom narrows the FOV -- markers near the '
        + 'frame edge can disappear.'},
};

let camcfgKeys = [];   // [['exposure','mode'], ...]

function camcfgSetMsg(s, ok) {
  const el = $('camcfg-msg'); if (!el) return;
  el.textContent = s || '';
  el.style.color = ok === false ? 'var(--bad)'
                 : ok === true ? 'var(--good)'
                 : '#aab';
}

function _renderField(key, sub, value) {
  const path = key + '.' + sub;
  const schema = CAMCFG_SCHEMA[path] || null;
  // Label cell with an info icon: hovering reveals the description
  // (browser-native title tooltip, same pattern as /tune's tune-info).
  const lbl = document.createElement('div');
  lbl.style.cssText = 'color:#cbd5e1; padding:.2rem 0;';
  const labelText = document.createElement('span');
  labelText.textContent = schema ? schema.label : path;
  lbl.appendChild(labelText);
  const helpText = schema ? schema.desc
    : '(no description for ' + path + ' -- firmware-specific field)';
  const info = document.createElement('span');
  info.className = 'tune-info';
  info.textContent = 'ⓘ';      // circled-i (the same glyph /tune uses)
  info.title = helpText;
  info.tabIndex = 0;
  lbl.appendChild(info);
  // Control cell.
  let ctrl;
  const fwHas = !(value === null || value === undefined || value === '');
  const cur = fwHas ? String(value) : '';
  // Schema default kicks in only when the firmware didn't supply a
  // value -- a non-null firmware reading always wins (so the form
  // reflects what the camera is actually doing).
  const fallback = schema && schema.default !== undefined
    ? String(schema.default) : '';
  const initial = fwHas ? cur : fallback;
  if (schema && schema.type === 'enum') {
    ctrl = document.createElement('select');
    const opts = schema.options.slice();
    // Surface the firmware's actual value even if the schema's
    // hardcoded list doesn't include it (older Olympe builds).
    if (cur && !opts.includes(cur)) opts.unshift(cur);
    for (const o of opts) {
      const op = document.createElement('option');
      op.value = o; op.textContent = o; ctrl.appendChild(op);
    }
    ctrl.value = initial;
    if (!ctrl.value && opts.length) ctrl.value = opts[0];
  } else if (schema && schema.type === 'number') {
    ctrl = document.createElement('input');
    ctrl.type = 'number';
    if (schema.min !== undefined) ctrl.min = schema.min;
    if (schema.max !== undefined) ctrl.max = schema.max;
    if (schema.step !== undefined) ctrl.step = schema.step;
    ctrl.value = initial;
  } else if (schema && schema.type === 'readonly') {
    ctrl = document.createElement('input');
    ctrl.type = 'text';
    ctrl.value = cur;
    ctrl.readOnly = true;
    ctrl.style.opacity = '.6';
    ctrl.style.cursor = 'not-allowed';
    ctrl.dataset.readonly = '1';
  } else {
    // Unknown axis -- text input fallback.
    ctrl = document.createElement('input');
    ctrl.type = (typeof value === 'number') ? 'number' : 'text';
    ctrl.value = cur;
  }
  ctrl.dataset.key = key;
  ctrl.dataset.sub = sub;
  ctrl.title = helpText;
  ctrl.style.cssText += 'background:#0c0f12; color:var(--fg);'
    + ' border:1px solid #2a3038; border-radius:4px;'
    + ' padding:.25rem .4rem; font-family:ui-monospace, monospace;'
    + ' font-size:.85rem; width:100%; box-sizing:border-box;';
  return [lbl, ctrl];
}

function camcfgPopulate(cfg) {
  const f = $('camcfg-form'); if (!f) return;
  f.innerHTML = '';
  camcfgKeys = [];
  if (!cfg || cfg.connected === false) {
    const div = document.createElement('div');
    div.style.cssText = 'grid-column:1/-1; color:#aab;';
    div.textContent = 'Drone not connected.';
    f.appendChild(div);
    return;
  }
  // Skip non-axis keys.
  const skip = new Set(['connected', 'cam_id']);
  const axes = Object.keys(cfg).filter(k => !skip.has(k));
  for (const k of axes) {
    const v = cfg[k];
    if (v && typeof v === 'object') {
      for (const sub of Object.keys(v)) {
        camcfgKeys.push([k, sub]);
        const cells = _renderField(k, sub, v[sub]);
        for (const c of cells) f.appendChild(c);
      }
    } else {
      camcfgKeys.push([k, '_value']);
      const cells = _renderField(k, '_value', v);
      for (const c of cells) f.appendChild(c);
    }
  }
}
function camcfgCollect() {
  const out = {};
  document.querySelectorAll('#camcfg-form input, #camcfg-form select')
    .forEach(inp => {
      const k = inp.dataset.key, s = inp.dataset.sub;
      if (!k || !s) return;
      // Skip read-only display fields (e.g. recording.bitrate) so we
      // don't fight the firmware by sending back its own derived value.
      if (inp.dataset.readonly === '1') return;
      let val = inp.value;
      if (inp.type === 'number') {
        const n = parseFloat(val); if (!Number.isFinite(n)) return;
        val = n;
      }
      if (s === '_value') { out[k] = val; return; }
      out[k] = out[k] || {};
      out[k][s] = val;
    });
  return out;
}
async function refreshCamcfg() {
  try {
    const r = await fetch('/api/camera/config', {cache:'no-store'});
    const j = await r.json();
    if (!j.ok) {
      camcfgSetMsg('Read failed: ' + (j.error || 'unknown'), false);
      camcfgPopulate(null);
      return;
    }
    camcfgPopulate(j.config || {});
    camcfgSetMsg('', null);
  } catch (e) {
    camcfgSetMsg('Read failed: ' + e, false);
  }
}
async function applyCamcfg() {
  const body = camcfgCollect();
  try {
    const r = await fetch('/api/camera/config',
                          {method:'POST',
                           headers:{'Content-Type':'application/json'},
                           body: JSON.stringify(body)});
    const j = await r.json();
    if (!j.ok) {
      camcfgSetMsg('Apply failed: ' + (j.error || 'unknown'), false);
      return;
    }
    const fail = Object.entries(j.results || {})
      .filter(([k, v]) => v === false && !k.endsWith('_error'))
      .map(([k]) => k);
    if (fail.length) {
      camcfgSetMsg('Applied with ' + fail.length + ' failures: '
                   + fail.join(', '), false);
    } else {
      camcfgSetMsg('Applied.', true);
    }
    if (j.config) camcfgPopulate(j.config);
  } catch (e) {
    camcfgSetMsg('Apply failed: ' + e, false);
  }
}
async function saveCamcfg() {
  const name = $('camcfg-name').value.trim();
  if (!name) { camcfgSetMsg('Type a name first.', false); return; }
  const body = camcfgCollect();
  try {
    const r = await fetch(
      '/api/camera/configs/' + encodeURIComponent(name),
      {method:'POST',
       headers:{'Content-Type':'application/json'},
       body: JSON.stringify(body)});
    const j = await r.json();
    if (j.ok) {
      camcfgSetMsg('Saved as ' + name + '.', true);
      await refreshCamSaves();
    } else {
      camcfgSetMsg('Save failed: ' + (j.error || 'unknown'), false);
    }
  } catch (e) {
    camcfgSetMsg('Save failed: ' + e, false);
  }
}
async function loadCamcfg(name) {
  try {
    const r = await fetch(
      '/api/camera/configs/' + encodeURIComponent(name) + '/load',
      {method:'POST'});
    const j = await r.json();
    if (!j.ok) {
      camcfgSetMsg('Load failed: ' + (j.error || 'unknown'), false);
      return;
    }
    // Populate the form from the saved JSON without pushing to drone.
    // The operator presses Apply explicitly so a mid-flight change
    // can't surprise them.
    const cfg = Object.assign({connected: true}, j.config || {});
    camcfgPopulate(cfg);
    camcfgSetMsg('Loaded ' + name + ' into form (press Apply to push).',
                 true);
  } catch (e) {
    camcfgSetMsg('Load failed: ' + e, false);
  }
}
async function deleteCamcfg(name) {
  if (!confirm('Delete saved camera config "' + name + '"?')) return;
  try {
    const r = await fetch(
      '/api/camera/configs/' + encodeURIComponent(name),
      {method:'DELETE'});
    const j = await r.json();
    if (j.ok) {
      camcfgSetMsg('Deleted ' + name + '.', true);
      await refreshCamSaves();
    } else {
      camcfgSetMsg('Delete failed: ' + (j.error || 'unknown'), false);
    }
  } catch (e) {
    camcfgSetMsg('Delete failed: ' + e, false);
  }
}
async function setCamDefault(name, isCurrentlyDefault) {
  try {
    await fetch(
      '/api/camera/configs/' + encodeURIComponent(name) + '/set-default',
      {method:'POST',
       headers:{'Content-Type':'application/json'},
       body: JSON.stringify(isCurrentlyDefault ? {clear: true} : {})});
    await refreshCamSaves();
  } catch (e) {
    camcfgSetMsg('Set default failed: ' + e, false);
  }
}
async function refreshCamSaves() {
  const tb = $('camcfg-body');
  if (!tb) return;
  try {
    const r = await fetch('/api/camera/configs', {cache:'no-store'});
    const j = await r.json();
    renderDefaultMissingBanner('camcfg-default-banner', j.default,
      (j.configs || []).some(x => x.name === j.default),
      () => clearDefault('/api/camera/configs', refreshCamSaves));
    tb.innerHTML = '';
    if (!j.configs || !j.configs.length) {
      tb.innerHTML = '<tr><td colspan="3" style="color:#6b7280;'
        + ' padding:.4rem;">No saved camera configs yet.</td></tr>';
      return;
    }
    const fmtTime = mt => new Date(mt*1000).toLocaleString();
    for (const s of j.configs) {
      const tr = document.createElement('tr');
      tr.style.borderTop = '1px dashed #2a3038';
      if (s.is_default) tr.style.background = 'rgba(22,163,74,0.08)';
      const nameTd = document.createElement('td');
      nameTd.style.padding = '.3rem .4rem';
      nameTd.style.color = '#e6edf3';
      nameTd.style.fontFamily = 'ui-monospace, monospace';
      nameTd.textContent = s.name;
      if (s.is_default) appendDefaultBadge(nameTd);
      const timeTd = document.createElement('td');
      timeTd.style.padding = '.3rem .4rem';
      timeTd.style.color = '#9ca3af';
      timeTd.textContent = fmtTime(s.mtime);
      const actTd = document.createElement('td');
      actTd.style.padding = '.3rem .4rem';
      const loadBtn = document.createElement('button');
      loadBtn.textContent = 'Load';
      loadBtn.type = 'button';
      loadBtn.title = 'Populate the form from this saved config'
        + ' (does NOT push to the drone). Press Apply to push.';
      loadBtn.style.cssText = 'padding:.2rem .55rem; border:0;'
        + ' border-radius:4px; background:var(--accent); color:#062633;'
        + ' font-weight:600; cursor:pointer; margin-right:.4rem;'
        + ' font-size:.8rem;';
      loadBtn.addEventListener('click', () => loadCamcfg(s.name));
      const defBtn = makeSetDefaultButton(s.is_default,
        () => setCamDefault(s.name, s.is_default));
      const delBtn = document.createElement('button');
      delBtn.textContent = 'Delete';
      delBtn.type = 'button';
      delBtn.style.cssText = 'padding:.2rem .55rem; border:1px solid #7f1d1d;'
        + ' border-radius:4px; background:transparent; color:#fca5a5;'
        + ' cursor:pointer; font-size:.8rem;';
      delBtn.addEventListener('click', () => deleteCamcfg(s.name));
      actTd.appendChild(loadBtn);
      actTd.appendChild(defBtn);
      actTd.appendChild(delBtn);
      tr.appendChild(nameTd);
      tr.appendChild(timeTd);
      tr.appendChild(actTd);
      tb.appendChild(tr);
    }
  } catch (e) {
    camcfgSetMsg('List failed: ' + e, false);
  }
}
$('camcfg-apply').addEventListener('click', applyCamcfg);
$('camcfg-refresh').addEventListener('click', refreshCamcfg);
$('camcfg-save').addEventListener('click', saveCamcfg);
refreshCamcfg();
refreshCamSaves();
})();
</script>
""" + _PAGE_GRID_CLOSE + _SHARED_SCRIPT


# ---------------------------------------------------------------------------
# Tuning page. Lists every UI-exposed config field grouped by purpose;
# numeric inputs are pre-filled with the current value and show the
# dataclass default next to each row. Apply pushes the new values to
# the running controller (PD gains, output clamps, smoother) without a
# restart. Save / Reload / Reset operate on the persisted JSON file.
# ---------------------------------------------------------------------------

_PAGE_TUNE = _PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT + _PAGE_GRID_OPEN + """
<section class="grid" style="grid-template-columns: minmax(0, 1fr);">
  <div class="card">
    <h2>Live RC</h2>
    <div style="display:flex; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
      <div style="text-align:center;">
        <canvas id="c-rc-lr-fb" width="180" height="180"
                style="background:#0c0f12; border-radius:6px;"></canvas>
        <div style="font-size:.75rem; color:#778; margin-top:.2rem;">lr / fb</div>
      </div>
      <div style="text-align:center;">
        <canvas id="c-rc-yaw-ud" width="180" height="180"
                style="background:#0c0f12; border-radius:6px;"></canvas>
        <div style="font-size:.75rem; color:#778; margin-top:.2rem;">yaw / ud</div>
      </div>
      <div style="flex:1; min-width:280px;">
        <canvas id="c-rc-history" width="700" height="180"
                style="background:#0c0f12; border-radius:6px;
                       width:100%; height:auto;"></canvas>
        <div style="font-size:.75rem; color:#778; margin-top:.2rem;
                    text-align:center;">history</div>
      </div>
    </div>
  </div>
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
    <div id="snap-default-banner" style="display:none;"></div>
    <div style="max-height:240px; overflow-y:auto; border:1px solid #2a3038;
                border-radius:6px;">
      <table id="snap-table" style="width:100%; border-collapse:collapse;
                                    font-size:.85rem;">
        <thead style="position:sticky; top:0; background:var(--panel);">
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
</section>

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
(() => {
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
      let inputType, stepAttr, valueAttr;
      if (it.kind === 'bool') {
        inputType = 'checkbox';
        stepAttr  = '';
        // Boolean values come through as JS true/false from the
        // /api/tune JSON payload; render the corresponding 'checked'
        // attribute. value="" is required because the apply path
        // also sends inp.value for non-bools, but for bools we
        // override with inp.checked further down.
        valueAttr = (it.value ? 'checked' : '') + ' value=""';
      } else if (it.kind === 'str') {
        inputType = 'text';
        stepAttr  = '';
        valueAttr = `value="${it.value}"`;
      } else {
        inputType = 'number';
        stepAttr  = `step="${it.step}"`;
        valueAttr = `value="${it.value}"`;
      }
      // Render warning banner for bool fields whose desc starts with
      // "WARNING:" -- highlights the dangerous prev-anchor toggle.
      const isWarning = (it.kind === 'bool'
                         && (it.desc || '').toUpperCase().includes('WARNING'));
      const warnHtml = isWarning
        ? `<span style="background:#3a1010; color:#fca5a5;
                        padding:.1rem .4rem; border-radius:3px;
                        font-size:.7rem; margin-left:.4rem;">⚠ DANGEROUS</span>`
        : '';
      row.innerHTML = `
        <span class="tune-label">${it.label} ${infoHtml}${warnHtml}
          <span style="color:#6b7280; font-size:.75rem;">(${it.name})</span></span>
        <input class="tune-input" type="${inputType}" data-name="${it.name}"
               data-kind="${it.kind}" ${stepAttr} ${valueAttr}
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
    const isBool = inp.dataset.kind === 'bool';
    const fieldVal = () => isBool
      ? (inp.checked ? 'true' : 'false')
      : String(inp.value);
    inp.addEventListener(isBool ? 'change' : 'input', () => {
      const name = inp.dataset.name;
      const def = String(TUNE_FIELDS[name].default);
      if (fieldVal() !== def) inp.classList.add('dirty');
      else inp.classList.remove('dirty');
      TUNE_DIRTY.add(name);
      // Bools commit on change immediately; non-bools debounce.
      if (isBool) applyFieldNow(name);
      else        scheduleAutoApply(name);
    });
    if (!isBool) {
      // Also commit immediately on blur / Enter (cancels any pending
      // debounce so the user gets fast feedback when leaving the field).
      inp.addEventListener('change', () => applyFieldNow(inp.dataset.name));
    }
  });
  document.querySelectorAll('.tune-resetbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.name;
      const inp = document.querySelector(`.tune-input[data-name="${name}"]`);
      if (!inp) return;
      const def = TUNE_FIELDS[name].default;
      if (inp.dataset.kind === 'bool') {
        inp.checked = !!def;
      } else {
        inp.value = def;
      }
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
function _inputApplyValue(inp) {
  // Booleans send "true"/"false"; everything else sends the raw input
  // value (server's update_from_dict casts based on the dataclass
  // field type).
  if (inp.dataset.kind === 'bool') {
    return inp.checked ? 'true' : 'false';
  }
  return inp.value;
}
async function applyFieldNow(name) {
  if (applyTimers.has(name)) {
    clearTimeout(applyTimers.get(name));
    applyTimers.delete(name);
  }
  const inp = document.querySelector(`.tune-input[data-name="${name}"]`);
  if (!inp) return;
  const v = _inputApplyValue(inp);
  const body = {}; body[name] = v;
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
    setStatus(name + ' = ' + v + ' applied to running controller.', true);
  } catch (e) {
    setStatus('Apply ' + name + ' failed: ' + e, false);
  }
}

function collectValues() {
  const out = {};
  document.querySelectorAll('.tune-input').forEach(inp => {
    out[inp.dataset.name] = _inputApplyValue(inp);
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
      renderDefaultMissingBanner('snap-default-banner', j.default,
        (j.snapshots || []).some(x => x.name === j.default),
        () => clearDefault('/api/tune/snapshots', refreshSnapshots));
      return;
    }
    for (const s of j.snapshots) {
      const tr = document.createElement('tr');
      tr.style.borderTop = '1px dashed #2a3038';
      if (s.is_default) tr.style.background = 'rgba(22, 163, 74, 0.08)';
      const nameTd = document.createElement('td');
      nameTd.style.padding = '.3rem .4rem';
      nameTd.style.color = '#e6edf3';
      nameTd.textContent = s.name;
      if (s.is_default) appendDefaultBadge(nameTd);
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
      const defBtn = makeSetDefaultButton(s.is_default, async () => {
        await fetch('/api/tune/snapshots/' + encodeURIComponent(s.name)
          + '/set-default',
          {method:'POST',
           headers:{'Content-Type':'application/json'},
           body: JSON.stringify(s.is_default ? {clear: true} : {})});
        await refreshSnapshots();
      });
      const delBtn = document.createElement('button');
      delBtn.textContent = 'Delete';
      delBtn.type = 'button';
      delBtn.style.cssText = 'padding:.2rem .55rem; border:1px solid #7f1d1d; '
        + 'border-radius:4px; background:transparent; color:#fca5a5; cursor:pointer; '
        + 'font-size:.8rem;';
      delBtn.addEventListener('click', () => deleteSnap(s.name));
      actTd.appendChild(loadBtn);
      actTd.appendChild(overBtn);
      actTd.appendChild(defBtn);
      actTd.appendChild(delBtn);
      tr.appendChild(nameTd);
      tr.appendChild(timeTd);
      tr.appendChild(actTd);
      tb.appendChild(tr);
    }
    renderDefaultMissingBanner('snap-default-banner', j.default,
      j.snapshots.some(x => x.name === j.default),
      () => clearDefault('/api/tune/snapshots', refreshSnapshots));
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
})();
</script>
""" + _PAGE_GRID_CLOSE + _SHARED_SCRIPT


# ---------------------------------------------------------------------------
# Arena tab: define the world layout + write the active config
# ---------------------------------------------------------------------------

_PAGE_ARENA = _PAGE_BASE_CSS + _PAGE_HEADER + _COMMON_SCRIPT + _PAGE_GRID_OPEN + """
  <section class="grid" style="grid-template-columns: minmax(0,1fr) minmax(0,1.4fr); gap:1rem;">

    <div class="card">
      <h2>Arena setup</h2>
      <table style="width:100%;">
        <tr><th>Width (front/back wall length)</th>
            <td><input id="ar-width"  type="number" step="0.01" style="width:8em;"> m</td></tr>
        <tr><th>Depth (left/right wall length)</th>
            <td><input id="ar-depth"  type="number" step="0.01" style="width:8em;"> m</td></tr>
        <tr><th>Top marker height</th>
            <td><input id="ar-topz"   type="number" step="0.01" style="width:8em;"> m</td></tr>
        <tr><th>Bottom marker height</th>
            <td><input id="ar-botz"   type="number" step="0.01" style="width:8em;"> m</td></tr>
        <tr><th>Marker size</th>
            <td><input id="ar-msize"  type="number" step="0.001" style="width:8em;"> m</td></tr>
        <tr><th>Magnetic north (arena yaw) <span class="tune-info"
                title="Arena-frame yaw (CW from front wall / +y) at which magnetic north points. Once set, the IPPE branch picker uses tel.yaw + offset to disambiguate the +/-15 deg planar-pose flip during APPROACH and across the per-marker world-position vote. Leave empty to keep the geometry-only fallback. Press &quot;Capture&quot; while the drone is reporting a fresh world position derived from at least 2 markers." tabindex="0">ⓘ</span></th>
            <td>
              <input id="ar-magnorth" type="number" step="0.1"
                     placeholder="uncalibrated"
                     style="width:8em;"> deg
              <button id="btn-ar-magcap" type="button"
                      title="Capture from current state. Requires fresh world position from >= 2 markers + valid tel.yaw."
                      style="margin-left:.4rem; padding:.2rem .55rem;
                             border:1px solid #2a3038; border-radius:4px;
                             background:#1f2937; color:#e6edf3;
                             cursor:pointer; font-size:.8rem;">Capture</button>
              <span id="ar-magcap-msg" style="margin-left:.4rem;
                    font-size:.78rem; color:#aab;"></span>
            </td></tr>
      </table>
      <div style="display:flex; gap:.5rem; margin-top:.6rem; flex-wrap:wrap;">
        <button id="btn-ar-reset" type="button"
                style="padding:.45rem .8rem; border:0; border-radius:5px;
                       background:var(--warn); color:#241a07; font-weight:600;
                       cursor:pointer; font-size:.85rem;">Reset to default</button>
        <button id="btn-ar-save" type="button"
                style="padding:.45rem .8rem; border:0; border-radius:5px;
                       background:var(--good); color:#072413; font-weight:600;
                       cursor:pointer; font-size:.85rem;">Save active</button>
        <span id="ar-msg" style="font-size:.85rem; color:#aab; align-self:center;"></span>
      </div>
      <div id="ar-err" style="color:var(--bad); font-size:.85rem;
                              margin-top:.4rem; min-height:1.1rem;"></div>
    </div>

    <div class="card">
      <h2>Top-down view (+y = front, +x = right)</h2>
      <canvas id="c-arena" width="700" height="700"
              style="background:#0c0f12; border-radius:6px;
                     width:100%; height:auto;"></canvas>
    </div>

  </section>

  <section class="card" style="margin-top:1rem;">
    <h2>Markers</h2>
    <table id="ar-markers" style="width:100%; font-variant-numeric: tabular-nums;">
      <thead>
        <tr style="color:#aab; font-weight:500;">
          <th style="width:3em;">id</th>
          <th style="width:5em;">wall</th>
          <th>x [m]</th>
          <th>y [m]</th>
          <th>z [m]</th>
          <th>label</th>
          <th style="width:3em;"></th>
        </tr>
      </thead>
      <tbody id="ar-rows"></tbody>
    </table>
    <div style="margin-top:.5rem;">
      <button id="btn-ar-add" type="button"
              style="padding:.35rem .7rem; border:0; border-radius:5px;
                     background:var(--accent); color:#062633; font-weight:600;
                     cursor:pointer; font-size:.85rem;">+ Add marker</button>
    </div>
  </section>

  <section class="card" style="margin-top:1rem;">
    <details>
      <summary style="cursor:pointer; font-size:.85rem; color:#aab;">
        Saved arenas
      </summary>
      <div style="display:flex; gap:.4rem; align-items:center; margin:.5rem 0;">
        <input id="ar-save-name" type="text" placeholder="name"
               style="flex:1; padding:.35rem .5rem; background:#0c0f12;
                      color:var(--fg); border:1px solid #2a3038;
                      border-radius:5px; font-size:.85rem;">
        <button id="btn-ar-save-as" type="button"
                style="padding:.35rem .7rem; border:0; border-radius:5px;
                       background:var(--accent); color:#062633; font-weight:600;
                       cursor:pointer; font-size:.85rem;">Save as ...</button>
      </div>
      <div id="ar-default-banner" style="display:none;"></div>
      <div id="ar-list" style="font-size:.85rem;"></div>
    </details>
  </section>

<script>
(() => {
const $ = id => document.getElementById(id);
const WALL_COLORS = {front:'#facc15', right:'#4ade80',
                     back:'#58c4ff',  left:'#f87171'};
let arena = null;   // {marker_size_m,width_m,depth_m,top_z_m,bottom_z_m,markers:[...]}

function setMsg(s, isErr) {
  $('ar-msg').textContent = isErr ? '' : (s || '');
  $('ar-err').textContent = isErr ? (s || '') : '';
}
function metaFromInputs() {
  // Magnetic-north field is optional: empty input -> null (omitted
  // from the saved JSON via to_json_dict). A valid float is passed
  // through and stored in arena.magnetic_north_arena_yaw_deg.
  const mn = $('ar-magnorth').value.trim();
  const mn_val = (mn === '') ? null : parseFloat(mn);
  return {
    marker_size_m: parseFloat($('ar-msize').value) || 0,
    width_m:  parseFloat($('ar-width').value)  || 0,
    depth_m:  parseFloat($('ar-depth').value)  || 0,
    top_z_m:  parseFloat($('ar-topz').value)   || 0,
    bottom_z_m: parseFloat($('ar-botz').value) || 0,
    magnetic_north_arena_yaw_deg:
      (Number.isFinite(mn_val) ? mn_val : null),
  };
}
function inputsFromArena() {
  $('ar-msize').value = arena.marker_size_m;
  $('ar-width').value = arena.width_m;
  $('ar-depth').value = arena.depth_m;
  $('ar-topz').value  = arena.top_z_m;
  $('ar-botz').value  = arena.bottom_z_m;
  $('ar-magnorth').value =
    (typeof arena.magnetic_north_arena_yaw_deg === 'number'
     ? arena.magnetic_north_arena_yaw_deg : '');
}
function _wrap180(d) {
  return ((d + 540.0) % 360.0) - 180.0;
}
async function captureMagNorth() {
  const msg = $('ar-magcap-msg');
  msg.style.color = '#aab'; msg.textContent = 'reading state...';
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const s = await r.json();
    const ay = s.arena_yaw_deg;
    const tyaw = (s.telemetry || {}).yaw;
    const used = (s.world_position_used_markers || []).length;
    const age = s.arena_yaw_age_s;
    if (typeof ay !== 'number' || typeof tyaw !== 'number') {
      msg.style.color = 'var(--bad)';
      msg.textContent = 'no fresh arena_yaw or tel.yaw available';
      return;
    }
    if (typeof age === 'number' && age > 1.0) {
      msg.style.color = 'var(--bad)';
      msg.textContent = 'arena_yaw is stale (' + age.toFixed(1) + 's old)';
      return;
    }
    if (used < 2) {
      msg.style.color = 'var(--bad)';
      msg.textContent = 'need >= 2 markers visible (have ' + used + ')';
      return;
    }
    // offset = arena_yaw - tel.yaw, wrapped to (-180, 180].
    const offset = _wrap180(ay - tyaw);
    $('ar-magnorth').value = offset.toFixed(2);
    msg.style.color = 'var(--good)';
    msg.textContent =
      'captured ' + offset.toFixed(2) + ' deg from '
      + used + ' markers (press Save active to persist)';
  } catch (e) {
    msg.style.color = 'var(--bad)';
    msg.textContent = 'capture failed: ' + e;
  }
}
function renderRows() {
  const tbody = $('ar-rows');
  tbody.innerHTML = '';
  arena.markers.forEach((m, idx) => {
    const tr = document.createElement('tr');
    tr.style.borderTop = '1px solid #2a3038';
    function cell(html) { const td = document.createElement('td');
      td.style.padding = '.2rem .3rem'; td.innerHTML = html; return td; }
    function inputCell(value, attrs) {
      const td = document.createElement('td');
      td.style.padding = '.2rem .3rem';
      const inp = document.createElement('input');
      Object.assign(inp, attrs);
      inp.value = value;
      inp.style.cssText = 'width:6em; background:#0c0f12; color:var(--fg);'
                       + ' border:1px solid #2a3038; border-radius:4px;'
                       + ' padding:.15rem .3rem; font-size:.8rem;';
      td.appendChild(inp);
      return [td, inp];
    }
    const [tdId, inpId] = inputCell(m.id, {type:'number', step:1, min:0});
    inpId.style.width = '4em';
    inpId.addEventListener('input', () => {
      arena.markers[idx].id = parseInt(inpId.value); drawArena();
    });
    tr.appendChild(tdId);

    const tdWall = document.createElement('td');
    tdWall.style.padding = '.2rem .3rem';
    const sel = document.createElement('select');
    sel.style.cssText = 'background:#0c0f12; color:var(--fg);'
                     + ' border:1px solid #2a3038; border-radius:4px;'
                     + ' padding:.15rem .3rem; font-size:.8rem;';
    ['front','back','left','right'].forEach(w => {
      const o = document.createElement('option');
      o.value = w; o.textContent = w;
      if (w === m.wall) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => {
      arena.markers[idx].wall = sel.value;
      // Snap fixed coordinate so the marker lands on the new wall.
      const meta = metaFromInputs();
      if (sel.value === 'front') arena.markers[idx].y = +meta.depth_m / 2;
      if (sel.value === 'back')  arena.markers[idx].y = -meta.depth_m / 2;
      if (sel.value === 'right') arena.markers[idx].x = +meta.width_m / 2;
      if (sel.value === 'left')  arena.markers[idx].x = -meta.width_m / 2;
      renderRows();
      drawArena();
    });
    tdWall.appendChild(sel);
    tr.appendChild(tdWall);

    [['x', m.x], ['y', m.y], ['z', m.z]].forEach(([key, val]) => {
      const [td, inp] = inputCell(val, {type:'number', step:0.01});
      inp.addEventListener('input', () => {
        arena.markers[idx][key] = parseFloat(inp.value); drawArena();
      });
      tr.appendChild(td);
    });

    const [tdLabel, inpLabel] = inputCell(m.label || '', {type:'text'});
    inpLabel.style.width = '12em';
    inpLabel.addEventListener('input', () => {
      arena.markers[idx].label = inpLabel.value;
    });
    tr.appendChild(tdLabel);

    const tdDel = cell('');
    const btn = document.createElement('button');
    btn.textContent = '\\u2715';
    btn.style.cssText = 'padding:.15rem .4rem; border:0; border-radius:4px;'
                     + ' background:var(--bad); color:#240707; cursor:pointer;'
                     + ' font-size:.8rem;';
    btn.addEventListener('click', () => {
      arena.markers.splice(idx, 1); renderRows(); drawArena();
    });
    tdDel.appendChild(btn);
    tr.appendChild(tdDel);

    tbody.appendChild(tr);
  });
}
function drawArena() {
  const c = $('c-arena');
  if (!c || !arena) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const meta = metaFromInputs();
  const w = meta.width_m, d = meta.depth_m;
  if (w <= 0 || d <= 0) return;
  const margin = 40;
  const px_per_m = Math.min((W - 2*margin) / w, (H - 2*margin) / d);
  const cx = W / 2, cy = H / 2;
  // arena -> canvas: +x_arena -> +x_canvas; +y_arena -> -y_canvas (top of canvas).
  const ax = (xm) => cx + xm * px_per_m;
  const ay = (ym) => cy - ym * px_per_m;
  // 1 m background grid clipped to the arena rectangle. Heavier line +
  // numeric label at every 5 m step so the operator can read the
  // marker positions off the diagram.
  const xMin = Math.ceil(-w/2),  xMax = Math.floor(w/2);
  const yMin = Math.ceil(-d/2),  yMax = Math.floor(d/2);
  const yTop = ay(d/2), yBot = ay(-d/2);
  const xLeft = ax(-w/2), xRight = ax(w/2);
  // 1 m grid -- very subtle.
  ctx.strokeStyle = '#1d2229'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let xi = xMin; xi <= xMax; xi++) {
    if (xi % 5 === 0) continue;          // 5 m step drawn separately
    const xp = ax(xi);
    ctx.moveTo(xp, yTop); ctx.lineTo(xp, yBot);
  }
  for (let yi = yMin; yi <= yMax; yi++) {
    if (yi % 5 === 0) continue;
    const yp = ay(yi);
    ctx.moveTo(xLeft, yp); ctx.lineTo(xRight, yp);
  }
  ctx.stroke();
  // 5 m grid -- more visible (matches the old centre-lines colour, so
  // the origin lines stay subtly highlighted as part of the 5 m grid).
  ctx.strokeStyle = '#2a3038'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let xi = xMin; xi <= xMax; xi++) {
    if (xi % 5 !== 0) continue;
    const xp = ax(xi);
    ctx.moveTo(xp, yTop); ctx.lineTo(xp, yBot);
  }
  for (let yi = yMin; yi <= yMax; yi++) {
    if (yi % 5 !== 0) continue;
    const yp = ay(yi);
    ctx.moveTo(xLeft, yp); ctx.lineTo(xRight, yp);
  }
  ctx.stroke();
  // 5 m labels: x along the bottom (just below the back wall), y along
  // the left edge (just left of the left wall). Skip 0 -- the origin
  // marker (drawn later) already labels it.
  ctx.fillStyle = '#778'; ctx.font = '10px ui-sans-serif';
  ctx.textAlign = 'center';
  for (let xi = xMin; xi <= xMax; xi++) {
    if (xi % 5 !== 0 || xi === 0) continue;
    ctx.fillText(`${xi} m`, ax(xi), yBot + 32);
  }
  ctx.textAlign = 'right';
  for (let yi = yMin; yi <= yMax; yi++) {
    if (yi % 5 !== 0 || yi === 0) continue;
    ctx.fillText(`${yi} m`, xLeft - 22, ay(yi) + 3);
  }
  ctx.textAlign = 'center';
  // arena rectangle
  ctx.strokeStyle = '#aab'; ctx.lineWidth = 2;
  ctx.strokeRect(ax(-w/2), ay(d/2), w * px_per_m, d * px_per_m);
  // wall labels
  ctx.fillStyle = '#aab'; ctx.font = '13px ui-sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('front (+y)', cx, ay(d/2) - 8);
  ctx.fillText('back (-y)',  cx, ay(-d/2) + 18);
  ctx.save();
  ctx.translate(ax(-w/2) - 14, cy); ctx.rotate(-Math.PI/2);
  ctx.fillText('left (-x)', 0, 0); ctx.restore();
  ctx.save();
  ctx.translate(ax( w/2) + 14, cy); ctx.rotate(Math.PI/2);
  ctx.fillText('right (+x)', 0, 0); ctx.restore();
  // coordinate system at the origin: small yellow +x and +y arrows with labels.
  const AXIS_LEN_PX = Math.min(50, px_per_m * 1.0);   // ~1 m, capped at 50 px
  const HEAD = 6;
  function drawArrow(x1, y1, x2, y2, label, labelDx, labelDy) {
    ctx.strokeStyle = '#facc15'; ctx.fillStyle = '#facc15';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    // arrow head
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - HEAD * Math.cos(ang - Math.PI/6),
               y2 - HEAD * Math.sin(ang - Math.PI/6));
    ctx.lineTo(x2 - HEAD * Math.cos(ang + Math.PI/6),
               y2 - HEAD * Math.sin(ang + Math.PI/6));
    ctx.closePath(); ctx.fill();
    ctx.font = 'bold 12px ui-sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x2 + labelDx, y2 + labelDy);
  }
  drawArrow(cx, cy, cx + AXIS_LEN_PX, cy, '+x', 10,  4);
  drawArrow(cx, cy, cx, cy - AXIS_LEN_PX, '+y',  0, -8);
  // origin dot
  ctx.fillStyle = '#facc15';
  ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2*Math.PI); ctx.fill();
  ctx.fillStyle = '#aab'; ctx.font = '11px ui-sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('(0, 0)', cx - 6, cy + 14);
  ctx.textAlign = 'center';
  // markers. Group by (x, y) so markers stacked vertically (same wall
  // location, different z) don't draw on top of each other -- offset
  // them along the canvas vertical with the higher-z marker drawn
  // higher on screen (smaller canvas y).
  const STACK_OFFSET_PX = 14;
  const groups = new Map();
  arena.markers.forEach((m, idx) => {
    const key = `${m.x.toFixed(2)},${m.y.toFixed(2)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(idx);
  });
  const stackOffset = new Map();
  groups.forEach(indices => {
    indices.sort((a, b) => arena.markers[b].z - arena.markers[a].z);
    const n = indices.length;
    indices.forEach((idx, i) => {
      stackOffset.set(idx, (i - (n - 1) / 2) * STACK_OFFSET_PX);
    });
  });
  arena.markers.forEach((m, idx) => {
    const x = ax(m.x);
    const y = ay(m.y) + (stackOffset.get(idx) || 0);
    ctx.fillStyle = WALL_COLORS[m.wall] || '#fff';
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = '#0e1116'; ctx.font = 'bold 10px ui-sans-serif';
    const label = String(m.id);
    ctx.fillText(label, x, y + 3.5);
    ctx.fillStyle = '#aab'; ctx.font = '10px ui-sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`(${m.x.toFixed(1)},${m.y.toFixed(1)},${m.z.toFixed(1)})`,
                 x + 9, y + 4);
    ctx.textAlign = 'center';
  });
}

async function loadArena() {
  try {
    const r = await fetch('/api/arena/active', {cache:'no-store'});
    arena = await r.json();
    inputsFromArena(); renderRows(); drawArena();
    setMsg('Loaded.');
  } catch (e) { setMsg('Load failed: ' + e, true); }
}
async function resetToDefault() {
  try {
    const r = await fetch('/api/arena/default', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(metaFromInputs())});
    arena = await r.json();
    inputsFromArena(); renderRows(); drawArena();
    setMsg('Reset to default.');
  } catch (e) { setMsg('Reset failed: ' + e, true); }
}
function arenaPayload() {
  // Pull current inputs into the payload (in case the user edited
  // the metadata but didn't yet trigger a re-render).
  const m = metaFromInputs();
  return {...m, markers: arena.markers};
}
async function saveActive() {
  try {
    const r = await fetch('/api/arena/active', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(arenaPayload())});
    const j = await r.json();
    if (j.ok) { setMsg('Saved active.'); refreshList(); }
    else setMsg('Save failed: ' + (j.error || 'unknown'), true);
  } catch (e) { setMsg('Save failed: ' + e, true); }
}
async function saveAs(name) {
  if (!name) { setMsg('Enter a name first.', true); return; }
  try {
    const r = await fetch('/api/arena/configs/' + encodeURIComponent(name),
      {method:'POST', headers:{'Content-Type':'application/json'},
       body: JSON.stringify(arenaPayload())});
    const j = await r.json();
    if (j.ok) { setMsg('Saved as ' + name + '.'); refreshList(); }
    else setMsg('Save failed: ' + (j.error || 'unknown'), true);
  } catch (e) { setMsg('Save failed: ' + e, true); }
}
async function loadNamed(name) {
  try {
    const r = await fetch('/api/arena/configs/' + encodeURIComponent(name)
                          + '/load', {method:'POST'});
    const j = await r.json();
    if (j.ok && j.config) {
      arena = j.config;
      inputsFromArena(); renderRows(); drawArena();
      setMsg('Loaded ' + name + '.');
    } else setMsg('Load failed: ' + (j.error || 'unknown'), true);
  } catch (e) { setMsg('Load failed: ' + e, true); }
}
async function deleteNamed(name) {
  try {
    const r = await fetch('/api/arena/configs/' + encodeURIComponent(name),
                          {method:'DELETE'});
    const j = await r.json();
    if (j.ok) { setMsg('Deleted ' + name + '.'); refreshList(); }
    else setMsg('Delete failed: ' + (j.error || 'unknown'), true);
  } catch (e) { setMsg('Delete failed: ' + e, true); }
}
async function refreshList() {
  try {
    const r = await fetch('/api/arena/configs', {cache:'no-store'});
    const j = await r.json();
    const list = $('ar-list');
    renderDefaultMissingBanner('ar-default-banner', j.default,
      (j.configs || []).some(x => x.name === j.default),
      () => clearDefault('/api/arena/configs', refreshList));
    if (!j.configs || !j.configs.length) {
      list.innerHTML = '<div style="color:#778;">No saved arenas.</div>'; return;
    }
    const fmtTime = mt => new Date(mt*1000).toLocaleString();
    list.innerHTML = j.configs.map(c => {
      const safe = c.name.replace(/"/g, '&quot;');
      const rowBg = c.is_default ? 'background:rgba(22,163,74,0.08);' : '';
      const badge = c.is_default
        ? '<span style="background:#16a34a; color:#062613; '
          + 'font-size:.7rem; font-weight:600; padding:.05rem .35rem; '
          + 'border-radius:3px; margin-left:.4rem;">default</span>' : '';
      const defLabel = c.is_default ? 'Clear default' : 'Set as default';
      return `
        <div style="display:flex; gap:.4rem; align-items:center;
                    padding:.25rem 0; border-top:1px solid #2a3038;${rowBg}">
          <span style="flex:1; font-family:ui-monospace, monospace;">${safe}${badge}</span>
          <span style="color:#778; font-size:.75rem;">${fmtTime(c.mtime)}</span>
          <button class="ar-load"      data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:var(--accent); color:#062633;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Load</button>
          <button class="ar-overwrite" data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:#facc15; color:#241a07;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Overwrite</button>
          <button class="ar-default" data-name="${safe}" data-isdef="${c.is_default ? '1' : '0'}"
                  style="padding:.2rem .5rem; border:1px solid #2a3038;
                         border-radius:4px; background:#1f2937; color:#e6edf3;
                         cursor:pointer; font-size:.75rem;">${defLabel}</button>
          <button class="ar-delete"    data-name="${safe}"
                  style="padding:.2rem .5rem; border:0; border-radius:4px;
                         background:var(--bad); color:#240707;
                         font-weight:600; cursor:pointer; font-size:.75rem;">Delete</button>
        </div>`;
    }).join('');
    list.querySelectorAll('.ar-load').forEach(b =>
      b.addEventListener('click', () => loadNamed(b.dataset.name)));
    list.querySelectorAll('.ar-overwrite').forEach(b =>
      b.addEventListener('click', () => saveAs(b.dataset.name)));
    list.querySelectorAll('.ar-delete').forEach(b =>
      b.addEventListener('click', () => deleteNamed(b.dataset.name)));
    list.querySelectorAll('.ar-default').forEach(b =>
      b.addEventListener('click', async () => {
        const isDef = b.dataset.isdef === '1';
        await fetch('/api/arena/configs/' + encodeURIComponent(b.dataset.name)
                    + '/set-default',
                    {method:'POST',
                     headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(isDef ? {clear: true} : {})});
        await refreshList();
      }));
  } catch (e) {}
}

['ar-width','ar-depth','ar-topz','ar-botz','ar-msize'].forEach(id => {
  const el = $(id); if (el) el.addEventListener('input', drawArena);
});
$('btn-ar-reset').addEventListener('click', resetToDefault);
$('btn-ar-save').addEventListener('click', saveActive);
$('btn-ar-magcap').addEventListener('click', captureMagNorth);
$('btn-ar-add').addEventListener('click', () => {
  const meta = metaFromInputs();
  arena.markers.push({id: (Math.max(0, ...arena.markers.map(m=>m.id))+1),
                      wall:'front', x:0, y:meta.depth_m/2,
                      z:meta.top_z_m, label:''});
  renderRows(); drawArena();
});
$('btn-ar-save-as').addEventListener('click', () =>
  saveAs($('ar-save-name').value.trim()));

loadArena();
refreshList();
})();
</script>
""" + _PAGE_GRID_CLOSE + _SHARED_SCRIPT


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
                 flight_dir_provider: Optional[Callable[[], Optional[Path]]] = None,
                 arena_holder=None,
                 api=None,
                 stream_reader=None,
                 external_app: Optional["Flask"] = None):
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
        # Optional DroneApi handle, used only by the /calibrate
        # camera-config proxy endpoints. None in view-only / replay
        # mode -- the proxy returns 503.
        self.api = api
        # MjpegStreamReader handle for the operator's "Restart camera"
        # button. None in replay mode. When the upstream MJPEG stops
        # delivering frames but doesn't close the socket, the reader
        # sits there forever; the button kicks it.
        self.stream_reader = stream_reader
        # Used to write parameter_changes.csv next to flight_log.csv.
        # mission.py rolls flight_dir between missions, so we receive
        # a callable rather than a fixed Path.
        self.flight_dir_provider = flight_dir_provider
        # Arena tab. Holder pattern so the running vision_worker picks
        # up a Save in the Arena tab without a restart. None when the
        # UI runs in view-only / replay-only mode.
        self.arena_holder = arena_holder
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
        # When ``external_app`` is provided (the combined entry point
        # in ``marker_mission/app.py``), attach our routes to that
        # Flask app instead of creating our own. ``self.owns_app`` then
        # tells :meth:`start` to skip ``app.run`` — the caller drives
        # the server. In standalone mode (existing behaviour),
        # ``external_app=None`` keeps the previous Flask-per-UI model.
        if external_app is not None:
            self.app = external_app
            self.owns_app = False
        else:
            self.app = Flask(__name__)
            self.owns_app = True
        self._register_routes()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------- active mission-script
    def _write_active_script(self, text: str) -> None:
        """Write ``text`` to the active mission-script draft path.
        Creates the parent directory if needed; raises on IO error.
        """
        from .config import ACTIVE_MISSION_SCRIPT_PATH
        ACTIVE_MISSION_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_MISSION_SCRIPT_PATH.write_text(text)

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

    @staticmethod
    def _serial_from_dirname(name: str) -> Optional[str]:
        """Trailing token of ``<timestamp>_<serial>``. Returns the
        token when it looks like a real serial, "unknown" when that
        was the explicit fallback, None when the dir name doesn't
        carry one at all."""
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            return None
        return parts[1] or None

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
            # Serial preference, most-authoritative first:
            #   1) ``outcome.serial`` -- written at flight end from
            #      live telemetry, even when calibration loaded
            #      against "unknown" (drone connected late).
            #   2) directory-name suffix -- set at TAKEOFF from
            #      tel_holder.get().serial_number; same value as (1)
            #      in normal flights but available without parsing
            #      mission_meta.
            #   3) ``calibration.serial`` -- the serial we loaded
            #      calibration against at startup; stale if drone
            #      connected late.
            dir_serial = self._serial_from_dirname(d.name)
            meta_path = d / "mission_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    outcome = meta.get("outcome") or {}
                    cal = meta.get("calibration") or {}
                    entry["serial"] = (outcome.get("serial")
                                       or dir_serial
                                       or cal.get("serial") or "?")
                    entry["final_phase"] = outcome.get("final_phase")
                except Exception:
                    entry["serial"] = dir_serial or "?"
            else:
                entry["serial"] = dir_serial or "?"
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

        # Combined-mode race: marker_mission/app.py starts the Flask
        # daemon serving unified_api_server's routes BEFORE this method
        # runs. Any deploy probe (ansible's wait_for + /api/telemetry)
        # that lands in that window finalises Flask's URL map, and
        # Flask 3.x's _check_setup_finished then raises AssertionError
        # the moment we try @app.get("/mission") below. The route
        # additions themselves work fine — Flask's check is
        # precautionary about lookup-time consistency — so we reset the
        # "first request seen" latch before re-opening setup. Safe in
        # this codebase because UiServer is the only thing that
        # registers routes post-startup (unified_api_server adds all
        # its routes inside init_backend_and_threads).
        try:
            app._got_first_request = False
        except AttributeError:
            pass

        # Per-request context. Returning a fresh dict each call lets
        # the live cfg.killswitch_key + self.drone_connected propagate
        # to the rendered HTML on every page load (operators can tune
        # the killswitch key via /tune without restarting the mission).
        def killswitch_key():
            return (self.cfg.killswitch_key
                    if self.cfg and getattr(self.cfg, "killswitch_key", None)
                    else "@")

        def live_ctx(**overrides):
            # Pull current phase off MissionState so the template can
            # render the Start/Stop buttons in their correct state from
            # the very first paint -- avoids the "Start mission" green
            # button briefly flashing on page load mid-flight, which
            # the operator could click and re-trigger the controller
            # before refresh() lands the live phase from /api/state.
            current_phase = "init"
            try:
                if self.state is not None:
                    current_phase = self.state.snapshot().get("phase") or "init"
            except Exception:
                pass
            base = dict(
                active="video",
                history_s=self.history_s,
                mode="live",
                state_url="/api/state",
                video_url="/video.mjpg",
                replay_id=None,
                camera_heading="Annotated camera view",
                header_label="phase: …",
                drone_connected=self.drone_connected,
                killswitch_key=killswitch_key(),
                initial_phase=current_phase,
            )
            base.update(overrides)
            return base

        # Operator dashboard root. In standalone mode this also serves
        # at "/" so existing bookmarks keep working. In the combined
        # app (marker_mission/app.py), "/" is owned by unified_api_server
        # — only "/mission" is registered to avoid the collision.
        @app.get("/mission")
        def mission_index():
            return render_template_string(_PAGE_VIDEO, **live_ctx())
        if self.owns_app:
            app.add_url_rule("/", endpoint="index_root",
                             view_func=mission_index)

        @app.get("/charts")
        def charts():
            return render_template_string(_PAGE_CHARTS,
                                           **live_ctx(active="charts"))

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
                killswitch_key=killswitch_key(),
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
                killswitch_key=killswitch_key(),
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
                killswitch_key=killswitch_key(),
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
                state_url="/api/state",
                video_url="/video.mjpg",
                drone_connected=self.drone_connected,
                killswitch_key=killswitch_key(),
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
            from .config import SNAPSHOTS_DIR, get_default
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            default_name = get_default("tune")
            items = []
            for p in SNAPSHOTS_DIR.glob("*.json"):
                try:
                    items.append({
                        "name":  p.stem,
                        "mtime": p.stat().st_mtime,
                        "size":  p.stat().st_size,
                        "is_default": (p.stem == default_name),
                    })
                except OSError:
                    continue
            items.sort(key=lambda x: -x["mtime"])
            return jsonify({"snapshots": items, "default": default_name})

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
                # If the deleted snapshot was the pinned default,
                # clear the pointer so the UI doesn't surface a
                # "default missing" banner the operator can't reach.
                from .config import get_default, set_default
                if get_default("tune") == name:
                    set_default("tune", None)
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/tune/snapshots/<name>/set-default")
        def api_snap_set_default(name):
            from .config import get_default, set_default
            data = request.get_json(silent=True) or {}
            # Toggle: clicking on the current default clears it.
            cur = get_default("tune")
            clear = bool(data.get("clear")) or cur == name
            try:
                if clear:
                    set_default("tune", None)
                    return jsonify({"ok": True, "default": None})
                p = _snap_path(name)
                if p is None or not p.exists():
                    return jsonify({"ok": False,
                                    "error": "snapshot not found"}), 404
                set_default("tune", name)
                return jsonify({"ok": True, "default": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        # ---- Arena tab + API -------------------------------------------
        # Page route, the active draft, the default-regenerator, and
        # the named save/load CRUD. Mirrors the layout of the mission
        # script + tuning snapshot routes above.

        @app.get("/arena")
        def arena_page():
            return render_template_string(
                _PAGE_ARENA, active="arena",
                history_s=self.history_s,
                header_label="arena layout",
                replay_id=None,
                mode="live",
                state_url="/api/state",
                video_url="/video.mjpg",
                drone_connected=self.drone_connected,
                killswitch_key=killswitch_key(),
            )

        def _arena_payload_to_config(data):
            from .arena import ArenaConfig
            markers = []
            for m in (data.get("markers") or []):
                markers.append({
                    "id": int(m["id"]),
                    "label": str(m.get("label", "")),
                    "wall": str(m["wall"]).lower(),
                    "x": float(m["x"]),
                    "y": float(m["y"]),
                    "z": float(m["z"]),
                })
            payload = {
                "marker_size_m": float(data.get("marker_size_m", 0.18)),
                "width_m":       float(data.get("width_m", 10.0)),
                "depth_m":       float(data.get("depth_m", 25.0)),
                "top_z_m":       float(data.get("top_z_m", 4.0)),
                "bottom_z_m":    float(data.get("bottom_z_m", 2.0)),
                "markers":       markers,
            }
            # Optional magnetic-north calibration. Pass through if
            # present (validated inside ArenaConfig.from_dict). null
            # / missing -> uncalibrated.
            mn = data.get("magnetic_north_arena_yaw_deg")
            if mn is not None:
                payload["magnetic_north_arena_yaw_deg"] = mn
            return ArenaConfig.from_dict(payload, source="<request>")

        def _validate_arena_layout(arena_obj):
            """Return error message string, or None if OK."""
            tol = 0.01    # 1 cm tolerance for "marker on wall" check
            if arena_obj.width_m <= 0 or arena_obj.depth_m <= 0:
                return "width_m and depth_m must be > 0"
            if arena_obj.marker_size_m <= 0:
                return "marker_size_m must be > 0"
            seen_ids = set()
            half_w, half_d = arena_obj.width_m / 2, arena_obj.depth_m / 2
            for mid, m in arena_obj.markers.items():
                if mid in seen_ids:
                    return f"duplicate marker id {mid}"
                seen_ids.add(mid)
                x, y, z = m.position_m
                if z < 0:
                    return f"marker {mid}: z={z:.3f} must be >= 0"
                if m.wall == "front":
                    if abs(y - half_d) > tol:
                        return (f"marker {mid}: y={y:.3f} not on front wall"
                                f" (expected {half_d:.3f})")
                    if abs(x) > half_w + tol:
                        return f"marker {mid}: x={x:.3f} outside front wall"
                elif m.wall == "back":
                    if abs(y - (-half_d)) > tol:
                        return (f"marker {mid}: y={y:.3f} not on back wall"
                                f" (expected {-half_d:.3f})")
                    if abs(x) > half_w + tol:
                        return f"marker {mid}: x={x:.3f} outside back wall"
                elif m.wall == "left":
                    if abs(x - (-half_w)) > tol:
                        return (f"marker {mid}: x={x:.3f} not on left wall"
                                f" (expected {-half_w:.3f})")
                    if abs(y) > half_d + tol:
                        return f"marker {mid}: y={y:.3f} outside left wall"
                elif m.wall == "right":
                    if abs(x - half_w) > tol:
                        return (f"marker {mid}: x={x:.3f} not on right wall"
                                f" (expected {half_w:.3f})")
                    if abs(y) > half_d + tol:
                        return f"marker {mid}: y={y:.3f} outside right wall"
            return None

        @app.get("/api/arena/active")
        def api_arena_active_get():
            from .arena import load_priority_arena
            try:
                arena_obj = load_priority_arena()
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            return jsonify(arena_obj.to_json_dict())

        @app.post("/api/arena/active")
        def api_arena_active_set():
            data = request.get_json(silent=True) or {}
            try:
                arena_obj = _arena_payload_to_config(data)
            except Exception as e:
                return jsonify({"ok": False,
                                "error": f"bad payload: {e}"}), 400
            err = _validate_arena_layout(arena_obj)
            if err:
                return jsonify({"ok": False, "error": err}), 400
            from .config import ACTIVE_ARENA_CONFIG_PATH
            try:
                ACTIVE_ARENA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                ACTIVE_ARENA_CONFIG_PATH.write_text(
                    json.dumps(arena_obj.to_json_dict(), indent=2))
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            # Hot-reload into the running vision_worker.
            if self.arena_holder is not None:
                try:
                    self.arena_holder.set(arena_obj)
                except Exception as e:
                    print(f"[ui] arena_holder.set failed: {e}")
            return jsonify({"ok": True})

        @app.post("/api/arena/default")
        def api_arena_default():
            from .arena import default_arena
            data = request.get_json(silent=True) or {}
            try:
                arena_obj = default_arena(
                    width_m=float(data.get("width_m", 10.0)),
                    depth_m=float(data.get("depth_m", 25.0)),
                    top_z_m=float(data.get("top_z_m", 4.0)),
                    bottom_z_m=float(data.get("bottom_z_m", 2.0)),
                    marker_size_m=float(data.get("marker_size_m", 0.18)))
            except Exception as e:
                return jsonify({"error": str(e)}), 400
            return jsonify(arena_obj.to_json_dict())

        def _arena_path(name: str) -> Optional[Path]:
            if not _NAME_RE.match(name):
                return None
            from .config import ARENA_CONFIGS_DIR
            ARENA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            return ARENA_CONFIGS_DIR / f"{name}.json"

        @app.get("/api/arena/configs")
        def api_arena_list():
            from .config import ARENA_CONFIGS_DIR, get_default
            ARENA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            default_name = get_default("arena")
            items = []
            for p in ARENA_CONFIGS_DIR.glob("*.json"):
                try:
                    items.append({
                        "name":  p.stem,
                        "mtime": p.stat().st_mtime,
                        "size":  p.stat().st_size,
                        "is_default": (p.stem == default_name),
                    })
                except OSError:
                    continue
            items.sort(key=lambda x: -x["mtime"])
            return jsonify({"configs": items, "default": default_name})

        @app.post("/api/arena/configs/<name>")
        def api_arena_save_named(name):
            data = request.get_json(silent=True) or {}
            try:
                arena_obj = _arena_payload_to_config(data)
            except Exception as e:
                return jsonify({"ok": False,
                                "error": f"bad payload: {e}"}), 400
            err = _validate_arena_layout(arena_obj)
            if err:
                return jsonify({"ok": False, "error": err}), 400
            p = _arena_path(name)
            if p is None:
                return jsonify({"ok": False,
                                "error": "invalid name"}), 400
            try:
                p.write_text(json.dumps(arena_obj.to_json_dict(), indent=2))
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/arena/configs/<name>/load")
        def api_arena_load_named(name):
            from .arena import ArenaConfig
            p = _arena_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "config not found"}), 404
            try:
                return jsonify({"ok": True, "name": name,
                                "config": ArenaConfig.load(p).to_json_dict()})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.delete("/api/arena/configs/<name>")
        def api_arena_delete_named(name):
            p = _arena_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "config not found"}), 404
            try:
                p.unlink()
                from .config import get_default, set_default
                if get_default("arena") == name:
                    set_default("arena", None)
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/arena/configs/<name>/set-default")
        def api_arena_set_default(name):
            from .config import get_default, set_default
            data = request.get_json(silent=True) or {}
            cur = get_default("arena")
            clear = bool(data.get("clear")) or cur == name
            try:
                if clear:
                    set_default("arena", None)
                    return jsonify({"ok": True, "default": None})
                p = _arena_path(name)
                if p is None or not p.exists():
                    return jsonify({"ok": False,
                                    "error": "config not found"}), 404
                set_default("arena", name)
                return jsonify({"ok": True, "default": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        # ---- Camera config saves (named) ------------------------------
        # Mirrors the snapshot / arena CRUD shape. Storage:
        # ~/.marker_mission/camera_configs/{name}.json containing the
        # dict that POST /api/camera/config accepts (partial axes OK).
        # Load returns the JSON without pushing to the drone -- the
        # operator presses Apply on the form to push.
        def _camera_path(name: str) -> Optional[Path]:
            if not _NAME_RE.match(name):
                return None
            from .config import CAMERA_CONFIGS_DIR
            CAMERA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            return CAMERA_CONFIGS_DIR / f"{name}.json"

        @app.get("/api/camera/configs")
        def api_camera_configs_list():
            from .config import CAMERA_CONFIGS_DIR, get_default
            CAMERA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            default_name = get_default("camera")
            items = []
            for p in CAMERA_CONFIGS_DIR.glob("*.json"):
                try:
                    items.append({
                        "name":  p.stem,
                        "mtime": p.stat().st_mtime,
                        "size":  p.stat().st_size,
                        "is_default": (p.stem == default_name),
                    })
                except OSError:
                    continue
            items.sort(key=lambda x: -x["mtime"])
            return jsonify({"configs": items, "default": default_name})

        @app.post("/api/camera/configs/<name>")
        def api_camera_config_save(name):
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"ok": False,
                                "error": "body must be a JSON object"}), 400
            p = _camera_path(name)
            if p is None:
                return jsonify({"ok": False,
                                "error": "invalid name (alphanum, _ - . space, "
                                         "max 64 chars, can't start with separator)"}), 400
            try:
                p.write_text(json.dumps(data, indent=2))
                return jsonify({"ok": True, "name": name, "path": str(p)})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/camera/configs/<name>/load")
        def api_camera_config_load(name):
            p = _camera_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "config not found"}), 404
            try:
                return jsonify({"ok": True, "name": name,
                                "config": json.loads(p.read_text())})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.delete("/api/camera/configs/<name>")
        def api_camera_config_delete(name):
            p = _camera_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "config not found"}), 404
            try:
                p.unlink()
                from .config import get_default, set_default
                if get_default("camera") == name:
                    set_default("camera", None)
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/camera/configs/<name>/set-default")
        def api_camera_config_set_default(name):
            from .config import get_default, set_default
            data = request.get_json(silent=True) or {}
            cur = get_default("camera")
            clear = bool(data.get("clear")) or cur == name
            try:
                if clear:
                    set_default("camera", None)
                    return jsonify({"ok": True, "default": None})
                p = _camera_path(name)
                if p is None or not p.exists():
                    return jsonify({"ok": False,
                                    "error": "config not found"}), 404
                set_default("camera", name)
                return jsonify({"ok": True, "default": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        # ---- Camera stream restart -----------------------------------
        # Operator-pressed button when the camera image freezes. Cycles
        # the upstream MJPEG stream (api.video_stop -> video_start_mjpeg)
        # AND the local MjpegStreamReader thread, in that order: kill
        # the upstream first so the reader gets a clean EOF, then
        # restart it so the reader's reconnect loop has something to
        # latch onto. Returns the stream reader's stats post-restart
        # so the UI can show "frames: N" feedback.
        @app.post("/api/video/restart")
        def api_video_restart():
            errors = []
            # 1) Cycle upstream. Both calls are best-effort: if the
            # unified server is itself down, we still want the local
            # reader to bounce so the operator gets unstuck the moment
            # the upstream comes back.
            if self.api is not None:
                try:
                    self.api.video_stop()
                except Exception as e:
                    errors.append(f"video_stop: {e}")
                try:
                    self.api.video_start_mjpeg()
                except Exception as e:
                    errors.append(f"video_start_mjpeg: {e}")
            # 2) Restart local reader.
            if self.stream_reader is not None:
                try:
                    self.stream_reader.restart()
                except Exception as e:
                    errors.append(f"reader.restart: {e}")
            stats = (self.stream_reader.stats
                     if self.stream_reader is not None else None)
            return jsonify({"ok": not errors,
                            "errors": errors,
                            "reader_stats": stats})

        # ---- Camera live config passthrough (proxy to upstream) -------
        # GET hits the unified API server and returns its config JSON
        # to the calibrate page. POST forwards the form payload back.
        # Routed at /api/mission/camera/config because /api/camera/config
        # is owned by unified_api_server in the combined Flask app —
        # this rename keeps both surfaces reachable on one port. In
        # standalone mode the old path is also registered so existing
        # frontend code keeps working without a coordinated rollout.
        @app.get("/api/mission/camera/config")
        def api_camera_config_get_proxy():
            try:
                cfg = self.api.camera_config_get() if self.api else None
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 503
            if cfg is None:
                return jsonify({"ok": False,
                                "error": "drone api not wired"}), 503
            # Upstream returns {ok, config} -- pass through.
            return jsonify(cfg)

        @app.post("/api/mission/camera/config")
        def api_camera_config_set_proxy():
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"ok": False,
                                "error": "body must be a JSON object"}), 400
            try:
                resp = self.api.camera_config_set(**data) if self.api else None
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 503
            if resp is None:
                return jsonify({"ok": False,
                                "error": "drone api not wired"}), 503
            return jsonify(resp)

        # Back-compat: standalone mode also exposes the old paths so
        # existing /calibrate frontend code keeps working without a
        # coordinated URL update. Skipped in combined mode because
        # unified_api_server owns /api/camera/config there.
        if self.owns_app:
            app.add_url_rule(
                "/api/camera/config",
                endpoint="api_camera_config_get_legacy",
                view_func=api_camera_config_get_proxy,
                methods=["GET"],
            )
            app.add_url_rule(
                "/api/camera/config",
                endpoint="api_camera_config_set_legacy",
                view_func=api_camera_config_set_proxy,
                methods=["POST"],
            )

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

        # ---- Calibration file CRUD (used by marker_mission_c2 to mirror
        # .npz files into a central library so a drone can be moved to a
        # different flight controller without manual file copies). Each
        # file is name-validated against the same anafi_<serial>_<res>
        # pattern that CalibrationStore writes so a malicious filename
        # can't escape calib_dir. ----------------------------------------
        import re as _cal_re
        _CAL_NAME_RE = _cal_re.compile(
            r"^anafi_[A-Za-z0-9_\-]+_[A-Za-z0-9]+\.npz$")

        def _cal_dir() -> Path:
            base = (self.cfg.calib_dir
                    if self.cfg and getattr(self.cfg, "calib_dir", None)
                    else CALIB_DIR)
            return Path(base)

        @app.get("/api/calibrate/files")
        def api_calibrate_files_list():
            d = _cal_dir()
            entries: list[dict] = []
            if d.exists():
                for path in sorted(d.glob("anafi_*.npz")):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    entry: dict = {
                        "name": path.name,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                    # Pull scalar metadata out of the .npz; the small
                    # fields we read are loaded directly without
                    # bringing the calibration matrix into memory.
                    try:
                        with np.load(path, allow_pickle=False) as npz:
                            if "serial" in npz:
                                entry["serial"] = str(npz["serial"])
                            if "resolution" in npz:
                                entry["resolution"] = str(npz["resolution"])
                            if "rms_error" in npz:
                                entry["rms_error"] = float(npz["rms_error"])
                            if "image_size" in npz:
                                entry["image_size"] = [
                                    int(v) for v in npz["image_size"]]
                            if "calibrated_at" in npz:
                                entry["calibrated_at"] = str(
                                    npz["calibrated_at"])
                            if "notes" in npz:
                                entry["notes"] = str(npz["notes"])
                    except Exception as e:
                        entry["read_error"] = str(e)
                    entries.append(entry)
            return jsonify({"files": entries})

        @app.get("/api/calibrate/files/<name>")
        def api_calibrate_files_get(name: str):
            if not _CAL_NAME_RE.match(name):
                return jsonify({"ok": False,
                                "error": "invalid filename"}), 400
            path = _cal_dir() / name
            if not path.is_file():
                return jsonify({"ok": False,
                                "error": "not found"}), 404
            return send_file(str(path),
                             mimetype="application/octet-stream")

        @app.post("/api/calibrate/files/<name>")
        def api_calibrate_files_put(name: str):
            if not _CAL_NAME_RE.match(name):
                return jsonify({"ok": False,
                                "error": "invalid filename"}), 400
            blob = request.get_data(cache=False, as_text=False)
            if not blob:
                return jsonify({"ok": False,
                                "error": "empty body"}), 400
            if len(blob) > 1_048_576:
                return jsonify({"ok": False,
                                "error": "calibration file > 1 MiB"}), 413
            # Validate as a NumPy .npz with the required arrays before
            # writing — rejects garbage at the door so a corrupt push
            # never replaces a working calibration.
            import io as _io
            try:
                with np.load(_io.BytesIO(blob), allow_pickle=False) as npz:
                    for k in ("camera_matrix", "dist_coeffs", "image_size"):
                        if k not in npz:
                            return jsonify({"ok": False,
                                            "error": f"npz missing {k}"}), 400
            except Exception as e:
                return jsonify({"ok": False,
                                "error": f"not a valid npz: {e}"}), 400
            d = _cal_dir()
            d.mkdir(parents=True, exist_ok=True)
            target = d / name
            tmp = target.with_suffix(target.suffix + ".tmp")
            try:
                with open(tmp, "wb") as f:
                    f.write(blob)
                    f.flush()
                    try:
                        import os as _os
                        _os.fsync(f.fileno())
                    except OSError:
                        pass
                import os as _os
                _os.replace(tmp, target)
            except OSError as e:
                return jsonify({"ok": False,
                                "error": f"write failed: {e}"}), 500
            return jsonify({"ok": True, "name": name,
                            "path": str(target), "size": len(blob)})

        @app.delete("/api/calibrate/files/<name>")
        def api_calibrate_files_delete(name: str):
            if not _CAL_NAME_RE.match(name):
                return jsonify({"ok": False,
                                "error": "invalid filename"}), 400
            path = _cal_dir() / name
            if not path.exists():
                return jsonify({"ok": False, "error": "not found"}), 404
            try:
                path.unlink()
            except OSError as e:
                return jsonify({"ok": False,
                                "error": f"delete failed: {e}"}), 500
            return jsonify({"ok": True})

        # ---- Identity: which physical drone is currently on this FC.
        # Used by marker_mission_c2 to (a) show "drone serial" in the
        # per-FC card and (b) warn when an operator pushes a library
        # calibration whose serial doesn't match this FC's drone.
        @app.get("/api/identity")
        def api_identity():
            import socket as _socket
            serial = None
            connected = False
            try:
                snap = self.state.snapshot()
                tel = snap.get("telemetry") or {}
                serial = (tel.get("serial_number") or tel.get("serial")
                          or None)
                connected = bool(snap.get("drone_connected")
                                  or self.drone_connected)
            except Exception:
                pass
            return jsonify({
                "hostname": _socket.gethostname(),
                "drone_serial": serial,
                "drone_connected": connected,
                "marker_mission_version": "1.x",
            })

        @app.post("/api/start")
        def api_start():
            if self.on_start is None:
                return jsonify({"ok": False,
                                "error": "no start handler registered"}), 500
            # Resolve the script in this priority order:
            #   1. ``script`` in the request body (Camera page sends the
            #      live textarea content here, including any unflushed
            #      edits within the autosave debounce).
            #   2. Active draft on disk (~/.marker_mission/active_mission_script.txt).
            #      The Camera-page editor autosaves to this file on every
            #      keystroke. Sidebars on /charts /tune /arena /calibrate
            #      have a Start button but no editor; they hit /api/start
            #      with no script body, and we want them to launch the
            #      operator's saved script -- NOT the hardcoded default.
            #   3. (No load) -- controller.trigger() falls back to the
            #      hardcoded default script.
            data = request.get_json(silent=True) or {}
            script_text = data.get("script") if isinstance(data, dict) else None
            if not script_text:                      # None or "" -> try draft
                from .mission_script import load_priority_script
                from .config import ACTIVE_MISSION_SCRIPT_PATH
                try:
                    script_text = load_priority_script(
                        ACTIVE_MISSION_SCRIPT_PATH, self.flights_root)
                except Exception as e:
                    print(f"[ui] active-draft load failed: {e}")
                    script_text = None
            if script_text and self.controller is not None and self.cfg is not None:
                from .mission_script import (parse as parse_script,
                                              defaults_from_cfg, ScriptError)
                try:
                    steps = parse_script(script_text,
                                         defaults_from_cfg(self.cfg))
                except ScriptError as e:
                    return jsonify({"ok": False,
                                    "error": "script parse error",
                                    "script_error": str(e)}), 400
                self.controller.set_script(steps)
                # Only write the active draft if the body carried it --
                # don't overwrite the file with what we just read from it.
                if data.get("script"):
                    try:
                        self._write_active_script(data["script"])
                    except Exception as e:
                        print(f"[ui] active draft save failed: {e}")
            try:
                started = bool(self.on_start())
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            if not started:
                return jsonify({"ok": False,
                                "error": "mission already started or not in INIT"}), 409
            return jsonify({"ok": True})

        @app.get("/api/mission/script")
        def api_mission_script_get():
            from .mission_script import load_priority_script
            from .config import ACTIVE_MISSION_SCRIPT_PATH
            text = load_priority_script(ACTIVE_MISSION_SCRIPT_PATH,
                                         self.flights_root)
            return jsonify({"text": text})

        @app.post("/api/mission/script")
        def api_mission_script_set():
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict) or "text" not in data:
                return jsonify({"ok": False,
                                "error": "missing 'text' field"}), 400
            try:
                self._write_active_script(str(data["text"]))
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            return jsonify({"ok": True})

        _SCRIPT_NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\. ]{0,63}$")

        def _script_path(name: str) -> Optional[Path]:
            if not _SCRIPT_NAME_RE.match(name):
                return None
            from .config import MISSION_SCRIPTS_DIR
            MISSION_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            return MISSION_SCRIPTS_DIR / f"{name}.txt"

        @app.get("/api/mission/scripts")
        def api_mission_scripts_list():
            from .config import MISSION_SCRIPTS_DIR, get_default
            MISSION_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            default_name = get_default("mission_script")
            items = []
            for p in MISSION_SCRIPTS_DIR.glob("*.txt"):
                try:
                    items.append({
                        "name":  p.stem,
                        "mtime": p.stat().st_mtime,
                        "size":  p.stat().st_size,
                        "is_default": (p.stem == default_name),
                    })
                except OSError:
                    continue
            items.sort(key=lambda x: -x["mtime"])
            return jsonify({"scripts": items, "default": default_name})

        @app.post("/api/mission/scripts/<name>")
        def api_mission_script_save(name):
            data = request.get_json(silent=True) or {}
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str):
                return jsonify({"ok": False,
                                "error": "missing 'text' field"}), 400
            p = _script_path(name)
            if p is None:
                return jsonify({"ok": False,
                                "error": "invalid name (alphanum, _ - . space, "
                                         "max 64 chars, can't start with separator)"}), 400
            try:
                p.write_text(text)
                return jsonify({"ok": True, "name": name, "path": str(p)})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/mission/scripts/<name>/load")
        def api_mission_script_load(name):
            p = _script_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "script not found"}), 404
            try:
                return jsonify({"ok": True, "name": name,
                                "text": p.read_text()})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.delete("/api/mission/scripts/<name>")
        def api_mission_script_delete(name):
            p = _script_path(name)
            if p is None or not p.exists():
                return jsonify({"ok": False,
                                "error": "script not found"}), 404
            try:
                p.unlink()
                from .config import get_default, set_default
                if get_default("mission_script") == name:
                    set_default("mission_script", None)
                return jsonify({"ok": True, "name": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.post("/api/mission/scripts/<name>/set-default")
        def api_mission_script_set_default(name):
            from .config import get_default, set_default
            data = request.get_json(silent=True) or {}
            cur = get_default("mission_script")
            clear = bool(data.get("clear")) or cur == name
            try:
                if clear:
                    set_default("mission_script", None)
                    return jsonify({"ok": True, "default": None})
                p = _script_path(name)
                if p is None or not p.exists():
                    return jsonify({"ok": False,
                                    "error": "script not found"}), 404
                set_default("mission_script", name)
                return jsonify({"ok": True, "default": name})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

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
            # Optional ?fps=N to cap the MJPEG output rate (clamped to
            # 1..30). Useful for the C2 dashboard when six streams
            # are running side-by-side and the browser's decode +
            # paint pipeline starts dropping work. Default ≈20 fps
            # preserves the historical behaviour for the FC's own
            # /mission page. We also de-dup: if no new frame has
            # arrived since the last send, sleep+continue rather than
            # re-emit the same JPEG bytes (so a slow drone doesn't
            # cause us to flood the wire with stale duplicates at the
            # cap rate).
            fps_raw = request.args.get("fps", "20")
            try:
                fps = float(fps_raw)
            except ValueError:
                fps = 20.0
            fps = max(1.0, min(30.0, fps))
            min_interval = 1.0 / fps

            def gen():
                last_ts = 0.0
                last_send = 0.0
                while True:
                    now = time.monotonic()
                    wait = (last_send + min_interval) - now
                    if wait > 0:
                        time.sleep(min(wait, 0.05))
                        continue
                    jpg, ts = self.frame.get()
                    if jpg and ts != last_ts:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg + b"\r\n")
                        last_ts = ts
                        last_send = time.monotonic()
                    else:
                        # No fresh frame from the drone yet — short
                        # sleep before polling self.frame again.
                        time.sleep(0.02)
            return Response(gen(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.get("/team_logo.png")
        def team_logo():
            # Repo asset, lives next to the package -- ../1_Doc/...
            # Browsers cache this for an hour so the 2.8 MB transfer
            # only happens on first load per session.
            p = (Path(__file__).resolve().parent.parent
                 / "1_Doc" / "team_logo_transparent.png")
            if not p.is_file():
                return ("logo not found", 404)
            resp = send_file(str(p), mimetype="image/png")
            resp.headers["Cache-Control"] = "public, max-age=3600"
            return resp

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.owns_app:
            # Combined mode: the embedding entry point (marker_mission/
            # app.py) owns the Flask lifecycle. Our routes are already
            # registered on its app from __init__; there's nothing
            # else for us to start.
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
