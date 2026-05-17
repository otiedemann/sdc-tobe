"""Shared HTML fragments: base CSS, topbar (incl. emergency-land button +
connection dots), and the global keydown handler.

The keydown handler is the one place that knows the configured
``emergency_land_key`` — both the top button and the key fire the same
``fetch('/api/c2/emergency-land-all')`` so emergency-land has exactly
one code path.
"""
from __future__ import annotations

# Inline style — matches marker_mission/ui.py's _PAGE_BASE_CSS pattern
# so both surfaces read the same way and stay visually consistent.
_BASE_CSS = """
<style>
  :root { --fg:#e8eaee; --bg:#0f1115; --card:#161a20; --accent:#7dd3fc;
          --good:#86efac; --bad:#f87171; --warn:#facc15; }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); margin: 0;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                      Oxygen, Ubuntu, sans-serif; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  header { background: var(--card); padding: .55rem 1rem; display: flex;
           align-items: center; gap: .9rem; flex-wrap: wrap; }
  header h1 { font-size: 1.05rem; margin: 0; font-weight: 600; }
  nav { display:flex; gap:.5rem; flex-wrap:wrap; }
  nav a { padding:.35rem .7rem; border-radius:5px; color:#aab;
          background:#0c0f12; font-size:.85rem; }
  nav a.active { color:#062633; background:var(--accent); font-weight:600; }
  main { padding: 1rem; }
  .card { background: var(--card); border-radius: 8px; padding: 1rem; }
  .card h2 { margin: 0 0 .5rem 0; font-size: .85rem; font-weight: 500;
             color: #aab; text-transform: uppercase; letter-spacing: .03em; }
  .grid { display: grid; gap: 1rem; }
  table { width: 100%; border-collapse: collapse;
          font-variant-numeric: tabular-nums; }
  td, th { padding: .25rem .5rem; text-align: left; }
  th { color: #aab; font-weight: 500; }
  tr + tr td, tr + tr th { border-top: 1px solid #2a3038; }
  input, textarea, select, button { font: inherit; color: var(--fg); }
  input[type=text], input[type=number], textarea, select {
      background: #0c0f12; border: 1px solid #2a3038;
      border-radius: 5px; padding: .35rem .5rem; }
  textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
                          monospace; font-size: .85rem; }
  button { cursor: pointer; }
  .btn { padding: .4rem .8rem; border: 0; border-radius: 6px;
         background: var(--accent); color: #062633; font-weight: 600;
         font-size: .85rem; }
  .btn-good { background: var(--good); color: #072413; }
  .btn-bad { background: var(--bad); color: #240707; }
  .btn-warn { background: var(--warn); color: #2a1d00; }
  .btn-ghost { background: #0c0f12; color: var(--fg);
               border: 1px solid #2a3038; }
  .dot { display:inline-block; width:.65rem; height:.65rem;
         border-radius:50%; vertical-align:middle; }
  .dot-on { background: var(--good); }
  .dot-off { background: var(--bad); }
  .dot-stale { background: #6b7280; }
  /* Emergency-land button: big, red, hard to miss, never disabled. */
  .btn-emergency { background:#dc2626; color:#fff; border:0;
                    padding:.7rem 1.1rem; border-radius:8px;
                    font-weight:700; font-size:1rem; cursor:pointer;
                    box-shadow:0 1px 0 #7f1d1d inset, 0 0 0 1px #991b1b;
                    letter-spacing:.02em; margin-left:.5rem; }
  .btn-emergency:hover { background:#ef4444; }
  /* Start-all button: green (fleet-level positive action), confirms
     before firing because it launches every drone you have. */
  .btn-start-all { background:#16a34a; color:#062613; border:0;
                    padding:.7rem 1.1rem; border-radius:8px;
                    font-weight:700; font-size:1rem; cursor:pointer;
                    box-shadow:0 1px 0 #14532d inset, 0 0 0 1px #166534;
                    letter-spacing:.02em; }
  .btn-start-all:hover { background:#22c55e; }
  /* Compact per-FC connection-dot strip in the topbar. */
  .fc-dots { display:flex; gap:.35rem; align-items:center;
              font-size:.78rem; color:#aab; }
  .fc-dots .item { display:inline-flex; align-items:center; gap:.25rem;
                    padding:.15rem .4rem; border-radius:4px;
                    background:#0c0f12; }
  /* Card flag: per-FC mission cards on the overview. CSS containment
     scopes layout + paint cost to the card so a single video frame
     never triggers a full-page reflow — critical when six MJPEG
     streams are decoding in parallel. */
  .fc-card { background: var(--card); border-radius: 8px;
              padding: .85rem; min-width: 0;
              contain: layout paint style;
              content-visibility: auto;
              contain-intrinsic-size: 480px; }
  .fc-card header { background:transparent; padding:0; margin-bottom:.5rem;
                     gap:.5rem; }
  .fc-card h3 { margin:0; font-size:1rem; font-weight:600;
                  display:flex; align-items:center; gap:.4rem; }
  /* Video slot: capped at 480 px so decode buffers stay small even
     when the FC delivers 1280×720 — the operator can pop the card's
     native UI for full-res. min-height keeps the card layout stable
     when video flips on/off. */
  .video-slot { background:#000; border-radius:6px; min-height:160px;
                 display:flex; align-items:center; justify-content:center;
                 color:#444; font-size:.8rem; margin:.5rem 0;
                 max-width:480px;
                 contain: layout paint style; }
  .video-slot img { width:100%; height:auto; display:block;
                     border-radius:6px; background:#000; }
  .stat-grid { display:grid; grid-template-columns: auto 1fr;
                gap:.15rem .55rem; font-size:.83rem;
                font-variant-numeric: tabular-nums; }
  .stat-grid dt { color:#aab; }
  .stat-grid dd { margin:0; }
  .pill { display:inline-block; padding:.05rem .4rem; border-radius:4px;
           font-size:.75rem; font-weight:600; }
  .pill-good { background:#0d3a1f; color:var(--good); }
  .pill-bad { background:#3a0d0d; color:var(--bad); }
  .pill-warn { background:#3a2d0d; color:var(--warn); }
  .pill-neutral { background:#1f2937; color:#aab; }
  details > summary { cursor:pointer; color:#aab; font-size:.85rem; }
  /* Emergency banner (added by JS at the top of the page). */
  .em-banner { position:fixed; top:0; left:0; right:0; z-index:9999;
                background:#dc2626; color:#fff; text-align:center;
                padding:.7rem; font-weight:700; font-size:1rem; }
  .em-banner.partial { background:#f59e0b; color:#1f1300; }
  .em-banner.ok { background:#16a34a; color:#062613; }
</style>
"""

# Header (every page). The topbar renders the connection-dot strip
# from the server-side FC list so the dots are present before the first
# JS poll lands. JS replaces the dot classes on each refresh.
_HEADER_HTML = """
<header>
  <img src="/team_logo.png" alt="team logo"
       style="width:64px; height:64px; object-fit:contain;
              display:block; margin-right:.6rem;">
  <h1 style="line-height:1.15;">to be defined<br>C2 mission control</h1>
  <nav>
    <a href="/" class="{{ 'active' if active=='overview' else '' }}">Overview</a>
    <a href="/arena" class="{{ 'active' if active=='arena' else '' }}">Arena</a>
    <a href="/tune" class="{{ 'active' if active=='tune' else '' }}">Tune</a>
    <a href="/calibrate" class="{{ 'active' if active=='calibrate' else '' }}">Calibrate</a>
    <a href="/scripts" class="{{ 'active' if active=='scripts' else '' }}">Scripts</a>
    <a href="/settings" class="{{ 'active' if active=='settings' else '' }}">Settings</a>
  </nav>
  <span class="fc-dots" id="topbar-dots">
    {% for fc in fc_specs %}
    <span class="item" data-fc="{{ fc.name }}">
      <span class="dot dot-stale" data-role="dot"></span>{{ fc.name }}
    </span>
    {% endfor %}
  </span>
  <button type="button" id="btn-start-all"
          class="btn-start-all" style="margin-left:auto;"
          title="Start every drone using its own active mission script">
    ▶ Start all
  </button>
  <button type="button" id="btn-emergency-land"
          class="btn-emergency"
          title="Land all configured drones (key: {{ emergency_land_key }})">
    EMERGENCY LAND ALL ({{ emergency_land_key }})
  </button>
</header>
"""

# Page-global script — runs on every page. Owns the keybinding, the
# emergency-land beep + banner + fan-out, and the topbar dot updates.
_COMMON_SCRIPT = """
<script>
const EMERGENCY_KEY = {{ emergency_land_key|tojson }};
const UI_REFRESH_MS = {{ ui_refresh_ms }};
const FC_NAMES = {{ fc_names|tojson }};

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
    g.gain.setValueAtTime(0.22, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + durMs / 1000);
    o.start(t0); o.stop(t0 + durMs / 1000 + 0.05);
  } catch (e) {}
}

function _emergencyBanner(text, kind) {
  let banner = document.getElementById('em-banner-live');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'em-banner-live';
    banner.className = 'em-banner';
    document.body.appendChild(banner);
  }
  banner.textContent = text || 'EMERGENCY LAND \\u2014 fanning out';
  banner.className = 'em-banner' + (kind ? ' ' + kind : '');
  banner.style.display = 'block';
  setTimeout(() => { if (banner) banner.style.display = 'none'; }, 8000);
}

function _emergencyResults(results) {
  // results is { fc_name: {ok, noop?, error?}, ... }
  let ok = 0, noop = 0, fail = 0;
  const failed = [];
  for (const name of Object.keys(results || {})) {
    const r = results[name] || {};
    if (r.ok && r.noop) { noop++; ok++; }
    else if (r.ok) ok++;
    else { fail++; failed.push(name); }
  }
  let text;
  let kind = 'ok';
  if (fail === 0) {
    text = 'EMERGENCY LAND acknowledged by ' + ok + '/' + (ok + fail) +
           (noop ? ' (' + noop + ' already idle)' : '');
  } else {
    text = 'EMERGENCY LAND: ' + ok + ' ok, ' + fail + ' failed (' +
           failed.join(', ') + ')';
    kind = 'partial';
  }
  _emergencyBanner(text, kind);
}

function fireEmergencyLand() {
  _emergencyBanner();
  _beep(180, 550);
  fetch('/api/c2/emergency-land-all', {method: 'POST'})
    .then(r => r.json())
    .then(j => _emergencyResults(j))
    .catch(e => _emergencyBanner('EMERGENCY LAND request failed: ' + e,
                                  'partial'));
}

function _startAllResults(results) {
  let started = 0, noop = 0, fail = 0;
  const failed = [];
  for (const name of Object.keys(results || {})) {
    const r = results[name] || {};
    if (r.ok && r.noop) noop++;
    else if (r.ok) started++;
    else { fail++; failed.push(name); }
  }
  let text, kind = 'ok';
  if (fail === 0) {
    text = 'Start all: ' + started + ' started'
            + (noop ? ', ' + noop + ' already running' : '');
  } else {
    text = 'Start all: ' + started + ' started, ' + fail
            + ' failed (' + failed.join(', ') + ')';
    kind = 'partial';
  }
  _emergencyBanner(text, kind);
}

function fireStartAll() {
  // Confirm before launching — every drone you have will take off
  // using its own active script.
  if (!confirm('Start every drone? Each FC will run its own active '
               + 'mission script. Continue?')) return;
  _emergencyBanner('Start all: launching every drone…', 'partial');
  _beep(420, 250);
  fetch('/api/c2/start-all', {method: 'POST'})
    .then(r => r.json())
    .then(j => _startAllResults(j))
    .catch(e => _emergencyBanner('Start all request failed: ' + e,
                                  'partial'));
}

document.addEventListener('keydown', e => {
  if (!EMERGENCY_KEY) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (String(e.key || '').toUpperCase()
      !== String(EMERGENCY_KEY).toUpperCase()) return;
  // Suppress when focus is inside any input/textarea/contenteditable so
  // typing '?' into the script editor doesn't land the swarm.
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
            || t.isContentEditable)) return;
  fireEmergencyLand();
});

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('btn-emergency-land');
  if (btn) btn.addEventListener('click', fireEmergencyLand);
  const btnStart = document.getElementById('btn-start-all');
  if (btnStart) btnStart.addEventListener('click', fireStartAll);
  // Kick off the topbar-dot poller. Each page may also use the overview
  // payload for its own widgets via window._latestOverview.
  if (!window._c2OverviewPollerStarted) {
    window._c2OverviewPollerStarted = true;
    pollOverviewLoop();
  }
});

function _updateTopbarDots(overview) {
  for (const name of FC_NAMES) {
    const item = document.querySelector('.fc-dots .item[data-fc="' + name + '"]');
    if (!item) continue;
    const dot = item.querySelector('[data-role=dot]');
    const fc = (overview && overview[name]) || null;
    let cls = 'dot dot-stale';
    if (fc && fc.connection_ok) cls = 'dot dot-on';
    else if (fc) cls = 'dot dot-off';
    dot.className = cls;
    item.title = fc && fc.last_error ? (name + ': ' + fc.last_error)
                                       : (name + ': ' + (fc && fc.connection_ok ? 'ok' : 'down'));
  }
}

async function pollOverviewLoop() {
  while (true) {
    try {
      const r = await fetch('/api/c2/overview', {cache:'no-store'});
      if (r.ok) {
        const j = await r.json();
        window._latestOverview = j;
        _updateTopbarDots(j);
        document.dispatchEvent(new CustomEvent('c2:overview',
                                                {detail: j}));
      }
    } catch (e) {}
    await new Promise(res => setTimeout(res, UI_REFRESH_MS));
  }
}
</script>
"""


def head_block() -> str:
    """The first thing every page renders: CSS + the meta tag."""
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>marker_mission C2</title>'
            + _BASE_CSS + '</head><body>')


def topbar() -> str:
    return _HEADER_HTML


def common_script() -> str:
    return _COMMON_SCRIPT


def tail_block() -> str:
    return "</body></html>"
