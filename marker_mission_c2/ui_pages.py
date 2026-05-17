"""HTML for the five C2 pages.

Each constant below is a Jinja template body, sandwiched between
:func:`ui_shared.head_block`, :func:`ui_shared.topbar`,
:func:`ui_shared.common_script`, and :func:`ui_shared.tail_block` by
the route handlers.
"""
from __future__ import annotations


# ---------------------------------------------------------------- overview

PAGE_OVERVIEW = """
<main>
  <div class="card" id="arena-viz-card" style="margin-bottom:1rem;">
    <div style="display:flex; align-items:center; gap:.5rem;
                margin-bottom:.4rem; flex-wrap:wrap;">
      <h2 style="margin:0;">Arena (C2 draft) — live drone positions</h2>
      <span id="arena-viz-status" style="color:#aab; font-size:.78rem;
                                          margin-left:auto;">…</span>
      <a href="/arena" style="font-size:.78rem;">edit ↗</a>
    </div>
    <div id="arena-viz-empty"
         style="color:#aab; font-size:.85rem; padding:.6rem;
                background:#0c0f12; border-radius:6px; display:none;">
      No arena draft saved yet. Go to
      <a href="/arena">/arena</a> → Load from FC or New empty arena → Save
      as C2 draft.
    </div>
    <svg id="arena-viz" width="100%" viewBox="0 0 800 480"
         style="background:#0c0f12; border-radius:6px;
                max-height:480px;"></svg>
  </div>
  <div style="display:flex; gap:.5rem; align-items:center;
              margin-bottom:.75rem; flex-wrap:wrap;">
    <button type="button" class="btn" id="btn-video-all">
      Show all video
    </button>
    <span id="video-all-msg" style="font-size:.78rem; color:#aab;"></span>
  </div>
  <div class="grid"
       style="grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));"
       id="fc-grid">
    {% for fc in fc_specs %}
    <div class="fc-card" data-fc="{{ fc.name }}">
      <h3>
        <span class="dot dot-stale" data-role="dot"></span>
        <span>{{ fc.name }}</span>
        <span style="margin-left:auto; font-size:.75rem; color:#aab;
                     font-weight:400;">
          <a href="http://{{ fc.host }}:{{ fc.port }}/mission"
             target="_blank" rel="noopener">native UI ↗</a>
        </span>
      </h3>
      <dl class="stat-grid">
        <dt>Phase</dt>
        <dd><span class="pill pill-neutral" data-role="phase">—</span></dd>
        <dt>Mission step</dt><dd data-role="step">—</dd>
        <dt>Battery</dt><dd data-role="battery">—</dd>
        <dt>Height</dt><dd data-role="height">—</dd>
        <dt>World pos</dt><dd data-role="wpos">—</dd>
        <dt>Dist to marker</dt><dd data-role="distance">—</dd>
        <dt>Drone</dt><dd data-role="drone">—</dd>
        <dt>Serial</dt>
        <dd data-role="serial"
            style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
                   monospace; font-size:.78rem;">—</dd>
      </dl>
      <div class="video-slot" data-role="video-slot">video off</div>
      <div style="display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.4rem;">
        <button type="button" class="btn btn-ghost" data-role="video-toggle"
                data-mjpeg-url="/video/{{ fc.name }}.mjpg">
          Show video
        </button>
        <button type="button" class="btn btn-good" data-role="start">
          Start
        </button>
        <button type="button" class="btn btn-bad" data-role="stop"
                style="display:none;">
          Stop &amp; land
        </button>
        <span data-role="msg" style="font-size:.78rem; color:#aab;
                                      align-self:center;"></span>
      </div>
      <details data-role="script-details" style="margin-top:.45rem;">
        <summary>Inline script editor</summary>
        <textarea data-role="script-text" rows="6" spellcheck="false"
                  placeholder="TAKEOFF&#10;APPROACH&#10;HOOVER&#10;LAND"
                  style="width:100%; margin-top:.3rem;"></textarea>
        <div style="display:flex; gap:.4rem; margin-top:.3rem;
                    align-items:center;">
          <button type="button" class="btn btn-ghost" data-role="script-load">
            Load from FC
          </button>
          <button type="button" class="btn" data-role="script-save">
            Save to this FC
          </button>
          <span data-role="script-msg"
                style="font-size:.78rem; color:#aab;"></span>
        </div>
      </details>
    </div>
    {% endfor %}
  </div>
</main>

<script>
(function() {
  function fmt(n, digits) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return Number(n).toFixed(digits);
  }
  function phaseClass(p) {
    if (!p) return 'pill pill-neutral';
    const s = String(p).toLowerCase();
    if (s === 'abort') return 'pill pill-bad';
    if (s === 'done' || s === 'init') return 'pill pill-neutral';
    return 'pill pill-good';
  }
  function batteryClass(b) {
    if (b === null || b === undefined) return 'pill pill-neutral';
    if (b < 30) return 'pill pill-bad';
    if (b < 58) return 'pill pill-warn';
    return 'pill pill-good';
  }
  function updateCards(overview) {
    for (const name of FC_NAMES) {
      const card = document.querySelector('.fc-card[data-fc="' + name + '"]');
      if (!card) continue;
      const fc = overview[name];
      const dot = card.querySelector('[data-role=dot]');
      if (!fc) {
        dot.className = 'dot dot-stale';
        continue;
      }
      dot.className = fc.connection_ok ? 'dot dot-on'
        : (fc.last_state_age_s !== null && fc.last_state_age_s < 5 ? 'dot dot-stale' : 'dot dot-off');
      const st = fc.state || {};
      const tel = st.telemetry || {};
      const phaseEl = card.querySelector('[data-role=phase]');
      phaseEl.textContent = st.phase || '—';
      phaseEl.className = phaseClass(st.phase);
      const stepEl = card.querySelector('[data-role=step]');
      if (typeof st.mission_step_idx === 'number'
          && Array.isArray(st.mission_script) && st.mission_script.length) {
        const line = st.mission_script[st.mission_step_idx]
                     || ('step ' + st.mission_step_idx);
        stepEl.textContent = (st.mission_step_idx + 1) + '/'
                             + st.mission_script.length + ' · ' + line;
      } else {
        stepEl.textContent = '—';
      }
      const batEl = card.querySelector('[data-role=battery]');
      const bat = tel.battery;
      batEl.innerHTML = '<span class="' + batteryClass(bat) + '">'
                       + (typeof bat === 'number' ? bat + '%' : '—')
                       + '</span>';
      card.querySelector('[data-role=height]').textContent =
        typeof tel.height_cm === 'number' ? fmt(tel.height_cm / 100, 2) + ' m'
        : (typeof st.height_target_m === 'number'
           ? '(target ' + fmt(st.height_target_m, 2) + ' m)' : '—');
      const w = st.world_position_m;
      card.querySelector('[data-role=wpos]').textContent =
        Array.isArray(w) && w.length >= 3
          ? '(' + fmt(w[0], 2) + ', ' + fmt(w[1], 2) + ', '
                  + fmt(w[2], 2) + ')'
          : '—';
      card.querySelector('[data-role=distance]').textContent =
        typeof st.distance_m === 'number' ? fmt(st.distance_m, 2) + ' m'
                                            : '—';
      card.querySelector('[data-role=drone]').textContent =
        (fc.drone_connected ? 'connected' : 'no link')
        + (tel.flying ? ' · flying' : '');
      card.querySelector('[data-role=serial]').textContent =
        fc.drone_serial || (tel.serial_number || '—');
      // Start / Stop button visibility — mirror marker_mission's logic:
      // active phases show Stop, inactive show Start.
      const startBtn = card.querySelector('[data-role=start]');
      const stopBtn = card.querySelector('[data-role=stop]');
      const phase = (st.phase || 'init').toLowerCase();
      const inactive = ['init', 'done', 'abort'].includes(phase);
      startBtn.style.display = inactive ? '' : 'none';
      stopBtn.style.display = inactive ? 'none' : '';
      // Disable Start when drone not connected.
      startBtn.disabled = !fc.drone_connected;
      startBtn.title = fc.drone_connected ? '' : 'drone not connected';
    }
  }
  document.addEventListener('c2:overview', e => updateCards(e.detail));

  // ------------------- arena viz (top-down, all drones overlaid) ----------
  // We render in SVG because the viz has at most ~30 markers + 6
  // drones — well under any perf threshold — and SVG keeps text crisp
  // at any zoom, plus we get free event handling on per-element
  // tooltips. The viewBox is 800×480 logical units; we map arena (x, y)
  // metres to SVG pixels with a single best-fit transform recomputed
  // each frame in case the operator edits the arena draft mid-flight.

  const ARENA_VIZ_VB = {w: 800, h: 480};
  // Wall → colour, matching marker_mission's /arena page palette so an
  // operator switching between native FC UI and the C2 sees consistent
  // colour coding.
  const WALL_COLORS = {front:'#facc15', right:'#4ade80',
                        back:'#58c4ff',  left:'#f87171'};
  // FC name → stable colour for drone dots. Six entries covers the
  // default fleet; extras wrap.
  const FC_COLORS = ['#fb7185', '#fbbf24', '#a3e635', '#22d3ee',
                      '#a78bfa', '#f472b6'];
  let _cachedArena = null;
  let _cachedArenaLoadedAt = 0;

  async function fetchArenaDraft() {
    // Cache for 4 s — overview polls 2 Hz but the arena rarely changes.
    if (_cachedArena !== null
        && (Date.now() - _cachedArenaLoadedAt) < 4000) {
      return _cachedArena;
    }
    try {
      const r = await fetch('/api/c2/settings', {cache:'no-store'});
      const j = await r.json();
      _cachedArena = j.arena_draft || null;
      _cachedArenaLoadedAt = Date.now();
    } catch (e) {
      _cachedArena = null;
    }
    return _cachedArena;
  }

  function _renderArenaSvg(svg, arena, overview) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!arena) return false;
    const w = Number(arena.width_m) || 10;
    const d = Number(arena.depth_m) || 25;
    const markers = arena.markers || [];
    // Arena origin is at the centre: spans [-w/2, +w/2] × [-d/2, +d/2].
    // We map metres → SVG units with a single best-fit scale, then
    // CENTRE the arena rect in the viewBox so we don't end up with
    // empty space on one side when the arena's aspect ratio doesn't
    // match the canvas (which is virtually always the case).
    const pad = 30;
    const VB_W = ARENA_VIZ_VB.w, VB_H = ARENA_VIZ_VB.h;
    const scale = Math.min((VB_W - 2 * pad) / w,
                            (VB_H - 2 * pad) / d);
    const arenaWpx = w * scale, arenaHpx = d * scale;
    // Centre of viewBox = centre of arena (origin). That makes the
    // x/y arrows + origin label land exactly in the middle.
    const cx = VB_W / 2, cy = VB_H / 2;
    // World (x, y) → SVG (px, py). World +y is "away" from camera;
    // in SVG y grows down, so we flip y. Net: a marker at
    // (0, +depth/2) lands at the TOP of the arena rect.
    function toSvg(x, y) {
      return [cx + x * scale, cy - y * scale];
    }
    const ns = 'http://www.w3.org/2000/svg';
    function rect(x, y, w_, h, attrs) {
      const e = document.createElementNS(ns, 'rect');
      e.setAttribute('x', x); e.setAttribute('y', y);
      e.setAttribute('width', w_); e.setAttribute('height', h);
      for (const k of Object.keys(attrs || {})) e.setAttribute(k, attrs[k]);
      svg.appendChild(e); return e;
    }
    function line(x1, y1, x2, y2, attrs) {
      const e = document.createElementNS(ns, 'line');
      e.setAttribute('x1', x1); e.setAttribute('y1', y1);
      e.setAttribute('x2', x2); e.setAttribute('y2', y2);
      for (const k of Object.keys(attrs || {})) e.setAttribute(k, attrs[k]);
      svg.appendChild(e); return e;
    }
    function circle(cx_, cy_, r, attrs) {
      const e = document.createElementNS(ns, 'circle');
      e.setAttribute('cx', cx_); e.setAttribute('cy', cy_);
      e.setAttribute('r', r);
      for (const k of Object.keys(attrs || {})) e.setAttribute(k, attrs[k]);
      svg.appendChild(e); return e;
    }
    function text(x, y, str, attrs) {
      const e = document.createElementNS(ns, 'text');
      e.setAttribute('x', x); e.setAttribute('y', y);
      for (const k of Object.keys(attrs || {})) e.setAttribute(k, attrs[k]);
      e.textContent = str;
      svg.appendChild(e); return e;
    }
    // Arena outline.
    const [tlx, tly] = toSvg(-w/2,  d/2);  // top-left  in SVG
    const [brx, bry] = toSvg( w/2, -d/2);  // bot-right in SVG
    rect(tlx, tly, arenaWpx, arenaHpx,
         {fill:'none', stroke:'#9ca3af', 'stroke-width':2});
    // 1 m grid (subtle).
    for (let mx = Math.ceil(-w/2); mx <= w/2; mx++) {
      const [px,] = toSvg(mx, 0);
      line(px, tly, px, bry,
           {stroke:'#1d2229', 'stroke-width':1});
    }
    for (let my = Math.ceil(-d/2); my <= d/2; my++) {
      const [, py] = toSvg(0, my);
      line(tlx, py, brx, py,
           {stroke:'#1d2229', 'stroke-width':1});
    }
    // 5 m grid (more visible) + numeric labels along bottom + left.
    for (let mx = Math.ceil(-w/2); mx <= w/2; mx++) {
      if (mx % 5 !== 0) continue;
      const [px,] = toSvg(mx, 0);
      line(px, tly, px, bry,
           {stroke:'#2a3038', 'stroke-width':1});
      if (mx !== 0) {
        text(px, bry + 14, mx + ' m',
             {fill:'#6b7280', 'font-size':'9',
              'text-anchor':'middle'});
      }
    }
    for (let my = Math.ceil(-d/2); my <= d/2; my++) {
      if (my % 5 !== 0) continue;
      const [, py] = toSvg(0, my);
      line(tlx, py, brx, py,
           {stroke:'#2a3038', 'stroke-width':1});
      if (my !== 0) {
        text(tlx - 6, py + 3, my + ' m',
             {fill:'#6b7280', 'font-size':'9',
              'text-anchor':'end'});
      }
    }
    // Origin cross + label.
    const [ox, oy] = toSvg(0, 0);
    line(ox - 22, oy, ox + 22, oy,
         {stroke:'#facc15', 'stroke-width':2});
    line(ox, oy - 22, ox, oy + 22,
         {stroke:'#facc15', 'stroke-width':2});
    circle(ox, oy, 3, {fill:'#facc15'});
    text(ox + 26, oy + 4, '+x',
         {fill:'#facc15', 'font-size':'11', 'font-weight':'700'});
    text(ox, oy - 26, '+y',
         {fill:'#facc15', 'font-size':'11', 'font-weight':'700',
          'text-anchor':'middle'});
    text(ox - 8, oy + 14, '(0,0)',
         {fill:'#aab', 'font-size':'10', 'text-anchor':'end'});
    // Wall labels (outside the arena rect).
    text((tlx + brx) / 2, tly - 8, 'front (+y)',
         {fill:'#aab', 'font-size':'11',
          'text-anchor':'middle', 'font-weight':'600'});
    text((tlx + brx) / 2, bry + 28, 'back (-y)',
         {fill:'#aab', 'font-size':'11',
          'text-anchor':'middle', 'font-weight':'600'});
    text(tlx - 24, (tly + bry) / 2, 'left',
         {fill:'#aab', 'font-size':'11',
          'text-anchor':'end', 'font-weight':'600',
          'transform': 'rotate(-90 ' + (tlx - 24) + ' '
                        + ((tly + bry) / 2) + ')'});
    text(brx + 24, (tly + bry) / 2, 'right',
         {fill:'#aab', 'font-size':'11',
          'text-anchor':'start', 'font-weight':'600',
          'transform': 'rotate(90 ' + (brx + 24) + ' '
                        + ((tly + bry) / 2) + ')'});
    // Markers — group by (x, y) so vertically-stacked markers (same
    // wall location, different z) don't draw on top of each other.
    // Higher z renders higher on screen (smaller SVG y), so the
    // visual order matches "looking at the wall from inside the
    // room". Mirrors marker_mission's /arena page exactly.
    const STACK_OFFSET = 14;
    const groups = new Map();
    markers.forEach((m, idx) => {
      const k = (Number(m.x) || 0).toFixed(2) + ','
              + (Number(m.y) || 0).toFixed(2);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(idx);
    });
    const stackOffset = new Map();
    groups.forEach(indices => {
      indices.sort((a, b) =>
        (Number(markers[b].z) || 0) - (Number(markers[a].z) || 0));
      const n = indices.length;
      indices.forEach((idx, i) => {
        stackOffset.set(idx, (i - (n - 1) / 2) * STACK_OFFSET);
      });
    });
    markers.forEach((m, idx) => {
      const x = Number(m.x), y = Number(m.y);
      if (Number.isNaN(x) || Number.isNaN(y)) return;
      const [px, py0] = toSvg(x, y);
      const py = py0 + (stackOffset.get(idx) || 0);
      const col = WALL_COLORS[m.wall] || '#9ca3af';
      circle(px, py, 8, {fill: col, stroke:'#000',
                          'stroke-width':1});
      text(px, py + 3.5, String(m.id),
           {fill:'#0e1116', 'font-size':'10', 'font-weight':'700',
            'text-anchor':'middle'});
      text(px + 11, py + 3.5,
           '(' + x.toFixed(1) + ',' + y.toFixed(1) + ','
            + (Number(m.z) || 0).toFixed(1) + ')',
           {fill:'#aab', 'font-size':'9'});
    });
    // Drone dots — read each enabled FC's world_position_m from the
    // overview payload and project. Use a stable per-FC colour so the
    // operator can match the dot to a card without hover.
    let droneCount = 0;
    for (const name of Object.keys(overview || {})) {
      const fc = overview[name];
      if (!fc || !fc.state) continue;
      const wp = fc.state.world_position_m;
      if (!Array.isArray(wp) || wp.length < 2) continue;
      const x = Number(wp[0]), y = Number(wp[1]);
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      const idx = FC_NAMES.indexOf(name);
      const col = FC_COLORS[idx >= 0 ? idx % FC_COLORS.length : 0];
      const [px, py] = toSvg(x, y);
      // Ring + filled dot — the ring helps when a drone sits on top
      // of a marker (otherwise the marker colour swallows the dot).
      circle(px, py, 9,
             {fill:'none', stroke: col, 'stroke-width':3,
              opacity:'0.5'});
      circle(px, py, 5,
             {fill: col, stroke:'#000', 'stroke-width':1});
      text(px + 11, py + 4, name,
           {fill: col, 'font-size':'10', 'font-weight':'600'});
      droneCount++;
    }
    return droneCount;
  }

  async function refreshArenaViz(overview) {
    const svg = document.getElementById('arena-viz');
    const empty = document.getElementById('arena-viz-empty');
    const status = document.getElementById('arena-viz-status');
    if (!svg) return;
    const arena = await fetchArenaDraft();
    if (!arena) {
      empty.style.display = '';
      svg.style.display = 'none';
      status.textContent = '';
      return;
    }
    empty.style.display = 'none';
    svg.style.display = '';
    const n = _renderArenaSvg(svg, arena, overview);
    status.textContent = n + ' drone' + (n === 1 ? '' : 's')
                          + ' on arena · ' + (arena.markers || []).length
                          + ' marker' + ((arena.markers || []).length === 1
                                          ? '' : 's');
  }

  document.addEventListener('c2:overview', e => refreshArenaViz(e.detail));
  // Initial paint before the first poll lands — uses cached overview
  // if there is one.
  refreshArenaViz(window._latestOverview || {});

  // Show/Hide all video — clicks each per-card video toggle so the
  // cache-busting + slot-replace logic stays in exactly one place.
  // Two-mode button: if any card is currently OFF, treat the click
  // as "turn ALL on" (clicks only the off cards). Otherwise treat it
  // as "turn ALL off" (clicks every card).
  const btnVideoAll = document.getElementById('btn-video-all');
  const videoAllMsg = document.getElementById('video-all-msg');
  function syncVideoAllLabel() {
    const toggles = document.querySelectorAll('[data-role=video-toggle]');
    if (!toggles.length) return;
    let on = 0;
    toggles.forEach(t => { if (t.dataset.on === '1') on++; });
    btnVideoAll.textContent = on === toggles.length
      ? 'Hide all video' : 'Show all video';
    videoAllMsg.textContent = on + '/' + toggles.length + ' streaming';
  }
  if (btnVideoAll) {
    btnVideoAll.addEventListener('click', () => {
      const toggles = Array.from(
        document.querySelectorAll('[data-role=video-toggle]'));
      if (!toggles.length) return;
      const allOn = toggles.every(t => t.dataset.on === '1');
      const target = !allOn;  // true = turn all ON, false = OFF
      // Stagger the clicks ~50 ms apart so the browser doesn't open
      // six MJPEG sockets in the exact same event-loop tick — keeps
      // the network panel readable and avoids head-of-line bursts.
      toggles.forEach((t, i) => {
        const isOn = t.dataset.on === '1';
        if (isOn !== target) {
          setTimeout(() => { t.click(); }, i * 50);
        }
      });
      setTimeout(syncVideoAllLabel, toggles.length * 50 + 100);
    });
    // Refresh label whenever any per-card toggle was clicked.
    document.addEventListener('click', ev => {
      const t = ev.target;
      if (t instanceof HTMLElement && t.dataset.role === 'video-toggle') {
        setTimeout(syncVideoAllLabel, 50);
      }
    });
    document.addEventListener('c2:overview', () => syncVideoAllLabel());
  }

  // Per-card button wiring (event-delegated so it survives DOM rerenders).
  document.addEventListener('click', async ev => {
    const t = ev.target;
    if (!(t instanceof HTMLElement)) return;
    const card = t.closest('.fc-card');
    if (!card) return;
    const fc = card.dataset.fc;
    const role = t.dataset.role;
    const msg = card.querySelector('[data-role=msg]');
    const scriptMsg = card.querySelector('[data-role=script-msg]');
    const setMsg = (el, text, kind) => {
      if (!el) return;
      el.textContent = text || '';
      el.style.color = kind === 'bad' ? '#f87171'
                      : kind === 'good' ? '#86efac' : '#aab';
    };
    if (role === 'video-toggle') {
      const slot = card.querySelector('[data-role=video-slot]');
      const url = t.dataset.mjpegUrl;
      if (t.dataset.on === '1') {
        slot.innerHTML = '';
        slot.textContent = 'video off';
        t.dataset.on = '';
        t.textContent = 'Show video';
      } else {
        slot.innerHTML = '';
        const img = document.createElement('img');
        // decoding=async pushes JPEG decode off the main thread so
        // six parallel streams don't starve UI interactivity. The fps
        // cap scales with concurrent streams: solo card gets 20 fps,
        // multi-card drops to 10 (a 6-stream sweep on a laptop GPU
        // is the difference between smooth and chunky).
        img.decoding = 'async';
        img.loading = 'eager';  // overview is the operator's primary surface
        const otherOn = document.querySelectorAll(
          '[data-role=video-toggle][data-on="1"]').length;
        const fps = otherOn >= 1 ? 10 : 20;
        // Cache-bust + fps so a previous closed-then-reopened stream
        // doesn't rebind to a stale connection.
        img.src = url + '?fps=' + fps + '&t=' + Date.now();
        slot.appendChild(img);
        t.dataset.on = '1';
        t.textContent = 'Hide video';
      }
    } else if (role === 'start') {
      setMsg(msg, 'starting…');
      const textarea = card.querySelector('[data-role=script-text]');
      const body = {};
      const txt = (textarea && textarea.value || '').trim();
      if (txt) body.script = txt;
      try {
        const r = await fetch('/api/c2/' + fc + '/start',
                              {method:'POST',
                               headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body)});
        const j = await r.json();
        if (j.ok) setMsg(msg, 'started', 'good');
        else setMsg(msg, 'start failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg(msg, 'start error: ' + e, 'bad'); }
    } else if (role === 'stop') {
      setMsg(msg, 'stopping…');
      try {
        const r = await fetch('/api/c2/' + fc + '/stop', {method:'POST'});
        const j = await r.json();
        if (j.ok) setMsg(msg, j.noop ? 'already idle' : 'stop sent', 'good');
        else setMsg(msg, 'stop failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg(msg, 'stop error: ' + e, 'bad'); }
    } else if (role === 'script-load') {
      setMsg(scriptMsg, 'loading…');
      try {
        const r = await fetch('/api/c2/' + fc + '/script', {cache:'no-store'});
        const j = await r.json();
        if (j.ok) {
          card.querySelector('[data-role=script-text]').value =
            (j.payload && j.payload.text) || '';
          setMsg(scriptMsg, 'loaded', 'good');
        } else setMsg(scriptMsg, 'load failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg(scriptMsg, 'load error: ' + e, 'bad'); }
    } else if (role === 'script-save') {
      const text = card.querySelector('[data-role=script-text]').value;
      setMsg(scriptMsg, 'saving…');
      try {
        const r = await fetch('/api/c2/' + fc + '/script',
                              {method:'POST',
                               headers:{'Content-Type':'application/json'},
                               body: JSON.stringify({text})});
        const j = await r.json();
        if (j.ok) setMsg(scriptMsg, 'saved', 'good');
        else setMsg(scriptMsg, 'save failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg(scriptMsg, 'save error: ' + e, 'bad'); }
    }
  });
})();
</script>
"""


# ---------------------------------------------------------------- arena

PAGE_ARENA = """
<main>
  <div class="card" style="margin-bottom:1rem;">
    <h2>Source</h2>
    <div style="display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.25rem;
                    font-size:.85rem;">
        <input type="radio" name="arena-src" value="{{ fc.name }}"
               {% if loop.first %}checked{% endif %}>
        {{ fc.name }}
      </label>
      {% endfor %}
      <button type="button" class="btn" id="btn-arena-load">
        Load active arena from FC
      </button>
      <button type="button" class="btn btn-ghost" id="btn-arena-load-draft"
              title="Load the C2-local draft used by the overview viz">
        Load C2 draft
      </button>
      <button type="button" class="btn btn-ghost" id="btn-arena-new">
        New empty arena
      </button>
      <span id="arena-load-msg" style="font-size:.78rem; color:#aab;"></span>
    </div>
  </div>

  <div class="card" style="margin-bottom:1rem;">
    <h2>Target FCs</h2>
    <div id="arena-fc-checks" style="display:flex; gap:.6rem; flex-wrap:wrap;
                                      margin-bottom:.5rem;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.3rem;
                    font-size:.85rem;">
        <input type="checkbox" data-fc="{{ fc.name }}" checked>
        {{ fc.name }}
      </label>
      {% endfor %}
    </div>
    <div style="display:flex; gap:.4rem; margin-bottom:.5rem;">
      <button type="button" class="btn btn-ghost" id="btn-arena-check-all">
        Select all
      </button>
      <button type="button" class="btn btn-ghost" id="btn-arena-uncheck-all">
        Clear
      </button>
    </div>
  </div>

  <div class="card">
    <h2>Arena JSON</h2>
    <textarea id="arena-text" rows="22" spellcheck="false"
              style="width:100%; box-sizing:border-box;"
              placeholder='{"width_m": 10, "depth_m": 25, "markers": [...]}'
              ></textarea>
    <div style="display:flex; gap:.5rem; align-items:center; margin-top:.5rem;
                flex-wrap:wrap;">
      <button type="button" class="btn btn-warn" id="btn-arena-push">
        Push to selected FCs
      </button>
      <button type="button" class="btn" id="btn-arena-save-draft"
              title="Save as the C2-local arena draft the overview viz uses">
        Save as C2 draft
      </button>
      <span id="arena-push-msg" style="font-size:.85rem; color:#aab;"></span>
    </div>
    <pre id="arena-results"
         style="margin-top:.6rem; background:#0c0f12; border-radius:6px;
                padding:.5rem; min-height:0; max-height:240px;
                overflow:auto; font-size:.8rem; color:#aab;"></pre>
  </div>
</main>

<script>
(function() {
  const $ = id => document.getElementById(id);
  $('btn-arena-check-all').addEventListener('click', () =>
    document.querySelectorAll('#arena-fc-checks input').forEach(
      el => el.checked = true));
  $('btn-arena-uncheck-all').addEventListener('click', () =>
    document.querySelectorAll('#arena-fc-checks input').forEach(
      el => el.checked = false));

  const NEW_ARENA_TEMPLATE = {
    width_m: 10, depth_m: 25, top_z_m: 4, bottom_z_m: 2,
    marker_size_m: 0.18,
    magnetic_north_arena_yaw_deg: null,
    markers: []
  };

  $('btn-arena-load').addEventListener('click', async () => {
    const src = (document.querySelector('input[name=arena-src]:checked')
                 || {}).value;
    if (!src) return;
    const msg = $('arena-load-msg');
    msg.textContent = 'loading from ' + src + '…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/arena/load-from/' + src,
                            {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        $('arena-text').value = JSON.stringify(j.payload, null, 2);
        msg.textContent = 'loaded from ' + src;
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'load failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'load error: ' + e; msg.style.color = '#f87171';
    }
  });

  $('btn-arena-load-draft').addEventListener('click', async () => {
    const msg = $('arena-load-msg');
    msg.textContent = 'loading C2 draft…'; msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/settings', {cache:'no-store'});
      const j = await r.json();
      if (j.arena_draft) {
        $('arena-text').value = JSON.stringify(j.arena_draft, null, 2);
        msg.textContent = 'loaded C2 draft';
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'no C2 draft saved yet';
        msg.style.color = '#facc15';
      }
    } catch (e) {
      msg.textContent = 'load error: ' + e; msg.style.color = '#f87171';
    }
  });

  $('btn-arena-new').addEventListener('click', () => {
    if ($('arena-text').value.trim()
        && !confirm('Replace current editor contents with an empty arena '
                    + 'template? Unsaved edits will be lost.')) return;
    $('arena-text').value = JSON.stringify(NEW_ARENA_TEMPLATE, null, 2);
    const msg = $('arena-load-msg');
    msg.textContent = 'started new arena from template';
    msg.style.color = '#aab';
  });

  $('btn-arena-save-draft').addEventListener('click', async () => {
    const msg = $('arena-push-msg');
    let payload;
    try { payload = JSON.parse($('arena-text').value); }
    catch (e) {
      msg.textContent = 'invalid JSON: ' + e; msg.style.color = '#f87171';
      return;
    }
    msg.textContent = 'saving C2 draft…'; msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/settings/arena-draft',
                            {method:'POST',
                             headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({payload})});
      const j = await r.json();
      if (j.ok) {
        msg.textContent = 'saved as C2 draft (overview viz now uses it)';
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'save failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'save error: ' + e; msg.style.color = '#f87171';
    }
  });

  // Preload C2 draft if there is one — operator usually wants to keep
  // editing where they left off.
  (async () => {
    try {
      const r = await fetch('/api/c2/settings', {cache:'no-store'});
      const j = await r.json();
      if (j.arena_draft && !$('arena-text').value) {
        $('arena-text').value = JSON.stringify(j.arena_draft, null, 2);
      }
    } catch (e) {}
  })();

  $('btn-arena-push').addEventListener('click', async () => {
    const fcs = Array.from(
      document.querySelectorAll('#arena-fc-checks input:checked')
    ).map(el => el.dataset.fc);
    const msg = $('arena-push-msg');
    if (!fcs.length) {
      msg.textContent = 'no FCs selected'; msg.style.color = '#f87171';
      return;
    }
    let payload;
    try { payload = JSON.parse($('arena-text').value); }
    catch (e) {
      msg.textContent = 'invalid JSON: ' + e; msg.style.color = '#f87171';
      return;
    }
    msg.textContent = 'pushing to ' + fcs.length + ' FC(s)…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/arena/push',
                            {method:'POST',
                             headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({fcs, payload})});
      const j = await r.json();
      const results = j.results || {};
      let ok = 0, fail = 0;
      const lines = [];
      for (const name of Object.keys(results)) {
        if (results[name].ok) { ok++; lines.push('OK    ' + name); }
        else { fail++;
               lines.push('FAIL  ' + name + '  '
                          + (results[name].error || '')); }
      }
      msg.textContent = 'pushed: ' + ok + ' ok, ' + fail + ' failed';
      msg.style.color = fail ? '#f87171' : '#86efac';
      $('arena-results').textContent = lines.join('\\n');
    } catch (e) {
      msg.textContent = 'push error: ' + e; msg.style.color = '#f87171';
    }
  });
})();
</script>
"""


# ---------------------------------------------------------------- tune

PAGE_TUNE = """
<main>
  <div class="card" style="margin-bottom:1rem;">
    <h2>Source FC</h2>
    <div style="display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.25rem;
                    font-size:.85rem;">
        <input type="radio" name="tune-src" value="{{ fc.name }}"
               {% if loop.first %}checked{% endif %}>
        {{ fc.name }}
      </label>
      {% endfor %}
      <button type="button" class="btn" id="btn-tune-load">
        Load tune from FC
      </button>
      <button type="button" class="btn btn-ghost" id="btn-tune-defaults"
              title="Static factory defaults from MissionConfig dataclass">
        Load defaults
      </button>
      <span id="tune-load-msg" style="font-size:.78rem; color:#aab;"></span>
    </div>
  </div>

  <div class="card" style="margin-bottom:1rem;">
    <h2>Target FCs</h2>
    <div id="tune-fc-checks" style="display:flex; gap:.6rem; flex-wrap:wrap;
                                     margin-bottom:.5rem;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.3rem;
                    font-size:.85rem;">
        <input type="checkbox" data-fc="{{ fc.name }}" checked>
        {{ fc.name }}
      </label>
      {% endfor %}
    </div>
    <label style="font-size:.85rem; display:inline-flex; align-items:center;
                  gap:.3rem;">
      <input type="checkbox" id="tune-also-save">
      Also persist with /api/tune/save on each selected FC
    </label>
    <label style="font-size:.85rem; display:inline-flex; align-items:center;
                  gap:.3rem; margin-left:1rem;">
      <input type="checkbox" id="tune-only-changed">
      Send only fields I edited (diff vs source FC)
    </label>
  </div>

  <div class="card">
    <h2>Tuning parameters</h2>
    <p style="color:#aab; font-size:.85rem; margin:0 0 .5rem 0;">
      Apply pushes every field shown to each selected FC — useful for
      sync'ing a source FC's full config to the rest of the fleet.
      Tick <em>Send only fields I edited</em> if you want to override
      just a few values without touching the rest.
    </p>
    <div id="tune-form"
         style="display:grid;
                grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
                gap:.4rem .75rem; align-items:center;
                font-size:.85rem;">
      <span style="color:#aab; grid-column:1/-1;">
        Load a source FC to populate the form.
      </span>
    </div>
    <div style="display:flex; gap:.5rem; align-items:center; margin-top:.6rem;
                flex-wrap:wrap;">
      <button type="button" class="btn btn-warn" id="btn-tune-push">
        Apply to selected FCs
      </button>
      <span id="tune-push-msg" style="font-size:.85rem; color:#aab;"></span>
    </div>
    <pre id="tune-results"
         style="margin-top:.6rem; background:#0c0f12; border-radius:6px;
                padding:.5rem; max-height:240px; overflow:auto;
                font-size:.8rem; color:#aab;"></pre>
  </div>
</main>

<script>
(function() {
  const $ = id => document.getElementById(id);
  // Holds the original values from the source FC so we can compute the
  // diff to send (avoids overwriting fields the operator didn't touch).
  let originalValues = {};

  function renderForm(values) {
    const form = $('tune-form');
    form.innerHTML = '';
    const keys = Object.keys(values).sort();
    for (const k of keys) {
      const wrap = document.createElement('label');
      wrap.style.cssText = 'display:flex; align-items:center; gap:.4rem;';
      const labelEl = document.createElement('span');
      labelEl.textContent = k;
      labelEl.style.cssText = 'color:#aab; min-width:11rem;';
      const v = values[k];
      const input = document.createElement('input');
      input.dataset.name = k;
      if (typeof v === 'boolean') {
        input.type = 'checkbox';
        input.checked = v;
        input.dataset.kind = 'bool';
      } else if (typeof v === 'number') {
        input.type = 'number'; input.step = 'any';
        input.value = v;
        input.dataset.kind = 'number';
      } else {
        input.type = 'text';
        input.value = (v === null || v === undefined) ? '' : String(v);
        input.dataset.kind = (v === null ? 'null' : 'string');
      }
      input.style.cssText = 'flex:1; min-width:5rem;';
      wrap.appendChild(labelEl);
      wrap.appendChild(input);
      form.appendChild(wrap);
    }
  }

  function collectUpdates(opts) {
    // Apply = "make the selected FCs match what the form shows" — so
    // by default we send EVERY field, not just the diff. That matches
    // the C2's "load from one FC, push to others" model where the
    // source FC is canonical. If the operator only wants to override
    // a subset of fields they can clear the rest (or use the "only
    // changed" checkbox if surfaced later).
    const onlyChanged = !!(opts && opts.onlyChanged);
    const updates = {};
    document.querySelectorAll('#tune-form input').forEach(el => {
      const k = el.dataset.name;
      const kind = el.dataset.kind;
      let v;
      if (kind === 'bool') v = el.checked;
      else if (kind === 'number') {
        if (el.value === '') return;
        const f = Number(el.value);
        if (Number.isNaN(f)) return;
        v = f;
      } else if (kind === 'null' && el.value === '') {
        // Field was null on the source FC and the operator left it
        // empty — skip rather than push the string "" which the FC
        // would type-check against a numeric field and reject.
        return;
      } else {
        v = el.value;
      }
      if (onlyChanged) {
        const orig = originalValues[k];
        const same = (kind === 'bool') ? (orig === v)
                      : (kind === 'number') ? (Number(orig) === v)
                      : (String(orig === null ? '' : orig) === String(v));
        if (same) return;
      }
      updates[k] = v;
    });
    return updates;
  }

  $('btn-tune-load').addEventListener('click', async () => {
    const src = (document.querySelector('input[name=tune-src]:checked')
                 || {}).value;
    if (!src) return;
    const msg = $('tune-load-msg');
    msg.textContent = 'loading…'; msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/tune/load-from/' + src,
                            {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        // Server returns the field/value map only — schema metadata
        // (form layout etc.) is FC-local and not propagated.
        originalValues = j.values || {};
        renderForm(originalValues);
        msg.textContent = 'loaded ' + Object.keys(originalValues).length
                          + ' fields from ' + src;
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'load failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'load error: ' + e; msg.style.color = '#f87171';
    }
  });

  // Auto-load the static defaults from the marker_mission dataclass
  // on page open — operator sees every parameter and its factory
  // default immediately, with no "Load from FC" round-trip needed.
  // "Load from FC" still overrides with live per-FC values.
  $('btn-tune-defaults') && $('btn-tune-defaults').addEventListener(
    'click', () => loadDefaults());
  async function loadDefaults() {
    const msg = $('tune-load-msg');
    msg.textContent = 'loading defaults from MissionConfig…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/tune/defaults', {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        originalValues = j.values || {};
        renderForm(originalValues);
        msg.textContent = 'loaded ' + Object.keys(originalValues).length
                          + ' default values';
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'defaults load failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'defaults error: ' + e; msg.style.color = '#f87171';
    }
  }
  loadDefaults();

  $('btn-tune-push').addEventListener('click', async () => {
    const fcs = Array.from(
      document.querySelectorAll('#tune-fc-checks input:checked')
    ).map(el => el.dataset.fc);
    const msg = $('tune-push-msg');
    if (!fcs.length) {
      msg.textContent = 'no FCs selected'; msg.style.color = '#f87171';
      return;
    }
    const onlyChanged = $('tune-only-changed') && $('tune-only-changed').checked;
    const updates = collectUpdates({onlyChanged});
    if (!Object.keys(updates).length) {
      msg.textContent = onlyChanged
        ? 'no field changes vs source FC to apply'
        : 'load a source FC first';
      msg.style.color = '#facc15';
      return;
    }
    const alsoSave = $('tune-also-save').checked;
    msg.textContent = 'applying ' + Object.keys(updates).length
                      + ' field(s) to ' + fcs.length + ' FC(s)…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/tune/apply',
                            {method:'POST',
                             headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({fcs, updates,
                                                    also_save: alsoSave})});
      const j = await r.json();
      const results = j.results || {};
      const lines = [];
      let ok = 0, fail = 0;
      for (const name of Object.keys(results)) {
        if (results[name].ok) { ok++; lines.push('OK    ' + name); }
        else { fail++; lines.push('FAIL  ' + name + '  '
                                   + (results[name].error || '')); }
      }
      msg.textContent = ok + ' ok, ' + fail + ' failed';
      msg.style.color = fail ? '#f87171' : '#86efac';
      $('tune-results').textContent = lines.join('\\n');
    } catch (e) {
      msg.textContent = 'apply error: ' + e; msg.style.color = '#f87171';
    }
  });
})();
</script>
"""


# ---------------------------------------------------------------- calibrate

PAGE_CALIBRATE = """
<main>
  <div class="card" style="margin-bottom:1rem;">
    <h2>Live calibration (per drone)</h2>
    <p style="color:#aab; font-size:.85rem; margin:0 0 .5rem 0;">
      Calibration drives the drone connected to one specific FC, so
      it only makes sense one at a time. Pick a tab to load that
      FC's own <code>/calibrate</code> page; other tabs stay
      un-mounted until clicked so we don't spin up six camera feeds
      at once.
    </p>
    <div class="cal-tabs" id="cal-tabs"
         style="display:flex; gap:.3rem; flex-wrap:wrap;
                border-bottom:1px solid #2a3038; margin-bottom:.5rem;
                padding-bottom:.4rem;">
      {% for fc in fc_specs %}
      <button type="button" class="cal-tab"
              data-fc="{{ fc.name }}"
              data-url="http://{{ fc.host }}:{{ fc.port }}/calibrate"
              style="background:#0c0f12; color:var(--fg);
                     border:1px solid #2a3038; border-radius:5px;
                     padding:.4rem .8rem; font-size:.85rem;
                     cursor:pointer;">
        {{ fc.name }}
      </button>
      {% endfor %}
      <span style="margin-left:auto; align-self:center;">
        <a id="cal-open-tab" href="#" target="_blank" rel="noopener"
           style="font-size:.78rem; display:none;">open in new tab ↗</a>
      </span>
    </div>
    <div id="cal-tab-host"
         style="background:#0c0f12; border-radius:6px;
                min-height:560px;
                display:flex; align-items:center; justify-content:center;
                color:#6b7280; font-size:.85rem;">
      Pick a flight controller above to load its calibration page.
    </div>
  </div>

  <div class="card">
    <div style="display:flex; align-items:center; gap:.6rem;
                margin-bottom:.5rem; flex-wrap:wrap;">
      <h2 style="margin:0;">Calibration library</h2>
      <button type="button" class="btn btn-ghost" id="btn-cal-refresh">
        Refresh now
      </button>
      <span id="cal-status" style="color:#aab; font-size:.8rem;"></span>
    </div>
    <p style="color:#aab; font-size:.85rem; margin:.2rem 0 .6rem 0;">
      C2 mirrors every FC's calibration <code>.npz</code> files into a
      central library so you can push the right one to a different FC
      when a drone is physically moved. FC → library is automatic;
      library → FC is the <em>Push to ▾</em> action.
    </p>
    <table id="cal-table">
      <thead>
        <tr>
          <th>Filename</th><th>Source FC</th><th>Serial</th>
          <th>Res</th><th>RMS</th><th>Calibrated</th>
          <th>Local mtime</th><th>Actions</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    <div id="cal-msg" style="margin-top:.5rem; font-size:.85rem;
                              color:#aab;"></div>
  </div>
</main>

<script>
(function() {
  // ----- single-active-tab iframe wiring -----
  const tabs = document.querySelectorAll('.cal-tab');
  const host = document.getElementById('cal-tab-host');
  const openLink = document.getElementById('cal-open-tab');
  function activate(btn) {
    tabs.forEach(t => {
      t.style.background = (t === btn) ? 'var(--accent)' : '#0c0f12';
      t.style.color = (t === btn) ? '#062633' : 'var(--fg)';
      t.style.fontWeight = (t === btn) ? '700' : '400';
    });
    // Replace the host's contents — destroying the old iframe is
    // important so that FC's camera MJPEG socket is closed before we
    // open the next FC's. We don't want six MJPEG streams stacking
    // up just because the operator clicked through every tab.
    host.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.src = btn.dataset.url;
    iframe.style.cssText = 'width:100%; height:680px; border:0; '
      + 'background:#0c0f12; border-radius:6px;';
    iframe.setAttribute('sandbox',
      'allow-scripts allow-same-origin allow-forms');
    host.style.display = 'block';
    host.style.minHeight = '';
    host.appendChild(iframe);
    openLink.href = btn.dataset.url;
    openLink.style.display = '';
  }
  tabs.forEach(t => t.addEventListener('click', () => activate(t)));
  // Don't auto-activate the first tab — the operator should
  // consciously pick which drone they're calibrating. Auto-loading
  // would silently start a camera stream on whatever FC happened to
  // be first in the inventory.

  // ----- library table -----
  function fmtTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts * 1000);
    return d.toLocaleString();
  }
  function pushOptions() {
    return FC_NAMES.map(n => '<option value="' + n + '">'
                              + n + '</option>').join('');
  }
  async function refresh() {
    const s = document.getElementById('cal-status');
    s.textContent = 'loading…';
    try {
      const r = await fetch('/api/c2/calibrations', {cache:'no-store'});
      const j = await r.json();
      const tbody = document.querySelector('#cal-table tbody');
      tbody.innerHTML = '';
      for (const row of j.entries || []) {
        const tr = document.createElement('tr');
        tr.innerHTML =
          '<td>' + row.name + '</td>' +
          '<td>' + (row.source_fc || '—') + '</td>' +
          '<td style="font-family:ui-monospace,monospace; font-size:.78rem;">'
            + (row.serial || '—') + '</td>' +
          '<td>' + (row.resolution || '—') + '</td>' +
          '<td>' + (typeof row.rms_error === 'number'
                    ? row.rms_error.toFixed(3) : '—') + '</td>' +
          '<td>' + (row.calibrated_at || '—') + '</td>' +
          '<td>' + fmtTime(row.mtime) + '</td>' +
          '<td style="white-space:nowrap;">'
            + '<select data-role="push-to">'
              + '<option value="">push to ▾</option>'
              + pushOptions() + '</select> '
            + '<button class="btn" data-role="push" data-name="'
              + row.name + '">push</button> '
            + '<button class="btn btn-ghost" data-role="delete" data-name="'
              + row.name + '">delete</button>'
          + '</td>';
        tbody.appendChild(tr);
      }
      s.textContent = 'last sweep ' + (j.last_sync_age_s
                       ? Math.round(j.last_sync_age_s) + ' s ago' : 'pending')
                      + ', ' + (j.entries || []).length + ' file(s)';
    } catch (e) {
      s.textContent = 'error: ' + e;
    }
  }
  document.getElementById('btn-cal-refresh').addEventListener('click', async () => {
    document.getElementById('cal-status').textContent = 'syncing…';
    await fetch('/api/c2/calibrations/sync', {method:'POST'});
    await refresh();
  });
  document.addEventListener('click', async ev => {
    const t = ev.target;
    if (!(t instanceof HTMLElement)) return;
    const msg = document.getElementById('cal-msg');
    const setMsg = (text, kind) => {
      msg.textContent = text;
      msg.style.color = kind === 'bad' ? '#f87171'
                       : kind === 'good' ? '#86efac' : '#aab';
    };
    if (t.dataset.role === 'push') {
      const row = t.closest('tr');
      const select = row.querySelector('[data-role=push-to]');
      const fc = select.value;
      const name = t.dataset.name;
      if (!fc) { setMsg('pick a target FC first', 'bad'); return; }
      setMsg('pushing ' + name + ' → ' + fc + '…');
      try {
        const r = await fetch('/api/c2/calibrations/'
                              + encodeURIComponent(name) + '/push',
                              {method:'POST',
                               headers:{'Content-Type':'application/json'},
                               body: JSON.stringify({fc})});
        const j = await r.json();
        if (j.ok) {
          let m = 'pushed ' + name + ' to ' + fc;
          if (j.warning) m += ' (warning: ' + j.warning + ')';
          setMsg(m, j.warning ? 'good' : 'good');
        } else setMsg('push failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg('push error: ' + e, 'bad'); }
      await refresh();
    } else if (t.dataset.role === 'delete') {
      const name = t.dataset.name;
      if (!confirm('Delete calibration ' + name
                   + ' from the C2 library?\\n\\n'
                   + 'This does NOT delete it on any FC.')) return;
      setMsg('deleting ' + name + '…');
      try {
        const r = await fetch('/api/c2/calibrations/'
                              + encodeURIComponent(name),
                              {method:'DELETE'});
        const j = await r.json();
        if (j.ok) setMsg('deleted ' + name, 'good');
        else setMsg('delete failed: ' + (j.error || ''), 'bad');
      } catch (e) { setMsg('delete error: ' + e, 'bad'); }
      await refresh();
    }
  });
  refresh();
  setInterval(refresh, 5000);
})();
</script>
"""


# ---------------------------------------------------------------- settings

PAGE_SETTINGS = """
<main>
  <div class="card" style="margin-bottom:1rem;">
    <h2>Flight controllers</h2>
    <p style="color:#aab; font-size:.85rem; margin:0 0 .75rem 0;">
      Disabled FCs are not polled, hidden from the overview, and
      excluded from EMERGENCY LAND ALL, Start all, and every
      arena / tune / script fan-out. Re-enable to bring them back
      into the fleet immediately — no restart needed.
    </p>
    <table id="settings-fc">
      <thead>
        <tr><th>FC</th><th>Host</th><th>Port</th><th>Enabled</th></tr>
      </thead>
      <tbody>
        {% for fc in fc_specs %}
        <tr data-fc="{{ fc.name }}">
          <td><strong>{{ fc.name }}</strong></td>
          <td>{{ fc.host }}</td>
          <td>{{ fc.port }}</td>
          <td>
            <label style="display:inline-flex; align-items:center; gap:.4rem;">
              <input type="checkbox" data-role="fc-enabled" checked>
              <span data-role="status" style="font-size:.78rem;
                                              color:#aab;">…</span>
            </label>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</main>

<script>
(function() {
  function refresh() {
    fetch('/api/c2/settings', {cache:'no-store'})
      .then(r => r.json())
      .then(j => {
        for (const fc of (j.fcs || [])) {
          const row = document.querySelector(
            '#settings-fc tr[data-fc="' + fc.name + '"]');
          if (!row) continue;
          row.querySelector('[data-role=fc-enabled]').checked = !!fc.enabled;
          const st = row.querySelector('[data-role=status]');
          st.textContent = fc.enabled ? 'enabled' : 'disabled';
          st.style.color = fc.enabled ? '#86efac' : '#aab';
        }
      });
  }
  document.addEventListener('change', async ev => {
    const t = ev.target;
    if (!(t instanceof HTMLElement)) return;
    if (t.dataset.role !== 'fc-enabled') return;
    const row = t.closest('tr');
    const fc = row.dataset.fc;
    const enabled = !!t.checked;
    const st = row.querySelector('[data-role=status]');
    st.textContent = 'saving…'; st.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/settings/fc-enabled',
                            {method:'POST',
                             headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({fc, enabled})});
      const j = await r.json();
      if (j.ok) {
        st.textContent = enabled ? 'enabled' : 'disabled';
        st.style.color = enabled ? '#86efac' : '#aab';
      } else {
        st.textContent = 'failed: ' + (j.error || '');
        st.style.color = '#f87171';
      }
    } catch (e) {
      st.textContent = 'error: ' + e; st.style.color = '#f87171';
    }
  });
  refresh();
})();
</script>
"""


# ---------------------------------------------------------------- scripts

PAGE_SCRIPTS = """
<main>
  <div class="card" style="margin-bottom:1rem;">
    <h2>Source FC</h2>
    <div style="display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.25rem;
                    font-size:.85rem;">
        <input type="radio" name="scr-src" value="{{ fc.name }}"
               {% if loop.first %}checked{% endif %}>
        {{ fc.name }}
      </label>
      {% endfor %}
      <button type="button" class="btn" id="btn-scr-load-active">
        Load active draft from FC
      </button>
      <select id="scr-named-select"
              style="font-size:.85rem;">
        <option value="">— named scripts —</option>
      </select>
      <button type="button" class="btn btn-ghost" id="btn-scr-refresh-named">
        ↻
      </button>
      <button type="button" class="btn" id="btn-scr-load-named">
        Load named
      </button>
      <span id="scr-load-msg" style="font-size:.78rem; color:#aab;"></span>
    </div>
  </div>

  <div class="card" style="margin-bottom:1rem;">
    <h2>Target FCs</h2>
    <div id="scr-fc-checks" style="display:flex; gap:.6rem; flex-wrap:wrap;
                                    margin-bottom:.5rem;">
      {% for fc in fc_specs %}
      <label style="display:inline-flex; align-items:center; gap:.3rem;
                    font-size:.85rem;">
        <input type="checkbox" data-fc="{{ fc.name }}" checked>
        {{ fc.name }}
      </label>
      {% endfor %}
    </div>
    <div style="display:flex; gap:.4rem;">
      <button type="button" class="btn btn-ghost" id="btn-scr-all">
        Select all
      </button>
      <button type="button" class="btn btn-ghost" id="btn-scr-none">
        Clear
      </button>
    </div>
  </div>

  <div class="card">
    <h2>Mission script</h2>
    <p style="color:#aab; font-size:.85rem; margin:0 0 .4rem 0;">
      One command per line. Comments start with <code>#</code>.
      Commands: TAKEOFF, APPROACH, HOOVER, AWAIT, PAUSE, LAND, HEIGHT,
      TO, DANCE.
    </p>
    <textarea id="scr-text" rows="16" spellcheck="false"
              style="width:100%; box-sizing:border-box;"
              placeholder="TAKEOFF&#10;APPROACH 4 1.0&#10;HOOVER 3&#10;LAND"></textarea>
    <div style="display:flex; gap:.5rem; margin-top:.5rem; flex-wrap:wrap;
                align-items:center;">
      <button type="button" class="btn" id="btn-scr-push">
        Push as draft to selected
      </button>
      <button type="button" class="btn btn-good" id="btn-scr-start">
        Push and START on selected
      </button>
      <button type="button" class="btn btn-warn" id="btn-scr-stop">
        STOP on selected
      </button>
      <span id="scr-msg" style="font-size:.85rem; color:#aab;"></span>
    </div>
    <pre id="scr-results"
         style="margin-top:.6rem; background:#0c0f12; border-radius:6px;
                padding:.5rem; max-height:240px; overflow:auto;
                font-size:.8rem; color:#aab;"></pre>
  </div>
</main>

<script>
(function() {
  const $ = id => document.getElementById(id);
  $('btn-scr-all').addEventListener('click', () =>
    document.querySelectorAll('#scr-fc-checks input').forEach(
      el => el.checked = true));
  $('btn-scr-none').addEventListener('click', () =>
    document.querySelectorAll('#scr-fc-checks input').forEach(
      el => el.checked = false));

  function srcFc() {
    return (document.querySelector('input[name=scr-src]:checked')
            || {}).value;
  }

  $('btn-scr-load-active').addEventListener('click', async () => {
    const src = srcFc();
    if (!src) return;
    const msg = $('scr-load-msg');
    msg.textContent = 'loading active draft from ' + src + '…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/scripts/load-from/' + src,
                            {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        $('scr-text').value = j.text || '';
        msg.textContent = 'loaded active draft from ' + src
                          + ' (' + (j.text || '').split('\\n').length
                          + ' line(s))';
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'load failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'load error: ' + e; msg.style.color = '#f87171';
    }
  });

  async function refreshNamed() {
    const src = srcFc();
    const sel = $('scr-named-select');
    sel.innerHTML = '<option value="">— named scripts —</option>';
    if (!src) return;
    try {
      const r = await fetch('/api/c2/scripts/list-saved/' + src,
                            {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        const scripts = j.scripts || [];
        for (const s of scripts) {
          const opt = document.createElement('option');
          opt.value = s.name;
          const isDefault = s.is_default ? ' (default)' : '';
          opt.textContent = s.name + isDefault;
          sel.appendChild(opt);
        }
        if (!scripts.length) {
          sel.innerHTML = '<option value="">'
            + '— no named scripts on ' + src + ' —</option>';
        }
      }
    } catch (e) {}
  }

  $('btn-scr-refresh-named').addEventListener('click', refreshNamed);
  document.addEventListener('change', ev => {
    if (ev.target && ev.target.name === 'scr-src') refreshNamed();
  });

  $('btn-scr-load-named').addEventListener('click', async () => {
    const src = srcFc();
    const name = $('scr-named-select').value;
    const msg = $('scr-load-msg');
    if (!src) { msg.textContent = 'pick a source FC'; msg.style.color='#f87171';
                return; }
    if (!name) { msg.textContent = 'pick a named script';
                  msg.style.color='#f87171'; return; }
    msg.textContent = 'loading ' + name + ' from ' + src + '…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/scripts/load-named/' + src
                            + '/' + encodeURIComponent(name),
                            {cache:'no-store'});
      const j = await r.json();
      if (j.ok) {
        $('scr-text').value = j.text || '';
        msg.textContent = 'loaded ' + (j.name || name) + ' from ' + src;
        msg.style.color = '#86efac';
      } else {
        msg.textContent = 'load failed: ' + (j.error || '');
        msg.style.color = '#f87171';
      }
    } catch (e) {
      msg.textContent = 'load error: ' + e; msg.style.color = '#f87171';
    }
  });

  // Initial named-list fetch.
  refreshNamed();

  function selected() {
    return Array.from(
      document.querySelectorAll('#scr-fc-checks input:checked')
    ).map(el => el.dataset.fc);
  }
  function renderResults(j) {
    const results = j.results || {};
    const lines = [];
    let ok = 0, fail = 0;
    for (const name of Object.keys(results)) {
      if (results[name].ok) { ok++; lines.push('OK    ' + name); }
      else { fail++; lines.push('FAIL  ' + name + '  '
                                 + (results[name].error || '')); }
    }
    $('scr-results').textContent = lines.join('\\n');
    return {ok, fail};
  }
  async function broadcast(action) {
    const fcs = selected();
    const msg = $('scr-msg');
    if (!fcs.length) {
      msg.textContent = 'no FCs selected'; msg.style.color = '#f87171';
      return;
    }
    const text = $('scr-text').value;
    msg.textContent = action + ' on ' + fcs.length + ' FC(s)…';
    msg.style.color = '#aab';
    try {
      const r = await fetch('/api/c2/scripts/' + action,
                            {method:'POST',
                             headers:{'Content-Type':'application/json'},
                             body: JSON.stringify({fcs, text})});
      const j = await r.json();
      const stat = renderResults(j);
      msg.textContent = stat.ok + ' ok, ' + stat.fail + ' failed';
      msg.style.color = stat.fail ? '#f87171' : '#86efac';
    } catch (e) {
      msg.textContent = action + ' error: ' + e;
      msg.style.color = '#f87171';
    }
  }
  $('btn-scr-push').addEventListener('click', () => broadcast('push'));
  $('btn-scr-start').addEventListener('click', () => broadcast('start'));
  $('btn-scr-stop').addEventListener('click', () => broadcast('stop'));
})();
</script>
"""
