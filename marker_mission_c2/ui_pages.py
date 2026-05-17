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
    <h2>Source FC</h2>
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
    <h2>Live calibration tiles (per FC)</h2>
    <p style="color:#aab; font-size:.85rem; margin:0 0 .5rem 0;">
      Each tile embeds the FC's own <code>/calibrate</code> page so it can
      drive that FC's connected drone directly. Iframes load lazily —
      scroll to wake them.
    </p>
    <div class="grid"
         style="grid-template-columns:repeat(auto-fit,minmax(420px,1fr));">
      {% for fc in fc_specs %}
      <div class="card" style="background:#0c0f12;">
        <div style="display:flex; align-items:center; gap:.5rem;
                    margin-bottom:.4rem;">
          <strong style="font-size:.9rem;">{{ fc.name }}</strong>
          <a href="http://{{ fc.host }}:{{ fc.port }}/calibrate"
             target="_blank" rel="noopener"
             style="margin-left:auto; font-size:.78rem;">open ↗</a>
        </div>
        <iframe src="http://{{ fc.host }}:{{ fc.port }}/calibrate"
                loading="lazy"
                sandbox="allow-scripts allow-same-origin allow-forms"
                style="width:100%; height:520px; border:0;
                       background:#0c0f12; border-radius:6px;"></iframe>
      </div>
      {% endfor %}
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


# ---------------------------------------------------------------- scripts

PAGE_SCRIPTS = """
<main>
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
