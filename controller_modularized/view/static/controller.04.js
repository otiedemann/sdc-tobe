
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
    if (!confirm('⚠ Wi-Fi change will disconnect the drone for ~5-10 s.\n\n' +
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
  const k = e.key.toLowerCase();
  // ── SAFETY HOTKEYS — always top priority ────────────────────────────
  // 0 (LAND ALL) and 9 (PAUSE / CONTINUE) are the fleet-level emergency
  // controls. They MUST fire even when a preset-name textbox, slider,
  // code editor, or any other input has focus — otherwise "press 0 to
  // land" becomes "sometimes presses 0 to land" which is unacceptable
  // for safety. We bypass the _isTyping() guard for these two keys.
  // Movement keys still respect _isTyping() so typing "w" in a textbox
  // doesn't fly the drone.
  if (k === '0') {
    e.preventDefault();
    // Blur any text field so the keystroke doesn't also land in it.
    if (document.activeElement && document.activeElement !== document.body) {
      try { document.activeElement.blur(); } catch {}
    }
    if (window._landAllInFlight) return;
    window._landAllInFlight = true;
    console.log('[LAND_ALL] 0 pressed — landing every drone (top-priority hotkey)');
    landAllDrones('0 hotkey').finally(() => { window._landAllInFlight = false; });
    return;
  }
  if (k === '9') {
    e.preventDefault();
    if (document.activeElement && document.activeElement !== document.body) {
      try { document.activeElement.blur(); } catch {}
    }
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    console.log('[PAUSE] 9 pressed — toggling fleet pause (top-priority hotkey)');
    if (window._globalPaused) {
      console.log('[PAUSE] 9 pressed — resuming fleet');
      resumeAllDrones('9 hotkey').finally(() => { window._pauseInFlight = false; });
    } else {
      console.log('[PAUSE] 9 pressed — pausing fleet');
      pauseAllDrones('9 hotkey').finally(() => { window._pauseInFlight = false; });
    }
    return;
  }
  // Below this point, non-safety keys respect the typing guard.
  if (_isTyping()) return;
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

// ── PAUSE ALL / CONTINUE MISSION wiring ───────────────────────────────
// The PAUSE button overrides any command — every drone freezes at its
// current position (autonomous missions abort, ArUco Seek drops out of
// LIVE, and a zero-RC brake goes out to every drone). Only manual WASD
// / RC remains active. CONTINUE MISSION clears the flag; nothing
// auto-restarts so the operator is never surprised by a darting drone.
window._globalPaused = false;
function applyPauseUI(paused, info) {
  window._globalPaused = !!paused;
  const pbtn = document.getElementById('pause_all_btn');
  const rbtn = document.getElementById('resume_all_btn');
  const ban  = document.getElementById('global_pause_banner');
  if (pbtn) pbtn.style.display = paused ? 'none' : '';
  if (rbtn) rbtn.style.display = paused ? '' : 'none';
  if (ban)  ban.style.display  = paused ? '' : 'none';
  document.body.classList.toggle('paused-mode', !!paused);
  if (paused && info && info.source) {
    if (ban) ban.title = 'paused via ' + info.source;
  }
}
async function pauseAllDrones(source) {
  try {
    const r = await fetch('/proxy/pause_all', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({source: source || 'ui'}),
    });
    const j = await r.json();
    console.log('[PAUSE] result:', j);
    applyPauseUI(true, j);
    showLandAllBanner(
      '\u23f8 PAUSED — ' + (j.braked || 0) + '/' + (j.total || 0) +
      ' drones braked' + (j.mission_stopped ? ' (mission stopped)' : ''),
      '#78350f', '#fde68a', 4000);
  } catch (err) {
    console.error('[PAUSE] failed:', err);
    showLandAllBanner('\u2717 PAUSE failed: ' + err, '#7f1d1d', '#fecaca', 5000);
  }
}
async function resumeAllDrones(source) {
  try {
    const r = await fetch('/proxy/resume_all', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({source: source || 'ui'}),
    });
    const j = await r.json();
    console.log('[RESUME] result:', j);
    applyPauseUI(false, j);
    showLandAllBanner('\u25b6 RESUMED — autonomous control re-enabled',
                      '#064e3b', '#a7f3d0', 3500);
  } catch (err) {
    console.error('[RESUME] failed:', err);
    showLandAllBanner('\u2717 RESUME failed: ' + err, '#7f1d1d', '#fecaca', 5000);
  }
}
(function wirePauseButtons(){
  const p = document.getElementById('pause_all_btn');
  if (p) p.addEventListener('click', () => {
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    pauseAllDrones('button').finally(() => { window._pauseInFlight = false; });
  });
  const r = document.getElementById('resume_all_btn');
  if (r) r.addEventListener('click', () => {
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    resumeAllDrones('button').finally(() => { window._pauseInFlight = false; });
  });
})();
// Pause state sync is now handled by the unified uiStatePoll() above.
// Retained applyPauseUI() — it's called from there and from hotkey handlers.

// ── Transport selector (per-subsystem WS ↔ HTTP) ──────────────────────
// Populated from /proxy/config/transport on load + every time the
// server is polled via the WS status check. POSTs on every dropdown
// change so the server state always matches the UI.
(function wireTransport(){
  const selectors = document.querySelectorAll('.transport-sel');
  const status = document.getElementById('transport_status');
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }
  async function apply(patch, label) {
    try {
      const r = await fetch('/proxy/config/transport', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(patch),
      });
      const j = await r.json();
      if (j.ok) {
        // Reflect server's authoritative state in case some keys were rejected.
        Object.keys(j.transport || {}).forEach(k => {
          const el = document.getElementById('transport_' + k);
          if (el) el.value = j.transport[k];
        });
        flash('\u2713 ' + (label || 'applied'), '#22c55e');
      } else {
        flash('error: ' + (j.error || 'unknown'), '#ef4444');
      }
    } catch (e) { flash('request failed', '#ef4444'); }
  }
  selectors.forEach(el => {
    el.addEventListener('change', () => {
      apply({[el.dataset.subsys]: el.value}, el.dataset.subsys + '=' + el.value);
    });
  });
  const allHttp = document.getElementById('transport_all_http');
  if (allHttp) allHttp.onclick = () =>
    apply({rc: 'http', telemetry: 'http', position: 'http'}, 'all → http');
  const allAuto = document.getElementById('transport_all_auto');
  if (allAuto) allAuto.onclick = () =>
    apply({rc: 'auto', telemetry: 'auto', position: 'auto'}, 'all → auto');

  // Initial load — fetch current state from the server.
  (async () => {
    try {
      const j = await (await fetch('/proxy/config/transport')).json();
      Object.keys(j.transport || {}).forEach(k => {
        const el = document.getElementById('transport_' + k);
        if (el) el.value = j.transport[k];
      });
      if (!j.ws_available) {
        document.querySelectorAll('.transport-sel').forEach(el => {
          // Disable the WS option when the client library is missing
          el.querySelectorAll('option[value="ws"]').forEach(o => o.disabled = true);
          if (el.value === 'ws') el.value = 'http';
        });
        flash('WS library missing — selectors locked to http/auto', '#fbbf24');
      }
    } catch {}
  })();
})();

// ── WebSocket status badge ────────────────────────────────────────────
// Polls /proxy/ws/status every 2 s. Green when the active drone has
// telemetry + position + rc sockets all up; amber if partial; red if
// the WS client lib is missing or all three are down.
(function wireWsStatus(){
  const el = document.getElementById('ws_status_badge');
  if (!el) return;
  async function tick() {
    try {
      applyWs(await (await fetch('/proxy/ws/status', {cache:'no-store'})).json());
    } catch {
      el.style.background = '#334155';
      el.style.color = '#94a3b8';
      el.textContent = 'WS ?';
    }
  }
  // Exposed on window so the unified uiStatePoll() can drive it without
  // the local timer firing separate /proxy/ws/status requests.
  function applyWs(j) {
    if (!j || !j.available) {
      el.style.background = '#7f1d1d';
      el.style.color = '#fecaca';
      el.textContent = 'WS off';
      el.title = 'websocket-client not installed on C2 — falling back to HTTP';
      return;
    }
    const did = String(window.activeDroneId || activeDroneId);
    const st = (j.drones || {})[did];
    if (!st) {
      el.style.background = '#334155';
      el.style.color = '#94a3b8';
      el.textContent = 'WS —';
      return;
    }
    const up = (st.telemetry ? 1 : 0) + (st.position ? 1 : 0) + (st.rc ? 1 : 0);
    if (up === 3) {
      const slowSend = (st.rc_send_ms != null && st.rc_send_ms > 50);
      el.style.background = slowSend ? '#78350f' : '#064e3b';
      el.style.color      = slowSend ? '#fde68a' : '#86efac';
      const parts = [];
      if (st.rc_send_ms     != null) parts.push('send ' + Math.round(st.rc_send_ms) + 'ms');
      if (st.rc_rtt_ms      != null) parts.push('rtt ' + st.rc_rtt_ms + 'ms');
      if (st.telemetry_age_ms != null) parts.push('tel ' + st.telemetry_age_ms + 'ms');
      if (st.position_age_ms  != null) parts.push('pos ' + st.position_age_ms  + 'ms');
      el.textContent = (slowSend ? 'WS ⚠ ' : 'WS ✓ ') + parts.join(' · ');
    } else if (up > 0) {
      el.style.background = '#78350f';
      el.style.color = '#fde68a';
      const flags = (st.telemetry?'T':'·') + (st.position?'P':'·') + (st.rc?'R':'·');
      el.textContent = 'WS ' + flags;
    } else {
      el.style.background = '#7f1d1d';
      el.style.color = '#fecaca';
      el.textContent = 'WS down';
    }
    el.title = 'tel=' + st.telemetry + ' pos=' + st.position + ' rc=' + st.rc;
  }
  window._applyWsStatus = applyWs;
  tick();  // one-time initial fetch; steady state is pushed by uiStatePoll()
})();

// ── Flight Logs list ──────────────────────────────────────────────────
// Lists archived per-flight JSONL files. Polls occasionally so a live
// recording's size updates visibly (the file grows as ticks accumulate).
(function wireFlightLogs(){
  const list = document.getElementById('flight_logs_list');
  const btn  = document.getElementById('flight_logs_refresh');
  if (!list) return;
  function fmtSize(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
    return (n/1024/1024).toFixed(2) + ' MB';
  }
  function fmtAge(ts) {
    const dt = Math.max(0, Date.now()/1000 - ts);
    if (dt < 60) return Math.round(dt) + 's ago';
    if (dt < 3600) return Math.round(dt/60) + 'm ago';
    if (dt < 86400) return Math.round(dt/3600) + 'h ago';
    return Math.round(dt/86400) + 'd ago';
  }
  async function refresh() {
    try {
      const r = await fetch('/proxy/flight_logs');
      const j = await r.json();
      if (!j.files || !j.files.length) {
        list.innerHTML = '<i style="color:#64748b;">no flights recorded yet</i>';
        return;
      }
      list.innerHTML = j.files.map(f => {
        const vid = f.video;
        const vidHtml = vid
          ? ('<a href="/proxy/flight_video/' + encodeURIComponent(vid.name) + '" ' +
             'style="color:#a78bfa;text-decoration:none;font-weight:600;" ' +
             'title="Download ' + vid.name + ' (' + fmtSize(vid.size || 0) + ')">' +
             '&#127916; Video</a>')
          : '<span style="color:#475569;font-style:italic;" title="No video recorded for this flight">no video</span>';
        return '<div style="display:flex;gap:10px;padding:2px 0;align-items:center;">' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
            '<a href="/proxy/flight_logs/' + encodeURIComponent(f.name) + '" ' +
               'style="color:#38bdf8;text-decoration:none;" title="Download ' + f.name + '">' +
               f.name + '</a>' +
          '</span>' +
          '<a href="/flight_log_viewer?file=' + encodeURIComponent(f.name) + '" ' +
             'target="_blank" ' +
             'style="color:#fbbf24;text-decoration:none;font-weight:600;" ' +
             'title="Open replay viewer in a new tab">&#128065; View</a>' +
          vidHtml +
          '<span style="color:#94a3b8;width:70px;text-align:right;">' + fmtSize(f.size) + '</span>' +
          '<span style="color:#64748b;width:70px;text-align:right;">' + fmtAge(f.mtime) + '</span>' +
        '</div>';
      }).join('');
    } catch (e) {
      list.innerHTML = '<i style="color:#ef4444;">error: ' + e + '</i>';
    }
  }
  if (btn) btn.onclick = refresh;
  refresh();
  setInterval(refresh, 15000);  // 15s — enough to see new flights appear
})();

// ── Flight guards: axis-lock + arena safety (manual + autonomous) ─
(function wireFlightGuards(){
  const axisTog   = document.getElementById('axis_locked_toggle');
  const arenaTog  = document.getElementById('arena_guard_toggle');
  const camFaceTog = document.getElementById('cam_face_center_toggle');
  const inp       = document.getElementById('safety_margin_input');
  const btn       = document.getElementById('safety_margin_apply');
  const status    = document.getElementById('autonomous_guards_status');
  const badge     = document.getElementById('arena_guard_engaged_badge');
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }

  async function load() {
    try {
      // axis-lock (autonomous-observer param)
      const p = await (await fetch('/proxy/aruco/params')).json();
      if (axisTog && typeof p.axis_locked !== 'undefined') axisTog.checked = !!p.axis_locked;
      // Camera-faces-arena-centre toggle (C2-local fleet-wide setting)
      try {
        const cf = await (await fetch('/proxy/config/camera_face_center')).json();
        if (camFaceTog && typeof cf.enabled !== 'undefined') camFaceTog.checked = !!cf.enabled;
      } catch {}
      // arena guard + margin (Pi-side, enforces on BOTH manual + auto)
      const a = await (await fetch('/proxy/config/arena_safety')).json();
      if (arenaTog && typeof a.enabled !== 'undefined') arenaTog.checked = !!a.enabled;
      if (inp && a.margin_m != null && document.activeElement !== inp)
        inp.value = Number(a.margin_m).toFixed(1);
      if (badge) badge.style.display = a.engaged ? '' : 'none';
      // Cache on window so drawArena + 3D + position readout can render
      // the dashed safety boundary + distance-to-wall readout.
      window._arenaSafety = {
        enabled:  !!a.enabled,
        margin_m: a.margin_m,
        engaged:  !!a.engaged,
        reasons:  a.reasons || [],
      };
      // Redraw the 2D arena so the updated margin rectangle appears
      // even if no fresh position event has arrived (e.g. operator
      // changed the margin value while the drone is stationary).
      if (typeof drawArena === 'function') drawArena();
      // Update the 3D highlight too — safety overlay mesh refresh
      if (window._arena3d && window._arena3d.updateSafetyMargin) {
        window._arena3d.updateSafetyMargin(a.margin_m, !!a.engaged);
      }
    } catch {}
  }
  // Initial load only — subsequent updates ride on the unified 1Hz
  // /proxy/ui_state poll (see uiStatePoll). Avoids adding another
  // per-2s fetch cycle that would compete with the browser's 6-
  // connection HTTP/1.1 pool alongside keydown/batch traffic.
  load();
  // Expose for the unified poller
  window._applyArenaSafety = (a) => {
    if (!a) return;
    if (arenaTog && typeof a.enabled !== 'undefined') arenaTog.checked = !!a.enabled;
    if (inp && a.margin_m != null && document.activeElement !== inp)
      inp.value = Number(a.margin_m).toFixed(1);
    if (badge) badge.style.display = a.engaged ? '' : 'none';
    window._arenaSafety = {
      enabled:  !!a.enabled,
      margin_m: a.margin_m,
      engaged:  !!a.engaged,
      reasons:  a.reasons || [],
    };
    if (typeof drawArena === 'function') drawArena();
    if (window._arena3d && window._arena3d.updateSafetyMargin) {
      window._arena3d.updateSafetyMargin(a.margin_m, !!a.engaged);
    }
  };

  if (axisTog) axisTog.addEventListener('change', async () => {
    try {
      await fetch('/proxy/aruco/params', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({axis_locked: axisTog.checked}),
      });
      flash(axisTog.checked ? '✓ axis-lock ON' : '✓ axis-lock OFF', '#22c55e');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (camFaceTog) camFaceTog.addEventListener('change', async () => {
    try {
      const r = await fetch('/proxy/config/camera_face_center', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({enabled: camFaceTog.checked}),
      });
      const j = await r.json();
      if (j.ok) flash(camFaceTog.checked
                        ? '✓ camera → arena centre (fleet)'
                        : '✓ camera free (mission-driven)',
                       '#22c55e');
      else flash('error: ' + (j.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (arenaTog) arenaTog.addEventListener('change', async () => {
    try {
      const r = await fetch('/proxy/config/arena_safety', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({enabled: arenaTog.checked}),
      });
      const j = await r.json();
      if (j.ok) flash(arenaTog.checked
                        ? '✓ arena guard ON (manual + auto)'
                        : '⚠ arena guard OFF — operator owns boundaries',
                       arenaTog.checked ? '#22c55e' : '#fbbf24');
      else flash('error: ' + (j.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (btn) btn.onclick = async () => {
    const v = parseFloat(inp.value);
    if (!isFinite(v) || v < 0.1 || v > 5.0) {
      flash('enter 0.1 - 5.0 m', '#ef4444');
      return;
    }
    // Push to BOTH the Pi-side guard (manual + auto) AND the observer
    // autonomous guard so they share a single source of truth.
    try {
      const [a, b] = await Promise.all([
        fetch('/proxy/config/arena_safety', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({margin_m: v}),
        }).then(r => r.json()).catch(e => ({ok:false, error:String(e)})),
        fetch('/proxy/aruco/safety_margin', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({safety_margin_m: v}),
        }).then(r => r.json()).catch(e => ({ok:false, error:String(e)})),
      ]);
      if (a.ok && b.ok) flash('✓ margin → ' + v + ' m (Pi + observers)', '#22c55e');
      else flash('partial: pi=' + (a.ok?'ok':'err') + ' obs=' + (b.ok?'ok':'err'), '#fbbf24');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
})();

// ── Ceiling safety wiring ─────────────────────────────────────────────
// The ceiling is enforced on the Pi (can't be bypassed by client code),
// but the UI mirrors its value and flashes an alert banner when the
// guard is actively clamping any drone. Poll every ~1 s so operators
// see engagement immediately on overshoot.
(function wireCeiling(){
  const inp = document.getElementById('ceiling_input');
  const btn = document.getElementById('ceiling_apply_btn');
  const badge = document.getElementById('ceiling_engaged_badge');
  const status = document.getElementById('ceiling_status');
  function apply(j) {
    if (!j) return;
    if (typeof j.ceiling_m === 'number') {
      if (document.activeElement !== inp) inp.value = Number(j.ceiling_m).toFixed(1);
    }
    if (badge) badge.style.display = j.engaged ? '' : 'none';
    if (status) {
      if (j.engaged && j.reasons && j.reasons.length) {
        status.textContent = '⚠ ' + j.reasons[0];
        status.style.color = '#fbbf24';
      } else {
        status.textContent = '';
      }
    }
  }
  // Exposed for the unified uiStatePoll — no local setInterval anymore.
  window._applyCeilingStatus = apply;
  async function load() {
    try {
      const j = await (await fetch('/proxy/config/ceiling', {cache:'no-store'})).json();
      apply(j);
    } catch {}
  }
  if (btn) btn.onclick = async () => {
    const v = parseFloat(inp.value);
    if (!isFinite(v) || v < 0.5 || v > 20) {
      status.textContent = 'enter 0.5 - 20 m';
      status.style.color = '#ef4444';
      return;
    }
    status.textContent = 'applying...';
    status.style.color = '#94a3b8';
    try {
      const r = await fetch('/proxy/config/ceiling', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ceiling_m: v}),
      });
      const j = await r.json();
      status.textContent = '✓ ' + (j.applied || 0) + '/' + (j.total || 0) + ' drones → ' + v + ' m';
      status.style.color = '#22c55e';
      console.log('[CEILING] apply result:', j);
    } catch (err) {
      status.textContent = 'failed: ' + err;
      status.style.color = '#ef4444';
    }
  };
  load();   // one-time initial fetch; steady state pushed by uiStatePoll()
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
        `no live drone telemetry
` +
        `api reachable: yes
` +
        `drone connected: ${t.connected}
` +
        `state age: ${t.state_age_s ?? '-'} s`;
      return;
    }

    const battery = (typeof t.battery === 'number') ? t.battery : null;
    setMeter('battery_bar', 'battery_val', battery, '%');

    document.getElementById('telemetry').textContent =
      `battery: ${t.battery ?? '-'} %
` +
      `temperature: ${t.temperature ?? '-'} °C
` +
      `height: ${t.height_cm ?? '-'} cm
` +
      `tof: ${t.tof_cm ?? '-'} cm
` +
      `barometer: ${t.barometer_cm ?? '-'} cm
` +
      `flight time: ${t.flight_time_s ?? '-'} s
` +
      `speed: ${t.speed ?? '-'}
` +
      `wifi snr: ${t.wifi_snr ?? '-'}
` +
      `attitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}
` +
      `velocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}
` +
      `accel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}
` +
      `sdk version: ${t.sdk_version ?? '-'}
` +
      `serial number: ${t.serial_number ?? '-'}
` +
      `mission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}
` +
      `gps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m
` +
      `gimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}
` +
      `state age: ${t.state_age_s ?? '-'} s
` +
      `flying: ${t.flying}
` +
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
// Fire a bit after page load so the C2 has booted + WS clients settled.
// Heartbeat is handled server-side now — the browser doesn't wait for it.
setTimeout(autoStartVideo, 300);

// ── Anafi camera zoom slider ─────────────────────────────────────────
(function(){
  const sl  = document.getElementById('video_zoom');
  const lbl = document.getElementById('video_zoom_val');
  const rst = document.getElementById('video_zoom_reset');
  if (!sl || !lbl) return;
  let _zoomTimer = null;
  function sendZoom(v) {
    fetch('/proxy/camera/zoom', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({zoom: v})}).catch(()=>{});
  }
  sl.addEventListener('input', () => {
    const v = Number(sl.value);
    lbl.textContent = v.toFixed(2) + '×';
    // Debounce — don't spam the drone with 100+ posts per drag
    if (_zoomTimer) clearTimeout(_zoomTimer);
    _zoomTimer = setTimeout(() => sendZoom(v), 80);
  });
  if (rst) rst.addEventListener('click', () => {
    sl.value = 1.0;
    lbl.textContent = '1.00×';
    sendZoom(1.0);
  });
})();

// ── Latency measurement ──────────────────────────────────────────────
// Polls /proxy/latency every 2s. Total displayed = c2_to_fc + fc_to_drone
// + video_offset (operator-configurable). When the "auto-set latency"
// checkbox is ticked, the total is pushed into the Position Tracker's
// latency_ms slider each poll so position fusion stays in sync with the
// actual comm-stack delay.
async function latPoll() {
  try {
    const r = await fetch('/proxy/latency', {cache:'no-store'});
    const d = await r.json();
    const fm = (x) => (x == null) ? '—' : Math.round(x) + ' ms';
    const c2fc = d.c2_to_fc_ms;
    const fcdr = d.fc_to_drone_ms;
    const videoEl = document.getElementById('lat_video_offset');
    const videoMs = videoEl ? (parseInt(videoEl.value, 10) || 0) : 0;
    const totalMs = (c2fc || 0) + (fcdr || 0) + videoMs;
    document.getElementById('lat_c2fc').textContent = fm(c2fc);
    document.getElementById('lat_fcdr').textContent = fm(fcdr);
    document.getElementById('lat_vid').textContent = fm(videoMs);
    const tEl = document.getElementById('lat_total');
    tEl.textContent = Math.round(totalMs) + ' ms';
    // Colour total: <80ms green, <200ms amber, else red
    tEl.style.color = (totalMs < 80) ? '#22c55e'
                    : (totalMs < 200) ? '#fbbf24' : '#ef4444';
    // Auto-push into Position Tracker slider if toggle on
    const auto = document.getElementById('lat_auto_apply');
    if (auto && auto.checked && c2fc != null && fcdr != null) {
      const slider = document.getElementById('pos_latency');
      const lbl = document.getElementById('pos_latency_val');
      if (slider) {
        slider.value = Math.round(totalMs);
        if (lbl) lbl.textContent = Math.round(totalMs);
        // Mirror to server so the positioning pipeline uses this too
        fetch('/proxy/position/config', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({latency_ms: Math.round(totalMs)})}).catch(()=>{});
      }
    }
  } catch {}
}
setInterval(latPoll, 2000);
setTimeout(latPoll, 800);

// ── Collapsible panels ───────────────────────────────────────────────
// Wrap a panel's existing content under a click-to-toggle header.
// If the panel already contains an h2/h3 as its first significant
// child, reuse it as the toggle; otherwise inject a new header with
// `defaultTitle`. State persists in localStorage per `storageKey` so
// each operator keeps their preferred layout across page reloads.
function makeCollapsible(el, defaultTitle, storageKey, startCollapsed) {
  if (!el || el.classList.contains('collapsible')) return;
  // Prefer an existing h2/h3 as the toggle header
  let header = el.querySelector(':scope > h2, :scope > h3');
  let body;
  if (header) {
    body = document.createElement('div');
    body.className = 'collapsible-body';
    const siblings = Array.from(el.children).filter(c => c !== header);
    siblings.forEach(c => body.appendChild(c));
    el.appendChild(body);
  } else {
    body = document.createElement('div');
    body.className = 'collapsible-body';
    while (el.firstChild) body.appendChild(el.firstChild);
    header = document.createElement('div');
    header.innerHTML = '<b>' + defaultTitle + '</b>';
    el.appendChild(header);
    el.appendChild(body);
  }
  header.classList.add('collapsible-toggle');
  el.classList.add('collapsible');
  header.addEventListener('click', (e) => {
    // Don't toggle if the click was on an interactive child
    const t = e.target;
    if (t !== header && (t.tagName === 'INPUT' || t.tagName === 'BUTTON' ||
                         t.tagName === 'SELECT' || t.tagName === 'A' ||
                         t.tagName === 'TEXTAREA')) return;
    el.classList.toggle('collapsed');
    try { localStorage.setItem(storageKey,
          el.classList.contains('collapsed') ? '1' : '0'); } catch {}
  });
  // Load persisted state (fall back to startCollapsed default)
  let saved = null;
  try { saved = localStorage.getItem(storageKey); } catch {}
  const shouldCollapse = (saved === null) ? Boolean(startCollapsed)
                                           : (saved === '1');
  if (shouldCollapse) el.classList.add('collapsed');
}

// Find a .panel that contains a <b> with the given exact text.
function _panelByBoldTitle(title) {
  const panels = Array.from(document.querySelectorAll('.panel'));
  for (const p of panels) {
    const b = p.querySelector('b');
    if (b && b.textContent.trim() === title) return p;
  }
  return null;
}

// Run after the DOM is settled — some panels are only fully populated
// after the first polls (e.g. drone bar) but their structure is fixed.
// ── Relocate tuning controls into the new unified Tuning Parameters
// panel. Runs BEFORE makeCollapsible so the panel's starting state is
// set correctly. Keeps existing DOM nodes intact (IDs, handlers, all
// event bindings) — we just re-parent them. The original ArUco Seek
// panel keeps only the live readout; the Position Tracker panel keeps
// only the canvas + coordinate readouts. Everything else moves here.
function relocateTuningControls() {
  const obsSlot = document.getElementById('tuning_observer_slot');
  const posSlot = document.getElementById('tuning_position_slot');
  if (!obsSlot || !posSlot) return;

  // Observer PD — the slider grid + Reload button container
  const arcParamsWrap = document.getElementById('arc_params');
  if (arcParamsWrap && arcParamsWrap.parentElement) {
    // The adjacent 'Live tuning parameters' label + Reload button are
    // siblings of #arc_params inside the same wrapper <div>. Move the
    // whole wrapper for convenience.
    const wrap = arcParamsWrap.parentElement;
    obsSlot.appendChild(wrap);
  }

  // Position Tracker — the .pos-cfg div holding all the fusion sliders.
  // The tracker panel keeps its canvas + numeric readouts; the config
  // rows (Profile / FOV / Latency / IMU blend / Kalman / Marker size /
  // Top-K / Outlier / Apply Config) move over.
  const posCfgWrap = document.querySelector('#pos_panel .pos-cfg') ||
                     Array.from(document.querySelectorAll('.pos-cfg'))
                       .find(el => el.querySelector('#pos_profile'));
  if (posCfgWrap) posSlot.appendChild(posCfgWrap);
}
relocateTuningControls();

setTimeout(() => {
  // id-addressable panels
  makeCollapsible(document.getElementById('tuning_panel'),     'Tuning Parameters',     'collapsed_tuning',           true);
  makeCollapsible(document.getElementById('mission_panel'),    'Mission Planner',       'collapsed_mission_planner',  true);
  makeCollapsible(document.getElementById('anafi_panel'),      'Anafi / Olympe controls','collapsed_anafi',           true);
  makeCollapsible(document.getElementById('video_panel'),      'Video stream',          'collapsed_video',            false);
  makeCollapsible(document.getElementById('aruco_panel'),      'ArUco Seek',            'collapsed_aruco',            true);
  makeCollapsible(document.getElementById('missions_panel'),   'Special Missions',      'collapsed_missions',         true);

  // Panels addressed by their <b>-wrapped title
  makeCollapsible(_panelByBoldTitle('Telemetry'),          'Telemetry',          'collapsed_telemetry',  false);
  makeCollapsible(_panelByBoldTitle('Position Tracker'),   'Position Tracker',   'collapsed_pos_tracker', false);
  makeCollapsible(_panelByBoldTitle('Arena Configuration'),'Arena Configuration','collapsed_arena_cfg',  true);

  // WASD grid panel — first .panel that contains a .grid
  (function wrapKeyPanel() {
    const p = document.querySelector('.panel:has(> .grid)') ||
              Array.from(document.querySelectorAll('.panel')).find(
                x => x.querySelector(':scope > .grid'));
    if (p) makeCollapsible(p, 'WASD Key controls', 'collapsed_keys', false);
  })();

  // Advanced SDK controls — .adv block with the "Advanced SDK controls" label
  (function wrapAdvPanel() {
    const advs = document.querySelectorAll('.adv');
    for (const a of advs) {
      const lbl = a.querySelector(':scope > .small');
      if (lbl && lbl.textContent.trim().startsWith('Advanced SDK controls')) {
        // Replace the tiny 'small' label with a proper header
        lbl.remove();
        makeCollapsible(a, 'Advanced SDK controls', 'collapsed_adv_sdk', true);
        return;
      }
    }
  })();
}, 150);

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
// ── UNIFIED STATUS POLL ─────────────────────────────────────────────
// Replaces /proxy/heartbeat + /proxy/pause_status + /proxy/ws/status +
// /proxy/config/ceiling + /proxy/config/transport + /proxy/missions/status
// with ONE request at 1 Hz. The C2 aggregates them all locally (all are
// in-memory dict reads) — this frees the browser's 6-connection pool so
// CONTROL traffic (WASD, takeoff, rc) goes through without queueing.
//
// Control latency (keypress → drone) is completely unaffected: those go
// via /proxy/key_down | key_up | key_batch which are NOT part of this
// unified poll. Only the low-frequency UI bookkeeping lives here.
let _uiPollSeq = 0;
async function uiStatePoll(){
  const mySeq = ++_uiPollSeq;
  try {
    const r = await fetch('/proxy/ui_state', {cache:'no-store'});
    if (_uiPollSeq !== mySeq) return;   // a newer poll already fired
    const j = await r.json();
    // Distribute to the individual consumers
    if (j.pause && typeof applyPauseUI === 'function') {
      if (!!j.pause.paused !== !!window._globalPaused) applyPauseUI(j.pause.paused, j.pause);
    }
    if (j.ceiling && typeof window._applyCeilingStatus === 'function') {
      window._applyCeilingStatus(j.ceiling);
    }
    if (j.arena_safety && typeof window._applyArenaSafety === 'function') {
      window._applyArenaSafety(j.arena_safety);
    }
    if (j.ws && typeof window._applyWsStatus === 'function') {
      window._applyWsStatus(j.ws);
    }
    if (j.transport && typeof window._applyTransportState === 'function') {
      window._applyTransportState(j.transport);
    }
    if (j.missions && typeof window._applyMissionsStatus === 'function') {
      window._applyMissionsStatus(j.missions);
    }
  } catch {}
}
setInterval(uiStatePoll, 1000);
uiStatePoll();

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
  const fmt = v => (v != null && !isNaN(v)) ? Number(v).toFixed(1) : '\u2014';
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
      `no live drone telemetry\napi reachable: yes\ndrone connected: ${t.connected}\nstate age: ${t.state_age_s ?? '-'} s`;
    return;
  }
  const battery = (typeof t.battery === 'number') ? t.battery : null;
  setMeter('battery_bar', 'battery_val', battery, '%');
  document.getElementById('telemetry').textContent =
    `battery: ${t.battery ?? '-'} %\ntemperature: ${t.temperature ?? '-'} °C\nheight: ${t.height_cm ?? '-'} cm\ntof: ${t.tof_cm ?? '-'} cm\nbarometer: ${t.barometer_cm ?? '-'} cm\nflight time: ${t.flight_time_s ?? '-'} s\nspeed: ${t.speed ?? '-'}\nwifi snr: ${t.wifi_snr ?? '-'}\nattitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\nvelocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\naccel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\nsdk version: ${t.sdk_version ?? '-'}\nserial number: ${t.serial_number ?? '-'}\nmission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}\ngps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m\ngimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}\nstate age: ${t.state_age_s ?? '-'} s\nflying: ${t.flying}\nconnected: ${t.connected}`;

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
  el.textContent += `[${t}] ${line}\n`;
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
        (function(){ const s = (msg.post.status||'').toLowerCase().replace(/\s+/g,'');
          const m = s.match(/axes=x(\d)y(\d)z(\d)/);
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
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

async function runMission() {
  if (missionRunning) return;
  const textarea = document.getElementById('mission_cmds');
  const lines = textarea.value.split('\n').map(l=>l.trim()).filter(l=>l && !l.startsWith('#'));
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

    const parts = line.toLowerCase().split(/\s+/);
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
let arenaW = 20, arenaD = 10.8, arenaOX = -10, arenaOY = 0;
// Tight 1 m border around the arena — the arena fills most of the view,
// and out-of-bounds positions beyond 1 m get clamped with an OOB arrow.
const VIEW_MARGIN = 1;  // metres
let viewOX = -11, viewOY = -1, viewW = 22, viewD = 12.8;

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

  // ── Team-zone shading (SDC26: red home zone LEFT, blue home zone RIGHT,
  //    neutral middle). Splits the 20 m length into three equal thirds.
  (function shadeZones() {
    const thirdW = (arenaW) / 3;
    // Red zone: x ∈ [arenaOX, arenaOX + thirdW]
    const [rx0] = arenaToCanvas(arenaOX, arenaOY);
    const [rx1] = arenaToCanvas(arenaOX + thirdW, arenaOY);
    ctx.fillStyle = 'rgba(239,68,68,0.06)';
    ctx.fillRect(rx0, ay0, rx1 - rx0, ay1 - ay0);
    // Blue zone: x ∈ [arenaOX + 2·thirdW, arenaOX + arenaW]
    const [bx0] = arenaToCanvas(arenaOX + 2 * thirdW, arenaOY);
    const [bx1] = arenaToCanvas(arenaOX + arenaW, arenaOY);
    ctx.fillStyle = 'rgba(59,130,246,0.06)';
    ctx.fillRect(bx0, ay0, bx1 - bx0, ay1 - ay0);
  })();

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

  // ── Safety margin rectangle ────────────────────────────────────
  // Draws the Pi-side arena guard's inner boundary as a dashed red
  // outline and shades the restricted zone (between the outline and
  // the arena wall) with a faint red tint. The drone is not allowed
  // to cross the inner boundary during autonomous flight OR manual
  // flight (when the guard is ON). Value comes from the last
  // /proxy/config/arena_safety poll, cached in window._arenaSafety.
  (function drawSafetyMargin() {
    const sa = window._arenaSafety || {};
    const margin = (typeof sa.margin_m === 'number' && sa.margin_m > 0)
                     ? sa.margin_m : null;
    if (margin == null) return;
    // Inner boundary rectangle — arena bounds minus margin.
    const [sx0, sy0] = arenaToCanvas(arenaOX + margin, arenaOY + arenaD - margin);
    const [sx1, sy1] = arenaToCanvas(arenaOX + arenaW - margin, arenaOY + margin);
    // Faint red shading in the restricted band (between margin and wall).
    // We paint four rectangles: top/bottom/left/right edges.
    ctx.fillStyle = 'rgba(239,68,68,0.08)';
    ctx.fillRect(ax0, ay0, ax1 - ax0, sy0 - ay0);                  // top (y=max zone)
    ctx.fillRect(ax0, sy1, ax1 - ax0, ay1 - sy1);                  // bottom
    ctx.fillRect(ax0, sy0, sx0 - ax0, sy1 - sy0);                  // left
    ctx.fillRect(sx1, sy0, ax1 - sx1, sy1 - sy0);                  // right
    // Dashed inner boundary — brighter if the guard is currently engaged.
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = sa.engaged ? 2 : 1.5;
    ctx.strokeStyle = sa.engaged ? '#ef4444' : '#f87171';
    ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
    ctx.restore();
    // Tiny label so operators know what the dashed rect means
    ctx.font = '9px monospace';
    ctx.fillStyle = '#fca5a5';
    ctx.textAlign = 'left';
    ctx.fillText('safe @ -' + margin.toFixed(1) + 'm', sx0 + 4, sy0 + 11);
  })();

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

  // ── Bucket markers that share the same (x,y) top-down projection ──
  // The SDC arena uses stacked pairs (low at z≈2 m, high at z≈4 m) on
  // every wall. On the 2D top-down they collapse to identical pixels.
  // Drawing them individually means the label of the later-iterated ID
  // hides the first. We group by rounded canvas coordinate and render
  // a single combined glyph with a multi-ID label ("1/2") where each
  // individual ID is coloured by its own seen/ref state.
  const _bucketKey = (x, y) => Math.round(x) + ',' + Math.round(y);
  const buckets = new Map();
  for (const [id, m] of Object.entries(arenaMarkers)) {
    if (!m.pos) continue;
    const [mx, my] = arenaToCanvas(m.pos[0], m.pos[1]);
    if (mx < PAD - 4 || mx > W - PAD + 4 || my < PAD - 4 || my > H - PAD + 4) continue;
    const key = _bucketKey(mx, my);
    let b = buckets.get(key);
    if (!b) {
      b = {cx: mx, cy: my, wall: m.wall, entries: []};
      buckets.set(key, b);
    }
    b.entries.push({
      id:   String(id),
      seen: _seenMarkers.has(String(id)),
      ref:  _refMarkers.has(String(id)),
      z:    (m.pos[2] != null) ? Number(m.pos[2]) : 0,
    });
  }

  for (const b of buckets.values()) {
    // Sort HIGH altitude first so the top row of the label is the
    // upper marker (matches the physical stacking on the wall: low ID
    // paints low, high ID paints high in the same pixel column).
    b.entries.sort((a, b) => b.z - a.z);
    const mx = b.cx, my = b.cy;
    const anySeen = b.entries.some(e => e.seen);
    const anyRef  = b.entries.some(e => e.ref);
    const baseColor = WALL_COLOR[b.wall] || '#94a3b8';

    // Halo whenever ANY marker in the stack is seen.
    if (anySeen) {
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.fillStyle = (anyRef ? 'rgba(34,197,94,0.35)' : 'rgba(251,191,36,0.35)');
      ctx.fill();
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.strokeStyle = (anyRef ? '#22c55e' : '#fbbf24');
      ctx.lineWidth = 1.5; ctx.stroke();
    }

    // Square marker glyph — same shape as before; brighter border if
    // any of the stacked IDs is currently seen.
    ctx.fillStyle = baseColor;
    ctx.fillRect(mx - 5, my - 5, 10, 10);
    ctx.strokeStyle = anySeen ? '#ffffff' : 'rgba(15,23,42,0.9)';
    ctx.lineWidth = anySeen ? 1.5 : 1;
    ctx.strokeRect(mx - 5.5, my - 5.5, 11, 11);

    // Vertical label stack — one row per ID. Upper-altitude marker sits
    // on the top row; lower marker on the bottom. This mirrors the
    // physical layout of the stacked wall pairs (high above low).
    const LINE_H = 13;
    const pillW  = Math.max(...b.entries.map(e => ctx.measureText(e.id).width)) + 10;
    const pillH  = b.entries.length * LINE_H + 4;

    // Side opposite the BACK/FRONT/LEFT/RIGHT text so the pill doesn't
    // collide with it. Offset includes pillH now since the pill is taller.
    const wall = (b.wall || '').toLowerCase();
    let labelX, labelY;
    if      (wall === 'front') { labelX = mx;                      labelY = my - pillH / 2 - 8; }
    else if (wall === 'back')  { labelX = mx;                      labelY = my + pillH / 2 + 8; }
    else if (wall === 'left')  { labelX = mx + 9 + pillW / 2;      labelY = my; }
    else if (wall === 'right') { labelX = mx - 9 - pillW / 2;      labelY = my; }
    else                        { labelX = mx;                      labelY = my - pillH / 2 - 8; }

    // Pill background + border (brighter border when anything seen)
    ctx.fillStyle = 'rgba(15,23,42,0.9)';
    ctx.fillRect(labelX - pillW / 2, labelY - pillH / 2, pillW, pillH);
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = anySeen ? 1.5 : 1;
    ctx.strokeRect(labelX - pillW / 2 + 0.5, labelY - pillH / 2 + 0.5, pillW - 1, pillH - 1);

    // Render each ID on its own row; seen → bold white, unseen → wall
    // colour. Thin separator line between rows makes the pair visually
    // obvious when both IDs are unseen (same colour otherwise).
    ctx.textAlign = 'center';
    b.entries.forEach((e, i) => {
      const rowY = labelY - pillH / 2 + (i + 0.5) * LINE_H + 2;
      ctx.fillStyle = e.seen ? '#ffffff' : baseColor;
      ctx.font      = e.seen ? 'bold 11px monospace' : '11px monospace';
      ctx.fillText(e.id, labelX, rowY);
      if (i < b.entries.length - 1) {
        ctx.strokeStyle = 'rgba(100,116,139,0.5)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(labelX - pillW / 2 + 2, labelY - pillH / 2 + (i + 1) * LINE_H + 2);
        ctx.lineTo(labelX + pillW / 2 - 2, labelY - pillH / 2 + (i + 1) * LINE_H + 2);
        ctx.stroke();
      }
    });
    ctx.font = 'bold 11px monospace';   // restore default for any later caller
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

  // ── Target boxes (from the capture-targets mission config) ──
  // Drawn on the floor as a labelled square with a diamond marker.
  // Colour indicates capture state when a mission is running:
  //   grey   = not yet visited
  //   yellow = currently claimed by a drone
  //   green  = captured
  const boxes = window._targetBoxes;
  if (Array.isArray(boxes) && boxes.length) {
    const claimed = (window._missionClaimedBoxes || {});
    const captured = new Set(window._missionCapturedBoxes || []);
    boxes.forEach((b, i) => {
      if (b == null || b.x == null || b.y == null) return;
      const [bx, by] = arenaToCanvas(Number(b.x), Number(b.y));
      const idx = (typeof b.idx === 'number') ? b.idx : i;
      const isCap = captured.has(idx);
      const isClaimed = Object.values(claimed).includes(idx);
      const team = (b.home_team || '').toLowerCase();
      // Default colour is the team the box BELONGS to (red/blue start colour
      // from the SDC26 rules). While being approached by us → yellow.
      // After capture → green (ours).
      const fill = isCap     ? 'rgba(34,197,94,0.55)'
                 : isClaimed ? 'rgba(250,204,21,0.55)'
                 : team === 'red'  ? 'rgba(239,68,68,0.55)'
                 : team === 'blue' ? 'rgba(59,130,246,0.55)'
                                   : 'rgba(148,163,184,0.45)';
      const stroke = isCap     ? '#16a34a'
                   : isClaimed ? '#eab308'
                   : team === 'red'  ? '#dc2626'
                   : team === 'blue' ? '#2563eb'
                                     : '#64748b';
      // Box body: 20 px diamond-ish square
      const s = 18;
      ctx.save();
      ctx.translate(bx, by);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = fill;
      ctx.fillRect(-s/2, -s/2, s, s);
      ctx.strokeStyle = stroke; ctx.lineWidth = 2;
      ctx.strokeRect(-s/2, -s/2, s, s);
      ctx.restore();
      // Label: box id + status
      const label = `#${b.id ?? idx+1}` +
                    (isCap ? ' ✓' : (isClaimed ? ' ⏱' : ''));
      ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(bx - tw/2 - 3, by + 14, tw + 6, 13);
      ctx.fillStyle = stroke;
      ctx.fillText(label, bx, by + 24);
    });
  }

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
        if (window._arena3d.syncTargetBoxes) window._arena3d.syncTargetBoxes();
        if (window._arena3d.updateDronePositionHUD) {
          window._arena3d.updateDronePositionHUD(d.observers);
        }
        // Aggregate which markers are currently visible / used as refs
        // across the whole fleet so the 3D highlight matches the 2D
        // halo. Prefer the position service's seen_markers field; fall
        // back to ref_markers-only when the observer isn't running.
        if (window._arena3d.updateVisibleMarkers) {
          const seenAll = new Set();
          const refAll  = new Set();
          for (const st of Object.values(d.observers || {})) {
            (st.seen_markers || []).forEach(m => seenAll.add(String(m)));
            (st.ref_markers  || []).forEach(m => {
              refAll.add(String(m));
              seenAll.add(String(m));   // ref implies seen
            });
          }
          window._arena3d.updateVisibleMarkers(
            Array.from(seenAll), Array.from(refAll));
        }
      }
    }
  } catch {}
}
setInterval(fleetPoll, 1000);   // was 500ms/2Hz → 1Hz; fleet poll fans out to all drones, heaviest endpoint
fleetPoll();

function updatePosUI(d) {
  const pos = d.pos;
  const vel = d.vel || [0, 0, 0];
  const lat = (d.latency_ms || 0) / 1000;
  const compPos = pos ? pos.map((v, i) => v + (vel[i] || 0) * lat) : null;

  document.getElementById('pos_x').textContent = pos ? pos[0].toFixed(2) : '\u2014';
  document.getElementById('pos_y').textContent = pos ? pos[1].toFixed(2) : '\u2014';
  document.getElementById('pos_z').textContent = pos ? pos[2].toFixed(2) : '\u2014';

  const dir = d.dir;
  const hdg = dir ? ((Math.atan2(dir[0], dir[1]) * 180 / Math.PI + 360) % 360).toFixed(1) : '\u2014';
  document.getElementById('pos_hdg').textContent = hdg;
  const spd = vel ? Math.sqrt((vel[0]||0)**2 + (vel[1]||0)**2).toFixed(2) : '\u2014';
  document.getElementById('pos_vel').textContent = spd + ' m/s';
  document.getElementById('pos_refs').textContent = d.ref_markers ? d.ref_markers.length : '\u2014';

  // ── Target Boxes table update ────────────────────────────────────
  // Resolves team/colour from the currently-loaded arena_cfg.target_teams
  // (ID-range list) + target_overrides (per-ID explicit). Targets carry
  // a 3 s TTL so a target briefly out of view doesn't flicker off; a
  // "stale" status flag is set when we haven't seen it in the latest
  // frame.
  (function updateTargetsPanel() {
    const tbody = document.getElementById('targets_tbody');
    const badge = document.getElementById('targets_badge');
    const table = document.getElementById('targets_table');
    const empty = document.getElementById('targets_empty');
    if (!tbody || !badge) return;
    // SDC26 convention: only IDs 31-36 (Blue) and 41-46 (Red) are real
    // target boxes. Any other ArUco ID — including rogue detections at
    // 30, 37-40, 47+, or reference-marker IDs that happened to land in
    // the tgt_indices pass — must be ignored so the operator doesn't
    // see (and the drone doesn't try to capture) a phantom target.
    const VALID_TARGET_IDS = new Set([31,32,33,34,35,36,41,42,43,44,45,46]);
    const allTargets = d.targets || {};
    const targets = {};
    for (const k of Object.keys(allTargets)) {
      if (VALID_TARGET_IDS.has(Number(k))) targets[k] = allTargets[k];
    }
    const ids = Object.keys(targets);
    // Resolve team+colour+box-number. SDC26 convention:
    //   - Marker ID = 10·(team_code) + box_number
    //   - First digit 3 → Blue team, first digit 4 → Red team
    //   - Box number = id % 10 (1..6)
    function teamFor(tid) {
      const n = Number(tid);
      const ac = window._arenaCfgCache || {};
      // Per-ID overrides always win
      for (const o of (ac.target_overrides || [])) {
        if (Number(o.id) === n) {
          return { team: o.team || 'Override', color: o.color || '#64748b',
                   box_num: (n % 10) || null };
        }
      }
      // Range mapping from arena config
      for (const r of (ac.target_teams || [])) {
        const rng = r.id_range || [0, 0];
        if (n >= Number(rng[0]) && n <= Number(rng[1])) {
          const box = n % 10;
          const label = r.team + (box >= 1 && box <= 9 ? ' Box ' + box : '');
          return { team: label, color: r.color || '#64748b', box_num: box };
        }
      }
      return { team: 'Unknown', color: '#64748b', box_num: null };
    }
    badge.textContent = ids.length + ' visible';
    badge.style.color = ids.length > 0 ? '#fbbf24' : '#64748b';
    if (ids.length === 0) {
      table.style.display = 'none';
      empty.style.display = '';
      return;
    }
    table.style.display = '';
    empty.style.display = 'none';
    // Sort by ID for stability
    ids.sort((a, b) => Number(a) - Number(b));
    const rows = ids.map(tid => {
      const t = targets[tid];
      const p = t.pos || [0, 0, 0];
      const age = t.age_s || 0;
      const fresh = !!t.fresh;
      const team = teamFor(tid);
      const statusHtml = fresh
        ? '<span style="color:#22c55e;">● live</span>'
        : '<span style="color:#94a3b8;">○ held ' + age.toFixed(1) + 's</span>';
      return (
        '<tr style="border-bottom:1px solid #1e293b;">' +
        '<td style="padding:4px 6px;font-family:monospace;color:#94a3b8;">' + tid + '</td>' +
        '<td style="padding:4px 6px;">' +
          '<span style="display:inline-block;width:12px;height:12px;border-radius:2px;' +
          'background:' + team.color + ';vertical-align:middle;margin-right:6px;"></span>' +
          '<span style="color:' + team.color + ';font-weight:600;">' + team.team + '</span>' +
        '</td>' +
        '<td style="padding:4px 6px;font-family:monospace;">' + p[0].toFixed(2) + '</td>' +
        '<td style="padding:4px 6px;font-family:monospace;">' + p[1].toFixed(2) + '</td>' +
        '<td style="padding:4px 6px;font-family:monospace;">' + p[2].toFixed(2) + '</td>' +
        '<td style="padding:4px 6px;font-family:monospace;color:#64748b;">' + age.toFixed(1) + 's</td>' +
        '<td style="padding:4px 6px;">' + statusHtml + '</td>' +
        '</tr>'
      );
    });
    tbody.innerHTML = rows.join('');
  })();

  // ── Safety distance readout ───────────────────────────────────────
  // How close is the drone to the nearest safety boundary? Green =
  // safe zone with margin; amber = within 30 cm of the boundary; red
  // = past the margin (i.e. inside the restricted band between the
  // dashed safety line and the arena wall).
  (function updateSafetyReadout() {
    const el = document.getElementById('pos_safety_readout');
    if (!el) return;
    const sa = window._arenaSafety || {};
    if (!sa.enabled) {
      el.textContent = 'guard OFF — operator owns boundaries';
      el.style.color = '#fbbf24';
      return;
    }
    if (sa.margin_m == null || !Array.isArray(pos) || pos.length < 2) {
      el.textContent = '—';
      el.style.color = '#64748b';
      return;
    }
    const m = sa.margin_m;
    const x = Number(pos[0]), y = Number(pos[1]);
    // Distance to nearest wall, then distance to the SAFE inner boundary.
    // Negative value = drone is inside the restricted band (past the margin).
    const dWall = Math.min(
      (arenaOX + arenaW) - x,      // right wall
      x - arenaOX,                  // left wall
      (arenaOY + arenaD) - y,      // back wall (y_max)
      y - arenaOY,                  // front wall (y_min)
    );
    const dSafe = dWall - m;        // metres from safe boundary
    const wallNames = {
      xr: (arenaOX + arenaW) - x,
      xl: x - arenaOX,
      yb: (arenaOY + arenaD) - y,
      yf: y - arenaOY,
    };
    let closest = 'right';
    let closestD = wallNames.xr;
    if (wallNames.xl < closestD) { closest = 'left';  closestD = wallNames.xl; }
    if (wallNames.yb < closestD) { closest = 'back';  closestD = wallNames.yb; }
    if (wallNames.yf < closestD) { closest = 'front'; closestD = wallNames.yf; }

    if (dSafe < 0) {
      // Inside restricted band — or worse, outside arena.
      el.textContent = '⚠ RESTRICTED — ' + Math.abs(dSafe).toFixed(2) +
                        ' m past ' + closest + ' margin  (wall ' +
                        closestD.toFixed(2) + ' m away)';
      el.style.color = '#ef4444';
    } else if (dSafe < 0.3) {
      el.textContent = '⚡ approaching ' + closest + ' — ' +
                        dSafe.toFixed(2) + ' m to margin  (' +
                        closestD.toFixed(2) + ' m to wall)';
      el.style.color = '#fbbf24';
    } else {
      el.textContent = '✓ safe — ' + dSafe.toFixed(2) +
                        ' m to ' + closest + ' margin  (' +
                        closestD.toFixed(2) + ' m to wall)';
      el.style.color = '#22c55e';
    }
  })();
  document.getElementById('pos_fps').textContent = d.fps != null ? d.fps : '\u2014';
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
    const ds = document.getElementById('pos_distance_scale');
    if (ds && c.distance_scale != null) ds.value = Number(c.distance_scale).toFixed(3);
    // Sync the Auto Positioning master toggle — also re-locks/unlocks
    // the manual tuning sliders accordingly.
    if (typeof window._applyPosAutoUI === 'function') {
      window._applyPosAutoUI(c.auto_positioning !== false);
    }
    // ── Populate precision (advanced) controls ──
    const setIf = (id, val, formatter) => {
      const el = document.getElementById(id);
      if (el && val != null) el.value = formatter ? formatter(val) : val;
    };
    setIf('pos_pose_hold',  c.pose_hold_sec);
    setIf('pos_min_refs',   c.min_ref_count);
    setIf('pos_min_ref_w',  c.min_ref_weight);
    setIf('pos_blend_min',  c.meas_blend_min);
    setIf('pos_blend_max',  c.meas_blend_max);
    setIf('pos_vel_blend',  c.vel_blend);
    setIf('pos_max_dt',     c.max_state_dt);
    setIf('pos_kf_q',       c.kalman_process_var, (v)=>Number(v).toPrecision(3));
    setIf('pos_kf_r',       c.kalman_meas_var,    (v)=>Number(v).toPrecision(3));
    setIf('pos_max_jump',   c.max_pose_jump_m);
    setIf('pos_target_size', c.target_marker_size_m);
    setIf('pos_zupt_speed',  c.zupt_speed_m_s);
    setIf('pos_zupt_hold',   c.zupt_hold_frames);
    // IMU LPF slider + its live-label
    if (c.imu_lowpass_hz != null) {
      const slider = document.getElementById('pos_imu_lpf');
      const label  = document.getElementById('pos_imu_lpf_val');
      if (slider) slider.value = Number(c.imu_lowpass_hz).toFixed(1);
      if (label)  label.textContent = (Number(c.imu_lowpass_hz) > 0
                                        ? Number(c.imu_lowpass_hz).toFixed(1) + ' Hz'
                                        : 'OFF');
    }
    const cs = document.getElementById('pos_calib_status');
    cs.textContent = d.has_calibration ? '\u2713 calibration loaded' : 'no calibration';
    cs.style.color = d.has_calibration ? '#22c55e' : '#94a3b8';
    if (c.enabled) startPosEvents();
  } catch {}
}

// ── Auto Positioning toggle ─────────────────────────────────────────
// Single master switch that forces the FC to use CLAUDE_AUTO_CONFIG
// instead of whatever the operator has tuned. While on, the manual
// slider/input controls below are disabled + visibly dimmed so the
// operator can't accidentally change a value that isn't being applied.
(function wireAutoPositioning(){
  const tgl = document.getElementById('pos_auto_toggle');
  const status = document.getElementById('pos_auto_status');
  if (!tgl) return;
  // Every ID of a control that belongs to the "manual tuning" surface.
  // When auto is ON, they're disabled. They stay visible for reference.
  const MANUAL_IDS = [
    'pos_profile','pos_fov','pos_imu_weight',
    'pos_kalman','pos_marker_size','pos_top_k','pos_outlier','pos_distance_scale',
    'pos_filters_apply','pos_filters_reset',
    'pos_pose_hold','pos_min_refs','pos_min_ref_w',
    'pos_blend_min','pos_blend_max','pos_vel_blend','pos_max_dt',
    'pos_kf_q','pos_kf_r','pos_imu_lpf',
    'pos_precision_apply','pos_precision_reset',
    'pos_preset_apply','pos_preset_name','pos_preset_save','pos_preset_delete','pos_preset_sel',
  ];
  function setManualEnabled(enabled) {
    for (const id of MANUAL_IDS) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.disabled = !enabled;
      // Also visually dim the surrounding <label> containers so it's
      // obvious the whole row is frozen, not just the input field.
      const lbl = el.closest('label');
      if (lbl) lbl.style.opacity = enabled ? '1' : '0.45';
      el.style.opacity = enabled ? '1' : '0.55';
    }
  }
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg; status.style.color = col || '#86efac';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }
  window._applyPosAutoUI = function(auto) {
    tgl.checked = !!auto;
    setManualEnabled(!auto);
    if (status) status.textContent = auto
      ? 'active — manual tuning locked'
      : 'off — manual tuning active';
    if (status) status.style.color = auto ? '#86efac' : '#fbbf24';
  };
  tgl.addEventListener('change', async () => {
    const auto = tgl.checked;
    window._applyPosAutoUI(auto);
    try {
      const r = await fetch('/proxy/position/config', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({auto_positioning: auto}),
      });
      const d = await r.json();
      if (d.ok) {
        flash(auto ? '\u2713 Claude preset applied' : '\u2713 manual mode', '#86efac');
        if (typeof loadPosConfig === 'function') setTimeout(loadPosConfig, 300);
      } else {
        flash('\u2717 ' + (d.error || 'apply failed'), '#ef4444');
      }
    } catch (e) {
      flash('\u2717 ' + e, '#ef4444');
    }
  });
  // Apply initial UI state (will be overwritten by loadPosConfig once the
  // FC replies with the persisted auto_positioning value).
  window._applyPosAutoUI(true);
})();

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
    const dsEl = document.getElementById('pos_distance_scale');
    const payload = {
      enable_kalman_filter: document.getElementById('pos_kalman').checked,
      marker_size_m: parseFloat(document.getElementById('pos_marker_size').value),
      top_k_markers: parseInt(document.getElementById('pos_top_k').value, 10),
      outlier_reject_m: parseFloat(document.getElementById('pos_outlier').value),
      distance_scale: dsEl ? parseFloat(dsEl.value) : 1.0,
    };
    try {
      const r = await fetch('/proxy/position/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.ok) flash('\u2713 applied', '#22c55e');
      else flash('error: ' + (d.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
  const resetBtn = document.getElementById('pos_filters_reset');
  if (resetBtn) resetBtn.onclick = () => {
    document.getElementById('pos_kalman').checked = true;
    document.getElementById('pos_marker_size').value = '0.50';
    document.getElementById('pos_top_k').value = '0';
    document.getElementById('pos_outlier').value = '2.5';
    const ds = document.getElementById('pos_distance_scale');
    if (ds) ds.value = '1.000';
    flash('defaults loaded — click Apply', '#94a3b8');
  };
})();

// ── Precision (advanced) controls — pose hold, Kalman variances,
// measurement/velocity blend, min-refs. Live-apply through the same
// endpoint; no restart required. Defaults mirror ctrl_position.py.
(function wirePrecisionControls(){
  const flash = (msg, col) => {
    const e = document.getElementById('pos_precision_status');
    if (!e) return;
    e.textContent = msg; e.style.color = col || '#64748b';
    setTimeout(() => { if (e.textContent === msg) e.textContent = ''; }, 2500);
  };
  const applyBtn = document.getElementById('pos_precision_apply');
  if (applyBtn) applyBtn.onclick = async () => {
    const num = (id) => parseFloat(document.getElementById(id).value);
    const int = (id) => parseInt(document.getElementById(id).value, 10);
    const payload = {
      pose_hold_sec:       num('pos_pose_hold'),
      min_ref_count:       int('pos_min_refs'),
      min_ref_weight:      num('pos_min_ref_w'),
      meas_blend_min:      num('pos_blend_min'),
      meas_blend_max:      num('pos_blend_max'),
      vel_blend:           num('pos_vel_blend'),
      max_state_dt:        num('pos_max_dt'),
      kalman_process_var:  num('pos_kf_q'),
      kalman_meas_var:     num('pos_kf_r'),
      imu_lowpass_hz:      num('pos_imu_lpf'),
      max_pose_jump_m:     num('pos_max_jump'),
      target_marker_size_m: num('pos_target_size'),
      zupt_speed_m_s:      num('pos_zupt_speed'),
      zupt_hold_frames:    int('pos_zupt_hold'),
    };
    try {
      const r = await fetch('/proxy/position/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.ok) flash('\u2713 applied', '#22c55e');
      else flash('error: ' + (d.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
  const resetBtn = document.getElementById('pos_precision_reset');
  if (resetBtn) resetBtn.onclick = () => {
    document.getElementById('pos_pose_hold').value = '0.8';
    document.getElementById('pos_min_refs').value  = '1';
    document.getElementById('pos_min_ref_w').value = '0';
    document.getElementById('pos_blend_min').value = '0.35';
    document.getElementById('pos_blend_max').value = '0.85';
    document.getElementById('pos_vel_blend').value = '0.25';
    document.getElementById('pos_max_dt').value    = '1.0';
    document.getElementById('pos_kf_q').value      = '1e-3';
    document.getElementById('pos_kf_r').value      = '0.1';
    const lpf = document.getElementById('pos_imu_lpf');
    if (lpf) {
      lpf.value = '5';
      const lbl = document.getElementById('pos_imu_lpf_val');
      if (lbl) lbl.textContent = '5.0 Hz';
    }
    const mj = document.getElementById('pos_max_jump');
    if (mj) mj.value = '0';
    const ts = document.getElementById('pos_target_size');
    if (ts) ts.value = '0.19';
    const zs = document.getElementById('pos_zupt_speed');
    if (zs) zs.value = '0.05';
    const zh = document.getElementById('pos_zupt_hold');
    if (zh) zh.value = '3';
    flash('defaults loaded — click Apply', '#94a3b8');
  };
})();

// ── Position-tracker preset management ─────────────────────────────
// Mirrors the mission preset system. Apply fans out to every drone
// via /proxy/position/config; Save gathers current UI values and
// POSTs to /proxy/position/presets.
(function wirePositionPresets(){
  const sel   = document.getElementById('pos_preset_sel');
  const aBtn  = document.getElementById('pos_preset_apply');
  const sBtn  = document.getElementById('pos_preset_save');
  const dBtn  = document.getElementById('pos_preset_delete');
  const nameI = document.getElementById('pos_preset_name');
  const status = document.getElementById('pos_preset_status');
  if (!sel) return;
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 3000);
  }
  function readCurrentParams() {
    // Gather everything live-tunable on the Position Tracker panel.
    // The server accepts unknown keys gracefully; we send a superset.
    const v = (id, parser) => {
      const el = document.getElementById(id);
      if (!el) return undefined;
      const raw = el.value;
      if (parser === 'float') { const f = parseFloat(raw); return isFinite(f) ? f : undefined; }
      if (parser === 'int')   { const i = parseInt(raw, 10); return isFinite(i) ? i : undefined; }
      if (parser === 'bool')  return !!el.checked;
      return raw;
    };
    return {
      detect_profile:        v('pos_profile'),
      fov_deg:               v('pos_fov', 'float'),
      imu_weight:            (v('pos_imu_weight', 'float') || 0) / 100.0,
      latency_ms:            v('pos_latency', 'float'),
      enable_kalman_filter:  v('pos_kalman', 'bool'),
      marker_size_m:         v('pos_marker_size', 'float'),
      top_k_markers:         v('pos_top_k', 'int'),
      outlier_reject_m:      v('pos_outlier', 'float'),
      distance_scale:        v('pos_distance_scale', 'float'),
      pose_hold_sec:         v('pos_pose_hold', 'float'),
      min_ref_count:         v('pos_min_refs', 'int'),
      min_ref_weight:        v('pos_min_ref_w', 'float'),
      meas_blend_min:        v('pos_blend_min', 'float'),
      meas_blend_max:        v('pos_blend_max', 'float'),
      vel_blend:             v('pos_vel_blend', 'float'),
      max_state_dt:          v('pos_max_dt', 'float'),
      kalman_process_var:    v('pos_kf_q', 'float'),
      kalman_meas_var:       v('pos_kf_r', 'float'),
      imu_lowpass_hz:        v('pos_imu_lpf', 'float'),
      max_pose_jump_m:       v('pos_max_jump', 'float'),
      target_marker_size_m:  v('pos_target_size', 'float'),
      zupt_speed_m_s:        v('pos_zupt_speed', 'float'),
      zupt_hold_frames:      v('pos_zupt_hold', 'int'),
    };
  }
  async function refresh() {
    try {
      const j = await (await fetch('/proxy/position/presets')).json();
      const presets = j.presets || {};
      const names = Object.keys(presets).sort();
      sel.innerHTML = '';
      if (!names.length) {
        const opt = document.createElement('option');
        opt.value = ''; opt.textContent = '(no presets)';
        sel.appendChild(opt);
      } else {
        names.forEach(n => {
          const opt = document.createElement('option');
          opt.value = n; opt.textContent = n;
          sel.appendChild(opt);
        });
        if (names.includes('balanced')) sel.value = 'balanced';
      }
    } catch {}
  }
  if (aBtn) aBtn.onclick = async () => {
    const name = sel.value;
    if (!name) { flash('no preset selected', '#ef4444'); return; }
    try {
      const r = await fetch('/proxy/position/presets/apply', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name}),
      });
      const j = await r.json();
      if (j.ok) {
        flash('✓ "' + name + '" → ' + (j.applied_to || 0) + '/' + (j.total || 0) + ' drones', '#22c55e');
        if (typeof loadPosConfig === 'function') loadPosConfig();   // pull fresh UI values
      } else flash('✗ ' + (j.error || 'apply failed'), '#ef4444');
    } catch (e) { flash('✗ ' + e, '#ef4444'); }
  };
  if (sBtn) sBtn.onclick = async () => {
    const name = (nameI.value.trim()) || sel.value;
    if (!name) { flash('enter a preset name', '#ef4444'); return; }
    const params = readCurrentParams();
    try {
      const r = await fetch('/proxy/position/presets', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, params}),
      });
      const j = await r.json();
      if (j.ok) { flash('✓ saved "' + name + '"', '#22c55e'); await refresh(); }
      else flash('✗ ' + (j.error || 'save failed'), '#ef4444');
    } catch (e) { flash('✗ ' + e, '#ef4444'); }
  };
  if (dBtn) dBtn.onclick = async () => {
    const name = sel.value;
    if (!name) return;
    if (!confirm('Delete position preset "' + name + '"?')) return;
    try {
      const r = await fetch('/proxy/position/presets?name=' + encodeURIComponent(name),
                             {method:'DELETE'});
      const j = await r.json();
      if (j.ok) { flash('✓ deleted', '#22c55e'); await refresh(); }
      else flash('✗ ' + (j.error || 'delete failed'), '#ef4444');
    } catch (e) { flash('✗ ' + e, '#ef4444'); }
  };
  refresh();
})();

// ── Claude Automatic Calibration ───────────────────────────────────
// Orchestrates the autonomous calibration flight:
//   1. Pre-flight warning (operator must have drone centred in arena)
//   2. POST /proxy/calibration/start to kick off the ~90 s sequence
//   3. Poll /proxy/calibration/status at 500 ms to drive the progress bar
//   4. On completion, show the download hint pointing to Flight Logs
//   5. Provide an Import Preset box so the operator can paste Claude's
//      tuned JSON and save it as a named preset via /proxy/position/presets
// ── Scan & Capture Targets mission ─────────────────────────────
// Kicks off /proxy/missions/scan_and_capture/start and polls
// /proxy/missions/scan_and_capture/status every 500 ms to drive the
// progress readout in the Target Boxes panel.
(function wireScanAndCapture(){
  const startBtn  = document.getElementById('scan_cap_start_btn');
  const abortBtn  = document.getElementById('scan_cap_abort_btn');
  const statusTxt = document.getElementById('scan_cap_status_txt');
  const progWrap  = document.getElementById('scan_cap_progress');
  const progBar   = document.getElementById('scan_cap_progress_bar');
  const phaseEl   = document.getElementById('scan_cap_phase');
  const stepEl    = document.getElementById('scan_cap_step');
  const ndetEl    = document.getElementById('scan_cap_ndet');
  const elapsedEl = document.getElementById('scan_cap_elapsed');
  const hoverI    = document.getElementById('scan_cap_hover');
  const aboveI    = document.getElementById('scan_cap_above');
  const stackI    = document.getElementById('scan_cap_stack');
  const sepI      = document.getElementById('scan_cap_sep');
  const dronesDiv = document.getElementById('scan_cap_drones');
  if (!startBtn) return;

  let pollTimer = null;
  let active = false;

  // Populate drone checkboxes so operator can select one-or-many.
  // Defaults to the currently-active drone. Re-polled every 10 s so a
  // drone coming online mid-session becomes selectable.
  async function loadDroneChecklist() {
    try {
      const r = await fetch('/proxy/drones');
      const d = await r.json();
      const drones = d.drones || {};
      const prev = {};
      dronesDiv.querySelectorAll('input[type=checkbox]').forEach(cb => {
        prev[cb.dataset.id] = cb.checked;
      });
      dronesDiv.innerHTML = '';
      const activeId = (d.active && String(d.active)) || '';
      const ids = Object.keys(drones).sort();
      ids.forEach(id => {
        const info = drones[id] || {};
        const wrap = document.createElement('label');
        wrap.style.cssText = 'display:flex;gap:4px;align-items:center;font-size:12px;color:#e2e8f0;cursor:pointer;';
        // If we've seen this drone before, preserve previous selection.
        // Otherwise default: only the active drone is checked so operators
        // don't accidentally fly every drone at once.
        const checked = (id in prev) ? prev[id] : (id === activeId);
        wrap.innerHTML = '<input type="checkbox" data-id="' + id + '"'
          + (checked ? ' checked' : '') + ' /> '
          + (info.name || ('Drone ' + id))
          + ' <span style="color:#64748b;">#' + id + '</span>';
        dronesDiv.appendChild(wrap);
      });
    } catch (e) {}
  }
  function getSelectedDroneIds() {
    return Array.from(dronesDiv.querySelectorAll('input[type=checkbox]'))
      .filter(cb => cb.checked).map(cb => cb.dataset.id);
  }
  loadDroneChecklist();
  setInterval(loadDroneChecklist, 10000);

  function setStatus(msg, col) {
    if (!statusTxt) return;
    statusTxt.textContent = msg;
    statusTxt.style.color = col || '#94a3b8';
  }
  function setActiveUI(on) {
    active = on;
    startBtn.style.display = on ? 'none' : '';
    abortBtn.style.display = on ? '' : 'none';
    progWrap.style.display = on ? '' : 'none';
    if (!on) progBar.style.width = '0%';
  }
  // Phase → percentage (rough progress bar). "scanning" 10-70%,
  // "capture" 70-100% (the capture mission runs after the handoff).
  function phaseProgress(phase, elapsed, n_detected) {
    if (phase === 'takeoff' || phase === 'starting') return 5;
    if (phase === 'scanning') {
      // 10s scan ≈ 70% target → scale with elapsed (after takeoff ~5s)
      return Math.min(70, 10 + elapsed * 6);
    }
    if (phase === 'capture' || phase === 'done')     return 90;
    if (phase === 'error' || phase === 'aborted')    return 100;
    return 0;
  }
  // Resolve a target ID to a human-readable "Team Box N (ID nn)" label
  // using the SDC26 convention: first digit = team, last digit = box.
  function targetLabel(tid) {
    const n = Number(tid);
    if (!Number.isFinite(n)) return String(tid);
    const team = (n >= 31 && n <= 36) ? 'Blue'
               : (n >= 41 && n <= 46) ? 'Red'
               : '?';
    const col  = team === 'Blue' ? '#3b82f6'
               : team === 'Red'  ? '#ef4444'
               : '#94a3b8';
    const box = n % 10;
    return '<span style="color:' + col + ';font-weight:600;">'
         + team + ' Box ' + box + '</span>'
         + ' <span style="color:#64748b;">(ID ' + n + ')</span>';
  }

  function phaseBadge(phase) {
    const p = String(phase || '').toUpperCase();
    const bg = p === 'PURSUE' ? '#713f12' :
               p === 'HOVER'  ? '#065f46' :
               p === 'SEARCH' ? '#1e293b' : '#1e293b';
    const fg = p === 'PURSUE' ? '#fde68a' :
               p === 'HOVER'  ? '#a7f3d0' :
               p === 'SEARCH' ? '#94a3b8' : '#94a3b8';
    return '<span style="padding:1px 7px;background:' + bg
         + ';color:' + fg + ';border-radius:10px;font-size:11px;font-weight:700;letter-spacing:0.3px;">'
         + (p || 'IDLE') + '</span>';
  }

  async function renderPursuitStatus() {
    // Pull the currently-running mission's status so we can surface
    // per-drone pursuit state (phase + target ID + note).
    const box = document.getElementById('scan_cap_drones_state');
    if (!box) return;
    try {
      const r = await fetch('/proxy/missions/status');
      const m = await r.json();
      if (!m || !m.has_mission || m.kind !== 'direct_target_pursuit') {
        box.style.display = 'none';
        return;
      }
      const drones = m.drones || {};
      const ids = Object.keys(drones).sort();
      if (!ids.length) { box.style.display = 'none'; return; }
      box.style.display = '';
      const rows = ids.map(did => {
        const st = drones[did] || {};
        const phase = st.phase || 'IDLE';
        const tid = st.target_id;
        const note = st.note || '';
        const phaseReason = st.phase_reason || '';
        const tgt = (tid != null) ? targetLabel(tid) : '<span style="color:#64748b;">—</span>';
        // Render the "camera currently sees" chips — one per SDC26-valid
        // target within TARGET_MAX_AGE_S, coloured by team, with the
        // detection age and a "taken" / "claimed" tag when another
        // drone already owns it.
        const inView = Array.isArray(st.targets_in_view) ? st.targets_in_view : [];
        const chips = inView.map(c => {
          const n = Number(c.id);
          const team = (n >= 31 && n <= 36) ? 'Blue' : (n >= 41 && n <= 46) ? 'Red' : '?';
          const col  = team === 'Blue' ? '#3b82f6' : team === 'Red' ? '#ef4444' : '#94a3b8';
          const box = n % 10;
          const isMe = Number(tid) === n;
          const tagLabel = c.taken
            ? (isMe ? ' • mine' : ' • taken')
            : (c.fresh ? '' : ' • ' + Number(c.age_s).toFixed(1) + 's');
          const border = isMe ? '2px solid ' + col : '1px solid ' + col + '55';
          return '<span style="display:inline-block;padding:1px 7px;border:' + border
               + ';color:' + col + ';border-radius:10px;font-size:11px;font-weight:600;margin-right:4px;background:rgba(0,0,0,0.25);">'
               + team + ' ' + box + tagLabel + '</span>';
        }).join('');
        const viewLine = inView.length
          ? '<div style="margin-top:2px;padding-left:56px;font-size:11px;color:#94a3b8;">'
            + '<span style="color:#64748b;">sees:</span> ' + chips
            + '</div>'
          : '';
        // If the drone is stuck in SEARCH, phase_reason explains WHY —
        // no valid SDC26 in view, or all targets claimed by others, etc.
        const reasonLine = (phase === 'SEARCH' && phaseReason)
          ? '<div style="margin-top:2px;padding-left:56px;font-size:11px;color:#fbbf24;">'
            + '↳ ' + phaseReason + '</div>'
          : '';
        return (
          '<div style="padding:3px 0;">' +
          '<div style="display:flex;gap:10px;align-items:center;">' +
          '<span style="color:#e2e8f0;font-weight:600;min-width:56px;">Drone ' + did + '</span>' +
          phaseBadge(phase) +
          '<span style="margin-left:6px;">' + tgt + '</span>' +
          '<span style="color:#94a3b8;margin-left:auto;font-size:11px;">' + note + '</span>' +
          '</div>' +
          viewLine +
          reasonLine +
          '</div>'
        );
      });
      // Header explains the "what's being approached" contract so the
      // operator sees it straight on the panel, not just in docs.
      box.innerHTML =
        '<div style="color:#64748b;margin-bottom:4px;font-size:11px;">' +
        'Only valid SDC26 target IDs are approached: <b style="color:#3b82f6;">31-36 Blue</b>, '
        + '<b style="color:#ef4444;">41-46 Red</b>. Wall markers and other ArUco IDs are ignored.' +
        '</div>' + rows.join('');
    } catch (e) {
      box.style.display = 'none';
    }
  }

  async function refreshStatus() {
    try {
      const r = await fetch('/proxy/missions/scan_and_capture/status');
      const d = await r.json();
      if (!d.ok) { setStatus('status error', '#ef4444'); return; }
      phaseEl.textContent = d.phase || '—';
      stepEl.textContent  = d.step_name || '—';
      ndetEl.textContent  = d.n_detected || 0;
      elapsedEl.textContent = (d.elapsed_s || 0).toFixed(1);
      progBar.style.width = phaseProgress(d.phase, d.elapsed_s || 0, d.n_detected || 0) + '%';
      // Always refresh the per-drone pursuit readout — it stays live as
      // long as the underlying CaptureAllTargetsMission /
      // DirectTargetPursuitMission is running, even after the outer
      // scan-and-capture thread has finished its handoff.
      renderPursuitStatus();
      if (d.active) {
        setActiveUI(true);
        setStatus('in progress — ' + (d.phase || ''), '#fbbf24');
      } else {
        if (active) {
          setActiveUI(false);
          if (d.result === 'ok') {
            setStatus('✓ scan done — ' + (d.n_detected || 0) +
                      ' target(s), capture mission launched', '#22c55e');
          } else if (d.result === 'aborted') {
            setStatus('aborted', '#f59e0b');
          } else if (d.result === 'error') {
            setStatus('✗ ' + (d.last_error || 'unknown error'), '#ef4444');
          }
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }
      }
    } catch (e) {
      setStatus('network error: ' + e, '#ef4444');
    }
  }

  // Reflect the global pause state on the Start button itself — the
  // #1 cause of "nothing happens" was the server rejecting starts with
  // a tiny "fleet paused" status message that the operator missed.
  // The button now shows its own state clearly.
  function refreshPausedState() {
    const paused = !!window._globalPaused;
    if (paused) {
      startBtn.style.background = '#450a0a';
      startBtn.style.borderColor = '#b91c1c';
      startBtn.style.color = '#fecaca';
      startBtn.textContent = '⏸ Fleet paused — press 9 or click to resume';
      startBtn.title = 'The fleet is paused (hotkey 9). Clicking will offer '
        + 'to resume autonomy and then start the mission.';
    } else {
      startBtn.style.background = '#713f12';
      startBtn.style.borderColor = '#fbbf24';
      startBtn.style.color = '#fef3c7';
      startBtn.textContent = '▶ Start Scan & Capture';
      startBtn.title = 'Drone takes off (if needed), rotates 6×60° scanning '
        + 'for target markers, then visits each box in nearest-neighbour order.';
    }
  }
  setInterval(refreshPausedState, 600);
  refreshPausedState();

  async function _resumeFleetIfPaused() {
    if (!window._globalPaused) return true;
    try {
      const r = await fetch('/proxy/resume_all', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({source: 'scan_and_capture_start'}),
      });
      const j = await r.json();
      if (j && j.ok) {
        window._globalPaused = false;
        if (typeof applyPauseUI === 'function') applyPauseUI(false, j);
        refreshPausedState();
        return true;
      }
      alert('Could not resume fleet — check the main PAUSED banner at the top of the page and click CONTINUE MISSION, then try again.');
      return false;
    } catch (e) {
      alert('Resume failed: ' + e);
      return false;
    }
  }

  startBtn.addEventListener('click', async () => {
    // Paused? Offer to resume first — don't silently fail with a status
    // line that's easy to miss.
    if (window._globalPaused) {
      const go = confirm(
        '⏸  Fleet is currently PAUSED (hotkey 9 / Pause All).\n\n' +
        'Resume the fleet AND start Scan & Capture now?\n\n' +
        'OK = resume and start\n' +
        'Cancel = leave paused (press 9 or CONTINUE MISSION to resume manually)'
      );
      if (!go) {
        setStatus('✗ fleet paused — cancelled', '#f59e0b');
        return;
      }
      const resumed = await _resumeFleetIfPaused();
      if (!resumed) return;
    }
    const selected = getSelectedDroneIds();
    if (!selected.length) {
      alert('✗ Select at least one drone (tick a Drones checkbox).');
      setStatus('✗ select at least one drone', '#ef4444');
      return;
    }
    const stackM = Math.max(0, parseFloat(stackI.value || '1.0'));
    const sepM   = Math.max(0, parseFloat(sepI.value || '1.2'));
    const multi = selected.length > 1;
    const altLines = multi
      ? ('\n• Each drone flies at a stacked altitude (drone N = baseline + '
         + stackM.toFixed(1) + ' m × N) so they cannot collide vertically.\n'
         + '• Minimum 3-D separation enforced at ' + sepM.toFixed(1) + ' m — if two drones\n'
         + '  get closer than that, the one about to move pauses in place.\n'
         + '• No two drones will approach the same target box (claim-based).')
      : '';
    if (!confirm(
      'Target Capture (reactive pursuit):\n\n' +
      '• ' + selected.length + ' drone' + (multi?'s':'') + ' selected'
        + (multi?' ('+selected.join(', ')+').':'.') + '\n' +
      '• Each drone takes off, then slowly rotates to search for valid\n' +
      '  SDC26 targets (IDs 31-36 Blue, 41-46 Red — others ignored).\n' +
      '• As soon as a target appears in the camera, the drone flies\n' +
      '  directly toward it (camera stays pointed at the target), then\n' +
      '  hovers ' + (hoverI.value||'3') + ' s at ' + (aboveI.value||'1.5') +
      ' m ABOVE the marker (waypoint z = marker_z + ' + (aboveI.value||'1.5') + ').\n' +
      '• After a capture the drone immediately looks for the next target —\n' +
      '  no pre-planned waypoint list.' +
      altLines + '\n' +
      '• Arena boundary guard + ceiling stay active throughout.\n\n' +
      'Start now?'
    )) return;
    setStatus('starting...', '#fbbf24');
    try {
      const body = {
        drone_ids:      selected,
        hover_seconds:  parseFloat(hoverI.value || '3.0'),
        hover_above_m:  parseFloat(aboveI.value || '1.5'),
        altitude_stack_m: stackM,
        min_separation_m: sepM,
      };
      const r = await fetch('/proxy/missions/scan_and_capture/start', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!d.ok) {
        const msg = (d.error || 'start failed') + (d.hint ? ' — ' + d.hint : '');
        // Loud: an alert() guarantees the operator sees the failure reason
        // rather than missing a 12-pixel status line in the corner.
        alert('✗ Could not start Scan & Capture:\n\n' + msg);
        setStatus('✗ ' + msg, '#ef4444');
        return;
      }
      setActiveUI(true);
      setStatus('in progress', '#fbbf24');
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshStatus, 500);
      refreshStatus();
    } catch (e) {
      alert('✗ Scan & Capture request failed:\n\n' + e);
      setStatus('✗ ' + e, '#ef4444');
    }
  });

  abortBtn.addEventListener('click', async () => {
    if (!confirm('Abort scan & capture?\n\nThe drone will stop rotating/visiting; it stays airborne — land it manually.')) return;
    try {
      const r = await fetch('/proxy/missions/scan_and_capture/abort', {method:'POST'});
      const d = await r.json();
      if (d.ok) setStatus('abort requested...', '#f59e0b');
      else setStatus('✗ ' + (d.error || 'abort failed'), '#ef4444');
    } catch (e) {
      setStatus('✗ ' + e, '#ef4444');
    }
  });

  // One status poll at page load in case a mission is already running
  refreshStatus();
  // Keep the per-drone pursuit state live even after the outer
  // scan_and_capture thread finishes its handoff — the
  // DirectTargetPursuitMission continues running and this is the
  // operator's window into what each drone is actually doing.
  setInterval(renderPursuitStatus, 700);
})();

(function wireCalibrationFlight(){
  const startBtn  = document.getElementById('calib_start_btn');
  const abortBtn  = document.getElementById('calib_abort_btn');
  const statusTxt = document.getElementById('calib_status_txt');
  const progWrap  = document.getElementById('calib_progress_wrap');
  const progBar   = document.getElementById('calib_progress_bar');
  const stepNum   = document.getElementById('calib_step_num');
  const stepTotal = document.getElementById('calib_step_total');
  const stepName  = document.getElementById('calib_step_name');
  const elapsed   = document.getElementById('calib_elapsed');
  const dlHint    = document.getElementById('calib_download_hint');
  const importBtn = document.getElementById('calib_import_btn');
  const presetJsonEl = document.getElementById('calib_preset_json');
  const presetNameEl = document.getElementById('calib_preset_name');
  const presetStatus = document.getElementById('calib_preset_status');
  if (!startBtn) return;

  let pollTimer = null;
  let active = false;

  function setStatus(msg, col) {
    if (!statusTxt) return;
    statusTxt.textContent = msg;
    statusTxt.style.color = col || '#94a3b8';
  }
  function setActiveUI(on) {
    active = on;
    startBtn.style.display = on ? 'none' : '';
    abortBtn.style.display = on ? '' : 'none';
    progWrap.style.display = on ? '' : 'none';
    if (!on) {
      progBar.style.width = '0%';
    }
  }
  async function refreshStatus() {
    try {
      const r = await fetch('/proxy/calibration/status');
      const d = await r.json();
      if (!d.ok) { setStatus('status error', '#ef4444'); return; }
      const total = d.total_steps || 1;
      const cur = d.current_step || 0;
      const pct = Math.round((cur / total) * 100);
      progBar.style.width = pct + '%';
      stepNum.textContent = cur;
      stepTotal.textContent = total;
      stepName.textContent = d.step_name || '—';
      elapsed.textContent = (d.elapsed_s || 0).toFixed(1);
      if (d.active) {
        setActiveUI(true);
        setStatus('in progress — step ' + cur + '/' + total, '#fbbf24');
      } else {
        if (active) {
          // Just finished
          setActiveUI(false);
          if (d.result === 'ok') {
            setStatus('✓ completed (' + (d.elapsed_s||0).toFixed(1) + 's)', '#22c55e');
            dlHint.style.display = '';
            // Refresh flight-logs list so the new calibration flight appears
            const fr = document.getElementById('flight_logs_refresh');
            if (fr) setTimeout(() => fr.click(), 800);
          } else if (d.result === 'aborted') {
            setStatus('aborted', '#f59e0b');
            dlHint.style.display = '';
          } else if (d.result === 'error') {
            setStatus('✗ error: ' + (d.last_error || 'unknown'), '#ef4444');
            dlHint.style.display = '';
          }
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }
      }
    } catch (e) {
      setStatus('network error: ' + e, '#ef4444');
    }
  }

  startBtn.addEventListener('click', async () => {
    if (!confirm(
      'Calibration Flight:\n\n' +
      '• The drone will take off, fly a ~90 s scan pattern, and land.\n' +
      '• It stays within ±1.5 m of its current position — make sure\n' +
      '  the drone is placed in the arena CENTRE before continuing.\n' +
      '• Arena boundary guard and ceiling remain active.\n\n' +
      'Start now?'
    )) return;
    setStatus('starting...', '#fbbf24');
    try {
      const r = await fetch('/proxy/calibration/start', {method:'POST'});
      const d = await r.json();
      if (!d.ok) {
        setStatus('✗ ' + (d.error || 'start failed'), '#ef4444');
        return;
      }
      stepTotal.textContent = d.total_steps || '?';
      setActiveUI(true);
      dlHint.style.display = 'none';
      setStatus('in progress', '#fbbf24');
      // Begin polling
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshStatus, 500);
      refreshStatus();
    } catch (e) {
      setStatus('✗ ' + e, '#ef4444');
    }
  });

  abortBtn.addEventListener('click', async () => {
    if (!confirm('Abort calibration? The drone will land at the next safe point.')) return;
    try {
      const r = await fetch('/proxy/calibration/abort', {method:'POST'});
      const d = await r.json();
      if (d.ok) setStatus('abort requested...', '#f59e0b');
      else setStatus('✗ ' + (d.error || 'abort failed'), '#ef4444');
    } catch (e) {
      setStatus('✗ ' + e, '#ef4444');
    }
  });

  // Toggle the paste-JSON textarea when operator clicks Import.
  importBtn.addEventListener('click', async () => {
    if (presetJsonEl.style.display === 'none') {
      presetJsonEl.style.display = '';
      presetJsonEl.focus();
      presetStatus.textContent = 'paste JSON above, then click Import again to save';
      presetStatus.style.color = '#fbbf24';
      return;
    }
    // Actually import
    const raw = presetJsonEl.value.trim();
    const name = (presetNameEl.value || '').trim();
    if (!name) {
      presetStatus.textContent = '✗ enter a preset name first';
      presetStatus.style.color = '#ef4444';
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      presetStatus.textContent = '✗ invalid JSON: ' + e.message;
      presetStatus.style.color = '#ef4444';
      return;
    }
    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
      presetStatus.textContent = '✗ JSON must be an object';
      presetStatus.style.color = '#ef4444';
      return;
    }
    try {
      const r = await fetch('/proxy/position/presets', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name, params: parsed}),
      });
      const d = await r.json();
      if (d.ok) {
        presetStatus.textContent = '✓ saved "' + name + '" — apply from Presets dropdown above';
        presetStatus.style.color = '#22c55e';
        presetJsonEl.style.display = 'none';
        presetJsonEl.value = '';
        // Refresh presets dropdown
        const presetSel = document.getElementById('pos_preset_sel');
        if (presetSel) {
          fetch('/proxy/position/presets').then(r=>r.json()).then(j=>{
            if (!j.presets) return;
            const names = Object.keys(j.presets).sort();
            presetSel.innerHTML = '';
            names.forEach(n => {
              const opt = document.createElement('option');
              opt.value = n; opt.textContent = n;
              presetSel.appendChild(opt);
            });
            presetSel.value = name;
          });
        }
      } else {
        presetStatus.textContent = '✗ save failed: ' + (d.error || 'unknown');
        presetStatus.style.color = '#ef4444';
      }
    } catch (e) {
      presetStatus.textContent = '✗ ' + e;
      presetStatus.style.color = '#ef4444';
    }
  });

  // One initial status poll — useful if operator loads page mid-flight
  refreshStatus();
})();

// ── IMU LPF slider live-label + debounced apply ─────────────────────
// Updates the 'X.X Hz' / 'OFF' label on every frame of the drag, and
// POSTs the value once the user stops moving it (~200 ms debounce).
(function wireImuLpfSlider(){
  const slider = document.getElementById('pos_imu_lpf');
  const label  = document.getElementById('pos_imu_lpf_val');
  if (!slider) return;
  let _t = null;
  function refreshLabel() {
    if (!label) return;
    const v = parseFloat(slider.value);
    label.textContent = (v > 0 ? v.toFixed(1) + ' Hz' : 'OFF');
  }
  slider.addEventListener('input', () => {
    refreshLabel();
    if (_t) clearTimeout(_t);
    _t = setTimeout(async () => {
      try {
        await fetch('/proxy/position/config', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({imu_lowpass_hz: parseFloat(slider.value)}),
        });
      } catch {}
    }, 200);
  });
  refreshLabel();
})();

// ── Parameter-info popup — click any ⓘ icon for an explanation ──
// One shared map drives every tuning knob. The modal picks up title,
// key, body (explanation), and an optional range/units hint.
window.PARAM_INFO = {
  // ===== Observer PD (visual servo — used by all missions) =====
  hover_distance_m: {title:'Hover distance', units:'metres', body:
    "Target stand-off distance from the marker during hover. The mission "+
    "flies forward until the marker is this far away, then holds.\n\n"+
    "Smaller = closer / more camera detail, but less safety margin against "+
    "the net. 2 m is the rules-safe default for SDC26.\n"+
    "Feeds into: approach phase (distance error) and hover phase (dead-band)."},
  fb_max:           {title:'Approach speed (forward)', units:'% RC', body:
    "Upper clamp for forward throttle when approaching a marker. Scales "+
    "the P gain output — higher = faster approach but larger overshoot.\n\n"+
    "Pair with dist_p: the effective command is min(fb_max, dist_p · err_dist)."},
  fb_back_max:      {title:'Retreat speed (backward)', units:'% RC', body:
    "Upper clamp for backward throttle when the drone is too close to the "+
    "target. Set lower than fb_max — retreats tend to happen near the net "+
    "and should be cautious."},
  dist_p:           {title:'Approach aggressiveness', units:'P · dist error (m)', body:
    "Proportional gain turning distance-error (m) into forward RC%. Higher "+
    "values react more sharply to distance error.\n\n"+
    "Start low (10-20) and raise until the drone reaches the hover distance "+
    "without overshoot. Tied to fb_max and the IMU D term (d_fb)."},
  ema_alpha:        {title:'EMA smoothing (α)', units:'0..1', body:
    "First-order low-pass on camera-derived errors. 1.0 = no smoothing "+
    "(fastest response, jitteriest), 0.05 = heavy smoothing (laggy).\n\n"+
    "Typical 0.25-0.5. Lower when the video is noisy or markers are small."},
  deadband_x:       {title:'Yaw/lateral dead-band (err_x)', units:'normalised', body:
    "Below this threshold, yaw + sideways commands are zero. Stops the "+
    "drone hunting around an already-centred marker.\n\n"+
    "Typical 0.03-0.08. Increase if the drone wiggles while hovered."},
  deadband_y:       {title:'Altitude dead-band (err_y)', units:'normalised', body:
    "Below this threshold vertical command is zero. Similar purpose to "+
    "deadband_x but on the vertical axis. 0.05-0.1 typical."},
  deadband_skew:    {title:'Skew dead-band', units:'normalised', body:
    "Below this threshold the perpendicular-alignment (strafe) command is "+
    "zero. Skew measures how tilted the marker appears — a small tilt "+
    "doesn't need correction."},
  deadband_dist_m:  {title:'Distance dead-band', units:'metres', body:
    "Below this distance-error threshold the forward/back command is zero. "+
    "Keeps the drone parked once it's ~within range of hover_distance_m.\n\n"+
    "Too small and you get buzzing; too large and the drone drifts."},
  yaw_p:            {title:'Yaw P-gain', units:'per err_x', body:
    "Proportional gain from horizontal image-error to yaw RC%. Higher = "+
    "snappier rotation toward the marker but more overshoot."},
  skew_p:           {title:'Lateral P-gain', units:'per skew', body:
    "Proportional gain from marker-tilt to sideways RC%. Used to strafe "+
    "around the marker for head-on approach."},
  alt_p:            {title:'Altitude P-gain', units:'per err_y', body:
    "Proportional gain from vertical image-error to vertical RC%. Higher = "+
    "snappier climb/descend to marker height."},
  d_yaw:            {title:'Yaw D-damping', units:'per °/s (gyro)', body:
    "Derivative damping on yaw using the gyro. Subtracts a fraction of the "+
    "current yaw-rate from the yaw command — cancels oscillation.\n\n"+
    "If the drone visibly orbits around the marker, increase."},
  d_lr:             {title:'Lateral D-damping', units:'per cm/s (vgy)', body:
    "Derivative damping on sideways RC using the body-frame Y velocity "+
    "from the IMU. Prevents over-strafing."},
  d_ud:             {title:'Vertical D-damping', units:'per cm/s (vgz)', body:
    "Derivative damping on vertical RC using the body-frame Z velocity. "+
    "Prevents vertical oscillation as the drone approaches target altitude."},
  d_fb:             {title:'Fwd/back D-damping', units:'per cm/s (vgx)', body:
    "Derivative damping on forward/back RC using the body-frame X velocity. "+
    "The most important D term for not slamming into the net.\n\n"+
    "Combined with the boundary guard (arena edge prediction) this is the "+
    "primary brake during fast approaches."},
  yaw_max:          {title:'Clamp · max yaw', units:'% RC', body:
    "Hard upper limit for yaw RC%. Applied after the PD math. Keeps the "+
    "drone from spinning wildly on large errors. 20-30 typical."},
  lr_max:           {title:'Clamp · max lateral', units:'% RC', body:
    "Hard upper limit for sideways RC%. 20-40 typical — too high and the "+
    "drone over-strafes during approach."},
  ud_max:           {title:'Clamp · max vertical', units:'% RC', body:
    "Hard upper limit for vertical RC%. Typical 20-40. Raise for faster "+
    "altitude acquisition if the ceiling guard allows."},
  rc_min:           {title:'RC dead-floor', units:'% RC', body:
    "Below this magnitude the RC output is forced to zero. Anafi ignores "+
    "very small RC values anyway — this prevents buzzing/whine while "+
    "hovered. Usually 2-3."},
  cam_hfov_deg:     {title:'Camera H-FOV (drawing only)', units:'degrees', body:
    "Used ONLY to draw the camera cone in the top-down view — does NOT "+
    "affect PnP or any control. 69° is the Anafi nominal."},
  marker_size_m:    {title:'Observer marker size', units:'metres', body:
    "Physical marker side length for the observer's own PnP. Must match "+
    "the printed markers. SDC26 markers are 0.5 m.\n\n"+
    "Tip: keep this in sync with the Position Tracker's marker size."},

  // ===== Position Tracker (arena-frame pose fusion) =====
  detect_profile:     {title:'Detection profile', body:
    "Preset for the ArUco detector parameters (corner refinement, adaptive "+
    "threshold window, min marker size). Profiles:\n"+
    "  · Balanced  — default, good speed + robustness.\n"+
    "  · Sensitive — lighter thresholds, catches distant / partially-lit "+
    "markers at higher CPU cost.\n"+
    "  · Strict    — tighter accept, rejects noisy detections."},
  fov_deg:            {title:'Camera H-FOV', units:'degrees', body:
    "Horizontal field-of-view used to synthesise the intrinsics matrix "+
    "when no calibration file is loaded. Anafi 4K ≈ 69°.\n\n"+
    "Uploading a .npz calibration overrides this."},
  latency_ms:         {title:'Video-to-IMU latency', units:'ms', body:
    "How long the camera frame is old relative to the current IMU sample. "+
    "The positioner rewinds the IMU buffer by this amount so the IMU "+
    "velocity used for dead-reckoning corresponds to the same moment as "+
    "the vision measurement.\n\n"+
    "Measure: the header Latency row shows c2→fc + fc→drone RTT plus a "+
    "video decode offset; enabling auto-set pushes that total here."},
  imu_weight:         {title:'IMU ↔ ArUco blend', units:'0..1', body:
    "Mix between pure ArUco pose (0) and pure IMU dead-reckoning (1). "+
    "Higher = smoother but drifts more during marker outages.\n\n"+
    "30 % is a good default. Raise to 50-60 % if markers flicker in/out, "+
    "lower if the IMU shows bias."},
  enable_kalman_filter:{title:'Kalman filter', body:
    "Per-axis 1-D Kalman filter on x/y/z. Models state as position+velocity "+
    "and fuses ArUco fixes as measurements.\n\n"+
    "ON (recommended): smoother, handles brief marker dropouts.\n"+
    "OFF: pose jumps straight to the last ArUco solution — noisier but "+
    "with zero added latency."},
  top_k_markers:      {title:'Top-K markers', body:
    "Use only the N closest markers in the weighted-mean fusion. 0 = auto "+
    "(picks 4). Smaller K = faster, less robust to outliers. Larger K = "+
    "more samples but includes distant, less-accurate detections."},
  outlier_reject_m:   {title:'Outlier reject distance', units:'metres', body:
    "Per-marker poses further than this from the weighted-mean position "+
    "are rejected as outliers before re-averaging.\n\n"+
    "Default 2.5 m. Tighten to 1.0 m for a smaller, tidy arena; loosen to "+
    "3.5 m if markers are far apart."},
  target_marker_size_m: {title:'Target-box marker size', units:'metres', body:
    "Physical side length of the ArUco stickers on SDC26 target boxes.\n\n"+
    "The drone's ArUco dictionary (DICT_4X4_100) has 100 IDs. IDs 0-29 are "+
    "reserved for arena wall/reference markers (50 cm) and IDs \u226530 for "+
    "target boxes (19 cm SDC26 default).\n\n"+
    "This matters because solvePnP needs the real marker size to compute "+
    "distance correctly — if we measure 19 cm markers with a 50 cm template, "+
    "we get distances 50/19 \u2248 2.6\u00d7 too large, and the target lands far "+
    "from where it really is.\n\n"+
    "Default 0.19 m. Adjust only if your arena uses a different sticker size."},
  zupt_speed_m_s:     {title:'ZUPT speed threshold', units:'m/s (0 = disabled)', body:
    "Zero-Velocity Update. When the IMU reports speed below this threshold "+
    "for \"hold\" consecutive frames, the positioner snaps the fresh ArUco "+
    "measurement to the last valid pose — eliminates the parked-drone drift "+
    "caused by sub-pixel corner noise + IPPE mirror ambiguity.\n\n"+
    "Below 0.05 m/s is \"not moving\" in practice. Raise to 0.10 m/s for "+
    "very noisy indoor flight (it'll also freeze during slow hovers), or "+
    "drop to 0.02 m/s if you want aggressive drift suppression.\n\n"+
    "0 disables."},
  zupt_hold_frames:   {title:'ZUPT hold frames', units:'frames', body:
    "How many consecutive slow frames are required before ZUPT engages. "+
    "Stops false-triggering during quick directional changes (where IMU "+
    "speed dips briefly as the drone reverses).\n\n"+
    "At 5 Hz positioning: 3 frames \u2248 600 ms. That's a good balance — long "+
    "enough to ignore brief lulls, short enough to lock in quickly when "+
    "actually parked."},
  max_pose_jump_m:    {title:'Pose-jump gate', units:'metres (0 = disabled)', body:
    "Safety limit on how far a fresh ArUco fix is allowed to disagree with "+
    "the Kalman-predicted state before it's rejected as an outlier.\n\n"+
    "Single-marker solvePnP occasionally produces mirror-pose glitches that "+
    "put the drone 10-20 m from where it actually is. The gate silently "+
    "drops those fixes (the filter keeps predicting forward) so the UI and "+
    "missions don't see the spike.\n\n"+
    "Default 0 = off. 3.0 m is a good starting value for indoor arenas — "+
    "large enough not to block legitimate fast motion, small enough to "+
    "catch the bad frames. Auto Positioning uses 3.0 m by default."},
  distance_scale:     {title:'Distance correction factor', units:'multiplier (1.0 = no correction)', body:
    "Multiplicative correction applied to the camera\u2194marker translation "+
    "from solvePnP. Compensates for systematic scale error, usually caused "+
    "by a mismatch between the real marker size and the configured "+
    "marker_size_m, or by an uncalibrated focal length.\n\n"+
    "Calibration recipe: put the drone a known distance D_real from a marker, "+
    "read the UI distance D_ui, then set distance_scale = D_real / D_ui.\n\n"+
    "Example: UI shows 7 m, tape measure says 9 m \u2192 9 / 7 \u2248 1.286.\n\n"+
    "The scale applies equally to every pose axis, so it works whether the "+
    "error shows up in x, y, z, or the combined 3-D distance. If the error "+
    "is direction-dependent, fix the camera calibration instead."},
  pose_hold_sec:      {title:'Pose hold (dead-reckon)', units:'seconds', body:
    "After the last valid ArUco fix, keep publishing pose based on IMU "+
    "dead-reckoning for this many seconds.\n\n"+
    "Too short → pose vanishes every time the camera blinks.\n"+
    "Too long  → stale pose during marker outages drifts meters.\n"+
    "0.5-1.0 s typical. Raise only if the IMU is well-calibrated."},
  min_ref_count:      {title:'Minimum reference markers', body:
    "Require at least this many markers to be visible before a fused pose "+
    "is accepted. 1 = accept a single marker (can be noisy). 2-3 gives a "+
    "much more robust fix by cross-checking."},
  min_ref_weight:     {title:'Minimum reference weight', units:'0..1', body:
    "Require the best-matching marker to have at least this fused weight. "+
    "Rejects very low-confidence fits (tiny / extreme-angle markers).\n\n"+
    "0 = accept any detection. 0.2-0.4 is restrictive but clean."},
  meas_blend_min:     {title:'Measurement blend — low', units:'α ∈ [0..1]', body:
    "Minimum EMA α applied to fresh ArUco measurements when fusing with "+
    "the Kalman state. Used when fix quality is high (trust the filter).\n\n"+
    "Lower values = more Kalman smoothing, less jitter."},
  meas_blend_max:     {title:'Measurement blend — high', units:'α ∈ [0..1]', body:
    "Maximum EMA α used when fix quality is low (trust the latest fresh "+
    "measurement more). The positioner interpolates between min and max "+
    "based on residual error and ref count."},
  vel_blend:          {title:'Velocity blend', units:'0..1', body:
    "Blend between IMU-measured velocity (0) and Kalman-state velocity "+
    "derivative (1). Raises 0.25 means 25 % Kalman-derived, 75 % raw IMU.\n\n"+
    "Higher = smoother vz/vy plots. Lower = faster reaction to real motion."},
  max_state_dt:       {title:'Max state Δt', units:'seconds', body:
    "If more than this amount of time passes between updates, the Kalman "+
    "state is reset instead of extrapolated. Prevents exploding covariance "+
    "during long outages (landing, lost camera link)."},
  kalman_process_var: {title:'Kalman process variance (Q)', body:
    "How much the state is expected to change between steps. Low Q → the "+
    "filter believes its model (smooth but sluggish). High Q → the filter "+
    "expects rapid changes (reacts faster but noisier).\n\n"+
    "Default 1e-3. Try 5e-4 for smooth hover, 5e-3 for dynamic missions."},
  imu_lowpass_hz:     {title:'IMU low-pass cut-off', units:'Hz', body:
    "First-order IIR low-pass filter applied to body-frame IMU "+
    "velocity (vgx, vgy, vgz) at telemetry ingestion on the Pi. Smooths "+
    "Anafi's noisy per-sample velocity before it reaches the position "+
    "fusion, the synchronisation buffer, and the arena view.\n\n"+
    "Parameter: cut-off frequency (Hz). Lower = more smoothing.\n"+
    "  • 0       → filter disabled (raw values passed through).\n"+
    "  • 1–3 Hz  → heavy smoothing, visibly laggy reaction.\n"+
    "  • 5 Hz    → default; cuts Anafi's ~15 Hz broadband jitter without\n"+
    "               dulling reaction to real motion.\n"+
    "  • 10-30 Hz→ minimal smoothing, reacts to faster manoeuvres.\n\n"+
    "Implemented as a time-based alpha = dt / (tau + dt), where "+
    "tau = 1 / (2π·fc), so uneven telemetry intervals still yield "+
    "the correct filter response. State is reset when the filter is "+
    "disabled so re-enabling starts clean."},
  kalman_meas_var:    {title:'Kalman measurement variance (R)', body:
    "How noisy the ArUco measurements are. Low R → trust the camera "+
    "more (snaps to detections). High R → trust the model more (smoother "+
    "but may lag).\n\n"+
    "Default 1e-1. Large markers at short range can use 1e-2; noisy, "+
    "distant markers may benefit from 3e-1."},

  // ===== Safety =====
  axis_locked: {title:'Axis-locked (Manhattan) autonomous flight', body:
    "When ON, autonomous motion (missions + ArUco LIVE + waypoint nav) "+
    "is constrained to wall-parallel axes only:\n\n"+
    "  1. Yaw snaps to the nearest 90° multiple (0° / 90° / 180° / 270° "+
    "in arena frame). The drone aligns to a cardinal heading before "+
    "moving. Rotations happen in 90° increments only.\n"+
    "  2. Only ONE horizontal axis moves at a time: either strafe (LR) "+
    "or forward/back (FB), whichever has the larger magnitude. The "+
    "drone never flies diagonally.\n"+
    "  3. Vertical (up/down) is unaffected.\n\n"+
    "Manual WASD / Q-E / R-F is unaffected — this constraint applies "+
    "exclusively to autonomous decisions (observer LIVE + waypoints).\n\n"+
    "Typical use: structured arena navigation where diagonals invite "+
    "unnecessary marker-tracking jitter. Default OFF."},
  camera_face_center: {title:'Camera → arena centre', body:
    "Fleet-wide toggle: when ON, every observer's waypoint face-target "+
    "defaults to the arena centre (x=0, y=half-depth). During autonomous "+
    "missions, the drone's camera then aims at the centre of the arena "+
    "regardless of which direction it's flying — keeping the maximum "+
    "number of ArUco markers in view at once.\n\n"+
    "Why: the position processor fuses poses from every visible marker. "+
    "More visible markers = tighter fused arena-frame pose. Pointing the "+
    "camera at one specific marker (the mission's target) means only a "+
    "handful of markers are in the FOV at a time; pointing at the centre "+
    "typically keeps 4-6 markers visible throughout a flight.\n\n"+
    "Missions can still override this on a per-call basis by passing an "+
    "explicit face target. If OFF, missions that don't set a face leave "+
    "the camera aligned with the drone's direction of travel.\n\n"+
    "Arena centre defaults to (0, 5.4) for a 20×10.8m arena. The value "+
    "is adjustable via POST /proxy/config/camera_face_center body "+
    "{\"xy\": [x, y]}."},
  arena_guard_enabled: {title:'Arena guard (manual + auto)', body:
    "When ON (default), the Pi's own RC tick loop runs a boundary "+
    "guard on EVERY command — manual WASD, autonomous missions, "+
    "ArUco LIVE, everything. If the drone is within the safety "+
    "margin of any arena wall AND the command would push it closer, "+
    "that axis of the command is clamped to zero.\n\n"+
    "Independent of C2 connection — the guard runs on the flight "+
    "controller itself using the position processor's arena-frame "+
    "pose. If the position fix is stale or missing, the guard "+
    "gracefully falls back (no clamping) and the operator owns the "+
    "boundary decision.\n\n"+
    "Turn OFF for test/debug flights where you want raw control."},
  safety_margin_m: {title:'Arena safety margin', units:'metres', body:
    "Minimum distance the drone will maintain from ANY arena wall "+
    "during autonomous flight. The boundary guard runs per-tick at "+
    "20 Hz and overrides any waypoint or PD command that would drive "+
    "the drone closer to a wall than this margin.\n\n"+
    "Defaults to 1.5 m per ops policy. Adjustable 0.1–5.0 m.\n\n"+
    "The guard uses the latency-aware lookahead (GUARD_LOOKAHEAD_S = "+
    "0.35 s) so it accounts for momentum — the drone brakes / retreats "+
    "BEFORE it would cross the margin, not after. Only active during "+
    "autonomous flight (observer LIVE / waypoint). Manual flight is "+
    "bounded only by the hard altitude ceiling."},
  ceiling_m: {title:'Hard altitude ceiling', units:'metres', body:
    "Maximum altitude above ground. The value is set from this UI, but "+
    "the enforcement is ENTIRELY on each drone's flight-controller Pi — "+
    "in its own 20 Hz RC tick loop, using its own height_cm telemetry. "+
    "Independent of any C2 connection: if this browser or the C2 server "+
    "crashes mid-flight, the Pi keeps clamping upward RC.\n\n"+
    "Behaviour:\n"+
    "  • approaching (within 50 cm): climb RC clamped proportionally to "+
    "remaining clearance.\n"+
    "  • at ceiling: all climb blocked.\n"+
    "  • above ceiling (+20 cm): forced active descent regardless of any "+
    "input — manual WASD or autonomous mission, no difference.\n\n"+
    "Persistence: the Pi writes the chosen value to flight_config.json "+
    "and reloads it on restart, so power-cycling the Pi still leaves "+
    "your last-set ceiling active. The firmware MaxAltitude is also "+
    "pushed to the Anafi as a second-line guard on every connect.\n\n"+
    "Default 5 m."},
};

window.showParamInfo = function(key, ev) {
  if (ev) { ev.stopPropagation(); ev.preventDefault(); }
  const info = window.PARAM_INFO[key];
  const m = document.getElementById('param_info_modal');
  if (!m) return;
  if (!info) {
    // Still show the modal with a graceful note — useful while adding new params.
    document.getElementById('pim_title').textContent = key;
    document.getElementById('pim_key').textContent   = key;
    document.getElementById('pim_body').textContent  = 'No description registered for this parameter yet.';
    document.getElementById('pim_range').textContent = '';
  } else {
    document.getElementById('pim_title').textContent = info.title || key;
    document.getElementById('pim_key').textContent   = key;
    document.getElementById('pim_body').textContent  = info.body || '';
    document.getElementById('pim_range').textContent = info.units ? ('Units: ' + info.units) : '';
  }
  m.style.display = 'flex';
};

// Event delegation — any element with class .info-icon and data-info="<key>"
// opens the modal. Works for icons injected later (observer PD rows) too.
document.addEventListener('click', function(ev){
  const el = ev.target.closest && ev.target.closest('.info-icon');
  if (!el) return;
  const key = el.dataset.info || el.getAttribute('data-info');
  if (!key) return;
  window.showParamInfo(key, ev);
});

// ── Light / dark theme toggle ──────────────────────────────────────
// Persists via localStorage. Default = dark (matches the original UI).
// The data-theme attribute drives the CSS overrides at the top of the
// <style> block. Using !important there lets the light theme defeat
// the many hard-coded inline style="" colours without rewriting every
// DOM element.
(function wireThemeToggle(){
  const KEY = 'sdc_theme';
  const btn = document.getElementById('theme_toggle');
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    if (btn) btn.innerHTML = (t === 'light') ? '\u2600\ufe0f Light' : '\ud83c\udf19 Dark';
  }
  const saved = (function(){ try { return localStorage.getItem(KEY) || 'dark'; } catch { return 'dark'; } })();
  applyTheme(saved === 'light' ? 'light' : 'dark');
  if (btn) btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem(KEY, next); } catch {}
    applyTheme(next);
  });
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
    cs.textContent = d.ok ? '\u2713 calibration saved' : ('error: ' + d.error);
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
    recBtn.textContent = _recActive ? '\u25a0 Stop Rec' : '\u25cf Record';
    recBtn.style.borderColor = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.color = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.background = _recActive ? '#3b0f0f' : '#1e3a2e';
    document.getElementById('rec_raw').disabled = _recActive;
    if (_recActive) recStatus.textContent = `${d.frames} frames • ${d.raw ? 'raw' : 'ann'} • ${d.path.split('/').pop()}`;
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
    // Cache target-box metadata for the Target Boxes panel — team &
    // colour come from here, keyed by marker ID.
    window._arenaCfgCache = {
      target_marker_size_m: d.target_marker_size_m,
      target_teams:      d.target_teams || [],
      target_overrides:  d.target_overrides || [],
    };
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
// Load arena config once at startup so the Target Boxes panel has
// team/colour metadata immediately (operator doesn't need to click
// "Show" on Arena Configuration first).
window.addEventListener('load', () => {
  try { loadArenaConfig(); } catch (e) {}
});

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
    st.textContent = d.ok ? `\u2713 saved (${d.marker_count} markers)` : ('error: ' + (d.error||'?'));
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
    if (d.ok) { await loadArenaConfig(); st.textContent = '\u2713 reset to defaults'; st.style.color = '#22c55e'; }
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
