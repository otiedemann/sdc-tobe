/**
 * Live Telemetry Graphs — real-time canvas charts from SSE telemetry stream.
 *
 * Connects to the unified API server's /api/telemetry/stream SSE endpoint
 * and renders rolling 10-second graphs for all telemetry channels.
 */

(function () {
  'use strict';

  const WINDOW_S = 10;        // seconds of history to show
  const MAX_SAMPLES = 200;    // max data points (10s at 10Hz + margin)
  const GRAPH_W = 360;
  const GRAPH_H = 120;
  const PADDING = { top: 18, right: 8, bottom: 22, left: 50 };

  // Colors per channel (dark theme friendly)
  const COLORS = [
    '#22d3ee', '#a78bfa', '#f472b6', '#34d399', '#fbbf24',
    '#fb923c', '#60a5fa', '#e879f9', '#4ade80', '#f87171',
    '#38bdf8', '#c084fc', '#fb7185', '#2dd4bf', '#facc15',
  ];

  // Channel definitions: which telemetry keys to graph and how to group them
  const CHANNEL_GROUPS = [
    { title: 'Position (m)',       keys: ['pos_x', 'pos_y', 'pos_z'],  unit: 'm',     decimals: 2 },
    { title: 'Altitude',           keys: ['height_cm'],                 unit: 'cm',    decimals: 0 },
    { title: 'Attitude',           keys: ['pitch', 'roll', 'yaw'],     unit: '\u00B0', decimals: 1 },
    { title: 'Velocity',           keys: ['vgx', 'vgy', 'vgz'],       unit: 'cm/s',  decimals: 1 },
    { title: 'Acceleration',       keys: ['agx', 'agy', 'agz'],       unit: '',      decimals: 1 },
    { title: 'Speed',              keys: ['speed'],                    unit: 'cm/s',  decimals: 0 },
    { title: 'Battery',            keys: ['battery'],                  unit: '%',     decimals: 0 },
    { title: 'Barometer',          keys: ['barometer_cm'],             unit: 'cm',    decimals: 0 },
    { title: 'ToF Distance',       keys: ['tof_cm'],                   unit: 'cm',    decimals: 0 },
    { title: 'Temperature',        keys: ['temperature'],              unit: '\u00B0C', decimals: 0 },
  ];

  // ─── State ───────────────────────────────────────────────────────────

  let eventSource = null;
  let posEventSource = null;
  let series = {};           // { key: [{t, v}] }
  let graphs = {};           // { groupIdx: {canvas, ctx} }
  let connected = false;
  let lastPosData = {};      // latest position SSE data
  let animFrameId = null;
  let tOrigin = null;        // first sample timestamp

  const panelEl     = document.getElementById('telemetryPanel');
  const graphsEl    = document.getElementById('telemetryGraphs');
  const toggleBtn   = document.getElementById('telemetryToggle');
  const connectBtn  = document.getElementById('telemetryConnect');
  const disconnBtn  = document.getElementById('telemetryDisconnect');
  const statusEl    = document.getElementById('telemetryStatus');
  const apiUrlInput = document.getElementById('telemetryApiUrl');

  // ─── Toggle panel ────────────────────────────────────────────────────

  toggleBtn.addEventListener('click', () => {
    const visible = panelEl.style.display !== 'none';
    panelEl.style.display = visible ? 'none' : 'block';
    toggleBtn.textContent = visible ? '\uD83D\uDCCA Telemetry' : '\u2716 Close Telemetry';
  });

  // ─── Connect / Disconnect ────────────────────────────────────────────

  connectBtn.addEventListener('click', startStream);
  disconnBtn.addEventListener('click', stopStream);

  function setStatus(text, color) {
    statusEl.textContent = text;
    statusEl.style.color = color || '#94a3b8';
  }

  function startStream() {
    if (connected) stopStream();

    const base = apiUrlInput.value.trim().replace(/\/+$/, '');
    if (!base) return;

    // Reset
    series = {};
    lastPosData = {};
    tOrigin = null;
    buildGraphs();

    // Telemetry SSE (10 Hz)
    try {
      eventSource = new EventSource(base + '/api/telemetry/stream');
      eventSource.onmessage = onTelemetryMessage;
      eventSource.onerror = () => {
        setStatus('connection error', '#f87171');
      };
      eventSource.onopen = () => {
        setStatus('connected (telemetry)', '#34d399');
      };
    } catch (e) {
      setStatus('failed: ' + e.message, '#f87171');
      return;
    }

    // Position SSE (ArUco position — may not be available)
    try {
      posEventSource = new EventSource(base + '/api/position/events');
      posEventSource.onmessage = onPositionMessage;
      posEventSource.onerror = () => {}; // silent — position may not be enabled
      posEventSource.onopen = () => {
        setStatus('connected (telemetry + position)', '#34d399');
      };
    } catch (e) {
      // Position stream optional
    }

    connected = true;
    connectBtn.disabled = true;
    disconnBtn.disabled = false;
    startRenderLoop();
  }

  function stopStream() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    if (posEventSource) { posEventSource.close(); posEventSource = null; }
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    connected = false;
    connectBtn.disabled = false;
    disconnBtn.disabled = true;
    setStatus('disconnected', '#94a3b8');
  }

  // ─── SSE handlers ────────────────────────────────────────────────────

  function onTelemetryMessage(ev) {
    try {
      const d = JSON.parse(ev.data);
      const now = performance.now() / 1000;
      if (tOrigin === null) tOrigin = now;
      const t = now - tOrigin;

      for (const key of Object.keys(d)) {
        const v = d[key];
        if (typeof v !== 'number' || v === null) continue;
        pushSample(key, t, v);
      }
    } catch (e) { /* ignore parse errors */ }
  }

  function onPositionMessage(ev) {
    try {
      const d = JSON.parse(ev.data);
      lastPosData = d;
      const now = performance.now() / 1000;
      if (tOrigin === null) tOrigin = now;
      const t = now - tOrigin;

      // Extract position if available
      const pos = d.pos;
      if (Array.isArray(pos) && pos.length >= 3) {
        pushSample('pos_x', t, pos[0]);
        pushSample('pos_y', t, pos[1]);
        pushSample('pos_z', t, pos[2]);
      }
    } catch (e) { /* ignore */ }
  }

  function pushSample(key, t, v) {
    if (!series[key]) series[key] = [];
    const arr = series[key];
    arr.push({ t, v });
    // Trim old samples (keep 2x window for safety)
    const cutoff = t - WINDOW_S * 2;
    while (arr.length > 0 && arr[0].t < cutoff) arr.shift();
    if (arr.length > MAX_SAMPLES * 2) arr.splice(0, arr.length - MAX_SAMPLES);
  }

  // ─── Build graph canvases ────────────────────────────────────────────

  function buildGraphs() {
    graphsEl.innerHTML = '';
    graphs = {};

    CHANNEL_GROUPS.forEach((group, idx) => {
      const wrap = document.createElement('div');
      wrap.className = 'telem-graph-wrap';

      const canvas = document.createElement('canvas');
      canvas.width = GRAPH_W * 2;   // 2x for retina
      canvas.height = GRAPH_H * 2;
      canvas.style.width = '100%';
      canvas.style.height = 'auto';
      const ctx = canvas.getContext('2d');

      wrap.appendChild(canvas);
      graphsEl.appendChild(wrap);
      graphs[idx] = { canvas, ctx, group };
    });
  }

  // ─── Render loop ─────────────────────────────────────────────────────

  function startRenderLoop() {
    function frame() {
      if (!connected) return;
      renderAllGraphs();
      animFrameId = requestAnimationFrame(frame);
    }
    animFrameId = requestAnimationFrame(frame);
  }

  function renderAllGraphs() {
    const now = (performance.now() / 1000) - (tOrigin || 0);

    for (const idx of Object.keys(graphs)) {
      const { ctx, group } = graphs[idx];
      renderGraph(ctx, group, now);
    }
  }

  function renderGraph(ctx, group, tNow) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    const P = {
      top: PADDING.top * 2, right: PADDING.right * 2,
      bottom: PADDING.bottom * 2, left: PADDING.left * 2
    };
    const plotW = w - P.left - P.right;
    const plotH = h - P.top - P.bottom;
    const tMin = tNow - WINDOW_S;
    const tMax = tNow;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Compute Y range from visible data across all keys in group
    let yMin = Infinity, yMax = -Infinity;
    let hasData = false;

    for (const key of group.keys) {
      const arr = series[key];
      if (!arr) continue;
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i].t < tMin) break;
        if (arr[i].t > tMax) continue;
        const v = arr[i].v;
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
        hasData = true;
      }
    }

    if (!hasData) {
      yMin = 0; yMax = 1;
    }

    // Add 10% padding to Y range
    const yRange = yMax - yMin || 1;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;

    // ─── Grid lines ──────────────────────────────────────────────
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;

    // Horizontal grid (4 lines)
    for (let i = 0; i <= 4; i++) {
      const y = P.top + (plotH * i / 4);
      ctx.beginPath();
      ctx.moveTo(P.left, y);
      ctx.lineTo(P.left + plotW, y);
      ctx.stroke();
    }

    // Vertical grid (every 2 seconds)
    for (let s = 0; s <= WINDOW_S; s += 2) {
      const x = P.left + (s / WINDOW_S) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, P.top);
      ctx.lineTo(x, P.top + plotH);
      ctx.stroke();
    }

    // ─── Y-axis labels ──────────────────────────────────────────
    ctx.fillStyle = '#64748b';
    ctx.font = '18px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const y = P.top + (plotH * i / 4);
      const val = yMax - (yMax - yMin) * (i / 4);
      ctx.fillText(formatVal(val, group.decimals), P.left - 6, y);
    }

    // ─── X-axis labels ──────────────────────────────────────────
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let s = 0; s <= WINDOW_S; s += 2) {
      const x = P.left + (s / WINDOW_S) * plotW;
      ctx.fillText('-' + (WINDOW_S - s) + 's', x, P.top + plotH + 4);
    }

    // ─── Title + legend ─────────────────────────────────────────
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 20px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(group.title, P.left, 4);

    // Legend + current values
    let legendX = P.left + ctx.measureText(group.title).width + 16;
    ctx.font = '17px monospace';
    group.keys.forEach((key, ki) => {
      const color = COLORS[ki % COLORS.length];
      const arr = series[key];
      const lastVal = arr && arr.length > 0 ? arr[arr.length - 1].v : null;
      const label = key + (lastVal !== null ? '=' + formatVal(lastVal, group.decimals) : '');

      ctx.fillStyle = color;
      ctx.fillText(label, legendX, 6);
      legendX += ctx.measureText(label).width + 14;
    });

    // Unit
    if (group.unit) {
      ctx.fillStyle = '#475569';
      ctx.font = '16px monospace';
      ctx.fillText(group.unit, legendX, 7);
    }

    // ─── Plot lines ─────────────────────────────────────────────
    group.keys.forEach((key, ki) => {
      const arr = series[key];
      if (!arr || arr.length < 2) return;

      ctx.strokeStyle = COLORS[ki % COLORS.length];
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();

      let started = false;
      for (let i = 0; i < arr.length; i++) {
        const s = arr[i];
        if (s.t < tMin) continue;
        if (s.t > tMax) break;

        const x = P.left + ((s.t - tMin) / WINDOW_S) * plotW;
        const y = P.top + plotH - ((s.v - yMin) / (yMax - yMin)) * plotH;

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });

    // ─── Border ─────────────────────────────────────────────────
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.strokeRect(P.left, P.top, plotW, plotH);
  }

  function formatVal(v, decimals) {
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    return v.toFixed(Math.min(decimals, 2));
  }

})();
