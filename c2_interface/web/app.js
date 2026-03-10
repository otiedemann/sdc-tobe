let state = { arena: null, drones: [], system: { mode: 'live', drone_count: 2 }, targets: [] };
let targetEditMode = false;
const simDronePos = new Map();
let simFramesKey = '';
let simSpawnKey = '';
let cardsRenderKey = '';
let droneSelectKey = '';
let currentDroneId = '';
let systemConfigDirty = false;
let gridActionInFlight = false;
const SIM_BUILD_TAG = '20260308-2108';

const gridEl = document.getElementById('grid');
const droneSelect = document.getElementById('droneSelect');
const cardsEl = document.getElementById('cards');
const modeSelect = document.getElementById('modeSelect');
const droneCountSelect = document.getElementById('droneCountSelect');
const applySystemBtn = document.getElementById('applySystem');
const targetModeBtn = document.getElementById('targetMode');
const simEmbedEl = document.getElementById('simEmbed');
const connectLiveBtn = document.getElementById('connectLive');
const droneOverlay = document.getElementById('droneOverlay');
const targetColorSelect = document.getElementById('targetColorSelect');
const targetListEl = document.getElementById('targetList');

function randomHomeTargets() {
  const targets = [];
  const pickUnique = (xmin, xmax, ymin, ymax, color) => {
    while (targets.length < (color === 'red' ? 3 : 6)) {
      const x = Math.floor(Math.random() * (xmax - xmin + 1)) + xmin;
      const y = Math.floor(Math.random() * (ymax - ymin + 1)) + ymin;
      if (targets.some(t => t.x === x && t.y === y)) continue;
      targets.push({ x, y, color });
    }
  };
  // 20x10 grid: red home in left half, blue home in right half.
  pickUnique(0, 6, 0, 9, 'red');
  pickUnique(13, 19, 0, 9, 'blue');
  return targets;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function postSimCommand(cmd) {
  const frames = document.querySelectorAll('iframe[data-sim-frame="1"]');
  frames.forEach(frame => {
    if (!frame?.contentWindow) return;
    frame.contentWindow.postMessage({ source: 'c2', ...cmd }, '*');
  });
}

function targetMap() {
  const m = new Map();
  for (const t of (state.targets || [])) m.set(`${t.x},${t.y}`, t.color || 'red');
  return m;
}

function render() {
  if (!state.arena) return;

  // Overlay simulator-reported positions onto C2 state for accurate map alignment.
  if (simDronePos.size) {
    state.drones = state.drones.map((d) => {
      const p = simDronePos.get(d.drone_id);
      if (!p) return d;
      return { ...d, position: { x: p.x, y: p.y } };
    });
  }

  if (!systemConfigDirty) {
    modeSelect.value = state.system.mode;
    droneCountSelect.value = String(state.system.drone_count);
  }
  targetModeBtn.textContent = targetEditMode ? 'Target mode ON (click cells)' : 'Set targets (max 6)';
  connectLiveBtn.disabled = state.system.mode !== 'live';

  const activeDrones = state.drones.filter(d => d.status !== 'offline');
  const nextSelectKey = activeDrones.map(d => `${d.drone_id}:${d.status}`).join('|');
  if (nextSelectKey !== droneSelectKey) {
    const prev = currentDroneId || droneSelect.value;
    droneSelect.innerHTML = activeDrones.map(d => `<option value="${d.drone_id}">${d.drone_id} (${d.status})</option>`).join('');
    const fallback = activeDrones[0]?.drone_id || '';
    currentDroneId = activeDrones.some(d => d.drone_id === prev) ? prev : fallback;
    if (currentDroneId) droneSelect.value = currentDroneId;
    droneSelectKey = nextSelectKey;
  } else if (currentDroneId) {
    droneSelect.value = currentDroneId;
  }

  const occupied = new Map(activeDrones.map(d => [`${d.position.x},${d.position.y}`, d.drone_id]));
  const tmap = targetMap();
  gridEl.innerHTML = state.arena.cells.map(c => {
    const key = `${c.x},${c.y}`;
    const occ = occupied.get(key);
    const color = tmap.get(key);
    const targetCls = color ? `target-${color}` : '';
    const label = color ? ` target:${color}` : '';
    return `<button class="cell ${c.zone} ${occ ? 'occupied' : ''} ${targetCls}" data-x="${c.x}" data-y="${c.y}" title="(${c.x},${c.y})${label} ${occ ? occ : ''}"></button>`;
  }).join('');

  droneOverlay.innerHTML = activeDrones.map((d) => {
    const left = d.position.x * 30 + 1;
    const top = d.position.y * 30 + 1;
    const num = (d.drone_id.split('-').pop() || '?');
    return `<div class="drone-icon" style="left:${left}px;top:${top}px" title="${d.drone_id}">🚁${num}</div>`;
  }).join('');

  targetListEl.innerHTML = `<strong>Targets:</strong> ` + (state.targets?.length
    ? state.targets.map((t, i) => `#${i+1} (${t.x},${t.y}) <b style="color:${t.color === 'red' ? '#f87171' : '#60a5fa'}">${t.color}</b>`).join(' | ')
    : 'none');

  let simUrl = state.system.simulator_url;
  if (simUrl) {
    try {
      const u = new URL(simUrl, window.location.origin);
      if (u.hostname === 'localhost' || u.hostname === '127.0.0.1') {
        u.hostname = window.location.hostname;
      }
      simUrl = u.pathname + u.search + u.hash;
    } catch {}
  }

  const nextCardsKey = JSON.stringify({
    mode: state.system.mode,
    simUrl,
    drones: activeDrones.map(d => d.drone_id),
  });
  if (nextCardsKey !== cardsRenderKey) {
    cardsEl.innerHTML = activeDrones.map(d => {
      let video = `<div class="video-preview"><div class="video-meta">960 × 720</div><div class="video">No video URL configured</div></div>`;
      if (state.system.mode === 'simulator' && simUrl) {
        const idx = Number((d.drone_id.split('-').pop() || '1'));
        const fpvUrl = `${simUrl}${simUrl.includes('?') ? '&' : '?'}panel=fpv&embed=1&team=red&drone=${idx}&cam=fpv&v=${SIM_BUILD_TAG}`;
        video = `<div class="video-preview"><div class="video-meta">960 × 720</div><iframe data-sim-frame="1" src="${fpvUrl}" class="video" style="border:0"></iframe></div>`;
      } else if (d.video_url) {
        video = `<div class="video-preview"><div class="video-meta">960 × 720</div><img src="${d.video_url}?t=${Date.now()}" class="video" alt="${d.drone_id} video" /></div>`;
      }
      return `
        <div class="card" data-drone-card="${d.drone_id}">
          <strong>${d.drone_id}</strong> — <span data-drone-status="${d.drone_id}">${d.status}</span><br/>
          <span data-drone-pos="${d.drone_id}">Pos: (${d.position.x}, ${d.position.y}) | Home: (${d.home.x}, ${d.home.y})</span>
          ${video}
          <div class="actions">
            <button data-target="${d.drone_id}" class="spot">Target in sight</button>
          </div>
        </div>
      `;
    }).join('');
    cardsRenderKey = nextCardsKey;
  } else {
    activeDrones.forEach((d) => {
      const s = cardsEl.querySelector(`[data-drone-status="${d.drone_id}"]`);
      if (s) s.textContent = d.status;
      const p = cardsEl.querySelector(`[data-drone-pos="${d.drone_id}"]`);
      if (p) p.textContent = `Pos: (${d.position.x}, ${d.position.y}) | Home: (${d.home.x}, ${d.home.y})`;
    });
  }

  if (state.system.mode === 'simulator' && simUrl) {
    const modelUrl = `${simUrl}${simUrl.includes('?') ? '&' : '?'}panel=model&embed=1&team=red&v=${SIM_BUILD_TAG}`;
    const framesKey = `${modelUrl}`;
    if (framesKey !== simFramesKey) {
      simEmbedEl.innerHTML = `
        <h3>Simulator (auto-connected)</h3>
        <div class="sim-controls">
          <button id="simRandomTargetsBtn" type="button">Place targets at random</button>
          <div class="speed-wrap">
            <label for="simSpeedRange">Drone speed</label>
            <input id="simSpeedRange" type="range" min="0.6" max="4" step="0.1" value="1.2" />
            <span id="simSpeedVal">1.2</span>
          </div>
        </div>
        <div class="sim-grid">
          <iframe data-sim-frame="1" class="sim-frame full" src="${modelUrl}"></iframe>
        </div>
      `;

      const rndBtn = document.getElementById('simRandomTargetsBtn');
      const speedRange = document.getElementById('simSpeedRange');
      const speedVal = document.getElementById('simSpeedVal');
      if (rndBtn) {
        rndBtn.addEventListener('click', async () => {
          // Simulator-only randomization: do NOT overwrite manual C2 target planning layer.
          postSimCommand({ kind: 'targets_random_home' });
        });
      }
      if (speedRange && speedVal) {
        const pushSpeed = () => {
          speedVal.textContent = Number(speedRange.value).toFixed(1);
          postSimCommand({ kind: 'set_speed', speed: Number(speedRange.value) });
        };
        speedRange.addEventListener('input', pushSpeed);
        setTimeout(pushSpeed, 150);
      }

      simFramesKey = framesKey;
      simSpawnKey = '';
    }
  } else {
    simEmbedEl.innerHTML = '';
    simFramesKey = '';
    simSpawnKey = '';
    simDronePos.clear();
  }

  if (state.system.mode === 'simulator') {
    const spawnKey = `${state.system.drone_count}|red|c2`;
    setTimeout(() => {
      if (spawnKey !== simSpawnKey) {
        simDronePos.clear();
        postSimCommand({ kind: 'spawn', droneCount: Number(state.system.drone_count), enemyDroneCount: 0, team: 'red', mode: 'c2' });
        postSimCommand({
          kind: 'sync_state',
          drones: (state.drones || [])
            .filter(d => d.status !== 'offline')
            .map(d => ({ drone_id: d.drone_id, x: d.position.x, y: d.position.y }))
        });
        simSpawnKey = spawnKey;
      }
    }, 120);
  }
}

async function applySystemConfig() {
  const desiredMode = modeSelect.value;
  const desiredCount = Number(droneCountSelect.value);
  try {
    await api('/api/system', {
      method: 'POST',
      body: JSON.stringify({ mode: desiredMode, drone_count: desiredCount }),
    });
    systemConfigDirty = false;
  } catch (err) {
    alert(`Failed to apply system config: ${err.message || err}`);
  }
}

applySystemBtn.addEventListener('click', applySystemConfig);
modeSelect.addEventListener('change', () => { systemConfigDirty = true; });
droneCountSelect.addEventListener('change', () => { systemConfigDirty = true; });

targetModeBtn.addEventListener('click', () => {
  targetEditMode = !targetEditMode;
  render();
});

droneSelect.addEventListener('change', () => {
  currentDroneId = droneSelect.value;
});

connectLiveBtn.addEventListener('click', async () => {
  const res = await api('/api/live/connect', { method: 'POST', body: JSON.stringify({}) });
  alert('Live connect result: ' + JSON.stringify(res.result));
});

gridEl.addEventListener('click', async (e) => {
  let x, y;
  const el = e.target.closest('.cell');
  if (el) {
    x = Number(el.dataset.x);
    y = Number(el.dataset.y);
  } else {
    // Also accept clicks on the grid gaps between cells.
    const rect = gridEl.getBoundingClientRect();
    const gx = Math.floor(((e.clientX - rect.left) / Math.max(1, rect.width)) * 20);
    const gy = Math.floor(((e.clientY - rect.top) / Math.max(1, rect.height)) * 10);
    x = Math.max(0, Math.min(19, gx));
    y = Math.max(0, Math.min(9, gy));
  }
  if (gridActionInFlight) return;

  gridActionInFlight = true;
  try {
    if (targetEditMode) {
      const prevTargets = [...(state.targets || [])];
      const targets = [...prevTargets];
      const idx = targets.findIndex(t => t.x === x && t.y === y);
      if (idx >= 0) targets.splice(idx, 1);
      else {
        if (targets.length >= 6) return alert('Maximum 6 targets');
        targets.push({ x, y, color: targetColorSelect.value });
      }
      // optimistic update for snappy, single-click behavior
      state.targets = targets;
      render();
      try {
        await api('/api/targets', { method: 'POST', body: JSON.stringify({ targets }) });
      } catch (err) {
        state.targets = prevTargets;
        render();
        alert(`Failed to update targets: ${err.message || err}`);
      }
      return;
    }

    const droneId = currentDroneId || droneSelect.value;
    if (!droneId) return alert('No drone selected');

    // Apply reroute immediately in simulator so course changes are responsive.
    if (state.system.mode === 'simulator') postSimCommand({ kind: 'goto', droneId, x, y });

    try {
      await api(`/api/drones/${droneId}/goto`, { method: 'POST', body: JSON.stringify({ x, y }) });
    } catch (err) {
      alert(`Failed to send goto: ${err.message || err}`);
    }
  } finally {
    // tiny release delay avoids duplicate multi-click bursts while websocket rerenders
    setTimeout(() => { gridActionInFlight = false; }, 120);
  }
});

cardsEl.addEventListener('click', async (e) => {
  const btn = e.target.closest('button.spot');
  if (!btn) return;
  const droneId = btn.dataset.target;
  await api(`/api/drones/${droneId}/target-spotted`, { method: 'POST', body: JSON.stringify({}) });
});

async function bootstrap() {
  state = await api('/api/state');
  render();

  window.addEventListener('message', (ev) => {
    const m = ev.data || {};
    if (m.source !== 'sim' || m.kind !== 'state' || !Array.isArray(m.drones)) return;
    const nextPos = new Map();
    for (const d of m.drones) {
      if (!d?.drone_id || !d?.grid) continue;
      nextPos.set(d.drone_id, { x: Number(d.grid.x), y: Number(d.grid.y) });
    }
    simDronePos.clear();
    for (const [k, v] of nextPos.entries()) simDronePos.set(k, v);
    // Mirror model-frame world state into FPV frames so feeds stay aligned with main simulator.
    postSimCommand({ kind: 'sync_world', drones: m.drones });
    render();
  });

  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/ws`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'state') {
      state.drones = msg.drones;
      state.system = msg.system;
      state.targets = msg.targets;
      render();
    }
  };
}

setInterval(() => {
  if (state.system.mode === 'simulator' || state.system.mode === 'live') {
    document.querySelectorAll('img.video').forEach(img => {
      const base = img.src.split('?')[0];
      img.src = `${base}?t=${Date.now()}`;
    });
  }
}, 1000);

bootstrap();
