let state = { arena: null, drones: [], system: { mode: 'live', drone_count: 2 }, targets: [] };
let targetEditMode = false;

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

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function postSimCommand(cmd) {
  const frame = simEmbedEl.querySelector('iframe');
  if (!frame?.contentWindow) return;
  frame.contentWindow.postMessage({ source: 'c2', ...cmd }, '*');
}

function targetMap() {
  const m = new Map();
  for (const t of (state.targets || [])) m.set(`${t.x},${t.y}`, t.color || 'red');
  return m;
}

function render() {
  if (!state.arena) return;

  modeSelect.value = state.system.mode;
  droneCountSelect.value = String(state.system.drone_count);
  targetModeBtn.textContent = targetEditMode ? 'Target mode ON (click cells)' : 'Set targets (max 6)';
  connectLiveBtn.disabled = state.system.mode !== 'live';

  const selected = droneSelect.value;
  const activeDrones = state.drones.filter(d => d.status !== 'offline');
  droneSelect.innerHTML = activeDrones.map(d => `<option value="${d.drone_id}">${d.drone_id} (${d.status})</option>`).join('');
  if (selected) droneSelect.value = selected;

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
    return `<div class="drone-icon" style="left:${left}px;top:${top}px" title="${d.drone_id}">🚁</div>`;
  }).join('');

  targetListEl.innerHTML = `<strong>Targets:</strong> ` + (state.targets?.length
    ? state.targets.map((t, i) => `#${i+1} (${t.x},${t.y}) <b style="color:${t.color === 'red' ? '#f87171' : '#60a5fa'}">${t.color}</b>`).join(' | ')
    : 'none');

  cardsEl.innerHTML = activeDrones.map(d => {
    const video = d.video_url
      ? `<img src="${d.video_url}?t=${Date.now()}" class="video" alt="${d.drone_id} video" />`
      : `<div class="video">No video URL configured</div>`;
    return `
      <div class="card">
        <strong>${d.drone_id}</strong> — ${d.status}<br/>
        Pos: (${d.position.x}, ${d.position.y}) | Home: (${d.home.x}, ${d.home.y})
        ${video}
        <div class="actions">
          <button data-target="${d.drone_id}" class="spot">Target in sight</button>
        </div>
      </div>
    `;
  }).join('');

  let simUrl = state.system.simulator_url;
  if (simUrl) {
    try {
      const u = new URL(simUrl);
      if (u.hostname === 'localhost' || u.hostname === '127.0.0.1') {
        u.hostname = window.location.hostname;
        simUrl = u.toString();
      }
    } catch {}
  }

  simEmbedEl.innerHTML = state.system.mode === 'simulator' && simUrl
    ? `<h3>Simulator (auto-connected)</h3><iframe id="simFrame" src="${simUrl}"></iframe>`
    : '';

  if (state.system.mode === 'simulator') {
    setTimeout(() => {
      postSimCommand({ kind: 'spawn', droneCount: Number(state.system.drone_count), enemyDroneCount: Number(state.system.drone_count), team: 'red', mode: 'swarm' });
      postSimCommand({ kind: 'targets', targets: state.targets || [] });
    }, 300);
  }
}

applySystemBtn.addEventListener('click', async () => {
  await api('/api/system', {
    method: 'POST',
    body: JSON.stringify({ mode: modeSelect.value, drone_count: Number(droneCountSelect.value) }),
  });
});

targetModeBtn.addEventListener('click', () => {
  targetEditMode = !targetEditMode;
  render();
});

connectLiveBtn.addEventListener('click', async () => {
  const res = await api('/api/live/connect', { method: 'POST', body: JSON.stringify({}) });
  alert('Live connect result: ' + JSON.stringify(res.result));
});

gridEl.addEventListener('click', async (e) => {
  const el = e.target.closest('.cell');
  if (!el) return;
  const x = Number(el.dataset.x);
  const y = Number(el.dataset.y);

  if (targetEditMode) {
    const targets = [...(state.targets || [])];
    const idx = targets.findIndex(t => t.x === x && t.y === y);
    if (idx >= 0) targets.splice(idx, 1);
    else {
      if (targets.length >= 6) return alert('Maximum 6 targets');
      targets.push({ x, y, color: targetColorSelect.value });
    }
    await api('/api/targets', { method: 'POST', body: JSON.stringify({ targets }) });
    if (state.system.mode === 'simulator') postSimCommand({ kind: 'targets', targets });
    return;
  }

  const droneId = droneSelect.value;
  if (!droneId) return alert('No drone selected');

  await api(`/api/drones/${droneId}/goto`, { method: 'POST', body: JSON.stringify({ x, y }) });
  if (state.system.mode === 'simulator') postSimCommand({ kind: 'goto', droneId, x, y });
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
