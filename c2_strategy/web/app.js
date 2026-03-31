// SDC26 Strategy C2 Dashboard
// Connects via WebSocket to strategy_server.py on port 9090

let ws = null;
let state = {};
let startTime = null;

// ---- WebSocket ----

function connect() {
    const host = location.host || 'localhost:9090';
    ws = new WebSocket(`ws://${host}/ws`);

    ws.onopen = () => {
        const el = document.getElementById('connection-status');
        el.textContent = 'Connected';
        el.className = 'status-disconnected status-connected';
    };

    ws.onclose = () => {
        const el = document.getElementById('connection-status');
        el.textContent = 'Disconnected';
        el.className = 'status-disconnected';
        setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.cmd_result) return; // command response, ignore
            if (msg.error) { console.warn('WS error:', msg.error); return; }
            state = msg;
            render();
        } catch (err) {
            console.error('WS parse error:', err);
        }
    };
}

function sendCmd(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
    }
}

// ---- Rendering ----

function render() {
    renderScoreboard();
    renderDroneCards();
    renderTargets();
    renderArena();
    renderLog();
}

function renderScoreboard() {
    const team = (state.our_team || 'blue').toUpperCase();
    document.getElementById('our-team-label').textContent = team;
    document.getElementById('score-us').textContent = state.score_us || 0;
    document.getElementById('score-them').textContent = state.score_them || 0;

    const phaseBadge = document.getElementById('game-phase');
    const phase = state.phase || 'setup';
    phaseBadge.textContent = phase.replace(/_/g, ' ');
    phaseBadge.className = 'phase-badge';
    if (['simultaneous_attack', 'instant_win_attempt'].includes(phase)) phaseBadge.classList.add('attack');
    else if (phase === 'game_over') phaseBadge.classList.add('danger');
    else if (phase !== 'setup') phaseBadge.classList.add('active');

    document.getElementById('phase-select').value = phase;
    document.getElementById('team-select').value = state.our_team || 'blue';
}

function renderDroneCards() {
    const container = document.getElementById('drone-cards');
    const drones = state.drones || {};
    let html = '';

    for (const [id, d] of Object.entries(drones)) {
        const statusClass = d.flying ? 'flying' : d.connected ? 'connected' : d.phase === 'lost' ? 'lost' : 'idle';
        const statusText = d.flying ? 'Flying' : d.connected ? 'Ready' : d.phase === 'lost' ? 'Lost' : 'Offline';
        const bat = Math.round(d.battery_pct || 0);
        const batClass = bat > 50 ? 'high' : bat > 20 ? 'medium' : 'low';
        const pos = d.position ? `(${d.position.x.toFixed(1)}, ${d.position.y.toFixed(1)})` : '--';

        html += `
        <div class="drone-card">
            <div class="drone-card-header">
                <span class="drone-name">${id}</span>
                <span class="drone-status ${statusClass}">${statusText}</span>
            </div>
            <div class="drone-info">
                <span>Role: ${d.role || '--'}</span>
                <span>Phase: ${d.phase || '--'}</span>
                <span>Pos: ${pos}</span>
                <span>Alt: ${(d.altitude_m || 0).toFixed(1)}m</span>
                <span>Hdg: ${Math.round(d.heading_deg || 0)}°</span>
                <span>Target: ${d.assigned_target || '--'}</span>
            </div>
            <div class="battery-bar">
                <div class="battery-fill ${batClass}" style="width:${bat}%"></div>
            </div>
            <div class="drone-actions">
                <button class="btn btn-green" onclick="sendCmd({cmd:'takeoff',drone_id:'${id}'})">Takeoff</button>
                <button class="btn btn-orange" onclick="sendCmd({cmd:'land',drone_id:'${id}'})">Land</button>
                <button class="btn btn-red" onclick="sendCmd({cmd:'emergency',drone_id:'${id}'})">E-Stop</button>
            </div>
        </div>`;
    }

    container.innerHTML = html || '<p style="color:var(--text-dim)">No drones configured</p>';
}

function renderTargets() {
    const container = document.getElementById('target-list');
    const targets = state.targets || {};
    let html = '';

    for (const [id, t] of Object.entries(targets)) {
        const owner = t.captured_by || t.owner || 'neutral';
        const captured = t.captured ? ' captured' : '';
        const pos = t.position ? `(${t.position.x.toFixed(1)}, ${t.position.y.toFixed(1)})` : '';
        html += `
        <div class="target-chip${captured}">
            <span class="dot ${owner}"></span>
            <span>${id}</span>
            <span style="color:var(--text-dim);font-size:0.8em">${pos}</span>
        </div>`;
    }

    container.innerHTML = html || '<p style="color:var(--text-dim)">No targets discovered</p>';
}

function renderLog() {
    const container = document.getElementById('captures-log');
    const captures = state.captures || [];
    let html = '';
    for (const c of captures.slice().reverse()) {
        html += `<div class="log-entry">${c.drone_id} captured box ${c.box_id} (+${c.points}pt)</div>`;
    }
    container.innerHTML = html || '<div class="log-entry" style="color:var(--text-dim)">No captures yet</div>';
}

// ---- Arena Canvas ----

function renderArena() {
    const canvas = document.getElementById('arena-canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Arena dimensions from config (default 20x10)
    const arenaW = 20;
    const arenaH = 10;
    const scale = Math.min(W / arenaW, H / arenaH);
    const ox = (W - arenaW * scale) / 2;
    const oy = (H - arenaH * scale) / 2;

    function toScreen(ax, ay) {
        return [ox + ax * scale, oy + (arenaH - ay) * scale];
    }

    // Clear
    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(48,54,61,0.5)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= arenaW; x++) {
        const [sx] = toScreen(x, 0);
        ctx.beginPath(); ctx.moveTo(sx, oy); ctx.lineTo(sx, oy + arenaH * scale); ctx.stroke();
    }
    for (let y = 0; y <= arenaH; y++) {
        const [, sy] = toScreen(0, y);
        ctx.beginPath(); ctx.moveTo(ox, sy); ctx.lineTo(ox + arenaW * scale, sy); ctx.stroke();
    }

    // Home zones
    const ourTeam = state.our_team || 'blue';

    // Red home: 0-7, Blue home: 13-20
    ctx.fillStyle = 'rgba(248,81,73,0.08)';
    const [rx1, ry1] = toScreen(0, 10);
    ctx.fillRect(rx1, ry1, 7 * scale, 10 * scale);

    ctx.fillStyle = 'rgba(88,166,255,0.08)';
    const [bx1, by1] = toScreen(13, 10);
    ctx.fillRect(bx1, by1, 7 * scale, 10 * scale);

    // Labels
    ctx.fillStyle = 'rgba(248,81,73,0.3)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    const [rlx, rly] = toScreen(3.5, 5);
    ctx.fillText('RED HOME', rlx, rly);

    ctx.fillStyle = 'rgba(88,166,255,0.3)';
    const [blx, bly] = toScreen(16.5, 5);
    ctx.fillText('BLUE HOME', blx, bly);

    // Boundary
    ctx.strokeStyle = 'rgba(139,148,158,0.4)';
    ctx.lineWidth = 2;
    ctx.strokeRect(ox, oy, arenaW * scale, arenaH * scale);

    // Center line
    ctx.strokeStyle = 'rgba(139,148,158,0.2)';
    ctx.setLineDash([5, 5]);
    const [cx, cy1] = toScreen(10, 0);
    const [, cy2] = toScreen(10, 10);
    ctx.beginPath(); ctx.moveTo(cx, cy1); ctx.lineTo(cx, cy2); ctx.stroke();
    ctx.setLineDash([]);

    // ArUco markers (diamonds)
    const showMarkers = document.getElementById('show-markers-toggle')?.checked !== false;
    const markers = state.pillar_markers || {};
    if (showMarkers) {
        for (const [mid, m] of Object.entries(markers)) {
            const p = m.position;
            if (!p) continue;
            const [sx, sy] = toScreen(p.x, p.y);
            const s = 5;

            // Wall-based color
            const wallColors = {
                front: 'rgba(210,153,34,0.7)',
                back: 'rgba(210,153,34,0.7)',
                left: 'rgba(188,140,255,0.7)',
                right: 'rgba(188,140,255,0.7)',
                pillar: 'rgba(139,148,158,0.7)',
            };
            ctx.fillStyle = wallColors[m.wall] || 'rgba(139,148,158,0.5)';

            // Diamond shape
            ctx.beginPath();
            ctx.moveTo(sx, sy - s);
            ctx.lineTo(sx + s, sy);
            ctx.lineTo(sx, sy + s);
            ctx.lineTo(sx - s, sy);
            ctx.closePath();
            ctx.fill();

            // ID label
            ctx.fillStyle = 'rgba(230,237,243,0.5)';
            ctx.font = '8px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(mid, sx, sy - s - 2);
        }
    }

    // Target boxes
    const targets = state.targets || {};
    for (const [id, t] of Object.entries(targets)) {
        if (!t.position) continue;
        const [sx, sy] = toScreen(t.position.x, t.position.y);
        const owner = t.captured_by || t.owner || 'neutral';

        // Box
        ctx.fillStyle = owner === 'red' ? 'rgba(248,81,73,0.7)' :
                        owner === 'blue' ? 'rgba(88,166,255,0.7)' :
                        'rgba(139,148,158,0.5)';
        ctx.fillRect(sx - 6, sy - 6, 12, 12);

        // Label
        ctx.fillStyle = '#e6edf3';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(id, sx, sy - 10);
    }

    // Drones
    const drones = state.drones || {};
    const droneColors = ['#58a6ff', '#3fb950', '#bc8cff', '#d29922', '#f778ba'];
    let di = 0;

    for (const [id, d] of Object.entries(drones)) {
        if (!d.position) { di++; continue; }
        const [sx, sy] = toScreen(d.position.x, d.position.y);
        const color = droneColors[di % droneColors.length];

        // Drone triangle (pointing in heading direction)
        const rad = (d.heading_deg || 0) * Math.PI / 180;
        const size = 8;
        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(-rad);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(0, -size);
        ctx.lineTo(-size * 0.6, size * 0.5);
        ctx.lineTo(size * 0.6, size * 0.5);
        ctx.closePath();
        ctx.fill();

        ctx.restore();

        // Label
        ctx.fillStyle = color;
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(id, sx, sy + 16);

        // Line to target
        if (d.assigned_target && targets[d.assigned_target]?.position) {
            const tp = targets[d.assigned_target].position;
            const [tx, ty] = toScreen(tp.x, tp.y);
            ctx.strokeStyle = color + '44';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(tx, ty); ctx.stroke();
            ctx.setLineDash([]);
        }

        di++;
    }

    // Locator info (ArUco positions overlay)
    const loc = state.locator;
    if (loc && loc.drones) {
        ctx.font = '8px sans-serif';
        for (const [src, info] of Object.entries(loc.drones)) {
            if (info.stale) continue;
            // Only show if not already a drone (ArUco sources)
            if (drones[src]) continue;
            const [sx, sy] = toScreen(info.x, info.y);
            ctx.fillStyle = 'rgba(188,140,255,0.5)';
            ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#bc8cff';
            ctx.textAlign = 'center';
            ctx.fillText(src, sx, sy - 6);
        }
    }
}

// ---- Actions ----

function setPhase() {
    const phase = document.getElementById('phase-select').value;
    sendCmd({ cmd: 'set_phase', phase });
}

function setTeam(team) {
    fetch('/api/config/team', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ team }),
    });
}

function addTarget() {
    const box_id = document.getElementById('new-target-id').value.trim();
    const x = parseFloat(document.getElementById('new-target-x').value);
    const y = parseFloat(document.getElementById('new-target-y').value);
    const owner = document.getElementById('new-target-owner').value;
    if (!box_id) return;

    fetch('/api/targets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ box_id, marker_id: parseInt(box_id) || 0, x, y, owner }),
    }).then(() => {
        document.getElementById('new-target-id').value = '';
    });
}

// ---- ArUco Markers Config ----

let markerConfigs = {}; // { "21": {marker_id, position:{x,y,z}, wall}, ... }

function renderMarkerList() {
    const container = document.getElementById('marker-list');
    const ids = Object.keys(markerConfigs).sort((a, b) => Number(a) - Number(b));
    if (ids.length === 0) {
        container.innerHTML = '<p style="color:var(--text-dim)">No markers configured</p>';
        return;
    }
    let html = '';
    for (const mid of ids) {
        const m = markerConfigs[mid];
        const p = m.position || {};
        html += `
        <div class="marker-row">
            <span class="marker-id-label">#${m.marker_id}</span>
            <label>X:</label><input type="number" step="0.1" value="${p.x ?? 0}" onchange="markerConfigs['${mid}'].position.x=parseFloat(this.value)" />
            <label>Y:</label><input type="number" step="0.1" value="${p.y ?? 0}" onchange="markerConfigs['${mid}'].position.y=parseFloat(this.value)" />
            <label>Z:</label><input type="number" step="0.1" value="${p.z ?? 0}" onchange="markerConfigs['${mid}'].position.z=parseFloat(this.value)" />
            <select onchange="markerConfigs['${mid}'].wall=this.value">
                <option value="front" ${m.wall==='front'?'selected':''}>Front</option>
                <option value="back" ${m.wall==='back'?'selected':''}>Back</option>
                <option value="left" ${m.wall==='left'?'selected':''}>Left</option>
                <option value="right" ${m.wall==='right'?'selected':''}>Right</option>
                <option value="pillar" ${m.wall==='pillar'?'selected':''}>Pillar</option>
            </select>
            <button class="btn btn-red" onclick="delete markerConfigs['${mid}'];renderMarkerList()">X</button>
        </div>`;
    }
    container.innerHTML = html;
}

function addMarker() {
    const id = document.getElementById('new-marker-id').value.trim();
    if (!id) return;
    const x = parseFloat(document.getElementById('new-marker-x').value) || 0;
    const y = parseFloat(document.getElementById('new-marker-y').value) || 0;
    const z = parseFloat(document.getElementById('new-marker-z').value) || 2.0;
    const wall = document.getElementById('new-marker-wall').value;
    markerConfigs[id] = {
        marker_id: parseInt(id),
        position: { x, y, z },
        wall,
    };
    document.getElementById('new-marker-id').value = '';
    renderMarkerList();
}

function saveMarkers() {
    fetch('/api/config/markers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(markerConfigs),
    }).then(r => r.json()).then(data => {
        if (data.ok) alert(`Saved ${data.marker_count} markers`);
    });
}

function loadDefaultMarkers() {
    if (!confirm('Reset markers to default positions?')) return;
    // Fetch defaults from server by resetting config
    fetch('/api/config/markers/defaults').then(r => r.json()).then(data => {
        markerConfigs = data;
        renderMarkerList();
        saveMarkers();
    });
}

// ---- Drone Config ----

let droneConfigs = [];

function renderDroneConfigs() {
    const container = document.getElementById('drone-config-list');
    let html = '';
    droneConfigs.forEach((dc, i) => {
        html += `
        <div class="drone-config-row">
            <input value="${dc.drone_id}" onchange="droneConfigs[${i}].drone_id=this.value" placeholder="ID" />
            <input value="${dc.api_base_url}" onchange="droneConfigs[${i}].api_base_url=this.value" placeholder="API URL" style="width:160px"/>
            <select onchange="droneConfigs[${i}].drone_type=this.value">
                <option value="simulator" ${dc.drone_type==='simulator'?'selected':''}>Simulator</option>
                <option value="tello" ${dc.drone_type==='tello'?'selected':''}>Tello</option>
                <option value="anafi" ${dc.drone_type==='anafi'?'selected':''}>Anafi</option>
            </select>
            <input value="${dc.sim_drone_id || ''}" onchange="droneConfigs[${i}].sim_drone_id=this.value||null" placeholder="Sim ID" style="width:60px"/>
            <button class="btn btn-red" onclick="droneConfigs.splice(${i},1);renderDroneConfigs()">X</button>
        </div>`;
    });
    container.innerHTML = html;
}

function addDroneConfig() {
    const n = droneConfigs.length + 1;
    droneConfigs.push({
        drone_id: `drone-${n}`,
        api_base_url: 'http://localhost:8080',
        drone_type: 'simulator',
        sim_drone_id: `B${n}`,
    });
    renderDroneConfigs();
}

function saveDroneConfig() {
    fetch('/api/config/drones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(droneConfigs),
    }).then(r => r.json()).then(data => {
        if (data.ok) alert(`Fleet updated: ${data.drone_count} drones`);
    });
}

// ---- Virtual Camera Feed ----

let cameraConnected = false;
let cameraInterval = null;

function toggleCameraFeed() {
    if (cameraConnected) {
        disconnectCamera();
    } else {
        connectCamera();
    }
}

function connectCamera() {
    const baseUrl = document.getElementById('cam-api-url').value.trim();
    const usePolling = document.getElementById('cam-poll-mode').checked;
    const img = document.getElementById('camera-feed');
    const statusEl = document.getElementById('camera-status');
    const overlayEl = document.getElementById('camera-overlay');
    const btn = document.getElementById('cam-connect-btn');

    if (usePolling) {
        // Polling mode: fetch individual frames + ArUco status
        let frameCount = 0;
        cameraInterval = setInterval(() => {
            // Refresh frame by changing src with cache-bust
            img.src = `${baseUrl}/api/aruco/frame?t=${Date.now()}`;
            frameCount++;

            // Also fetch ArUco status for overlay data
            fetch(`${baseUrl}/api/aruco`)
                .then(r => r.json())
                .then(data => {
                    overlayEl.style.display = 'block';
                    document.getElementById('cam-markers').textContent = data.markers_detected || 0;
                    document.getElementById('cam-refs').textContent =
                        (data.ref_markers && data.ref_markers.length > 0) ? data.ref_markers.join(', ') : '-';
                    const pos = data.position;
                    if (pos && pos.aruco_x !== undefined) {
                        document.getElementById('cam-arena-pos').textContent =
                            `(${pos.aruco_x.toFixed(1)}, ${pos.aruco_y.toFixed(1)})`;
                        document.getElementById('cam-arena-pos').style.color = '#50fa7b';
                    } else {
                        document.getElementById('cam-arena-pos').textContent = '-';
                        document.getElementById('cam-arena-pos').style.color = '#ff5555';
                    }
                })
                .catch(() => {});
        }, 200);  // 5fps polling
    } else {
        // MJPEG stream mode
        img.src = `${baseUrl}/api/aruco/stream`;
        overlayEl.style.display = 'block';
    }

    cameraConnected = true;
    statusEl.textContent = 'Connected';
    statusEl.style.background = 'rgba(80,250,123,0.2)';
    statusEl.style.color = '#50fa7b';
    btn.textContent = 'Disconnect';
    btn.classList.add('btn-red');
}

function disconnectCamera() {
    const img = document.getElementById('camera-feed');
    const statusEl = document.getElementById('camera-status');
    const overlayEl = document.getElementById('camera-overlay');
    const btn = document.getElementById('cam-connect-btn');

    if (cameraInterval) {
        clearInterval(cameraInterval);
        cameraInterval = null;
    }
    img.src = '';
    cameraConnected = false;
    overlayEl.style.display = 'none';
    statusEl.textContent = 'Disconnected';
    statusEl.style.background = 'var(--bg-tertiary)';
    statusEl.style.color = 'var(--text-dim)';
    btn.textContent = 'Connect';
    btn.classList.remove('btn-red');
}

// ---- Init ----

// Load initial config
fetch('/api/config').then(r => r.json()).then(cfg => {
    if (cfg.drone_configs) {
        droneConfigs = cfg.drone_configs;
        renderDroneConfigs();
    }
    if (cfg.pillar_markers) {
        markerConfigs = cfg.pillar_markers;
        renderMarkerList();
    }
}).catch(() => {});

connect();

// Resize canvas on window resize
function resizeCanvas() {
    const canvas = document.getElementById('arena-canvas');
    const container = document.getElementById('arena-container');
    if (container) {
        canvas.width = container.clientWidth;
        canvas.height = Math.round(container.clientWidth / 2);
        renderArena();
    }
}
window.addEventListener('resize', resizeCanvas);
setTimeout(resizeCanvas, 100);
