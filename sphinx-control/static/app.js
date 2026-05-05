// Sphinx Control — vanilla JS frontend.
// Polls /api/drones + /api/environment every 2s; mutating actions go
// through fetch + reload. The flow is two-stage: start an environment
// (UE4-only) first, then spawn a drone (sphinx) into it.

const REFRESH_MS = 2000;
let selectedDroneId = null;
let envCache = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

async function jget(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

async function jdelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

function formatUptime(s) {
  if (s == null) return "—";
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatEndpoint(d) {
  if (d.drone_port) return `${d.drone_ip}:${d.drone_port}`;
  return d.drone_ip;
}

async function loadProfiles() {
  const profiles = await jget("/api/profiles");
  const sel = $("#profile-select");
  sel.innerHTML = "";
  for (const p of profiles) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.available
      ? `${p.name} — ${p.description || p.descriptor}`
      : `${p.name} (unavailable: ${p.unavailable_reason})`;
    if (!p.available) opt.disabled = true;
    sel.appendChild(opt);
  }
}

async function loadWorlds() {
  const worlds = await jget("/api/worlds");
  const sel = $("#world-select");
  sel.innerHTML = "";
  // Default selection: prefer "sdc_arena" if it's available; else fall
  // back to the first available world. (When running this control plane,
  // we almost always want to spawn into the SDC arena, so make that
  // the one-click choice.)
  const DEFAULT_WORLD = "sdc_arena";
  let defaultIdx = -1;
  worlds.forEach((w, i) => {
    if (w.name === DEFAULT_WORLD && w.available) defaultIdx = i;
  });
  if (defaultIdx === -1) {
    defaultIdx = worlds.findIndex(w => w.available);
  }
  worlds.forEach((w, i) => {
    const opt = document.createElement("option");
    opt.value = w.name;
    opt.textContent = w.available
      ? `${w.name} — ${w.description || w.binary}`
      : `${w.name} (unavailable: ${w.unavailable_reason})`;
    if (!w.available) opt.disabled = true;
    if (i === defaultIdx) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadDrones() {
  let drones;
  try {
    drones = await jget("/api/drones");
  } catch (e) {
    console.error("loadDrones", e);
    return;
  }
  const tbody = $("#drones-tbody");
  const countEl = $("#drones-count");
  if (countEl) countEl.textContent = drones.length;
  if (drones.length === 0) {
    tbody.innerHTML = `<tr class="empty"><td colspan="9">No environment running. Use the form above to start one.</td></tr>`;
    return;
  }
  tbody.innerHTML = "";
  for (const d of drones) {
    const tr = document.createElement("tr");
    tr.dataset.droneId = d.drone_id;
    if (d.drone_id === selectedDroneId) tr.classList.add("selected");
    tr.innerHTML = `
      <td>${d.instance_id}</td>
      <td><code>${d.drone_id}</code></td>
      <td><span class="status status-${d.status}">${d.status}</span></td>
      <td>${d.drone_type}</td>
      <td>${d.world_app}</td>
      <td class="endpoint">${formatEndpoint(d)}</td>
      <td class="pids">sphinx=${d.sphinx_pid ?? "—"} ue4=${d.ue4_pid ?? "—"}</td>
      <td>${formatUptime(d.uptime_s)}</td>
      <td class="row-actions">
        <button class="secondary" data-action="restart">Restart</button>
        <button class="secondary" data-action="stop">Stop</button>
        <button class="danger" data-action="delete">Delete</button>
      </td>
    `;
    tr.addEventListener("click", (ev) => {
      if (ev.target.tagName === "BUTTON") return;
      selectDrone(d.drone_id);
    });
    tr.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const action = btn.dataset.action;
        try {
          if (action === "restart") await jpost(`/api/drones/${d.drone_id}/restart`);
          if (action === "stop")    await jpost(`/api/drones/${d.drone_id}/stop`);
          if (action === "delete")  await jdelete(`/api/drones/${d.drone_id}`);
        } catch (e) {
          alert(`${action} failed: ${e.message}`);
        }
        await loadDrones();
      });
    });
    tbody.appendChild(tr);
  }
}

async function selectDrone(droneId) {
  selectedDroneId = droneId;
  $$("#drones-tbody tr").forEach((tr) => {
    tr.classList.toggle("selected", tr.dataset.droneId === droneId);
  });
  if (!droneId) {
    $("#connections-output").textContent = "(select a drone)";
    return;
  }
  try {
    const data = await jget(`/api/drones/${droneId}/connections`);
    $("#connections-output").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#connections-output").textContent = `error: ${e.message}`;
  }
}

async function loadSystem() {
  try {
    const data = await jget("/api/system");
    $("#system-output").textContent = JSON.stringify(data, null, 2);
    const ts = data.tailscale || {};
    const tsBadge = $("#tailscale-badge");
    if (!ts.enabled) {
      tsBadge.textContent = "tailscale: off";
    } else if (!ts.available) {
      tsBadge.textContent = `tailscale: ${ts.error || "unavailable"}`;
      tsBadge.classList.add("err");
    } else {
      const ip = (ts.self_addrs && ts.self_addrs[0]) || "?";
      tsBadge.textContent = `tailnet: ${ip}`;
      tsBadge.classList.add("ok");
    }
    const sx = data.sphinx || {};
    const sxBadge = $("#sphinx-badge");
    if (!sx.available) {
      sxBadge.textContent = "sphinx: not installed";
      sxBadge.classList.add("warn");
    } else {
      sxBadge.textContent = `sphinx: ${(sx.version || "ok").slice(0, 32)}`;
      sxBadge.classList.add("ok");
    }
    const sess = data.active_session;
    const sessBadge = $("#session-badge");
    sessBadge.classList.remove("ok", "warn", "err");
    if (data.session_attach === "off") {
      sessBadge.textContent = "session: off";
    } else if (sess) {
      const display = sess.display || "?";
      sessBadge.textContent = `session: ${sess.user}@${sess.type}:${display}`;
      sessBadge.classList.add("ok");
    } else {
      sessBadge.textContent = "session: none detected";
      sessBadge.classList.add("warn");
    }
  } catch (e) {
    $("#system-output").textContent = `error: ${e.message}`;
  }
}

async function loadFC() {
  let fc;
  try {
    fc = await jget("/api/fc");
  } catch (e) {
    console.error("loadFC", e);
    return;
  }
  const pill = $("#fc-state-pill");
  const statusSpan = pill.querySelector(".fc-status");
  pill.classList.remove("env-pill-running", "env-pill-stopped",
                        "env-pill-starting", "env-pill-error");
  const startBtn = $("#fc-start-btn");
  const stopBtn = $("#fc-stop-btn");
  const restartBtn = $("#fc-restart-btn");
  if (!fc || fc.status !== "running") {
    pill.classList.add("env-pill-stopped");
    statusSpan.textContent = fc
      ? `last: ${fc.script} (${fc.status})`
      : "no flight controller running";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    restartBtn.disabled = true;
  } else {
    pill.classList.add("env-pill-running");
    const uptime = formatUptime(fc.uptime_s);
    statusSpan.textContent = `running: ${fc.script} on :${fc.http_port} — pid=${fc.pid} uptime ${uptime}`;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    restartBtn.disabled = false;
  }
}

async function loadEnvironment() {
  let env;
  try {
    env = await jget("/api/environment");
  } catch (e) {
    console.error("loadEnvironment", e);
    return;
  }
  envCache = env;
  const pill = $("#env-state-pill");
  const statusSpan = pill.querySelector(".env-status");
  pill.classList.remove("env-pill-running", "env-pill-stopped",
                        "env-pill-starting", "env-pill-error");
  const startBtn = $("#env-start-btn");
  const stopBtn = $("#env-stop-btn");
  const restartBtn = $("#env-restart-btn");
  const droneBtn = $("#drone-spawn-btn");
  if (!env || env.status !== "running") {
    pill.classList.add("env-pill-stopped");
    statusSpan.textContent = env
      ? `last: ${env.world_name} (${env.status})`
      : "no environment running";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    restartBtn.disabled = true;
    droneBtn.disabled = true;
  } else {
    pill.classList.add("env-pill-running");
    const uptime = formatUptime(env.uptime_s);
    statusSpan.textContent = `running: ${env.world_name} — ue4_pid=${env.ue4_pid} uptime ${uptime}`;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    restartBtn.disabled = false;
    droneBtn.disabled = false;
  }
}

// Helper: while an async operation is in flight, disable the relevant
// FC buttons and show a "working..." label on the active one. Without
// this, the FC stop/restart looks completely unresponsive for the
// 3-5 seconds it takes the kill+port-release dance to complete, and
// users assume the click was lost (or click twice, racing the API).
function fcButtonsLock(activeBtn, label) {
  const btns = ["#fc-start-btn", "#fc-stop-btn", "#fc-restart-btn"];
  const orig = {};
  for (const sel of btns) {
    const b = $(sel);
    if (!b) continue;
    orig[sel] = b.textContent;
    b.disabled = true;
    if (sel === activeBtn) b.textContent = label;
  }
  return () => {
    for (const sel of btns) {
      const b = $(sel);
      if (!b) continue;
      b.disabled = false;
      b.textContent = orig[sel];
    }
  };
}

$("#fc-start-btn").addEventListener("click", async () => {
  const unlock = fcButtonsLock("#fc-start-btn", "starting…");
  try { await jpost("/api/fc", {}); }
  catch (e) { unlock(); alert(`fc start failed: ${e.message}`); return; }
  unlock();
  await loadFC();
});

$("#fc-stop-btn").addEventListener("click", async () => {
  if (!confirm("Stop the flight controller?")) return;
  const unlock = fcButtonsLock("#fc-stop-btn", "stopping…");
  try { await jdelete("/api/fc"); }
  catch (e) { unlock(); alert(`fc stop failed: ${e.message}`); return; }
  unlock();
  await loadFC();
});

$("#fc-restart-btn").addEventListener("click", async () => {
  if (!confirm("Restart the flight controller?")) return;
  const unlock = fcButtonsLock("#fc-restart-btn", "restarting…");
  try { await jpost("/api/fc/restart"); }
  catch (e) { unlock(); alert(`fc restart failed: ${e.message}`); return; }
  unlock();
  await loadFC();
});

$("#env-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  const body = { world_name: fd.get("world_name") };
  // Optional UE window controls — only include if the user filled
  // them in / changed defaults. Sending null is fine, the API treats
  // missing fields as "use launcher defaults".
  const rx = parseInt(fd.get("res_x"), 10);
  const ry = parseInt(fd.get("res_y"), 10);
  if (!Number.isNaN(rx)) body.res_x = rx;
  if (!Number.isNaN(ry)) body.res_y = ry;
  body.hide_panels = $("#env-hide-panels").checked;
  try {
    await jpost("/api/environment", body);
  } catch (e) {
    alert(`environment start failed: ${e.message}`);
    return;
  }
  await loadEnvironment();
  await loadDrones();
});

$("#env-stop-btn").addEventListener("click", async () => {
  if (!confirm("Stop the running environment? Any attached drone will also stop.")) return;
  try { await jdelete("/api/environment"); }
  catch (e) { alert(`stop failed: ${e.message}`); return; }
  await loadEnvironment();
  await loadDrones();
});

$("#env-restart-btn").addEventListener("click", async () => {
  if (!confirm("Restart the environment? The drone will be stopped.")) return;
  try { await jpost("/api/environment/restart"); }
  catch (e) { alert(`restart failed: ${e.message}`); return; }
  await loadEnvironment();
  await loadDrones();
});

$("#spawn-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!envCache || envCache.status !== "running") {
    alert("Start the environment first (Step 1 above).");
    return;
  }
  const fd = new FormData(ev.target);
  const body = { drone_profile: fd.get("drone_profile") };
  const iid = fd.get("instance_id");
  if (iid) body.instance_id = parseInt(iid, 10);
  try {
    await jpost("/api/drones", body);
  } catch (e) {
    alert(`spawn failed: ${e.message}`);
    return;
  }
  ev.target.reset();
  await loadDrones();
});

// Bulk controls only exist when max_drones > 1; guard the listeners.
const restartAllBtn = $("#restart-all-btn");
if (restartAllBtn) restartAllBtn.addEventListener("click", async () => {
  if (!confirm("Restart all running drones?")) return;
  try { await jpost("/api/drones/restart-all"); }
  catch (e) { alert(`restart-all: ${e.message}`); }
  await loadDrones();
});

const stopAllBtn = $("#stop-all-btn");
if (stopAllBtn) stopAllBtn.addEventListener("click", async () => {
  if (!confirm("Stop all running drones?")) return;
  try { await jpost("/api/drones/stop-all"); }
  catch (e) { alert(`stop-all: ${e.message}`); }
  await loadDrones();
});

(async function init() {
  await Promise.all([loadProfiles(), loadWorlds(), loadSystem()]);
  await loadEnvironment();
  await loadDrones();
  await loadFC();
  setInterval(loadEnvironment, REFRESH_MS);
  setInterval(loadDrones, REFRESH_MS);
  setInterval(loadFC, REFRESH_MS);
  setInterval(loadSystem, REFRESH_MS * 5);
})();
