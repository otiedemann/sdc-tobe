// Sphinx Control — vanilla JS frontend.
// Polls /api/drones every 2s; mutating actions go through fetch + reload.

const REFRESH_MS = 2000;
let selectedDroneId = null;

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
  for (const w of worlds) {
    const opt = document.createElement("option");
    opt.value = w.name;
    opt.textContent = w.available
      ? `${w.name} — ${w.description || w.binary}`
      : `${w.name} (unavailable: ${w.unavailable_reason})`;
    if (!w.available) opt.disabled = true;
    sel.appendChild(opt);
  }
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
  $("#drones-count").textContent = drones.length;
  if (drones.length === 0) {
    tbody.innerHTML = `<tr class="empty"><td colspan="9">No drones spawned yet.</td></tr>`;
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
  } catch (e) {
    $("#system-output").textContent = `error: ${e.message}`;
  }
}

$("#spawn-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  const body = {
    drone_profile: fd.get("drone_profile"),
    world_name: fd.get("world_name"),
  };
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

$("#restart-all-btn").addEventListener("click", async () => {
  if (!confirm("Restart all running drones?")) return;
  try { await jpost("/api/drones/restart-all"); }
  catch (e) { alert(`restart-all: ${e.message}`); }
  await loadDrones();
});

$("#stop-all-btn").addEventListener("click", async () => {
  if (!confirm("Stop all running drones?")) return;
  try { await jpost("/api/drones/stop-all"); }
  catch (e) { alert(`stop-all: ${e.message}`); }
  await loadDrones();
});

(async function init() {
  await Promise.all([loadProfiles(), loadWorlds(), loadSystem()]);
  await loadDrones();
  setInterval(loadDrones, REFRESH_MS);
  setInterval(loadSystem, REFRESH_MS * 5);
})();
