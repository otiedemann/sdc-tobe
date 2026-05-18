"""Flask web UI + JSON API for the strategy layer.

Endpoints
---------
GET  /                                — HTML dashboard
GET  /api/state                       — combined snapshot (settings + runner
                                        state + markers + recent events)
GET  /api/settings                    — current settings JSON
POST /api/settings/drone/<fc>         — patch a drone (team/role/altitudes)
POST /api/settings/match              — patch match-level fields
POST /api/settings/markers            — patch live-marker IDs
POST /api/strategy/arm                — arm the runner (allow pushes)
POST /api/strategy/disarm             — disarm
POST /api/strategy/target/<fc>        — body {marker_id: int|null} → set/clear
POST /api/strategy/emergency-land     — emergency-land everyone
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from .markers import MarkerTracker
from .roles import all_roles
from .runner import SwarmRunner
from .settings import VALID_ROLES, VALID_TEAMS, SettingsStore

logger = logging.getLogger(__name__)


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def build_app(
    *,
    settings: SettingsStore,
    runner: SwarmRunner,
    markers: MarkerTracker,
    loop: asyncio.AbstractEventLoop,
) -> Flask:
    app = Flask(__name__)

    def _run_coro(coro):
        """Submit a coroutine to the runner's event loop from a Flask thread."""
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return _no_cache(app.response_class(_INDEX_HTML, mimetype="text/html"))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @app.route("/api/state")
    def api_state():
        return _no_cache(jsonify({
            "settings": settings.to_dict(),
            "runner": runner.snapshot(),
            "markers": markers.to_dict(),
            "events": runner.events.to_list()[-50:],
            "roles": sorted(all_roles().keys()),
            "valid_teams": list(VALID_TEAMS),
            "valid_roles": list(VALID_ROLES),
        }))

    @app.route("/api/settings", methods=["GET"])
    def api_settings():
        return _no_cache(jsonify(settings.to_dict()))

    @app.route("/api/settings/drone/<fc_name>", methods=["POST"])
    def api_settings_drone(fc_name: str):
        body = request.get_json(silent=True) or {}
        patched = settings.update_drone(fc_name, **body)
        runner.events.add("settings",
                          f"drone updated: {body}",
                          drone=fc_name)
        return _no_cache(jsonify({"ok": True, "drone": _drone_dict(patched)}))

    @app.route("/api/settings/match", methods=["POST"])
    def api_settings_match():
        body = request.get_json(silent=True) or {}
        settings.update_match(**body)
        runner.events.add("settings", f"match updated: {list(body.keys())}")
        return _no_cache(jsonify({"ok": True, "match": settings.to_dict()["match"]}))

    @app.route("/api/settings/markers", methods=["POST"])
    def api_settings_markers():
        body = request.get_json(silent=True) or {}
        settings.update_markers(**body)
        runner.events.add("settings", f"markers updated: {list(body.keys())}")
        return _no_cache(jsonify({"ok": True, "markers": settings.to_dict()["markers"]}))

    # ------------------------------------------------------------------
    # Strategy control
    # ------------------------------------------------------------------

    @app.route("/api/strategy/arm", methods=["POST"])
    def api_arm():
        runner.arm("web")
        return _no_cache(jsonify({"ok": True, "armed": runner.is_armed()}))

    @app.route("/api/strategy/disarm", methods=["POST"])
    def api_disarm():
        runner.disarm("web")
        return _no_cache(jsonify({"ok": True, "armed": runner.is_armed()}))

    @app.route("/api/strategy/target/<fc_name>", methods=["POST"])
    def api_target(fc_name: str):
        body = request.get_json(silent=True) or {}
        raw = body.get("marker_id")
        marker_id: Optional[int]
        if raw is None or raw == "" or raw == "null":
            marker_id = None
        else:
            try:
                marker_id = int(raw)
            except (TypeError, ValueError):
                return _no_cache(jsonify({"ok": False, "error": "invalid marker_id"})), 400
        runner.assign_target(fc_name, marker_id)
        return _no_cache(jsonify({"ok": True, "target": marker_id}))

    @app.route("/api/strategy/emergency-land", methods=["POST"])
    def api_emergency():
        from .c2_client import C2Client  # local import to avoid cycle
        # We don't have a direct handle here, but the runner uses one; talk
        # via the runner's loop-bound coroutine call so we share the client.
        # Simpler: open a one-shot client.
        # (Strategy server lives next to C2 on localhost.)
        async def _land():
            c = C2Client()
            try:
                await c.emergency_land_all()
            finally:
                await c.aclose()

        _run_coro(_land())
        runner.disarm("emergency-land")
        runner.events.add("emergency", "emergency-land triggered from UI")
        return _no_cache(jsonify({"ok": True}))

    return app


def _drone_dict(d) -> Dict[str, Any]:
    return {
        "fc_name": d.fc_name,
        "team": d.team,
        "role": d.role,
        "enabled": d.enabled,
        "scout_alt_m": d.scout_alt_m,
        "attack_alt_m": d.attack_alt_m,
        "home_alt_m": d.home_alt_m,
    }


# ---------------------------------------------------------------------------
# HTML (single-page, vanilla JS, no build step)
# ---------------------------------------------------------------------------


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SDC26 Strategy</title>
<style>
  :root {
    --bg: #0e0f12;
    --panel: #1a1c20;
    --panel2: #23262c;
    --text: #e6e6e6;
    --muted: #9aa0a6;
    --red: #c0392b;
    --blue: #2475c2;
    --green: #2ecc71;
    --yellow: #f1c40f;
    --border: #2c3038;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.4 system-ui, sans-serif; }
  header { padding: 12px 16px; background: var(--panel);
           border-bottom: 1px solid var(--border);
           display: flex; gap: 12px; align-items: center; }
  header h1 { font-size: 16px; margin: 0; }
  header .spacer { flex: 1; }
  header .pill { padding: 4px 10px; border-radius: 999px;
                 background: var(--panel2); color: var(--muted);
                 font-size: 12px; }
  header .pill.armed { background: var(--green); color: #000; }
  header .pill.disarmed { background: #444; }
  button { background: var(--panel2); color: var(--text);
           border: 1px solid var(--border);
           padding: 6px 12px; border-radius: 6px; cursor: pointer; }
  button:hover { background: #2c3038; }
  button.primary { background: var(--green); color: #000; border-color: transparent; }
  button.danger { background: var(--red); color: #fff; border-color: transparent; }
  button.warn { background: var(--yellow); color: #000; border-color: transparent; }
  main { display: grid; grid-template-columns: 2fr 1fr;
         gap: 16px; padding: 16px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  section { background: var(--panel); border: 1px solid var(--border);
            border-radius: 8px; padding: 12px; }
  section h2 { margin: 0 0 8px; font-size: 14px;
                text-transform: uppercase; color: var(--muted);
                letter-spacing: 0.05em; }
  .grid { display: grid; gap: 10px; }
  .drone { background: var(--panel2); border: 1px solid var(--border);
           border-radius: 6px; padding: 10px; }
  .drone .head { display: flex; align-items: center; gap: 8px;
                 margin-bottom: 8px; }
  .drone .head .name { font-weight: 600; flex: 1; }
  .drone .badge { padding: 2px 8px; border-radius: 999px;
                  font-size: 11px; background: #2c3038; color: var(--muted); }
  .drone .badge.red { background: var(--red); color: #fff; }
  .drone .badge.blue { background: var(--blue); color: #fff; }
  .drone .badge.green { background: var(--green); color: #000; }
  .drone .row { display: grid; grid-template-columns: 1fr 1fr 1fr;
                gap: 8px; margin-top: 6px; }
  .drone label { display: block; font-size: 11px; color: var(--muted);
                 margin-bottom: 2px; }
  .drone input[type=number], .drone select {
        width: 100%; padding: 4px 6px; background: #16181d;
        color: var(--text); border: 1px solid var(--border);
        border-radius: 4px; }
  .drone .phase { font-size: 12px; color: var(--muted);
                  font-family: ui-monospace, monospace; }
  .marker-grid { display: grid;
                 grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
                 gap: 8px; }
  .marker { padding: 8px; border-radius: 6px; background: var(--panel2);
            text-align: center; font-family: ui-monospace, monospace;
            border: 1px solid var(--border); }
  .marker.red { border-color: var(--red); }
  .marker.blue { border-color: var(--blue); }
  .marker.captured { background: #1a3a25; border-color: var(--green); }
  .marker .id { font-size: 16px; font-weight: 600; }
  .marker .sub { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .events { max-height: 320px; overflow-y: auto;
            font-family: ui-monospace, monospace; font-size: 12px; }
  .events .ev { padding: 3px 0; border-bottom: 1px solid var(--border);
                color: var(--muted); }
  .events .ev .k { display: inline-block; min-width: 72px;
                   color: var(--text); }
  .events .ev.script_push .k { color: var(--green); }
  .events .ev.disarmed_skip .k { color: var(--yellow); }
  .events .ev.role .k, .events .ev.target .k { color: var(--yellow); }
  .events .ev.emergency .k, .events .ev.stop .k { color: var(--red); }
  .small { font-size: 11px; color: var(--muted); }
  .target-row { display: flex; gap: 4px; margin-top: 6px; align-items: center; }
  .target-row input { flex: 1; padding: 4px 6px; background: #16181d;
                       color: var(--text); border: 1px solid var(--border);
                       border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>SDC26 Strategy</h1>
  <span id="armed-pill" class="pill disarmed">disarmed</span>
  <span class="pill">tick <span id="tick">–</span></span>
  <span class="spacer"></span>
  <button id="arm-btn" class="primary">Arm</button>
  <button id="disarm-btn">Disarm</button>
  <button id="land-btn" class="danger">Emergency Land</button>
</header>

<main>
  <section>
    <h2>Drones</h2>
    <div id="drones" class="grid"></div>
  </section>

  <section>
    <h2>Markers</h2>
    <div id="markers" class="marker-grid"></div>
    <h2 style="margin-top:16px">Events</h2>
    <div id="events" class="events"></div>
  </section>
</main>

<script>
async function api(path, opts) {
  const r = await fetch(path, Object.assign({headers: {"Content-Type":"application/json"}}, opts||{}));
  return r.json();
}
async function patchDrone(fc, patch) {
  return api(`/api/settings/drone/${encodeURIComponent(fc)}`, {
    method:"POST", body: JSON.stringify(patch)});
}
async function assignTarget(fc, mid) {
  return api(`/api/strategy/target/${encodeURIComponent(fc)}`, {
    method:"POST", body: JSON.stringify({marker_id: mid})});
}

function renderDrones(state) {
  const root = document.getElementById("drones");
  const drones = (state.settings.drones || {});
  const role_states = (state.runner && state.runner.drones) || {};
  const names = Object.keys(drones).sort();
  if (!names.length) {
    root.innerHTML = '<div class="small">No drones configured. '
      + 'Edit <code>settings.json</code> or wait for C2 sync.</div>';
    return;
  }
  root.innerHTML = "";
  for (const fc of names) {
    const d = drones[fc];
    const rs = role_states[fc] || {};
    const team = d.team || "";
    const role = d.role || "idle";
    const teamBadge = team === "red" ? "red" : team === "blue" ? "blue" : "";
    const roleBadge = role === "idle" ? "" : "green";
    const targetVal = rs.target_marker_id == null ? "" : rs.target_marker_id;
    const phase = rs.phase || "idle";
    const reason = rs.last_decision_reason || "";
    const html = `
      <div class="drone">
        <div class="head">
          <span class="name">${fc}</span>
          <span class="badge ${teamBadge}">${team || "no team"}</span>
          <span class="badge ${roleBadge}">${role}</span>
        </div>
        <div class="row">
          <div>
            <label>Team</label>
            <select data-fc="${fc}" data-field="team">
              <option value="" ${team===""?"selected":""}>—</option>
              <option value="red" ${team==="red"?"selected":""}>red</option>
              <option value="blue" ${team==="blue"?"selected":""}>blue</option>
            </select>
          </div>
          <div>
            <label>Role</label>
            <select data-fc="${fc}" data-field="role">
              <option value="idle" ${role==="idle"?"selected":""}>idle</option>
              <option value="scout" ${role==="scout"?"selected":""}>scout</option>
              <option value="attacker" ${role==="attacker"?"selected":""}>attacker</option>
            </select>
          </div>
          <div>
            <label>Enabled</label>
            <select data-fc="${fc}" data-field="enabled">
              <option value="true" ${d.enabled?"selected":""}>yes</option>
              <option value="false" ${!d.enabled?"selected":""}>no</option>
            </select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Scout alt (m)</label>
            <input type="number" step="0.1" min="0.4" max="3.0"
                   value="${d.scout_alt_m}" data-fc="${fc}" data-field="scout_alt_m">
          </div>
          <div>
            <label>Attack alt (m)</label>
            <input type="number" step="0.1" min="0.4" max="3.0"
                   value="${d.attack_alt_m}" data-fc="${fc}" data-field="attack_alt_m">
          </div>
          <div>
            <label>Home alt (m)</label>
            <input type="number" step="0.1" min="0.4" max="3.0"
                   value="${d.home_alt_m}" data-fc="${fc}" data-field="home_alt_m">
          </div>
        </div>
        <div class="target-row" ${role!=="attacker"?'style="display:none"':''}>
          <input type="number" min="1" max="60"
                 placeholder="target marker id"
                 value="${targetVal}" data-target="${fc}">
          <button data-assign="${fc}">Assign</button>
          <button data-clear="${fc}">Clear</button>
        </div>
        <div class="phase" title="${reason}">
          phase=${phase}${rs.target_marker_id!=null?` target=${rs.target_marker_id}`:''}
          ${reason?` · ${reason}`:''}
        </div>
      </div>
    `;
    root.insertAdjacentHTML("beforeend", html);
  }
}

function renderMarkers(state) {
  const root = document.getElementById("markers");
  const items = state.markers || {};
  const ids = Object.keys(items).map(x=>parseInt(x,10)).sort((a,b)=>a-b);
  const liveIds = new Set([
    ...(state.settings.markers.red_live_ids || []),
    ...(state.settings.markers.blue_live_ids || []),
  ]);
  // Always render the live markers even if never seen yet.
  for (const lid of liveIds) {
    if (!ids.includes(lid)) ids.push(lid);
  }
  ids.sort((a,b)=>a-b);
  if (!ids.length) {
    root.innerHTML = '<div class="small">No marker observations yet. '
      + 'Scout some drones to populate.</div>';
    return;
  }
  root.innerHTML = "";
  for (const id of ids) {
    const s = items[id] || {team: liveIds.has(id) ? (state.settings.markers.red_live_ids.includes(id) ? "red" : "blue") : "other", captured: false};
    const cls = [s.team, s.captured ? "captured" : ""].join(" ");
    const age = s.last_seen_age_s == null ? "never" : `${s.last_seen_age_s.toFixed(1)}s`;
    const by = s.last_seen_by || "—";
    const status = s.captured ? "CAPTURED" : (s.seen_by_recent && s.seen_by_recent.length ? "visible" : "stale");
    root.insertAdjacentHTML("beforeend", `
      <div class="marker ${cls}" title="seen by: ${(s.seen_by_recent||[]).join(", ")||"none"}">
        <div class="id">${id}</div>
        <div class="sub">${status}</div>
        <div class="sub">${age}</div>
      </div>
    `);
  }
}

function renderEvents(state) {
  const root = document.getElementById("events");
  const evs = (state.events || []).slice().reverse();
  root.innerHTML = "";
  for (const e of evs) {
    const t = new Date((e.unix_s||0)*1000).toLocaleTimeString();
    const drone = e.drone ? `${e.drone} · ` : "";
    root.insertAdjacentHTML("beforeend",
      `<div class="ev ${e.kind}"><span class="k">${e.kind}</span> `
      + `<span class="small">${t}</span> ${drone}${e.text}</div>`);
  }
}

function renderHeader(state) {
  const armed = state.runner && state.runner.armed;
  const pill = document.getElementById("armed-pill");
  pill.className = "pill " + (armed ? "armed" : "disarmed");
  pill.textContent = armed ? "ARMED" : "disarmed";
  document.getElementById("tick").textContent = state.runner ? state.runner.tick_count : "–";
}

let last = null;
async function refresh() {
  try {
    const state = await api("/api/state");
    last = state;
    renderHeader(state);
    renderDrones(state);
    renderMarkers(state);
    renderEvents(state);
  } catch (e) {
    console.error("refresh failed", e);
  }
}

// Field editing
document.addEventListener("change", async (ev) => {
  const t = ev.target;
  if (!t.matches("[data-fc][data-field]")) return;
  const fc = t.getAttribute("data-fc");
  const field = t.getAttribute("data-field");
  let val = t.value;
  if (field === "enabled") val = (val === "true");
  if (["scout_alt_m","attack_alt_m","home_alt_m"].includes(field)) val = parseFloat(val);
  await patchDrone(fc, {[field]: val});
  refresh();
});

// Target assignment
document.addEventListener("click", async (ev) => {
  const t = ev.target;
  if (t.matches("[data-assign]")) {
    const fc = t.getAttribute("data-assign");
    const input = document.querySelector(`[data-target="${fc}"]`);
    const v = input.value.trim();
    if (!v) return;
    await assignTarget(fc, parseInt(v, 10));
    refresh();
  } else if (t.matches("[data-clear]")) {
    const fc = t.getAttribute("data-clear");
    await assignTarget(fc, null);
    refresh();
  }
});

document.getElementById("arm-btn").addEventListener("click", async () => {
  await api("/api/strategy/arm", {method:"POST"});
  refresh();
});
document.getElementById("disarm-btn").addEventListener("click", async () => {
  await api("/api/strategy/disarm", {method:"POST"});
  refresh();
});
document.getElementById("land-btn").addEventListener("click", async () => {
  if (!confirm("Emergency-land all drones?")) return;
  await api("/api/strategy/emergency-land", {method:"POST"});
  refresh();
});

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
