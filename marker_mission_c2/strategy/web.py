"""Flask web UI + JSON API for the strategy layer.

Endpoints
---------
GET  /                                — HTML dashboard
GET  /api/state                       — combined snapshot (settings + runner
                                        state + slot statuses + recent events)
GET  /api/settings                    — current settings JSON
POST /api/settings/drone/<fc>         — patch a drone (team/role/altitudes)
POST /api/settings/match              — patch match-level fields
POST /api/settings/markers            — patch our_team / active_slots
POST /api/strategy/arm                — arm the runner (allow pushes)
POST /api/strategy/disarm             — disarm
POST /api/strategy/target/<fc>        — body {slot: int|null} → set/clear slot
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
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    @app.route("/")
    def index():
        return _no_cache(app.response_class(_INDEX_HTML, mimetype="text/html"))

    @app.route("/api/state")
    def api_state():
        import time as _time
        s = settings.to_dict()
        our_team = s["markers"]["our_team"]
        return _no_cache(jsonify({
            "settings": s,
            "runner": runner.snapshot(),
            "slots": markers.to_dict(our_team=our_team),
            "events": runner.events.to_list()[-50:],
            "roles": sorted(all_roles().keys()),
            "valid_teams": list(VALID_TEAMS),
            "valid_roles": list(VALID_ROLES),
            # v7 scoring (regs §1.4.3):
            #   1 pt   single drone enemy capture
            #   5 pts  full-sortie (all our drones outside home) ← future
            #  10 pts  double-strike (2 enemy slots within 1 s, same team)
            #          → split as 5 pts × 2 in the per-flip events
            # ``score`` is the running per-team total of points awarded so
            # far. ``match_status`` reflects v7 Special Maneuver state for
            # OUR team: dwell timer + instant-win flag.
            "score": markers.score(),
            "capture_events": markers.capture_events(limit=15),
            "match_status": markers.match_status(our_team, _time.time()),
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
        # Apply role change immediately so a follow-up target assignment
        # isn't wiped by the runner's lazy reset on the next tick.
        if "role" in body:
            runner.sync_role_state(fc_name)
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

    @app.route("/api/strategy/arm", methods=["POST"])
    def api_arm():
        runner.arm("web")
        return _no_cache(jsonify({"ok": True, "armed": runner.is_armed()}))

    @app.route("/api/strategy/disarm", methods=["POST"])
    def api_disarm():
        runner.disarm("web")
        return _no_cache(jsonify({"ok": True, "armed": runner.is_armed()}))

    @app.route("/api/strategy/roster/reset", methods=["POST"])
    def api_roster_reset():
        """Drop every drone from this strategy's roster (and per-drone role
        memory). Auto-adopt fills it back in from the C2 overview on the
        next tick with fresh defaults (e.g. the current scout_alt_m, not a
        stale persisted value)."""
        removed = runner.reset_roster("web")
        return _no_cache(jsonify({"ok": True, "removed": int(removed)}))

    @app.route("/api/strategy/mode", methods=["POST"])
    def api_mode():
        """Switch MANUAL <-> AUTO. Body: {"mode":"auto"|"manual"} or {"auto":bool}."""
        body = request.get_json(silent=True) or {}
        if "auto" in body:
            auto = bool(body["auto"])
        else:
            auto = str(body.get("mode", "")).lower() == "auto"
        runner.set_mode(auto, "web")
        return _no_cache(jsonify({
            "ok": True,
            "auto": runner.is_auto(),
            "mode": "auto" if runner.is_auto() else "manual",
        }))

    @app.route("/api/strategy/target/<fc_name>", methods=["POST"])
    def api_target(fc_name: str):
        body = request.get_json(silent=True) or {}
        raw = body.get("slot", body.get("marker_id"))   # legacy marker_id accepted
        slot: Optional[int]
        if raw is None or raw == "" or raw == "null":
            slot = None
        else:
            try:
                slot = int(raw)
            except (TypeError, ValueError):
                return _no_cache(jsonify({"ok": False, "error": "invalid slot"})), 400
            if not (1 <= slot <= 6):
                return _no_cache(jsonify(
                    {"ok": False, "error": "slot out of range 1..6"})), 400
        runner.assign_target(fc_name, slot)
        return _no_cache(jsonify({"ok": True, "slot": slot}))

    @app.route("/api/strategy/emergency-land", methods=["POST"])
    def api_emergency():
        from .c2_client import C2Client

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
# HTML
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
           display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; }
  header .spacer { flex: 1; }
  header .pill { padding: 4px 10px; border-radius: 999px;
                 background: var(--panel2); color: var(--muted);
                 font-size: 12px; }
  header .pill.armed { background: var(--green); color: #000; }
  header .pill.disarmed { background: #444; }
  header .pill.auto { background: var(--yellow); color: #000; }
  header .pill.manual { background: var(--panel2); color: var(--muted); }
  header .pill.red { background: var(--red); color: #fff; }
  header .pill.blue { background: var(--blue); color: #fff; }
  header select { background: #16181d; color: var(--text);
                  border: 1px solid var(--border); padding: 4px 8px;
                  border-radius: 4px; }
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
  /* Live per-drone status: link dots + compact field grid. */
  .drone .head .status-dot { display: inline-block; width: 8px; height: 8px;
                             border-radius: 50%; background: var(--muted);
                             margin-left: 2px; }
  .drone .head .status-dot.good { background: #4ade80; }
  .drone .head .status-dot.warn { background: #facc15; }
  .drone .head .status-dot.bad  { background: var(--red); }
  .drone .status-grid { display: grid; grid-template-columns: repeat(3, 1fr);
                        gap: 3px 10px; margin: 6px 0 4px; font-size: 12px; }
  .drone .status-grid > div { display: flex; gap: 6px; align-items: baseline;
                              white-space: nowrap; overflow: hidden;
                              text-overflow: ellipsis; }
  .drone .status-grid .k { color: var(--muted); }
  .drone .status-grid .v { color: var(--text); }
  .drone .status-grid .v.good { color: #4ade80; }
  .drone .status-grid .v.warn { color: #facc15; }
  .drone .status-grid .v.bad  { color: var(--red); }
  .drone .status-grid .v.mono { font-family: ui-monospace,Menlo,monospace;
                                font-size: 11px; }
  .drone .phase { font-size: 12px; color: var(--muted);
                  font-family: ui-monospace, monospace; margin-top: 6px; }
  .target-row { display: flex; gap: 4px; margin-top: 8px; align-items: center; }
  .target-row select { flex: 1; padding: 4px 6px; background: #16181d;
                       color: var(--text); border: 1px solid var(--border);
                       border-radius: 4px; }
  .slot-grid { display: grid;
               grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
               gap: 8px; }
  .slot { padding: 10px; border-radius: 6px; background: var(--panel2);
          border: 2px solid var(--border); font-family: ui-monospace, monospace;
          position: relative; }
  .slot.red    { border-color: var(--red); }
  .slot.blue   { border-color: var(--blue); }
  .slot.unknown { border-color: var(--border); opacity: 0.7; }
  .slot.ours   { box-shadow: 0 0 0 2px var(--green) inset; }
  .slot .title { display: flex; justify-content: space-between;
                 align-items: baseline; margin-bottom: 4px; }
  .slot .num { font-size: 14px; font-weight: 700; }
  .slot .pair { font-size: 10px; color: var(--muted); }
  .slot .holder { font-size: 13px; font-weight: 600; text-transform: uppercase;
                  letter-spacing: 0.04em; }
  .slot.red .holder { color: var(--red); }
  .slot.blue .holder { color: var(--blue); }
  .slot.unknown .holder { color: var(--muted); }
  .slot .info { font-size: 10px; color: var(--muted); margin-top: 4px;
                line-height: 1.3; }
  .slot .ours-tag { position: absolute; top: 4px; right: 6px;
                    font-size: 10px; padding: 1px 6px; border-radius: 999px;
                    background: var(--green); color: #000; font-weight: 700; }
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
</style>
</head>
<body>
<header>
  <h1>SDC26 Strategy</h1>
  <span id="armed-pill" class="pill disarmed">disarmed</span>
  <span id="mode-pill" class="pill" title="MANUAL: you assign targets. AUTO: C2 reacts to slot changes.">MANUAL</span>
  <span class="pill" title="v7 scoring (regs §1.4.3): 1 pt singleton, 10 pt double-strike sync (auto-detected within 1 s window). 5 pt full-sortie bonus pending.">
    score: <span id="score-red" style="color:var(--red)">0</span>
         · <span id="score-blue" style="color:var(--blue)">0</span>
  </span>
  <span id="special-pill" class="pill" style="display:none" title="v7 §1.4.3 Special Maneuver: holding all 6 slots in our colour for ≥ 5 s ends the match — INSTANT WIN">
    Special: <span id="special-text">–</span>
  </span>
  <span class="pill">tick <span id="tick">–</span></span>
  <span class="small">our team:</span>
  <select id="our-team">
    <option value="red">red</option>
    <option value="blue">blue</option>
  </select>
  <span class="spacer"></span>
  <button id="mode-btn" class="warn" title="Toggle MANUAL / AUTO">Go AUTO</button>
  <button id="arm-btn" class="primary">Arm</button>
  <button id="disarm-btn">Disarm</button>
  <button id="reset-btn" title="Drop every drone from the roster — auto-adopt then re-fills it from the C2 with fresh defaults. Useful for clearing ghost drones / stale per-drone settings.">Reset roster</button>
  <button id="land-btn" class="danger">Emergency Land</button>
</header>

<main>
  <section>
    <h2>Drones</h2>
    <div id="drones" class="grid"></div>
  </section>

  <section>
    <h2>Target Slots</h2>
    <div id="slots" class="slot-grid"></div>
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
async function patchMarkers(patch) {
  return api(`/api/settings/markers`, {method:"POST", body: JSON.stringify(patch)});
}
async function assignTarget(fc, slot) {
  return api(`/api/strategy/target/${encodeURIComponent(fc)}`, {
    method:"POST", body: JSON.stringify({slot: slot})});
}

function ageStr(s) {
  if (s == null) return "never";
  if (s < 1) return s.toFixed(1) + "s";
  if (s < 60) return Math.round(s) + "s";
  return Math.round(s/60) + "m";
}

function renderDrones(state) {
  const root = document.getElementById("drones");
  const drones = (state.settings.drones || {});
  const role_states = (state.runner && state.runner.drones) || {};
  const overview = (state.runner && state.runner.overview) || {};
  const activeSlots = (state.settings.markers && state.settings.markers.active_slots) || [1,2,3,4,5,6];
  const names = Object.keys(drones).sort();
  if (!names.length) {
    root.innerHTML = '<div class="small">No drones configured.</div>';
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
    const targetVal = rs.target_slot == null ? "" : rs.target_slot;
    const phase = rs.phase || "idle";
    const reason = (rs.last_decision_reason || "").replace(/"/g, '&quot;');
    const attackId = rs.last_attack_marker_id;
    const cruiseAlts = (state.runner && state.runner.cruise_alts) || {};
    const cruiseAlt = cruiseAlts[fc] != null ? cruiseAlts[fc] : null;
    const slotOpts = ['<option value="">— no target —</option>']
      .concat(activeSlots.map(s => `<option value="${s}" ${targetVal===s?"selected":""}>slot ${s}</option>`))
      .join("");

    // ── Live status from the C2 overview (per-FC live telemetry). ──
    // overview shape: { fc: {connection_ok, drone_connected, state:{...}} }.
    // Be defensive — fields may be missing while a drone is offline.
    const ov = overview[fc] || {};
    const ovState = (ov && ov.state) || {};
    const tel = (ovState && ovState.telemetry) || {};
    const c2Ok = !!ov.connection_ok;
    const droneOk = !!ov.drone_connected;
    const c2DotCls = c2Ok ? "good" : "bad";
    const droneDotCls = droneOk ? "good" : (c2Ok ? "warn" : "bad");
    const fcPhase = ovState.phase || "—";
    const heightCm = (tel.height_cm != null) ? Number(tel.height_cm)
                     : (ovState.height_cm != null ? Number(ovState.height_cm) : null);
    const heightStr = heightCm != null ? (heightCm/100).toFixed(2) + "m" : "—";
    const batteryRaw = (tel.battery_percent != null) ? tel.battery_percent
                     : (tel.battery != null ? tel.battery
                     : (ovState.battery_percent != null ? ovState.battery_percent : null));
    const battery = batteryRaw != null ? Math.round(Number(batteryRaw)) : null;
    const battStr = battery != null ? battery + "%" : "—";
    const battCls = battery == null ? "" : (battery < 30 ? "bad" : battery < 58 ? "warn" : "good");
    const wpos = ovState.world_position_m;
    const posStr = (Array.isArray(wpos) && wpos.length >= 2 && wpos[0] != null && wpos[1] != null)
      ? `(${Number(wpos[0]).toFixed(1)}, ${Number(wpos[1]).toFixed(1)})` : "—";
    // in-home computed client-side (mirrors strategy/roles.in_home_zone).
    let inHome = "?", inHomeCls = "";
    if (Array.isArray(wpos) && wpos.length >= 2 && team && wpos[0] != null && wpos[1] != null) {
      const x = Number(wpos[0]), y = Number(wpos[1]);
      const inX = Math.abs(x) <= 5.0;
      if (team === "red")  inHome = (inX && y >= -10 && y <= -5) ? "yes" : "no";
      if (team === "blue") inHome = (inX && y >= 5 && y <= 10) ? "yes" : "no";
      inHomeCls = inHome === "yes" ? "good" : (inHome === "no" ? "warn" : "");
    }
    const visible = ovState.visible_marker_ids;
    const visStr = (Array.isArray(visible) && visible.length) ? visible.join(",") : "—";
    const missionRun = fcPhase && !["init", "idle", "done", "", "—"].includes(String(fcPhase).toLowerCase());
    const c2DotTitle = c2Ok ? `C2 link to ${fc}: ok` : `C2 link to ${fc}: DOWN`;
    const droneDotTitle = droneOk ? `drone link: connected` : `drone link: not connected`;
    const lastPush = rs.last_pushed_age_s != null ? `pushed ${ageStr(rs.last_pushed_age_s)} ago` : "";

    const html = `
      <div class="drone">
        <div class="head">
          <span class="name">${fc}</span>
          <span class="badge ${teamBadge}">${team || "no team"}</span>
          <span class="badge ${roleBadge}">${role}</span>
          <span class="status-dot ${c2DotCls}" title="${c2DotTitle}"></span>
          <span class="status-dot ${droneDotCls}" title="${droneDotTitle}"></span>
        </div>
        <div class="status-grid">
          <div><span class="k">battery</span><span class="v ${battCls}">${battStr}</span></div>
          <div><span class="k">height</span><span class="v">${heightStr}</span></div>
          <div><span class="k">FC phase</span><span class="v">${fcPhase}${missionRun?" ●":""}</span></div>
          <div><span class="k">in home</span><span class="v ${inHomeCls}">${inHome}</span></div>
          <div><span class="k">world pos</span><span class="v mono">${posStr}</span></div>
          <div><span class="k">sees</span><span class="v mono" title="visible ArUco IDs">${visStr}</span></div>
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
              <option value="defender" ${role==="defender"?"selected":""}>defender</option>
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
            <input type="number" step="0.1" min="0.4" max="5.0"
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
        <div class="target-row" ${(role!=="attacker"&&role!=="defender")?'style="display:none"':''}>
          <label class="small">${role==="defender"?"Defend slot:":"Target slot:"}</label>
          <select data-target="${fc}">${slotOpts}</select>
          <button data-clear="${fc}">Clear</button>
        </div>
        <div class="phase" title="${reason}">
          phase=${phase}${rs.target_slot!=null?` slot=${rs.target_slot}`:''}${attackId?` last_aim=${attackId}`:''}${cruiseAlt!=null?` · cruise ${cruiseAlt}m`:''}${lastPush?` · ${lastPush}`:''}
          ${reason?` · ${reason}`:''}
        </div>
      </div>
    `;
    root.insertAdjacentHTML("beforeend", html);
  }
}

function renderSlots(state) {
  const root = document.getElementById("slots");
  const slots = state.slots || {};
  const our = state.settings.markers.our_team || "red";
  const enemy = our === "red" ? "blue" : "red";
  const activeSlots = state.settings.markers.active_slots || [1,2,3,4,5,6];
  if (!activeSlots.length) {
    root.innerHTML = '<div class="small">No active slots.</div>';
    return;
  }
  root.innerHTML = "";
  for (const slotNum of activeSlots) {
    const s = slots[String(slotNum)] || {
      slot: slotNum, holder: "unknown",
      red_face_id: 40 + slotNum, blue_face_id: 30 + slotNum,
      last_seen_age_s: null, last_seen_by: null,
      seen_by_recent: [], last_observed_face_id: null,
    };
    const holder = s.holder || "unknown";
    const ours = holder === our;
    const cls = [holder, ours ? "ours" : ""].join(" ").trim();
    const age = ageStr(s.last_seen_age_s);
    const by = s.last_seen_by || "—";
    const recent = (s.seen_by_recent || []).join(", ") || "none";
    const lastFace = s.last_observed_face_id ? `id ${s.last_observed_face_id}` : "—";
    const lockHtml = s.locked
        ? `<br><span style="color:var(--yellow)">🔒 lock ${(s.lock_remaining_s||0).toFixed(1)}s</span>`
        : "";
    root.insertAdjacentHTML("beforeend", `
      <div class="slot ${cls}" title="seen by recent: ${recent}">
        ${ours ? '<span class="ours-tag">OURS</span>' : ''}
        <div class="title">
          <span class="num">Slot ${slotNum}</span>
          <span class="pair">${s.blue_face_id}/${s.red_face_id}</span>
        </div>
        <div class="holder">${holder}</div>
        <div class="info">last face: ${lastFace}<br>seen: ${age}${by !== "—" ? " · " + by : ""}${lockHtml}</div>
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
  const auto = !!(state.runner && state.runner.auto);
  const modePill = document.getElementById("mode-pill");
  modePill.className = "pill " + (auto ? "auto" : "manual");
  modePill.textContent = auto ? "AUTO" : "MANUAL";
  const modeBtn = document.getElementById("mode-btn");
  modeBtn.textContent = auto ? "Go MANUAL" : "Go AUTO";
  document.getElementById("tick").textContent = state.runner ? state.runner.tick_count : "–";
  const sc = state.score || {};
  document.getElementById("score-red").textContent  = sc.red  ?? 0;
  document.getElementById("score-blue").textContent = sc.blue ?? 0;
  // v7 §1.4.3 Special Maneuver banner. Three states the operator cares about:
  //   - hidden:          no all-ours dwell yet (nothing imminent)
  //   - "5/6 → X.Xs/5s": all-ours running, win imminent (count up to 5 s)
  //   - "WON!":          dwell crossed the threshold; instant-win triggered
  const ms = state.match_status || {};
  const specialPill = document.getElementById("special-pill");
  const specialText = document.getElementById("special-text");
  if (ms.won) {
    specialPill.style.display = "";
    specialPill.style.background = "var(--green, #1b6e2c)";
    specialPill.style.color = "#fff";
    specialText.textContent = "MATCH WON!";
  } else if (ms.all_ours && ms.dwell_s != null) {
    specialPill.style.display = "";
    specialPill.style.background = "";  // default pill bg
    specialPill.style.color = "";
    specialText.textContent = `holding 6/6 — ${ms.dwell_s.toFixed(1)}/${ms.threshold_s.toFixed(1)}s`;
  } else {
    specialPill.style.display = "none";
  }
  const teamSel = document.getElementById("our-team");
  // Only update if user isn't currently interacting with it.
  if (document.activeElement !== teamSel) {
    teamSel.value = state.settings.markers.our_team || "red";
  }
}

let last = null;
// Suppress the full re-render of the drones panel while the user is
// interacting with it — otherwise an open <select> dropdown gets ripped
// out from under them by the 1Hz refresh. Triggered on mousedown (works
// even before focus settles, and across Safari/Chrome). Cleared on
// `change` (selection committed) and a few seconds later as a safety net.
let dronesEditingUntil = 0;
function markEditingDrones(extra_ms = 8000) {
  dronesEditingUntil = Date.now() + extra_ms;
}
const dronesEl = document.getElementById("drones");
dronesEl.addEventListener("mousedown", (e) => {
  if (e.target.matches("select, input")) markEditingDrones();
});
dronesEl.addEventListener("focusin", (e) => {
  if (e.target.matches("select, input")) markEditingDrones();
});
dronesEl.addEventListener("change", () => {
  // Once the user has committed a change, let the refresh resume.
  dronesEditingUntil = 0;
});

function dronesIsEditing() {
  if (Date.now() < dronesEditingUntil) return true;
  const ae = document.activeElement;
  if (!ae) return false;
  if (!dronesEl.contains(ae)) return false;
  return ae.tagName === "SELECT" || ae.tagName === "INPUT";
}

async function refresh() {
  try {
    const state = await api("/api/state");
    last = state;
    renderHeader(state);
    renderSlots(state);
    renderEvents(state);
    if (!dronesIsEditing()) {
      renderDrones(state);
    }
  } catch (e) {
    console.error("refresh failed", e);
  }
}

document.addEventListener("change", async (ev) => {
  const t = ev.target;
  if (t.id === "our-team") {
    await patchMarkers({our_team: t.value});
    refresh();
    return;
  }
  if (t.matches("[data-fc][data-field]")) {
    const fc = t.getAttribute("data-fc");
    const field = t.getAttribute("data-field");
    let val = t.value;
    if (field === "enabled") val = (val === "true");
    if (["scout_alt_m","attack_alt_m","home_alt_m"].includes(field)) val = parseFloat(val);
    await patchDrone(fc, {[field]: val});
    refresh();
    return;
  }
  if (t.matches("[data-target]")) {
    const fc = t.getAttribute("data-target");
    const raw = t.value;
    const slot = raw === "" ? null : parseInt(raw, 10);
    await assignTarget(fc, slot);
    refresh();
    return;
  }
});

document.addEventListener("click", async (ev) => {
  const t = ev.target;
  if (t.matches("[data-clear]")) {
    const fc = t.getAttribute("data-clear");
    await assignTarget(fc, null);
    refresh();
  }
});

document.getElementById("mode-btn").addEventListener("click", async () => {
  // Toggle based on the last known mode.
  const auto = !!(last && last.runner && last.runner.auto);
  await api("/api/strategy/mode", {method:"POST", body: JSON.stringify({auto: !auto})});
  refresh();
});
document.getElementById("arm-btn").addEventListener("click", async () => {
  await api("/api/strategy/arm", {method:"POST"});
  refresh();
});
document.getElementById("disarm-btn").addEventListener("click", async () => {
  await api("/api/strategy/disarm", {method:"POST"});
  refresh();
});
document.getElementById("reset-btn").addEventListener("click", async () => {
  if (!confirm("Reset the roster? Every drone will be dropped; auto-adopt then refills it from the C2 with fresh defaults.")) return;
  const r = await api("/api/strategy/roster/reset", {method:"POST"});
  console.log("roster reset:", r);
  refresh();
});
document.getElementById("land-btn").addEventListener("click", async () => {
  // No confirmation — emergency-land must be IMMEDIATE. Fire the
  // request before anything else so a click always lands the swarm.
  await api("/api/strategy/emergency-land", {method:"POST"});
  refresh();
});

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
