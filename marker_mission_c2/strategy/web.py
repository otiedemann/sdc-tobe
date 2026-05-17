"""Strategy operator UI.

A tiny Flask app, embedded on a background thread inside
:mod:`marker_mission_c2.strategy.app`. Two pages plus a small JSON
API:

  /           live top-down arena view (drones, zones, last-seen marker)
  /settings   form-bound editor for :class:`StrategySettings`

  GET  /api/state            current SwarmState (drones + zones)
  GET  /api/settings         current settings dict
  POST /api/settings         apply (validates, mutates the live instance,
                             then ``settings.save()`` writes to disk)
  POST /api/settings/reload  re-read settings.json from disk
  GET  /api/topology         FC list from the C2 config (drives the form's
                             per-FC table)

The UI mutates the *same* :class:`StrategySettings` instance the
:class:`SwarmRunner` reads from — every derived property recomputes
on access, so a Save takes effect on the next strategy tick without
a restart.

Defaults: binds 0.0.0.0:8091 (the C2 dashboard runs on 8090). Open
``http://<sphinx-host>:8091/`` once the strategy app is up.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

from .settings import (
    DEFAULT_PATH,
    StrategySettings,
    _from_dict,
)
from .settings import load as load_settings
from .settings import save as save_settings
from .world_model import SwarmWorldModel

log = logging.getLogger("c2.strategy.web")


# ---------------------------------------------------------------------------
# Live-mutate helper
# ---------------------------------------------------------------------------

def _replace_in_place(target: StrategySettings,
                      source: StrategySettings) -> None:
    """Copy every public field from ``source`` into ``target`` so
    callers sharing the ``target`` reference see the new values on
    the next read. Used by the POST handlers so the running planner
    sees edits without us having to rebuild the SwarmRunner."""
    target.team_color               = source.team_color
    target.own_target_ids_override  = source.own_target_ids_override
    target.enemy_target_ids_override = source.enemy_target_ids_override
    target.live_targets_only        = source.live_targets_only
    target.arena                    = source.arena
    target.drones                   = source.drones
    target.attack                   = source.attack
    target.defender                 = source.defender


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def make_app(
    settings_obj: StrategySettings,
    world_model: SwarmWorldModel,
    c2_cfg,
    settings_path: Optional[Path] = None,
) -> Flask:
    app = Flask("c2-strategy-web")
    path = Path(settings_path) if settings_path else DEFAULT_PATH

    @app.after_request
    def _no_cache(response):
        # The pages and JSON endpoints are tiny and re-read fresh state
        # every poll. Stale caches have already caused one round of
        # confusion (browser kept rendering an old broken SVG layout
        # after the server-side fix). Force every response to be
        # uncacheable — these are operator-facing pages, not assets.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # ---- pages ----

    @app.get("/")
    def page_index():
        return _INDEX_HTML

    @app.get("/settings")
    def page_settings():
        return _SETTINGS_HTML

    # ---- json api ----

    @app.get("/api/topology")
    def api_topology():
        return jsonify({
            "fcs": [
                {"name": fc.name, "host": fc.host, "port": fc.port}
                for fc in c2_cfg.fcs
            ],
            "settings_path": str(path),
        })

    @app.get("/api/settings")
    def api_get_settings():
        return jsonify(settings_obj.to_dict())

    @app.post("/api/settings")
    def api_post_settings():
        try:
            payload = request.get_json(force=True) or {}
            new_obj = _from_dict(payload)
        except (ValueError, TypeError) as e:
            return jsonify(error=str(e)), 400
        _replace_in_place(settings_obj, new_obj)
        try:
            saved_to = save_settings(settings_obj, str(path))
        except Exception as e:
            log.exception("settings save to %s failed", path)
            return jsonify(
                ok=True,
                warning=f"settings applied in-memory but disk write failed: {e}",
            ), 200
        return jsonify(ok=True, saved_to=str(saved_to))

    @app.post("/api/settings/reload")
    def api_reload_settings():
        try:
            fresh = load_settings(str(path))
        except (ValueError, FileNotFoundError) as e:
            return jsonify(error=str(e)), 400
        _replace_in_place(settings_obj, fresh)
        return jsonify(ok=True, loaded_from=str(path))

    @app.get("/api/state")
    def api_state():
        try:
            state = world_model.observe()
        except Exception as e:
            log.exception("world_model.observe() failed")
            return jsonify(error=str(e)), 500
        own_targets = sorted(settings_obj.own_target_ids)
        enemy_targets = sorted(settings_obj.enemy_target_ids)
        drones_out = {}
        for name, obs in state.drones.items():
            drones_out[name] = {
                "name": obs.name,
                "online": obs.online,
                "drone_connected": obs.drone_connected,
                "flying": obs.flying,
                "battery_pct": obs.battery_pct,
                "phase": obs.phase,
                "pose": list(obs.pose) if obs.pose else None,
                "yaw_deg": obs.yaw_deg,
                "last_marker_id": obs.last_marker_id,
                "serial": obs.serial,
                "age_s": obs.age_s,
                "role": settings_obj.role_for(name).value,
                "cruise_altitude_m": settings_obj.cruise_altitude_for(name),
            }
        return jsonify({
            "t": state.t,
            "drones": drones_out,
            "team_color": settings_obj.team_color.value,
            "zones": {
                "our_home_x_m": list(settings_obj.our_home_x_m),
                "enemy_home_x_m": list(settings_obj.enemy_home_x_m),
                "neutral_zone_x_m": list(settings_obj.arena.neutral_zone_x_m),
            },
            "arena": {
                "width_m": settings_obj.arena.width_m,
                "depth_m": settings_obj.arena.depth_m,
                "safety_margin_m": settings_obj.arena.safety_margin_m,
            },
            "targets": {
                "own": own_targets,
                "enemy": enemy_targets,
            },
        })

    return app


# ---------------------------------------------------------------------------
# Background-thread runner
# ---------------------------------------------------------------------------

def start_in_background(
    app: Flask,
    host: str = "0.0.0.0",
    port: int = 8091,
) -> threading.Thread:
    """Run the Flask app on a daemon thread. Returns the thread handle."""
    def _run() -> None:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True, name="c2-strategy-web")
    t.start()
    log.info("strategy web UI listening on http://%s:%d/", host, port)
    return t


# ---------------------------------------------------------------------------
# HTML — kept inline so there's nothing extra to ship.  Pure CSS + JS,
# polls /api/state at 2 Hz for the live arena view.  Orientation mirrors
# the canonical master image: image-right = arena +Y (front wall),
# image-top = arena -X (left wall).
# ---------------------------------------------------------------------------

_BASE_CSS = """
  :root { color-scheme: dark; }
  body { background:#0e0f12; color:#dde; font:14px/1.4 -apple-system,system-ui,Segoe UI,sans-serif;
         margin:0; padding:18px 24px; }
  a { color:#7cf; text-decoration:none; } a:hover { text-decoration:underline; }
  h1 { margin:0 0 4px; font-size:17px; font-weight:600; }
  h2 { margin:18px 0 6px; font-size:13px; font-weight:600; text-transform:uppercase;
       letter-spacing:.05em; color:#9aa; }
  .nav { margin-bottom:14px; color:#789; font-size:12px; }
  .nav span { margin-right:14px; }
  button { background:#1d2025; border:1px solid #333; color:#dde; padding:6px 12px;
           border-radius:4px; cursor:pointer; font:inherit; }
  button:hover { background:#262a30; }
  button.primary { background:#246; border-color:#48a; }
  button.primary:hover { background:#367; }
  input[type=number], input[type=text], select {
    background:#161922; border:1px solid #333; color:#dde; padding:4px 6px;
    border-radius:3px; font:inherit; width:80px;
  }
  table { border-collapse:collapse; margin:6px 0; }
  th, td { text-align:left; padding:4px 10px; border-bottom:1px solid #222; }
  th { color:#9aa; font-weight:600; font-size:12px; }
  .pill { display:inline-block; padding:1px 7px; border-radius:10px;
          font-size:11px; vertical-align:middle; }
  .pill.red { background:#642; color:#fcb; }
  .pill.blue { background:#246; color:#bdf; }
  .pill.off { background:#321; color:#caa; }
  .row { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .card { background:#15171c; border:1px solid #2a2d33; border-radius:6px;
          padding:12px 16px; margin-bottom:12px; }
  .muted { color:#789; }
  .ok { color:#9d6; }  .warn { color:#fc8; }  .err { color:#f88; }
  fieldset { border:1px solid #2a2d33; border-radius:6px; padding:10px 14px;
             margin:8px 0; }
  legend { color:#9aa; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  label { display:inline-block; min-width:160px; color:#bbc; }
"""


_INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy — live arena</title>
<style>{_BASE_CSS}
  .arena-wrap {{ background:#0a0b0e; border:1px solid #222; border-radius:6px;
                padding:8px; }}
  svg.arena {{ width:100%; max-width:880px; height:auto; display:block; }}
  .drone-table th, .drone-table td {{ font-size:12px; }}
  /* Zone legend — HTML, not SVG text, so font-size is reliable across
     every browser. SVG <text> sized in mixed pixel/viewBox units is
     a portability landmine; HTML <span> just renders at 12px. */
  .zone-legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px;
                  font-size:12px; color:#9aa; }}
  .zone-legend .swatch {{ display:inline-block; width:14px; height:14px;
                          border-radius:3px; vertical-align:middle;
                          margin-right:6px; }}
</style>
</head>
<body>
<div class="nav">
  <span><b>Strategy</b></span>
  <span><a href="/">Live arena</a></span>
  <span><a href="/settings">Settings</a></span>
  <span class="muted" id="status">connecting…</span>
</div>

<h1>Live arena</h1>
<div class="row">
  <div class="arena-wrap">
    <svg class="arena" id="arena" viewBox="-12 -7 24 14" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="zone-legend" id="zone-legend"></div>
  </div>
  <div style="min-width:280px;">
    <h2>Drones</h2>
    <table class="drone-table" id="drone-table">
      <thead><tr><th>FC</th><th>role</th><th>alt</th><th>pos</th><th>batt</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const SVG = document.getElementById("arena");
const TBODY = document.querySelector("#drone-table tbody");
const LEGEND = document.getElementById("zone-legend");
const STATUS = document.getElementById("status");

// Arena → image transform: image_x = arena_y, image_y = arena_x.
// SVG y increases downward, so arena_x=+5 (right wall) lands at the
// bottom of the image; arena_y=+10 (front wall) lands on the right.
function toImg(arena_x, arena_y) {{ return {{ ix: arena_y, iy: arena_x }}; }}

function draw(state) {{
  const A = state.arena;
  const Z = state.zones;
  const team = state.team_color;
  const halfW = A.width_m / 2, halfD = A.depth_m / 2;
  const pad = 2;
  SVG.setAttribute("viewBox",
    `${{-halfD - pad}} ${{-halfW - pad}} ${{A.depth_m + 2*pad}} ${{A.width_m + 2*pad}}`);

  let svg = "";

  // Arena outline
  svg += `<rect x="${{-halfD}}" y="${{-halfW}}" width="${{A.depth_m}}" height="${{A.width_m}}" `
       + `fill="#0e1217" stroke="#3a4250" stroke-width="0.06"/>`;

  // Safety margin (dashed)
  const m = A.safety_margin_m;
  svg += `<rect x="${{-halfD+m}}" y="${{-halfW+m}}" width="${{A.depth_m-2*m}}" height="${{A.width_m-2*m}}" `
       + `fill="none" stroke="#445" stroke-dasharray="0.2 0.2" stroke-width="0.04"/>`;

  // Zone bands along arena X axis (= image Y axis). Solid colour only;
  // labels go in the HTML legend below to dodge SVG-text sizing
  // weirdness across browsers.
  const ourColor   = team === "red" ? "rgba(255,80,80,0.18)"  : "rgba(80,160,255,0.18)";
  const enemyColor = team === "red" ? "rgba(80,160,255,0.18)" : "rgba(255,80,80,0.18)";
  const neutralColor = "rgba(200,200,200,0.08)";
  function band(xRange, fill) {{
    const [lo, hi] = xRange;
    svg += `<rect x="${{-halfD}}" y="${{lo}}" width="${{A.depth_m}}" height="${{hi - lo}}" fill="${{fill}}" />`;
  }}
  band(Z.our_home_x_m,     ourColor);
  band(Z.enemy_home_x_m,   enemyColor);
  band(Z.neutral_zone_x_m, neutralColor);

  // Drones
  let rows = "";
  for (const [name, d] of Object.entries(state.drones)) {{
    if (!d.pose) {{
      rows += `<tr><td>${{name}}</td><td>${{d.role}}</td><td>${{d.cruise_altitude_m.toFixed(1)}}m</td>`
           +  `<td class="muted">no pose</td><td>${{d.battery_pct == null ? "–" : Math.round(d.battery_pct) + "%"}}</td></tr>`;
      continue;
    }}
    const [ax, ay, az] = d.pose;
    const {{ ix, iy }} = toImg(ax, ay);
    const color = d.flying ? "#9fe88a" : (d.online ? "#fc8" : "#866");
    const yaw = (d.yaw_deg ?? 0) - 90;  // 0° = nose +Y in arena
    svg += `<g transform="translate(${{ix}},${{iy}}) rotate(${{yaw}})">`
        +  `<polygon points="0,-0.35 0.25,0.25 -0.25,0.25" fill="${{color}}" stroke="#000" stroke-width="0.03"/>`
        +  `</g>`;
    // No SVG <text> for drone names — the side table is the canonical
    // source of "which dot is which". Keeps the arena viz robust to
    // SVG-text browser quirks.
    rows += `<tr><td>${{name}}</td><td>${{d.role}}</td><td>${{d.cruise_altitude_m.toFixed(1)}}m</td>`
         +  `<td>${{ax.toFixed(1)}}, ${{ay.toFixed(1)}}, ${{az.toFixed(1)}}</td>`
         +  `<td>${{d.battery_pct == null ? "–" : Math.round(d.battery_pct) + "%"}}</td></tr>`;
  }}

  SVG.innerHTML = svg;
  TBODY.innerHTML = rows || `<tr><td colspan="5" class="muted">no drones reporting</td></tr>`;

  // Zone legend (HTML, immune to SVG-text sizing weirdness)
  LEGEND.innerHTML =
      `<span><span class="swatch" style="background:${{ourColor}}"></span>OUR HOME (${{team}})</span>`
    + `<span><span class="swatch" style="background:${{neutralColor}}"></span>NEUTRAL</span>`
    + `<span><span class="swatch" style="background:${{enemyColor}}"></span>ENEMY HOME</span>`
    + `<span><span class="swatch" style="background:rgba(159,232,138,0.7); border:1px solid #000"></span>flying</span>`
    + `<span><span class="swatch" style="background:rgba(255,204,136,0.7); border:1px solid #000"></span>on ground</span>`;
}}

async function tick() {{
  try {{
    const r = await fetch("/api/state");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    STATUS.className = "muted"; STATUS.textContent = "ok · team=" + data.team_color;
    draw(data);
  }} catch (e) {{
    STATUS.className = "err"; STATUS.textContent = "fetch failed: " + e.message;
  }}
}}
tick(); setInterval(tick, 500);
</script>
</body>
</html>"""


_SETTINGS_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy — settings</title>
<style>{_BASE_CSS}
  .grid {{ display:grid; grid-template-columns: 200px 1fr; gap:6px 12px; align-items:center; }}
  .grid input[type=number] {{ width:88px; }}
  .row2 input {{ width:60px; margin-right:4px; }}
</style>
</head>
<body>
<div class="nav">
  <span><b>Strategy</b></span>
  <span><a href="/">Live arena</a></span>
  <span><a href="/settings">Settings</a></span>
  <span class="muted" id="status">loading…</span>
</div>

<h1>Settings</h1>

<form id="form">

<fieldset><legend>Team</legend>
  <div class="grid">
    <label>team_color</label>
    <div>
      <label style="min-width:0;"><input type="radio" name="team_color" value="red"> red</label>
      <label style="min-width:0; margin-left:14px;"><input type="radio" name="team_color" value="blue"> blue</label>
    </div>
    <label>live_targets_only</label>
    <input type="checkbox" id="live_targets_only">
  </div>
</fieldset>

<fieldset><legend>Arena</legend>
  <div class="grid">
    <label>width_m (along X)</label><input type="number" step="0.1" id="arena.width_m">
    <label>depth_m (along Y)</label><input type="number" step="0.1" id="arena.depth_m">
    <label>safety_margin_m</label><input type="number" step="0.05" id="arena.safety_margin_m">
    <label>red_home_x_m</label>
    <div class="row2"><input type="number" step="0.1" id="arena.red_home_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.red_home_x_m.hi"></div>
    <label>blue_home_x_m</label>
    <div class="row2"><input type="number" step="0.1" id="arena.blue_home_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.blue_home_x_m.hi"></div>
    <label>neutral_zone_x_m</label>
    <div class="row2"><input type="number" step="0.1" id="arena.neutral_zone_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.neutral_zone_x_m.hi"></div>
  </div>
</fieldset>

<fieldset><legend>Drones</legend>
  <table id="drones-table">
    <thead><tr><th>FC name</th><th>role</th><th>altitude (m)</th></tr></thead>
    <tbody></tbody>
  </table>
  <p class="muted" style="font-size:12px;">FC list comes from <code>marker_mission_c2/config.json</code>.
     Edit it there to add or remove drones.</p>
</fieldset>

<fieldset><legend>Attack (10-pt maneuver)</legend>
  <div class="grid">
    <label>hover_alt_m</label><input type="number" step="0.1" id="attack.hover_alt_m">
    <label>strike_alt_m</label><input type="number" step="0.1" id="attack.strike_alt_m">
    <label>sync_window_s</label><input type="number" step="0.1" id="attack.sync_window_s">
    <label>home_zone_clear</label><input type="checkbox" id="attack.home_zone_clear">
    <label>pair_targets_override</label>
    <input type="text" id="attack.pair_targets_override" placeholder="e.g. 41,42 ; 42,43" style="width:240px">
  </div>
  <p class="muted" style="font-size:12px;">
    Leave <code>pair_targets_override</code> blank to auto-derive every 2-combination of own targets.
    Otherwise enter pairs separated by <code>;</code>, ids by <code>,</code>.
  </p>
</fieldset>

<fieldset><legend>Defender</legend>
  <div class="grid">
    <label>intercept_radius_m</label><input type="number" step="0.1" id="defender.intercept_radius_m">
  </div>
</fieldset>

<div style="margin-top:14px;">
  <button class="primary" type="button" id="save">Save</button>
  <button type="button" id="reload">Reload from disk</button>
  <span id="msg" class="muted" style="margin-left:14px;"></span>
</div>
</form>

<script>
const $ = (s) => document.getElementById(s);
const STATUS = $("status"); const MSG = $("msg");

function setVal(id, v) {{
  const el = $(id); if (!el) return;
  if (el.type === "checkbox") el.checked = !!v;
  else el.value = (v == null ? "" : v);
}}

function getNum(id) {{
  const v = $(id).value;
  return v === "" ? null : parseFloat(v);
}}

function pairsToText(arr) {{
  if (!arr) return "";
  return arr.map(p => p.join(",")).join("; ");
}}

function textToPairs(s) {{
  s = (s || "").trim();
  if (!s) return null;
  return s.split(";").map(part => part.split(",").map(x => parseInt(x.trim(), 10)));
}}

async function load() {{
  STATUS.textContent = "loading…";
  const [top, cfg] = await Promise.all([
    fetch("/api/topology").then(r => r.json()),
    fetch("/api/settings").then(r => r.json()),
  ]);

  // team
  for (const el of document.querySelectorAll("input[name=team_color]")) {{
    el.checked = (el.value === cfg.team_color);
  }}
  setVal("live_targets_only", cfg.live_targets_only);

  // arena
  setVal("arena.width_m", cfg.arena.width_m);
  setVal("arena.depth_m", cfg.arena.depth_m);
  setVal("arena.safety_margin_m", cfg.arena.safety_margin_m);
  setVal("arena.red_home_x_m.lo", cfg.arena.red_home_x_m[0]);
  setVal("arena.red_home_x_m.hi", cfg.arena.red_home_x_m[1]);
  setVal("arena.blue_home_x_m.lo", cfg.arena.blue_home_x_m[0]);
  setVal("arena.blue_home_x_m.hi", cfg.arena.blue_home_x_m[1]);
  setVal("arena.neutral_zone_x_m.lo", cfg.arena.neutral_zone_x_m[0]);
  setVal("arena.neutral_zone_x_m.hi", cfg.arena.neutral_zone_x_m[1]);

  // drones table — one row per FC from topology, prefilled with current assignment
  const tbody = document.querySelector("#drones-table tbody");
  tbody.innerHTML = "";
  for (const fc of top.fcs) {{
    const cur = cfg.drones[fc.name] || {{role: "idle", altitude_m: 1.5}};
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${{fc.name}}</td>
      <td><select data-fc="${{fc.name}}" data-k="role">
        ${{["attacker","scout","defender","idle"].map(r =>
            `<option value="${{r}}"${{r===cur.role?" selected":""}}>${{r}}</option>`).join("")}}
      </select></td>
      <td><input type="number" step="0.1" data-fc="${{fc.name}}" data-k="altitude_m" value="${{cur.altitude_m}}"></td>`;
    tbody.appendChild(tr);
  }}

  // attack
  setVal("attack.hover_alt_m", cfg.attack.hover_alt_m);
  setVal("attack.strike_alt_m", cfg.attack.strike_alt_m);
  setVal("attack.sync_window_s", cfg.attack.sync_window_s);
  setVal("attack.home_zone_clear", cfg.attack.home_zone_clear);
  setVal("attack.pair_targets_override", pairsToText(cfg.attack.pair_targets_override));

  // defender
  setVal("defender.intercept_radius_m", cfg.defender.intercept_radius_m);

  STATUS.className = "muted"; STATUS.textContent = "ok · " + top.settings_path;
}}

function gather() {{
  const team = document.querySelector("input[name=team_color]:checked").value;

  const drones = {{}};
  for (const sel of document.querySelectorAll("[data-fc]")) {{
    const fc = sel.dataset.fc; const k = sel.dataset.k;
    drones[fc] = drones[fc] || {{role: "idle", altitude_m: 1.5}};
    drones[fc][k] = (k === "altitude_m") ? parseFloat(sel.value) : sel.value;
  }}

  return {{
    team_color: team,
    own_target_ids_override: null,
    enemy_target_ids_override: null,
    live_targets_only: $("live_targets_only").checked,
    arena: {{
      width_m:          getNum("arena.width_m"),
      depth_m:          getNum("arena.depth_m"),
      safety_margin_m:  getNum("arena.safety_margin_m"),
      red_home_x_m:     [getNum("arena.red_home_x_m.lo"),     getNum("arena.red_home_x_m.hi")],
      blue_home_x_m:    [getNum("arena.blue_home_x_m.lo"),    getNum("arena.blue_home_x_m.hi")],
      neutral_zone_x_m: [getNum("arena.neutral_zone_x_m.lo"), getNum("arena.neutral_zone_x_m.hi")],
    }},
    drones: drones,
    attack: {{
      hover_alt_m:           getNum("attack.hover_alt_m"),
      strike_alt_m:          getNum("attack.strike_alt_m"),
      sync_window_s:         getNum("attack.sync_window_s"),
      pair_targets_override: textToPairs($("attack.pair_targets_override").value),
      home_zone_clear:       $("attack.home_zone_clear").checked,
    }},
    defender: {{
      intercept_radius_m: getNum("defender.intercept_radius_m"),
    }},
  }};
}}

$("save").onclick = async () => {{
  MSG.className = "muted"; MSG.textContent = "saving…";
  try {{
    const body = gather();
    const r = await fetch("/api/settings", {{
      method: "POST", headers: {{"content-type": "application/json"}},
      body: JSON.stringify(body),
    }});
    const j = await r.json();
    if (!r.ok || j.error) {{ MSG.className = "err"; MSG.textContent = "save failed: " + (j.error || r.status); return; }}
    MSG.className = "ok"; MSG.textContent = "saved → " + (j.saved_to || "memory") + (j.warning ? "  (" + j.warning + ")" : "");
  }} catch (e) {{ MSG.className = "err"; MSG.textContent = "save failed: " + e.message; }}
}};

$("reload").onclick = async () => {{
  MSG.className = "muted"; MSG.textContent = "reloading…";
  try {{
    const r = await fetch("/api/settings/reload", {{method: "POST"}});
    const j = await r.json();
    if (!r.ok || j.error) {{ MSG.className = "err"; MSG.textContent = "reload failed: " + (j.error || r.status); return; }}
    await load();
    MSG.className = "ok"; MSG.textContent = "reloaded from " + j.loaded_from;
  }} catch (e) {{ MSG.className = "err"; MSG.textContent = "reload failed: " + e.message; }}
}};

load();
</script>
</body>
</html>"""
