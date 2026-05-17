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
    planner=None,
    match_state=None,
    safety=None,
    event_log=None,
    runner=None,
) -> Flask:
    """Build the strategy web app.

    ``planner`` is optional — if supplied (and is a
    :class:`RoleAssignmentPlanner` or anything with
    ``current_role_for(fc)``), the ``/api/state`` endpoint reports the
    *live* role per drone instead of the operator-pinned role from
    settings. Pinning still drives the planner; the UI just shows what
    the strategy actually decided.

    ``match_state`` is optional — :class:`marker_mission_c2.strategy.match.MatchState`
    instance shared with the runner. When supplied, ``/api/state``
    embeds tick counter + match clock, and the ``/api/match/*``
    endpoints become operative.
    """
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
            live_role = None
            score = None
            if planner is not None and hasattr(planner, "current_role_for"):
                live_role = planner.current_role_for(name)
                if hasattr(planner, "current_score_for"):
                    score = planner.current_score_for(name)
            # Show the live role when the planner has one, else fall
            # back to the settings-pinned role. Both come from the
            # same Role enum, so the UI doesn't have to special-case.
            role_value = live_role or settings_obj.role_for(name).value
            task_name = None
            last_script = None
            if runner is not None:
                task_name = runner.current_task_for(name)
                last_script = runner.current_script_for(name)
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
                "role": role_value,
                "role_pinned": (
                    settings_obj.drones.get(name).role.value
                    if name in settings_obj.drones else "idle"
                ),
                "role_score": score,
                "cruise_altitude_m": settings_obj.cruise_altitude_for(name),
                "task": task_name,
                "last_script": last_script,
            }
        payload = {
            "t": state.t,
            "drones": drones_out,
            "team_color": settings_obj.team_color.value,
            "zones": {
                "our_home_x_m":     list(settings_obj.our_home_x_m),
                "our_home_y_m":     list(settings_obj.our_home_y_m),
                "enemy_home_x_m":   list(settings_obj.enemy_home_x_m),
                "enemy_home_y_m":   list(settings_obj.enemy_home_y_m),
                "neutral_zone_x_m": list(settings_obj.arena.neutral_zone_x_m),
                "neutral_zone_y_m": list(settings_obj.arena.neutral_zone_y_m),
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
        }
        if match_state is not None:
            payload.update(match_state.snapshot())
        if safety is not None:
            payload["armed"] = bool(safety.is_armed())
        return jsonify(payload)

    # ---- strategy arm/disarm ----

    @app.get("/api/strategy/armed")
    def api_strategy_armed_get():
        if safety is None:
            return jsonify(error="safety not wired"), 503
        return jsonify(armed=bool(safety.is_armed()))

    @app.post("/api/strategy/arm")
    def api_strategy_arm():
        if safety is None:
            return jsonify(error="safety not wired"), 503
        safety.arm()
        if event_log is not None:
            event_log.add("arm", "strategy ARMED")
        return jsonify(ok=True, armed=True)

    @app.post("/api/strategy/disarm")
    def api_strategy_disarm():
        if safety is None:
            return jsonify(error="safety not wired"), 503
        safety.disarm()
        if event_log is not None:
            event_log.add("disarm", "strategy DISARMED")
        return jsonify(ok=True, armed=False)

    # ---- event log ----

    @app.get("/api/strategy/events")
    def api_strategy_events():
        if event_log is None:
            return jsonify(error="event_log not wired"), 503
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
        since = request.args.get("since_seq")
        since_seq = int(since) if since and since.isdigit() else None
        return jsonify(
            events=event_log.recent(limit=limit, since_seq=since_seq),
            latest_seq=event_log.latest_seq(),
        )

    # ---- match clock ----

    @app.get("/api/match")
    def api_match_get():
        if match_state is None:
            return jsonify(error="match_state not wired"), 503
        return jsonify(match_state.snapshot())

    @app.post("/api/match/start")
    def api_match_start():
        if match_state is None:
            return jsonify(error="match_state not wired"), 503
        match_state.start_match()
        if safety is not None:
            safety.arm()
        if event_log is not None:
            event_log.add("match_start",
                          f"match started (duration {match_state.snapshot()['match']['duration_s']:.0f}s)")
        return jsonify(ok=True, **match_state.snapshot())

    @app.post("/api/match/stop")
    def api_match_stop():
        if match_state is None:
            return jsonify(error="match_state not wired"), 503
        match_state.stop_match()
        if event_log is not None:
            event_log.add("match_stop", "match stopped")
        return jsonify(ok=True, **match_state.snapshot())

    @app.post("/api/match/duration")
    def api_match_set_duration():
        if match_state is None:
            return jsonify(error="match_state not wired"), 503
        data = request.get_json(force=True) or {}
        try:
            new_dur = float(data["duration_s"])
        except (KeyError, TypeError, ValueError) as e:
            return jsonify(error=f"bad duration_s: {e}"), 400
        match_state.set_duration_s(new_dur)
        settings_obj.match.duration_s = new_dur
        # Persist so it survives a runner restart.
        try:
            save_settings(settings_obj, str(path))
        except Exception:
            log.exception("settings save (match.duration) failed")
        return jsonify(ok=True, **match_state.snapshot())

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

<div class="card" id="match-card" style="margin-bottom:14px;">
  <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
    <div>
      <div class="muted" style="font-size:11px; text-transform:uppercase;">strategy</div>
      <div id="strategy-status" style="font-size:14px;">—</div>
    </div>
    <div>
      <div class="muted" style="font-size:11px; text-transform:uppercase;">match</div>
      <div id="match-clock" style="font-size:24px; font-family:monospace; letter-spacing:.05em;">—:—</div>
    </div>
    <div>
      <div class="muted" style="font-size:11px; text-transform:uppercase;">duration</div>
      <div><input type="number" min="10" step="10" id="match-duration" style="width:90px;"> s
        <button type="button" id="duration-save" style="padding:4px 8px;">save</button>
      </div>
    </div>
    <div>
      <div class="muted" style="font-size:11px; text-transform:uppercase;">strategy</div>
      <button type="button" id="arm-toggle" style="padding:6px 14px; font-weight:600;">— </button>
    </div>
    <div style="margin-left:auto;">
      <button class="primary" type="button" id="match-start">Start match</button>
      <button type="button" id="match-stop">Stop</button>
    </div>
  </div>
  <div id="disarm-banner" class="warn" style="display:none; margin-top:8px;
       padding:6px 10px; border:1px solid #aa6; border-radius:4px; background:#3a2f10;">
    ⚠ <b>Strategy disarmed.</b> Planner is running and reporting roles
    on the arena view, but no commands are being pushed to FCs — the
    drone stays grounded until you click <b>Arm</b> or <b>Start match</b>.
  </div>
</div>

<div class="row">
  <div class="arena-wrap">
    <svg class="arena" id="arena" viewBox="-12 -7 24 14" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="zone-legend" id="zone-legend"></div>
  </div>
  <div style="min-width:360px;">
    <h2>Drones</h2>
    <table class="drone-table" id="drone-table">
      <thead><tr><th>FC</th><th>role</th><th>task</th><th>alt</th><th>pos</th><th>batt</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<h2>Events</h2>
<div class="card" style="padding:0; max-height:320px; overflow-y:auto;">
  <table id="events-table" style="width:100%; font-size:12px;">
    <thead style="position:sticky; top:0; background:#15171c; z-index:1;">
      <tr><th style="width:80px;">time</th>
          <th style="width:100px;">drone</th>
          <th style="width:110px;">kind</th>
          <th>message</th></tr>
    </thead>
    <tbody><tr><td colspan="4" class="muted" style="padding:8px;">no events yet</td></tr></tbody>
  </table>
</div>

<h2 style="margin-top:18px;">Last pushed script per drone</h2>
<div class="card" id="scripts-card">
  <pre id="scripts-pre" style="margin:0; font-size:12px; line-height:1.4;
       color:#bdf; white-space:pre-wrap;">(no scripts pushed yet)</pre>
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

  // Zone rectangles: each zone has an X range (left/right band) AND
  // a Y range (depth slice). Image: image_x = arena_y, image_y =
  // arena_x. Solid colour only; labels live in the HTML legend below
  // to dodge SVG-text sizing weirdness across browsers.
  const ourColor   = team === "red" ? "rgba(255,80,80,0.18)"  : "rgba(80,160,255,0.18)";
  const enemyColor = team === "red" ? "rgba(80,160,255,0.18)" : "rgba(255,80,80,0.18)";
  const neutralColor = "rgba(200,200,200,0.08)";
  function zone(xRange, yRange, fill) {{
    // Defensively normalise lo/hi in case the API ships a swapped
    // tuple — a zone with lo > hi would otherwise render as a
    // zero-or-negative-size rect (invisible). Python side also
    // auto-sorts on load; this is belt-and-braces.
    const xLo = Math.min(xRange[0], xRange[1]);
    const xHi = Math.max(xRange[0], xRange[1]);
    const yLo = Math.min(yRange[0], yRange[1]);
    const yHi = Math.max(yRange[0], yRange[1]);
    svg += `<rect x="${{yLo}}" y="${{xLo}}" `
         + `width="${{yHi - yLo}}" height="${{xHi - xLo}}" fill="${{fill}}" />`;
  }}
  zone(Z.our_home_x_m,     Z.our_home_y_m,     ourColor);
  zone(Z.enemy_home_x_m,   Z.enemy_home_y_m,   enemyColor);
  zone(Z.neutral_zone_x_m, Z.neutral_zone_y_m, neutralColor);

  // Drones
  let rows = "";
  for (const [name, d] of Object.entries(state.drones)) {{
    if (!d.pose) {{
      rows += `<tr><td>${{name}}</td><td>${{d.role}}</td>`
           +  `<td>${{d.task || "—"}}</td>`
           +  `<td>${{d.cruise_altitude_m.toFixed(1)}}m</td>`
           +  `<td class="muted">no pose</td>`
           +  `<td>${{d.battery_pct == null ? "–" : Math.round(d.battery_pct) + "%"}}</td></tr>`;
      continue;
    }}
    const [ax, ay, az] = d.pose;
    const {{ ix, iy }} = toImg(ax, ay);
    // Colour code by role: attacker = green-yellow, defender = orange,
    // scout = teal, idle = grey. Dim if not online.
    const ROLE_FILL = {{
      "attacker": "#9fe88a", "defender": "#ffb070",
      "scout":    "#7fd7e0", "idle":     "#8a8e94",
    }};
    let color = ROLE_FILL[d.role] || "#8a8e94";
    if (!d.online) color = "#444a52";
    const yaw = (d.yaw_deg ?? 0) - 90;  // 0° = nose +Y in arena
    svg += `<g transform="translate(${{ix}},${{iy}}) rotate(${{yaw}})">`
        +  `<polygon points="0,-0.35 0.25,0.25 -0.25,0.25" fill="${{color}}" stroke="#000" stroke-width="0.03"/>`
        +  `</g>`;
    // No SVG <text> for drone names — the side table is the canonical
    // source of "which dot is which". Keeps the arena viz robust to
    // SVG-text browser quirks.
    const roleDisplay = (d.role_pinned && d.role_pinned !== "idle" && d.role_pinned !== d.role)
        ? `${{d.role}} <span class="muted">(pin:${{d.role_pinned}})</span>`
        : (d.role_pinned && d.role_pinned !== "idle")
            ? `${{d.role}} <span class="muted">(pin)</span>`
            : d.role;
    rows += `<tr><td>${{name}}</td><td>${{roleDisplay}}</td>`
         +  `<td>${{d.task || "—"}}</td>`
         +  `<td>${{d.cruise_altitude_m.toFixed(1)}}m</td>`
         +  `<td>${{ax.toFixed(1)}}, ${{ay.toFixed(1)}}, ${{az.toFixed(1)}}</td>`
         +  `<td>${{d.battery_pct == null ? "–" : Math.round(d.battery_pct) + "%"}}</td></tr>`;
  }}

  SVG.innerHTML = svg;
  TBODY.innerHTML = rows || `<tr><td colspan="6" class="muted">no drones reporting</td></tr>`;

  // Per-drone last-pushed-script panel
  const scriptsPre = document.getElementById("scripts-pre");
  if (scriptsPre) {{
    const parts = [];
    for (const [name, d] of Object.entries(state.drones)) {{
      if (!d.last_script) continue;
      parts.push(`# ${{name}}\\n${{d.last_script.trim()}}`);
    }}
    scriptsPre.textContent = parts.length ? parts.join("\\n\\n") : "(no scripts pushed yet)";
  }}

  // Zone + role legend (HTML, immune to SVG-text sizing weirdness)
  LEGEND.innerHTML =
      `<span><span class="swatch" style="background:${{ourColor}}"></span>OUR HOME (${{team}})</span>`
    + `<span><span class="swatch" style="background:${{neutralColor}}"></span>NEUTRAL</span>`
    + `<span><span class="swatch" style="background:${{enemyColor}}"></span>ENEMY HOME</span>`
    + `<span><span class="swatch" style="background:#9fe88a; border:1px solid #000"></span>attacker</span>`
    + `<span><span class="swatch" style="background:#ffb070; border:1px solid #000"></span>defender</span>`
    + `<span><span class="swatch" style="background:#7fd7e0; border:1px solid #000"></span>scout</span>`
    + `<span><span class="swatch" style="background:#8a8e94; border:1px solid #000"></span>idle</span>`;
}}

function fmtTime(s) {{
  if (s == null || !isFinite(s)) return "—:—";
  s = Math.max(0, Math.floor(s));
  const m = Math.floor(s / 60), r = s % 60;
  return String(m).padStart(2,"0") + ":" + String(r).padStart(2,"0");
}}

function drawMatch(data) {{
  const STR = document.getElementById("strategy-status");
  const CLOCK = document.getElementById("match-clock");
  const DUR_INPUT = document.getElementById("match-duration");
  const BTN_START = document.getElementById("match-start");
  const BTN_STOP = document.getElementById("match-stop");
  const ARM_BTN = document.getElementById("arm-toggle");
  const DISARM_BANNER = document.getElementById("disarm-banner");

  // strategy alive indicator
  const ticks = data.ticks || {{}};
  const hz = (ticks.hz != null) ? ticks.hz.toFixed(2) + " Hz" : "—";
  const lastAge = ticks.last_age_s;
  const aliveText = lastAge == null ? "starting…" :
    (lastAge < 3.0 ? "ticking" : "stalled (" + lastAge.toFixed(1) + "s ago)");
  STR.textContent = aliveText + " · " + (ticks.count || 0) + " ticks · " + hz;
  STR.className = (lastAge == null || lastAge < 3.0) ? "ok" : "err";

  // arm state
  const armed = !!data.armed;
  ARM_BTN.textContent = armed ? "ARMED — click to disarm" : "DISARMED — click to arm";
  ARM_BTN.style.background = armed ? "#3a2c14" : "#10401a";
  ARM_BTN.style.borderColor = armed ? "#a73" : "#3a6";
  ARM_BTN.style.color = armed ? "#fc8" : "#9d6";
  DISARM_BANNER.style.display = armed ? "none" : "block";

  // match clock
  const m = data.match || {{}};
  if (m.running) {{
    CLOCK.textContent = fmtTime(m.remaining_s) + " / " + fmtTime(m.duration_s);
    CLOCK.style.color = m.expired ? "#f88" : (m.remaining_s < 30 ? "#fc8" : "#9d6");
    BTN_START.style.display = "none";
    BTN_STOP.style.display = "";
  }} else {{
    CLOCK.textContent = "NOT STARTED";
    CLOCK.style.color = "#9aa";
    BTN_START.style.display = "";
    BTN_STOP.style.display = "none";
  }}

  // duration input — only update if the user isn't editing it
  if (document.activeElement !== DUR_INPUT && m.duration_s != null) {{
    DUR_INPUT.value = m.duration_s;
  }}
}}

async function postMatch(suffix) {{
  try {{
    await fetch("/api/match/" + suffix, {{method: "POST"}});
    tick();
  }} catch (e) {{ console.error(e); }}
}}

document.getElementById("match-start").onclick = () => postMatch("start");
document.getElementById("match-stop").onclick  = () => postMatch("stop");
document.getElementById("arm-toggle").onclick = async () => {{
  // Read CURRENT armed state from button text — avoids stale closure
  // over an old data snapshot if the user clicks fast.
  const armed = document.getElementById("arm-toggle").textContent.startsWith("ARMED");
  await fetch("/api/strategy/" + (armed ? "disarm" : "arm"), {{method: "POST"}});
  tick();
}};
document.getElementById("duration-save").onclick = async () => {{
  const v = parseFloat(document.getElementById("match-duration").value);
  if (!isFinite(v) || v < 1) return;
  await fetch("/api/match/duration", {{
    method: "POST", headers: {{"content-type":"application/json"}},
    body: JSON.stringify({{duration_s: v}})
  }});
  tick();
}};

// Events panel — colour by kind, sticky timestamp.
const EVENTS_TBODY = document.querySelector("#events-table tbody");
const EVENT_COLORS = {{
  role_change: "#fc8",
  script_push: "#7cf",
  safety:      "#f88",
  arm:         "#9d6",
  disarm:      "#fc8",
  match_start: "#9d6",
  match_stop:  "#fc8",
  info:        "#9aa",
}};
let LAST_EVENT_SEQ = 0;
let EVENT_HISTORY = [];   // newest-first, capped to MAX

const MAX_EVENT_HISTORY = 200;

function fmtWall(t) {{
  const d = new Date(t * 1000);
  return d.toLocaleTimeString("en-GB", {{hour12: false}});
}}

function renderEvents() {{
  if (!EVENT_HISTORY.length) {{
    EVENTS_TBODY.innerHTML =
      `<tr><td colspan="4" class="muted" style="padding:8px;">no events yet</td></tr>`;
    return;
  }}
  EVENTS_TBODY.innerHTML = EVENT_HISTORY.map(e => {{
    const color = EVENT_COLORS[e.kind] || "#9aa";
    const msg = e.msg.length > 200 ? e.msg.slice(0, 200) + "…" : e.msg;
    return `<tr>
      <td class="muted" style="font-family:monospace; font-size:11px;">${{fmtWall(e.t_wall)}}</td>
      <td style="font-size:11px;">${{e.drone || "—"}}</td>
      <td style="color:${{color}}; font-weight:600;">${{e.kind}}</td>
      <td style="font-family:monospace; font-size:11px;">${{msg}}</td>
    </tr>`;
  }}).join("");
}}

async function pollEvents() {{
  try {{
    const url = "/api/strategy/events?limit=50"
              + (LAST_EVENT_SEQ ? "&since_seq=" + LAST_EVENT_SEQ : "");
    const r = await fetch(url);
    if (!r.ok) return;
    const data = await r.json();
    const fresh = (data.events || []).filter(e => e.seq > LAST_EVENT_SEQ);
    if (fresh.length) {{
      LAST_EVENT_SEQ = data.latest_seq;
      // fresh are newest-first; prepend to history.
      EVENT_HISTORY = fresh.concat(EVENT_HISTORY).slice(0, MAX_EVENT_HISTORY);
      renderEvents();
    }}
  }} catch (e) {{ /* ignore transient fetch errors */ }}
}}

async function tick() {{
  try {{
    const r = await fetch("/api/state");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    STATUS.className = "muted"; STATUS.textContent = "ok · team=" + data.team_color;
    drawMatch(data);
    draw(data);
  }} catch (e) {{
    STATUS.className = "err"; STATUS.textContent = "fetch failed: " + e.message;
  }}
}}
tick(); setInterval(tick, 500);
pollEvents(); setInterval(pollEvents, 1000);
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
    <label>red_home_x_m (left band)</label>
    <div class="row2"><input type="number" step="0.1" id="arena.red_home_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.red_home_x_m.hi"></div>
    <label>red_home_y_m (depth)</label>
    <div class="row2"><input type="number" step="0.1" id="arena.red_home_y_m.lo"> to
                       <input type="number" step="0.1" id="arena.red_home_y_m.hi"></div>
    <label>blue_home_x_m (right band)</label>
    <div class="row2"><input type="number" step="0.1" id="arena.blue_home_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.blue_home_x_m.hi"></div>
    <label>blue_home_y_m (depth)</label>
    <div class="row2"><input type="number" step="0.1" id="arena.blue_home_y_m.lo"> to
                       <input type="number" step="0.1" id="arena.blue_home_y_m.hi"></div>
    <label>neutral_zone_x_m</label>
    <div class="row2"><input type="number" step="0.1" id="arena.neutral_zone_x_m.lo"> to
                       <input type="number" step="0.1" id="arena.neutral_zone_x_m.hi"></div>
    <label>neutral_zone_y_m</label>
    <div class="row2"><input type="number" step="0.1" id="arena.neutral_zone_y_m.lo"> to
                       <input type="number" step="0.1" id="arena.neutral_zone_y_m.hi"></div>
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
  setVal("arena.red_home_y_m.lo", cfg.arena.red_home_y_m[0]);
  setVal("arena.red_home_y_m.hi", cfg.arena.red_home_y_m[1]);
  setVal("arena.blue_home_x_m.lo", cfg.arena.blue_home_x_m[0]);
  setVal("arena.blue_home_x_m.hi", cfg.arena.blue_home_x_m[1]);
  setVal("arena.blue_home_y_m.lo", cfg.arena.blue_home_y_m[0]);
  setVal("arena.blue_home_y_m.hi", cfg.arena.blue_home_y_m[1]);
  setVal("arena.neutral_zone_x_m.lo", cfg.arena.neutral_zone_x_m[0]);
  setVal("arena.neutral_zone_x_m.hi", cfg.arena.neutral_zone_x_m[1]);
  setVal("arena.neutral_zone_y_m.lo", cfg.arena.neutral_zone_y_m[0]);
  setVal("arena.neutral_zone_y_m.hi", cfg.arena.neutral_zone_y_m[1]);

  // drones table — one row per FC from topology, prefilled with current assignment
  const tbody = document.querySelector("#drones-table tbody");
  tbody.innerHTML = "";
  for (const fc of top.fcs) {{
    const cur = cfg.drones[fc.name] || {{role: "idle", altitude_m: 1.5}};
    const tr = document.createElement("tr");
    // "idle" is the "auto / let the strategy decide" value — pin a
    // specific role (attacker/scout/defender) only when you need to
    // force one drone. The default `idle` lets RoleAssignmentPlanner
    // pick each tick from live state.
    tr.innerHTML = `<td>${{fc.name}}</td>
      <td><select data-fc="${{fc.name}}" data-k="role">
        ${{[["idle","(auto — strategy decides)"],
            ["attacker","attacker (pinned)"],
            ["scout","scout (pinned)"],
            ["defender","defender (pinned)"]].map(([r, label]) =>
              `<option value="${{r}}"${{r===cur.role?" selected":""}}>${{label}}</option>`).join("")}}
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
      red_home_y_m:     [getNum("arena.red_home_y_m.lo"),     getNum("arena.red_home_y_m.hi")],
      blue_home_x_m:    [getNum("arena.blue_home_x_m.lo"),    getNum("arena.blue_home_x_m.hi")],
      blue_home_y_m:    [getNum("arena.blue_home_y_m.lo"),    getNum("arena.blue_home_y_m.hi")],
      neutral_zone_x_m: [getNum("arena.neutral_zone_x_m.lo"), getNum("arena.neutral_zone_x_m.hi")],
      neutral_zone_y_m: [getNum("arena.neutral_zone_y_m.lo"), getNum("arena.neutral_zone_y_m.hi")],
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
