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
POST /api/strategy/fc/<fc>/endpoint   — point ONE FC at a real drone / sim
POST /api/strategy/fc/all             — body {mode: "real"|"sim"} → switch ALL FCs
POST /api/strategy/emergency-land     — emergency-land everyone
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from . import arena_state
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
    sim_url: Optional[str] = None,
) -> Flask:
    app = Flask(__name__)
    _sim_url = sim_url.rstrip("/") if sim_url else None

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

        # Team isolation (req 2 + regs §1.3 — "no access to enemy drones"):
        # filter every per-drone collection in the response to OUR-team
        # FCs (+ any still-unassigned drones the operator can still label
        # as ours). Enemy drones are stripped from settings, from the live
        # C2 overview, from the per-drone role state and from the cruise-
        # alt deconfliction map BEFORE they ever leave the server, so the
        # UI / API consumers never see them at all.
        def _is_ours_team(t):
            return (t == our_team) or (t is None) or (t == "")
        drones_dict = (s.get("drones") or {})
        our_fcs = {fc for fc, dd in drones_dict.items()
                   if _is_ours_team(dd.get("team"))}
        s["drones"] = {fc: dd for fc, dd in drones_dict.items() if fc in our_fcs}
        runner_snap = runner.snapshot()
        for key in ("overview", "drones", "cruise_alts"):
            sub = runner_snap.get(key)
            if isinstance(sub, dict):
                runner_snap[key] = {fc: v for fc, v in sub.items()
                                    if fc in our_fcs}

        return _no_cache(jsonify({
            "settings": s,
            "runner": runner_snap,
            "arena": arena_state.to_client_dict(),
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

    @app.route("/api/missions")
    def api_missions():
        """The exact mission scripts the C2 has pushed to the FCs.

        For copy-paste onto a live drone. ``?format=text`` returns the
        same copy-paste layout as the on-disk log (``# header`` comment +
        raw verbs); default is JSON. ``?fc=red3`` / ``?limit=20`` filter.
        Only OUR-team drones are ever in the roster, so nothing here can
        leak an enemy drone (regs §1.3).
        """
        limit = request.args.get("limit", type=int) or 50
        fc = request.args.get("fc") or None
        fmt = (request.args.get("format") or "json").lower()
        ml = runner.mission_log
        entries = ml.recent(limit=limit, fc=fc)
        if fmt in ("text", "txt", "raw"):
            blocks = [
                f"# [#{e['seq']} {e['time']}] {e['fc']} · {e['reason']}\n"
                f"{e['script']}"
                for e in entries
            ]
            body = ("\n\n".join(blocks) + "\n") if blocks else "# (no missions yet)\n"
            return _no_cache(app.response_class(body, mimetype="text/plain"))
        return _no_cache(jsonify({"path": ml.path, "missions": entries}))

    @app.route("/api/missions/<fc_name>/download")
    def api_missions_download(fc_name: str):
        """Download ONE drone's COMPLETE command list as a .log file.

        Returns every mission script the C2 pushed to ``fc_name`` this match,
        in order, as a copy-paste-ready marker_mission script: a ``#`` comment
        header per run (the FC's DSL ignores ``#`` lines) followed by the raw
        verbs. Served as an attachment so the browser saves it directly — for
        replaying the exact game on a real drone.
        """
        import time as _t
        ml = runner.mission_log
        # Newest-last (chronological) so the file reads top-to-bottom as the
        # match played out. recent() returns newest-first, so reverse it.
        entries = list(reversed(ml.recent(limit=100000, fc=fc_name)))
        header = (
            f"# marker_mission command list — drone {fc_name}\n"
            f"# exported {_t.strftime('%Y-%m-%d %H:%M:%S')} · "
            f"{len(entries)} mission(s)\n"
            f"# '#' lines are comments (ignored by the FC). Paste a block — or\n"
            f"# the whole file — onto a real drone to replay the game.\n\n"
        )
        blocks = [
            f"# {'=' * 60}\n"
            f"# [#{e['seq']} {e['time']}] {e['reason']}\n"
            f"# {'=' * 60}\n"
            f"{e['script']}"
            for e in entries
        ]
        body = header + ("\n\n".join(blocks) + "\n"
                         if blocks else "# (no missions recorded for this drone yet)\n")
        resp = app.response_class(body, mimetype="text/plain")
        resp.headers["Content-Disposition"] = (
            f'attachment; filename="{fc_name}_commands.log"')
        return _no_cache(resp)

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

    @app.route("/api/arena", methods=["GET"])
    def api_arena_get():
        """Active arena geometry the dashboard renders from."""
        return _no_cache(jsonify({"ok": True, "arena": arena_state.to_client_dict()}))

    @app.route("/api/arena/switch", methods=["POST"])
    def api_arena_switch():
        """Live arena switch. Body ``{"name": "real"|"gvz"}``.

        The SIM owns the active arena (the shared physical world), so we proxy
        the switch to the sim UI — its wall markers + boxes change without a
        restart. This strategy then follows (the runner polls the sim each few
        ticks); we also set it locally now for instant dashboard feedback, and
        the OTHER team's strategy follows the sim independently. On real HW
        (no sim URL) the switch still re-points this strategy and is pushed to
        the FCs by the C2 (see Phase 5)."""
        body = request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return _no_cache(jsonify({"ok": False, "error": "missing 'name'"})), 400
        if name not in (arena_state.arena_names() or [name]):
            return _no_cache(jsonify({"ok": False,
                                      "error": f"unknown arena {name!r}",
                                      "available": arena_state.arena_names()})), 400
        sim_result = None
        if _sim_url:
            try:
                req = urllib.request.Request(
                    f"{_sim_url}/api/arena",
                    data=json.dumps({"name": name}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    sim_result = json.loads(
                        resp.read().decode("utf-8", "replace") or "{}")
            except Exception as exc:
                return _no_cache(jsonify({
                    "ok": False,
                    "error": f"sim switch failed: {type(exc).__name__}: {exc}",
                })), 502
        try:
            st = arena_state.set_active(name)
        except Exception as exc:
            return _no_cache(jsonify({
                "ok": False, "error": f"{type(exc).__name__}: {exc}"})), 500
        runner.events.add("arena", f"operator switched arena -> {name}")
        return _no_cache(jsonify({
            "ok": True, "name": st.name, "sim": sim_result,
            "arena": arena_state.to_client_dict(),
        }))

    @app.route("/api/strategy/fc/<fc_name>/endpoint", methods=["POST"])
    def api_fc_endpoint(fc_name):
        """Switch a drone between the SIM and a REAL flight controller by IP.

        Proxies to the C2 (which owns the FC connections). Body:
          {"host": "192.168.42.1", "port": 8080}  -> point at a real FC
          {"reset": true}                          -> back to the sim endpoint
        Team isolation (regs §1.3): only OUR-team (or unassigned) drones. The
        drone's team is taken from our roster, else sniffed from the FC-name
        prefix ('red*'/'blue*') — matching the runner's auto-adopt isolation —
        so we reject the enemy's drones even though they're not in our roster."""
        s = settings.to_dict()
        our_team = s["markers"]["our_team"]
        dd = (s.get("drones") or {}).get(fc_name) or {}
        nl = fc_name.lower()
        sniffed = "red" if nl.startswith("red") else ("blue" if nl.startswith("blue") else None)
        t = dd.get("team") or sniffed
        if t and t != our_team:
            return _no_cache(jsonify({
                "ok": False, "error": "not our drone (team isolation)"})), 403
        body = request.get_json(silent=True) or {}
        reset = bool(body.get("reset"))
        host = body.get("host")
        port = body.get("port")
        try:
            ok, payload = _run_coro(runner._c2.set_fc_endpoint(
                fc_name, host=host, port=port, reset=reset))
        except Exception as exc:
            return _no_cache(jsonify({
                "ok": False, "error": f"{type(exc).__name__}: {exc}"})), 502
        where = "sim (reset)" if reset else (
            (payload or {}).get("base_url") if isinstance(payload, dict) else host)
        runner.events.add("fc", f"{fc_name} flight controller -> {where}")
        return _no_cache(jsonify({"ok": bool(ok), "payload": payload}))

    @app.route("/api/strategy/fc/all", methods=["POST"])
    def api_fc_all():
        """Bulk-switch EVERY flight controller between REAL hardware and the SIM.
        Body: {"mode": "real"|"sim"}.
          real -> point each FC at flightctrl<N>:8080, where N is the trailing
                  digit of the FC name (flightctrl2 -> flightctrl2, sim-3 ->
                  flightctrl3), else its 1-based sorted position.
          sim  -> reset each FC to its configured (sim) endpoint — identical to
                  the per-drone "Use sim" button, applied to all."""
        import re as _re
        body = request.get_json(silent=True) or {}
        mode = str(body.get("mode") or "").lower()
        if mode not in ("real", "sim"):
            return _no_cache(jsonify({
                "ok": False, "error": "mode must be 'real' or 'sim'"})), 400
        try:
            names = sorted(_run_coro(runner._c2.fc_names()) or [])
        except Exception as exc:
            return _no_cache(jsonify({
                "ok": False, "error": f"{type(exc).__name__}: {exc}"})), 502
        if not names:
            return _no_cache(jsonify({
                "ok": False, "error": "no flight controllers known to the C2"})), 404
        results: dict = {}
        for i, fc in enumerate(names, start=1):
            try:
                if mode == "sim":
                    ok, payload = _run_coro(
                        runner._c2.set_fc_endpoint(fc, reset=True))
                else:
                    m = _re.search(r"(\d+)\s*$", fc)
                    n = int(m.group(1)) if m else i
                    ok, payload = _run_coro(runner._c2.set_fc_endpoint(
                        fc, host=f"flightctrl{n}", port=8080))
                results[fc] = {"ok": bool(ok), "payload": payload}
            except Exception as exc:
                results[fc] = {"ok": False,
                               "error": f"{type(exc).__name__}: {exc}"}
        n_ok = sum(1 for r in results.values() if r.get("ok"))
        runner.events.add(
            "fc", f"ALL flight controllers -> {mode} "
                  f"({n_ok}/{len(names)} ok)")
        return _no_cache(jsonify({
            "ok": n_ok == len(names), "mode": mode, "results": results}))

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

    @app.route("/api/strategy/target-marker/<fc_name>", methods=["POST"])
    def api_target_marker(fc_name: str):
        """TESTING: pin (or clear) an explicit target MARKER id for an attacker.
        Body: {"marker_id": int|null}. Bypasses the slot->face mapping."""
        body = request.get_json(silent=True) or {}
        raw = body.get("marker_id", body.get("marker"))
        marker_id: Optional[int]
        if raw is None or raw == "" or raw == "null":
            marker_id = None
        else:
            try:
                marker_id = int(raw)
            except (TypeError, ValueError):
                return _no_cache(jsonify(
                    {"ok": False, "error": "invalid marker_id"})), 400
            if not (0 < marker_id < 1000):
                return _no_cache(jsonify(
                    {"ok": False, "error": "marker_id out of range 1..999"})), 400
        runner.assign_target_marker(fc_name, marker_id)
        return _no_cache(jsonify({"ok": True, "marker_id": marker_id}))

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
        "go_home_angle_deg": d.go_home_angle_deg,
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
  /* Persistent fleet-video strip — rendered ONCE per roster change so
     the MJPEG streams survive the 1Hz state polling re-renders. */
  .video-strip { display: grid; gap: 8px;
                 grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                 margin-bottom: 12px; }
  .video-tile { position: relative; background: #111; border-radius: 6px;
                border: 1px solid var(--border); overflow: hidden;
                aspect-ratio: 16 / 10; cursor: zoom-in; }
  .video-tile.disabled { cursor: default; opacity: .5;
                         border-style: dashed; }
  .video-tile img { width: 100%; height: 100%; object-fit: cover;
                    display: block; background: #222; }
  .video-tile .label { position: absolute; top: 4px; left: 6px;
                       padding: 1px 6px; border-radius: 999px;
                       background: rgba(0,0,0,.55); color: #fff;
                       font-size: 11px; line-height: 1.4; font-weight: 600; }
  .video-tile .label.red { background: var(--red); }
  .video-tile .label.blue { background: var(--blue); }
  .video-tile .nope { position: absolute; inset: 0; display: flex;
                      align-items: center; justify-content: center;
                      color: #666; font-size: 11px; }
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
  .dl-log { display:inline-block; margin-top:6px; font-size:11px;
            color:#7fb3e6; text-decoration:none; border:1px solid #2c3038;
            border-radius:4px; padding:3px 8px; }
  .dl-log:hover { border-color:#3a4350; background:#15171c; }
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
  <span class="pill" id="arena-pill" title="Active arena — shared by both teams. Switching changes the whole stack (sim + both strategies) live, no restart.">arena: <b id="arena-name">–</b></span>
  <span class="small">our team:</span>
  <select id="our-team">
    <option value="red">red</option>
    <option value="blue">blue</option>
  </select>
  <span class="small">arena:</span>
  <select id="arena-sel" title="Switch the WHOLE stack (sim + both strategies) between the Competition and GVZ testing arena — live, no restart"></select>
  <span class="spacer"></span>
  <label class="small" title="TESTING: every attacker loops these marker IDs — capture each, return home (scores), turn, repeat — never landing between attacks.">
    <input type="checkbox" id="test-mode-chk"> Test loop
  </label>
  <input type="text" id="test-markers-inp" placeholder="31,32,33" style="width:6.5em"
         title="Comma-separated marker IDs the test loop attacks, in order.">
  <span class="spacer"></span>
  <button id="mode-btn" class="warn" title="Toggle MANUAL / AUTO">Go AUTO</button>
  <button id="arm-btn" class="primary">Arm</button>
  <button id="disarm-btn">Disarm</button>
  <button id="reset-btn" title="Drop every drone from the roster — auto-adopt then re-fills it from the C2 with fresh defaults. Useful for clearing ghost drones / stale per-drone settings.">Reset roster</button>
  <button id="all-real-btn" title="Point EVERY flight controller at the REAL drones (flightctrl1-5 on :8080).">All real</button>
  <button id="all-sim-btn" title="Point EVERY flight controller back at the SIMULATOR (its configured endpoint).">All sim</button>
  <button id="land-btn" class="danger">Emergency Land</button>
</header>

<main>
  <section>
    <h2>Arena <span class="small" style="color:#888">(top-down · live drone positions)</span></h2>
    <canvas id="arena" width="760" height="400"
            style="width:100%;max-width:760px;background:#0d0f13;border:1px solid #222;border-radius:8px;display:block"></canvas>
  </section>

  <section>
    <h2>Fleet video</h2>
    <div id="videos" class="video-strip"></div>
  </section>

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
async function assignTargetMarker(fc, markerId) {
  return api(`/api/strategy/target-marker/${encodeURIComponent(fc)}`, {
    method:"POST", body: JSON.stringify({marker_id: markerId})});
}

function ageStr(s) {
  if (s == null) return "never";
  if (s < 1) return s.toFixed(1) + "s";
  if (s < 60) return Math.round(s) + "s";
  return Math.round(s/60) + "m";
}

// Build the fleet-video strip ONCE per roster change. Re-rendering on
// every tick would tear down + re-establish every MJPEG connection,
// which causes visible flicker AND quickly piles up dropped sockets on
// the sim. So we cache the last-seen drone + URL set and only rebuild
// the strip when something CHANGES.
let _videoCache = "";  // signature: "fc1=url1|fc2=url2|…"
function renderVideos(state) {
  const root = document.getElementById("videos");
  if (!root) return;
  const drones = state.settings.drones || {};
  const overview = (state.runner && state.runner.overview) || {};
  const names = Object.keys(drones).sort();
  // Build a signature of "fc=videoUrl,team" — when it changes we rebuild,
  // otherwise the existing <img>s keep streaming.
  const sig = names.map(fc => {
    const ov = overview[fc] || {};
    const url = ov.base_url ? `${ov.base_url}/video.mjpg` : "";
    const team = (drones[fc] || {}).team || "";
    const en = (drones[fc] || {}).enabled !== false;   // disabled -> no feed
    return `${fc}=${url}=${team}=${en}`;
  }).join("|");
  if (sig === _videoCache) return;
  _videoCache = sig;
  if (!names.length) {
    root.innerHTML = '<div class="small">No drones — strip will appear once the C2 reports any.</div>';
    return;
  }
  root.innerHTML = names.map(fc => {
    const ov = overview[fc] || {};
    const url = ov.base_url ? `${ov.base_url}/video.mjpg` : "";
    const team = (drones[fc] || {}).team || "";
    const teamCls = team === "red" ? "red" : team === "blue" ? "blue" : "";
    // A DISABLED drone must NOT show any feed — a stale sim stream would
    // confuse the operator's read of the live feeds. Show a placeholder
    // instead of the <img> (no MJPEG connection opened at all).
    const enabled = (drones[fc] || {}).enabled !== false;
    const click = (enabled && url) ? `onclick="window.open('${url}', '_blank')"` : "";
    const content = !enabled
      ? `<div class="nope">disabled</div>`
      : (url
        ? `<img src="${url}" alt="${fc} live feed" onerror="this.style.display='none'; this.nextElementSibling && (this.nextElementSibling.style.display='flex');" />
           <div class="nope" style="display:none">camera offline</div>`
        : `<div class="nope">no video</div>`);
    return `<div class="video-tile${enabled ? "" : " disabled"}" ${click} title="${fc} ${enabled ? "live feed" : "(disabled)"}">
              <span class="label ${teamCls}">${fc}</span>
              ${content}
            </div>`;
  }).join("");
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
    // overview shape: { fc: {base_url, connection_ok, drone_connected,
    //                        state:{phase, mission_script, mission_step_idx, …}} }.
    // Be defensive — fields may be missing while a drone is offline.
    const ov = overview[fc] || {};
    const ovState = (ov && ov.state) || {};
    const tel = (ovState && ovState.telemetry) || {};
    // Video URL: every FC (sim + real Olympe) serves an MJPEG stream at
    // <base_url>/video.mjpg. base_url comes through the overview so it
    // works for sim FCs (127.0.0.1:9101+) and remote ones (sphinx3:8080).
    const videoUrl = ov.base_url ? `${ov.base_url}/video.mjpg` : null;
    // FC source (sim vs real). base_url comes from the C2 overview; a loopback
    // host = the simulator, anything else = a real flight controller by IP.
    const fcBaseUrl = ov.base_url || "—";
    const fcIsSim = /(^|\/\/)(127\.0\.0\.1|localhost|\[?::1\]?)[:/]/.test(fcBaseUrl);
    // Mission script progress (the script currently EXECUTING on the FC,
    // distinct from rs.last_pushed_script which is what we last pushed).
    const fcScript = ovState.mission_script || [];
    const fcStepIdx = ovState.mission_step_idx;
    const fcScriptLen = fcScript.length;
    const fcStepStr = (fcScriptLen > 0 && fcStepIdx != null)
      ? `${fcStepIdx + 1}/${fcScriptLen}` : "—";
    const currentStep = (fcScriptLen > 0 && fcStepIdx != null
                         && fcStepIdx >= 0 && fcStepIdx < fcScriptLen)
      ? fcScript[fcStepIdx] : "";
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
      const A = state.arena || _ARENA_FALLBACK;
      const rh = A.red_home_y || _ARENA_FALLBACK.red_home_y;
      const bh = A.blue_home_y || _ARENA_FALLBACK.blue_home_y;
      const inX = Math.abs(x) <= (A.x_max != null ? A.x_max : 5.0);
      if (team === "red")  inHome = (inX && y >= rh[0] && y <= rh[1]) ? "yes" : "no";
      if (team === "blue") inHome = (inX && y >= bh[0] && y <= bh[1]) ? "yes" : "no";
      inHomeCls = inHome === "yes" ? "good" : (inHome === "no" ? "warn" : "");
    }
    const visible = ovState.visible_marker_ids;
    const visStr = (Array.isArray(visible) && visible.length) ? visible.join(",") : "—";
    const missionRun = fcPhase && !["init", "idle", "done", "", "—"].includes(String(fcPhase).toLowerCase());
    const c2DotTitle = c2Ok ? `C2 link to ${fc}: ok` : `C2 link to ${fc}: DOWN`;
    const droneDotTitle = droneOk ? `drone link: connected` : `drone link: not connected`;
    const lastPush = rs.last_pushed_age_s != null ? `pushed ${ageStr(rs.last_pushed_age_s)} ago` : "";

    // Drone WiFi (which AP this FC's Pi is connected to) — the SSID is the
    // drone's WiFi name, so the operator sees at a glance which drone is on each FC.
    const wlan = ov.wlan || {};
    const wifiName = wlan.ssid || "—";
    const wifiDetail = [
      (wlan.signal_dbm != null && wlan.signal_dbm !== "") ? `${wlan.signal_dbm} dBm` : "",
      wlan.band || "",
      wlan.link_quality ? `Q ${wlan.link_quality}%` : "",
      wlan.bit_rate || "",
    ].filter(Boolean).join("  ");
    const wifiTitle = wlan.ssid
      ? `Drone WiFi: ${wlan.ssid}${wifiDetail ? "  —  " + wifiDetail : ""}`
      : "drone WiFi: unknown";
    const wifiCls = wlan.ssid ? "good" : "warn";

    const html = `
      <div class="drone">
        <div class="head">
          <span class="name">${fc}</span>
          <span class="badge ${teamBadge}">${team || "no team"}</span>
          <span class="badge ${roleBadge}">${role}</span>
          <span class="status-dot ${c2DotCls}" title="${c2DotTitle}"></span>
          <span class="status-dot ${droneDotCls}" title="${droneDotTitle}"></span>
          <span class="badge ${wifiCls}" title="${wifiTitle}" style="margin-left:auto">📶 ${wifiName}</span>
        </div>
        <div class="status-grid">
          <div><span class="k">battery</span><span class="v ${battCls}">${battStr}</span></div>
          <div><span class="k">height</span><span class="v">${heightStr}</span></div>
          <div><span class="k">FC phase</span><span class="v">${fcPhase}${missionRun?" ●":""}</span></div>
          <div><span class="k">in home</span><span class="v ${inHomeCls}">${inHome}</span></div>
          <div><span class="k">world pos</span><span class="v mono">${posStr}</span></div>
          <div><span class="k">sees</span><span class="v mono" title="visible ArUco IDs">${visStr}</span></div>
          <div><span class="k">wifi</span><span class="v mono ${wifiCls}" title="${wifiTitle}">${wifiName}${wlan.signal_dbm!=null&&wlan.signal_dbm!==""?` (${wlan.signal_dbm} dBm)`:""}</span></div>
        </div>
        <div class="mission-row" style="margin-top:6px">
          <div class="small" title="The mission script currently EXECUTING on the FC (from C2 overview), with the current step highlighted.">
            <b>script ${fcStepStr}</b>
            ${currentStep ? `<span class="mono" style="color:var(--green,#5a5)"> · ${currentStep.replace(/</g,'&lt;')}</span>` : ""}
          </div>
          ${rs.last_pushed_script ? `<details style="margin-top:4px">
            <summary class="small" style="cursor:pointer">view full pushed script (${rs.last_pushed_script.split('\\n').filter(Boolean).length} lines)</summary>
            <pre class="mono" style="font-size:11px;max-height:140px;overflow:auto;
                                     background:#111;padding:6px;border-radius:4px;
                                     margin:4px 0 0">${rs.last_pushed_script.replace(/</g,'&lt;')}</pre>
          </details>` : ""}
          <a class="dl-log" href="/api/missions/${fc}/download" download="${fc}_commands.log"
             title="Download this drone's COMPLETE command list for the whole game as a .log file — replay it on a real drone.">⤓ download command list</a>
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
          <div>
            <label title="RTH GO_HOME arrival bearing around the home wall marker (-80..+80°). Blank = auto-fan (-45/0/+45).">GO_HOME angle (°)</label>
            <input type="number" step="5" min="-80" max="80" placeholder="auto"
                   value="${d.go_home_angle_deg==null?'':d.go_home_angle_deg}"
                   data-fc="${fc}" data-field="go_home_angle_deg">
          </div>
          <div>
            <label title="Scout rotation speed: yaw-stick 1..100. The drone spins at ~(this/100) × the FC max rotation speed (~150°/s). Higher = faster scanning.">Scout rot. speed</label>
            <input type="number" step="5" min="1" max="100"
                   value="${d.scout_yaw_stick}" data-fc="${fc}" data-field="scout_yaw_stick">
          </div>
        </div>
        <div class="fc-source" style="margin-top:6px">
          <label class="small">Flight controller
            <span class="badge" style="background:${fcIsSim?'#3a566e':'#1b6e2c'};color:#fff">${fcIsSim?'SIM':'REAL'}</span>
            <span class="mono small" style="color:#8a93a0"> ${fcBaseUrl}</span>
          </label>
          <div style="display:flex;gap:4px;margin-top:2px">
            <input class="fc-ip" data-fc="${fc}" placeholder="real FC IP (e.g. 192.168.42.1 or 192.168.42.1:8080)"
                   style="flex:1;min-width:0" title="Set this drone's flight-controller IP to use a REAL drone instead of the simulator.">
            <button class="fc-ip-apply" data-fc="${fc}" title="Point this drone at a REAL flight controller by IP">Use real</button>
            <button class="fc-ip-reset" data-fc="${fc}" title="Restore the simulator endpoint for this drone">Use sim</button>
          </div>
        </div>
        <div class="target-row" ${(role!=="attacker"&&role!=="defender")?'style="display:none"':''}>
          <label class="small">${role==="defender"?"Defend slot:":"Target slot:"}</label>
          <select data-target="${fc}">${slotOpts}</select>
          <button data-clear="${fc}">Clear</button>
        </div>
        <div class="target-row" ${role!=="attacker"?'style="display:none"':''}
             title="TESTING: fly the attack script straight at this marker id, bypassing the slot->face mapping. One-shot.">
          <label class="small">Test marker:</label>
          <input type="number" min="1" max="999" step="1" data-marker-input="${fc}"
                 value="${rs.target_marker_id==null?'':rs.target_marker_id}"
                 placeholder="id" style="width:5em">
          <button data-marker-go="${fc}">Attack</button>
          <button data-marker-clear="${fc}">Clear</button>
        </div>
        <div class="phase" title="${reason}">
          phase=${phase}${rs.target_marker_id!=null?` <b>test→${rs.target_marker_id}</b>`:''}${rs.target_slot!=null?` slot=${rs.target_slot}`:''}${attackId?` last_aim=${attackId}`:''}${cruiseAlt!=null?` · cruise ${cruiseAlt}m`:''}${lastPush?` · ${lastPush}`:''}
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
  // Active arena (shared) + keep the switch dropdown in sync (follows the sim,
  // so a switch by the other team is reflected here too).
  const arena = state.arena || {};
  const an = document.getElementById("arena-name");
  if (an) an.textContent = arena.name || "–";
  const asel = document.getElementById("arena-sel");
  if (asel && !asel.dataset.busy) {
    const avail = arena.available || (arena.name ? [arena.name] : []);
    const labels = arena.labels || {};
    const want = avail.join(",");
    if (asel.dataset.opts !== want) {       // rebuild options only when the set changes
      asel.innerHTML = "";
      for (const nm of avail) {
        const o = document.createElement("option");
        o.value = nm; o.textContent = labels[nm] || nm;
        asel.appendChild(o);
      }
      asel.dataset.opts = want;
    }
    if (arena.name && asel.value !== arena.name) asel.value = arena.name;
  }
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
  // Test-loop control: reflect current match settings unless the operator is
  // editing the field/checkbox right now.
  const match = (state.settings && state.settings.match) || {};
  const tChk = document.getElementById("test-mode-chk");
  const tInp = document.getElementById("test-markers-inp");
  if (tChk && document.activeElement !== tChk) tChk.checked = !!match.test_mode;
  if (tInp && document.activeElement !== tInp) {
    tInp.value = (match.test_markers || []).join(",");
  }
}

async function applyTestMode() {
  const tChk = document.getElementById("test-mode-chk");
  const tInp = document.getElementById("test-markers-inp");
  const body = { test_mode: !!tChk.checked, test_markers: (tInp.value || "").trim() };
  if (body.test_mode && !body.test_markers) {
    alert("Enter the marker IDs to test-attack (e.g. 31,32,33) first.");
    tChk.checked = false; return;
  }
  await api("/api/settings/match", {method:"POST", body: JSON.stringify(body)});
  refresh();
}
document.getElementById("test-mode-chk").addEventListener("change", applyTestMode);
document.getElementById("test-markers-inp").addEventListener("change", applyTestMode);

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

// ── 2D top-down arena view ───────────────────────────────────────────────
// Self-contained canvas (no external deps -> always loads offline at the
// arena). Arena frame: long axis Y in [-10,+10] (20 m), width X in [-5,+5]
// (10 m). Red home y in [-10,-5], blue home y in [+5,+10], neutral in between.
// We draw with Y as the HORIZONTAL screen axis (the arena is wide & short on
// screen) so the 20 m length uses the canvas width.
// Fallback geometry (used only if the server didn't send state.arena, e.g. an
// older server). Normally the canvas is DATA-DRIVEN from state.arena so it
// redraws the right markers/boxes/home-bands when the arena is switched live.
const _ARENA_FALLBACK = {
  y_min:-10, y_max:10, x_min:-5, x_max:5,
  red_home_y:[-10,-5], blue_home_y:[5,10],
  boxes: {1:[-3,-6.5], 2:[0,-9], 3:[3,-6.5], 4:[-3,6.5], 5:[0,9], 6:[3,6.5]},
};
function renderArena(state) {
  const cv = document.getElementById("arena");
  if (!cv) return;
  const A = state.arena || _ARENA_FALLBACK;
  const ARENA = { yMin:A.y_min, yMax:A.y_max, xMin:A.x_min, xMax:A.x_max };
  const BOXES = A.boxes || _ARENA_FALLBACK.boxes;
  const redHome = A.red_home_y || _ARENA_FALLBACK.red_home_y;
  const blueHome = A.blue_home_y || _ARENA_FALLBACK.blue_home_y;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 26;
  // arena Y -> screen X ; arena X -> screen Y (flip so +X is up on screen)
  const sx = y => pad + (y - ARENA.yMin) / (ARENA.yMax - ARENA.yMin) * (W - 2*pad);
  const sy = x => pad + (ARENA.xMax - x) / (ARENA.xMax - ARENA.xMin) * (H - 2*pad);
  ctx.clearRect(0, 0, W, H);

  // zones: red home (left), neutral, blue home (right) — from active arena
  const yTop = sy(ARENA.xMax), yBot = sy(ARENA.xMin);
  ctx.fillStyle = "rgba(192,57,43,0.13)";
  ctx.fillRect(sx(redHome[0]), yTop, sx(redHome[1])-sx(redHome[0]), yBot-yTop);
  ctx.fillStyle = "rgba(36,117,194,0.13)";
  ctx.fillRect(sx(blueHome[0]), yTop, sx(blueHome[1])-sx(blueHome[0]), yBot-yTop);
  // arena border
  ctx.strokeStyle = "#2c3038"; ctx.lineWidth = 1;
  ctx.strokeRect(sx(ARENA.yMin), yTop, sx(ARENA.yMax)-sx(ARENA.yMin), yBot-yTop);
  // home-boundary lines
  ctx.setLineDash([4,4]); ctx.strokeStyle = "#444";
  for (const yb of [redHome[1], blueHome[0]]) { ctx.beginPath(); ctx.moveTo(sx(yb),yTop); ctx.lineTo(sx(yb),yBot); ctx.stroke(); }
  ctx.setLineDash([]);

  const ourTeam = (state.settings && state.settings.markers && state.settings.markers.our_team) || "";
  const slots = state.slots || {};
  // target boxes, coloured by current holder
  for (const [sl,[bx,by]] of Object.entries(BOXES)) {
    const holder = (slots[sl] && slots[sl].holder) || "unknown";
    const col = holder === "red" ? "#c0392b" : holder === "blue" ? "#2475c2" : "#666";
    const px = sx(by), py = sy(bx), r = 9;
    ctx.fillStyle = col; ctx.fillRect(px-r, py-r, 2*r, 2*r);
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.strokeRect(px-r, py-r, 2*r, 2*r);
    ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.textAlign = "center";
    ctx.fillText("#"+sl, px, py+3);
  }

  // drones from the live C2 overview (our team only — enemy is filtered server-side)
  const ov = (state.runner && state.runner.overview) || {};
  const drones = (state.settings && state.settings.drones) || {};
  const roleStates = (state.runner && state.runner.drones) || {};
  for (const fc of Object.keys(ov)) {
    const st = (ov[fc] && ov[fc].state) || {};
    const pos = st.world_position_m;
    if (!pos || pos.length < 2) continue;
    const team = (drones[fc] && drones[fc].team) || ourTeam;
    const col = team === "blue" ? "#3a9bf0" : "#e74c3c";
    const px = sx(pos[1]), py = sy(pos[0]);
    // heading arrow: heading_deg is CW from +Y (arena front). On screen +Y is
    // rightward, +X is up. forward = (sin h, cos h) in (X,Y) arena -> screen.
    const h = (st.heading_deg || 0) * Math.PI / 180;
    const fY = Math.cos(h), fX = Math.sin(h);          // arena forward unit
    const dpx = (fY) , dpy = (-fX);                    // to screen dir (Y->x, X-> -y)
    const L = 13;
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px+dpx*L, py+dpy*L); ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.textAlign = "center";
    const rp = (roleStates[fc] && roleStates[fc].role) || "";
    ctx.fillText(fc + (rp?(" ("+rp[0]+")"):""), px, py-9);
  }
  // legend
  ctx.fillStyle = "#888"; ctx.font = "10px system-ui"; ctx.textAlign = "left";
  ctx.fillText("◀ red home", sx(ARENA.yMin)+3, yBot+14);
  ctx.textAlign = "right";
  ctx.fillText("blue home ▶", sx(ARENA.yMax)-3, yBot+14);
}

async function refresh() {
  try {
    const state = await api("/api/state");
    last = state;
    renderHeader(state);
    renderArena(state);
    renderVideos(state);   // signature-cached → only rebuilds when roster changes
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
    if (field === "scout_yaw_stick") val = parseInt(val, 10);
    // GO_HOME angle: blank -> null (auto-fan), else numeric.
    if (field === "go_home_angle_deg") val = (val === "" ? null : parseFloat(val));
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
  // TESTING: launch / clear an explicit target-marker attack.
  if (t.matches("[data-marker-go]")) {
    const fc = t.getAttribute("data-marker-go");
    const inp = document.querySelector(`[data-marker-input="${fc}"]`);
    const raw = (inp && inp.value || "").trim();
    const id = raw === "" ? null : parseInt(raw, 10);
    if (id == null || !(id > 0)) { alert("Enter a marker id (1..999) first."); return; }
    await assignTargetMarker(fc, id);
    refresh();
  }
  if (t.matches("[data-marker-clear]")) {
    const fc = t.getAttribute("data-marker-clear");
    await assignTargetMarker(fc, null);
    refresh();
  }
  // Switch a drone to a REAL flight controller by IP.
  if (t.matches(".fc-ip-apply")) {
    const fc = t.getAttribute("data-fc");
    const inp = document.querySelector(`.fc-ip[data-fc="${fc}"]`);
    const host = (inp && inp.value || "").trim();
    if (!host) { alert("Enter the real flight-controller IP first."); return; }
    t.disabled = true;
    try {
      const r = await api(`/api/strategy/fc/${encodeURIComponent(fc)}/endpoint`,
                          {method:"POST", body: JSON.stringify({host})});
      if (r && r.ok === false) alert("Set FC IP failed: " + (r.error || (r.payload && r.payload.error) || "?"));
      else if (inp) inp.value = "";
    } catch (err) { alert("Set FC IP error: " + err); }
    finally { t.disabled = false; refresh(); }
  }
  // Restore the simulator endpoint for a drone.
  if (t.matches(".fc-ip-reset")) {
    const fc = t.getAttribute("data-fc");
    t.disabled = true;
    try {
      const r = await api(`/api/strategy/fc/${encodeURIComponent(fc)}/endpoint`,
                          {method:"POST", body: JSON.stringify({reset:true})});
      if (r && r.ok === false) alert("Reset to sim failed: " + (r.error || (r.payload && r.payload.error) || "?"));
    } catch (err) { alert("Reset to sim error: " + err); }
    finally { t.disabled = false; refresh(); }
  }
});

document.getElementById("mode-btn").addEventListener("click", async () => {
  // Toggle based on the last known mode.
  const auto = !!(last && last.runner && last.runner.auto);
  await api("/api/strategy/mode", {method:"POST", body: JSON.stringify({auto: !auto})});
  refresh();
});
document.getElementById("arena-sel").addEventListener("change", async (e) => {
  // Switch the WHOLE stack to the chosen arena (the sim swaps its physical
  // world; both strategies follow). Live, no restart.
  const sel = e.target, name = sel.value;
  sel.dataset.busy = "1"; sel.disabled = true;
  try {
    const r = await api("/api/arena/switch", {method:"POST", body: JSON.stringify({name})});
    if (r && r.ok === false) alert("Arena switch failed: " + (r.error || "?"));
  } catch (err) {
    alert("Arena switch error: " + err);
  } finally {
    delete sel.dataset.busy; sel.disabled = false; refresh();
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
async function switchAllFc(mode) {
  const label = mode === "real" ? "REAL drones (flightctrl1-5:8080)" : "the SIMULATOR";
  if (!confirm(`Point EVERY flight controller at ${label}?`)) return;
  const btn = document.getElementById(mode === "real" ? "all-real-btn" : "all-sim-btn");
  if (btn) btn.disabled = true;
  try {
    const r = await api("/api/strategy/fc/all", {method:"POST", body: JSON.stringify({mode})});
    console.log(`all ${mode}:`, r);
    if (r && r.ok === false) alert(`Some FCs failed to switch to ${mode}. See console / event log.`);
  } catch (e) {
    alert(`All-${mode} failed: ${e}`);
  } finally {
    if (btn) btn.disabled = false;
    refresh();
  }
}
document.getElementById("all-real-btn").addEventListener("click", () => switchAllFc("real"));
document.getElementById("all-sim-btn").addEventListener("click", () => switchAllFc("sim"));

// Keyboard killswitch: "?" emergency-lands the WHOLE swarm immediately — the
// same panic key as marker_mission's per-drone UI. Fires on every "?" press
// (any focus, incl. while typing in a field — emergency stop must always win);
// only Ctrl/Cmd/Alt+"?" is left to the browser. The land request goes out
// before the banner/beep so a press always reaches the swarm. Additive: we do
// NOT preventDefault, so "?" still types normally where intended.
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;     // browser shortcuts win
  if ((e.key || "") !== "?") return;
  // Fire FIRST — disarms + emergency-lands every drone on the server.
  try { api("/api/strategy/emergency-land", {method:"POST"}); } catch (err) {}
  // Visual confirmation banner.
  try {
    const b = document.createElement("div");
    b.textContent = "EMERGENCY LAND (?) — landing all drones";
    b.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;"
      + "background:#f87171;color:#240707;text-align:center;padding:.6rem;"
      + "font-weight:700;font-size:1rem;letter-spacing:.02em;";
    document.body.appendChild(b);
    setTimeout(() => b.remove(), 4000);
  } catch (err) {}
  // Audible low tone, distinct from any UI click.
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    if (ac.state === "suspended") ac.resume();
    const o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = 220; o.type = "sine";
    o.connect(g); g.connect(ac.destination);
    const t0 = ac.currentTime;
    g.gain.setValueAtTime(0.18, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.35);
    o.start(t0); o.stop(t0 + 0.4);
  } catch (err) {}
  refresh();
});

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
