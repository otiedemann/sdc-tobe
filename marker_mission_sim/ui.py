"""Live 3D web view for the simulator.

Serves a small Three.js dashboard (no build step — modules + import map
from a CDN) plus a JSON feed of the world. The static page fetches
``/api/world`` once to build the arena geometry, then either subscribes to
``/api/world/stream`` (Server-Sent Events) or polls ``/api/world`` to keep
drone poses, box colours and the HUD live.

Public interface (kept stable for the launcher)::

    from marker_mission_sim.ui import make_ui_app
    app = make_ui_app(world, cfg)

The :class:`~marker_mission_sim.world.World` is read-only here — we only
ever call ``world.world_snapshot()``.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .config import (
    SimConfig, arena_names, default_arena_name, load_arena_by_name,
)
from .world import World

WEB_DIR = Path(__file__).resolve().parent / "web"

# Well-known local strategy servers (must match run_local.sh + hub.html).
# The hub's "Arm both teams" button proxies through the sim to these so the
# browser makes a single same-origin call (no cross-origin/CORS to the
# strategy ports). Override via the SDC_STRATEGY_PORTS env if you remap them.
STRATEGY_PORTS = {"red": 8091, "blue": 8092}


def make_ui_app(world: World, cfg: SimConfig) -> Flask:
    """Build the Flask app that serves the 3D view + world feed."""
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index() -> Response:
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/hub")
    def hub() -> Response:
        # Single landing page linking to every local service (C2,
        # strategy, sim 3D view, and the per-drone marker_mission FCs).
        return send_from_directory(WEB_DIR, "hub.html")

    @app.get("/web/<path:filename>")
    def web_static(filename: str) -> Response:
        return send_from_directory(WEB_DIR, filename)

    @app.get("/api/world")
    def api_world():
        return jsonify(world.world_snapshot())

    @app.get("/api/world/stream")
    def api_world_stream() -> Response:
        """Push the world snapshot ~5x/s as Server-Sent Events.

        The generator simply exits when the client disconnects (Flask
        raises while writing to the closed socket), so there is nothing to
        clean up explicitly.
        """

        def gen():
            while True:
                payload = json.dumps(world.world_snapshot())
                yield f"data: {payload}\n\n"
                time.sleep(0.2)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # disable proxy buffering
            "Connection": "keep-alive",
        }
        return Response(gen(), mimetype="text/event-stream", headers=headers)

    @app.get("/api/arena")
    def api_arena_get():
        """Active arena name + the registry's available names."""
        return jsonify({
            "ok": True,
            "active": world.arena_name,
            "available": arena_names() or [world.arena_name],
            "default": default_arena_name(),
        })

    @app.post("/api/arena")
    def api_arena_set():
        """Hot-swap the WHOLE physical arena (wall markers + boxes) live.

        Body ``{"name": "real"|"gvz"}`` (a registry name). Rebuilds the
        ArenaDef from ``arenas.json`` and swaps it into the running World
        without dropping drones. The strategies follow within a few ticks by
        reading ``/api/world``'s ``arena_name``.
        """
        body = request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return jsonify({"ok": False, "error": "missing 'name'"}), 400
        avail = arena_names()
        if avail and name not in avail:
            return jsonify({"ok": False,
                            "error": f"unknown arena {name!r}",
                            "available": avail}), 400
        try:
            new_arena = load_arena_by_name(name)
        except Exception as exc:  # bad/missing config file
            return jsonify({"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}), 500
        result = world.swap_arena(new_arena, name)
        return jsonify(result)

    @app.post("/api/match/<action>")
    def api_match(action: str):
        """Arm/disarm BOTH strategy servers in one click (hub convenience).

        The hub is served from the sim, but the Arm/Disarm endpoints live on
        the strategy servers (:8091 red, :8092 blue) — a different origin. To
        avoid CORS in the browser we proxy here: the page POSTs same-origin to
        the sim, and the sim relays to each strategy server-side and reports a
        per-team result. Arming both at once kicks off a full simulated match
        (both strategies default to AUTO, so Arm is the only step needed).
        """
        if action not in ("arm", "disarm"):
            return jsonify({"ok": False, "error": f"unknown action {action!r}"}), 400

        # Optional ?teams=red,blue to target a subset (defaults to both).
        want = (request.args.get("teams") or "").strip()
        teams = ([t for t in want.split(",") if t in STRATEGY_PORTS]
                 if want else list(STRATEGY_PORTS))

        results: dict[str, dict] = {}
        ok_any = False
        for team in teams:
            port = STRATEGY_PORTS[team]
            url = f"http://127.0.0.1:{port}/api/strategy/{action}"
            try:
                req = urllib.request.Request(url, data=b"", method="POST")
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    body = resp.read().decode("utf-8", "replace")
                payload = json.loads(body) if body.strip() else {}
                results[team] = {"ok": True, "armed": payload.get("armed")}
                ok_any = True
            except Exception as exc:  # unreachable / not started / error
                results[team] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        status = 200 if ok_any else 502
        return jsonify({"ok": ok_any, "action": action, "teams": results}), status

    return app


# ---------------------------------------------------------------------------
# Self-test: build a world, make the app, exercise the routes via test_client.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    cfg = SimConfig()
    world = World(cfg)
    app = make_ui_app(world, cfg)
    client = app.test_client()

    r = client.get("/")
    assert r.status_code == 200, f"GET / -> {r.status_code}"
    body = r.get_data(as_text=True).lower()
    assert "<html" in body and "<canvas" in body, "index.html looks wrong"

    r = client.get("/api/world")
    assert r.status_code == 200, f"GET /api/world -> {r.status_code}"
    snap = r.get_json()
    for key in ("arena", "boxes", "drones", "wall_markers", "t"):
        assert key in snap, f"/api/world missing {key!r}"
    arena = snap["arena"]
    for key in ("width_m", "depth_m", "ceiling_m"):
        assert key in arena, f"arena missing {key!r}"

    r = client.get("/web/app.js")
    assert r.status_code == 200, f"GET /web/app.js -> {r.status_code}"

    print("ui.py self-test OK:")
    print(f"  arena      {arena['width_m']}x{arena['depth_m']}x{arena['ceiling_m']} m")
    print(f"  markers    {len(snap['wall_markers'])}")
    print(f"  boxes      {len(snap['boxes'])}")
    print(f"  drones     {len(snap['drones'])}")


if __name__ == "__main__":
    _self_test()
