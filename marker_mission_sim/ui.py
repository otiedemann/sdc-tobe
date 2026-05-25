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
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

from .config import SimConfig
from .world import World

WEB_DIR = Path(__file__).resolve().parent / "web"


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
