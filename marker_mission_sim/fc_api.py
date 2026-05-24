"""Per-drone marker_mission flight-controller HTTP API, backed by the live World.

``make_fc_app(world, drone_id, cfg)`` returns a fully-wired Flask app that
serves exactly the HTTP surface ``marker_mission_c2`` (``fc_client.py``) reads
and writes, so the C2 talks to a simulated drone the same way it talks to a real
FC. Unlike ``marker_mission_c2/tools/fake_fc.py`` (whose state is static), every
read here is sourced from the shared :class:`World`:

  * ``GET /api/state`` returns ``drone.to_state_dict(world.visible_marker_ids(...))``
    so phase, visible markers, world position and telemetry all reflect the
    live simulation as it ticks.
  * ``POST /api/start`` / ``POST /api/stop`` drive ``world.queue_script`` /
    ``world.stop_script`` for this drone.

Per-app mutable state (the active arena dict, the tune dict, and the active
mission script text) lives in closures created below, so two drones served by
two apps keep independent copies. Every handler returns JSON via ``jsonify`` and
never raises: risky work is wrapped and surfaced as ``500 {"ok": False,
"error": <str>}``.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, Response, jsonify, request

from .config import SimConfig
from .world import World

# A 1x1 white JPEG (same bytes as tools/fake_fc.py) used for the /video.mjpg
# stub so the C2's calibrate iframe / overview video <img> doesn't 404.
_WHITE_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb0043"
    "00080606070605080707070909080a0c140d0c0b0b0c1912"
    "130f141d1a1f1e1d1a1c1c20242e2720222c231c1c283728"
    "2c30313434341f273a3d3832dc323434ffc0000b08000100"
    "01010110ffc4001f0000010501010101010100000000000000"
    "000102030405060708090a0bffc400b510000201030302040305"
    "0504040000017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a161718191a25"
    "262728292a3435363738393a434445464748494a535455565758"
    "595a636465666768696a737475767778797a838485868788898a"
    "92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9"
    "bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7"
    "e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbffd9"
)

_DEFAULT_SCRIPT = "TAKEOFF\nHOOVER 2\nLAND\n"


def _seed_arena(cfg: SimConfig) -> dict:
    """Build the initial /api/arena/active dict in the marker_mission JSON shape.

    Dimensions come from ``cfg.arena`` and the markers list is the arena's wall
    markers (id / wall / x,y,z). ``top_z_m`` / ``bottom_z_m`` describe the wall
    marker height band; we derive them from the actual marker z values (falling
    back to sensible defaults), and the marker size from the capture config-free
    default used by marker_mission.
    """
    arena = cfg.arena
    markers = []
    zs: list[float] = []
    for m in arena.wall_markers:
        zs.append(float(m.pos.z))
        markers.append({
            "id": int(m.id),
            "wall": str(m.wall),
            "x": round(float(m.pos.x), 4),
            "y": round(float(m.pos.y), 4),
            "z": round(float(m.pos.z), 4),
        })
    top_z = round(max(zs), 4) if zs else 4.0
    bottom_z = round(min(zs), 4) if zs else 2.0
    return {
        "width_m": float(arena.width_m),
        "depth_m": float(arena.depth_m),
        "top_z_m": top_z,
        "bottom_z_m": bottom_z,
        "marker_size_m": 0.5,
        "markers": markers,
    }


def _default_tune() -> dict:
    """Sensible sim defaults for the tuner dict (marker_mission shape)."""
    return {
        "fwd_rc_max": 10,
        "lat_rc_max": 10,
        "fwd_kp": 30,
        "lat_kp": 30,
        "fwd_kv": 0.4,
        "lat_kv": 0.4,
        "distance_floor_factor": 0.7,
        "target_distance_m": 2.0,
        "killswitch_key": "@",
        "target_marker_id": 4,
    }


def make_fc_app(world: World, drone_id: str, cfg: SimConfig) -> Flask:
    """Return a Flask app serving one simulated drone's FC HTTP API."""
    app = Flask(f"sim_fc_{drone_id}")

    # ---- per-app mutable state (closures) --------------------------------
    active_arena: dict = _seed_arena(cfg)
    tune_values: dict = _default_tune()
    # Single-element list so nested handlers can rebind the string in place.
    active_script: list[str] = [_DEFAULT_SCRIPT]

    def _fail(e: Exception):
        return jsonify({"ok": False, "error": str(e)}), 500

    # ---- telemetry -------------------------------------------------------
    @app.get("/api/state")
    def api_state():
        try:
            drone = world.get_drone(drone_id)
            if drone is None:
                return jsonify({"ok": False, "error": "unknown drone"}), 503
            visible = world.visible_marker_ids(drone_id)
            return jsonify(drone.to_state_dict(visible))
        except Exception as e:  # never raise
            return _fail(e)

    @app.get("/api/identity")
    def api_identity():
        try:
            return jsonify({
                "hostname": f"sim-{drone_id}",
                "drone_serial": drone_id,
                "drone_connected": True,
                "marker_mission_version": "sim",
            })
        except Exception as e:
            return _fail(e)

    # ---- mission control -------------------------------------------------
    @app.post("/api/start")
    def api_start():
        try:
            body = request.get_json(silent=True) or {}
            script_text = body.get("script")
            if not script_text:
                script_text = active_script[0] or ""
            world.queue_script(drone_id, script_text)
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    @app.post("/api/stop")
    def api_stop():
        try:
            if not world.stop_script(drone_id):
                return jsonify({"ok": False,
                                "error": "mission not running"}), 409
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    # ---- arena -----------------------------------------------------------
    @app.get("/api/arena/active")
    def api_arena_get():
        try:
            return jsonify(active_arena)
        except Exception as e:
            return _fail(e)

    @app.post("/api/arena/active")
    def api_arena_set():
        try:
            body = request.get_json(silent=True)
            if isinstance(body, dict):
                active_arena.clear()
                active_arena.update(body)
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    # ---- tune ------------------------------------------------------------
    @app.get("/api/tune")
    def api_tune_get():
        try:
            return jsonify({"current_values": tune_values})
        except Exception as e:
            return _fail(e)

    @app.post("/api/tune/apply")
    def api_tune_apply():
        try:
            body = request.get_json(silent=True) or {}
            if isinstance(body, dict):
                for k, v in body.items():
                    tune_values[k] = v
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    @app.post("/api/tune/save")
    def api_tune_save():
        try:
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    # ---- mission script --------------------------------------------------
    @app.get("/api/mission/script")
    def api_script_get():
        try:
            return jsonify({"text": active_script[0]})
        except Exception as e:
            return _fail(e)

    @app.post("/api/mission/script")
    def api_script_set():
        try:
            body = request.get_json(silent=True) or {}
            active_script[0] = str(body.get("text") or "")
            return jsonify({"ok": True})
        except Exception as e:
            return _fail(e)

    @app.get("/api/mission/scripts")
    def api_scripts_list():
        try:
            return jsonify({"scripts": []})
        except Exception as e:
            return _fail(e)

    @app.post("/api/mission/scripts/<name>/load")
    def api_scripts_load(name: str):
        try:
            return jsonify({"text": active_script[0]})
        except Exception as e:
            return _fail(e)

    # ---- calibration stubs ----------------------------------------------
    @app.get("/api/calibrate/files")
    def api_calibrate_list():
        try:
            return jsonify({"files": []})
        except Exception as e:
            return _fail(e)

    @app.get("/api/calibrate/files/<name>")
    def api_calibrate_get(name: str):
        try:
            return jsonify({"ok": False, "error": "not found"}), 404
        except Exception as e:
            return _fail(e)

    @app.post("/api/calibrate/files/<name>")
    def api_calibrate_put(name: str):
        try:
            return jsonify({"ok": True, "name": name})
        except Exception as e:
            return _fail(e)

    @app.delete("/api/calibrate/files/<name>")
    def api_calibrate_delete(name: str):
        try:
            return jsonify({"ok": False, "error": "not found"}), 404
        except Exception as e:
            return _fail(e)

    # ---- video stub ------------------------------------------------------
    @app.get("/video.mjpg")
    def video_mjpg():
        # Single-frame multipart stream of a 1x1 white JPEG. One frame only so
        # the generator finishes immediately and never blocks the worker.
        def gen():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + _WHITE_JPEG + b"\r\n")
        return Response(
            gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .config import SimConfig, SimDroneConfig

    drone_id = "SIM1"
    cfg = SimConfig(drones=[SimDroneConfig(id=drone_id, team="red", fc_port=9001)])
    world = World(cfg)
    app = make_fc_app(world, drone_id, cfg)
    client = app.test_client()

    print("=== marker_mission_sim.fc_api smoke test ===")

    r = client.get("/api/state")
    st: dict[str, Any] = r.get_json()
    assert r.status_code == 200, r.status_code
    assert "visible_marker_ids" in st, st
    assert "phase" in st, st
    print(f"GET /api/state -> {r.status_code} phase={st['phase']!r} "
          f"visible_marker_ids={st['visible_marker_ids']} "
          f"world_position_m={st['world_position_m']}")

    r = client.get("/api/identity")
    print(f"GET /api/identity -> {r.status_code} {r.get_json()}")

    r = client.post("/api/start", json={"script": "TAKEOFF\nHOOVER 1\nLAND\n"})
    body = r.get_json()
    assert r.status_code == 200 and body.get("ok") is True, body
    print(f"POST /api/start -> {r.status_code} {body}")

    # Advance the sim past cmd_latency so the queued script applies, and show
    # the phase moving off "init".
    for _ in range(40):
        world.tick(0.1)
    r = client.get("/api/state")
    st = r.get_json()
    print(f"GET /api/state (after 40x tick 0.1s) -> {r.status_code} "
          f"phase={st['phase']!r} mission_step_idx={st['mission_step_idx']} "
          f"flying={st['telemetry']['flying']} "
          f"height_cm={st['telemetry']['height_cm']}")

    r = client.post("/api/stop")
    print(f"POST /api/stop -> {r.status_code} {r.get_json()}")

    # A second stop should be a 409 no-op (mission already done/idle).
    r = client.post("/api/stop")
    print(f"POST /api/stop (again) -> {r.status_code} {r.get_json()}")

    r = client.get("/api/arena/active")
    arena = r.get_json()
    print(f"GET /api/arena/active -> {r.status_code} "
          f"width_m={arena.get('width_m')} depth_m={arena.get('depth_m')} "
          f"n_markers={len(arena.get('markers', []))}")

    print("=== all smoke-test assertions passed ===")
