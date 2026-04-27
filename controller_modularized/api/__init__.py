"""Flask blueprints, one per concern area.

PR #3 introduces the blueprint architecture: every route is decorated on
a blueprint object instead of directly on the Flask app, and the
``register_all`` factory wires them all up.

PR #4 will move each blueprint's route bodies into the corresponding
``api/<area>.py`` module (currently the bodies still live in
``remote_web_controller.py``).
"""
from __future__ import annotations

from flask import Blueprint


# ── Blueprint objects ─────────────────────────────────────────────
# No url_prefix — each route's rule still spells out its full URL, so
# this refactor doesn't change a single client-visible path.
bp_system    = Blueprint("system",    __name__)
bp_drones    = Blueprint("drones",    __name__)
bp_flight    = Blueprint("flight",    __name__)
bp_safety    = Blueprint("safety",    __name__)
bp_camera    = Blueprint("camera",    __name__)
bp_telemetry = Blueprint("telemetry", __name__)
bp_settings  = Blueprint("settings",  __name__)
bp_magneto   = Blueprint("magneto",   __name__)
bp_logs      = Blueprint("logs",      __name__)
bp_arena     = Blueprint("arena",     __name__)
bp_aruco     = Blueprint("aruco",     __name__)
bp_missions  = Blueprint("missions",  __name__)


_ALL_BLUEPRINTS = (
    bp_system, bp_drones, bp_flight, bp_safety, bp_camera, bp_telemetry,
    bp_settings, bp_magneto, bp_logs, bp_arena, bp_aruco, bp_missions,
)


def register_all(app) -> None:
    """Register every blueprint on the given Flask app. Call once at
    boot, after every route handler has been imported (decoration is
    when ``@bp_X.get(...)`` actually attaches the rule to the
    blueprint)."""
    for bp in _ALL_BLUEPRINTS:
        app.register_blueprint(bp)


def assign_blueprint(url: str) -> str:
    """Map a URL to its blueprint name. Used by tooling that rewrites
    @app.<method> decorators to @bp_<x>.<method>. Keep the rules in sync
    with how the routes are conceptually grouped — the URL prefix is
    almost always the deciding factor."""
    p = url
    if p in ("/", "/logo.png", "/flight_log_viewer"):
        return "system"
    if (p.startswith("/proxy/git_rev")
            or p.startswith("/proxy/c2_version")
            or p.startswith("/proxy/diagnostics")
            or p.startswith("/proxy/fc_version")):
        return "system"
    if p.startswith("/proxy/drones") or p.startswith("/proxy/switch"):
        return "drones"
    if (p.startswith("/proxy/pause")
            or p.startswith("/proxy/resume")
            or p.startswith("/proxy/safety/")
            or p.startswith("/proxy/config/ceiling")
            or p.startswith("/proxy/config/arena_safety")):
        return "safety"
    if (p.startswith("/proxy/camera")
            or p.startswith("/proxy/gimbal")
            or p.startswith("/proxy/stream")
            or p.startswith("/proxy/video")):
        return "camera"
    if (p.startswith("/proxy/telemetry")
            or p.startswith("/proxy/heartbeat")
            or p == "/proxy/position"
            or p.startswith("/proxy/position/events")
            or p.startswith("/proxy/position/video")):
        return "telemetry"
    if (p.startswith("/proxy/settings")
            or p.startswith("/proxy/environment")
            or p.startswith("/proxy/wifi/")
            or p.startswith("/proxy/config/transport")
            or p.startswith("/proxy/config/camera_face_center")):
        return "settings"
    if p.startswith("/proxy/magneto"):
        return "magneto"
    if (p.startswith("/proxy/logging")
            or p.startswith("/proxy/flight_logs")
            or p.startswith("/proxy/flight_video")):
        return "logs"
    if (p.startswith("/proxy/arena")
            or p.startswith("/proxy/position/config")
            or p.startswith("/proxy/position/calibration")
            or p.startswith("/proxy/calibration")
            or p.startswith("/proxy/position/presets")):
        return "arena"
    if p.startswith("/proxy/aruco"):
        return "aruco"
    if p.startswith("/proxy/missions"):
        return "missions"
    if p.startswith("/proxy/"):
        return "flight"
    return "system"
