"""Drone-control core, extracted from the unified_api_server.py route bodies.

The HTTP layer (Flask routes in unified_api_server.py) is now a thin
adapter: it parses the request, calls a ``do_*`` function from this
module, and wraps the returned ``(payload, status)`` in ``jsonify``.
Direct callers (e.g. marker_mission running in the same process) can
import this module and call ``do_*`` directly with no JSON / no HTTP /
no socket round-trip — see ``marker_mission/drone_api_inproc.py``.

State coupling
--------------
The drone backend, connection lock, RC state, flying flag, etc. still
live as module-level globals in ``controller_unified.unified_api_server``.
This module accesses them via a lazy ``from . import unified_api_server
as _srv`` inside each function — a deliberate first-step compromise that
keeps the prototype diff small. The follow-up commit migrates those
globals into a ``DroneCoreState`` dataclass owned by this module; once
that lands, ``unified_api_server`` becomes a pure HTTP shim.

Return contract
---------------
Each ``do_*`` returns ``(payload: dict, http_status: int)`` so the route
handler can do ``jsonify(payload), http_status``. Errors that previously
produced ``HTTP 5xx`` come back as a status >= 400; the route handler
applies that status code to the Flask response. The payload dict shape
matches the JSON contract that external clients already depend on.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def get_telemetry_payload() -> tuple[dict, int]:
    """Snapshot the current telemetry dict and decorate it with the
    same fields the /api/telemetry endpoint has always returned. Used
    by both the HTTP route and the in-proc client."""
    import unified_api_server as _srv  # sibling import; see header note
    payload = _srv._build_telemetry_payload()
    return payload, 200


# ---------------------------------------------------------------------------
# Takeoff / land / emergency
# ---------------------------------------------------------------------------

def do_takeoff() -> tuple[dict, int]:
    """Discrete takeoff command. Mirrors the previous /api/takeoff
    handler exactly, including the auto-record side thread."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    with _srv.conn_lock:
        connected = _srv.conn_state["connected"]
    if not connected or b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    try:
        if not _srv.flying:
            hold_s = _srv.SAFE_TAKEOFF_S if _srv.safe_takeoff_enabled else 3.0
            _srv.start_discrete_window(hold_s)
            b.before_discrete_command()
            ok, msg = b.takeoff()
            b.after_discrete_command()
            if ok:
                _srv.flying = True
                _srv.takeoff_cooldown_until = time.time() + hold_s
                try:
                    threading.Thread(
                        target=_srv._auto_record_after_takeoff,
                        kwargs={"grace_s": 2.5},
                        daemon=True, name="auto-record",
                    ).start()
                except Exception as te:
                    print(f"[REC] Could not spawn auto-record thread: {te}")
            else:
                print(f"[{_srv.drone_type.upper()}] Takeoff returned ok=False msg={msg}")
                return {"ok": False, "error": msg}, 500
        return {"ok": True, "flying": _srv.flying,
                "safe_takeoff": _srv.safe_takeoff_enabled}, 200
    except Exception as e:
        traceback.print_exc()
        if _srv.drone_type == "tello":
            rok, rmsg = b.recover()
            return {"ok": False, "error": "takeoff_failed",
                    "recovered": rok, "message": rmsg}, 500
        return {"ok": False, "error": str(e)}, 500


def do_land() -> tuple[dict, int]:
    """Discrete land command. Clears RC overrides + pressed keys to
    avoid the drone fighting the operator on the way down, then auto-
    stops any in-progress recording."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    with _srv.conn_lock:
        connected = _srv.conn_state["connected"]
    if not connected or b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    try:
        if not _srv.flying:
            return {"ok": True, "flying": False}, 200
        _srv.start_discrete_window(3.0)
        with _srv.pressed_lock:
            _srv.pressed_web.clear()
            _srv.key_last_seen.clear()
        with _srv.rc_lock:
            _srv.rc_override = None
            _srv.rc_override_until = 0.0
        b.before_discrete_command()
        ok, msg = b.land()
        b.after_discrete_command()
        if ok:
            _srv.flying = False
            try:
                _srv._stop_recording_internal(reason="land")
            except Exception as se:
                print(f"[REC] Auto-stop on land failed: {se}")
            return {"ok": True, "flying": False}, 200
        print(f"[{_srv.drone_type.upper()}] Land returned ok=False msg={msg}")
        return {"ok": False, "error": msg}, 500
    except Exception as e:
        traceback.print_exc()
        if _srv.drone_type == "tello":
            rok, rmsg = b.recover()
            return {"ok": False, "error": "land_failed",
                    "recovered": rok, "message": rmsg}, 500
        return {"ok": False, "error": str(e)}, 500


def do_emergency() -> tuple[dict, int]:
    """Cut motors immediately. Always clears the flying flag, even on
    backend exception — the drone is presumed unsafe and the next
    /api/telemetry should report flying=False to all clients."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    if b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    try:
        ok, _msg = b.emergency()
        _srv.flying = False
        return {"ok": ok}, 200
    except Exception as e:
        _srv.flying = False
        return {"ok": False, "error": str(e)}, 500


# ---------------------------------------------------------------------------
# RC sticks
# ---------------------------------------------------------------------------

def _clamp_rc(v: Any) -> int:
    try:
        return max(-100, min(100, int(v)))
    except Exception:
        return 0


def do_rc(lr: Any = 0, fb: Any = 0, ud: Any = 0, yaw: Any = 0,
          duration_ms: Any = 250) -> tuple[dict, int]:
    """Push a continuous RC stick command. All inputs are clamped to
    [-100, 100]; ``duration_ms`` is clamped to [50, 2000] so a stuck
    client can't pin the override indefinitely."""
    import unified_api_server as _srv  # sibling import; see header note
    lr = _clamp_rc(lr)
    fb = _clamp_rc(fb)
    ud = _clamp_rc(ud)
    yaw = _clamp_rc(yaw)
    try:
        dur_ms = max(50, min(2000, int(duration_ms)))
    except (TypeError, ValueError):
        dur_ms = 250
    with _srv.rc_lock:
        _srv.rc_override = (lr, fb, ud, yaw)
        _srv.rc_override_until = time.time() + (dur_ms / 1000.0)
    return {
        "ok": True,
        "rc": {"lr": lr, "fb": fb, "ud": ud, "yaw": yaw},
        "duration_ms": dur_ms,
    }, 200
