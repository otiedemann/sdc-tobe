"""Fleet-wide pause guard. ``state._global_paused`` blocks autonomous
control (missions, ArUco Seek LIVE) — manual WASD/RC/keepalive remain.

Set by /proxy/pause_all, cleared by /proxy/resume_all. UI hotkey '9'.
"""
from __future__ import annotations

from flask import jsonify

from controller_modularized import state


def is_paused() -> bool:
    with state._pause_lock:
        return state._global_paused


def pause_guard_response():
    if is_paused():
        return jsonify(
            ok=False,
            error="fleet paused — autonomous control is disabled",
            hint="press CONTINUE MISSION (button) or 9 (hotkey) to resume",
            paused=True,
        ), 409
    return None
