"""Routes for the **safety** blueprint.

Moved out of ``remote_web_controller.py`` in PR 4. The route bodies are
unchanged; only their location moved. Helpers used by these routes are
imported from the entrypoint module (which still owns the C2-wide
state)."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

from flask import (
    Response, jsonify, render_template, request, send_file,
)

from controller_modularized import state
from controller_modularized.api import bp_safety
from controller_modularized.config import (
    C2_CODE_VERSION, FLIGHT_LOG_DIR, FLIGHT_LOG_HZ,
    HTTP_HOST, HTTP_PORT,
    TIMEOUT_CMD, TIMEOUT_FAST, TIMEOUT_SLOW, TIMEOUT_STATUS,
    VIDEO_FPS, VIDEO_JPEG_QUALITY, VIDEO_UDP_FORWARD_PORT,
)
from controller_modularized.http_client import (
    _http_session, _heartbeat_pool,
    _record_pi_call, _slow_calls, _slow_calls_lock,
    command_log_enabled, command_log_last, command_log_path,
    log_command, pi_get, pi_post,
)
from controller_modularized.controller.pause import (
    is_paused as _is_paused,
    pause_guard_response as _pause_guard_response,
)

from controller_modularized.state import DRONES, save_drones_config
from controller_modularized.model.drone_ws import drone_ws
from controller_modularized.remote_web_controller import (  # noqa: E402
    aruco_fleet, mission_manager, flight_logger, _GIT_REVISION,
)

# Helpers that still live in the entrypoint. The import resolves at
# api/<bp>.py load time, which the entrypoint does at the END of its
# own initialization — so by the time this runs every name below is
# already defined in remote_web_controller.__dict__.
from controller_modularized.remote_web_controller import (
    aruco_fleet, mission_manager,)

@bp_safety.get("/proxy/config/ceiling")
def proxy_ceiling_get():
    """Aggregate the soft ceiling and engaged state from every drone in
    the fleet. Returns the minimum ceiling across drones (most
    conservative) and engaged=True if ANY drone is currently clamped."""
    results = {}
    any_engaged = False
    min_ceiling = None
    reasons = []
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/config/ceiling", timeout=1.0)
            j = r.json()
            results[str(did)] = j
            if j.get("engaged"):
                any_engaged = True
                if j.get("reason"):
                    reasons.append(f"{did}: {j['reason']}")
            if j.get("ceiling_m") is not None:
                c = float(j["ceiling_m"])
                if min_ceiling is None or c < min_ceiling:
                    min_ceiling = c
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    return jsonify(
        ok=True,
        ceiling_m=min_ceiling,
        engaged=any_engaged,
        reasons=reasons,
        per_drone=results,
    )


@bp_safety.get("/proxy/config/arena_safety")
def proxy_arena_safety_get():
    """Aggregate arena-safety state across the fleet — min margin wins,
    any engaged → engaged, any disabled → disabled (most conservative)."""
    results = {}
    any_engaged = False
    all_enabled = True
    min_margin = None
    reasons = []
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/config/arena_safety",
                                   timeout=1.0)
            j = r.json() if r.ok else {"ok": False}
            results[str(did)] = j
            if j.get("engaged"):
                any_engaged = True
                if j.get("reason"):
                    reasons.append(f"{did}: {j['reason']}")
            if not j.get("enabled", True):
                all_enabled = False
            m = j.get("margin_m")
            if isinstance(m, (int, float)):
                if min_margin is None or m < min_margin:
                    min_margin = float(m)
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    return jsonify(
        ok=True,
        enabled=all_enabled,
        margin_m=min_margin,
        engaged=any_engaged,
        reasons=reasons,
        per_drone=results,
    )


@bp_safety.post("/proxy/config/arena_safety")
def proxy_arena_safety_set():
    """Fan-out arena safety settings to every drone's Pi."""
    data = request.get_json(silent=True) or {}
    log_command("arena_safety_set", data)
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/config/arena_safety",
                                    json=data, timeout=2.0)
            results[str(did)] = r.json() if r.ok else {"ok": False, "status": r.status_code}
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[ARENA] fleet-wide: {data} → {ok_count}/{len(results)} drones acknowledged")
    return jsonify(ok=True, applied=ok_count, total=len(results), results=results)


@bp_safety.post("/proxy/config/ceiling")
def proxy_ceiling_set():
    """Push a new soft ceiling to every drone in the fleet. Body:
        {"ceiling_m": <float, metres>}
    """
    data = request.get_json(silent=True) or {}
    try:
        v = float(data.get("ceiling_m") or data.get("max_altitude_m"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="ceiling_m required"), 400
    log_command("ceiling_set", {"ceiling_m": v})
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/config/ceiling",
                                   json={"ceiling_m": v}, timeout=2.0)
            results[str(did)] = r.json()
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[CEILING] fleet-wide set to {v}m: {ok_count}/{len(results)} drones acknowledged")
    return jsonify(ok=True, ceiling_m=v, applied=ok_count, total=len(results), results=results)


@bp_safety.post("/proxy/pause_all")
def proxy_pause_all():
    """Fleet-wide PAUSE. Overrides any command — every drone stays
    airborne at its current position (Anafi auto-hovers with zero RC).

    Side-effects, in order:
      1. Raise the global pause flag (blocks mission start + ArUco LIVE).
      2. Abort any running mission (do NOT land — the drones hover).
      3. Yank every ArUco observer out of LIVE back to OBSERVE and clear
         its search RC override, so it stops pushing commands.
      4. Send a hard zero-RC (full brake) to each drone so any residual
         translation command stops immediately.

    The call is idempotent — pressing it again while already paused is
    a no-op beyond re-issuing the zero-RC.
    """
    data = request.get_json(silent=True) or {}
    source = str(data.get("source") or "unknown")
    log_command("pause_all", {"source": source})

    # 1) Raise the flag first so guards start rejecting autonomous starts
    #    even before we're done tearing down the existing activity.
    with state._pause_lock:
        state._global_paused = True
        state._global_paused_at = time.time()
        state._global_paused_src = source

    # 2) Stop any running mission (don't land).
    mission_stopped = False
    try:
        if mission_manager is not None and mission_manager.current is not None:
            mission_stopped = mission_manager.stop(land=False)
    except Exception as e:
        print(f"[PAUSE_ALL] mission stop failed: {e}")

    # 3) Observers → OBSERVE, clear search overrides.
    try:
        for did, obs in aruco_fleet._obs.items():
            try:
                obs.set_search_rc(0, 0, 0, 0)
                obs.set_mode("observe")
            except Exception:
                pass
    except Exception:
        pass

    # 4) Full brake — POST rc(0,0,0,0) to every drone in parallel-ish.
    results: dict[str, dict] = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            # Some APIs want the full RC tuple, others accept a rc_stop
            # shortcut. The unified API exposes /api/rc; send zeros.
            resp = _http_session.post(
                f"{base.rstrip('/')}/api/rc",
                json={"lr": 0, "fb": 0, "ud": 0, "yaw": 0},
                timeout=2.0,
            )
            try:
                j = resp.json()
            except Exception:
                j = {"raw": resp.text[:120]}
            results[str(did)] = {
                "ok": bool(j.get("ok", resp.status_code == 200)),
                "status": resp.status_code,
            }
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[PAUSE_ALL] fleet paused (source={source}) — "
          f"{ok_count}/{len(results)} drones braked, "
          f"mission_stopped={mission_stopped}")
    return jsonify(
        ok=True, paused=True,
        source=source,
        mission_stopped=mission_stopped,
        braked=ok_count, total=len(results),
        results=results,
    )


@bp_safety.post("/proxy/resume_all")
def proxy_resume_all():
    """Clear the global pause. Autonomous endpoints (missions, ArUco
    LIVE) become reachable again, but nothing auto-restarts — the
    operator must re-arm any mission themselves. That's intentional:
    coming out of pause should never surprise the pilot with a drone
    suddenly darting off."""
    data = request.get_json(silent=True) or {}
    source = str(data.get("source") or "unknown")
    log_command("resume_all", {"source": source})
    with state._pause_lock:
        was_paused = state._global_paused
        state._global_paused = False
        state._global_paused_at = time.time()
        state._global_paused_src = ""
    print(f"[RESUME_ALL] fleet resumed (source={source}, was_paused={was_paused})")
    return jsonify(ok=True, paused=False, source=source, was_paused=was_paused)


@bp_safety.get("/proxy/pause_status")
def proxy_pause_status():
    """UI poll target — lets multiple open tabs sync their button state."""
    with state._pause_lock:
        return jsonify(
            paused=state._global_paused,
            since=state._global_paused_at,
            source=state._global_paused_src,
        )


@bp_safety.get("/proxy/safety/takeoff")
def proxy_safe_takeoff_get():
    try:
        r = pi_get("/api/safety/takeoff", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_safety.post("/proxy/safety/takeoff")
def proxy_safe_takeoff_set():
    data = request.get_json(silent=True) or {}
    log_command("safe_takeoff_set", data)
    r = pi_post("/api/safety/takeoff", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
