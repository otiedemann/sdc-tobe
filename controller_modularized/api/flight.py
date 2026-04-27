"""Routes for the **flight** blueprint.

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
from controller_modularized.api import bp_flight
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
    _transport, _transport_lock,
    _transport_mode, aruco_fleet,
    drone_ws, mission_manager,)

@bp_flight.post("/proxy/key_down")
def proxy_key_down():
    data = request.get_json(silent=True) or {}
    log_command("key_down", data)
    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        k = str(data.get("key", ""))
        ws = drone_ws.get(str(state.active_drone_id))
        if ws and ws.send_key(k, "down"):
            return jsonify(ok=True, via="ws")
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503
    # HTTP path — tight TIMEOUT_FAST so an unreachable drone fails fast
    # (was TIMEOUT_CMD=8s, which made every key burn 8s when the Pi
    # didn't answer — the real cause of "HTTP controls don't work").
    # No more _active_drone_reachable() gate: HTTP shouldn't be vetoed
    # by WS status. If HTTP itself is broken the timeout tells us.
    try:
        r = pi_post("/api/key_down", data, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@bp_flight.post("/proxy/key_up")
def proxy_key_up():
    data = request.get_json(silent=True) or {}
    log_command("key_up", data)
    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        k = str(data.get("key", ""))
        ws = drone_ws.get(str(state.active_drone_id))
        if ws and ws.send_key(k, "up"):
            return jsonify(ok=True, via="ws")
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503
    try:
        r = pi_post("/api/key_up", data, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@bp_flight.post("/proxy/key_batch")
def proxy_key_batch():
    """Batch key_down for all currently held keys in a single request.

    Fires at ~10 Hz from the browser — it's the keep-alive that stops
    the Pi's KEY_STALE_S=1 timer from dropping keys mid-flight. Historic
    path did N serial HTTP POSTs per batch which is the real WASD
    latency culprit (3 held keys × 10 Hz × slow HTTP = backed-up fetch
    queue).

    New path: one WS send per held key on the persistent socket. With
    fire-and-forget semantics, the whole batch clears in microseconds.
    Fallback to HTTP on WS disconnect."""
    data = request.get_json(silent=True) or {}
    keys = data.get("keys", [])
    if not keys:
        return jsonify(ok=True)
    # Dedup so the log and the wire don't carry redundant presses
    uniq = list(dict.fromkeys(keys))

    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        ws = drone_ws.get(str(state.active_drone_id))
        if ws:
            all_sent = True
            for k in uniq:
                if not ws.send_key(str(k), "down"):
                    all_sent = False
                    break
            if all_sent:
                log_command("key_batch", {"keys": uniq, "via": "ws"})
                return jsonify(ok=True, via="ws", n=len(uniq))
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503

    log_command("key_batch", {"keys": uniq, "via": "http"})
    # SINGLE HTTP POST to the Pi's batch endpoint — was N serial
    # per-key calls previously, which saturated the browser's
    # 6-connection pool whenever WS fell back to HTTP.
    try:
        r = pi_post("/api/key_batch", {"keys": uniq}, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@bp_flight.post("/proxy/takeoff")
def proxy_takeoff():
    """Relay takeoff to the active drone's Pi.

    Uses TIMEOUT_SLOW (default 15s) not the default TIMEOUT_CMD (8s)
    because Anafi takeoff routinely takes 3-5 seconds for the armament
    sensors + motor spin-up before /api/takeoff returns success. The
    earlier 2s default truncated this and led to "network error
    contacting drone: TypeError: Failed to fetch" on every attempt.
    Also wraps the call in try/except so a slow drone returns clean
    JSON to the UI instead of a Flask stack-trace HTML page.
    """
    log_command("takeoff")
    try:
        r = pi_post("/api/takeoff", timeout=TIMEOUT_SLOW)
    except Exception as e:
        import requests as _rq
        if isinstance(e, _rq.exceptions.ReadTimeout):
            return jsonify(ok=False,
                           error=f"takeoff timed out after {TIMEOUT_SLOW}s — "
                                 f"drone may still be arming; check telemetry"), 504
        return jsonify(ok=False, error=f"network error: {e}"), 502
    return (r.text, r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/land")
def proxy_land():
    """Relay land to the active drone's Pi. Same slow-command timeout
    as takeoff — Anafi land waits for ground-contact before returning."""
    log_command("land")
    try:
        r = pi_post("/api/land", timeout=TIMEOUT_SLOW)
    except Exception as e:
        import requests as _rq
        if isinstance(e, _rq.exceptions.ReadTimeout):
            return jsonify(ok=False,
                           error=f"land timed out after {TIMEOUT_SLOW}s — "
                                 f"check drone state before retrying"), 504
        return jsonify(ok=False, error=f"network error: {e}"), 502
    return (r.text, r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/land_all")
def proxy_land_all():
    """Emergency panic-button: land every configured drone and halt any
    running mission. Used by the 'q' hotkey in the UI. Tolerates
    individual drones being unreachable — collects per-drone outcomes
    and always returns 200 so the client can render the summary."""
    log_command("land_all")

    # 1) Stop any running mission first so its LIVE-mode state machine
    #    doesn't immediately re-push RC commands that conflict with land.
    mission_stopped = False
    try:
        if mission_manager is not None and mission_manager.current is not None:
            mission_stopped = mission_manager.stop(land=False)
    except Exception as e:
        print(f"[LAND_ALL] mission stop failed: {e}")

    # 2) Send /api/land to every configured drone in parallel-ish
    #    (sequentially but with a short timeout per drone).
    results: dict[str, dict] = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            # Use the slow-command timeout — Anafi land blocks until
            # ground contact. 3s wasn't enough for gentle descent.
            resp = _http_session.post(f"{base.rstrip('/')}/api/land",
                                      json={}, timeout=TIMEOUT_SLOW)
            try:
                j = resp.json()
            except Exception:
                j = {"raw": resp.text[:120]}
            results[str(did)] = {
                "ok": bool(j.get("ok", resp.status_code == 200)),
                "status": resp.status_code,
                "msg": j.get("msg") or j.get("error") or "",
            }
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}

    # 3) Also put every ArUco observer back into OBSERVE mode and clear
    #    any lingering search RC so the drones don't fight the land.
    try:
        for did, obs in aruco_fleet._obs.items():
            try:
                obs.set_search_rc(0, 0, 0, 0)
                obs.set_mode("observe")
            except Exception:
                pass
    except Exception:
        pass

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[LAND_ALL] {ok_count}/{len(results)} drones acknowledged land "
          f"(mission_stopped={mission_stopped})")
    return jsonify(
        ok=True,
        landed=ok_count,
        total=len(results),
        mission_stopped=mission_stopped,
        results=results,
    )


@bp_flight.post("/proxy/flip")
def proxy_flip():
    data = request.get_json(silent=True) or {}
    log_command("flip", data)
    r = pi_post("/api/flip", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/recover")
def proxy_recover():
    log_command("recover")
    r = pi_post("/api/recover")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/emergency")
def proxy_emergency():
    log_command("emergency")
    r = pi_post("/api/emergency")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/speed")
def proxy_speed():
    data = request.get_json(silent=True) or {}
    log_command("speed", data)
    r = pi_post("/api/speed", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/move")
def proxy_move():
    data = request.get_json(silent=True) or {}
    log_command("move", data)
    r = pi_post("/api/move", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/rotate")
def proxy_rotate():
    data = request.get_json(silent=True) or {}
    log_command("rotate", data)
    r = pi_post("/api/rotate", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/go")
def proxy_go():
    data = request.get_json(silent=True) or {}
    log_command("go", data)
    r = pi_post("/api/go", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/curve")
def proxy_curve():
    data = request.get_json(silent=True) or {}
    log_command("curve", data)
    r = pi_post("/api/curve", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/sdk")
def proxy_sdk():
    data = request.get_json(silent=True) or {}
    log_command("sdk", data)
    r = pi_post("/api/sdk", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/rth")
def proxy_rth():
    data = request.get_json(silent=True) or {}
    log_command("rth", data)
    r = pi_post("/api/rth", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.post("/proxy/moveto")
def proxy_moveto():
    data = request.get_json(silent=True) or {}
    log_command("moveto", data)
    r = pi_post("/api/moveto", data, timeout=65)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_flight.get("/proxy/latency")
def proxy_latency():
    """Returns the three-leg latency picture for the ACTIVE drone:
      c2_to_fc_ms     — fresh HTTP round-trip to /api/heartbeat on the Pi
      fc_to_drone_ms  — cached flight-controller-to-drone ICMP ping from /api/drone_ping
      sample_age_s    — age of the fc_to_drone sample
    The client adds these together (plus a user-tuned video-processing
    offset) to get the total command-loop latency."""
    t0 = time.time()
    c2_rtt = None
    fc = None
    try:
        r = _http_session.get(f"{state.PI_BASE}/api/heartbeat", timeout=1.0)
        c2_rtt = (time.time() - t0) * 1000.0
        fc_ok = (r.status_code == 200)
    except Exception:
        fc_ok = False
    try:
        r2 = _http_session.get(f"{state.PI_BASE}/api/drone_ping", timeout=1.0)
        if r2.ok:
            fc = r2.json()
    except Exception:
        pass
    resp = {
        "ok": True,
        "c2_to_fc_ms": round(c2_rtt, 2) if c2_rtt is not None else None,
        "fc_reachable": fc_ok,
        "fc_to_drone_ms": (fc.get("rtt_ms") if fc else None),
        "fc_to_drone_host": (fc.get("host") if fc else None),
        "fc_to_drone_sample_age_s": (fc.get("sample_age_s") if fc else None),
        "pi_base": state.PI_BASE,
    }
    return jsonify(resp)


@bp_flight.get("/proxy/ui_state")
def proxy_ui_state():
    """Single consolidated poll — replaces the many 0.5-5Hz polls the UI
    used to fire for tiny pieces of state (heartbeat, pause, ws_status,
    ceiling, transport, mission status). All of them are cheap reads
    of local C2 state; making them one request means the browser keeps
    its 6 HTTP/1.1 connections free for real traffic (takeoff, key
    events) instead of queueing behind 8 small polls.

    Heartbeat to the Pi is STILL sent here so the Pi watchdog stays
    happy — we reuse the existing parallel-ping logic.
    """
    out: dict = {}

    # --- Pause state ---
    with state._pause_lock:
        out["pause"] = {
            "paused":  state._global_paused,
            "since":   state._global_paused_at,
            "source":  state._global_paused_src,
        }

    # --- Ceiling + arena-safety (aggregated from per-drone endpoints) ---
    # Combined fan-out: for every live drone we fetch BOTH endpoints in
    # sequence. That's 2 HTTP calls per drone per uiStatePoll tick (1Hz)
    # instead of two separate per-2s polls issued by the browser — less
    # load on the C2's HTTP pool and on the browser's 6-connection limit.
    try:
        ceilings = []
        any_c_engaged = False
        c_reasons = []
        margins = []
        all_arena_enabled = True
        any_a_engaged = False
        a_reasons = []
        for did, info in DRONES.items():
            base = (info or {}).get("base")
            if not base: continue
            cli = drone_ws.get(str(did)) if 'drone_ws' in globals() else None
            if cli is not None:
                all_down = (not cli._ws_connected.get("telemetry") and
                            not cli._ws_connected.get("position") and
                            not cli._ws_connected.get("rc"))
                if all_down:
                    continue
            # Ceiling
            try:
                r = _http_session.get(f"{base.rstrip('/')}/api/config/ceiling",
                                       timeout=0.25)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("ceiling_m") is not None:
                        ceilings.append(float(j["ceiling_m"]))
                    if j.get("engaged"):
                        any_c_engaged = True
                        if j.get("reason"):
                            c_reasons.append(f"{did}: {j['reason']}")
            except Exception:
                pass
            # Arena safety
            try:
                r = _http_session.get(f"{base.rstrip('/')}/api/config/arena_safety",
                                       timeout=0.25)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("margin_m") is not None:
                        margins.append(float(j["margin_m"]))
                    if not j.get("enabled", True):
                        all_arena_enabled = False
                    if j.get("engaged"):
                        any_a_engaged = True
                        if j.get("reason"):
                            a_reasons.append(f"{did}: {j['reason']}")
            except Exception:
                pass
        out["ceiling"] = {
            "ceiling_m": min(ceilings) if ceilings else None,
            "engaged":   any_c_engaged,
            "reasons":   c_reasons,
        }
        out["arena_safety"] = {
            "enabled":  all_arena_enabled,
            "margin_m": min(margins) if margins else None,
            "engaged":  any_a_engaged,
            "reasons":  a_reasons,
        }
    except Exception:
        out["ceiling"] = {"ceiling_m": None, "engaged": False, "reasons": []}
        out["arena_safety"] = {"enabled": True, "margin_m": None,
                                "engaged": False, "reasons": []}

    # --- WS status (per drone) ---
    try:
        out["ws"] = {
            "available": HAS_WSCLIENT,
            "drones": {did: cli.status() for did, cli in drone_ws.items()},
        }
    except Exception:
        out["ws"] = {"available": HAS_WSCLIENT, "drones": {}}

    # --- Transport selector state ---
    try:
        with _transport_lock:
            out["transport"] = dict(_transport)
    except Exception:
        out["transport"] = {}

    # --- Missions ---
    try:
        out["missions"] = mission_manager.status()
    except Exception:
        out["missions"] = {}

    # Heartbeat removed — now handled by the C2's _heartbeat_loop()
    # background thread directly. No reason for the browser to be
    # involved in keeping the Pi watchdog alive.

    return jsonify(ok=True, **out)


@bp_flight.get("/proxy/ws/status")
def proxy_ws_status():
    """Per-drone WS connection snapshot for the UI badge."""
    out = {did: cli.status() for did, cli in drone_ws.items()}
    return jsonify(
        ok=True,
        available=HAS_WSCLIENT,
        drones=out,
    )
