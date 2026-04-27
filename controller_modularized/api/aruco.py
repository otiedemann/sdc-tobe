"""Routes for the **aruco** blueprint.

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
from controller_modularized.api import bp_aruco
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
    _aruco_require_live, _aruco_resolve,
    _transport_mode, aruco_fleet,
    asdict_hover_defaults, drone_ws,)

@bp_aruco.get("/proxy/aruco/state")
def proxy_aruco_state():
    """Snapshot of ONE observer (query ?id=1, defaults to active drone)."""
    did = request.args.get("id") or state.active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(running=False, drone_id=did, error="unknown drone"), 404
    return jsonify(obs.get_state())


@bp_aruco.get("/proxy/aruco/fleet")
def proxy_aruco_fleet():
    """Snapshot of every observer in the fleet.

    Merged data flow:
      1. Start from the observer's own state dict (pos + telemetry) if
         the observer thread is actually running on this drone.
      2. For every drone in DRONES, fan out a lightweight GET to
         <base>/api/position in parallel (300 ms timeout). That endpoint
         returns the live fused pose even when no observer is started,
         which is exactly what manual-flight mode needs — otherwise the
         3D arena view stays empty until someone arms ArUco Seek.

    Merge policy: the observer's pos wins if present (it's been cached
    with IMU dead-reckoning); the /api/position fallback fills any gap.
    """
    observers = dict(aruco_fleet.all_states())
    # Ensure every configured drone has at least an entry to populate
    for did in DRONES.keys():
        did = str(did)
        if did not in observers:
            observers[did] = {"drone_id": did, "running": False}

    # Two data paths, in order of preference:
    #   1. WS cache (zero HTTP on the fleet-poll tick — <1 ms)
    #   2. Parallel fan-out to each Pi's /api/position
    #
    # We SKIP the HTTP fan-out for any drone whose WS is known-disconnected.
    # If the WS can't reach the Pi, HTTP won't either — trying anyway wastes
    # ~200 ms per drone per poll on connection-refused / timeout, which was
    # the real cause of the "500 timeout" and button-press delays (the fleet
    # poll was holding up Flask threads that other endpoints needed).
    need_http: list[tuple[str, str]] = []
    pos_mode = _transport_mode("position")
    for did, info in DRONES.items():
        did = str(did)
        base = (info or {}).get("base")
        if not base:
            continue
        # WS cache first (unless position transport is forced to http)
        cli = drone_ws.get(did) if pos_mode in ("auto", "ws") else None
        if cli is not None:
            pj, age = cli.latest_position()
            if pj is not None and age < 1.5:
                entry = observers.setdefault(did, {"drone_id": did})
                if entry.get("pos") is None and pj.get("pos") is not None:
                    entry["pos"] = pj["pos"]
                if entry.get("dir") is None and pj.get("dir") is not None:
                    entry["dir"] = pj["dir"]
                if entry.get("pos_vel") is None and pj.get("vel") is not None:
                    entry["pos_vel"] = pj["vel"]
                if entry.get("pos_stale") is None and pj.get("stale") is not None:
                    entry["pos_stale"] = pj["stale"]
                if entry.get("ref_markers") is None and pj.get("ref_markers") is not None:
                    entry["ref_markers"] = pj["ref_markers"]
                # Currently-visible markers — MUST be propagated so the 2D
                # halo + 3D highlight light up on manual flight too. When
                # the DroneObserver isn't started, the per-drone position
                # service is the only source for this, and we were dropping
                # it in both the WS-cache and HTTP-fallback paths.
                if entry.get("seen_markers") is None and pj.get("seen_markers") is not None:
                    entry["seen_markers"] = pj["seen_markers"]
                if entry.get("altitude_m") is None and pj.get("pos"):
                    try:
                        entry["altitude_m"] = float(pj["pos"][2])
                    except (IndexError, TypeError, ValueError):
                        pass
                entry["_pos_source"] = "ws"
                continue
            # WS exists but stale — skip HTTP if ALL three sockets are down
            # (i.e. the Pi is unreachable). Keep trying HTTP if only position
            # is down but telemetry/rc still connected (different failure mode).
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                observers.setdefault(did, {"drone_id": did, "ws_all_down": True})
                continue
        # Position forced to ws-only: don't attempt HTTP.
        if pos_mode == "ws":
            observers.setdefault(did, {"drone_id": did, "ws_only": True})
            continue
        need_http.append((did, base))

    def _fetch(did: str, base: str):
        try:
            resp = _http_session.get(f"{base.rstrip('/')}/api/position", timeout=0.3)
            if resp.status_code == 200:
                return did, resp.json()
        except Exception:
            pass
        return did, None

    if need_http:
        import concurrent.futures as _cf
        jobs = {}
        pool = _cf.ThreadPoolExecutor(max_workers=max(1, len(need_http)))
        try:
            for did, base in need_http:
                jobs[pool.submit(_fetch, did, base)] = did
            # Loop with a per-future timeout rather than as_completed's
            # global timeout — that one raises TimeoutError instead of
            # returning partial results, which used to bubble up as a
            # 500 whenever any Pi was slow/unreachable.
            deadline = time.time() + 0.6
            for fut in list(jobs):
                remaining = max(0.0, deadline - time.time())
                try:
                    did, pj = fut.result(timeout=remaining)
                except Exception:
                    continue
                if not pj:
                    continue
                entry = observers.setdefault(did, {"drone_id": did})
                if entry.get("pos") is None and pj.get("pos") is not None:
                    entry["pos"] = pj["pos"]
                if entry.get("dir") is None and pj.get("dir") is not None:
                    entry["dir"] = pj["dir"]
                if entry.get("pos_vel") is None and pj.get("vel") is not None:
                    entry["pos_vel"] = pj["vel"]
                if entry.get("pos_stale") is None and pj.get("stale") is not None:
                    entry["pos_stale"] = pj["stale"]
                if entry.get("ref_markers") is None and pj.get("ref_markers") is not None:
                    entry["ref_markers"] = pj["ref_markers"]
                # Currently-visible markers — MUST be propagated so the 2D
                # halo + 3D highlight light up on manual flight too. When
                # the DroneObserver isn't started, the per-drone position
                # service is the only source for this, and we were dropping
                # it in both the WS-cache and HTTP-fallback paths.
                if entry.get("seen_markers") is None and pj.get("seen_markers") is not None:
                    entry["seen_markers"] = pj["seen_markers"]
                if entry.get("altitude_m") is None and pj.get("pos"):
                    try:
                        entry["altitude_m"] = float(pj["pos"][2])
                    except (IndexError, TypeError, ValueError):
                        pass
                entry["_pos_source"] = "http"
        finally:
            # Don't wait for slow workers — let them finish in the
            # background. If we joined here the endpoint could block 4+
            # seconds when a drone is unreachable.
            pool.shutdown(wait=False)

    return jsonify(active=state.active_drone_id,
                   allow_live=aruco_fleet.allow_live,
                   observers=observers)


@bp_aruco.get("/proxy/aruco/params")
def proxy_aruco_params_get():
    did = request.args.get("id") or state.active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(asdict_hover_defaults()), 200
    return jsonify(obs.get_params())


@bp_aruco.post("/proxy/aruco/params")
def proxy_aruco_params_set():
    data = request.get_json(silent=True) or {}
    # If no id, broadcast to EVERY observer (fleet-wide knobs like
    # axis_locked should apply to the whole swarm, not just one drone).
    did = data.pop("id", None) or request.args.get("id")
    if did:
        obs, did = _aruco_resolve(str(did))
        if obs is None:
            return jsonify(ok=False, error="unknown drone"), 404
        applied = obs.update_params(data)
        return jsonify(ok=True, applied=applied, drone_id=did)
    # Fleet-wide fan-out
    applied_all = {}
    for d_id, o in aruco_fleet._obs.items():
        try:
            applied_all[d_id] = o.update_params(data)
        except Exception as e:
            applied_all[d_id] = {"error": str(e)[:80]}
    return jsonify(ok=True, applied=applied_all, fleet=True)


@bp_aruco.get("/proxy/aruco/safety_margin")
def proxy_aruco_safety_margin_get():
    """Return the current autonomous-mode arena safety margin (metres).
    Reads from the active drone's observer since the margin is
    per-observer state; fleet-wide consistency is maintained by the
    POST endpoint broadcasting to every observer."""
    obs = aruco_fleet.get(str(state.active_drone_id))
    if obs is None:
        return jsonify(ok=False, safety_margin_m=None), 200
    with obs._lock:
        return jsonify(
            ok=True,
            safety_margin_m=float(obs._safety_margin_m),
            arena_bounds=dict(obs._arena_bounds),
        )


@bp_aruco.post("/proxy/aruco/safety_margin")
def proxy_aruco_safety_margin_set():
    """Set the autonomous-mode arena safety margin. Body:
        {"safety_margin_m": 1.5}
    Broadcasts to every drone's observer so the fleet shares one value.
    """
    data = request.get_json(silent=True) or {}
    try:
        v = float(data.get("safety_margin_m"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="safety_margin_m required"), 400
    v = max(0.1, min(5.0, v))   # clamp to a sane range
    for d_id, o in aruco_fleet._obs.items():
        try:
            o.set_arena_bounds({}, safety_margin_m=v)
        except Exception as e:
            print(f"[SAFETY] drone {d_id} set failed: {e}")
    log_command("safety_margin_set", {"safety_margin_m": v})
    print(f"[SAFETY] arena margin set to {v}m (fleet-wide)")
    return jsonify(ok=True, safety_margin_m=v)


@bp_aruco.post("/proxy/aruco/start")
def proxy_aruco_start():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_start", {"id": did})
    obs.start()
    return jsonify(ok=True, drone_id=did)


@bp_aruco.post("/proxy/aruco/stop")
def proxy_aruco_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_stop", {"id": did})
    obs.stop()
    return jsonify(ok=True, drone_id=did)


@bp_aruco.post("/proxy/aruco/target")
def proxy_aruco_target():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    mid = data.get("marker")
    if mid is not None:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            mid = None
    obs.set_target(mid)
    log_command("aruco_target", {"id": did, "marker": mid})
    return jsonify(ok=True, drone_id=did, marker=mid)


@bp_aruco.post("/proxy/aruco/mode")
def proxy_aruco_mode():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    requested = (data.get("mode") or "").lower()
    # Respect global PAUSE — switching to LIVE while paused is exactly
    # the kind of autonomous-command surprise PAUSE exists to prevent.
    # Going back to OBSERVE is always allowed (it's a safety downgrade).
    if requested == "live":
        guarded = _pause_guard_response()
        if guarded is not None:
            return guarded
    if requested == "live" and not obs.allow_live:
        return jsonify(ok=False, mode=obs.mode,
                       error="LIVE mode disabled on this server (REMOTE_NO_LIVE=1)"), 403
    actual = obs.set_mode(requested)
    log_command("aruco_mode", {"id": did, "mode": actual})
    return jsonify(ok=(actual == requested), drone_id=did, mode=actual)


@bp_aruco.post("/proxy/aruco/takeoff")
def proxy_aruco_takeoff():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_takeoff", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_takeoff())


@bp_aruco.post("/proxy/aruco/land")
def proxy_aruco_land():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_land", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_land())


@bp_aruco.post("/proxy/aruco/emergency")
def proxy_aruco_emergency():
    # Allowed at any mode — killswitch must always be reachable
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_emergency", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_emergency())


@bp_aruco.post("/proxy/aruco/rc_stop")
def proxy_aruco_rc_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or state.active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_rc_stop", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_rc_stop())


@bp_aruco.get("/proxy/aruco/video.mjpg")
def proxy_aruco_video():
    """Pass-through MJPEG for the observer's drone (separate from /proxy/video
    so the main UI's video stream and the ArUco Seek panel don't collide)."""
    did = request.args.get("id") or state.active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return Response(b"", status=404)
    upstream_url = f"{obs.api_base}/api/position/video"
    try:
        upstream = requests.get(upstream_url, stream=True, timeout=10)
    except Exception as e:
        return Response(f"upstream error: {e}".encode(), status=502, mimetype="text/plain")
    content_type = upstream.headers.get(
        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
    )

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        except (GeneratorExit, requests.exceptions.RequestException):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass
    return Response(generate(), content_type=content_type)
