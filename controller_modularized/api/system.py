"""Routes for the **system** blueprint.

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
from controller_modularized.api import bp_system
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
    _GIT_REVISION, drone_ws,
    flight_logger,)

@bp_system.get("/")
def index():
    return render_template("index.html")


@bp_system.get("/logo.png")
def serve_logo():
    """Serve the team logo for the header of the UI.

    Prefers the alpha-masked variant (team_logo_transparent.png) so the
    logo blends with the page's dark background rather than showing a
    white rectangle. Falls back to the original PNG if the masked file
    isn't present.
    """
    from pathlib import Path as _P
    base = _P(__file__).resolve().parent.parent / "1_Doc"
    for name in ("team_logo_transparent.png", "team_logo.png"):
        p = base / name
        if p.exists():
            return send_file(str(p), mimetype="image/png",
                             max_age=86400)   # cache-1-day
    return jsonify(ok=False, error="logo not found"), 404


@bp_system.get("/flight_log_viewer")
def serve_flight_log_viewer():
    """Interactive replay UI for flight logs.

    Opens ?file=<name> auto-loaded via /proxy/flight_logs/<name>?dl=0.
    Visualises trajectory + events + timeline so an operator can scrub
    through a flight and see exactly where something went wrong.
    """
    from pathlib import Path as _P
    p = _P(__file__).with_name("flight_log_viewer.html")
    if not p.exists():
        return jsonify(ok=False, error="viewer html missing"), 404
    return send_file(str(p), mimetype="text/html", max_age=0)


@bp_system.get("/proxy/fc_version")
def proxy_fc_version():
    """Fetch the active drone's FC /api/version (short 1 s timeout) and
    compare its code_version string to the C2's. Lets the UI flash a
    banner when the operator deployed a fix to C2 but the FC is still
    running old code — which was silently happening for days."""
    did = str(request.args.get("drone_id") or state.active_drone_id)
    info = DRONES.get(did) or {}
    base = (info.get("base") or "").rstrip("/")
    fc_code = None
    fc_sha = None
    fc_err = None
    if base:
        try:
            r = _http_session.get(f"{base}/api/version", timeout=1.0)
            if r.ok:
                j = r.json() or {}
                fc_code = j.get("code_version")
                fc_sha = (j.get("git_revision") or {}).get("short_sha")
            else:
                fc_err = f"HTTP {r.status_code}"
        except Exception as e:
            fc_err = str(e)
    # Match = version strings are equal ignoring the trailing parenthetical
    # description (so a small comment change doesn't flag every restart).
    def _strip_tag(v):
        if not v: return ""
        return str(v).split(" ", 1)[0].strip()
    match = (fc_code is not None
             and _strip_tag(fc_code) == _strip_tag(C2_CODE_VERSION))
    return jsonify(
        ok=True,
        drone_id=did,
        c2_code_version=C2_CODE_VERSION,
        c2_git_sha=_GIT_REVISION.get("short_sha"),
        fc_code_version=fc_code,
        fc_git_sha=fc_sha,
        fc_error=fc_err,
        match=match,
    )


@bp_system.get("/proxy/diagnostics")
def proxy_diagnostics():
    """C2-side + Pi-side diagnostic snapshot.

    Everything that SHOULD NOT grow monotonically across flights is
    here. If a counter keeps climbing flight-after-flight, that's the
    leak. To use:
      1. Open this URL in a browser tab before flight 1.
      2. Note the thread_count + flight_logger.active_files + http_pool counts.
      3. Fly, land, fly, land.
      4. Refresh. Any number growing is a direct clue.
    """
    import threading as _th, gc as _gc
    with _slow_calls_lock:
        slow = list(_slow_calls)

    # --- C2 threads ---
    threads = _th.enumerate()
    by_name: dict[str, int] = {}
    for t in threads:
        n = t.name
        for prefix in ("ThreadPoolExecutor-", "Thread-", "obs-", "ws-"):
            if n.startswith(prefix):
                n = prefix + "*"
                break
        by_name[n] = by_name.get(n, 0) + 1

    # --- HTTP session connection pool ---
    http_pool_stats = {}
    try:
        adapter = _http_session.get_adapter("http://x")
        if hasattr(adapter, "poolmanager"):
            pm = adapter.poolmanager
            http_pool_stats = {
                "pool_connections_limit": getattr(adapter, "_pool_connections", None),
                "pool_maxsize_limit":     getattr(adapter, "_pool_maxsize",     None),
                "pools_cached":           len(pm.pools) if hasattr(pm, "pools") else None,
            }
    except Exception as e:
        http_pool_stats = {"error": str(e)[:120]}

    # --- FlightLogger ---
    try:
        with flight_logger._lock:
            fl_state = {
                "active_files": len(flight_logger._flights),
                "drone_ids":    list(flight_logger._flights.keys()),
                "running":      flight_logger._running,
                "log_dir":      str(flight_logger.log_dir),
            }
    except Exception as e:
        fl_state = {"error": str(e)[:120]}

    # --- DroneWS clients ---
    ws_clients = {}
    for did, cli in drone_ws.items():
        try:
            s = cli.status()
            ws_clients[did] = {
                "rc": s.get("rc"),
                "telemetry": s.get("telemetry"),
                "position":  s.get("position"),
                "rc_rtt_ms": s.get("rc_rtt_ms"),
                "rc_send_ms": s.get("rc_send_ms"),
                "consec_failures": dict(cli._consec_failures),
            }
        except Exception as e:
            ws_clients[did] = {"error": str(e)[:80]}

    # --- Per-drone Pi diagnostics ---
    per_drone = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        # Skip fan-out for drones whose WS is all-down — same pattern
        # as fleet poll. Keeps this endpoint responsive under partial
        # outage.
        cli = drone_ws.get(str(did))
        if cli is not None:
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                per_drone[str(did)] = {"error": "offline (ws all down)"}
                continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/diagnostics", timeout=0.5)
            per_drone[str(did)] = r.json() if r.ok else {"error": f"http {r.status_code}"}
        except Exception as e:
            per_drone[str(did)] = {"error": str(e)[:120]}

    return jsonify(
        ok=True,
        # C2-side counters — these are what usually leak
        c2={
            "thread_count":    len(threads),
            "threads_by_name": by_name,
            "gc_counts":       list(_gc.get_count()),
            "http_pool":       http_pool_stats,
            "flight_logger":   fl_state,
            "ws_clients":      ws_clients,
            "slow_calls":      slow,
        },
        per_drone=per_drone,
        active_drone=state.active_drone_id,
    )
