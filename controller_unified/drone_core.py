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
from typing import Any, Optional


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
    _t0 = time.monotonic()
    print(f"[TKOFF-T] {_t0:.3f} do_takeoff: ENTER")
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
            print(f"[TKOFF-T] {time.monotonic():.3f} do_takeoff: calling "
                  f"backend.takeoff() (+{time.monotonic() - _t0:.3f}s)")
            ok, msg = b.takeoff()
            print(f"[TKOFF-T] {time.monotonic():.3f} do_takeoff: backend.takeoff()"
                  f" returned ok={ok} (+{time.monotonic() - _t0:.3f}s)")
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


# ---------------------------------------------------------------------------
# Closed-loop position-relative move (moveBy under the hood)
# ---------------------------------------------------------------------------

def do_move(direction: Any = "", cm: Any = 20) -> tuple[dict, int]:
    """Discrete closed-loop move using Anafi's moveBy. ``direction`` is
    one of ``forward``/``back``/``left``/``right``/``up``/``down``;
    ``cm`` is the distance in centimetres. Synchronous on the FC side
    — the call blocks until the firmware confirms the move completed.

    Same wire shape as the existing ``/api/move`` route — that route
    now delegates here so HTTP and in-proc consumers agree on
    semantics + the operator-visible safety clamp [20, 500] cm. The
    in-proc caller (mission's FB_IMU/LR_IMU/UD_IMU steps) goes
    around the clamp when it needs sub-20cm precision."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    with _srv.conn_lock:
        connected = _srv.conn_state["connected"]
    if not connected or b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    direction = str(direction or "").lower()
    if direction not in {"forward", "back", "left", "right", "up", "down"}:
        return {"ok": False,
                "error": "dir must be one of "
                         "forward|back|left|right|up|down"}, 400
    try:
        dist_cm = int(cm)
    except (TypeError, ValueError):
        return {"ok": False, "error": "cm must be an integer"}, 400
    try:
        _srv.start_discrete_window(
            1.0 if _srv.drone_type == "tello"
            else max(1.0, abs(dist_cm) / 50)
        )
        # Wait at least one RC-loop cycle (50 ms @ 20 Hz) so the loop
        # notices in_discrete=True and stops sending PCMD before
        # _stop_piloting() is called inside b.move().
        import time as _time; _time.sleep(0.1)
        ok, msg = b.move(direction, dist_cm)
        if ok:
            return {"ok": True, "dir": direction, "cm": dist_cm}, 200
        return {"ok": False, "error": msg}, 500
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


def do_go(x: Any = 0, y: Any = 0, z: Any = 0,
          speed: Any = 50) -> tuple[dict, int]:
    """Combined relative move (Olympe moveBy via ``go_xyz``): ``x`` forward,
    ``y`` right, ``z`` up — all in CENTIMETRES — in ONE moveBy, so a script can
    move on several axes at once (e.g. forward + up) instead of sequential
    single-axis IMU steps. In-proc twin of the ``/api/go`` route."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    with _srv.conn_lock:
        connected = _srv.conn_state["connected"]
    if not connected or b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    try:
        xi, yi, zi = int(x), int(y), int(z)
        sp = max(10, min(100, int(speed)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "x/y/z/speed must be integers"}, 400
    try:
        reach = max(abs(xi), abs(yi), abs(zi))
        _srv.start_discrete_window(
            1.0 if _srv.drone_type == "tello" else max(1.0, reach / 50))
        import time as _time; _time.sleep(0.1)
        ok, msg = b.go_xyz(xi, yi, zi, sp)
        if ok:
            return {"ok": True, "x": xi, "y": yi, "z": zi, "speed": sp}, 200
        return {"ok": False, "error": msg}, 500
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def do_rotate(direction: Any = "", deg: Any = 45,
              speed: Any = None) -> tuple[dict, int]:
    """Discrete CW/CCW rotation. Degrees clamped to [1, 360].

    ``speed`` (deg/s) is an optional per-rotation angular-speed override
    (YAW_IMU's 2nd arg). None -> use the FC's global MaxRotationSpeed. On
    Anafi it temporarily reshapes MaxRotationSpeed for this turn; on Tello
    it is accepted and ignored.
    """
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    with _srv.conn_lock:
        connected = _srv.conn_state["connected"]
    if not connected or b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    direction = str(direction or "").lower()
    if direction not in {"cw", "ccw"}:
        return {"ok": False, "error": "dir must be one of cw|ccw"}, 400
    spd: Optional[int] = None
    if speed is not None:
        try:
            spd = max(1, min(200, int(speed)))
        except (TypeError, ValueError):
            spd = None
    try:
        degrees = max(1, min(360, int(deg)))
        # A faster turn finishes sooner; size the discrete window to the
        # actual angular rate so we don't hold the command window open
        # longer than the rotation takes (default rate ~90 deg/s).
        rate = float(spd) if spd else 90.0
        _srv.start_discrete_window(
            1.0 if _srv.drone_type == "tello"
            else max(1.0, degrees / max(1.0, rate))
        )
        import time as _time; _time.sleep(0.1)
        ok, msg = b.rotate(direction, degrees, spd)
        if ok:
            payload = {"ok": True, "dir": direction, "deg": degrees}
            if spd is not None:
                payload["speed"] = spd
            return payload, 200
        return {"ok": False, "error": msg}, 500
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ---------------------------------------------------------------------------
# Wi-Fi AP band / channel
# ---------------------------------------------------------------------------

def do_set_wifi_channel(band: Any = "5_GHz", channel: Any = 0,
                        auto: Any = True) -> tuple[dict, int]:
    """Set the Anafi's Wi-Fi AP band + channel (in-proc twin of the HTTP
    ``/api/wifi/channel``).

    auto=True (recommended): the drone scans and picks the cleanest channel in
    ``band`` (``"5_GHz"`` / ``"2_4_GHz"``). auto=False: pin ``channel`` (1-13 for
    2.4 GHz; 36/40/44/48/149/153/157/161/165 for 5 GHz, region-dependent).

    Changing the AP channel briefly DROPS the Wi-Fi link (the drone re-associates
    on the new channel; the watchdog reconnects). Only issue it on the ground.
    """
    import unified_api_server as _srv  # sibling import; see header note
    if not getattr(_srv, "HAS_WIFI_CTRL", False) or _srv.WifiSetApChannel is None:
        return {"ok": False, "error": "Wi-Fi control not supported"}, 501
    b = _srv._anafi_backend()
    if b is None:
        return {"ok": False, "error": "drone not connected"}, 503
    band_str = str(band or "5_GHz").lower()
    auto_mode = bool(auto)
    try:
        ch = int(channel)
    except (TypeError, ValueError):
        ch = 0
    band_enum = _srv._wifi_band_enum(band_str)
    if auto_mode:
        which = ("auto_5" if band_str.startswith("5")
                 else "auto_2_4" if band_str.startswith("2") else "auto_all")
        sel = _srv._wifi_sel_type_enum(which)
    else:
        sel = _srv._wifi_sel_type_enum("manual")
    try:
        with _srv.command_lock:
            b.drone(_srv.WifiSetApChannel(
                type=sel, band=band_enum,
                channel=0 if auto_mode else ch)).wait(_timeout=5)
        mode = "auto" if auto_mode else "manual"
        print(f"[ANAFI] Wi-Fi: {mode.upper()} -> band={band_str}, "
              f"channel={'any' if auto_mode else ch}")
        out = {"ok": True, "mode": mode, "band": band_str,
               "message": "channel change submitted; the link drops briefly"}
        if not auto_mode:
            out["channel"] = ch
        return out, 200
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ---------------------------------------------------------------------------
# Video pipeline (MJPEG / forward)
# ---------------------------------------------------------------------------

def do_video_start(mode: str = "mjpeg",
                   target_host: str | None = None,
                   target_port: int | None = None,
                   stream_url: str | None = None) -> tuple[dict, int]:
    """Start the backend video pipeline. ``stream_url`` is only used
    in the response payload — the HTTP route fills it with the
    request's ``http://<host>/api/video``; in-proc callers leave it
    as None and consume frames via :func:`iter_video_jpegs` instead."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    if b is None:
        return {"ok": False, "error": "controller not ready"}, 503
    try:
        if mode == "mjpeg" and _srv._video_streaming and _srv._video_mode == "mjpeg":
            return {"ok": True, "mode": "mjpeg",
                    "message": "already streaming",
                    "stream_url": stream_url}, 200
        b.video_stop_all()
        if mode == "mjpeg":
            ok, msg = b.video_start_mjpeg()
            if ok:
                _srv._video_mode = "mjpeg"
                return {"ok": True, "mode": "mjpeg",
                        "message": msg, "stream_url": stream_url}, 200
            print(f"[{_srv.drone_type.upper()}] video_start_mjpeg failed: {msg}")
            return {"ok": False, "error": msg}, 500
        elif mode == "forward":
            if not target_host:
                return {"ok": False, "error": "target_host required"}, 400
            tport = int(target_port if target_port is not None
                        else _srv.VIDEO_UDP_FORWARD_PORT)
            ok, msg = b.video_start_forward(target_host, tport)
            if ok:
                _srv._video_mode = "forward"
                return {
                    "ok": True, "mode": "forward", "message": msg,
                    "target": f"{target_host}:{tport}",
                    "viewer_cmd": (
                        f"ffplay -fflags nobuffer -flags low_delay "
                        f"-framedrop -probesize 32 -analyzeduration 0 "
                        f"udp://0.0.0.0:{tport}"
                    ),
                }, 200
            print(f"[{_srv.drone_type.upper()}] video_start_forward failed: {msg}")
            return {"ok": False, "error": msg}, 500
        return {"ok": False, "error": f"unknown mode: {mode}"}, 400
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}, 500


def do_video_stop() -> tuple[dict, int]:
    """Tear down the backend video pipeline. Idempotent."""
    import unified_api_server as _srv  # sibling import; see header note
    b = _srv.backend
    if b:
        b.video_stop_all()
    return {"ok": True, "mode": "off"}, 200


# ---------------------------------------------------------------------------
# Flight envelope (soft ceiling)
# ---------------------------------------------------------------------------

def do_set_ceiling(ceiling_m: Any) -> tuple[dict, int]:
    """Push the soft altitude ceiling to the FC. Same effect as
    ``POST /api/config/ceiling`` — sets ``MAX_ALTITUDE_M`` (the value
    the RC tick loop clamps every climb stick against), persists it
    to ``flight_config.json``, and best-effort updates the Anafi
    firmware ``MaxAltitude`` cap so the autopilot itself enforces it
    as a second line of defence.

    marker_mission calls this at startup with ``MissionConfig.max_height_m``
    so the FC ceiling never disagrees with what the mission's PD
    output would otherwise produce — without this the FC clamped any
    mission climb to its persisted ``flight_config.json`` value
    (typically 2 m), leaving altitude-limited markers unreachable."""
    import unified_api_server as _srv  # sibling import; see header note
    try:
        v = float(ceiling_m)
    except (TypeError, ValueError):
        return {"ok": False, "error": "ceiling_m required (float, metres)"}, 400
    v = max(0.5, min(150.0, v))
    _srv.MAX_ALTITUDE_M = v
    firmware_result = None
    b = _srv.backend
    try:
        if b is not None and hasattr(b, "set_settings"):
            r = b.set_settings({"max_altitude_m": v})
            firmware_result = r.get("max_altitude_m",
                                    r.get("max_altitude_m_error"))
    except Exception as e:
        firmware_result = f"firmware_error: {e}"
    try:
        _srv._save_flight_config({"max_altitude_m": v})
    except Exception:
        pass
    print(f"[CEILING] set to {v}m (firmware={firmware_result})")
    return {"ok": True, "ceiling_m": v,
            "firmware_result": firmware_result}, 200


def iter_video_jpegs():
    """Yield the latest JPEG frame whenever one is available — an
    in-proc replacement for the ``/api/video`` MJPEG endpoint. Caller
    decides what to do with each frame (decode + OpenCV, save to disk,
    forward to another consumer). Stops when the server is shutting
    down or the video pipeline transitions away from mjpeg mode."""
    import unified_api_server as _srv  # sibling import; see header note
    if _srv._video_mode != "mjpeg":
        return
    b = _srv.backend
    while _srv.running and _srv._video_mode == "mjpeg":
        jpg = b.get_video_jpeg() if b else b""
        if jpg:
            yield jpg
        time.sleep(1.0 / max(1, _srv.VIDEO_FPS))
