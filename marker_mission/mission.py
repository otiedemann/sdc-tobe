"""
Top-level entry point.

Usage::

    python -m marker_mission.mission [options]

Examples::

    # Default mission: marker id 4, 1 m standoff, 90 deg final heading
    python -m marker_mission.mission

    # Run an offline calibration on a video, store under the drone's serial
    python -m marker_mission.mission calibrate \\
        --video calib.mp4 --serial PI040421AA1234 --resolution 720p \\
        --pattern 9x6 --square 0.025

The default subcommand ``fly`` performs the full mission:

    1. Take off
    2. Search for the configured marker (yaw in place)
    3. Approach until yaw=0 and distance=target_distance_m
    4. Orbit to relative_heading=target_relative_heading_deg while keeping
       distance and yaw=0
    5. Hover for ``hold_time_s`` seconds
    6. Land

In any phase, if the marker is lost for longer than ``pose_max_age_s``
the controller falls back to SEARCH; if a full sweep does not reacquire
it, the drone lands.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import asdict, replace as _dc_replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .aruco_detector import (ArucoDetector, MarkerPose, annotate_frame,
                              draw_arena_minimap)
from .calibration_store import (Calibration, CalibrationStore,
                                calibrate_from_video)
from .arena import (ArenaConfig, ARENA_MAG_SLACK_DEG,
                    estimate_position, load_priority_arena,
                    _arena_yaw_for_branch, _wrap_180, _yaw_diff,
                    MarkerPositionCache)
from .config import (CALIB_DIR, FLIGHTS_DIR, DEFAULT_DATA_DIR,
                     PER_FLIGHT_SCRIPT_FILENAME, MissionConfig)
from .controller import MissionController, MissionState, Phase
from . import mission_script as ms
from .drone_api import DroneApi, MjpegStreamReader, TelemetrySnapshot
from .drone_api_inproc import DroneApiInProc, InProcMjpegReader
from .kalman import PositionKalman, ned_velocity_to_arena
from .recorder import FlightRecorder, make_flight_dir, write_meta
from . import cpu_monitor
from .ui import LatestFrame, UiServer
from .web_calibration import CalibrationCapture


# ---------------------------------------------------------------------------
# Shared providers used by the controller
# ---------------------------------------------------------------------------

class _PoseHolder:
    """Latest detected MarkerPose for the configured target ID, plus a lock."""
    def __init__(self):
        self._lock = threading.Lock()
        self._pose: Optional[MarkerPose] = None

    def set(self, pose: Optional[MarkerPose]) -> None:
        with self._lock:
            self._pose = pose

    def get(self) -> Optional[MarkerPose]:
        with self._lock:
            return self._pose


class _TelemetryHolder:
    def __init__(self):
        self._lock = threading.Lock()
        self._tel: Optional[TelemetrySnapshot] = None
        # Process-monotonic timestamp of the last set() call with non-None
        # data. The stall detector compares (time.monotonic() -
        # _last_set_at) against cfg.telemetry_stall_*_threshold_s to
        # decide whether to mark / gate. 0.0 means "never received a
        # sample yet" (pre-flight, drone not connected).
        self._last_set_at: float = 0.0

    def set(self, tel: Optional[TelemetrySnapshot]) -> None:
        with self._lock:
            self._tel = tel
            if tel is not None:
                self._last_set_at = time.monotonic()

    def get(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._tel

    def age_s(self) -> Optional[float]:
        """Seconds since the last set() with non-None data, or None if
        no sample has arrived yet. Used by the controller's stall
        detector -- when this exceeds cfg.telemetry_stall_gate_threshold_s,
        the controller zeros its RC output instead of acting on a
        stale TelemetrySnapshot."""
        with self._lock:
            if self._last_set_at == 0.0:
                return None
            return time.monotonic() - self._last_set_at


class _ArenaHolder:
    """Latest active ArenaConfig (or None if world positioning is off).

    Wrapped in a holder so the Arena tab's POST /api/arena/active can
    swap in a new layout without restarting the mission -- vision_worker
    reads arena_holder.get() each tick.
    """
    def __init__(self, arena: Optional[ArenaConfig] = None):
        self._lock = threading.Lock()
        self._arena: Optional[ArenaConfig] = arena

    def set(self, arena: Optional[ArenaConfig]) -> None:
        with self._lock:
            self._arena = arena

    def get(self) -> Optional[ArenaConfig]:
        with self._lock:
            return self._arena


class _DroneThreatHolder:
    """Thread-safe: largest confirmed enemy-drone blob width (px) + lateral
    movement direction in camera frame (dx, pixels/frame, positive = moving right).
    max_w=0 means no threat detected."""
    def __init__(self):
        self._lock = threading.Lock()
        self._max_w: int = 0
        self._dx: float = 0.0   # camera-frame lateral movement of nearest blob

    def set(self, max_w: int, dx: float = 0.0) -> None:
        with self._lock:
            self._max_w = max_w
            self._dx = dx

    def get(self) -> tuple:
        """Returns (max_w: int, dx: float)."""
        with self._lock:
            return self._max_w, self._dx


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> int:
    cols, rows = (int(x) for x in args.pattern.lower().split("x"))
    cal = calibrate_from_video(Path(args.video),
                               pattern_size=(cols, rows),
                               square_size_m=float(args.square),
                               frame_stride=int(args.stride))
    cal.serial = args.serial
    cal.resolution = args.resolution
    if args.notes:
        cal.notes = (cal.notes + " | " + args.notes).strip(" |")
    store = CalibrationStore(CALIB_DIR)
    path = store.save(cal)
    print(f"calibration saved: {path}")
    print(cal.short_summary())
    if cal.rms_error > 1.0:
        print(f"WARNING: reprojection error {cal.rms_error:.2f} px is high "
              f"(>1 px). Re-run with more / better-lit / more varied views.")
    return 0


def cmd_fly(args: argparse.Namespace) -> int:
    cfg = MissionConfig.load()
    # CLI overrides
    if args.marker_id is not None:    cfg.target_marker_id = args.marker_id
    if args.distance is not None:     cfg.target_distance_m = args.distance
    if args.heading is not None:      cfg.target_relative_heading_deg = args.heading
    if args.hold is not None:         cfg.hold_time_s = args.hold
    if args.api_url is not None:      cfg.api_base_url = args.api_url
    if args.ui_port is not None:      cfg.ui_port = args.ui_port
    if args.marker_size is not None:  cfg.marker_size_m = args.marker_size
    cfg.save()  # persist effective config

    print(f"[mission] config: marker_id={cfg.target_marker_id} "
          f"distance={cfg.target_distance_m}m "
          f"heading={cfg.target_relative_heading_deg}deg "
          f"hold={cfg.hold_time_s}s api={cfg.api_base_url}")

    # ---------- 1. Connect to API server, fetch serial, load calibration ---
    # The drone doesn't have to be reachable at startup -- the UI comes
    # up either way so the operator can browse recorded flights via the
    # Replay tab. We try briefly here so a drone that IS connected gets
    # its proper per-serial calibration; if not, we fall back to the
    # default Anafi intrinsics and the telemetry_worker re-checks
    # connectivity in the background so the Start button unlocks as
    # soon as the drone shows up.
    in_process = bool(getattr(args, "in_process", False))
    if in_process:
        api = DroneApiInProc()
        print(f"[mission] in-process mode: drone_core direct dispatch "
              f"(--api-url ignored)")
        # Push the mission's altitude envelope to the FC so the FC's
        # MAX_ALTITUDE_M (which clamps every climb stick) doesn't
        # silently veto mission climbs above whatever the operator
        # last persisted in flight_config.json. Only meaningful
        # in-proc — the HTTP path can't get here before the user has
        # done the corresponding /tune step manually.
        try:
            res = api.set_ceiling(cfg.max_height_m)
            print(f"[mission] FC ceiling aligned to "
                  f"max_height_m={cfg.max_height_m}m "
                  f"(firmware={res.get('firmware_result')})")
        except Exception as e:
            print(f"[mission] WARNING: failed to push ceiling to FC: {e}")
    else:
        api = DroneApi(cfg.api_base_url, cfg.request_timeout_s)
        print(f"[mission] contacting API server at {cfg.api_base_url} ...")
    serial = "unknown"
    initially_connected = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            tel = api.telemetry()
            if tel.connected:
                serial = tel.serial_number or "unknown"
                initially_connected = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    if initially_connected:
        print(f"[mission] connected. drone serial = {serial}")
        # Apply the operator-pinned default camera config (if any)
        # before the calibration / video-start dance below. Non-fatal:
        # a startup that fails to push the default still flies; the
        # operator can re-apply on /calibrate.
        try:
            from .config import get_default, CAMERA_CONFIGS_DIR
            cam_default = get_default("camera")
            if cam_default:
                p = CAMERA_CONFIGS_DIR / f"{cam_default}.json"
                if p.exists():
                    payload = json.loads(p.read_text())
                    api.camera_config_set(**payload)
                    print(f"[mission] applied default camera config:"
                          f" {cam_default}")
                else:
                    print(f"[mission] default camera config"
                          f" {cam_default!r} not found; skipped")
        except Exception as e:
            print(f"[mission] default camera config push failed: {e}"
                  f" (non-fatal)")
    else:
        print(f"[mission] no drone yet -- starting UI anyway. "
              f"telemetry_worker will keep retrying.")

    store = CalibrationStore(CALIB_DIR)
    calibration = store.load(serial, args.resolution, allow_default=True)
    print(f"[mission] calibration: {calibration.short_summary()}")
    if calibration.is_default:
        print("[mission] WARNING: no per-drone calibration found -- using "
              "Anafi default intrinsics. Pose accuracy will be reduced.")

    # ---------- 2. Open video stream, start detector ----------------------
    # Both calls are best-effort -- video_start_mjpeg fails if the drone
    # isn't reachable yet, and the MjpegStreamReader auto-reconnects on
    # its own loop, so we just let it retry in the background.
    print("[mission] starting video stream ...")
    if initially_connected:
        try:
            api.video_start_mjpeg()
        except Exception as e:
            print(f"[mission] video_start_mjpeg failed: {e} (will retry)")
    if in_process:
        reader = InProcMjpegReader()
    else:
        reader = MjpegStreamReader(api.video_url())
    reader.start()
    detector = ArucoDetector(calibration, cfg.marker_size_m, cfg.aruco_dict)
    detector.enable_mirror_collapse = cfg.enable_ippe_mirror_collapse

    # ---------- 2b. Arena world-position estimator -----------------------
    # vision_worker computes the camera's arena-frame world position
    # each frame as a weighted average of the per-marker pose
    # inversions. Source priority for the active layout:
    #   1. --arena-config <path> CLI override.
    #   2. ~/.marker_mission/active_arena_config.json (saved from the
    #      Arena tab in the UI).
    #   3. Hard-coded default 16-marker layout (10 m x 25 m).
    # The Arena tab's POST /api/arena/active swaps the layout via the
    # _ArenaHolder so saves take effect without restart.
    arena: Optional[ArenaConfig] = None
    if args.arena_config:
        try:
            arena = ArenaConfig.load(Path(args.arena_config))
            print(f"[mission] arena loaded from {args.arena_config}: "
                  f"{len(arena.markers)} markers, "
                  f"size={arena.marker_size_m:.3f}m")
        except Exception as e:
            print(f"[mission] arena_config load failed ({e}); "
                  f"falling back to active config / default")
            arena = None
    if arena is None:
        try:
            arena = load_priority_arena()
            print(f"[mission] arena loaded (active or default): "
                  f"{len(arena.markers)} markers, "
                  f"size={arena.marker_size_m:.3f}m, "
                  f"{arena.width_m:.1f}m x {arena.depth_m:.1f}m")
        except Exception as e:
            print(f"[mission] could not load any arena ({e}); "
                  f"world position estimator disabled")
            arena = None
    # When an arena is loaded, the arena's marker_size_m wins over the
    # cfg default: the Arena tab is the operator-facing source of truth
    # for marker geometry, and a mismatch would scale every distance /
    # world-position estimate by the ratio. vision_worker continues to
    # sync on every tick so live edits in /arena propagate without a
    # restart.
    if arena is not None and abs(arena.marker_size_m - cfg.marker_size_m) > 1e-3:
        print(f"[mission] marker_size_m: arena={arena.marker_size_m:.3f} "
              f"vs cfg={cfg.marker_size_m:.3f} -- using arena value "
              f"(detector + solvePnP)")
        detector.set_marker_size(arena.marker_size_m)
    arena_holder = _ArenaHolder(arena)

    pose_holder = _PoseHolder()
    tel_holder = _TelemetryHolder()
    drone_threat_holder = _DroneThreatHolder()
    # Background sampler for host + process CPU / memory usage. Daemon
    # thread, idempotent start. state.snapshot() reads the latest
    # sample so every CSV row + /api/state response carries it; the
    # UI uses it to surface a live resource indicator. Costs ~5 reads
    # from /proc per second.
    cpu_monitor.start(sample_period_s=1.0)
    latest_ann_frame = LatestFrame()

    # ---------- 3. Set up mission state, recorder, UI ---------------------
    # The recorder + flight directory are created LAZILY at TAKEOFF so a
    # script run that never flies (e.g., the operator only browsing
    # replays) doesn't litter ~/.marker_mission/flights/ with empty
    # directories. recorder_box[0] / flight_dir_box[0] are None when
    # there is no active recording; vision_worker / log_worker /
    # _log_param_changes all None-check before writing.
    state = MissionState()
    recorder_box: list = [None]
    flight_dir_box: list = [None]
    # Start paused -- INIT is not part of the flight.
    recording_paused = threading.Event()
    recording_paused.set()

    # Phases during which we record video + log CSV. Defined as a
    # denylist (anything that's NOT pre-flight INIT or terminal
    # DONE/ABORT) so a future Phase added to the enum is recorded by
    # default -- the previous allowlist silently dropped Phase.GOTO
    # and truncated flight-log.csv at TAKEOFF on every TO-using
    # mission until 2026-05-05.
    NON_AIRBORNE_PHASES = {Phase.INIT, Phase.LAND, Phase.DONE, Phase.ABORT}
    AIRBORNE_PHASES = {p for p in Phase if p not in NON_AIRBORNE_PHASES}

    def on_phase_change(old_phase, new_phase, note):
        # First-tick of TAKEOFF in this mission: create the flight dir
        # and recorder. We do this BEFORE clearing recording_paused so
        # vision_worker / log_worker never see a "recording active but
        # recorder is None" race.
        if new_phase == Phase.TAKEOFF and recorder_box[0] is None:
            # Refresh the serial from live telemetry. The startup
            # capture (5 s window) returns "unknown" if the drone
            # wasn't connected yet; telemetry_worker keeps the holder
            # current after a late connect, so by TAKEOFF time the
            # real serial is usually available. Without this refresh
            # the flight directory ends up named "..._unknown" even
            # though the drone is plainly connected.
            nonlocal serial, calibration
            tel_now = tel_holder.get()
            if tel_now is not None and tel_now.serial_number:
                serial = tel_now.serial_number
            # If we loaded calibration against a stale serial at
            # startup (drone connected after the 5 s startup window),
            # retry with the now-known serial. Otherwise the entire
            # flight runs with default Anafi intrinsics -- cx/cy
            # offset by ~5 % of image dimension, fx/fy off by ~3 % --
            # and every marker pose is biased proportionally.
            if calibration.serial != serial:
                try:
                    new_cal = store.load(serial, calibration.resolution,
                                          allow_default=True)
                    # Only swap if the new load IS a calibrated one,
                    # OR if the existing was already a default (so
                    # mission_meta at least records the live serial
                    # in calibration.serial instead of "unknown").
                    if not new_cal.is_default or calibration.is_default:
                        calibration = new_cal
                        detector.calibration = new_cal
                        print(f"[mission] calibration refreshed at "
                              f"TAKEOFF: {calibration.short_summary()}")
                except Exception as e:
                    print(f"[mission] calibration refresh failed: {e}")
            new_dir = make_flight_dir(FLIGHTS_DIR, serial)
            recorder_box[0] = FlightRecorder(new_dir, fps=cfg.record_fps)
            flight_dir_box[0] = new_dir
            print(f"[mission] flight artefacts: {new_dir} (serial={serial})")
            try:
                cfg.save(new_dir / "cfg_start.json")
            except Exception as e:
                print(f"[mission] cfg_start.json save failed: {e}")
            # Per-flight copy of the executing mission script. Rendered
            # from the controller's installed step list so what we save
            # matches what the drone actually walks (post-default-fill,
            # post-canonicalisation), not the raw textarea contents.
            try:
                with state.lock:
                    steps = list(state.mission_script)
                if steps:
                    (new_dir / PER_FLIGHT_SCRIPT_FILENAME).write_text(
                        ms.format(steps) + "\n")
            except Exception as e:
                print(f"[mission] mission_script.txt save failed: {e}")
            # Per-flight snapshot of the active arena config. The
            # active path can be re-saved between flights; pinning the
            # exact arena layout that this flight was estimated against
            # makes per-flight forensics independent of what the
            # operator subsequently edits in the Arena tab.
            try:
                arena_now = arena_holder.get()
                if arena_now is not None:
                    (new_dir / "arena_config.json").write_text(
                        json.dumps(arena_now.to_json_dict(), indent=2))
            except Exception as e:
                print(f"[mission] arena_config.json save failed: {e}")
            # Code provenance: which git commit was running when this
            # flight started? Lets us match flight-log behaviour to the
            # exact source even after subsequent commits to the
            # marker_mission package.
            try:
                from .git_provenance import describe_marker_mission_commit
                commit_text = describe_marker_mission_commit()
                if commit_text:
                    (new_dir / "git_commit.txt").write_text(commit_text + "\n")
            except Exception as e:
                print(f"[mission] git_commit.txt save failed: {e}")

        # On every takeoff: reset zoom to 1x, then trigger the same
        # full camera restart as the "Restart Camera" button in the UI.
        if new_phase == Phase.TAKEOFF:
            try:
                api.camera_config_set(zoom=1.0)
            except Exception as e:
                print(f"[mission] zoom reset at TAKEOFF failed: {e}")
            errs = ui.restart_camera()
            if errs:
                print(f"[mission] camera restart at TAKEOFF: {errs}")

        # Recording envelope.
        if new_phase in AIRBORNE_PHASES:
            if recording_paused.is_set():
                print(f"[rec] starting recording (phase={new_phase.value})")
            recording_paused.clear()
        else:
            if not recording_paused.is_set():
                print(f"[rec] stopping recording (phase={new_phase.value})")
            recording_paused.set()

        # End-of-flight snapshot. Only if a flight actually happened
        # (cfg_start.json was written above).
        if new_phase in (Phase.DONE, Phase.ABORT):
            fdir = flight_dir_box[0]
            if fdir is not None and (fdir / "cfg_start.json").exists():
                try:
                    cfg.save(fdir / "cfg_end.json")
                except Exception as e:
                    print(f"[mission] cfg_end.json save failed: {e}")

    # Build the controller before the UI so the UI's "Start mission" button
    # can call into it.
    controller = MissionController(api, cfg, state,
                                   frame_pose_provider=pose_holder.get,
                                   telemetry_provider=tel_holder.get,
                                   on_phase_change=on_phase_change,
                                   arena_provider=arena_holder.get,
                                   drone_threat_provider=drone_threat_holder.get,
                                   telemetry_age_provider=tel_holder.age_s)

    # The Stop button calls controller.stop() which blocks for up to ~25 s
    # waiting for the safe-shutdown to finish. We don't want the Flask
    # request thread hanging that long, so kick the stop off on a daemon
    # thread and return immediately. The control thread itself does the
    # rc_zero + land + telemetry poll synchronously.
    def request_stop_async():
        # Set abort_reason synchronously BEFORE returning so the next
        # CSV row (and the forced flush below) timestamp the
        # emergency-land moment precisely. The shutdown thread will
        # call controller.stop("UI stop request") which sets the same
        # field again -- idempotent, but capturing it here means the
        # log row recorded right now already shows the abort.
        with state.lock:
            if not state.abort_reason:
                state.abort_reason = "UI stop request"
        rec = recorder_box[0]
        if rec is not None:
            try:
                rec.log_row(state, pose_holder.get(), tel_holder.get())
            except Exception as e:
                print(f"[mission] emergency-stop log_row failed: {e}")
        threading.Thread(
            target=controller.stop,
            args=("UI stop request",),
            name="ui-stop-trigger",
            daemon=True,
        ).start()
        return True

    cal_capture = CalibrationCapture(reader, CALIB_DIR, fps=cfg.record_fps)

    # In combined-app mode the Flask lifecycle is owned by the
    # unified-server side; hand UiServer its app so our routes register
    # there instead of UiServer creating its own.
    external_app = None
    if in_process:
        import unified_api_server as _srv
        external_app = _srv.app
    ui = UiServer(state, latest_ann_frame,
                  host=cfg.ui_host, port=cfg.ui_port,
                  history_s=cfg.ui_telemetry_history_s,
                  on_start=controller.trigger,
                  on_stop=request_stop_async,
                  flights_root=FLIGHTS_DIR,
                  drone_connected=initially_connected,
                  calibration_capture=cal_capture,
                  cfg=cfg,
                  controller=controller,
                  flight_dir_provider=lambda: flight_dir_box[0],
                  arena_holder=arena_holder,
                  api=api,
                  stream_reader=reader,
                  external_app=external_app)
    ui.start()
    if in_process:
        print(f"[mission] UI routes registered on combined Flask app "
              f"(served by marker_mission.app)")
    else:
        print(f"[mission] UI running at {ui.url()} (camera) and {ui.url()}/charts")

    # ---------- 4. Background workers ------------------------------------
    stop = threading.Event()

    def telemetry_worker():
        # Poll telemetry independently of the control loop so the UI stays
        # responsive even when the controller is busy. Also keeps
        # ui.drone_connected in sync, and (re-)issues video_start_mjpeg
        # the first time the drone appears -- so the operator can plug
        # the drone in AFTER starting mission and the video feed kicks
        # in automatically.
        REARM_WAIT_S      = 3.0   # seconds after confirmed landing before re-takeoff
        REARM_TIMEOUT     = 120.0 # give up + stop after this many seconds
        NOT_FLYING_STREAK = 3     # consecutive not-flying readings to confirm landing
        period = 0.25
        was_connected = initially_connected
        video_started = initially_connected
        video_retry_after: float = 0.0
        was_flying = False
        not_flying_count = 0      # consecutive ticks with flying=False during active flight
        emergency_land_at: float = 0.0
        in_emergency_rearm = False
        takeoff_retry_after: float = 0.0
        while not stop.is_set():
            try:
                tel = api.telemetry()
                tel_holder.set(tel)
                connected_now = bool(tel.connected)
                flying_now = bool(tel.flying)
            except Exception:
                connected_now = False
                flying_now = False
            ui.drone_connected = connected_now

            # ── Emergency rearm: cancel immediately if operator stopped ──────
            # If abort_reason is set the stop was intentional (killswitch,
            # UI stop button) — never auto-rearm after an intentional stop.
            if in_emergency_rearm:
                with state.lock:
                    operator_stopped = bool(state.abort_reason)
                if operator_stopped:
                    print("[mission] operator stop detected — cancelling emergency rearm")
                    with state.lock:
                        state.emergency_rearm = False
                    in_emergency_rearm = False
                    not_flying_count = 0

            # ── Emergency rearm: clear immediately if drone is flying ──────
            # Always check first so a WLAN blip that triggered rearm gets
            # cleared the moment we see flying=True again.
            if in_emergency_rearm and flying_now:
                print("[mission] drone is flying — clearing emergency rearm, "
                      "resuming mission")
                with state.lock:
                    state.emergency_rearm = False
                    state.note = "mission resumed after re-takeoff"
                in_emergency_rearm = False
                not_flying_count = 0

            # ── Unexpected landing detection (require N consecutive ticks) ─
            # Only trigger if the drone was actually airborne (was_flying)
            # and no operator stop is pending (abort_reason == "").
            # This prevents spurious rearm at initial power-on (drone never
            # flew yet) and after a killswitch-triggered landing.
            if not flying_now and not in_emergency_rearm:
                with state.lock:
                    active_phase = state.phase
                    operator_stopped = bool(state.abort_reason)
                if (active_phase not in NON_AIRBORNE_PHASES
                        and was_flying
                        and not operator_stopped):
                    not_flying_count += 1
                    if not_flying_count >= NOT_FLYING_STREAK:
                        print(f"[mission] WARN: drone confirmed landed during "
                              f"phase '{active_phase.value}' — will re-takeoff")
                        emergency_land_at = time.monotonic()
                        in_emergency_rearm = True
                        takeoff_retry_after = emergency_land_at + REARM_WAIT_S
                        with state.lock:
                            state.emergency_rearm = True
                            state.note = "emergency rearm: waiting to re-takeoff"
                else:
                    not_flying_count = 0
            elif flying_now:
                not_flying_count = 0

            # ── Re-takeoff logic ───────────────────────────────────────────
            if in_emergency_rearm and not flying_now:
                elapsed = time.monotonic() - emergency_land_at
                if elapsed > REARM_TIMEOUT:
                    print("[mission] emergency rearm timeout — stopping mission")
                    with state.lock:
                        state.emergency_rearm = False
                    stop.set()
                    in_emergency_rearm = False
                elif (connected_now
                      and time.monotonic() >= takeoff_retry_after):
                    print(f"[mission] attempting re-takeoff "
                          f"({elapsed:.0f}s after unexpected landing)...")
                    try:
                        ok, msg = api.takeoff()
                        if ok:
                            print("[mission] re-takeoff command sent OK")
                        else:
                            print(f"[mission] re-takeoff failed: {msg} — retrying in 5s")
                        takeoff_retry_after = time.monotonic() + 5.0
                    except Exception as e:
                        print(f"[mission] re-takeoff error: {e} — retrying in 5s")
                        takeoff_retry_after = time.monotonic() + 5.0

            was_flying = flying_now
            if connected_now and not was_connected:
                print("[mission] drone connected (was offline at startup) "
                      "-- starting video stream")
                video_started = False       # force re-start after reconnect
                video_retry_after = 0.0
            if connected_now and not video_started:
                if time.monotonic() >= video_retry_after:
                    try:
                        ok, msg = api.video_start_mjpeg()
                        if ok:
                            video_started = True
                        else:
                            print(f"[mission] video_start_mjpeg failed: {msg}"
                                  f" — retrying in 5 s")
                            video_retry_after = time.monotonic() + 5.0
                    except Exception as e:
                        print(f"[mission] video_start_mjpeg error: {e}"
                              f" — retrying in 5 s")
                        video_retry_after = time.monotonic() + 5.0
            elif not connected_now and was_connected:
                print("[mission] drone went offline -- waiting to reconnect")
                video_started = False
                video_retry_after = 0.0
            was_connected = connected_now
            time.sleep(period)

    def vision_worker():
        # Pull the latest frame, run detection, push to recorder + UI.
        # If the upstream MJPEG stream stalls (no fresh JPEG decoded for
        # longer than pose_max_age_s), expire the held pose. Without this
        # the controller would keep reading the LAST detection from
        # pose_holder forever and act on it as if it were live, which is
        # what caused the "drone hits the wall" incident in flight
        # 2026-04-28_18-06-58_unknown (pose frozen identical for 120 s).
        last_seen_ts = 0.0
        last_expired = False

        # ── Other-drone detection ──────────────────────────────────────
        # Sparse optical flow + RANSAC homography compensates for camera
        # movement; residual motion blobs ≥ 20×20 px are drone candidates.
        # Temporal filter: a blob must appear in ≥ 3 of the last 5 frames.
        import collections as _col
        _dd_prev_gray  = None
        _dd_prev_threat_cx: float = None  # camera-x centre of last nearest blob
        _DD_MIN_W      = 30     # min blob width  (2:1 aspect → drones are wide)
        _DD_MIN_H      = 15     # min blob height
        _DD_MAX_W      = 300    # max blob width  (filters walls/ceiling)
        _DD_MAX_H      = 150    # max blob height (300×15/30 — keeps 2:1 ratio)
        _DD_CONSEC_REQ = 8      # consecutive frames required to confirm
        _DD_TOP_N      = 4      # max candidates shown (1 red + 3 yellow)
        _DD_MATCH_PX   = 40     # centre-distance for track→detection match
        # Active tracks: list of {'cx','cy','w','h','streak'}
        _dd_tracks: list = []

        def _detect_drones(raw_frame):
            """Return list of (x,y,w,h) bounding boxes of drone candidates,
            sorted largest→smallest (nearest first). Draws nothing."""
            nonlocal _dd_prev_gray
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            if _dd_prev_gray is None or _dd_prev_gray.shape != gray.shape:
                _dd_prev_gray = gray
                return []
            # Feature points in previous frame
            pts = cv2.goodFeaturesToTrack(
                _dd_prev_gray, maxCorners=300,
                qualityLevel=0.01, minDistance=8,
                blockSize=7)
            if pts is None or len(pts) < 12:
                _dd_prev_gray = gray
                return []
            # Track with LK optical flow
            pts1, st, _ = cv2.calcOpticalFlowPyrLK(
                _dd_prev_gray, gray, pts, None,
                winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            p0, p1 = pts[ok], pts1[ok]
            if len(p0) < 10:
                _dd_prev_gray = gray
                return []
            # RANSAC homography = camera-motion model
            H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
            if H is None:
                _dd_prev_gray = gray
                return []
            h, w = gray.shape
            warped = cv2.warpPerspective(
                _dd_prev_gray, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
            diff = cv2.absdiff(warped, gray)
            _, thr = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, ker)
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  ker)
            cnts, _ = cv2.findContours(
                thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = []
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw >= _DD_MIN_W and bh >= _DD_MIN_H \
                        and bw <= _DD_MAX_W and bh <= _DD_MAX_H:
                    blobs.append((x, y, bw, bh))
            blobs.sort(key=lambda b: -(b[2]*b[3]))
            _dd_prev_gray = gray
            return blobs[:_DD_TOP_N * 2]

        def _temporal_filter(current_blobs):
            """Update tracks with current detections.
            A track must have streak >= _DD_CONSEC_REQ consecutive frames
            to appear in the overlay. Returns confirmed boxes sorted by area."""
            nonlocal _dd_tracks
            matched_track_idx = set()
            matched_blob_idx  = set()
            # Match current blobs to existing tracks by centre proximity.
            for ti, tr in enumerate(_dd_tracks):
                best_d, best_bi = 9999, -1
                for bi, box in enumerate(current_blobs):
                    if bi in matched_blob_idx:
                        continue
                    cx = box[0] + box[2] // 2
                    cy = box[1] + box[3] // 2
                    d = abs(cx - tr['cx']) + abs(cy - tr['cy'])
                    if d < _DD_MATCH_PX and d < best_d:
                        best_d, best_bi = d, bi
                if best_bi >= 0:
                    box = current_blobs[best_bi]
                    tr['cx'] = box[0] + box[2] // 2
                    tr['cy'] = box[1] + box[3] // 2
                    tr['w']  = box[2]; tr['h'] = box[3]
                    tr['streak'] += 1
                    matched_track_idx.add(ti)
                    matched_blob_idx.add(best_bi)
                else:
                    tr['streak'] = 0   # missed this frame
            # Remove dead tracks (missed); add new tracks for unmatched blobs.
            _dd_tracks = [t for t in _dd_tracks if t['streak'] > 0]
            for bi, box in enumerate(current_blobs):
                if bi not in matched_blob_idx:
                    _dd_tracks.append({
                        'cx': box[0]+box[2]//2, 'cy': box[1]+box[3]//2,
                        'w': box[2], 'h': box[3], 'streak': 1})
            # Return confirmed tracks sorted largest→smallest (nearest first).
            confirmed = [
                (t['cx']-t['w']//2, t['cy']-t['h']//2, t['w'], t['h'])
                for t in _dd_tracks if t['streak'] >= _DD_CONSEC_REQ
            ]
            confirmed.sort(key=lambda b: -(b[2]*b[3]))
            return confirmed[:_DD_TOP_N]
        # ──────────────────────────────────────────────────────────────

        # ── Arena Validator: Track visible markers + 30s history ──────────
        _VALIDATOR_HIST_MAX = 30.0  # seconds to keep historical markers
        _validator_markers = {}  # {mid: {wx, wy, wz, detected_at, last_seen_at, conf, is_visible}}
        _validator_last_update = 0.0  # time of last state exposure
        _VALIDATOR_UPDATE_INTERVAL = 0.5  # ~2x per second
        # ──────────────────────────────────────────────────────────────────

        # Freeze detection: track last frame timestamp change and restart cooldown.
        _freeze_last_ts: float = 0.0
        _freeze_restart_after: float = 0.0   # monotonic cooldown (vision_worker local)
        _FREEZE_TIMEOUT_S  = 3.0   # stale frame age that counts as frozen
        _FREEZE_COOLDOWN_S = 3.0   # min seconds between auto-restarts
        _MOVE_SPEED_CMS    = 5.0   # drone speed threshold (cm/s) for "moving"
        # Position Kalman filter state. Lives across loop iterations
        # (function-local closure variables persist). Reset on
        # disable -> enable transitions and on dt > kalman_reset_gap_s.
        position_kf = PositionKalman()
        kf_last_t = 0.0
        kf_enabled_prev = False
        # Marker position history for IMU dead-reckoning.
        # imu_disp accumulates arena-frame displacement (m) since the
        # last cache reset; stored per-entry so correction =
        # current_imu_disp - stored_imu_disp.
        marker_cache = MarkerPositionCache()
        imu_disp = np.zeros(3, dtype=float)
        imu_last_t: float = 0.0
        # IMU dead-reckoning anchor: last marker-confirmed position and
        # the imu_disp value at that moment. Used for 25% IMU blend.
        imu_anchor_pos: Optional[np.ndarray] = None
        imu_anchor_disp = np.zeros(3, dtype=float)
        _IMU_WEIGHT = 0.25   # fraction of IMU estimate in final blend
        while not stop.is_set():
            frame, jpg, ts = reader.latest()
            # ── Freeze detection (runs even when no new frame arrives) ──
            now_ft = time.monotonic()
            stats = reader.stats
            frame_age = stats.get("age_s") or 0.0
            if frame_age > _FREEZE_TIMEOUT_S and now_ft >= _freeze_restart_after:
                tel_now = tel_holder.get()
                if tel_now is not None and tel_now.flying:
                    try:
                        spd = math.hypot(
                            float(tel_now.raw.get("vgx", 0) or 0),
                            float(tel_now.raw.get("vgy", 0) or 0),
                        )
                    except (TypeError, ValueError):
                        spd = 0.0
                    if spd >= _MOVE_SPEED_CMS:
                        print(f"[vision] frame frozen {frame_age:.1f}s while "
                              f"moving ({spd:.0f} cm/s) — restarting camera")
                        errs = ui.restart_camera()
                        _freeze_restart_after = now_ft + _FREEZE_COOLDOWN_S
                        with state.lock:
                            state.camera_restart_count += 1
                        if errs:
                            print(f"[vision] camera restart errors: {errs}")
            # ────────────────────────────────────────────────────────────

            if frame is None or ts == last_seen_ts:
                if (last_seen_ts and not last_expired
                        and time.monotonic() - last_seen_ts
                            > cfg.pose_max_age_s):
                    pose_holder.set(None)
                    last_expired = True
                    print(f"[vision] no new frame for "
                          f">{cfg.pose_max_age_s:.1f}s -- expiring held pose")
                time.sleep(0.01)
                continue
            last_seen_ts = ts
            last_expired = False

            try:
                # Pick up runtime cfg toggles (e.g. operator flips
                # mirror_collapse on /tune mid-flight). Cheap attribute
                # write; the detector reads it per-frame inside the
                # IPPE branch picker.
                detector.enable_mirror_collapse = (
                    cfg.enable_ippe_mirror_collapse)
                detector.mirror_collapse_sum_hdg_deg = float(
                    cfg.mirror_collapse_sum_hdg_deg)
                detector.mirror_collapse_max_hdg_deg = float(
                    cfg.mirror_collapse_max_hdg_deg)
                # Live-sync marker size from the active arena. The
                # operator-facing source of truth is /arena, not cfg
                # -- a Save in the Arena tab swaps the active arena
                # via arena_holder, and the detector picks up the new
                # size on the next frame. set_marker_size is a no-op
                # when the value is unchanged.
                _arena_now = arena_holder.get()
                detector.set_marker_size(
                    _arena_now.marker_size_m if _arena_now is not None
                    else cfg.marker_size_m)
                # Per-marker size resolver. Targets (ids 31..46 by
                # default) are 18cm boxes while wall markers are 50cm;
                # without this the detector would assume the arena
                # default for both and inflate distance estimates by
                # ~2.8× for whichever family doesn't match. Arena
                # JSON can also carry a per-marker ``size_m`` override
                # that wins over the cfg fallback below.
                def _size_for(mid: int,
                              _arena=_arena_now,
                              _min=int(cfg.target_marker_id_min),
                              _max=int(cfg.target_marker_id_max),
                              _tsize=float(cfg.target_marker_size_m)) -> float:
                    if _arena is not None:
                        m = _arena.markers.get(int(mid))
                        if m is not None and m.size_m is not None:
                            return float(m.size_m)
                    if _min <= int(mid) <= _max:
                        return _tsize
                    if _arena is not None:
                        return float(_arena.marker_size_m)
                    return float(cfg.marker_size_m)
                detector.set_size_resolver(_size_for)
                # Detect every visible marker so the operator sees them
                # all in the video overlay -- the script can target any
                # marker, and even non-target markers in frame are
                # useful situational context. The controller still gets
                # only the active target via pose_holder.
                poses = detector.detect(frame, wanted_id=None)
                # state.active_marker_id is the runtime target -- a
                # script's APPROACH step can change it without
                # polluting cfg.target_marker_id. Fall back to cfg if
                # state hasn't been seeded yet (shouldn't happen --
                # MissionController.__init__ seeds it).
                with state.lock:
                    active_mid = state.active_marker_id
                if active_mid is None:
                    active_mid = cfg.target_marker_id
                target = next((p for p in poses
                               if p.marker_id == active_mid), None)
                # Magnetometer-aware swap of the active marker's
                # relative_heading_deg. When the chosen IPPE branch
                # disagrees with tel.yaw + arena_offset by more than
                # the alt branch does, swap to alt's heading -- the
                # controller's body-frame projection during APPROACH
                # / ALIGN / HOLD then tracks the right side of the
                # marker normal from the very first frame instead
                # of flickering +/-15 deg with corner-noise jitter.
                # All four conditions must hold; otherwise we leave
                # the chosen-branch pose alone (geometry-only).
                arena_for_swap = arena_holder.get()
                tel_for_swap = tel_holder.get()
                tel_yaw_for_swap = (float(tel_for_swap.yaw_deg)
                                    if tel_for_swap is not None
                                    and tel_for_swap.yaw_deg is not None
                                    else None)
                if (target is not None
                        and target.alt_relative_heading_deg is not None
                        and arena_for_swap is not None
                        and arena_for_swap.magnetic_north_arena_yaw_deg is not None
                        and tel_yaw_for_swap is not None):
                    marker_for_swap = arena_for_swap.markers.get(
                        int(target.marker_id))
                    if marker_for_swap is not None:
                        expected_yaw = _wrap_180(
                            tel_yaw_for_swap
                            + float(arena_for_swap
                                    .magnetic_north_arena_yaw_deg))
                        chosen_yaw = _arena_yaw_for_branch(
                            target, marker_for_swap, "chosen")
                        alt_yaw = _arena_yaw_for_branch(
                            target, marker_for_swap, "alt")
                        if (chosen_yaw is not None
                                and alt_yaw is not None):
                            d_chosen = _yaw_diff(chosen_yaw, expected_yaw)
                            d_alt = _yaw_diff(alt_yaw, expected_yaw)
                            if (min(d_chosen, d_alt)
                                    <= ARENA_MAG_SLACK_DEG
                                    and d_alt < d_chosen):
                                target = _dc_replace(
                                    target,
                                    relative_heading_deg=
                                        target.alt_relative_heading_deg,
                                    pose_method="ippe_mag_swap")
                pose_holder.set(target)
                # Publish the full visible-marker set to state so AWAIT
                # can early-exit when its target id appears, even if
                # that id isn't the controller's active marker.
                with state.lock:
                    state.visible_marker_ids = [int(p.marker_id)
                                                 for p in poses]
                # Arena world-position estimate from every visible
                # reference marker (weighted by inverse distance).
                # When no reference marker is visible (or arena_config
                # is missing) we leave the previous fix in place and
                # let world_position_age_s grow -- the operator still
                # gets a stale-but-useful position estimate during
                # marker loss, colour-coded by age in the UI.
                arena = arena_for_swap   # reuse holder snapshot above
                if arena is not None:
                    # Snapshot the previous sticky world position +
                    # its age so estimate_position can anchor branch
                    # selection to it: a chosen IPPE pose that's
                    # metres from the last good fix is the wrong
                    # branch on a noisy frame, not a real teleport.
                    # Also honour reset_position_estimator: the
                    # controller raises it on TAKEOFF entry so we
                    # don't carry a poisoned anchor across missions.
                    with state.lock:
                        reset_req = state.reset_position_estimator
                        if reset_req:
                            state.world_position_m = None
                            state.world_position_updated_at = 0.0
                            state.reset_position_estimator = False
                        prev_wp = state.world_position_m
                        prev_at = state.world_position_updated_at
                    if reset_req:
                        # Clear the Kalman state too so the next fresh
                        # measurement initialises it cleanly instead
                        # of being pulled toward the stale mean.
                        position_kf.reset()
                        kf_last_t = 0.0
                        kf_enabled_prev = False
                        # Reset marker history and IMU state so stale
                        # pre-takeoff observations don't bleed in.
                        marker_cache.reset()
                        imu_disp = np.zeros(3, dtype=float)
                        imu_last_t = 0.0
                        imu_anchor_pos = None
                        imu_anchor_disp = np.zeros(3, dtype=float)
                        print("[vision] position estimator reset (TAKEOFF)")

                    # Integrate arena-frame IMU velocity for dead-reckoning.
                    now_mono = time.monotonic()
                    # Integrate arena-frame IMU velocity for dead-reckoning.
                    if (arena_for_swap is not None
                            and arena_for_swap.magnetic_north_arena_yaw_deg
                                is not None
                            and tel_for_swap is not None
                            and imu_last_t > 0.0):
                        try:
                            _vN = float(tel_for_swap.raw["vgx"]) / 100.0
                            _vE = float(tel_for_swap.raw["vgy"]) / 100.0
                            _vD = float(tel_for_swap.raw["vgz"]) / 100.0
                            _v  = ned_velocity_to_arena(
                                _vN, _vE, _vD,
                                float(arena_for_swap
                                      .magnetic_north_arena_yaw_deg))
                            imu_disp += _v * (now_mono - imu_last_t)
                        except (KeyError, TypeError, ValueError):
                            pass
                    if imu_last_t == 0.0:
                        imu_last_t = now_mono
                    imu_last_t = now_mono
                    marker_cache.prune(now_mono)

                    prev_age_s = ((time.monotonic() - prev_at)
                                   if prev_at > 0.0 else None)
                    # Drone altimeter (barometer) reading for the
                    # altitude-aware IPPE branch picker. Optional: None
                    # disables that layer.
                    tel_height_m: Optional[float] = None
                    if tel_for_swap is not None:
                        h_cm = tel_for_swap.raw.get("height_cm")
                        if h_cm is not None:
                            try:
                                tel_height_m = float(h_cm) / 100.0
                            except (TypeError, ValueError):
                                tel_height_m = None
                    est = estimate_position(
                        arena, poses,
                        prev_position_m=prev_wp,
                        prev_age_s=prev_age_s,
                        tel_yaw_deg=tel_yaw_for_swap,
                        tel_height_m=tel_height_m,
                        enable_arena_oob_filter=cfg.enable_ippe_arena_oob_filter,
                        enable_alt_branch_swap=cfg.enable_ippe_alt_branch_swap,
                        enable_prev_anchor=cfg.enable_ippe_prev_anchor,
                        enable_aggregate_oob_discard=
                            cfg.enable_ippe_aggregate_oob_discard,
                        marker_cache=marker_cache,
                        cache_now=now_mono,
                        cache_imu_disp=imu_disp)
                    # Refresh the marker cache with fresh primary
                    # observations (dist available from poses).
                    _pose_dist = {int(p.marker_id): float(p.distance_m)
                                  for p in poses}
                    if est is not None:
                        for mid, pos_w in est.per_marker_position_m.items():
                            dist = _pose_dist.get(mid, 1.0)
                            marker_cache.record(
                                mid, pos_w, dist, now_mono, imu_disp)
                        # 75 % marker estimate + 25 % IMU dead-reckoning.
                        # imu_anchor_pos is the last marker-confirmed
                        # position; imu_disp − imu_anchor_disp is the
                        # displacement since that anchor.
                        marker_pos = np.asarray(est.position_m, dtype=float)
                        if imu_anchor_pos is not None:
                            imu_pos = (imu_anchor_pos
                                       + (imu_disp - imu_anchor_disp))
                            blended = ((1.0 - _IMU_WEIGHT) * marker_pos
                                       + _IMU_WEIGHT * imu_pos)
                        else:
                            blended = marker_pos
                        # Update anchor to the blended result.
                        imu_anchor_pos = blended.copy()
                        imu_anchor_disp = imu_disp.copy()
                        # Override est.position_m with blended value.
                        est = type(est)(
                            position_m=blended,
                            used_markers=est.used_markers,
                            per_marker_position_m=est.per_marker_position_m,
                            weights=est.weights,
                            per_marker_method=est.per_marker_method,
                            confidence=est.confidence,
                        )
                    # Always update confidence (0 if no estimate).
                    with state.lock:
                        state.world_position_confidence = float(
                            est.confidence if est is not None else 0.0)
                    if est is not None:
                        # Optional position Kalman filter. When enabled,
                        # the world_position_m we publish is the KF
                        # smoothed value; the per-marker raw values
                        # below are unaffected (operator can still see
                        # what each marker individually voted for).
                        if cfg.enable_position_kalman:
                            kf_now = time.monotonic()
                            kf_dt = ((kf_now - kf_last_t)
                                     if kf_last_t > 0.0 else 0.0)
                            # Reset on disable -> enable, or large gap.
                            if (not kf_enabled_prev
                                    or kf_dt > cfg.kalman_reset_gap_s):
                                position_kf.reset()
                                kf_dt = 0.0
                            # Sync params live so /tune-time changes
                            # take effect on the next tick.
                            position_kf.accel_var = float(cfg.kalman_accel_var)
                            position_kf.pos_meas_var = float(
                                cfg.kalman_pos_meas_var)
                            position_kf.vel_meas_var = float(
                                cfg.kalman_vel_meas_var)
                            if kf_dt > 0.0 and position_kf.initialised:
                                position_kf.predict(kf_dt)
                            position_kf.update_position(
                                np.asarray(est.position_m, dtype=float))
                            # IMU velocity update (NED -> arena, 25% weight
                            # already applied via blended position above).
                            if (arena_for_swap is not None
                                    and arena_for_swap
                                    .magnetic_north_arena_yaw_deg is not None
                                    and tel_for_swap is not None):
                                try:
                                    vN = float(
                                        tel_for_swap.raw["vgx"]) / 100.0
                                    vE = float(
                                        tel_for_swap.raw["vgy"]) / 100.0
                                    vD = float(
                                        tel_for_swap.raw["vgz"]) / 100.0
                                    arena_vel = ned_velocity_to_arena(
                                        vN, vE, vD,
                                        float(arena_for_swap
                                              .magnetic_north_arena_yaw_deg))
                                    position_kf.update_velocity(arena_vel)
                                except (KeyError, TypeError, ValueError):
                                    pass
                            kf_last_t = kf_now
                            kf_pos = position_kf.position()
                            kf_vel = position_kf.velocity()
                            world_pos = (
                                (float(kf_pos[0]), float(kf_pos[1]),
                                 float(kf_pos[2]))
                                if kf_pos is not None
                                else (float(est.position_m[0]),
                                      float(est.position_m[1]),
                                      float(est.position_m[2])))
                            world_vel: Optional[tuple[float, float, float]] = (
                                (float(kf_vel[0]), float(kf_vel[1]),
                                 float(kf_vel[2]))
                                if kf_vel is not None else None)
                        else:
                            # KF disabled: clean up state so re-enable
                            # starts fresh, and pass the raw aggregate
                            # through unchanged.
                            if kf_enabled_prev:
                                position_kf.reset()
                                kf_last_t = 0.0
                            world_pos = (float(est.position_m[0]),
                                         float(est.position_m[1]),
                                         float(est.position_m[2]))
                            world_vel = None
                        kf_enabled_prev = cfg.enable_position_kalman
                        with state.lock:
                            state.world_position_m = world_pos
                            state.world_position_updated_at = time.monotonic()
                            state.world_position_used_markers = list(
                                est.used_markers)
                            state.world_position_pose_methods = [
                                est.per_marker_method.get(mid, "")
                                for mid in est.used_markers]
                            state.world_position_per_marker = [
                                tuple(float(c) for c in
                                      est.per_marker_position_m[mid])
                                for mid in est.used_markers]
                            state.world_velocity_m_kf = world_vel
                            # Update arena validator with visible markers.
                            now_val = time.monotonic()
                            for mid, pos_m in est.per_marker_position_m.items():
                                if mid not in _validator_markers:
                                    _validator_markers[mid] = {
                                        'wx': float(pos_m[0]),
                                        'wy': float(pos_m[1]),
                                        'wz': float(pos_m[2]),
                                        'detected_at': now_val,
                                        'last_seen_at': now_val,
                                        'conf': float(est.weights.get(mid, 0.5)),
                                        'is_visible': True,
                                    }
                                else:
                                    m = _validator_markers[mid]
                                    m['wx'] = float(pos_m[0])
                                    m['wy'] = float(pos_m[1])
                                    m['wz'] = float(pos_m[2])
                                    m['last_seen_at'] = now_val
                                    m['conf'] = float(est.weights.get(mid, m['conf']))
                                    m['is_visible'] = True
                            # Prune old markers (>30s) and mark stale ones.
                            prune_before = now_val - _VALIDATOR_HIST_MAX
                            to_delete = []
                            for mid, m in _validator_markers.items():
                                if m['detected_at'] < prune_before:
                                    to_delete.append(mid)
                                elif mid not in est.per_marker_position_m:
                                    m['is_visible'] = False
                            for mid in to_delete:
                                del _validator_markers[mid]
                            # Expose to state ~2x per second (already holding state.lock).
                            if now_val - _validator_last_update >= _VALIDATOR_UPDATE_INTERVAL:
                                state.arena_validation_map = dict(_validator_markers)
                                _validator_last_update = now_val
                # Active target's pose method (or empty if not in view).
                with state.lock:
                    state.target_pose_method = (
                        target.pose_method if target is not None else "")
                # Arena-frame drone yaw -- published whenever ANY visible
                # marker is also a reference marker in the arena config.
                # The world-position estimate gives the bearing of that
                # marker from the drone in the arena, and the per-pose
                # camera-frame angle (yaw_deg) is what's left over once
                # we subtract that bearing. We prefer the active marker
                # (matches the controller's smoothed yaw_to_marker
                # tracking) but fall back to any other ref-marker so a
                # TO step can run even when the active marker has left
                # the frame.
                if arena is not None and state.world_position_m is not None and poses:
                    candidate = None
                    if target is not None and int(target.marker_id) in arena.markers:
                        candidate = target
                    else:
                        for p in poses:
                            if int(p.marker_id) in arena.markers:
                                candidate = p
                                break
                    if candidate is not None:
                        marker_arena = arena.markers[int(candidate.marker_id)]
                        wp_now = state.world_position_m
                        dxm = float(marker_arena.position_m[0]) - wp_now[0]
                        dym = float(marker_arena.position_m[1]) - wp_now[1]
                        bearing_deg = math.degrees(math.atan2(dxm, dym))
                        with state.lock:
                            state.arena_yaw_deg = (
                                bearing_deg - float(candidate.yaw_deg))
                            state.arena_yaw_updated_at = time.monotonic()
                # Status overlay -------------------------------------------
                # Use the smoothed values from state.snapshot() so the
                # baked-in overlay matches what the CSV records and what
                # the controller actually acts on. Earlier we used the
                # raw per-frame pose (target.*), which diverged from the
                # smoothed CSV columns whenever the smoother absorbed an
                # IPPE branch flip or jitter -- the operator then saw
                # video showing -16 deg next to a CSV row at +12 deg
                # at the same instant.
                snap = state.snapshot()
                # Marker size shown on the overlay must match what
                # solvePnP is actually using — detector.marker_size is
                # kept in sync with arena_holder.get().marker_size_m
                # every tick above, so it's the single source of truth.
                # cfg.marker_size_m was the older path and is stale
                # whenever an arena has been loaded (which is always,
                # in production).
                # Current camera digital-zoom factor. The controller drives it
                # (1.0x normally; SEARCH escalates to 1.25/2/3x to spot distant
                # markers, then resets to 1.0x on approach), so reading its
                # tracked value shows the operator the live zoom on the overlay.
                zoom_x = float(getattr(controller, "_search_zoom", 1.0) or 1.0)
                lines = [
                    f"phase: {snap['phase']}  zoom={zoom_x:.2f}x",
                    f"target id={active_mid}  "
                    f"size={detector.marker_size*100:.0f}cm",
                ]
                d_s = snap.get("distance_m")
                y_s = snap.get("yaw_to_marker_deg")
                h_s = snap.get("relative_heading_deg")
                if target and d_s is not None:
                    lines.append(
                        f"d={d_s:.2f}m  yaw={y_s:+.1f}deg  "
                        f"hdg={h_s:+.1f}deg")
                else:
                    lines.append("marker: NOT VISIBLE")
                ann = annotate_frame(frame, poses,
                                     target_id=active_mid,
                                     extra_lines=lines)
                # Arena mini-map in the upper-right corner: 1 m
                # grid, all known markers (wall-coloured dots),
                # currently-visible markers ringed in white, yellow
                # drone dot at state.world_position_m + yellow yaw
                # arrow from state.arena_yaw_deg.
                if arena_for_swap is not None:
                    with state.lock:
                        wp = state.world_position_m
                        ay_deg = state.arena_yaw_deg
                        seen_ids = list(state.visible_marker_ids)
                    draw_arena_minimap(
                        ann,
                        arena_width_m=arena_for_swap.width_m,
                        arena_depth_m=arena_for_swap.depth_m,
                        world_pos=wp,
                        arena_yaw_deg=ay_deg,
                        markers=arena_for_swap.markers,
                        visible_marker_ids=seen_ids,
                    )
                # ── Other-drone detection + overlay ───────────────────
                raw_blobs = _detect_drones(frame)
                confirmed  = _temporal_filter(raw_blobs)
                if confirmed:
                    _bx, _by, _bw, _bh = confirmed[0]
                    _threat_cx = _bx + _bw / 2.0
                    _threat_dx = ((_threat_cx - _dd_prev_threat_cx)
                                  if _dd_prev_threat_cx is not None else 0.0)
                    _dd_prev_threat_cx = _threat_cx
                    drone_threat_holder.set(_bw, _threat_dx)
                else:
                    _dd_prev_threat_cx = None
                    drone_threat_holder.set(0, 0.0)
                if confirmed:
                    # Largest = nearest (red), rest yellow
                    for i, (bx, by, bw, bh) in enumerate(confirmed):
                        color = (0, 0, 255) if i == 0 else (0, 255, 255)
                        cv2.rectangle(ann, (bx, by),
                                      (bx + bw, by + bh), color, 2)
                        label = "drone" if i > 0 else "NEAREST"
                        cv2.putText(ann, label,
                                    (bx, max(by - 4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, color, 1, cv2.LINE_AA)
                # ──────────────────────────────────────────────────────
                latest_ann_frame.set(ann)
                # Record both raw and annotated ----------------------------
                if not recording_paused.is_set():
                    rec = recorder_box[0]
                    if rec is not None:
                        rec.push_raw_frame(frame)
                        rec.push_annotated_frame(ann)
            except Exception as e:
                print(f"[vision] error: {e}")

    def log_worker():
        # Periodic CSV row; this gives us a sample even when no frames are
        # arriving (useful for diagnosing video drops).
        period = 1.0 / max(2.0, cfg.control_rate_hz / 2)
        while not stop.is_set():
            if not recording_paused.is_set():
                rec = recorder_box[0]
                if rec is not None:
                    rec.log_row(state, pose_holder.get(), tel_holder.get())
            time.sleep(period)

    threading.Thread(target=telemetry_worker, daemon=True,
                     name="telemetry").start()
    threading.Thread(target=vision_worker, daemon=True,
                     name="vision").start()
    threading.Thread(target=log_worker, daemon=True,
                     name="log").start()

    # ---------- 5. Mission ------------------------------------------------
    # Graceful shutdown on Ctrl-C / SIGTERM. We set ``stop`` BEFORE asking
    # the controller to stop (vs. after) so the main wait loop below can
    # exit promptly even if the control thread is stuck in a blocking
    # HTTP call (api.takeoff/land have a 20 s timeout, api.rc has 2 s) --
    # otherwise we'd block on its .join until the HTTP call returns.
    #
    # The control thread runs _safe_shutdown on its way out (rc_zero +
    # land + ~15 s telemetry poll), which is why this signal handler
    # blocks for up to ~25 s. Repeated Ctrl-C during shutdown is harmless
    # but slow, so we early-return if we're already in shutdown.
    def handle_signal(signum, _frame):
        if stop.is_set():
            print(f"\n[mission] signal {signum} -- already shutting down; "
                  f"please wait for the drone to land "
                  f"(or kill -9 if it hangs)")
            return
        print(f"\n[mission] signal {signum} -- requesting safe stop "
              f"(drone will land if airborne)")
        stop.set()
        controller.stop("operator interrupt")

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    # Arm the controller for the FIRST mission. It parks in INIT until
    # the operator presses "Start mission" in the web UI (which calls
    # controller.trigger via POST /api/start). If the drone wasn't
    # reachable at startup the button stays disabled until
    # telemetry_worker sees a connection.
    if initially_connected:
        print(f"[mission] armed -- open {ui.url()} and press 'Start mission' "
              f"to take off. Ctrl-C to quit.")
    else:
        print(f"[mission] UI ready at {ui.url()}. Browse recorded flights "
              f"at {ui.url()}/replay.")
        print(f"[mission] Start mission will unlock once the drone "
              f"connects via the API server.")

    # ---------- 5b. Mission loop -----------------------------------------
    # Each iteration runs one start-to-end mission cycle: arm the
    # controller, wait for it to terminate (operator pressed Start ->
    # full flight, or Stop -> safe shutdown, or no-op exit), finalize
    # that flight's artefacts, roll a fresh flight directory, reset the
    # state machine, and loop. Ctrl-C / SIGTERM sets ``stop`` and breaks
    # us out of the loop into final cleanup.
    last_phase = "init"
    mission_idx = 0
    try:
        while not stop.is_set():
            mission_idx += 1
            controller.start()
            # Block until the controller thread exits (mission finished
            # or stop requested). Daemon thread, so even if it overruns
            # we'll exit on stop.is_set().
            while controller.is_running() and not stop.is_set():
                time.sleep(0.5)
            if stop.is_set():
                break

            final = state.snapshot()
            last_phase = final["phase"]
            print(f"[mission #{mission_idx}] final phase: {last_phase} "
                  f"(reason: {final['note'] or final['abort_reason'] or '—'})")

            # ---- Finalize this flight's artefacts (write meta + close
            # video / csv + run the H.264 re-encode). Only fires if a
            # flight actually happened -- a controller exit without ever
            # entering TAKEOFF (operator pressed Stop in INIT, etc.)
            # leaves recorder_box[0] / flight_dir_box[0] both None and we
            # have nothing to finalize.
            recording_paused.set()
            if recorder_box[0] is not None and flight_dir_box[0] is not None:
                outcome = {
                    "final_phase": last_phase,
                    "abort_reason": final["abort_reason"],
                    "note": final["note"],
                    "serial": serial,
                }
                write_meta(flight_dir_box[0], cfg, calibration, outcome)
                recorder_box[0].stop(meta=None)
                print(f"[mission #{mission_idx}] artefacts saved in "
                      f"{flight_dir_box[0]}")
            else:
                print(f"[mission #{mission_idx}] no flight performed -- "
                      f"nothing to save")
            # Drop the box references. The next mission's TAKEOFF will
            # create a fresh flight_dir + recorder via on_phase_change.
            recorder_box[0] = None
            flight_dir_box[0] = None
            if stop.is_set():
                break

            # Reset the controller + shared state so the next iteration
            # arms cleanly back in INIT.
            controller.reset()
            print(f"[mission] re-armed -- press 'Start mission' to fly "
                  f"another, or Ctrl-C to quit.")
    except KeyboardInterrupt:
        stop.set()
        controller.stop("KeyboardInterrupt")

    # ---------- 6. Cleanup -----------------------------------------------
    stop.set()
    time.sleep(0.5)
    reader.stop()
    try:
        api.video_stop()
    except Exception:
        pass
    # Finalize whatever recorder is current, if it hasn't been stopped
    # already by the mission loop above. Skip silently if there was no
    # active recording (lazy creation never fired -- replay-only run).
    try:
        if recorder_box[0] is not None and flight_dir_box[0] is not None:
            last_outcome = {
                "final_phase": state.snapshot()["phase"],
                "abort_reason": state.snapshot()["abort_reason"],
                "note": state.snapshot()["note"],
                "serial": serial,
            }
            write_meta(flight_dir_box[0], cfg, calibration, last_outcome)
            recorder_box[0].stop(meta=None)
            print(f"[mission] artefacts saved in {flight_dir_box[0]}")
    except Exception as e:
        print(f"[mission] cleanup recorder.stop failed: {e}")
    return 0 if last_phase == "done" else 2


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marker_mission",
                                description="ArUco-based drone approach mission.")
    sub = p.add_subparsers(dest="cmd")

    pf = sub.add_parser("fly", help="run the full mission (default)")
    pf.add_argument("--marker-id", type=int, default=None,
                    help="target ArUco ID (default from config)")
    pf.add_argument("--marker-size", type=float, default=None,
                    help="physical side length of marker in metres")
    pf.add_argument("--distance", type=float, default=None,
                    help="final standoff distance from marker [m]")
    pf.add_argument("--heading", type=float, default=None,
                    help="final relative heading around marker [deg]")
    pf.add_argument("--hold", type=float, default=None,
                    help="hover duration before landing [s]")
    pf.add_argument("--api-url", default=None,
                    help="base URL of the unified API server "
                         "(e.g. http://127.0.0.1:5050)")
    pf.add_argument("--drone-ip", default=None,
                    help="Override the drone's IP address (default "
                         "192.168.42.1 for a real Anafi). Useful when "
                         "talking to the Sphinx simulator where the drone "
                         "may live at 127.0.0.1 (ports mode) or "
                         "10.202.0.1 (netns mode). Only effective when "
                         "the FC is started in-process by "
                         "marker_mission.app; HTTP mode (--api-url) "
                         "ignores it because the remote FC is already "
                         "configured with its own drone IP.")
    pf.add_argument("--ui-port", type=int, default=None,
                    help="port for the operator UI (default 8080)")
    pf.add_argument("--in-process", action="store_true",
                    help="Talk to unified_api_server directly in-memory "
                         "via drone_core (no HTTP). Requires that "
                         "marker_mission/app.py was the entry point so "
                         "the unified backend is loaded in this process. "
                         "Ignores --api-url.")
    pf.add_argument("--resolution", default="720p",
                    help="video resolution label for calibration lookup")
    pf.add_argument("--arena-config", default=None,
                    help="path to an arena_config*.json (marker world "
                         "positions + walls). Without it the world-position "
                         "estimator is disabled but everything else runs.")

    pc = sub.add_parser("calibrate", help="run intrinsic calibration on a video")
    pc.add_argument("--video", required=True, help="path to checkerboard video")
    pc.add_argument("--serial", required=True, help="drone serial number")
    pc.add_argument("--resolution", default="720p",
                    help="video resolution label (default 720p)")
    pc.add_argument("--pattern", default="9x6",
                    help="inner-corner count, COLSxROWS (default 9x6)")
    pc.add_argument("--square", type=float, default=0.025,
                    help="square side length in metres (default 0.025)")
    pc.add_argument("--stride", type=int, default=10,
                    help="use every Nth frame (default 10)")
    pc.add_argument("--notes", default="",
                    help="free-form notes saved with the calibration")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("fly", "calibrate"):
        # Default subcommand is fly
        argv = ["fly"] + argv
    parser = _build_parser()
    args = parser.parse_args(argv)
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    FLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    return cmd_fly(args)


if __name__ == "__main__":
    sys.exit(main())
