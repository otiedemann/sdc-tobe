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
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .aruco_detector import ArucoDetector, MarkerPose, annotate_frame
from .calibration_store import (Calibration, CalibrationStore,
                                calibrate_from_video)
from .config import (CALIB_DIR, FLIGHTS_DIR, DEFAULT_DATA_DIR,
                     MissionConfig)
from .controller import MissionController, MissionState, Phase
from .drone_api import DroneApi, MjpegStreamReader, TelemetrySnapshot
from .recorder import FlightRecorder, make_flight_dir, write_meta
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

    def set(self, tel: Optional[TelemetrySnapshot]) -> None:
        with self._lock:
            self._tel = tel

    def get(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._tel


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
    reader = MjpegStreamReader(api.video_url())
    reader.start()
    detector = ArucoDetector(calibration, cfg.marker_size_m, cfg.aruco_dict)

    pose_holder = _PoseHolder()
    tel_holder = _TelemetryHolder()
    latest_ann_frame = LatestFrame()

    # ---------- 3. Set up mission state, recorder, UI ---------------------
    # The recorder is rotated per mission (see the mission loop below)
    # so each run lands in its own ~/.marker_mission/flights/<ts>_<serial>/
    # directory with its own log + videos. The workers reference whichever
    # recorder is current via recorder_box[0]; recording_paused governs
    # WHEN they actually write -- it's set during pre-flight (INIT) and
    # post-flight (DONE/ABORT), and cleared while the drone is airborne
    # (TAKEOFF / SEARCH / ALIGN / APPROACH / HOLD / LAND). The
    # phase transitions below drive that flag so each saved flight only
    # contains the actual flight, not the operator dry-run period.
    state = MissionState()
    flight_dir = make_flight_dir(FLIGHTS_DIR, serial)
    recorder = FlightRecorder(flight_dir, fps=cfg.record_fps)
    print(f"[mission] flight artefacts: {flight_dir}")
    recorder_box = [recorder]
    flight_dir_box = [flight_dir]
    # Start paused -- INIT is not part of the flight.
    recording_paused = threading.Event()
    recording_paused.set()

    AIRBORNE_PHASES = {Phase.TAKEOFF, Phase.SEARCH, Phase.ALIGN,
                       Phase.APPROACH, Phase.HOLD, Phase.LAND}

    def on_phase_change(old_phase, new_phase, note):
        # Recording follows the airborne envelope: starts at TAKEOFF,
        # captures everything through LAND, stops at DONE / ABORT.
        if new_phase in AIRBORNE_PHASES:
            if recording_paused.is_set():
                print(f"[rec] starting recording (phase={new_phase.value})")
            recording_paused.clear()
        else:
            if not recording_paused.is_set():
                print(f"[rec] stopping recording (phase={new_phase.value})")
            recording_paused.set()

    # Build the controller before the UI so the UI's "Start mission" button
    # can call into it.
    controller = MissionController(api, cfg, state,
                                   frame_pose_provider=pose_holder.get,
                                   telemetry_provider=tel_holder.get,
                                   on_phase_change=on_phase_change)

    # The Stop button calls controller.stop() which blocks for up to ~25 s
    # waiting for the safe-shutdown to finish. We don't want the Flask
    # request thread hanging that long, so kick the stop off on a daemon
    # thread and return immediately. The control thread itself does the
    # rc_zero + land + telemetry poll synchronously.
    def request_stop_async():
        threading.Thread(
            target=controller.stop,
            args=("UI stop request",),
            name="ui-stop-trigger",
            daemon=True,
        ).start()
        return True

    cal_capture = CalibrationCapture(reader, CALIB_DIR, fps=cfg.record_fps)

    ui = UiServer(state, latest_ann_frame,
                  host=cfg.ui_host, port=cfg.ui_port,
                  history_s=cfg.ui_telemetry_history_s,
                  on_start=controller.trigger,
                  on_stop=request_stop_async,
                  flights_root=FLIGHTS_DIR,
                  drone_connected=initially_connected,
                  calibration_capture=cal_capture)
    ui.start()
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
        period = 0.25
        was_connected = initially_connected
        video_started = initially_connected
        while not stop.is_set():
            try:
                tel = api.telemetry()
                tel_holder.set(tel)
                connected_now = bool(tel.connected)
            except Exception:
                connected_now = False
            ui.drone_connected = connected_now
            if connected_now and not was_connected:
                print("[mission] drone connected (was offline at startup) "
                      "-- starting video stream")
            if connected_now and not video_started:
                try:
                    api.video_start_mjpeg()
                    video_started = True
                except Exception as e:
                    print(f"[mission] video_start_mjpeg failed: {e}")
            elif not connected_now and was_connected:
                print("[mission] drone went offline -- waiting to reconnect")
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
        while not stop.is_set():
            frame, jpg, ts = reader.latest()
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
                poses = detector.detect(frame, wanted_id=cfg.target_marker_id)
                target = poses[0] if poses else None
                pose_holder.set(target)
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
                lines = [
                    f"phase: {snap['phase']}",
                    f"target id={cfg.target_marker_id}  "
                    f"size={cfg.marker_size_m*100:.0f}cm",
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
                                     target_id=cfg.target_marker_id,
                                     extra_lines=lines)
                latest_ann_frame.set(ann)
                # Record both raw and annotated ----------------------------
                if not recording_paused.is_set():
                    rec = recorder_box[0]
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
                recorder_box[0].log_row(
                    state, pose_holder.get(), tel_holder.get())
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
            # video / csv + run the H.264 re-encode). Pause the workers
            # for the moment we swap recorders so they don't race against
            # FlightRecorder's _video_q and _csv_fp teardown.
            outcome = {
                "final_phase": last_phase,
                "abort_reason": final["abort_reason"],
                "note": final["note"],
                "serial": serial,
            }
            # Pause recording for the swap. on_phase_change keeps it
            # paused for the whole next INIT phase too, so the new
            # flight_dir only fills up once the next TAKEOFF fires.
            recording_paused.set()
            write_meta(flight_dir_box[0], cfg, calibration, outcome)
            recorder_box[0].stop(meta=None)
            print(f"[mission #{mission_idx}] artefacts saved in "
                  f"{flight_dir_box[0]}")
            if stop.is_set():
                break
            # ---- Roll a fresh flight dir + recorder for the next run.
            new_dir = make_flight_dir(FLIGHTS_DIR, serial)
            flight_dir_box[0] = new_dir
            recorder_box[0] = FlightRecorder(new_dir, fps=cfg.record_fps)
            print(f"[mission] new flight artefacts: {new_dir}")

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
    # already by the mission loop's stop() above.
    try:
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
    pf.add_argument("--ui-port", type=int, default=None,
                    help="port for the operator UI (default 8080)")
    pf.add_argument("--resolution", default="720p",
                    help="video resolution label for calibration lookup")

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
