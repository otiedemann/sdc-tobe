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
    api = DroneApi(cfg.api_base_url, cfg.request_timeout_s)
    print(f"[mission] contacting API server at {cfg.api_base_url} ...")
    deadline = time.monotonic() + 15.0
    serial = "unknown"
    while time.monotonic() < deadline:
        try:
            tel = api.telemetry()
            if tel.connected:
                serial = tel.serial_number or "unknown"
                break
        except Exception as e:
            print(f"[mission]   waiting: {e}")
        time.sleep(1.0)
    else:
        print("[mission] API server unreachable or drone not connected -- abort.")
        return 1
    print(f"[mission] connected. drone serial = {serial}")

    store = CalibrationStore(CALIB_DIR)
    calibration = store.load(serial, args.resolution, allow_default=True)
    print(f"[mission] calibration: {calibration.short_summary()}")
    if calibration.is_default:
        print("[mission] WARNING: no per-drone calibration found -- using "
              "Anafi default intrinsics. Pose accuracy will be reduced.")

    # ---------- 2. Open video stream, start detector ----------------------
    print("[mission] starting video stream ...")
    api.video_start_mjpeg()
    reader = MjpegStreamReader(api.video_url())
    reader.start()
    detector = ArucoDetector(calibration, cfg.marker_size_m, cfg.aruco_dict)

    pose_holder = _PoseHolder()
    tel_holder = _TelemetryHolder()
    latest_ann_frame = LatestFrame()

    # ---------- 3. Set up mission state, recorder, UI ---------------------
    state = MissionState()
    flight_dir = make_flight_dir(FLIGHTS_DIR, serial)
    recorder = FlightRecorder(flight_dir, fps=cfg.record_fps)
    print(f"[mission] flight artefacts: {flight_dir}")

    # Build the controller before the UI so the UI's "Start mission" button
    # can call into it.
    controller = MissionController(api, cfg, state,
                                   frame_pose_provider=pose_holder.get,
                                   telemetry_provider=tel_holder.get)

    ui = UiServer(state, latest_ann_frame,
                  host=cfg.ui_host, port=cfg.ui_port,
                  history_s=cfg.ui_telemetry_history_s,
                  on_start=controller.trigger)
    ui.start()
    print(f"[mission] UI running at {ui.url()} (camera) and {ui.url()}/charts")

    # ---------- 4. Background workers ------------------------------------
    stop = threading.Event()

    def telemetry_worker():
        # Poll telemetry independently of the control loop so the UI stays
        # responsive even when the controller is busy.
        period = 0.25
        while not stop.is_set():
            try:
                tel = api.telemetry()
                tel_holder.set(tel)
            except Exception:
                pass
            time.sleep(period)

    def vision_worker():
        # Pull the latest frame, run detection, push to recorder + UI.
        last_seen_ts = 0.0
        while not stop.is_set():
            frame, jpg, ts = reader.latest()
            if frame is None or ts == last_seen_ts:
                time.sleep(0.01)
                continue
            last_seen_ts = ts
            try:
                poses = detector.detect(frame, wanted_id=cfg.target_marker_id)
                target = poses[0] if poses else None
                pose_holder.set(target)
                # Status overlay -------------------------------------------
                snap = state.snapshot()
                lines = [
                    f"phase: {snap['phase']}",
                    f"target id={cfg.target_marker_id}  "
                    f"size={cfg.marker_size_m*100:.0f}cm",
                ]
                if target:
                    lines.append(
                        f"d={target.distance_m:.2f}m  yaw={target.yaw_deg:+.1f}deg  "
                        f"hdg={target.relative_heading_deg:+.1f}deg")
                else:
                    lines.append("marker: NOT VISIBLE")
                ann = annotate_frame(frame, poses,
                                     target_id=cfg.target_marker_id,
                                     extra_lines=lines)
                latest_ann_frame.set(ann)
                # Record both raw and annotated ----------------------------
                recorder.push_raw_frame(frame)
                recorder.push_annotated_frame(ann)
            except Exception as e:
                print(f"[vision] error: {e}")

    def log_worker():
        # Periodic CSV row; this gives us a sample even when no frames are
        # arriving (useful for diagnosing video drops).
        period = 1.0 / max(2.0, cfg.control_rate_hz / 2)
        while not stop.is_set():
            recorder.log_row(state, pose_holder.get(), tel_holder.get())
            time.sleep(period)

    threading.Thread(target=telemetry_worker, daemon=True,
                     name="telemetry").start()
    threading.Thread(target=vision_worker, daemon=True,
                     name="vision").start()
    threading.Thread(target=log_worker, daemon=True,
                     name="log").start()

    # ---------- 5. Mission ------------------------------------------------
    # Graceful shutdown on Ctrl-C / SIGTERM
    def handle_signal(signum, _frame):
        print(f"\n[mission] signal {signum} -- requesting safe stop")
        controller.stop("operator interrupt")
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    # Arm the controller. It parks in INIT until the operator presses
    # "Start mission" in the web UI (which calls controller.trigger via
    # POST /api/start).
    controller.start()
    print(f"[mission] armed -- open {ui.url()} and press 'Start mission' "
          f"to take off. Ctrl-C to abort.")

    # Wait for completion
    try:
        while controller.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        controller.stop("KeyboardInterrupt")

    final_phase = state.snapshot()
    print(f"[mission] final phase: {final_phase['phase']} "
          f"(reason: {final_phase['note'] or final_phase['abort_reason'] or '—'})")

    # ---------- 6. Cleanup -----------------------------------------------
    stop.set()
    time.sleep(0.5)
    reader.stop()
    try:
        api.video_stop()
    except Exception:
        pass
    outcome = {
        "final_phase": final_phase["phase"],
        "abort_reason": final_phase["abort_reason"],
        "note": final_phase["note"],
        "serial": serial,
    }
    write_meta(flight_dir, cfg, calibration, outcome)
    recorder.stop(meta=None)  # meta already written separately above
    print(f"[mission] artefacts saved in {flight_dir}")
    return 0 if final_phase["phase"] in ("done",) else 2


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
