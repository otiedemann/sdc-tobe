import base64
import json
import socket
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from cv2 import aruco

import pi_position_core as core
import prediction as prediction_module

from motion_estimator import MotionEstimator, MotionState
from fusion import fuse_delayed_vision_update


# ============================================================
# Optional imports for future modules
# ============================================================

try:
    import input as input_module
except Exception:
    input_module = None


# ============================================================
# Fallbacks
# ============================================================

def fallback_get_motion_input() -> Dict[str, Any]:
    """
    Erwartetes späteres Format aus input-Modul:
    {
        "timestamp": float,
        "vx_body": float,
        "vy_body": float,
        "vz_body": float,
        "yaw_rate": float,
    }
    """
    return {
        "timestamp": time.monotonic(),
        "vx_body": 0.0,
        "vy_body": 0.0,
        "vz_body": 0.0,
        "yaw_rate": 0.0,
    }


def get_motion_input() -> Dict[str, Any]:
    if input_module is not None and hasattr(input_module, "get_motion_input"):
        return input_module.get_motion_input()
    return fallback_get_motion_input()


def estimate_velocity_from_history(history: list[MotionState]) -> tuple[float, float, float]:
    """
    Einfache Geschwindigkeitsabschätzung aus den letzten zwei Zuständen.
    """
    if len(history) < 2:
        return 0.0, 0.0, 0.0

    s0 = history[-2]
    s1 = history[-1]
    dt = s1.timestamp - s0.timestamp
    if dt <= 1e-6:
        return 0.0, 0.0, 0.0

    vx = (s1.x - s0.x) / dt
    vy = (s1.y - s0.y) / dt
    vz = (s1.z - s0.z) / dt
    return vx, vy, vz


# ============================================================
# Main
# ============================================================

def main():
    import sys

    # --------------------------------------------------------
    # CLI args
    # --------------------------------------------------------
    camera_src = core.CAMERA_SOURCE
    target_ip = core.UDP_DEST_IP
    verbose_mode = False

    if "--src" in sys.argv:
        try:
            src_val = sys.argv[sys.argv.index("--src") + 1]
            camera_src = int(src_val) if src_val.isdigit() else src_val
        except Exception:
            print("❌ Invalid source provided, using default.")

    if "--target-ip" in sys.argv:
        try:
            target_ip = sys.argv[sys.argv.index("--target-ip") + 1]
        except Exception:
            print("❌ Invalid target IP provided, using default.")

    if "--verbose" in sys.argv:
        verbose_mode = True

    min_ref_weight = core.MIN_REF_WEIGHT
    min_ref_count = core.MIN_REF_COUNT
    outlier_pos_thresh = core.OUTLIER_POS_THRESH
    pose_hold_sec = core.POSE_HOLD_SEC
    target_z_pos = core.TARGET_Z_POS

    if "--min-ref-weight" in sys.argv:
        try:
            min_ref_weight = float(sys.argv[sys.argv.index("--min-ref-weight") + 1])
        except Exception:
            print("⚠️ Invalid --min-ref-weight value, using default.")

    if "--min-ref-count" in sys.argv:
        try:
            min_ref_count = int(sys.argv[sys.argv.index("--min-ref-count") + 1])
        except Exception:
            print("⚠️ Invalid --min-ref-count value, using default.")

    if "--outlier-thresh" in sys.argv:
        try:
            outlier_pos_thresh = float(sys.argv[sys.argv.index("--outlier-thresh") + 1])
        except Exception:
            print("⚠️ Invalid --outlier-thresh value, using default.")

    if "--pose-hold" in sys.argv:
        try:
            pose_hold_sec = float(sys.argv[sys.argv.index("--pose-hold") + 1])
        except Exception:
            print("⚠️ Invalid --pose-hold value, using default.")

    if "--target-z-pos" in sys.argv:
        try:
            target_z_pos = float(sys.argv[sys.argv.index("--target-z-pos") + 1])
        except Exception:
            print("⚠️ Invalid --target-z-pos value, using default.")

    preview_requested = ("--preview" in sys.argv)
    gui_enabled = preview_requested and core.has_gui() and ("--force-headless" not in sys.argv)
    gui_available = True

    detect_profile = "balanced"
    if "--detect" in sys.argv:
        try:
            val = sys.argv[sys.argv.index("--detect") + 1].strip().lower()
            if val in ("sensitive", "balanced", "strict"):
                detect_profile = val
            else:
                print("⚠️ Unknown detect profile, using 'balanced'.")
        except Exception:
            print("⚠️ Missing value for --detect, using 'balanced'.")

    # Runtime-Tuning zurück ins Core-Modul
    core.MIN_REF_WEIGHT = min_ref_weight
    core.MIN_REF_COUNT = max(1, min_ref_count)
    core.OUTLIER_POS_THRESH = max(0.1, outlier_pos_thresh)
    core.POSE_HOLD_SEC = max(0.0, pose_hold_sec)
    core.TARGET_Z_POS = target_z_pos

    print(f"🚀 Node -> {target_ip}:{core.UDP_PORT} (Debug CMD on {core.UDP_CMD_PORT})")
    print(f"📷 Camera Source: {camera_src}")
    print(f"📝 Verbose Mode: {'ON' if verbose_mode else 'OFF'}")
    print(f"🔎 Detect Profile: {detect_profile}")
    print(
        f"⚙️ min_ref_weight={core.MIN_REF_WEIGHT} "
        f"min_ref_count={core.MIN_REF_COUNT} "
        f"outlier={core.OUTLIER_POS_THRESH} "
        f"pose_hold={core.POSE_HOLD_SEC} "
        f"target_z_pos={core.TARGET_Z_POS}"
    )
    print(f"🖥️ Preview Requested: {'YES' if preview_requested else 'NO'}")
    print(f"🖥️ GUI Overlay: {'ON' if gui_enabled else 'OFF'}")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------
    cm = np.array(
        [[850.0, 0.0, 320.0],
         [0.0, 850.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    dc = np.zeros(5, dtype=float)

    if "--calib" in sys.argv:
        try:
            d = np.load(sys.argv[sys.argv.index("--calib") + 1])
            cm, dc = d["camera_matrix"], d["dist_coeffs"]
            print("✅ Calibration loaded.")
        except Exception:
            print("❌ Failed to load calibration.")

    # --------------------------------------------------------
    # Vision processor
    # --------------------------------------------------------
    vision_processor = core.HeadlessAruCoPositioning(
        cm,
        dc,
        detect_profile=detect_profile,
    )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------
    motion_rate_hz = 25.0
    vision_rate_hz = 5.0

    motion_dt = 1.0 / motion_rate_hz
    vision_dt = 1.0 / vision_rate_hz

    # --------------------------------------------------------
    # Motion estimator
    # --------------------------------------------------------
    motion_estimator = MotionEstimator(
        history_seconds=2.0,
        nominal_dt=motion_dt,
        initial_pose=(0.0, 0.0, 0.0, 0.0),
        initial_variance=(0.01, 0.01, 0.01, 0.01),
        body_y_positive_is_left=True,
        body_z_positive_is_down=True,
        yaw_positive_is_ccw=True,
        initial_timestamp=time.monotonic(),
    )

    last_motion_state: Optional[MotionState] = motion_estimator.get_current_state()
    last_vision_update: Optional[Dict[str, Any]] = None
    pending_vision_update: Optional[Dict[str, Any]] = None
    last_fused_state: Optional[Dict[str, Any]] = None
    last_prediction: Optional[Dict[str, Any]] = None
    last_fusion_result: Optional[Dict[str, Any]] = None

    # --------------------------------------------------------
    # Video source setup
    # --------------------------------------------------------
    use_tello_stream = core._is_tello_source(camera_src)
    use_anafi_stream = core._is_anafi_source(camera_src)

    tello = None
    tello_frame_reader = None

    anafi_drone = None
    anafi_stream_api = None
    anafi_frame_state = {"frame": None}
    anafi_frame_lock = threading.Lock()

    cap = None

    if use_tello_stream:
        if core.Tello is None:
            raise RuntimeError("djitellopy not installed. Install with: pip install djitellopy")

        print("🛩️ Connecting to DJI Tello…")
        tello = core.Tello()
        tello.connect()
        print(f"🔋 Tello battery: {tello.get_battery()}%")
        tello.streamon()
        tello_frame_reader = tello.get_frame_read()
        time.sleep(0.2)
        print("✅ Tello videostream active")

    elif use_anafi_stream:
        if core.olympe is None:
            raise RuntimeError("Parrot Olympe not installed. Install with: pip install parrot-olympe")

        anafi_ip = core._parse_anafi_ip(camera_src)
        print(f"🛩️ Connecting to Parrot Anafi at {anafi_ip}…")
        anafi_drone = core.olympe.Drone(anafi_ip)
        anafi_drone.connect()

        def _anafi_frame_cb(yuv_frame):
            try:
                yuv_frame.ref()
            except Exception:
                pass

            try:
                cv_cvt = cv2.COLOR_YUV2BGR_I420
                try:
                    info = yuv_frame.info()
                    yuv_fmt = None

                    if isinstance(info, dict):
                        if "yuv" in info and isinstance(info["yuv"], dict):
                            yuv_fmt = info["yuv"].get("format")
                        elif "format" in info:
                            yuv_fmt = info["format"]
                        elif "raw" in info and isinstance(info["raw"], dict):
                            yuv_fmt = info["raw"].get("format")

                    if yuv_fmt is not None:
                        for attr, flag in (
                            ("VDEF_I420", cv2.COLOR_YUV2BGR_I420),
                            ("VDEF_NV12", cv2.COLOR_YUV2BGR_NV12),
                        ):
                            fmt_const = getattr(core.olympe, attr, None)
                            if fmt_const is not None and fmt_const == yuv_fmt:
                                cv_cvt = flag
                                break
                except Exception:
                    pass

                frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv_cvt)
                with anafi_frame_lock:
                    anafi_frame_state["frame"] = frame
            finally:
                try:
                    yuv_frame.unref()
                except Exception:
                    pass

        if hasattr(anafi_drone, "streaming") and hasattr(anafi_drone.streaming, "set_callbacks"):
            anafi_drone.streaming.set_callbacks(raw_cb=_anafi_frame_cb, flush_raw_cb=core._anafi_flush_cb)
            anafi_drone.streaming.start()
            anafi_stream_api = "modern"
        elif hasattr(anafi_drone, "set_streaming_callbacks"):
            anafi_drone.set_streaming_callbacks(raw_cb=_anafi_frame_cb)
            anafi_drone.start_video_streaming()
            anafi_stream_api = "legacy"
        else:
            raise RuntimeError("No compatible Olympe streaming API found")

        time.sleep(0.2)
        print("✅ Anafi videostream active")

    else:
        cap = cv2.VideoCapture(camera_src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd.bind(("0.0.0.0", core.UDP_CMD_PORT))
    sock_cmd.setblocking(False)

    debug_mode = False
    last_send_time = 0.0
    last_img_time = 0.0
    last_heartbeat_time = 0.0

    next_motion_time = time.monotonic()
    next_vision_time = time.monotonic()

    def read_frame():
        if use_tello_stream:
            frame = tello_frame_reader.frame if tello_frame_reader is not None else None
            if frame is None:
                return False, None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        if use_anafi_stream:
            with anafi_frame_lock:
                frame = anafi_frame_state.get("frame")
                if frame is not None:
                    frame = frame.copy()
            return (frame is not None), frame

        if cap is not None:
            ret, frame = cap.read()
            return ret, frame

        return False, None

    try:
        while True:
            now = time.monotonic()

            # --------------------------------------
            # Debug command socket
            # --------------------------------------
            try:
                data, _ = sock_cmd.recvfrom(1024)
                cmd = json.loads(data.decode())
                if "debug" in cmd:
                    debug_mode = bool(cmd["debug"])
            except BlockingIOError:
                pass
            except Exception:
                pass

            # --------------------------------------
            # Motion update @ 25 Hz
            # --------------------------------------
            if now >= next_motion_time:
                # TODO: Input fuer motion aus telemetrie erzeugen #help
                motion_sample = get_motion_input()

                ts = float(motion_sample.get("timestamp", now))
                vx_body = float(motion_sample.get("vx_body", 0.0))
                vy_body = float(motion_sample.get("vy_body", 0.0))
                vz_body = float(motion_sample.get("vz_body", 0.0))
                yaw_rate = float(motion_sample.get("yaw_rate", 0.0))

                last_motion_state = motion_estimator.update_world_frame(
                    timestamp=ts,
                    vx_body=vx_body,
                    vy_body=vy_body,
                    vz_body=vz_body,
                    yaw_rate=yaw_rate,
                )

                next_motion_time += motion_dt
                if now - next_motion_time > motion_dt:
                    next_motion_time = now + motion_dt

            # --------------------------------------
            # Vision update @ 5 Hz
            # --------------------------------------
            current_frame = None

            if now >= next_vision_time:
                ret, current_frame = read_frame()

                if ret and current_frame is not None:
                    result = vision_processor.process_frame(
                        current_frame,
                        frame_ts=now,
                        now_ts=now,
                    )

                    if result is None:
                        result = {"cam": None, "dir": None, "targets": {}}

                    result["timestamp"] = now
                    result["debug"] = debug_mode

                    last_vision_update = result

                    # Nur wirklich neue verwertbare Vision-Updates in die Fusion geben
                    if result.get("cam") is not None:
                        pending_vision_update = result

                next_vision_time += vision_dt
                if now - next_vision_time > vision_dt:
                    next_vision_time = now + vision_dt

            # Falls wir für Preview/Debug ein Frame brauchen
            if current_frame is None and (gui_enabled or debug_mode):
                ret, current_frame = read_frame()
                if not ret:
                    current_frame = None

            # --------------------------------------
            # Fusion nur bei neuem Vision-Update
            # --------------------------------------
            if pending_vision_update is not None:
                last_fusion_result = fuse_delayed_vision_update(
                    motion_history=motion_estimator.get_history(),
                    vision_update=pending_vision_update,
                    use_yaw_if_available=True,
                    prefer_repropagation=True,
                )

                updated_history = last_fusion_result.get("updated_history")
                if updated_history:
                    last_motion_state = motion_estimator.apply_fused_history(updated_history)

                pending_vision_update = None

            # --------------------------------------
            # Aktuellen gefuseten Zustand aufbauen
            # --------------------------------------
            current_state = motion_estimator.get_current_state()
            history = motion_estimator.get_history()
            est_vx, est_vy, est_vz = estimate_velocity_from_history(history)

            last_fused_state = {
                "timestamp": current_state.timestamp,
                "x": current_state.x,
                "y": current_state.y,
                "z": current_state.z,
                "yaw": current_state.yaw,
                "vx": est_vx,
                "vy": est_vy,
                "vz": est_vz,
                "var_x": current_state.var_x,
                "var_y": current_state.var_y,
                "var_z": current_state.var_z,
                "var_yaw": current_state.var_yaw,
                "source": "fusion" if last_vision_update is not None else "motion_only",
                "history": [
                    {
                        "timestamp": s.timestamp,
                        "x": s.x,
                        "y": s.y,
                        "z": s.z,
                        "yaw": s.yaw,
                    }
                    for s in history
                ],
            }

            # --------------------------------------
            # Prediction auf jetzt
            # --------------------------------------
            last_prediction = predict_to_now(
                fused_state=last_fused_state,
                now_ts=now,
            )


            # --------------------------------------
            # Drone Position Controller
            # --------------------------------------

            # TODO: Zielposition aus C2 ist hier input
            # 
            # Regler erstellen der eine Fluganweisung fuer die Drohne erzeugt
            # aus last_prediction und target_position (kommt vom C2)
            # -> ggf regelt er den yaw immer nach 0, darueber habe ich aber noch nicht genug nachgedacht
            #    evtl brauchen wir die drohne auch in Blickrichtung da wir ggf objekten ausweichen wollen
            
            
            # --------------------------------------
            # Outgoing payload
            # Alte Daten beibehalten + neue ergänzen
            # --------------------------------------
            result = {"cam": None, "dir": None, "targets": {}, "debug": debug_mode}

            if last_vision_update is not None:
                result.update(last_vision_update)
                result["debug"] = debug_mode

            if last_motion_state is not None:
                result["motion"] = {
                    "x": last_motion_state.x,
                    "y": last_motion_state.y,
                    "z": last_motion_state.z,
                    "yaw": last_motion_state.yaw,
                    "var_x": last_motion_state.var_x,
                    "var_y": last_motion_state.var_y,
                    "var_z": last_motion_state.var_z,
                    "var_yaw": last_motion_state.var_yaw,
                    "timestamp": last_motion_state.timestamp,
                }

            if last_fused_state is not None:
                result["fused"] = {
                    "x": last_fused_state["x"],
                    "y": last_fused_state["y"],
                    "z": last_fused_state["z"],
                    "yaw": last_fused_state["yaw"],
                    "vx": last_fused_state["vx"],
                    "vy": last_fused_state["vy"],
                    "vz": last_fused_state["vz"],
                    "timestamp": last_fused_state["timestamp"],
                    "source": last_fused_state["source"],
                }

            if last_prediction is not None:
                result["pred"] = {
                    "x": last_prediction["x"],
                    "y": last_prediction["y"],
                    "z": last_prediction["z"],
                    "yaw": last_prediction["yaw"],
                    "vx": last_prediction["vx"],
                    "vy": last_prediction["vy"],
                    "vz": last_prediction["vz"],
                    "std_x": last_prediction["std_x"],
                    "std_y": last_prediction["std_y"],
                    "std_z": last_prediction["std_z"],
                    "std_pos": last_prediction["std_pos"],
                    "timestamp": last_prediction["timestamp"],
                    "source": last_prediction["source"],
                    "method": last_prediction["method"],
                }

            if last_fusion_result is not None:
                result["fusion"] = {
                    "reference_index": last_fusion_result.get("reference_index"),
                    "correction": last_fusion_result.get("correction"),
                    "innovation": last_fusion_result.get("innovation"),
                }

            # --------------------------------------
            # Local preview
            # --------------------------------------
            if gui_enabled and gui_available and current_frame is not None:
                preview = current_frame.copy()
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                c_dbg, i_dbg, _ = vision_processor.detector.detectMarkers(gray)
                if i_dbg is not None:
                    aruco.drawDetectedMarkers(preview, c_dbg, i_dbg)

                cv2.putText(
                    preview,
                    f"Debug: {'ON' if debug_mode else 'OFF'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if last_prediction is not None:
                    txt1 = (
                        f"PRED x={last_prediction['x']:+.2f} "
                        f"y={last_prediction['y']:+.2f} "
                        f"z={last_prediction['z']:+.2f}"
                    )
                    txt2 = (
                        f"{last_prediction['method']} | "
                        f"std={last_prediction['std_pos']:.2f} m"
                    )

                    cv2.putText(
                        preview,
                        txt1,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        preview,
                        txt2,
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                try:
                    cv2.imshow("pi_position Preview", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    gui_available = False
                    print("\n⚠️ OpenCV HighGUI not available. Disabling local preview window.")

            # --------------------------------------
            # Debug image (max 10 FPS)
            # --------------------------------------
            if debug_mode and current_frame is not None and now - last_img_time > 0.1:
                small = cv2.resize(current_frame, (320, 240))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                result["img"] = base64.b64encode(buf).decode()
                last_img_time = now

            # --------------------------------------
            # Send data
            # --------------------------------------
            should_send_tracking = (result.get("cam") is not None) and (now - last_send_time > 0.03)
            should_send_debug = debug_mode and (now - last_send_time > 0.03)
            should_send_heartbeat = (now - last_heartbeat_time > core.HEARTBEAT_INTERVAL)

            if should_send_tracking or should_send_debug or should_send_heartbeat:
                sock_send.sendto(json.dumps(result).encode(), (target_ip, core.UDP_PORT))

                if should_send_tracking or should_send_debug:
                    last_send_time = now
                if should_send_heartbeat:
                    last_heartbeat_time = now

                if verbose_mode and result.get("cam") is not None and result.get("dir") is not None:
                    cam = result["cam"]
                    dirv = result["dir"]

                    targets_txt = ""
                    if result.get("targets"):
                        parts = []
                        for tid, tpos in result["targets"].items():
                            parts.append(f"T{tid}:[{tpos[0]:+.2f},{tpos[1]:+.2f},{tpos[2]:+.2f}]")
                        targets_txt = " | " + " ".join(parts)

                    marker_txt = ""
                    if "ref_markers" in result and "marker_weights" in result:
                        marker_parts = []
                        for mid in result["ref_markers"]:
                            w = result["marker_weights"].get(str(mid), 0.0)
                            marker_parts.append(f"M{mid}:{w:.3f}")
                        marker_txt = " | REF: " + " ".join(marker_parts)

                    pred_txt = ""
                    if result.get("pred") is not None:
                        p = result["pred"]
                        pred_txt = f" | PRED:[{p['x']:+.2f},{p['y']:+.2f},{p['z']:+.2f}]"

                    print(
                        f"\rCAM: [{cam[0]:+.3f}, {cam[1]:+.3f}, {cam[2]:+.3f}] "
                        f"DIR: [{dirv[0]:+.3f}, {dirv[1]:+.3f}, {dirv[2]:+.3f}] "
                        f"Targets: {len(result.get('targets', {}))} "
                        f"Debug: {'ON' if debug_mode else 'OFF'}"
                        f"{marker_txt}{targets_txt}{pred_txt}",
                        end="",
                    )
                else:
                    tgt_count = len(result["targets"]) if result.get("targets") else 0
                    src = result.get("pred", {}).get("source", "-")
                    print(
                        f"\rTracking: {tgt_count} Targets | Debug: {'Yes' if debug_mode else 'No'} | PredSrc: {src}",
                        end="",
                    )

            time.sleep(0.001)

    finally:
        if cap is not None:
            cap.release()

        if tello is not None:
            try:
                tello.streamoff()
            except Exception:
                pass
            try:
                tello.end()
            except Exception:
                pass

        if anafi_drone is not None:
            try:
                if anafi_stream_api == "modern":
                    anafi_drone.streaming.stop()
                else:
                    anafi_drone.stop_video_streaming()
            except Exception:
                try:
                    anafi_drone.streaming.stop()
                except Exception:
                    try:
                        anafi_drone.stop_video_streaming()
                    except Exception:
                        pass
            try:
                anafi_drone.disconnect()
            except Exception:
                pass

        sock_send.close()
        sock_cmd.close()

        if gui_enabled and gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
