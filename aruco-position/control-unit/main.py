import base64
import json
import socket
import threading
import time
import sys

import cv2
import numpy as np
from cv2 import aruco

from pi_position_core import (
    UDP_DEST_IP,
    UDP_PORT,
    UDP_CMD_PORT,
    CAMERA_SOURCE,
    HEARTBEAT_INTERVAL,
    MIN_REF_WEIGHT,
    MIN_REF_COUNT,
    OUTLIER_POS_THRESH,
    POSE_HOLD_SEC,
    TARGET_Z_POS,
    Tello,
    olympe,
    has_gui,
    _is_tello_source,
    _is_anafi_source,
    _parse_anafi_ip,
    _anafi_flush_cb,
    HeadlessAruCoPositioning,
)


def main():
    camera_src = CAMERA_SOURCE
    target_ip = UDP_DEST_IP
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

    min_ref_weight = MIN_REF_WEIGHT
    min_ref_count = MIN_REF_COUNT
    outlier_pos_thresh = OUTLIER_POS_THRESH
    pose_hold_sec = POSE_HOLD_SEC
    target_z_pos = TARGET_Z_POS

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

    # Runtime tuning global im Core-Modul setzen
    import pi_position_core as core
    core.MIN_REF_WEIGHT = min_ref_weight
    core.MIN_REF_COUNT = max(1, min_ref_count)
    core.OUTLIER_POS_THRESH = max(0.1, outlier_pos_thresh)
    core.POSE_HOLD_SEC = max(0.0, pose_hold_sec)
    core.TARGET_Z_POS = target_z_pos

    preview_requested = "--preview" in sys.argv
    gui_enabled = preview_requested and has_gui() and ("--force-headless" not in sys.argv)
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

    print(f"🚀 Headless Node -> {target_ip}:{UDP_PORT} (Debug CMD on {UDP_CMD_PORT})")
    print(f"📷 Camera Source: {camera_src}")
    print(f"📝 Verbose Mode: {'ON' if verbose_mode else 'OFF'}")

    use_tello_stream = _is_tello_source(camera_src)
    use_anafi_stream = _is_anafi_source(camera_src)

    tello = None
    tello_frame_reader = None
    anafi_drone = None
    anafi_stream_api = None
    anafi_frame_state = {"frame": None}
    anafi_frame_lock = threading.Lock()

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

    cm = np.array(
        [[850.0, 0.0, 320.0], [0.0, 850.0, 240.0], [0.0, 0.0, 1.0]],
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

    ap = HeadlessAruCoPositioning(cm, dc, detect_profile=detect_profile)

    cap = None
    if use_tello_stream:
        if Tello is None:
            raise RuntimeError(
                "djitellopy nicht installiert. Install with: pip install djitellopy"
            )

        print("🛩️ Connecting to DJI Tello…")
        tello = Tello()
        tello.connect()
        print(f"🔋 Tello battery: {tello.get_battery()}%")
        tello.streamon()
        tello_frame_reader = tello.get_frame_read()
        time.sleep(0.2)
        print("✅ Tello videostream active")

    elif use_anafi_stream:
        if olympe is None:
            raise RuntimeError(
                "Parrot Olympe nicht installiert. Install with: pip install parrot-olympe"
            )

        anafi_ip = _parse_anafi_ip(camera_src)
        print(f"🛩️ Connecting to Parrot Anafi at {anafi_ip}…")
        anafi_drone = olympe.Drone(anafi_ip)
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
                            fmt_const = getattr(olympe, attr, None)
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
            anafi_drone.streaming.set_callbacks(
                raw_cb=_anafi_frame_cb,
                flush_raw_cb=_anafi_flush_cb,
            )
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

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd.bind(("0.0.0.0", UDP_CMD_PORT))
    sock_cmd.setblocking(False)

    debug_mode = False
    last_send_time = 0
    last_img_time = 0
    last_heartbeat_time = 0

    try:
        while True:
            ret, frame = False, None

            if use_tello_stream:
                frame = tello_frame_reader.frame if tello_frame_reader is not None else None
                ret = frame is not None
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            elif use_anafi_stream:
                with anafi_frame_lock:
                    frame = anafi_frame_state.get("frame")
                    if frame is not None:
                        frame = frame.copy()
                ret = frame is not None

            else:
                if cap is not None:
                    ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            try:
                data, _ = sock_cmd.recvfrom(1024)
                cmd = json.loads(data.decode())
                if "debug" in cmd:
                    debug_mode = bool(cmd["debug"])
            except BlockingIOError:
                pass

            result = ap.process_frame(frame)
            now = time.time()

            if gui_enabled and gui_available:
                preview = frame.copy()
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                c_dbg, i_dbg, _ = ap.detector.detectMarkers(gray)
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

                try:
                    cv2.imshow("pi_position Preview", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    gui_available = False
                    print("\n⚠️ OpenCV HighGUI nicht verfügbar. Preview wird deaktiviert.")

            if result is None:
                result = {"cam": None, "dir": None, "targets": {}}

            result["debug"] = debug_mode

            if debug_mode and now - last_img_time > 0.1:
                small = cv2.resize(frame, (320, 240))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                result["img"] = base64.b64encode(buf).decode()
                last_img_time = now

            should_send_tracking = (result["cam"] is not None) and (now - last_send_time > 0.03)
            should_send_debug = debug_mode and (now - last_send_time > 0.03)
            should_send_heartbeat = now - last_heartbeat_time > HEARTBEAT_INTERVAL

            if should_send_tracking or should_send_debug or should_send_heartbeat:
                sock_send.sendto(json.dumps(result).encode(), (target_ip, UDP_PORT))

                if should_send_tracking or should_send_debug:
                    last_send_time = now
                if should_send_heartbeat:
                    last_heartbeat_time = now

                if verbose_mode and result["cam"] is not None and result["dir"] is not None:
                    cam = result["cam"]
                    dirv = result["dir"]

                    targets_txt = ""
                    if result["targets"]:
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

                    print(
                        f"\rCAM: [{cam[0]:+.3f}, {cam[1]:+.3f}, {cam[2]:+.3f}] "
                        f"DIR: [{dirv[0]:+.3f}, {dirv[1]:+.3f}, {dirv[2]:+.3f}] "
                        f"Targets: {len(result['targets'])} Debug: {'ON' if debug_mode else 'OFF'}"
                        f"{marker_txt}{targets_txt}",
                        end="",
                    )
                else:
                    tgt_count = len(result["targets"]) if result["targets"] else 0
                    print(f"\rTracking: {tgt_count} Targets | Debug: {'Yes' if debug_mode else 'No'}", end="")

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
