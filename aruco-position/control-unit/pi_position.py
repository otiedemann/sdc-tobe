import base64
import json
import os
import platform
import socket
import time

import cv2
import numpy as np
from cv2 import aruco

# --- CONFIGURATION ---
UDP_DEST_IP = "127.0.0.1"  # Default IP des Laptops (Relay)
UDP_PORT = 5005
UDP_CMD_PORT = 5006  # Port für eingehende Befehle vom Relay
MARKER_SIZE = 0.5
CAMERA_SOURCE = 0
HEARTBEAT_INTERVAL = 1.0  # Sekunden: Status senden auch ohne Marker

# Pose robustness settings
MIN_REF_WEIGHT = 0.00  # Ignore very weak refs, but keep detection usable
MIN_REF_COUNT = 1  # Allow single-marker pose as fallback
POSE_HOLD_SEC = 0.8  # Hold last valid pose briefly when refs drop out
OUTLIER_POS_THRESH = 2.5  # meters: looser outlier reject for real-world noise
TARGET_Y_OFFSET = 0.0  # add constant offset to target Y output (e.g. +3.0 for absolute arena height)

def has_gui():
    system = platform.system().lower()
    if system == "windows":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class KalmanFilter1D:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = None
        self.covariance = np.eye(2)
        self.last_time = None

    def update(self, measurement):
        now = time.time()
        if self.state is None:
            self.state = np.array([measurement, 0.0])
            self.last_time = now
            return measurement
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.5:
            self.state[0], self.state[1] = measurement, 0.0
            return measurement
        F = np.array([[1, dt], [0, 1]])
        self.state = F @ self.state
        Q = np.array([[1e-4, 0], [0, 1e-2]]) * self.process_variance
        self.covariance = F @ self.covariance @ F.T + Q
        H = np.array([[1, 0]])
        y = measurement - (H @ self.state)
        S = H @ self.covariance @ H.T + self.measurement_variance
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ H) @ self.covariance
        return self.state[0]

    def reset(self):
        self.state, self.last_time = None, None


class ExponentialMovingAverage:
    def __init__(self, alpha=0.3):
        self.alpha, self.value = alpha, None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class HeadlessAruCoPositioning:
    MARKER_3D_POINTS = np.array([
        [-0.25, 0.25, 0.0],
        [0.25, 0.25, 0.0],
        [0.25, -0.25, 0.0],
        [-0.25, -0.25, 0.0],
    ], dtype=np.float32)

    def __init__(self, camera_matrix, dist_coeffs, detect_profile="balanced"):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = MARKER_SIZE
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self._apply_detection_profile(detect_profile)
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_positions = self._initialize_marker_positions()
        self._wall_rotations = {
            'back': np.eye(3),
            'front': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'left': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            'right': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        }
        # IMPORTANT: Explicit wall mapping by marker ID (robust against axis/coordinate refactors)
        self.marker_wall_type = self._initialize_marker_wall_types()
        self.kf_pos = [KalmanFilter1D() for _ in range(3)]
        self.direction_filter = ExponentialMovingAverage(alpha=0.2)
        self.target_filters = {}

        # last valid pose cache for temporary marker loss
        self.last_valid_pose = None
        self.last_valid_dir = None
        self.last_valid_ts = 0.0

    def _apply_detection_profile(self, profile):
        p = self.aruco_params

        # Common baseline
        p.useAruco3Detection = False
        p.maxMarkerPerimeterRate = 4.0

        if profile == "sensitive":
            # Better far/small marker detection, more false positives
            p.minMarkerPerimeterRate = 0.015
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 31
            p.adaptiveThreshWinSizeStep = 4
            p.adaptiveThreshConstant = 7
            p.polygonalApproxAccuracyRate = 0.05
            p.errorCorrectionRate = 0.8
            p.minCornerDistanceRate = 0.03
            p.minDistanceToBorder = 3
        elif profile == "strict":
            # Fewer false positives, may miss small/far markers
            p.minMarkerPerimeterRate = 0.03
            p.adaptiveThreshWinSizeMin = 5
            p.adaptiveThreshWinSizeMax = 31
            p.adaptiveThreshWinSizeStep = 6
            p.adaptiveThreshConstant = 10
            p.polygonalApproxAccuracyRate = 0.04
            p.errorCorrectionRate = 0.6
            p.minCornerDistanceRate = 0.06
            p.minDistanceToBorder = 8
        else:
            # balanced (default)
            p.minMarkerPerimeterRate = 0.025
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 31
            p.adaptiveThreshWinSizeStep = 4
            p.adaptiveThreshConstant = 9
            p.polygonalApproxAccuracyRate = 0.04
            p.errorCorrectionRate = 0.6
            p.minCornerDistanceRate = 0.05
            p.minDistanceToBorder = 5

    def _initialize_marker_positions(self):
        pos = {0: np.array([0.0, 0.0, 0.0])}
        pos[1], pos[2] = np.array([-10.0, -1.0, 6.667]), np.array([-10.0, 1.0, 6.667])
        pos[3], pos[4] = np.array([-10.0, -1.0, 3.333]), np.array([-10.0, 1.0, 3.333])
        pos[5], pos[6] = np.array([-6.0, -1.0, 0.0]), np.array([-6.0, 1.0, 0.0])
        pos[7], pos[8] = np.array([-2.0, -1.0, 0.0]), np.array([-2.0, 1.0, 0.0])
        pos[9], pos[10] = np.array([2.0, -1.0, 0.0]), np.array([2.0, 1.0, 0.0])
        pos[11], pos[12] = np.array([6.0, -1.0, 0.0]), np.array([6.0, 1.0, 0.0])
        pos[13], pos[14] = np.array([10.0, -1.0, 3.333]), np.array([10.0, 1.0, 3.333])
        pos[15], pos[16] = np.array([10.0, -1.0, 6.667]), np.array([10.0, 1.0, 6.667])
        pos[17], pos[18] = np.array([6.0, -1.0, 10.0]), np.array([6.0, 1.0, 10.0])
        pos[19], pos[20] = np.array([2.0, -1.0, 10.0]), np.array([2.0, 1.0, 10.0])
        pos[21], pos[22] = np.array([-2.0, -1.0, 10.0]), np.array([-2.0, 1.0, 10.0])
        pos[23], pos[24] = np.array([-6.0, -1.0, 10.0]), np.array([-6.0, 1.0, 10.0])
        return pos

    def _initialize_marker_wall_types(self):
        """
        Wall mapping by marker ID (SDC layout specific):
        - back: 0, 5..12
        - front: 17..24
        - right: 1..4
        - left: 13..16

        NOTE: This is intentionally ID-based (not coordinate-threshold-based),
        so coordinate sign/axis changes won't break orientation assignment.
        """
        mapping = {}

        for mid in [0, 5, 6, 7, 8, 9, 10, 11, 12]:
            mapping[mid] = 'back'
        for mid in [17, 18, 19, 20, 21, 22, 23, 24]:
            mapping[mid] = 'front'
        for mid in [1, 2, 3, 4]:
            mapping[mid] = 'left'
        for mid in [13, 14, 15, 16]:
            mapping[mid] = 'right'

        return mapping

    def _get_marker_orientation(self, mid):
        return self._wall_rotations.get(self.marker_wall_type.get(mid), np.eye(3))

    def _calculate_marker_weight(self, corner, distance, rvec=None, tvec=None):
        """
        Combined quality weight:
        - image area
        - distance
        - shape squareness
        - reprojection error (if pose is available)
        """
        pts = corner.reshape(4, 2)

        # area term
        v1, v2 = pts[1] - pts[0], pts[3] - pts[0]
        area = abs(np.cross(v1, v2))
        area_w = np.clip(area / 5000.0, 0.1, 1.0)

        # distance term
        dist_w = np.exp(-distance / 10.0)

        # shape term
        side_lens = [np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4)]
        edge_v = np.std(side_lens) / (np.mean(side_lens) + 1e-6)
        shape_w = np.exp(-edge_v * 5.0)

        # reprojection term
        reproj_w = 1.0
        if rvec is not None and tvec is not None:
            proj, _ = cv2.projectPoints(self.MARKER_3D_POINTS, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape(-1, 2)
            err = np.mean(np.linalg.norm(proj - pts, axis=1))
            reproj_w = float(np.exp(-err / 5.0))
            reproj_w = float(np.clip(reproj_w, 0.05, 1.0))

        return float(area_w * dist_w * shape_w * reproj_w)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            # short hold to avoid immediate dropouts
            if self.last_valid_pose is not None and (time.time() - self.last_valid_ts) <= POSE_HOLD_SEC:
                return {
                    "cam": self.last_valid_pose.tolist(),
                    "dir": self.last_valid_dir.tolist() if self.last_valid_dir is not None else None,
                    "targets": {},
                    "ref_markers": [],
                    "marker_weights": {},
                    "stale": True
                }
            for kf in self.kf_pos: kf.reset()
            self.direction_filter.reset()
            return None
        cached_poses = {}
        marker_points = np.array([[-0.25, 0.25, 0], [0.25, 0.25, 0], [0.25, -0.25, 0], [-0.25, -0.25, 0]],
                                 dtype=np.float32)
        for i, mid_raw in enumerate(ids.flatten()):
            mid = int(mid_raw)
            ok, rvec, tvec = cv2.solvePnP(marker_points, corners[i].reshape(-1, 2), self.camera_matrix,
                                          self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok: cached_poses[mid] = (rvec, tvec.reshape(3))
        ref_indices = [i for i, mid in enumerate(ids.flatten()) if
                       int(mid) in self.marker_positions and int(mid) in cached_poses]
        tgt_indices = [i for i, mid in enumerate(ids.flatten()) if int(mid) >= 30 and int(mid) in cached_poses]
        if not ref_indices:
            if self.last_valid_pose is not None and (time.time() - self.last_valid_ts) <= POSE_HOLD_SEC:
                return {
                    "cam": self.last_valid_pose.tolist(),
                    "dir": self.last_valid_dir.tolist() if self.last_valid_dir is not None else None,
                    "targets": {},
                    "ref_markers": [],
                    "marker_weights": {},
                    "stale": True
                }
            return None
        cam_positions, cam_dirs, rot_mats, weights, ref_marker_ids = [], [], [], [], []
        for idx in ref_indices:
            mid = int(ids.flatten()[idx])
            rvec, tvec = cached_poses[mid]

            # solvePnP result:
            # X_cam = R_m_c * X_marker + t_m_c
            # where R_m_c maps marker-frame -> camera-frame
            R_m_c, _ = cv2.Rodrigues(rvec)
            t_m_c = tvec.reshape(3)

            # Fixed marker transform in world (must stay unchanged by request)
            R_m_w = self._get_marker_orientation(mid)
            t_m_w = self.marker_positions[mid]

            # Camera origin in marker frame: C_m = -R_m_c^T * t_m_c
            C_m = -R_m_c.T @ t_m_c

            # Camera position in world frame: C_w = R_m_w * C_m + t_m_w
            c_pos_w = (R_m_w @ C_m) + t_m_w

            # Camera rotation in world frame:
            # R_c_m = R_m_c^T (camera -> marker)
            # R_c_w = R_m_w * R_c_m (camera -> world)
            R_c_w = R_m_w @ R_m_c.T

            # Camera forward direction = +Z axis of camera frame mapped to world
            c_dir_w = R_c_w @ np.array([0.0, 0.0, 1.0])
            c_dir_norm = np.linalg.norm(c_dir_w)
            if c_dir_norm > 1e-9:
                c_dir_w = c_dir_w / c_dir_norm

            # keep existing quality weighting logic
            w = self._calculate_marker_weight(corners[idx], np.linalg.norm(t_m_c), rvec=rvec, tvec=t_m_c)
            if w < MIN_REF_WEIGHT:
                continue

            cam_positions.append(c_pos_w)
            cam_dirs.append(c_dir_w)
            rot_mats.append(R_c_w)
            weights.append(w)
            ref_marker_ids.append(mid)

        # If too few high-quality refs remain, hold recent pose instead of producing bad estimates
        if len(weights) < MIN_REF_COUNT:
            if self.last_valid_pose is not None and (time.time() - self.last_valid_ts) <= POSE_HOLD_SEC:
                return {
                    "cam": self.last_valid_pose.tolist(),
                    "dir": self.last_valid_dir.tolist() if self.last_valid_dir is not None else None,
                    "targets": {},
                    "ref_markers": ref_marker_ids,
                    "marker_weights": {str(mid): float(w) for mid, w in zip(ref_marker_ids, weights)},
                    "stale": True
                }
            return None

        # Keep only top-4 best reference markers when more are visible
        if len(weights) > 4:
            top_idx = np.argsort(weights)[-4:]
            cam_positions = [cam_positions[i] for i in top_idx]
            cam_dirs = [cam_dirs[i] for i in top_idx]
            rot_mats = [rot_mats[i] for i in top_idx]
            weights = [weights[i] for i in top_idx]
            ref_marker_ids = [ref_marker_ids[i] for i in top_idx]

        # Outlier rejection in position domain
        center = np.mean(np.array(cam_positions), axis=0)
        keep = [i for i, p in enumerate(cam_positions) if np.linalg.norm(p - center) <= OUTLIER_POS_THRESH]
        if len(keep) >= MIN_REF_COUNT:
            cam_positions = [cam_positions[i] for i in keep]
            cam_dirs = [cam_dirs[i] for i in keep]
            rot_mats = [rot_mats[i] for i in keep]
            weights = [weights[i] for i in keep]
            ref_marker_ids = [ref_marker_ids[i] for i in keep]

        w_arr = np.array(weights) / (sum(weights) + 1e-6)
        raw_pos = sum(p * w for p, w in zip(cam_positions, w_arr))
        raw_dir = sum(d * w for d, w in zip(cam_dirs, w_arr))
        f_pos = np.array([self.kf_pos[j].update(raw_pos[j]) for j in range(3)])

        raw_dir_norm = np.linalg.norm(raw_dir)
        if raw_dir_norm > 1e-9:
            raw_dir = raw_dir / raw_dir_norm

        f_dir = self.direction_filter.update(raw_dir)
        f_dir_norm = np.linalg.norm(f_dir)
        if f_dir_norm > 1e-9:
            f_dir = f_dir / f_dir_norm
        # Use per-reference target projection and weighted fusion (more robust)
        targets = {}
        w_ref = np.array(weights, dtype=float)
        w_ref = w_ref / (np.sum(w_ref) + 1e-9)

        for idx in tgt_indices:
            tid = int(ids.flatten()[idx])
            tvec_target = cached_poses[tid][1]

            # Estimate target world position from each reference marker independently
            target_estimates = []
            for c_pos_w, R_c_w in zip(cam_positions, rot_mats):
                target_estimates.append(c_pos_w + (R_c_w @ tvec_target))

            # Weighted fusion of all estimates
            t_w = np.zeros(3, dtype=float)
            for est, w in zip(target_estimates, w_ref):
                t_w += est * w

            # Apply configurable Y-offset (useful after origin shift to marker 0)
            t_w[1] += TARGET_Y_OFFSET

            if tid not in self.target_filters:
                self.target_filters[tid] = ExponentialMovingAverage(alpha=0.15)
            targets[str(tid)] = self.target_filters[tid].update(t_w).tolist()

        # cache valid pose
        self.last_valid_pose = f_pos.copy()
        self.last_valid_dir = f_dir.copy()
        self.last_valid_ts = time.time()

        marker_weights = {
            str(mid): float(w) for mid, w in zip(ref_marker_ids, weights)
        }

        return {
            "cam": f_pos.tolist(),
            "dir": f_dir.tolist(),
            "targets": targets,
            "ref_markers": [int(m) for m in ref_marker_ids],
            "marker_weights": marker_weights
        }


def main():
    import sys
    # Handle command line arguments
    camera_src = CAMERA_SOURCE
    target_ip = UDP_DEST_IP
    verbose_mode = False

    if '--src' in sys.argv:
        try:
            src_val = sys.argv[sys.argv.index('--src') + 1]
            camera_src = int(src_val) if src_val.isdigit() else src_val
        except:
            print("❌ Invalid source provided, using default.")

    if '--target-ip' in sys.argv:
        try:
            target_ip = sys.argv[sys.argv.index('--target-ip') + 1]
        except:
            print("❌ Invalid target IP provided, using default.")

    if '--verbose' in sys.argv:
        verbose_mode = True

    min_ref_weight = MIN_REF_WEIGHT
    min_ref_count = MIN_REF_COUNT
    outlier_pos_thresh = OUTLIER_POS_THRESH
    pose_hold_sec = POSE_HOLD_SEC
    target_y_offset = TARGET_Y_OFFSET

    if '--min-ref-weight' in sys.argv:
        try:
            min_ref_weight = float(sys.argv[sys.argv.index('--min-ref-weight') + 1])
        except:
            print("⚠️ Invalid --min-ref-weight value, using default.")

    if '--min-ref-count' in sys.argv:
        try:
            min_ref_count = int(sys.argv[sys.argv.index('--min-ref-count') + 1])
        except:
            print("⚠️ Invalid --min-ref-count value, using default.")

    if '--outlier-thresh' in sys.argv:
        try:
            outlier_pos_thresh = float(sys.argv[sys.argv.index('--outlier-thresh') + 1])
        except:
            print("⚠️ Invalid --outlier-thresh value, using default.")

    if '--pose-hold' in sys.argv:
        try:
            pose_hold_sec = float(sys.argv[sys.argv.index('--pose-hold') + 1])
        except:
            print("⚠️ Invalid --pose-hold value, using default.")

    if '--target-y-offset' in sys.argv:
        try:
            target_y_offset = float(sys.argv[sys.argv.index('--target-y-offset') + 1])
        except:
            print("⚠️ Invalid --target-y-offset value, using default.")

    # Apply runtime tuning globally (used inside process_frame)
    globals()["MIN_REF_WEIGHT"] = min_ref_weight
    globals()["MIN_REF_COUNT"] = max(1, min_ref_count)
    globals()["OUTLIER_POS_THRESH"] = max(0.1, outlier_pos_thresh)
    globals()["POSE_HOLD_SEC"] = max(0.0, pose_hold_sec)
    globals()["TARGET_Y_OFFSET"] = target_y_offset

    preview_requested = ('--preview' in sys.argv)
    gui_enabled = preview_requested and has_gui() and ('--force-headless' not in sys.argv)
    gui_available = True

    detect_profile = "balanced"
    if '--detect' in sys.argv:
        try:
            val = sys.argv[sys.argv.index('--detect') + 1].strip().lower()
            if val in ("sensitive", "balanced", "strict"):
                detect_profile = val
            else:
                print("⚠️ Unknown detect profile, using 'balanced'.")
        except:
            print("⚠️ Missing value for --detect, using 'balanced'.")

    print(f"🚀 Headless Node -> {target_ip}:{UDP_PORT} (Debug CMD on {UDP_CMD_PORT})")
    print(f"📷 Camera Source: {camera_src}")
    print(f"📝 Verbose Mode: {'ON' if verbose_mode else 'OFF'}")
    print(f"🔎 Detect Profile: {detect_profile}")
    print(
        f"⚙️ min_ref_weight={MIN_REF_WEIGHT} min_ref_count={MIN_REF_COUNT} outlier={OUTLIER_POS_THRESH} pose_hold={POSE_HOLD_SEC} target_y_offset={TARGET_Y_OFFSET}")
    print(f"🖥️ Preview Requested: {'YES' if preview_requested else 'NO'}")
    print(f"🖥️ GUI Overlay: {'ON' if gui_enabled else 'OFF'}")

    cm = np.array([[850.0, 0.0, 320.0], [0.0, 850.0, 240.0], [0.0, 0.0, 1.0]], dtype=float)
    dc = np.zeros(5, dtype=float)

    if '--calib' in sys.argv:
        try:
            d = np.load(sys.argv[sys.argv.index('--calib') + 1])
            cm, dc = d['camera_matrix'], d['dist_coeffs']
            print("✅ Calibration loaded.")
        except:
            print("❌ Failed to load calibration.")

    ap = HeadlessAruCoPositioning(cm, dc, detect_profile=detect_profile)
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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Check for commands (Debug On/Off)
            try:
                data, _ = sock_cmd.recvfrom(1024)
                cmd = json.loads(data.decode())
                if "debug" in cmd:
                    debug_mode = bool(cmd["debug"])
            except BlockingIOError:
                pass

            result = ap.process_frame(frame)
            now = time.time()

            # Optional local preview window if GUI is available
            if gui_enabled and gui_available:
                preview = frame.copy()
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                c_dbg, i_dbg, _ = ap.detector.detectMarkers(gray)
                if i_dbg is not None:
                    aruco.drawDetectedMarkers(preview, c_dbg, i_dbg)
                cv2.putText(preview, f"Debug: {'ON' if debug_mode else 'OFF'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                try:
                    cv2.imshow("pi_position Preview", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except cv2.error:
                    gui_available = False
                    print("\n⚠️ OpenCV HighGUI nicht verfügbar (headless build). Deaktiviere lokales Preview-Fenster.")

            # Ensure we have a result dict even if no markers are found
            if result is None:
                result = {"cam": None, "dir": None, "targets": {}}

            # Always include the current debug state in the heartbeat/payload
            result["debug"] = debug_mode

            # Debug Video (max 10 FPS) - now independent of detection
            if debug_mode and now - last_img_time > 0.1:
                small = cv2.resize(frame, (320, 240))
                _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                result["img"] = base64.b64encode(buf).decode()
                last_img_time = now

            # Send data if markers found OR debug mode active OR periodic heartbeat
            should_send_tracking = (result["cam"] is not None) and (now - last_send_time > 0.03)
            should_send_debug = debug_mode and (now - last_send_time > 0.03)
            should_send_heartbeat = (now - last_heartbeat_time > HEARTBEAT_INTERVAL)

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
                        end=""
                    )
                else:
                    tgt_count = len(result['targets']) if result['targets'] else 0
                    print(f"\rTracking: {tgt_count} Targets | Debug: {'Yes' if debug_mode else 'No'}", end="")
    finally:
        cap.release()
        sock_send.close()
        if gui_enabled and gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__": main()
