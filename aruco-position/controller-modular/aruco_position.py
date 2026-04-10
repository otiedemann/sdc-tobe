import json
import time

import cv2
import numpy as np
from cv2 import aruco

def _load_arena_config(path, default_marker_size=0.5):
    """Load marker positions, wall types, and marker size from an arena_config.json file.

    Returns (marker_positions, marker_wall_type, marker_size_m) or raises on error.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    marker_positions = {}
    marker_wall_type = {}
    for m in cfg.get("markers", []):
        mid = int(m["id"])
        marker_positions[mid] = np.array([float(m["x"]), float(m["y"]), float(m["z"])], dtype=np.float64)
        marker_wall_type[mid] = str(m["wall"])
    marker_size_m = float(cfg.get("marker_size_m", default_marker_size))
    return marker_positions, marker_wall_type, marker_size_m


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

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        detect_profile="balanced",
        marker_size=None,
        enable_kalman_filter=True,
        arena_config_path=None,
        min_ref_weight=0.00,
        min_ref_count=1,
        pose_hold_sec=0.8,
        outlier_pos_thresh=2.5,
        target_z_pos=-1.5,
        max_state_dt=1.0,
        vel_blend=0.25,
        meas_blend_min=0.35,
        meas_blend_max=0.85,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.min_ref_weight = min_ref_weight
        self.min_ref_count = min_ref_count
        self.pose_hold_sec = pose_hold_sec
        self.outlier_pos_thresh = outlier_pos_thresh
        self.target_z_pos = target_z_pos
        self.max_state_dt = max_state_dt
        self.vel_blend = vel_blend
        self.meas_blend_min = meas_blend_min
        self.meas_blend_max = meas_blend_max

        _default_marker_size = 0.5
        # Load arena config from file when provided
        if arena_config_path is not None:
            loaded_positions, loaded_wall_types, loaded_marker_size = _load_arena_config(arena_config_path, default_marker_size=_default_marker_size)
            self.marker_positions = loaded_positions
            self.marker_wall_type = loaded_wall_types
            self.marker_size = marker_size if marker_size is not None else loaded_marker_size
            print(f"✅ Arena config loaded from {arena_config_path} ({len(self.marker_positions)} markers, marker_size={self.marker_size}m)")
        else:
            self.marker_size = marker_size if marker_size is not None else _default_marker_size
            self.marker_positions = self._initialize_marker_positions()
            self.marker_wall_type = self._initialize_marker_wall_types()

        half = self.marker_size / 2.0
        self.MARKER_3D_POINTS = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        self.aruco_params = aruco.DetectorParameters()
        self._apply_detection_profile(detect_profile)
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self._wall_rotations = self._initialize_wall_rotations()
        self.enable_kalman_filter = bool(enable_kalman_filter)
        self.kf_pos = [KalmanFilter1D() for _ in range(3)]
        self.direction_filter = ExponentialMovingAverage(alpha=0.2)
        self.target_filters = {}

        # last valid pose cache for temporary marker loss
        self.last_valid_pose = None
        self.last_valid_dir = None
        self.last_valid_ts = 0.0

        # Constant-velocity world-state used for latency-aware prediction.
        # State time is in wall-clock seconds and is advanced with capture/eval timestamps.
        self.state_pos = None
        self.state_vel = np.zeros(3, dtype=float)
        self.state_ts = None
        self.imu_vel = np.zeros(3, dtype=float)  # IMU velocity in arena frame (m/s)

    def _reset_motion_state(self):
        self.state_pos = None
        self.state_vel = np.zeros(3, dtype=float)
        self.state_ts = None
        self.imu_vel = np.zeros(3, dtype=float)  # latest IMU velocity in arena frame

    def set_imu_velocity(self, vel_arena):
        """Update the latest IMU velocity (arena frame, m/s).

        Call this before process_frame so the motion model can fuse it.
        """
        if vel_arena is not None:
            self.imu_vel = np.asarray(vel_arena, dtype=float)

    def _predict_state(self, target_ts):
        if self.state_pos is None or self.state_ts is None:
            return None
        dt = float(target_ts) - float(self.state_ts)
        if dt <= 0.0:
            return self.state_pos.copy()
        dt = min(dt, self.max_state_dt)
        return self.state_pos + self.state_vel * dt

    def _predict_imu(self, target_ts):
        """Dead-reckon using IMU velocity only (no ArUco measurement)."""
        if self.state_pos is None or self.state_ts is None:
            return None
        dt = float(target_ts) - float(self.state_ts)
        if dt <= 0.0:
            return self.state_pos.copy()
        if dt > 2.0:
            return None  # too stale
        pos = self.state_pos + self.imu_vel * dt
        # Update state so dead-reckoning accumulates
        self.state_pos = pos.copy()
        self.state_vel = self.imu_vel.copy()
        self.state_ts = target_ts
        return pos

    def _update_motion_state(self, meas_pos, meas_ts, blend_alpha):
        meas_pos = np.asarray(meas_pos, dtype=float)
        meas_ts = float(meas_ts)
        blend_alpha = float(np.clip(blend_alpha, self.meas_blend_min, self.meas_blend_max))
        if self.state_ts is not None and meas_ts < float(self.state_ts):
            meas_ts = float(self.state_ts)

        pred = self._predict_state(meas_ts)
        if pred is None:
            fused = meas_pos.copy()
        else:
            innov = meas_pos - pred
            if np.linalg.norm(innov) > (self.outlier_pos_thresh * 1.5):
                blend_alpha *= 0.5
            fused = pred + blend_alpha * innov

        if self.state_pos is not None and self.state_ts is not None:
            dt = meas_ts - float(self.state_ts)
            if 1e-3 < dt <= self.max_state_dt:
                # Blend vision-derived velocity with IMU velocity
                inst_vel = (fused - self.state_pos) / dt
                imu_weight = 0.3  # trust IMU for 30% of velocity estimate
                blended_vel = (1.0 - imu_weight) * inst_vel + imu_weight * self.imu_vel
                self.state_vel = (1.0 - self.vel_blend) * self.state_vel + self.vel_blend * blended_vel
            elif dt > self.max_state_dt:
                self.state_vel = self.imu_vel.copy()  # use IMU when vision gap is large
        self.state_pos = fused.copy()
        self.state_ts = meas_ts
        return fused

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

    @staticmethod
    def _rotz(theta_deg):
        t = np.deg2rad(theta_deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    def _initialize_wall_rotations(self):
        """
        World orientation of each wall marker frame (marker -> world).

        Chosen front base mapping (validated in live tests):
        - marker X -> world -X
        - marker Y -> world +Z
        - marker Z -> world +Y

        This gives the expected arena output behavior (front viewed from inside),
        while staying a proper rotation (right-handed): det(front_base_rot) = +1.

        Wall rotations are applied around WORLD Z relative to front:
        - back  = +180°
        - left  = -90°
        - right = +90°

        If marker print orientation changes in the future, adjust ONLY front_base_rot.
        """
        front_base_rot = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        # Safety check: must stay a pure rotation (no mirror/reflection).
        if np.linalg.det(front_base_rot) < 0.0:
            raise ValueError("front_base_rot is left-handed (det<0). Use a proper rotation matrix.")
        # Important: wall yaw is a WORLD-frame rotation relative to front.
        # Therefore pre-multiply: R_wall = Rz(yaw_world) @ front_base_rot
        return {
            'front': front_base_rot,
            'back': self._rotz(180.0) @ front_base_rot,
            'left': self._rotz(-90.0) @ front_base_rot,
            'right': self._rotz(90.0) @ front_base_rot,
        }

    def _initialize_marker_positions(self):
        pos = {0: np.array([0.0, 0.0, 0.0])}
        pos[1], pos[2] = np.array([10.0, 6.667, -1.0]), np.array([10.0, 6.667, 1.0])
        pos[3], pos[4] = np.array([10.0, 3.333, -1.0]), np.array([10.0, 3.333, 1.0])
        pos[5], pos[6] = np.array([6.0, 0.0, -1.0]), np.array([6.0, 0.0, 1.0])
        pos[7], pos[8] = np.array([2.0, 0.0, -1.0]), np.array([2.0, 0.0, 1.0])
        pos[9], pos[10] = np.array([-2.0, 0.0, -1.0]), np.array([-2.0, 0.0, 1.0])
        pos[11], pos[12] = np.array([-6.0, 0.0, -1.0]), np.array([-6.0, 0.0, 1.0])
        pos[13], pos[14] = np.array([-10.0, 3.333, -1.0]), np.array([-10.0, 3.333, 1.0])
        pos[15], pos[16] = np.array([-10.0, 6.667, -1.0]), np.array([-10.0, 6.667, 1.0])
        pos[17], pos[18] = np.array([-6.0, 10.0, -1.0]), np.array([-6.0, 10.0, 1.0])
        pos[19], pos[20] = np.array([-2.0, 10.0, -1.0]), np.array([-2.0, 10.0, 1.0])
        pos[21], pos[22] = np.array([2.0, 10.0, -1.0]), np.array([2.0, 10.0, 1.0])
        pos[23], pos[24] = np.array([6.0, 10.0, -1.0]), np.array([6.0, 10.0, 1.0])
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
            mapping[mid] = 'front'
        for mid in [17, 18, 19, 20, 21, 22, 23, 24]:
            mapping[mid] = 'back'
        for mid in [1, 2, 3, 4]:
            mapping[mid] = 'right'
        for mid in [13, 14, 15, 16]:
            mapping[mid] = 'left'

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

    def process_frame(self, frame, frame_ts=None, latency_s=0.0, now_ts=None):
        now_wall = float(now_ts) if now_ts is not None else time.time()
        frame_ts = float(frame_ts) if frame_ts is not None else now_wall
        latency_s = max(0.0, float(latency_s))
        capture_ts = frame_ts - latency_s
        eval_ts = max(capture_ts, now_wall)

        def _stale_payload(refs=None, marker_weights=None, seen_ids=None):
            if refs is None:
                refs = []
            if marker_weights is None:
                marker_weights = {}
            if seen_ids is None:
                seen_ids = []
            pred = self._predict_state(eval_ts)
            cam_out = pred if pred is not None else self.last_valid_pose
            if cam_out is None:
                return None
            return {
                "cam": np.asarray(cam_out, dtype=float).tolist(),
                "dir": self.last_valid_dir.tolist() if self.last_valid_dir is not None else None,
                "targets": {},
                "ref_markers": refs,
                "marker_weights": marker_weights,
                "seen_markers": [int(m) for m in seen_ids],
                "seen_count": len(seen_ids),
                "capture_ts": capture_ts,
                "eval_ts": eval_ts,
                "state_vel": self.state_vel.tolist(),
                "stale": True,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            # Try IMU dead-reckoning when we have state + IMU velocity
            imu_speed = float(np.linalg.norm(self.imu_vel))
            if self.state_pos is not None and imu_speed > 0.01:
                dr_pos = self._predict_imu(eval_ts)
                if dr_pos is not None:
                    self.last_valid_pose = dr_pos.copy()
                    self.last_valid_ts = now_wall
                    return {
                        "cam": dr_pos.tolist(),
                        "dir": self.last_valid_dir.tolist() if self.last_valid_dir is not None else None,
                        "targets": {},
                        "ref_markers": [],
                        "marker_weights": {},
                        "seen_markers": [],
                        "seen_count": 0,
                        "capture_ts": capture_ts,
                        "eval_ts": eval_ts,
                        "state_vel": self.state_vel.tolist(),
                        "stale": True,
                        "dead_reckoning": True,
                    }
            # short hold to avoid immediate dropouts
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= self.pose_hold_sec:
                return _stale_payload()
            for kf in self.kf_pos:
                kf.reset()
            self.direction_filter.reset()
            self._reset_motion_state()
            return None

        seen_ids = [int(mid) for mid in ids.flatten()]
        cached_poses = {}
        # Use self.MARKER_3D_POINTS so that the configured marker_size_m is
        # taken into account when estimating camera-to-marker distance via solvePnP.
        marker_points = self.MARKER_3D_POINTS.astype(np.float32)
        for i, mid_raw in enumerate(seen_ids):
            mid = int(mid_raw)
            ok, rvec, tvec = cv2.solvePnP(marker_points, corners[i].reshape(-1, 2), self.camera_matrix,
                                          self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cached_poses[mid] = (rvec, tvec.reshape(3))
        ref_indices = [i for i, mid in enumerate(seen_ids) if
                       int(mid) in self.marker_positions and int(mid) in cached_poses]
        tgt_indices = [i for i, mid in enumerate(seen_ids) if int(mid) >= 30 and int(mid) in cached_poses]
        if not ref_indices:
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= self.pose_hold_sec:
                return _stale_payload(seen_ids=seen_ids)
            return None
        cam_positions, cam_dirs, rot_mats, weights, ref_marker_ids = [], [], [], [], []
        for idx in ref_indices:
            mid = int(seen_ids[idx])
            rvec, tvec = cached_poses[mid]

            # ===================== Pose math (single marker) =====================
            # solvePnP gives marker->camera transform:
            #   x_c = R_m_c * x_m + t_m_c
            #
            # with:
            #   R_m_c : marker frame -> camera frame
            #   t_m_c : marker origin expressed in camera frame
            R_m_c, _ = cv2.Rodrigues(rvec)
            t_m_c = tvec.reshape(3)

            # Known, fixed marker transform in world:
            #   x_w = R_m_w * x_m + t_m_w
            R_m_w = self._get_marker_orientation(mid)
            t_m_w = self.marker_positions[mid]

            # Camera origin in marker frame:
            # set x_c = 0 => 0 = R_m_c * C_m + t_m_c
            # => C_m = -R_m_c^T * t_m_c
            C_m = -R_m_c.T @ t_m_c

            # Camera position in world:
            #   C_w = R_m_w * C_m + t_m_w
            c_pos_w = (R_m_w @ C_m) + t_m_w

            # Camera rotation in world:
            #   solvePnP: x_c = R_m_c x_m + t_m_c  (marker -> camera)
            #   so camera->marker is R_c_m = R_m_c^T
            #   and camera->world is R_c_w = R_m_w @ R_c_m
            R_c_w = R_m_w @ R_m_c.T

            # Camera forward vector in world:
            # d_w = R_c_w * [0,0,1]^T
            c_dir_w = R_c_w @ np.array([0.0, 0.0, 1.0])
            c_dir_norm = np.linalg.norm(c_dir_w)
            if c_dir_norm > 1e-9:
                c_dir_w = c_dir_w / c_dir_norm

            # keep existing quality weighting logic
            w = self._calculate_marker_weight(corners[idx], np.linalg.norm(t_m_c), rvec=rvec, tvec=t_m_c)
            if w < self.min_ref_weight:
                continue

            cam_positions.append(c_pos_w)
            cam_dirs.append(c_dir_w)
            rot_mats.append(R_c_w)
            weights.append(w)
            ref_marker_ids.append(mid)

        # If too few high-quality refs remain, hold recent pose instead of producing bad estimates
        if len(weights) < self.min_ref_count:
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= self.pose_hold_sec:
                return _stale_payload(
                    refs=[int(m) for m in ref_marker_ids],
                    marker_weights={str(mid): float(w) for mid, w in zip(ref_marker_ids, weights)},
                    seen_ids=seen_ids
                )
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
        keep = [i for i, p in enumerate(cam_positions) if np.linalg.norm(p - center) <= self.outlier_pos_thresh]
        if len(keep) >= self.min_ref_count:
            cam_positions = [cam_positions[i] for i in keep]
            cam_dirs = [cam_dirs[i] for i in keep]
            rot_mats = [rot_mats[i] for i in keep]
            weights = [weights[i] for i in keep]
            ref_marker_ids = [ref_marker_ids[i] for i in keep]

        # ===================== Multi-marker fusion =====================
        # Weighted mean of per-marker camera position/direction estimates:
        #   C_w = sum(w_i * C_w_i) / sum(w_i)
        #   d_w = sum(w_i * d_w_i) / sum(w_i)
        w_arr = np.array(weights) / (sum(weights) + 1e-6)
        raw_pos = sum(p * w for p, w in zip(cam_positions, w_arr))
        raw_dir = sum(d * w for d, w in zip(cam_dirs, w_arr))
        if self.enable_kalman_filter:
            f_pos_meas = np.array([self.kf_pos[j].update(raw_pos[j]) for j in range(3)])
        else:
            f_pos_meas = raw_pos.copy()

        # Apply delayed measurement update at capture time, then predict to evaluation time.
        quality = float(np.mean(weights)) * min(1.0, len(weights) / 3.0)
        blend_alpha = np.clip(0.35 + 0.5 * quality, self.meas_blend_min, self.meas_blend_max)
        self._update_motion_state(f_pos_meas, capture_ts, blend_alpha)
        f_pos = self._predict_state(eval_ts)
        if f_pos is None:
            f_pos = f_pos_meas.copy()

        raw_dir_norm = np.linalg.norm(raw_dir)
        if raw_dir_norm > 1e-9:
            raw_dir = raw_dir / raw_dir_norm

        f_dir = self.direction_filter.update(raw_dir)
        f_dir_norm = np.linalg.norm(f_dir)
        if f_dir_norm > 1e-9:
            f_dir = f_dir / f_dir_norm
        # ===================== Target projection =====================
        # For each target pose t_target (in camera frame), project via each ref estimate:
        #   T_w_i = C_w_i + R_c_w_i * t_target
        # then weighted fusion across references.
        targets = {}
        w_ref = np.array(weights, dtype=float)
        w_ref = w_ref / (np.sum(w_ref) + 1e-9)

        for idx in tgt_indices:
            tid = int(seen_ids[idx])
            tvec_target = cached_poses[tid][1]

            # Estimate target world position from each reference marker independently
            target_estimates = []
            for c_pos_w, R_c_w in zip(cam_positions, rot_mats):
                target_estimates.append(c_pos_w + (R_c_w @ tvec_target))

            # Weighted fusion of all estimates
            t_w = np.zeros(3, dtype=float)
            for est, w in zip(target_estimates, w_ref):
                t_w += est * w

            # Keep target on fixed height axis (arena Y)
            t_w[2] = self.target_z_pos

            if tid not in self.target_filters:
                self.target_filters[tid] = ExponentialMovingAverage(alpha=0.15)
            targets[str(tid)] = self.target_filters[tid].update(t_w).tolist()

        # cache valid pose
        self.last_valid_pose = f_pos.copy()
        self.last_valid_dir = f_dir.copy()
        self.last_valid_ts = now_wall

        marker_weights = {
            str(mid): float(w) for mid, w in zip(ref_marker_ids, weights)
        }

        return {
            "cam": f_pos.tolist(),
            "dir": f_dir.tolist(),
            "targets": targets,
            "ref_markers": [int(m) for m in ref_marker_ids],
            "marker_weights": marker_weights,
            "seen_markers": seen_ids,
            "seen_count": len(seen_ids),
            "capture_ts": capture_ts,
            "eval_ts": eval_ts,
            "state_vel": self.state_vel.tolist(),
        }
