import base64
import json
import os
import platform
import socket
import threading
import time

import cv2
import numpy as np
from cv2 import aruco

try:
    from djitellopy import Tello
except Exception:
    Tello = None

try:
    import olympe
except Exception:
    olympe = None

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
TARGET_Z_POS = -1.5  # fixed target height position (internal Z axis)

# Motion model / delayed-measurement tuning
MAX_STATE_DT = 1.0
VEL_BLEND = 0.25
MEAS_BLEND_MIN = 0.35
MEAS_BLEND_MAX = 0.85

# Re-acquisition consensus (item 3 of position-tracker stability fix).
# After a marker dropout longer than POSE_HOLD_SEC the previous state is no
# longer trustworthy as a prior — but a single bad solvePnP fix shouldn't
# teleport the published position either. The first fresh fix after such a
# gap is cached, and only accepted once a SECOND fix lands within
# RECOVERY_DT_S of the first AND within RECOVERY_DIST_M of it. The
# confirmed measurement is then partially blended toward the predicted
# state via RECOVERY_ALPHA_SCALE, so even after confirmation the published
# position doesn't snap. Eliminates the "0→1 ref re-acquisition teleport"
# class — see flightlog analysis 2026-04-26 (43 m p99 jumps in baseline).
RECOVERY_GAP_S = 0.5
RECOVERY_DT_S = 0.5
RECOVERY_DIST_M = 1.5
RECOVERY_ALPHA_SCALE = 0.3


def has_gui():
    system = platform.system().lower()
    if system == "windows":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _is_tello_source(camera_source):
    return str(camera_source).strip().lower() in {"tello", "dji", "dji-tello", "tello-udp"}


def _is_anafi_source(camera_source):
    source = str(camera_source).strip().lower()
    return source in {"anafi", "parrot", "parrot-anafi"} or source.startswith("anafi:") or source.startswith("anafi://")


def _parse_anafi_ip(camera_source):
    source = str(camera_source).strip()
    source_lower = source.lower()
    if source_lower in {"anafi", "parrot", "parrot-anafi"}:
        return os.getenv("ANAFI_IP") or os.getenv("DRONE_IP") or "192.168.42.1"
    if source_lower.startswith("anafi://"):
        return source.split("://", 1)[1].strip() or "192.168.42.1"
    if source_lower.startswith("anafi:"):
        return source.split(":", 1)[1].strip() or "192.168.42.1"
    return "192.168.42.1"


def _anafi_flush_cb(stream):
    try:
        if hasattr(stream, "empty"):
            while not stream.empty():
                try:
                    stream.get(timeout=0.005).unref()
                except Exception:
                    break
        elif hasattr(stream, "get"):
            while True:
                try:
                    stream.get(timeout=0.005).unref()
                except Exception:
                    break
    except Exception:
        pass
    return True


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
        imu_weight=0.3,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size if marker_size is not None else MARKER_SIZE
        # SDC26 target boxes wear 19 cm ArUco stickers, while the arena
        # wall markers are 50 cm. solvePnP needs the RIGHT physical
        # size for each to return a correct distance — if we measure a
        # 19 cm target with 50 cm corners we get distances 50/19 ≈ 2.63×
        # too large and the target lands ~30 m away from reality.
        # Kept as a per-instance attribute so the FC can live-patch it.
        self.target_marker_size = 0.19
        self._build_marker_point_sets()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        self.aruco_params = aruco.DetectorParameters()
        self._apply_detection_profile(detect_profile)
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_positions = self._initialize_marker_positions()
        self._wall_rotations = self._initialize_wall_rotations()
        # IMPORTANT: Explicit wall mapping by marker ID (robust against axis/coordinate refactors)
        self.marker_wall_type = self._initialize_marker_wall_types()
        self.enable_kalman_filter = bool(enable_kalman_filter)
        self.kf_pos = [KalmanFilter1D() for _ in range(3)]
        self.direction_filter = ExponentialMovingAverage(alpha=0.2)
        self.target_filters = {}
        # Live-tunable fusion knobs
        self.top_k_markers = 4           # keep best N of visible refs; 0 = unlimited
        self.outlier_reject_m = OUTLIER_POS_THRESH  # drop per-marker estimates > this from centroid
        # ── Distance correction factor ──────────────────────────────────
        # Multiplier applied to the solvePnP-produced marker→camera
        # translation (tvec). Compensates for systematic scale error in
        # the fused pose, typically caused by a mismatch between the real
        # marker physical size and the configured marker_size_m, or by an
        # uncalibrated-camera focal length. Use when the UI shows e.g.
        # 7 m but the real distance to the marker is 9 m → set to 9/7
        # ≈ 1.286. Default 1.0 = no correction.
        self.distance_scale = 1.0
        # ── Pose-jump gate ─────────────────────────────────────────────
        # Reject a fused pose estimate if it disagrees with the Kalman-
        # predicted state by more than this many metres. This kills the
        # catastrophic >10 m single-marker glitches we saw in the field
        # logs (single-marker solvePnP is occasionally wildly wrong, e.g.
        # mirror-pose ambiguity at certain viewing angles). Set 0 to
        # disable. Applies only when a prior state exists AND marker
        # coverage has been continuous — after a reset or long stale
        # gap, the re-acquisition consensus check below takes over.
        # Default 3.0 m — flightlog analysis (2026-04-26, 107 flights)
        # showed gate=3 m would have rejected 147 catastrophic single-
        # frame outliers (0.49 % of fresh fixes) including every |Z|>10 m
        # glitch.
        self.max_pose_jump_m = 3.0

        # last valid pose cache for temporary marker loss
        self.last_valid_pose = None
        self.last_valid_dir = None
        self.last_valid_ts = 0.0

        # Constant-velocity world-state used for latency-aware prediction.
        # State time is in wall-clock seconds and is advanced with capture/eval timestamps.
        self.state_pos = None
        self.state_vel = np.zeros(3, dtype=float)
        self.state_ts = None

        # Re-acquisition consensus state (paired with RECOVERY_* constants).
        # When tracking has been lost (no fresh fix within POSE_HOLD_SEC),
        # the next fresh fix becomes a candidate; we publish a coast until
        # a second fresh fix confirms it.
        self._recovery_candidate_pos = None
        self._recovery_candidate_ts = 0.0

        # ── IMU fusion ───────────────────────────────────────────────────
        # IMU velocity in ARENA frame (m/s), set externally via set_imu_velocity()
        # before process_frame. imu_weight controls how much we trust it vs vision:
        #   0.0 → pure ArUco (vision-only velocity)
        #   1.0 → pure IMU (marker fix only updates position, never velocity)
        #   0.3 → default: 70% vision, 30% IMU
        self.imu_vel = np.zeros(3, dtype=float)
        self.imu_vel_ts = 0.0
        self.imu_weight = float(np.clip(imu_weight, 0.0, 1.0))

    def _build_marker_point_sets(self):
        """(Re)build the two 3D corner arrays used for solvePnP — one at
        the reference-marker size (arena walls, 0.5 m default) and one
        at the target-marker size (SDC26 target boxes, 0.19 m default).
        Call this after changing either self.marker_size or
        self.target_marker_size at runtime."""
        half_ref = self.marker_size / 2.0
        self.MARKER_3D_POINTS = np.array([
            [-half_ref,  half_ref, 0.0],
            [ half_ref,  half_ref, 0.0],
            [ half_ref, -half_ref, 0.0],
            [-half_ref, -half_ref, 0.0],
        ], dtype=np.float32)
        half_tgt = self.target_marker_size / 2.0
        self.MARKER_3D_POINTS_TARGET = np.array([
            [-half_tgt,  half_tgt, 0.0],
            [ half_tgt,  half_tgt, 0.0],
            [ half_tgt, -half_tgt, 0.0],
            [-half_tgt, -half_tgt, 0.0],
        ], dtype=np.float32)

    def _marker_points_for_id(self, mid: int) -> np.ndarray:
        """Pick the correct 3D corner set for a given marker ID.

        SDC26 convention: IDs ≥ 30 are target boxes (19 cm stickers);
        anything below is an arena reference (50 cm wall marker). The
        tgt_indices filter later restricts which target IDs are
        actually *used* (strict {31-36, 41-46}); this helper just
        makes sure solvePnP uses the right corners so the distance is
        right whichever class the marker turns out to belong to.
        """
        if int(mid) >= 30:
            return self.MARKER_3D_POINTS_TARGET
        return self.MARKER_3D_POINTS

    def set_imu_velocity(self, vel_arena, ts=None):
        """
        Set the latest IMU-derived velocity in ARENA frame (m/s).
        Call this before each process_frame() so the motion model can fuse it.

        Args:
            vel_arena: 3-element vector [vx, vy, vz] in m/s, ARENA frame
            ts: capture timestamp (wall-clock seconds). Defaults to time.time().
        """
        if vel_arena is None:
            return
        self.imu_vel = np.asarray(vel_arena, dtype=float)
        self.imu_vel_ts = float(ts) if ts is not None else time.time()

    def set_imu_weight(self, weight):
        """Live-tune the IMU-vs-ArUco velocity blend (0=pure vision, 1=pure IMU)."""
        self.imu_weight = float(np.clip(weight, 0.0, 1.0))

    def _predict_imu(self, target_ts):
        """
        Dead-reckon position from last known state using IMU velocity only.
        Used when ArUco markers are briefly lost. Returns None if state is stale
        (>2s since last vision update) — don't blindly integrate forever.
        """
        if self.state_pos is None or self.state_ts is None:
            return None
        dt = float(target_ts) - float(self.state_ts)
        if dt <= 0.0 or dt > 2.0:
            return None
        pos = self.state_pos + self.imu_vel * dt
        # Update state so subsequent dead-reckoning accumulates from this point
        self.state_pos = pos.copy()
        self.state_vel = self.imu_vel.copy()
        self.state_ts = target_ts
        return pos

    def _reset_motion_state(self):
        """Clear in-flight prediction state. Called on marker dropouts
        inside `process_frame`. Does NOT clear `last_valid_pose` (the
        next frame may still want to publish it as a stale hold)."""
        self.state_pos = None
        self.state_vel = np.zeros(3, dtype=float)
        self.state_ts = None
        self.imu_vel = np.zeros(3, dtype=float)
        self.imu_vel_ts = 0.0
        self._recovery_candidate_pos = None
        self._recovery_candidate_ts = 0.0

    def reset_tracker_state(self):
        """Full reset of every piece of position-tracker state. Call this
        from the FC's takeoff path so a previous flight's poisoned state
        cannot leak into the new flight.

        Why this matters: the FC positioner is a long-lived process. If
        the previous flight ended with `state_pos` pointing at, say,
        [643, -315, -1866] (a real example from flight 2026-04-24 09-44-43,
        caused by a bad single-marker fix), that state persists across
        takeoffs. The EMA filter then takes ~5 s to drag the published
        position back to reality, during which every consumer (mission
        logic, UI, autonomous pursuit) sees impossible coordinates.
        Resetting on takeoff makes each flight start from a clean slate.
        """
        self._reset_motion_state()
        self.last_valid_pose = None
        self.last_valid_dir = None
        self.last_valid_ts = 0.0
        for kf in self.kf_pos:
            kf.reset()
        self.direction_filter.reset()
        self.target_filters = {}

    def _predict_state(self, target_ts):
        if self.state_pos is None or self.state_ts is None:
            return None
        dt = float(target_ts) - float(self.state_ts)
        if dt <= 0.0:
            return self.state_pos.copy()
        dt = min(dt, MAX_STATE_DT)
        return self.state_pos + self.state_vel * dt

    def _update_motion_state(self, meas_pos, meas_ts, blend_alpha):
        meas_pos = np.asarray(meas_pos, dtype=float)
        meas_ts = float(meas_ts)
        blend_alpha = float(np.clip(blend_alpha, MEAS_BLEND_MIN, MEAS_BLEND_MAX))
        if self.state_ts is not None and meas_ts < float(self.state_ts):
            meas_ts = float(self.state_ts)

        pred = self._predict_state(meas_ts)
        if pred is None:
            fused = meas_pos.copy()
        else:
            innov = meas_pos - pred
            if np.linalg.norm(innov) > (OUTLIER_POS_THRESH * 1.5):
                blend_alpha *= 0.5
            fused = pred + blend_alpha * innov

        if self.state_pos is not None and self.state_ts is not None:
            dt = meas_ts - float(self.state_ts)
            if 1e-3 < dt <= MAX_STATE_DT:
                inst_vel = (fused - self.state_pos) / dt
                # Blend vision-derived velocity with IMU velocity for smoother estimate
                # imu_weight=0 → pure vision, 1 → pure IMU
                blended_vel = (1.0 - self.imu_weight) * inst_vel + self.imu_weight * self.imu_vel
                self.state_vel = (1.0 - VEL_BLEND) * self.state_vel + VEL_BLEND * blended_vel
            elif dt > MAX_STATE_DT:
                # Vision gap too large — fall back to IMU velocity instead of zeroing
                self.state_vel = self.imu_vel.copy()
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
            p.minMarkerPerimeterRate = 0.01
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
            # balanced (default) — matched to the C2-side VideoMarkerTracker
            # params (tools/aruco_seek.py) so that if the operator-facing
            # observer detects a marker, the FC-side position tracker
            # sees it too. The old 0.025 perimeter threshold was
            # silently losing markers at 3–5 m during flights, leaving
            # the tracker stuck on last_valid_pose with seen_markers=[].
            p.minMarkerPerimeterRate = 0.01
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 23
            p.adaptiveThreshWinSizeStep = 5
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
                "marker_pixel_sizes": {},
                "marker_centers": {},
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
            # short hold to avoid immediate dropouts
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= POSE_HOLD_SEC:
                return _stale_payload()
            # Extended IMU dead-reckoning: if markers lost but IMU velocity is
            # fresh and we're moving, keep outputting a live position by
            # integrating IMU velocity forward from the last known state.
            imu_fresh = (now_wall - self.imu_vel_ts) < 0.5
            imu_moving = float(np.linalg.norm(self.imu_vel)) > 0.01
            if imu_fresh and imu_moving:
                dr_pos = self._predict_imu(eval_ts)
                if dr_pos is not None:
                    return {
                        "cam": dr_pos.tolist(),
                        "dir": (self.last_valid_dir.tolist()
                                if self.last_valid_dir is not None else [0.0, 1.0, 0.0]),
                        "targets": {},
                        "ref_markers": [],
                        "marker_weights": {},
                        "marker_pixel_sizes": {},
                        "marker_centers": {},
                        "seen_markers": [],
                        "seen_count": 0,
                        "capture_ts": capture_ts,
                        "eval_ts": eval_ts,
                        "state_vel": self.state_vel.tolist(),
                        "stale": False,
                        "dead_reckoning": True,
                    }
            for kf in self.kf_pos:
                kf.reset()
            self.direction_filter.reset()
            self._reset_motion_state()
            return None

        seen_ids = [int(mid) for mid in ids.flatten()]
        cached_poses = {}
        # Pick the correct 3D-corner array PER MARKER ID. Arena wall
        # markers (IDs < 30) use self.MARKER_3D_POINTS (0.5 m corners);
        # SDC26 target boxes (IDs ≥ 30) use self.MARKER_3D_POINTS_TARGET
        # (0.19 m corners). Before this per-ID split, every target was
        # measured at 2.63× its real distance (50/19 = 2.63), which put
        # detected targets at X≈34 m — 20 m outside the arena — and
        # made pursuit completely fail to find them.
        ref_points = self.MARKER_3D_POINTS.astype(np.float32)
        tgt_points = self.MARKER_3D_POINTS_TARGET.astype(np.float32)
        # Optional camera↔marker-distance correction. We keep the RAW
        # tvec in the cache because downstream needs it at its original
        # scale for:
        #   - reprojection-error weighting (projectPoints must match the
        #     actual detected corners or reproj_w collapses to 0.05)
        #   - distance-based marker weighting
        # The correction is applied separately inside the ref/target
        # loops at the point where we compute world coordinates.
        try:
            dist_scale = float(getattr(self, "distance_scale", 1.0))
        except Exception:
            dist_scale = 1.0
        if dist_scale <= 0 or not np.isfinite(dist_scale):
            dist_scale = 1.0
        for i, mid_raw in enumerate(seen_ids):
            mid = int(mid_raw)
            pts = tgt_points if mid >= 30 else ref_points
            ok, rvec, tvec = cv2.solvePnP(pts, corners[i].reshape(-1, 2), self.camera_matrix,
                                          self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cached_poses[mid] = (rvec, tvec.reshape(3))
        ref_indices = [i for i, mid in enumerate(seen_ids) if
                       int(mid) in self.marker_positions and int(mid) in cached_poses]
        # SDC26 target boxes: Blue 31-36, Red 41-46. Anything else in the
        # ≥30 range (30, 33, 37-40, 47-99) is NOT a valid target box and
        # must be ignored — otherwise a phantom detection can pollute the
        # targets dict and confuse the capture mission.
        _SDC26_TARGET_IDS = {31,32,33,34,35,36,41,42,43,44,45,46}
        tgt_indices = [i for i, mid in enumerate(seen_ids)
                       if int(mid) in _SDC26_TARGET_IDS and int(mid) in cached_poses]
        if not ref_indices:
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= POSE_HOLD_SEC:
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

            # Apply the distance correction here — ONLY on the position
            # output, never on t_m_c used for weighting. Scaling C_m
            # stretches the offset from marker to camera by `dist_scale`
            # without touching rotation, so C_w lands at the corrected
            # distance from the marker in world coords.
            if dist_scale != 1.0:
                C_m = C_m * dist_scale

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
            if w < MIN_REF_WEIGHT:
                continue

            cam_positions.append(c_pos_w)
            cam_dirs.append(c_dir_w)
            rot_mats.append(R_c_w)
            weights.append(w)
            ref_marker_ids.append(mid)

        # If too few high-quality refs remain, hold recent pose instead of producing bad estimates
        if len(weights) < MIN_REF_COUNT:
            if self.last_valid_pose is not None and (now_wall - self.last_valid_ts) <= POSE_HOLD_SEC:
                return _stale_payload(
                    refs=[int(m) for m in ref_marker_ids],
                    marker_weights={str(mid): float(w) for mid, w in zip(ref_marker_ids, weights)},
                    seen_ids=seen_ids
                )
            return None

        # Keep only top-K best reference markers when more are visible
        top_k = int(getattr(self, "top_k_markers", 4) or 0)
        if top_k > 0 and len(weights) > top_k:
            top_idx = np.argsort(weights)[-top_k:]
            cam_positions = [cam_positions[i] for i in top_idx]
            cam_dirs = [cam_dirs[i] for i in top_idx]
            rot_mats = [rot_mats[i] for i in top_idx]
            weights = [weights[i] for i in top_idx]
            ref_marker_ids = [ref_marker_ids[i] for i in top_idx]

        # Outlier rejection in position domain
        outlier_thr = float(getattr(self, "outlier_reject_m", OUTLIER_POS_THRESH))
        center = np.mean(np.array(cam_positions), axis=0)
        keep = [i for i, p in enumerate(cam_positions) if np.linalg.norm(p - center) <= outlier_thr]
        if len(keep) >= MIN_REF_COUNT:
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

        # Shared "tracking is currently fresh" predicate — used by both
        # the pose-jump gate (continuous-tracking outlier reject) and the
        # re-acquisition consensus check (long-stale recovery).
        tracking_fresh = (self.last_valid_pose is not None and
                          self.state_pos is not None and
                          self.state_ts is not None and
                          (now_wall - self.last_valid_ts) <= POSE_HOLD_SEC)

        # ── Pose-jump gate ────────────────────────────────────────────
        # If the fresh fix disagrees with the predicted state by more
        # than `max_pose_jump_m`, reject it rather than poisoning the
        # filter. The gate only fires when we have an existing state
        # AND tracking has been continuous — a bootstrap or a genuinely
        # long dropout falls through to the re-acquisition consensus
        # check below.
        try:
            gate_m = float(getattr(self, "max_pose_jump_m", 0.0))
        except Exception:
            gate_m = 0.0
        gate_m = max(0.0, gate_m)
        if gate_m > 0 and tracking_fresh:
            pred_gate = self._predict_state(capture_ts)
            if pred_gate is not None:
                jump = float(np.linalg.norm(f_pos_meas - pred_gate))
                if jump > gate_m:
                    # Outlier — return a stale payload coasting on the
                    # existing state. Next frame will try again with
                    # fresh markers; if several in a row all agree on
                    # the jump, POSE_HOLD_SEC eventually elapses and
                    # the consensus path bootstraps to the new location.
                    return _stale_payload(
                        refs=[int(m) for m in ref_marker_ids],
                        marker_weights={str(mid): float(w)
                                        for mid, w in zip(ref_marker_ids, weights)},
                        seen_ids=seen_ids,
                    )

        # ── Re-acquisition consensus ──────────────────────────────────
        # If tracking was lost for longer than POSE_HOLD_SEC the previous
        # state is no longer trustworthy as a prior — but a single bad
        # solvePnP fix shouldn't teleport the published position either.
        # Cache the first fresh fix and only accept it once a second fix
        # confirms it within RECOVERY_DT_S / RECOVERY_DIST_M. The
        # confirmed fix is partially blended toward the predicted state
        # (RECOVERY_ALPHA_SCALE) so even after confirmation we don't snap.
        if (not tracking_fresh) and self.state_pos is not None:
            cand_pos = self._recovery_candidate_pos
            cand_ts = float(self._recovery_candidate_ts or 0.0)
            cand_age = float(capture_ts) - cand_ts
            if cand_pos is None or cand_age > RECOVERY_DT_S:
                # No candidate (or candidate timed out) — start a new one
                # and coast.
                self._recovery_candidate_pos = f_pos_meas.copy()
                self._recovery_candidate_ts = float(capture_ts)
                return _stale_payload(
                    refs=[int(m) for m in ref_marker_ids],
                    marker_weights={str(mid): float(w)
                                    for mid, w in zip(ref_marker_ids, weights)},
                    seen_ids=seen_ids,
                )
            cand_dist = float(np.linalg.norm(f_pos_meas - cand_pos))
            if cand_dist > RECOVERY_DIST_M:
                # Two fresh fixes disagree — replace the candidate and
                # keep coasting. We never confirm an inconsistent pair.
                self._recovery_candidate_pos = f_pos_meas.copy()
                self._recovery_candidate_ts = float(capture_ts)
                return _stale_payload(
                    refs=[int(m) for m in ref_marker_ids],
                    marker_weights={str(mid): float(w)
                                    for mid, w in zip(ref_marker_ids, weights)},
                    seen_ids=seen_ids,
                )
            # Confirmed re-acquisition — partially blend the measurement
            # toward the predicted state so the published position
            # doesn't snap. _update_motion_state below will then absorb
            # the (already-softened) measurement at its normal alpha.
            pred_now = self._predict_state(capture_ts)
            if pred_now is not None:
                f_pos_meas = pred_now + RECOVERY_ALPHA_SCALE * (f_pos_meas - pred_now)
            self._recovery_candidate_pos = None
            self._recovery_candidate_ts = 0.0
        else:
            # Continuous tracking (or genuine cold-start with no state).
            # Drop any pending candidate from a previous gap.
            self._recovery_candidate_pos = None
            self._recovery_candidate_ts = 0.0

        # Apply delayed measurement update at capture time, then predict to evaluation time.
        quality = float(np.mean(weights)) * min(1.0, len(weights) / 3.0)
        blend_alpha = np.clip(0.35 + 0.5 * quality, MEAS_BLEND_MIN, MEAS_BLEND_MAX)
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
            # Apply distance correction to the camera→target offset so the
            # projected target position scales consistently with the
            # camera pose (which was computed with the same correction).
            if dist_scale != 1.0:
                tvec_target = tvec_target * dist_scale

            # Estimate target world position from each reference marker independently
            target_estimates = []
            for c_pos_w, R_c_w in zip(cam_positions, rot_mats):
                target_estimates.append(c_pos_w + (R_c_w @ tvec_target))

            # Weighted fusion of all estimates
            t_w = np.zeros(3, dtype=float)
            for est, w in zip(target_estimates, w_ref):
                t_w += est * w

            # Clamp the target Z to a physically-sane band instead of
            # forcing it to the legacy TARGET_Z_POS constant. With the
            # correct 19 cm marker size the per-marker Z estimate is now
            # meaningful (target box sits on the floor ≈ 0.1-0.5 m, stand-
            # mounted up to ~1 m). Anything way outside that range is a
            # mirror-pose / noise artefact; fall back to 0 instead of
            # propagating a Z that would send the drone into the floor
            # or the ceiling.
            z_meas = float(t_w[2])
            if z_meas < -0.5 or z_meas > 2.0 or not np.isfinite(z_meas):
                t_w[2] = 0.0
            else:
                t_w[2] = max(0.0, z_meas)

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

        # Compute average marker pixel sizes and center pixel coords for all seen markers
        marker_pixel_sizes = {}
        marker_centers = {}
        for i, mid_raw in enumerate(seen_ids):
            mid = int(mid_raw)
            pts = corners[i].reshape(4, 2)
            side_lens = [float(np.linalg.norm(pts[j] - pts[(j + 1) % 4])) for j in range(4)]
            marker_pixel_sizes[str(mid)] = round(float(np.mean(side_lens)), 2)
            cx = round(float(np.mean(pts[:, 0])), 1)
            cy = round(float(np.mean(pts[:, 1])), 1)
            marker_centers[str(mid)] = [cx, cy]

        return {
            "cam": f_pos.tolist(),
            "dir": f_dir.tolist(),
            "targets": targets,
            "ref_markers": [int(m) for m in ref_marker_ids],
            "marker_weights": marker_weights,
            "marker_pixel_sizes": marker_pixel_sizes,
            "marker_centers": marker_centers,
            "seen_markers": seen_ids,
            "seen_count": len(seen_ids),
            "capture_ts": capture_ts,
            "eval_ts": eval_ts,
            "state_vel": self.state_vel.tolist(),
            "stale": False,
        }
