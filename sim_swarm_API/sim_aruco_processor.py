"""
ArUco marker detection for the sim_swarm_API virtual camera.

Processes JPEG frames from the browser's virtual drone camera,
runs OpenCV ArUco detection, and computes the drone's position
using solvePnP against known marker world positions.

Uses the same solvePnP pipeline as aruco-position/aruco_positioning.py
but with marker positions matching the sim's pillar layout.
The sim uses DICT_ARUCO_ORIGINAL (5x5 classic) markers, not DICT_4X4_50.

Coordinate system (pi_position world coords):
  x: left/right (-10..10), y: height (0..6), z: forward/back (0..10)
  Maps to sim Three.js: sim_x = world_x, sim_y = world_y, sim_z = -(world_z - 5)
  Maps to arena coords: arena_x = world_x + 10, arena_y = world_z
"""

import base64
import json
import logging
import math
import socket
import time
from typing import Optional

import cv2
import numpy as np
from cv2 import aruco

log = logging.getLogger("sim_aruco")


class SimArucoProcessor:
    """
    Processes virtual camera frames and detects ArUco markers.
    Uses the exact same DICT_4X4_50 dictionary as the real pipeline.
    """

    # 0.5m x 0.5m marker corners (same as aruco_positioning.py)
    MARKER_3D_POINTS = np.array([
        [-0.25,  0.25, 0],
        [ 0.25,  0.25, 0],
        [ 0.25, -0.25, 0],
        [-0.25, -0.25, 0],
    ], dtype=np.float32)

    def __init__(
        self,
        frame_width: int = 320,
        frame_height: int = 240,
        fov_deg: float = 75.0,
        udp_target: str = "127.0.0.1",
        udp_port: int = 5005,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Build camera matrix from virtual camera FOV
        fov_rad = fov_deg * math.pi / 180
        fx = (frame_width / 2) / math.tan(fov_rad / 2)
        fy = fx  # square pixels
        cx = frame_width / 2
        cy = frame_height / 2

        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        # No lens distortion in the virtual camera
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # ArUco detector — the sim SVGs use DICT_ARUCO_ORIGINAL (5x5 classic)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Known marker world positions (pi_position coord system)
        # Built from the sim's addBoundaryPillars layout
        self.marker_positions = self._build_sim_marker_positions()

        # Wall type per marker for orientation
        self.marker_wall_type = {}
        for mid, info in self.marker_positions.items():
            self.marker_wall_type[mid] = info["wall"]

        # Wall rotation matrices (marker normal → world frame)
        # These define how each marker face is oriented in world coords
        self._wall_rotations = {
            "back":  np.eye(3),  # z=0 wall, markers face +z
            "front": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),  # z=10, face -z
            "left":  np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),   # x=-10, face +x
            "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),   # x=10, face -x
        }

        # Kalman filters for smoothing
        self._kf_pos = [_KF1D() for _ in range(3)]
        self._dir_filter = _EMA(alpha=0.3)

        # UDP output
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_target = (udp_target, udp_port)

        # State
        self.last_position: Optional[np.ndarray] = None
        self.last_direction: Optional[np.ndarray] = None
        self.last_ref_markers: list[int] = []
        self.last_targets: dict = {}
        self._last_process_time = 0.0

        log.info(
            f"SimArucoProcessor ready: {frame_width}x{frame_height}, "
            f"FOV={fov_deg}°, fx={fx:.1f}, "
            f"{len(self.marker_positions)} reference markers"
        )

    def _build_sim_marker_positions(self) -> dict:
        """
        Build marker world positions matching the sim's addBoundaryPillars().

        Sim Three.js coords:
          Pillars at x = [-5, 0, 5] on top edge (z = -arena.y/2 ≈ -4.89)
          Pillars at x = [-5, 0, 5] on bottom edge (z = +arena.y/2 ≈ +4.89)
          Pillars at z = [-arena.y/6, +arena.y/6] on left (x = -9.89) and right (x = +9.89)

        pi_position world coords (what aruco_positioning.py uses):
          x = sim_x                  (range: -10..10)
          y = -sim_y (height, but inverted: sim y goes up, world y goes... actually same)
          z = sim_z + arena.y/2      (range: 0..10)

        Actually let me re-derive. The pi_position system:
          Back wall: z=0, markers at x = [-6, -2, 2, 6]
          Front wall: z=10, markers at x = [-6, -2, 2, 6]
          Left wall: x=-10, markers at z = [3.333, 6.667]
          Right wall: x=10, markers at z = [3.333, 6.667]

        Sim pillar positions (Three.js x,z):
          Top edge (z=-5): x = [-5, 0, 5]     → world: x=[-5,0,5], z=0
          Bottom edge (z=+5): x = [-5, 0, 5]  → world: x=[-5,0,5], z=10
          Left (x=-10): z = [-1.667, 1.667]   → world: x=-10, z=[3.333, 6.667]
          Right (x=+10): z = [-1.667, 1.667]  → world: x=10, z=[3.333, 6.667]

        Marker IDs start at 21, each pillar gets 2 heights (2m, 4m).
        Mirrored markers use markerId+10.
        """
        markers = {}
        arena_y = 10  # arena.y in sim
        arena_x = 20  # arena.x in sim

        marker_id = 21

        # Pillar definitions: (sim_x, sim_z, wall_type, inward_axis)
        pillars = []

        # Top edge (sim z = -arena_y/2), facing inward (z+) → back wall in world
        for x in [-5, 0, 5]:
            pillars.append((x, -arena_y / 2, "back", "z+"))
        # Bottom edge (sim z = +arena_y/2), facing inward (z-) → front wall in world
        for x in [-5, 0, 5]:
            pillars.append((x, arena_y / 2, "front", "z-"))
        # Left edge (sim x = -arena_x/2), facing inward (x+) → left wall
        for z in [-arena_y / 6, arena_y / 6]:
            pillars.append((-arena_x / 2, z, "left", "x+"))
        # Right edge (sim x = +arena_x/2), facing inward (x-) → right wall
        for z in [-arena_y / 6, arena_y / 6]:
            pillars.append((arena_x / 2, z, "right", "x-"))

        for sim_x, sim_z, wall, inward in pillars:
            for height in [2.0, 4.0]:
                # Convert sim coords to pi_position world coords
                world_x = sim_x
                world_y = -height  # In aruco_positioning, y is typically the vertical with markers at y=-1, +1
                world_z = sim_z + arena_y / 2  # shift so back wall is z=0, front is z=10

                markers[marker_id] = {
                    "pos": np.array([world_x, world_y, world_z], dtype=np.float64),
                    "wall": wall,
                    "height": height,
                    "sim_pos": (sim_x, height, sim_z),
                }
                # Also register the mirrored marker (markerId + 10)
                mirror_id = marker_id + 10
                markers[mirror_id] = {
                    "pos": np.array([world_x, world_y, world_z], dtype=np.float64),
                    "wall": self._mirror_wall(wall),
                    "height": height,
                    "sim_pos": (sim_x, height, sim_z),
                }
                marker_id += 1

        return markers

    @staticmethod
    def _mirror_wall(wall: str) -> str:
        return {"back": "front", "front": "back", "left": "right", "right": "left"}.get(wall, wall)

    def process_frame(self, jpeg_bytes: bytes, drone_id: str = "unknown") -> dict:
        """
        Process a JPEG camera frame. Detect ArUco markers and compute position.

        Returns dict with:
          cam: [x, y, z] in pi_position world coords (or None)
          dir: [dx, dy, dz] camera direction
          targets: {marker_id: [x, y, z]} for unknown markers
          ref_markers: [id, ...] detected reference marker IDs
          markers_detected: total markers found
        """
        t0 = time.monotonic()

        # Decode JPEG
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return {"cam": None, "error": "decode_failed"}

        # Detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self._reset_filters()
            return {
                "cam": None,
                "dir": None,
                "targets": {},
                "ref_markers": [],
                "markers_detected": 0,
            }

        # Separate reference markers from unknown targets
        ref_poses = []
        tgt_poses = {}

        for i, corner in enumerate(corners):
            mid = int(ids[i][0])

            # solvePnP for this marker
            ok, rvec, tvec = cv2.solvePnP(
                self.MARKER_3D_POINTS,
                corner.reshape(-1, 2),
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            tvec = tvec.reshape(3)

            if mid in self.marker_positions:
                # Reference marker — compute camera world position
                info = self.marker_positions[mid]
                R_m_c, _ = cv2.Rodrigues(rvec)
                R_m_w = self._wall_rotations.get(info["wall"], np.eye(3))

                cam_pos = info["pos"] + R_m_w @ (-R_m_c.T @ tvec)

                # Weight based on marker quality
                w = self._marker_weight(corner, np.linalg.norm(tvec))
                ref_poses.append((mid, cam_pos, R_m_w @ R_m_c, w))
            elif mid >= 30:
                # Unknown target marker
                tgt_poses[mid] = tvec

        if not ref_poses:
            return {
                "cam": None,
                "dir": None,
                "targets": {},
                "ref_markers": [int(ids[i][0]) for i in range(len(ids))],
                "markers_detected": len(ids),
            }

        # Weighted average of camera positions from all reference markers
        total_w = sum(w for _, _, _, w in ref_poses)
        if total_w < 1e-9:
            return {"cam": None, "ref_markers": [], "markers_detected": len(ids)}

        raw_pos = sum(pos * (w / total_w) for _, pos, _, w in ref_poses)

        # Direction from best-weight marker
        best_idx = max(range(len(ref_poses)), key=lambda i: ref_poses[i][3])
        best_R = ref_poses[best_idx][2]
        raw_dir = best_R @ np.array([0, 0, 1])
        raw_dir /= (np.linalg.norm(raw_dir) + 1e-9)

        # Kalman filter smoothing
        filt_pos = np.array([self._kf_pos[j].update(raw_pos[j]) for j in range(3)])
        filt_dir = self._dir_filter.update(raw_dir)
        filt_dir /= (np.linalg.norm(filt_dir) + 1e-9)

        # Resolve target markers relative to camera
        targets = {}
        for tid, tvec in tgt_poses.items():
            t_world = filt_pos + best_R @ tvec
            targets[int(tid)] = t_world.tolist()

        self.last_position = filt_pos
        self.last_direction = filt_dir
        self.last_ref_markers = [mid for mid, _, _, _ in ref_poses]
        self.last_targets = targets

        elapsed = time.monotonic() - t0
        result = {
            "cam": filt_pos.tolist(),
            "dir": filt_dir.tolist(),
            "targets": {str(k): v for k, v in targets.items()},
            "ref_markers": self.last_ref_markers,
            "markers_detected": len(ids),
            "process_ms": round(elapsed * 1000, 1),
        }

        return result

    def send_udp(self, result: dict, stale: bool = False) -> None:
        """Send position result as UDP packet (pi_position format)."""
        if result.get("cam") is None:
            return
        msg = {
            "cam": result["cam"],
            "dir": result.get("dir", [0, 0, 0]),
            "targets": result.get("targets", {}),
            "ref_markers": result.get("ref_markers", []),
            "stale": stale,
        }
        try:
            data = json.dumps(msg).encode("utf-8")
            self._udp_sock.sendto(data, self._udp_target)
        except Exception as e:
            log.debug(f"UDP send failed: {e}")

    def get_arena_position(self, result: dict) -> Optional[dict]:
        """
        Convert pi_position world coords to arena coords for telemetry.
        arena_x = world_x + 10, arena_y = world_z
        """
        cam = result.get("cam")
        if cam is None:
            return None
        return {
            "aruco_x": round(cam[0] + 10.0, 3),
            "aruco_y": round(cam[2], 3),
            "aruco_z": round(-cam[1], 3),  # world_y is inverted height
            "aruco_ref_markers": result.get("ref_markers", []),
            "aruco_markers_detected": result.get("markers_detected", 0),
        }

    def _marker_weight(self, corner, distance: float) -> float:
        pts = corner.reshape(4, 2)
        v1, v2 = pts[1] - pts[0], pts[3] - pts[0]
        area = abs(float(np.cross(v1, v2)))
        area_w = min(max(area / 5000.0, 0.1), 1.0)
        dist_w = math.exp(-distance / 10.0)
        side_lens = [float(np.linalg.norm(pts[j] - pts[(j + 1) % 4])) for j in range(4)]
        mean_side = sum(side_lens) / 4
        std_side = (sum((s - mean_side) ** 2 for s in side_lens) / 4) ** 0.5
        shape_w = math.exp(-std_side / (mean_side + 1e-6) * 5.0)
        return area_w * dist_w * shape_w

    def _reset_filters(self):
        for kf in self._kf_pos:
            kf.reset()
        self._dir_filter.reset()


class _KF1D:
    """Minimal 1D Kalman filter (same as aruco_positioning.py)."""
    def __init__(self):
        self.state = None
        self.cov = np.eye(2)
        self.last_t = None

    def update(self, z):
        now = time.time()
        if self.state is None:
            self.state = np.array([z, 0.0])
            self.last_t = now
            return z
        dt = now - self.last_t
        self.last_t = now
        if dt > 0.5:
            self.state = np.array([z, 0.0])
            return z
        F = np.array([[1, dt], [0, 1]])
        self.state = F @ self.state
        Q = np.array([[1e-4, 0], [0, 1e-2]]) * 1e-3
        self.cov = F @ self.cov @ F.T + Q
        H = np.array([[1, 0]])
        y = z - H @ self.state
        S = H @ self.cov @ H.T + 0.1
        K = self.cov @ H.T / S
        self.state += (K * y).flatten()
        self.cov = (np.eye(2) - K @ H) @ self.cov
        return float(self.state[0])

    def reset(self):
        self.state = None
        self.last_t = None


class _EMA:
    """Exponential moving average."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = self.alpha * v + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None
