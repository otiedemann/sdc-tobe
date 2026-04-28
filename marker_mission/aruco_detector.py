"""
ArUco marker detection and pose extraction.

Given a frame and a calibration, this module returns the marker(s) in the
scene with full 6-DOF pose information.

Conventions (aerospace, with gimbal-stabilised camera)
------------------------------------------------------

* Camera +z = optical axis (forward).  Camera +x = image right.
  Camera +y = image down.  This is OpenCV's convention.
* The Anafi's gimbal keeps the camera's x-z plane horizontal in the
  WORLD: world-up is exactly ``-y_cam``.  We exploit this so the
  ``yaw_to_marker_deg`` and ``relative_heading_deg`` outputs are
  invariant to drone roll, drone pitch, and how the marker is rotated
  in its own plane (sideways or upside-down on the wall).

Output angles use a single sign convention -- aerospace right-hand rule
about world-up:

* ``yaw_to_marker_deg``  -- camera-frame bearing of the marker centre
  in the horizontal plane.  ``+`` = marker is to the camera's right,
  i.e. the camera must yaw RIGHT (CW from above) to face it.
  ``0`` = marker dead ahead.  Range (-180, 180].
* ``relative_heading_deg``  -- the drone's compass bearing AROUND the
  marker, measured at the marker, with the marker's outward normal as
  the 0° reference and CW (from above) as positive.  ``0`` = drone is
  directly in front of the marker face.  ``+90`` = drone is to the
  marker's RIGHT (as the marker would describe it, looking outward).
  Range (-180, 180].
* ``marker_normal_bearing_deg``  -- bearing of the marker's outward
  normal in the camera's horizontal plane.  Tells the operator how the
  marker is oriented on the wall.  Same sign rule as ``yaw_to_marker``.
* ``marker_tilt_deg``  -- how far the marker plane deviates from
  vertical.  ``0`` = perfectly vertical wall.  ``+90`` = marker on
  ceiling facing down.  ``-90`` = marker on floor facing up.
* ``marker_inplane_rot_deg``  -- how the marker is rotated within its
  own plane (i.e. is it upside down or sideways on the wall).  Purely
  diagnostic; the heading/yaw outputs are already invariant to this.

A short wrapper :func:`annotate_frame` draws a clean overlay on a copy
of the frame for the operator UI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .calibration_store import Calibration


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class MarkerPose:
    marker_id: int
    corners: np.ndarray              # shape (4, 2), float32, TL-TR-BR-BL
    rvec: np.ndarray                 # shape (3,), Rodrigues rotation
    tvec: np.ndarray                 # shape (3,), marker centre in camera frame
    distance_m: float                # Euclidean camera -> marker
    yaw_deg: float                   # camera bearing of marker, aerospace yaw
    relative_heading_deg: float      # drone bearing around the marker
    marker_normal_bearing_deg: float # bearing of marker's outward normal in camera frame
    marker_tilt_deg: float           # how far the marker plane is from vertical
    marker_inplane_rot_deg: float    # marker's rotation within its own plane
    timestamp: float                 # monotonic seconds when the frame was decoded
    frame_size: tuple[int, int]      # (width, height) of the source frame


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

_ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoDetector:
    """Detect ArUco markers and recover 6-DOF pose using OpenCV solvePnP."""

    def __init__(self, calibration: Calibration,
                 marker_size_m: float,
                 dict_name: str = "DICT_4X4_50"):
        if dict_name not in _ARUCO_DICTS:
            raise ValueError(f"unknown aruco dict: {dict_name}")
        self.calibration = calibration
        self.marker_size = float(marker_size_m)
        self.dict_id = _ARUCO_DICTS[dict_name]
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.dict_id)
        self._params = cv2.aruco.DetectorParameters()
        # Tighten the corner refinement for sub-pixel quality.
        self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._params.cornerRefinementWinSize = 5
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._params)

        # Per-marker cache of the last accepted relative_heading_deg and
        # its timestamp. Used to break the IPPE_SQUARE planar-pose
        # ambiguity by temporal continuity: pick whichever IPPE candidate
        # has the relative_heading closest to the previous accepted
        # value. relative_heading_deg is purely geometric (angle between
        # marker normal and marker->drone vector in horizontal plane)
        # and gimbal-stabilised, so it does NOT need correction for
        # drone yaw drift. An earlier version cached the camera-frame
        # normal vector and compared by dot product, but that comparison
        # was insensitive enough that the planar mirror could still flip
        # in (flight 22-38-36 orbit: hdg jumped +28 -> -4 -> -25 in two
        # frames while distance stayed put -- an instantaneous mirror,
        # not motion). Comparing the actual hdg quantity catches it.
        self._prev_relhdg: dict[int, tuple[float, float]] = {}
        self._ambiguity_max_age_s = 5.0  # cache invalidates after this gap

        # 3D marker corners in marker frame, OpenCV ArUco order TL-TR-BR-BL,
        # marker plane is z=0. With z pointing AWAY from the marker face,
        # the camera sits at z>0 when looking at the marker face-on.
        s = self.marker_size / 2.0
        self._obj_pts = np.array([[-s,  s, 0.0],
                                  [ s,  s, 0.0],
                                  [ s, -s, 0.0],
                                  [-s, -s, 0.0]], dtype=np.float64)

    # ------------------------------------------------------------------ scan
    def detect(self, frame_bgr: np.ndarray,
               wanted_id: Optional[int] = None) -> list[MarkerPose]:
        """Return a list of :class:`MarkerPose` for every detected marker.

        If ``wanted_id`` is given only that marker is returned (or [] if
        not visible).
        """
        ts = time.monotonic()
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        out: list[MarkerPose] = []
        if ids is None:
            return out
        K = self.calibration.camera_matrix
        D = self.calibration.dist_coeffs
        for c, id_arr in zip(corners, ids):
            mid = int(id_arr[0])
            if wanted_id is not None and mid != wanted_id:
                continue
            img_pts = c[0].astype(np.float64)

            # SOLVEPNP_IPPE_SQUARE returns BOTH planar-ambiguity solutions
            # (sorted by reprojection error). We pick between them using
            # temporal continuity instead of just trusting the lowest-error
            # one -- at low tilt the two are within numerical noise of each
            # other and which one wins flips between frames.
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self._obj_pts, img_pts, K, D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
            if ok:
                for r, t, e in zip(rvecs, tvecs, errs):
                    r = r.ravel(); t = t.ravel()
                    if np.all(np.isfinite(r)) and np.all(np.isfinite(t)):
                        candidates.append((r, t, float(np.asarray(e).ravel()[0])))

            chosen_rvec: Optional[np.ndarray] = None
            chosen_tvec: Optional[np.ndarray] = None
            chosen_pose: Optional[MarkerPose] = None
            if candidates:
                # Build the full pose for each candidate so we can
                # compare on the actual relative_heading we care about
                # rather than a proxy (camera-frame normal dot product
                # was not discriminative enough at low tilt).
                scored = []
                for r, t, e in candidates:
                    pose = self._build_pose(mid, img_pts, r, t, ts, (W, H))
                    scored.append((pose, r, t, e))
                # Always start from "lowest reprojection error first" --
                # this is the geometric truth when one solution clearly
                # fits better. Temporal continuity only overrides when
                # the candidates are within sub-pixel tie-territory (the
                # truly ambiguous case where IPPE flips on noise).
                scored.sort(key=lambda x: x[3])

                prev = self._prev_relhdg.get(mid)
                prev_hdg = (prev[0] if prev is not None
                            and (ts - prev[1]) < self._ambiguity_max_age_s
                            else None)
                if prev_hdg is not None and len(scored) > 1:
                    best_err = scored[0][3]
                    second_err = scored[1][3]
                    # "Ambiguous" = both fit reasonably AND errs are close.
                    # Otherwise the lower-err candidate is the geometric
                    # winner, even if it disagrees with the cached value
                    # (this prevents cache lock-in: an early wrong branch
                    # would otherwise self-perpetuate).
                    ambiguous = (best_err < 1.0
                                 and second_err < 2.0 * best_err + 0.2)
                    if ambiguous:
                        def hdg_delta(item):
                            d = ((item[0].relative_heading_deg - prev_hdg
                                  + 540.0) % 360.0) - 180.0
                            return abs(d)
                        scored.sort(key=hdg_delta)

                chosen_pose, chosen_rvec, chosen_tvec, chosen_err = scored[0]

                # Mirror-collapse for the irrecoverable near-frontal zone:
                # when two IPPE candidates fit equally and their hdgs are
                # exact mirrors (sum ~ 0), the image carries no information
                # about the sign. Reporting either value produces +/-15
                # flicker that oscillates the lateral PD. Reporting the
                # midpoint (~0) tells the controller "we're centered, no
                # strong heading signal" -- which is the truthful
                # interpretation and falls cleanly inside the deadband.
                if len(scored) > 1:
                    hdg0 = scored[0][0].relative_heading_deg
                    hdg1 = scored[1][0].relative_heading_deg
                    err0, err1 = scored[0][3], scored[1][3]
                    mirrored = abs(hdg0 + hdg1) < 5.0
                    similar_err = (err0 < 1.0
                                   and err1 < 2.0 * err0 + 0.2)
                    if mirrored and similar_err:
                        chosen_pose.relative_heading_deg = (hdg0 + hdg1) / 2.0

                # Reproj-err sanity gate: a winning IPPE candidate that still
                # mis-projects badly is suspect -- fall back to ITERATIVE.
                proj, _ = cv2.projectPoints(self._obj_pts,
                                            chosen_rvec, chosen_tvec, K, D)
                reproj_err = float(np.linalg.norm(
                    proj.reshape(-1, 2) - img_pts, axis=1).mean())
                if reproj_err > 2.0:           # TUNE: pixel threshold
                    chosen_rvec = None
                    chosen_pose = None

            if chosen_rvec is None:
                ok2, rvec, tvec = cv2.solvePnP(self._obj_pts, img_pts, K, D,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
                if (not ok2
                        or not np.all(np.isfinite(rvec))
                        or not np.all(np.isfinite(tvec))):
                    continue
                chosen_rvec = rvec.ravel()
                chosen_tvec = tvec.ravel()
                chosen_pose = self._build_pose(mid, img_pts, chosen_rvec,
                                               chosen_tvec, ts, (W, H))

            self._prev_relhdg[mid] = (chosen_pose.relative_heading_deg, ts)
            out.append(chosen_pose)
        return out

    # ----------------------------------------------------------------- build
    @staticmethod
    def _build_pose(marker_id: int, corners: np.ndarray,
                    rvec: np.ndarray, tvec: np.ndarray,
                    timestamp: float,
                    frame_size: tuple[int, int]) -> MarkerPose:
        """Compute pose with gimbal-aware horizontal-plane geometry.

        We rely on the Anafi gimbal keeping the camera level (i.e. the
        camera's x-z plane is the world horizontal plane).  All
        horizontal angles are measured in that plane so the outputs are
        invariant to:

          * how the marker is rotated WITHIN its plane on the wall
            (upside down, sideways, etc.),
          * the drone's pitch and roll angles (the gimbal removes them),
          * any roll the operator might command.
        """
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        distance = math.sqrt(tx * tx + ty * ty + tz * tz)

        # --- Marker outward normal in the camera frame ----------------------
        # Marker frame has +z OUT OF the marker face (toward the camera, when
        # the camera looks at the front of the marker).  We rotate (0,0,1)
        # by R to express it in the camera frame.
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        n_cam = (R @ np.array([0.0, 0.0, 1.0])).astype(float)
        # n_cam.z should be NEGATIVE when the camera is looking at the
        # marker's front face (the normal points back towards the camera).
        # solvePnP with a planar square sometimes returns the front/back-
        # ambiguous mirror solution (identity rotation) for a perfectly
        # centred, fronto-parallel marker.  Since ArUco IDs are only
        # readable from the front, any apparent "back-facing" pose is
        # the spurious one -- flip the normal so all downstream geometry
        # behaves as if the marker correctly faces the camera.
        if n_cam[2] > 0:
            n_cam = -n_cam

        # --- Yaw to marker (aerospace: + = nose right, CW from above) -------
        # Project the camera->marker vector onto the camera's horizontal
        # plane (camera x-z), which equals the WORLD horizontal plane
        # because of the gimbal.  Bearing of the marker:
        #     atan2(tx, tz) is the angle from +z_cam (forward) toward
        #     +x_cam (right). When the marker is on the camera's right,
        #     tx>0 -> atan2 returns positive -> our yaw_to_marker is positive.
        # That's the aerospace convention: positive yaw = nose-right.
        yaw_deg = math.degrees(math.atan2(tx, tz))

        # --- Relative heading around the marker -----------------------------
        # Reference direction = marker's outward normal (in horizontal plane).
        # Drone direction (from marker) = -t (in horizontal plane) because
        #   marker is at t (camera frame) and the drone is at the camera = 0.
        # Heading = signed horizontal angle from n_h to d_h, positive CW
        # when viewed from world-up (= -y_cam).
        n_h = np.array([n_cam[0], 0.0, n_cam[2]])
        d_h = np.array([-tx,      0.0, -tz])     # marker -> drone
        n_norm = float(np.linalg.norm(n_h))
        d_norm = float(np.linalg.norm(d_h))
        if n_norm < 1e-6 or d_norm < 1e-6:
            # Degenerate: marker on floor/ceiling (no horizontal normal),
            # or the camera is exactly at the marker centre.  Heading is
            # undefined; report 0 as a safe placeholder and let the
            # operator notice via marker_tilt_deg ~= +/-90.
            relative_heading_deg = 0.0
        else:
            n_hn = n_h / n_norm
            d_hn = d_h / d_norm
            dot = float(np.dot(n_hn, d_hn))
            # +y_cam points DOWN in the world.  Right-hand-rule cross
            # gives the y-component of (n x d):
            #    cross_y = n.z * d.x - n.x * d.z
            # Positive cross_y -> rotation from n_h to d_h is CCW about
            # +y_cam, i.e. CW when viewed from world-up.  That's our
            # "positive heading" sign convention.
            cross_y = float(n_hn[2] * d_hn[0] - n_hn[0] * d_hn[2])
            relative_heading_deg = math.degrees(math.atan2(cross_y, dot))

        # --- Marker normal bearing in camera frame --------------------------
        # Where on the wall does the marker face?  Same horizontal plane,
        # measured from camera +z (forward).  Diagnostic only.
        if n_norm < 1e-6:
            marker_normal_bearing_deg = 0.0
        else:
            marker_normal_bearing_deg = math.degrees(math.atan2(n_cam[0],
                                                                n_cam[2]))

        # --- Marker tilt: how far is it from vertical? ----------------------
        # n_cam.y is the world-vertical component of the normal (remember
        # +y_cam = world DOWN, so positive n_cam.y means the normal points
        # downward = marker on floor facing up; negative means the normal
        # points up = marker on ceiling facing down).  asin(-n_cam.y) is
        # signed tilt from a vertical wall:
        #   0   -> marker on a vertical wall
        #   +90 -> marker on the ceiling, facing the floor
        #   -90 -> marker on the floor, facing up
        ny = max(-1.0, min(1.0, float(n_cam[1])))
        marker_tilt_deg = math.degrees(math.asin(-ny))

        # --- Marker in-plane rotation ---------------------------------------
        # The marker's local +x axis, expressed in the camera frame.  Project
        # it onto the camera's image plane (x,y of camera frame).  Compare
        # against camera +x (image right) to get the marker's rotation in
        # the image.  Useful operator diagnostic ("is this marker upside
        # down?").
        x_marker_in_cam = (R @ np.array([1.0, 0.0, 0.0])).astype(float)
        # Image-plane projection (drop z), with y flipped because +y_cam
        # is image down -- we want image-up to be positive for this
        # diagnostic so that "0 deg = marker upright" reads naturally.
        rot_image = math.degrees(math.atan2(-x_marker_in_cam[1],
                                             x_marker_in_cam[0]))
        marker_inplane_rot_deg = rot_image

        return MarkerPose(
            marker_id=marker_id,
            corners=corners.astype(np.float32),
            rvec=rvec.astype(np.float64),
            tvec=tvec.astype(np.float64),
            distance_m=float(distance),
            yaw_deg=float(yaw_deg),
            relative_heading_deg=float(relative_heading_deg),
            marker_normal_bearing_deg=float(marker_normal_bearing_deg),
            marker_tilt_deg=float(marker_tilt_deg),
            marker_inplane_rot_deg=float(marker_inplane_rot_deg),
            timestamp=timestamp,
            frame_size=frame_size,
        )


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def annotate_frame(frame_bgr: np.ndarray,
                   poses: list[MarkerPose],
                   target_id: Optional[int] = None,
                   extra_lines: Optional[list[str]] = None) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with marker overlays drawn on it.

    The overlay intentionally avoids OpenCV's built-in drawDetectedMarkers
    so we can highlight the target marker in green and others in grey.
    """
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for p in poses:
        is_target = (target_id is None) or (p.marker_id == target_id)
        col = (0, 220, 0) if is_target else (140, 140, 140)
        pts = p.corners.astype(np.int32)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, col, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(out, (int(cx), int(cy)), 4, col, -1)
        # Keep the per-marker label tiny: just the ID. Pose values live in the
        # top-left HUD so we avoid drawing them twice (and avoid overlapping
        # the marker itself when it's small in the frame).
        cv2.putText(out, f"id={p.marker_id}", (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    # Optional extra status lines in the top-left corner.
    if extra_lines:
        y = 22
        for line in extra_lines:
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                        cv2.LINE_AA)
            y += 22
    return out
