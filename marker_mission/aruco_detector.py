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
    # Which solvePnP path produced this pose. One of:
    #   "ippe_only"     -- IPPE_SQUARE returned only one front-facing
    #                       candidate (or only one survived back-facing
    #                       filter), used directly.
    #   "ippe_lowerr"   -- two IPPE candidates; picked the lower
    #                       reproj-err winner without temporal override.
    #   "ippe_temporal" -- two IPPE candidates with similar reproj-err;
    #                       temporal-continuity heading match overrode
    #                       the lower-err winner.
    #   "ippe_collapsed"-- two IPPE candidates were heading-mirrors with
    #                       similar reproj-err (the near-frontal blind
    #                       window); the chosen pose's relative_heading
    #                       and per-marker world-position vote were both
    #                       collapsed to the midpoint. Without this the
    #                       world position would flip across the marker
    #                       normal every time IPPE picks the other branch.
    #   "iterative"     -- IPPE failed entirely (no front-facing
    #                       candidate or reproj-err > 2px), fell through
    #                       to SOLVEPNP_ITERATIVE.
    # Logged per tick so a flight log can be sliced by method when
    # diagnosing position jumps.
    pose_method: str = ""

    # Camera position in MARKER frame, populated only by the
    # near-frontal mirror-collapse: the midpoint of the two IPPE
    # candidates' ``-R.T @ t`` vectors. ``arena.position_from_marker``
    # uses this when set so the per-marker world-position vote stays
    # on the marker normal during the planar-ambiguity blind window
    # instead of flipping with the chosen IPPE branch.
    collapsed_camera_position_m: Optional[np.ndarray] = None

    # Camera position in MARKER frame derived from the LOSER IPPE
    # candidate (the branch we did NOT pick). Set whenever IPPE
    # returned 2 front-facing candidates. ``arena.estimate_position``
    # uses this as a swap target when the chosen branch's world
    # position falls outside the arena -- i.e. IPPE picked the wrong
    # branch on reprojection-error noise. None when only one
    # candidate survived (no ambiguity to resolve).
    alt_camera_position_m: Optional[np.ndarray] = None

    # The loser IPPE candidate's ``relative_heading_deg`` -- the
    # mirror of ``relative_heading_deg`` across the marker normal.
    # Populated alongside ``alt_camera_position_m`` whenever both
    # branches survived. ``vision_worker`` uses this to swap the
    # active marker's heading when the magnetometer says the chosen
    # branch is wrong, so the controller's body-frame projection
    # tracks the right side of the marker normal during APPROACH /
    # ALIGN / HEIGHT_ALIGN / HOLD. None when the mirror-collapse
    # path already overrode ``relative_heading_deg`` to the midpoint.
    alt_relative_heading_deg: Optional[float] = None


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

        # Per-marker cache of the last accepted rvec + timestamp. Used to
        # break the IPPE_SQUARE planar-pose ambiguity by temporal
        # continuity: the two IPPE candidates for a planar marker differ
        # by a large rotation (often ~180 deg about an axis in the
        # marker plane), so the wrong one is far from the previous
        # frame's rvec while the right one stays close. Earlier versions
        # cached only relative_heading_deg, which is just the horizontal
        # projection of the marker normal -- it doesn't distinguish the
        # two solutions when their tilt difference is mostly vertical
        # (flights 2026-05-02_19-10-53 etc. flipped world-X by O(2 m)
        # frame-to-frame on off-centre markers because of this). Caching
        # the full rvec uses the entire 3-DOF rotation as the
        # discriminator, which is reliable.
        self._prev_rvec: dict[int, tuple[np.ndarray, float]] = {}
        self._ambiguity_max_age_s = 5.0  # cache invalidates after this gap
        # Toggleable via /tune. When False, the detector skips the
        # mirror-collapse path that overrides the chosen pose's
        # heading + position with the midpoint of the two IPPE
        # candidates (used for the truly-frontal blind window).
        self.enable_mirror_collapse = True
        # Thresholds for the mirror-collapse gate (also tunable via
        # /tune). Sum-near-zero is the real "is this a mirror pair"
        # check; max|hdg| restricts firing to the truly-frontal blind
        # window. Defaults match the dataclass values in MissionConfig.
        self.mirror_collapse_sum_hdg_deg = 5.0
        self.mirror_collapse_max_hdg_deg = 10.0

        # 3D marker corners in marker frame, OpenCV ArUco order TL-TR-BR-BL,
        # marker plane is z=0. With z pointing AWAY from the marker face,
        # the camera sits at z>0 when looking at the marker face-on.
        self._obj_pts: np.ndarray = np.zeros((4, 3), dtype=np.float64)
        self._rebuild_obj_pts()

    def _rebuild_obj_pts(self) -> None:
        s = self.marker_size / 2.0
        self._obj_pts = np.array([[-s,  s, 0.0],
                                  [ s,  s, 0.0],
                                  [ s, -s, 0.0],
                                  [-s, -s, 0.0]], dtype=np.float64)

    def set_marker_size(self, marker_size_m: float) -> None:
        """Live update of the physical marker side length. Cheap when
        unchanged (no-op); rebuilds the solvePnP object points when
        it actually changes. Called every tick from ``vision_worker``
        so a Save in the Arena tab propagates without a restart."""
        new = float(marker_size_m)
        if abs(new - self.marker_size) <= 1e-6:
            return
        self.marker_size = new
        self._rebuild_obj_pts()

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
                    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(t))):
                        continue
                    # Reject the back-facing IPPE solution. For a
                    # near-fronto-parallel marker the planar ambiguity
                    # can degenerate so that one of the two candidates
                    # places the marker normal in +cam_z (i.e., facing
                    # away from the camera) -- physically impossible
                    # because we just decoded the marker's pixels, but
                    # mathematically a valid second solution. R @ (0,0,1)
                    # is the marker normal in camera frame, and its
                    # z-component is R[2,2]; front-facing must have it
                    # negative. Filtering here means the rvec stored on
                    # MarkerPose is always front-facing, so downstream
                    # consumers (arena.position_from_marker) don't have
                    # to rediscover the issue.
                    R, _ = cv2.Rodrigues(r.reshape(3, 1))
                    if R[2, 2] >= 0.0:
                        continue
                    candidates.append((r, t, float(np.asarray(e).ravel()[0])))

            chosen_rvec: Optional[np.ndarray] = None
            chosen_tvec: Optional[np.ndarray] = None
            chosen_pose: Optional[MarkerPose] = None
            chosen_method: str = ""
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

                # Default tag before any temporal override fires.
                chosen_method = ("ippe_only" if len(scored) == 1
                                 else "ippe_lowerr")

                prev = self._prev_rvec.get(mid)
                prev_rvec = (prev[0] if prev is not None
                             and (ts - prev[1]) < self._ambiguity_max_age_s
                             else None)
                if prev_rvec is not None and len(scored) > 1:
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
                        before = scored[0]
                        scored.sort(key=lambda item:
                                    self._rvec_angle(item[1], prev_rvec))
                        if scored[0] is not before:
                            chosen_method = "ippe_temporal"

                chosen_pose, chosen_rvec, chosen_tvec, chosen_err = scored[0]

                # Always stash the loser branch's camera-in-marker
                # position when both candidates survived. The arena
                # estimator swaps to it when the chosen branch's
                # world position falls outside the arena -- the
                # off-axis case where IPPE's reproj-error winner is
                # inside sub-pixel noise of the wrong branch.
                if len(scored) > 1:
                    R_alt, _ = cv2.Rodrigues(
                        np.asarray(scored[1][1]).reshape(3, 1))
                    t_alt = np.asarray(scored[1][2],
                                       dtype=float).reshape(3)
                    chosen_pose.alt_camera_position_m = -R_alt.T @ t_alt
                    # The loser candidate's full pose (built by
                    # _build_pose during scoring above) carries the
                    # mirrored relative_heading_deg -- vision_worker's
                    # magnetometer-aware branch picker uses it to fix
                    # the +/-15 deg flicker the controller sees during
                    # APPROACH when the active marker's chosen branch
                    # is wrong.
                    chosen_pose.alt_relative_heading_deg = (
                        scored[1][0].relative_heading_deg)

                # Mirror-collapse for the irrecoverable near-frontal zone:
                # when two IPPE candidates fit equally and their hdgs are
                # exact mirrors (sum ~ 0), the image carries no information
                # about the sign. Reporting either value produces +/-15
                # flicker that oscillates the lateral PD. Reporting the
                # midpoint (~0) tells the controller "we're centered, no
                # strong heading signal" -- which is the truthful
                # interpretation and falls cleanly inside the deadband.
                #
                # The exact same ambiguity flips the per-marker world
                # position vote: -R.T @ t for the two candidates places
                # the camera at mirrored points across the marker
                # normal. We compute the midpoint of those two C_m
                # vectors here and stash it on chosen_pose so
                # ``arena.position_from_marker`` can use it instead of
                # re-deriving from the single winning rvec/tvec.
                if len(scored) > 1:
                    hdg0 = scored[0][0].relative_heading_deg
                    hdg1 = scored[1][0].relative_heading_deg
                    err0, err1 = scored[0][3], scored[1][3]
                    # Mirror-collapse should only fire when the drone is
                    # genuinely near the marker normal -- the truly
                    # ambiguous "no signal about which side" case. The
                    # sum-near-zero check alone is necessary but not
                    # sufficient: a drone at significant lateral offset
                    # (e.g. 1 m off-normal at 3 m, hdg = +/-18 deg) also
                    # has IPPE candidates summing to ~0 deg, but there
                    # the geometrically-correct candidate IS distinct
                    # and we'd much rather report it than collapse to
                    # the marker normal (= the wrong answer by ~1 m).
                    # Both individual headings being small is what
                    # restricts us to the actual blind window.
                    mirrored = (abs(hdg0 + hdg1) < self.mirror_collapse_sum_hdg_deg
                                and max(abs(hdg0), abs(hdg1)) < self.mirror_collapse_max_hdg_deg)
                    similar_err = (err0 < 1.0
                                   and err1 < 2.0 * err0 + 0.2)
                    if mirrored and similar_err and self.enable_mirror_collapse:
                        chosen_pose.relative_heading_deg = (hdg0 + hdg1) / 2.0
                        R0, _ = cv2.Rodrigues(
                            np.asarray(scored[0][1]).reshape(3, 1))
                        R1, _ = cv2.Rodrigues(
                            np.asarray(scored[1][1]).reshape(3, 1))
                        t0 = np.asarray(scored[0][2], dtype=float).reshape(3)
                        t1 = np.asarray(scored[1][2], dtype=float).reshape(3)
                        C0 = -R0.T @ t0
                        C1 = -R1.T @ t1
                        chosen_pose.collapsed_camera_position_m = (
                            (C0 + C1) * 0.5)
                        chosen_method = "ippe_collapsed"

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
                # Same back-facing rejection as the IPPE branch above --
                # ITERATIVE doesn't suffer the planar ambiguity but a
                # poorly conditioned image can still converge wrong.
                R_iter, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                if R_iter[2, 2] >= 0.0:
                    continue
                chosen_rvec = rvec.ravel()
                chosen_tvec = tvec.ravel()
                chosen_pose = self._build_pose(mid, img_pts, chosen_rvec,
                                               chosen_tvec, ts, (W, H))
                chosen_method = "iterative"

            chosen_pose.pose_method = chosen_method
            self._prev_rvec[mid] = (np.asarray(chosen_rvec).copy(), ts)
            out.append(chosen_pose)
        return out

    @staticmethod
    def _rvec_angle(r1: np.ndarray, r2: np.ndarray) -> float:
        """Angular distance (radians) between two Rodrigues rotations.
        Used as the IPPE planar-ambiguity tie-breaker."""
        R1, _ = cv2.Rodrigues(np.asarray(r1).reshape(3, 1))
        R2, _ = cv2.Rodrigues(np.asarray(r2).reshape(3, 1))
        cos_angle = (np.trace(R1 @ R2.T) - 1.0) * 0.5
        return math.acos(max(-1.0, min(1.0, cos_angle)))

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
        # n_cam.z is guaranteed negative (front-facing) here: the IPPE
        # candidate loop in detect() already filters out any solution
        # with R[2,2] >= 0 before this is called. If a future change
        # ever feeds a back-facing pose in, downstream geometry
        # (relative_heading, arena.position_from_marker) silently
        # produces wrong values -- catch it loudly instead.
        assert n_cam[2] < 0.0, (
            f"_build_pose given back-facing pose for marker {marker_id}: "
            f"n_cam={n_cam!r}; detect() should have filtered it")

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

# Per-wall colour conventions matching the /arena tab's WALL_COLORS
# in ui.py. Stored as BGR (OpenCV) -- the RGB hex is in the comment.
_MINIMAP_WALL_COLORS_BGR = {
    "front": (21, 204, 250),    # #facc15 (yellow)
    "right": (128, 222, 74),    # #4ade80 (green)
    "back":  (255, 196, 88),    # #58c4ff (light blue)
    "left":  (113, 113, 248),   # #f87171 (red-pink)
}


def draw_arena_minimap(frame_bgr: np.ndarray,
                        arena_width_m: float,
                        arena_depth_m: float,
                        world_pos: Optional[tuple] = None,
                        arena_yaw_deg: Optional[float] = None,
                        markers: Optional[object] = None,
                        visible_marker_ids: Optional[object] = None,
                        size_px: int = 180,
                        margin_px: int = 10,
                        title: str = "") -> None:
    """Draw an arena top-down mini-map onto the upper-right corner
    of ``frame_bgr`` (mutates in place).

    Layout: 1 m grid, arena rectangle (white outline), origin tick,
    yellow dot for the drone position, yellow line+arrow for the
    drone's arena yaw. Coordinates are arena +x = right, +y = front
    (top of the mini-map). When ``world_pos`` is None the dot/arrow
    are skipped and the mini-map shows just the arena layout. When
    ``arena_yaw_deg`` is None the dot is drawn without an arrow.

    ``title`` is rendered in the top-left of the mini-map panel
    (small, for offline-reprocess comparisons -- "OLD" / "NEW").
    """
    if arena_width_m <= 0 or arena_depth_m <= 0:
        return
    H, W = frame_bgr.shape[:2]
    side = int(size_px)
    x0 = W - int(margin_px) - side
    y0 = int(margin_px)
    if x0 < 0 or y0 + side > H:
        return
    # Background panel (semi-transparent dark).
    panel = frame_bgr[y0:y0+side, x0:x0+side].copy()
    panel[:] = (28, 28, 28)
    cv2.addWeighted(panel, 0.85,
                    frame_bgr[y0:y0+side, x0:x0+side], 0.15, 0,
                    frame_bgr[y0:y0+side, x0:x0+side])
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + side - 1, y0 + side - 1),
                  (60, 60, 60), 1)

    pad = 14
    inner_w = side - 2 * pad
    inner_h = side - 2 * pad
    px_per_m = float(min(inner_w / float(arena_width_m),
                          inner_h / float(arena_depth_m)))
    cxp = x0 + side // 2
    cyp = y0 + side // 2

    def to_px(x_m: float, y_m: float):
        return (int(round(cxp + x_m * px_per_m)),
                int(round(cyp - y_m * px_per_m)))

    half_w = float(arena_width_m) / 2.0
    half_d = float(arena_depth_m) / 2.0

    # 1 m grid (drawn faintly).
    grid_col = (55, 55, 55)
    x_lo = -int(math.floor(half_w))
    x_hi = +int(math.floor(half_w))
    for xi in range(x_lo, x_hi + 1):
        p1 = to_px(xi, -half_d); p2 = to_px(xi, +half_d)
        cv2.line(frame_bgr, p1, p2, grid_col, 1)
    y_lo = -int(math.floor(half_d))
    y_hi = +int(math.floor(half_d))
    for yi in range(y_lo, y_hi + 1):
        p1 = to_px(-half_w, yi); p2 = to_px(+half_w, yi)
        cv2.line(frame_bgr, p1, p2, grid_col, 1)

    # Origin tick.
    cv2.drawMarker(frame_bgr, to_px(0, 0), (200, 200, 200),
                   cv2.MARKER_TILTED_CROSS, 6, 1)

    # Arena rectangle.
    cv2.rectangle(frame_bgr,
                  to_px(-half_w, +half_d),
                  to_px(+half_w, -half_d),
                  (170, 170, 170), 1)

    # Wall labels (F = front, top of map).
    cv2.putText(frame_bgr, "F",
                (cxp - 4, to_px(0, half_d)[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "B",
                (cxp - 4, to_px(0, -half_d)[1] + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)

    # Markers. Each known marker gets a small wall-coloured dot at
    # its arena position; markers currently visible in this frame
    # get an additional white outline ring so the operator can tell
    # at a glance which references the position estimate is using.
    # Conventions match /arena's WALL_COLORS in the web UI.
    #
    # Vertical stacking: two markers that share (x, y) but differ in
    # z (e.g. the default 16-marker layout has a high-z and low-z
    # marker per slot) would otherwise overlap in the top-down view.
    # We group by rounded (x, y), sort descending by z, and offset
    # each dot a few pixels along screen-y so the stack is visible
    # as separate touching dots from top (high z) to bottom (low z).
    if markers is not None:
        seen_ids = set()
        if visible_marker_ids is not None:
            for v in visible_marker_ids:
                try: seen_ids.add(int(v))
                except (TypeError, ValueError): pass
        try:
            iter_ms = (list(markers.values()) if hasattr(markers, "values")
                       else list(markers))
        except Exception:
            iter_ms = []
        # Group co-located markers, biggest-z first within each group.
        from collections import defaultdict
        groups = defaultdict(list)
        for m in iter_ms:
            try:
                mid = int(getattr(m, "id"))
                wall = str(getattr(m, "wall", "")).lower()
                pos = getattr(m, "position_m")
                mx = float(pos[0]); my = float(pos[1]); mz = float(pos[2])
            except Exception:
                continue
            key = (round(mx, 2), round(my, 2))
            groups[key].append((mz, mid, wall, mx, my))
        for key, items in groups.items():
            items.sort(key=lambda it: -it[0])     # high z first
            n = len(items)
            # Same convention as the web /arena canvas (drawArena's
            # STACK_OFFSET_PX = 14 on a 700 px canvas, ratio ~ 2.3 x
            # dot-radius). Mini-map dot radius is 3 px so step = 7
            # gives the same ratio: high-z marker drawn slightly
            # higher on screen, low-z slightly lower.
            for idx, (mz, mid, wall, mx, my) in enumerate(items):
                base = to_px(mx, my)
                step = 7
                offset = -step * (n - 1) / 2.0 + idx * step
                mp = (base[0], int(round(base[1] + offset)))
                col = _MINIMAP_WALL_COLORS_BGR.get(wall, (180, 180, 180))
                cv2.circle(frame_bgr, mp, 3, col, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, mp, 3, (0, 0, 0), 1, cv2.LINE_AA)
                if mid in seen_ids:
                    cv2.circle(frame_bgr, mp, 6, (255, 255, 255),
                               1, cv2.LINE_AA)

    # Drone position + yaw arrow.
    if world_pos is not None:
        wx = float(world_pos[0]); wy = float(world_pos[1])
        in_arena = abs(wx) <= half_w + 0.5 and abs(wy) <= half_d + 0.5
        dx, dy = to_px(wx, wy)
        # Always draw the dot, even if just outside the rectangle, so the
        # operator can see it when the position is briefly drifting OOB.
        if (x0 + 3 <= dx <= x0 + side - 3
                and y0 + 3 <= dy <= y0 + side - 3):
            yellow = (0, 220, 220)         # BGR yellow
            cv2.circle(frame_bgr, (dx, dy), 5, yellow, -1)
            cv2.circle(frame_bgr, (dx, dy), 5, (0, 0, 0), 1)
            if arena_yaw_deg is not None:
                ang = math.radians(float(arena_yaw_deg))
                L = 16
                ax = int(round(dx + L * math.sin(ang)))
                ay = int(round(dy - L * math.cos(ang)))
                cv2.line(frame_bgr, (dx, dy), (ax, ay), yellow, 2)
                base = math.atan2(ay - dy, ax - dx)
                head_size = 5
                hx1 = int(round(ax - head_size * math.cos(base - math.pi/6)))
                hy1 = int(round(ay - head_size * math.sin(base - math.pi/6)))
                hx2 = int(round(ax - head_size * math.cos(base + math.pi/6)))
                hy2 = int(round(ay - head_size * math.sin(base + math.pi/6)))
                cv2.line(frame_bgr, (ax, ay), (hx1, hy1), yellow, 2)
                cv2.line(frame_bgr, (ax, ay), (hx2, hy2), yellow, 2)
        if not in_arena:
            cv2.putText(frame_bgr, "OOB",
                        (x0 + 6, y0 + side - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 220, 220), 1, cv2.LINE_AA)
        # Numeric label.
        label = f"({wx:+.2f}, {wy:+.2f})"
        cv2.putText(frame_bgr, label,
                    (x0 + 6, y0 + side - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, label,
                    (x0 + 6, y0 + side - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1,
                    cv2.LINE_AA)

    # Optional title (e.g., OLD / NEW for the offline reprocessor).
    if title:
        cv2.putText(frame_bgr, title,
                    (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, title,
                    (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)


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
        # Per-marker label: id + raw pose values (distance / yaw_to_marker /
        # relative_heading). Drawn with a black outline + coloured fill so
        # it stays legible against varied backgrounds. The top-left HUD
        # shows the smoothed values for the target only; per-marker raw
        # values let the operator spot non-target markers and judge their
        # geometry at a glance.
        lines = [
            f"id={p.marker_id}",
            f"d={p.distance_m:.2f}m  y={p.yaw_deg:+.1f}deg",
            f"h={p.relative_heading_deg:+.1f}deg",
        ]
        ox = int(cx) + 12
        oy = int(cy) - 12
        for i, line in enumerate(lines):
            y = oy + i * 21
            cv2.putText(out, line, (ox, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.63, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (ox, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.63, col, 1, cv2.LINE_AA)
    # Optional extra status lines in the top-left corner.
    y = 33
    if extra_lines:
        for line in extra_lines:
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.825, (255, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.825, (0, 0, 0), 1,
                        cv2.LINE_AA)
            y += 33
    # Per-marker list in the top-left, colour-coded the same way the
    # marker outlines are (green = target, grey = other). Sorted with
    # the target first so the operator's eye lands on it. Skipped when
    # nothing is visible.
    if poses:
        # spacer line
        y += 6
        cv2.putText(out, "Markers seen:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.675, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.putText(out, "Markers seen:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.675, (0, 0, 0), 1,
                    cv2.LINE_AA)
        y += 27
        ordered = sorted(poses, key=lambda p: (
            0 if (target_id is not None and p.marker_id == target_id) else 1,
            p.marker_id))
        for p in ordered:
            is_target = (target_id is not None and p.marker_id == target_id)
            col = (0, 220, 0) if is_target else (140, 140, 140)
            line = (f"id={p.marker_id:>3}  d={p.distance_m:.2f}m  "
                    f"y={p.yaw_deg:+6.1f}deg  "
                    f"h={p.relative_heading_deg:+6.1f}deg")
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.675, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.675, col, 1, cv2.LINE_AA)
            y += 26
    return out
