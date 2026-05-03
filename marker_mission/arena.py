"""
Arena-frame world positioning from visible ArUco markers.

Loads an arena layout JSON (``arena_config*.json`` -- same schema as the
``aruco-position/controller-modular`` package), and given a list of
:class:`MarkerPose` from the detector, returns the camera's world
position in arena coordinates as a weighted average of the per-marker
inversions.

This is the **basic** estimator -- one shot per frame, no temporal
filtering, no IMU. Plenty good for a first cut: every visible reference
marker votes, votes are weighted by inverse distance, the result is
plotted / logged / exposed via ``state.world_position_m``.

Future work:

* Per-marker quality weighting beyond pure 1/distance -- reprojection
  error, image area, shape squareness -- mirroring what
  ``aruco-position/controller-modular/aruco_position.py`` does.
* Outlier rejection in position space (drop markers whose voted-position
  is more than N standard deviations from the centroid).
* Kalman fusion across ticks for smoothing + velocity estimate.
* IMU dead-reckoning during marker dropouts.

Math, per visible reference marker:

* ``solvePnP`` already gave us ``(rvec, tvec)`` describing the
  marker->camera transform: ``x_c = R_m_c x_m + t_m_c``.
* Camera origin in marker frame: set ``x_c = 0``,
  ``C_m = -R_m_c^T t_m_c``.
* Marker pose in world: from arena config we know the marker's world
  position ``t_m_w`` and a fixed rotation ``R_m_w`` derived from the
  ``wall`` it's mounted on.
* Camera position in world: ``C_w = R_m_w C_m + t_m_w``.

Wall rotations are inherited verbatim from the controller-modular
package so positions transfer cleanly between the two systems.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .aruco_detector import MarkerPose


VALID_WALLS = ("front", "back", "left", "right")


# ---------------------------------------------------------------------------
# Arena configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArenaMarker:
    id: int
    label: str
    position_m: np.ndarray   # shape (3,), world coordinates [m]
    wall: str                # one of VALID_WALLS


@dataclass
class ArenaConfig:
    """Loaded arena layout. ``marker_size_m`` is informational here -- the
    controller still drives the detector with ``cfg.marker_size_m``;
    this field is what the layout file says it printed at."""

    marker_size_m: float
    markers: Dict[int, ArenaMarker]

    @classmethod
    def load(cls, path: Path) -> "ArenaConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        marker_size_m = float(data.get("marker_size_m", 0.18))
        markers: Dict[int, ArenaMarker] = {}
        for m in data.get("markers", []):
            mid = int(m["id"])
            wall = str(m["wall"]).lower()
            if wall not in VALID_WALLS:
                raise ValueError(
                    f"arena_config {path}: marker {mid} has unknown "
                    f"wall {wall!r} (expected one of {VALID_WALLS})")
            pos = np.array([float(m["x"]), float(m["y"]), float(m["z"])],
                           dtype=float)
            label = str(m.get("label", ""))
            markers[mid] = ArenaMarker(id=mid, label=label,
                                       position_m=pos, wall=wall)
        if not markers:
            raise ValueError(f"arena_config {path}: empty marker list")
        return cls(marker_size_m=marker_size_m, markers=markers)

    def __contains__(self, marker_id: int) -> bool:
        return marker_id in self.markers


# ---------------------------------------------------------------------------
# Wall rotations (marker frame -> world frame)
# ---------------------------------------------------------------------------
#
# Inherited verbatim from
# aruco-position/controller-modular/aruco_position.py:_initialize_wall_rotations,
# so position estimates produced here line up with that package.
#
# Front-wall mapping (marker axes -> world axes):
#   marker X -> world -X
#   marker Y -> world +Z
#   marker Z -> world +Y
# Other walls are this base rotated about world-Z by the wall's yaw.

def _rotz(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


_FRONT_BASE_ROT = np.array([
    [-1, 0, 0],
    [ 0, 0, 1],
    [ 0, 1, 0],
], dtype=float)

WALL_ROTATIONS: Dict[str, np.ndarray] = {
    "front": _FRONT_BASE_ROT,
    "back":  _rotz(180.0) @ _FRONT_BASE_ROT,
    "left":  _rotz(-90.0) @ _FRONT_BASE_ROT,
    "right": _rotz( 90.0) @ _FRONT_BASE_ROT,
}


# ---------------------------------------------------------------------------
# Per-marker position vote + weighted aggregation
# ---------------------------------------------------------------------------

@dataclass
class PositionEstimate:
    """One frame's world-position estimate plus the per-marker breakdown."""
    position_m: np.ndarray                  # (3,) world coordinates
    used_markers: List[int]                 # ids that contributed
    per_marker_position_m: Dict[int, np.ndarray] = field(default_factory=dict)
    weights: Dict[int, float] = field(default_factory=dict)   # normalised, sums to 1
    # solvePnP path used for each contributing marker (same key set as
    # weights). See ``MarkerPose.pose_method`` for the value vocabulary.
    per_marker_method: Dict[int, str] = field(default_factory=dict)


def position_from_marker(pose: MarkerPose,
                         marker: ArenaMarker) -> np.ndarray:
    """Camera world position from one marker's pose + its known world placement.

    ``pose.rvec`` / ``pose.tvec`` were produced by solvePnP inside the
    detector, so we just consume them; no second solver call. The
    detector guarantees ``rvec`` describes a front-facing pose
    (back-facing IPPE candidates are filtered out during selection),
    so we don't need to re-check the planar ambiguity here.
    """
    R_m_c, _ = cv2.Rodrigues(pose.rvec)
    t_m_c = np.asarray(pose.tvec, dtype=float).reshape(3)
    R_m_w = WALL_ROTATIONS[marker.wall]
    C_m = -R_m_c.T @ t_m_c
    return R_m_w @ C_m + marker.position_m


def estimate_position(arena: ArenaConfig,
                      poses: Iterable[MarkerPose]
                      ) -> Optional[PositionEstimate]:
    """Weighted average of camera world positions derived from each visible
    reference marker.

    Weight is currently ``1 / max(0.1, distance_m)`` -- close markers
    dominate. Returns ``None`` if no visible marker is in the arena
    config.
    """
    contributions: List[tuple] = []   # (mid, pos_w, weight, method)
    for p in poses:
        marker = arena.markers.get(int(p.marker_id))
        if marker is None:
            continue
        pos_w = position_from_marker(p, marker)
        weight = 1.0 / max(0.1, float(p.distance_m))
        contributions.append(
            (int(p.marker_id), pos_w, weight, str(p.pose_method or "")))

    if not contributions:
        return None

    total_w = sum(w for _, _, w, _ in contributions)
    weighted_sum = np.zeros(3, dtype=float)
    for _, pos_w, w, _ in contributions:
        weighted_sum += pos_w * w
    avg = weighted_sum / total_w

    return PositionEstimate(
        position_m=avg,
        used_markers=[mid for mid, _, _, _ in contributions],
        per_marker_position_m={mid: pos for mid, pos, _, _ in contributions},
        weights={mid: w / total_w for mid, _, w, _ in contributions},
        per_marker_method={mid: meth for mid, _, _, meth in contributions},
    )
