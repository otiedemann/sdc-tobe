"""
Arena-frame world positioning from visible ArUco markers.

Loads an arena layout JSON (``arena_config*.json``), and given a list of
:class:`MarkerPose` from the detector, returns the camera's world
position in arena coordinates as a weighted average of the per-marker
inversions.

This is the **basic** estimator -- one shot per frame, no temporal
filtering, no IMU. Plenty good for a first cut: every visible reference
marker votes, votes are weighted by inverse distance, the result is
plotted / logged / exposed via ``state.world_position_m``.

Future work:

* Per-marker quality weighting beyond pure 1/distance -- reprojection
  error, image area, shape squareness.
* Outlier rejection in position space (drop markers whose voted-position
  is more than N standard deviations from the centroid).
* Kalman fusion across ticks for smoothing + velocity estimate.
* IMU dead-reckoning during marker dropouts.

Coordinate convention (centred-origin)
--------------------------------------

Origin ``(0, 0, 0)`` is the centre of the arena floor. Looking down at
the floor:

* ``+x`` is to the right.
* ``+y`` is toward the front wall (top of the top-down view).
* ``+z`` is up.

Walls are at ``x = ±width/2`` (left / right) and ``y = ±depth/2``
(front / back). Markers sit on the inside face of those walls with
their normal pointing toward the origin.

Math, per visible reference marker:

* ``solvePnP`` already gave us ``(rvec, tvec)`` describing the
  marker->camera transform: ``x_c = R_m_c x_m + t_m_c``.
* Camera origin in marker frame: set ``x_c = 0``,
  ``C_m = -R_m_c^T t_m_c``.
* Marker pose in world: from arena config we know the marker's world
  position ``t_m_w`` and a fixed rotation ``R_m_w`` derived from the
  ``wall`` it's mounted on (see ``WALL_ROTATIONS`` below).
* Camera position in world: ``C_w = R_m_w C_m + t_m_w``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .aruco_detector import MarkerPose


VALID_WALLS = ("front", "back", "left", "right")

# Sensible defaults used both for ``ArenaConfig.load`` (when the JSON
# doesn't carry the metadata fields) and for ``default_arena``.
DEFAULT_WIDTH_M = 10.0
DEFAULT_DEPTH_M = 20.0
DEFAULT_TOP_Z_M = 4.0
DEFAULT_BOTTOM_Z_M = 2.0
DEFAULT_MARKER_SIZE_M = 0.18


# ---------------------------------------------------------------------------
# Wall rotations (marker frame -> arena frame)
# ---------------------------------------------------------------------------
#
# Each rotation maps the marker's local axes (ArUco convention: +x to
# the right when looking at the printed face from outside the wall, +y
# up, +z out of the face toward the camera) to arena axes (+x right,
# +y toward front wall, +z up).
#
# Front wall (y = +depth/2, normal points -y into arena):
#   marker +x -> arena +x       (camera looking at front sees its +x = arena +x)
#   marker +y -> arena +z       (top of marker = up)
#   marker +z -> arena -y       (face normal points back into arena)
#
# Other walls follow by symmetry. Each is a proper rotation
# (det = +1, columns right-handed); verified by hand and by the
# per-wall smoke test in ``estimate_position``.

WALL_ROTATIONS: Dict[str, np.ndarray] = {
    "front": np.array([[1, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]], dtype=float),
    "back":  np.array([[-1, 0, 0],
                       [ 0, 0, 1],
                       [ 0, 1, 0]], dtype=float),
    "left":  np.array([[0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]], dtype=float),
    "right": np.array([[ 0, 0, -1],
                       [-1, 0,  0],
                       [ 0, 1,  0]], dtype=float),
}


# ---------------------------------------------------------------------------
# Arena configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArenaMarker:
    id: int
    label: str
    position_m: np.ndarray   # shape (3,), arena coordinates [m]
    wall: str                # one of VALID_WALLS


@dataclass
class ArenaConfig:
    """Loaded arena layout. ``marker_size_m`` is informational here -- the
    controller still drives the detector with ``cfg.marker_size_m``;
    this field is what the layout file says it printed at.

    The ``width_m`` / ``depth_m`` / ``top_z_m`` / ``bottom_z_m`` fields
    are metadata used by the Arena tab to re-render the arena on the
    top-down canvas and to validate marker positions on save. They are
    optional in the JSON; loaders that don't care about them can ignore
    them.
    """

    marker_size_m: float
    markers: Dict[int, ArenaMarker]
    width_m: float = DEFAULT_WIDTH_M
    depth_m: float = DEFAULT_DEPTH_M
    top_z_m: float = DEFAULT_TOP_Z_M
    bottom_z_m: float = DEFAULT_BOTTOM_Z_M

    @classmethod
    def load(cls, path: Path) -> "ArenaConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict, source: str = "<dict>") -> "ArenaConfig":
        marker_size_m = float(data.get("marker_size_m", DEFAULT_MARKER_SIZE_M))
        width_m = float(data.get("width_m", DEFAULT_WIDTH_M))
        depth_m = float(data.get("depth_m", DEFAULT_DEPTH_M))
        top_z_m = float(data.get("top_z_m", DEFAULT_TOP_Z_M))
        bottom_z_m = float(data.get("bottom_z_m", DEFAULT_BOTTOM_Z_M))
        markers: Dict[int, ArenaMarker] = {}
        for m in data.get("markers", []):
            mid = int(m["id"])
            wall = str(m["wall"]).lower()
            if wall not in VALID_WALLS:
                raise ValueError(
                    f"arena_config {source}: marker {mid} has unknown "
                    f"wall {wall!r} (expected one of {VALID_WALLS})")
            pos = np.array([float(m["x"]), float(m["y"]), float(m["z"])],
                           dtype=float)
            label = str(m.get("label", ""))
            markers[mid] = ArenaMarker(id=mid, label=label,
                                       position_m=pos, wall=wall)
        if not markers:
            raise ValueError(f"arena_config {source}: empty marker list")
        return cls(marker_size_m=marker_size_m, markers=markers,
                   width_m=width_m, depth_m=depth_m,
                   top_z_m=top_z_m, bottom_z_m=bottom_z_m)

    def to_json_dict(self) -> dict:
        """Round-trippable JSON representation. The Arena tab uses this
        when writing the active config or a named save."""
        return {
            "marker_size_m": float(self.marker_size_m),
            "width_m": float(self.width_m),
            "depth_m": float(self.depth_m),
            "top_z_m": float(self.top_z_m),
            "bottom_z_m": float(self.bottom_z_m),
            "markers": [
                {"id": int(m.id), "label": m.label, "wall": m.wall,
                 "x": float(m.position_m[0]),
                 "y": float(m.position_m[1]),
                 "z": float(m.position_m[2])}
                for m in sorted(self.markers.values(), key=lambda x: x.id)
            ],
        }

    def __contains__(self, marker_id: int) -> bool:
        return marker_id in self.markers


# ---------------------------------------------------------------------------
# Default-arena generator
# ---------------------------------------------------------------------------

def default_arena(width_m: float = DEFAULT_WIDTH_M,
                  depth_m: float = DEFAULT_DEPTH_M,
                  top_z_m: float = DEFAULT_TOP_Z_M,
                  bottom_z_m: float = DEFAULT_BOTTOM_Z_M,
                  marker_size_m: float = DEFAULT_MARKER_SIZE_M
                  ) -> ArenaConfig:
    """Generate the default 16-marker layout.

    * 1 marker on each of front and back walls (centred horizontally).
    * 3 evenly-spaced markers on each of left and right walls.
    * Repeated at ``top_z_m`` (IDs 1-8) and ``bottom_z_m`` (IDs 9-16).

    Clockwise traversal looking down at the floor: front (1) -> right (3)
    -> back (1) -> left (3). At each height that gives 8 markers.
    """
    half_w = width_m / 2.0
    half_d = depth_m / 2.0
    quarter_d = depth_m / 4.0

    # (wall, x, y) tuples for one z-layer, clockwise starting at front.
    layer = [
        ("front",  0.0,        +half_d),     # 1 / 9
        ("right", +half_w,     +quarter_d),  # 2 / 10
        ("right", +half_w,      0.0),        # 3 / 11
        ("right", +half_w,     -quarter_d),  # 4 / 12
        ("back",   0.0,        -half_d),     # 5 / 13
        ("left",  -half_w,     -quarter_d),  # 6 / 14
        ("left",  -half_w,      0.0),        # 7 / 15
        ("left",  -half_w,     +quarter_d),  # 8 / 16
    ]

    markers: Dict[int, ArenaMarker] = {}
    for layer_idx, z in enumerate((top_z_m, bottom_z_m)):
        id_offset = layer_idx * 8       # 0 for top, 8 for bottom
        height_label = "high" if layer_idx == 0 else "low"
        for k, (wall, x, y) in enumerate(layer):
            mid = id_offset + k + 1
            label = f"{wall.capitalize()} {height_label} #{mid}"
            markers[mid] = ArenaMarker(
                id=mid, label=label, wall=wall,
                position_m=np.array([x, y, z], dtype=float))
    return ArenaConfig(marker_size_m=marker_size_m, markers=markers,
                       width_m=width_m, depth_m=depth_m,
                       top_z_m=top_z_m, bottom_z_m=bottom_z_m)


# ---------------------------------------------------------------------------
# Active-config persistence
# ---------------------------------------------------------------------------

def load_priority_arena(active_path: Optional[Path] = None
                        ) -> ArenaConfig:
    """Load the active arena config, falling back to the default layout.

    Priority:
      1. ``active_path`` (defaults to ``ACTIVE_ARENA_CONFIG_PATH`` from
         :mod:`marker_mission.config`) if it exists.
      2. ``default_arena()`` -- a ready-to-use 10 m x 25 m, 16-marker
         layout, in case the operator hasn't saved one yet.
    """
    if active_path is None:
        # Local import to avoid a circular dependency between
        # arena.py and config.py at module load time.
        from .config import ACTIVE_ARENA_CONFIG_PATH
        active_path = ACTIVE_ARENA_CONFIG_PATH
    p = Path(active_path)
    if p.is_file():
        try:
            return ArenaConfig.load(p)
        except Exception as e:
            print(f"[arena] failed to load {p}: {e}; using default")
    return default_arena()


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

    When ``pose.collapsed_camera_position_m`` is set the detector has
    already pre-computed the camera-in-marker-frame position as the
    midpoint of two IPPE mirror candidates (the near-frontal blind
    window). We use that value as ``C_m`` instead of re-deriving it
    from the single winning rvec/tvec, which would jump across the
    marker normal whenever IPPE picks the other branch.
    """
    if pose.collapsed_camera_position_m is not None:
        C_m = np.asarray(pose.collapsed_camera_position_m,
                         dtype=float).reshape(3)
    else:
        R_m_c, _ = cv2.Rodrigues(pose.rvec)
        t_m_c = np.asarray(pose.tvec, dtype=float).reshape(3)
        C_m = -R_m_c.T @ t_m_c
    R_m_w = WALL_ROTATIONS[marker.wall]
    return R_m_w @ C_m + marker.position_m


ARENA_BOUNDS_MARGIN_M = 1.0


def _vote_in_bounds(pos_w: np.ndarray, arena: ArenaConfig) -> bool:
    """True if ``pos_w`` lies inside the arena's bounding box plus
    ``ARENA_BOUNDS_MARGIN_M`` slack on every axis. The margin absorbs
    legitimate jitter and the small overshoot a drone can have just
    outside a wall, while still rejecting the IPPE-mirror votes that
    place the camera several metres outside the arena -- those are
    the votes that yank the weighted average across the room when
    they enter the visibility set.
    """
    half_w = float(arena.width_m) / 2.0 + ARENA_BOUNDS_MARGIN_M
    half_d = float(arena.depth_m) / 2.0 + ARENA_BOUNDS_MARGIN_M
    z_lo = float(arena.bottom_z_m) - ARENA_BOUNDS_MARGIN_M
    z_hi = float(arena.top_z_m) + ARENA_BOUNDS_MARGIN_M
    return (-half_w <= float(pos_w[0]) <= half_w
            and -half_d <= float(pos_w[1]) <= half_d
            and z_lo <= float(pos_w[2]) <= z_hi)


def estimate_position(arena: ArenaConfig,
                      poses: Iterable[MarkerPose]
                      ) -> Optional[PositionEstimate]:
    """Weighted average of camera world positions derived from each visible
    reference marker.

    Weight is currently ``1 / max(0.1, distance_m)`` -- close markers
    dominate. Returns ``None`` if no visible marker is in the arena
    config.

    Per-marker votes that fall outside the arena bounding box (plus
    ``ARENA_BOUNDS_MARGIN_M``) are discarded before the average. The
    drone is physically constrained to the arena so any vote that
    places it metres outside is the IPPE-mirror branch -- the
    detector picked the wrong planar-ambiguity solution for that
    marker. Mixing it into the weighted average produces the
    "position jumps to (-1, 0, 0.9) every time marker 3 comes back
    into frame" behaviour observed on a stationary drone. If every
    vote is out-of-bounds we fall back to using all of them rather
    than returning None -- a stale wrong answer is better than no
    answer when something is genuinely off.
    """
    contributions: List[tuple] = []   # (mid, pos_w, weight, method)
    for p in poses:
        marker = arena.markers.get(int(p.marker_id))
        if marker is None:
            continue
        pos_w = position_from_marker(p, marker)
        method = str(p.pose_method or "")
        # When the chosen IPPE branch lands outside the arena AND the
        # detector kept the loser branch on the pose, try swapping
        # to the loser. At significant off-normal angle the two
        # IPPE candidates' reproj-errs can sit within sub-pixel of
        # each other and IPPE picks wrong on noise; the right
        # branch is the loser. A successful swap recovers the
        # marker's contribution to the average instead of just
        # dropping it (the OOB filter below would otherwise discard
        # it). The mirror-collapse path takes precedence: if it
        # already overrode the camera position via
        # collapsed_camera_position_m, we don't second-guess it.
        if (p.alt_camera_position_m is not None
                and p.collapsed_camera_position_m is None
                and not _vote_in_bounds(pos_w, arena)):
            R_m_w = WALL_ROTATIONS[marker.wall]
            alt_pos_w = (R_m_w @ np.asarray(p.alt_camera_position_m,
                                            dtype=float).reshape(3)
                         + marker.position_m)
            if _vote_in_bounds(alt_pos_w, arena):
                pos_w = alt_pos_w
                method = "ippe_swapped"
        weight = 1.0 / max(0.1, float(p.distance_m))
        contributions.append(
            (int(p.marker_id), pos_w, weight, method))

    if not contributions:
        return None

    in_bounds = [c for c in contributions if _vote_in_bounds(c[1], arena)]
    if in_bounds:
        contributions = in_bounds
    # else: every vote is out-of-bounds -- keep them all rather than
    # silently dropping the estimate; the operator at least sees a
    # nonsensical value and can diagnose, instead of losing the fix
    # entirely.

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
