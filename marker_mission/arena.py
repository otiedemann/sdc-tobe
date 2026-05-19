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
import math
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
DEFAULT_MARKER_SIZE_M = 0.5


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
    # Optional per-marker physical side length. When None the arena's
    # global ``marker_size_m`` is used. Use this for target markers
    # whose physical size differs from the wall markers (SDC26: walls
    # 50cm, targets 18cm); the aruco detector picks the right
    # objectPoints per detected id, so distance estimates stay correct
    # for mixed sizes in the same frame.
    size_m: Optional[float] = None


@dataclass
class ArenaConfig:
    """Loaded arena layout. ``marker_size_m`` is the operator-facing
    source of truth for marker geometry: ``mission.py`` syncs it into
    the detector every tick (``detector.set_marker_size``), so a /arena
    Save propagates to solvePnP without a restart. The ``cfg.marker_size_m``
    field on :class:`MissionConfig` is only consulted as a fallback when
    no arena is loaded — in production an arena always is, and the
    cfg value is effectively ignored. The video overlay reads
    ``detector.marker_size`` to stay consistent with this convention.

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
    # Arena-frame yaw (CW from front wall / +y) at which magnetic
    # north points. Used by the IPPE branch picker to disambiguate
    # planar-pose mirror flips: expected drone arena yaw at any tick
    # = ``tel.yaw_deg + magnetic_north_arena_yaw_deg``. None means
    # "uncalibrated" -- the picker silently falls back to
    # geometry-only logic. See vision_worker / estimate_position.
    magnetic_north_arena_yaw_deg: Optional[float] = None

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
        # Optional magnetic-north calibration. Old files don't have
        # this key -- legacy load returns None, picker stays disabled.
        mag_raw = data.get("magnetic_north_arena_yaw_deg", None)
        if mag_raw is None:
            magnetic_north_arena_yaw_deg = None
        else:
            try:
                m = float(mag_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"arena_config {source}: "
                    f"magnetic_north_arena_yaw_deg must be a number, "
                    f"got {mag_raw!r}")
            if not (-360.0 <= m <= 360.0):
                raise ValueError(
                    f"arena_config {source}: "
                    f"magnetic_north_arena_yaw_deg out of range "
                    f"[-360, 360]: {m}")
            magnetic_north_arena_yaw_deg = m
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
            size_raw = m.get("size_m")
            size_override: Optional[float] = None
            if size_raw is not None:
                try:
                    size_override = float(size_raw)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"arena_config {source}: marker {mid}.size_m must "
                        f"be a number, got {size_raw!r}")
                if not (0.01 <= size_override <= 5.0):
                    raise ValueError(
                        f"arena_config {source}: marker {mid}.size_m out "
                        f"of [0.01, 5.0]: {size_override}")
            markers[mid] = ArenaMarker(id=mid, label=label,
                                       position_m=pos, wall=wall,
                                       size_m=size_override)
        if not markers:
            raise ValueError(f"arena_config {source}: empty marker list")
        return cls(marker_size_m=marker_size_m, markers=markers,
                   width_m=width_m, depth_m=depth_m,
                   top_z_m=top_z_m, bottom_z_m=bottom_z_m,
                   magnetic_north_arena_yaw_deg=magnetic_north_arena_yaw_deg)

    def to_json_dict(self) -> dict:
        """Round-trippable JSON representation. The Arena tab uses this
        when writing the active config or a named save."""
        out: dict = {
            "marker_size_m": float(self.marker_size_m),
            "width_m": float(self.width_m),
            "depth_m": float(self.depth_m),
            "top_z_m": float(self.top_z_m),
            "bottom_z_m": float(self.bottom_z_m),
            "markers": [
                {
                    "id": int(m.id), "label": m.label, "wall": m.wall,
                    "x": float(m.position_m[0]),
                    "y": float(m.position_m[1]),
                    "z": float(m.position_m[2]),
                    # size_m is optional — only emitted when overridden.
                    **({"size_m": float(m.size_m)}
                       if m.size_m is not None else {}),
                }
                for m in sorted(self.markers.values(), key=lambda x: x.id)
            ],
        }
        # Only emit the magnetic offset when it's actually set, so a
        # round-tripped legacy arena config doesn't grow a noise key.
        if self.magnetic_north_arena_yaw_deg is not None:
            out["magnetic_north_arena_yaw_deg"] = float(
                self.magnetic_north_arena_yaw_deg)
        return out

    def __contains__(self, marker_id: int) -> bool:
        return marker_id in self.markers

    def size_for(self, marker_id: int) -> float:
        """Per-marker physical side length [m].

        Returns the marker's own ``size_m`` if it's overridden in the
        arena config, otherwise the arena's default ``marker_size_m``.
        Markers not in the arena fall back to the default too.
        """
        m = self.markers.get(int(marker_id))
        if m is not None and m.size_m is not None:
            return float(m.size_m)
        return float(self.marker_size_m)


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
      2. operator-pinned default-by-name (Arena tab, "Set as default").
      3. ``default_arena()`` -- a ready-to-use 10 m x 25 m, 16-marker
         layout, in case the operator hasn't saved one yet.

    Default-by-name sits between the active path (which the operator
    saves explicitly via the Arena tab) and the built-in default --
    same shape as ``mission_script.load_priority_script``.
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
            print(f"[arena] failed to load {p}: {e}; trying default-by-name")
    try:
        from .config import get_default, ARENA_CONFIGS_DIR
        name = get_default("arena")
        if name:
            q = ARENA_CONFIGS_DIR / f"{name}.json"
            if q.is_file():
                return ArenaConfig.load(q)
            else:
                print(f"[arena] default-by-name {name!r} not found;"
                      f" using built-in default")
    except Exception as e:
        print(f"[arena] default-by-name load failed: {e};"
              f" using built-in default")
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
    ``ARENA_BOUNDS_MARGIN_M`` slack on every axis.

    Floor reference is the arena origin (z = 0), NOT
    ``arena.bottom_z_m`` -- that field is the lowest reference
    marker's z (typically ~0.94 m), so the drone can legitimately
    fly below it. We use a symmetric margin around z=0 / top_z_m
    instead, otherwise a stationary drone near the floor reads
    "out of bounds" and the branch selector rejects perfectly good
    IPPE votes.
    """
    half_w = float(arena.width_m) / 2.0 + ARENA_BOUNDS_MARGIN_M
    half_d = float(arena.depth_m) / 2.0 + ARENA_BOUNDS_MARGIN_M
    z_lo = -ARENA_BOUNDS_MARGIN_M
    z_hi = float(arena.top_z_m) + ARENA_BOUNDS_MARGIN_M
    return (-half_w <= float(pos_w[0]) <= half_w
            and -half_d <= float(pos_w[1]) <= half_d
            and z_lo <= float(pos_w[2]) <= z_hi)


# Per-frame jump cap when anchoring to the previous fix. The Anafi
# caps near 12 m/s; at 25 fps that's < 0.5 m / frame, so 1.0 m gives
# us 2x slack for low-fps moments without admitting metre-scale
# IPPE-mirror flips.
ARENA_MAX_STEP_M = 1.0
# Beyond this age the previous position is too stale to anchor
# branch selection -- fall back to in-bounds filtering.
ARENA_PREV_STALE_S = 2.0


def _alt_world_position(pose: 'MarkerPose',
                         marker: 'ArenaMarker') -> Optional[np.ndarray]:
    if pose.alt_camera_position_m is None:
        return None
    R_m_w = WALL_ROTATIONS[marker.wall]
    return R_m_w @ np.asarray(pose.alt_camera_position_m,
                              dtype=float).reshape(3) + marker.position_m


# Magnetometer slack: how far a candidate's implied arena yaw may
# disagree from ``tel.yaw + magnetic_north_arena_yaw_deg`` and still
# count as a match. 12 deg covers magnetometer drift over a 60s flight
# (~1-2 deg) plus the sub-degree yaw_to_marker approximation across
# IPPE branches plus a few deg of corner-detector jitter. Above 12 deg
# both branches "look wrong" and we fall back to geometry.
ARENA_MAG_SLACK_DEG = 12.0


def _wrap_180(deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    return ((float(deg) + 540.0) % 360.0) - 180.0


def _yaw_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference, in degrees."""
    return abs(_wrap_180(a - b))


def _arena_yaw_for_branch(pose: 'MarkerPose',
                           marker: 'ArenaMarker',
                           branch: str) -> Optional[float]:
    """Drone arena yaw implied by ``pose``'s chosen-or-alt IPPE
    branch. Used by the magnetometer-aware branch picker in both
    ``estimate_position`` (per-marker world-position) and
    ``vision_worker`` (active-marker relative_heading_deg).

    The math: arena_yaw = bearing(marker - drone_world) - yaw_to_marker,
    where bearing is CW from arena +y and ``yaw_to_marker`` (camera-frame
    angle to the marker) is approximated as branch-invariant -- the
    marker tvec is nearly the same for both IPPE candidates because
    they differ in *rotation* of a marker whose centre projects to
    the same image pixel. Sub-degree error vs the 12 deg
    ARENA_MAG_SLACK_DEG; safe.

    Returns None when ``branch == 'alt'`` and ``pose`` doesn't carry
    an alt camera position (e.g. the mirror-collapse path nulled it
    or only one IPPE candidate survived).
    """
    if branch == "chosen":
        drone_world = position_from_marker(pose, marker)
    elif branch == "alt":
        drone_world = _alt_world_position(pose, marker)
        if drone_world is None:
            return None
    else:
        raise ValueError(f"unknown branch {branch!r}")
    dx = float(marker.position_m[0]) - float(drone_world[0])
    dy = float(marker.position_m[1]) - float(drone_world[1])
    if dx == 0.0 and dy == 0.0:
        return None
    bearing_deg = math.degrees(math.atan2(dx, dy))
    return _wrap_180(bearing_deg - float(pose.yaw_deg))


def estimate_position(arena: ArenaConfig,
                      poses: Iterable[MarkerPose],
                      prev_position_m: Optional[np.ndarray] = None,
                      prev_age_s: Optional[float] = None,
                      tel_yaw_deg: Optional[float] = None,
                      tel_height_m: Optional[float] = None,
                      enable_arena_oob_filter: bool = True,
                      enable_alt_branch_swap: bool = True,
                      enable_prev_anchor: bool = False,
                      enable_aggregate_oob_discard: bool = True,
                      ) -> Optional[PositionEstimate]:
    """Weighted average of camera world positions derived from each visible
    reference marker.

    Weight is currently ``1 / max(0.1, distance_m)`` -- close markers
    dominate. Returns ``None`` if no visible marker contributes.

    Branch selection per marker:

    1. **Anchored** (``prev_position_m`` is fresh, age <
       ``ARENA_PREV_STALE_S``): for each marker we score both IPPE
       branches against the previous fix and pick whichever is closer,
       provided it's within ``ARENA_MAX_STEP_M`` of the previous fix.
       If neither branch lands within that window, the marker is
       skipped entirely -- IPPE produced two unreliable poses on this
       frame and a wrong vote that's "barely in bounds" is what
       caused the displayed position to jump every few frames on a
       stationary drone with marker 3 visible.
    2. **Unanchored** (no fresh previous fix): fall back to the
       original arena-bounds filter. Drop chosen votes that fall
       outside the arena box (plus ``ARENA_BOUNDS_MARGIN_M`` slack);
       try the alt branch when chosen is OOB. Drop the marker if
       both branches are OOB.

    If every marker is skipped/dropped we return ``None`` so the
    caller's sticky ``state.world_position_m`` is preserved -- the
    display dot doesn't jump to a clearly-wrong location on a single
    bad detection frame.
    """
    # Use the previous fix as a branch-selection anchor only when it
    # is recent AND itself in-bounds. An OOB prev is a wrong-branch
    # IPPE result that snuck through cold-start; trusting it would
    # lock the estimator into the wrong branch indefinitely (every
    # frame the wrong-but-near-prev chosen wins, the right
    # well-in-bounds alt gets rejected as "too far"). Falling back
    # to the cold-start (in-bounds) path lets the swap recover.
    prev_arr = (np.asarray(prev_position_m, dtype=float).reshape(3)
                if prev_position_m is not None else None)
    use_anchor = (enable_prev_anchor
                  and prev_arr is not None
                  and prev_age_s is not None
                  and prev_age_s < ARENA_PREV_STALE_S
                  and _vote_in_bounds(prev_arr, arena))

    # Magnetometer-aware branch picker. Active when the operator has
    # calibrated the arena's magnetic-north offset AND tel.yaw is
    # available. Short-circuits the prev-anchor / OOB blocks because
    # the magnetometer is an independent signal that doesn't share
    # corner-noise / pixel-size failure modes.
    use_mag = (arena.magnetic_north_arena_yaw_deg is not None
               and tel_yaw_deg is not None)
    expected_arena_yaw = (
        _wrap_180(float(tel_yaw_deg)
                  + float(arena.magnetic_north_arena_yaw_deg))
        if use_mag else None)

    contributions: List[tuple] = []   # (mid, pos_w, weight, method)
    for p in poses:
        marker = arena.markers.get(int(p.marker_id))
        if marker is None:
            continue
        chosen_pos = position_from_marker(p, marker)
        # collapsed camera position takes precedence: the
        # mirror-collapse path already produced a midpoint we should
        # trust, no branch swap.
        if p.collapsed_camera_position_m is not None:
            alt_pos = None
        else:
            alt_pos = _alt_world_position(p, marker)
        chosen_method = str(p.pose_method or "")
        weight = 1.0 / max(0.1, float(p.distance_m))

        pos_w = None
        method = chosen_method
        alt_validated = False  # set when altimeter agreed within tolerance

        # Layer -1: altimeter-aware Z disambiguator. The barometer-derived
        # ``tel.height_cm`` is the most reliable single source we have for
        # the drone's actual altitude. Three cases:
        #
        #   (a) both IPPE branches exist, only one Z matches altimeter →
        #       pick the matching one (clear winner)
        #   (b) only chosen branch exists, Z matches altimeter → trust
        #       chosen, mark alt_validated (lets it through the
        #       cold-start quorum gate)
        #   (c) chosen exists, Z doesn't match → no validation, fall
        #       through to mag/anchor/OOB layers
        #
        # ALT_TOL_M=0.5 absorbs the ~10–20 cm Anafi altimeter drift while
        # still rejecting metre-scale mirror flips.
        if (tel_height_m is not None
                and p.collapsed_camera_position_m is None
                and tel_height_m > 0.05):
            ALT_TOL_M = 0.5
            chosen_z_err = abs(float(chosen_pos[2]) - float(tel_height_m))
            if alt_pos is not None:
                alt_z_err = abs(float(alt_pos[2]) - float(tel_height_m))
                if chosen_z_err <= ALT_TOL_M and alt_z_err > ALT_TOL_M:
                    pos_w = chosen_pos
                    method = "ippe_alt_z"
                    alt_validated = True
                elif alt_z_err <= ALT_TOL_M and chosen_z_err > ALT_TOL_M:
                    pos_w = alt_pos
                    method = "ippe_alt_z_swap"
                    alt_validated = True
                elif chosen_z_err <= ALT_TOL_M and alt_z_err <= ALT_TOL_M:
                    # both match — chosen wins by default, no flag
                    pos_w = chosen_pos
                    method = chosen_method
            else:
                # Single-branch IPPE result (mirror was back-facing or
                # numerically degenerate). Trust the chosen branch IFF
                # it matches the altimeter. If it doesn't, leave pos_w
                # None so the cold-start quorum rejects it.
                if chosen_z_err <= ALT_TOL_M:
                    pos_w = chosen_pos
                    method = "ippe_alt_z_single"
                    alt_validated = True

        # Layer 0: magnetometer pick. Only fires when both branches
        # exist (otherwise there's nothing to disambiguate against)
        # and at least one of them lands within ARENA_MAG_SLACK_DEG
        # of the magnetometer-expected arena yaw. Skipped when the
        # altimeter layer already picked — it's the more reliable
        # signal when available.
        if (pos_w is None
                and use_mag and alt_pos is not None
                and p.collapsed_camera_position_m is None):
            chosen_yaw = _arena_yaw_for_branch(p, marker, "chosen")
            alt_yaw    = _arena_yaw_for_branch(p, marker, "alt")
            if chosen_yaw is not None and alt_yaw is not None:
                d_chosen = _yaw_diff(chosen_yaw, expected_arena_yaw)
                d_alt    = _yaw_diff(alt_yaw,    expected_arena_yaw)
                if min(d_chosen, d_alt) <= ARENA_MAG_SLACK_DEG:
                    if d_alt < d_chosen:
                        pos_w = alt_pos
                        method = "ippe_mag_swap"
                    else:
                        pos_w = chosen_pos
                        method = chosen_method
                # else: both > slack -- magnetic field disturbance
                # or stale offset; fall through to the anchor / OOB
                # blocks rather than picking a confidently-wrong
                # branch.

        if pos_w is not None:
            # Magnetometer (or altimeter) made the call -- skip the
            # anchor / OOB gates. Final aggregate OOB check still
            # applies below.
            contributions.append((int(p.marker_id), pos_w, weight, method,
                                  alt_validated))
            continue

        if use_anchor:
            d_chosen = float(np.linalg.norm(chosen_pos - prev_arr))
            d_alt = (float(np.linalg.norm(alt_pos - prev_arr))
                     if alt_pos is not None else None)
            chosen_ok = d_chosen <= ARENA_MAX_STEP_M
            alt_ok = (d_alt is not None and d_alt <= ARENA_MAX_STEP_M)
            if chosen_ok and (not alt_ok or d_chosen <= d_alt):
                pos_w = chosen_pos
            elif alt_ok:
                pos_w = alt_pos
                method = "ippe_swapped"
            # else: both branches are too far from the previous fix
            # to be plausible; skip this marker so its bad pose
            # doesn't pull the average.
        else:
            # Cold-start path. enable_arena_oob_filter gates the
            # in-bounds check on the chosen vote; without it, the
            # chosen vote is taken regardless of where it lands (the
            # pre-2026-05-04 behaviour). enable_alt_branch_swap
            # toggles whether we rescue OOB chosen by trying alt.
            chosen_in = _vote_in_bounds(chosen_pos, arena)
            if chosen_in or not enable_arena_oob_filter:
                pos_w = chosen_pos
            elif (enable_alt_branch_swap and alt_pos is not None
                    and _vote_in_bounds(alt_pos, arena)):
                pos_w = alt_pos
                method = "ippe_swapped"
            # else: both OOB; skip.

        # Per-marker bounds revision: even when a vote passed the
        # anchor gate, an OOB result is still physically impossible.
        # Try the alt branch as a second-chance fix; if both branches
        # are OOB, drop the marker. The drone can't be outside the
        # arena, no matter how close to prev the IPPE pose looks.
        # Gated by enable_aggregate_oob_discard (same toggle as the
        # final-average discard below -- both are layer-5 sanity).
        if (enable_aggregate_oob_discard
                and pos_w is not None
                and not _vote_in_bounds(pos_w, arena)):
            if (method != "ippe_swapped" and alt_pos is not None
                    and _vote_in_bounds(alt_pos, arena)):
                pos_w = alt_pos
                method = "ippe_swapped"
            else:
                pos_w = None

        if pos_w is not None:
            contributions.append((int(p.marker_id), pos_w, weight, method,
                                  alt_validated))

    if not contributions:
        return None

    # Cold-start single-marker reject. With only one wall marker visible
    # and no fresh anchor to disambiguate against, the IPPE planar-pose
    # mirror branch can be confidently inside the arena AND wrong. EXCEPT
    # when the altimeter layer (above) disambiguated the branch from the
    # drone's barometric height — that's a robust independent signal
    # that lets single-marker fixes through safely.
    if ((not use_anchor)
            and len(contributions) < 2
            and not any(av for _, _, _, _, av in contributions)):
        return None

    total_w = sum(w for _, _, w, _, _ in contributions)
    weighted_sum = np.zeros(3, dtype=float)
    for _, pos_w, w, _, _ in contributions:
        weighted_sum += pos_w * w
    avg = weighted_sum / total_w

    # Final aggregate sanity gate: even with all per-marker votes
    # in-bounds, the weighted average can sit just outside the
    # arena box if the contributions hug the boundary asymmetrically.
    # Discard the whole estimate in that case so the caller's sticky
    # state.world_position_m is preserved -- a fix that places the
    # drone outside the arena is by definition wrong, no matter how
    # confident the per-marker votes look.
    if enable_aggregate_oob_discard and not _vote_in_bounds(avg, arena):
        return None

    return PositionEstimate(
        position_m=avg,
        used_markers=[mid for mid, _, _, _, _ in contributions],
        per_marker_position_m={mid: pos for mid, pos, _, _, _ in contributions},
        weights={mid: w / total_w for mid, _, w, _, _ in contributions},
        per_marker_method={mid: meth for mid, _, _, meth, _ in contributions},
    )
