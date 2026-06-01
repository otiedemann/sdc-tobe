"""Runtime-active arena geometry for the strategy.

The strategy used to hardcode ``HOME_WALL_MARKER`` / ``SLOT_POSITIONS_M`` and
the home-zone bounds as module-level literals. To support a LIVE arena switch
(the operator toggles real<->gvz from the dashboard; the sim swaps its physical
arena and the strategy FOLLOWS), those facts now live here — derived from the
active arena config + target layout and swappable at runtime via
:func:`set_active`.

Source of truth = the named-arena registry ``marker_mission_c2/arena/arenas.json``
(the same file the sim loads), so the strategy's geometry can never drift from
the sim's physical world. Accessors are evaluated per script-dispatch, so a
swap takes effect on the very next mission the runner builds.

Arena geometry that actually varies between arenas: the wall markers (hence the
derived home back-wall markers) and the target-box positions. Tuning constants
(defender hold dwell, capture altitudes, ...) and team conventions
(``enemy_heading_deg``) are NOT arena geometry and stay where they are.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# repo root = .../<repo>/marker_mission_c2/strategy/arena_state.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "marker_mission_c2" / "arena" / "arenas.json"

# Regs §1.1: each home zone is the outer 5 m of the 20 m-long field.
HOME_DEPTH_M: float = 5.0

# Last-resort triangle (used only if a target layout is missing/unreadable).
_FALLBACK_SLOTS: dict[int, tuple[float, float]] = {
    1: (-3.0, -6.5), 2: (0.0, -9.0), 3: (3.0, -6.5),
    4: (-3.0, 6.5), 5: (0.0, 9.0), 6: (3.0, 6.5),
}


@dataclass(frozen=True)
class ArenaState:
    name: str
    width_m: float
    depth_m: float
    # {"red": (marker_id, (x, y)), "blue": (marker_id, (x, y))}
    home_wall_marker: dict[str, tuple[int, tuple[float, float]]]
    slot_positions_m: dict[int, tuple[float, float]]   # {slot: (x, y)}

    @property
    def half_width_m(self) -> float:
        return self.width_m / 2.0

    @property
    def red_home_y(self) -> tuple[float, float]:
        return (-self.depth_m / 2.0, -self.depth_m / 2.0 + HOME_DEPTH_M)

    @property
    def blue_home_y(self) -> tuple[float, float]:
        return (self.depth_m / 2.0 - HOME_DEPTH_M, self.depth_m / 2.0)


# ---------------------------------------------------------------------------
# Loading / derivation
# ---------------------------------------------------------------------------

def _read_registry() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except (OSError, ValueError):
        return {"default": "real", "arenas": {}}


def default_name() -> str:
    return str(_read_registry().get("default", "real"))


def arena_names() -> list[str]:
    return list(_read_registry().get("arenas", {}).keys())


def _slot_of_face(fid: int) -> Optional[int]:
    if 31 <= fid <= 36:
        return fid - 30
    if 41 <= fid <= 46:
        return fid - 40
    return None


def _derive_home_markers(arena) -> dict[str, tuple[int, tuple[float, float]]]:
    """Red's home back-wall marker = the ``back`` wall marker at flight-camera
    height (lowest z); blue's = the ``front`` wall marker (lowest z). Derived,
    so a future arena with different marker IDs Just Works."""
    out: dict[str, tuple[int, tuple[float, float]]] = {}
    for team, wall in (("red", "back"), ("blue", "front")):
        cands = [m for m in arena.markers.values() if str(m.wall) == wall]
        if not cands:
            continue
        m = min(cands, key=lambda mm: (float(mm.position_m[2]), int(mm.id)))
        out[team] = (int(m.id), (round(float(m.position_m[0]), 2),
                                 round(float(m.position_m[1]), 2)))
    return out


def _load_slot_positions(target_layout_path: Optional[str]) -> dict[int, tuple[float, float]]:
    if not target_layout_path:
        return dict(_FALLBACK_SLOTS)
    try:
        tl = json.loads((REPO_ROOT / target_layout_path).read_text())
    except (OSError, ValueError):
        return dict(_FALLBACK_SLOTS)
    slots: dict[int, tuple[float, float]] = {}
    for b in tl.get("boxes", []):
        if not b.get("enabled", True):
            continue
        try:
            s = _slot_of_face(int(b["id"]))
        except (TypeError, ValueError, KeyError):
            continue
        if s is not None:
            slots[s] = (round(float(b["x"]), 2), round(float(b["y"]), 2))
    return slots or dict(_FALLBACK_SLOTS)


def _load_state(name: str) -> ArenaState:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from marker_mission.arena import ArenaConfig, default_arena  # type: ignore

    entry = _read_registry().get("arenas", {}).get(name) or {}
    acp = entry.get("arena_config_path")
    arena = (ArenaConfig.load(REPO_ROOT / acp) if acp else default_arena())
    return ArenaState(
        name=name,
        width_m=float(arena.width_m),
        depth_m=float(arena.depth_m),
        home_wall_marker=_derive_home_markers(arena),
        slot_positions_m=_load_slot_positions(entry.get("target_layout_path")),
    )


# ---------------------------------------------------------------------------
# Current active state (thread-safe) + accessors
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_current: Optional[ArenaState] = None


def current() -> ArenaState:
    global _current
    with _lock:
        if _current is None:
            _current = _load_state(default_name())
        return _current


def set_active(name: str) -> ArenaState:
    """Re-point the strategy at a registered arena. Returns the new state."""
    global _current
    with _lock:
        _current = _load_state(name)
        return _current


def active_name() -> str:
    return current().name


def home_wall_marker(team: str) -> Optional[tuple[int, tuple[float, float]]]:
    return current().home_wall_marker.get(team)


def slot_positions() -> dict[int, tuple[float, float]]:
    return dict(current().slot_positions_m)


def in_home_zone(team: str, x: float, y: float) -> bool:
    st = current()
    if abs(x) > st.half_width_m:
        return False
    if team == "red":
        lo, hi = st.red_home_y
        return lo <= y <= hi
    if team == "blue":
        lo, hi = st.blue_home_y
        return lo <= y <= hi
    return False


def home_park_xy(team: str, lane: int = 0, n_lanes: int = 1) -> tuple[float, float]:
    """Shallow bank-park point: 0.5 m inside the inner home boundary (in FRONT
    of the boxes, not over them), x spread across the home-zone width by lane."""
    st = current()
    y = (st.red_home_y[1] - 0.5) if team == "red" else (st.blue_home_y[0] + 0.5)
    if n_lanes <= 1:
        return (0.0, y)
    span = min(7.0, st.width_m - 3.0)
    x = -span / 2.0 + span * (lane / (n_lanes - 1))
    return (round(x, 2), y)


def neutral_wait_y() -> float:
    """|y| of the defender's neutral hold station = 0.5 m INTO neutral from the
    inner home boundary (e.g. 5.0 - 0.5 = 4.5 for a 20 m field)."""
    return abs(current().red_home_y[1]) - 0.5


def arena_dims() -> tuple[float, float]:
    st = current()
    return (st.width_m, st.depth_m)


def to_client_dict() -> dict:
    """Geometry the dashboard canvas renders from (replaces its hardcoded
    ARENA/BOXES constants)."""
    st = current()
    return {
        "name": st.name,
        "x_min": -st.half_width_m, "x_max": st.half_width_m,
        "y_min": -st.depth_m / 2.0, "y_max": st.depth_m / 2.0,
        "red_home_y": list(st.red_home_y),
        "blue_home_y": list(st.blue_home_y),
        "boxes": {str(s): [x, y] for s, (x, y) in st.slot_positions_m.items()},
        "home_markers": {t: v[0] for t, v in st.home_wall_marker.items()},
        "available": arena_names(),
    }
