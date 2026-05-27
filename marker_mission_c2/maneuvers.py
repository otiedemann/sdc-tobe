"""Build attack maneuvers from the *existing* FC mission-script commands.

The C2 pushes operator-triggered attack runs to a flight controller as a
plain mission script the FC already understands — NO special AUTO_ATTACK
phase. Positioning is assistance only: ``APPROACH`` (drift-free vision
bearing/distance) drives every precise stop; ``TO`` is coarse
pre-positioning that the FC self-protects (it falls back to a yaw-search
if the world fix goes stale, so it can't run open-loop on a dead estimate).

Generated script shape::

    TAKEOFF
    HEIGHT   <cruise_alt>          # climb to a camera-level altitude first
    TO       <tx> <ty>             # (assist) coarse move toward the slot's
                                   #   approx arena position — only bites when
                                   #   the target marker isn't visible yet
    APPROACH <enemy_face> <standoff>   # VISION: home onto the box face
    HEIGHT   <over_alt>            # rise to clear the box top
    FB_RC    <stick> <secs>        # drift forward to sit OVER the box ("drop")
    HOOVER   <dwell>               # dwell over the box
    YAW_IMU  180                   # turn toward home
    TO       <hx> <hy>             # (assist) coarse cruise back toward home
    APPROACH <home_marker> <home_standoff>   # VISION: home onto the home wall
    YAW_IMU  180                   # face the enemy again, ready for next run
    LAND                          # (optional — omit to hover, ready)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Arena slot layout — per SDC26 v7 regs §1.4.2 Table 1 + Figure 1.
# Each physical target carries two ArUco faces: blue = 30 + slot,
# red = 40 + slot. Long axis is Y (20 m), width is X (10 m).
# Boxes 1-3 sit in the RED home zone at y = -7.5 (default markers
# 41/42/43); boxes 4-6 in the BLUE home zone at y = +7.5 (default
# markers 34/35/36); spread across x in {-3, 0, +3}. Mirrors the
# canonical layout used by strategy/attacker.py + regs §1.1 home-zone
# bounds (in_home_zone in strategy/roles.py).
SLOT_POSITIONS_M: dict[int, tuple[float, float]] = {
    1: (-3.0, -7.5), 2: (0.0, -7.5), 3: (3.0, -7.5),   # RED zone
    4: (-3.0,  7.5), 5: (0.0,  7.5), 6: (3.0,  7.5),   # BLUE zone
}

# Our home-wall marker — the marker the drone APPROACHes on the way back —
# per team: team -> (marker_id, (arena_x, arena_y)).
HOME_WALL_MARKER: dict[str, tuple[int, tuple[float, float]]] = {
    "red": (13, (0.0, -10.0)),
    "blue": (9, (0.0, 10.0)),
}

VALID_SLOTS: tuple[int, ...] = tuple(SLOT_POSITIONS_M.keys())
VALID_TEAMS: tuple[str, ...] = tuple(HOME_WALL_MARKER.keys())


def enemy_face_id(slot: int, our_team: str) -> int:
    """ArUco face we attack = the *enemy's* face on the slot.

    blue faces are 30 + slot, red faces are 40 + slot. If we are red the
    enemy is blue (30 + slot); if we are blue the enemy is red (40 + slot).
    """
    enemy = "blue" if str(our_team).lower() == "red" else "red"
    base = 30 if enemy == "blue" else 40
    return base + int(slot)


@dataclass
class ManeuverParams:
    """Tunable knobs for the attack maneuver (all have safe sim defaults)."""
    cruise_alt_m: float = 1.5       # climb-first / approach altitude (clears boxes)
    over_alt_m: float = 1.8         # rise to clear the box top before the drop
    target_standoff_m: float = 1.2  # APPROACH stop distance from the box face
    home_standoff_m: float = 2.0    # APPROACH stop distance from the home wall
    over_push_rc: int = 50          # forward stick for the drift-over-box
    over_push_s: float = 0.8        # drift-over-box duration (s)
    dwell_s: float = 3.0            # hover over the box (s); 0 to skip
    use_to_assist: bool = True      # emit TO coarse pre-positioning (assist)
    land_at_end: bool = True        # LAND vs leave hovering, ready for next run


def build_attack_script(
    slot: int,
    our_team: str,
    params: Optional[ManeuverParams] = None,
) -> str:
    """Return a mission script that attacks ``slot`` and returns home,
    built entirely from existing FC commands.

    Raises ``ValueError`` on an unknown slot or team.
    """
    p = params or ManeuverParams()
    slot = int(slot)
    our_team = str(our_team).lower()
    if slot not in SLOT_POSITIONS_M:
        raise ValueError(f"slot must be one of {VALID_SLOTS}, got {slot}")
    if our_team not in HOME_WALL_MARKER:
        raise ValueError(f"team must be red|blue, got {our_team!r}")

    target_face = enemy_face_id(slot, our_team)
    tx, ty = SLOT_POSITIONS_M[slot]
    home_marker, (hx, hy) = HOME_WALL_MARKER[our_team]

    lines: list[str] = ["TAKEOFF", f"HEIGHT {p.cruise_alt_m:g}"]
    # ── outbound: coarse TO (assist) → vision APPROACH (precise) ──
    if p.use_to_assist:
        lines.append(f"TO {tx:g} {ty:g}")
    lines.append(f"APPROACH {target_face} {p.target_standoff_m:g}")
    # ── over-the-box: rise to clear the top, then drift forward ──
    lines.append(f"HEIGHT {p.over_alt_m:g}")
    lines.append(f"FB_RC {int(p.over_push_rc)} {p.over_push_s:g}")
    if p.dwell_s > 0:
        lines.append(f"HOOVER {p.dwell_s:g}")
    # ── return: turn home, coarse TO (assist) → vision APPROACH ──
    lines.append("YAW_IMU 180")
    if p.use_to_assist:
        lines.append(f"TO {hx:g} {hy:g}")
    lines.append(f"APPROACH {home_marker} {p.home_standoff_m:g}")
    lines.append("YAW_IMU 180")   # face the enemy again, ready for next run
    if p.land_at_end:
        lines.append("LAND")
    return "\n".join(lines) + "\n"


def params_from_dict(d: Optional[dict]) -> ManeuverParams:
    """Build :class:`ManeuverParams` from a JSON body, ignoring unknown
    keys and falling back to defaults for missing/invalid values."""
    p = ManeuverParams()
    if not isinstance(d, dict):
        return p

    def _num(key: str, cur):
        if key not in d or d[key] is None:
            return cur
        try:
            return type(cur)(d[key])
        except (TypeError, ValueError):
            return cur

    p.cruise_alt_m = _num("cruise_alt_m", p.cruise_alt_m)
    p.over_alt_m = _num("over_alt_m", p.over_alt_m)
    p.target_standoff_m = _num("target_standoff_m", p.target_standoff_m)
    p.home_standoff_m = _num("home_standoff_m", p.home_standoff_m)
    p.over_push_rc = _num("over_push_rc", p.over_push_rc)
    p.over_push_s = _num("over_push_s", p.over_push_s)
    p.dwell_s = _num("dwell_s", p.dwell_s)
    if "use_to_assist" in d:
        p.use_to_assist = bool(d["use_to_assist"])
    if "land_at_end" in d:
        p.land_at_end = bool(d["land_at_end"])
    return p
