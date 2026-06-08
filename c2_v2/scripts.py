"""Never-land role scripts for C2 V2.

The FC safety-LANDS whenever a mission script COMPLETES, and ``/api/stop`` lands
too. So to keep a drone airborne for the whole match (operator requirement: only
the '?' emergency-land ever puts a drone down), every role must run an
effectively-INFINITE script that never reaches "mission complete":

  * scout    — centre, then rotate in place for ~1 h (one RC yaw drive).
  * defender — park in front of a side marker near our boxes, slow-rotate ~1 h
               (presence blocks enemy capture; the slow rotation keeps it
               "actively flying", not a sitting dead duck).
  * attacker — a long CHAIN of capture legs over the enemy boxes (capture →
               return home to score → turn → next), repeated far more times
               than a 10-min match allows, so it loops without ever landing.

All movement is VISION-based (APPROACH / GO_HOME home on ArUco markers; no
absolute coordinates), reusing the operator's PROVEN capture geometry:

    APPROACH <face> 1.50 0.20      # stop ~1.5 m short, ±20 cm band
    FB_UD_IMU 1.50 0.50            # slide the full standoff fwd while rising 0.5 m
    YAW_IMU 180                    # turn to face home
    GO_HOME <wall> 3.50 0.5 <hdg>  # vision-home shallow inside our home zone

These constants are the values the operator validated by hand in
``marker_mission`` (see marker_mission_c2/strategy/attacker.py) — copied here so
c2_v2 doesn't import V1's role machinery.
"""
from __future__ import annotations

from typing import List, Optional

# Pure helper (no import side effects).
from marker_mission_c2.strategy.settings import face_id

# ── PROVEN capture geometry (operator-validated) ────────────────────────────
ATTACK_STANDOFF_M = 1.50     # APPROACH stop distance from the box face
ATTACK_DIST_TOL_M = 0.20     # arrival band (fast, not fussy)
OVER_BOX_FORWARD_M = 1.50    # FB_UD_IMU forward (= full standoff -> over the box)
CAPTURE_RISE_M = 0.50        # FB_UD_IMU rise (into the 1-2 m RFID band)
RTH_WALL_STANDOFF_M = 3.50   # GO_HOME standoff -> shallow inside home zone
RTH_ARRIVE_TOL_M = 0.50      # GO_HOME loose arrival band

# ── Altitudes (deconflicted: scout low, movers above it, ≥30 cm apart) ──────
SCOUT_ALT_M = 1.0            # operator: scout ~1 m
ABOVE_SCOUT_ALT_M = 1.30     # mover cruise floor (above the scout)
MOVER_ALT_STEP_M = 0.30      # ≥30 cm between movers
MAX_MOVER_ALT_M = 2.50

# ── Scout / defender rotation ───────────────────────────────────────────────
SCOUT_YAW_STICK = 70         # ~105 deg/s; slowed when detection is poor (Phase 4)
DEFENDER_YAW_STICK = 35      # slow scan so boxes are read cleanly; not a dead duck
ROTATE_FOREVER_S = 3600      # one RC yaw drive that never completes
CENTER_STANDOFF_M = 5.0      # GO_HOME 5 m from a side marker -> arena centre
CENTER_TOL_M = 0.6
DEFENDER_SIDE_STANDOFF_M = 2.5
DEFENDER_SIDE_TOL_M = 0.5

# Centre side-wall markers (y≈0), present in both real + gvz arenas.
CENTER_MARKERS = (11, 15)

# How many capture passes to chain per attacker push. Far more than a 10-min
# match can fly, so the script never ends -> the drone never lands.
ATTACK_CHAIN_PASSES = 30


def _fmt(lines: List[str]) -> str:
    return "\n".join(l for l in lines if l) + "\n"


def mover_alt(lane: int) -> float:
    """Distinct cruise altitude for mover ``lane`` (0-based), ≥30 cm apart and
    above the scout, capped."""
    return min(MAX_MOVER_ALT_M, ABOVE_SCOUT_ALT_M + MOVER_ALT_STEP_M * max(0, lane))


# ── Scout ───────────────────────────────────────────────────────────────────
def scout_script(center_marker: int, alt_m: float = SCOUT_ALT_M,
                 yaw_stick: int = SCOUT_YAW_STICK) -> str:
    """Fly to the arena centre via a side marker, then rotate in place forever.

    GO_HOME (loose, holds altitude) onto the centre marker -> ~arena centre at
    the set altitude; then one long RC yaw drive scans every box. Never lands."""
    return _fmt([
        "TAKEOFF",
        f"HEIGHT {alt_m:.2f}",
        f"GO_HOME {int(center_marker)} {CENTER_STANDOFF_M:.2f} {CENTER_TOL_M:g} 0",
        f"HEIGHT {alt_m:.2f}",
        f"RC 0 0 0 {int(yaw_stick)} {ROTATE_FOREVER_S}",
    ])


# ── Defender ────────────────────────────────────────────────────────────────
def defender_script(side_marker: int, alt_m: float) -> str:
    """Park in front of a side-wall marker near our home boxes and slow-rotate
    forever — presence blocks enemy capture; the rotation keeps it active."""
    return _fmt([
        "TAKEOFF",
        f"HEIGHT {alt_m:.2f}",
        f"GO_HOME {int(side_marker)} {DEFENDER_SIDE_STANDOFF_M:.2f} {DEFENDER_SIDE_TOL_M:g} 0",
        f"RC 0 0 0 {int(DEFENDER_YAW_STICK)} {ROTATE_FOREVER_S}",
    ])


# ── Attacker ────────────────────────────────────────────────────────────────
def attack_leg(face_id_: int, wall_marker: int, cruise_alt_m: float,
               rth_hdg_deg: float) -> List[str]:
    """One capture-and-return leg (no TAKEOFF). Cruise high -> APPROACH the box
    face -> slide up+over to flip -> turn -> cruise high -> GO_HOME our wall.
    After GO_HOME the box point scores; the next leg's APPROACH re-finds the
    next enemy face, so legs chain with NO landing in between."""
    cruise = f"HEIGHT {cruise_alt_m:.2f}"
    return [
        cruise,
        f"APPROACH {int(face_id_)} {ATTACK_STANDOFF_M:.2f} {ATTACK_DIST_TOL_M:.2f}",
        f"FB_UD_IMU {OVER_BOX_FORWARD_M:.2f} {CAPTURE_RISE_M:.2f}",
        "YAW_IMU 180",
        cruise,
        f"GO_HOME {int(wall_marker)} {RTH_WALL_STANDOFF_M:.2f} {RTH_ARRIVE_TOL_M:g} {rth_hdg_deg:g}",
    ]


def attacker_loop_script(enemy_faces: List[int], wall_marker: int,
                         cruise_alt_m: float, rth_hdg_deg: float = 0.0,
                         passes: int = ATTACK_CHAIN_PASSES) -> str:
    """A long chain: ``passes`` loops over ``enemy_faces``, each a full
    capture-and-return leg. Never ends within a match -> never lands. The enemy
    boxes flip back to enemy colour when we leave, so re-attacking them stays
    productive (continuous offense)."""
    lines: List[str] = ["TAKEOFF"]
    for _ in range(max(1, int(passes))):
        for f in enemy_faces:
            lines.extend(attack_leg(int(f), wall_marker, cruise_alt_m, rth_hdg_deg))
    return _fmt(lines)


def enemy_home_faces(our_team: str) -> List[int]:
    """The ArUco faces shown by the ENEMY's home boxes at default — what an
    attacker APPROACHes. Red attacks blue's boxes 4-6 (faces 34/35/36); blue
    attacks red's boxes 1-3 (faces 41/42/43)."""
    enemy = "blue" if our_team == "red" else "red"
    enemy_slots = (4, 5, 6) if our_team == "red" else (1, 2, 3)
    return [face_id(s, enemy) for s in enemy_slots]
