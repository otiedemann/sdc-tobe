"""Never-land role scripts for C2 V2.

The FC safety-LANDS whenever a mission script COMPLETES, and ``/api/stop`` lands
too. So to keep a drone airborne for the whole match (operator requirement: only
the '?' emergency-land ever puts a drone down), every role must run an
effectively-INFINITE script that never reaches "mission complete":

  * attacker — FIXED LANE: a never-land loop on ONE enemy box (fly the straight
               home<->box lane, capture it whenever the enemy face is up, hold it
               by presence). 3 attackers on 3 distinct boxes => collision-safe
               spatially-separated lanes (plus the altitude ladder + RTH fan-out).
  * defender — wait JUST OUTSIDE our home in the NEUTRAL zone, slow-rotate to
               scan our 3 boxes; the runner splices in a recapture (no landing)
               the instant one of our boxes flips, then it returns to the scan.
  * scout    — (optional) centre + rotate ~1 h. The simplified strategy runs
               SCOUT-LESS (3 attackers + 2 defenders do their own scouting).

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

# ── Altitudes (deconflicted: launch low ~1 m; never below the box top) ──────
SCOUT_ALT_M = 1.0            # operator: scout ~1 m
ABOVE_SCOUT_ALT_M = 1.30     # mover cruise floor (above the scout)
MOVER_ALT_STEP_M = 0.30      # ≥30 cm between movers
BOX_TOP_M = 0.73             # hard floor — never command below the 0.73 m box top
MAX_MOVER_ALT_M = 2.50

# ── Scout / defender rotation ───────────────────────────────────────────────
SCOUT_YAW_STICK = 70         # ~105 deg/s; slowed when detection is poor (Phase 4)
DEFENDER_YAW_STICK = 35      # slow scan so boxes are read cleanly; not a dead duck
ROTATE_FOREVER_S = 3600      # one RC yaw drive that never completes
CENTER_STANDOFF_M = 5.0      # GO_HOME 5 m from a side marker -> arena centre
CENTER_TOL_M = 0.6
DEFENDER_SIDE_STANDOFF_M = 2.5
DEFENDER_SIDE_TOL_M = 0.5

# A never-land HOVER/SCAN must be a CHAIN of SHORT RC steps, not one long
# ``RC .. 3600``. Reason: a mission SPLICE (live re-target, no land) KEEPS the
# currently-executing step and appends the new tail; the FC only walks into that
# tail once the current step COMPLETES. A single 3600 s RC step would make the FC
# sit out the whole timer (~1 h) before a spliced attack/recapture ever runs — so
# the attacker would never strike within a match. Short steps complete every few
# seconds, so a splice engages within ~RC_HOLD_TICK_S. The chain still outlasts a
# match by a wide margin (matches the old single-step 3600 s horizon) so the drone
# never lands on its own — if it were exhausted the FC safety-lands, so keep the
# total well above any match incl. overtime/pauses.
RC_HOLD_TICK_S = 5
RC_HOLD_REPEATS = 720        # 720 x 5 s = 3600 s (~60 min) >> any match

# Centre side-wall markers (y≈0), present in both real + gvz arenas.
CENTER_MARKERS = (11, 15)

# How many capture passes to chain per attacker push. Far more than a 10-min
# match can fly, so the script never ends -> the drone never lands.
ATTACK_CHAIN_PASSES = 30


def _fmt(lines: List[str]) -> str:
    return "\n".join(l for l in lines if l) + "\n"


def _rc_hold(lr: int, fb: int, ud: int, yaw: int) -> List[str]:
    """A never-land hover (yaw=0) or slow scan (yaw>0) built as a CHAIN of SHORT
    RC steps so a mission SPLICE walks into its tail within ~RC_HOLD_TICK_S
    instead of waiting out a single long RC timer (~1 h). See RC_HOLD_* above."""
    step = f"RC {int(lr)} {int(fb)} {int(ud)} {int(yaw)} {RC_HOLD_TICK_S}"
    return [step] * RC_HOLD_REPEATS


# All builders take a live ``Tunables`` (operator-editable). Defaults equal the
# proven constants above, so passing Tunables() reproduces the validated values.
from .tunables import Tunables   # noqa: E402  (after the constants, by design)


def _clamp_alt(a: float) -> float:
    """Floor at the 0.73 m box top, cap at MAX_MOVER_ALT_M."""
    return max(BOX_TOP_M, min(MAX_MOVER_ALT_M, a))


def attacker_alt(t: Tunables) -> float:
    """FLAT altitude shared by all attackers — their fixed lanes never cross, so
    they don't need vertical separation. ~1 m for spotting the low box markers."""
    return _clamp_alt(t.mover_base_alt_m)


def defender_alt(rank: int, t: Tunables) -> float:
    """Defender altitude = the attacker altitude PLUS a fixed offset
    (``defender_alt_above_attacker_m``, default 0.80 m) so defenders always wait
    well ABOVE the attackers and can never collide with them, plus a small
    per-defender step so two defenders sharing a neutral station are vertically
    separated. Clamped to the 0.73-2.50 m band."""
    return _clamp_alt(attacker_alt(t) + t.defender_alt_above_attacker_m
                      + t.mover_alt_step_m * max(0, rank))


def mover_alt(lane: int, t: Tunables) -> float:  # back-compat (scout/legacy)
    return _clamp_alt(t.mover_base_alt_m + t.mover_alt_step_m * max(0, lane))


# ── Scout ───────────────────────────────────────────────────────────────────
def scout_script(center_marker: int, t: Tunables) -> str:
    """Fly to the arena centre via a side marker, then rotate in place forever.

    GO_HOME (loose, holds altitude) onto the centre marker -> ~arena centre at
    the set altitude; then one long RC yaw drive scans every box. Never lands."""
    return _fmt([
        "TAKEOFF",
        f"HEIGHT {t.scout_alt_m:.2f}",
        f"GO_HOME {int(center_marker)} {CENTER_STANDOFF_M:.2f} {CENTER_TOL_M:g} 0",
        f"HEIGHT {t.scout_alt_m:.2f}",
        f"RC 0 0 0 {int(t.scout_yaw_stick)} {ROTATE_FOREVER_S}",
    ])


# ── Defender ────────────────────────────────────────────────────────────────
def defender_script(side_marker: int, alt_m: float, t: Tunables) -> str:
    """Park in front of a side-wall marker near our home boxes and slow-rotate
    forever — presence blocks enemy capture; the rotation keeps it active."""
    return _fmt([
        "TAKEOFF",
        f"HEIGHT {alt_m:.2f}",
        f"GO_HOME {int(side_marker)} {t.defender_standoff_m:.2f} {DEFENDER_SIDE_TOL_M:g} 0",
        *_rc_hold(0, 0, 0, int(t.defender_yaw_stick)),     # slow scan (short-RC chain)
    ])


# ── Attacker ────────────────────────────────────────────────────────────────
def attack_leg(face_id_: int, wall_marker: int, cruise_alt_m: float,
               rth_hdg_deg: float, t: Tunables) -> List[str]:
    """One capture-and-return leg (no TAKEOFF). Cruise high -> APPROACH the box
    face -> slide up+over to flip -> turn -> cruise high -> GO_HOME our wall.
    After GO_HOME the box point scores; the next leg's APPROACH re-finds the
    next enemy face, so legs chain with NO landing in between."""
    cruise = f"HEIGHT {cruise_alt_m:.2f}"
    return [
        cruise,
        f"APPROACH {int(face_id_)} {t.attack_standoff_m:.2f} {t.attack_dist_tol_m:.2f}",
        f"FB_UD_IMU {t.over_box_forward_m:.2f} {t.capture_rise_m:.2f}",
        "YAW_IMU 180",
        cruise,
        f"GO_HOME {int(wall_marker)} {t.rth_standoff_m:.2f} {RTH_ARRIVE_TOL_M:g} {rth_hdg_deg:g}",
    ]


def attacker_loop_script(enemy_faces: List[int], wall_marker: int,
                         cruise_alt_m: float, rth_hdg_deg: float,
                         t: Tunables) -> str:
    """A long chain: ``t.attack_passes`` loops over ``enemy_faces``, each a full
    capture-and-return leg. Never ends within a match -> never lands. The enemy
    boxes flip back to enemy colour when we leave, so re-attacking them stays
    productive (continuous offense)."""
    lines: List[str] = ["TAKEOFF"]
    for _ in range(max(1, int(t.attack_passes))):
        for f in enemy_faces:
            lines.extend(attack_leg(int(f), wall_marker, cruise_alt_m,
                                    rth_hdg_deg, t))
    return _fmt(lines)


def enemy_home_faces(our_team: str) -> List[int]:
    """The ArUco faces shown by the ENEMY's home boxes at default — what an
    attacker APPROACHes. Red attacks blue's boxes 4-6 (faces 34/35/36); blue
    attacks red's boxes 1-3 (faces 41/42/43)."""
    enemy = "blue" if our_team == "red" else "red"
    enemy_slots = (4, 5, 6) if our_team == "red" else (1, 2, 3)
    return [face_id(s, enemy) for s in enemy_slots]


def enemy_target_faces(our_team: str, holders: dict) -> List[int]:
    """Enemy faces still worth attacking, given the scout's live box state:
    the enemy slots NOT currently held by us (enemy-held or unknown). Skips
    boxes already ours — re-attacking those wastes a pass and (since an
    already-ours box shows OUR face, e.g. 44 not 34) makes APPROACH hunt a
    marker that isn't up. Falls back to ALL enemy faces when we already hold
    them all, so the loop keeps pressure as boxes flip back to enemy colour.

    ``holders`` is {slot -> 'red'|'blue'|'unknown'} (the decayed effective
    holder from MatchState.holders())."""
    enemy = "blue" if our_team == "red" else "red"
    enemy_slots = (4, 5, 6) if our_team == "red" else (1, 2, 3)
    targets = [s for s in enemy_slots if holders.get(s) != our_team]
    if not targets:                       # we hold them all -> keep full loop
        targets = list(enemy_slots)
    return [face_id(s, enemy) for s in targets]


def attacker_splice_script(enemy_faces: List[int], wall_marker: int,
                           cruise_alt_m: float, rth_hdg_deg: float,
                           t: Tunables) -> str:
    """Like attacker_loop_script but with NO TAKEOFF — a long capture chain
    for LIVE-SPLICING into an already-airborne attacker (re-target without
    landing). The FC keeps the current step running and walks into this tail,
    so it must not re-take-off. Long enough (``t.attack_passes`` loops) that
    it never ends within a match -> the drone never lands."""
    lines: List[str] = []
    for _ in range(max(1, int(t.attack_passes))):
        for f in enemy_faces:
            lines.extend(attack_leg(int(f), wall_marker, cruise_alt_m,
                                    rth_hdg_deg, t))
    return _fmt(lines)


# ── Simplified fixed-lane strategy (3 attackers + 2 neutral defenders) ───────

def our_box_enemy_faces(our_team: str) -> dict:
    """{our_slot -> the ArUco face it shows WHEN THE ENEMY HOLDS IT} — what a
    defender APPROACHes to recapture it. Red's boxes 1-3 show 31/32/33 when blue
    holds them; blue's boxes 4-6 show 44/45/46 when red holds them."""
    enemy = "blue" if our_team == "red" else "red"
    return {s: face_id(s, enemy) for s in _our_slots(our_team)}


def _our_slots(our_team: str):
    return (1, 2, 3) if our_team == "red" else (4, 5, 6)


def attacker_lane_script(enemy_face: int, wall_marker: int, cruise_alt_m: float,
                         rth_hdg_deg: float, t: Tunables) -> str:
    """FIXED-TARGET attacker — a SELF-CONTAINED never-land loop on ONE enemy box,
    built with the FC's WAIT_AND_ATTACK + REPEAT verbs. No C2 splice / box-state /
    FC mission-state needed: the mission itself shuttles the straight lane
    home<->box. Each pass: fly out and WAIT_AND_ATTACK the box (it acquires the box
    by EITHER face — the enemy face OR our sibling face — so it never loses the
    marker when the box is ours; it holds at standoff until the box shows the ENEMY
    face, then closes in), slide up+over to flip it, turn, GO_HOME the back-wall
    marker to score, then HOVER in our home zone for ``attacker_home_dwell_s`` —
    then REPEAT loops back to the first non-TAKEOFF step (the leg) forever, skipping
    TAKEOFF, so the drone NEVER lands on its own. Distinct face per attacker =>
    distinct boxes => collision-safe straight lanes.

    Using WAIT_AND_ATTACK (not plain APPROACH) is what makes "attack only when the
    target is enemy-held" robust AND never-land: a box that is currently OURS shows
    the sibling face, which WAIT_AND_ATTACK still tracks, so the drone holds at
    standoff awaiting the flip instead of searching for an absent marker and
    eventually landing. The home dwell still returns it home to score between
    strikes; it only holds forward at the box when it arrives before the flip."""
    cruise = f"HEIGHT {cruise_alt_m:.2f}"
    dwell = max(1, int(round(t.attacker_home_dwell_s)))
    return _fmt([
        "TAKEOFF",
        cruise,
        # Acquire the box by EITHER face; hold at standoff until it shows the ENEMY
        # face, then close in. Never searches-then-lands when the box is ours.
        f"WAIT_AND_ATTACK {int(enemy_face)} {t.attack_standoff_m:.2f} {t.attack_dist_tol_m:.2f}",
        f"FB_UD_IMU {t.over_box_forward_m:.2f} {t.capture_rise_m:.2f}",
        "YAW_IMU 180",
        cruise,
        f"GO_HOME {int(wall_marker)} {t.rth_standoff_m:.2f} {RTH_ARRIVE_TOL_M:g} {rth_hdg_deg:g}",
        f"RC 0 0 0 0 {dwell}",            # hover at home (score), then strike again
        "REPEAT",                          # loop the leg forever, skipping TAKEOFF -> never lands
    ])


def enemy_face_for_slot(slot: int, our_team: str) -> int:
    """The ArUco face this ENEMY box shows when the enemy holds it (what an
    attacker APPROACHes). Red attacking blue slot 4 -> 34; blue attacking red
    slot 1 -> 41."""
    enemy = "blue" if our_team == "red" else "red"
    return face_id(int(slot), enemy)


# ── Swimlanes (3 attackers shuttle straight vertical lanes) ──────────────────
# The 6 boxes mirror across the centre line at matching x: box 1<->4 (x=-3),
# box 2<->5 (x=0), box 3<->6 (x=+3). Each swimlane pairs the two boxes at one x,
# so an attacker shuttling that lane flies a STRAIGHT vertical path and the 3
# lanes never cross. Lane L (0,1,2) = boxes (L+1) [red home] and (L+4) [blue home].
NUM_SWIMLANES = 3


def swimlane_faces(lane: int, our_team: str) -> tuple:
    """``(enemy_box_face, our_box_face)`` for swimlane ``lane`` (0..2), from OUR
    team's view. BOTH are the box's ENEMY-colour face, so a WAIT_AND_ATTACK on
    each captures the enemy box / recaptures our box whenever it shows the enemy
    colour (and holds at standoff awaiting the flip otherwise). The lane pairs box
    (lane+1) [red home] with box (lane+4) [blue home]; which is "ours" vs "enemy"
    flips with team."""
    enemy = "blue" if our_team == "red" else "red"
    red_slot = (lane % NUM_SWIMLANES) + 1          # 1,2,3
    blue_slot = (lane % NUM_SWIMLANES) + 4         # 4,5,6
    if our_team == "red":
        our_slot, enemy_slot = red_slot, blue_slot
    else:
        our_slot, enemy_slot = blue_slot, red_slot
    return face_id(enemy_slot, enemy), face_id(our_slot, enemy)


def attacker_swimlane_script(enemy_box_face: int, our_box_face: int,
                             cruise_alt_m: float, t: Tunables) -> str:
    """SWIMLANE attacker — shuttles a straight vertical lane between an ENEMY box
    and OUR box, capturing/recapturing each whenever it shows the enemy colour.

    WAIT_AND_ATTACK at BOTH ends (acquires the box by EITHER face, holds at
    standoff until it shows the ENEMY colour, then closes in to flip it);
    FB_UD_IMU slides up+over to capture; YAW_IMU 180 turns toward the other end;
    REPEAT loops back to the first non-TAKEOFF step forever (skipping TAKEOFF) so
    the drone NEVER lands. Distinct lane per attacker => the 3 lanes never cross
    (no fan-out / altitude ladder needed)."""
    cruise = f"HEIGHT {cruise_alt_m:.2f}"
    std = f"{t.attack_standoff_m:.2f} {t.attack_dist_tol_m:.2f}"
    fbud = f"FB_UD_IMU {t.over_box_forward_m:.2f} {t.capture_rise_m:.2f}"
    return _fmt([
        "TAKEOFF",
        cruise,
        f"WAIT_AND_ATTACK {int(enemy_box_face)} {std}",   # enemy box: capture when enemy-held
        fbud,
        "YAW_IMU 180",
        f"WAIT_AND_ATTACK {int(our_box_face)} {std}",      # our box: recapture when enemy-held
        fbud,
        "YAW_IMU 180",
        "REPEAT",                                          # loop forever, skip TAKEOFF -> never lands
    ])


def attacker_park_script(wall_marker: int, cruise_alt_m: float,
                         rth_hdg_deg: float, t: Tunables) -> str:
    """Fallback hover for an EXTRA attacker with no free swimlane (more attackers
    than the 3 lanes): take off, settle in our HOME zone, and hover forever (never
    lands). Attackers that DO own a lane fly ``attacker_swimlane_script`` instead."""
    cruise = f"HEIGHT {cruise_alt_m:.2f}"
    return _fmt([
        "TAKEOFF",
        cruise,
        f"GO_HOME {int(wall_marker)} {t.rth_standoff_m:.2f} {RTH_ARRIVE_TOL_M:g} {rth_hdg_deg:g}",
        cruise,
        *_rc_hold(0, 0, 0, 0),                # hover at home (short-RC chain), never land
    ])


def defender_neutral_script(wall_marker: int, neutral_standoff_m: float,
                            alt_m: float, t: Tunables) -> str:
    """Wait ``neutral_standoff_m`` (operator: 5 m) off our home-wall marker, facing
    our boxes, and slow-rotate to scan them. GO_HOME onto the home-wall marker at
    that LOOSE standoff (5 m ~ the home/neutral boundary; raise it to ~5.5-6 m to
    sit clearly in neutral), then rotate forever. The runner splices in a recapture
    the instant one of our boxes flips."""
    return _fmt([
        "TAKEOFF",
        f"HEIGHT {alt_m:.2f}",
        f"GO_HOME {int(wall_marker)} {neutral_standoff_m:.2f} {DEFENDER_SIDE_TOL_M:g} 0",
        *_rc_hold(0, 0, 0, int(t.defender_yaw_stick)),     # neutral scan (short-RC chain)
    ])


def defender_recapture_splice(enemy_face: int, wall_marker: int,
                              neutral_standoff_m: float, alt_m: float,
                              t: Tunables) -> str:
    """NO-TAKEOFF splice: dash IN to the flipped own box and recapture it with
    WAIT_AND_ATTACK (acquire by either face, attack the moment it shows the enemy
    colour -> slide up+over to flip it back), then return to the neutral scan
    station 5 m off our home marker and rotate. Spliced into an airborne defender,
    so no landing. WAIT_AND_ATTACK (vs plain APPROACH) keeps the recapture from
    searching-then-landing if the box momentarily flips during the dash-in."""
    cruise = f"HEIGHT {alt_m:.2f}"
    return _fmt([
        cruise,
        f"WAIT_AND_ATTACK {int(enemy_face)} {t.attack_standoff_m:.2f} {t.attack_dist_tol_m:.2f}",
        f"FB_UD_IMU {t.over_box_forward_m:.2f} {t.capture_rise_m:.2f}",
        "YAW_IMU 180",
        cruise,
        f"GO_HOME {int(wall_marker)} {neutral_standoff_m:.2f} {DEFENDER_SIDE_TOL_M:g} 0",
        *_rc_hold(0, 0, 0, int(t.defender_yaw_stick)),     # back to neutral scan
    ])
