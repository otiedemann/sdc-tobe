"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role pushes ONE complete
mission script that does everything end-to-end:

    TAKEOFF
    HEIGHT <capture_ascend_m>      # ascend BEFORE closing in so the
                                   # camera has line-of-sight to the
                                   # distant marker
    FB_IMU <close_in_m>            # cruise forward in IMU frame until
                                   # within ArUco range (~3-4 m)
    APPROACH <enemy-face-id> <approach_distance_m>
    HEIGHT <capture_ascend_m>      # re-assert in case approach drifted
    FB_IMU <capture_forward_m>     # drift over the box
    YAW_IMU 180
    HOOVER <capture_hover_s>
    TO_HOME <home_alt>
    LAND

The drone closes in toward the enemy side, ascends, locks onto the
marker via APPROACH, drifts over it, hovers, then RTHs and lands. After
the script fully ends (FC drops back to Phase.INIT), the role marks
itself done and the target is cleared.

OPERATOR CONVENTION
-------------------
The FB_IMU close-in assumes the drone is placed at takeoff with its
front camera pointing toward the enemy side (i.e. red drones face arena
+y, blue drones face -y). FB_IMU is forward-in-drone-frame, not arena
frame — without the right takeoff orientation the drone would fly off
into a wall. We don't try to align via YAW_IMU because the absolute
IMU↔arena yaw transform is unreliable (no compass calibration in sim,
magnetic_north_arena_yaw_deg drifts). Operator-set takeoff orientation
is simple and reliable.

We deliberately keep it as a single push (rather than separate attack +
RTH scripts) because pushing a second script while the first is still
running is rejected by marker_mission_c2's start endpoint. Mid-script
phase==IDLE (during HOOVER) is *not* "script finished" — it's just
station-keeping within the current mission.

Phases:
    idle    → no target assigned, or target already captured by us
    running → attack script pushed, drone executing the whole sequence
    done    → script ended (FC back to INIT); reset to idle

If the operator assigns a slot we already hold (holder == our_team) the
role short-circuits straight to ``done``.

Verify-by-scout (i.e. abort RTH if the capture is not confirmed) is a
future enhancement; for now the script always commits to the full
approach-capture-RTH-land sequence.
"""
from __future__ import annotations

import logging
import math

from .roles import Decision, Role, RoleContext, noop, push, register
from .settings import face_id

logger = logging.getLogger(__name__)


# Physical target slot positions in arena frame (x, y) — z is fixed at
# 1.0 m by the box pedestals and isn't needed for the close-in distance.
# Pulled from tools/sphinx-arena/default_target_layout.json and the
# §1.1 regs: red slots 4-6 at y=-7.5, blue slots 1-3 at y=+7.5, all
# spread across x ∈ {-3, 0, +3}.
SLOT_POSITIONS_M: dict[int, tuple[float, float]] = {
    1: (-3.0, 7.5),  2: (0.0, 7.5),  3: (3.0, 7.5),
    4: (-3.0, -7.5), 5: (0.0, -7.5), 6: (3.0, -7.5),
}

# Stop the FB_IMU close-in this far short of the target so APPROACH has
# room to do its precision dance. 7 m gives 18 cm markers comfortable
# headroom for detection — empirically ArUco picks them up at <8 m on
# the sim's 70° FOV / 1280×720 cam.
CLOSE_IN_STANDOFF_M: float = 7.0

# mission_script enforces |FB_IMU| <= 5 m per step (anti-typo safety
# cap). But empirically chained 5m moveBy steps stall short (one round
# of 3×5+1.5 commanded → only ~7 m actual). Cap our chunk size at 4 m
# to stay well clear of that regime while respecting the parser cap.
FB_IMU_CHUNK_M: float = 4.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _chunk_close_in(total_m: float) -> list[float]:
    """Split a total forward-cruise distance into <= FB_IMU_CHUNK_M chunks.

    Drops chunks below 0.1 m (parser min is 0.01 m but tiny steps add
    latency for no useful motion). Returns [] for non-positive totals.
    """
    out: list[float] = []
    remaining = float(total_m)
    while remaining > FB_IMU_CHUNK_M:
        out.append(FB_IMU_CHUNK_M)
        remaining -= FB_IMU_CHUNK_M
    if remaining >= 0.1:
        out.append(remaining)
    return out


def _close_in_distance_m(home_xy: tuple[float, float], slot: int) -> float:
    """Straight-line distance from home to target slot, minus standoff.

    Returns 0.0 if the slot is unknown or if home is already inside the
    standoff (don't emit a degenerate FB_IMU that drives backward).
    """
    target = SLOT_POSITIONS_M.get(int(slot))
    if target is None:
        return 0.0
    dx = target[0] - home_xy[0]
    dy = target[1] - home_xy[1]
    raw = math.hypot(dx, dy)
    return max(0.0, raw - CLOSE_IN_STANDOFF_M)


def _full_attack_script(
    ctx: RoleContext, attack_marker_id: int, slot: int
) -> str:
    """Close in toward the enemy side, capture, RTH, land.

    Script shape:
        TAKEOFF
        HEIGHT <capture_ascend_m>      # ascend before closing in so
                                       # the camera has line-of-sight
        FB_IMU <close_in_m>            # cruise within ArUco range
        APPROACH <id> <approach_d>     # precision lock + slow advance
        HEIGHT <capture_ascend_m>      # re-assert post-approach
        FB_IMU <capture_forward_m>     # drift over the box
        YAW_IMU 180                    # turn to face home
        HOOVER <capture_hover_s>       # capture window
        TO_HOME <home_alt>             # straight-line cruise to home
        LAND

    The close-in step (FB_IMU before APPROACH) is what lets APPROACH
    actually see the target — 18 cm markers at 15 m are below the
    detector's reliable range. By ascending to capture_ascend_m first
    and then cruising forward in IMU frame, we arrive at standoff
    distance with the marker centred in the front cam.

    The YAW_IMU 180 after capture is the key to a fast RTH: the
    subsequent TO_HOME step drives the drone forward (not backward),
    giving the fwd PD channel — already the fastest-tuned — the full
    control range.
    """
    m = ctx.match
    approach_d = max(0.2, float(m.approach_distance_m))
    ascend = max(0.6, float(m.capture_ascend_m))
    forward = max(0.1, float(m.capture_forward_m))
    # Lower floor than the previous 2.0 — the operator wants race-pace
    # captures (the capture window is mostly "give the scout a chance
    # to register the new face"; 1 s is enough at 5 Hz vision).
    hover_s = max(1.0, float(m.capture_hover_s))
    # Home zone follows OUR team (the operator-level "our_team" field),
    # not the per-drone team. The drone.team dropdown can drift out of
    # sync — e.g. the operator flips us from blue to red but a single
    # drone still shows team=blue. The right semantic is "every drone
    # of our swarm flies back to OUR base", so use ctx.our_team.
    home = (
        ctx.match.home_red if ctx.our_team == "red" else ctx.match.home_blue
    )
    # RTH cruise altitude. Per-drone home_alt_m (exposed in the dashboard)
    # wins over match.home_red.alt / match.home_blue.alt — the operator's
    # mental model is "I set the slider for THIS drone." The match-level
    # value is kept as a fallback so a fresh deploy still produces a
    # working TO step. Floor at 0.6 m so a typo can't fly into the floor.
    home_alt = max(0.6, float(ctx.drone.home_alt_m or home.alt))
    close_in = _close_in_distance_m((home.x, home.y), slot)
    # Chain multiple FB_IMU steps to respect the script parser's 5 m
    # safety cap (anti-typo). e.g. 11.5 m → 5 + 5 + 1.5.
    close_in_steps = "\n".join(
        f"FB_IMU {chunk:.2f}" for chunk in _chunk_close_in(close_in)
    )
    # RTH uses TO_HOME (drone returns to its takeoff snapshot in
    # ArUco frame) rather than TO with absolute arena coords. The
    # ArUco solution currently has an unsolved absolute offset error;
    # navigating relative to "where I took off from" sidesteps it
    # because motion deltas are reliable even when absolute is not.
    # home_alt is passed as the cruise altitude override so the
    # drone climbs above target boxes during RTH, then LAND
    # descends at home.
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {ascend:.2f}",
        close_in_steps,
        f"APPROACH {int(attack_marker_id)} {approach_d:.2f}",
        f"HEIGHT {ascend:.2f}",
        f"FB_IMU {forward:.2f}",
        "YAW_IMU 180",
        f"HOOVER {hover_s:.1f}",
        f"TO_HOME {home_alt:.2f}",
        "LAND",
    )


def _enemy_face_for(slot: int, our_team: str) -> int:
    enemy = "blue" if our_team == "red" else "red"
    return face_id(slot, enemy)


def _slot_is_ours(ctx: RoleContext, slot: int) -> bool:
    return ctx.markers.slot_holder(slot) == ctx.our_team


class AttackerRole(Role):
    name = "attacker"

    def decide(self, ctx: RoleContext) -> Decision:
        rs = ctx.role_state
        drone = ctx.drone

        if not drone.enabled:
            return noop("drone disabled")
        if not drone.team:
            return noop("attacker: no team assigned")
        if not ctx.state.drone_connected:
            return noop("attacker: drone not connected")

        slot = rs.target_slot

        # ---- idle: waiting for a target slot ------------------------------
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                return noop("attacker: idle (no target slot)")
            if _slot_is_ours(ctx, slot):
                rs.advance_phase(
                    "done", f"slot {slot} already captured by us — no action"
                )
                return noop(f"attacker: slot {slot} already ours")
            attack_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = attack_id
            return push(
                _full_attack_script(ctx, attack_id, slot),
                new_phase="running",
                reason=f"attacker: full attack on slot {slot} (id={attack_id})",
            )

        # ---- running: waiting for the entire script to finish -------------
        if rs.phase == "running":
            if slot is None:
                # Target cleared while in flight — let the script finish on
                # its own (it ends with LAND); we just reset our state.
                rs.advance_phase("idle", "target cleared mid-flight")
                return noop("attacker: target cleared mid-flight")
            # Only "init" reliably means the script has fully ended (FC has
            # landed and gone back to its resting state). Phase=IDLE
            # during the HOOVER step is mid-script and must not be
            # treated as completion.
            if ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase(
                    "done",
                    f"attack on slot {slot} complete (FC back to init)",
                )
                return noop("attacker: attack complete, target cleared")
            return noop(
                f"attacker: running (fc phase={ctx.state.phase})"
            )

        # Unknown phase — reset to idle.
        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("attacker: reset to idle")


register(AttackerRole())
