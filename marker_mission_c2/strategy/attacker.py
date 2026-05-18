"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role pushes ONE complete
mission script that does everything end-to-end:

    TAKEOFF
    APPROACH <enemy-face-id> <approach_distance_m>
    HEIGHT <capture_ascend_m>
    FB_IMU <capture_forward_m>
    HOOVER <capture_hover_s>
    TO <home.x> <home.y> <home_alt>
    LAND

The drone approaches the enemy face, ascends, drifts over the marker,
hovers briefly, returns to home, and lands. After the script is fully
done (FC drops back to Phase.INIT), the role marks itself done and the
target is cleared.

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

from .roles import Decision, Role, RoleContext, noop, push, register
from .settings import face_id

logger = logging.getLogger(__name__)


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _full_attack_script(ctx: RoleContext, attack_marker_id: int) -> str:
    """Approach the enemy face, fly over, turn around, RTH, land.

    Script shape:
        TAKEOFF
        APPROACH <id> <approach_distance_m>
        HEIGHT <capture_ascend_m>
        FB_IMU <capture_forward_m>     # drift forward to be *over* the box
        YAW_IMU 180                     # turn to face home
        HOOVER <capture_hover_s>        # capture window (already facing home)
        TO <home.x> <home.y> <home_alt> # forward cruise back to home
        LAND

    The YAW_IMU 180 step is the key to a fast return: by the time the
    HOOVER capture window ends, the drone is already pointed at home,
    so the subsequent TO step drives the drone forward (not backward
    or sideways), which gives the fwd PD channel — already the
    fastest-tuned — the full control range. Tested before this change,
    the drone had to swing 180° during TO, sliding sideways while
    yawing; with this change it's a clean straight-line cruise.
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
    return _format_script(
        "TAKEOFF",
        f"APPROACH {int(attack_marker_id)} {approach_d:.2f}",
        f"HEIGHT {ascend:.2f}",
        f"FB_IMU {forward:.2f}",
        "YAW_IMU 180",
        f"HOOVER {hover_s:.1f}",
        f"TO {home.x:.2f} {home.y:.2f} {home_alt:.2f}",
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
                _full_attack_script(ctx, attack_id),
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
