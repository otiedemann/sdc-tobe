"""Defender role — protect our slots without ever camping a box.

Regs DQ a drone that "sits like a dead duck" on a target, so the defender
NEVER lingers over any single box. Two modes:

  * UN-CAP: if ``drone.assigned_slot`` points at one of our slots that an
    enemy has flipped, fly there at defender_uncap_altitude_m and HOVER
    capture_hover_s to revert it, then drop back to patrol.

  * PATROL: otherwise sweep our home zone, cycling GOTO between our slot
    positions at defender_patrol_altitude_m. Advance to the next slot once
    we arrive OR after anti_camp_max_s at the current one — whichever comes
    first — so we are always moving.

State in ctx.memory:
    "def_patrol_i"      index into our_slots() currently targeted.
    "def_patrol_since"  ctx.now we started toward the current patrol slot.
    "def_uncap_start"   ctx.now the un-cap hover began (None until at alt).
"""

from __future__ import annotations

from ..models import (
    DroneState, WorldSnapshot, RoleAction, ActionKind, DronePhase, Vec3,
)
from .base import RoleContext


class DefenderRole:
    name = "defender"

    def _patrol(self, drone: DroneState, ctx: RoleContext) -> RoleAction:
        """Cycle through our slots, never lingering on any one."""
        strat = ctx.config.strategy
        our_slots = ctx.config.our_slots()
        if not our_slots:
            hold = Vec3(drone.position.x, drone.position.y, strat.defender_patrol_altitude_m)
            return RoleAction(
                kind=ActionKind.GOTO, target=hold,
                next_phase=DronePhase.ORBIT, note="defender patrol (no slots)",
            )

        i = ctx.memory.get("def_patrol_i", 0) % len(our_slots)
        ctx.memory["def_patrol_i"] = i          # seed so the UI reflects it
        since = ctx.memory.get("def_patrol_since")
        if since is None:
            since = ctx.now
            ctx.memory["def_patrol_since"] = since

        slot = our_slots[i]
        nominal = ctx.config.nominal_slot_positions[slot]
        target = Vec3(nominal.x, nominal.y, strat.defender_patrol_altitude_m)

        arrived = drone.position.dist_xy(target) <= strat.arrival_tol_m
        lingered = (ctx.now - since) >= strat.anti_camp_max_s

        # Advance to the next slot on arrival OR anti-camp timeout.
        if arrived or lingered:
            i = (i + 1) % len(our_slots)
            ctx.memory["def_patrol_i"] = i
            ctx.memory["def_patrol_since"] = ctx.now
            slot = our_slots[i]
            nominal = ctx.config.nominal_slot_positions[slot]
            target = Vec3(nominal.x, nominal.y, strat.defender_patrol_altitude_m)

        return RoleAction(
            kind=ActionKind.GOTO,
            target=target,
            next_phase=DronePhase.ORBIT,
            note=f"defender patrol slot {slot}",
        )

    def tick(self, drone: DroneState, world: WorldSnapshot, ctx: RoleContext) -> RoleAction:
        if not drone.flying or drone.position is None:
            return RoleAction.none(note="defender (grounded)")

        strat = ctx.config.strategy
        our_team = ctx.config.our_team
        slot = drone.assigned_slot
        ts = world.slots.get(slot) if slot is not None else None

        # Is a specific own slot under threat (no longer our colour)?
        threatened = (
            slot is not None
            and slot in ctx.config.our_slots()
            and ts is not None
            and not ts.owned_by(our_team)
        )

        if threatened:
            slot_xy = ts.position if ts.position is not None \
                else ctx.config.nominal_slot_positions[slot]
            uncap_target = Vec3(slot_xy.x, slot_xy.y, strat.defender_uncap_altitude_m)
            at_alt = abs(uncap_target.z - drone.position.z) <= strat.altitude_tol_m
            arrived = drone.position.dist_xy(uncap_target) <= strat.arrival_tol_m

            # Fly down onto the box first.
            if not (arrived and at_alt):
                ctx.memory["def_uncap_start"] = None
                return RoleAction(
                    kind=ActionKind.GOTO,
                    target=uncap_target,
                    next_phase=DronePhase.UNCAPPING,
                    note=f"defender un-cap -> slot {slot}",
                )

            # On the box: hover to revert its colour.
            start = ctx.memory.get("def_uncap_start")
            if start is None:
                start = ctx.now
                ctx.memory["def_uncap_start"] = start
            held = ctx.now - start

            # Done when it flips back to us OR the hover time elapses.
            if ts.owned_by(our_team) or held >= strat.capture_hover_s:
                ctx.memory["def_uncap_start"] = None
                # Reset patrol timer so we move off immediately, not camp.
                ctx.memory["def_patrol_since"] = None
                return self._patrol(drone, ctx)

            return RoleAction(
                kind=ActionKind.HOVER,
                target=uncap_target,
                hold_s=strat.capture_hover_s,
                next_phase=DronePhase.UNCAPPING,
                note=f"defender un-cap slot {slot} ({held:.1f}/"
                     f"{strat.capture_hover_s:.0f}s)",
            )

        # No threat -> keep patrolling.
        return self._patrol(drone, ctx)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from c2.config import C2Config
    from c2.models import (
        DroneState, WorldSnapshot, TargetSlot, GameMode, DroneRole,
        SlotColor, Vec3,
    )
    from c2.roles.base import RoleContext

    cfg = C2Config()  # our_team RED -> defends slots 4..6
    slots = {
        i: TargetSlot(
            index=i,
            home_team=cfg.slot_map.home_team_of(i),
            color=SlotColor.from_team(cfg.slot_map.home_team_of(i)),
            position=cfg.nominal_slot_positions[i],
        )
        for i in range(1, 7)
    }
    world = WorldSnapshot(t=0.0, mode=GameMode.AUTO, our_team=cfg.our_team, slots=slots)

    home = cfg.arena.home_center(cfg.our_team)
    drone = DroneState(
        drone_id="def-1", role=DroneRole.DEFENDER, flying=True,
        position=Vec3(home.x, home.y, cfg.strategy.defender_patrol_altitude_m),
        heading_deg=0.0, base_altitude_m=2.0,
    )
    role = DefenderRole()
    ctx = RoleContext(config=cfg, now=0.0, memory={})

    print("== defender PATROL smoke test (anti-camp) ==")
    for t in range(7):
        ctx.now = float(t)
        act = role.tick(drone, world, ctx)
        print(f"t={t} patrol_i={ctx.memory.get('def_patrol_i')} "
              f"{act.kind.value:<6} target={act.target.to_list() if act.target else None} "
              f"-> {act.next_phase.value:<10} | {act.note}")
        # Never reach the waypoint -> proves anti-camp timeout advances us.

    print("\n== defender UN-CAP smoke test ==")
    threatened_slot = cfg.our_slots()[0]
    slots[threatened_slot].color = SlotColor.from_team(cfg.enemy_team())  # enemy grabbed it
    drone.assigned_slot = threatened_slot
    slot_xy = cfg.nominal_slot_positions[threatened_slot]
    ctx2 = RoleContext(config=cfg, now=0.0, memory={})
    for t in range(6):
        ctx2.now = float(t)
        act = role.tick(drone, world, ctx2)
        print(f"t={t} {act.kind.value:<6} -> {act.next_phase.value:<10} | {act.note}")
        if t == 0:  # arrive on the box at un-cap altitude
            drone.position = Vec3(slot_xy.x, slot_xy.y, cfg.strategy.defender_uncap_altitude_m)
        elif t == 3:  # hover elapsed -> box reverts to our colour
            slots[threatened_slot].color = SlotColor.from_team(cfg.our_team)
