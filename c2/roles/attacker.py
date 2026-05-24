"""Attacker role — drive to an assigned enemy slot and capture it.

The commander sets ``drone.assigned_slot`` to an ENEMY slot index. This
role runs a small phase machine (stored in ``ctx.memory["atk_phase"]``)
to fly there, descend into the 1-2 m detection band, hover long enough to
flip the box to our colour, then carry the capture home for validation.

Phases (ctx.memory["atk_phase"]):
    "transit"  GOTO slot xy at base altitude.
    "capture"  descend to capture_altitude_m, HOVER capture_hover_s.
    "return"   GOTO our home centre at base altitude.
    "done"     idle, capture validated; await reassignment.

The machine resets whenever ``assigned_slot`` changes (tracked via
``ctx.memory["atk_slot"]``). Timers use ``ctx.now``.
"""

from __future__ import annotations

from typing import Optional

from ..models import (
    DroneState, WorldSnapshot, RoleAction, ActionKind, DronePhase, Vec3,
)
from .base import RoleContext


class AttackerRole:
    name = "attacker"

    def _slot_position(self, slot: int, world: WorldSnapshot, ctx: RoleContext) -> Vec3:
        """Best-known xy of a slot: observed position, else nominal."""
        ts = world.slots.get(slot)
        if ts is not None and ts.position is not None:
            return ts.position
        return ctx.config.nominal_slot_positions[slot]

    def _reset(self, ctx: RoleContext, slot: Optional[int]) -> None:
        ctx.memory["atk_phase"] = "transit"
        ctx.memory["atk_slot"] = slot
        ctx.memory["capture_start"] = None

    def tick(self, drone: DroneState, world: WorldSnapshot, ctx: RoleContext) -> RoleAction:
        if not drone.flying or drone.position is None:
            return RoleAction.none(note="attacker (grounded)")

        strat = ctx.config.strategy
        slot = drone.assigned_slot

        # No assignment: hold over the current xy at base altitude.
        if slot is None:
            ctx.memory["atk_slot"] = None
            hold = Vec3(drone.position.x, drone.position.y, drone.base_altitude_m)
            return RoleAction(
                kind=ActionKind.GOTO,
                target=hold,
                next_phase=DronePhase.HOLDING,
                note="attacker idle, no slot",
            )

        # Reset the phase machine if the target changed.
        if ctx.memory.get("atk_slot") != slot or "atk_phase" not in ctx.memory:
            self._reset(ctx, slot)

        our_team = ctx.config.our_team
        ts = world.slots.get(slot)
        already_ours = ts is not None and ts.owned_by(our_team)
        slot_xy = self._slot_position(slot, world, ctx)
        phase = ctx.memory.get("atk_phase", "transit")

        # ---- transit: fly to the slot at cruise altitude -----------------
        if phase == "transit":
            # If it's somehow already ours, skip straight home.
            if already_ours:
                ctx.memory["atk_phase"] = "return"
                phase = "return"
            else:
                target = Vec3(slot_xy.x, slot_xy.y, drone.base_altitude_m)
                if drone.position.dist_xy(target) <= strat.arrival_tol_m:
                    ctx.memory["atk_phase"] = "capture"
                    ctx.memory["capture_start"] = None
                    phase = "capture"
                else:
                    return RoleAction(
                        kind=ActionKind.GOTO,
                        target=target,
                        next_phase=DronePhase.TRANSIT,
                        note=f"attacker transit -> slot {slot}",
                    )

        # ---- capture: descend into the band + hover to flip the box -------
        if phase == "capture":
            cap_target = Vec3(slot_xy.x, slot_xy.y, strat.capture_altitude_m)
            at_alt = abs(cap_target.z - drone.position.z) <= strat.altitude_tol_m

            # Still descending: keep going down to capture altitude.
            if not at_alt:
                return RoleAction(
                    kind=ActionKind.GOTO,
                    target=cap_target,
                    next_phase=DronePhase.CAPTURING,
                    note=f"attacker descend -> slot {slot}",
                )

            # At altitude: run the hover timer.
            start = ctx.memory.get("capture_start")
            if start is None:
                start = ctx.now
                ctx.memory["capture_start"] = start
            held = ctx.now - start

            # Done when the box flips to our colour OR the hover time elapses.
            if already_ours or held >= strat.capture_hover_s:
                ctx.memory["atk_phase"] = "return"
                phase = "return"
            else:
                return RoleAction(
                    kind=ActionKind.HOVER,
                    target=cap_target,
                    hold_s=strat.capture_hover_s,
                    next_phase=DronePhase.CAPTURING,
                    note=f"attacker capture slot {slot} ({held:.1f}/"
                         f"{strat.capture_hover_s:.0f}s)",
                )

        # ---- return: carry the capture home for validation ---------------
        if phase == "return":
            home = ctx.config.arena.home_center(our_team)
            target = Vec3(home.x, home.y, drone.base_altitude_m)
            if ctx.config.arena.in_home_zone(drone.position, our_team):
                ctx.memory["atk_phase"] = "done"
                phase = "done"
            else:
                return RoleAction(
                    kind=ActionKind.GOTO,
                    target=target,
                    next_phase=DronePhase.RETURNING,
                    note=f"attacker return home (slot {slot})",
                )

        # ---- done --------------------------------------------------------
        return RoleAction.none(note="capture validated, awaiting orders")


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

    cfg = C2Config()  # our_team RED -> attacks blue slots 1..3
    enemy_slot = cfg.enemy_slots()[0]
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

    # Start in our home zone, assigned an enemy slot.
    home = cfg.arena.home_center(cfg.our_team)
    drone = DroneState(
        drone_id="atk-1", role=DroneRole.ATTACKER, flying=True,
        position=Vec3(home.x, home.y, 2.0), heading_deg=0.0,
        base_altitude_m=2.0, assigned_slot=enemy_slot,
    )
    role = AttackerRole()
    ctx = RoleContext(config=cfg, now=0.0, memory={})
    slot_xy = cfg.nominal_slot_positions[enemy_slot]

    print(f"== attacker smoke test (enemy slot {enemy_slot}) ==")
    for t in range(8):
        ctx.now = float(t)
        act = role.tick(drone, world, ctx)
        print(f"t={t} phase={ctx.memory.get('atk_phase'):<8} "
              f"{act.kind.value:<6} -> {act.next_phase.value:<10} | {act.note}")
        # Drive a plausible world response so phases actually advance:
        ph = ctx.memory.get("atk_phase")
        if t == 0:   # arrived over the slot (cruise altitude)
            drone.position = Vec3(slot_xy.x, slot_xy.y, drone.base_altitude_m)
        elif t == 1:  # descended into the capture band
            drone.position = Vec3(slot_xy.x, slot_xy.y, cfg.strategy.capture_altitude_m)
        elif t == 4:  # hover elapsed -> box flips to our colour
            slots[enemy_slot].color = SlotColor.from_team(cfg.our_team)
        elif ph == "return":  # snap home so we reach the zone
            drone.position = Vec3(home.x, home.y, drone.base_altitude_m)
