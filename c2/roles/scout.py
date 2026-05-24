"""Scout role — a single watcher that rotates in the arena centre.

The operator wants ONE scout parked over the centre of the arena, spinning
in place at high altitude so its downward/forward camera sweeps all six
target boxes and keeps every slot's colour + age fresh for the world model.

Two phases, tracked in ``ctx.memory["scout_phase"]``:

    "goto_centre" -> GOTO (0, 0, scout_altitude_m) until centred.
    "scan"        -> YAW_SCAN in place at scout_yaw_rate_dps.

If the drone ever drifts off-centre (e.g. wind, manual nudge) it falls
back to "goto_centre" automatically.
"""

from __future__ import annotations

from ..models import (
    DroneState, WorldSnapshot, RoleAction, ActionKind, DronePhase, Vec3,
)
from .base import RoleContext


class ScoutRole:
    name = "scout"

    def tick(self, drone: DroneState, world: WorldSnapshot, ctx: RoleContext) -> RoleAction:
        if not drone.flying or drone.position is None:
            return RoleAction.none(note="scout (grounded)")

        strat = ctx.config.strategy
        centre = ctx.config.arena.center()                 # Vec2(0, 0)
        alt = strat.scout_altitude_m
        target = Vec3(centre.x, centre.y, alt)

        phase = ctx.memory.get("scout_phase", "goto_centre")

        # Are we centred (xy) and at altitude?
        dxy = drone.position.dist_xy(target)
        dz = abs(target.z - drone.position.z)
        centred = (dxy <= strat.arrival_tol_m) and (dz <= strat.altitude_tol_m)

        # Phase 1: fly to the centre at scout altitude.
        if phase != "scan":
            if centred:
                ctx.memory["scout_phase"] = "scan"
                phase = "scan"
            else:
                ctx.memory["scout_phase"] = "goto_centre"
                return RoleAction(
                    kind=ActionKind.GOTO,
                    target=target,
                    next_phase=DronePhase.TRANSIT,
                    note="scout -> centre",
                )

        # Drifted off the centre while scanning? Re-acquire.
        if not centred:
            ctx.memory["scout_phase"] = "goto_centre"
            return RoleAction(
                kind=ActionKind.GOTO,
                target=target,
                next_phase=DronePhase.TRANSIT,
                note="scout re-centre",
            )

        # Phase 2: rotate in place so the camera sweeps all six boxes.
        return RoleAction(
            kind=ActionKind.YAW_SCAN,
            yaw_rate=strat.scout_yaw_rate_dps,
            target=target,                       # holds the centre + altitude
            next_phase=DronePhase.ORBIT,
            note="scout scan",
        )


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

    cfg = C2Config()
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

    # Start the scout off-centre on the floor edge, climbing.
    drone = DroneState(
        drone_id="scout-1", role=DroneRole.SCOUT, flying=True,
        position=Vec3(-3.0, 4.0, 2.0), heading_deg=0.0,
        base_altitude_m=2.0,
    )
    role = ScoutRole()
    ctx = RoleContext(config=cfg, now=0.0, memory={})

    print("== scout smoke test ==")
    for t in range(5):
        ctx.now = float(t)
        act = role.tick(drone, world, ctx)
        print(f"t={t} phase_mem={ctx.memory.get('scout_phase'):<11} "
              f"{act.kind.value:<9} yaw={act.yaw_rate:>4} "
              f"target={act.target.to_list() if act.target else None} "
              f"-> {act.next_phase.value:<8} | {act.note}")
        # Simulate the nav executing the GOTO: snap to the centre after tick 1.
        if t == 1:
            drone.position = Vec3(0.0, 0.0, cfg.strategy.scout_altitude_m)
