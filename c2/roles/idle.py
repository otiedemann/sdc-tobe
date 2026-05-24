"""Idle role — hold position (or do nothing if landed).

Used for drones with no assignment. Keeps them hovering at their base
altitude so they remain controllable and out of the way.
"""

from __future__ import annotations

from ..models import (
    DroneState, WorldSnapshot, RoleAction, ActionKind, DronePhase, Vec3,
)
from .base import RoleContext


class IdleRole:
    name = "idle"

    def tick(self, drone: DroneState, world: WorldSnapshot, ctx: RoleContext) -> RoleAction:
        if not drone.flying or drone.position is None:
            return RoleAction.none(note="idle (grounded)")
        # Hold current xy at the assigned base altitude.
        hold = Vec3(drone.position.x, drone.position.y, drone.base_altitude_m)
        return RoleAction(
            kind=ActionKind.GOTO,
            target=hold,
            next_phase=DronePhase.HOLDING,
            note="idle hold",
        )
