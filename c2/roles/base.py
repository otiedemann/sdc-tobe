"""Role protocol + shared context for C2 drone behaviours.

A Role is a policy object with a single method::

    action = role.tick(drone, world, ctx)

where:
    drone : models.DroneState   — the drone this role is driving
    world : models.WorldSnapshot — full fused world (slots + all drones)
    ctx   : RoleContext         — config + clock + per-drone scratch memory

It returns a models.RoleAction describing the high-level intent
(GOTO / HOVER / ORBIT / YAW_SCAN / LAND / NONE). navigation.py executes it.

Roles MUST be side-effect-free except for mutating ``ctx.memory`` (their
own scratch dict). They must not perform I/O — that's fc_client's job.
This makes them unit-testable with synthetic world snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any

from ..models import DroneState, WorldSnapshot, RoleAction


@dataclass
class RoleContext:
    """Everything a role needs besides the drone + world.

    ``config`` is the live C2Config (arena geometry, strategy tuning,
    slot map, team). ``now`` is the tick wall-clock. ``memory`` is a
    per-drone scratch dict the commander persists across ticks (use it
    for phase timers, orbit angle, chosen waypoint, etc.).
    """
    config: Any                       # c2.config.C2Config (avoid import cycle)
    now: float
    memory: dict = field(default_factory=dict)


class Role(Protocol):
    """A drone behaviour policy."""

    #: stable name for logging / UI
    name: str

    def tick(self, drone: DroneState, world: WorldSnapshot, ctx: RoleContext) -> RoleAction:
        """Return the next high-level action for ``drone``."""
        ...
