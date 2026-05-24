"""Drone role behaviours for the SDC26 C2.

Each role is a pure-ish policy: given a drone's state and the world
snapshot, it returns a high-level RoleAction. navigation.py turns that
action into concrete RC commands, and fc_client.py performs the I/O.
This keeps the roles testable without a live drone.

Public factory: ``make_role(role: DroneRole) -> Role``.
"""

from __future__ import annotations

from ..models import DroneRole
from .base import Role, RoleContext


def make_role(role: DroneRole) -> "Role":
    # Import ONLY the requested role's module, lazily, so a missing or
    # broken sibling role file doesn't break the others.
    if role == DroneRole.SCOUT:
        from .scout import ScoutRole
        return ScoutRole()
    if role == DroneRole.ATTACKER:
        from .attacker import AttackerRole
        return AttackerRole()
    if role == DroneRole.DEFENDER:
        from .defender import DefenderRole
        return DefenderRole()
    from .idle import IdleRole
    return IdleRole()


__all__ = ["Role", "RoleContext", "make_role"]
