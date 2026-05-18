"""Swarm strategy for SDC26.

A lightweight orchestration layer that sits next to marker_mission_c2 (the
main C2 server) and drives multiple drones into role-based behaviours
(SCOUT, ATTACKER, ...).

The C2 server (marker_mission_c2.server) remains the single source of truth
for FC connections, mission scripts and per-drone state. The strategy
layer talks to it via plain HTTP (see ``c2_client``).

Public entry-point: ``python -m marker_mission_c2.strategy``.
"""

__all__ = ["app", "settings", "c2_client", "markers", "roles", "runner"]
