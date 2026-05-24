"""SDC26 Command & Control (C2) strategy server.

A team-aware C2 that drives 2-5 drones (each fronted by its own
controller_unified ``unified_api_server`` instance) in the Swarm Drone
Challenge "capture the target boxes" game.

Design goals (see README.md):
  * MANUAL mode  — operator issues every command (takeoff, assign role,
                   attack slot N, defend, scout, land) for testing.
  * AUTO mode    — the commander reacts to world events (e.g. an enemy
                   captures one of our target slots -> dispatch the
                   nearest defender to re-capture it).
  * Always surface the live state of all six target slots.
  * One scout drone orbits the arena centre to keep the slot states fresh.
  * Each drone flies at a distinct base altitude to avoid mid-air
    collisions.
  * The Pi-side ``arena_safety`` guard is disabled on every drone at
    startup (the C2 owns boundary keeping).

Layout is inherited from ``marker_mission.arena`` so the C2, the
simulator build and the position estimator all agree on geometry.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
