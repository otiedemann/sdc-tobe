"""C2 V2 — a clean swarm Ground Control Station for the Swarm Drone Challenge.

A from-scratch strategy brain that connects DIRECTLY to the five real flight
controllers (flightctrl1-5:8080), reads their live state, and (in later phases)
commands them to win the v7 Capture-the-Flag game. ``marker_mission`` (the FC)
is never touched.

It deliberately REUSES the battle-proven primitives from ``marker_mission_c2``
by import only:
  * ``marker_mission_c2.fc_client.AsyncFCClient`` — the debugged per-FC HTTP client
  * ``marker_mission_c2.config.FCSpec`` — the FC endpoint spec
  * ``marker_mission_c2.strategy.arena_state`` — live arena geometry (real/gvz)
  * ``marker_mission_c2.strategy.markers`` — the box-state tracker
  * ``marker_mission_c2.strategy.settings`` face-id helpers

and rebuilds the part that was tangled in V1: the per-drone role state machines
and the single swarm COORDINATOR that plays for the 5/10/Special scoring tiers
and never lands the drones except on the '?' emergency-land.

Phases (each shipped + verified separately):
  1. FC client + live per-drone world + read-only dashboard   <-- this commit
  2. Box-state + arena + 6-box status panel + team/arena toggles
  3. Roles + the never-land chained-script mechanism
  4. The coordinator (unified assignment + 5/10/Special plays + recovery)
  5. Arming, '?' kill-all-land, C2-link-loss safety, tuning UI, verify
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
