"""marker_mission_sim — a lightweight local drone+arena simulator.

A "Sphinx-lite": it simulates one or more drones behind the SAME HTTP
flight-controller API that ``marker_mission_c2`` already speaks, so each
simulated drone drops into the C2 config exactly like a real flight
controller. It also serves a 3D web view of the arena.

What it simulates:
  * Kinematic flight that executes the FC mission-script DSL (TAKEOFF,
    HEIGHT, TO, FB_RC, FB_BRAKE, RC, UD_RC, YAW_IMU, HOOVER, APPROACH,
    APPROACH_HOME, LAND, …) — the exact verbs the C2 emits. No NEW FC
    verbs are added.
  * A vision model that reports ``visible_marker_ids`` from each drone's
    pose vs the wall markers + target-box faces (with noise + dropout),
    so the C2's slot tracker and AUTO reactions work end-to-end.
  * A capture model: a drone dwelling over an enemy target box flips the
    box's face to its team — the C2 sees the colour change.

Run with ``python -m marker_mission_sim --config sim_config.json``.
Geometry is inherited from ``marker_mission.arena`` so the sim, the C2,
and the real FC all agree on the arena.
"""

__version__ = "0.1.0"
