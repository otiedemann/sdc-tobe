"""Vision model: what marker IDs does a drone's camera see this frame?

Each tick the sim asks every drone for the set of ArUco marker IDs in view.
That list is what the C2's slot tracker / AUTO reactions consume, so this is
the bridge between the simulated world and the real C2 logic.

A marker is reported visible iff ALL of:
  * it is within ``noise.vision_range_m`` of the drone, and
  * it falls inside the horizontal camera FOV (``noise.vision_fov_deg``),
    measured as the absolute wrapped angle between the drone heading and the
    bearing drone->marker, and
  * a per-marker dropout roll passes (``rng.random() >= marker_dropout_prob``)
    so detection flickers a little, like a real camera.

Candidate markers are the 16 wall markers plus each target box's *current*
face id (which flips when the box is captured). Position noise of stddev
``noise.pos_noise_m`` is added to the drone pose before the geometry so
localisation error perturbs detection subtly.

Pure stdlib + package modules. ``visible_marker_ids`` never raises (returns
[] on any unexpected error). The caller already holds ``world.lock``.
"""

from __future__ import annotations

from .geometry import Vec3, heading_vec, heading_of, bearing_to, wrap_deg


def all_marker_positions(world) -> list[tuple[int, Vec3]]:
    """Every candidate marker as ``(id, pos)``.

    Wall markers (static) followed by each target box's *current* face id at
    the box position. The box face id changes with its holder, so a captured
    box advertises its new team's face here automatically.
    """
    out: list[tuple[int, Vec3]] = []
    for m in world.wall_markers:
        out.append((int(m.id), m.pos))
    for box in world.boxes.values():
        out.append((int(box.current_face_id), box.pos))
    return out


def visible_marker_ids(drone, world, now: float) -> list[int]:
    """IDs of markers this drone's camera sees this frame.

    See module docstring for the visibility rules. Returns a sorted list of
    unique ints; never raises.
    """
    try:
        noise = world.cfg.noise
        rng = world.rng

        # Subtle localisation jitter on the drone pose (xy is what matters for
        # range + bearing). Kept at pos_noise_m stddev so detection stays
        # mostly stable frame-to-frame.
        sigma = float(getattr(noise, "pos_noise_m", 0.0) or 0.0)
        if sigma > 0.0:
            obs = Vec3(
                drone.pos.x + rng.gauss(0.0, sigma),
                drone.pos.y + rng.gauss(0.0, sigma),
                drone.pos.z + rng.gauss(0.0, sigma),
            )
        else:
            obs = drone.pos

        range_m = float(noise.vision_range_m)
        half_fov = float(noise.vision_fov_deg) / 2.0
        dropout = float(noise.marker_dropout_prob)

        # Heading the camera points along (deg CW from +Y).
        hx, hy = heading_vec(drone.heading_deg)
        cam_heading = heading_of(hx, hy)

        seen: set[int] = set()
        for mid, mpos in all_marker_positions(world):
            # Range gate (full 3D distance).
            if obs.dist(mpos) > range_m:
                continue
            # FOV gate: wrapped angle between camera heading and bearing.
            bearing = bearing_to(obs, mpos)
            if abs(wrap_deg(bearing - cam_heading)) > half_fov:
                continue
            # Dropout roll (use world.rng so a seed reproduces the run).
            if rng.random() < dropout:
                continue
            seen.add(int(mid))

        return sorted(seen)
    except Exception:
        # Vision must never kill the tick loop.
        return []


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from marker_mission_sim.config import SimConfig
    from marker_mission_sim.world import World

    cfg = SimConfig()
    world = World(cfg)

    # Place a drone a few metres in front of the blue box in slot 1
    # (pos [-3.0, 7.5, 1.0]) looking toward it (+Y, heading 0).
    drone = next(iter(world.drones.values())) if world.drones else None
    if drone is None:
        from marker_mission_sim.world import SimDrone
        drone = SimDrone(id="sim-1", team="red", fc_port=8000, pos=Vec3())
        world.drones["sim-1"] = drone

    box1 = world.boxes[1]
    drone.pos = Vec3(box1.pos.x, box1.pos.y - 3.0, 1.2)
    drone.heading_deg = 0.0   # facing +Y, toward the box / front wall
    drone.flying = True

    print(f"drone @ {drone.pos.to_list()} heading {drone.heading_deg} deg")
    print(f"box1 face id {box1.current_face_id} @ {box1.pos.to_list()}")
    print(f"vision_range_m={cfg.noise.vision_range_m} fov={cfg.noise.vision_fov_deg} "
          f"dropout={cfg.noise.marker_dropout_prob}")

    seen_counts: dict[int, int] = {}
    N = 8
    for i in range(N):
        ids = visible_marker_ids(drone, world, now=float(i))
        for mid in ids:
            seen_counts[mid] = seen_counts.get(mid, 0) + 1
        print(f"  call {i}: visible = {ids}")

    assert box1.current_face_id in seen_counts, "box face should be seen at least once"
    print(f"per-id hit counts over {N} calls: "
          f"{dict(sorted(seen_counts.items()))}")
    print("OK: in-FOV markers detected; dropout causes occasional drops.")
