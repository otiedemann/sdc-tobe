"""Capture model: a drone dwelling over an enemy box flips it to its team.

Called once per tick from ``World.tick`` (already holding ``world.lock``).
For each drone we find the nearest box it is currently "over" — meaning it is
flying, inside the capture altitude band, slow enough, and within
``capture.radius_m`` in the xy plane. If that box belongs to the enemy and the
drone holds position over it for ``capture.hold_s`` seconds, the box flips to
the drone's team (its advertised face id changes with the holder, so the C2
sees the colour change through the vision model).

Dwell bookkeeping lives on the drone (``capture_slot`` / ``capture_since``,
owned here). Moving off a box, going too fast, leaving the altitude band, or
switching to a different box resets the dwell. ``update`` never raises.
"""

from __future__ import annotations


def update(world, dt: float, now: float) -> list[str]:
    """Advance capture dwell for every drone; return flip events this tick.

    See module docstring for the rules. Returns the (often empty) list of
    human-readable event strings produced this tick; never raises.
    """
    events: list[str] = []
    try:
        cap = world.cfg.capture
        radius = float(cap.radius_m)
        z_min = float(cap.z_min_m)
        z_max = float(cap.z_max_m)
        hold_s = float(cap.hold_s)
        max_speed = float(cap.max_speed_mps)

        for drone in world.drones.values():
            try:
                # Find the nearest box this drone currently qualifies to capture.
                best_box = None
                best_d = None
                if drone.flying and (z_min <= drone.pos.z <= z_max) \
                        and (drone.speed_mps <= max_speed):
                    for box in world.boxes.values():
                        d_xy = drone.pos.dist_xy(box.pos)
                        if d_xy <= radius and (best_d is None or d_xy < best_d):
                            best_d = d_xy
                            best_box = box

                # Not over any capturable box -> reset dwell.
                if best_box is None:
                    drone.capture_slot = None
                    drone.capture_since = 0.0
                    continue

                # Already ours -> nothing to capture; reset dwell.
                if best_box.holder == drone.team:
                    drone.capture_slot = None
                    drone.capture_since = 0.0
                    continue

                # Enemy box: accumulate dwell. (Re)start the timer if we just
                # arrived over this slot.
                if drone.capture_slot != best_box.slot:
                    drone.capture_slot = best_box.slot
                    drone.capture_since = now

                # Held long enough -> flip the box to this drone's team.
                if (now - drone.capture_since) >= hold_s:
                    best_box.holder = drone.team
                    best_box.last_flip_unix_s = now
                    drone.capture_slot = None
                    drone.capture_since = 0.0
                    events.append(
                        f"{drone.id} ({drone.team}) captured slot "
                        f"{best_box.slot} -> {drone.team} "
                        f"(face {best_box.current_face_id})"
                    )
            except Exception:
                # One drone's error must not stop the others or the tick.
                continue
    except Exception:
        return events
    return events


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from marker_mission_sim.config import SimConfig
    from marker_mission_sim.world import World, SimDrone
    from marker_mission_sim.geometry import Vec3

    cfg = SimConfig()
    world = World(cfg)

    # Blue box in slot 1 at [-3.0, 7.5, 1.0]; a RED drone hovering over it.
    box1 = world.boxes[1]
    assert box1.holder == "blue", f"expected blue home box, got {box1.holder}"

    drone = SimDrone(
        id="red-1", team="red", fc_port=8000,
        pos=Vec3(box1.pos.x, box1.pos.y, 1.5),  # directly over, in z band
        heading_deg=0.0,
    )
    drone.flying = True
    drone.speed_mps = 0.0
    world.drones[drone.id] = drone

    hold_s = cfg.capture.hold_s
    print(f"box1 slot {box1.slot} holder={box1.holder} face={box1.current_face_id}")
    print(f"red drone @ {drone.pos.to_list()} speed {drone.speed_mps} "
          f"hold_s={hold_s} radius={cfg.capture.radius_m} "
          f"z=[{cfg.capture.z_min_m},{cfg.capture.z_max_m}]")

    all_events: list[str] = []
    # Tick repeatedly, advancing `now` until past hold_s.
    t = 0.0
    dt = 0.5
    steps = int(hold_s / dt) + 3
    for _ in range(steps):
        evs = update(world, dt=dt, now=t)
        if evs:
            all_events.extend(evs)
            print(f"  t={t:.1f}: {evs}")
        else:
            print(f"  t={t:.1f}: holder={box1.holder} "
                  f"dwell={t - drone.capture_since:.1f}s")
        t += dt

    assert box1.holder == "red", f"box should have flipped to red, got {box1.holder}"
    assert box1.current_face_id == box1.red_face_id, "face id should be the red face"
    assert all_events, "a capture event should have been returned"
    print(f"OK: box flipped blue->red, face now {box1.current_face_id}; "
          f"event: {all_events[-1]!r}")

    # A second pass should NOT re-capture (already ours).
    more = update(world, dt=dt, now=t)
    assert not more, f"should not re-capture an own box, got {more}"
    print("OK: own box is not re-captured.")
