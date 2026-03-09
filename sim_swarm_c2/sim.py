import json
import math
from dataclasses import asdict
from typing import List

from models import Drone, World, Vec2
from mission import assign_initial_roles, update_mission, safety_separation
from comms import CommsModel


def _move_towards(d: Drone, dt: float, world: World):
    if d.target is None or not d.active:
        return
    dx = d.target.x - d.pos.x
    dy = d.target.y - d.pos.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    step = min(d.speed_mps * dt, dist)
    d.pos.x += dx / dist * step
    d.pos.y += dy / dist * step
    d.pos.x = max(0.0, min(world.width_m, d.pos.x))
    d.pos.y = max(0.0, min(world.height_m, d.pos.y))


def run_sim(config: dict):
    world_cfg = config["world"]
    world = World(
        width_m=world_cfg["width_m"],
        height_m=world_cfg["height_m"],
        home=world_cfg["home"],
        target=world_cfg["target"],
    )
    drones: List[Drone] = [Drone(id=d["id"], pos=d["start"]) for d in config["drones"]]
    comms = CommsModel(**config["comms"])

    assign_initial_roles(drones)

    t = 0.0
    dt = config["dt"]
    duration = config["duration_s"]
    log = []

    mission_success = False
    completion_time = None

    while t < duration:
        # mission updates + comms effects
        for d in drones:
            if d.active and comms.drone_drops():
                d.active = False
                d.state = "DROPOUT"
                d.link_ok = False
                world.link_loss_events += 1

            if not d.active:
                continue

            delivered = comms.command_delivered()
            if not delivered:
                d.link_ok = False
                world.link_loss_events += 1
            else:
                d.link_ok = True
                update_mission(d, world)

        for d in drones:
            if d.active:
                safety_separation(d, drones, world)
                _move_towards(d, dt, world)
                d.battery_pct = max(0.0, d.battery_pct - 0.035)  # simple drain model

                # target capture
                if not world.target_captured and d.pos.dist(world.target) <= world.capture_radius_m:
                    world.target_captured = True
                    d.state = "TRACK"

                # return/land
                if d.state == "RETURN" and d.pos.dist(world.home) < 0.4:
                    d.state = "LAND"

        if world.target_captured:
            all_done = all((not d.active) or d.state in ("LAND", "DROPOUT", "ABORT") for d in drones)
            if all_done:
                mission_success = True
                completion_time = t
                break

        if int(t / dt) % 10 == 0:  # every ~1s
            log.append(
                {
                    "t": round(t, 2),
                    "target_captured": world.target_captured,
                    "drones": [
                        {
                            "id": d.id,
                            "x": round(d.pos.x, 2),
                            "y": round(d.pos.y, 2),
                            "state": d.state,
                            "bat": round(d.battery_pct, 1),
                            "link": d.link_ok,
                        }
                        for d in drones
                    ],
                }
            )

        t += dt

    avg_battery = sum(d.battery_pct for d in drones) / max(1, len(drones))
    result = {
        "mission_success": mission_success,
        "completion_time_s": None if completion_time is None else round(completion_time, 2),
        "near_collisions": world.near_collision_count,
        "link_loss_events": world.link_loss_events,
        "avg_battery_pct": round(avg_battery, 2),
        "events_logged": len(log),
        "final_states": {d.id: d.state for d in drones},
    }
    return result, log
