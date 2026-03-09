from models import Drone, World, Vec2


def assign_initial_roles(drones):
    for d in drones:
        d.state = "SEARCH"


def update_mission(d: Drone, world: World):
    if not d.active or d.state == "DROPOUT":
        return

    if d.battery_pct <= 15 and d.state not in ("RETURN", "LAND", "ABORT"):
        d.state = "RETURN"
        d.target = world.home

    if world.target_captured and d.state not in ("LAND", "ABORT", "DROPOUT"):
        d.state = "RETURN"
        d.target = world.home

    if d.state == "SEARCH":
        d.target = world.target
    elif d.state == "TRACK":
        d.target = world.target
    elif d.state == "RETURN":
        d.target = world.home


def safety_separation(d: Drone, others, world: World):
    # Very simple repulsion: if too close, nudge away
    for o in others:
        if o.id == d.id or not o.active:
            continue
        dist = d.pos.dist(o.pos)
        if dist < world.min_sep_m:
            world.near_collision_count += 1
            # nudge in opposite axis direction
            if d.pos.x >= o.pos.x:
                d.pos.x += 0.05
            else:
                d.pos.x -= 0.05
            if d.pos.y >= o.pos.y:
                d.pos.y += 0.05
            else:
                d.pos.y -= 0.05
