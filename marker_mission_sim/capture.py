"""Target-box capture model — SDC26 regs v7 §1.4.3.

Per tick, for every target box, we work out which of our drones (if any) are
currently "over" it — flying, inside the 1-2 m capture altitude band, slow
enough to count as hovering, and within ``capture.radius_m`` in the xy plane.
From the set of drones over each box we apply the v7 rules:

  * **Capture lock (5 s after a flip)** — while ``now < box.lock_until`` no
    further captures are possible (in either direction). Drone dwell timers
    over this box are reset so they don't carry across the lock window.
  * **Home-zone recapture is instant** — if any drone of the box's *home*
    team is over a box that is currently NOT in the home colour, the box
    flips back to the home colour immediately (no 2 s hover). This is worth
    **0 game points** — the scoring side awards none for self-recovery —
    but the colour change is mechanical here.
  * **Defender blocks capture** — if any home-team drone is over the box,
    enemy drones cannot accumulate dwell on it (their timers reset). v7
    requires that "no defending drone is detected in that box" for an
    enemy capture to count.
  * **Enemy capture requires >=2 s hover** — accumulate per-drone dwell on
    enemy boxes; first attacker to reach ``capture.hold_s`` flips the box
    to its team and starts the 5 s lock.

The whole function is wrapped in defensive try/excepts so a bad drone state
cannot stop the tick. Returns the (often empty) list of human-readable event
strings produced this tick. The C2 marker tracker independently observes
the resulting marker-ID changes through its drones' vision and updates its
own lock state from there.
"""

from __future__ import annotations


def _drones_over_box(world, box, z_min, z_max, max_speed, radius):
    out = []
    for d in world.drones.values():
        if (d.flying
                and z_min <= d.pos.z <= z_max
                and d.speed_mps <= max_speed
                and d.pos.dist_xy(box.pos) <= radius):
            out.append(d)
    return out


def _reset_dwell_for_slot(world, slot: int) -> None:
    for d in world.drones.values():
        if d.capture_slot == slot:
            d.capture_slot = None
            d.capture_since = 0.0


def update(world, dt: float, now: float) -> list[str]:
    """Advance v7 capture state for every box; return flip events this tick."""
    events: list[str] = []
    try:
        cap = world.cfg.capture
        radius = float(cap.radius_m)
        z_min = float(cap.z_min_m)
        z_max = float(cap.z_max_m)
        hold_s = float(cap.hold_s)
        max_speed = float(cap.max_speed_mps)
        lock_s = float(cap.post_capture_lock_s)

        for box in world.boxes.values():
            try:
                # ---- 5-s post-capture lock --------------------------------
                if now < box.lock_until:
                    _reset_dwell_for_slot(world, box.slot)
                    continue

                drones_over = _drones_over_box(
                    world, box, z_min, z_max, max_speed, radius)
                home_team = box.home_team
                home_drones = [d for d in drones_over if d.team == home_team]
                enemy_drones = [d for d in drones_over if d.team != home_team]

                # ---- home-zone INSTANT recapture (0 pts, but box flips) ---
                if home_drones and box.holder != home_team:
                    box.holder = home_team
                    box.last_flip_unix_s = now
                    box.lock_until = now + lock_s
                    events.append(
                        f"{home_drones[0].id} ({home_team}) home-recap "
                        f"slot {box.slot} -> {home_team} (instant, 0 pts)"
                    )
                    _reset_dwell_for_slot(world, box.slot)
                    continue

                # ---- defender presence blocks any enemy capture -----------
                if home_drones:
                    # Reset enemy dwells: a defender showing up cancels their
                    # in-progress hover for this slot.
                    for d in enemy_drones:
                        if d.capture_slot == box.slot:
                            d.capture_slot = None
                            d.capture_since = 0.0
                    # Home drone just patrols / holds presence — no flip.
                    continue

                # ---- enemy capture attempt (>=2 s hover, no defender) -----
                flipped = False
                for d in enemy_drones:
                    if box.holder == d.team:
                        # Box is already this drone's colour (rare edge);
                        # nothing to flip and no dwell to track.
                        if d.capture_slot == box.slot:
                            d.capture_slot = None
                            d.capture_since = 0.0
                        continue
                    # Start or check this drone's dwell on this slot.
                    if d.capture_slot != box.slot:
                        d.capture_slot = box.slot
                        d.capture_since = now
                    elif (now - d.capture_since) >= hold_s:
                        box.holder = d.team
                        box.last_flip_unix_s = now
                        box.lock_until = now + lock_s
                        events.append(
                            f"{d.id} ({d.team}) captured slot {box.slot} "
                            f"-> {d.team} (face {box.current_face_id})"
                        )
                        _reset_dwell_for_slot(world, box.slot)
                        flipped = True
                        break
                # If no flip this tick, dwell continues to accumulate for the
                # drones over this box; drones that LEFT the box have their
                # dwell cleared via the per-drone "not over any box" path below.
                if flipped:
                    continue
            except Exception:
                continue

        # Final safety: any drone whose ``capture_slot`` points at a slot it
        # is no longer over (e.g. flew away) gets its dwell cleared.
        # (Per-slot resets above already handle most cases; this catches the
        # "drifted off the box but no other event fired" case.)
        for d in world.drones.values():
            sl = d.capture_slot
            if sl is None:
                continue
            box = world.boxes.get(sl)
            if box is None:
                d.capture_slot = None
                d.capture_since = 0.0
                continue
            still_over = (d.flying and z_min <= d.pos.z <= z_max
                          and d.speed_mps <= max_speed
                          and d.pos.dist_xy(box.pos) <= radius)
            if not still_over:
                d.capture_slot = None
                d.capture_since = 0.0
    except Exception:
        return events
    return events
