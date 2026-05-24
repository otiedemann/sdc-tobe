"""Fleet assignment helpers for the SDC26 C2 commander.

Three small, *pure-ish* helpers the commander uses to set up and re-balance
the fleet. They depend only on :mod:`c2.models` and :mod:`c2.config` so this
module imports cleanly even before the rest of the C2 (fc_client, navigation,
world_model, roles) is finished.

  * :func:`assign_altitude_bands` — give every drone a DISTINCT cruise
    altitude so two drones never share a height (collision deconfliction).
  * :func:`default_roles` — pick a sensible starting role per drone
    (exactly one scout, the rest biased toward attackers).
  * :func:`pick_uncap_defender` — choose the best free defender to send to
    re-capture (un-cap) one of our slots the enemy just flipped.

None of these perform I/O. ``assign_altitude_bands`` mutates the passed-in
``DroneState`` objects in place (that is its documented job); the others are
read-only and return their decision.
"""

from __future__ import annotations

import math
from typing import Optional

from .config import C2Config
from .models import (
    DroneRole,
    DroneState,
    SlotColor,
    Vec3,
    WorldSnapshot,
)

# Lowest cruise altitude we will ever assign to a non-scout drone. Kept above
# the 1-2 m capture/detection band so a *cruising* drone is clearly distinct
# from one that has descended to flip a box.
MIN_BAND_ALTITUDE_M = 1.8

# Vertical separation we aim for between adjacent cruise bands. We compress
# below this only when there are too many drones to fit between
# MIN_BAND_ALTITUDE_M and the scout altitude at the full step.
PREFERRED_BAND_STEP_M = 0.5


# ---------------------------------------------------------------------------
# Altitude deconfliction
# ---------------------------------------------------------------------------

def assign_altitude_bands(drones: list[DroneState], cfg: C2Config) -> None:
    """Give every drone a distinct cruise altitude (mutates in place).

    Rules:
      * The SCOUT (if any) gets the highest band,
        ``cfg.strategy.scout_altitude_m`` — it wants the widest view and to
        stay clear of the capture traffic below.
      * Every other drone gets a distinct altitude spread in ~0.5 m steps
        starting at ``MIN_BAND_ALTITUDE_M`` and staying strictly *below* the
        scout altitude. Order is deterministic by sorted ``drone_id`` so the
        same fleet always lands on the same bands (reproducible runs).

    The result is collision-avoidance by construction: no two drones share a
    cruise height, so the closed-loop nav can fly them to xy waypoints
    without them stacking on top of each other.
    """
    if not drones:
        return

    scout_alt = float(cfg.strategy.scout_altitude_m)

    # Deterministic order.
    ordered = sorted(drones, key=lambda d: d.drone_id)

    scouts = [d for d in ordered if d.role == DroneRole.SCOUT]
    others = [d for d in ordered if d.role != DroneRole.SCOUT]

    # The scout owns the top band.
    for d in scouts:
        d.base_altitude_m = scout_alt

    n = len(others)
    if n == 0:
        return

    # Spread the remaining drones between MIN_BAND_ALTITUDE_M and just under
    # the scout altitude. We need n distinct values; reserve a small gap below
    # the scout so a non-scout never collides with it.
    top = scout_alt - PREFERRED_BAND_STEP_M  # highest non-scout band
    if not scouts:
        # No scout in this fleet: we may use the whole range up to the scout
        # altitude itself for the topmost drone.
        top = scout_alt

    lo = MIN_BAND_ALTITUDE_M
    if top <= lo:
        # Degenerate geometry (scout altitude basically at the floor band):
        # fall back to fixed 0.5 m steps from lo so values stay distinct.
        for i, d in enumerate(others):
            d.base_altitude_m = round(lo + i * PREFERRED_BAND_STEP_M, 3)
        return

    if n == 1:
        # A single non-scout sits at the bottom band (predictable + low).
        others[0].base_altitude_m = round(lo, 3)
        return

    # n >= 2: evenly space n drones across [lo, top] inclusive. The natural
    # step is (top-lo)/(n-1); if that exceeds the preferred 0.5 m we cap it so
    # bands stay tight (no point flying one drone way up high for no reason).
    span = top - lo
    step = min(PREFERRED_BAND_STEP_M, span / (n - 1))
    for i, d in enumerate(others):
        d.base_altitude_m = round(lo + i * step, 3)


# ---------------------------------------------------------------------------
# Default role assignment
# ---------------------------------------------------------------------------

def default_roles(drone_ids: list[str], cfg: C2Config) -> dict[str, DroneRole]:
    """Pick a starting role for each drone id.

    Policy (kept deliberately simple + deterministic):
      * Exactly ONE scout: the first drone by sorted id. It orbits the centre
        keeping every slot's colour fresh for the world model.
      * Of the remaining drones we bias toward ATTACKERS (offence wins the
        capture game): ``ceil(remaining / 2)`` attackers, the rest defenders.

    Worked examples::

        1 drone  -> {a: scout}
        2 drones -> {a: scout, b: attacker}            (1 left -> ceil(0.5)=1 atk)
        3 drones -> {a: scout, b: attacker, c: attacker}
        4 drones -> {a: scout, b: attacker, c: attacker, d: defender}
        5 drones -> scout + 2 attackers + 2 defenders
    """
    roles: dict[str, DroneRole] = {}
    if not drone_ids:
        return roles

    ordered = sorted(drone_ids)

    # First drone is the scout.
    roles[ordered[0]] = DroneRole.SCOUT

    rest = ordered[1:]
    n_rest = len(rest)
    if n_rest == 0:
        return roles

    n_attackers = math.ceil(n_rest / 2)
    for i, did in enumerate(rest):
        roles[did] = DroneRole.ATTACKER if i < n_attackers else DroneRole.DEFENDER

    return roles


# ---------------------------------------------------------------------------
# Defender selection for un-capping a flipped slot
# ---------------------------------------------------------------------------

def _slot_xy(slot: int, world: WorldSnapshot, cfg: C2Config) -> Optional[Vec3]:
    """Best-known position of a slot: last observed, else nominal, else None."""
    ts = world.slots.get(slot)
    if ts is not None and ts.position is not None:
        return ts.position
    return cfg.nominal_slot_positions.get(slot)


def pick_uncap_defender(
    slot: int, world: WorldSnapshot, cfg: C2Config
) -> Optional[str]:
    """Choose the best DEFENDER to send to un-cap ``slot`` (or None).

    A candidate defender must be:
      * role == DEFENDER,
      * flying (it can actually respond),
      * not already committed to a *different* threatened slot — i.e. its
        ``assigned_slot`` is None, or already this slot. (A defender already
        un-capping another box is left where it is.)

    Among the candidates we pick the one nearest (xy) to the slot position so
    the response is as fast as possible. A defender already assigned to *this*
    slot is the natural winner (distance is moot — it's on the job already),
    but we still return it so the caller's assignment is idempotent.

    Returns the chosen ``drone_id`` or ``None`` if no suitable defender exists.
    """
    target = _slot_xy(slot, world, cfg)

    candidates: list[tuple[float, str]] = []
    for did, d in world.drones.items():
        if d.role != DroneRole.DEFENDER:
            continue
        if not d.flying:
            continue
        # Already busy un-capping a *different* slot -> skip.
        if d.assigned_slot is not None and d.assigned_slot != slot:
            continue

        # Distance metric: prefer the one already on this slot, then nearest.
        if d.assigned_slot == slot:
            dist = -1.0  # already committed here -> top priority
        elif d.position is not None and target is not None:
            dist = d.position.dist_xy(target)
        else:
            # No position fix and not yet assigned here: usable but lowest
            # priority (we can't measure how far it is).
            dist = math.inf
        candidates.append((dist, did))

    if not candidates:
        return None

    # Nearest wins; ties broken by drone_id for determinism.
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Standalone demo (no drones, no siblings required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from c2.config import C2Config
    from c2.models import (
        DroneRole,
        DroneState,
        GameMode,
        TargetSlot,
        Vec3,
        WorldSnapshot,
        now_s,
    )

    cfg = C2Config()  # our_team RED by default
    ids = ["delta", "alpha", "charlie", "bravo", "echo"]

    print("== default_roles ==")
    for k in range(1, len(ids) + 1):
        subset = ids[:k]
        roles = default_roles(subset, cfg)
        pretty = {d: roles[d].value for d in sorted(subset)}
        print(f"  {k} drones: {pretty}")

    print("\n== assign_altitude_bands (5 drones) ==")
    roles5 = default_roles(ids, cfg)
    drones = [
        DroneState(drone_id=d, role=roles5[d], flying=True)
        for d in ids
    ]
    assign_altitude_bands(drones, cfg)
    for d in sorted(drones, key=lambda x: x.base_altitude_m):
        print(f"  {d.drone_id:<8} {d.role.value:<9} -> {d.base_altitude_m:.3f} m")
    alts = [d.base_altitude_m for d in drones]
    assert len(set(alts)) == len(alts), f"altitudes not distinct: {alts}"
    print("  all altitudes distinct: OK")

    print("\n== pick_uncap_defender ==")
    # our_team RED -> our slots are 4,5,6. Enemy just flipped slot 5.
    slot = 5
    slot_pos = cfg.nominal_slot_positions[slot]
    slots = {
        i: TargetSlot(
            index=i,
            home_team=cfg.slot_map.home_team_of(i),
            color=SlotColor.from_team(cfg.slot_map.home_team_of(i)),
            position=cfg.nominal_slot_positions[i],
        )
        for i in range(1, 7)
    }
    # Place the world drones: two defenders at different distances, plus a busy
    # one already assigned elsewhere, plus a (non-eligible) attacker.
    wm_drones = {
        "bravo": DroneState(  # near defender, free
            drone_id="bravo", role=DroneRole.DEFENDER, flying=True,
            position=Vec3(slot_pos.x + 0.5, slot_pos.y + 0.5, 2.2),
        ),
        "charlie": DroneState(  # far defender, free
            drone_id="charlie", role=DroneRole.DEFENDER, flying=True,
            position=Vec3(slot_pos.x + 4.0, slot_pos.y - 3.0, 2.2),
        ),
        "delta": DroneState(  # defender already busy on slot 4 -> skipped
            drone_id="delta", role=DroneRole.DEFENDER, flying=True,
            assigned_slot=4, position=Vec3(slot_pos.x, slot_pos.y, 2.2),
        ),
        "echo": DroneState(  # attacker -> not eligible
            drone_id="echo", role=DroneRole.ATTACKER, flying=True,
            position=Vec3(slot_pos.x, slot_pos.y, 2.0),
        ),
    }
    world = WorldSnapshot(
        t=now_s(), mode=GameMode.AUTO, our_team=cfg.our_team,
        drones=wm_drones, slots=slots,
    )
    chosen = pick_uncap_defender(slot, world, cfg)
    print(f"  enemy flipped slot {slot}; nearest free defender -> {chosen!r}")
    assert chosen == "bravo", chosen

    # If bravo is already assigned to this slot, it stays the winner.
    wm_drones["bravo"].assigned_slot = slot
    chosen2 = pick_uncap_defender(slot, world, cfg)
    print(f"  bravo already on slot {slot}: idempotent pick -> {chosen2!r}")
    assert chosen2 == "bravo", chosen2

    # No eligible defenders -> None.
    empty_world = WorldSnapshot(
        t=now_s(), mode=GameMode.AUTO, our_team=cfg.our_team,
        drones={"echo": wm_drones["echo"]}, slots=slots,
    )
    print(f"  no free defender -> {pick_uncap_defender(slot, empty_world, cfg)!r}")
    assert pick_uncap_defender(slot, empty_world, cfg) is None

    print("\nOK assignments demo")
