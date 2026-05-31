"""SDC26 v7 play planner — picks the highest-EV play available each tick.

The planner is a pure function:

    PlayAssignment[] = plan(settings, markers, overview, role_states, now)

It enumerates the FEASIBLE v7 plays given the live world (slot holders, lock
windows, drone in-home presence, FC INIT state, etc.) and returns the list
of (drone, slot) assignments the runner should apply. The runner then calls
``assign_target`` and the per-drone role builds the actual FC mission script
(attacker: 1-pt / 10-pt; defender: re-capture).

v7 scoring (regs §1.4.3):
    1 pt   one drone captures + returns home
    5 pts  ALL of our drones outside home at capture moment, then all return
   10 pts  two drones flip two different enemy boxes within <1 s
   instant special: all six boxes our colour for >=5 s -> match won

Priority (highest first):
   defensive recap  > 10-pt double-strike > 1-pt singleton > idle

5-pt full sortie is not scheduled here yet (it needs whole-team timed
choreography; flagged for follow-up). 10-pt sync uses the "push both at the
same tick" approximation — adequate when both attackers' flight profiles
are similar (they spawn at symmetric home positions). A tighter ETA-based
dispatcher can replace this without changing the planner's public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .roles import in_home_zone


def slot_home_team(slot: int) -> str:
    """v7 §1.4.2: boxes 1-3 in RED home, 4-6 in BLUE home."""
    return "red" if 1 <= slot <= 3 else "blue"


@dataclass(frozen=True)
class PlayAssignment:
    fc_name: str
    slot: Optional[int]      # target slot to attack/defend; None clears
    play_kind: str           # "defend_uncap" | "double_strike_10" | "single_strike_1"
    reason: str              # human-readable, surfaces in the event log


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _drone_state(overview: dict, fc: str) -> dict:
    item = overview.get(fc) or {}
    return (item.get("state") or {}) if isinstance(item, dict) else {}


def _fc_ready_to_start(overview: dict, fc: str) -> bool:
    """FC must be back in INIT to accept a new mission start."""
    phase = _drone_state(overview, fc).get("phase") or "unknown"
    return phase in ("init", "done", "")


def _drone_in_home(overview: dict, fc: str, team: str) -> bool:
    pos = _drone_state(overview, fc).get("world_position_m")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False
    try:
        return in_home_zone(team, float(pos[0]), float(pos[1]))
    except (TypeError, ValueError):
        return False


def _is_free(rs: Any) -> bool:
    """Drone has no current target and isn't mid-run."""
    if rs is None:
        return False
    if rs.target_slot is not None:
        return False
    return rs.phase in ("", "idle", "done")


def _drone_xy(overview: dict, fc: str):
    """(x, y) of drone fc from the C2 overview, or None if unknown."""
    pos = _drone_state(overview, fc).get("world_position_m")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        return (float(pos[0]), float(pos[1]))
    except (TypeError, ValueError):
        return None


def _dist_to_slot(overview: dict, fc: str, slot: int) -> float:
    """Horizontal distance from drone fc to a slot (inf if pos unknown).

    Used to pick the NEAREST free attacker to pull into a recapture so the
    "attacker becomes defender" handoff costs the least flight time.
    """
    import math as _m
    from .attacker import SLOT_POSITIONS_M as _SP
    xy = _drone_xy(overview, fc)
    if xy is None:
        return float("inf")
    sx, sy = _SP.get(int(slot), (0.0, 0.0))
    return _m.hypot(xy[0] - sx, xy[1] - sy)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def plan(settings, markers, overview: dict, role_states: dict, now: float,
         team_phase: str = "sortie") -> list[PlayAssignment]:
    """Pick the v7 plays to schedule this tick.

    Parameters
    ----------
    settings : StrategySettings
        The snapshot of operator-set drones/teams/roles + match settings.
    markers : MarkerTracker
        Live slot holders + lock windows.
    overview : dict[fc, {connection_ok, drone_connected, state: {phase,
        world_position_m, visible_marker_ids, ...}}]
        Latest C2 overview (one entry per FC).
    role_states : dict[fc, RoleState]
        Per-drone role/phase/target.
    now : float
        Wall-clock seconds (``time.time()``).

    Returns
    -------
    A list of PlayAssignment instances the runner should apply via
    ``assign_target``. Empty if there's nothing useful to schedule.
    """
    our = settings.markers.our_team
    enemy = "blue" if our == "red" else "red"
    active = [int(x) for x in settings.markers.active_slots]

    # Two belief views:
    #  * DECAYED (effective_holder): a slot unseen for HOLDER_TRUST_S reads
    #    "unknown" so we re-check enemy boxes instead of trusting a stale
    #    "we hold it". Used for ATTACK target selection.
    #  * RAW (slot_holder): the last face we actually observed, persists until
    #    re-observed. Used for OUR-OWN-slot DEFENCE — if the last thing we saw
    #    on our box was the enemy face, it stays "threatened" until a drone
    #    sees it flipped back. (Decaying a threatened own box to "unknown"
    #    would make us STOP defending it — the exact bug where blue boxes sat
    #    un-recaptured in red's home.)
    holder = {sl: markers.effective_holder(sl, now=now) for sl in active}
    raw_holder = {sl: markers.slot_holder(sl) for sl in active}
    locked = {sl for sl in active if markers.slot_locked(sl, now)}

    # Slots an in-flight (non-free) drone of ours is already heading to. We
    # must not assign these to a second drone — but the moment that drone
    # finishes its run and clears its target, the slot is up for grabs again
    # (which, with the decay above, is what re-launches the next capture run).
    targeted = {rs.target_slot for rs in role_states.values()
                if rs is not None and rs.target_slot is not None}

    our_slots = sorted(sl for sl in active if slot_home_team(sl) == our)
    enemy_slots = sorted(sl for sl in active if slot_home_team(sl) == enemy)

    # Free drones, partitioned by operator-set role. A drone is "free" only
    # when it has no target, isn't mid-run, AND can actually start a new
    # mission right now (FC INIT + in own home zone per v7 §1.4.4).
    # MUST also be on OUR team — both strategies (red + blue) auto-adopt
    # every FC, so without this filter the red planner would happily
    # dispatch blue drones (and vice-versa).
    # Two eligibility tiers:
    #
    #  RECAPTURE-eligible (defence): our team, FC-ready (init/done), not
    #  mid-run, no current target. NO in-home gate — a home recapture is a
    #  0-pt defensive flip the drone flies INTO home to do (the §1.4.4
    #  home-presence gate only guards new SCORING attempts on enemy boxes),
    #  and the defender now waits in the NEUTRAL zone so it would never be
    #  "in home". Role-agnostic: a defender does this by default, but a free
    #  attacker is pulled in too ("attacker becomes defender" near a flipped
    #  own box, then reverts to attacking) — this is what the operator asked
    #  for. We keep them split so we can PREFER the defender.
    #
    #  ATTACK-eligible (scoring): the above PLUS in-home (§1.4.4) — only
    #  attackers, and only while not banking.
    recap_defenders: list[str] = []   # free defenders, any zone
    recap_attackers: list[str] = []   # free attackers, any zone
    free_attackers: list[str] = []    # free attackers, in-home (for scoring)
    for d in settings.drones:
        if not d.enabled or not d.team:
            continue
        if d.team != our:
            continue   # other team's drone — not ours to command
        rs = role_states.get(d.fc_name)
        if not _is_free(rs):
            continue
        if not _fc_ready_to_start(overview, d.fc_name):
            continue
        if d.role == "defender":
            recap_defenders.append(d.fc_name)
        elif d.role == "attacker":
            recap_attackers.append(d.fc_name)
            if _drone_in_home(overview, d.fc_name, d.team):
                free_attackers.append(d.fc_name)
        # scouts/idle: never participate in scoring/defence plays

    assignments: list[PlayAssignment] = []
    taken: set[int] = set()

    # ----- (0) Special-Maneuver hold — don't break our own all-ours -----
    # When we already hold all 6 slots, the v7 §1.4.3 Special Maneuver
    # ("sudden-death") win is just 5 s of dwell away. Pushing new attack
    # scripts during this window would (a) waste battery, (b) leave home
    # under-defended, (c) risk a flip-back race if the enemy ALSO arrives.
    # Best behaviour: emit no new assignments and let the marker tracker's
    # all-ours timer expire into a match_won event. The defender role's
    # existing patrol script keeps drones present in the RFID zones,
    # blocking any in-flight enemy capture during the 5-s window.
    all_ours = bool(active) and all(holder.get(sl) == our for sl in active)
    if all_ours:
        return []  # let the win timer expire — no further dispatch

    # ----- (1) Defensive re-capture — HIGHEST priority, runs every tick -----
    # An own slot whose LAST-OBSERVED face is the enemy's is captured and stays
    # threatened until a drone sees it flipped back (raw_holder, not the decayed
    # view — see above). Recapture it instantly: 0 pts, but it stops the enemy
    # bleeding us and is required to ever reach the all-6 instant win. Skip
    # slots inside the 5-s post-capture lock and slots already being handled.
    #
    # Assignment: the free DEFENDER goes first (its whole job), then the
    # NEAREST free attacker is pulled in for any remaining threats — i.e. an
    # attacker near one of our flipped boxes becomes a defender for one run,
    # then reverts to attacking. Neither needs to be in-home (0-pt defence).
    threatened = [sl for sl in our_slots
                  if raw_holder.get(sl) == enemy
                  and sl not in locked and sl not in taken]
    # Order threats so the closest-to-any-recapturer is handled first.
    recap_pool = list(recap_defenders) + list(recap_attackers)
    for sl in threatened:
        if not recap_pool:
            break
        # Prefer a defender if one is still free; else the nearest attacker.
        free_def = [fc for fc in recap_defenders if fc in recap_pool]
        if free_def:
            fc = free_def[0]
        else:
            fc = min(recap_pool, key=lambda f: _dist_to_slot(overview, f, sl))
        recap_pool.remove(fc)
        if fc in recap_attackers:
            recap_attackers.remove(fc)
            if fc in free_attackers:
                free_attackers.remove(fc)
        if fc in recap_defenders:
            recap_defenders.remove(fc)
        assignments.append(PlayAssignment(
            fc_name=fc, slot=sl, play_kind="defend_uncap",
            reason=f"v7 defend: slot {sl} held by {enemy} — recapture now",
        ))
        taken.add(sl)

    # During a BANK the whole team is securing the 5-pt attempt (returning
    # home); the only thing we schedule is the defensive recapture above.
    # New enemy-box ATTACK plays wait until we sortie again.
    if team_phase == "bank":
        return assignments

    # Enemy slots we can actually attack right now: not (freshly) confirmed
    # ours, not in the 5-s post-capture lock, not already taken by a defend
    # assignment this tick, and not already being attacked by an in-flight
    # drone. With the decay above, a slot we captured and left decays out of
    # "ours" within HOLDER_TRUST_S, so it re-enters this list and the next
    # free attacker is dispatched to re-capture it — continuous play.
    attackable = [sl for sl in enemy_slots
                  if holder.get(sl) != our and sl not in locked
                  and sl not in taken and sl not in targeted]

    # ----- (2) 5-pt full-sortie — preferred when we can cover EVERYTHING --
    # v7 §1.4.3 awards 5 pts per capture when ALL our drones are outside
    # home at the moment of capture. The easiest way to set that up is to
    # send every free attacker out on the same planner tick AND have them
    # cover every attackable slot — so no drone is left parked at home
    # waiting for a target. We prefer this over a 10-pt double-strike
    # when feasible: a 10-pt sync nets 10 (one event); a 3-slot full
    # sortie can net 5+5+5 = 15 (three captures, all 5-pt) AND the first
    # two captures may also land within 1 s of each other, retro-upgraded
    # to 5+5 of double-strike anyway.
    if (len(free_attackers) >= len(attackable) >= 2):
        from .attacker import SLOT_POSITIONS_M as _SP
        import math as _m

        def _d2(pos, slot):
            sx, sy = _SP.get(slot, (0.0, 0.0))
            return _m.hypot(float(pos[0]) - sx, float(pos[1]) - sy)

        positions = {fc: _drone_state(overview, fc).get("world_position_m")
                     or [0.0, 0.0, 0.0]
                     for fc in free_attackers}
        # Greedy nearest-slot assignment over the full attackable set.
        sortie: list[tuple[str, int]] = []
        used_fcs: set[str] = set()
        for sl in attackable:
            cand = [fc for fc in free_attackers if fc not in used_fcs]
            if not cand:
                break
            fc = min(cand, key=lambda f: _d2(positions[f], sl))
            sortie.append((fc, sl))
            used_fcs.add(fc)
        if len(sortie) >= 2:
            reason = ("v7 5-pt full-sortie: "
                      + " + ".join(f"{fc}->{sl}" for fc, sl in sortie))
            for fc, sl in sortie:
                assignments.append(PlayAssignment(
                    fc_name=fc, slot=sl, play_kind="full_sortie_5",
                    reason=reason,
                ))
                taken.add(sl)
            free_attackers = [fc for fc in free_attackers if fc not in used_fcs]
            # Refresh attackable for the next stages.
            attackable = [sl for sl in attackable if sl not in taken]

    # ----- (3) 10-pt double-strike — sync two strikers ------------------
    # Two free attackers + two unlocked enemy slots we don't already hold.
    # For sub-second sync we want the pairing whose two flight times match
    # as closely as possible — i.e. minimise MAX(distance) across the pair.
    # With <= 3 enemy slots and 2 drones it's a tiny brute-force search.
    if len(free_attackers) >= 2 and len(attackable) >= 2:
        from .attacker import SLOT_POSITIONS_M
        import math
        def _dist(pos, slot):
            sx, sy = SLOT_POSITIONS_M.get(slot, (0.0, 0.0))
            return math.hypot(float(pos[0]) - sx, float(pos[1]) - sy)
        # Cache positions for every free attacker.
        attacker_pos = {fc: _drone_state(overview, fc).get("world_position_m")
                        or [0.0, 0.0, 0.0]
                        for fc in free_attackers}
        # Search over every (fc1, fc2) pair (any two from free attackers) and
        # every (sl_a, sl_b) ordered slot pair; pick the (drone->slot,
        # drone->slot) assignment that minimises MAX(flight time delta).
        # Symmetric pairings (e.g. drone-at-x=-2 -> slot-at-x=-3 paired with
        # drone-at-x=+2 -> slot-at-x=+3) score best.
        best = None  # (max_flight, fc1, sl1, fc2, sl2)
        for i in range(len(free_attackers)):
            for j in range(i + 1, len(free_attackers)):
                fa, fb = free_attackers[i], free_attackers[j]
                pa, pb = attacker_pos[fa], attacker_pos[fb]
                for sa in attackable:
                    for sb in attackable:
                        if sa == sb:
                            continue
                        m1 = max(_dist(pa, sa), _dist(pb, sb))
                        if best is None or m1 < best[0]:
                            best = (m1, fa, sa, fb, sb)
        if best:
            _, fc1, sl1, fc2, sl2 = best
            # Drop the two chosen drones from free_attackers so the
            # 1-pt singleton path doesn't reuse them.
            free_attackers = [fc for fc in free_attackers
                              if fc != fc1 and fc != fc2]
            reason = (f"v7 10-pt sync: {fc1}->{sl1} + {fc2}->{sl2} "
                      f"(max flight {best[0]:.1f} m)")
            assignments.append(PlayAssignment(
                fc_name=fc1, slot=sl1, play_kind="double_strike_10",
                reason=reason,
            ))
            assignments.append(PlayAssignment(
                fc_name=fc2, slot=sl2, play_kind="double_strike_10",
                reason=reason,
            ))
            taken.add(sl1); taken.add(sl2)

    # ----- (4) 1-pt singletons — any remaining free attackers -----------
    for sl in attackable:
        if sl in taken:
            continue
        if not free_attackers:
            break
        fc = free_attackers.pop(0)
        assignments.append(PlayAssignment(
            fc_name=fc, slot=sl, play_kind="single_strike_1",
            reason=f"v7 1-pt singleton: {fc}->{sl}",
        ))
        taken.add(sl)

    return assignments
