"""Role assignment + role-specific task factories.

The strategy decides each drone's role dynamically every tick based on
:class:`SwarmState` + :class:`StrategySettings`. An operator can pin a
role per drone in settings (``drones.<fc>.role``); ``None`` (the
default) means "let the strategy choose".

Three scoring functions, one per role. Each returns a float — higher
is better. Returning ``-inf`` means "do NOT pick this drone for this
role" (battery too low, drone not flying-capable, etc.).
:class:`marker_mission_c2.strategy.planner.RoleAssignmentPlanner` runs
all three scorers and greedy-assigns each drone to the highest-scoring
role under per-role caps.

For each role we also expose a factory that builds the right
:class:`DroneTask` given the drone's current observation and the
settings — e.g. "attacker on Sphinx3 should be SyncAttackPair against
target 41 with Sphinx5 as partner".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .settings import Role, StrategySettings
from .tasks import (
    DroneTask,
    HoldAboveTarget,
    Idle,
    ReclaimOnIntrusion,
    SyncAttackPair,
    WaitInNeutral,
)
from .world_model import SwarmState

log = logging.getLogger("c2.strategy.roles")


# ---------------------------------------------------------------------------
# Default target layout — used to translate marker ID → arena (x, y) so
# tasks can fly to a target without each module re-loading
# ``default_target_layout.json``. Kept here so the strategy doesn't
# couple to the tools/sphinx-arena/ build path.
#
# Coordinates mirror tools/sphinx-arena/default_target_layout.json
# (commit-tracked, won't drift in practice — that file is the layout
# of the physical arena). If you ever move the targets, edit both.
# ---------------------------------------------------------------------------

TARGET_POSITIONS: Dict[int, Tuple[float, float, float]] = {
    # Red team
    41: (-5.0,  8.0, 1.0),
    42: (-8.0,  5.4, 1.0),
    43: (-5.0,  3.0, 1.0),
    44: ( 0.0,  0.0, 1.0),   # spare, normally disabled
    45: ( 0.0,  0.0, 1.0),
    46: ( 0.0,  0.0, 1.0),
    # Blue team
    31: ( 5.0,  8.0, 1.0),
    32: ( 8.0,  5.4, 1.0),
    33: ( 5.0,  3.0, 1.0),
    34: ( 0.0,  0.0, 1.0),
    35: ( 0.0,  0.0, 1.0),
    36: ( 0.0,  0.0, 1.0),
}


def target_pos(target_id: int) -> Optional[Tuple[float, float, float]]:
    return TARGET_POSITIONS.get(int(target_id))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScoreCtx:
    """Bundled context passed to every score fn — keeps the signatures
    short and makes adding new inputs (e.g. recent kills, enemy
    positions) painless."""
    state: SwarmState
    settings: StrategySettings


# Per-role minimum battery — under this, the role is disqualified.
# Higher than safety.battery_critical_pct because we want to retire
# drones *before* safety has to STOP them mid-attack.
_MIN_BATTERY_ATTACKER = 30.0
_MIN_BATTERY_DEFENDER = 25.0
_MIN_BATTERY_SCOUT    = 20.0


def score_attacker(ctx: _ScoreCtx, fc: str) -> float:
    """Higher = better attacker candidate. Returns ``-inf`` if the
    drone simply isn't fit to attack (offline, low battery, etc.)."""
    obs = ctx.state.drones.get(fc)
    if obs is None or not obs.online or not obs.drone_connected:
        return float("-inf")
    if obs.battery_pct is None or obs.battery_pct < _MIN_BATTERY_ATTACKER:
        return float("-inf")
    # Pinned attacker → priority over everything.
    if ctx.settings.role_for(fc) == Role.ATTACKER:
        return 1000.0
    # Pinned to a *different* role → score very low so the planner
    # doesn't accidentally promote them.
    pinned = ctx.settings.drones.get(fc)
    if pinned is not None and pinned.role != Role.IDLE:
        return -100.0
    score = 10.0
    # Higher battery = stronger preference. 100 % adds +5, 30 % adds 0.
    score += (obs.battery_pct - 30.0) / 14.0
    # Drones already in our home zone are pre-staged (closer to the
    # target staging point) — prefer them so the attack starts faster.
    if obs.pose is not None and ctx.settings.is_in_home_zone(obs.pose[0]):
        score += 3.0
    return score


def score_defender(ctx: _ScoreCtx, fc: str) -> float:
    obs = ctx.state.drones.get(fc)
    if obs is None or not obs.online or not obs.drone_connected:
        return float("-inf")
    if obs.battery_pct is None or obs.battery_pct < _MIN_BATTERY_DEFENDER:
        return float("-inf")
    if ctx.settings.role_for(fc) == Role.DEFENDER:
        return 1000.0
    pinned = ctx.settings.drones.get(fc)
    if pinned is not None and pinned.role != Role.IDLE:
        return -100.0
    score = 5.0
    # Battery contribution lighter than attacker's — defenders don't
    # have to commit a full strike.
    score += (obs.battery_pct - 25.0) / 25.0
    # Bonus if an enemy is currently within intercept_radius of one of
    # our targets — there's actual work for the defender to do.
    if _intrusion_detected(ctx):
        score += 7.0
    return score


def score_scout(ctx: _ScoreCtx, fc: str) -> float:
    obs = ctx.state.drones.get(fc)
    if obs is None or not obs.online or not obs.drone_connected:
        return float("-inf")
    if obs.battery_pct is None or obs.battery_pct < _MIN_BATTERY_SCOUT:
        return float("-inf")
    if ctx.settings.role_for(fc) == Role.SCOUT:
        return 1000.0
    pinned = ctx.settings.drones.get(fc)
    if pinned is not None and pinned.role != Role.IDLE:
        return -100.0
    # Scout is a fallback role — lower default score than attacker /
    # defender so the planner only picks scouts after attacker and
    # defender slots are filled.
    return 2.0


def _intrusion_detected(ctx: _ScoreCtx) -> bool:
    """Best-effort intrusion check: any drone reported in this snapshot
    that isn't ours, within intercept_radius of one of our targets.

    Currently a stub — we don't have a friend/foe channel in
    :class:`SwarmState`, so this always returns False until we wire
    enemy positions (e.g. via the drone-detector YOLO output).
    Returning False just makes the defender's score the same as the
    scout's baseline, so behaviour stays sane."""
    return False


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------

def _safe_hover_xy(s: StrategySettings, marker_xy: Tuple[float, float],
                   inset_m: float = 1.5) -> Tuple[float, float]:
    """Compute a safe (x, y) for the drone to hover at, given a wall-
    mounted marker's position. We do TWO transforms:

    1. **Pull toward arena centre by ``inset_m``** so the drone is
       ~1.5 m clear of the wall the marker is mounted on. Without this,
       the drone would be commanded TO the wall coordinate (e.g.
       target 31 sits at (5, 8) which is the +X wall itself) and
       overshoot it.
    2. **Clamp to the arena's safe inner box** (half-extent minus
       ``arena.safety_margin_m``). Belt-and-braces — if the marker
       position drifts past the wall (sometimes happens with the SDC
       arena tooling) we still don't generate an OOB script.
    """
    mx, my = float(marker_xy[0]), float(marker_xy[1])
    half_w = s.arena.width_m / 2.0 - s.arena.safety_margin_m
    half_d = s.arena.depth_m / 2.0 - s.arena.safety_margin_m
    # Step 1: pull toward origin.  unit-vector from marker→(0,0)
    import math
    r = math.hypot(mx, my)
    if r > 1e-6:
        ux, uy = -mx / r, -my / r
        sx = mx + ux * inset_m
        sy = my + uy * inset_m
    else:
        sx, sy = mx, my
    # Step 2: clamp to safe arena box.
    sx = max(-half_w, min(half_w, sx))
    sy = max(-half_d, min(half_d, sy))
    return (sx, sy)


def build_attacker_task(
    fc: str,
    ctx: _ScoreCtx,
    target_marker_id: int,
    partner_fc: Optional[str] = None,
    partner_target_marker_id: Optional[int] = None,
) -> DroneTask:
    """Pre-strike: stage above the assigned target. If a partner is
    also an attacker, build a :class:`SyncAttackPair` that will flip
    into the dive script once both are in position.

    Falls back to a plain :class:`HoldAboveTarget` if no partner is
    available (solo attacker — never executes the 10-pt move, just
    holds).

    The hover position is **not** the marker's coordinate (which sits
    on the wall) — it's pulled inward by :func:`_safe_hover_xy` so the
    drone stages safely inside the arena.
    """
    s = ctx.settings
    pos = target_pos(target_marker_id)
    if pos is None:
        log.warning("attacker on %s: target_id %s has no known position",
                    fc, target_marker_id)
        return Idle(fc)
    safe_xy = _safe_hover_xy(s, (pos[0], pos[1]))
    partner_pos = (target_pos(partner_target_marker_id)
                   if partner_target_marker_id is not None else None)
    partner_safe = (_safe_hover_xy(s, (partner_pos[0], partner_pos[1]))
                    if partner_pos is not None else None)

    if partner_fc is not None and partner_safe is not None:
        return SyncAttackPair(
            target=fc,
            target_marker_id=target_marker_id,
            target_pos=safe_xy,
            hover_alt_m=s.attack.hover_alt_m,
            strike_alt_m=s.attack.strike_alt_m,
            partner_fc=partner_fc,
            partner_target_pos=partner_safe,
            sync_window_s=s.attack.sync_window_s,
            ready_radius_m=1.5,
        )
    return HoldAboveTarget(
        target=fc, target_marker_id=target_marker_id,
        hover_alt_m=s.attack.hover_alt_m, target_pos=safe_xy,
    )


def build_defender_task(
    fc: str,
    ctx: _ScoreCtx,
) -> DroneTask:
    """Defender: wait in the neutral zone unless intrusion detected,
    in which case fly toward the threatened own-target.

    Intrusion detection is a placeholder until we wire enemy
    positions; meanwhile the defender just parks like a scout.
    """
    s = ctx.settings
    if _intrusion_detected(ctx):
        # Pick our nearest own-target as the reclaim point.
        own_ids = sorted(s.own_target_ids)
        if own_ids:
            return ReclaimOnIntrusion(fc, target_marker_id=own_ids[0])
    x, y, z = _neutral_slot(fc, ctx)
    return WaitInNeutral(fc, x=x, y=y, alt_m=z)


def build_scout_task(
    fc: str,
    ctx: _ScoreCtx,
) -> DroneTask:
    """Scout: park in neutral zone at the drone's cruise altitude."""
    x, y, z = _neutral_slot(fc, ctx)
    return WaitInNeutral(fc, x=x, y=y, alt_m=z)


def build_idle_task(fc: str, ctx: _ScoreCtx) -> DroneTask:
    return Idle(fc)


def _neutral_slot(fc: str, ctx: _ScoreCtx) -> Tuple[float, float, float]:
    """Per-drone parking slot inside the neutral band. Deterministic
    spread along Y so two drones with the same cruise altitude don't
    sit on top of each other (XY separation as a fallback to altitude
    layering).

    Strategy: lay slots along the depth (Y) axis, evenly spaced, in
    insertion order of the drones in settings.drones. Each slot sits
    at the centre of the neutral X band, at the drone's cruise
    altitude.
    """
    s = ctx.settings
    fcs = list(s.drones.keys())
    n = max(1, len(fcs))
    try:
        idx = fcs.index(fc)
    except ValueError:
        idx = 0
    lo, hi = s.arena.neutral_zone_x_m
    x_mid = (lo + hi) / 2.0
    # Slot the drones along arena Y inside the neutral band — keep a
    # 1.5 m margin from each end of the depth axis so they don't
    # bracket the front/back walls.
    half_d = s.arena.depth_m / 2.0
    span = max(0.0, 2.0 * half_d - 3.0)
    y_step = span / max(1, n - 1) if n > 1 else 0.0
    y_slot = -half_d + 1.5 + idx * y_step if n > 1 else 0.0
    z_slot = s.cruise_altitude_for(fc)
    return (x_mid, y_slot, z_slot)


# ---------------------------------------------------------------------------
# Top-level decision: which role for each drone, then build the task
# ---------------------------------------------------------------------------

@dataclass
class RoleDecision:
    fc: str
    role: Role
    score: float
    task: DroneTask


def decide_roles(
    state: SwarmState,
    settings: StrategySettings,
    *,
    max_attackers: int = 2,
    max_defenders: int = 1,
) -> Mapping[str, RoleDecision]:
    """Greedy role assignment.

    Algorithm:
      1. Score each (fc, role) tuple with the three score fns.
      2. Walk roles in priority order: attacker → defender → scout →
         idle. For each role, assign up to ``max_<role>`` drones with
         the highest score for that role (and score > -inf).
      3. Remaining drones get :class:`Role.IDLE`.

    Caps are operator-tunable parameters; the default ``2 attackers /
    1 defender`` matches the 10-pt sync-strike doctrine + one
    home-zone watcher.

    Returns ``{fc → RoleDecision}`` containing the assigned role, its
    score, and the concrete :class:`DroneTask` to run this tick.
    """
    ctx = _ScoreCtx(state=state, settings=settings)
    fcs = list(state.drones.keys())

    # Score table: roles[fc][role] = score
    scores: Dict[str, Dict[Role, float]] = {fc: {} for fc in fcs}
    for fc in fcs:
        scores[fc][Role.ATTACKER] = score_attacker(ctx, fc)
        scores[fc][Role.DEFENDER] = score_defender(ctx, fc)
        scores[fc][Role.SCOUT] = score_scout(ctx, fc)
        scores[fc][Role.IDLE] = 0.1  # Universal fallback

    assigned: Dict[str, Role] = {}
    caps = {Role.ATTACKER: max_attackers, Role.DEFENDER: max_defenders,
            Role.SCOUT: 999, Role.IDLE: 999}

    for role in (Role.ATTACKER, Role.DEFENDER, Role.SCOUT, Role.IDLE):
        cap = caps[role]
        # Candidates not yet assigned, sorted by score desc.
        candidates = sorted(
            ((fc, scores[fc][role]) for fc in fcs if fc not in assigned),
            key=lambda x: x[1],
            reverse=True,
        )
        for fc, sc in candidates:
            if cap <= 0:
                break
            if sc == float("-inf"):
                continue
            assigned[fc] = role
            cap -= 1
        if cap == 0 and role == Role.IDLE:
            break

    # Build attacker pairs so SyncAttackPair gets a partner reference.
    attackers = [fc for fc, r in assigned.items() if r == Role.ATTACKER]
    own_targets = sorted(settings.own_target_ids)
    pairs_cfg = settings.attack_pair_targets()
    # Default: assign the first qualifying pair to the first two
    # attackers in alphabetical order. More attackers than slots → the
    # rest fall through to plain HoldAboveTarget (no partner).
    attacker_assignments: Dict[str, Tuple[Optional[int], Optional[str], Optional[int]]] = {}
    if len(attackers) >= 2 and pairs_cfg:
        a1, a2 = sorted(attackers)[:2]
        t1, t2 = pairs_cfg[0]
        attacker_assignments[a1] = (t1, a2, t2)
        attacker_assignments[a2] = (t2, a1, t1)
        # remaining attackers (if any) get solo HoldAboveTarget on the
        # rest of our targets, cycling through
        rest = [a for a in sorted(attackers) if a not in (a1, a2)]
        spare_targets = [t for t in own_targets if t not in (t1, t2)] or own_targets
        for i, a in enumerate(rest):
            attacker_assignments[a] = (
                spare_targets[i % len(spare_targets)],
                None, None,
            )
    elif len(attackers) >= 1:
        # Solo attacker (or no pairs defined): everyone gets a target
        # in round-robin, no partner.
        targets = own_targets or [None]
        for i, a in enumerate(sorted(attackers)):
            attacker_assignments[a] = (
                (int(targets[i % len(targets)])
                 if targets and targets[i % len(targets)] is not None
                 else None),
                None, None,
            )

    # Now build tasks per drone.
    out: Dict[str, RoleDecision] = {}
    for fc in fcs:
        role = assigned.get(fc, Role.IDLE)
        score = scores[fc][role]
        if role == Role.ATTACKER:
            tgt, partner, partner_tgt = attacker_assignments.get(
                fc, (None, None, None))
            if tgt is None:
                task: DroneTask = Idle(fc)
            else:
                task = build_attacker_task(
                    fc, ctx,
                    target_marker_id=tgt,
                    partner_fc=partner,
                    partner_target_marker_id=partner_tgt,
                )
        elif role == Role.DEFENDER:
            task = build_defender_task(fc, ctx)
        elif role == Role.SCOUT:
            task = build_scout_task(fc, ctx)
        else:
            task = build_idle_task(fc, ctx)
        out[fc] = RoleDecision(fc=fc, role=role, score=score, task=task)
    return out
