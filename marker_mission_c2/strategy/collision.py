"""Runtime collision avoidance for OUR-team drones.

The static altitude grid (``runner._deconflicted_cruise_alts``) already gives
every drone of our team a distinct CRUISE height 0.6 m apart, so in steady
cruise the 3D separation is always >= 0.6 m. The residual risk is during
vertical TRANSITIONS — a drone climbing to cruise, descending to capture, or
climbing for RTH crosses another drone's altitude band, and if their xy happens
to be close at that moment the 3D gap can collapse.

This module is the safety net for exactly that. Each tick it:

  1. estimates every drone's velocity from its last two reported positions,
  2. predicts the closest point of approach (CPA) for every pair over a short
     horizon (straight-line extrapolation), and
  3. if a pair is predicted to come within ``SAFETY_RADIUS_M``, tells the
     LOWER-PRIORITY drone to climb/descend to a clear altitude and hold until
     the conflict passes.

Only ONE drone of each conflicting pair yields (deterministic priority), so the
two never both dodge into each other. The drone that yields changes ALTITUDE
(not heading) — vertical separation is the cheapest, most reliable dodge and
matches how the rest of the swarm stays apart.

Enemy drones are out of scope: regs §1.3 gives us no access to them, so we can
neither see nor predict them. Cross-team separation is handled statically by
the red-even / blue-odd altitude convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Our-team cruise spacing is 0.6 m, so a 0.45 m trigger never false-fires in
# steady cruise but DOES catch a transition that crosses another drone's lane.
SAFETY_RADIUS_M = 0.45
# Predict this far ahead (s): long enough to react with a ~0.6 m climb (~2 s),
# short enough that the straight-line velocity extrapolation stays valid.
LOOKAHEAD_S = 4.0
# When a drone yields, place it at least this far (m) from the drone it yields
# to AND from every other drone, so the dodge doesn't create a fresh conflict.
CLEAR_GAP_M = 0.60
MAX_ALT_M = 5.0      # 1 m below the 6 m ceiling
MIN_ALT_M = 0.80     # don't dodge into the floor
# A drone moving faster than this (m/s, horizontal) is treated as TRANSITING
# (attack run / flying home) and is never told to dodge — the altitude grid
# keeps lanes apart and the crossing passes in a second or two. Only slower
# (loitering / drifting) drones get an altitude dodge.
MOVING_SPEED_MPS = 0.4

# Priority: higher stays, lower yields. The scout holds station at the centre
# (others route around it); a drone mid-capture outranks one merely cruising.
_ROLE_RANK = {"scout": 3, "defender": 2, "attacker": 1, "idle": 0}


@dataclass
class Track:
    fc: str
    pos: Tuple[float, float, float]
    vel: Tuple[float, float, float]
    role: str


def _cpa(a: Track, b: Track, horizon: float) -> Tuple[float, float]:
    """(time, distance) of closest approach over [0, horizon], linear motion."""
    dp = (a.pos[0] - b.pos[0], a.pos[1] - b.pos[1], a.pos[2] - b.pos[2])
    dv = (a.vel[0] - b.vel[0], a.vel[1] - b.vel[1], a.vel[2] - b.vel[2])
    dvv = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]
    if dvv < 1e-9:
        t = 0.0                      # parallel / both ~static -> closest is now
    else:
        t = -(dp[0] * dv[0] + dp[1] * dv[1] + dp[2] * dv[2]) / dvv
        t = max(0.0, min(horizon, t))
    cx, cy, cz = dp[0] + t * dv[0], dp[1] + t * dv[1], dp[2] + t * dv[2]
    return t, math.sqrt(cx * cx + cy * cy + cz * cz)


def _priority(t: Track) -> Tuple[int, str]:
    # Higher tuple = stays. Lower = yields. Tie-break: larger fc name stays.
    return (_ROLE_RANK.get(t.role, 0), t.fc)


class CollisionAvoider:
    """Velocity-tracking CPA predictor that emits per-drone altitude dodges."""

    def __init__(self) -> None:
        # fc -> (last_pos, last_unix_s) for velocity estimation.
        self._prev: Dict[str, Tuple[Tuple[float, float, float], float]] = {}

    def update_velocities(
        self, positions: Dict[str, Tuple[float, float, float]], now: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """Estimate each drone's velocity from its last two positions."""
        vel: Dict[str, Tuple[float, float, float]] = {}
        for fc, pos in positions.items():
            prev = self._prev.get(fc)
            if prev is not None and now > prev[1]:
                dt = now - prev[1]
                vel[fc] = (
                    (pos[0] - prev[0][0]) / dt,
                    (pos[1] - prev[0][1]) / dt,
                    (pos[2] - prev[0][2]) / dt,
                )
            else:
                vel[fc] = (0.0, 0.0, 0.0)
            self._prev[fc] = (pos, now)
        for fc in list(self._prev):           # forget drones that vanished
            if fc not in positions:
                del self._prev[fc]
        return vel

    def resolve(self, tracks: List[Track]) -> Dict[str, Tuple[float, str]]:
        """Return ``{fc: (target_alt_m, reason)}`` for drones that must yield.

        Only LOITERING drones (scout rotating, defender/attacker holding) can
        be told to dodge. A drone transiting horizontally (an attacker on a
        capture run, anyone flying home) is left alone: the static altitude
        grid already keeps within-team cruise lanes 0.6 m apart, and the
        crossing is transient — it passes in 1-2 s. Overriding a transiting
        attacker with a HEIGHT/HOOVER hold was stalling capture runs (the
        "massive delay" the operator saw), so we trust the grid for movers and
        reserve the dodge for two slow drones genuinely drifting together.
        """
        overrides: Dict[str, Tuple[float, str]] = {}
        alts: Dict[str, float] = {t.fc: t.pos[2] for t in tracks}

        def _moving(t: Track) -> bool:
            return math.hypot(t.vel[0], t.vel[1]) > MOVING_SPEED_MPS

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                t_cpa, dmin = _cpa(a, b, LOOKAHEAD_S)
                if dmin >= SAFETY_RADIUS_M:
                    continue
                yielder, stayer = (
                    (a, b) if _priority(a) < _priority(b) else (b, a)
                )
                # Never break a drone that's actively transiting — the grid
                # handles it and the crossing is fleeting.
                if _moving(yielder):
                    continue
                target = self._clear_alt(
                    alts[stayer.fc],
                    {fc: z for fc, z in alts.items() if fc != yielder.fc},
                )
                overrides[yielder.fc] = (
                    target,
                    f"collision risk with {stayer.fc} "
                    f"(~{dmin:.2f} m in {t_cpa:.1f} s) — change altitude to "
                    f"{target:.2f} m",
                )
                alts[yielder.fc] = target      # account for it in later pairs
        return overrides

    @staticmethod
    def _clear_alt(stayer_alt: float, others: Dict[str, float]) -> float:
        """Nearest altitude >= CLEAR_GAP_M from the stayer and every other drone."""
        def is_clear(z: float) -> bool:
            return all(abs(z - o) >= CLEAR_GAP_M for o in others.values())

        for step in range(1, 16):
            up = round(stayer_alt + step * CLEAR_GAP_M, 2)
            if up <= MAX_ALT_M and is_clear(up):
                return up
            down = round(stayer_alt - step * CLEAR_GAP_M, 2)
            if down >= MIN_ALT_M and is_clear(down):
                return down
        # Arena saturated — climb as high as allowed (best effort).
        return min(MAX_ALT_M, round(stayer_alt + CLEAR_GAP_M, 2))
