"""Live-tunable strategy + capture parameters for C2 V2.

These are the levers the operator hand-tuned all season (capture geometry,
altitudes, rotation speeds, strategy bias). Surfaced in the dashboard so they
can be adjusted at the venue WITHOUT a code change or restart; the runner reads
them on every launch, so a change takes effect on the next (re)launch.

Defaults are the PROVEN values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Tunables:
    # capture geometry (operator-validated)
    attack_standoff_m: float = 1.50
    attack_dist_tol_m: float = 0.20
    over_box_forward_m: float = 1.50
    capture_rise_m: float = 0.50
    rth_standoff_m: float = 3.50
    # altitudes. Operator: ATTACKERS fly fixed lanes that never cross, so they
    # all share ONE low altitude (~1 m, good for spotting the low ~0.4-0.73 m box
    # markers). DEFENDERS fly a HIGHER tier so they never conflict with the
    # attackers crossing beneath them; the step separates the two defenders that
    # share a neutral station. Never below the 0.73 m box top.
    scout_alt_m: float = 1.00
    mover_base_alt_m: float = 1.00     # attacker altitude (flat — all attackers)
    defender_alt_above_attacker_m: float = 0.80   # defender altitude = attacker + this (avoid collisions)
    mover_alt_step_m: float = 0.20     # vertical separation between the defenders
    # rotation speeds (RC yaw stick 1..100)
    scout_yaw_stick: int = 70
    defender_yaw_stick: int = 35
    # defender standoff
    defender_standoff_m: float = 2.50
    # Defender neutral-wait distance: how far OFF our home wall marker the
    # defender holds (in the neutral zone, scanning our boxes). Operator: 5 m.
    defender_neutral_m: float = 5.0
    # offense persistence + strategy bias
    attack_passes: int = 30
    # Attacker home dwell: after each capture+return the attacker HOVERS in our
    # home zone this long (scoring, never landing) before flying its straight
    # lane back out to strike its fixed target again — the "hover at home between
    # strikes" cadence, which also gives the box time to flip back to enemy.
    attacker_home_dwell_s: float = 5.0
    # Baseline home defenders (the simplified strategy: 3 attackers + 2
    # defenders). In AUTO this is the default defender count when no own box is
    # threatened; more defenders are added per threatened box.
    baseline_defenders: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# (key, label, min, max, step) — drives the tuning panel + validates writes.
TUNE_SPEC = [
    ("attack_standoff_m", "Attack standoff (m)", 0.5, 4.0, 0.05),
    ("attack_dist_tol_m", "Attack arrival band (m)", 0.05, 1.0, 0.05),
    ("over_box_forward_m", "FB_UD_IMU forward (m)", 0.5, 3.0, 0.05),
    ("capture_rise_m", "FB_UD_IMU rise (m)", 0.2, 1.5, 0.05),
    ("rth_standoff_m", "RTH wall standoff (m)", 1.0, 6.0, 0.1),
    ("scout_alt_m", "Scout altitude (m)", 0.73, 2.5, 0.05),
    ("mover_base_alt_m", "Attacker altitude (m)", 0.73, 3.0, 0.05),
    ("defender_alt_above_attacker_m", "Defender height above attacker (m)", 0.0, 2.0, 0.05),
    ("mover_alt_step_m", "Defender separation step (m)", 0.0, 1.0, 0.05),
    ("scout_yaw_stick", "Scout rotate speed", 5, 100, 5),
    ("defender_yaw_stick", "Defender rotate speed", 5, 100, 5),
    ("defender_standoff_m", "Defender standoff (m)", 1.0, 5.0, 0.1),
    ("defender_neutral_m", "Defender neutral distance (m)", 1.0, 8.0, 0.5),
    ("attack_passes", "Attack chain passes", 5, 100, 1),
    ("attacker_home_dwell_s", "Attacker home dwell (s)", 0.0, 30.0, 0.5),
    ("baseline_defenders", "Baseline defenders", 0, 4, 1),
]
_SPEC_BY_KEY = {k: (lo, hi, st) for k, _l, lo, hi, st in TUNE_SPEC}
_INT_KEYS = {"scout_yaw_stick", "defender_yaw_stick", "attack_passes",
             "baseline_defenders"}


def coerce(key: str, value: Any):
    """Clamp + type a tunable write; return None if the key is unknown/invalid."""
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return None
    lo, hi, _ = spec
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    v = max(lo, min(hi, v))
    return int(round(v)) if key in _INT_KEYS else round(v, 3)


def from_dict(d: Dict[str, Any]) -> Tunables:
    t = Tunables()
    if isinstance(d, dict):
        for k, v in d.items():
            cv = coerce(k, v)
            if cv is not None:
                setattr(t, k, cv)
    return t
