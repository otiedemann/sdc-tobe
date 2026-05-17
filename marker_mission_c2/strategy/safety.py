"""Safety gate: vetoes / modifies a planner's command before it reaches
:class:`marker_mission_c2.fc_pool.FCPool`.

Rules are opt-in via :class:`SafetyConfig`. They run **after** the
planner+task picked an :class:`FCCommand`, in this order:

  1. FC offline / stale poll  → force STOP_MISSION
  2. Drone link dropped       → force STOP_MISSION
  3. Battery < critical %     → force STOP_MISSION (mission-script lands)
  4. Manual emergency hold    → force STOP_MISSION on every FC
  5. Geofence (per-drone)     → currently a soft warn only; the C2 has
     no per-tick RC channel to enforce a fence at this layer. Planners
     should pick scripts whose envelope they trust; safety logs if a
     reported pose leaves the configured bounds so an operator notices.

Add custom rules by composing a :class:`SafetyGate` subclass and
overriding :meth:`SafetyGate.gate`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .tasks import CmdKind, FCCommand
from .world_model import DroneObservation, SwarmState

log = logging.getLogger("c2.strategy.safety")


@dataclass
class SafetyConfig:
    # Stop the mission if we haven't successfully polled the FC for this
    # long (Real-world: a brief poll miss is noise, anything past a few
    # seconds usually means the FC died).
    poll_stale_s: float = 6.0

    # Stop if battery drops below this. Drone-side land logic is what
    # actually performs the descent; STOP_MISSION just ends the script.
    battery_critical_pct: float = 12.0

    # Geofence in arena coords. None = no fence on that axis. Used for
    # logging-only warnings (see module docstring).
    bounds_x_m: Optional[Tuple[float, float]] = None
    bounds_y_m: Optional[Tuple[float, float]] = None
    bounds_z_m: Optional[Tuple[float, float]] = None

    # Throttle repeated stop spam: don't re-issue STOP_MISSION to a drone
    # more often than this. Without it, every tick where the FC is still
    # reporting "stale" would fire another STOP.
    stop_cooldown_s: float = 5.0


@dataclass
class SafetyVerdict:
    """What :meth:`SafetyGate.gate` returns."""
    cmd: FCCommand
    reason: str = ""
    overridden: bool = False    # True iff cmd != input cmd


class SafetyGate:
    """Single-instance gate; share across all FCs.

    Carries a small amount of per-FC cooldown state internally so it
    doesn't spam STOP. Stateless w.r.t. the rest of the strategy
    pipeline — calling ``gate`` on the same inputs always produces a
    deterministic result modulo the cooldown clock.
    """

    def __init__(self, cfg: SafetyConfig, manual_hold_all: bool = False):
        self.cfg = cfg
        self._manual_hold_all = bool(manual_hold_all)
        self._last_stop_at: dict[str, float] = {}

    # ----- operator switches -----

    def hold_all(self, on: bool) -> None:
        """Engage / release the manual emergency hold. While engaged,
        every gate() call returns STOP_MISSION regardless of input."""
        self._manual_hold_all = bool(on)
        log.warning("safety: manual hold_all=%s", self._manual_hold_all)

    # ----- main entry -----

    def gate(self, cmd: FCCommand, state: SwarmState) -> SafetyVerdict:
        obs = state.drones.get(cmd.target)
        if obs is None:
            return SafetyVerdict(cmd, "")

        if self._manual_hold_all:
            return self._stop(cmd, "manual emergency hold engaged", state.t)

        if not obs.online or (obs.age_s is not None
                              and obs.age_s > self.cfg.poll_stale_s):
            return self._stop(
                cmd,
                f"FC offline / stale poll (age={obs.age_s})",
                state.t,
            )

        if obs.flying and not obs.drone_connected:
            return self._stop(
                cmd,
                "drone link dropped while flying",
                state.t,
            )

        if (obs.battery_pct is not None
                and obs.battery_pct < self.cfg.battery_critical_pct
                and obs.flying):
            return self._stop(
                cmd,
                (f"battery {obs.battery_pct:.0f}% < critical "
                 f"{self.cfg.battery_critical_pct:.0f}%"),
                state.t,
            )

        # Geofence: warn, don't override. Mission script owns enforcement.
        self._maybe_warn_geofence(obs)

        return SafetyVerdict(cmd, "")

    # ----- internals -----

    def _stop(self, original: FCCommand, reason: str,
              t_now: float) -> SafetyVerdict:
        target = original.target
        last = self._last_stop_at.get(target, 0.0)
        if (t_now - last) < self.cfg.stop_cooldown_s:
            # Already issued a STOP recently — emit IDLE to keep quiet.
            return SafetyVerdict(
                FCCommand.idle(target),
                f"{reason} (stop on cooldown)",
                overridden=True,
            )
        self._last_stop_at[target] = t_now
        log.warning("safety: STOP %s (%s)", target, reason)
        return SafetyVerdict(
            FCCommand(target=target, kind=CmdKind.STOP_MISSION),
            reason, overridden=True,
        )

    def _maybe_warn_geofence(self, obs: DroneObservation) -> None:
        if obs.pose is None:
            return
        x, y, z = obs.pose
        for axis, value, bounds in (
            ("x", x, self.cfg.bounds_x_m),
            ("y", y, self.cfg.bounds_y_m),
            ("z", z, self.cfg.bounds_z_m),
        ):
            if bounds is None:
                continue
            lo, hi = bounds
            if value < lo or value > hi:
                log.warning(
                    "safety: %s %s=%.2f outside fence [%.2f, %.2f] "
                    "(planner must enforce)",
                    obs.name, axis, value, lo, hi,
                )
