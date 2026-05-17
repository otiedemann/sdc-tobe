"""Per-drone tasks the strategy layer can ask of the fleet.

A task is anything with two methods:

  - ``tick(state) -> FCCommand``  picks the side-effect to apply this tick
  - ``done(state) -> bool``       True once the task is satisfied

Tasks own no I/O — they translate :class:`SwarmState` into an
:class:`FCCommand` describing *what* should happen on *which* FC.
The runner is the only thing that talks to :class:`FCPool`. This split
keeps every task trivially unit-testable: ``MyTask(...).tick(state)``
returns a dataclass, period.

Currently shipped tasks: ``Idle``, ``StartMission``, ``StopMission``,
``ApplyTune``, ``SetArena``. Add more by subclassing :class:`DroneTask`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .world_model import DroneObservation, SwarmState

log = logging.getLogger("c2.strategy.tasks")


class CmdKind(str, Enum):
    """The verb in an :class:`FCCommand`. Mirrors the existing
    :class:`marker_mission_c2.fc_client.AsyncFCClient` surface 1:1, so
    :class:`marker_mission_c2.strategy.runner.SwarmRunner` can dispatch
    without a giant if-tree."""
    IDLE = "idle"
    START_MISSION = "start_mission"
    STOP_MISSION = "stop_mission"
    APPLY_TUNE = "apply_tune"
    SET_ARENA = "set_arena"


@dataclass(frozen=True)
class FCCommand:
    """The output of a :meth:`DroneTask.tick`. Describes *which FC* should
    have *which side effect* applied this tick. ``IDLE`` is the no-op:
    the runner skips the HTTP call entirely so 90 % of ticks (when most
    drones are mid-mission and the planner has nothing new to say) cost
    nothing.
    """
    target: str
    kind: CmdKind
    payload: dict = field(default_factory=dict)

    @classmethod
    def idle(cls, target: str) -> "FCCommand":
        return cls(target=target, kind=CmdKind.IDLE)


class DroneTask:
    """Base class. Subclass and override ``tick`` (+ optionally ``done``,
    ``reset``).

    Subclasses set ``name`` as a class attribute — used for logging and
    the on-tick records the runner emits.
    """
    name: str = "task"

    def __init__(self, target: str) -> None:
        self.target = target

    def reset(self) -> None:
        """Called by the planner whenever it (re)assigns this task to a
        drone. Use to clear any per-instance timers."""
        pass

    def tick(self, state: SwarmState) -> FCCommand:
        return FCCommand.idle(self.target)

    def done(self, state: SwarmState) -> bool:
        return False

    # Convenience: get my observation from a state, or None if my FC
    # isn't in the snapshot (e.g. removed from config mid-flight).
    def _obs(self, state: SwarmState) -> Optional[DroneObservation]:
        return state.drones.get(self.target)


class Idle(DroneTask):
    """Default fall-through. Emits IDLE forever; ``done`` always False
    so :class:`SwarmRunner` keeps re-evaluating the planner each tick."""
    name = "idle"

    def tick(self, state: SwarmState) -> FCCommand:
        return FCCommand.idle(self.target)


class StartMission(DroneTask):
    """Push a mission script to the FC and start it. Fires exactly once
    per task instance — subsequent ticks emit IDLE. ``done`` flips True
    as soon as the FC's reported phase leaves ``"idle"``/``"ready"``.

    Pass ``script`` as the full text the FC accepts via ``POST
    /api/mission/script`` (then ``POST /api/start``). The C2's
    :meth:`AsyncFCClient.start_mission` handles both.
    """
    name = "start_mission"

    def __init__(self, target: str, script: Optional[str] = None) -> None:
        super().__init__(target)
        self.script = script
        self._fired: bool = False

    def reset(self) -> None:
        self._fired = False

    def tick(self, state: SwarmState) -> FCCommand:
        if self._fired:
            return FCCommand.idle(self.target)
        self._fired = True
        return FCCommand(
            target=self.target,
            kind=CmdKind.START_MISSION,
            payload={"script": self.script} if self.script else {},
        )

    def done(self, state: SwarmState) -> bool:
        obs = self._obs(state)
        if obs is None or obs.phase is None:
            return False
        return obs.phase.lower() not in ("idle", "ready", "")


class StopMission(DroneTask):
    """Stop the running mission. One-shot; ``done`` flips True as soon
    as the FC reports a stopped phase OR is no longer flying."""
    name = "stop_mission"

    def __init__(self, target: str) -> None:
        super().__init__(target)
        self._fired: bool = False

    def reset(self) -> None:
        self._fired = False

    def tick(self, state: SwarmState) -> FCCommand:
        if self._fired:
            return FCCommand.idle(self.target)
        self._fired = True
        return FCCommand(target=self.target, kind=CmdKind.STOP_MISSION)

    def done(self, state: SwarmState) -> bool:
        obs = self._obs(state)
        if obs is None:
            return False
        return (not obs.flying) or (obs.phase or "").lower() in ("idle", "ready", "")


class ApplyTune(DroneTask):
    """One-shot push of a tune update dict (POST /api/tune).
    ``done`` flips True after the first tick."""
    name = "apply_tune"

    def __init__(self, target: str, updates: dict) -> None:
        super().__init__(target)
        self.updates = dict(updates)
        self._fired: bool = False

    def reset(self) -> None:
        self._fired = False

    def tick(self, state: SwarmState) -> FCCommand:
        if self._fired:
            return FCCommand.idle(self.target)
        self._fired = True
        return FCCommand(
            target=self.target, kind=CmdKind.APPLY_TUNE,
            payload={"updates": self.updates},
        )

    def done(self, state: SwarmState) -> bool:
        return self._fired


class SetArena(DroneTask):
    """One-shot push of an arena config dict (POST /api/arena/active)."""
    name = "set_arena"

    def __init__(self, target: str, arena: dict) -> None:
        super().__init__(target)
        self.arena = dict(arena)
        self._fired: bool = False

    def reset(self) -> None:
        self._fired = False

    def tick(self, state: SwarmState) -> FCCommand:
        if self._fired:
            return FCCommand.idle(self.target)
        self._fired = True
        return FCCommand(
            target=self.target, kind=CmdKind.SET_ARENA,
            payload={"arena": self.arena},
        )

    def done(self, state: SwarmState) -> bool:
        return self._fired


# ---------------------------------------------------------------------------
# Flight tasks (role-driven)
#
# These tasks generate marker_mission script text (the same DSL the FC's
# /api/mission/script endpoint accepts — TAKEOFF / APPROACH / HOOVER /
# TO / HEIGHT / LAND / etc., documented in marker_mission/mission_script.py)
# and push it via START_MISSION. The strategy reasons about WHICH script
# to run; the FC's mission player executes it, talks to Olympe, owns
# stick-level control. That split keeps the strategy at 1 Hz (script-push
# cadence) while flight control stays at marker_mission's native rate.
#
# Tasks emit a *fresh* script only when they actually want to change the
# drone's behaviour — re-emitting the same script every tick would
# constantly restart the FC's mission player. Each task tracks its
# "last pushed script" in ``self._last_script`` and only re-pushes when
# the content changes.
# ---------------------------------------------------------------------------

def _format_script(*lines: str) -> str:
    """Join script lines with newlines, dropping blanks. The FC's script
    parser is tolerant of trailing newlines."""
    return "\n".join(l for l in lines if l) + "\n"


class _ScriptingTask(DroneTask):
    """Base class for tasks that work by pushing a marker_mission script.
    Subclasses build the script in :meth:`_compose_script` (returning
    ``None`` for "no change"); the wrapper handles change detection so we
    only push when the script content differs from the last push."""

    name = "scripting_task"

    def __init__(self, target: str) -> None:
        super().__init__(target)
        self._last_script: Optional[str] = None
        self._done: bool = False

    def reset(self) -> None:
        self._last_script = None
        self._done = False

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        """Return the script text we want the FC to be running right now,
        or ``None`` to leave the FC's current mission untouched."""
        return None

    def tick(self, state: SwarmState) -> FCCommand:
        script = self._compose_script(state)
        if script is None:
            return FCCommand.idle(self.target)

        # Re-push if either:
        #   (a) the script content has changed since we last pushed, or
        #   (b) the FC reports its mission phase is "init" / blank —
        #       meaning the FC has nothing running (it was restarted,
        #       the mission completed, the operator hit stop, etc.).
        #       Without (b) the task would emit IDLE forever after the
        #       first push and the drone would sit on the ground while
        #       the strategy thinks it's already executing.
        obs = state.drones.get(self.target)
        fc_idle = (obs is not None and
                   (obs.phase is None or obs.phase.lower() in ("init", "ready", "done", "")))

        if script == self._last_script and not fc_idle:
            return FCCommand.idle(self.target)
        self._last_script = script
        return FCCommand(
            target=self.target, kind=CmdKind.START_MISSION,
            payload={"script": script},
        )

    def done(self, state: SwarmState) -> bool:
        return self._done


class Goto(_ScriptingTask):
    """Fly to a specified (x, y, z) arena point and hover there.

    Generates a one-line ``TO`` script. Used as the baseline navigation
    task and as a building block for the other role tasks.
    """
    name = "goto"

    def __init__(self, target: str, x: float, y: float, z: float,
                 hover_s: float = 9999.0) -> None:
        super().__init__(target)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.hover_s = float(hover_s)

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        return _format_script(
            "TAKEOFF",
            f"TO {self.x:.2f} {self.y:.2f} {self.z:.2f}",
            f"HOOVER {self.hover_s:.0f}",
        )


class HoldAboveTarget(_ScriptingTask):
    """Stage at ``hover_alt_m`` directly above ``target_id`` and hold.

    This is the attacker's *pre-strike* posture during the 10-pt
    maneuver — both attackers reach this state, then
    :class:`SyncAttackPair` flips them into the dive script.
    """
    name = "hold_above"

    def __init__(self, target: str, target_marker_id: int,
                 hover_alt_m: float, target_pos: tuple) -> None:
        super().__init__(target)
        self.target_marker_id = int(target_marker_id)
        self.hover_alt_m = float(hover_alt_m)
        # target_pos is (x, y) of the marker in arena coords; the
        # planner passes this in from arena_config so this task doesn't
        # have to know about marker layouts.
        self.target_x = float(target_pos[0])
        self.target_y = float(target_pos[1])

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        return _format_script(
            "TAKEOFF",
            f"TO {self.target_x:.2f} {self.target_y:.2f} {self.hover_alt_m:.2f}",
            "HOOVER 9999",
        )


class WaitInNeutral(_ScriptingTask):
    """Park a non-attacking drone in the neutral band at its cruise
    altitude. The slot (x, y) is computed by the planner from the
    arena geometry + per-drone offset so multiple drones don't collide.
    """
    name = "wait_in_neutral"

    def __init__(self, target: str, x: float, y: float, alt_m: float) -> None:
        super().__init__(target)
        self.x = float(x)
        self.y = float(y)
        self.alt_m = float(alt_m)

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        return _format_script(
            "TAKEOFF",
            f"TO {self.x:.2f} {self.y:.2f} {self.alt_m:.2f}",
            "HOOVER 9999",
        )


class ReclaimOnIntrusion(_ScriptingTask):
    """Defender behaviour: a friendly target is being approached by an
    enemy drone — fly over and reclaim. Built as ``APPROACH <our_id>``
    at a short distance; the FC's existing approach controller handles
    the close-in.
    """
    name = "reclaim"

    def __init__(self, target: str, target_marker_id: int,
                 hover_distance_m: float = 1.0) -> None:
        super().__init__(target)
        self.target_marker_id = int(target_marker_id)
        self.hover_distance_m = float(hover_distance_m)

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        return _format_script(
            "TAKEOFF",
            f"APPROACH {self.target_marker_id} {self.hover_distance_m:.2f}",
            "HOOVER 9999",
        )


class SyncAttackPair(_ScriptingTask):
    """The 10-point simultaneous-strike maneuver.

    Phases:
      ``hold``     — staging at ``hover_alt_m`` above ``target_marker_id``.
                     We emit the same script as :class:`HoldAboveTarget`
                     and watch for our partner to also report a pose
                     close to *their* target.
      ``strike``   — once both attackers report "in position" within
                     ``sync_window_s``, push a dive script: descend to
                     ``strike_alt_m``, brief hover, then RTH (LAND).
                     The strategy pushes this script on the same tick
                     to both attackers — sub-second sync at the
                     script-push level.
      ``done``     — drone reports ``not flying`` after RTH; planner
                     can re-assign the slot.

    The partner-readiness check is duck-typed against world state — we
    look at ``state.drones[partner_fc].pose`` and require it within
    ``ready_radius_m`` of the *partner's* target. The planner sets
    that up when it builds the pair.
    """
    name = "sync_attack"

    PHASE_HOLD = "hold"
    PHASE_STRIKE = "strike"
    PHASE_DONE = "done"

    def __init__(
        self,
        target: str,
        target_marker_id: int,
        target_pos: tuple,
        hover_alt_m: float,
        strike_alt_m: float,
        partner_fc: Optional[str] = None,
        partner_target_pos: Optional[tuple] = None,
        sync_window_s: float = 1.5,
        ready_radius_m: float = 1.5,
    ) -> None:
        super().__init__(target)
        self.target_marker_id = int(target_marker_id)
        self.target_x = float(target_pos[0])
        self.target_y = float(target_pos[1])
        self.hover_alt_m = float(hover_alt_m)
        self.strike_alt_m = float(strike_alt_m)
        self.partner_fc = partner_fc
        self.partner_target_pos = partner_target_pos
        self.sync_window_s = float(sync_window_s)
        self.ready_radius_m = float(ready_radius_m)
        self._phase: str = self.PHASE_HOLD
        self._strike_start_t: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._phase = self.PHASE_HOLD
        self._strike_start_t = None

    def _am_in_position(self, state: SwarmState) -> bool:
        obs = state.drones.get(self.target)
        if obs is None or obs.pose is None:
            return False
        dx = obs.pose[0] - self.target_x
        dy = obs.pose[1] - self.target_y
        return (dx * dx + dy * dy) <= (self.ready_radius_m ** 2)

    def _partner_in_position(self, state: SwarmState) -> bool:
        if self.partner_fc is None or self.partner_target_pos is None:
            return True  # solo strike — no partner gating
        obs = state.drones.get(self.partner_fc)
        if obs is None or obs.pose is None:
            return False
        dx = obs.pose[0] - float(self.partner_target_pos[0])
        dy = obs.pose[1] - float(self.partner_target_pos[1])
        return (dx * dx + dy * dy) <= (self.ready_radius_m ** 2)

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        if self._phase == self.PHASE_DONE:
            return None

        # Hold phase: emit (and keep emitting) the staging script.
        if self._phase == self.PHASE_HOLD:
            both_ready = self._am_in_position(state) and self._partner_in_position(state)
            if both_ready:
                # Flip to strike. The script we return on THIS tick is
                # the strike script — the runner pushes it now.
                self._phase = self.PHASE_STRIKE
                self._strike_start_t = state.t
                return _format_script(
                    f"TO {self.target_x:.2f} {self.target_y:.2f} {self.strike_alt_m:.2f}",
                    "HOOVER 2",
                    "LAND",
                )
            return _format_script(
                "TAKEOFF",
                f"TO {self.target_x:.2f} {self.target_y:.2f} {self.hover_alt_m:.2f}",
                "HOOVER 9999",
            )

        # Strike phase: don't change the script, just wait for the
        # drone to report not-flying (LAND completed).
        obs = state.drones.get(self.target)
        if obs is not None and not obs.flying:
            self._phase = self.PHASE_DONE
            self._done = True
        return None  # leave the strike script running on the FC

    @property
    def phase(self) -> str:
        return self._phase


class ScoreAndCaptureLoop(_ScriptingTask):
    """Cycle: score one own target → return home → capture one enemy
    target → return home → next own → next enemy → repeat.

    Generates a fresh mission-script for each cycle phase.
    marker_mission's mission player runs the script to completion
    (TAKEOFF / TO target / HOOVER / TO home / HOOVER), at which point
    the FC reports phase=init. We detect that init transition, advance
    our internal cycle pointer, and push the next phase's script.

    Hover altitude defaults to 2.5 m — markers sit at z=1.0 with a
    0.5 m box on top, so 2.5 m puts the drone 1 m clear above the
    target. Operator can adjust via :attr:`StrategySettings.match.score_hover_alt_m`.

    Per-drone state machine:

      phase = score_own:    fly to settings.own_target_ids[own_idx],
                            hover ``hover_s`` seconds, return home.
                            On completion: advance to capture_enemy.

      phase = capture_enemy: fly to settings.enemy_target_ids[enemy_idx],
                            hover, return home.
                            On completion: bump own_idx, back to score_own.
    """
    name = "score_capture"

    PHASE_SCORE  = "score_own"
    PHASE_CAPTURE = "capture_enemy"

    def __init__(self, target: str, settings,
                 hover_alt_m: float = 2.5,
                 hover_s: float = 5.0) -> None:
        super().__init__(target)
        self.settings = settings
        self.hover_alt_m = float(hover_alt_m)
        self.hover_s = float(hover_s)
        self._phase = self.PHASE_SCORE
        self._own_idx = 0
        self._enemy_idx = 0
        # FC-was-active flag detects the falling edge from
        # mission-running → mission-ended so we can advance the
        # cycle exactly once per FC cycle.
        self._fc_was_active = False

    def reset(self) -> None:
        super().reset()
        self._phase = self.PHASE_SCORE
        self._own_idx = 0
        self._enemy_idx = 0
        self._fc_was_active = False

    def _current_target_xy(self) -> Optional[tuple]:
        """The safe hover (x, y) for whichever target this phase
        wants. Returns ``None`` if the target id has no known
        position (configuration error)."""
        # Late import to avoid the tasks → roles → tasks loop.
        from .roles import _safe_hover_xy, target_pos
        s = self.settings
        if self._phase == self.PHASE_SCORE:
            ids = sorted(s.own_target_ids)
            if not ids:
                return None
            tid = ids[self._own_idx % len(ids)]
        else:
            ids = sorted(s.enemy_target_ids)
            if not ids:
                return None
            tid = ids[self._enemy_idx % len(ids)]
        pos = target_pos(tid)
        if pos is None:
            return None
        return _safe_hover_xy(s, (pos[0], pos[1]))

    def _home_xy(self) -> tuple:
        """Centroid of our team's home zone — the rally point between
        cycles. Centroid of the (X, Y) rect."""
        s = self.settings
        x_lo, x_hi = s.our_home_x_m
        y_lo, y_hi = s.our_home_y_m
        return ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)

    def _compose_script(self, state: SwarmState) -> Optional[str]:
        # Detect the FC ending its previous mission. obs.phase
        # transitions runtime → init when the mission completes
        # (LAND, end of script, abort, etc.). We advance the cycle
        # on the falling edge so each FC mission corresponds to
        # exactly one phase of our loop.
        obs = state.drones.get(self.target)
        fc_active = (obs is not None and obs.phase is not None
                     and obs.phase.lower()
                     not in ("init", "ready", "done", ""))
        if self._fc_was_active and not fc_active:
            # Mission cycle completed — advance to the next phase
            # and bump the index of the phase we just left.
            if self._phase == self.PHASE_SCORE:
                self._own_idx += 1
                self._phase = self.PHASE_CAPTURE
            else:
                self._enemy_idx += 1
                self._phase = self.PHASE_SCORE
        self._fc_was_active = fc_active

        target_xy = self._current_target_xy()
        if target_xy is None:
            return None
        home_x, home_y = self._home_xy()
        cruise_alt = self.settings.cruise_altitude_for(self.target)

        return _format_script(
            "TAKEOFF",
            f"TO {target_xy[0]:.2f} {target_xy[1]:.2f} {self.hover_alt_m:.2f}",
            f"HOOVER {self.hover_s:.0f}",
            f"TO {home_x:.2f} {home_y:.2f} {cruise_alt:.2f}",
            "HOOVER 1",
        )

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def own_idx(self) -> int:
        return self._own_idx

    @property
    def enemy_idx(self) -> int:
        return self._enemy_idx
