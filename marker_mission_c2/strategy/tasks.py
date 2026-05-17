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
