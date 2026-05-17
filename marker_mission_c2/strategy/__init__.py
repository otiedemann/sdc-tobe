"""marker_mission_c2.strategy — reactive strategy layer for the fleet C2.

Sits on top of :mod:`marker_mission_c2` (which already polls each FC's
``/api/state`` and exposes :class:`marker_mission_c2.fc_pool.FCPool`)
and chooses what each drone should do *next*. Five modules, narrow
contracts between them, nothing in ``marker_mission_c2/`` proper is
modified — strategy is opt-in.

Pipeline (one iteration per tick, default 1 Hz):

    world_model.SwarmWorldModel.observe()    -> SwarmState (read-only)
    planner.SwarmPlanner.decide(state)       -> dict[fc_name -> DroneTask]
    DroneTask.tick(state)                    -> FCCommand
    safety.SafetyGate.gate(cmd, state)       -> FCCommand (possibly modified)
    runner.SwarmRunner._dispatch(cmd)        -> applies via FCPool

See ``README.md`` in this directory for the design rationale and
extension points.
"""
from __future__ import annotations

from .world_model import (
    DroneObservation,
    SwarmState,
    SwarmWorldModel,
)
from .tasks import (
    FCCommand,
    CmdKind,
    DroneTask,
    Idle,
    StartMission,
    StopMission,
    ApplyTune,
    SetArena,
)
from .planner import (
    SwarmPlanner,
    StaticAssignmentPlanner,
    UtilityPlanner,
)
from .safety import (
    SafetyConfig,
    SafetyGate,
    SafetyVerdict,
)
from .runner import (
    SwarmRunner,
    TickRecord,
)

__all__ = [
    "DroneObservation", "SwarmState", "SwarmWorldModel",
    "FCCommand", "CmdKind", "DroneTask",
    "Idle", "StartMission", "StopMission", "ApplyTune", "SetArena",
    "SwarmPlanner", "StaticAssignmentPlanner", "UtilityPlanner",
    "SafetyConfig", "SafetyGate", "SafetyVerdict",
    "SwarmRunner", "TickRecord",
]
