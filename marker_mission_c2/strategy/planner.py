"""Planners: pick a :class:`DroneTask` per FC for the current tick.

Two ship-with-the-package implementations cover the two common shapes
of swarm strategy:

  - :class:`StaticAssignmentPlanner` — fixed per-FC role. Each FC
    runs a single :class:`DroneTask` until it's ``done``, then falls
    back to :class:`Idle`. Closest analog to a hand-written deployment
    script.

  - :class:`UtilityPlanner` — per tick, scores a set of candidate
    tasks for the swarm and assigns the highest-scoring task to each
    FC, optionally constrained by per-task assignment limits. Use
    when "which drone should do X" depends on live state.

Anything with a ``decide(state) -> Mapping[str, DroneTask]`` method
is a valid planner. The runner doesn't care about its internals.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Mapping, MutableMapping, Optional

from .tasks import DroneTask, Idle
from .world_model import SwarmState

log = logging.getLogger("c2.strategy.planner")


class SwarmPlanner:
    """Protocol: anything that implements ``decide(state)`` works."""

    def decide(self, state: SwarmState) -> Mapping[str, DroneTask]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Static one-task-per-drone planner
# ---------------------------------------------------------------------------

class StaticAssignmentPlanner(SwarmPlanner):
    """Each FC gets a single task that runs to completion.

    After a task's ``done()`` returns True, the FC falls through to
    :class:`Idle`. Re-run :meth:`assign` to give it more work later.

    Example::

        planner = StaticAssignmentPlanner({
            "flightctrl1": StartMission("flightctrl1", script=script_a),
            "flightctrl2": StartMission("flightctrl2", script=script_b),
        })
    """

    def __init__(self, assignment: Mapping[str, DroneTask]):
        self._tasks: MutableMapping[str, DroneTask] = dict(assignment)
        for t in self._tasks.values():
            t.reset()

    def assign(self, target: str, task: DroneTask) -> None:
        """Replace the current task for one drone. Resets it."""
        task.reset()
        self._tasks[target] = task

    def decide(self, state: SwarmState) -> Mapping[str, DroneTask]:
        out: dict[str, DroneTask] = {}
        for name in state.drones.keys():
            task = self._tasks.get(name)
            if task is None or task.done(state):
                out[name] = self._tasks.setdefault(name, Idle(name))
                # Replace done tasks with Idle so we don't keep
                # re-evaluating their done() forever.
                if task is not None and task.done(state):
                    log.info("planner: %s on %s done — idling",
                             task.name, name)
                    self._tasks[name] = Idle(name)
                    out[name] = self._tasks[name]
                continue
            out[name] = task
        return out


# ---------------------------------------------------------------------------
# Utility-scored planner
# ---------------------------------------------------------------------------

ScoreFn = Callable[[SwarmState, str], float]
TaskFactory = Callable[[str], DroneTask]


class UtilityPlanner(SwarmPlanner):
    """Per-tick: for every candidate (factory, score) tuple, score it
    against every FC; assign the highest-scoring candidate to each FC
    (greedy, with optional per-candidate ``limit``).

    Each candidate is a tuple ``(name, factory, score_fn, limit)``:

      - ``name``       human-readable label for logs
      - ``factory(target)`` builds a fresh task for one drone
      - ``score_fn(state, target) -> float``  higher = more wanted;
        ``float("-inf")`` to mean "do NOT pick this for this drone"
      - ``limit``      max simultaneous assignments of this candidate
        across the fleet (None = no limit)

    The greedy assignment isn't globally optimal — for now we accept
    that, swarms are tiny (≤ ~10 FCs).
    """

    def __init__(
        self,
        candidates: Iterable[tuple[str, TaskFactory, ScoreFn, Optional[int]]],
        fallback: TaskFactory = Idle,
    ):
        self._candidates = list(candidates)
        self._fallback = fallback
        # Per-FC currently-running task, so we don't reset() it every tick
        # just because the planner picks the same candidate again.
        self._current: dict[str, tuple[str, DroneTask]] = {}

    def decide(self, state: SwarmState) -> Mapping[str, DroneTask]:
        out: dict[str, DroneTask] = {}
        assigned_per_candidate: dict[str, int] = {}

        # Build per-FC ranked candidate list once
        ranks: dict[str, list[tuple[float, str, TaskFactory]]] = {}
        for fc in state.drones.keys():
            scored: list[tuple[float, str, TaskFactory]] = []
            for cand_name, factory, score_fn, _ in self._candidates:
                try:
                    s = float(score_fn(state, fc))
                except Exception:
                    log.exception(
                        "planner: score_fn raised for %s/%s — treating as -inf",
                        cand_name, fc,
                    )
                    s = float("-inf")
                scored.append((s, cand_name, factory))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranks[fc] = scored

        # Greedy: walk FCs in stable name order; pick each FC's top
        # candidate that still has capacity. Falls back to ``self._fallback``.
        limits = {c[0]: c[3] for c in self._candidates}
        for fc in sorted(state.drones.keys()):
            chosen_name: Optional[str] = None
            chosen_factory: Optional[TaskFactory] = None
            for score, cand_name, factory in ranks[fc]:
                if score == float("-inf"):
                    continue
                lim = limits.get(cand_name)
                used = assigned_per_candidate.get(cand_name, 0)
                if lim is not None and used >= lim:
                    continue
                chosen_name = cand_name
                chosen_factory = factory
                assigned_per_candidate[cand_name] = used + 1
                break

            prev = self._current.get(fc)
            if chosen_name is None or chosen_factory is None:
                task = self._fallback(fc)
                self._current[fc] = ("__fallback__", task)
            elif prev is None or prev[0] != chosen_name:
                task = chosen_factory(fc)
                task.reset()
                self._current[fc] = (chosen_name, task)
                log.info("planner: %s -> %s", fc, chosen_name)
            else:
                task = prev[1]
            out[fc] = task
        return out
