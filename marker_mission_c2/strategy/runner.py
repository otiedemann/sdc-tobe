"""SwarmRunner: the tick loop that wires every other strategy module
together and applies the resulting :class:`FCCommand` via FCPool.

Runs on the same asyncio event loop as :mod:`marker_mission_c2.server`
so it can re-use the pool's :class:`httpx.AsyncClient`. Pull-pattern:

    while running:
        state = world_model.observe()              # sync
        tasks = planner.decide(state)              # sync
        for fc, task in tasks.items():
            cmd  = task.tick(state)                 # sync
            verd = safety.gate(cmd, state)          # sync
            await self._dispatch(verd.cmd)          # async — single FC call

Default tick rate is 1 Hz; the C2's per-FC ``/api/state`` poll already
runs at ``state_poll_hz`` (default 5 Hz), so 1 Hz strategy decisions
are about as fast as the inputs change. Tune via ``tick_hz=``.

This module owns no policy. Plug your own planner/safety in if you need
something exotic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Optional

from .safety import SafetyGate, SafetyVerdict
from .tasks import CmdKind, DroneTask, FCCommand
from .world_model import SwarmState, SwarmWorldModel

log = logging.getLogger("c2.strategy.runner")


@dataclass
class TickRecord:
    """One iteration of the loop, exposed via the optional ``on_tick``
    callback. Cheap to log or push to a SSE channel.
    """
    t: float
    state: SwarmState
    decisions: Mapping[str, str]             # fc_name -> task.name
    dispatched: Mapping[str, FCCommand]      # fc_name -> command actually sent
    safety_overrides: Mapping[str, str]      # fc_name -> reason


class SwarmRunner:
    """The strategy event loop. Construct once, ``await start()``; call
    ``await stop()`` to wind down.

    Wired to live FCPool but doesn't import the pool's classes — keeps
    the strategy testable with a duck-typed fake (anything with
    ``snapshot()`` for the world_model and ``client(name)`` for
    dispatch).

    Optional ``on_tick`` is fired with a :class:`TickRecord` after each
    iteration completes; useful for SSE-streaming the latest strategy
    decisions into the C2 dashboard, or for offline replay.
    """

    def __init__(
        self,
        pool,
        world_model: Optional[SwarmWorldModel] = None,
        planner=None,
        safety: Optional[SafetyGate] = None,
        tick_hz: float = 1.0,
        on_tick: Optional[Callable[[TickRecord], Awaitable[None] | None]] = None,
    ):
        self._pool = pool
        self._world = world_model or SwarmWorldModel(pool)
        if planner is None:
            raise ValueError("SwarmRunner needs a planner")
        self._planner = planner
        if safety is None:
            from .safety import SafetyConfig
            safety = SafetyGate(SafetyConfig())
        self._safety = safety
        self._dt = 1.0 / max(0.1, float(tick_hz))
        self._on_tick = on_tick
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="c2-strategy-runner")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    # ---------- the loop ----------

    async def _run(self) -> None:
        log.info("strategy runner started @ %.1f Hz", 1.0 / self._dt)
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    state = self._world.observe()
                    assignments = self._planner.decide(state)
                    decisions = {fc: t.name for fc, t in assignments.items()}
                    dispatched: dict[str, FCCommand] = {}
                    overrides: dict[str, str] = {}

                    # Dispatch all FCs in parallel; one failing call must
                    # not delay the others.
                    coros = []
                    for fc, task in assignments.items():
                        try:
                            cmd = task.tick(state)
                        except Exception:
                            log.exception("task.tick crashed on %s — IDLE", fc)
                            cmd = FCCommand.idle(fc)
                        verd = self._safety.gate(cmd, state)
                        if verd.overridden:
                            overrides[fc] = verd.reason
                        dispatched[fc] = verd.cmd
                        if verd.cmd.kind != CmdKind.IDLE:
                            coros.append(self._dispatch(verd.cmd))

                    if coros:
                        await asyncio.gather(*coros, return_exceptions=True)

                    if self._on_tick is not None:
                        rec = TickRecord(
                            t=t0, state=state, decisions=decisions,
                            dispatched=dispatched, safety_overrides=overrides,
                        )
                        try:
                            result = self._on_tick(rec)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            log.exception("on_tick callback failed")
                except Exception:
                    log.exception("strategy tick failed — continuing")

                # Sleep the remainder of the tick budget.
                slack = self._dt - (time.monotonic() - t0)
                if slack > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=slack)
                    except asyncio.TimeoutError:
                        pass
        finally:
            log.info("strategy runner stopped")

    # ---------- dispatch ----------

    async def _dispatch(self, cmd: FCCommand) -> None:
        """Apply one command to one FC via the existing
        :class:`AsyncFCClient` surface. Logs and swallows errors so a
        single dead FC never crashes the loop.
        """
        client = self._pool.client(cmd.target)
        if client is None:
            log.warning("dispatch: unknown target %s", cmd.target)
            return
        try:
            if cmd.kind == CmdKind.START_MISSION:
                await client.start_mission(script=cmd.payload.get("script"))
            elif cmd.kind == CmdKind.STOP_MISSION:
                await client.stop_mission()
            elif cmd.kind == CmdKind.APPLY_TUNE:
                await client.apply_tune(cmd.payload.get("updates") or {})
            elif cmd.kind == CmdKind.SET_ARENA:
                await client.set_active_arena(cmd.payload.get("arena") or {})
            elif cmd.kind == CmdKind.IDLE:
                pass
            else:
                log.warning("dispatch: unknown CmdKind %s", cmd.kind)
        except Exception:
            log.exception("dispatch %s to %s failed", cmd.kind.value, cmd.target)
