"""Stand-alone entry point. Boots an FCPool from the same config the
main C2 reads, attaches a strategy stack, runs it.

Usage::

    python -m marker_mission_c2.strategy.app                # default config
    python -m marker_mission_c2.strategy.app --config my.json
    python -m marker_mission_c2.strategy.app --tick-hz 2

By default it runs the no-op :class:`StaticAssignmentPlanner` with no
assignments — every drone is :class:`Idle` until you call
``planner.assign(...)``. Edit :func:`build_planner` to point at your
own mission scripts, or import :class:`SwarmRunner` from your own
top-level program.

This module is intentionally separate from ``marker_mission_c2.server``:
the strategy doesn't *need* the C2 web UI running, and the C2 web UI
doesn't *need* the strategy running. Either can be active without the
other. If you want both in one process, see
:class:`marker_mission_c2.server.create_app` and wire the runner into
its lifespan there — that's the only place ``marker_mission_c2``'s
existing code would have to change.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from marker_mission_c2.config import load_config
from marker_mission_c2.fc_pool import FCPool

from .planner import StaticAssignmentPlanner, SwarmPlanner
from .runner import SwarmRunner
from .safety import SafetyConfig, SafetyGate
from .settings import StrategySettings, TeamColor, load as load_settings, save as save_settings
from .tasks import Idle
from .world_model import SwarmWorldModel

log = logging.getLogger("c2.strategy.app")


def build_planner(cfg) -> SwarmPlanner:
    """Default planner: every configured FC is Idle. Override this
    function (or pass your own planner to :class:`SwarmRunner`) to
    inject real assignments.

    Examples::

        from .tasks import StartMission
        return StaticAssignmentPlanner({
            "flightctrl1": StartMission("flightctrl1", script=open("scripts/scout.txt").read()),
            "flightctrl2": StartMission("flightctrl2", script=open("scripts/attack.txt").read()),
        })

    or, utility-based::

        from .planner import UtilityPlanner
        return UtilityPlanner(candidates=[
            ("scout",  lambda fc: StartMission(fc, script=SCOUT_SCRIPT),
                       lambda s, fc: 1.0 if needs_scout(s) else -inf,  1),
            ("attack", lambda fc: StartMission(fc, script=ATK_SCRIPT),
                       lambda s, fc: score_attacker(s, fc),            None),
        ])
    """
    return StaticAssignmentPlanner({spec.name: Idle(spec.name)
                                    for spec in cfg.fcs})


def build_safety(cfg) -> SafetyGate:
    """Default safety: stop on stale poll / battery <12 % / link drop.
    Override for arena-specific geofence or different thresholds."""
    return SafetyGate(SafetyConfig(
        poll_stale_s=max(3.0, 2.0 / max(0.1, cfg.state_poll_hz) * 6.0),
        battery_critical_pct=12.0,
    ))


async def run(cfg, tick_hz: float,
              settings_obj: StrategySettings) -> None:
    pool = FCPool(cfg)
    await pool.start()
    log.info("strategy: FCPool up with %d FC(s)", len(cfg.fcs))
    log.info(
        "strategy: team=%s  own=%s  enemy=%s",
        settings_obj.team_color.value,
        sorted(settings_obj.own_target_ids),
        sorted(settings_obj.enemy_target_ids),
    )
    planner = build_planner(cfg)
    safety = build_safety(cfg)
    runner = SwarmRunner(
        pool=pool,
        world_model=SwarmWorldModel(pool),
        planner=planner,
        safety=safety,
        tick_hz=tick_hz,
        on_tick=_log_record,
    )
    await runner.start()
    try:
        # Park until SIGINT / SIGTERM
        await _wait_for_signal()
    finally:
        await runner.stop()
        await pool.stop()


def _log_record(rec) -> None:
    if rec.safety_overrides:
        for fc, reason in rec.safety_overrides.items():
            log.warning("override %s: %s", fc, reason)
    busy = [f"{fc}:{n}" for fc, n in rec.decisions.items() if n != "idle"]
    if busy:
        log.info("tick: %s", ", ".join(busy))


async def _wait_for_signal() -> None:
    """Resolve when SIGINT/SIGTERM arrive. On Windows we just sleep
    forever (no UNIX signal hooks)."""
    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()
    try:
        for sig_name in ("SIGINT", "SIGTERM"):
            import signal
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                loop.add_signal_handler(sig, stop_evt.set)
    except NotImplementedError:
        pass  # Windows
    await stop_evt.wait()


def main() -> int:
    p = argparse.ArgumentParser(
        prog="marker_mission_c2.strategy.app",
        description="Stand-alone strategy runner for the fleet C2.",
    )
    p.add_argument("--config", default=None,
                   help="Override C2 config path (else uses the same "
                        "resolution as `python -m marker_mission_c2`).")
    p.add_argument("--strategy-settings", default=None,
                   help="Path to a strategy settings JSON (else resolves "
                        "via $MARKER_MISSION_C2_STRATEGY_SETTINGS, the "
                        "package's settings.json, then settings.example.json).")
    p.add_argument("--team", choices=["red", "blue"], default=None,
                   help="Override team_color for this run only. Combined "
                        "with --save-settings to persist.")
    p.add_argument("--save-settings", action="store_true",
                   help="Write the effective settings back to settings.json "
                        "before starting the runner (so the next launch "
                        "picks them up).")
    p.add_argument("--tick-hz", type=float, default=1.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)

    settings_obj = load_settings(args.strategy_settings)
    if args.team:
        settings_obj.team_color = TeamColor.parse(args.team)
    if args.save_settings:
        path = save_settings(settings_obj, args.strategy_settings)
        log.info("strategy: settings saved to %s", path)

    asyncio.run(run(cfg, args.tick_hz, settings_obj))
    return 0


if __name__ == "__main__":
    sys.exit(main())
