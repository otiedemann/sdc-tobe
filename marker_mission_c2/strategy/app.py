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
from marker_mission_c2.settings import SettingsStore as C2SettingsStore

from .match import MatchState
from .planner import RoleAssignmentPlanner, StaticAssignmentPlanner, SwarmPlanner
from .runner import SwarmRunner
from .safety import SafetyConfig, SafetyGate
from .settings import StrategySettings, TeamColor, load as load_settings, save as save_settings
from .tasks import Idle
from .web import make_app as make_web_app, start_in_background as start_web
from .world_model import SwarmWorldModel

log = logging.getLogger("c2.strategy.app")


def build_planner(cfg, settings_obj: StrategySettings) -> SwarmPlanner:
    """Default planner: :class:`RoleAssignmentPlanner` — every tick the
    strategy decides each drone's role (attacker / scout / defender /
    idle) from live state + settings, then builds a role-specific
    task. Operator pins (``settings.drones[fc].role``) override the
    auto-assignment.

    Override this function (or pass your own planner to
    :class:`SwarmRunner`) if you want a hand-crafted assignment
    instead.
    """
    return RoleAssignmentPlanner(
        settings_obj,
        max_attackers=2,
        max_defenders=1,
    )


def build_safety(cfg, settings_obj: StrategySettings) -> SafetyGate:
    """Default safety: stop on stale poll / battery <12 % / link drop /
    geofence breach. Bounds are read from
    :class:`StrategySettings.arena` and pulled inward by
    ``safety_margin_m`` so the enforcement boundary sits inside the
    physical wall — the drone must STOP_MISSION *before* reaching the
    wall, not after clipping through it.
    """
    a = settings_obj.arena
    half_w = a.width_m / 2.0 - a.safety_margin_m
    half_d = a.depth_m / 2.0 - a.safety_margin_m
    return SafetyGate(SafetyConfig(
        poll_stale_s=max(3.0, 2.0 / max(0.1, cfg.state_poll_hz) * 6.0),
        battery_critical_pct=12.0,
        bounds_x_m=(-half_w, half_w),
        bounds_y_m=(-half_d, half_d),
        bounds_z_m=(0.2, 5.5),  # 20 cm floor → 5.5 m ceiling
        geofence_enforce=True,
    ))


async def run(cfg, tick_hz: float,
              settings_obj: StrategySettings,
              web_host: str, web_port: int, web_enabled: bool,
              settings_path) -> None:
    # FCPool needs the C2's own SettingsStore (disabled_fcs etc.).
    # Point at the canonical runtime/settings.json so the strategy
    # honours whatever the main C2 dashboard's /settings page has
    # silenced. The strategy never writes to it — only the dashboard
    # process does — so there's no lock contention with running both
    # at once.
    c2_settings = C2SettingsStore()
    pool = FCPool(cfg, c2_settings)
    await pool.start()
    log.info("strategy: FCPool up with %d FC(s)", len(cfg.fcs))
    log.info(
        "strategy: team=%s  own=%s  enemy=%s",
        settings_obj.team_color.value,
        sorted(settings_obj.own_target_ids),
        sorted(settings_obj.enemy_target_ids),
    )
    world_model = SwarmWorldModel(pool)
    planner = build_planner(cfg, settings_obj)
    safety = build_safety(cfg, settings_obj)
    match_state = MatchState(duration_s=settings_obj.match.duration_s)
    runner = SwarmRunner(
        pool=pool,
        world_model=world_model,
        planner=planner,
        safety=safety,
        tick_hz=tick_hz,
        on_tick=_log_record,
        match_state=match_state,
    )
    await runner.start()
    if web_enabled:
        web_app = make_web_app(
            settings_obj=settings_obj,
            world_model=world_model,
            c2_cfg=cfg,
            settings_path=settings_path,
            planner=planner,
            match_state=match_state,
            safety=safety,
        )
        start_web(web_app, host=web_host, port=web_port)
        log.info("strategy: open http://%s:%d/ to view live arena + settings",
                 "localhost" if web_host == "0.0.0.0" else web_host, web_port)
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
    p.add_argument("--web-host", default="0.0.0.0",
                   help="Bind host for the operator UI (default 0.0.0.0).")
    p.add_argument("--web-port", type=int, default=8091,
                   help="Bind port for the operator UI (default 8091). "
                        "8090 is the main C2 dashboard — keep them apart.")
    p.add_argument("--no-web", action="store_true",
                   help="Do not start the operator UI. The strategy "
                        "runner still runs headless.")
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

    asyncio.run(run(
        cfg, args.tick_hz, settings_obj,
        web_host=args.web_host,
        web_port=args.web_port,
        web_enabled=not args.no_web,
        settings_path=args.strategy_settings,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
