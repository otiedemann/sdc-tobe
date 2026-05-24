"""The SDC26 C2 commander — the orchestration brain.

The :class:`Commander` owns the live game loop. It is the only module that
ties every other C2 piece together:

    fc_client   — one DroneFC per drone (the HTTP link to each FC)
    world_model — the fused picture of slots + drones
    events      — pure slot-diff -> GameEvent detector
    roles       — per-drone behaviour policies (scout/attacker/defender/idle)
    navigation  — turns a RoleAction into concrete RC stick values
    assignments — altitude deconfliction + default roles + defender picking

Lifecycle::

    cmd = Commander(cfg)
    await cmd.start()        # connect, deconflict, launch the tick loop
    cmd.submit(Command(...)) # operator pokes (thread-safe enqueue)
    cmd.set_mode(GameMode.AUTO)
    snap = cmd.snapshot()    # for the web UI
    await cmd.stop()         # cancel the loop, close the HTTP clients

Each tick (``cfg.strategy.tick_hz``):
    1. poll every drone's position + telemetry (concurrently) -> world_model
    2. drain the manual-command queue and apply it
    3. detect_events() vs last tick's slots; in AUTO, react
    4. run each drone's role -> RoleAction -> navigation -> send_rc / land /
       takeoff / hover
    5. sleep to hold the loop rate

The whole tick body is wrapped in try/except so one bad tick (a transient
network blip, a malformed payload, a half-finished sibling) logs and the loop
keeps running rather than dying.

IMPORTS: only :mod:`c2.models` and :mod:`c2.config` are imported at module
load. Everything else (fc_client, world_model, events, roles, navigation) is
imported lazily inside :meth:`start` / the tick loop, so ``import c2.commander``
succeeds even while a sibling module is still being written.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

from .config import C2Config
from .models import (
    Command,
    CommandType,
    DronePhase,
    DroneRole,
    DroneState,
    GameMode,
    Vec3,
    WorldSnapshot,
    now_s,
)

log = logging.getLogger("c2.commander")


class Commander:
    """Owns the world model, the fleet links, and the async game loop."""

    def __init__(self, cfg: C2Config) -> None:
        self.cfg = cfg

        # World model is safe to build now (depends only on models/config).
        from .world_model import WorldModel  # local: keep import-time light
        self.world = WorldModel(cfg)

        # Per-drone links / behaviours / scratch memory. Populated in start().
        self.fcs: dict[str, "object"] = {}          # drone_id -> DroneFC
        self.roles: dict[str, "object"] = {}        # drone_id -> Role instance
        self.memory: dict[str, dict] = {}           # drone_id -> scratch dict

        # Game mode (operator-toggleable).
        self.mode: GameMode = GameMode.AUTO if cfg.mode_start_auto else GameMode.MANUAL

        # Recent human-readable event / log strings for the UI.
        self.recent_events: deque[str] = deque(maxlen=30)

        # Pending manual commands (thread/async-safe FIFO).
        self.cmd_queue: "asyncio.Queue[Command]" = asyncio.Queue()

        # Previous tick's slot dict, for event detection.
        self._prev_slots: dict = {}

        # The background tick task.
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------ helpers

    def _log_event(self, msg: str) -> None:
        """Record a human-readable line for the UI + the logger."""
        line = f"{now_s():.1f} {msg}"
        self.recent_events.append(line)
        log.info(msg)

    def _drone_ids(self) -> list[str]:
        return [d.drone_id for d in self.cfg.drones]

    def _reset_memory(self, drone_id: str) -> None:
        """Wipe a drone's scratch dict (role phase machines start fresh)."""
        self.memory[drone_id] = {}

    def _set_role(self, drone_id: str, role: DroneRole) -> None:
        """Swap a drone's behaviour policy + mirror it into the world model."""
        from .roles import make_role
        self.roles[drone_id] = make_role(role)
        self.world.set_role(drone_id, role)
        self._reset_memory(drone_id)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Connect to every FC, deconflict altitudes, launch the tick loop."""
        from .fc_client import DroneFC
        from .roles import make_role

        ids = self._drone_ids()

        # 1. Decide starting roles. Honour an explicit per-drone initial_role
        #    if the operator set one in the config; otherwise fall back to the
        #    automatic default_roles() policy (one scout, rest biased attacker).
        from .assignments import assign_altitude_bands, default_roles
        auto_roles = default_roles(ids, self.cfg)
        roles: dict[str, DroneRole] = {}
        for dc in self.cfg.drones:
            if dc.initial_role != DroneRole.IDLE:
                roles[dc.drone_id] = dc.initial_role
            else:
                roles[dc.drone_id] = auto_roles.get(dc.drone_id, DroneRole.IDLE)

        # 2. Build FC clients, register drones, disable the Pi-side arena guard.
        for dc in self.cfg.drones:
            did = dc.drone_id
            fc = DroneFC(did, dc.fc_base_url, timeout=4.0)
            self.fcs[did] = fc
            self.memory[did] = {}
            self.roles[did] = make_role(roles[did])
            self.world.ensure_drone(did, dc.base_altitude_m, roles[did])
            try:
                ok = await fc.disable_arena_guard()
            except Exception as e:  # noqa: BLE001 — never let one FC abort start
                ok = False
                log.warning("[%s] disable_arena_guard raised: %s", did, e)
            self._log_event(
                f"{did}: link up, role={roles[did].value}, "
                f"arena_guard_off={ok}"
            )

        # 3. Deconflict cruise altitudes across the live fleet, then push the
        #    chosen bands back into the world model.
        states = [self.world.get_drone(d) for d in ids]
        assign_altitude_bands(states, self.cfg)
        for st in states:
            self.world.ensure_drone(st.drone_id, st.base_altitude_m, st.role)
        bands = {st.drone_id: round(st.base_altitude_m, 2) for st in states}
        if bands:
            self._log_event(f"altitude bands: {bands}")

        # 4. Launch the loop.
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="c2-tick-loop")
        self._log_event(
            f"commander started: {len(ids)} drone(s), mode={self.mode.value}, "
            f"tick={self.cfg.strategy.tick_hz} Hz"
        )

    async def stop(self) -> None:
        """Cancel the tick loop and close every FC client. Does NOT land."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                log.warning("tick loop exited with: %s", e)
            self._task = None

        for did, fc in list(self.fcs.items()):
            try:
                await fc.close()
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] fc.close failed: %s", did, e)
        self._log_event("commander stopped")

    # ------------------------------------------------------------------ public API

    def set_mode(self, mode: GameMode) -> None:
        """Switch MANUAL <-> AUTO (operator control)."""
        if not isinstance(mode, GameMode):
            mode = GameMode(mode)
        if mode != self.mode:
            self._log_event(f"mode -> {mode.value}")
        self.mode = mode

    def submit(self, cmd: Command) -> dict:
        """Validate + enqueue a manual command. Returns ``{"ok": bool, ...}``.

        The command is applied on the next tick (so all state mutation happens
        in one place, on the loop). Validation here is cheap structural
        checking; semantic application happens in :meth:`_apply_command`.
        """
        if not isinstance(cmd, Command):
            return {"ok": False, "error": "not a Command"}
        if not isinstance(cmd.type, CommandType):
            return {"ok": False, "error": f"bad command type: {cmd.type!r}"}

        known = set(self._drone_ids())

        # SET_MODE needs a mode.
        if cmd.type == CommandType.SET_MODE and cmd.mode is None:
            return {"ok": False, "error": "set_mode requires 'mode'"}

        # Commands that target a specific drone must name a known one (or a
        # fleet-wide keyword where allowed).
        fleet_ok_types = {
            CommandType.TAKEOFF, CommandType.LAND,
            CommandType.STOP, CommandType.EMERGENCY,
        }
        if cmd.type in {
            CommandType.ASSIGN_ROLE, CommandType.ATTACK,
            CommandType.DEFEND, CommandType.SCOUT, CommandType.RETURN_HOME,
        }:
            if cmd.drone_id not in known:
                return {"ok": False,
                        "error": f"unknown drone_id: {cmd.drone_id!r}"}
        elif cmd.type in fleet_ok_types:
            if cmd.drone_id not in (None, "all") and cmd.drone_id not in known:
                return {"ok": False,
                        "error": f"unknown drone_id: {cmd.drone_id!r}"}

        # ATTACK needs a slot.
        if cmd.type == CommandType.ATTACK and cmd.slot is None:
            return {"ok": False, "error": "attack requires 'slot'"}
        # ASSIGN_ROLE needs a role.
        if cmd.type == CommandType.ASSIGN_ROLE and cmd.role is None:
            return {"ok": False, "error": "assign_role requires 'role'"}

        self.cmd_queue.put_nowait(cmd)
        return {"ok": True, "queued": cmd.type.value, "drone_id": cmd.drone_id}

    def snapshot(self) -> WorldSnapshot:
        """World snapshot with the recent-events log attached for the UI."""
        snap = self.world.snapshot(self.mode)
        snap.events = list(self.recent_events)
        return snap

    # ------------------------------------------------------------------ tick loop

    async def _run_loop(self) -> None:
        """Background task: poll, apply commands, react, drive, repeat."""
        hz = self.cfg.strategy.tick_hz or 4.0
        period = 1.0 / hz if hz > 0 else 0.25
        while self._running:
            t0 = now_s()
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill us
                log.exception("tick error (continuing): %s", e)
                self._log_event(f"tick error: {e}")
            # Sleep the remainder of the tick budget.
            elapsed = now_s() - t0
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _tick(self) -> None:
        now = now_s()
        ids = self._drone_ids()

        # 1. Poll every drone concurrently (position + telemetry), ingest.
        await self._poll_fleet(ids, now)

        # 2. Apply any queued manual commands.
        await self._drain_commands()

        # 3. Event detection vs the previous tick's slots.
        curr_slots = self.world.slots
        if self._prev_slots:
            try:
                from .events import detect_events
                evs = detect_events(self._prev_slots, curr_slots, self.cfg)
            except Exception as e:  # noqa: BLE001
                evs = []
                log.warning("detect_events failed: %s", e)
            for ev in evs:
                self._log_event(str(ev))
                if self.mode == GameMode.AUTO:
                    self._react_to_event(ev)
        self._prev_slots = curr_slots

        # 4. Drive every drone via its role + navigation.
        snap = self.world.snapshot(self.mode)
        for did in ids:
            await self._drive_drone(did, snap, now)

    async def _poll_fleet(self, ids: list[str], now: float) -> None:
        """Concurrently read position+telemetry for all drones, ingest each."""

        async def _one(did: str) -> None:
            fc = self.fcs.get(did)
            if fc is None:
                return
            pos, tel = await asyncio.gather(
                fc.get_position(), fc.get_telemetry()
            )
            try:
                self.world.ingest_position(did, pos or {}, now)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] ingest_position failed: %s", did, e)
            try:
                self.world.ingest_telemetry(did, tel or {}, now)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] ingest_telemetry failed: %s", did, e)

        if ids:
            await asyncio.gather(*(_one(d) for d in ids))

    async def _drain_commands(self) -> None:
        """Apply every queued command (FIFO) until the queue is empty."""
        while True:
            try:
                cmd = self.cmd_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await self._apply_command(cmd)
            except Exception as e:  # noqa: BLE001 — a bad command shouldn't crash the tick
                log.warning("apply_command %s failed: %s", cmd.type, e)
                self._log_event(f"command {cmd.type.value} failed: {e}")
            finally:
                self.cmd_queue.task_done()

    # ------------------------------------------------------------------ command application

    def _fleet_targets(self, drone_id: Optional[str]) -> list[str]:
        """Resolve a command's drone_id (None / 'all' -> whole fleet)."""
        if drone_id in (None, "all"):
            return self._drone_ids()
        return [drone_id]

    async def _apply_command(self, cmd: Command) -> None:
        """Mutate fleet/world state for one manual command (runs on the loop).

        Mapping (see models.CommandType):
            SET_MODE     -> set_mode(mode)
            ASSIGN_ROLE  -> swap Role (+ world role), reset memory
            ATTACK       -> role ATTACKER + assigned_slot=slot, reset memory
            DEFEND       -> role DEFENDER + assigned_slot=slot|None, reset memory
            SCOUT        -> role SCOUT, reset memory
            RETURN_HOME  -> role IDLE + memory["return_home"]=True (the tick
                            loop flies it home, then it idle-holds; see
                            _drive_drone)
            TAKEOFF/LAND/EMERGENCY -> direct FC call (one drone or all)
            STOP         -> role IDLE (hover in place), clear assigned_slot
        """
        t = cmd.type

        if t == CommandType.SET_MODE:
            self.set_mode(cmd.mode)
            return

        if t == CommandType.ASSIGN_ROLE:
            self._set_role(cmd.drone_id, cmd.role)
            # A bare role assignment clears any stale slot from a prior order.
            self.world.set_assigned_slot(cmd.drone_id, None)
            self._log_event(f"{cmd.drone_id}: role -> {cmd.role.value}")
            return

        if t == CommandType.ATTACK:
            self._set_role(cmd.drone_id, DroneRole.ATTACKER)
            self.world.set_assigned_slot(cmd.drone_id, cmd.slot)
            self._log_event(f"{cmd.drone_id}: ATTACK slot {cmd.slot}")
            return

        if t == CommandType.DEFEND:
            self._set_role(cmd.drone_id, DroneRole.DEFENDER)
            self.world.set_assigned_slot(cmd.drone_id, cmd.slot)
            tail = f"slot {cmd.slot}" if cmd.slot is not None else "patrol"
            self._log_event(f"{cmd.drone_id}: DEFEND {tail}")
            return

        if t == CommandType.SCOUT:
            self._set_role(cmd.drone_id, DroneRole.SCOUT)
            self.world.set_assigned_slot(cmd.drone_id, None)
            self._log_event(f"{cmd.drone_id}: SCOUT")
            return

        if t == CommandType.RETURN_HOME:
            # RETURN_HOME design choice: rather than overload a real role, we
            # switch the drone to IDLE and raise a per-drone memory flag. The
            # tick loop (see _drive_drone) intercepts that flag and issues a
            # direct GOTO our-home-centre at the drone's cruise altitude until
            # it is inside the home zone, then clears the flag — at which point
            # the IdleRole takes over and simply hovers there. This actually
            # flies the drone home (unlike a plain DEFENDER/IDLE hold), needs
            # no sibling I/O, and degrades safely to a hover on arrival.
            self._set_role(cmd.drone_id, DroneRole.IDLE)
            self.world.set_assigned_slot(cmd.drone_id, None)
            self.memory.setdefault(cmd.drone_id, {})["return_home"] = True
            self._log_event(f"{cmd.drone_id}: RETURN_HOME")
            return

        if t == CommandType.STOP:
            for did in self._fleet_targets(cmd.drone_id):
                self._set_role(did, DroneRole.IDLE)
                self.world.set_assigned_slot(did, None)
            self._log_event(f"STOP {cmd.drone_id or 'all'} (hover in place)")
            return

        if t in (CommandType.TAKEOFF, CommandType.LAND, CommandType.EMERGENCY):
            await self._fc_action(t, cmd.drone_id)
            return

        log.warning("unhandled command type: %s", t)

    async def _fc_action(self, t: CommandType, drone_id: Optional[str]) -> None:
        """Issue a direct flight command to one drone or the whole fleet."""
        targets = self._fleet_targets(drone_id)

        async def _one(did: str) -> None:
            fc = self.fcs.get(did)
            if fc is None:
                return
            if t == CommandType.TAKEOFF:
                await fc.takeoff()
            elif t == CommandType.LAND:
                await fc.land()
            elif t == CommandType.EMERGENCY:
                await fc.emergency()

        if targets:
            await asyncio.gather(*(_one(d) for d in targets))
        self._log_event(f"{t.value} -> {drone_id or 'all'}")

    # ------------------------------------------------------------------ AUTO reactions

    def _react_to_event(self, ev) -> None:
        """AUTO-mode response to a GameEvent (only called when mode==AUTO)."""
        kind = getattr(ev, "kind", "")
        slot = getattr(ev, "slot", None)

        if kind == "enemy_captured_our_slot" and slot is not None:
            from .assignments import pick_uncap_defender
            snap = self.world.snapshot(self.mode)
            did = pick_uncap_defender(slot, snap, self.cfg)
            if did is not None:
                # Send that defender to un-cap the flipped box.
                self.world.set_assigned_slot(did, slot)
                self._log_event(
                    f"AUTO: dispatch defender {did} to un-cap slot {slot}"
                )
            else:
                self._log_event(
                    f"AUTO: slot {slot} flipped but no free defender available"
                )
            return

        if kind == "we_captured_enemy_slot" and slot is not None:
            # Keep it simple + safe: just note it. Freeing the attacker to grab
            # a fresh enemy slot is left to the attacker role / operator so we
            # don't thrash assignments mid-capture-validation.
            self._log_event(f"AUTO: captured enemy slot {slot} (holding plan)")
            return

        # slot_color_changed / slot_lost: logged by the caller already; no
        # automatic reaction (avoid over-reacting to noisy detections).

    # ------------------------------------------------------------------ per-drone drive

    async def _drive_drone(self, did: str, snap: WorldSnapshot, now: float) -> None:
        """Run one drone's role -> RoleAction -> navigation -> FC command."""
        fc = self.fcs.get(did)
        role = self.roles.get(did)
        if fc is None or role is None:
            return

        drone = snap.drones.get(did) or self.world.get_drone(did)
        mem = self.memory.setdefault(did, {})

        # --- RETURN_HOME override -----------------------------------------
        # Handled here (not in a role) so it's fully self-contained. Fly a
        # straight GOTO our-home-centre until inside the home zone, then drop
        # the flag and let the IdleRole hover.
        if mem.get("return_home"):
            action = self._return_home_action(drone, mem)
            if action is not None:
                # Mirror the RETURNING phase so the UI reflects the override.
                try:
                    self.world.set_phase(did, action.next_phase)
                except Exception:  # noqa: BLE001
                    pass
                await self._execute_action(fc, drone, action, now)
                return
            # action is None -> arrived home; flag cleared, fall through to the
            # (now IDLE) role which will hover.

        # --- normal role-driven behaviour ---------------------------------
        from .roles.base import RoleContext
        ctx = RoleContext(config=self.cfg, now=now, memory=mem)
        try:
            action = role.tick(drone, snap, ctx)
        except Exception as e:  # noqa: BLE001 — a broken role shouldn't kill the loop
            log.warning("[%s] role.tick failed: %s", did, e)
            return

        # Mirror the role's reported phase into the world model.
        try:
            self.world.set_phase(did, action.next_phase)
        except Exception:  # noqa: BLE001
            pass

        await self._execute_action(fc, drone, action, now)

    def _return_home_action(self, drone: DroneState, mem: dict):
        """Build the GOTO-home action, or None once the drone has arrived.

        Returns a RoleAction (GOTO home) while still en route, or None when in
        the home zone (clearing the flag) / grounded (nothing to do).
        """
        from .models import ActionKind, RoleAction

        if not drone.flying or drone.position is None:
            mem.pop("return_home", None)
            return None

        home = self.cfg.arena.home_center(self.cfg.our_team)
        if self.cfg.arena.in_home_zone(drone.position, self.cfg.our_team):
            mem.pop("return_home", None)
            self._log_event(f"{drone.drone_id}: reached home, holding")
            return None

        target = Vec3(home.x, home.y, drone.base_altitude_m)
        return RoleAction(
            kind=ActionKind.GOTO,
            target=target,
            next_phase=DronePhase.RETURNING,
            note="return home",
        )

    async def _execute_action(self, fc, drone: DroneState, action, now: float) -> None:
        """Turn a RoleAction into a concrete FC command for this tick."""
        from .models import ActionKind

        kind = action.kind

        if kind == ActionKind.LAND:
            await fc.land()
            return
        if kind == ActionKind.TAKEOFF:
            await fc.takeoff()
            return
        if kind == ActionKind.NONE:
            # "do nothing this tick" == hold position with neutral sticks.
            await fc.send_rc(0, 0, 0, 0)
            return

        # GOTO / HOVER / YAW_SCAN / ORBIT all resolve to RC stick values via
        # navigation.action_to_rc. navigation may not exist yet during early
        # integration; if so, fail safe to a hover (zeros) rather than crash.
        try:
            from .navigation import action_to_rc
        except Exception as e:  # noqa: BLE001 — sibling not ready yet
            log.warning("navigation unavailable (%s); hovering", e)
            await fc.send_rc(0, 0, 0, 0)
            return

        try:
            lr, fb, ud, yaw = action_to_rc(action, drone, self.cfg, self.memory)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] action_to_rc failed: %s; hovering", drone.drone_id, e)
            await fc.send_rc(0, 0, 0, 0)
            return

        await fc.send_rc(int(lr), int(fb), int(ud), int(yaw))


# ---------------------------------------------------------------------------
# Self-test: exercise the command plumbing WITHOUT real drones.
# ---------------------------------------------------------------------------

async def _selftest() -> None:
    """Prove submit() / set_mode() / snapshot() work with zero drones.

    Builds a Commander on a C2Config that has an empty fleet, so no DroneFC is
    ever constructed and no network I/O happens. We drive a few ticks of the
    loop by hand (start() then a brief sleep) to show the plumbing is alive,
    then exercise the public API and assert the results.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = C2Config()           # zero drones (no `drones:` -> empty fleet)
    assert cfg.drones == [], "selftest expects an empty fleet"

    cmd = Commander(cfg)
    print(f"initial mode: {cmd.mode.value}")
    assert cmd.mode == GameMode.MANUAL  # mode_start_auto defaults False

    # --- start the loop (no FCs to connect; loop just idles) ---
    await cmd.start()
    assert cmd._task is not None and not cmd._task.done(), "loop should be live"

    # --- set_mode ---
    cmd.set_mode(GameMode.AUTO)
    assert cmd.mode == GameMode.AUTO
    print(f"after set_mode: {cmd.mode.value}")

    # --- submit: a valid SET_MODE round-trips through validation + queue ---
    r1 = cmd.submit(Command(type=CommandType.SET_MODE, mode=GameMode.MANUAL))
    print("submit SET_MODE(manual):", r1)
    assert r1["ok"] is True

    # --- submit: invalid (unknown drone) is rejected synchronously ---
    r2 = cmd.submit(Command(type=CommandType.ATTACK, drone_id="ghost", slot=1))
    print("submit ATTACK ghost:", r2)
    assert r2["ok"] is False and "unknown drone_id" in r2["error"]

    # --- submit: ATTACK without a slot is rejected ---
    r3 = cmd.submit(Command(type=CommandType.ATTACK, drone_id=None))
    print("submit ATTACK (no slot):", r3)
    assert r3["ok"] is False

    # --- submit: fleet-wide TAKEOFF 'all' validates even with no drones ---
    r4 = cmd.submit(Command(type=CommandType.TAKEOFF, drone_id="all"))
    print("submit TAKEOFF all:", r4)
    assert r4["ok"] is True

    # --- let the loop run a few ticks so the queued cmds are drained ---
    await asyncio.sleep(0.6)
    # The queued SET_MODE(manual) should have been applied by now.
    print(f"mode after draining queue: {cmd.mode.value}")
    assert cmd.mode == GameMode.MANUAL, cmd.mode

    # --- snapshot: well-formed, no drones, mode attached, events present ---
    snap = cmd.snapshot()
    print(
        "snapshot: mode=%s our_team=%s drones=%d slots=%d events=%d"
        % (
            snap.mode.value,
            snap.our_team.value,
            len(snap.drones),
            len(snap.slots),
            len(snap.events),
        )
    )
    assert snap.mode == GameMode.MANUAL
    assert len(snap.drones) == 0
    assert len(snap.slots) == 6                  # world model always seeds 1..6
    assert isinstance(snap.to_dict(), dict)      # JSON-serialisable for the web
    assert len(snap.events) >= 1                 # we logged start + mode changes

    print("recent events:")
    for line in snap.events:
        print("   ", line)

    # --- stop cleanly ---
    await cmd.stop()
    assert cmd._task is None
    assert not cmd._running
    print("\nOK commander selftest (command plumbing + snapshot verified)")


if __name__ == "__main__":
    asyncio.run(_selftest())
