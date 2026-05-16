"""Per-FC state cache, polled in the background on the C2 event loop.

Flask handlers read snapshots via :meth:`FCPool.snapshot` — deep-copied
so the UI never sees a half-updated dict. The pool also owns the one
``httpx.AsyncClient`` and the per-FC poll tasks.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import C2Config, FCSpec
from .fc_client import AsyncFCClient, make_client


@dataclass
class FCState:
    name: str
    host: str
    port: int
    last_state: Optional[dict] = None
    last_state_monotonic: float = 0.0   # time.monotonic() of last successful poll
    last_state_wall: Optional[float] = None  # time.time() of last successful poll
    connection_ok: bool = False
    drone_serial: Optional[str] = None
    drone_connected: bool = False
    last_error: Optional[str] = None
    identity: Optional[dict] = None
    identity_monotonic: float = 0.0
    calibrations: list[dict] = field(default_factory=list)
    calibrations_monotonic: float = 0.0


class FCPool:
    """Holds one :class:`AsyncFCClient` + :class:`FCState` per configured FC.

    Lifecycle: ``start()`` spawns per-FC poll tasks; ``stop()`` cancels
    them and closes the HTTP client. Both must be awaited on the C2
    event loop.
    """

    def __init__(self, cfg: C2Config):
        self.cfg = cfg
        self.http = make_client(cfg.fc_request_timeout_s)
        self.clients: dict[str, AsyncFCClient] = {}
        self.states: dict[str, FCState] = {}
        for spec in cfg.fcs:
            self.clients[spec.name] = AsyncFCClient(
                spec, self.http, default_timeout_s=cfg.fc_request_timeout_s)
            self.states[spec.name] = FCState(
                name=spec.name, host=spec.host, port=spec.port)
        self._tasks: list[asyncio.Task] = []
        self._stop_evt = asyncio.Event()
        self._log = logging.getLogger("c2.pool")
        self._locks: dict[str, asyncio.Lock] = {
            spec.name: asyncio.Lock() for spec in cfg.fcs
        }

    # --------------------------------------------------------- lifecycle
    async def start(self) -> None:
        interval = 1.0 / max(0.1, self.cfg.state_poll_hz)
        for spec in self.cfg.fcs:
            self._tasks.append(asyncio.create_task(
                self._poll_loop(spec, interval),
                name=f"c2-poll-{spec.name}",
            ))
            self._log.info("FC poll task started: %s (%s)",
                           spec.name, spec.base_url)

    async def stop(self) -> None:
        self._stop_evt.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await self.http.aclose()

    # --------------------------------------------------------- poll
    async def _poll_loop(self, spec: FCSpec, interval: float) -> None:
        client = self.clients[spec.name]
        # Stagger first-poll across FCs by a few ms so we don't hit the
        # connection pool with a 6-way burst at startup.
        await asyncio.sleep(0.05 * list(self.states.keys()).index(spec.name))
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            ok, payload = await client.get_state()
            await self._update_state(spec.name, ok, payload)
            elapsed = time.monotonic() - t0
            try:
                await asyncio.wait_for(
                    self._stop_evt.wait(),
                    timeout=max(0.0, interval - elapsed),
                )
                return  # stop event set
            except asyncio.TimeoutError:
                continue

    async def _update_state(self, name: str, ok: bool, payload: Any) -> None:
        async with self._locks[name]:
            st = self.states[name]
            if ok and isinstance(payload, dict):
                st.last_state = payload
                st.last_state_monotonic = time.monotonic()
                st.last_state_wall = time.time()
                st.connection_ok = True
                st.last_error = None
                tel = payload.get("telemetry") or {}
                serial = tel.get("serial_number") or tel.get("serial")
                if isinstance(serial, str) and serial:
                    st.drone_serial = serial
                st.drone_connected = bool(payload.get("drone_connected"))
            else:
                st.connection_ok = False
                st.last_error = (
                    payload if isinstance(payload, str)
                    else (payload.get("body") if isinstance(payload, dict)
                          else "unknown")
                )

    # --------------------------------------------------------- public read
    def snapshot(self) -> dict[str, dict]:
        """Return a JSON-safe deep copy of every FC's state.

        Called from Flask handlers; the deep copy is essential because the
        poll loop mutates ``FCState.last_state`` under a lock that Flask
        doesn't hold.
        """
        out: dict[str, dict] = {}
        now = time.monotonic()
        for name, st in self.states.items():
            last_state = copy.deepcopy(st.last_state) if st.last_state else None
            age = (now - st.last_state_monotonic
                   if st.last_state_monotonic else None)
            out[name] = {
                "name": st.name,
                "host": st.host,
                "port": st.port,
                "base_url": f"http://{st.host}:{st.port}",
                "connection_ok": st.connection_ok,
                "drone_serial": st.drone_serial,
                "drone_connected": st.drone_connected,
                "last_error": st.last_error,
                "last_state_age_s": age,
                "last_state_wall": st.last_state_wall,
                "state": last_state,
                "identity": copy.deepcopy(st.identity),
                "calibrations": copy.deepcopy(st.calibrations),
                "calibrations_age_s": (
                    now - st.calibrations_monotonic
                    if st.calibrations_monotonic else None
                ),
            }
        return out

    def all_clients(self) -> list[AsyncFCClient]:
        return list(self.clients.values())

    def client(self, name: str) -> Optional[AsyncFCClient]:
        return self.clients.get(name)

    # ------------------------------ helpers used by calibration_sync etc.
    async def set_calibrations(self, name: str, entries: list[dict]) -> None:
        async with self._locks[name]:
            self.states[name].calibrations = list(entries)
            self.states[name].calibrations_monotonic = time.monotonic()

    async def set_identity(self, name: str, identity: Optional[dict]) -> None:
        async with self._locks[name]:
            self.states[name].identity = identity
            if identity:
                self.states[name].identity_monotonic = time.monotonic()
                serial = identity.get("drone_serial")
                if isinstance(serial, str) and serial:
                    # Only override the state-derived serial when /api/state
                    # hasn't seen one yet; /api/state's serial wins because
                    # it's live, /api/identity may report a remembered one
                    # from a previous flight.
                    if not self.states[name].drone_serial:
                        self.states[name].drone_serial = serial
