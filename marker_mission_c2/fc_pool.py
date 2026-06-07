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
from .settings import SettingsStore


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
    # Latest WLAN info from the FC host (iwconfig of its wl* interface): the
    # SSID is the connected drone's WiFi name. Polled at a slow cadence.
    wlan: Optional[dict] = None
    wlan_monotonic: float = 0.0


class FCPool:
    """Holds one :class:`AsyncFCClient` + :class:`FCState` per configured FC.

    Lifecycle: ``start()`` spawns per-FC poll tasks; ``stop()`` cancels
    them and closes the HTTP client. Both must be awaited on the C2
    event loop.
    """

    def __init__(self, cfg: C2Config, settings: SettingsStore):
        self.cfg = cfg
        self.settings = settings
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
            # Skip the network round-trip entirely when the operator
            # has disabled this FC from /settings — saves bandwidth,
            # avoids spurious "down" alerts when an FC is intentionally
            # offline, and prevents the calibration sync worker from
            # picking up changes from FCs the operator has silenced.
            if self.settings.is_fc_enabled(spec.name):
                ok, payload = await client.get_state()
                await self._update_state(spec.name, ok, payload)
                # WLAN (which drone WiFi this FC is on) changes rarely and
                # iwconfig is a subprocess on the Pi — poll it at a SLOW cadence
                # (~ every WLAN_POLL_INTERVAL_S) rather than every state tick.
                st = self.states[spec.name]
                if time.monotonic() - st.wlan_monotonic >= self.WLAN_POLL_INTERVAL_S:
                    wok, wpayload = await client.get_wlan()
                    await self._update_wlan(spec.name, wok, wpayload)
            else:
                await self._mark_disabled(spec.name)
            elapsed = time.monotonic() - t0
            try:
                await asyncio.wait_for(
                    self._stop_evt.wait(),
                    timeout=max(0.0, interval - elapsed),
                )
                return  # stop event set
            except asyncio.TimeoutError:
                continue

    # WLAN changes rarely; poll it this often (seconds) instead of every tick.
    WLAN_POLL_INTERVAL_S: float = 4.0

    @staticmethod
    def _primary_wlan(payload: Any) -> Optional[dict]:
        """Pick the connected wireless interface from an /api/wlan payload:
        the first ``wl*`` iface that actually has an SSID. Returns a compact
        dict (ssid + the headline radio fields) or None."""
        if not isinstance(payload, dict):
            return None
        for iface, info in payload.items():
            if (isinstance(iface, str) and iface.startswith("wl")
                    and isinstance(info, dict) and info.get("ssid")):
                out = {"iface": iface}
                for k in ("ssid", "signal_dbm", "link_quality", "bit_rate",
                          "frequency", "band", "tx_power", "mode"):
                    if info.get(k) not in (None, ""):
                        out[k] = info[k]
                return out
        return None

    async def _update_wlan(self, name: str, ok: bool, payload: Any) -> None:
        async with self._locks[name]:
            st = self.states[name]
            st.wlan_monotonic = time.monotonic()
            if ok:
                st.wlan = self._primary_wlan(payload)

    async def _mark_disabled(self, name: str) -> None:
        async with self._locks[name]:
            st = self.states[name]
            st.connection_ok = False
            st.last_error = "disabled"
            # Wipe the last known state so the overview doesn't show
            # stale telemetry next to a "disabled" badge.
            st.last_state = None
            st.wlan = None

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
    def snapshot(self, include_disabled: bool = False) -> dict[str, dict]:
        """Return a JSON-safe deep copy of every enabled FC's state.

        Disabled FCs are excluded by default (so handlers don't have
        to filter — the overview, fleet broadcasts, and overview-viz
        all get the right behaviour for free). Pass
        ``include_disabled=True`` from the /settings page to render
        the full inventory with their on/off badges.
        """
        out: dict[str, dict] = {}
        now = time.monotonic()
        disabled = self.settings.disabled_fcs()
        for name, st in self.states.items():
            is_disabled = name in disabled
            if is_disabled and not include_disabled:
                continue
            last_state = copy.deepcopy(st.last_state) if st.last_state else None
            age = (now - st.last_state_monotonic
                   if st.last_state_monotonic else None)
            out[name] = {
                "name": st.name,
                "host": st.host,
                "port": st.port,
                "base_url": f"http://{st.host}:{st.port}",
                "enabled": not is_disabled,
                "connection_ok": st.connection_ok,
                "drone_serial": st.drone_serial,
                "drone_connected": st.drone_connected,
                "wlan": copy.deepcopy(st.wlan),
                "wlan_age_s": (now - st.wlan_monotonic
                               if st.wlan_monotonic else None),
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

    def enabled_clients(self) -> list[AsyncFCClient]:
        """Clients for FCs the operator hasn't silenced in /settings.

        Fleet broadcasts (start-all, emergency-land-all, push-to-all)
        should use this so a disabled FC never receives a fan-out
        command.
        """
        disabled = self.settings.disabled_fcs()
        return [c for n, c in self.clients.items() if n not in disabled]

    def client(self, name: str) -> Optional[AsyncFCClient]:
        return self.clients.get(name)

    # ------------------------------ runtime endpoint re-point (sim <-> real)
    def config_endpoint(self, name: str) -> Optional[tuple[str, int]]:
        """The (host, port) this FC was originally CONFIGURED with — used by
        'reset to sim' to undo a runtime re-point."""
        for spec in self.cfg.fcs:
            if spec.name == name:
                return (spec.host, spec.port)
        return None

    async def set_endpoint(self, name: str, host: str, port: int) -> bool:
        """Re-point an existing FC at a new flight-controller endpoint — e.g.
        switch a simulated drone to a REAL drone's IP, or back.

        Runtime-only: a C2 restart reverts to the configured endpoint (the
        config JSON stays the source of truth for permanent fleets). The
        running poll task keeps its client reference and reads
        ``client.spec.base_url`` per request, so swapping the spec re-points it
        with no task restart. Stale connection state is dropped so the overview
        reflects the change immediately."""
        client = self.clients.get(name)
        if client is None:
            return False
        async with self._locks[name]:
            client.spec = FCSpec(name=name, host=str(host), port=int(port))
            st = self.states[name]
            st.host = str(host)
            st.port = int(port)
            st.connection_ok = False
            st.last_error = "endpoint changed — reconnecting"
            st.last_state = None
            st.drone_serial = None
            st.drone_connected = False
        self._log.info("FC %s re-pointed -> %s", name, client.spec.base_url)
        return True

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
