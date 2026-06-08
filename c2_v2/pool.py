"""Async FC pool — polls every flight controller's ``/api/state`` and keeps a
live ``DroneWorld`` per drone.

One ``asyncio`` task per FC reads ``/api/state`` at ``PollConfig.state_hz`` and
(slowly) ``/api/identity`` + ``/api/wlan`` for the serial and WiFi SSID. The
pool exposes a thread-safe ``snapshot()`` for the Flask dashboard, which runs on
a different thread.

Reuses the proven per-FC HTTP client (``marker_mission_c2.fc_client``) and FC
spec; adds nothing to ``marker_mission``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

# Proven, debugged per-FC HTTP client + spec — reuse, don't re-implement.
from marker_mission_c2.config import FCSpec
from marker_mission_c2.fc_client import AsyncFCClient, make_client

from .config import PollConfig
from .world import DroneWorld

logger = logging.getLogger("c2_v2.pool")


class FCPool:
    def __init__(self, specs: List[FCSpec], poll: Optional[PollConfig] = None,
                 match: Any = None):
        # ``match`` (a MatchState) receives each drone's visible markers every
        # tick to maintain the live box-state. Optional so the pool also works
        # standalone (Phase 1). Duck-typed: only ``.ingest(fc, ids)`` is used.
        self._match = match
        self.poll = poll or PollConfig()
        self._http: httpx.AsyncClient = make_client(self.poll.request_timeout_s)
        self.clients: Dict[str, AsyncFCClient] = {
            s.name: AsyncFCClient(s, self._http, self.poll.request_timeout_s)
            for s in specs
        }
        self.worlds: Dict[str, DroneWorld] = {
            s.name: DroneWorld(name=s.name, base_url=s.base_url) for s in specs
        }
        self._tasks: List[asyncio.Task] = []
        self._wifi: Dict[str, str] = {}     # fc -> connected SSID
        self._running = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for name in self.clients:
            self._tasks.append(asyncio.create_task(
                self._poll_loop(name), name=f"c2v2-poll-{name}"))
        logger.info("c2_v2 pool: polling %d FC(s): %s",
                    len(self.clients), ", ".join(self.clients))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.sleep(0.05)
        try:
            await self._http.aclose()
        except Exception:
            pass

    # ------------------------------------------------------------------ poll loop
    async def _poll_loop(self, name: str) -> None:
        client = self.clients[name]
        world = self.worlds[name]
        period = 1.0 / max(0.5, self.poll.state_hz)
        # Stagger the five loops so requests don't all fire on the same tick.
        await asyncio.sleep(0.05 * list(self.clients).index(name))
        next_identity = 0.0
        next_wlan = 0.0
        while self._running:
            t0 = time.monotonic()
            now = time.time()
            ok, payload = await client.get_state()
            if ok and isinstance(payload, dict):
                world.update_from_state(payload, now)
                # Feed the box-state tracker with this drone's observations.
                if self._match is not None and world.visible_marker_ids:
                    try:
                        self._match.ingest(name, world.visible_marker_ids)
                    except Exception as exc:
                        logger.debug("ingest(%s) failed: %s", name, exc)
            else:
                err = payload if isinstance(payload, str) else str(payload)
                world.mark_disconnected(err, now, self.poll.stale_after_s)

            # Slow-cadence reads: serial (identity) + WiFi SSID (wlan).
            if t0 >= next_identity:
                next_identity = t0 + self.poll.identity_period_s
                iok, ipayload = await client.get_identity()
                if iok and isinstance(ipayload, dict):
                    sn = ipayload.get("serial_number") or ipayload.get("serial")
                    if sn:
                        world.serial = str(sn)
            if t0 >= next_wlan:
                next_wlan = t0 + self.poll.wlan_period_s
                wok, wpayload = await client.get_wlan()
                if wok and isinstance(wpayload, dict):
                    self._wifi[name] = _extract_ssid(wpayload)

            dt = time.monotonic() - t0
            await asyncio.sleep(max(0.02, period - dt))

    # ------------------------------------------------------------------ commands
    async def push_and_start(self, name: str, script: str) -> tuple[bool, Any]:
        """Set + start a mission on one FC in a single call (the FC's /api/start
        accepts the script). The FC only starts from idle, so callers gate on
        the drone's phase being init/done/''."""
        client = self.clients.get(name)
        if client is None:
            return False, "unknown fc"
        return await client.start_mission(script)

    async def stop(self, name: str) -> tuple[bool, Any]:
        """Stop one FC's mission. NOTE: the FC LANDS on stop (rc_zero + land)."""
        client = self.clients.get(name)
        if client is None:
            return False, "unknown fc"
        return await client.stop_mission()

    async def emergency_land_all(self) -> Dict[str, bool]:
        """Land EVERY drone immediately (stop each mission -> rc_zero + land).
        409 'not running' is treated as success (already idle)."""
        results: Dict[str, bool] = {}
        for name, client in self.clients.items():
            ok, _ = await client.stop_mission()
            results[name] = bool(ok)
        return results

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe-enough read for the dashboard. The worlds are mutated by
        the asyncio thread; we build a fresh dict of plain values here, which is
        atomic per-field for the dashboard's purposes (read-only display)."""
        drones = []
        for name, w in self.worlds.items():
            d = w.to_client()
            d["wifi_ssid"] = self._wifi.get(name, "")
            drones.append(d)
        connected = sum(1 for d in drones if d["connected"])
        return {
            "now": time.time(),
            "fc_count": len(self.worlds),
            "connected_count": connected,
            "drones": drones,
        }


def _extract_ssid(wlan: Dict[str, Any]) -> str:
    """Pull the connected SSID from /api/wlan's {iface: {ssid, ...}} body."""
    for iface, info in wlan.items():
        if not isinstance(info, dict):
            continue
        ssid = info.get("ssid")
        if ssid and str(iface).startswith("wl"):
            return str(ssid)
    # Fallback: any iface with an SSID.
    for info in wlan.values():
        if isinstance(info, dict) and info.get("ssid"):
            return str(info["ssid"])
    return ""
