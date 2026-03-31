"""
Unified async HTTP client for controlling drones via the API.

Supports all three backends:
- sim_api_server (simulator) — single server, multiple drones via select_drone
- tello_pi_api_server — one server per Tello drone
- olympe_pi_api_server — one server per Anafi drone

For the simulator, commands are serialized per API server to prevent
select_drone race conditions (select + command must be atomic).
"""

import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger("drone_client")


class DroneClient:
    """Controls a single drone through its API server."""

    def __init__(
        self,
        drone_id: str,
        api_base_url: str,
        drone_type: str = "simulator",
        sim_drone_id: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.drone_id = drone_id
        self.api_base_url = api_base_url.rstrip("/")
        self.drone_type = drone_type
        self.sim_drone_id = sim_drone_id  # e.g. "B1" for simulator
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, path: str, body: dict | None = None) -> dict:
        try:
            r = await self._client.post(f"{self.api_base_url}{path}", json=body or {})
            return r.json()
        except Exception as e:
            log.warning(f"[{self.drone_id}] POST {path} failed: {e}")
            return {"ok": False, "error": str(e)}

    async def _get(self, path: str) -> dict:
        try:
            r = await self._client.get(f"{self.api_base_url}{path}")
            return r.json()
        except Exception as e:
            log.warning(f"[{self.drone_id}] GET {path} failed: {e}")
            return {"ok": False, "error": str(e)}

    async def select(self) -> dict:
        """Select this drone on the simulator (no-op for real drones)."""
        if self.drone_type == "simulator" and self.sim_drone_id:
            return await self._post("/api/select_drone", {"drone_id": self.sim_drone_id})
        return {"ok": True}

    async def takeoff(self) -> dict:
        return await self._post("/api/takeoff")

    async def land(self) -> dict:
        return await self._post("/api/land")

    async def emergency(self) -> dict:
        return await self._post("/api/emergency")

    async def move(self, direction: str, distance_cm: int) -> dict:
        return await self._post("/api/move", {"direction": direction, "distance": distance_cm})

    async def rotate(self, direction: str, degrees: int) -> dict:
        return await self._post("/api/rotate", {"direction": direction, "degrees": degrees})

    async def rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> dict:
        return await self._post("/api/rc", {
            "left_right": lr, "forward_back": fb, "up_down": ud, "yaw": yaw,
        })

    async def move_to(self, x: float, y: float, z: float = 1.0) -> dict:
        """Move to a position (Olympe/sim only)."""
        return await self._post("/api/moveto", {"x": x, "y": y, "z": z})

    async def go(self, x: int, y: int, z: int, speed: int = 50) -> dict:
        return await self._post("/api/go", {"x": x, "y": y, "z": z, "speed": speed})

    async def get_telemetry(self) -> dict:
        # Request per-drone telemetry using sim_drone_id (e.g. "B1")
        if self.sim_drone_id:
            return await self._get(f"/api/telemetry?drone_id={self.sim_drone_id}")
        return await self._get("/api/telemetry")

    async def recover(self) -> dict:
        return await self._post("/api/recover")

    async def key_down(self, key: str) -> dict:
        return await self._post("/api/key_down", {"key": key})

    async def key_up(self, key: str) -> dict:
        return await self._post("/api/key_up", {"key": key})

    async def close(self) -> None:
        await self._client.aclose()


class FleetClient:
    """
    Manages a fleet of DroneClients.

    For simulator mode, uses a per-API-server lock to serialize
    select_drone + command sequences.
    """

    def __init__(self) -> None:
        self.drones: dict[str, DroneClient] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def add_drone(
        self,
        drone_id: str,
        api_base_url: str,
        drone_type: str = "simulator",
        sim_drone_id: Optional[str] = None,
    ) -> DroneClient:
        client = DroneClient(
            drone_id=drone_id,
            api_base_url=api_base_url,
            drone_type=drone_type,
            sim_drone_id=sim_drone_id,
        )
        self.drones[drone_id] = client
        # Ensure a lock exists per API base URL
        if api_base_url not in self._locks:
            self._locks[api_base_url] = asyncio.Lock()
        return client

    def _lock_for(self, drone_id: str) -> asyncio.Lock:
        client = self.drones[drone_id]
        return self._locks[client.api_base_url]

    async def command_drone(self, drone_id: str, method: str, **kwargs) -> dict:
        """
        Execute a command on a specific drone.
        For simulator drones, serializes select + command behind a lock.
        """
        client = self.drones.get(drone_id)
        if not client:
            return {"ok": False, "error": f"Unknown drone: {drone_id}"}

        lock = self._lock_for(drone_id)
        async with lock:
            if client.drone_type == "simulator":
                await client.select()
            fn = getattr(client, method, None)
            if fn is None:
                return {"ok": False, "error": f"Unknown method: {method}"}
            return await fn(**kwargs)

    async def takeoff(self, drone_id: str) -> dict:
        return await self.command_drone(drone_id, "takeoff")

    async def land(self, drone_id: str) -> dict:
        return await self.command_drone(drone_id, "land")

    async def move(self, drone_id: str, direction: str, distance_cm: int) -> dict:
        return await self.command_drone(drone_id, "move", direction=direction, distance_cm=distance_cm)

    async def rc(self, drone_id: str, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> dict:
        return await self.command_drone(drone_id, "rc", lr=lr, fb=fb, ud=ud, yaw=yaw)

    async def emergency(self, drone_id: str) -> dict:
        return await self.command_drone(drone_id, "emergency")

    async def get_telemetry(self, drone_id: str) -> dict:
        """Telemetry doesn't need the select lock (read-only)."""
        client = self.drones.get(drone_id)
        if not client:
            return {}
        return await client.get_telemetry()

    async def poll_all_telemetry(self) -> dict[str, dict]:
        """Poll telemetry from all drones concurrently."""
        results = {}
        tasks = {
            drone_id: asyncio.create_task(client.get_telemetry())
            for drone_id, client in self.drones.items()
        }
        for drone_id, task in tasks.items():
            try:
                results[drone_id] = await task
            except Exception:
                results[drone_id] = {}
        return results

    async def takeoff_all(self) -> dict[str, dict]:
        """Take off all drones (serialized for simulator)."""
        results = {}
        for drone_id in self.drones:
            results[drone_id] = await self.takeoff(drone_id)
        return results

    async def land_all(self) -> dict[str, dict]:
        """Land all drones."""
        results = {}
        for drone_id in self.drones:
            results[drone_id] = await self.land(drone_id)
        return results

    async def emergency_all(self) -> dict[str, dict]:
        """Emergency stop all drones."""
        results = {}
        for drone_id in self.drones:
            results[drone_id] = await self.emergency(drone_id)
        return results

    async def close_all(self) -> None:
        for client in self.drones.values():
            await client.close()
