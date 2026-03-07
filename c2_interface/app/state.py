import asyncio
from collections.abc import Callable

from .models import DroneState


class StateStore:
    def __init__(self) -> None:
        self._drones: dict[str, DroneState] = {}
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    async def _notify(self) -> None:
        for fn in self._listeners:
            fn()

    async def upsert_drone(self, drone: DroneState) -> None:
        async with self._lock:
            self._drones[drone.drone_id] = drone
        await self._notify()

    async def update_drone(self, drone_id: str, **kwargs) -> DroneState:
        async with self._lock:
            d = self._drones[drone_id]
            updated = d.model_copy(update=kwargs)
            self._drones[drone_id] = updated
        await self._notify()
        return updated

    async def get_drone(self, drone_id: str) -> DroneState:
        async with self._lock:
            return self._drones[drone_id]

    async def all_drones(self) -> list[DroneState]:
        async with self._lock:
            return list(self._drones.values())
