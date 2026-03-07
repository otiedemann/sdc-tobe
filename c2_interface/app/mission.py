import asyncio

from .aruco_nav import ArucoNavigator
from .config import settings
from .drone_adapter import DroneAdapter
from .models import DroneStatus, GridPoint
from .state import StateStore


class MissionController:
    def __init__(self, store: StateStore, adapter: DroneAdapter, aruco: ArucoNavigator) -> None:
        self.store = store
        self.adapter = adapter
        self.aruco = aruco
        self._tasks: dict[str, asyncio.Task] = {}

    async def send_drone_to(self, drone_id: str, cell: GridPoint) -> None:
        if drone_id in self._tasks and not self._tasks[drone_id].done():
            self._tasks[drone_id].cancel()

        self._tasks[drone_id] = asyncio.create_task(self._run_move_and_search(drone_id, cell))

    async def mark_target_spotted(self, drone_id: str) -> None:
        d = await self.store.get_drone(drone_id)
        await self.store.update_drone(drone_id, target_seen=True, status=DroneStatus.TARGET_HOVER)
        await self.adapter.hover(drone_id, settings.mission.hover_seconds_on_target)
        await asyncio.sleep(settings.mission.hover_seconds_on_target)

        await self.store.update_drone(drone_id, status=DroneStatus.RETURNING_HOME)
        await self.adapter.return_home(drone_id, d.home)
        await self._simulate_travel(drone_id, d.home)
        await self.store.update_drone(
            drone_id,
            status=DroneStatus.IDLE,
            target_seen=False,
            target_cell=None,
        )

    async def _run_move_and_search(self, drone_id: str, cell: GridPoint) -> None:
        await self.store.update_drone(drone_id, status=DroneStatus.MOVING, target_cell=cell, target_seen=False)
        await self.adapter.fly_to_cell(drone_id, cell)
        await self._simulate_travel(drone_id, cell)

        localized = self.aruco.localize(drone_id, cell)
        await self.store.update_drone(drone_id, position=localized, status=DroneStatus.SEARCHING)

    async def _simulate_travel(self, drone_id: str, dst: GridPoint) -> None:
        d = await self.store.get_drone(drone_id)
        src = d.position
        dist = abs(src.x - dst.x) + abs(src.y - dst.y)
        duration = dist / settings.mission.speed_cells_per_sec
        await asyncio.sleep(max(0.3, duration))
        await self.store.update_drone(drone_id, position=dst)
