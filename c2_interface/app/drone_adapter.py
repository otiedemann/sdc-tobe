"""
Drone adapter abstraction.

For real drones, implement movement + hover + return commands here.
"""

from .models import GridPoint


class DroneAdapter:
    async def fly_to_cell(self, drone_id: str, target: GridPoint) -> None:
        # TODO: call simulator or flight stack command.
        return

    async def hover(self, drone_id: str, seconds: int) -> None:
        # TODO: call hold/hover command on drone.
        return

    async def return_home(self, drone_id: str, home: GridPoint) -> None:
        # TODO: send return-to-home mission command.
        return
