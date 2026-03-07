from .config import settings
from .models import ArenaCell


class ArenaGrid:
    def __init__(self) -> None:
        self.width = settings.arena.width_m // settings.arena.cell_size_m
        self.height = settings.arena.height_m // settings.arena.cell_size_m

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def zone_for_x(self, x: int) -> str:
        red = settings.arena.red_width_cells
        neutral = settings.arena.neutral_width_cells
        if x < red:
            return "red"
        if x < red + neutral:
            return "neutral"
        return "blue"

    def to_cells(self) -> list[ArenaCell]:
        cells: list[ArenaCell] = []
        for y in range(self.height):
            for x in range(self.width):
                cells.append(ArenaCell(x=x, y=y, zone=self.zone_for_x(x)))
        return cells
