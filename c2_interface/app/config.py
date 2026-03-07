from pydantic import BaseModel


class ArenaConfig(BaseModel):
    width_m: int = 20
    height_m: int = 10
    cell_size_m: int = 1

    # Split along X (20m): red(0-5), neutral(6-13), blue(14-19)
    red_width_cells: int = 6
    neutral_width_cells: int = 8
    blue_width_cells: int = 6


class MissionConfig(BaseModel):
    hover_seconds_on_target: int = 5
    speed_cells_per_sec: float = 1.0


class Settings(BaseModel):
    arena: ArenaConfig = ArenaConfig()
    mission: MissionConfig = MissionConfig()


settings = Settings()
