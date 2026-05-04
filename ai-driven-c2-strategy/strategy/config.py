from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AppConfig,
    Arena,
    GameplayConfig,
    LlmConfig,
    MockEnemyConfig,
    OpenRouterConfig,
    SimulationConfig,
    TargetState,
    TeamConfig,
    Vec3,
)


def load_config(path: str | Path) -> AppConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(data)


def parse_config(data: dict[str, Any]) -> AppConfig:
    team = TeamConfig(**data["team"])
    arena = Arena(**data["arena"])
    gameplay = GameplayConfig(**data["gameplay"])
    llm = LlmConfig(**data["llm"])
    openrouter = OpenRouterConfig(**data["openrouter"])
    simulation = SimulationConfig(**data["simulation"])
    mock_enemy = MockEnemyConfig(**data["mock_enemy"])
    targets = []
    for target in data["targets"]:
        side = target["side"]
        owner = team.color if side == team.home_side else team.opponent_color
        position = Vec3(**target["position"])
        targets.append(TargetState(target["id"], side, position, owner, validated_for_owner=owner))
    return AppConfig(
        team=team,
        arena=arena,
        gameplay=gameplay,
        llm=llm,
        openrouter=openrouter,
        simulation=simulation,
        mock_enemy=mock_enemy,
        targets=targets,
    )
