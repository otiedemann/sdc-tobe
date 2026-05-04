from __future__ import annotations

from typing import Any

from .models import DroneCommand, MatchState, PlannerResult, Pose
from .openrouter_client import OpenRouterClient
from .prompts import SYSTEM_PROMPT, build_user_prompt


class LlmPlanner:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def plan(self, state: MatchState) -> PlannerResult:
        raw = self.client.chat_json(SYSTEM_PROMPT, build_user_prompt(state))
        return PlannerResult(
            mode=str(raw.get("mode", "llm_plan")),
            strategy_summary=str(raw.get("strategy_summary", "LLM tactical plan.")),
            commands=_parse_commands(raw.get("commands", [])),
            source="llm",
            raw_response=raw,
        )


def _parse_commands(items: list[Any]) -> list[DroneCommand]:
    commands: list[DroneCommand] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        position = item.get("target_position") or {}
        commands.append(
            DroneCommand(
                drone_id=str(item.get("drone_id", "")),
                role=str(item.get("role", "flex")),
                objective=str(item.get("objective", "regroup")),  # type: ignore[arg-type]
                target_id=item.get("target_id"),
                target_position=Pose(
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 1.8)),
                    float(position.get("yaw", 0.0)),
                ),
                hold_seconds=float(item.get("hold_seconds", 0.0)),
                priority=int(item.get("priority", 0)),
                reason=str(item.get("reason", "")),
            )
        )
    return commands
