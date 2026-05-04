from __future__ import annotations

from .fallback_policy import plan_fallback
from .llm_planner import LlmPlanner
from .models import MatchState, PlannerResult
from .validator import validate_commands


class StrategyController:
    def __init__(self, llm_planner: LlmPlanner | None = None, use_llm: bool = True) -> None:
        self.llm_planner = llm_planner
        self.use_llm = use_llm
        self.last_error: str | None = None

    def next_plan(self, state: MatchState) -> PlannerResult:
        if self.use_llm and self.llm_planner:
            try:
                llm_result = self.llm_planner.plan(state)
                validation = validate_commands(state, llm_result.commands)
                if validation.ok:
                    llm_result.commands = validation.commands
                    self.last_error = None
                    return llm_result
                self.last_error = "; ".join(validation.errors + validation.warnings)
            except Exception as exc:  # noqa: BLE001 - strategy loop must degrade safely.
                self.last_error = str(exc)

        fallback = plan_fallback(state)
        validation = validate_commands(state, fallback.commands)
        fallback.commands = validation.commands
        if validation.errors:
            self.last_error = "; ".join(validation.errors)
        return fallback
