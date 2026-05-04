from __future__ import annotations

import copy
import threading
import time
from collections import deque

from .controller import StrategyController
from .mock_enemy import MockEnemyPlanner
from .models import (
    AppConfig,
    DroneCommand,
    DroneState,
    MatchEvent,
    MatchState,
    PlannerResult,
    Pose,
    ScoreState,
    TargetState,
    TeamColor,
)


class Simulation:
    def __init__(self, config: AppConfig, controller: StrategyController) -> None:
        self.config = config
        self.controller = controller
        self.enemy_planner = MockEnemyPlanner(config.mock_enemy.style)
        self.score = ScoreState()
        self.match_time_seconds = 0.0
        self.events: deque[MatchEvent] = deque(maxlen=80)
        self.own_drones = self._spawn_drones("own", config.simulation.own_drone_count)
        self.enemy_drones = self._spawn_drones("enemy", config.simulation.enemy_drone_count)
        self.targets = copy.deepcopy(config.targets)
        self.current_plan = PlannerResult("startup", "Starting strategy loop.", [], "fallback")
        self.enemy_commands: list[DroneCommand] = []
        self.lock = threading.Lock()
        self.running = False
        self._last_strategy_time = -999.0
        self._last_major_event_time = -999.0

    def start_background(self) -> None:
        self.running = True
        thread = threading.Thread(target=self.run_forever, daemon=True)
        thread.start()

    def run_forever(self) -> None:
        last = time.monotonic()
        while self.running:
            now = time.monotonic()
            dt = min(now - last, 0.25)
            last = now
            self.step(dt)
            time.sleep(self.config.simulation.tick_seconds)

    def step(self, dt: float) -> None:
        with self.lock:
            if self.match_time_seconds >= self.config.gameplay.match_duration_seconds:
                return
            self.match_time_seconds += dt
            state = self._build_state_unlocked()
            if self._should_plan_unlocked():
                self.current_plan = self.controller.next_plan(state)
                self._apply_commands_unlocked(self.own_drones, self.current_plan.commands)
                self._last_strategy_time = self.match_time_seconds
                if self.controller.last_error:
                    self._event_unlocked("planner_warning", self.controller.last_error)

            if self.config.mock_enemy.enabled:
                self.enemy_commands = self.enemy_planner.plan(state)
                self._apply_commands_unlocked(self.enemy_drones, self.enemy_commands)

            self._move_drones_unlocked(self.own_drones, self.config.gameplay.drone_speed_mps, dt)
            self._move_drones_unlocked(self.enemy_drones, self.config.gameplay.enemy_speed_mps, dt)
            self._update_captures_unlocked(dt)

    def snapshot(self) -> dict:
        with self.lock:
            state = self._build_state_unlocked()
            return {
                "match_state": state.to_prompt_dict(),
                "plan": {
                    "mode": self.current_plan.mode,
                    "strategy_summary": self.current_plan.strategy_summary,
                    "source": self.current_plan.source,
                    "commands": [command.to_dict() for command in self.current_plan.commands],
                    "last_error": self.controller.last_error,
                },
                "enemy_commands": [command.to_dict() for command in self.enemy_commands],
                "config": {
                    "gameplay": self.config.gameplay.__dict__,
                    "llm_enabled": self.config.llm.enabled,
                    "llm_model": self.config.llm.model,
                },
            }

    def _spawn_drones(self, team: str, count: int) -> list[DroneState]:
        zone = self.config.own_home_zone if team == "own" else self.config.enemy_home_zone
        x = (zone.x_min + zone.x_max) / 2.0
        spacing = 10.0 / (count + 1)
        drones: list[DroneState] = []
        for index in range(count):
            drones.append(
                DroneState(
                    id=f"{team}_{index + 1}",
                    team=team,  # type: ignore[arg-type]
                    pose=Pose(x, spacing * (index + 1), self.config.gameplay.transit_z, 0.0),
                )
            )
        return drones

    def _build_state_unlocked(self) -> MatchState:
        return MatchState(
            match_time_seconds=self.match_time_seconds,
            time_remaining_seconds=max(
                0.0, self.config.gameplay.match_duration_seconds - self.match_time_seconds
            ),
            team=self.config.team,
            arena=self.config.arena,
            gameplay=self.config.gameplay,
            own_home_zone=self.config.own_home_zone,
            enemy_home_zone=self.config.enemy_home_zone,
            neutral_zone=self.config.arena.neutral_zone(),
            score=self.score,
            own_drones=self.own_drones,
            enemy_drones=self.enemy_drones,
            targets=self.targets,
            recent_events=list(self.events),
        )

    def _should_plan_unlocked(self) -> bool:
        interval_due = (
            self.match_time_seconds - self._last_strategy_time
            >= self.config.llm.decision_interval_seconds
        )
        major_due = (
            self.config.simulation.llm_on_major_events
            and self._last_major_event_time > self._last_strategy_time
        )
        return interval_due or major_due

    def _apply_commands_unlocked(
        self,
        drones: list[DroneState],
        commands: list[DroneCommand],
    ) -> None:
        by_id = {command.drone_id: command for command in commands}
        for drone in drones:
            command = by_id.get(drone.id)
            if not command:
                continue
            drone.role = command.role
            drone.objective = command.objective
            drone.assigned_target_id = command.target_id
            drone.command_position = command.target_position
            drone.hold_until = self.match_time_seconds + command.hold_seconds
            if command.objective == "land":
                drone.command_position = Pose(drone.pose.x, drone.pose.y, 0.0, drone.pose.yaw)

    def _move_drones_unlocked(self, drones: list[DroneState], speed: float, dt: float) -> None:
        max_step = speed * dt
        for drone in drones:
            if drone.status != "active" or drone.command_position is None:
                continue
            target = drone.command_position
            distance = drone.pose.distance_3d(target)
            if distance <= max_step or distance <= 0.001:
                drone.pose = Pose(target.x, target.y, target.z, target.yaw)
                if drone.objective == "land" and target.z <= 0.05:
                    drone.status = "landed"
                continue
            ratio = max_step / distance
            drone.pose = Pose(
                drone.pose.x + (target.x - drone.pose.x) * ratio,
                drone.pose.y + (target.y - drone.pose.y) * ratio,
                drone.pose.z + (target.z - drone.pose.z) * ratio,
                target.yaw,
            )

    def _update_captures_unlocked(self, dt: float) -> None:
        for target in self.targets:
            own_near = self._team_near_target_unlocked(self.own_drones, target)
            enemy_near = self._team_near_target_unlocked(self.enemy_drones, target)
            if own_near and not enemy_near:
                self._advance_capture_unlocked(target, self.config.team.color, dt)
            elif enemy_near and not own_near:
                self._advance_capture_unlocked(target, self.config.team.opponent_color, dt)
            else:
                target.capture_progress.clear()

            if target.owner != target.validated_for_owner:
                stable_for = self.match_time_seconds - target.stable_owner_since_seconds
                if stable_for >= self.config.gameplay.validation_seconds:
                    self._score_validation_unlocked(target)

    def _team_near_target_unlocked(self, drones: list[DroneState], target: TargetState) -> bool:
        for drone in drones:
            if drone.status == "active" and drone.pose.distance_2d(target.position) <= self.config.gameplay.capture_radius_m:
                if abs(drone.pose.z - self.config.gameplay.capture_z) <= 0.8:
                    return True
        return False

    def _advance_capture_unlocked(self, target: TargetState, color: TeamColor, dt: float) -> None:
        target.capture_progress[color] = target.capture_progress.get(color, 0.0) + dt
        other_color = self.config.team.opponent_color if color == self.config.team.color else self.config.team.color
        target.capture_progress.pop(other_color, None)
        if target.owner != color and target.capture_progress[color] >= self.config.gameplay.capture_hold_seconds:
            target.owner = color
            target.last_changed_seconds = self.match_time_seconds
            target.stable_owner_since_seconds = self.match_time_seconds
            target.validated_for_owner = None
            target.capture_progress.clear()
            self._last_major_event_time = self.match_time_seconds
            self._event_unlocked("target_changed", f"{target.id} changed to {color}.", {"target_id": target.id, "owner": color})

    def _score_validation_unlocked(self, target: TargetState) -> None:
        target.validated_for_owner = target.owner
        if target.owner == self.config.team.color:
            self.score.own += 1
            side = "own"
        else:
            self.score.opponent += 1
            side = "opponent"
        self._last_major_event_time = self.match_time_seconds
        self._event_unlocked(
            "score",
            f"Validated capture for {side} on {target.id}.",
            {"target_id": target.id, "owner": target.owner},
        )
        self._score_simultaneous_unlocked(target.owner)

    def _score_simultaneous_unlocked(self, owner: TeamColor) -> None:
        window = self.config.gameplay.simultaneous_window_seconds
        recent_scores = [
            event
            for event in self.events
            if event.type == "score"
            and event.data.get("owner") == owner
            and self.match_time_seconds - event.time_seconds <= window
        ]
        target_ids = {event.data.get("target_id") for event in recent_scores}
        if len(target_ids) >= 2:
            if owner == self.config.team.color:
                self.score.own += 9
                side = "own"
            else:
                self.score.opponent += 9
                side = "opponent"
            self._event_unlocked("simultaneous_score", f"Simultaneous capture bonus for {side}.")

    def _event_unlocked(self, event_type: str, message: str, data: dict | None = None) -> None:
        self.events.append(MatchEvent(self.match_time_seconds, event_type, message, data or {}))
