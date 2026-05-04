from __future__ import annotations

import itertools

from .models import DroneCommand, DroneState, MatchState, Pose, TargetState


class MockEnemyPlanner:
    def __init__(self, style: str = "balanced") -> None:
        self.style = style
        self._counter = itertools.count()

    def plan(self, state: MatchState) -> list[DroneCommand]:
        active = [drone for drone in state.enemy_drones if drone.status == "active"]
        own_targets = [target for target in state.targets if target.relation_to_team(state.team) == "own"]
        enemy_targets = [target for target in state.targets if target.relation_to_team(state.team) == "enemy"]
        if not active:
            return []

        step = next(self._counter)
        commands: list[DroneCommand] = []
        for index, drone in enumerate(active):
            if self.style == "defensive":
                attack = index == 0
            elif self.style == "aggressive":
                attack = index < max(1, len(active) - 1)
            else:
                attack = (index + step // 10) % 3 != 0

            if attack:
                target = own_targets[index % len(own_targets)]
                commands.append(_enemy_capture_command(drone, target, state))
            else:
                target = enemy_targets[index % len(enemy_targets)]
                commands.append(_enemy_defense_command(drone, target, state, index))
        return commands


def _enemy_capture_command(drone: DroneState, target: TargetState, state: MatchState) -> DroneCommand:
    return DroneCommand(
        drone_id=drone.id,
        role="enemy_attacker",
        objective="capture_target",
        target_id=target.id,
        target_position=Pose(target.position.x, target.position.y, state.gameplay.capture_z, 0.0),
        hold_seconds=state.gameplay.capture_hold_seconds,
        priority=80,
        reason="Mock enemy attacking own target.",
    )


def _enemy_defense_command(
    drone: DroneState,
    target: TargetState,
    state: MatchState,
    index: int,
) -> DroneCommand:
    x_offset = -0.8 if target.side == "right" else 0.8
    y_offset = 1.0 if index % 2 == 0 else -1.0
    return DroneCommand(
        drone_id=drone.id,
        role="enemy_defender",
        objective="defense_patrol",
        target_id=target.id,
        target_position=state.arena.clamp_pose(
            Pose(target.position.x + x_offset, target.position.y + y_offset, state.gameplay.transit_z, 0.0)
        ),
        hold_seconds=0.0,
        priority=50,
        reason="Mock enemy defending own target.",
    )
