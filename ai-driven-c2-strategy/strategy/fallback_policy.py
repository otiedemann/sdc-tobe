from __future__ import annotations

from .models import DroneCommand, DroneState, MatchState, PlannerResult, Pose, TargetState, Vec3


def plan_fallback(state: MatchState) -> PlannerResult:
    active = state.active_own_drones()
    if len(active) < 2:
        return PlannerResult(
            mode="failsafe",
            strategy_summary="Fewer than two active drones; landing active drones for safety.",
            commands=[_land_command(drone) for drone in active],
            source="fallback",
        )

    own_targets = [target for target in state.targets if target.relation_to_team(state.team) == "own"]
    enemy_targets = [target for target in state.targets if target.relation_to_team(state.team) == "enemy"]
    compromised = [target for target in own_targets if target.owner == state.team.opponent_color]

    commands: list[DroneCommand] = []
    available = active.copy()
    if compromised:
        for target in compromised:
            if not available:
                break
            drone = _nearest_drone(available, target.position)
            available.remove(drone)
            commands.append(_recover_command(drone, target, state))

    if available:
        if _should_bias_defense(state):
            commands.extend(_assign_defense(available, own_targets, state))
        else:
            attack_count = min(3, max(1, len(available) - 1))
            if len(active) == 2:
                attack_count = 1
            attackers = available[:attack_count]
            defenders = available[attack_count:]
            commands.extend(_assign_attackers(attackers, enemy_targets, state))
            commands.extend(_assign_defense(defenders, own_targets, state))

    return PlannerResult(
        mode="fallback_tactical",
        strategy_summary="Recover compromised own targets, otherwise stage attackers and patrol defense.",
        commands=commands,
        source="fallback",
    )


def _should_bias_defense(state: MatchState) -> bool:
    late_game = state.time_remaining_seconds < 90
    return late_game and state.score.own >= state.score.opponent


def _assign_attackers(
    drones: list[DroneState],
    enemy_targets: list[TargetState],
    state: MatchState,
) -> list[DroneCommand]:
    commands: list[DroneCommand] = []
    for index, drone in enumerate(drones):
        target = enemy_targets[index % len(enemy_targets)]
        objective = "capture_target" if _near_enemy_zone(drone, state) else "stage_attack"
        position = _capture_pose(target, state) if objective == "capture_target" else _stage_pose(target, state)
        commands.append(
            DroneCommand(
                drone_id=drone.id,
                role="attacker",
                objective=objective,  # type: ignore[arg-type]
                target_id=target.id,
                target_position=position,
                hold_seconds=state.gameplay.capture_hold_seconds if objective == "capture_target" else 0.0,
                priority=80,
                reason="Stage or execute synchronized pressure on enemy targets.",
            )
        )
    return commands


def _assign_defense(
    drones: list[DroneState],
    own_targets: list[TargetState],
    state: MatchState,
) -> list[DroneCommand]:
    commands: list[DroneCommand] = []
    for index, drone in enumerate(drones):
        target = own_targets[index % len(own_targets)]
        commands.append(
            DroneCommand(
                drone_id=drone.id,
                role="defender",
                objective="defense_patrol",
                target_id=target.id,
                target_position=_patrol_pose(target, state, index),
                hold_seconds=0.0,
                priority=70,
                reason="Patrol own targets without static blocking.",
            )
        )
    return commands


def _recover_command(drone: DroneState, target: TargetState, state: MatchState) -> DroneCommand:
    return DroneCommand(
        drone_id=drone.id,
        role="defender",
        objective="recover_target",
        target_id=target.id,
        target_position=_capture_pose(target, state),
        hold_seconds=state.gameplay.capture_hold_seconds + 0.3,
        priority=100,
        reason="Own target is opponent-colored and must be recovered before validation.",
    )


def _land_command(drone: DroneState) -> DroneCommand:
    return DroneCommand(
        drone_id=drone.id,
        role="failsafe",
        objective="land",
        target_position=Pose(drone.pose.x, drone.pose.y, 0.0, drone.pose.yaw),
        priority=100,
        reason="Safety fallback landing.",
    )


def _nearest_drone(drones: list[DroneState], point: Vec3) -> DroneState:
    return min(drones, key=lambda drone: drone.pose.distance_2d(point))


def _near_enemy_zone(drone: DroneState, state: MatchState) -> bool:
    return state.enemy_home_zone.contains(drone.pose) or state.neutral_zone.contains(drone.pose)


def _capture_pose(target: TargetState, state: MatchState) -> Pose:
    return state.arena.clamp_pose(Pose(target.position.x, target.position.y, state.gameplay.capture_z, 0.0))


def _stage_pose(target: TargetState, state: MatchState) -> Pose:
    direction = -1.0 if target.side == "right" else 1.0
    return state.arena.clamp_pose(
        Pose(target.position.x + direction * 2.0, target.position.y, state.gameplay.transit_z, 0.0)
    )


def _patrol_pose(target: TargetState, state: MatchState, index: int) -> Pose:
    side_offset = 1.0 if index % 2 == 0 else -1.0
    x_offset = 0.8 if target.side == "left" else -0.8
    return state.arena.clamp_pose(
        Pose(
            target.position.x + x_offset,
            target.position.y + side_offset,
            state.gameplay.transit_z,
            0.0,
        )
    )
