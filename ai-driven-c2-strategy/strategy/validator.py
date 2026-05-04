from __future__ import annotations

from .models import DroneCommand, MatchState, Pose, ValidationResult


VALID_OBJECTIVES = {
    "stage_attack",
    "capture_target",
    "recover_target",
    "defense_patrol",
    "offset_loiter",
    "return_home",
    "hold_home",
    "regroup",
    "land",
}


def validate_commands(state: MatchState, commands: list[DroneCommand]) -> ValidationResult:
    active_ids = {drone.id for drone in state.active_own_drones()}
    target_ids = {target.id for target in state.targets}
    errors: list[str] = []
    warnings: list[str] = []
    repaired: list[DroneCommand] = []
    seen_ids: set[str] = set()

    for command in commands:
        if command.drone_id not in active_ids:
            errors.append(f"Command references inactive or unknown drone {command.drone_id}.")
            continue
        if command.drone_id in seen_ids:
            warnings.append(f"Duplicate command for {command.drone_id}; keeping first command.")
            continue
        if command.objective not in VALID_OBJECTIVES:
            errors.append(f"Invalid objective {command.objective} for {command.drone_id}.")
            continue
        if command.target_id is not None and command.target_id not in target_ids:
            errors.append(f"Unknown target_id {command.target_id} for {command.drone_id}.")
            continue

        repaired_command = _repair_command(state, command, warnings)
        repaired.append(repaired_command)
        seen_ids.add(command.drone_id)

    missing = sorted(active_ids - seen_ids)
    if missing:
        warnings.append(f"Missing commands for active drones: {', '.join(missing)}.")

    separation_errors = _check_command_separation(state, repaired)
    errors.extend(separation_errors)
    return ValidationResult(ok=not errors and not missing, commands=repaired, errors=errors, warnings=warnings)


def _repair_command(
    state: MatchState,
    command: DroneCommand,
    warnings: list[str],
) -> DroneCommand:
    original = command.target_position
    position = state.arena.clamp_pose(original)
    if (position.x, position.y, position.z) != (original.x, original.y, original.z):
        warnings.append(f"Clamped out-of-bounds command for {command.drone_id}.")

    if command.objective in {"capture_target", "recover_target"}:
        z = min(max(position.z, 1.2), 1.8)
        position = Pose(position.x, position.y, z, position.yaw)
    elif command.objective != "land":
        z = min(max(position.z, 1.2), 2.8)
        position = Pose(position.x, position.y, z, position.yaw)

    return DroneCommand(
        drone_id=command.drone_id,
        role=command.role,
        objective=command.objective,
        target_id=command.target_id,
        target_position=position,
        hold_seconds=max(0.0, min(command.hold_seconds, 8.0)),
        priority=command.priority,
        reason=command.reason,
    )


def _check_command_separation(state: MatchState, commands: list[DroneCommand]) -> list[str]:
    errors: list[str] = []
    min_sep = state.gameplay.min_drone_separation_m
    for index, first in enumerate(commands):
        for second in commands[index + 1 :]:
            if first.target_position.distance_3d(second.target_position) < min_sep:
                errors.append(
                    f"Commands for {first.drone_id} and {second.drone_id} are closer than {min_sep} m."
                )
    return errors
