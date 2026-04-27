from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import math


def wrap_angle_pi(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def spielfeld_to_ned(pos_spiel: Tuple[float, float, float], yaw_spiel: float,
                     vx_spiel: float = 0.0, vy_spiel: float = 0.0, vz_spiel: float = 0.0, yaw_rate_spiel: float = 0.0) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Transformiert Position, Yaw und Geschwindigkeiten vom Spielfeld-System (X East, Y Up, Z North) zu NED (X North, Y East, Z Down).
    
    Rückgabe: (x_ned, y_ned, z_ned, yaw_ned, vx_ned, vy_ned, vz_ned, yaw_rate_ned)
    """
    x_spiel, y_spiel, z_spiel = pos_spiel
    
    # Position
    x_ned = z_spiel      # Z_spiel (North) -> X_ned (North)
    y_ned = x_spiel      # X_spiel (East) -> Y_ned (East)
    z_ned = -y_spiel     # Y_spiel (Up) -> -Z_ned (Down)
    
    # Yaw bleibt gleich, da die vertikale Achse (Up/Down) und die Drehrichtung um sie herum konsistent sind
    yaw_ned = yaw_spiel
    
    # Geschwindigkeiten
    vx_ned = vz_spiel      # vz_spiel -> vx_ned
    vy_ned = vx_spiel      # vx_spiel -> vy_ned
    vz_ned = -vy_spiel     # vy_spiel -> -vz_ned
    
    yaw_rate_ned = yaw_rate_spiel  # Gleich
    
    return x_ned, y_ned, z_ned, yaw_ned, vx_ned, vy_ned, vz_ned, yaw_rate_ned


def ned_to_spielfeld(pos_ned: Tuple[float, float, float], yaw_ned: float,
                     vx_ned: float = 0.0, vy_ned: float = 0.0, vz_ned: float = 0.0, yaw_rate_ned: float = 0.0) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Transformiert Position, Yaw und Geschwindigkeiten von NED (X North, Y East, Z Down) zum Spielfeld-System (X East, Y Up, Z North).
    
    Rückgabe: (x_spiel, y_spiel, z_spiel, yaw_spiel, vx_spiel, vy_spiel, vz_spiel, yaw_rate_spiel)
    """
    x_ned, y_ned, z_ned = pos_ned
    
    # Position
    x_spiel = y_ned       # Y_ned (East) -> X_spiel (East)
    y_spiel = -z_ned      # -Z_ned (Down) -> Y_spiel (Up)
    z_spiel = x_ned       # X_ned (North) -> Z_spiel (North)
    
    # Yaw
    yaw_spiel = yaw_ned
    
    # Geschwindigkeiten
    vx_spiel = vy_ned       # vy_ned -> vx_spiel
    vy_spiel = -vz_ned      # -vz_ned -> vy_spiel
    vz_spiel = vx_ned       # vx_ned -> vz_spiel
    
    yaw_rate_spiel = yaw_rate_ned
    
    return x_spiel, y_spiel, z_spiel, yaw_spiel, vx_spiel, vy_spiel, vz_spiel, yaw_rate_spiel


def rotate_xy(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def kalman_update_scalar(prior_mean: float, prior_var: float, meas_mean: float, meas_var: float) -> Tuple[float, float, float]:
    """
    Returns:
        posterior_mean, posterior_var, innovation
    """
    if meas_var <= 1e-9: #c apping close to zero division
        meas_var = 1e-9
    if prior_var <= 1e-9: # capping close to zero division
        prior_var = 1e-9

    innovation = meas_mean - prior_mean
    k = prior_var / (prior_var + meas_var)
    post_mean = prior_mean + k * innovation
    post_var = (1.0 - k) * prior_var
    return post_mean, post_var, innovation


def find_closest_state_index(motion_history: List[Any], timestamp: float) -> Optional[int]:
    if not motion_history:
        return None

    best_idx = 0
    best_dt = abs(motion_history[0].timestamp - timestamp)

    for i in range(1, len(motion_history)):
        dt = abs(motion_history[i].timestamp - timestamp)
        if dt < best_dt:
            best_dt = dt
            best_idx = i

    return best_idx


def extract_vision_measurement(vision_update: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Erwartete Formate:
    A)
        {
            "timestamp": ...,
            "cam": [x, y, z],
            "yaw": ... optional,
            "var_x": ...,
            "var_y": ...,
            "var_z": ...,
            "var_yaw": ... optional
        }

    B)
        {
            "timestamp": ...,
            "x": ..., "y": ..., "z": ...,
            ...
        }
    """
    if vision_update is None:
        return None

    ts = vision_update.get("timestamp")
    if ts is None:
        return None

    if vision_update.get("cam") is not None:
        cam = vision_update["cam"]
        x, y, z = float(cam[0]), float(cam[1]), float(cam[2])
    elif all(k in vision_update for k in ("x", "y", "z")):
        x = float(vision_update["x"])
        y = float(vision_update["y"])
        z = float(vision_update["z"])
    else:
        return None

    result = {
        "timestamp": float(ts),
        "x": x,
        "y": y,
        "z": z,
        "var_x": float(vision_update.get("var_x", 0.05)),
        "var_y": float(vision_update.get("var_y", 0.05)),
        "var_z": float(vision_update.get("var_z", 0.08)),
    }

    if "yaw" in vision_update and vision_update["yaw"] is not None:
        result["yaw"] = float(vision_update["yaw"])
        result["var_yaw"] = float(vision_update.get("var_yaw", 0.05))

    return result


def repropagate_history_from_index(updated_history: List[Any], start_idx: int) -> List[Any]:
    """
    Baut alle Zustände nach start_idx aus den gespeicherten Deltas und Prozessvarianzen neu auf.

    Erwartet idealerweise in jedem MotionState:
      delta_x, delta_y, delta_z, delta_yaw
      proc_var_x, proc_var_y, proc_var_z, proc_var_yaw
    """
    if start_idx >= len(updated_history) - 1:
        return updated_history

    new_history = list(updated_history)

    for i in range(start_idx + 1, len(new_history)):
        prev_state = new_history[i - 1]
        old_state = new_history[i]

        delta_x = getattr(old_state, "delta_x", old_state.x - updated_history[i - 1].x)
        delta_y = getattr(old_state, "delta_y", old_state.y - updated_history[i - 1].y)
        delta_z = getattr(old_state, "delta_z", old_state.z - updated_history[i - 1].z)
        delta_yaw = getattr(old_state, "delta_yaw", wrap_angle_pi(old_state.yaw - updated_history[i - 1].yaw))

        proc_var_x = getattr(old_state, "proc_var_x", max(0.0, old_state.var_x - updated_history[i - 1].var_x))
        proc_var_y = getattr(old_state, "proc_var_y", max(0.0, old_state.var_y - updated_history[i - 1].var_y))
        proc_var_z = getattr(old_state, "proc_var_z", max(0.0, old_state.var_z - updated_history[i - 1].var_z))
        proc_var_yaw = getattr(old_state, "proc_var_yaw", max(0.0, old_state.var_yaw - updated_history[i - 1].var_yaw))

        # delta_x/delta_y werden als Weltkoordinaten-Inkremente interpretiert
        new_x = prev_state.x + delta_x
        new_y = prev_state.y + delta_y
        new_z = prev_state.z + delta_z
        new_yaw = wrap_angle_pi(prev_state.yaw + delta_yaw)

        new_var_x = prev_state.var_x + proc_var_x
        new_var_y = prev_state.var_y + proc_var_y
        new_var_z = prev_state.var_z + proc_var_z
        new_var_yaw = prev_state.var_yaw + proc_var_yaw

        new_history[i] = replace(
            old_state,
            x=new_x,
            y=new_y,
            z=new_z,
            yaw=new_yaw,
            var_x=new_var_x,
            var_y=new_var_y,
            var_z=new_var_z,
            var_yaw=new_var_yaw,
        )

    return new_history


def transform_newer_points_rigid(updated_history: List[Any], ref_idx: int, dx: float, dy: float, dz: float, dyaw: float) -> List[Any]:
    """
    Fallback/Alternative:
    Wendet eine starre 2D-Transformation + Z-Translation auf alle neueren Punkte an.
    Das berücksichtigt den Hebelarm bei yaw-Korrektur.
    """
    new_history = list(updated_history)
    ref_old = updated_history[ref_idx]
    ref_new = new_history[ref_idx]

    for i in range(ref_idx + 1, len(new_history)):
        s = new_history[i]

        rel_x = s.x - ref_old.x
        rel_y = s.y - ref_old.y
        rot_x, rot_y = rotate_xy(rel_x, rel_y, dyaw)

        new_x = ref_new.x + rot_x
        new_y = ref_new.y + rot_y
        new_z = s.z + dz
        new_yaw = wrap_angle_pi(s.yaw + dyaw)

        new_history[i] = replace(
            s,
            x=new_x,
            y=new_y,
            z=new_z,
            yaw=new_yaw,
            var_x=min(s.var_x, max(ref_new.var_x, 1e-9) + max(0.0, s.var_x - ref_old.var_x)),
            var_y=min(s.var_y, max(ref_new.var_y, 1e-9) + max(0.0, s.var_y - ref_old.var_y)),
            var_z=min(s.var_z, max(ref_new.var_z, 1e-9) + max(0.0, s.var_z - ref_old.var_z)),
            var_yaw=min(s.var_yaw, max(ref_new.var_yaw, 1e-9) + max(0.0, s.var_yaw - ref_old.var_yaw)),
        )

    return new_history


def fuse_delayed_vision_update(
    motion_history: List[Any],
    vision_update: Dict[str, Any],
    use_yaw_if_available: bool = True,
    prefer_repropagation: bool = True,
) -> Dict[str, Any]:
    """
    Führt ein verzögertes Vision-Update in die Motion-Historie ein.

    Rückgabe:
    {
        "updated_history": [...],
        "reference_index": int,
        "correction": {"dx":..., "dy":..., "dz":..., "dyaw":...},
        "innovation": {...},
        "used_vision": {...}
    }
    """
    if not motion_history:
        return {
            "updated_history": motion_history,
            "reference_index": None,
            "correction": None,
            "innovation": None,
            "used_vision": None,
        }

    meas = extract_vision_measurement(vision_update)
    if meas is None:
        return {
            "updated_history": motion_history,
            "reference_index": None,
            "correction": None,
            "innovation": None,
            "used_vision": None,
        }

    # Transformiere Vision-Messung von Spielfeld zu NED
    pos_spiel = (meas["x"], meas["y"], meas["z"])
    yaw_spiel = meas.get("yaw", 0.0)
    x_ned, y_ned, z_ned, yaw_ned, _, _, _, _ = spielfeld_to_ned(pos_spiel, yaw_spiel)
    meas_ned = meas.copy()
    meas_ned["x"] = x_ned
    meas_ned["y"] = y_ned
    meas_ned["z"] = z_ned
    meas_ned["yaw"] = yaw_ned

    ref_idx = find_closest_state_index(motion_history, meas_ned["timestamp"])
    if ref_idx is None:
        return {
            "updated_history": motion_history,
            "reference_index": None,
            "correction": None,
            "innovation": None,
            "used_vision": meas,
        }

    old_ref = motion_history[ref_idx]

    # Kalman Update für Position
    new_x, new_var_x, innov_x = kalman_update_scalar(old_ref.x, old_ref.var_x, meas_ned["x"], meas_ned["var_x"])
    new_y, new_var_y, innov_y = kalman_update_scalar(old_ref.y, old_ref.var_y, meas_ned["y"], meas_ned["var_y"])
    new_z, new_var_z, innov_z = kalman_update_scalar(old_ref.z, old_ref.var_z, meas_ned["z"], meas_ned["var_z"])

    if use_yaw_if_available and "yaw" in meas_ned:
        meas_yaw = wrap_angle_pi(meas_ned["yaw"])
        yaw_innov = wrap_angle_pi(meas_yaw - old_ref.yaw)

        # Update um Winkelumschlag korrekt zu behandeln
        tmp_yaw_meas = old_ref.yaw + yaw_innov
        new_yaw_raw, new_var_yaw, _ = kalman_update_scalar(
            old_ref.yaw,
            old_ref.var_yaw,
            tmp_yaw_meas,
            meas_ned.get("var_yaw", 0.05),
        )
        new_yaw = wrap_angle_pi(new_yaw_raw)
        innov_yaw = yaw_innov
    else:
        new_yaw = old_ref.yaw
        new_var_yaw = old_ref.var_yaw
        innov_yaw = 0.0

    new_ref = replace(
        old_ref,
        x=new_x,
        y=new_y,
        z=new_z,
        yaw=new_yaw,
        var_x=new_var_x,
        var_y=new_var_y,
        var_z=new_var_z,
        var_yaw=new_var_yaw,
        vx=old_ref.vx,
        vy=old_ref.vy,
        vz=old_ref.vz,
        yaw_rate=old_ref.yaw_rate,
    )

    updated_history = list(motion_history)
    updated_history[ref_idx] = new_ref

    dx = new_ref.x - old_ref.x
    dy = new_ref.y - old_ref.y
    dz = new_ref.z - old_ref.z
    dyaw = wrap_angle_pi(new_ref.yaw - old_ref.yaw)

    if prefer_repropagation:
        # Beste Variante, falls MotionState Deltas/Prozessvarianzen speichert
        updated_history = repropagate_history_from_index(updated_history, ref_idx)
    else:
        # Fallback: starre Korrektur inkl. yaw-Hebelarm
        updated_history = transform_newer_points_rigid(updated_history, ref_idx, dx, dy, dz, dyaw)

    # Transformiere die gesamte Historie von NED zurück zu Spielfeld
    transformed_history = []
    for state in updated_history:
        pos_ned = (state.x, state.y, state.z)
        yaw_ned = state.yaw
        vx_ned = state.vx
        vy_ned = state.vy
        vz_ned = state.vz
        yaw_rate_ned = state.yaw_rate
        x_spiel, y_spiel, z_spiel, yaw_spiel, vx_spiel, vy_spiel, vz_spiel, yaw_rate_spiel = ned_to_spielfeld(
            pos_ned, yaw_ned, vx_ned, vy_ned, vz_ned, yaw_rate_ned
        )
        # Setze yaw_rate auf 0 nach Fusion, da sie nach der Korrektur inkonsistent ist.
        # Die nächste Messung wird neue yaw_rate aus dem absoluten Yaw generieren.
        transformed_state = replace(
            state,
            x=x_spiel,
            y=y_spiel,
            z=z_spiel,
            yaw=yaw_spiel,
            vx=vx_spiel,
            vy=vy_spiel,
            vz=vz_spiel,
            yaw_rate=0.0,  # Reset nach Fusion
        )
        transformed_history.append(transformed_state)
    updated_history = transformed_history

    return {
        "updated_history": updated_history,
        "reference_index": ref_idx,
        "correction": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dyaw": dyaw,
        },
        "innovation": {
            "x": innov_x,
            "y": innov_y,
            "z": innov_z,
            "yaw": innov_yaw,
        },
        "used_vision": meas,
    }
