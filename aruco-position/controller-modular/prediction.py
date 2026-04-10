from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_dt(t1: float, t0: float) -> float:
    dt = t1 - t0
    return dt if dt > 1e-9 else 1e-9


def _estimate_velocity_from_points(
    history: List[Dict[str, float]],
    max_points: int = 5,
) -> Tuple[float, float, float]:
    """
    Schätzt Geschwindigkeit aus den letzten max_points Zuständen.
    Nutzt einfache Endpunkt-Schätzung über das betrachtete Fenster.
    """
    if len(history) < 2:
        return 0.0, 0.0, 0.0

    pts = history[-max_points:]
    p0 = pts[0]
    p1 = pts[-1]

    dt = _safe_dt(p1["timestamp"], p0["timestamp"])
    vx = (p1["x"] - p0["x"]) / dt
    vy = (p1["y"] - p0["y"]) / dt
    vz = (p1["z"] - p0["z"]) / dt
    return vx, vy, vz


def _linear_fit_predict(
    history: List[Dict[str, float]],
    target_ts: float,
    axis: str,
) -> Tuple[float, float]:
    """
    Linearer Least-Squares-Fit:
        value = a * t_rel + b

    Rückgabe:
        predicted_value, residual_std

    residual_std ist eine grobe Schätzung der Modellgüte.
    """
    n = len(history)
    if n == 0:
        return 0.0, 1.0

    if n == 1:
        return float(history[0][axis]), 1.0

    t0 = history[0]["timestamp"]
    xs = [float(p["timestamp"] - t0) for p in history]
    ys = [float(p[axis]) for p in history]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ys[-1], 1.0

    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = mean_y - a * mean_x

    x_target = target_ts - t0
    y_pred = a * x_target + b

    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    if n >= 3:
        rss = sum(r * r for r in residuals)
        residual_std = math.sqrt(rss / max(1, n - 2))
    else:
        residual_std = abs(residuals[-1]) if residuals else 0.1

    return y_pred, max(1e-4, residual_std)


def _extract_history_points(
    fused_state: Optional[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """
    Erwartet idealerweise in fused_state:
        fused_state["history"] = [
            {"timestamp": ..., "x": ..., "y": ..., "z": ..., "yaw": ...},
            ...
        ]

    Falls keine Historie da ist, wird aus dem fused_state selbst ein Eintrag erzeugt.
    """
    if fused_state is None:
        return []

    raw_history = fused_state.get("history")
    if isinstance(raw_history, list) and len(raw_history) > 0:
        cleaned = []
        for p in raw_history:
            if not isinstance(p, dict):
                continue
            if not all(k in p for k in ("timestamp", "x", "y", "z")):
                continue
            cleaned.append(
                {
                    "timestamp": float(p["timestamp"]),
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]),
                    "yaw": float(p.get("yaw", 0.0)),
                }
            )
        if cleaned:
            return cleaned

    if all(k in fused_state for k in ("timestamp", "x", "y", "z")):
        return [{
            "timestamp": float(fused_state["timestamp"]),
            "x": float(fused_state["x"]),
            "y": float(fused_state["y"]),
            "z": float(fused_state["z"]),
            "yaw": float(fused_state.get("yaw", 0.0)),
        }]

    return []


def predict_to_now(
    fused_state: Optional[Dict[str, Any]],
    now_ts: float,
    prefer_constant_velocity: bool = True,
    fit_min_points: int = 5,
    fit_full_history: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Extrapoliert Position auf now_ts.

    Anforderungen:
    1. erhält aktuelle Motion/Fusion
    2. erhält aktuellen Zeitstempel
    3. extrapoliert aus min. letzten 5 Positionen oder Fit über Historie
    4. wenn letzte Geschwindigkeit bekannt ist, wird konstante Geschwindigkeit angenommen
    5. gibt Position + grobe Genauigkeit zurück

    Erwartete Eingaben in fused_state:
    - timestamp, x, y, z, yaw
    - optional vx, vy, vz
    - optional std_x, std_y, std_z oder var_x, var_y, var_z
    - optional history: Liste vergangener Zustände
    """
    if fused_state is None:
        return None

    history = _extract_history_points(fused_state)
    if not history:
        return None

    latest = history[-1]
    dt = max(0.0, now_ts - latest["timestamp"])

    latest_x = float(latest["x"])
    latest_y = float(latest["y"])
    latest_z = float(latest["z"])
    latest_yaw = float(latest.get("yaw", fused_state.get("yaw", 0.0)))

    vx_known = float(fused_state.get("vx", 0.0))
    vy_known = float(fused_state.get("vy", 0.0))
    vz_known = float(fused_state.get("vz", 0.0))
    speed_known = math.sqrt(vx_known * vx_known + vy_known * vy_known + vz_known * vz_known)

    # Basis-Unsicherheit aus fused_state
    if "std_x" in fused_state and "std_y" in fused_state and "std_z" in fused_state:
        base_std_x = float(fused_state["std_x"])
        base_std_y = float(fused_state["std_y"])
        base_std_z = float(fused_state["std_z"])
    else:
        base_std_x = math.sqrt(max(0.0, float(fused_state.get("var_x", 0.04))))
        base_std_y = math.sqrt(max(0.0, float(fused_state.get("var_y", 0.04))))
        base_std_z = math.sqrt(max(0.0, float(fused_state.get("var_z", 0.09))))

    # --------------------------------------------------
    # Modus A: bekannte Geschwindigkeit -> konstante Bewegung
    # --------------------------------------------------
    use_constant_velocity = prefer_constant_velocity and speed_known > 1e-4

    if use_constant_velocity:
        pred_x = latest_x + vx_known * dt
        pred_y = latest_y + vy_known * dt
        pred_z = latest_z + vz_known * dt

        # Unsicherheit wächst grob mit Zeit und Geschwindigkeit
        std_growth_xy = 0.15 * dt + 0.10 * speed_known * dt
        std_growth_z = 0.20 * dt + 0.08 * abs(vz_known) * dt

        std_x = base_std_x + std_growth_xy
        std_y = base_std_y + std_growth_xy
        std_z = base_std_z + std_growth_z

        method = "constant_velocity"

    # --------------------------------------------------
    # Modus B: Fit / Historie
    # --------------------------------------------------
    else:
        if len(history) >= fit_min_points:
            fit_points = history if fit_full_history else history[-fit_min_points:]

            pred_x, fit_std_x = _linear_fit_predict(fit_points, now_ts, "x")
            pred_y, fit_std_y = _linear_fit_predict(fit_points, now_ts, "y")
            pred_z, fit_std_z = _linear_fit_predict(fit_points, now_ts, "z")

            # Grobe Zunahme mit Vorhersagehorizont
            horizon_factor = 1.0 + 0.5 * dt
            std_x = base_std_x + fit_std_x * horizon_factor
            std_y = base_std_y + fit_std_y * horizon_factor
            std_z = base_std_z + fit_std_z * horizon_factor

            # Geschwindigkeit aus Historie als Zusatzinfo
            est_vx, est_vy, est_vz = _estimate_velocity_from_points(history, max_points=fit_min_points)
            vx_known, vy_known, vz_known = est_vx, est_vy, est_vz

            method = "linear_fit"
        else:
            # Zu wenig Historie -> letzte bekannte Pose halten
            pred_x = latest_x
            pred_y = latest_y
            pred_z = latest_z

            std_x = base_std_x + 0.25 * dt
            std_y = base_std_y + 0.25 * dt
            std_z = base_std_z + 0.30 * dt

            if len(history) >= 2:
                vx_known, vy_known, vz_known = _estimate_velocity_from_points(history, max_points=2)

            method = "hold_last"

    std_pos = math.sqrt(std_x * std_x + std_y * std_y + std_z * std_z)

    return {
        "timestamp": float(now_ts),
        "x": float(pred_x),
        "y": float(pred_y),
        "z": float(pred_z),
        "yaw": float(latest_yaw),
        "vx": float(vx_known),
        "vy": float(vy_known),
        "vz": float(vz_known),
        "std_x": float(max(1e-4, std_x)),
        "std_y": float(max(1e-4, std_y)),
        "std_z": float(max(1e-4, std_z)),
        "std_pos": float(max(1e-4, std_pos)),
        "source": fused_state.get("source", "fusion"),
        "method": method,
    }
