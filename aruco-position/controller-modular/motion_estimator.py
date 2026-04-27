from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple
import math
import time


def wrap_angle_pi(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


@dataclass
class MotionState:
    timestamp: float

    # absolute pose in world frame (NED)
    x: float  # North
    y: float  # East
    z: float  # Down
    yaw: float

    # velocities
    vx: float
    vy: float
    vz: float
    yaw_rate: float

    # accumulated variances
    var_x: float
    var_y: float
    var_z: float
    var_yaw: float

    # last-step increments used to reach this state from previous state
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    delta_yaw: float = 0.0

    # process variance added in this last step
    proc_var_x: float = 0.0
    proc_var_y: float = 0.0
    proc_var_z: float = 0.0
    proc_var_yaw: float = 0.0


class MotionEstimator:
    """
    Reine Bewegungsfortschreibung mit Trägheitsmodellierung.

    Weltkoordinatensystem: NED
        +X = North
        +Y = East
        +Z = Down

    Eingangsgrößen (NED, absolute Werte):
        vx_world, vy_world, vz_world in m/s
        yaw absoluter Yaw-Winkel in rad (wird intern zu yaw_rate differentiert)
        timestamp in Sekunden

    Wichtige Features:
        - Trägheit: Geschwindigkeitsänderungen begrenzt durch max_acc_*
        - Interne yaw_rate Berechnung aus absolutem Yaw
        - Varianzaktualisierung basierend auf Bewegungsgeschwindigkeit
    """

    def __init__(
        self,
        history_seconds: float = 2.0,
        nominal_dt: float = 0.1,
        initial_pose: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        initial_velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        initial_variance: Tuple[float, float, float, float] = (0.01, 0.01, 0.01, 0.01),
        base_process_var_pos: Tuple[float, float, float] = (0.0005, 0.0005, 0.0005),
        base_process_var_yaw: float = 0.0005,
        dyn_gain_pos: float = 0.01,
        dyn_gain_yaw: float = 0.01,
        yaw_positive_is_ccw: bool = True,
        max_acc_x: float = 1.0,
        max_acc_y: float = 1.0,
        max_acc_z: float = 1.0,
        max_acc_yaw_rate: float = 1.0,
        initial_timestamp: float = 0.0,
    ):
        self.history_seconds = history_seconds
        self.nominal_dt = nominal_dt

        maxlen = max(4, int(math.ceil(history_seconds / nominal_dt)) + 2)
        self._buffer: Deque[MotionState] = deque(maxlen=maxlen)

        self.base_process_var_pos = base_process_var_pos
        self.base_process_var_yaw = base_process_var_yaw
        self.dyn_gain_pos = dyn_gain_pos
        self.dyn_gain_yaw = dyn_gain_yaw
        self.yaw_positive_is_ccw = yaw_positive_is_ccw
        self.max_acc_x = max_acc_x
        self.max_acc_y = max_acc_y
        self.max_acc_z = max_acc_z
        self.max_acc_yaw_rate = max_acc_yaw_rate

        x, y, z, yaw = initial_pose
        vx, vy, vz, yr = initial_velocity
        var_x, var_y, var_z, var_yaw = initial_variance

        now = initial_timestamp
        self._buffer.append(
            MotionState(
                timestamp=now,
                x=x,
                y=y,
                z=z,
                yaw=wrap_angle_pi(yaw),
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=yr,
                var_x=var_x,
                var_y=var_y,
                var_z=var_z,
                var_yaw=var_yaw,
            )
        )

    def _drop_old_entries(self, current_time: float) -> None:
        cutoff = current_time - self.history_seconds
        while len(self._buffer) > 1 and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def _compute_process_variance(
        self,
        vx_world: float,
        vy_world: float,
        vz_world: float,
        yaw_rate: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        speed = math.sqrt(vx_world * vx_world + vy_world * vy_world) #no need to include vz in variance calculation

        proc_var_x = self.base_process_var_pos[0] * dt + self.dyn_gain_pos * abs(vx_world) * dt
        proc_var_y = self.base_process_var_pos[1] * dt + self.dyn_gain_pos * abs(vy_world) * dt
        proc_var_z = self.base_process_var_pos[2] * dt + self.dyn_gain_pos * abs(vz_world) * dt
        proc_var_yaw = self.base_process_var_yaw * dt + self.dyn_gain_yaw * abs(yaw_rate) * dt

        coupling = 0.002 * speed * dt
        proc_var_x += coupling
        proc_var_y += coupling
        proc_var_z += coupling

        return proc_var_x, proc_var_y, proc_var_z, proc_var_yaw

    def update_world_frame(
        self,
        timestamp: float,
        vx_world: float,
        vy_world: float,
        vz_world: float,
        yaw: float = 0.0,
    ) -> MotionState:
        last = self._buffer[-1]
        dt = timestamp - last.timestamp

        if dt <= 0.0:
            return last

        # Desired velocities (direct NED coordinates)
        vx_desired = vx_world
        vy_desired = vy_world
        vz_desired = vz_world
        
        # Berechne desired yaw_rate aus absolutem yaw
        yaw_desired = wrap_angle_pi(yaw)
        yaw_diff = wrap_angle_pi(yaw_desired - last.yaw)
        yr_desired = yaw_diff / dt if dt > 0 else 0.0

        # Current velocities
        vx_current = last.vx
        vy_current = last.vy
        vz_current = last.vz
        yr_current = last.yaw_rate

        # Compute velocity changes, limited by acceleration
        delta_vx = vx_desired - vx_current
        delta_vy = vy_desired - vy_current
        delta_vz = vz_desired - vz_current
        delta_yr = yr_desired - yr_current

        max_delta_vx = self.max_acc_x * dt
        max_delta_vy = self.max_acc_y * dt
        max_delta_vz = self.max_acc_z * dt
        max_delta_yr = self.max_acc_yaw_rate * dt

        delta_vx = max(-max_delta_vx, min(max_delta_vx, delta_vx))
        delta_vy = max(-max_delta_vy, min(max_delta_vy, delta_vy))
        delta_vz = max(-max_delta_vz, min(max_delta_vz, delta_vz))
        delta_yr = max(-max_delta_yr, min(max_delta_yr, delta_yr))

        # Updated velocities
        new_vx = vx_current + delta_vx
        new_vy = vy_current + delta_vy
        new_vz = vz_current + delta_vz
        new_yr = yr_current + delta_yr

        # Position updates using average velocity for better accuracy
        delta_x = dt * (new_vx + vx_current) / 2
        delta_y = dt * (new_vy + vy_current) / 2
        delta_z = dt * (new_vz + vz_current) / 2
        delta_yaw = dt * (new_yr + yr_current) / 2

        new_x = last.x + delta_x
        new_y = last.y + delta_y
        new_z = last.z + delta_z
        new_yaw = wrap_angle_pi(last.yaw + delta_yaw)

        proc_var_x, proc_var_y, proc_var_z, proc_var_yaw = self._compute_process_variance(
            vx_world=vx_desired,
            vy_world=vy_desired,
            vz_world=vz_desired,
            yaw_rate=yr_desired,
            dt=dt,
        )

        state = MotionState(
            timestamp=timestamp,
            x=new_x,
            y=new_y,
            z=new_z,
            yaw=new_yaw,
            vx=new_vx,
            vy=new_vy,
            vz=new_vz,
            yaw_rate=new_yr,
            var_x=last.var_x + proc_var_x,
            var_y=last.var_y + proc_var_y,
            var_z=last.var_z + proc_var_z,
            var_yaw=last.var_yaw + proc_var_yaw,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z,
            delta_yaw=delta_yaw,
            proc_var_x=proc_var_x,
            proc_var_y=proc_var_y,
            proc_var_z=proc_var_z,
            proc_var_yaw=proc_var_yaw,
        )

        self._buffer.append(state)
        self._drop_old_entries(timestamp)
        return state

    def get_current_state(self) -> MotionState:
        return self._buffer[-1]

    def get_history(self) -> List[MotionState]:
        return list(self._buffer)

    def apply_fused_history(self, new_history: List[MotionState]) -> MotionState:
        """
        Ersetzt den aktuellen Buffer durch eine bereits gefusete Historie.
        Gibt den neuesten Zustand zurück.
        """
        if not new_history:
            return self.get_current_state()

        self._buffer.clear()
        for state in new_history:
            self._buffer.append(state)

        return self._buffer[-1]

    def reset(
        self,
        timestamp: float,
        x: float,
        y: float,
        z: float,
        yaw: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
        var_x: float = 0.01,
        var_y: float = 0.01,
        var_z: float = 0.01,
        var_yaw: float = 0.01,
    ) -> None:
        self._buffer.clear()
        self._buffer.append(
            MotionState(
                timestamp=timestamp,
                x=x,
                y=y,
                z=z,
                yaw=wrap_angle_pi(yaw),
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=yaw_rate,
                var_x=var_x,
                var_y=var_y,
                var_z=var_z,
                var_yaw=var_yaw,
            )
        )
