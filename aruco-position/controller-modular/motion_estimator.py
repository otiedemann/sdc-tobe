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

    # absolute pose in world frame
    x: float
    y: float
    z: float
    yaw: float

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
    Reine Bewegungsfortschreibung ohne Filterung.

    Weltkoordinatensystem:
        +X nach vorne beim Start
        +Y nach links
        +Z nach oben

    Body-System der Drohne:
        +X nach vorne
        +Z nach unten (falls body_z_positive_is_down=True)

    Eingangsgrößen:
        vx_body, vy_body, vz_body, yaw_rate
    """

    def __init__(
        self,
        history_seconds: float = 2.0,
        nominal_dt: float = 0.1,
        initial_pose: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        initial_variance: Tuple[float, float, float, float] = (0.01, 0.01, 0.01, 0.01),
        base_process_var_pos: Tuple[float, float, float] = (0.0005, 0.0005, 0.0005),
        base_process_var_yaw: float = 0.0005,
        dyn_gain_pos: float = 0.01,
        dyn_gain_yaw: float = 0.01,
        body_y_positive_is_left: bool = True,
        body_z_positive_is_down: bool = True,
        yaw_positive_is_ccw: bool = True,
    ):
        self.history_seconds = history_seconds
        self.nominal_dt = nominal_dt

        maxlen = max(4, int(math.ceil(history_seconds / nominal_dt)) + 2)
        self._buffer: Deque[MotionState] = deque(maxlen=maxlen)

        self.base_process_var_pos = base_process_var_pos
        self.base_process_var_yaw = base_process_var_yaw
        self.dyn_gain_pos = dyn_gain_pos
        self.dyn_gain_yaw = dyn_gain_yaw

        self.body_y_positive_is_left = body_y_positive_is_left
        self.body_z_positive_is_down = body_z_positive_is_down
        self.yaw_positive_is_ccw = yaw_positive_is_ccw

        x, y, z, yaw = initial_pose
        var_x, var_y, var_z, var_yaw = initial_variance

        now = time.monotonic()
        self._buffer.append(
            MotionState(
                timestamp=now,
                x=x,
                y=y,
                z=z,
                yaw=wrap_angle_pi(yaw),
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

    def _normalize_body_axes(
        self,
        vx_body: float,
        vy_body: float,
        vz_body: float,
        yaw_rate: float,
    ) -> Tuple[float, float, float, float]:
        vx = vx_body
        vy = vy_body if self.body_y_positive_is_left else -vy_body
        vz = -vz_body if self.body_z_positive_is_down else vz_body
        yr = yaw_rate if self.yaw_positive_is_ccw else -yaw_rate
        return vx, vy, vz, yr

    def _body_to_world_velocity(
        self,
        vx_body: float,
        vy_body: float,
        vz_body: float,
        yaw: float,
    ) -> Tuple[float, float, float]:
        c = math.cos(yaw)
        s = math.sin(yaw)

        vx_world = c * vx_body - s * vy_body
        vy_world = s * vx_body + c * vy_body
        vz_world = vz_body

        return vx_world, vy_world, vz_world

    def _compute_process_variance(
        self,
        vx_world: float,
        vy_world: float,
        vz_world: float,
        yaw_rate: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        speed = math.sqrt(vx_world * vx_world + vy_world * vy_world + vz_world * vz_world)

        proc_var_x = self.base_process_var_pos[0] * dt + self.dyn_gain_pos * abs(vx_world) * dt
        proc_var_y = self.base_process_var_pos[1] * dt + self.dyn_gain_pos * abs(vy_world) * dt
        proc_var_z = self.base_process_var_pos[2] * dt + self.dyn_gain_pos * abs(vz_world) * dt
        proc_var_yaw = self.base_process_var_yaw * dt + self.dyn_gain_yaw * abs(yaw_rate) * dt

        coupling = 0.002 * speed * dt
        proc_var_x += coupling
        proc_var_y += coupling
        proc_var_z += coupling

        return proc_var_x, proc_var_y, proc_var_z, proc_var_yaw

    def update_body_frame(
        self,
        timestamp: float,
        vx_body: float,
        vy_body: float,
        vz_body: float,
        yaw_rate: float = 0.0,
    ) -> MotionState:
        last = self._buffer[-1]
        dt = timestamp - last.timestamp

        if dt <= 0.0:
            return last

        vx_body_n, vy_body_n, vz_body_n, yaw_rate_n = self._normalize_body_axes(
            vx_body, vy_body, vz_body, yaw_rate
        )

        # midpoint yaw is a bit more accurate than using only last.yaw
        yaw_mid = wrap_angle_pi(last.yaw + 0.5 * yaw_rate_n * dt)

        vx_world, vy_world, vz_world = self._body_to_world_velocity(
            vx_body=vx_body_n,
            vy_body=vy_body_n,
            vz_body=vz_body_n,
            yaw=yaw_mid,
        )

        delta_x = vx_world * dt
        delta_y = vy_world * dt
        delta_z = vz_world * dt
        delta_yaw = yaw_rate_n * dt

        new_x = last.x + delta_x
        new_y = last.y + delta_y
        new_z = last.z + delta_z
        new_yaw = wrap_angle_pi(last.yaw + delta_yaw)

        proc_var_x, proc_var_y, proc_var_z, proc_var_yaw = self._compute_process_variance(
            vx_world=vx_world,
            vy_world=vy_world,
            vz_world=vz_world,
            yaw_rate=yaw_rate_n,
            dt=dt,
        )

        state = MotionState(
            timestamp=timestamp,
            x=new_x,
            y=new_y,
            z=new_z,
            yaw=new_yaw,
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
        var_x: float,
        var_y: float,
        var_z: float,
        var_yaw: float,
    ) -> None:
        self._buffer.clear()
        self._buffer.append(
            MotionState(
                timestamp=timestamp,
                x=x,
                y=y,
                z=z,
                yaw=wrap_angle_pi(yaw),
                var_x=var_x,
                var_y=var_y,
                var_z=var_z,
                var_yaw=var_yaw,
            )
        )
