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
    x: float
    y: float
    z: float
    yaw: float
    var_x: float
    var_y: float
    var_z: float
    var_yaw: float


class MotionEstimator:
    """
    Reine Bewegungsfortschreibung ohne Filterung.

    Annahmen:
    - Eingangsdaten liegen im Body-Frame vor.
    - Weltkoordinatensystem:
        +X nach vorne beim Start
        +Y nach links
        +Z nach oben
    - Body-System der Drohne:
        +X nach vorne
        +Z nach unten
    - Startausrichtung der Drohne entspricht yaw = 0 im Welt-CS.
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

        now = time.time()
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
        """
        Bringt SDK-/Drohnenkonventionen in die gewünschte Modellkonvention:
        - +X forward
        - +Y left
        - +Z up
        - +yaw gegen Uhrzeigersinn
        """

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
        """
        Rotation aus Body nach Welt unter Nutzung von yaw.
        """
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
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
        """
        Fortschreiben aus Body-Frame Geschwindigkeiten.
        """
        last = self._buffer[-1]
        dt = timestamp - last.timestamp

        if dt <= 0.0:
            return last

        vx_body_n, vy_body_n, vz_body_n, yaw_rate_n = self._normalize_body_axes(
            vx_body, vy_body, vz_body, yaw_rate
        )

        new_yaw = wrap_angle_pi(last.yaw + yaw_rate_n * dt)

        vx_world, vy_world, vz_world = self._body_to_world_velocity(
            vx_body=vx_body_n,
            vy_body=vy_body_n,
            vz_body=vz_body_n,
            yaw=last.yaw,
        )

        new_x = last.x + vx_world * dt
        new_y = last.y + vy_world * dt
        new_z = last.z + vz_world * dt

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
        )

        self._buffer.append(state)
        self._drop_old_entries(timestamp)
        return state

    def get_current_state(self) -> MotionState:
        return self._buffer[-1]

    def get_history(self) -> List[MotionState]:
        return list(self._buffer)
