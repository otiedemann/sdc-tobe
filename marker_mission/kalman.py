"""
6-state constant-velocity Kalman filter for arena-frame drone position.

State: ``[px, py, pz, vx, vy, vz]`` in metres / metres per second.

Two heterogeneous measurement updates:

* ``update_position(z_pos, R_diag=None)`` -- ArUco aggregate position
  from ``arena.estimate_position`` (slow, intermittent, accurate).
* ``update_velocity(z_vel, R_diag=None)`` -- IMU velocity from the
  Anafi ``vgx/vgy/vgz`` telemetry, rotated into the arena frame by
  the caller (fast, every-tick, moderately noisy).

Process noise is the standard CV-model Q with a single ``accel_var``
(drone-agility) parameter, scaled by powers of dt.

Used both by the offline reprocessor (``tools.replay_kalman``) and,
when wired in, the live ``vision_worker`` -- the math is identical;
only the data source differs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PositionKalman:
    """3D constant-velocity KF with separate position + velocity updates."""

    def __init__(self,
                 accel_var: float = 0.1,
                 pos_meas_var: float = 0.0025,
                 vel_meas_var: float = 0.0025,
                 init_pos_var: float = 1.0,
                 init_vel_var: float = 1.0):
        # Process and measurement noise (variances, not std-devs).
        self.accel_var = float(accel_var)
        self.pos_meas_var = float(pos_meas_var)
        self.vel_meas_var = float(vel_meas_var)
        self.init_pos_var = float(init_pos_var)
        self.init_vel_var = float(init_vel_var)

        self._x: Optional[np.ndarray] = None         # 6,
        self._P: Optional[np.ndarray] = None         # 6x6
        self._I6 = np.eye(6)

    # ------------------------------------------------------------ life cycle
    def reset(self) -> None:
        self._x = None
        self._P = None

    @property
    def initialised(self) -> bool:
        return self._x is not None

    # -------------------------------------------------------------- accessors
    def position(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x[0:3].copy()

    def velocity(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x[3:6].copy()

    def state(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x.copy()

    # ------------------------------------------------------------- prediction
    def predict(self, dt: float) -> None:
        """Advance state by ``dt`` seconds. No-op if uninitialised."""
        if self._x is None or dt <= 0.0:
            return
        F = self._I6.copy()
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Standard CV-model Q. Treats acceleration as zero-mean
        # white noise with variance ``accel_var`` (per-axis, m^2/s^4).
        # Result: position grows by dt^4/4, position-velocity cross
        # by dt^3/2, velocity by dt^2.
        q = self.accel_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (dt4 / 4.0) * np.eye(3) * q
        Q[0:3, 3:6] = (dt3 / 2.0) * np.eye(3) * q
        Q[3:6, 0:3] = (dt3 / 2.0) * np.eye(3) * q
        Q[3:6, 3:6] = (dt2)        * np.eye(3) * q

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    # ------------------------------------------------------------ update -- position
    def update_position(self, z_pos: np.ndarray,
                        R_diag: Optional[float] = None) -> None:
        """Fuse a 3D position measurement (arena frame, metres).

        Initialises the filter on the first call: position = z_pos,
        velocity = 0, covariance set from ``init_pos_var`` /
        ``init_vel_var``.
        """
        z = np.asarray(z_pos, dtype=float).reshape(3)
        if not np.all(np.isfinite(z)):
            return
        if self._x is None:
            self._x = np.zeros(6)
            self._x[0:3] = z
            self._P = np.eye(6)
            self._P[0:3, 0:3] *= self.init_pos_var
            self._P[3:6, 3:6] *= self.init_vel_var
            return
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        r = self.pos_meas_var if R_diag is None else float(R_diag)
        R = np.eye(3) * r
        self._update(H, R, z)

    # ------------------------------------------------------------ update -- velocity
    def update_velocity(self, z_vel: np.ndarray,
                        R_diag: Optional[float] = None) -> None:
        """Fuse a 3D velocity measurement (arena frame, m/s).

        Skips silently when the filter hasn't been initialised by
        a position measurement yet -- a velocity-only initialisation
        leaves position completely undetermined and is worse than no
        estimate at all.
        """
        if self._x is None:
            return
        z = np.asarray(z_vel, dtype=float).reshape(3)
        if not np.all(np.isfinite(z)):
            return
        H = np.zeros((3, 6))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0
        r = self.vel_meas_var if R_diag is None else float(R_diag)
        R = np.eye(3) * r
        self._update(H, R, z)

    # ----------------------------------------------------------------- shared
    def _update(self, H: np.ndarray, R: np.ndarray, z: np.ndarray) -> None:
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (self._I6 - K @ H) @ self._P


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def ned_velocity_to_arena(vN: float, vE: float, vD: float,
                           magnetic_north_arena_yaw_deg: float
                           ) -> np.ndarray:
    """Rotate a NED velocity into the arena frame.

    ``magnetic_north_arena_yaw_deg`` is the value stored on the
    ``ArenaConfig`` after operator calibration: arena yaw at which
    magnetic north points (CW from arena +y / front wall). The arena
    frame uses ``+x right, +y to front wall, +z up``.

    Returns ``np.array([arena_vx, arena_vy, arena_vz])`` in m/s.
    """
    import math
    # Direction of magnetic north in arena frame, CW from arena +y.
    # offset = "where magnetic north points in arena". So the world-N
    # axis IS at angle ``offset`` CW from arena +y.
    a = math.radians(float(magnetic_north_arena_yaw_deg))
    sin_a, cos_a = math.sin(a), math.cos(a)
    # A unit-N vector lies along (sin_a, cos_a) in arena (x, y).
    # A unit-E vector is 90 deg CW from N in NED == 90 deg CW in
    # arena too -> (cos_a, -sin_a).
    arena_vx = vN * sin_a + vE * cos_a
    arena_vy = vN * cos_a - vE * sin_a
    arena_vz = -float(vD)   # NED-down -> arena-up
    return np.array([arena_vx, arena_vy, arena_vz], dtype=float)
