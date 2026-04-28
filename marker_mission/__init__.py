"""Marker-approach mission for the Parrot Anafi.

Modules:

* :mod:`marker_mission.config`             -- typed dataclass with all knobs
* :mod:`marker_mission.calibration_store`  -- per-serial intrinsic store
* :mod:`marker_mission.aruco_detector`     -- detection + 6-DOF pose
* :mod:`marker_mission.drone_api`          -- HTTP client + MJPEG reader
* :mod:`marker_mission.controller`         -- PD controllers + state machine
* :mod:`marker_mission.recorder`           -- video + flight log writer
* :mod:`marker_mission.ui`                 -- two-screen operator UI
* :mod:`marker_mission.mission`            -- CLI entry point
"""

__all__ = [
    "config", "calibration_store", "aruco_detector",
    "drone_api", "controller", "recorder", "ui", "mission",
]
