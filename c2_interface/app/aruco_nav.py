"""
Aruco navigation module boundary.

Replace internals with your simulator/real implementation:
- Read camera frames
- Detect Aruco markers on pillars
- Triangulate drone pose in arena grid
- Return corrected position for C2 and autopilot loops
"""

from .models import GridPoint


class ArucoNavigator:
    def localize(self, drone_id: str, fallback: GridPoint) -> GridPoint:
        # TODO: integrate with existing aruco-position/simulator pipeline.
        # For now, return the fallback (estimated) location.
        return fallback
