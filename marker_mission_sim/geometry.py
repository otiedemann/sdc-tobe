"""Tiny vector helpers for the simulator.

Arena coordinates (marker_mission convention): centred origin,
+x = right, +y = toward the front (blue) wall, +z = up. Headings are
degrees clockwise from +Y (the front wall), matching marker_mission
(``heading = atan2(dx, dy)``), so a unit heading vector is
``(sin θ, cos θ)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def scaled(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def dist(self, o: "Vec3") -> float:
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)

    def dist_xy(self, o: "Vec3") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def to_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.z, 4)]

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)


def heading_vec(heading_deg: float) -> tuple[float, float]:
    """Unit (dx, dy) for a heading measured CW from +Y (front wall)."""
    r = math.radians(heading_deg)
    return (math.sin(r), math.cos(r))


def heading_of(dx: float, dy: float) -> float:
    """Heading (deg CW from +Y) of a direction vector."""
    return math.degrees(math.atan2(dx, dy))


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def bearing_to(frm: Vec3, to: Vec3) -> float:
    """Heading (deg CW from +Y) from one point toward another (xy plane)."""
    return heading_of(to.x - frm.x, to.y - frm.y)
