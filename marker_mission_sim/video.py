"""Synthetic per-drone camera feed.

Renders what a drone "sees": a pinhole projection of the markers the
vision model reports visible (wall markers + target-box faces), over a
sky/ground background, with a telemetry HUD. Served as MJPEG by the FC
API's ``/video.mjpg`` so the C2 overview shows a live picture for every
simulated drone — just like a real flight controller.

Pure Pillow; no OpenCV needed.
"""

from __future__ import annotations

import io
import math
from typing import Optional

from PIL import Image, ImageDraw

from .geometry import heading_vec
from .world import World

# Frame size + physical marker sizes (m) used for apparent-size scaling.
W, H = 640, 360
_WALL_MARKER_M = 0.5
_TARGET_MARKER_M = 0.19

_SKY = (120, 160, 200)
_GROUND = (70, 110, 70)
_HUD_BG = (0, 0, 0)


def _marker_world_pos(world: World, mid: int):
    """Return (x, y, z, kind, size_m) for a visible marker id, or None."""
    # Target face? (current face of some box)
    for b in world.boxes.values():
        if b.current_face_id == mid:
            return (b.pos.x, b.pos.y, b.pos.z, "target", _TARGET_MARKER_M)
    # Wall marker?
    for m in world.wall_markers:
        if m.id == mid:
            return (m.pos.x, m.pos.y, m.pos.z, "wall", _WALL_MARKER_M)
    return None


def _marker_color(mid: int, kind: str) -> tuple[int, int, int]:
    if kind == "target":
        if 41 <= mid <= 46:
            return (210, 70, 60)    # red face
        if 31 <= mid <= 36:
            return (70, 120, 210)   # blue face
    return (235, 235, 235)          # wall marker (white)


def render_frame(world: World, drone_id: str) -> Image.Image:
    """Render one camera frame for ``drone_id`` as a PIL image."""
    with world.lock:
        d = world.drones.get(drone_id)
        if d is None:
            img = Image.new("RGB", (W, H), (20, 20, 20))
            ImageDraw.Draw(img).text((10, 10), f"no drone {drone_id}", fill=(255, 0, 0))
            return img
        # Snapshot the bits we need under the lock.
        dx, dy, dz = d.pos.x, d.pos.y, d.pos.z
        heading = d.heading_deg
        team = d.team
        phase = d.phase
        flying = d.flying
        try:
            visible = world.visible_marker_ids(drone_id)
        except Exception:
            visible = []
        markers = [(mid, _marker_world_pos(world, mid)) for mid in visible]

    fov = float(world.cfg.noise.vision_fov_deg)
    f = (W / 2) / max(0.2, math.tan(math.radians(fov) / 2))
    fwd_x, fwd_y = heading_vec(heading)         # (sin h, cos h)
    right_x, right_y = math.cos(math.radians(heading)), -math.sin(math.radians(heading))

    img = Image.new("RGB", (W, H), _SKY)
    draw = ImageDraw.Draw(img)
    # Ground (lower half) + simple horizon.
    draw.rectangle([0, H // 2, W, H], fill=_GROUND)
    draw.line([0, H // 2, W, H // 2], fill=(150, 170, 190), width=1)
    # Crosshair.
    draw.line([W // 2 - 10, H // 2, W // 2 + 10, H // 2], fill=(255, 255, 255), width=1)
    draw.line([W // 2, H // 2 - 10, W // 2, H // 2 + 10], fill=(255, 255, 255), width=1)

    # Project + draw each visible marker.
    for mid, wp in markers:
        if wp is None:
            continue
        mx, my, mz, kind, size_m = wp
        rx, ry, rz = mx - dx, my - dy, mz - dz
        fdepth = rx * fwd_x + ry * fwd_y          # forward distance
        if fdepth <= 0.15:
            continue
        rcomp = rx * right_x + ry * right_y       # right offset
        ucomp = rz                                # up offset
        sx = W / 2 + f * (rcomp / fdepth)
        sy = H / 2 - f * (ucomp / fdepth)
        side = max(6.0, f * (size_m / fdepth))
        half = side / 2
        if sx < -half or sx > W + half:
            continue
        color = _marker_color(mid, kind)
        draw.rectangle([sx - half, sy - half, sx + half, sy + half],
                       outline=color, width=max(2, int(side / 12)))
        # inner fill hint + id label
        draw.rectangle([sx - half * 0.5, sy - half * 0.5, sx + half * 0.5, sy + half * 0.5],
                       fill=color)
        draw.text((sx - half, sy - half - 12), str(mid), fill=color)

    # HUD.
    draw.rectangle([0, 0, W, 26], fill=_HUD_BG)
    draw.text((6, 7),
              f"{drone_id}  team={team}  phase={phase}  "
              f"{'FLYING' if flying else 'GROUND'}",
              fill=(0, 255, 0) if flying else (200, 200, 200))
    draw.rectangle([0, H - 22, W, H], fill=_HUD_BG)
    draw.text((6, H - 17),
              f"pos=({dx:+.1f},{dy:+.1f},{dz:+.1f})  hdg={heading:.0f}  "
              f"markers={len(markers)}",
              fill=(200, 200, 200))
    return img


def render_frame_jpeg(world: World, drone_id: str, quality: int = 70) -> bytes:
    buf = io.BytesIO()
    render_frame(world, drone_id).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
