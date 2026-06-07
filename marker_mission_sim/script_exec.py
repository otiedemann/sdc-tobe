"""Kinematic executor for the flight-controller mission-script DSL.

The simulator does NOT model aerodynamics. It interprets the *existing*
FC mission-script verbs (the same language ``marker_mission.mission_script``
parses for the real drone) and moves a :class:`SimDrone` toward the target
each frame at fixed rates. ``World.tick`` calls :func:`advance` once per
drone per frame while already holding ``world.lock``.

Two public entry points:

* :func:`parse_script` — turn script text into a flat list of :class:`Step`
  (verb + parsed args + raw line + optional ``marker_id``). This is a
  deliberately thin parser: it does NOT validate arity or ranges the way
  the real FC parser does — the sim only needs enough structure to drive
  the kinematics, and unknown/oddly-shaped lines degrade gracefully.
* :func:`advance` — execute the current step for one frame.

Headings are degrees clockwise from +Y (front wall), matching the rest of
the package (``heading_vec(θ) = (sin θ, cos θ)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .geometry import Vec3, heading_vec, heading_of, wrap_deg, clamp, bearing_to
from .world import World, SimDrone


# ---------------------------------------------------------------------------
# Kinematic constants (tune here)
# ---------------------------------------------------------------------------

CRUISE_SPEED = 2.0      # m/s — forward speed at full FB stick / cruising moves
LATERAL_SPEED = 1.5     # m/s — sideways speed at full LR stick
CLIMB_SPEED = 1.0       # m/s — vertical speed (climb/descend)
YAW_RATE = 90.0         # deg/s — rotation rate at full yaw stick / IMU turns

ARRIVE_XY = 0.25        # m  — horizontal arrival tolerance
ARRIVE_Z = 0.1          # m  — vertical arrival tolerance
ARRIVE_YAW = 5.0        # deg — heading arrival tolerance

TAKEOFF_ALT = 0.9       # m  — hover altitude reached by TAKEOFF
DEFAULT_APPROACH_DIST = 1.0   # m  — APPROACH stop distance when unspecified
DEFAULT_HOLD_S = 1.0    # s  — HOOVER/PAUSE/AWAIT default dwell
UNKNOWN_MARKER_HOLD_S = 1.5   # s — hold when a target marker can't be located
HOME_INSET = 2.5        # m  — TO_HOME sits this far in from the team's wall


# ---------------------------------------------------------------------------
# Step model + parser
# ---------------------------------------------------------------------------

# Verbs whose FIRST argument is a marker id. ``parse_script`` copies that
# token into ``Step.marker_id`` so world.py can report the active target
# marker on /api/state without re-parsing.
_MARKER_ID_VERBS = frozenset({"APPROACH", "GO_HOME", "FB_BRAKE", "AWAIT"})


@dataclass
class Step:
    verb: str
    args: list = field(default_factory=list)   # tokens after verb (float if numeric else str)
    raw: str = ""                              # original line text (world.py reads s.raw)
    marker_id: Optional[int] = None            # APPROACH/FB_BRAKE target (world.py reads s.marker_id)


def _maybe_float(tok: str):
    """Parse ``tok`` as float when it looks numeric, else return the raw str."""
    try:
        return float(tok)
    except ValueError:
        return tok


def parse_script(text: str) -> list:
    """Parse mission-script ``text`` into a flat list of :class:`Step`.

    Lines are split on newlines; blank lines and lines whose first
    non-space character is ``#`` are ignored (an inline ``# comment`` tail
    is also stripped). The first token is the verb (upper-cased); the rest
    are parsed as floats where numeric, otherwise kept as strings. For
    marker-targeting verbs the first arg is also stored in ``marker_id``.
    """
    steps: list = []
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Drop an inline comment tail (e.g. "TO 1 2  # go there").
        code = stripped.split("#", 1)[0].strip()
        if not code:
            continue
        tokens = code.split()
        verb = tokens[0].upper()
        args = [_maybe_float(t) for t in tokens[1:]]

        marker_id: Optional[int] = None
        if verb in _MARKER_ID_VERBS and args:
            first = args[0]
            if isinstance(first, float) and float(first).is_integer():
                marker_id = int(first)
        steps.append(Step(verb=verb, args=args, raw=raw.rstrip("\n"),
                          marker_id=marker_id))
    return steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def marker_world_pos(world: World, marker_id) -> Optional[Vec3]:
    """World position of a marker id, or None if unknown.

    Resolves wall markers (``world.wall_markers``) and target-box faces
    (a box whose ``current_face_id`` / ``blue_face_id`` / ``red_face_id``
    matches the id → that box's position).
    """
    if marker_id is None:
        return None
    try:
        mid = int(marker_id)
    except (TypeError, ValueError):
        return None

    for m in world.wall_markers:
        if m.id == mid:
            return m.pos.copy()

    for box in world.boxes.values():
        if mid in (box.current_face_id, box.blue_face_id, box.red_face_id):
            return box.pos.copy()
    return None


def _marker_inward_normal(world: World, marker_id) -> Optional[tuple]:
    """Inward (into-arena) unit normal of a wall marker, by its wall side, or
    None if the marker isn't a wall marker. Used by GO_HOME's fan-out arrival."""
    try:
        mid = int(marker_id)
    except (TypeError, ValueError):
        return None
    for m in world.wall_markers:
        if m.id == mid:
            wall = str(getattr(m, "wall", "")).lower()
            return {"back": (0.0, 1.0), "front": (0.0, -1.0),
                    "left": (1.0, 0.0), "right": (-1.0, 0.0)}.get(wall)
    return None


def _home_pos(world: World, team: str, alt: Optional[float]) -> Vec3:
    """Team home: red near the back (-y) wall, blue near the front (+y)."""
    arena = world.arena
    if str(team).lower() == "blue":
        y = arena.y_max - HOME_INSET
    else:
        y = arena.y_min + HOME_INSET
    z = alt if alt is not None else None
    return Vec3(0.0, y, z if z is not None else 0.0)


def _arg(step: Step, idx: int, default=None):
    """Numeric arg at ``idx`` (as float) or ``default`` if absent/non-numeric."""
    if idx < len(step.args):
        v = step.args[idx]
        if isinstance(v, (int, float)):
            return float(v)
    return default


def _clamp_to_arena(world: World, pos: Vec3) -> None:
    """Clamp ``pos`` in place to the arena box (x/y bounds, z in [0, ceiling])."""
    a = world.arena
    pos.x = clamp(pos.x, a.x_min, a.x_max)
    pos.y = clamp(pos.y, a.y_min, a.y_max)
    pos.z = clamp(pos.z, 0.0, a.ceiling_m)


def _step_xy(drone: SimDrone, world: World, tx: float, ty: float,
             dt: float, speed: float) -> tuple[float, bool]:
    """Move horizontally toward (tx, ty). Returns (ground_speed, arrived)."""
    dx, dy = tx - drone.pos.x, ty - drone.pos.y
    dist = math.hypot(dx, dy)
    if dist <= ARRIVE_XY:
        return 0.0, True
    step = speed * dt
    if step >= dist:
        drone.pos.x, drone.pos.y = tx, ty
        applied = dist / dt if dt > 0 else 0.0
    else:
        ux, uy = dx / dist, dy / dist
        drone.pos.x += ux * step
        drone.pos.y += uy * step
        applied = speed
    # Face the direction of travel so vision/heading stay sensible.
    drone.heading_deg = heading_of(dx, dy)
    _clamp_to_arena(world, drone.pos)
    return applied, math.hypot(tx - drone.pos.x, ty - drone.pos.y) <= ARRIVE_XY


def _step_z(drone: SimDrone, world: World, tz: float, dt: float) -> bool:
    """Climb/descend toward ``tz`` at CLIMB_SPEED. Returns arrived?"""
    dz = tz - drone.pos.z
    if abs(dz) <= ARRIVE_Z:
        return True
    step = math.copysign(min(CLIMB_SPEED * dt, abs(dz)), dz)
    drone.pos.z += step
    _clamp_to_arena(world, drone.pos)
    return abs(tz - drone.pos.z) <= ARRIVE_Z


def _apply_body_velocity(drone: SimDrone, world: World, dt: float,
                         fb: float, lr: float, ud: float,
                         yaw_rate: float) -> float:
    """Integrate body-frame stick velocities for one frame.

    ``fb``/``lr``/``ud`` are m/s (forward / right / up). ``yaw_rate`` is
    deg/s (CW). Returns the horizontal ground speed applied this frame.
    """
    if yaw_rate:
        drone.heading_deg = wrap_deg(drone.heading_deg + yaw_rate * dt)
    fdx, fdy = heading_vec(drone.heading_deg)        # forward unit (x, y)
    rdx, rdy = heading_vec(drone.heading_deg + 90.0)  # right unit (x, y)
    vx = fb * fdx + lr * rdx
    vy = fb * fdy + lr * rdy
    drone.pos.x += vx * dt
    drone.pos.y += vy * dt
    drone.pos.z += ud * dt
    _clamp_to_arena(world, drone.pos)
    return math.hypot(vx, vy)


# ---------------------------------------------------------------------------
# advance(): per-frame step execution
# ---------------------------------------------------------------------------

def advance(drone: SimDrone, world: World, dt: float, now: float) -> None:
    """Advance ``drone`` by one frame of mission-script execution."""
    # 1) Command latency: install a pending script once its apply-time hits.
    if (drone.pending_script_text is not None
            and now >= drone.pending_apply_at):
        text = drone.pending_script_text
        drone.script = parse_script(text)
        drone.script_text = text
        drone.step_idx = 0
        drone.step_t0 = now
        drone.mem = {}
        drone.phase = "running"
        drone.pending_script_text = None

    # 2) Not running, or script exhausted → hold position.
    if drone.phase != "running" or drone.step_idx >= len(drone.script):
        if drone.phase == "running" and drone.step_idx >= len(drone.script):
            drone.phase = "done"
        drone.speed_mps = 0.0
        return

    # 3) Execute the current step for this frame.
    step = drone.script[drone.step_idx]
    elapsed = now - drone.step_t0
    speed, done = _exec_step(drone, world, step, dt, now, elapsed)

    # 4) Record the ground speed applied (capture gating reads this).
    drone.speed_mps = max(0.0, speed)

    # 5) Step bookkeeping: advance / finish.
    if done:
        drone.step_idx += 1
        drone.step_t0 = now
        drone.mem = {}
        if drone.step_idx >= len(drone.script):
            drone.phase = "done"


def _exec_step(drone: SimDrone, world: World, step: Step, dt: float,
               now: float, elapsed: float) -> tuple[float, bool]:
    """Run one verb for one frame. Returns (ground_speed, step_done)."""
    verb = step.verb
    mem = drone.mem

    # -- vertical-only -----------------------------------------------------
    if verb == "TAKEOFF":
        drone.flying = True
        done = _step_z(drone, world, TAKEOFF_ALT, dt)
        # Treat "already at/above hover alt" as immediately satisfied.
        if drone.pos.z >= TAKEOFF_ALT - ARRIVE_Z:
            done = True
        return 0.0, done

    if verb == "HEIGHT":
        h = _arg(step, 0, TAKEOFF_ALT)
        return 0.0, _step_z(drone, world, h, dt)

    if verb == "LAND":
        done = _step_z(drone, world, 0.0, dt)
        if drone.pos.z <= ARRIVE_Z:
            drone.pos.z = 0.0
            drone.flying = False
            done = True
        return 0.0, done

    # -- horizontal goto ---------------------------------------------------
    if verb == "TO":
        tx = _arg(step, 0, drone.pos.x)
        ty = _arg(step, 1, drone.pos.y)
        tz = _arg(step, 2, None)
        z_ok = True if tz is None else _step_z(drone, world, tz, dt)
        spd, xy_ok = _step_xy(drone, world, tx, ty, dt, CRUISE_SPEED)
        return spd, (xy_ok and z_ok)

    if verb == "TO_HOME":
        alt = _arg(step, 0, None)
        home = _home_pos(world, drone.team, alt)
        z_ok = True if alt is None else _step_z(drone, world, alt, dt)
        spd, xy_ok = _step_xy(drone, world, home.x, home.y, dt, CRUISE_SPEED)
        return spd, (xy_ok and z_ok)

    # -- timed open-loop RC sticks ----------------------------------------
    if verb == "FB_RC":
        stick = _arg(step, 0, 0.0)
        dur = _arg(step, 1, 1.0)
        spd = _apply_body_velocity(drone, world, dt,
                                   fb=(stick / 100.0) * CRUISE_SPEED,
                                   lr=0.0, ud=0.0, yaw_rate=0.0)
        return spd, elapsed >= dur

    if verb == "LR_RC":
        stick = _arg(step, 0, 0.0)
        dur = _arg(step, 1, 1.0)
        spd = _apply_body_velocity(drone, world, dt, fb=0.0,
                                   lr=(stick / 100.0) * LATERAL_SPEED,
                                   ud=0.0, yaw_rate=0.0)
        return spd, elapsed >= dur

    if verb == "UD_RC":
        stick = _arg(step, 0, 0.0)
        dur = _arg(step, 1, 1.0)
        _apply_body_velocity(drone, world, dt, fb=0.0, lr=0.0,
                             ud=(stick / 100.0) * CLIMB_SPEED, yaw_rate=0.0)
        return 0.0, elapsed >= dur

    if verb == "YAW_RC":
        stick = _arg(step, 0, 0.0)
        dur = _arg(step, 1, 1.0)
        _apply_body_velocity(drone, world, dt, fb=0.0, lr=0.0, ud=0.0,
                             yaw_rate=(stick / 100.0) * YAW_RATE)
        return 0.0, elapsed >= dur

    if verb == "RC":
        # RC lr fb ud yaw dur — combined body-frame sticks.
        lr = _arg(step, 0, 0.0)
        fb = _arg(step, 1, 0.0)
        ud = _arg(step, 2, 0.0)
        yaw = _arg(step, 3, 0.0)
        dur = _arg(step, 4, 1.0)
        spd = _apply_body_velocity(
            drone, world, dt,
            fb=(fb / 100.0) * CRUISE_SPEED,
            lr=(lr / 100.0) * LATERAL_SPEED,
            ud=(ud / 100.0) * CLIMB_SPEED,
            yaw_rate=(yaw / 100.0) * YAW_RATE,
        )
        return spd, elapsed >= dur

    # -- closed-loop relative moves (Anafi moveBy: forward/right/up by N m) --
    # FB_IMU/LR_IMU/UD_IMU <meters> — move a FIXED body-frame distance, sensor-
    # fused on real hardware, blocking until the move completes. Relative (no
    # absolute position needed). We track the distance moved in mem and stop
    # once it reaches the request. Sign: +fwd/-back, +right/-left, +up/-down.
    if verb in ("FB_IMU", "LR_IMU", "UD_IMU"):
        meters = _arg(step, 0, 0.0)
        moved = mem.get("imu_moved", 0.0)
        remaining = abs(meters) - moved
        if remaining <= ARRIVE_XY:
            return 0.0, True
        sign = 1.0 if meters >= 0 else -1.0
        speed = (CLIMB_SPEED if verb == "UD_IMU"
                 else LATERAL_SPEED if verb == "LR_IMU" else CRUISE_SPEED)
        leg = min(speed * dt, remaining)        # don't overshoot the request
        fb = sign * leg / dt if verb == "FB_IMU" and dt > 0 else 0.0
        lr = sign * leg / dt if verb == "LR_IMU" and dt > 0 else 0.0
        ud = sign * leg / dt if verb == "UD_IMU" and dt > 0 else 0.0
        spd = _apply_body_velocity(drone, world, dt, fb=fb, lr=lr, ud=ud,
                                   yaw_rate=0.0)
        mem["imu_moved"] = moved + leg
        return spd, (moved + leg) >= abs(meters) - ARRIVE_XY

    # FB_UD_IMU <forward> <up> — combined forward+up in ONE move (rise while
    # advancing). Drive both body axes together so they finish simultaneously.
    if verb == "FB_UD_IMU":
        fwd_m = _arg(step, 0, 0.0)
        up_m = _arg(step, 1, 0.0)
        total = math.hypot(fwd_m, up_m)
        moved = mem.get("imu_moved", 0.0)
        if total <= 1e-6 or (total - moved) <= ARRIVE_XY:
            return 0.0, True
        leg = min(CRUISE_SPEED * dt, total - moved)
        frac = (leg / total) if total > 0 else 0.0
        fb = (fwd_m * frac) / dt if dt > 0 else 0.0
        ud = (up_m * frac) / dt if dt > 0 else 0.0
        spd = _apply_body_velocity(drone, world, dt, fb=fb, lr=0.0, ud=ud,
                                   yaw_rate=0.0)
        mem["imu_moved"] = moved + leg
        return spd, (moved + leg) >= total - ARRIVE_XY

    # -- closed-loop yaw ---------------------------------------------------
    if verb == "YAW":
        # ABSOLUTE heading hold (arena frame, deg CW from +Y). Matches the real
        # marker_mission YAW verb. 0 = face +Y (front/enemy for red), 180 = -Y.
        # Rotate toward the absolute target regardless of current heading.
        target = wrap_deg(_arg(step, 0, drone.heading_deg))
        err = wrap_deg(target - drone.heading_deg)
        if abs(err) <= ARRIVE_YAW:
            drone.heading_deg = target
            return 0.0, True
        step_deg = math.copysign(min(YAW_RATE * dt, abs(err)), err)
        drone.heading_deg = wrap_deg(drone.heading_deg + step_deg)
        return 0.0, abs(wrap_deg(target - drone.heading_deg)) <= ARRIVE_YAW

    if verb == "YAW_IMU":
        # Relative rotation by `deg` at `rate`; latch a target heading.
        # Optional 2nd arg overrides the rotation rate (deg/s) — mirrors the
        # FC's YAW_IMU speed override (temporary MaxRotationSpeed). None ->
        # the default YAW_RATE.
        if "yaw_target" not in mem:
            deg = _arg(step, 0, 0.0)
            mem["yaw_target"] = wrap_deg(drone.heading_deg + deg)
        spd = _arg(step, 1, None)
        rate = YAW_RATE
        if spd is not None:
            try:
                rate = max(1.0, min(180.0, float(spd)))
            except (TypeError, ValueError):
                rate = YAW_RATE
        target = mem["yaw_target"]
        err = wrap_deg(target - drone.heading_deg)
        if abs(err) <= ARRIVE_YAW:
            drone.heading_deg = wrap_deg(target)
            return 0.0, True
        step_deg = math.copysign(min(rate * dt, abs(err)), err)
        drone.heading_deg = wrap_deg(drone.heading_deg + step_deg)
        return 0.0, abs(wrap_deg(target - drone.heading_deg)) <= ARRIVE_YAW

    if verb == "SCOUT":
        # Slow 360° spin in place: accumulate yaw until a full turn.
        acc = mem.get("yaw_acc", 0.0)
        d = YAW_RATE * dt
        remaining = 360.0 - acc
        if d >= remaining:
            d = remaining
        _apply_body_velocity(drone, world, dt, fb=0.0, lr=0.0, ud=0.0,
                             yaw_rate=d / dt if dt > 0 else 0.0)
        acc += d
        mem["yaw_acc"] = acc
        return 0.0, acc >= 360.0 - 1e-6

    # -- hold --------------------------------------------------------------
    if verb in ("HOOVER", "PAUSE", "AWAIT"):
        dur = _arg(step, 0, DEFAULT_HOLD_S)
        # AWAIT's first arg is a marker id; its dwell is the 2nd token.
        if verb == "AWAIT":
            dur = _arg(step, 1, DEFAULT_HOLD_S)
        return 0.0, elapsed >= dur

    # -- vision-style arrivals --------------------------------------------
    if verb == "APPROACH":
        dist = _arg(step, 1, DEFAULT_APPROACH_DIST)
        tgt = marker_world_pos(world, step.marker_id)
        if tgt is None:
            # Unknown marker: hold briefly then give up.
            return 0.0, elapsed >= UNKNOWN_MARKER_HOLD_S
        # Home to a standoff `dist` from the marker (xy plane). The real FC's
        # forward PD drives toward the marker when too far AND backs off when too
        # close, holding the standoff — so we aim for the point on the standoff
        # ring along the current bearing (marker -> drone), not the marker xy.
        # Without this, a drone that starts CLOSER than `dist` (e.g. the scout,
        # already within 6 m of a side marker) would count as "arrived" without
        # ever centring.
        cur_d = drone.pos.dist_xy(tgt)
        if abs(cur_d - dist) <= ARRIVE_XY:
            return 0.0, True
        if cur_d > 1e-6:
            ux = (drone.pos.x - tgt.x) / cur_d
            uy = (drone.pos.y - tgt.y) / cur_d
        else:
            ux, uy = 0.0, 1.0
        spd, _ = _step_xy(drone, world, tgt.x + dist * ux, tgt.y + dist * uy,
                          dt, CRUISE_SPEED)
        return spd, abs(drone.pos.dist_xy(tgt) - dist) <= ARRIVE_XY

    if verb == "GO_HOME":
        # Loose APPROACH onto the home back-wall marker: settle anywhere
        # within ±tol of `dist` ("be roughly in the home zone"). Mirrors the
        # FC's GO_HOME (APPROACH with a wide target_distance_tol_m).
        # args: [marker_id, dist=3.5, tol=0.5, hdg=0]
        dist = _arg(step, 1, 3.5)
        tol = _arg(step, 2, 0.5)
        hdg = _arg(step, 3, None)        # arrival bearing off the marker normal
        try:
            band = max(0.05, float(tol))
        except (TypeError, ValueError):
            band = 0.5
        tgt = marker_world_pos(world, step.marker_id)
        if tgt is None:
            return 0.0, elapsed >= UNKNOWN_MARKER_HOLD_S
        if hdg is not None:
            # Fan-out arrival: aim for the point on the standoff ring at `hdg`
            # degrees off the marker's INWARD normal, so several drones with
            # different hdg land at DIFFERENT points around the marker (no
            # convergence). Mirrors the FC's target_relative_heading_deg.
            nrm = _marker_inward_normal(world, step.marker_id) or (0.0, 1.0)
            th = math.radians(float(hdg))
            nx, ny = nrm
            ux = nx * math.cos(th) - ny * math.sin(th)
            uy = nx * math.sin(th) + ny * math.cos(th)
            aim_x, aim_y = tgt.x + dist * ux, tgt.y + dist * uy
            if math.hypot(drone.pos.x - aim_x, drone.pos.y - aim_y) <= band:
                return 0.0, True
            spd, _ = _step_xy(drone, world, aim_x, aim_y, dt, CRUISE_SPEED)
            return spd, math.hypot(drone.pos.x - aim_x,
                                   drone.pos.y - aim_y) <= band
        cur_d = drone.pos.dist_xy(tgt)
        # Arrived once we are within the [dist-band, dist+band] standoff ring.
        if abs(cur_d - dist) <= band:
            return 0.0, True
        # Aim for the point on the standoff ring along the CURRENT bearing
        # (marker -> drone), so we drive toward the marker when too far AND
        # back away from it when too close — mirrors the FC's forward PD, which
        # retreats on a negative distance error. Driving straight at the marker
        # xy (the plain-APPROACH behaviour) would overshoot a large standoff.
        if cur_d > 1e-6:
            ux = (drone.pos.x - tgt.x) / cur_d
            uy = (drone.pos.y - tgt.y) / cur_d
        else:
            ux, uy = 0.0, 1.0   # degenerate: sitting on the marker, pick +Y
        ring_x = tgt.x + dist * ux
        ring_y = tgt.y + dist * uy
        spd, _ = _step_xy(drone, world, ring_x, ring_y, dt, CRUISE_SPEED)
        return spd, abs(drone.pos.dist_xy(tgt) - dist) <= band

    if verb == "FB_BRAKE":
        stop_m = _arg(step, 1, 0.5)
        stick = _arg(step, 2, 100.0)
        timeout = _arg(step, 3, 10.0)
        wx = _arg(step, 4, None)
        wy = _arg(step, 5, None)
        if wx is not None and wy is not None:
            tgt = Vec3(wx, wy, drone.pos.z)
        else:
            tgt = marker_world_pos(world, step.marker_id)
        if tgt is None:
            # No target to brake on: cruise forward until timeout.
            spd = _apply_body_velocity(
                drone, world, dt,
                fb=(stick / 100.0) * CRUISE_SPEED, lr=0.0, ud=0.0,
                yaw_rate=0.0)
            return spd, elapsed >= timeout
        if drone.pos.dist_xy(tgt) <= stop_m or elapsed >= timeout:
            return 0.0, True
        spd, _ = _step_xy(drone, world, tgt.x, tgt.y, dt,
                          (abs(stick) / 100.0) * CRUISE_SPEED)
        return spd, (drone.pos.dist_xy(tgt) <= stop_m or elapsed >= timeout)

    # -- unknown verb: warn once and skip ----------------------------------
    world.log("warn", f"unknown verb {verb}")
    return 0.0, True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from marker_mission_sim.config import SimConfig, SimDroneConfig
    from marker_mission_sim.world import World

    cfg = SimConfig(drones=[SimDroneConfig(id="sim1", team="red", fc_port=8551,
                                           spawn_x=0.0, spawn_y=0.0,
                                           spawn_heading_deg=0.0)])
    world = World(cfg)

    t = 0.0
    dt = 0.1
    # Drive a synthetic clock so command latency is deterministic.
    world.queue_script("sim1", "TAKEOFF\nHEIGHT 2.0\nTO 3 -7.5\nHOOVER 1\nLAND\n",
                       now=t)
    d = world.drones["sim1"]

    print(f"{'tick':>4} {'t':>5} {'phase':>8} {'step':>4} "
          f"{'pos':>26} {'spd':>5} fly")
    for i in range(200):
        world.tick(dt=dt, now=t)
        if i % 20 == 0 or d.phase == "done":
            p = d.pos
            print(f"{i:>4} {t:>5.1f} {d.phase:>8} {d.step_idx:>4} "
                  f"({p.x:>6.2f},{p.y:>6.2f},{p.z:>6.2f}) "
                  f"{d.speed_mps:>5.2f} {d.flying}")
            if d.phase == "done":
                break
        t += dt

    print(f"\nfinal: phase={d.phase} pos={d.pos.to_list()} "
          f"flying={d.flying} step_idx={d.step_idx}/{len(d.script)}")
