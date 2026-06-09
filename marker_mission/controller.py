"""
Mission state machine and PD controllers.

The controller produces RC commands (lr, fb, ud, yaw) at a fixed rate by
combining:

* the most recent marker pose from the detector,
* the most recent telemetry sample from the drone,
* the configured target distance and target relative heading,
* and PD gains from :mod:`config`.

State machine
-------------

::

    INIT -> TAKEOFF -> SEARCH -> ALIGN -> APPROACH -> HOLD -> LAND -> DONE
                          ^         |        |          |
                          |         v        v          v
                          \\---------+--------+----------/   (marker lost)

Phases:

* TAKEOFF   -- request takeoff and wait until ``flying`` becomes True.
* SEARCH    -- yaw in place. If the marker becomes visible, go to ALIGN.
               If a full sweep completes without a sighting, LAND.
* ALIGN     -- orbit at the start radius until heading is roughly 0
               (drone roughly faces the marker straight on) so that the
               subsequent APPROACH stays inside the detector's reliable
               tilt range.
* APPROACH  -- close yaw error to zero AND distance error to zero,
               while gently holding heading near 0.
* HOLD      -- hover for ``hold_time_s`` while station-keeping
               (yaw / distance / heading all actively corrected) at the
               post-approach position. Then transitions to LAND.
* LAND      -- request land and wait until ``flying`` becomes False.
* DONE      -- terminal state.

Any phase except TAKEOFF/LAND/DONE will fall back to SEARCH if the marker
is lost for longer than ``pose_max_age_s``.
"""

from __future__ import annotations

import enum
import math
import random as _random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .aruco_detector import MarkerPose
from .config import MissionConfig
from .drone_api import DroneApi, DroneApiError, TelemetrySnapshot
from .mission_script import (HARDCODED_DEFAULT_SCRIPT, ScriptError, Step,
                              defaults_from_cfg, parse as parse_script)
from . import cpu_monitor


def _host_resource_dict() -> dict:
    """Latest CPU / memory sample as a serialisable dict for
    state.snapshot(). Returns all-None fields if the sampler hasn't
    produced a delta yet."""
    s = cpu_monitor.last_sample()
    return {
        "system_cpu_pct": (None if s.system_cpu_pct is None
                           else round(s.system_cpu_pct, 1)),
        "process_cpu_pct": (None if s.process_cpu_pct is None
                            else round(s.process_cpu_pct, 1)),
        "load_1m": s.load_1m,
        "mem_used_pct": (None if s.mem_used_pct is None
                         else round(s.mem_used_pct, 1)),
        "sample_age_s": (round(time.monotonic() - s.sampled_at, 2)
                         if s.sampled_at > 0.0 else None),
    }


# ---------------------------------------------------------------------------
# State machine enum
# ---------------------------------------------------------------------------

class Phase(enum.Enum):
    INIT      = "init"
    TAKEOFF   = "takeoff"
    SEARCH    = "search"
    ALIGN     = "align"     # orbit at start radius until heading ~ 0
    HEIGHT_ALIGN = "height_align"   # match drone height to marker height
    APPROACH  = "approach"  # close distance to target with heading held near 0
    HOLD      = "hold"      # station-keeping hover for hold_time_s, then LAND
    IDLE      = "idle"      # mission-script HOOVER w/o prior APPROACH
    HEIGHT    = "height"    # mission-script HEIGHT step (climb/descend to target)
    GOTO      = "goto"      # mission-script TO step (drive to arena-frame point)
    DANCE     = "dance"     # mission-script DANCE step (programmed RC routine)
    RC        = "rc"        # mission-script LR / FB / UD / RC (raw stick + timer)
    FB_BRAKE  = "fb_brake"  # mission-script FB_BRAKE (open-loop fwd + vision brake)
    ROTATE    = "rotate"    # mission-script YAW_IMU (discrete rotation by N deg)
    SCOUT     = "scout"     # mission-script SCOUT (slow 360° yaw spin)
    MOVE_IMU  = "move_imu"  # mission-script FB_IMU / LR_IMU / UD_IMU
    AUTO_ATTACK = "auto_attack"  # reactive end-to-end attack (own state machine)
    THREAT_OBSERVE = "threat_observe"  # enemy drone ≥150 px: hover + watch
    THREAT_EVADE   = "threat_evade"    # still present after 5 s: lateral evade
    LAND      = "land"
    DONE      = "done"
    ABORT     = "abort"


# ---------------------------------------------------------------------------
# PD controller building block
# ---------------------------------------------------------------------------

class PIDController:
    """Proportional-integral-derivative controller with output clamp
    and back-calculation anti-windup.

    The derivative is computed from filtered sample-to-sample differences
    so that quick measurement jitter does not blow up the D term.

    The integral term defaults to OFF (``ki = 0.0``) so the class
    behaves exactly like the older PD controller until an operator
    bumps ``ki`` via /tune. Anti-windup is two-layered:

    * Hard clamp: ``|integral| <= i_clip``. Last-line guard against
      an integrator runaway during marker loss / sustained saturation.
    * Back-calculation: when the unclamped output exceeds out_clip,
      the integrator is wound back by ``(u_unclipped - u_clipped) / ki``
      so it doesn't keep growing while the actuator is pinned.

    The integrator is also frozen in a few obvious cases (``dt > 1 s``
    suggests a missed tick / phase change; ``ki == 0`` skips the
    integration math entirely).
    """

    def __init__(self, kp: float, kd: float, out_clip: float,
                 ki: float = 0.0, i_clip: float = 1.0,
                 d_filter_alpha: float = 0.4):
        self.kp = float(kp)
        self.kd = float(kd)
        self.ki = float(ki)
        self.out_clip = float(out_clip)
        self.i_clip = float(i_clip)
        self.alpha = float(d_filter_alpha)
        self._last_err: Optional[float] = None
        self._last_t: Optional[float] = None
        self._d_filt: float = 0.0
        self._integral: float = 0.0

    def reset(self) -> None:
        self._last_err = None
        self._last_t = None
        self._d_filt = 0.0
        self._integral = 0.0

    def step(self, error: float, now: float) -> float:
        # Derivative term -------------------------------------------------
        if self._last_err is None or self._last_t is None or now - self._last_t < 1e-3:
            d_err = 0.0
        else:
            d_raw = (error - self._last_err) / (now - self._last_t)
            self._d_filt = self.alpha * d_raw + (1.0 - self.alpha) * self._d_filt
            d_err = self._d_filt

        # Integral term ---------------------------------------------------
        # Skip the math when ki == 0 so the integrator state stays at
        # zero and the controller is provably equivalent to today's PD.
        if self.ki != 0.0 and self._last_t is not None:
            dt = now - self._last_t
            if 0.0 < dt < 1.0:
                self._integral += error * dt
                # Hard clamp.
                if self._integral > self.i_clip:
                    self._integral = self.i_clip
                elif self._integral < -self.i_clip:
                    self._integral = -self.i_clip

        self._last_err = error
        self._last_t = now

        u_unclipped = (self.kp * error
                       + self.kd * d_err
                       + self.ki * self._integral)
        if u_unclipped > self.out_clip:
            u = self.out_clip
        elif u_unclipped < -self.out_clip:
            u = -self.out_clip
        else:
            u = u_unclipped

        # Back-calculation anti-windup. When the unclamped command
        # was past saturation, unwind the integrator by exactly the
        # excess so it can't keep growing while the actuator is
        # pinned. Only meaningful with ki != 0.
        if self.ki != 0.0 and u != u_unclipped:
            self._integral -= (u_unclipped - u) / self.ki
            if self._integral > self.i_clip:
                self._integral = self.i_clip
            elif self._integral < -self.i_clip:
                self._integral = -self.i_clip

        return u


# Backwards-compatible alias. The old PD-only class name is widely
# used in this module and in commit messages; keep it pointing at
# the new PID class so the rest of the file (and any external tests)
# don't have to change.
PDController = PIDController


# ---------------------------------------------------------------------------
# Smoothed pose
# ---------------------------------------------------------------------------

class PoseSmoother:
    """Exponential moving average of the most recent marker pose.

    We smooth distance / yaw / heading independently because they have
    different noise characteristics. We do NOT smooth across angle wrap
    boundaries -- the values come out of solvePnP in the (-180, 180]
    range and never wrap during a normal mission, so plain EMA is fine.

    The heading channel additionally has an outlier-rejection guard:
    a single-tick change beyond ``hdg_jump_max_deg`` is physically
    impossible (would require >140 deg/s tangential motion at d=1.5m),
    so it must be a planar-pose mirror flip from the IPPE solver. We
    hold the previous smoothed value through such a sample. If the
    "jump" persists for ``hdg_jump_consecutive_max`` consecutive ticks,
    we accept it as real motion and resume normal smoothing -- this
    way a real but fast manoeuvre still gets tracked, while a single
    branch flip gets ignored.
    """

    HDG_JUMP_MAX_DEG = 5.0            # max plausible per-tick change
    HDG_JUMP_CONSECUTIVE_MAX = 20     # accept after this many "jump" ticks
                                      # (was 5 = 0.5s @ 10Hz; raised to 20 = 2s
                                      # after IPPE branch flips at >10m distance
                                      # persisted 5+ ticks and contaminated ALIGN)

    def __init__(self, alpha: float, max_age_s: float):
        self.alpha = float(alpha)
        self.max_age_s = float(max_age_s)
        self._d: Optional[float] = None
        self._yaw: Optional[float] = None
        self._hdg: Optional[float] = None
        self._last_seen: float = 0.0
        self._hdg_jump_streak: int = 0

    def reset(self) -> None:
        self._d = None
        self._yaw = None
        self._hdg = None
        self._last_seen = 0.0
        self._hdg_jump_streak = 0

    def update(self, pose: Optional[MarkerPose], now: float) -> None:
        if pose is None:
            return
        if self._d is None:
            self._d = pose.distance_m
            self._yaw = pose.yaw_deg
            self._hdg = pose.relative_heading_deg
            self._hdg_jump_streak = 0
        else:
            a = self.alpha
            self._d   = a * pose.distance_m            + (1 - a) * self._d
            self._yaw = a * pose.yaw_deg               + (1 - a) * self._yaw
            # Heading: outlier-reject single-frame jumps. shortest-arc
            # delta in (-180, 180].
            delta = ((pose.relative_heading_deg - self._hdg + 540.0)
                     % 360.0) - 180.0
            if abs(delta) > self.HDG_JUMP_MAX_DEG:
                self._hdg_jump_streak += 1
                if self._hdg_jump_streak < self.HDG_JUMP_CONSECUTIVE_MAX:
                    # treat as IPPE flip; hold previous smoothed hdg
                    pass
                else:
                    # persistent -> real motion, resync
                    self._hdg = pose.relative_heading_deg
                    self._hdg_jump_streak = 0
            else:
                self._hdg = a * pose.relative_heading_deg + (1 - a) * self._hdg
                self._hdg_jump_streak = 0
        self._last_seen = now

    def get(self, now: float) -> Optional[tuple[float, float, float]]:
        if self._d is None:
            return None
        if (now - self._last_seen) > self.max_age_s:
            return None
        return (self._d, self._yaw, self._hdg)

    @property
    def last_seen(self) -> float:
        return self._last_seen


# ---------------------------------------------------------------------------
# Raw-RC step (LR / FB / UD / RC) brake settings.
# After the drive timer expires _step_rc sends zero sticks and waits
# until the body-frame velocity (vgx² + vgy² + vgz²)^½ drops below
# ``RC_BRAKE_SPEED_THRESHOLD_CMS`` — at which point the drone has
# essentially stopped and the next step starts from a clean baseline.
# ``RC_BRAKE_MAX_SETTLE_S`` caps the wait so a stuck telemetry feed
# never hangs the script. Numbers tuned for Anafi indoor — the
# stabiliser brakes hard, so most steps settle in well under 1 s.
# ---------------------------------------------------------------------------
RC_BRAKE_SPEED_THRESHOLD_CMS: float = 8.0
RC_BRAKE_MAX_SETTLE_S: float = 1.5
# Hard cap on the mid-approach height-align pause. If it can't settle within
# this (e.g. the marker is mounted below the safe min-altitude floor and the
# camera keeps asking to descend, which can never be satisfied), force-release
# into the final approach so the drone never hovers metres short of the target.
MID_HA_MAX_S: float = 5.0
# Hard timeout for the HEIGHT_ALIGN phase. If the marker is mounted below the
# drone's physical minimum altitude the settle condition can never be met
# (min_height floor guard clamps e_h=0 but yaw may still be outside band, or
# the drone simply cannot descend far enough). Force APPROACH after this many
# seconds so the mission doesn't stall indefinitely.
HEIGHT_ALIGN_MAX_S: float = 5.0
# Hard timeout for the lost-box SEARCH "descend back to the approach-start
# height" gate. Bounds the descend so a noisy altimeter (or a target below the
# min-height floor) can never stall the search before the yaw sweep starts.
SEARCH_DESCEND_MAX_S: float = 6.0

# HEIGHT_ALIGN uses a much shorter pose staleness limit than the general
# pose_max_age_s (2 s). Flying on a 2-s-old pose during active altitude
# control causes full-throttle blind descent: the smoother serves the last
# bad estimate and the drone sinks until the barometric floor guard fires.
# 0.3 s gives one missed frame at 10 Hz before HEIGHT_ALIGN calls marker_lost.
HEIGHT_ALIGN_POSE_MAX_AGE_S: float = 0.3

# During HEIGHT_ALIGN, stop any downward correction when the barometer reads
# at or below this altitude. The standard min_height_m (0.20 m) only fires
# when the drone is nearly on the ground; this earlier guard (default 0.40 m)
# prevents a bad pose estimate from driving the drone into the floor.
HEIGHT_ALIGN_DESCENT_FLOOR_M: float = 0.40

# FB_BRAKE world-position fallback parameters. When vision can't see
# the target marker (yaw drift, occlusion, marker outside FOV), the
# brake uses the drone's positioning-subsystem estimate of its own
# arena-frame position vs the marker's KNOWN arena-frame position
# from the arena config. The estimator drifts ~3-5 m at cruise
# speeds, so the brake threshold gets a generous extra margin so we
# brake EARLY (safe distance from wall) rather than risk crashing
# through it by trusting a stale estimate.
# Bumped from 2.0 → 3.0 after observing the position estimator can
# be 2-3 m off even at spawn (drone stationary, multiple wall markers
# visible). 3 m margin keeps us safely short of any wall behind the
# target marker.
WORLD_FALLBACK_MARGIN_M: float = 3.0
# Don't trust position estimates older than this. If positioning
# has been silent (no marker observations) for longer, the estimate
# is too unreliable for a fast-flight brake decision; we fall back
# to the hard timeout instead.
WORLD_POSITION_MAX_AGE_S: float = 1.5
# A marker pose newer than this counts as "vision live" — recent
# enough to trust for closed-loop braking. While vision is live the
# world-position fallback is suppressed so it can't preempt a clean
# visual approach (the fallback is for when the marker is LOST).
VISION_LIVE_MAX_AGE_S: float = 0.5

# ── AUTO_ATTACK reactive controller tunables ────────────────────────
# Safe cruise altitude (above the 1.4 m slot boxes). The CLIMB
# sub-state drives ud until the drone reaches this.
AA_CRUISE_ALT_M: float = 1.6
AA_ALT_TOLERANCE_M: float = 0.2
# Forward/yaw control gains. fb is RC per metre of remaining distance;
# yaw is RC per degree of heading error. Both saturate at the caps.
AA_FB_KP: float = 18.0          # RC per metre
AA_FB_MAX: int = 55             # cruise stick ceiling (~2.2 m/s)
AA_FB_MIN: int = 12             # creep stick when very close
AA_YAW_KP: float = 2.2          # RC per degree
AA_YAW_MAX: int = 40
AA_UD_KP: float = 60.0          # RC per metre of altitude error
AA_UD_MAX: int = 80
# Only drive forward when roughly facing the goal; otherwise turn
# in place first (prevents wide curving arcs into walls).
AA_FACING_TOLERANCE_DEG: float = 35.0
# Vision homing: when the goal marker is visible, steer to centre it
# (drift-free) and brake on its measured distance.
AA_VISION_YAW_KP: float = 1.4   # RC per degree of camera bearing
AA_TARGET_STOP_M: float = 1.2   # capture standoff from the slot marker
AA_HOME_STOP_M: float = 2.0     # landing standoff from the home wall
# World-position capture trigger: only commit to capture-by-world
# when this close to the goal AND vision hasn't been live recently.
# Tight (vs the old 3.0 "arrive radius") so it can't preempt a vision
# approach that's still closing in.
AA_WORLD_CAPTURE_M: float = 1.5
# Grace window: if the goal marker was visible within this many
# seconds, keep approaching by vision even on a momentary dropout —
# do NOT fall back to a premature world-capture. (This was the bug
# that made the drone "capture" 6 m short and skip the hover.)
AA_VISION_GRACE_S: float = 1.2
# Sub-state timeouts (safety — never hang forever).
AA_CLIMB_TIMEOUT_S: float = 8.0
AA_GO_TIMEOUT_S: float = 30.0
# Capture maneuver. When the drone gets within AA_CAPTURE_TRIGGER_M of
# the target marker (by vision), it records that distance D, RISES to
# AA_CAPTURE_RISE_ALT_M (above the box top), then moves FORWARD by D so
# it ends up directly OVER the marker/box (the "drop on top" position).
AA_CAPTURE_TRIGGER_M: float = 2.0      # start the over-marker move here
AA_CAPTURE_RISE_ALT_M: float = 1.5     # rise to this before going over
AA_OVER_FORWARD_RC: int = 40           # forward stick for the over-move
AA_OVER_FORWARD_SPEED_MPS: float = 1.6 # ~stick 40 → 1.6 m/s (duration calc)
AA_OVER_FORWARD_MAX_S: float = 6.0     # safety cap on the forward move
AA_BRAKE_SPEED_CMS: float = 15.0   # consider "stopped" below this
AA_BRAKE_MAX_S: float = 2.0        # cap the brake phase

# After the over-marker "drop" the drone does NOT land. It dwells over
# the box (rotating to face our home zone and acquiring the home marker),
# flies back home, turns to face the enemy home zone, and then HOVERS —
# ready for the next attack run.
AA_DWELL_S: float = 5.0            # hover over the box (face home + find home marker)
AA_FACE_DONE_DEG: float = 12.0     # "now facing the enemy" tolerance at home
AA_FACE_TIMEOUT_S: float = 6.0     # stop rotating to face enemy after this

# SCOUT step — operator-facing "slow 360° look-around" command. Yaw
# stick is held at ``cfg.scout_yaw_stick`` (tunable, default 15) until
# cumulative yaw telemetry has rotated by SCOUT_TARGET_DEG, then we
# enter the standard brake phase. SCOUT_MAX_DRIVE_S is a safety cap
# that defeats a stuck-telemetry hang.
SCOUT_TARGET_DEG: float = 360.0
SCOUT_MAX_DRIVE_S: float = 30.0


# ---------------------------------------------------------------------------
# Dead-reckoning position estimator
# ---------------------------------------------------------------------------
#
# Used by the DANCE script step to bound the routine within a small
# radius of its entry position. The frame question (NED earth-frame vs
# body-frame rotated through yaw) was settled empirically by
# tools/analyze_velocity_frame.py: NED matched the marker-derived
# ground truth 2.2x to 3.9x better than body+yaw across every flight
# with > 1 m of motion. So vgx is treated as vN, vgy as vE.
#
# This is dead-reckoning only -- expect cm/s of drift per second of
# flight. Good enough for "stay within half a metre of where the
# dance started", not for anything tighter.

class PositionEstimator:
    MAX_DT_S = 1.0   # if we get a sample with > this gap, restart from rest

    def __init__(self):
        self.x_east_m: float = 0.0
        self.y_north_m: float = 0.0
        self._last_t: Optional[float] = None
        self._has_data: bool = False

    def reset(self) -> None:
        self.x_east_m = 0.0
        self.y_north_m = 0.0
        self._last_t = None
        self._has_data = False

    def update(self, vgx_cms: Optional[float], vgy_cms: Optional[float],
               now: float) -> None:
        if vgx_cms is None or vgy_cms is None:
            return
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0 or dt > self.MAX_DT_S:
            return
        self.x_east_m  += (vgy_cms / 100.0) * dt
        self.y_north_m += (vgx_cms / 100.0) * dt
        self._has_data = True

    @property
    def position_m(self) -> tuple[float, float]:
        return (self.x_east_m, self.y_north_m)


# ---------------------------------------------------------------------------
# Shared mission state -- read by UI / recorder, written by controller
# ---------------------------------------------------------------------------

@dataclass
class MissionState:
    phase: Phase = Phase.INIT
    phase_started_at: float = 0.0
    started_at: float = 0.0

    last_pose: Optional[MarkerPose] = None
    smoothed: Optional[tuple[float, float, float]] = None  # (d, yaw, hdg)
    last_telemetry: Optional[TelemetrySnapshot] = None

    last_rc: tuple[int, int, int, int] = (0, 0, 0, 0)        # (lr, fb, ud, yaw)
    # Active runtime targets. Populated from cfg defaults at every
    # mission start AND overridden per-step by _apply_step_to_phase.
    # Phases READ these (NOT cfg.target_*) so a script step that
    # specifies an explicit value can't pollute the cfg defaults that
    # subsequent missions parse from.
    target_distance_m: float = 1.0
    # Loose arrival band (metres) for GO_HOME: when set, the approach
    # phase completes as soon as the forward error is within ±this many
    # metres of target_distance_m, instead of the tight cfg.distance_deadband_m.
    # None -> fall back to the tight deadband (a precise APPROACH).
    target_distance_tol_m: Optional[float] = None
    # True only for a GO_HOME (loose POSITIONING hop): the approach homes at the
    # CURRENT altitude and does NOT vision-climb to the marker. A plain/capture
    # APPROACH leaves this False so it still aligns to the marker's height (even
    # when it carries its own approach_dist_tol_m loose band). Decouples the
    # "hold altitude" behaviour from target_distance_tol_m (which both verbs set).
    approach_positioning: bool = False
    # Heading tolerance (deg) for GO_HOME settle: when set, the approach also
    # gates on abs(e_hdg) < arrive_hdg_tol_deg before advancing.
    # None -> no heading gate (plain APPROACH or GO_HOME without hdg arg).
    arrive_hdg_tol_deg: Optional[float] = None
    # Yaw-settle tolerance (deg) for APPROACH's 3rd arg: when set, the approach
    # advances as soon as abs(e_yaw) < this AND the distance band is met, with NO
    # settle-time wait — a fast hand-off the instant the target distance is
    # reached (from an angle). None -> tight head-on settle (cfg.yaw_deadband_deg
    # + cfg.approach_settle_time_s).
    arrive_yaw_tol_deg: Optional[float] = None
    target_relative_heading_deg: float = 90.0
    active_marker_id: Optional[int] = None
    hold_time_s: float = 60.0
    # WAIT_AND_ATTACK tracks two marker ids for the same physical target
    # box: ``wait_attack_target_id`` is the face we actually want to
    # attack (passed by the operator); ``wait_attack_sibling_id`` is the
    # opposing-team face on the same box (derived as target ±
    # cfg.target_marker_sibling_offset, default 10). The pre-tick swap
    # in ``_wait_attack_pre_tick`` keeps ``active_marker_id`` pointing
    # at whichever id is currently visible (preferring target when both
    # are), and the approach-settle gate refuses to advance the script
    # while ``active_marker_id != wait_attack_target_id`` — so the
    # drone holds at standoff on the sibling face until the box flips.
    # Both fields are cleared on the step transition out of WAIT_AND_ATTACK.
    wait_attack_target_id: Optional[int] = None
    wait_attack_sibling_id: Optional[int] = None

    settle_began_at: Optional[float] = None
    hold_began_at: Optional[float] = None
    search_began_at: Optional[float] = None
    search_yaw_swept_deg: float = 0.0
    # When SEARCH was entered after an APPROACH-class phase, this is
    # the monotonic timestamp at which the retreat-then-yaw transition
    # should switch to plain yaw-in-place. None disables the retreat
    # for this entry (e.g. SEARCH at script start).
    search_retreat_until: Optional[float] = None
    last_marker_seen_at: float = 0.0
    # Captured on entry to Phase.ALIGN so the forward channel can hold the
    # current radius (we don't want to close distance until heading is
    # centred). Cleared when leaving ALIGN.
    align_distance_m: Optional[float] = None
    # Altimeter height (m) captured the FIRST time a capture-APPROACH onto a
    # target box (ids target_marker_id_min..max) begins. A lost-box SEARCH
    # descends back to this before yawing, so a brief HEIGHT_ALIGN to a high /
    # oblique marker -- or a GO_HOME recovery anchor -- can't strand the drone
    # above the (low) box where it would never re-enter the search FOV. None
    # when the active target isn't a box or no telemetry was available.
    approach_start_height_m: Optional[float] = None

    # Mission-script execution state. Set by trigger() -- the controller
    # walks the script one step at a time; _advance_script consumes it.
    mission_script: List[Step] = field(default_factory=list)
    mission_step_idx: int = -1            # -1 = before first step
    current_step_kind: Optional[str] = None
    last_completed_step_kind: Optional[str] = None
    # Live mission-splice forensics (C2 airborne re-target, splice_script()).
    # splice_count steps up by 1 on each successful splice so flight_log.csv
    # pins the exact tick it happened; splice_note carries the latest splice's
    # detail (new step count + APPROACH target ids). Both flow through
    # snapshot() into the CSV next to note / abort_reason.
    splice_count: int = 0
    splice_note: str = ""
    # Per-step transient state used by IDLE / HEIGHT / DANCE / GOTO.
    idle_until: Optional[float] = None
    height_target_m: Optional[float] = None
    dance_until: Optional[float] = None
    dance_mode: Optional[str] = None
    dance_origin_xy_m: Optional[tuple[float, float]] = None
    dance_origin_height_m: Optional[float] = None
    # TO step: arena-frame target. ``goto_target_z_m`` is None when
    # the operator only specified x/y -- _step_goto then leaves the
    # ud channel at zero and the drone holds whatever altitude it's
    # currently at via the Anafi's onboard stabiliser.
    goto_target_x_m: Optional[float] = None
    goto_target_y_m: Optional[float] = None
    goto_target_z_m: Optional[float] = None
    # Optional arena-frame yaw target for the TO step (degrees, CW
    # from arena +y / front wall). None means "do not drive yaw" --
    # _step_goto leaves rc_yaw at 0 and the drone keeps its current
    # heading. Set either from an explicit numeric arg in the script
    # or by the best-view auto-yaw heuristic at TO step entry.
    goto_target_yaw_deg: Optional[float] = None
    # When set, _step_goto stays in GOTO until ``now >=
    # goto_hold_until`` and advances on the timer instead of on
    # settle. Used by HOOVER-after-TO so the drone station-keeps at
    # the TO target for the requested seconds, the analogue of
    # HOOVER-after-APPROACH (HOLD on the marker).
    goto_hold_until: Optional[float] = None
    # RC step (FB / UD / YAW): pin a single stick to the parsed
    # value for ``rc_step_until - now`` seconds. Others ride 0.
    # All zeros = effectively PAUSE; the parser never produces that.
    rc_step_lr: int = 0
    rc_step_fb: int = 0
    rc_step_ud: int = 0
    rc_step_yaw: int = 0
    rc_step_until: Optional[float] = None
    # FB_BRAKE world-position hint. When the script supplies world
    # coords (e.g. for slot face markers not present in arena_config),
    # the controller falls back to these when vision can't see the
    # marker AND arena.markers doesn't have it.
    fb_brake_world_x: Optional[float] = None
    fb_brake_world_y: Optional[float] = None
    # AUTO_ATTACK reactive state machine. substate ∈ {climb, go_target,
    # capture, go_home, land}. The target/home marker IDs + their known
    # arena positions come from the AUTO_ATTACK step.
    aa_substate: str = ""
    aa_substate_started: float = 0.0
    aa_target_marker: Optional[int] = None
    aa_target_xy: Optional[tuple] = None
    aa_home_marker: Optional[int] = None
    aa_home_xy: Optional[tuple] = None
    # Per-mission AUTO_ATTACK tuning (set from the step; default to the
    # module constants). aa_altitude_m = cruise/hover altitude;
    # aa_approach_speed = forward RC stick ceiling (cruise/approach
    # speed). Read by _aa_alt_hold / _aa_steer / _aa_vision_home.
    aa_altitude_m: float = AA_CRUISE_ALT_M
    aa_approach_speed: int = AA_FB_MAX
    # Last monotonic time the *current goal* marker was vision-live.
    # Used to keep approaching by vision across momentary dropouts
    # instead of prematurely capturing by world position.
    aa_last_vision_at: float = 0.0
    # Whether the capture brake has settled (drone stopped) — once true
    # the capture sub-state switches from braking to vision-hold hover.
    aa_capture_braked: bool = False
    # Capture "over-marker" maneuver: distance recorded when the drone
    # was in front of the target (so it can move forward by exactly
    # that much to end up over it), and the deadline for the forward move.
    aa_capture_distance_m: float = 0.0
    aa_over_forward_until: float = 0.0
    # Deadline for the post-drop dwell over the box (face home + acquire
    # the home marker) before flying back to the home zone.
    aa_dwell_until: float = 0.0
    # Two-phase RC step: once the drive timer expires we enter a
    # brake phase that holds zero sticks until the IMU reports near-
    # zero body-frame velocity (or a max-settle timeout fires).
    # Stops the drone before the NEXT step starts so commands like
    # FB +20 then FB -20 don't ride on top of each other's residual
    # momentum.
    rc_step_braking: bool = False
    rc_step_brake_started: Optional[float] = None
    # SCOUT step (slow 360° yaw spin). Drives the yaw stick at
    # SCOUT_YAW_STICK and tracks ``cur_yaw - prev_yaw`` (wrap-safe)
    # into ``scout_accumulated_deg`` until |accumulated| >= 360°.
    # Reuses the rc_step_braking flag for the post-drive brake.
    scout_started_at: Optional[float] = None
    scout_last_yaw_deg: Optional[float] = None
    scout_accumulated_deg: float = 0.0
    # AWAIT step: when set, _step_idle / _step_hold advance the
    # script as soon as this marker id appears in
    # state.visible_marker_ids (see vision_worker). Cleared on every
    # _apply_step_to_phase call so it doesn't leak across steps.
    await_marker_id: Optional[int] = None
    # All marker ids the detector reports this frame. Maintained by
    # vision_worker on every detect tick. Used by AWAIT for early
    # exit and by anything else that wants a "what's visible right
    # now" signal.
    visible_marker_ids: List[int] = field(default_factory=list)

    # Arena-frame world position (camera centre), populated by
    # vision_worker once an ArenaConfig is loaded. None until the
    # FIRST reference-marker sighting; after that we keep the last
    # known fix even when no marker is currently visible, so the
    # operator still sees a position estimate during marker loss.
    # Use ``world_position_age_s`` from the snapshot to tell live
    # from stale.
    world_position_m: Optional[tuple[float, float, float]] = None
    # Monotonic timestamp of the last fresh fix. 0.0 means "never
    # computed yet" -- snapshot exposes None as the age in that
    # case. Set every time vision_worker writes a non-None estimate.
    world_position_updated_at: float = 0.0
    # When True, ``vision_worker`` clears ``world_position_m`` AND
    # resets the position Kalman filter on its next iteration. The
    # controller raises this on entry to ``Phase.TAKEOFF`` so each
    # mission starts cold — otherwise a stale or wrong-anchor fix
    # from the previous mission (worst case: a 7+ m IPPE branch error
    # that latched into anchor) survives across the cycle and gates
    # every subsequent measurement. vision_worker clears the flag
    # after honouring it.
    reset_position_estimator: bool = False
    # Markers / methods / per-marker votes from the LAST fresh fix
    # (kept in sync with ``world_position_m``). They describe how the
    # currently-displayed position was computed, so they go stale
    # together. Cleared only by ``reset``.
    world_position_used_markers: List[int] = field(default_factory=list)
    world_position_pose_methods: List[str] = field(default_factory=list)
    world_position_per_marker: List[tuple[float, float, float]] = field(
        default_factory=list)
    # Position Kalman filter velocity output (arena frame, m/s). None
    # whenever ``cfg.enable_position_kalman`` is False or the filter
    # hasn't yet seen a position measurement to initialise. Logged in
    # the per-flight CSV alongside the position so post-flight
    # analysis can plot the KF velocity track without re-running the
    # offline replay.
    world_velocity_m_kf: Optional[tuple[float, float, float]] = None
    camera_restart_count: int = 0   # auto-restarts triggered by freeze detection
    world_position_confidence: float = 0.0   # base confidence from last estimate
    arena_validation_map: dict = field(default_factory=dict)  # marker validator map
    # solvePnP method for the controller's active TARGET marker
    # (whatever pose_holder.get() returned this tick). "" when the
    # target marker isn't currently in view.
    target_pose_method: str = ""

    # Drone yaw in arena frame (CW from +y / front wall) computed by
    # vision_worker whenever the active marker is visible alongside a
    # fresh world fix. ``arena_yaw_updated_at`` is the monotonic
    # timestamp of the last fresh value (0.0 = never), used by
    # _step_goto to gate driving on a fresh-enough yaw.
    arena_yaw_deg: Optional[float] = None
    arena_yaw_updated_at: float = 0.0

    abort_reason: str = ""
    note: str = ""                                            # informational

    # Enemy-drone threat response. Set when a ≥150 px blob is first seen
    # and cleared once the threat disappears. The suspended_* fields save
    # the phase + step index so we can resume exactly where we left off.
    threat_observe_since: Optional[float] = None    # monotonic when threat began
    threat_suspended_phase: Optional['Phase'] = None
    threat_suspended_step_idx: int = -1
    # Emergency re-arm: set by telemetry_worker when the drone lands
    # unexpectedly mid-mission. While True the controller sends only
    # RC(0,0,0,0) and waits. Cleared once a re-takeoff succeeds.
    emergency_rearm: bool = False

    # Two-stage telemetry-stall flags. ``telemetry_stalled`` is the
    # diagnostic level (age > detect_threshold) -- marks the CSV /
    # journal so we can quantify stalls without intervention. The
    # higher-threshold ``telemetry_rc_gated`` reflects ESCALATION:
    # _send_rc has actually zeroed its output because the link's been
    # stalled long enough (age > gate_threshold) that acting on the
    # frozen TelemetrySnapshot is unsafe. Either flag clears on the
    # first fresh sample.
    telemetry_stalled: bool = False
    telemetry_rc_gated: bool = False

    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, cfg: MissionConfig) -> None:
        """Clear per-mission fields so the same MissionState instance can
        be re-used for a fresh run. Telemetry is intentionally left
        untouched -- the telemetry_worker keeps it fresh across missions.
        """
        with self.lock:
            self.phase = Phase.INIT
            self.phase_started_at = 0.0
            self.started_at = 0.0
            self.last_pose = None
            self.smoothed = None
            self.last_rc = (0, 0, 0, 0)
            self.target_distance_m = cfg.target_distance_m
            self.target_relative_heading_deg = cfg.target_relative_heading_deg
            self.active_marker_id = cfg.target_marker_id
            self.hold_time_s = cfg.hold_time_s
            self.settle_began_at = None
            self.hold_began_at = None
            self.search_began_at = None
            self.search_yaw_swept_deg = 0.0
            self.search_retreat_until = None
            self.last_marker_seen_at = 0.0
            self.align_distance_m = None
            self.approach_start_height_m = None
            self.mission_script = []
            self.mission_step_idx = -1
            self.current_step_kind = None
            self.last_completed_step_kind = None
            self.idle_until = None
            self.height_target_m = None
            self.dance_until = None
            self.goto_target_x_m = None
            self.goto_target_y_m = None
            self.goto_target_z_m = None
            self.goto_target_yaw_deg = None
            self.goto_hold_until = None
            self.rc_step_lr = 0
            self.rc_step_fb = 0
            self.rc_step_ud = 0
            self.rc_step_yaw = 0
            self.rc_step_until = None
            self.rc_step_braking = False
            self.rc_step_brake_started = None
            self.scout_started_at = None
            self.scout_last_yaw_deg = None
            self.scout_accumulated_deg = 0.0
            self.await_marker_id = None
            self.visible_marker_ids = []
            self.dance_mode = None
            self.dance_origin_xy_m = None
            self.dance_origin_height_m = None
            self.world_position_m = None
            self.world_position_updated_at = 0.0
            self.world_position_used_markers = []
            self.world_position_pose_methods = []
            self.world_position_per_marker = []
            self.world_velocity_m_kf = None
            self.target_pose_method = ""
            self.arena_yaw_deg = None
            self.arena_yaw_updated_at = 0.0
            self.abort_reason = ""
            self.note = ""

    def snapshot(self) -> dict:
        with self.lock:
            d = yaw = hdg = None
            if self.smoothed is not None:
                d, yaw, hdg = self.smoothed
            tel = self.last_telemetry.raw if self.last_telemetry else {}
            # Mission script for the live runtime view: one canonicalised
            # text line per step plus the active index (-1 before the
            # first _advance_script). The UI swaps the textarea for a
            # highlighted list whenever this is non-empty.
            from . import mission_script as _ms
            script_lines = (_ms.format(self.mission_script).splitlines()
                             if self.mission_script else [])
            return {
                "phase": self.phase.value,
                "phase_age_s": time.monotonic() - self.phase_started_at,
                "uptime_s": time.monotonic() - self.started_at if self.started_at else 0.0,
                "distance_m": d,
                "yaw_to_marker_deg": yaw,
                "relative_heading_deg": hdg,
                "target_distance_m": self.target_distance_m,
                "target_relative_heading_deg": self.target_relative_heading_deg,
                "active_marker_id": self.active_marker_id,
                "hold_time_s": self.hold_time_s,
                "rc": {"lr": self.last_rc[0], "fb": self.last_rc[1],
                       "ud": self.last_rc[2], "yaw": self.last_rc[3]},
                "telemetry": tel,
                "telemetry_stalled": bool(self.telemetry_stalled),
                "telemetry_rc_gated": bool(self.telemetry_rc_gated),
                "host": _host_resource_dict(),
                "marker_seen_age_s": (time.monotonic() - self.last_marker_seen_at)
                                       if self.last_marker_seen_at else None,
                "mission_script": script_lines,
                "mission_step_idx": self.mission_step_idx,
                "current_step_kind": self.current_step_kind,
                "splice_count": self.splice_count,
                "splice_note": self.splice_note,
                "height_target_m": self.height_target_m,
                "world_position_m": (list(self.world_position_m)
                                     if self.world_position_m is not None
                                     else None),
                "world_position_age_s": (
                    (time.monotonic() - self.world_position_updated_at)
                    if self.world_position_updated_at > 0.0
                    else None),
                "visible_marker_ids": list(self.visible_marker_ids),
                "world_position_used_markers": list(
                    self.world_position_used_markers),
                "world_position_pose_methods": list(
                    self.world_position_pose_methods),
                "world_position_per_marker": [list(p) for p in
                                              self.world_position_per_marker],
                "world_velocity_m_kf": (list(self.world_velocity_m_kf)
                                        if self.world_velocity_m_kf is not None
                                        else None),
                "camera_restart_count": self.camera_restart_count,
                "arena_validation_map": dict(self.arena_validation_map),
                "world_position_confidence": round(
                    self.world_position_confidence
                    * max(0.0, 1.0 - (
                        (time.monotonic() - self.world_position_updated_at)
                        / 10.0
                        if self.world_position_updated_at > 0 else 10.0)),
                    3),
                "target_pose_method": self.target_pose_method,
                "arena_yaw_deg": self.arena_yaw_deg,
                "arena_yaw_age_s": (
                    (time.monotonic() - self.arena_yaw_updated_at)
                    if self.arena_yaw_updated_at > 0.0
                    else None),
                "goto_target_m": (
                    [self.goto_target_x_m, self.goto_target_y_m,
                     self.goto_target_z_m]
                    if self.goto_target_x_m is not None
                    else None),
                "abort_reason": self.abort_reason,
                "note": self.note,
            }


# ---------------------------------------------------------------------------
# The controller itself
# ---------------------------------------------------------------------------

class MissionController:
    """Runs the state machine in a background thread."""

    def __init__(self, api: DroneApi, cfg: MissionConfig,
                 state: MissionState,
                 frame_pose_provider,           # callable -> Optional[MarkerPose]
                 telemetry_provider,            # callable -> TelemetrySnapshot
                 on_phase_change=None,
                 arena_provider=None,           # callable -> Optional[ArenaConfig]
                 drone_threat_provider=None,    # callable -> int (max blob width px)
                 telemetry_age_provider=None,   # callable -> Optional[float] (s since last fresh tel)
                 ):
        self.api = api
        self.cfg = cfg
        self.state = state
        self.get_pose = frame_pose_provider
        self.get_tel = telemetry_provider
        # Optional callable returning seconds since the last fresh
        # telemetry sample (mission.py wires this to
        # tel_holder.age_s). When provided AND
        # cfg.enable_telemetry_stall_detector is True AND the returned
        # age exceeds cfg.telemetry_stall_gate_threshold_s, _send_rc
        # zeros the output so we don't keep slamming the drone with
        # stale-state-derived RC during a wifi / Olympe stream stall.
        self.get_tel_age = telemetry_age_provider
        # Stall tracking for one-shot logs + state.snapshot exposure.
        # Two latched levels: _stall_detect_active = "marked in CSV"
        # (past cfg.telemetry_stall_detect_threshold_s), _stall_gate_active
        # = "RC actually zeroed" (past cfg.telemetry_stall_gate_threshold_s).
        # Each transition (enter/leave) logs exactly once so journal volume
        # stays sane on a flaky link.
        self._stall_detect_active: bool = False
        self._stall_gate_active: bool = False
        self.on_phase_change = on_phase_change
        # Lets the TO step's auto-yaw heuristic enumerate reference
        # markers and score candidate yaws. Optional; without an
        # arena_provider TO falls back to "no yaw drive" for auto.
        self.get_arena = arena_provider
        self.get_drone_threat = drone_threat_provider   # callable -> (int px, float dx) or None

        # PD/PID controllers. ``ki`` defaults to 0 in cfg, so today's
        # behaviour (pure PD) is preserved unless the operator bumps
        # the I gains via /tune.
        self.pd_yaw = PIDController(cfg.yaw_kp, cfg.yaw_kd, cfg.yaw_rc_max,
                                     ki=cfg.yaw_ki, i_clip=cfg.yaw_i_clip)
        self.pd_fwd = PIDController(cfg.fwd_kp, cfg.fwd_kd, cfg.fwd_rc_max,
                                     ki=cfg.fwd_ki, i_clip=cfg.fwd_i_clip)
        self.pd_lat = PIDController(cfg.lat_kp, cfg.lat_kd, cfg.lat_rc_max,
                                     ki=cfg.lat_ki, i_clip=cfg.lat_i_clip)
        self.pd_height = PIDController(cfg.height_kp, cfg.height_kd, cfg.ud_rc_max,
                                        ki=cfg.height_ki, i_clip=cfg.height_i_clip)

        self.smoother = PoseSmoother(cfg.pose_smoothing_alpha, cfg.pose_max_age_s)
        # Dead-reckoning position estimator. Used by the DANCE script
        # step to bound the routine within dance_radius_m of its
        # entry position. Runs continuously from controller
        # construction; never reset.
        self.position = PositionEstimator()

        self._stop = threading.Event()
        self._go = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Seed the runtime targets from cfg so vision_worker (which
        # may run before the first controller.start()) reads a
        # sensible state.active_marker_id.
        with self.state.lock:
            self.state.active_marker_id = self.cfg.target_marker_id
            self.state.hold_time_s = self.cfg.hold_time_s
            self.state.target_distance_m = self.cfg.target_distance_m
            self.state.target_distance_tol_m = None
            self.state.approach_positioning = False
            self.state.arrive_hdg_tol_deg = None
            self.state.arrive_yaw_tol_deg = None
            self.state.target_relative_heading_deg = (
                self.cfg.target_relative_heading_deg)

        # Initialise all per-mission scratch state. reset() is only called
        # between missions, so __init__ must also initialise these so the
        # very first start() finds a clean controller without AttributeErrors.
        self._land_requested = False
        self._descent_started_at = None
        self._search_start_yaw = None
        self._search_swept = 0.0
        self._search_prev_yaw = None
        # Two-stage SEARCH protocol state (see _step_search):
        #   0 = stationary check at zoom 2.0x for cfg.search_zoom_check_s
        #   1 = yaw spin at zoom 1.25x for one full revolution
        # After stage 1 without acquisition -> _recover_via_central_marker.
        self._search_stage: int = 0
        self._search_stage_started: float = 0.0
        self._search_visited_markers: set = set()
        self._goto_on_settle_phase = None
        self._goto_on_settle_restart_mission = False
        # Set by the marker-lost centre-marker recovery: restart the mission
        # when the recovery APPROACH settles (vision-only, no TO).
        # See _recover_via_central_marker.
        self._approach_on_settle_restart_mission = False
        # Bounded recovery budget: consecutive GO_HOME re-anchor hops without
        # re-acquiring the TARGET. Reset to 0 on real target re-acquisition.
        # Once it hits _MAX_RECOVERY_HOPS we stop the active-flight recovery and
        # fall through to an in-place spin (low power) waiting for the C2 to
        # re-task -- so a persistently-unrecoverable target can't loop forever.
        self._recovery_count = 0
        self._goto_last_wp = None
        self._goto_jump_until: float = 0.0
        self._goto_frozen_yaw = None
        self._goto_scan_state: str = 'fly'
        self._goto_scan_until: float = 0.0
        self._approach_start_d = None
        self._approach_lat_unlocked: bool = False
        self._approach_mid_ha_active: bool = False
        self._approach_mid_ha_settled_at: Optional[float] = None
        self._approach_mid_ha_started_at: Optional[float] = None
        # Track last zoom applied so we skip redundant camera_config_set calls.
        # Each set_zoom_target triggers an Olympe _media_removed stream restart
        # (~2s freeze), so we only call when the level actually changes.
        self._camera_zoom: float = 1.0

    def _apply_zoom(self, zoom: float) -> None:
        """Set digital zoom only when it differs from the last applied level."""
        if abs(self._camera_zoom - zoom) < 0.01:
            return
        try:
            self.api.camera_config_set(zoom=zoom)
            self._camera_zoom = zoom
        except Exception as e:
            print(f"[ctrl] zoom set failed: {e}")

    def _apply_stream_settings(self) -> None:
        """Apply stream mode and recording framerate from cfg once at mission start."""
        fps_str = f"fps_{self.cfg.video_framerate_fps}"
        try:
            result = self.api.camera_config_set(
                stream_mode={"mode": self.cfg.video_stream_mode},
                recording={"mode": "standard", "resolution": "res_1080p",
                           "framerate": fps_str, "hyperlapse": "ratio_15"},
            )
            print(f"[ctrl] stream settings: mode={self.cfg.video_stream_mode}"
                  f" fps={self.cfg.video_framerate_fps} → {result}")
        except Exception as e:
            print(f"[ctrl] stream settings failed: {e}")

    def apply_config_changes(self) -> None:
        """Re-sync per-instance state that was copied out of cfg at
        construction. Call after the cfg dataclass has been mutated
        externally (e.g., from the live-tuning UI) so the running
        controller picks up the new gains / clips immediately."""
        cfg = self.cfg
        self.pd_yaw.kp = float(cfg.yaw_kp)
        self.pd_yaw.kd = float(cfg.yaw_kd)
        self.pd_yaw.ki = float(cfg.yaw_ki)
        self.pd_yaw.i_clip = float(cfg.yaw_i_clip)
        self.pd_yaw.out_clip = float(cfg.yaw_rc_max)
        self.pd_fwd.kp = float(cfg.fwd_kp)
        self.pd_fwd.kd = float(cfg.fwd_kd)
        self.pd_fwd.ki = float(cfg.fwd_ki)
        self.pd_fwd.i_clip = float(cfg.fwd_i_clip)
        self.pd_fwd.out_clip = float(cfg.fwd_rc_max)
        self.pd_lat.kp = float(cfg.lat_kp)
        self.pd_lat.kd = float(cfg.lat_kd)
        self.pd_lat.ki = float(cfg.lat_ki)
        self.pd_lat.i_clip = float(cfg.lat_i_clip)
        self.pd_lat.out_clip = float(cfg.lat_rc_max)
        self.pd_height.kp = float(cfg.height_kp)
        self.pd_height.kd = float(cfg.height_kd)
        self.pd_height.ki = float(cfg.height_ki)
        self.pd_height.i_clip = float(cfg.height_i_clip)
        self.pd_height.out_clip = float(cfg.ud_rc_max)
        self.smoother.alpha = float(cfg.pose_smoothing_alpha)
        self.smoother.max_age_s = float(cfg.pose_max_age_s)

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        """Spin up the control thread. The thread parks in INIT until
        :meth:`trigger` is called -- this lets the operator review the
        UI / camera feed before committing to takeoff."""
        if self._thread and self._thread.is_alive():
            return
        with self.state.lock:
            self.state.started_at = time.monotonic()
            self.state.target_distance_m = self.cfg.target_distance_m
            self.state.target_distance_tol_m = None
            self.state.approach_positioning = False
            self.state.arrive_hdg_tol_deg = None
            self.state.arrive_yaw_tol_deg = None
            self.state.target_relative_heading_deg = self.cfg.target_relative_heading_deg
            self.state.active_marker_id = self.cfg.target_marker_id
            self.state.hold_time_s = self.cfg.hold_time_s
            self.state.note = "waiting for operator to start mission"
        self._stop.clear()
        self._go.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mission-ctrl")
        self._thread.start()

    def reset(self) -> None:
        """Prepare for another start() after the previous run has
        terminated. Clears all per-mission internal state so a fresh
        Phase.INIT-and-park cycle starts clean.

        Caller must ensure the previous _run thread has exited (or
        is_running() is False) before calling.
        """
        if self._thread and self._thread.is_alive():
            raise RuntimeError("controller still running -- call stop() first")
        self._stop.clear()
        self._go.clear()
        self.pd_yaw.reset()
        self.pd_fwd.reset()
        self.pd_lat.reset()
        self.pd_height.reset()
        self.smoother.reset()
        # Per-phase scratch state (set lazily in the relevant _step_*).
        self._land_requested = False
        self._descent_started_at = None
        self._search_start_yaw = None
        self._search_swept = 0.0
        self._search_prev_yaw = None
        # Two-stage SEARCH protocol state (see _step_search):
        #   0 = stationary check at zoom 2.0x for cfg.search_zoom_check_s
        #   1 = yaw spin at zoom 1.25x for one full revolution
        self._search_stage: int = 0
        self._search_stage_started: float = 0.0
        self._search_visited_markers: set = set()  # wall-marker IDs already navigated to
        # Lost-box "descend back to approach-start height" state.
        # _box_height_marker remembers which box id approach_start_height_m was
        # captured for, so a recovery-driven mission RESTART (which re-runs the
        # APPROACH step from idx -1) cannot overwrite the original low altitude
        # with the climbed one. _search_descend_active gates the descend that
        # runs before the yaw sweep on each box SEARCH entry.
        self._box_height_marker: Optional[int] = None
        self._search_descend_active: bool = False
        self._search_descend_deadline: float = 0.0
        self._goto_on_settle_phase: Optional[Phase] = None  # override post-GOTO transition
        self._goto_on_settle_restart_mission: bool = False  # restart mission from step 0 on GOTO settle
        self._approach_on_settle_restart_mission: bool = False  # restart mission on recovery-GO_HOME settle
        self._recovery_count: int = 0                 # bounded marker-lost recovery hops
        self._goto_last_wp: Optional[tuple] = None    # last accepted world pos in GOTO
        self._goto_jump_until: float = 0.0            # stop+reorient until this monotonic time
        self._goto_frozen_yaw: Optional[float] = None  # yaw_arena frozen at GOTO entry
        # Scan-pause state machine (periodic 360° scan during GOTO)
        # States: 'fly' | 'scan' | 'realign'
        self._goto_scan_state: str = 'fly'
        self._goto_scan_until: float = 0.0   # end of current scan/realign phase
        self._approach_start_d: Optional[float] = None
        self._approach_lat_unlocked: bool = False
        self._approach_mid_ha_active: bool = False
        self._approach_mid_ha_settled_at: Optional[float] = None
        self._approach_mid_ha_started_at: Optional[float] = None
        self.state.reset(self.cfg)

    def set_script(self, steps: List[Step]) -> None:
        """Install the parsed mission script that the controller will
        walk on the next trigger(). Caller must have parsed and
        validated the steps; the controller does no further
        validation here."""
        with self.state.lock:
            self.state.mission_script = list(steps)
            self.state.mission_step_idx = -1
            self.state.current_step_kind = None
            self.state.last_completed_step_kind = None
        # A freshly-loaded mission (a NEW C2 task) gets a fresh marker-lost
        # recovery budget. The recovery's own restart re-runs the SAME script
        # via _advance_script (NOT set_script), so it does not reset this — the
        # budget only resets on a genuinely new task or a real re-acquisition.
        self._recovery_count = 0
        self._approach_on_settle_restart_mission = False
        # A new task re-snapshots the approach-start height per box. The
        # recovery restart re-runs via _advance_script (NOT set_script), so the
        # original (low) snapshot survives a restart -- the whole point of the
        # per-marker guard below.
        self._box_height_marker = None
        with self.state.lock:
            self.state.approach_start_height_m = None

    def splice_script(self, steps: List[Step]) -> tuple[bool, str]:
        """Live-replace the REMAINING mission steps WITHOUT landing.

        Unlike set_script() (which only takes effect on the next trigger()
        from INIT, i.e. landed), this mutates the running script in place:
        it KEEPS the currently-executing step and replaces everything after
        it, so the next _advance_script() walks straight into ``steps``. Used
        by the C2 to re-target an airborne attacker (new enemy boxes) without
        the stop -> land -> relaunch cost.

        The step list + index are updated atomically under state.lock, which
        also serialises against _advance_script() (it reads the list + index
        under the same lock). We always anchor the kept prefix to the *current*
        mission_step_idx, so whichever of splice/advance runs first the
        invariant "keep the current step, replace the tail" holds.

        Returns (ok, message). Rejects when the drone is not airborne, has no
        active script, or the splice itself contains TAKEOFF (already flying).
        On any rejection the C2 falls back to the stop->relaunch retask path.
        """
        if not steps:
            return False, "empty splice"
        if any(s.kind == "TAKEOFF" for s in steps):
            return False, "splice must not contain TAKEOFF (already airborne)"
        with self.state.lock:
            phase = self.state.phase
            if phase in (Phase.INIT, Phase.LAND, Phase.DONE, Phase.ABORT):
                return False, f"not airborne (phase={phase.value})"
            if not self.state.mission_script:
                return False, "no active mission to splice"
            # Keep the current step (mission_step_idx) executing; replace the
            # tail. idx<0 can't happen airborne (TAKEOFF is step 0 and the
            # index has advanced) but clamp defensively so kept is never empty.
            idx = self.state.mission_step_idx
            if idx < 0:
                idx = 0
            kept = list(self.state.mission_script[:idx + 1])
            self.state.mission_script = kept + list(steps)
            # mission_step_idx / current_step_kind are deliberately left as-is:
            # the current step finishes, then _advance_script() loads idx+1 =
            # the first spliced step.
            n_new = len(steps)
            # Flight-log forensics: step the counter + record what we spliced
            # (the distinct APPROACH target ids) so flight_log.csv pins the
            # tick and the new targets. See MissionState.splice_count.
            targets = sorted({s.marker_id for s in steps
                              if s.kind == "APPROACH" and s.marker_id is not None})
            self.state.splice_count += 1
            note = (f"splice#{self.state.splice_count}: +{n_new} steps after "
                    f"idx {idx}; targets {targets}")
            self.state.splice_note = note
        print(f"[ctrl] mission splice: {note}")
        return True, note

    def trigger(self) -> bool:
        """Release the run loop so it proceeds into the script's first
        step. If no script was set via set_script(), the controller
        falls back to the hardcoded default. Returns True if this
        call actually started the mission, False if it was already
        started or the controller is no longer in INIT."""
        if self._go.is_set():
            return False
        with self.state.lock:
            if self.state.phase != Phase.INIT:
                return False
            # Fall back to the hardcoded default script if the caller
            # didn't install one. Keeps the legacy "press Start with
            # nothing else configured" path working.
            if not self.state.mission_script:
                self.state.mission_script = parse_script(
                    HARDCODED_DEFAULT_SCRIPT, defaults_from_cfg(self.cfg))
                self.state.mission_step_idx = -1
                self.state.current_step_kind = None
                self.state.last_completed_step_kind = None
        self._go.set()
        return True

    def is_armed(self) -> bool:
        """True if the thread is alive and still parked in INIT."""
        if not self.is_running():
            return False
        with self.state.lock:
            return self.state.phase == Phase.INIT and not self._go.is_set()

    def stop(self, reason: str = "external stop") -> None:
        with self.state.lock:
            if self.state.phase not in (Phase.DONE, Phase.ABORT):
                self.state.abort_reason = reason
        self._stop.set()
        # Make sure a thread parked in INIT can wake up and exit cleanly.
        self._go.set()
        if self._thread:
            # The control thread runs _safe_shutdown on its way out if the
            # drone is potentially airborne -- that's an HTTP rc_zero +
            # land + ~15 s telemetry poll, so allow generous time before
            # giving up. Mission's main wait loop also watches the stop
            # event so we don't block the user forever if this overruns.
            self._thread.join(timeout=25.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----------------------------------------------------------------- phase
    def _set_phase(self, phase: Phase, note: str = "") -> None:
        with self.state.lock:
            old = self.state.phase
            self.state.phase = phase
            self.state.phase_started_at = time.monotonic()
            self.state.note = note
            self.state.settle_began_at = None
            self.state.hold_began_at = None
            self.state.search_began_at = None if phase != Phase.SEARCH else time.monotonic()
            self.state.search_yaw_swept_deg = 0.0 if phase == Phase.SEARCH else self.state.search_yaw_swept_deg
        # Reset the controller's internal sweep tracker on every entry to
        # SEARCH. Otherwise the second SEARCH after a successful APPROACH
        # would re-use the previous sweep counter and hit search_total_deg
        # almost immediately -- the drone would land instead of actually
        # looking around again.
        if phase == Phase.APPROACH:
            self._approach_start_d = None
            self._approach_lat_unlocked = False
            self._approach_mid_ha_active = False
            self._approach_mid_ha_settled_at = None
            self._approach_mid_ha_started_at = None
        if phase == Phase.GOTO:
            self._goto_last_wp = None
            self._goto_jump_until = 0.0
            self._goto_scan_state = 'fly'
            self._goto_scan_until = 0.0
            self._goto_frozen_yaw = None  # will be set on first valid yaw in _step_goto
        if phase == Phase.SEARCH:
            self._search_start_yaw = None
            self._search_swept = 0.0
            self._search_prev_yaw = None
            # Retreat-then-search: if we entered SEARCH from a phase
            # where the drone was actively tracking the marker (and
            # therefore had a real reason to look "behind" itself),
            # back up a bit before yawing in place. Skips the retreat
            # when entering SEARCH at the start of the script (from
            # TAKEOFF / INIT) — there's no prior viewpoint to retreat
            # toward.
            tracking_phases = (
                Phase.APPROACH, Phase.HOLD, Phase.HEIGHT_ALIGN, Phase.ALIGN,
            )
            retreat_s = int(getattr(self.cfg, "search_retreat_s", 0))
            if old in tracking_phases and retreat_s > 0:
                with self.state.lock:
                    self.state.search_retreat_until = (
                        time.monotonic() + retreat_s
                    )
            else:
                with self.state.lock:
                    self.state.search_retreat_until = None
        # Latch the HOLD timer on entry, not on the first marker-visible
        # tick. Otherwise a marker loss right at the start of HOLD never
        # starts the timer, and the phase can run forever (eventually
        # escalating to SEARCH). _step_hold always reads from this.
        if phase == Phase.HOLD:
            with self.state.lock:
                self.state.hold_began_at = time.monotonic()
        # Reset the position estimator on every TAKEOFF so a stale or
        # wrong-anchor world_position_m from the previous mission
        # (worst case: a 7+ m IPPE branch error that latched into
        # anchor) doesn't poison the new mission's branch selection.
        # vision_worker clears the flag after honouring it.
        if phase == Phase.TAKEOFF:
            with self.state.lock:
                self.state.reset_position_estimator = True
        # LAND can be re-entered (script with multiple LAND steps, or
        # LAND followed by TAKEOFF). Reset the per-LAND scratch state
        # so the second entry actually issues api.land() instead of
        # skipping it because _land_requested is still True from the
        # first entry.
        if phase == Phase.LAND:
            self._land_requested = False
            self._descent_started_at = None
        # Capture / release ALIGN's hold-radius setpoint at the phase
        # boundary so _step_align doesn't have to discover it itself.
        if phase == Phase.ALIGN:
            with self.state.lock:
                meas = self.state.smoothed
                tgt = self.state.target_distance_m
            self.state.align_distance_m = (meas[0] if meas is not None
                                           else tgt)
        elif old == Phase.ALIGN:
            self.state.align_distance_m = None
        # Reset PD integrators on phase change
        self.pd_yaw.reset(); self.pd_fwd.reset(); self.pd_lat.reset(); self.pd_height.reset()
        if self.on_phase_change:
            try:
                self.on_phase_change(old, phase, note)
            except Exception as e:
                print(f"[ctrl] on_phase_change error: {e}")
        print(f"[ctrl] phase {old.value} -> {phase.value} ({note})")

    # ------------------------------------------------------- input refresh
    def _refresh_inputs(self, now: float) -> Optional[TelemetrySnapshot]:
        """Pull the latest telemetry + marker pose, push them into
        ``MissionState`` and the pose smoother, and return the telemetry
        for callers that want it.

        Called every control tick, AND from the pre-flight wait loop so
        the UI panel shows live data before the operator presses Start.
        """
        tel = self.get_tel()
        if tel is not None:
            with self.state.lock:
                self.state.last_telemetry = tel
            # Feed the dead-reckoning position estimator (used by
            # DANCE for radius-bounded routines).
            try:
                vgx = float(tel.raw.get("vgx") or 0.0)
                vgy = float(tel.raw.get("vgy") or 0.0)
                self.position.update(vgx, vgy, now)
            except (TypeError, ValueError):
                pass
        pose = self.get_pose()
        self.smoother.update(pose, now)
        if pose is not None:
            with self.state.lock:
                self.state.last_pose = pose
                self.state.last_marker_seen_at = now
        with self.state.lock:
            self.state.smoothed = self.smoother.get(now)
        return tel

    # -------------------------------------------------------------- main loop
    def _run(self) -> None:
        cfg = self.cfg
        period = 1.0 / max(1.0, cfg.control_rate_hz)

        # Wait for the operator to press "Start" in the UI (or trigger() to
        # be called programmatically). The thread stays in INIT phase until
        # then, so the UI / camera feed are usable for pre-flight review.
        # Keep refreshing state so the UI status panel is live during the
        # wait (otherwise distance / yaw / battery all show "—").
        #
        # Critical: stop() also sets _go (to unblock a parked thread on
        # shutdown). So after the wait we MUST re-check _stop before doing
        # anything irreversible -- otherwise Ctrl-C in pre-flight would
        # take off the drone on the way out the door.
        while not self._go.is_set() and not self._stop.is_set():
            now = time.monotonic()
            tel = self._refresh_inputs(now)
            # Compute what the controller WOULD command (for the UI), but
            # don't send anything to the drone. With props off, this lets
            # the operator verify yaw/fwd/lat sign and magnitude by walking
            # the marker around.
            self._step_preflight_dryrun(tel, now)
            self._go.wait(timeout=period)
        if self._stop.is_set():
            print("[ctrl] stop received before takeoff -- exiting cleanly")
            return

        self._apply_stream_settings()

        # Hand off to the mission script. The first step (typically
        # TAKEOFF) is loaded here; _apply_step_to_phase handles the
        # api.takeoff() call so a script that opens with a non-TAKEOFF
        # step (e.g., DANCE while already airborne, in some testing
        # scenario) doesn't unconditionally request takeoff.
        self._advance_script("mission start")

        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(period, next_tick - now))
                continue
            next_tick = now + period

            try:
                # Emergency rearm: drone landed unexpectedly. Hold all RC
                # until telemetry_worker issues re-takeoff and clears flag.
                # Safety watchdog: if flag is stuck while drone is flying,
                # clear it automatically after 3 s so the mission resumes.
                with self.state.lock:
                    rearm = self.state.emergency_rearm
                if rearm:
                    tel_now = self._refresh_inputs(now)
                    flying = (tel_now is not None and tel_now.flying)
                    if flying:
                        if not hasattr(self, '_rearm_flying_since'):
                            self._rearm_flying_since = now
                        elif now - self._rearm_flying_since > 3.0:
                            print("[ctrl] emergency_rearm stuck while flying — "
                                  "force-clearing")
                            with self.state.lock:
                                self.state.emergency_rearm = False
                            self._rearm_flying_since = None
                    else:
                        self._rearm_flying_since = None
                    self._send_rc(0, 0, 0, 0)
                    continue
                self._rearm_flying_since = None

                tel = self._refresh_inputs(now)

                # ── Enemy-drone threat check ─────────────────────────────────
                self._check_drone_threat(now)

                # WAIT_AND_ATTACK target/sibling swap. Runs BEFORE the
                # phase dispatch so the per-phase code sees the freshly-
                # routed active_marker_id on the same tick.
                self._wait_attack_pre_tick()

                # Dispatch on phase ------------------------------------------
                phase = self.state.phase
                if phase == Phase.TAKEOFF:
                    self._step_takeoff(tel, now)
                elif phase == Phase.SEARCH:
                    self._step_search(now)
                elif phase == Phase.ALIGN:
                    self._step_align(tel, now)
                elif phase == Phase.HEIGHT_ALIGN:
                    self._step_height_align(tel, now)
                elif phase == Phase.APPROACH:
                    self._step_approach(tel, now)
                elif phase == Phase.HOLD:
                    self._step_hold(tel, now)
                elif phase == Phase.IDLE:
                    self._step_idle(tel, now)
                elif phase == Phase.HEIGHT:
                    self._step_height(tel, now)
                elif phase == Phase.GOTO:
                    self._step_goto(tel, now)
                elif phase == Phase.DANCE:
                    self._step_dance(tel, now)
                elif phase == Phase.RC:
                    self._step_rc(tel, now)
                elif phase == Phase.FB_BRAKE:
                    self._step_fb_brake(tel, now)
                elif phase == Phase.AUTO_ATTACK:
                    self._step_auto_attack(tel, now)
                elif phase == Phase.SCOUT:
                    self._step_scout(tel, now)
                elif phase == Phase.ROTATE:
                    # YAW_IMU step runs synchronously in
                    # _apply_step_to_phase and calls _advance_script
                    # before the controller ever ticks this phase.
                    # The branch is here for completeness so an
                    # unexpected stuck-in-ROTATE state can't pin the
                    # loop; just zero the sticks and idle until the
                    # next _advance_script lands.
                    self._send_rc(0, 0, 0, 0)
                elif phase == Phase.MOVE_IMU:
                    # FB/LR/UD_IMU steps also run synchronously in
                    # _apply_step_to_phase (api.move blocks on
                    # moveBy.wait). Same safety-net behaviour as
                    # Phase.ROTATE.
                    self._send_rc(0, 0, 0, 0)
                elif phase == Phase.THREAT_OBSERVE:
                    self._step_threat_observe(tel, now)
                elif phase == Phase.THREAT_EVADE:
                    self._step_threat_evade(tel, now)
                elif phase == Phase.LAND:
                    self._step_land(tel, now)
                elif phase in (Phase.DONE, Phase.ABORT):
                    return
            except DroneApiError as e:
                # Drone disconnected / refused command -- don't crash the
                # mission thread, just log and try again next tick.
                print(f"[ctrl] api error: {e}")
            except Exception as e:
                print(f"[ctrl] unexpected error: {e}")

        # Main loop exited because of _stop. If we never made it past
        # takeoff, or already reached DONE / ABORT, there's nothing to do.
        # Otherwise the drone is potentially airborne and we MUST land it
        # before this thread dies (operator pressed Ctrl-C; the Anafi
        # would otherwise just hover indefinitely waiting for commands).
        with self.state.lock:
            phase = self.state.phase
        if phase not in (Phase.DONE, Phase.ABORT, Phase.INIT):
            self._safe_shutdown(self.state.abort_reason or "operator stop")

    # ------------------------------------------------------- safe shutdown
    def _safe_shutdown(self, reason: str) -> None:
        """Best-effort: zero RC, request land, wait briefly for the drone
        to be on the ground. Called from ``_run`` when the main loop
        exits while the drone is (or might be) airborne -- typically an
        operator Ctrl-C mid-flight.

        Synchronous and blocking; ``stop()``'s join timeout is sized to
        accommodate the rc_zero (2 s) + land (20 s) + telemetry poll (up
        to 15 s) worst case. Each HTTP call has its own timeout, so we
        never block forever.
        """
        print(f"[ctrl] safe shutdown ({reason}): zeroing RC and requesting land")
        with self.state.lock:
            self.state.note = f"safe shutdown: {reason}"
        try:
            self.api.rc_zero()
        except DroneApiError as e:
            print(f"[ctrl] safe shutdown: rc_zero failed: {e}")
        try:
            self.api.land()
        except DroneApiError as e:
            print(f"[ctrl] safe shutdown: land() failed: {e}")
            self._set_phase(Phase.ABORT, f"safe shutdown failed: {e}")
            return
        # Wait for the drone to report flying=False.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                tel = self.api.telemetry()
                if not tel.flying:
                    print("[ctrl] safe shutdown: drone landed")
                    self._set_phase(Phase.DONE,
                                    f"landed via safe shutdown ({reason})")
                    return
            except DroneApiError:
                pass
            time.sleep(0.5)
        print("[ctrl] safe shutdown: timed out waiting for flying=False")
        self._set_phase(Phase.ABORT, f"safe shutdown timed out: {reason}")

    # --------------------------------------------------------------- takeoff
    def _step_takeoff(self, tel: Optional[TelemetrySnapshot], now: float) -> None:
        cfg = self.cfg
        # Sub-stage 1: wait for the airframe to leave the ground.
        if not (tel and tel.flying):
            self._send_rc(0, 0, 0, 0)
            if now - self.state.phase_started_at > 15.0:
                self._abort("timed out waiting for flying=True after takeoff")
            return
        # Sub-stage 2: climb to default_height_m before searching. Anafi's
        # takeoff command parks the drone at ~1 m; default_height_m is
        # usually a bit higher and we want a known starting altitude
        # for the rest of the mission.
        try:
            height_cm = float(tel.raw.get("height_cm")) if tel.raw.get("height_cm") is not None else None
        except (TypeError, ValueError):
            height_cm = None
        if height_cm is None:
            # Without height telemetry we can't drive a climb; advance
            # to the next script step anyway so the rest of the flight
            # still works.
            self._advance_script("airborne (no height telemetry)")
            return
        drone_h = height_cm / 100.0
        e_h = cfg.default_height_m - drone_h
        if abs(e_h) < cfg.height_deadband_m:
            self._advance_script(f"takeoff complete (h={drone_h:.2f}m)")
            return
        u_ud = self.pd_height.step(e_h, now)
        self._send_rc(0, 0, int(u_ud), 0)
        # Hard timeout for the whole TAKEOFF phase (climb included).
        if now - self.state.phase_started_at > 30.0:
            self._abort("timed out climbing to default height")

    # ---------------------------------------------------------------- search
    # Minimum yaw RC stick value for the escalating search (≈5°/s at
    # typical indoor Anafi yaw-rate settings). Below this the drone barely
    # rotates and detection becomes unreliable — we stop escalating.
    _SEARCH_MIN_YAW_RC: int = 3
    # Maximum digital zoom (Anafi hardware cap).
    _SEARCH_MAX_ZOOM: float = 3.0
    # Marker-lost recovery (NO absolute TO): approach the most CENTRAL marker on
    # whichever WALL is currently in view (up to four candidates, one per wall,
    # all computed from the live arena at runtime -- never a hard-coded id),
    # with a standoff computed so a head-on approach lands the drone
    # approximately at the arena origin (0,0), then restart the mission. The
    # standoff sits in the arena interior in FRONT of the centre marker, so the
    # recovery never backs the drone toward a side/back net. Standoff bounds are
    # derived from the arena's own width/depth (see _recovery_candidates), not
    # fixed magic numbers.
    _RECOVERY_TOL_M: float = 1.0
    # Cap on consecutive recover->restart hops without re-acquiring the target.
    # Past this we stop the active-flight recovery and spin in place waiting for
    # the C2 to re-task, so an unrecoverable target can't loop (and burn the
    # battery) forever.
    _MAX_RECOVERY_HOPS: int = 3

    def _recovery_candidates(self) -> list[tuple[int, float]]:
        """Up to four marker-lost recovery targets -- the most CENTRAL marker on
        EACH wall -- computed from the LIVE arena (no hard-coded ids). Returns a
        list of ``(marker_id, standoff_m)``; the recovery approaches whichever is
        in view, so a single unseen marker can never strand the drone.

        Per wall: the marker closest to the origin in the horizontal plane (i.e.
        nearest the wall's centre, where its inward normal passes through the
        origin), preferring the LOWER of a high/low pair (easier to approach at
        flight altitude). ``cfg.recovery_marker_id``, if set, forces a single
        explicit candidate instead.

        Standoff: the marker's horizontal distance to the origin. Arena markers
        sit on the inside wall face with their normal pointing toward the origin
        (see arena.py), so a head-on approach at this standoff lands the drone
        approximately at (0,0). Bounded by the arena's OWN geometry -- a
        wall-clearance floor (10% of the smaller span) and the arena
        half-diagonal ceiling -- so the value scales with the arena instead of
        fixed magic numbers."""
        arena = self.get_arena() if callable(self.get_arena) else None
        if arena is None or not arena.markers:
            return []

        width = float(getattr(arena, "width_m", 0.0) or 0.0)
        depth = float(getattr(arena, "depth_m", 0.0) or 0.0)
        clearance = 0.1 * min(width, depth) if (width > 0 and depth > 0) else 0.0
        ceiling = math.hypot(width, depth) / 2.0 if (width > 0 or depth > 0) \
            else float("inf")

        def _bound(dist0: float) -> float:
            return max(clearance, min(dist0, ceiling))

        def _key(m) -> Optional[tuple[float, float]]:
            # (horizontal distance to origin, height) -- most central, then
            # lowest. None if the marker has no usable position.
            try:
                return (math.hypot(float(m.position_m[0]),
                                   float(m.position_m[1])),
                        float(m.position_m[2]))
            except (TypeError, IndexError, ValueError):
                return None

        override = getattr(self.cfg, "recovery_marker_id", None)
        if override is not None and int(override) in arena.markers:
            k = _key(arena.markers[int(override)])
            if k is not None:
                return [(int(override), _bound(k[0]))]

        # Most-central marker per wall -> at most one candidate per wall.
        best_per_wall: dict[str, tuple[tuple[float, float], int]] = {}
        for mid, m in arena.markers.items():
            k = _key(m)
            if k is None:
                continue
            wall = str(getattr(m, "wall", "")) or "?"
            cur = best_per_wall.get(wall)
            if cur is None or k < cur[0]:
                best_per_wall[wall] = (k, int(mid))
        return [(mid, _bound(k[0])) for k, mid in best_per_wall.values()]

    def _recover_via_central_marker(self, now: float) -> bool:
        """Marker-lost recovery (NO absolute TO): approach the most CENTRAL
        marker on whichever wall is currently in view by vision (holding
        altitude), then RESTART the mission so the C2's task re-runs.

        Unlike the old "anchor on whatever marker is visible" recovery -- which
        could repeatedly pick the same corner marker (drone stuck in a corner)
        and back the drone toward a side net to reach standoff -- this homes
        onto a wall-centre marker, whose standoff sits in the arena interior in
        FRONT of it (~origin). It considers all four walls, so an unseen marker
        on one wall doesn't strand the drone -- another wall's centre marker is
        used instead.

        Returns False (caller keeps rotating + zoom-escalating) when: the hop
        budget is exhausted, no arena candidate is known, or NONE of the
        candidates are in view yet. We only COMMIT (and spend a hop) once a
        centre marker is actually visible, so the approach is never open-loop."""
        if self._recovery_count >= self._MAX_RECOVERY_HOPS:
            return False
        candidates = self._recovery_candidates()
        if not candidates:
            return False
        with self.state.lock:
            visible_ids = set(self.state.visible_marker_ids)
        visible = [(mid, so) for (mid, so) in candidates if mid in visible_ids]
        if not visible:
            # No wall-centre marker in the current FOV -- let the caller keep
            # sweeping and escalating zoom until one rotates into view.
            return False
        # Prefer the nearest candidate (smallest standoff): it's the easiest to
        # detect and approach, and still lands ~at the origin.
        central, standoff = min(visible, key=lambda c: c[1])
        # Reset the smoother so it doesn't blend the OLD target's pose with the
        # centre marker's pose across the switch.
        self.smoother.reset()
        with self.state.lock:
            self.state.active_marker_id = central
            self.state.target_distance_m = standoff
            self.state.target_distance_tol_m = self._RECOVERY_TOL_M
            self.state.approach_positioning = True    # vision-home, hold altitude
            self.state.arrive_hdg_tol_deg = None
            self.state.arrive_yaw_tol_deg = None
            self.state.target_relative_heading_deg = 0.0
            self.state.settle_began_at = None
        self._approach_on_settle_restart_mission = True
        self._recovery_count += 1
        self._set_phase(
            Phase.SEARCH,
            f"recovery {self._recovery_count}/{self._MAX_RECOVERY_HOPS}: "
            f"approach CENTRE marker {central} @ {standoff:.1f}m (~origin) "
            f"then restart mission")
        print(f"[ctrl] marker-lost recovery "
              f"{self._recovery_count}/{self._MAX_RECOVERY_HOPS} → approach "
              f"CENTRE marker {central} @ {standoff:.1f}m (~origin, vision) "
              f"then restart")
        return True

    def _step_search(self, now: float) -> None:
        cfg = self.cfg
        # If we already see the marker, switch immediately. ALIGN orbits
        # at the current radius until the relative heading is roughly
        # zero (marker faces the camera) before APPROACH closes distance
        # -- this keeps the marker out of the oblique-angle range where
        # the ArUco detector starts to fail.
        if self.smoother.get(now) is not None:
            # Stop rotation immediately, then step back (CCW — opposite to
            # the CW search sweep) by search_backrotate_deg so the marker
            # is better centred in the frame before HEIGHT_ALIGN starts.
            self._send_rc(0, 0, 0, 0)
            # Re-acquired a GENUINE target (a capture APPROACH or a real
            # GO_HOME step — NOT a recovery anchor, which keeps the restart
            # flag set): the marker-lost recovery succeeded, so clear the hop
            # budget. (The recovery anchor acquisition leaves the flag True and
            # must NOT reset it, else the budget could never bound the loop.)
            if not self._approach_on_settle_restart_mission:
                self._recovery_count = 0
            back_deg = int(getattr(cfg, "search_backrotate_deg", 10))
            swept = getattr(self, "_search_swept", 0.0)
            # Only back-rotate when the drone actually swept past the back_deg
            # threshold — if the marker appeared almost immediately (e.g. on
            # a HOLD→new-APPROACH transition where the target was already near
            # the centre of frame) a back-rotation would steer the drone away.
            if back_deg > 0 and swept >= back_deg:
                try:
                    self.api.rotate("ccw", back_deg)
                except Exception as e:
                    print(f"[ctrl] SEARCH back-rotate failed: {e}")
            # Keep zoom as-is; _step_approach resets to 1x at the
            # angle-correction-start threshold.
            #
            # Real mission GO_HOME steps now go through HEIGHT_ALIGN, same as
            # capture APPROACHes: the drone aligns to the home marker's altitude
            # before closing distance.
            #
            # Exception: marker-lost RECOVERY hops (_approach_on_settle_restart_mission
            # True) still skip HEIGHT_ALIGN and home at the current altitude.
            # Recovery hops use whatever marker happens to be visible (could be
            # a high centre/wall marker) and their only goal is re-establishing a
            # known position cheaply — climbing to a 2 m marker during recovery
            # would be undesirable.
            with self.state.lock:
                loose_home = self.state.approach_positioning
            is_recovery_hop = self._approach_on_settle_restart_mission
            if loose_home and is_recovery_hop:
                self._set_phase(Phase.APPROACH,
                                "marker acquired (recovery GO_HOME) -- homing at "
                                "current altitude (no climb to marker)")
            else:
                self._set_phase(Phase.HEIGHT_ALIGN,
                                f"marker acquired -- stop + {back_deg}° back -- aligning altitude")
            return

        # Detect new SEARCH phase entry via phase_started_at so the two-
        # stage protocol resets cleanly on every fresh SEARCH.
        with self.state.lock:
            phase_entry = self.state.phase_started_at
        if getattr(self, "_search_phase_entry", None) != phase_entry:
            self._search_phase_entry = phase_entry
            self._search_start_yaw = None
            self._search_swept = 0.0
            self._search_prev_yaw = None
            self._search_yaw_rc = float(cfg.search_yaw_rc)
            # Two-stage SEARCH protocol:
            #   stage 0 = stationary at zoom 2.0x (cfg.search_zoom_check_s).
            #             Acquires distant markers without rotating, so a
            #             marker that's "there but too small" pops out before
            #             we change yaw.
            #   stage 1 = yaw spin at zoom 1.25x (one full revolution).
            # After stage 1 without acquisition: _recover_via_central_marker;
            # if recovery can't commit, _terminate_script (yield to C2).
            self._search_stage = 0
            # Initialise lazily on first stage-0 execution so descent /
            # retreat time doesn't count against the stationary window.
            self._search_stage_started = None
            self._search_zoom = 2.0
            self._apply_zoom(2.0)
            # Arm the descend-to-approach-height for a lost TARGET BOX: a brief
            # HEIGHT_ALIGN to a high/oblique marker -- or a GO_HOME recovery
            # anchor -- can strand the drone above the (low) box, out of the
            # search FOV. Only for a capture APPROACH (not a loose GO_HOME,
            # which holds altitude by design) onto a box id, and only when we
            # have a snapshot to return to.
            self._search_descend_active = False
            self._search_descend_deadline = now + SEARCH_DESCEND_MAX_S
            if getattr(cfg, "search_descend_to_start_height", True):
                with self.state.lock:
                    _h_start = self.state.approach_start_height_m
                    _amid = self.state.active_marker_id
                    _loose = self.state.approach_positioning
                if (_h_start is not None and _amid is not None and not _loose
                        and cfg.target_marker_id_min
                        <= _amid <= cfg.target_marker_id_max):
                    self._search_descend_active = True

        # Descend back to the approach-start height BEFORE the retreat/yaw sweep
        # so a lost box is back in the camera's vertical FOV. Descend-only (it
        # never climbs back up); the _send_rc min-height floor still applies, and
        # the target is clamped to that floor so e_h can always reach the
        # deadband. Bounded by SEARCH_DESCEND_MAX_S against a noisy altimeter.
        if self._search_descend_active:
            if now >= self._search_descend_deadline:
                self._search_descend_active = False
            else:
                with self.state.lock:
                    tel = self.state.last_telemetry
                    _h_start = self.state.approach_start_height_m
                drone_h = (tel.height_cm / 100.0
                           if tel is not None and tel.height_cm is not None
                           else None)
                if drone_h is None or _h_start is None:
                    self._search_descend_active = False
                else:
                    _tgt = max(_h_start, cfg.min_height_m)
                    e_h = drone_h - _tgt              # > 0 => too high
                    if e_h > cfg.height_deadband_m:
                        ud = -max(0, min(int(e_h * cfg.height_kp),
                                         cfg.ud_rc_max))
                        self._send_rc(0, 0, ud, 0)
                        with self.state.lock:
                            self.state.note = (
                                f"SEARCH descend {drone_h:.2f}->{_tgt:.2f}m "
                                f"ud={ud}")
                        return
                    self._search_descend_active = False

        # Retreat-then-yaw: when we entered SEARCH after losing a
        # marker the controller was actively tracking, _set_phase
        # stamps state.search_retreat_until. While that timer hasn't
        # expired, fly straight back instead of yawing — bringing the
        # marker back into view if we just overshot it (small-target
        # case where FB_IMU drove past the box). After the timer the
        # standard yaw-in-place sweep takes over.
        with self.state.lock:
            retreat_until = self.state.search_retreat_until
        if retreat_until is not None and now < retreat_until:
            retreat_rc = int(getattr(cfg, "search_retreat_rc", 25))
            self._send_rc(0, -abs(retreat_rc), 0, 0)
            with self.state.lock:
                self.state.note = (
                    f"SEARCH retreat: fb={-abs(retreat_rc)} "
                    f"{retreat_until - now:.1f}s left"
                )
            return

        # ── Stage 0: stationary at zoom 2.0x ─────────────────────────
        # Hold position with RC zero so the detector gets a window of
        # frames at higher magnification. The acquisition-check at the
        # top of _step_search exits SEARCH the instant the smoother
        # receives a pose, so this stage is bounded by either
        # cfg.search_zoom_check_s (no acquisition) or detector success.
        if getattr(self, "_search_stage", 1) == 0:
            self._send_rc(0, 0, 0, 0)
            # Lazy start: the timer begins on the first tick that reaches
            # stage 0 (i.e. after descent + retreat have completed), so
            # the operator always gets the full stationary window worth
            # of detector frames.
            if self._search_stage_started is None:
                self._search_stage_started = now
            elapsed = now - self._search_stage_started
            with self.state.lock:
                self.state.note = (
                    f"SEARCH stage 0 (stationary zoom 2.0x) "
                    f"{elapsed:.1f}/{cfg.search_zoom_check_s:.1f}s")
            if elapsed >= float(cfg.search_zoom_check_s):
                # No acquisition at zoom 2.0x — drop to 1.25x and start
                # the one-revolution yaw sweep.
                self._search_stage = 1
                self._search_stage_started = now
                self._search_zoom = 1.25
                self._apply_zoom(1.25)
                self._search_start_yaw = None
                self._search_swept = 0.0
                self._search_prev_yaw = None
                print("[ctrl] SEARCH stage 0 → stage 1: zoom 2.0x check "
                      f"({cfg.search_zoom_check_s:.1f}s) yielded nothing, "
                      "spinning at zoom 1.25x")
            return

        # ── Stage 1: yaw spin at zoom 1.25x (target-only) ────────────
        # Look for the active marker for one full revolution. NO
        # recovery polling here: the target may be visible mid-rotation
        # and we don't want to bail out to a wall-marker hop while still
        # actively hunting the script's intended target. The smoother
        # acquisition check at the top of _step_search exits SEARCH the
        # instant the active marker is detected.
        if getattr(self, "_search_stage", 1) == 1:
            self._send_rc(0, 0, 0, int(self._search_yaw_rc),
                          enforce_cfg_caps=False)
            with self.state.lock:
                tel = self.state.last_telemetry
            if tel and tel.yaw_deg is not None:
                if self._search_start_yaw is None:
                    self._search_start_yaw = tel.yaw_deg
                    self._search_swept = 0.0
                    self._search_prev_yaw = tel.yaw_deg
                else:
                    delta = (tel.yaw_deg - self._search_prev_yaw + 540.0) % 360.0 - 180.0
                    self._search_swept += abs(delta)
                    self._search_prev_yaw = tel.yaw_deg
                    with self.state.lock:
                        self.state.search_yaw_swept_deg = self._search_swept
                    if self._search_swept >= cfg.search_total_deg:
                        # One full revolution at zoom 1.25x without
                        # acquiring the target. Hand off to stage 2.
                        self._search_stage = 2
                        self._search_stage_started = now
                        self._search_start_yaw = None
                        self._search_swept = 0.0
                        self._search_prev_yaw = None
                        print("[ctrl] SEARCH stage 1 → stage 2: full "
                              "rotation at zoom 1.25x yielded no target, "
                              "scanning for central-marker recovery")
            elif now - phase_entry > 30.0:
                self._terminate_script("search timed out (no telemetry yaw)")
            return

        # ── Stage 2: yaw spin at zoom 1.25x (recovery scan) ──────────
        # Same rotation as stage 1, but every tick we poll
        # _recover_via_central_marker. As soon as a centre-wall
        # candidate is in view (and the hop budget allows), commit:
        # the recovery's APPROACH→settle→mission-restart re-runs the
        # operator's script from step 0. ~60 polling chances per
        # rotation — vs the single end-of-sweep window that landed the
        # drone on fc1 because the timing was off.
        if self._recover_via_central_marker(now):
            self._search_start_yaw = None
            self._search_swept = 0.0
            self._search_prev_yaw = None
            return
        self._send_rc(0, 0, 0, int(self._search_yaw_rc),
                      enforce_cfg_caps=False)
        with self.state.lock:
            tel = self.state.last_telemetry
        if tel and tel.yaw_deg is not None:
            if self._search_start_yaw is None:
                self._search_start_yaw = tel.yaw_deg
                self._search_swept = 0.0
                self._search_prev_yaw = tel.yaw_deg
            else:
                delta = (tel.yaw_deg - self._search_prev_yaw + 540.0) % 360.0 - 180.0
                self._search_swept += abs(delta)
                self._search_prev_yaw = tel.yaw_deg
                with self.state.lock:
                    self.state.search_yaw_swept_deg = self._search_swept
                if self._search_swept >= cfg.search_total_deg:
                    # One full revolution at zoom 1.25x AND not a single
                    # tick saw a centre-wall recovery candidate. Yield
                    # to C2 so the strategy can re-task with new context.
                    self._terminate_script(
                        "marker-lost: zoom 2.0x check + target sweep at "
                        "1.25x + recovery-scan sweep at 1.25x all "
                        "yielded nothing — ending mission, yielding to C2")
                    return
        elif now - phase_entry > 30.0:
            self._terminate_script("search timed out (no telemetry yaw)")
            return

    # ----------------------------------------------------------------- align
    def _step_align(self, tel: Optional[TelemetrySnapshot],
                   now: float) -> None:
        """Orbit the drone around the marker at its current radius until
        relative_heading is ~0 (marker normal pointing at the camera).

        Lateral PD slides the drone tangentially around the marker to
        drive heading toward 0. Forward setpoint is pinned to the radius
        captured on phase entry (state.align_distance_m) -- we don't
        close distance here, that's APPROACH's job. We want APPROACH to
        start with the marker head-on so its tilt stays inside the
        detector's reliable range.
        """
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        # Hard distance floor: prevent forward motion when too close, but
        # let yaw and lateral PDs keep running. The previous "zero all
        # channels" pattern froze the drone in place if it dipped just
        # past the floor (flight 22-09-49: hovered at d=1.37 with hdg=22
        # for 8 s, never recovering). The fwd PD will naturally produce
        # a backward command when d<target, pushing the drone out.
        with self.state.lock:
            tgt_d = self.state.target_distance_m
        floor_active = d < cfg.distance_floor_factor * tgt_d

        # ALIGN target: heading=0, distance=d_align_start.
        d_set = self.state.align_distance_m or d
        e_yaw = yaw_to_marker
        e_fwd = d - d_set
        # Heading error: shortest-arc to 0.
        e_hdg = ((0.0 - hdg) + 540.0) % 360.0 - 180.0

        # ALIGN uses its own (more lenient) heading deadband -- the
        # detector is noisy at long range / off-axis and HOLD's tight
        # 2 deg threshold is impossible to satisfy here.
        align_dead = cfg.align_heading_deadband_deg

        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        if floor_active:
            u_fwd = min(0.0, u_fwd)         # no forward push when too close
        if abs(e_hdg) < align_dead:
            u_lat_raw = 0.0
        else:
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw))

        # Settle when heading is within the lenient ALIGN deadband.
        # yaw_to_marker still uses the tight yaw deadband -- yaw is
        # measured in pixels and is far less noisy than relative
        # heading at oblique angles.
        in_band = (abs(e_hdg) < align_dead
                   and abs(e_yaw) < cfg.yaw_deadband_deg)
        # Capture transition decision under the lock, fire _set_phase
        # OUTSIDE -- threading.Lock isn't reentrant.
        settled = False
        with self.state.lock:
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.align_settle_time_s:
                    settled = True
            else:
                self.state.settle_began_at = None
        if settled:
            self._set_phase(Phase.APPROACH,
                            "aligned (heading ~ 0) -- closing distance")

    # ---------------------------------------------------------- height-align
    def _step_height_align(self, tel: Optional[TelemetrySnapshot],
                          now: float) -> None:
        """Drive the drone's altitude to match the marker's altitude -- by
        VISION ONLY, never the altimeter (the barometric height is unreliable).

        tvec[1] is the marker's vertical offset from the camera in the
        gimbal-stabilised camera frame (camera +y points world-DOWN, so a
        positive tvec[1] means the marker is BELOW the drone). Driving
        e_h = -tvec[1] toward zero centres the marker vertically in the camera
        viewport -- i.e. puts the drone AT the marker's altitude -- without ever
        reading the (unreliable) altimeter. There is no min/max clamp here
        because that clamp could only be computed from the altimeter; the marker
        position itself bounds the motion (once centred, e_h -> 0 and it stops),
        and the PID output is capped at ud_rc_max.

        Yaw is kept on the marker so it doesn't drift out of frame.
        Forward / lateral channels stay zero -- altitude only.
        """
        cfg = self.cfg
        meas = self.smoother.get(now)
        pose = self.state.last_pose
        if meas is None or pose is None:
            self._marker_lost(now); return
        # Reject stale poses: a 2-s-old estimate from the general smoother is
        # fine for position navigation but not for active altitude control.
        # A bad ippe_temporal/ippe_collapsed pose held for 2 s causes full-
        # throttle blind descent until the barometric floor fires at ~0.2 m.
        if now - self.smoother.last_seen > HEIGHT_ALIGN_POSE_MAX_AGE_S:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        # Barometric height — used only for safety floor/ceiling guards and the
        # diagnostic note. Vision drives e_h; the altimeter is untrusted for
        # the correction itself.
        drone_h: Optional[float] = (
            tel.height_cm / 100.0
            if tel is not None and tel.height_cm is not None else None
        )

        # VISION-ONLY vertical error: -tvec[1] (marker's camera-frame downward
        # offset). No altimeter. Positive e_h => marker is above => fly up.
        try:
            marker_y = float(pose.tvec[1])
        except Exception:
            marker_y = 0.0

        # Camera-based height error: drive marker_y toward 0 so the marker
        # is level with the drone. Positive e_h = need to rise; negative = descend.
        # This is used instead of the altimeter-derived (target_h - drone_h) so
        # that barometer drift/offset cannot cause a premature "settled" verdict.
        e_h_raw = -marker_y  # unclamped; used for the settle check below
        e_h = e_h_raw

        # Safety floor (altimeter only): if the barometer says we are already
        # at or below min_height, do not command further descent regardless of
        # what the camera says — the altimeter may be wrong but we keep this as
        # a hard lower bound to avoid hitting the ground.
        if drone_h is not None and drone_h <= cfg.min_height_m and e_h < 0:
            e_h = 0.0
        # Early descent guard: stop downward correction at HEIGHT_ALIGN_DESCENT_FLOOR_M
        # (default 0.40 m) — well above min_height_m. This catches the failure mode
        # where a bad/stale pose (e.g. ippe_temporal holding +1.75 m) drives the
        # drone to the floor before the smoother staleness check can fire.
        if drone_h is not None and drone_h <= HEIGHT_ALIGN_DESCENT_FLOOR_M and e_h < 0:
            e_h = 0.0
        # Safety ceiling: symmetric guard.
        if drone_h is not None and drone_h >= cfg.max_height_m and e_h > 0:
            e_h = 0.0

        e_yaw = yaw_to_marker
        u_yaw = (0.0 if abs(e_yaw) < cfg.yaw_deadband_deg
                 else self.pd_yaw.step(e_yaw, now))
        u_ud = (0.0 if abs(e_h) < cfg.height_deadband_m
                else self.pd_height.step(e_h, now))

        # Camera-based: skip the barometric altitude envelope so that a low
        # barometer reading cannot block vision-commanded descent.
        self._send_rc(lr=0, fb=0, ud=int(u_ud), yaw=int(u_yaw),
                      altitude_envelope=False)

        drone_h_str = f"{drone_h:.2f}m" if drone_h is not None else "?"
        with self.state.lock:
            self.state.note = (f"HEIGHT_ALIGN: drone={drone_h_str}  "
                               f"marker_y={marker_y:+.2f}m  "
                               f"e={e_h:+.2f}m")

        # Settle: both height and yaw inside their deadbands for the
        # configured time -> transition to APPROACH.
        # Use e_h_raw (the unclamped camera error) so that a floor/ceiling guard
        # clamping e_h to 0 cannot produce a false "settled" verdict while the
        # marker is still significantly off-centre. If the drone has hit the
        # descent floor but the marker is still below, the settle timer must not
        # run — the height-align timeout will force APPROACH after HEIGHT_ALIGN_MAX_S.
        in_band = (abs(e_h_raw) < cfg.height_deadband_m
                   and abs(e_yaw) < cfg.yaw_deadband_deg)
        settled = False
        with self.state.lock:
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.height_settle_time_s:
                    settled = True
            else:
                self.state.settle_began_at = None
        if settled:
            self._set_phase(Phase.APPROACH,
                            f"height aligned (marker centred, e={e_h:+.2f}m) -- "
                            f"closing distance (angle correction at yaw_start_fraction)")
            return

        # Hard timeout: if the marker is physically unreachable at this altitude
        # (e.g. mounted on the floor, drone can't descend below ~48 cm), HEIGHT_ALIGN
        # would stall forever. After HEIGHT_ALIGN_MAX_S accept the current height
        # and start the approach anyway.
        with self.state.lock:
            _ha_started = self.state.phase_started_at
        if now - _ha_started >= HEIGHT_ALIGN_MAX_S:
            print(f"[ctrl] HEIGHT_ALIGN timeout ({HEIGHT_ALIGN_MAX_S:g}s) "
                  f"-- forcing APPROACH (e_h={e_h:+.2f}m marker_y={marker_y:+.2f}m)")
            self._set_phase(Phase.APPROACH,
                            f"height align timeout ({HEIGHT_ALIGN_MAX_S:g}s) -- "
                            f"starting approach at e_h={e_h:+.2f}m")

    # -------------------------------------------------------------- approach
    def _step_approach(self, tel: Optional[TelemetrySnapshot],
                      now: float) -> None:
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        # Capture starting distance on first tick of this APPROACH phase.
        if self._approach_start_d is None:
            self._approach_start_d = d

        # Hard distance floor: prevent FORWARD motion inside the floor;
        # yaw and lateral PDs keep running so the drone can recenter
        # and correct heading. The PD's natural backward output (when
        # d<target) pushes the drone out. Previously we zeroed all
        # channels on floor entry, which froze the drone in place
        # (flight 22-09-49: stuck at d=1.37, hdg=22 with rc=0,0,0,0).
        with self.state.lock:
            tgt_d = self.state.target_distance_m
            tol_override = self.state.target_distance_tol_m
        # Effective forward arrival band. GO_HOME sets a loose
        # target_distance_tol_m (e.g. 0.5 m) so the drone settles anywhere
        # within ±tol of tgt_d ("be roughly in the home zone"); a plain
        # APPROACH leaves it None -> tight cfg.distance_deadband_m for a
        # precise capture-distance hold.
        eff_tol = tol_override if tol_override is not None else cfg.distance_deadband_m
        floor_active = d < cfg.distance_floor_factor * tgt_d

        # Errors
        e_yaw = yaw_to_marker                  # want yaw_to_marker -> 0
        e_fwd = d - tgt_d                      # positive: too far -> move forward
        # Hold relative_heading at 0 during approach. ALIGN may exit while
        # the drone still has residual lateral velocity (the deadband is
        # met but the drone hasn't fully stopped). Without an active
        # lateral correction here, that residual integrates into a heading
        # drift over the long approach -- on flight 21-28-02 the drone
        # drifted from hdg=+15 to hdg=-46 across 9 s of approach, even
        # though approach itself never commanded any lateral motion.
        with self.state.lock:
            _tgt_hdg = self.state.target_relative_heading_deg
        e_hdg = ((_tgt_hdg - hdg) + 540.0) % 360.0 - 180.0

        # Yaw gated by latch (see below). Default = 0; overridden after latch.
        u_yaw = 0.0
        if abs(e_fwd) < eff_tol:
            u_fwd_raw = 0.0
        else:
            u_fwd_raw = self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        if floor_active:
            u_fwd = min(0.0, u_fwd)
        # Two-stage approach speed: hard-cap the forward channel when
        # we're within ``approach_slow_zone_m`` of the target distance.
        # This produces a "cruise then slow capture" profile — fwd_rc_max
        # can stay high for fast closing from far away, and the cap
        # below takes over near the marker so the drone doesn't barrel
        # in and overshoot. Capping AFTER velocity damping so the
        # damping can still pull u_fwd negative for braking and not
        # be clipped to a small positive.
        #
        # For GO_HOME (tol_override set): the loose arrive_tol can be
        # larger than slow_zone, which would place the slow zone entirely
        # inside the deadband (where u_fwd_raw is already 0). Fix: expand
        # the effective slow zone by tol_override so it reaches beyond the
        # deadband boundary and caps the PD output during final approach.
        slow_zone = float(getattr(cfg, "approach_slow_zone_m", 0.0))
        slow_cap = int(getattr(cfg, "approach_slow_rc_max", cfg.fwd_rc_max))
        eff_slow_zone = slow_zone
        if (slow_zone > 0.0
                and tol_override is not None
                and tol_override > slow_zone):
            eff_slow_zone = slow_zone + tol_override
        if eff_slow_zone > 0.0 and abs(e_fwd) < eff_slow_zone:
            cap = max(1, slow_cap)
            u_fwd = max(-cap, min(cap, u_fwd))

        # Yaw + lateral gated: drone flies forward-only until the latch
        # fires at approach_yaw_start_fraction of the leg (start_d→target_d).
        # Formula: threshold = start - frac*(start - target)
        #   frac=0 → fires immediately   frac=0.5 → midpoint   frac=1 → at target
        _lat_frac = float(getattr(cfg, "approach_yaw_start_fraction", 0.5))
        _lat_thr  = (self._approach_start_d
                     - _lat_frac * (self._approach_start_d - tgt_d))
        # Reset zoom to 1x 0.3 m before the lateral-correction threshold so
        # the detector has full FOV when angle correction starts (one-shot).
        if (not self._approach_lat_unlocked
                and d <= _lat_thr + 0.3
                and getattr(self, "_search_zoom", 1.0) > 1.0):
            self._search_zoom = 1.0
            self._apply_zoom(1.0)
        # Height error: VISION ONLY, never the altimeter (it's unreliable).
        # tvec[1] is the marker's downward offset in the gimbal-stabilised
        # camera frame; driving e_h = -tvec[1] -> 0 keeps the marker centred in
        # the viewport, i.e. holds the drone AT the marker's altitude. The same
        # law is used both in the mid-HA pause and in normal approach flight --
        # no absolute-height / altimeter branch.
        #   tvec[1] < 0 = marker above = fly up (e_h > 0)
        #
        # EXCEPTION: a loose GO_HOME (state.approach_positioning) is positioning,
        # not a capture -- it homes at the CURRENT altitude and must not climb to
        # the marker's mounted height (which would drag a low scout up to a
        # 1.7-2.0 m centre/wall marker). Hold altitude (ud=0, Anafi hover). A
        # capture APPROACH with a loose approach_dist_tol_m keeps positioning
        # False, so it still vision-aligns to the marker here.
        with self.state.lock:
            _loose_home = self.state.approach_positioning
        u_ud = 0.0
        e_h = 0.0
        pose = self.state.last_pose
        if pose is not None and not _loose_home:
            try:
                marker_y = float(pose.tvec[1])
            except Exception:
                marker_y = 0.0
            e_h = -marker_y
            # Safety floor / ceiling (SAME guard as the HEIGHT_ALIGN phase): if
            # the altimeter says we are already at/below min_height (or at/above
            # max_height) and the marker is below/above us, clamp e_h to 0. This
            # (a) stops descent below the safe floor toward a marker mounted
            # LOWER than min_height (the low box faces), and crucially (b) lets
            # the mid-HA height-settle COMPLETE (h_in_band) instead of freezing
            # forever chasing an unreachable below-floor marker -- the exact bug
            # behind the "hover 5 m short of the target" stall.
            try:
                _h_cm = (float(tel.raw.get("height_cm"))
                         if tel and tel.raw.get("height_cm") is not None
                         else None)
            except (TypeError, ValueError):
                _h_cm = None
            _drone_h = _h_cm / 100.0 if _h_cm is not None else None
            if _drone_h is not None and _drone_h <= cfg.min_height_m and e_h < 0:
                e_h = 0.0
            if _drone_h is not None and _drone_h >= cfg.max_height_m and e_h > 0:
                e_h = 0.0
            if abs(e_h) >= cfg.height_deadband_m:
                u_ud = self.pd_height.step(e_h, now)

        if not self._approach_lat_unlocked and d <= _lat_thr:
            self._approach_lat_unlocked = True
            self._approach_mid_ha_active = True
            self._approach_mid_ha_settled_at = None
            self._approach_mid_ha_started_at = now
            print(f"[ctrl] APPROACH latch: mid height-align starting "
                  f"d={d:.2f}m e_h={e_h:.2f}m tgt={tgt_d:.2f}m")

        # Mid-approach height re-align: hold distance, stabilise altitude,
        # then release into final corrected approach.
        if self._approach_mid_ha_active:
            # Active brake: counteract residual forward velocity from the
            # straight-in phase. _velocity_damp_fwd short-circuits at u_raw==0,
            # so compute body-forward velocity directly and apply a counter-RC.
            # Clamped to ≤0 so we never reverse past a hover.
            ws = self._telemetry_world_speed(tel)
            if ws is not None:
                _vN, _vE, _yaw = ws
                v_fwd_b, _ = self._world_to_body(_vN, _vE, _yaw)
                _bkv = cfg.approach_mid_ha_brake_kv
                u_fwd = max(-cfg.fwd_rc_max, min(0.0, -_bkv * v_fwd_b))
            else:
                u_fwd = 0.0
            h_in_band = abs(e_h) < cfg.height_deadband_m
            # Also gate on speed: don't release until body speed is low so
            # residual forward momentum can't carry the drone through the
            # pause and into the marker.
            # Exception: loose_home (GO_HOME positioning) already forces e_h=0
            # so h_in_band is always True; skip the speed gate and rely on the
            # approach slow_zone to control speed near the target.
            speed = self._body_speed_cms(tel)
            speed_ok = (_loose_home
                        or speed is None
                        or speed < RC_BRAKE_SPEED_THRESHOLD_CMS)
            if h_in_band and speed_ok:
                if self._approach_mid_ha_settled_at is None:
                    self._approach_mid_ha_settled_at = now
                if now - self._approach_mid_ha_settled_at >= cfg.height_settle_time_s:
                    self._approach_mid_ha_active = False
                    print("[ctrl] APPROACH mid height-align done — final approach")
            else:
                self._approach_mid_ha_settled_at = None
            # Hard timeout safety net: NEVER let the mid-HA pause stall the whole
            # approach. If it hasn't settled within MID_HA_MAX_S (e.g. the marker
            # is unreachable below the floor, or height telemetry is noisy),
            # force-release into the final approach so the drone keeps closing
            # instead of hovering metres short of the target forever.
            started = getattr(self, "_approach_mid_ha_started_at", None)
            if (self._approach_mid_ha_active and started is not None
                    and now - started >= MID_HA_MAX_S):
                self._approach_mid_ha_active = False
                self._approach_mid_ha_settled_at = None
                print(f"[ctrl] APPROACH mid height-align TIMEOUT "
                      f"({MID_HA_MAX_S:g}s) — forcing final approach "
                      f"(e_h={e_h:+.2f}m)")
            with self.state.lock:
                _spd_str = f" spd={speed:.0f}" if speed is not None else ""
                self.state.note = (f"mid height-align e_h={e_h:+.2f}m "
                                   f"d={d:.2f}m{_spd_str}")

        # Both yaw and lateral are gated by the same latch.
        # Before latch or during mid-HA: no lateral steering.
        # After latch + mid-HA done: yaw + lateral correct heading.
        if not self._approach_lat_unlocked or self._approach_mid_ha_active:
            u_lat_raw = 0.0
        elif abs(e_hdg) < cfg.approach_heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        # Yaw: full correction after latch; small pre-latch correction to
        # keep the marker centred in frame during the straight-in phase.
        pre_latch_limit = float(getattr(cfg, "approach_pre_latch_yaw_deg", 10.0))
        if self._approach_lat_unlocked:
            if abs(e_yaw) >= cfg.yaw_deadband_deg:
                u_yaw = self.pd_yaw.step(e_yaw, now)
        elif pre_latch_limit > 0 and abs(e_yaw) >= cfg.yaw_deadband_deg:
            u_yaw = self.pd_yaw.step(
                max(-pre_latch_limit, min(pre_latch_limit, e_yaw)), now
            )

        # Height correction is camera-based: skip the barometric altitude
        # envelope so a low barometer reading cannot block vision-commanded
        # descent.
        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=int(u_ud),
                      yaw=int(u_yaw), altitude_envelope=False)

        # Settle detection. We capture the "settled long enough" decision
        # under the lock, then release before calling _set_phase -- which
        # itself takes the lock and would self-deadlock since
        # threading.Lock is non-reentrant.
        with self.state.lock:
            _hdg_tol = self.state.arrive_hdg_tol_deg
            _yaw_tol_override = self.state.arrive_yaw_tol_deg
        hdg_ok = (_hdg_tol is None or abs(e_hdg) < _hdg_tol)
        # APPROACH's optional 3rd arg (state.arrive_yaw_tol_deg): a WIDE yaw
        # tolerance lets the approach advance the instant the target distance is
        # reached, even from an angle (e.g. 30/45° off the marker normal), with
        # NO settle-time wait -- so the next step (e.g. FB_UD_IMU over the box)
        # fires immediately. None -> tight head-on settle (cfg defaults).
        eff_yaw_tol = (_yaw_tol_override
                       if _yaw_tol_override is not None
                       else cfg.yaw_deadband_deg)
        eff_settle = (0.0 if _yaw_tol_override is not None
                      else cfg.approach_settle_time_s)
        in_band = (abs(e_yaw) < eff_yaw_tol
                   and abs(e_fwd) < eff_tol
                   and hdg_ok)
        settled = False
        with self.state.lock:
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= eff_settle:
                    settled = True
            else:
                self.state.settle_began_at = None
        if settled:
            # ORDER MATTERS. The marker-lost recovery restart check must
            # run BEFORE the WAIT_AND_ATTACK sibling-hold gate: recovery
            # parks the drone on a wall marker (state.active_marker_id =
            # 11 or similar) which differs from wait_attack_target_id
            # (e.g. 46), so the gate would otherwise refuse to advance
            # AND the recovery's mission_step_idx = -1 restart would
            # never fire — the drone would loop forever on the wall
            # marker. Recovery restart re-enters _apply_step_to_phase
            # from step 0; when the WAIT_AND_ATTACK step is reached
            # again it re-seeds wait_attack_target_id and active so
            # SEARCH for the box restarts cleanly.
            if self._approach_on_settle_restart_mission:
                self._approach_on_settle_restart_mission = False
                print("[ctrl] recovery GO_HOME settled at visible marker — "
                      "restarting mission")
                with self.state.lock:
                    self.state.mission_step_idx = -1
                    self.state.last_completed_step_kind = None
                self._advance_script("marker-lost recovery: mission restart")
                return
            # WAIT_AND_ATTACK gate: when we settle while parked on the
            # SIBLING face (target face not yet exposed), DO NOT advance
            # the script — the mission step expects an attack on the
            # target. Stay in APPROACH; the next tick re-runs the PD on
            # whatever active_marker_id the swap routes us to. When the
            # box flips and target becomes visible, the swap puts
            # active = target, the PD re-settles on target, this gate
            # passes, and the script advances. The settle_began_at
            # reset prevents an accidental flash-advance on the swap
            # tick: the new active pose must re-satisfy the in-band
            # condition AND the settle window before we move on.
            with self.state.lock:
                _wa_target = self.state.wait_attack_target_id
                _wa_active = self.state.active_marker_id
                if (_wa_target is not None
                        and _wa_active != _wa_target):
                    self.state.settle_began_at = None
                    return
            self._advance_script("approach settled")

    # ------------------------------------------------------------------ hold
    def _step_hold(self, tel: Optional[TelemetrySnapshot],
                  now: float) -> None:
        cfg = self.cfg
        # Check the HOLD timer FIRST. Once hold_time_s has elapsed we
        # commit to LAND regardless of marker visibility -- otherwise a
        # brief marker loss at the end of HOLD pre-empts the LAND
        # transition via _marker_lost and escalates to SEARCH instead
        # (the user sees "drone hovered then started spinning around
        # again" instead of "drone landed").
        with self.state.lock:
            began = self.state.hold_began_at
            hold_time_s = self.state.hold_time_s
            await_id = self.state.await_marker_id
            visible_now = (set(self.state.visible_marker_ids)
                           if await_id is not None else None)
        # AWAIT early-exit: an AWAIT step routed us here, and the
        # awaited marker is now in view. Advance the script before
        # the timer expires.
        if await_id is not None and visible_now and await_id in visible_now:
            self._send_rc(0, 0, 0, 0)
            self._advance_script(f"AWAIT marker {await_id} seen")
            return
        if began is not None and (now - began) >= hold_time_s:
            self._send_rc(0, 0, 0, 0)
            self._advance_script(f"hold of {hold_time_s:.0f}s complete")
            return

        meas = self.smoother.get(now)
        if meas is None:
            # During hold, brief marker loss is tolerable -- just zero out.
            # If sustained loss, _marker_lost will eventually escalate.
            self._send_rc(0, 0, 0, 0)
            self._marker_lost(now, escalate=True); return
        d, yaw_to_marker, hdg = meas

        # Hard distance floor: prevent forward push only; yaw and lateral
        # keep running so HOLD can still recenter while pushed inside the floor.
        with self.state.lock:
            tgt_d = self.state.target_distance_m
            tgt_h = self.state.target_relative_heading_deg
        floor_active = d < cfg.distance_floor_factor * tgt_d

        # Station-keeping PD: yaw / distance / heading are all actively
        # corrected so wind, rotor wash and sensor drift don't push the
        # drone off station during the timed hover.
        e_yaw = yaw_to_marker
        e_fwd = d - tgt_d
        e_hdg = ((tgt_h - hdg) + 540.0) % 360.0 - 180.0
        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        if floor_active:
            u_fwd = min(0.0, u_fwd)
        if abs(e_hdg) < cfg.heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            # Lateral channel: slide the drone tangentially to correct any
            # angular drift around the marker. Sign reasoning (top-down,
            # drone facing marker): +heading => drone is on the marker's
            # right (CW from normal). To bring heading back DOWN toward
            # target, drone must walk CCW around the marker, which from
            # its own POV is stepping body-RIGHT -- so u_lat has the
            # OPPOSITE sign of e_hdg: u_lat = -k * e_hdg, expressed via
            # arc length d * radians(e_hdg) so the gain stays sensibly
            # tuned at any distance.
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        # Continuous height alignment to the marker, same as APPROACH --
        # VISION ONLY, never the altimeter. HOLD is only entered after APPROACH
        # (HOOVER step rule), so the operator's intent is "stay at the marker's
        # height". Keep the marker centred in the camera viewport
        # (e_h = -tvec[1] -> 0). The IDLE phase (HOOVER without prior APPROACH)
        # keeps ud=0.
        u_ud = 0.0
        pose = self.state.last_pose
        if pose is not None:
            try:
                marker_y = float(pose.tvec[1])
            except Exception:
                marker_y = 0.0
            e_h = -marker_y
            if abs(e_h) >= cfg.height_deadband_m:
                u_ud = self.pd_height.step(e_h, now)

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=int(u_ud),
                      yaw=int(u_yaw))
        # Timer transition is handled at the top of this method so it
        # wins over a marker-lost escalation on the same tick.

    # -------------------------------------------------------------- idle (HOOVER)
    def _step_idle(self, tel: Optional[TelemetrySnapshot], now: float) -> None:
        """Mission-script HOOVER step that didn't follow APPROACH (or
        a PAUSE / AWAIT step routed here): zero RC and let the Anafi
        auto-stabilise. Advances when the per-step timer set in
        _apply_step_to_phase expires, OR -- if AWAIT installed an
        await_marker_id -- as soon as that marker becomes visible.
        """
        self._send_rc(0, 0, 0, 0)
        with self.state.lock:
            until = self.state.idle_until
            await_id = self.state.await_marker_id
            visible_now = (set(self.state.visible_marker_ids)
                           if await_id is not None else None)
            self.state.note = (f"IDLE: {(until - now):.1f}s remaining"
                               if until is not None else "IDLE")
        if await_id is not None and visible_now and await_id in visible_now:
            self._advance_script(f"AWAIT marker {await_id} seen")
            return
        if until is not None and now >= until:
            self._advance_script("idle complete")

    # -------------------------------------------------------------- rc (script)
    def _step_rc(self, tel: Optional[TelemetrySnapshot], now: float) -> None:
        """Mission-script LR / FB / UD / RC step. Two-phase:

        1. DRIVE — pin the operator-typed sticks until ``rc_step_until``.
        2. BRAKE — once the timer expires, hold zero sticks until the
           IMU body-frame velocity drops below RC_BRAKE_SPEED_THRESHOLD_CMS
           (or RC_BRAKE_MAX_SETTLE_S elapses). The Anafi stabiliser does
           the actual braking; we just wait long enough that the next
           step starts from a hover, not on top of residual momentum.

        Bypasses ``cfg.*_rc_max`` (those are PD-tuning bounds for the
        closed-loop controllers — applying them to operator input
        silently shrinks LR 20 to 4 on a host with ``lat_rc_max=4``).
        FC ceiling and arena guard still clamp at the wire."""
        with self.state.lock:
            lr = int(self.state.rc_step_lr)
            fb = int(self.state.rc_step_fb)
            ud = int(self.state.rc_step_ud)
            yaw = int(self.state.rc_step_yaw)
            until = self.state.rc_step_until
            braking = bool(self.state.rc_step_braking)
            brake_started = self.state.rc_step_brake_started

        if not braking:
            # ── DRIVE phase ─────────────────────────────────────────
            spd = max(0.0, min(1.0, float(getattr(self.cfg, "goto_speed_factor", 1.0))))
            self._send_rc(int(lr * spd), int(fb * spd), int(ud * spd),
                          int(yaw * spd), enforce_cfg_caps=False)
            with self.state.lock:
                remain = (until - now) if until is not None else None
                self.state.note = (
                    f"RC drive: lr={lr} fb={fb} ud={ud} yaw={yaw}"
                    + (f", {remain:.1f}s remaining" if remain is not None
                       else "")
                )
            if until is not None and now >= until:
                # Drive timer expired — flip into brake phase. Next
                # tick starts the velocity-based settle.
                with self.state.lock:
                    self.state.rc_step_braking = True
                    self.state.rc_step_brake_started = now
            return

        # ── BRAKE phase ─────────────────────────────────────────────
        self._send_rc(0, 0, 0, 0, enforce_cfg_caps=False)
        speed = self._body_speed_cms(tel)
        elapsed = (now - brake_started) if brake_started is not None else 0.0
        with self.state.lock:
            self.state.note = (
                f"RC brake: speed={speed:.0f}cm/s elapsed={elapsed:.2f}s"
                if speed is not None
                else f"RC brake: speed=? elapsed={elapsed:.2f}s"
            )
        # Settle conditions:
        #   - body speed below threshold (drone is effectively still), OR
        #   - max-settle elapsed (don't hang on stuck telemetry)
        settled = (
            speed is not None and speed < RC_BRAKE_SPEED_THRESHOLD_CMS
        ) or elapsed >= RC_BRAKE_MAX_SETTLE_S
        if settled:
            reason = (f"speed={speed:.0f}cm/s < {RC_BRAKE_SPEED_THRESHOLD_CMS:g}"
                      if speed is not None and speed < RC_BRAKE_SPEED_THRESHOLD_CMS
                      else f"timeout {elapsed:.2f}s")
            self._advance_script(
                f"rc step complete (brake {elapsed:.2f}s, {reason})"
            )

    # ----------------------------------------------------------- fb_brake
    def _step_fb_brake(self, tel: Optional[TelemetrySnapshot],
                       now: float) -> None:
        """FB_BRAKE step: pin forward stick, brake when marker close.

        Two-phase like _step_rc:

        1. DRIVE — pin ``rc_step_fb`` until ONE OF:
              (a) the active marker is visible AND its measured
                  ``distance_m`` < ``target_distance_m`` (vision trip), OR
              (b) the active marker is NOT currently visible BUT we
                  know its arena-frame position from the arena config
                  AND the drone's world-position estimate is fresh,
                  AND the horizontal distance from drone to marker
                  (in arena frame) is < ``target_distance_m + WORLD_FALLBACK_MARGIN_M``
                  (world fallback — keeps the drone from crashing
                  through a wall when vision blacks out / yaw drift
                  pushes the marker out of FOV), OR
              (c) ``rc_step_until`` elapses (hard timeout — neither
                  vision nor world-position estimate said brake;
                  prevents an infinite cruise).
        2. BRAKE — identical to _step_rc's brake: zero sticks, wait
           for IMU body-speed < RC_BRAKE_SPEED_THRESHOLD_CMS or
           RC_BRAKE_MAX_SETTLE_S, then advance.

        Vision trip is preferred (sub-decimetre precision via IPPE).
        World trip is the safety net — the world-position estimator
        drifts a few metres at cruise speeds, so the margin is
        generous (we err on braking too EARLY rather than crashing).
        Timeout is the last-resort guard.

        Pose freshness: ``last_pose`` is cleared on step entry so a
        stale pose from a previous APPROACH (which may be on a
        DIFFERENT marker) can't false-trip the brake. We additionally
        gate on ``pose.marker_id == active_marker_id``.
        """
        with self.state.lock:
            fb = int(self.state.rc_step_fb)
            until = self.state.rc_step_until
            braking = bool(self.state.rc_step_braking)
            brake_started = self.state.rc_step_brake_started
            active = self.state.active_marker_id
            stop_m = float(self.state.target_distance_m)
            pose = self.state.last_pose
            drone_pos = self.state.world_position_m
            drone_pos_at = self.state.world_position_updated_at
            hint_x = self.state.fb_brake_world_x
            hint_y = self.state.fb_brake_world_y

        if not braking:
            # ── DRIVE phase ─────────────────────────────────────────
            self._send_rc(0, fb, 0, 0, enforce_cfg_caps=False)

            # Is vision LIVE for the active marker? Fresh pose, right
            # marker. When vision is live we let it drive the brake
            # (precise IPPE distance) and SUPPRESS the world fallback
            # — otherwise the conservative world margin trips first and
            # the drone brakes metres short of a marker it can clearly
            # see. This is the "approach the marker you can see;
            # fall back to its remembered position only when you've
            # lost sight of it" behaviour.
            vision_live = (
                pose is not None
                and active is not None
                and int(pose.marker_id) == int(active)
                and (now - float(pose.timestamp)) < VISION_LIVE_MAX_AGE_S
            )

            # Trip 1: vision. Live pose below the stop threshold.
            trip_vision = (
                vision_live and float(pose.distance_m) < stop_m
            )

            # Trip 2: world-position fallback — ONLY when vision is not
            # live. Prefer the explicit script hint (slot face markers,
            # not in arena config); else arena.markers (wall markers).
            # The brake fires when the drone's *estimated* world
            # position is close to the marker's *known* world position
            # — the safety net for when vision can't see the marker
            # (yaw drift, occlusion, out of FOV, too far for the small
            # slot markers). Suppressed while vision_live so it never
            # preempts a clean visual approach.
            trip_world = False
            world_dist = None
            world_src = None
            if (not vision_live
                    and drone_pos is not None
                    and drone_pos_at > 0.0
                    and (now - drone_pos_at) < WORLD_POSITION_MAX_AGE_S):
                marker_xy = None
                if hint_x is not None and hint_y is not None:
                    marker_xy = (float(hint_x), float(hint_y))
                    world_src = "hint"
                elif (active is not None
                        and self.get_arena is not None):
                    arena = self.get_arena()
                    if (arena is not None and arena.markers
                            and int(active) in arena.markers):
                        m_pos = arena.markers[int(active)].position_m
                        marker_xy = (float(m_pos[0]), float(m_pos[1]))
                        world_src = "arena"
                if marker_xy is not None:
                    dx = marker_xy[0] - float(drone_pos[0])
                    dy = marker_xy[1] - float(drone_pos[1])
                    world_dist = math.hypot(dx, dy)
                    if world_dist < stop_m + WORLD_FALLBACK_MARGIN_M:
                        trip_world = True

            trip_timeout = (until is not None and now >= until)

            if trip_vision or trip_world or trip_timeout:
                with self.state.lock:
                    self.state.rc_step_braking = True
                    self.state.rc_step_brake_started = now
                    if trip_vision:
                        msg = (f"FB_BRAKE: marker {active} at "
                               f"{pose.distance_m:.2f}m < {stop_m:.2f}m "
                               f"(vision) — braking")
                    elif trip_world:
                        msg = (f"FB_BRAKE: marker {active} world-dist "
                               f"{world_dist:.2f}m < "
                               f"{stop_m + WORLD_FALLBACK_MARGIN_M:.2f}m "
                               f"(world fallback via {world_src}, vision "
                               f"missing) — braking")
                    else:
                        msg = (f"FB_BRAKE: timeout (marker {active} not "
                               f"reached by vision OR world) — braking")
                    self.state.note = msg
                print(f"[FB_BRAKE] {msg}", flush=True)
                return

            # Periodic drive-phase log so we can see WHY brake didn't
            # trip when it should have. Log once per second.
            if not hasattr(self, "_fb_brake_last_log") or (
                    now - self._fb_brake_last_log >= 1.0):
                self._fb_brake_last_log = now
                d_vis = (f"{pose.distance_m:.2f}m"
                         if pose is not None
                         and active is not None
                         and int(pose.marker_id) == int(active)
                         else "—")
                d_world = (f"{world_dist:.2f}m"
                           if world_dist is not None else "—")
                pos_age = ((now - drone_pos_at)
                           if drone_pos_at > 0.0 else None)
                print(
                    f"[FB_BRAKE] DRIVE marker={active} d_vis={d_vis} "
                    f"d_world={d_world} ({world_src or 'no-src'}) "
                    f"pos={drone_pos} pos_age={pos_age:.1f}s "
                    f"hint=({hint_x},{hint_y}) stop<{stop_m:.2f}m"
                    if pos_age is not None
                    else
                    f"[FB_BRAKE] DRIVE marker={active} d_vis={d_vis} "
                    f"d_world={d_world} ({world_src or 'no-src'}) "
                    f"pos={drone_pos} pos_age=NO_POS "
                    f"hint=({hint_x},{hint_y}) stop<{stop_m:.2f}m",
                    flush=True,
                )

            # Still cruising — diagnostic note for the UI.
            with self.state.lock:
                d_vis = (f"{pose.distance_m:.2f}m"
                         if pose is not None
                         and active is not None
                         and int(pose.marker_id) == int(active)
                         else "—")
                d_world = (f"{world_dist:.2f}m"
                           if world_dist is not None else "—")
                remain = (until - now) if until is not None else None
                self.state.note = (
                    f"FB_BRAKE drive: fb={fb} marker={active} "
                    f"d_vis={d_vis} d_world={d_world} stop<{stop_m:.2f}m"
                    + (f" ({remain:.1f}s deadline)" if remain is not None
                       else "")
                )
            return

        # ── BRAKE phase ─────────────────────────────────────────────
        self._send_rc(0, 0, 0, 0, enforce_cfg_caps=False)
        speed = self._body_speed_cms(tel)
        elapsed = (now - brake_started) if brake_started is not None else 0.0
        with self.state.lock:
            self.state.note = (
                f"FB_BRAKE brake: speed={speed:.0f}cm/s elapsed={elapsed:.2f}s"
                if speed is not None
                else f"FB_BRAKE brake: speed=? elapsed={elapsed:.2f}s"
            )
        settled = (
            speed is not None and speed < RC_BRAKE_SPEED_THRESHOLD_CMS
        ) or elapsed >= RC_BRAKE_MAX_SETTLE_S
        if settled:
            reason = (f"speed={speed:.0f}cm/s < {RC_BRAKE_SPEED_THRESHOLD_CMS:g}"
                      if speed is not None and speed < RC_BRAKE_SPEED_THRESHOLD_CMS
                      else f"timeout {elapsed:.2f}s")
            self._advance_script(
                f"fb_brake complete (brake {elapsed:.2f}s, {reason})"
            )

    # --------------------------------------------------------- auto_attack
    @staticmethod
    def _aa_wrap180(deg: float) -> float:
        return ((float(deg) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _aa_height_m(tel: Optional[TelemetrySnapshot]) -> Optional[float]:
        try:
            h_cm = (float(tel.raw.get("height_cm"))
                    if tel and tel.raw.get("height_cm") is not None
                    else None)
        except (TypeError, ValueError):
            h_cm = None
        return (h_cm / 100.0) if h_cm is not None else None

    def _aa_alt_hold(self, h: Optional[float],
                     target: Optional[float] = None) -> int:
        """UD stick to hold an altitude. Defaults to the per-mission
        cruise altitude; pass ``target`` to hold a different one (e.g.
        the over-marker rise altitude). 0 if altitude unknown."""
        if h is None:
            return 0
        tgt = float(self.state.aa_altitude_m) if target is None else float(target)
        err = tgt - h
        return int(max(-AA_UD_MAX, min(AA_UD_MAX, AA_UD_KP * err)))

    def _aa_steer(self, goal_xy, drone_xy, drone_yaw_deg):
        """Arena-frame position steering. Turn to face the goal, then
        drive forward (capped at the per-mission approach speed).
        Returns (fb, yaw, dist)."""
        fb_max = int(self.state.aa_approach_speed)
        dx = goal_xy[0] - drone_xy[0]
        dy = goal_xy[1] - drone_xy[1]
        dist = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dx, dy))   # CW from +y
        err = self._aa_wrap180(desired_heading - drone_yaw_deg)
        yaw = int(max(-AA_YAW_MAX, min(AA_YAW_MAX, AA_YAW_KP * err)))
        if abs(err) < AA_FACING_TOLERANCE_DEG:
            fb = int(max(AA_FB_MIN, min(fb_max, AA_FB_KP * dist)))
        else:
            fb = 0   # turn in place first to avoid curving into a wall
        return fb, yaw, dist

    def _aa_vision_home(self, pose, stop_m):
        """Vision-relative homing — drift-free. Centre the marker by its
        camera bearing and approach by its measured distance, capped at
        the per-mission approach speed. Returns (fb, yaw)."""
        fb_max = int(self.state.aa_approach_speed)
        bearing = float(pose.yaw_deg)            # marker bearing off centre
        yaw = int(max(-AA_YAW_MAX,
                      min(AA_YAW_MAX, AA_VISION_YAW_KP * bearing)))
        err = float(pose.distance_m) - stop_m
        if abs(bearing) < AA_FACING_TOLERANCE_DEG:
            fb = int(max(0, min(fb_max, AA_FB_KP * max(0.0, err))))
        else:
            fb = 0
        return fb, yaw

    def _step_auto_attack(self, tel: Optional[TelemetrySnapshot],
                          now: float) -> None:
        """Reactive end-to-end attack. Internal sub-state machine:

            climb        → ud until safe cruise altitude
            go_target    → approach the enemy slot. Vision-home when the
                           target marker is visible; when within
                           AA_CAPTURE_TRIGGER_M record that distance D.
                           (No world-distance capture — that drifts and
                           caused premature turn-arounds.)
            over_rise    → rise to the rise altitude (≥1.5 m), above box
            over_forward → move forward by D so the drone ends up
                           directly OVER the marker/box ("drop" position)
            over_dwell   → hover over the box for AA_DWELL_S, rotating to
                           face our home zone and acquiring the home
                           marker ("score" dwell — does NOT land)
            go_home      → fly back to the home-wall marker; vision-home
                           when visible; → face_enemy when close
            face_enemy   → at home, rotate to face the enemy home zone
                           (so the next run starts already oriented)
            ready        → terminal hover at home, NOT landed — waits
                           airborne for the next attack run

        The drone never lands on the normal path: after the drop it
        returns home and holds station, ready to attack again. (LAND
        remains only as a failsafe for lost-positioning timeouts.)

        Steering uses the ArUco arena position when the marker is out of
        view, and drift-free vision bearing/distance when it's visible.
        Wall safety is the rc_loop arena guard (active reverse-brake),
        which clamps every RC we send here.
        """
        with self.state.lock:
            sub = self.state.aa_substate
            sub_started = self.state.aa_substate_started
            tgt_marker = self.state.aa_target_marker
            tgt_xy = self.state.aa_target_xy
            home_marker = self.state.aa_home_marker
            home_xy = self.state.aa_home_xy
            drone_pos = self.state.world_position_m
            drone_yaw = self.state.arena_yaw_deg
            pose = self.state.last_pose
            active = self.state.active_marker_id

        h = self._aa_height_m(tel)
        elapsed = now - sub_started

        def advance(new_sub: str, reason: str, set_active=None):
            with self.state.lock:
                self.state.aa_substate = new_sub
                self.state.aa_substate_started = now
                if set_active is not None:
                    self.state.active_marker_id = int(set_active)
                self.state.note = f"AUTO_ATTACK {new_sub}: {reason}"
            print(f"[AUTO_ATTACK] {sub} -> {new_sub}: {reason}", flush=True)

        def vision_live(marker):
            return (pose is not None and active is not None
                    and marker is not None
                    and int(pose.marker_id) == int(marker)
                    and (now - float(pose.timestamp)) < VISION_LIVE_MAX_AGE_S)

        # ── CLIMB ───────────────────────────────────────────────────
        if sub == "climb":
            target_alt = float(self.state.aa_altitude_m)
            if h is not None and h >= target_alt - AA_ALT_TOLERANCE_M:
                advance("go_target", f"reached alt {h:.2f}m",
                        set_active=tgt_marker)
                self._send_rc(0, 0, 0, 0, enforce_cfg_caps=False)
                return
            if elapsed > AA_CLIMB_TIMEOUT_S:
                advance("go_target",
                        f"climb timeout (alt {h if h is None else round(h,2)})",
                        set_active=tgt_marker)
                return
            self._send_rc(0, 0, AA_UD_MAX, 0, enforce_cfg_caps=False)
            return

        # ── GO_TARGET ───────────────────────────────────────────────
        if sub == "go_target":
            # APPROACH the target by vision until we're "in front of it"
            # (within AA_CAPTURE_TRIGGER_M). Record that distance D and
            # hand off to the over-marker maneuver. We do NOT capture by
            # world-position estimate — that drifts and was triggering a
            # premature turn-around before the drone actually reached the
            # target. Vision distance is the only capture trigger.
            if vision_live(tgt_marker):
                with self.state.lock:
                    self.state.aa_last_vision_at = now
                d = float(pose.distance_m)
                if d < AA_CAPTURE_TRIGGER_M:
                    with self.state.lock:
                        self.state.aa_capture_distance_m = d
                    advance("over_rise",
                            f"in front of target at {d:.2f}m → rise+over")
                    self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                                  enforce_cfg_caps=False)
                    return
                fb, yaw = self._aa_vision_home(pose, AA_CAPTURE_TRIGGER_M)
                self._send_rc(0, fb, self._aa_alt_hold(h), yaw,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (
                        f"AUTO_ATTACK go_target VISION d={d:.1f}m "
                        f"brg={pose.yaw_deg:.0f} fb={fb} yaw={yaw}")
                return
            # No vision THIS tick — steer toward the target's known arena
            # position to bring the marker into the camera's view. Keep
            # going (the marker should appear); only the safety timeout
            # ends this without ever seeing the marker. If the target has
            # no known position (vision-only target not in the arena
            # config), just hold + wait for the marker to appear.
            if (tgt_xy is not None and drone_pos is not None
                    and drone_yaw is not None):
                fb, yaw, dist = self._aa_steer(tgt_xy, drone_pos, drone_yaw)
                self._send_rc(0, fb, self._aa_alt_hold(h), yaw,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (
                        f"AUTO_ATTACK go_target WORLD-seek d={dist:.1f}m "
                        f"fb={fb} yaw={yaw} (waiting for marker)")
            else:
                # No target position to steer to — hold and wait for
                # vision to pick up the marker.
                self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (
                        "AUTO_ATTACK go_target: waiting for target marker "
                        "(vision-only, no world position)")
            if elapsed > AA_GO_TIMEOUT_S:
                # Safety: never saw the marker close. Do the over-marker
                # move anyway using the trigger distance as a best guess.
                with self.state.lock:
                    self.state.aa_capture_distance_m = AA_CAPTURE_TRIGGER_M
                advance("over_rise", "go_target timeout — over anyway")
            return

        # ── OVER_RISE ── climb to the rise altitude above the box ────
        # Rise target is max(cruise, 1.5 m) so we never descend here and
        # always clear the box top before moving over it.
        rise_target = max(float(self.state.aa_altitude_m),
                          AA_CAPTURE_RISE_ALT_M)
        if sub == "over_rise":
            if h is not None and h >= rise_target - AA_ALT_TOLERANCE_M:
                # At rise altitude — start the forward-over move. Forward
                # duration = recorded distance D ÷ over-move speed.
                with self.state.lock:
                    d = float(self.state.aa_capture_distance_m)
                    dur = max(0.0, d / AA_OVER_FORWARD_SPEED_MPS)
                    dur = min(dur, AA_OVER_FORWARD_MAX_S)
                    self.state.aa_over_forward_until = now + dur
                advance("over_forward",
                        f"risen to {h:.2f}m → forward {d:.2f}m ({dur:.1f}s)")
                return
            # Rise straight up, hold horizontal position.
            self._send_rc(0, 0, AA_UD_MAX, 0, enforce_cfg_caps=False)
            with self.state.lock:
                self.state.note = (
                    f"AUTO_ATTACK over_rise h={h if h is None else round(h,2)}m "
                    f"→ {rise_target:.1f}m")
            return

        # ── OVER_FORWARD ── move forward by D to sit over the marker ──
        if sub == "over_forward":
            with self.state.lock:
                until = self.state.aa_over_forward_until
            if now >= until:
                self._send_rc(0, 0, self._aa_alt_hold(h, rise_target), 0,
                              enforce_cfg_caps=False)
                # Don't fly home yet — dwell over the box (face home +
                # acquire the home marker) for AA_DWELL_S first.
                with self.state.lock:
                    self.state.aa_dwell_until = now + AA_DWELL_S
                    self.state.aa_last_vision_at = 0.0
                advance("over_dwell", "over the marker → dwell + face home",
                        set_active=home_marker)
                return
            # Drive forward at the over-move speed, holding the rise
            # altitude (NOT cruise — we must stay above the box).
            self._send_rc(0, AA_OVER_FORWARD_RC,
                          self._aa_alt_hold(h, rise_target), 0,
                          enforce_cfg_caps=False)
            with self.state.lock:
                self.state.note = (
                    f"AUTO_ATTACK over_forward (moving over marker, "
                    f"{until - now:.1f}s left)")
            return

        # ── OVER_DWELL ── hover over the box: face home + find home marker ──
        if sub == "over_dwell":
            with self.state.lock:
                until = self.state.aa_dwell_until
            # Yaw toward home: prefer centering the home marker if it's
            # already in view ("focus on the home marker"); otherwise
            # turn toward the home position in the arena frame.
            yaw = 0
            facing = "?"
            if vision_live(home_marker):
                with self.state.lock:
                    self.state.aa_last_vision_at = now
                yaw = int(max(-AA_YAW_MAX, min(AA_YAW_MAX,
                              AA_VISION_YAW_KP * float(pose.yaw_deg))))
                facing = f"home-marker brg={pose.yaw_deg:.0f}"
            elif (drone_pos is not None and drone_yaw is not None
                  and home_xy is not None):
                dx = home_xy[0] - drone_pos[0]
                dy = home_xy[1] - drone_pos[1]
                desired = math.degrees(math.atan2(dx, dy))   # CW from +y
                err = self._aa_wrap180(desired - drone_yaw)
                yaw = int(max(-AA_YAW_MAX, min(AA_YAW_MAX, AA_YAW_KP * err)))
                facing = f"world err={err:.0f}"
            # Stay put over the box at the rise altitude — only rotate.
            self._send_rc(0, 0, self._aa_alt_hold(h, rise_target), yaw,
                          enforce_cfg_caps=False)
            if now >= until:
                advance("go_home", "dwell done → fly home",
                        set_active=home_marker)
                with self.state.lock:
                    self.state.aa_last_vision_at = 0.0
                return
            with self.state.lock:
                self.state.note = (
                    f"AUTO_ATTACK over_dwell ({until - now:.1f}s left) "
                    f"facing home [{facing}]")
            return

        # ── GO_HOME ─────────────────────────────────────────────────
        if sub == "go_home":
            if vision_live(home_marker):
                with self.state.lock:
                    self.state.aa_last_vision_at = now
                if float(pose.distance_m) < AA_HOME_STOP_M:
                    advance("face_enemy",
                            f"home vision dist {pose.distance_m:.2f}m")
                    self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                                  enforce_cfg_caps=False)
                    return
                fb, yaw = self._aa_vision_home(pose, AA_HOME_STOP_M)
                self._send_rc(0, fb, self._aa_alt_hold(h), yaw,
                              enforce_cfg_caps=False)
                return
            recent_vision = (now - self.state.aa_last_vision_at) < AA_VISION_GRACE_S
            if drone_pos is not None and drone_yaw is not None:
                fb, yaw, dist = self._aa_steer(home_xy, drone_pos, drone_yaw)
                if dist < AA_HOME_STOP_M + WORLD_FALLBACK_MARGIN_M and not recent_vision:
                    advance("face_enemy", f"world-arrived home {dist:.2f}m")
                    return
                self._send_rc(0, fb, self._aa_alt_hold(h), yaw,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (
                        f"AUTO_ATTACK go_home WORLD d={dist:.1f}m "
                        f"fb={fb} yaw={yaw}")
                return
            self._send_rc(0, AA_FB_MIN, self._aa_alt_hold(h), 0,
                          enforce_cfg_caps=False)
            if elapsed > AA_GO_TIMEOUT_S:
                # Lost positioning the whole way home — land as a failsafe
                # (can't navigate to hold station blind).
                advance("land", "go_home timeout (no pos)")
            return

        # ── FACE_ENEMY ── back home: turn to face the enemy home zone ──
        if sub == "face_enemy":
            # Enemy direction in the arena frame: from our home toward the
            # target we just attacked (the enemy side). Fall back to the
            # arena centre when the target had no world position.
            enemy_heading = None
            if home_xy is not None and tgt_xy is not None:
                enemy_heading = math.degrees(math.atan2(
                    tgt_xy[0] - home_xy[0], tgt_xy[1] - home_xy[1]))
            elif home_xy is not None:
                enemy_heading = math.degrees(math.atan2(
                    -home_xy[0], -home_xy[1]))
            if enemy_heading is None or drone_yaw is None:
                # No facing reference — just hold, ready for the next run.
                self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                              enforce_cfg_caps=False)
                advance("ready", "home (no facing reference)")
                return
            err = self._aa_wrap180(enemy_heading - drone_yaw)
            if abs(err) < AA_FACE_DONE_DEG or elapsed > AA_FACE_TIMEOUT_S:
                self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                              enforce_cfg_caps=False)
                advance("ready", f"facing enemy (err {err:.0f}°)")
                return
            yaw = int(max(-AA_YAW_MAX, min(AA_YAW_MAX, AA_YAW_KP * err)))
            self._send_rc(0, 0, self._aa_alt_hold(h), yaw,
                          enforce_cfg_caps=False)
            with self.state.lock:
                self.state.note = (
                    f"AUTO_ATTACK face_enemy err={err:.0f}° yaw={yaw}")
            return

        # ── READY ── terminal hover: run complete, NOT landed ────────
        if sub == "ready":
            # Hold station at home, hovering, facing the enemy. We do NOT
            # advance the script here: the end-of-script path triggers a
            # safety LAND, and the whole point is to stay airborne and
            # ready for the next attack run. If the operator's script has
            # an explicit step AFTER this AUTO_ATTACK, run it instead.
            self._send_rc(0, 0, self._aa_alt_hold(h), 0,
                          enforce_cfg_caps=False)
            with self.state.lock:
                idx = self.state.mission_step_idx
                n_steps = len(self.state.mission_script)
            if idx + 1 < n_steps:
                self._advance_script("auto_attack complete → next step")
                return
            with self.state.lock:
                self.state.note = (
                    "AUTO_ATTACK ready — hovering at home facing enemy "
                    "(no land, awaiting next run)")
            return

        # ── LAND ────────────────────────────────────────────────────
        if sub == "land":
            self._send_rc(0, 0, 0, 0, enforce_cfg_caps=False)
            try:
                self.api.land()
            except DroneApiError as e:
                print(f"[AUTO_ATTACK] land error: {e}", flush=True)
            self._advance_script("auto_attack complete")
            return

        # Unknown sub-state — fail safe to land.
        advance("land", f"unknown substate {sub!r}")

    # ------------------------------------------------------------- scout
    def _step_scout(self, tel: Optional[TelemetrySnapshot],
                    now: float) -> None:
        """Mission-script SCOUT step: slow 360° yaw spin.

        Two-phase like _step_rc:
        1. DRIVE — pin yaw stick at SCOUT_YAW_STICK and accumulate
           ``cur_yaw - prev_yaw`` (wrap-safe to [-180, +180]) until
           |accumulated| >= SCOUT_TARGET_DEG, or until
           SCOUT_MAX_DRIVE_S elapses (safety cap if telemetry hangs).
        2. BRAKE — zero sticks, wait for body-frame speed to drop
           below RC_BRAKE_SPEED_THRESHOLD_CMS (or the 1.5 s timeout).
           Shares state with _step_rc's brake so the next step starts
           from a clean hover.
        """
        with self.state.lock:
            started = self.state.scout_started_at or now
            accumulated = float(self.state.scout_accumulated_deg)
            last_yaw = self.state.scout_last_yaw_deg
            braking = bool(self.state.rc_step_braking)
            brake_started = self.state.rc_step_brake_started

        if not braking:
            # DRIVE phase ─ keep the yaw stick at SCOUT_YAW_STICK and
            # accumulate rotation from telemetry.
            self._send_rc(0, 0, 0, int(self.cfg.scout_yaw_stick),
                          enforce_cfg_caps=False)
            cur_yaw = tel.yaw_deg if tel is not None else None
            if cur_yaw is not None:
                with self.state.lock:
                    if self.state.scout_last_yaw_deg is None:
                        self.state.scout_last_yaw_deg = float(cur_yaw)
                    else:
                        # Wrap-safe delta: project (cur - prev) into
                        # [-180, +180] before adding to the cumulative
                        # rotation. Without this a -179 → +179 wrap
                        # registers as +358° in one tick.
                        delta = (
                            (float(cur_yaw) - self.state.scout_last_yaw_deg
                             + 540.0) % 360.0 - 180.0
                        )
                        self.state.scout_accumulated_deg += delta
                        self.state.scout_last_yaw_deg = float(cur_yaw)
                        accumulated = self.state.scout_accumulated_deg
            elapsed = now - started
            with self.state.lock:
                self.state.note = (
                    f"SCOUT: {accumulated:+.0f}° / {SCOUT_TARGET_DEG:g}° "
                    f"(t={elapsed:.1f}s)"
                )
            if (abs(accumulated) >= SCOUT_TARGET_DEG
                    or elapsed >= SCOUT_MAX_DRIVE_S):
                reason = (
                    f"target reached ({accumulated:+.0f}°)"
                    if abs(accumulated) >= SCOUT_TARGET_DEG
                    else f"timeout {elapsed:.1f}s ({accumulated:+.0f}°)"
                )
                with self.state.lock:
                    self.state.rc_step_braking = True
                    self.state.rc_step_brake_started = now
                print(f"[ctrl] scout drive done: {reason}")
            return

        # BRAKE phase ─ same shape as _step_rc's brake. Drone has
        # rotational momentum; zero the sticks and wait for the
        # body-frame speed to drop below threshold (the Anafi yaw
        # damps fast — usually well under 1 s — but we keep the
        # standard cap so a stuck telemetry feed can't hang the
        # mission).
        self._send_rc(0, 0, 0, 0, enforce_cfg_caps=False)
        speed = self._body_speed_cms(tel)
        elapsed_brake = (now - brake_started) if brake_started is not None else 0.0
        with self.state.lock:
            self.state.note = (
                f"SCOUT brake: speed={speed:.0f}cm/s "
                f"elapsed={elapsed_brake:.2f}s"
                if speed is not None
                else f"SCOUT brake: speed=? elapsed={elapsed_brake:.2f}s"
            )
        settled = (
            (speed is not None and speed < RC_BRAKE_SPEED_THRESHOLD_CMS)
            or elapsed_brake >= RC_BRAKE_MAX_SETTLE_S
        )
        if settled:
            self._advance_script(
                f"scout complete (rotated {accumulated:+.0f}°, "
                f"brake {elapsed_brake:.2f}s)"
            )

    @staticmethod
    def _body_speed_cms(tel: Optional[TelemetrySnapshot]) -> Optional[float]:
        """Body-frame speed magnitude in cm/s from the Anafi
        ``vgx/vgy/vgz`` telemetry. Returns None if the telemetry
        snapshot is missing any axis — caller treats that as
        ``don't settle yet``, falling back to the brake timeout."""
        if tel is None or tel.raw is None:
            return None
        try:
            vgx = float(tel.raw.get("vgx", 0.0))
            vgy = float(tel.raw.get("vgy", 0.0))
            vgz = float(tel.raw.get("vgz", 0.0))
        except (TypeError, ValueError):
            return None
        return (vgx * vgx + vgy * vgy + vgz * vgz) ** 0.5

    # ------------------------------------------------------------ height (script)
    def _step_height(self, tel: Optional[TelemetrySnapshot],
                    now: float) -> None:
        """Mission-script HEIGHT step: drive the drone's altitude to
        ``state.height_target_m`` and advance once it has settled
        within the height deadband for the configured settle time.
        """
        cfg = self.cfg
        with self.state.lock:
            target = self.state.height_target_m
            phase_entry = self.state.phase_started_at
        if target is None:
            self._advance_script("height: no target set")
            return
        # On first entry into this HEIGHT phase: pre-set zoom to 1.25x while
        # the camera is not needed for barometer-based altitude control.
        # This avoids the Olympe stream reconnect (~2s freeze) that would
        # otherwise fire at the start of the subsequent SEARCH phase.
        if getattr(self, '_height_zoom_phase_entry', None) != phase_entry:
            self._height_zoom_phase_entry = phase_entry
            self._apply_zoom(1.25)
        try:
            h_cm = (float(tel.raw.get("height_cm"))
                    if tel and tel.raw.get("height_cm") is not None
                    else None)
        except (TypeError, ValueError):
            h_cm = None
        if h_cm is None:
            # No height telemetry -- can't drive a controlled climb.
            # Hold the drone briefly then move on.
            self._send_rc(0, 0, 0, 0)
            if now - self.state.phase_started_at > 5.0:
                self._advance_script("height: no telemetry, advancing")
            return
        drone_h = h_cm / 100.0

        # Dead time: hover without PD for height_warmup_s after phase entry so
        # the barometer can recover from pressure artefacts (e.g. after move_imu).
        warmup = float(getattr(cfg, "height_warmup_s", 0.0))
        if warmup > 0.0 and now - self.state.phase_started_at < warmup:
            self._send_rc(0, 0, 0, 0)
            with self.state.lock:
                self.state.note = (f"HEIGHT warmup: drone={drone_h:.2f}m  "
                                   f"target={target:.2f}m  "
                                   f"t={now - self.state.phase_started_at:.1f}s/{warmup:.1f}s")
            return

        e_h = target - drone_h
        u_ud = (0.0 if abs(e_h) < cfg.height_deadband_m
                else self.pd_height.step(e_h, now))
        self._send_rc(0, 0, int(u_ud), 0)

        in_band = abs(e_h) < cfg.height_deadband_m
        settled = False
        with self.state.lock:
            self.state.note = (f"HEIGHT: drone={drone_h:.2f}m  "
                               f"target={target:.2f}m  e={e_h:+.2f}m")
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.height_settle_time_s:
                    settled = True
            else:
                self.state.settle_began_at = None
        if settled:
            self._advance_script(f"height reached ({drone_h:.2f}m)")

    def _best_view_yaw_deg(self, tx: float, ty: float) -> Optional[float]:
        """Pick the arena-frame yaw that maximises the score of
        reference markers visible from ``(tx, ty)``.

        Score per marker = ``cos(off_normal) / max(0.5, distance)``,
        clipped to zero when:

        * the drone is on the back side of the marker (can't see the face),
        * the off-normal angle exceeds 80 deg (extreme oblique),
        * the marker bearing falls outside the camera horizontal FOV
          for the candidate yaw.

        Sweeps yaws every 5 deg over the full circle; returns None
        when ``arena_provider`` isn't set or the arena has no
        markers (caller falls back to "no yaw drive").
        """
        if self.get_arena is None:
            return None
        arena = self.get_arena()
        if arena is None or not arena.markers:
            return None
        cfg = self.cfg
        half_fov_rad = math.radians(float(cfg.camera_fov_h_deg) / 2.0)
        # Per-marker pre-computed (bearing, distance, normal_bearing)
        # in arena frame. Bearing is CW from arena +y so it lines up
        # with state.arena_yaw_deg without further conversion.
        # Marker normals (outward into the arena) per wall:
        #   front -> -y, back -> +y, left -> +x, right -> -x.
        wall_normal_arena = {
            "front": (0.0, -1.0),
            "back":  (0.0, +1.0),
            "left":  (+1.0, 0.0),
            "right": (-1.0, 0.0),
        }
        marker_data = []
        for m in arena.markers.values():
            dx = float(m.position_m[0]) - tx
            dy = float(m.position_m[1]) - ty
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue
            bearing = math.degrees(math.atan2(dx, dy))    # CW from +y
            n = wall_normal_arena.get(m.wall)
            if n is None:
                continue
            # Drone -> marker direction in arena (unit).
            ux, uy = dx / dist, dy / dist
            # Drone is in front of the face only when (drone-to-marker)
            # has a NEGATIVE projection onto the outward normal -- i.e.
            # the drone is on the side the normal points away from.
            # Then off_normal is the angle between -(drone->marker)
            # and the normal (both pointing roughly the same way).
            face_proj = ux * n[0] + uy * n[1]            # cos angle
            if face_proj >= 0.0:
                continue                                  # behind marker
            off_normal_rad = math.acos(max(-1.0, min(1.0, -face_proj)))
            if off_normal_rad > math.radians(80.0):
                continue                                  # too oblique
            marker_data.append((bearing, dist, off_normal_rad))
        if not marker_data:
            return None
        best_score = -1.0
        best_yaw = None
        for yaw_int in range(0, 360, 5):
            yaw_deg = float(yaw_int)
            score = 0.0
            for bearing, dist, off_normal_rad in marker_data:
                # Angular offset of marker bearing from drone heading,
                # wrapped to [-180, 180].
                d = bearing - yaw_deg
                d = (d + 540.0) % 360.0 - 180.0
                if abs(d) > math.degrees(half_fov_rad):
                    continue
                score += math.cos(off_normal_rad) / max(0.5, dist)
            if score > best_score:
                best_score = score
                # Wrap the chosen yaw to (-180, 180] so the
                # downstream PD error wrap-around behaves cleanly.
                best_yaw = ((yaw_deg + 540.0) % 360.0) - 180.0
        return best_yaw if best_score > 0.0 else None

    # ----------------------------------------------------------- goto (script)
    def _step_goto(self, tel: Optional[TelemetrySnapshot],
                  now: float) -> None:
        """Mission-script TO step: drive the drone to an arena-frame
        target ``(goto_target_x_m, goto_target_y_m, goto_target_z_m)``.

        Drives only when both the world-position estimate AND the
        arena-yaw estimate are fresh (vision_worker stamps both with
        ``*_updated_at`` whenever the active marker is visible). When
        either goes stale the controller falls back to a yaw-search
        in place -- the drone never flies open-loop on a dead fix.

        ``goto_target_z_m`` is None when the operator omitted the third
        argument: x/y are still driven, ud stays at 0 and the Anafi's
        onboard stabiliser holds whatever altitude the drone arrived at.
        """
        cfg = self.cfg
        with self.state.lock:
            tx = self.state.goto_target_x_m
            ty = self.state.goto_target_y_m
            tz = self.state.goto_target_z_m
            t_yaw = self.state.goto_target_yaw_deg
            wp = self.state.world_position_m
            wp_at = self.state.world_position_updated_at
            yaw_arena_live = self.state.arena_yaw_deg
            yaw_at = self.state.arena_yaw_updated_at
            hold_until = self.state.goto_hold_until

        # Freeze yaw_arena at GOTO entry to prevent IPPE branch-flip
        # jumps from reversing the body-frame decomposition mid-flight.
        # Use the first valid live estimate; fall back to live if never set.
        if (self._goto_frozen_yaw is None and yaw_arena_live is not None):
            self._goto_frozen_yaw = yaw_arena_live
        yaw_arena = (self._goto_frozen_yaw
                     if self._goto_frozen_yaw is not None
                     else yaw_arena_live)
        if tx is None or ty is None:
            self._advance_script("goto: no target set")
            return

        wp_age = (now - wp_at) if wp_at > 0.0 else None
        yaw_age = (now - yaw_at) if yaw_at > 0.0 else None
        wp_fresh = (wp is not None and wp_age is not None
                    and wp_age <= cfg.pose_max_age_s)
        yaw_fresh = (yaw_arena is not None and yaw_age is not None
                     and yaw_age <= cfg.pose_max_age_s)
        # Only require yaw freshness when a yaw target is set; without
        # a yaw target the drone tracks x/y only, and a stale arena_yaw
        # was resetting settle_began_at every tick, preventing settle.
        fresh = wp_fresh and (t_yaw is None or yaw_fresh)
        if not fresh:
            # Position lost — stop translation but keep rotating with
            # scan_yaw so arena markers come back into view.
            scan_yaw_stale = int(getattr(cfg, "goto_scan_yaw_rc", 0))
            self._send_rc(0, 0, 0, scan_yaw_stale, enforce_cfg_caps=False)
            with self.state.lock:
                self.state.settle_began_at = None
                age_txt = (f"{wp_age:.1f}s" if wp_age is not None
                           else "never")
                self.state.note = (f"GOTO: position stale ({age_txt}) "
                                   f"-- scanning for markers")
            return

        # Position-jump guard: if the measured position jumped more than
        # _GOTO_JUMP_MAX_M in one tick, it's a bad IPPE branch flip.
        # Stop the drone and yaw-scan for _GOTO_JUMP_REORIENT_S seconds
        # to re-acquire stable marker references.
        _GOTO_JUMP_MAX_M   = 2.0
        _GOTO_JUMP_REORIENT_S = 3.0
        scan_yaw_rc = int(getattr(cfg, "goto_scan_yaw_rc", 0))
        if self._goto_last_wp is not None and wp is not None:
            dx_jump = abs(wp[0] - self._goto_last_wp[0])
            dy_jump = abs(wp[1] - self._goto_last_wp[1])
            if math.hypot(dx_jump, dy_jump) > _GOTO_JUMP_MAX_M:
                self._goto_jump_until = now + _GOTO_JUMP_REORIENT_S
                print(f"[ctrl] GOTO: position jump "
                      f"{math.hypot(dx_jump,dy_jump):.2f}m — reorienting")
        if now < self._goto_jump_until:
            # Re-orient: stop translation, yaw slowly to find markers.
            self._send_rc(0, 0, 0, scan_yaw_rc, enforce_cfg_caps=False)
            with self.state.lock:
                self.state.settle_began_at = None
                self.state.note = (f"GOTO: jump detected — reorienting "
                                   f"({self._goto_jump_until - now:.1f}s)")
            return
        if wp is not None:
            self._goto_last_wp = tuple(wp)

        ex = tx - wp[0]
        ey = ty - wp[1]
        # Arena +y is the front wall; yaw_arena is CW from +y. Body
        # forward unit vector in arena = (sin(yaw), cos(yaw)); body
        # right = (cos(yaw), -sin(yaw)) (90 deg CW from forward).
        yaw_rad = math.radians(yaw_arena)
        err_fwd   = ex * math.sin(yaw_rad) + ey * math.cos(yaw_rad)
        err_right = ex * math.cos(yaw_rad) - ey * math.sin(yaw_rad)

        # Reuse the marker-relative PD gains. Both legs operate on
        # metres of error and are clipped to *_rc_max, so they're a
        # reasonable fit; the integrators were reset on entry by
        # _set_phase. The deadband is the GOTO-specific
        # ``goto_deadband_m`` (default 0.3m) rather than the tight
        # marker-relative ``distance_deadband_m`` (0.05m), because
        # TO targets an arena-frame point reached from far-wall
        # ArUco, where 5cm is not realistically achievable.
        u_fwd = (0.0 if abs(err_fwd) < cfg.goto_deadband_m
                 else self.pd_fwd.step(err_fwd, now))
        u_lat = (0.0 if abs(err_right) < cfg.goto_deadband_m
                 else self.pd_lat.step(err_right, now))

        # Optional altitude leg.
        e_h = 0.0
        u_ud = 0.0
        height_ok = True
        if tz is not None:
            try:
                h_cm = (float(tel.raw.get("height_cm"))
                        if tel and tel.raw.get("height_cm") is not None
                        else None)
            except (TypeError, ValueError):
                h_cm = None
            if h_cm is not None:
                drone_h = h_cm / 100.0
                e_h = tz - drone_h
                u_ud = (0.0 if abs(e_h) < cfg.height_deadband_m
                        else self.pd_height.step(e_h, now))
                height_ok = abs(e_h) < cfg.height_deadband_m
            # No height telemetry -- skip the leg, treat altitude as
            # already-OK rather than blocking the step forever.

        # Yaw leg disabled during normal flight.
        u_yaw = 0.0
        e_yaw = 0.0
        yaw_ok = True

        spd = max(0.0, min(1.0, float(getattr(cfg, "goto_speed_factor", 1.0))))
        scan_yaw_rc = int(getattr(cfg, "goto_scan_yaw_rc", 0))
        _SCAN_DUR_S    = 2.0   # seconds of yaw scan per pause
        _REALIGN_DUR_S = 1.5   # seconds of re-alignment after scan
        _CONF_THRESHOLD = 0.3  # trigger scan when confidence drops below this

        # Position RC values capped for scan/realign (half speed to avoid
        # overshooting while the yaw is changing the body frame).
        pos_lr  = int(u_lat * spd * 0.5)
        pos_fb  = int(u_fwd * spd * 0.5)
        pos_ud  = int(u_ud  * spd * 0.5)

        if self._goto_scan_state == 'scan':
            if now >= self._goto_scan_until:
                self._goto_scan_state = 'realign'
                self._goto_scan_until = now + _REALIGN_DUR_S
            else:
                # Yaw scan AND position hold simultaneously.
                self._send_rc(pos_lr, pos_fb, pos_ud, scan_yaw_rc,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (f"GOTO: scan pause "
                                       f"({self._goto_scan_until-now:.1f}s)")
                return
        elif self._goto_scan_state == 'realign':
            if now >= self._goto_scan_until:
                self._goto_scan_state = 'fly'
            else:
                if yaw_arena is not None:
                    target_heading = math.degrees(math.atan2(ex, ey))
                    e_align = ((target_heading - yaw_arena + 540.0)
                               % 360.0 - 180.0)
                    align_rc = max(-scan_yaw_rc,
                                   min(scan_yaw_rc, int(e_align * 0.5)))
                else:
                    align_rc = 0
                # Realign yaw AND position hold simultaneously.
                self._send_rc(pos_lr, pos_fb, pos_ud, align_rc,
                              enforce_cfg_caps=False)
                with self.state.lock:
                    self.state.note = (f"GOTO: realigning "
                                       f"({self._goto_scan_until-now:.1f}s)")
                return
        else:  # 'fly'
            # Trigger scan when confidence drops below threshold.
            with self.state.lock:
                conf = self.state.world_position_confidence
            if conf < _CONF_THRESHOLD:
                self._goto_scan_state = 'scan'
                self._goto_scan_until = now + _SCAN_DUR_S
                # Send position hold immediately (no free drift at trigger).
                self._send_rc(pos_lr, pos_fb, pos_ud, 0)
                with self.state.lock:
                    self.state.note = (f"GOTO: low confidence ({conf:.2f})"
                                       f" — scanning")
                return
            # Normal flight — no continuous yaw.
            self._send_rc(int(u_lat * spd), int(u_fwd * spd),
                          int(u_ud * spd), 0)

        e_xy = math.hypot(ex, ey)
        # Settle uses per-axis box (|ex| < deadband AND |ey| < deadband)
        # instead of Euclidean radius so the tolerance is symmetric in x/y.
        in_band = (abs(ex) < cfg.goto_deadband_m
                   and abs(ey) < cfg.goto_deadband_m
                   and height_ok and yaw_ok)
        settled = False
        with self.state.lock:
            z_txt = (f"  z_e={e_h:+.2f}m" if tz is not None else "")
            yaw_txt = (f"  yaw_e={e_yaw:+.0f}deg"
                       if t_yaw is not None else "")
            mode_txt = ("station-keep" if hold_until is not None
                        else "GOTO")
            self.state.note = (
                f"{mode_txt}: at ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})m "
                f"-> ({tx:.2f},{ty:.2f}"
                f"{(',' + format(tz, '.2f')) if tz is not None else ''})m"
                f"  ex={ex:+.2f}m ey={ey:+.2f}m{z_txt}{yaw_txt}")
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.approach_settle_time_s:
                    settled = True
            else:
                self.state.settle_began_at = None
        if hold_until is not None:
            # HOOVER-after-TO branch: stay until the timer expires;
            # ignore the settle condition so a brief excursion doesn't
            # cut the hover short.
            if now >= hold_until:
                self._advance_script(
                    f"goto-hold complete at "
                    f"({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})m")
        elif settled:
            if self._goto_on_settle_restart_mission:
                self._goto_on_settle_restart_mission = False
                self._goto_on_settle_phase = None
                print("[ctrl] GOTO settled at arena marker — restarting mission")
                with self.state.lock:
                    self.state.mission_step_idx = -1
                    self.state.last_completed_step_kind = None
                self._advance_script("arena-nav recovery: mission restart")
            elif self._goto_on_settle_phase is not None:
                next_phase = self._goto_on_settle_phase
                self._goto_on_settle_phase = None
                self._set_phase(
                    next_phase,
                    f"goto reached ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})m"
                    f" → resuming {next_phase.value}")
            else:
                self._advance_script(
                    f"goto reached ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})m")

    # --------------------------------------------------------- dance (script)
    def _step_dance(self, tel: Optional[TelemetrySnapshot],
                   now: float) -> None:
        """Mission-script DANCE step: programmed RC routine, bounded
        within ``cfg.dance_radius_m`` of the entry position.
        """
        cfg = self.cfg
        with self.state.lock:
            until = self.state.dance_until
            mode = (self.state.dance_mode or "wobble")
            origin_xy = self.state.dance_origin_xy_m
            origin_h = self.state.dance_origin_height_m
            started = self.state.phase_started_at
        if until is None:
            self._advance_script("dance: no timer set")
            return
        if now >= until:
            self._send_rc(0, 0, 0, 0)
            self._advance_script("dance complete")
            return

        elapsed = max(0.0, now - started)
        # Mode-specific RC pattern.
        if mode == "spin":
            lr = 0; fb = 0; ud = 0
            yaw = int(cfg.yaw_rc_max)
        elif mode == "random":
            amp_lr  = max(2, cfg.lat_rc_max // 2)
            amp_fb  = max(2, cfg.fwd_rc_max // 2)
            amp_ud  = max(4, cfg.ud_rc_max // 4)
            amp_yaw = max(4, cfg.yaw_rc_max // 2)
            lr  = int(amp_lr  * math.sin(2 * math.pi * 0.40 * elapsed))
            fb  = int(amp_fb  * math.sin(2 * math.pi * 0.30 * elapsed + 1.0))
            ud  = int(amp_ud  * math.sin(2 * math.pi * 0.50 * elapsed + 2.0))
            yaw = int(amp_yaw * math.sin(2 * math.pi * 0.25 * elapsed + 0.5))
        else:  # wobble (default)
            amp_yaw = int(cfg.yaw_rc_max)
            amp_ud  = max(4, cfg.ud_rc_max // 4)
            lr = 0; fb = 0
            yaw = int(amp_yaw * math.sin(2 * math.pi * 0.5 * elapsed))
            ud  = int(amp_ud  * math.sin(2 * math.pi * 1.0 * elapsed))

        # Horizontal radius bound. PositionEstimator is dead-reckoning
        # NED (vgx -> vN, vgy -> vE).
        r_xy = 0.0
        if origin_xy is not None:
            x, y = self.position.position_m
            dx_e = x - origin_xy[0]
            dy_n = y - origin_xy[1]
            r_xy = math.hypot(dx_e, dy_n)
            if r_xy > cfg.dance_radius_m:
                yaw_deg = None
                if tel is not None:
                    try:
                        yaw_deg = float(tel.raw.get("yaw"))
                    except (TypeError, ValueError):
                        yaw_deg = None
                if yaw_deg is not None:
                    # Inward direction in world (NED): (-dy_n, -dx_e).
                    in_v_fwd, in_v_right = self._world_to_body(
                        -dy_n, -dx_e, yaw_deg)
                    # Zero any commanded channel that's pointing
                    # outward (sign opposite to the inward vector).
                    if fb * in_v_fwd < 0:
                        fb = 0
                    if lr * in_v_right < 0:
                        lr = 0
                    # Add an inward correction proportional to the
                    # excess radius, capped at the per-channel max.
                    excess = min(1.0,
                                 (r_xy - cfg.dance_radius_m)
                                 / max(0.1, cfg.dance_radius_m))
                    push_fb = int(cfg.fwd_rc_max * excess
                                  * (1 if in_v_fwd >= 0 else -1))
                    push_lr = int(cfg.lat_rc_max * excess
                                  * (1 if in_v_right >= 0 else -1))
                    fb += push_fb
                    lr += push_lr
                else:
                    # No yaw telemetry -- can't transform to body.
                    # Conservatively zero horizontal channels so we
                    # don't drift further out.
                    lr = 0; fb = 0

        # Vertical bound.
        h_now = None
        if tel is not None:
            try:
                h_cm_v = (float(tel.raw.get("height_cm"))
                          if tel.raw.get("height_cm") is not None else None)
                if h_cm_v is not None:
                    h_now = h_cm_v / 100.0
            except (TypeError, ValueError):
                pass
        dh = 0.0
        if h_now is not None and origin_h is not None:
            dh = h_now - origin_h
            if abs(dh) > cfg.dance_radius_m:
                # If ud is pushing further from origin (same sign as
                # dh), invert it to drive back inward.
                if ud * dh > 0:
                    ud = -int(min(cfg.ud_rc_max, abs(ud)) or cfg.ud_rc_max // 4)

        self._send_rc(lr, fb, ud, yaw)
        with self.state.lock:
            self.state.note = (f"DANCE {mode}  r={r_xy:.2f}m  dh={dh:+.2f}m  "
                               f"t={elapsed:.1f}/{(until - started):.1f}s")

    # ------------------------------------------------- enemy drone avoidance

    # Phases that may be interrupted by a drone threat.
    _THREAT_INTERRUPTIBLE = frozenset({
        Phase.SEARCH, Phase.ALIGN, Phase.HEIGHT_ALIGN,
        Phase.APPROACH, Phase.HOLD, Phase.IDLE, Phase.HEIGHT,
        Phase.GOTO, Phase.SCOUT,
    })
    # Blob width (px) threshold that triggers threat mode.
    _THREAT_BLOB_PX = 150
    # Seconds the threat must persist before evade manoeuvre starts.
    _THREAT_OBSERVE_S = 5.0
    # RC magnitude for lateral evade (toward arena centre).
    _THREAT_EVADE_RC = 12

    def _check_drone_threat(self, now: float) -> None:
        """Called each control tick before phase dispatch.

        Threat response is currently disabled — detected enemy drones are
        shown in the camera overlay only and do not affect flight.
        """
        return
        if self.get_drone_threat is None:  # noqa: unreachable
            return
        threat_w, _threat_dx = self.get_drone_threat()
        phase = self.state.phase

        if threat_w >= self._THREAT_BLOB_PX:
            if phase in self._THREAT_INTERRUPTIBLE:
                # Suspend current phase.
                with self.state.lock:
                    self.state.threat_suspended_phase = phase
                    self.state.threat_suspended_step_idx = self.state.mission_step_idx
                    self.state.threat_observe_since = now
                self._set_phase(Phase.THREAT_OBSERVE,
                                f"enemy drone {threat_w}px wide")
            elif phase == Phase.THREAT_OBSERVE:
                # Still threatened — keep counting (handled in _step_threat_observe)
                pass
            elif phase == Phase.THREAT_EVADE:
                # Still evading — let _step_threat_evade handle it
                pass
        else:
            # Threat cleared
            if phase in (Phase.THREAT_OBSERVE, Phase.THREAT_EVADE):
                self._resume_after_threat(now)

    def _resume_after_threat(self, now: float) -> None:
        """Restore the suspended phase. If the target marker is no longer
        visible, re-enter the current mission step from scratch so SEARCH
        picks up the marker again."""
        with self.state.lock:
            saved_phase = self.state.threat_suspended_phase
            saved_idx   = self.state.threat_suspended_step_idx
            script      = list(self.state.mission_script)
            self.state.threat_observe_since = None
            self.state.threat_suspended_phase = None
            self.state.threat_suspended_step_idx = -1

        pose = self.get_pose()
        marker_visible = (pose is not None)

        if not marker_visible and 0 <= saved_idx < len(script):
            # Target marker lost — restart current step.
            print("[ctrl] threat cleared, marker lost — restarting step "
                  f"{saved_idx}: {script[saved_idx].kind}")
            with self.state.lock:
                self.state.mission_step_idx = saved_idx
                self.state.current_step_kind = script[saved_idx].kind
            self._apply_step_to_phase(script[saved_idx], "threat cleared, marker lost")
        elif saved_phase is not None:
            # Marker still visible (or no step to re-enter) — resume.
            print(f"[ctrl] threat cleared — resuming {saved_phase.value}")
            self._set_phase(saved_phase, "threat cleared")
        else:
            # No suspended phase recorded (shouldn't happen).
            self._set_phase(Phase.SEARCH, "threat cleared, no saved phase")

    def _step_threat_observe(self,
                             tel: Optional['TelemetrySnapshot'],
                             now: float) -> None:
        """Hover in place and observe. After _THREAT_OBSERVE_S transition
        to THREAT_EVADE."""
        self._send_rc(0, 0, 0, 0)
        with self.state.lock:
            since = self.state.threat_observe_since or now
        threat_w, _dx = self.get_drone_threat() if self.get_drone_threat else (0, 0.0)
        with self.state.lock:
            self.state.note = (
                f"THREAT observe {now - since:.1f}/{self._THREAT_OBSERVE_S:.0f}s "
                f"blob={threat_w}px")
        if now - since >= self._THREAT_OBSERVE_S:
            with self.state.lock:
                self.state.threat_observe_since = now  # reset timer for evade
            self._set_phase(Phase.THREAT_EVADE, "threat persists — evading")

    def _step_threat_evade(self,
                           tel: Optional['TelemetrySnapshot'],
                           now: float) -> None:
        """Fly laterally in the direction opposite to the enemy drone's
        movement (camera-frame dx). If dx ≈ 0 (stationary threat) fall
        back to rc_lr = +_THREAT_EVADE_RC (right)."""
        threat_w, threat_dx = (self.get_drone_threat()
                                if self.get_drone_threat else (0, 0.0))
        # Enemy moves right (dx > 0) → we go left (negative rc_lr), and vice-versa.
        # Dead-zone of ±1 px/frame to ignore noise when blob is nearly still.
        if threat_dx > 1.0:
            rc_lr = -self._THREAT_EVADE_RC   # enemy rightward → evade left
        elif threat_dx < -1.0:
            rc_lr = self._THREAT_EVADE_RC    # enemy leftward  → evade right
        else:
            rc_lr = self._THREAT_EVADE_RC    # stationary: default right

        self._send_rc(rc_lr, 0, 0, 0)
        with self.state.lock:
            since = self.state.threat_observe_since or now
            self.state.note = (
                f"THREAT evade rc_lr={rc_lr:+d}  blob={threat_w}px  "
                f"dx={threat_dx:+.1f}px  t={now - since:.1f}s")

    # --------------------------------------------------- mission script driver
    def _terminate_script(self, reason: str) -> None:
        """Cut the script short and land. Used when an APPROACH
        sub-stage (SEARCH) gives up without finding the marker --
        we don't want the post-LAND _advance_script to walk into
        the next script step as if APPROACH had succeeded.
        """
        with self.state.lock:
            self.state.mission_script = []
            self.state.mission_step_idx = 0
            # Mark the current step as if we had executed a LAND so
            # the post-LAND _advance_script("landed") routes to DONE.
            self.state.current_step_kind = "LAND"
            self.state.last_completed_step_kind = None
        self._set_phase(Phase.LAND, reason)

    def _wait_attack_sibling_of(self, target_id: int) -> Optional[int]:
        """Compute the sibling face id for a WAIT_AND_ATTACK target.
        SDC26 convention: each target box has two opposing-team faces
        whose ids differ by ``cfg.target_marker_sibling_offset``
        (default 10 -> 31↔41, 36↔46, ...). Returns None if no in-range
        sibling exists (target is outside the target-marker range, or
        both target±offset are out of range)."""
        cfg = self.cfg
        offset = int(getattr(cfg, "target_marker_sibling_offset", 10))
        if offset <= 0:
            return None
        low, high = int(cfg.target_marker_id_min), int(cfg.target_marker_id_max)
        if not (low <= target_id <= high):
            return None
        for cand in (target_id - offset, target_id + offset):
            if low <= cand <= high:
                return cand
        return None

    def _wait_attack_pre_tick(self) -> None:
        """Per-tick swap of ``state.active_marker_id`` for a running
        WAIT_AND_ATTACK step. Preference order:
          1. If the target id is currently visible → active = target.
          2. Else if the sibling id is visible → active = sibling.
          3. Else: no change (the smoother carries the previous active
             pose forward through brief marker losses).

        Called from the main control loop BEFORE phase dispatch so
        _step_search / _step_align / _step_approach / _step_hold all
        see the freshly-routed active id on the same tick.

        No-ops when no WAIT_AND_ATTACK is active (wait_attack_target_id
        is None) OR when a marker-lost recovery hop is in progress
        (_approach_on_settle_restart_mission == True). The recovery
        hook deliberately switches active_marker_id to a wall marker
        for re-localisation; we must not yank it back to target/sibling
        mid-hop, or the recovery's APPROACH would chase the wrong
        marker and the mission-restart trigger would never fire."""
        if self._approach_on_settle_restart_mission:
            return
        with self.state.lock:
            target = self.state.wait_attack_target_id
            sibling = self.state.wait_attack_sibling_id
            active = self.state.active_marker_id
            visible = (set(self.state.visible_marker_ids)
                       if self.state.visible_marker_ids else set())
        if target is None:
            return
        if target in visible:
            new_active = target
        elif sibling is not None and sibling in visible:
            new_active = sibling
        else:
            return  # neither visible; keep current active so smoother
                    # can carry the previous pose through a brief loss
        if new_active != active:
            with self.state.lock:
                self.state.active_marker_id = new_active

    def _advance_script(self, reason: str) -> None:
        """Consume the current script step and load the next one. If
        the script is exhausted, terminate with LAND (or DONE if the
        last step was already LAND).
        """
        with self.state.lock:
            self.state.last_completed_step_kind = self.state.current_step_kind
            self.state.mission_step_idx += 1
            idx = self.state.mission_step_idx
            script = list(self.state.mission_script)
            last_done = self.state.last_completed_step_kind
        if idx >= len(script):
            if last_done == "LAND":
                self._set_phase(Phase.DONE,
                                f"script complete ({reason})")
            else:
                # Safety LAND for scripts that don't end with LAND.
                self._set_phase(Phase.LAND,
                                f"script complete -- safety land ({reason})")
            return
        step = script[idx]
        with self.state.lock:
            self.state.current_step_kind = step.kind
        self._apply_step_to_phase(step, reason)

    def _apply_step_to_phase(self, step: Step, reason: str) -> None:
        """Configure cfg overrides + per-step state for ``step`` and
        call _set_phase with the appropriate initial phase."""
        cfg = self.cfg
        note = f"script[{step.line_no}] {step.kind} ({reason})"
        # AWAIT installs an early-exit marker id; every other step
        # clears it so a previous AWAIT can't accidentally short-circuit
        # subsequent IDLE / HOLD ticks. WAIT_AND_ATTACK does NOT
        # piggyback on this hook (the previous implementation did, but
        # it skipped the search phase); see the WAIT_AND_ATTACK branch
        # below for the SEARCH-entering + sibling-aware approach.
        with self.state.lock:
            self.state.await_marker_id = (int(step.marker_id)
                                          if step.kind == "AWAIT"
                                          else None)
            # Clear wait_attack target/sibling tracking on transition
            # OUT of WAIT_AND_ATTACK. The pre-tick swap consults these
            # to route active_marker_id, and the approach-settle gate
            # uses them to refuse advancement while we're parked on
            # the sibling face.
            if step.kind != "WAIT_AND_ATTACK":
                self.state.wait_attack_target_id = None
                self.state.wait_attack_sibling_id = None
        if step.kind == "TAKEOFF":
            self._set_phase(Phase.TAKEOFF, note)
            try:
                self.api.takeoff()
            except DroneApiError as e:
                self._abort(f"takeoff API error: {e}")
            return
        if step.kind == "APPROACH":
            # Per-step runtime targets live in MissionState. cfg.target_*
            # stays untouched so it remains the authoritative source of
            # parse-time defaults (defaults_from_cfg) and the user's
            # tune-page values. Phases read from state.target_*.
            # A real mission APPROACH/GO_HOME step: never the recovery hop, so
            # clear the recovery restart hook (only _recover_via_central_marker
            # sets it).
            self._approach_on_settle_restart_mission = False
            with self.state.lock:
                self.state.active_marker_id = int(step.marker_id)
                self.state.target_distance_m = float(step.distance)
                # Loose arrival band: GO_HOME carries step.arrive_tol_m; a
                # capture APPROACH may carry its own step.approach_dist_tol_m
                # (e.g. 0.2 m). Either widens the arrival band; None -> tight
                # cfg.distance_deadband_m.
                _arrive_tol = (step.arrive_tol_m
                               if step.arrive_tol_m is not None
                               else getattr(step, "approach_dist_tol_m", None))
                self.state.target_distance_tol_m = (
                    float(_arrive_tol) if _arrive_tol is not None else None)
                # POSITIONING (hold altitude, no climb to marker) is GO_HOME-only
                # -- keyed on arrive_tol_m, NOT on the band above, so a capture
                # APPROACH with approach_dist_tol_m STILL vision-aligns to the
                # marker's height.
                self.state.approach_positioning = (step.arrive_tol_m is not None)
                # GO_HOME may specify an arrival heading (±80°, ±12° tol).
                # Plain APPROACH always closes head-on (0°); HOLD inherits
                # this setpoint via state.target_relative_heading_deg.
                self.state.target_relative_heading_deg = (
                    float(step.arrive_hdg_deg)
                    if step.arrive_hdg_deg is not None else 0.0)
                self.state.arrive_hdg_tol_deg = (
                    15.0 if step.arrive_hdg_deg is not None else None)
                # APPROACH's optional 3rd arg: a wide yaw-settle tolerance that
                # makes the approach advance the instant the target DISTANCE is
                # reached (no head-on settle, no settle-time wait) so the next
                # step (e.g. FB_UD_IMU over the box) fires immediately, from an
                # angle. None -> tight head-on settle.
                self.state.arrive_yaw_tol_deg = (
                    float(step.arrive_yaw_tol_deg)
                    if getattr(step, "arrive_yaw_tol_deg", None) is not None
                    else None)
                # Snapshot the altitude this capture-APPROACH onto a target box
                # begins at, so a lost-box SEARCH can descend back to it instead
                # of yawing from whatever height a brief HEIGHT_ALIGN or a
                # GO_HOME recovery anchor dragged us up to. Capture ONCE per
                # distinct box id: a recovery restart re-runs this branch from
                # mission_step_idx=-1, and must NOT overwrite the original (low)
                # height with the already-climbed one. arrive_tol_m is None for a
                # capture APPROACH (a loose GO_HOME carries it) -- GO_HOME homes
                # at the current altitude anyway, so it never needs this.
                _mid = int(step.marker_id)
                _is_box = (step.arrive_tol_m is None
                           and self.cfg.target_marker_id_min
                           <= _mid <= self.cfg.target_marker_id_max)
                if _is_box:
                    if self._box_height_marker != _mid:
                        _tel0 = self.state.last_telemetry
                        self._box_height_marker = _mid
                        self.state.approach_start_height_m = (
                            _tel0.height_cm / 100.0
                            if _tel0 is not None
                            and _tel0.height_cm is not None else None)
                else:
                    self._box_height_marker = None
                    self.state.approach_start_height_m = None
            self._set_phase(Phase.SEARCH,
                            note + f" id={int(step.marker_id)}"
                                   f" d={float(step.distance):g}m"
                            + (f" tol={float(step.arrive_tol_m):g}m"
                               if step.arrive_tol_m is not None else ""))
            return
        if step.kind == "HOOVER":
            with self.state.lock:
                last = self.state.last_completed_step_kind
                tx = self.state.goto_target_x_m
                ty = self.state.goto_target_y_m
            if last == "APPROACH":
                with self.state.lock:
                    self.state.hold_time_s = float(step.seconds)
                self._set_phase(Phase.HOLD,
                                note + f" station-keep {step.seconds:g}s")
            elif last == "TO" and tx is not None and ty is not None:
                # World-frame analogue of HOLD: stay on the TO target
                # for ``step.seconds``. _step_goto switches its advance
                # condition from "settled" to "timer expired" when
                # goto_hold_until is set.
                with self.state.lock:
                    self.state.goto_hold_until = (time.monotonic()
                                                   + float(step.seconds))
                self._set_phase(Phase.GOTO,
                                note + f" station-keep at"
                                       f" ({tx:.2f},{ty:.2f}) "
                                       f"{step.seconds:g}s")
            else:
                with self.state.lock:
                    self.state.idle_until = (time.monotonic()
                                              + float(step.seconds))
                self._set_phase(Phase.IDLE,
                                note + f" idle {step.seconds:g}s")
            return
        if step.kind == "AWAIT":
            # Same shape as HOOVER (HOLD if previous was APPROACH else
            # IDLE) plus an early-exit when state.await_marker_id (set
            # above) appears in state.visible_marker_ids -- the per-tick
            # check lives in _step_hold / _step_idle.
            with self.state.lock:
                last = self.state.last_completed_step_kind
            if last == "APPROACH":
                with self.state.lock:
                    self.state.hold_time_s = float(step.seconds)
                self._set_phase(Phase.HOLD,
                                note + f" station-keep,"
                                       f" await marker {step.marker_id}"
                                       f" timeout {step.seconds:g}s")
            else:
                with self.state.lock:
                    self.state.idle_until = (time.monotonic()
                                              + float(step.seconds))
                self._set_phase(Phase.IDLE,
                                note + f" idle, await marker {step.marker_id}"
                                       f" timeout {step.seconds:g}s")
            return
        if step.kind == "PAUSE":
            # Unconditional IDLE for ``seconds``. No early-exit, no
            # marker tracking, no station-keeping. Mirrors the
            # HOOVER-without-prior-APPROACH branch above.
            with self.state.lock:
                self.state.idle_until = (time.monotonic()
                                          + float(step.seconds))
            self._set_phase(Phase.IDLE,
                            note + f" pause {step.seconds:g}s")
            return
        if step.kind == "WAIT_AND_ATTACK":
            # Behaves like APPROACH on the target id, but with two
            # additions:
            #   1. The sibling-face id (target ± cfg.target_marker_sibling_offset,
            #      clipped to the target-marker range) is also tracked.
            #      The pre-tick swap in _wait_attack_pre_tick keeps
            #      active_marker_id pointing at whichever id is
            #      currently visible (preferring the target). So a
            #      SEARCH yaw-spin acquires the box even when the wrong
            #      face is shown, and the APPROACH chain proceeds on
            #      the sibling face to bring the drone to standoff.
            #   2. The approach-settle gate refuses to advance the
            #      script while active_marker_id != wait_attack_target_id.
            #      So when the drone settles at parsed-distance on the
            #      sibling face, the script does NOT advance — instead
            #      we hold the position. As soon as the box flips
            #      (target id becomes visible, sibling id does not),
            #      the swap puts active = target, the PD re-targets
            #      the (now-target) face, the drone re-settles, and
            #      the gate advances the script.
            mid = int(step.marker_id)
            sibling = self._wait_attack_sibling_of(mid)
            self._approach_on_settle_restart_mission = False
            with self.state.lock:
                self.state.active_marker_id = mid
                self.state.wait_attack_target_id = mid
                self.state.wait_attack_sibling_id = sibling
                self.state.target_distance_m = float(step.distance)
                _dist_tol = getattr(step, "approach_dist_tol_m", None)
                self.state.target_distance_tol_m = (
                    float(_dist_tol) if _dist_tol is not None else None)
                self.state.approach_positioning = False
                self.state.target_relative_heading_deg = 0.0
                self.state.arrive_hdg_tol_deg = None
                _yaw_tol = getattr(step, "arrive_yaw_tol_deg", None)
                self.state.arrive_yaw_tol_deg = (
                    float(_yaw_tol) if _yaw_tol is not None else None)
                # Box-height capture mirrors the APPROACH branch so
                # SEARCH's lost-box descent (which keys on
                # active_marker_id being in the target-marker range)
                # has a snapshot to descend back to. The sibling face
                # sits at the same physical height by construction,
                # so we don't need to re-capture on swap.
                _is_box = (cfg.target_marker_id_min
                           <= mid <= cfg.target_marker_id_max)
                if _is_box and self._box_height_marker != mid:
                    _tel0 = self.state.last_telemetry
                    self._box_height_marker = mid
                    self.state.approach_start_height_m = (
                        _tel0.height_cm / 100.0
                        if _tel0 is not None
                        and _tel0.height_cm is not None else None)
                elif not _is_box:
                    self._box_height_marker = None
                    self.state.approach_start_height_m = None
            sibling_str = f", sibling {sibling}" if sibling is not None else ""
            self._set_phase(Phase.SEARCH,
                            note + f" id={mid}{sibling_str}"
                                   f" d={float(step.distance):g}m"
                                   f" (hold on sibling face)")
            return
        if step.kind == "REPEAT":
            # Loop back to the first non-TAKEOFF step in the script
            # without landing in between. The parser guarantees REPEAT
            # is the last step AND has at least one non-TAKEOFF step
            # before it, so loop_to < idx_now is always true on a
            # well-formed script; the runtime guard below is
            # belt-and-suspenders for the operator who hand-edits the
            # in-memory script via the splice path.
            with self.state.lock:
                script = list(self.state.mission_script)
                idx_now = self.state.mission_step_idx
            loop_to = 0
            while loop_to < len(script) and script[loop_to].kind == "TAKEOFF":
                loop_to += 1
            if loop_to >= idx_now or loop_to >= len(script):
                self._set_phase(Phase.LAND,
                                note + " (REPEAT with no viable loop target"
                                       " — landing instead)")
                return
            with self.state.lock:
                # _advance_script increments mission_step_idx by 1 then
                # loads script[idx], so target idx = loop_to - 1.
                self.state.mission_step_idx = loop_to - 1
                # Clear last_completed_step_kind so the safety-LAND
                # branch (last == LAND -> DONE) can't fire on the next
                # advance just because the immediately-prior real step
                # happened to be LAND.
                self.state.last_completed_step_kind = None
            self._advance_script(
                f"REPEAT: looping to script[{loop_to}]"
                f" (kind={script[loop_to].kind})")
            return
        if step.kind == "RC":
            # Raw RC step (LR / FB / UD / RC): pin sticks for `seconds`,
            # then enter the brake phase. The FC ceiling and arena
            # guards still apply because _send_rc goes through the
            # same /api/rc path the operator's joystick uses.
            with self.state.lock:
                self.state.rc_step_lr = int(step.rc_lr)
                self.state.rc_step_fb = int(step.rc_fb)
                self.state.rc_step_ud = int(step.rc_ud)
                self.state.rc_step_yaw = int(step.rc_yaw)
                self.state.rc_step_until = (time.monotonic()
                                             + float(step.seconds))
                # Reset brake state so the new step starts in the
                # DRIVE phase, even if the previous one was mid-brake.
                self.state.rc_step_braking = False
                self.state.rc_step_brake_started = None
            tag = (f"fb={step.rc_fb}" if step.rc_fb else
                   f"ud={step.rc_ud}" if step.rc_ud else
                   f"yaw={step.rc_yaw}" if step.rc_yaw else
                   f"lr={step.rc_lr}")
            self._set_phase(Phase.RC,
                            note + f" {tag} {step.seconds:g}s")
            return
        if step.kind == "AUTO_ATTACK":
            # Reactive end-to-end attack. Seed the state machine and
            # hand control to _step_auto_attack (climb → go_target →
            # over_rise → over_forward → go_home → land).
            #
            # Resolve each marker's arena (x,y) from the ACTIVE arena
            # config (loaded from ~/.marker_mission/active_arena_config.json
            # via get_arena()). The explicit coords in the step are only
            # a fallback for markers NOT present in that config — this is
            # what lets a bare `AUTO_ATTACK <tid> <hid>` work in any
            # arena: the FC looks the positions up itself.
            arena = self.get_arena() if self.get_arena is not None else None

            def _resolve_xy(mid, fx, fy):
                if (arena is not None and arena.markers
                        and int(mid) in arena.markers):
                    p = arena.markers[int(mid)].position_m
                    return (float(p[0]), float(p[1]), "config")
                if fx is not None and fy is not None:
                    return (float(fx), float(fy), "coords")
                return (None, None, "UNRESOLVED")

            tgt_x, tgt_y, tgt_src = _resolve_xy(
                step.marker_id, step.world_x, step.world_y)
            home_x, home_y, home_src = _resolve_xy(
                step.aa_home_marker_id, step.aa_home_x, step.aa_home_y)

            if home_x is None:
                # Can't navigate home without a home position — abort
                # rather than fly blind.
                self._advance_script(
                    f"auto_attack: home marker {step.aa_home_marker_id} "
                    f"not in arena config and no coords given — aborting")
                return

            with self.state.lock:
                self.state.aa_target_marker = int(step.marker_id)
                self.state.aa_target_xy = (
                    (tgt_x, tgt_y) if tgt_x is not None else None)
                self.state.aa_home_marker = int(step.aa_home_marker_id)
                self.state.aa_home_xy = (home_x, home_y)
                self.state.aa_altitude_m = (
                    float(step.aa_altitude_m)
                    if step.aa_altitude_m is not None else AA_CRUISE_ALT_M)
                self.state.aa_approach_speed = (
                    int(step.aa_approach_speed)
                    if step.aa_approach_speed is not None else AA_FB_MAX)
                self.state.aa_substate = "climb"
                self.state.aa_substate_started = time.monotonic()
                self.state.active_marker_id = int(step.marker_id)
                self.state.last_pose = None
            tgt_pos_str = (f"({tgt_x:g},{tgt_y:g})" if tgt_x is not None
                           else "vision-only")
            self._set_phase(
                Phase.AUTO_ATTACK,
                note + f" target={step.marker_id}@{tgt_pos_str}/{tgt_src} "
                       f"home={step.aa_home_marker_id}@"
                       f"({home_x:g},{home_y:g})/{home_src} "
                       f"alt={self.state.aa_altitude_m:g} "
                       f"spd={self.state.aa_approach_speed}"
            )
            print(f"[AUTO_ATTACK] resolved target={step.marker_id} "
                  f"{tgt_pos_str} ({tgt_src}), home={step.aa_home_marker_id} "
                  f"({home_x:g},{home_y:g}) ({home_src})", flush=True)
            return
        if step.kind == "FB_BRAKE":
            # Open-loop forward stick + vision-triggered brake.
            # Reuses the RC step's brake-phase machinery: when the
            # vision threshold trips OR the timeout expires, we flip
            # rc_step_braking=True and the next tick zeros the sticks
            # + waits for IMU velocity to settle.
            with self.state.lock:
                self.state.active_marker_id = int(step.marker_id)
                self.state.target_distance_m = float(step.distance)
                self.state.rc_step_lr = 0
                self.state.rc_step_fb = int(step.rc_fb)
                self.state.rc_step_ud = 0
                self.state.rc_step_yaw = 0
                # rc_step_until is the HARD TIMEOUT (not the drive
                # duration). Vision / world-pos normally trip the
                # brake long before this; the deadline only fires
                # if neither pipeline says we've arrived.
                self.state.rc_step_until = (time.monotonic()
                                             + float(step.seconds))
                self.state.rc_step_braking = False
                self.state.rc_step_brake_started = None
                # Stash optional world-pos hint from the script.
                # _step_fb_brake will prefer arena.markers[id] when
                # available, then fall back to these, then to timeout.
                self.state.fb_brake_world_x = (float(step.world_x)
                                                if step.world_x is not None
                                                else None)
                self.state.fb_brake_world_y = (float(step.world_y)
                                                if step.world_y is not None
                                                else None)
                # Clear last_pose so a stale pose from the previous
                # APPROACH (e.g. on the target marker) can't trip
                # this brake instantly. The vision loop populates
                # last_pose with the new active marker on its next
                # frame; until then _step_fb_brake just drives.
                self.state.last_pose = None
            self._set_phase(
                Phase.FB_BRAKE,
                note + f" marker={step.marker_id} stop={step.distance:g}m "
                       f"fb={step.rc_fb} timeout={step.seconds:g}s"
            )
            return
        if step.kind == "SCOUT":
            # Slow 360° yaw spin for visual situational awareness.
            # Drive phase uses Phase.SCOUT (separate handler tracks
            # cumulative yaw against SCOUT_TARGET_DEG); brake phase
            # reuses the rc_step_braking flag once the drive is done,
            # so the next step starts from a clean hover.
            with self.state.lock:
                self.state.scout_started_at = time.monotonic()
                self.state.scout_last_yaw_deg = None
                self.state.scout_accumulated_deg = 0.0
                self.state.rc_step_braking = False
                self.state.rc_step_brake_started = None
            self._set_phase(
                Phase.SCOUT,
                note + f" yaw_stick={int(self.cfg.scout_yaw_stick)} "
                       f"target=±{SCOUT_TARGET_DEG:g}°"
            )
            return
        if step.kind == "MOVE_IMU":
            # Closed-loop FB_IMU / LR_IMU / UD_IMU step. Calls
            # api.move (Olympe moveBy) synchronously; the firmware
            # handles accel/cruise/decel and stops at the target.
            # Symmetric +/- (unlike the open-loop *_RC family that
            # rides Anafi's asymmetric velocity controller).
            direction = step.move_direction or "forward"
            meters = float(step.move_distance_m or 0.0)
            cm = int(round(meters * 100.0))
            self._set_phase(
                Phase.MOVE_IMU,
                note + f" {direction} {meters:g}m ({cm}cm)"
            )
            if cm <= 0:
                self._advance_script(f"move_imu {direction} 0cm — no-op")
                return
            try:
                self.api.move(direction, cm)
                self._advance_script(
                    f"move_imu complete ({direction} {meters:g}m)"
                )
            except DroneApiError as e:
                print(f"[ctrl] move_imu {direction} {meters:g}m failed: {e}")
                self._advance_script(f"move_imu failed: {e}")
            return
        if step.kind == "MOVE_FBUD":
            # Combined forward+up moveBy (FB_UD_IMU): one diagonal move instead
            # of HEIGHT then FB_IMU, so the drone RISES WHILE it ADVANCES —
            # faster capture positioning. go_xyz(x=fwd, y=0, z=up) in cm.
            fwd_cm = int(round(float(step.move_fb_m or 0.0) * 100.0))
            up_cm = int(round(float(step.move_up_m or 0.0) * 100.0))
            self._set_phase(
                Phase.MOVE_IMU,
                note + f" fwd {float(step.move_fb_m or 0):g}m "
                       f"+ up {float(step.move_up_m or 0):g}m")
            if fwd_cm == 0 and up_cm == 0:
                self._advance_script("fb_ud_imu 0 — no-op")
                return
            try:
                self.api.go_xyz(fwd_cm, 0, up_cm)
                self._advance_script(
                    f"fb_ud_imu complete (fwd {float(step.move_fb_m or 0):g}m "
                    f"up {float(step.move_up_m or 0):g}m)")
            except DroneApiError as e:
                print(f"[ctrl] fb_ud_imu failed: {e}")
                self._advance_script(f"fb_ud_imu failed: {e}")
            return
        if step.kind == "YAW":
            # Discrete rotation by ``rotation_deg`` degrees, +CW.
            # api.rotate is synchronous (blocks on the FC's discrete-
            # command window), so by the time we _advance_script the
            # firmware has confirmed the rotation completed. No phase
            # tick handler is needed; the brief Phase.ROTATE shows on
            # the UI for the operator's situational awareness.
            deg = int(step.rotation_deg or 0)
            # Optional per-step rotation-speed override (deg/s). None ->
            # the FC uses its global MaxRotationSpeed. Validated 1..180 by
            # the parser; clamp defensively here too.
            spd = step.rotation_speed
            spd = max(1, min(180, int(spd))) if spd is not None else None
            self._set_phase(Phase.ROTATE,
                            note + f" {deg:+d}deg"
                            + (f" @{spd}deg/s" if spd is not None else ""))
            if deg == 0:
                # No-op rotation — skip the API call entirely.
                self._advance_script("yaw 0 — no-op")
                return
            direction = "cw" if deg > 0 else "ccw"
            magnitude = max(1, min(180, abs(deg)))
            try:
                self.api.rotate(direction, magnitude, speed=spd)
                self._advance_script(
                    f"yaw {deg:+d}deg complete ({direction} {magnitude}"
                    + (f" @{spd}deg/s)" if spd is not None else ")")
                )
            except DroneApiError as e:
                # The backend already waits-for-hover + retries the moveBy a few
                # times, so a DroneApiError here means the rotation PERSISTENTLY
                # failed. Do NOT silently advance: a missed turn (esp. a 180°)
                # leaves the drone facing the WRONG way, and the next FB/APPROACH
                # then drives it in the wrong direction (operator saw exactly
                # this). Abort to a safe hold instead of flying blind.
                print(f"[ctrl] yaw rotate {deg:+d}deg FAILED after retries: {e}")
                self._abort(f"yaw rotate {deg:+d}deg failed after retries — "
                            f"holding to avoid a wrong-heading flight")
            return
        if step.kind == "LAND":
            self._set_phase(Phase.LAND, note)
            return
        if step.kind == "HEIGHT":
            target = max(cfg.min_height_m,
                         min(cfg.max_height_m, float(step.height)))
            with self.state.lock:
                self.state.height_target_m = target
            self._set_phase(Phase.HEIGHT,
                            note + f" -> {target:.2f}m")
            return
        if step.kind == "TO":
            tx = float(step.world_x)
            ty = float(step.world_y)
            tz = (max(cfg.min_height_m,
                      min(cfg.max_height_m, float(step.height)))
                  if step.height is not None else None)
            # Yaw target: explicit float wins; "auto" / None /
            # missing -> best-view heuristic. The heuristic returns
            # None when there's no arena_provider or no scoring
            # marker fits, in which case we leave the yaw target
            # unset and the drone keeps its current heading
            # throughout the move.
            yaw_arg = step.yaw
            if isinstance(yaw_arg, (int, float)):
                yaw_target = float(yaw_arg)
                yaw_src = "explicit"
            else:
                # No explicit yaw: keep current heading so the settle
                # condition (xy + height) fires without waiting for yaw.
                # The best-view heuristic caused 90°+ residual errors that
                # prevented settling entirely when goto_speed_factor is low.
                yaw_target = None
                yaw_src = "keep"
            with self.state.lock:
                self.state.goto_target_x_m = tx
                self.state.goto_target_y_m = ty
                self.state.goto_target_z_m = tz
                self.state.goto_target_yaw_deg = yaw_target
                # Plain TO advances on settle, not on a hold timer --
                # clear any goto_hold_until left over from a previous
                # HOOVER-after-TO.
                self.state.goto_hold_until = None
            z_txt = f" z={tz:.2f}" if tz is not None else " z=keep"
            yaw_txt = (f" yaw={yaw_target:+.0f}({yaw_src})"
                       if yaw_target is not None else " yaw=keep")
            self._set_phase(Phase.GOTO,
                            note + f" -> ({tx:.2f},{ty:.2f}){z_txt}m"
                                   f"{yaw_txt}")
            return
        if step.kind == "DANCE":
            now = time.monotonic()
            origin_xy = self.position.position_m
            with self.state.lock:
                tel = self.state.last_telemetry
            try:
                h_cm = (float(tel.raw.get("height_cm"))
                        if tel and tel.raw.get("height_cm") is not None
                        else None)
            except (TypeError, ValueError):
                h_cm = None
            origin_h = (h_cm / 100.0 if h_cm is not None
                        else cfg.default_height_m)
            with self.state.lock:
                self.state.dance_until = now + float(step.seconds)
                self.state.dance_mode = step.mode or "wobble"
                self.state.dance_origin_xy_m = origin_xy
                self.state.dance_origin_height_m = origin_h
            self._set_phase(Phase.DANCE,
                            note + f" {self.state.dance_mode}"
                                   f" {step.seconds:g}s")
            return
        self._abort(f"unknown script step kind: {step.kind!r}")

    # ------------------------------------------------------------------ land
    def _step_land(self, tel: Optional[TelemetrySnapshot], now: float) -> None:
        # Send zero RC for one tick before requesting land so that the
        # drone is stable.
        self._send_rc(0, 0, 0, 0)
        if not getattr(self, "_land_requested", False):
            try:
                self.api.land()
                self._land_requested = True
            except DroneApiError as e:
                print(f"[ctrl] land() failed: {e} (will retry)")
                return
        # The Anafi flips tel.flying to False the moment it enters the
        # 'landing' SDK state -- i.e., when descent starts, not when
        # the drone is on the ground. If we transitioned to DONE here
        # the recording would stop mid-air. Stay in LAND (recording
        # active) until either ground contact (height ~ 0) or a
        # reasonable descent window elapses.
        if tel and not tel.flying:
            try:
                height_cm = float(tel.raw.get("height_cm") or 0.0)
            except (TypeError, ValueError):
                height_cm = 0.0
            if getattr(self, "_descent_started_at", None) is None:
                self._descent_started_at = now
            descent_elapsed = now - self._descent_started_at
            if height_cm < 10.0 or descent_elapsed > 5.0:
                # Hand off to the script. If LAND was the last step,
                # _advance_script terminates with DONE; if more steps
                # follow (rare -- e.g., LAND followed by TAKEOFF for
                # a multi-flight script), the next step is loaded.
                self._advance_script("landed")
                return
        # Hard timeout
        if now - self.state.phase_started_at > 30.0:
            self._abort("timed out waiting for flying=False after land")

    # ---------------------------------------------------------------- helpers
    def _send_rc(self, lr: int, fb: int, ud: int, yaw: int,
                 dry_run: bool = False,
                 enforce_cfg_caps: bool = True,
                 altitude_envelope: bool = True) -> None:
        cfg = self.cfg
        if enforce_cfg_caps:
            # Final clamp -- in case PD output and individual clips disagree.
            lr  = max(-cfg.lat_rc_max, min(cfg.lat_rc_max, int(round(lr))))
            fb  = max(-cfg.fwd_rc_max, min(cfg.fwd_rc_max, int(round(fb))))
            ud  = max(-cfg.ud_rc_max,  min(cfg.ud_rc_max,  int(round(ud))))
            yaw = max(-cfg.yaw_rc_max, min(cfg.yaw_rc_max, int(round(yaw))))
        else:
            # Operator-typed raw RC (mission-script FB / UD / LR / RC):
            # the per-channel cfg.* caps are PD-tuning bounds designed
            # for closed-loop control output (where any larger stick
            # would put the tuned response in non-linear territory).
            # They were silently shrinking operator-typed values —
            # YAW 90 became 18 on a host with yaw_rc_max=18, LR 20
            # became 4 on a host with lat_rc_max=4. For raw steps we
            # only clamp to the protocol limit [-100, +100]; the FC
            # ceiling, arena guard and watchdog still clamp at the wire.
            lr  = max(-100, min(100, int(round(lr))))
            fb  = max(-100, min(100, int(round(fb))))
            ud  = max(-100, min(100, int(round(ud))))
            yaw = max(-100, min(100, int(round(yaw))))
        # Altitude envelope: forbid further climb above max_height_m and
        # forbid further descent below min_height_m. Applies to every
        # phase (HEIGHT_ALIGN's PD output and the takeoff climb both
        # respect the bounds because they're driven through the same
        # PD instance, but this clamp also covers any residual or
        # operator-style commands that try to push outside the
        # envelope).
        with self.state.lock:
            tel = self.state.last_telemetry
        if tel is not None:
            try:
                h_cm = float(tel.raw.get("height_cm"))
            except (TypeError, ValueError):
                h_cm = None
            if h_cm is not None and altitude_envelope:
                h_m = h_cm / 100.0
                if h_m >= cfg.max_height_m:
                    ud = min(0, ud)
                if h_m <= cfg.min_height_m:
                    ud = max(0, ud)
        # ── Telemetry stall: two-stage (detect + gate) ───────────────
        # If the drone-link has stalled (tel_holder hasn't been
        # refreshed for a while), we don't want the controller pushing
        # the PD-derived RC onto the wire -- it was computed against a
        # stale TelemetrySnapshot and is almost certainly wrong. Two
        # thresholds:
        #   - detect (~200ms): mark the stall in CSV / journal. No
        #     RC change. Catches short blips for diagnostics.
        #   - gate   (~500ms): escalate -- replace lr/fb/ud/yaw with
        #     zero so the drone reverts to onboard hover-hold.
        # dry_run bypasses (pre-flight dry-run has no live drone).
        if (not dry_run
                and getattr(cfg, "enable_telemetry_stall_detector", True)
                and self.get_tel_age is not None):
            try:
                age = self.get_tel_age()
            except Exception:
                age = None
            detect_th = float(getattr(
                cfg, "telemetry_stall_detect_threshold_s", 0.2))
            gate_th = float(getattr(
                cfg, "telemetry_stall_gate_threshold_s", 0.5))
            in_detect = (age is not None and age > detect_th)
            in_gate = (age is not None and age > gate_th)
            # ----- detect-level transitions (diagnostic only) -----
            if in_detect and not self._stall_detect_active:
                print(f"[ctrl] telemetry stall (age={age:.2f}s "
                      f"> {detect_th:.2f}s) -- marking CSV")
                self._stall_detect_active = True
            elif not in_detect and self._stall_detect_active:
                age_s = "n/a" if age is None else f"{age:.2f}s"
                print(f"[ctrl] telemetry recovered (age={age_s})")
                self._stall_detect_active = False
            # ----- gate-level transitions (RC override) -----
            if in_gate and not self._stall_gate_active:
                print(f"[ctrl] telemetry stall escalated "
                      f"(age={age:.2f}s > {gate_th:.2f}s) "
                      f"-- gating RC to zero")
                self._stall_gate_active = True
            elif not in_gate and self._stall_gate_active:
                age_s = "n/a" if age is None else f"{age:.2f}s"
                print(f"[ctrl] telemetry gate released (age={age_s}) "
                      f"-- resuming RC")
                self._stall_gate_active = False
            if self._stall_gate_active:
                lr = fb = ud = yaw = 0
            with self.state.lock:
                self.state.telemetry_stalled = self._stall_detect_active
                self.state.telemetry_rc_gated = self._stall_gate_active
        with self.state.lock:
            self.state.last_rc = (lr, fb, ud, yaw)
        if dry_run:
            # Caller wanted the values reflected on the UI but no command
            # actually sent to the drone (used in pre-flight dry-run).
            return
        try:
            self.api.rc(lr=lr, fb=fb, ud=ud, yaw=yaw,
                        duration_ms=cfg.rc_command_duration_ms)
        except DroneApiError as e:
            # Don't crash the loop on transient errors.
            print(f"[ctrl] rc() failed: {e}")

    # ----------------------------------------------------- pre-flight dry-run
    def _step_preflight_dryrun(self, tel: Optional[TelemetrySnapshot],
                              now: float) -> None:
        """Compute the RC commands a station-keeping controller WOULD send
        and write them to ``state.last_rc`` without sending to the drone.

        Runs in INIT phase while the operator is walking the marker around
        with props off, so the UI displays live PD response. Distance
        floor and dead-bands behave the same as in flight.

        We call ``_set_phase(TAKEOFF)`` later, which resets the PD
        integrators, so any history accrued here is discarded before the
        real flight starts.
        """
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._send_rc(0, 0, 0, 0, dry_run=True)
            with self.state.lock:
                self.state.note = "pre-flight dry-run: marker not visible"
            return

        d, yaw_to_marker, hdg = meas

        # Hard distance floor: same as in flight phases -- only clamp the
        # forward channel to <=0; yaw / lateral PDs keep running so the
        # operator can verify them with the marker held inside the floor.
        with self.state.lock:
            tgt_d = self.state.target_distance_m
            tgt_h = self.state.target_relative_heading_deg
        floor_active = d < cfg.distance_floor_factor * tgt_d

        e_yaw = yaw_to_marker
        e_fwd = d - tgt_d
        e_hdg = ((tgt_h - hdg) + 540.0) % 360.0 - 180.0

        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        if floor_active:
            u_fwd = min(0.0, u_fwd)
        if abs(e_hdg) < cfg.heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw),
                      dry_run=True)
        with self.state.lock:
            self.state.note = (f"pre-flight dry-run: e_yaw={e_yaw:+.1f}° "
                               f"e_fwd={e_fwd:+.2f}m e_hdg={e_hdg:+.1f}° "
                               f"(motors NOT commanded)")

    # ----------------------------------------------- velocity feedforward
    #
    # vgx/vgy are interpreted as earth-frame NED (vN, vE) and rotated
    # through the drone's yaw to get body-frame (forward, right) -- this
    # is the only interpretation that matches the per-frame closing
    # rate observed in the recent approach flights (e.g., 00-43-35
    # closing at ~24 cm/s body-fwd; body-frame interpretation gave 0).
    #
    # Both dampers are now guarded with a "no-sign-flip" rule: if
    # damping would invert the PD command's sign (i.e., the brake
    # correction is larger and opposite to PD's intent), discard the
    # damping for that tick and fall back to PD-only. This protects
    # against the failure mode that crashed flight 00-00-47 -- if the
    # frame convention ever happens to be wrong (e.g., yaw briefly
    # bogus during a fast yaw rate), the worst-case is a single tick
    # of PD-only output, never a damping-driven reverse command.

    @staticmethod
    def _telemetry_world_speed(tel: Optional[TelemetrySnapshot]
                                ) -> Optional[tuple[float, float, float]]:
        """Return (vN, vE, yaw_deg) in cm/s + degrees, or None if any
        component is missing."""
        if tel is None:
            return None
        try:
            vN = float(tel.raw["vgx"])
            vE = float(tel.raw["vgy"])
            yaw = float(tel.raw["yaw"])
        except (KeyError, TypeError, ValueError):
            return None
        return (vN, vE, yaw)

    @staticmethod
    def _world_to_body(vN: float, vE: float,
                      yaw_deg: float) -> tuple[float, float]:
        """Project world-NED velocity to (body-forward, body-right)
        using yaw (deg, CW from north)."""
        th = math.radians(yaw_deg)
        v_fwd   =  vN * math.cos(th) + vE * math.sin(th)
        v_right = -vN * math.sin(th) + vE * math.cos(th)
        return v_fwd, v_right

    @staticmethod
    def _no_flip(u_raw: float, u_damped: float) -> float:
        """If damping flipped the sign of u_raw, reject it (return
        u_raw). Otherwise return u_damped. Protects against bad-frame
        damping driving the drone opposite to PD intent."""
        if u_raw > 0 and u_damped < 0:
            return u_raw
        if u_raw < 0 and u_damped > 0:
            return u_raw
        return u_damped

    def _velocity_damp_fwd(self, u_raw: float,
                          tel: Optional[TelemetrySnapshot]) -> float:
        cfg = self.cfg
        # Damping only fires when PD has a non-zero intent. In deadband,
        # active braking commands would drive the drone away from rest
        # for as long as the residual velocity persists -- exactly the
        # 1.5s "rc_lr=+4 while hdg=0" misbehaviour observed in flight
        # 2026-04-29_00-57-20. Inside the deadband, drone is permitted
        # to coast and slow from natural drag.
        if u_raw == 0.0:
            return 0.0
        ws = self._telemetry_world_speed(tel)
        if ws is None:
            return max(-cfg.fwd_rc_max, min(cfg.fwd_rc_max, u_raw))
        vN, vE, yaw = ws
        v_fwd, _ = self._world_to_body(vN, vE, yaw)
        u = self._no_flip(u_raw, u_raw - cfg.fwd_kv * v_fwd)
        return max(-cfg.fwd_rc_max, min(cfg.fwd_rc_max, u))

    def _velocity_damp_lat(self, u_raw: float,
                          tel: Optional[TelemetrySnapshot]) -> float:
        cfg = self.cfg
        if u_raw == 0.0:
            return 0.0
        ws = self._telemetry_world_speed(tel)
        if ws is None:
            return max(-cfg.lat_rc_max, min(cfg.lat_rc_max, u_raw))
        vN, vE, yaw = ws
        _, v_right = self._world_to_body(vN, vE, yaw)
        u = self._no_flip(u_raw, u_raw - cfg.lat_kv * v_right)
        return max(-cfg.lat_rc_max, min(cfg.lat_rc_max, u))

    def _marker_lost(self, now: float, escalate: bool = True) -> None:
        """Handle missing pose with a grace window before falling back to SEARCH.

        Always commands zero RC (do NOT hold the last command -- the first
        flight crashed because the drone coasted forward at ~1 m/s through a
        1.5 s grace window with rc_fb pinned at +30).
        """
        cfg = self.cfg
        self._send_rc(0, 0, 0, 0)
        with self.state.lock:
            last_seen = self.state.last_marker_seen_at
        if last_seen and (now - last_seen) < cfg.search_marker_lost_grace_s:
            return
        # Sustained loss -- escalate to SEARCH (unless already there).
        if escalate and self.state.phase != Phase.SEARCH:
            self._set_phase(Phase.SEARCH, "marker lost -- searching")

    def _abort(self, reason: str) -> None:
        with self.state.lock:
            self.state.abort_reason = reason
        # Try to land safely.
        try:
            self.api.rc_zero()
            self.api.land()
        except DroneApiError:
            pass
        self._set_phase(Phase.ABORT, reason)
