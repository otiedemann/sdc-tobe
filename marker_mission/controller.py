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
    target_relative_heading_deg: float = 90.0
    active_marker_id: Optional[int] = None
    hold_time_s: float = 60.0

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

    # Mission-script execution state. Set by trigger() -- the controller
    # walks the script one step at a time; _advance_script consumes it.
    mission_script: List[Step] = field(default_factory=list)
    mission_step_idx: int = -1            # -1 = before first step
    current_step_kind: Optional[str] = None
    last_completed_step_kind: Optional[str] = None
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
                "marker_seen_age_s": (time.monotonic() - self.last_marker_seen_at)
                                       if self.last_marker_seen_at else None,
                "mission_script": script_lines,
                "mission_step_idx": self.mission_step_idx,
                "current_step_kind": self.current_step_kind,
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
                 ):
        self.api = api
        self.cfg = cfg
        self.state = state
        self.get_pose = frame_pose_provider
        self.get_tel = telemetry_provider
        self.on_phase_change = on_phase_change
        # Lets the TO step's auto-yaw heuristic enumerate reference
        # markers and score candidate yaws. Optional; without an
        # arena_provider TO falls back to "no yaw drive" for auto.
        self.get_arena = arena_provider

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
            self.state.target_relative_heading_deg = (
                self.cfg.target_relative_heading_deg)

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
                tel = self._refresh_inputs(now)

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
            # Reset zoom to 1.0 so APPROACH detection uses native FOV.
            if getattr(self, "_search_zoom", 1.0) > 1.0:
                try:
                    self.api.camera_config_set(zoom=1.0)
                except Exception:
                    pass
            self._set_phase(Phase.HEIGHT_ALIGN,
                            f"marker acquired -- stop + {back_deg}° back -- aligning altitude")
            return

        # Detect new SEARCH phase entry via phase_started_at so escalation
        # state (yaw speed, zoom) resets cleanly on every fresh SEARCH.
        with self.state.lock:
            phase_entry = self.state.phase_started_at
        if getattr(self, "_search_phase_entry", None) != phase_entry:
            self._search_phase_entry = phase_entry
            self._search_start_yaw = None
            self._search_swept = 0.0
            self._search_prev_yaw = None
            self._search_yaw_rc = float(cfg.search_yaw_rc)
            self._search_zoom = 1.0

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

        # Yaw in place using current (possibly escalation-reduced) speed.
        self._send_rc(0, 0, 0, int(self._search_yaw_rc),
                      enforce_cfg_caps=False)
        # Track how far we've yawed from telemetry to decide when to escalate.
        with self.state.lock:
            tel = self.state.last_telemetry
        if tel and tel.yaw_deg is not None:
            if self._search_start_yaw is None:
                self._search_start_yaw = tel.yaw_deg
                self._search_swept = 0.0
                self._search_prev_yaw = tel.yaw_deg
            else:
                # Accumulate signed delta, handling wrap.
                delta = (tel.yaw_deg - self._search_prev_yaw + 540.0) % 360.0 - 180.0
                self._search_swept += abs(delta)
                self._search_prev_yaw = tel.yaw_deg
                with self.state.lock:
                    self.state.search_yaw_swept_deg = self._search_swept
                if self._search_swept >= cfg.search_total_deg:
                    # Full revolution without sighting — escalate: 25% slower,
                    # 25% more zoom, capped at _SEARCH_MIN_YAW_RC / _SEARCH_MAX_ZOOM.
                    new_yaw_rc = max(self._SEARCH_MIN_YAW_RC,
                                     self._search_yaw_rc * 0.75)
                    new_zoom = min(self._SEARCH_MAX_ZOOM,
                                   self._search_zoom * 1.25)
                    at_speed_floor = new_yaw_rc <= self._SEARCH_MIN_YAW_RC
                    at_zoom_max = new_zoom >= self._SEARCH_MAX_ZOOM
                    print(f"[ctrl] SEARCH escalation: "
                          f"yaw_rc {self._search_yaw_rc:.1f} → {new_yaw_rc:.1f}, "
                          f"zoom {self._search_zoom:.2f}x → {new_zoom:.2f}x")
                    self._search_yaw_rc = new_yaw_rc
                    self._search_zoom = new_zoom
                    try:
                        self.api.camera_config_set(zoom=new_zoom)
                    except Exception as e:
                        print(f"[ctrl] SEARCH zoom set failed: {e}")
                    # Reset sweep counter for the next revolution.
                    self._search_start_yaw = None
                    self._search_swept = 0.0
                    with self.state.lock:
                        self.state.note = (
                            f"SEARCH escalated: yaw_rc={new_yaw_rc:.0f} "
                            f"zoom={new_zoom:.2f}x"
                            + (" [speed floor]" if at_speed_floor else "")
                            + (" [max zoom]" if at_zoom_max else "")
                        )
                    if at_speed_floor and at_zoom_max:
                        self._terminate_script(
                            "no marker found after escalated sweep -- landing")
                    return
        # Fallback timeout if no telemetry yaw is ever reported.
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
        """Drive the drone's altitude to match the marker's altitude.

        Marker world height = drone_height - tvec[1] (camera +y points
        DOWN in the world thanks to the gimbal, so a positive tvec[1]
        means the marker is below the drone). Target is clamped to
        [min_height_m, max_height_m] -- a marker mounted lower than
        min_height_m holds the drone at min_height_m instead of
        descending below the safety floor.

        Yaw is kept on the marker so it doesn't drift out of frame.
        Forward / lateral channels stay zero -- altitude only.
        """
        cfg = self.cfg
        meas = self.smoother.get(now)
        pose = self.state.last_pose
        if meas is None or pose is None:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        try:
            height_cm = float(tel.raw.get("height_cm")) if tel and tel.raw.get("height_cm") is not None else None
        except (TypeError, ValueError):
            height_cm = None
        if height_cm is None:
            # Without height telemetry, hand off to ALIGN. The state
            # machine continues to function, just without height
            # alignment for this run.
            self._set_phase(Phase.ALIGN,
                            "no height telemetry -- skipping HEIGHT_ALIGN")
            return

        drone_h = height_cm / 100.0
        # tvec[1] is the marker's downward offset from the camera in
        # the camera frame (gimbal-stabilised, so the camera y-axis
        # aligns with world-down). Positive => marker is below drone.
        try:
            marker_y = float(pose.tvec[1])
        except Exception:
            marker_y = 0.0
        marker_height_m = drone_h - marker_y
        target_h = max(cfg.min_height_m,
                       min(cfg.max_height_m, marker_height_m))
        e_h = target_h - drone_h

        e_yaw = yaw_to_marker
        u_yaw = (0.0 if abs(e_yaw) < cfg.yaw_deadband_deg
                 else self.pd_yaw.step(e_yaw, now))
        u_ud = (0.0 if abs(e_h) < cfg.height_deadband_m
                else self.pd_height.step(e_h, now))

        self._send_rc(lr=0, fb=0, ud=int(u_ud), yaw=int(u_yaw))

        with self.state.lock:
            self.state.note = (f"HEIGHT_ALIGN: drone={drone_h:.2f}m  "
                               f"marker={marker_height_m:.2f}m  "
                               f"target={target_h:.2f}m  e={e_h:+.2f}m")

        # Settle: both height and yaw inside their deadbands for the
        # configured time -> transition to ALIGN.
        in_band = (abs(e_h)   < cfg.height_deadband_m
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
            self._set_phase(Phase.ALIGN,
                            f"height aligned ({drone_h:.2f}m) -- "
                            f"now aligning heading")

    # -------------------------------------------------------------- approach
    def _step_approach(self, tel: Optional[TelemetrySnapshot],
                      now: float) -> None:
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        # Hard distance floor: prevent FORWARD motion inside the floor;
        # yaw and lateral PDs keep running so the drone can recenter
        # and correct heading. The PD's natural backward output (when
        # d<target) pushes the drone out. Previously we zeroed all
        # channels on floor entry, which froze the drone in place
        # (flight 22-09-49: stuck at d=1.37, hdg=22 with rc=0,0,0,0).
        with self.state.lock:
            tgt_d = self.state.target_distance_m
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
        e_hdg = ((0.0 - hdg) + 540.0) % 360.0 - 180.0

        # Apply dead-bands
        if abs(e_yaw) < cfg.yaw_deadband_deg:
            u_yaw = 0.0
        else:
            u_yaw = self.pd_yaw.step(e_yaw, now)
        if abs(e_fwd) < cfg.distance_deadband_m:
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
        slow_zone = float(getattr(cfg, "approach_slow_zone_m", 0.0))
        slow_cap = int(getattr(cfg, "approach_slow_rc_max", cfg.fwd_rc_max))
        if slow_zone > 0.0 and abs(e_fwd) < slow_zone:
            cap = max(1, slow_cap)
            u_fwd = max(-cap, min(cap, u_fwd))

        # Lateral: same tangential-slide PD as ALIGN but pinned to
        # heading 0 for the whole approach. We use approach_heading_deadband_deg
        # (~5 deg) -- much tighter than ALIGN. ALIGN's +/-30 deg gave
        # APPROACH 60 deg of free space inside which lateral PD never
        # fired; heading drift integrated unchecked until it exited the
        # band already carrying tangential momentum (flight 21-42-17:
        # +15 -> -45 with rc_lr=0 for 6 s of the drift). With lat_rc_max=4
        # the lateral correction is still bounded to ~16 cm/s so it
        # doesn't fight the forward channel.
        if abs(e_hdg) < cfg.approach_heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        # Continuous height alignment to the marker's altitude. Same
        # geometry as HEIGHT_ALIGN: marker_height = drone_height -
        # tvec[1] (camera y points world-down thanks to the gimbal).
        # HEIGHT_ALIGN's one-shot settle isn't enough -- approach can
        # take many seconds and small altitude drift (Anafi self-stabilise
        # bias, wind, residual ud from prior phases) lets the marker
        # walk out of frame. Soft no-op if telemetry / pose isn't
        # available so approach itself keeps working.
        u_ud = 0.0
        pose = self.state.last_pose
        try:
            h_cm = (float(tel.raw.get("height_cm"))
                    if tel and tel.raw.get("height_cm") is not None
                    else None)
        except (TypeError, ValueError):
            h_cm = None
        if h_cm is not None and pose is not None:
            drone_h = h_cm / 100.0
            try:
                marker_y = float(pose.tvec[1])
            except Exception:
                marker_y = 0.0
            target_h = max(cfg.min_height_m,
                           min(cfg.max_height_m, drone_h - marker_y))
            e_h = target_h - drone_h
            if abs(e_h) >= cfg.height_deadband_m:
                u_ud = self.pd_height.step(e_h, now)

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=int(u_ud),
                      yaw=int(u_yaw))

        # Settle detection. We capture the "settled long enough" decision
        # under the lock, then release before calling _set_phase -- which
        # itself takes the lock and would self-deadlock since
        # threading.Lock is non-reentrant.
        in_band = (abs(e_yaw) < cfg.yaw_deadband_deg
                   and abs(e_fwd) < cfg.distance_deadband_m)
        settled = False
        with self.state.lock:
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.approach_settle_time_s:
                    settled = True
            else:
                self.state.settle_began_at = None
        if settled:
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

        # Continuous height alignment to the marker, same as APPROACH.
        # HOLD is only entered after APPROACH (HOOVER step rule), so
        # the operator's intent is "stay at the marker's height". The
        # IDLE phase (HOOVER without prior APPROACH) keeps ud=0.
        u_ud = 0.0
        pose = self.state.last_pose
        try:
            h_cm = (float(tel.raw.get("height_cm"))
                    if tel and tel.raw.get("height_cm") is not None
                    else None)
        except (TypeError, ValueError):
            h_cm = None
        if h_cm is not None and pose is not None:
            drone_h = h_cm / 100.0
            try:
                marker_y = float(pose.tvec[1])
            except Exception:
                marker_y = 0.0
            target_h = max(cfg.min_height_m,
                           min(cfg.max_height_m, drone_h - marker_y))
            e_h = target_h - drone_h
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
            self._send_rc(lr, fb, ud, yaw, enforce_cfg_caps=False)
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
        if target is None:
            self._advance_script("height: no target set")
            return
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
            yaw_arena = self.state.arena_yaw_deg
            yaw_at = self.state.arena_yaw_updated_at
            hold_until = self.state.goto_hold_until
        if tx is None or ty is None:
            self._advance_script("goto: no target set")
            return

        wp_age = (now - wp_at) if wp_at > 0.0 else None
        yaw_age = (now - yaw_at) if yaw_at > 0.0 else None
        fresh = (wp is not None and wp_age is not None
                 and wp_age <= cfg.pose_max_age_s
                 and yaw_arena is not None and yaw_age is not None
                 and yaw_age <= cfg.pose_max_age_s)
        if not fresh:
            # Position / yaw lost -- yaw in place to bring a marker
            # back into view. Reset settle timer so we don't latch a
            # spurious "arrived" while drifting unobserved.
            self._send_rc(0, 0, 0, cfg.search_yaw_rc,
                          enforce_cfg_caps=False)
            with self.state.lock:
                self.state.settle_began_at = None
                age_txt = (f"{wp_age:.1f}s" if wp_age is not None
                           else "never")
                self.state.note = (f"GOTO: position stale ({age_txt}) "
                                   f"-- searching")
            return

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

        # Optional yaw leg. Drives drone arena yaw to t_yaw. Uses
        # state.arena_yaw_deg (already published by vision_worker)
        # which is fresh by the time we got here -- the freshness
        # gate above failed if it weren't.
        u_yaw = 0.0
        e_yaw = 0.0
        yaw_ok = True
        if t_yaw is not None and yaw_arena is not None:
            e_yaw = ((float(t_yaw) - float(yaw_arena) + 540.0) % 360.0
                     - 180.0)
            u_yaw = (0.0 if abs(e_yaw) < cfg.yaw_deadband_deg
                     else self.pd_yaw.step(e_yaw, now))
            yaw_ok = abs(e_yaw) < cfg.yaw_deadband_deg

        self._send_rc(int(u_lat), int(u_fwd), int(u_ud), int(u_yaw))

        e_xy = math.hypot(ex, ey)
        in_band = e_xy < cfg.goto_deadband_m and height_ok and yaw_ok
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
                f"  e_xy={e_xy:.2f}m{z_txt}{yaw_txt}")
            if in_band:
                if self.state.settle_began_at is None:
                    self.state.settle_began_at = now
                if now - self.state.settle_began_at >= cfg.height_settle_time_s:
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
        # subsequent IDLE / HOLD ticks.
        with self.state.lock:
            self.state.await_marker_id = (int(step.marker_id)
                                          if step.kind == "AWAIT"
                                          else None)
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
            with self.state.lock:
                self.state.active_marker_id = int(step.marker_id)
                self.state.target_distance_m = float(step.distance)
                # APPROACH always closes head-on so the marker stays
                # inside the detector's reliable angle range. HOLD
                # inherits this 0 setpoint via state.target_relative_heading_deg.
                self.state.target_relative_heading_deg = 0.0
            self._set_phase(Phase.SEARCH,
                            note + f" id={int(step.marker_id)}"
                                   f" d={float(step.distance):g}m")
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
        if step.kind == "YAW":
            # Discrete rotation by ``rotation_deg`` degrees, +CW.
            # api.rotate is synchronous (blocks on the FC's discrete-
            # command window), so by the time we _advance_script the
            # firmware has confirmed the rotation completed. No phase
            # tick handler is needed; the brief Phase.ROTATE shows on
            # the UI for the operator's situational awareness.
            deg = int(step.rotation_deg or 0)
            self._set_phase(Phase.ROTATE,
                            note + f" {deg:+d}deg")
            if deg == 0:
                # No-op rotation — skip the API call entirely.
                self._advance_script("yaw 0 — no-op")
                return
            direction = "cw" if deg > 0 else "ccw"
            magnitude = max(1, min(180, abs(deg)))
            try:
                self.api.rotate(direction, magnitude)
                self._advance_script(
                    f"yaw {deg:+d}deg complete ({direction} {magnitude})"
                )
            except DroneApiError as e:
                # Don't abort the whole mission for a transient
                # rotate failure; log and move on so subsequent
                # steps (notably LAND) still execute.
                print(f"[ctrl] yaw rotate {deg:+d}deg failed: {e}")
                self._advance_script(f"yaw rotate failed: {e}")
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
                yaw_target = self._best_view_yaw_deg(tx, ty)
                yaw_src = ("auto" if yaw_target is not None
                           else "auto-skipped")
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
                 enforce_cfg_caps: bool = True) -> None:
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
            if h_cm is not None:
                h_m = h_cm / 100.0
                if h_m >= cfg.max_height_m:
                    ud = min(0, ud)
                if h_m <= cfg.min_height_m:
                    ud = max(0, ud)
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
