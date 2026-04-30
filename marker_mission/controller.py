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
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .aruco_detector import MarkerPose
from .config import MissionConfig
from .drone_api import DroneApi, DroneApiError, TelemetrySnapshot


# ---------------------------------------------------------------------------
# State machine enum
# ---------------------------------------------------------------------------

class Phase(enum.Enum):
    INIT      = "init"
    TAKEOFF   = "takeoff"
    SEARCH    = "search"
    ALIGN     = "align"     # orbit at start radius until heading ~ 0
    APPROACH  = "approach"  # close distance to target with heading held near 0
    HOLD      = "hold"      # station-keeping hover for hold_time_s, then LAND
    RTH       = "rth"       # dead-reckon back to previous mission's takeoff pos
    LAND      = "land"
    DONE      = "done"
    ABORT     = "abort"


# ---------------------------------------------------------------------------
# PD controller building block
# ---------------------------------------------------------------------------

class PDController:
    """Plain proportional-derivative controller with output clamp.

    The derivative is computed from filtered sample-to-sample differences
    so that quick measurement jitter does not blow up the D term.
    """

    def __init__(self, kp: float, kd: float, out_clip: float,
                 d_filter_alpha: float = 0.4):
        self.kp = float(kp)
        self.kd = float(kd)
        self.out_clip = float(out_clip)
        self.alpha = float(d_filter_alpha)
        self._last_err: Optional[float] = None
        self._last_t: Optional[float] = None
        self._d_filt: float = 0.0

    def reset(self) -> None:
        self._last_err = None
        self._last_t = None
        self._d_filt = 0.0

    def step(self, error: float, now: float) -> float:
        if self._last_err is None or self._last_t is None or now - self._last_t < 1e-3:
            d_err = 0.0
        else:
            d_raw = (error - self._last_err) / (now - self._last_t)
            self._d_filt = self.alpha * d_raw + (1.0 - self.alpha) * self._d_filt
            d_err = self._d_filt
        self._last_err = error
        self._last_t = now
        u = self.kp * error + self.kd * d_err
        return max(-self.out_clip, min(self.out_clip, u))


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
    HDG_JUMP_CONSECUTIVE_MAX = 5      # accept after this many "jump" ticks

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
# Dead-reckoning position estimator
# ---------------------------------------------------------------------------
#
# Integrates earth-frame velocity (vgx = vN, vgy = vE in cm/s) into a
# 2D world-frame position (x_east, y_north) in metres.
#
# The "are vgx/vgy body-frame or earth-frame?" question was settled
# empirically with tools/analyze_velocity_frame.py against eight real
# flights: NED matched the marker-derived ground truth 2.2x to 3.9x
# better than body+yaw rotation across every flight with > 1 m of
# motion. So no yaw rotation is needed in the integration; vgx and
# vgy go in directly.
#
# This is purely dead-reckoning -- there is no marker fix, no GPS,
# no SLAM correction. Drift accumulates as roughly cm/s of velocity
# noise per second of flight. After ~60 s expect 1-2 m position
# error. So this is fine for a "rough RTH" function but not for
# anything requiring better than ~1 m accuracy.

class PositionEstimator:
    """World-frame x/y from integrating earth-frame NED velocity."""

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
        """Integrate one telemetry sample. ``vgx`` is earth-frame
        north velocity, ``vgy`` is earth-frame east, both in cm/s."""
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
        """Return (x_east, y_north) in metres."""
        return (self.x_east_m, self.y_north_m)

    @property
    def has_data(self) -> bool:
        return self._has_data


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
    target_distance_m: float = 1.0
    target_relative_heading_deg: float = 90.0

    settle_began_at: Optional[float] = None
    hold_began_at: Optional[float] = None
    search_began_at: Optional[float] = None
    search_yaw_swept_deg: float = 0.0
    last_marker_seen_at: float = 0.0
    # Captured on entry to Phase.ALIGN so the forward channel can hold the
    # current radius (we don't want to close distance until heading is
    # centred). Cleared when leaving ALIGN.
    align_distance_m: Optional[float] = None

    # Dead-reckoning bookkeeping for the RTH (return-to-home) feature.
    # Position is in PositionEstimator's frame -- (x_east, y_north)
    # metres relative to the estimator's origin (which is fixed for
    # the script's lifetime; the estimator runs continuously from
    # boot). Yaw is the compass heading recorded at takeoff; RTH
    # rotates the airframe back to this heading before landing so
    # the drone ends pointed the same way it started.
    # current_mission_takeoff_{world,yaw_deg} are set when the current
    # mission's TAKEOFF fires; on DONE both are promoted to the
    # previous_* fields, which are what RTH navigates to.
    current_mission_takeoff_world: Optional[tuple[float, float]] = None
    current_mission_takeoff_yaw_deg: Optional[float] = None
    previous_mission_takeoff_world: Optional[tuple[float, float]] = None
    previous_mission_takeoff_yaw_deg: Optional[float] = None
    # Set by trigger_rth() before the next TAKEOFF; consumed in
    # _step_takeoff and cleared on DONE/ABORT. When True, takeoff
    # leads to RTH instead of SEARCH.
    rth_armed: bool = False

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
            self.settle_began_at = None
            self.hold_began_at = None
            self.search_began_at = None
            self.search_yaw_swept_deg = 0.0
            self.last_marker_seen_at = 0.0
            self.align_distance_m = None
            self.current_mission_takeoff_world = None
            self.current_mission_takeoff_yaw_deg = None
            # NB: previous_mission_takeoff_{world,yaw_deg} are
            # intentionally NOT reset -- they have to survive between
            # missions for RTH to have a target. rth_armed is also NOT
            # reset (the RTH button sets it before reset() runs at
            # end-of-flight).
            self.abort_reason = ""
            self.note = ""

    def snapshot(self) -> dict:
        with self.lock:
            d = yaw = hdg = None
            if self.smoothed is not None:
                d, yaw, hdg = self.smoothed
            tel = self.last_telemetry.raw if self.last_telemetry else {}
            return {
                "phase": self.phase.value,
                "phase_age_s": time.monotonic() - self.phase_started_at,
                "uptime_s": time.monotonic() - self.started_at if self.started_at else 0.0,
                "distance_m": d,
                "yaw_to_marker_deg": yaw,
                "relative_heading_deg": hdg,
                "target_distance_m": self.target_distance_m,
                "target_relative_heading_deg": self.target_relative_heading_deg,
                "rc": {"lr": self.last_rc[0], "fb": self.last_rc[1],
                       "ud": self.last_rc[2], "yaw": self.last_rc[3]},
                "telemetry": tel,
                "marker_seen_age_s": (time.monotonic() - self.last_marker_seen_at)
                                       if self.last_marker_seen_at else None,
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
                 on_phase_change=None):
        self.api = api
        self.cfg = cfg
        self.state = state
        self.get_pose = frame_pose_provider
        self.get_tel = telemetry_provider
        self.on_phase_change = on_phase_change

        # PD controllers
        self.pd_yaw = PDController(cfg.yaw_kp, cfg.yaw_kd, cfg.yaw_rc_max)
        self.pd_fwd = PDController(cfg.fwd_kp, cfg.fwd_kd, cfg.fwd_rc_max)
        self.pd_lat = PDController(cfg.lat_kp, cfg.lat_kd, cfg.lat_rc_max)

        self.smoother = PoseSmoother(cfg.pose_smoothing_alpha, cfg.pose_max_age_s)

        # Dead-reckoning position estimator. Runs continuously from
        # construction (no reset between missions) so RTH has a stable
        # frame to navigate in.
        self.position = PositionEstimator()

        self._stop = threading.Event()
        self._go = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def apply_config_changes(self) -> None:
        """Re-sync per-instance state that was copied out of cfg at
        construction. Call after the cfg dataclass has been mutated
        externally (e.g., from the live-tuning UI) so the running
        controller picks up the new gains / clips immediately."""
        cfg = self.cfg
        self.pd_yaw.kp = float(cfg.yaw_kp)
        self.pd_yaw.kd = float(cfg.yaw_kd)
        self.pd_yaw.out_clip = float(cfg.yaw_rc_max)
        self.pd_fwd.kp = float(cfg.fwd_kp)
        self.pd_fwd.kd = float(cfg.fwd_kd)
        self.pd_fwd.out_clip = float(cfg.fwd_rc_max)
        self.pd_lat.kp = float(cfg.lat_kp)
        self.pd_lat.kd = float(cfg.lat_kd)
        self.pd_lat.out_clip = float(cfg.lat_rc_max)
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
        self.smoother.reset()
        # Per-phase scratch state (set lazily in the relevant _step_*).
        self._land_requested = False
        self._descent_started_at = None
        self._search_start_yaw = None
        self._search_swept = 0.0
        self._search_prev_yaw = None
        self.state.reset(self.cfg)

    def trigger(self) -> bool:
        """Release the run loop so it proceeds from INIT into TAKEOFF.
        Returns True if this call actually started the mission, False if
        it was already started or the controller is no longer in INIT."""
        if self._go.is_set():
            return False
        with self.state.lock:
            if self.state.phase != Phase.INIT:
                return False
        self._go.set()
        return True

    def is_armed(self) -> bool:
        """True if the thread is alive and still parked in INIT."""
        if not self.is_running():
            return False
        with self.state.lock:
            return self.state.phase == Phase.INIT and not self._go.is_set()

    def trigger_rth(self) -> bool:
        """Set the RTH-armed flag and trigger the mission. The next
        TAKEOFF will route to Phase.RTH (dead-reckoning navigate to
        previous_mission_takeoff_world, then yaw back to the recorded
        takeoff heading) instead of Phase.SEARCH.

        Returns True if RTH was successfully armed and triggered.
        Refuses if the run loop has already left INIT, if no previous
        mission start is recorded yet, or if the position estimator
        has no telemetry data (so we don't even know where 'home' is
        relative to where we are)."""
        with self.state.lock:
            if self.state.phase != Phase.INIT:
                return False
            if self.state.previous_mission_takeoff_world is None:
                return False
            if not self.position.has_data:
                return False
            self.state.rth_armed = True
        return self.trigger()

    def rth_available(self) -> bool:
        """True if trigger_rth() would succeed right now."""
        if not self.is_armed():
            return False
        with self.state.lock:
            if self.state.previous_mission_takeoff_world is None:
                return False
        return self.position.has_data

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
        # Latch the HOLD timer on entry, not on the first marker-visible
        # tick. Otherwise a marker loss right at the start of HOLD never
        # starts the timer, and the phase can run forever (eventually
        # escalating to SEARCH). _step_hold always reads from this.
        if phase == Phase.HOLD:
            with self.state.lock:
                self.state.hold_began_at = time.monotonic()
        # RTH bookkeeping. On TAKEOFF, snapshot the current world
        # position AND compass yaw so the next mission can navigate
        # back here pointing in the same direction. On DONE/ABORT,
        # promote them to "previous" (the actual RTH target), and
        # clear rth_armed so a fresh "Start mission" press doesn't
        # trigger another RTH.
        if phase == Phase.TAKEOFF:
            with self.state.lock:
                tel = self.state.last_telemetry
                yaw_now = None
                if tel is not None:
                    try:
                        yaw_now = float(tel.raw.get("yaw") or 0.0)
                    except (TypeError, ValueError):
                        yaw_now = None
                self.state.current_mission_takeoff_world = self.position.position_m
                self.state.current_mission_takeoff_yaw_deg = yaw_now
        if phase in (Phase.DONE, Phase.ABORT):
            with self.state.lock:
                if self.state.current_mission_takeoff_world is not None:
                    self.state.previous_mission_takeoff_world = (
                        self.state.current_mission_takeoff_world)
                if self.state.current_mission_takeoff_yaw_deg is not None:
                    self.state.previous_mission_takeoff_yaw_deg = (
                        self.state.current_mission_takeoff_yaw_deg)
                self.state.current_mission_takeoff_world = None
                self.state.current_mission_takeoff_yaw_deg = None
                self.state.rth_armed = False
        # Capture / release ALIGN's hold-radius setpoint at the phase
        # boundary so _step_align doesn't have to discover it itself.
        if phase == Phase.ALIGN:
            with self.state.lock:
                meas = self.state.smoothed
            self.state.align_distance_m = (meas[0] if meas is not None
                                           else self.cfg.target_distance_m)
        elif old == Phase.ALIGN:
            self.state.align_distance_m = None
        # Reset PD integrators on phase change
        self.pd_yaw.reset(); self.pd_fwd.reset(); self.pd_lat.reset()
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
            # Feed dead-reckoning estimator (used by RTH). vgx/vgy are
            # earth-frame NED velocity (settled empirically; see the
            # PositionEstimator docstring), so no yaw rotation here.
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

        self._set_phase(Phase.TAKEOFF, "requesting takeoff")
        try:
            self.api.takeoff()
        except DroneApiError as e:
            self._abort(f"takeoff API error: {e}")
            return

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
                elif phase == Phase.APPROACH:
                    self._step_approach(tel, now)
                elif phase == Phase.HOLD:
                    self._step_hold(tel, now)
                elif phase == Phase.RTH:
                    self._step_rth(tel, now)
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
        # Hold zero RC while we wait for the airframe to leave the ground.
        self._send_rc(0, 0, 0, 0)
        if tel and tel.flying:
            # If the operator pressed Return home before this takeoff,
            # skip the marker-search/align/approach pipeline and go
            # straight into dead-reckoning RTH navigation.
            with self.state.lock:
                go_rth = self.state.rth_armed
            if go_rth:
                self._set_phase(Phase.RTH,
                                "airborne, navigating to previous takeoff position")
            else:
                self._set_phase(Phase.SEARCH,
                                "airborne, beginning marker search")
            return
        # If takeoff hasn't completed in 15 s, abort.
        if now - self.state.phase_started_at > 15.0:
            self._abort("timed out waiting for flying=True after takeoff")

    # ---------------------------------------------------------------- search
    def _step_search(self, now: float) -> None:
        cfg = self.cfg
        # If we already see the marker, switch immediately. ALIGN orbits
        # at the current radius until the relative heading is roughly
        # zero (marker faces the camera) before APPROACH closes distance
        # -- this keeps the marker out of the oblique-angle range where
        # the ArUco detector starts to fail.
        if self.smoother.get(now) is not None:
            self._set_phase(Phase.ALIGN,
                            "marker acquired -- aligning to face it")
            return
        # Otherwise, yaw in place. Direction is arbitrary; we pick CW (+ yaw).
        self._send_rc(0, 0, 0, cfg.search_yaw_rc)
        # Track how far we've yawed from telemetry to decide when to give up.
        with self.state.lock:
            tel = self.state.last_telemetry
        # We don't know exact start heading -- compare to whatever we had when
        # the search began. The first time we have a telemetry yaw, latch it.
        if tel and tel.yaw_deg is not None:
            if not hasattr(self, "_search_start_yaw") or self._search_start_yaw is None:
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
                    self._search_start_yaw = None
                    self._set_phase(Phase.LAND,
                                    "no marker found in full sweep -- landing")
                    return
        # Fallback timeout if no telemetry yaw is ever reported.
        elif now - self.state.phase_started_at > 30.0:
            self._set_phase(Phase.LAND, "search timed out (no telemetry yaw)")
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
        floor_active = d < cfg.distance_floor_factor * cfg.target_distance_m

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
        floor_active = d < cfg.distance_floor_factor * cfg.target_distance_m

        # Errors
        e_yaw = yaw_to_marker                  # want yaw_to_marker -> 0
        e_fwd = d - cfg.target_distance_m      # positive: too far -> move forward
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

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw))

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
            self._set_phase(Phase.HOLD, "approach settled -- station-keeping")

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
        if began is not None and (now - began) >= cfg.hold_time_s:
            self._send_rc(0, 0, 0, 0)
            self._set_phase(Phase.LAND, f"hold of {cfg.hold_time_s:.0f}s complete")
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
        floor_active = d < cfg.distance_floor_factor * cfg.target_distance_m

        # Station-keeping PD: yaw / distance / heading are all actively
        # corrected so wind, rotor wash and sensor drift don't push the
        # drone off station during the timed hover.
        e_yaw = yaw_to_marker
        e_fwd = d - cfg.target_distance_m
        e_hdg = ((cfg.target_relative_heading_deg - hdg) + 540.0) % 360.0 - 180.0
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
        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw))
        # Timer transition is handled at the top of this method so it
        # wins over a marker-lost escalation on the same tick.

    # ------------------------------------------------------------------- rth
    # Dead-reckoning return-to-home. No marker / no GPS fix: we drive
    # purely off the integrated PositionEstimator, which has roughly
    # cm/s position-error drift per second of flight. This is a "rough"
    # RTH -- expect to land within a couple of metres of the original
    # takeoff spot, not centimetres.
    #
    # Strategy:
    #   - target = state.previous_mission_takeoff_world (set when the
    #     previous mission's TAKEOFF fired).
    #   - error = target - estimator.position. Distance + bearing in
    #     world frame.
    #   - Yaw the drone to face the bearing (pd_yaw on tel_yaw).
    #   - Once roughly facing target (|yaw_err| < gate), push forward
    #     proportional to remaining distance (pd_fwd, capped to
    #     fwd_rc_max).
    #   - Land when within RTH_ARRIVE_RADIUS_M, or after RTH_TIMEOUT_S
    #     (so a runaway dead-reckoning can't fly the drone into a wall
    #     forever).
    RTH_ARRIVE_RADIUS_M = 0.6
    RTH_TIMEOUT_S = 60.0
    RTH_YAW_GATE_DEG = 25.0   # only push forward when within this of target
    # Once at home position, rotate the airframe back to its takeoff
    # yaw. We tolerate this much error before committing to LAND
    # (avoids spinning forever on noisy yaw telemetry).
    RTH_FINAL_YAW_TOLERANCE_DEG = 8.0

    def _step_rth(self, tel: Optional[TelemetrySnapshot], now: float) -> None:
        cfg = self.cfg
        target = self.state.previous_mission_takeoff_world
        if target is None:
            self._set_phase(Phase.LAND, "RTH: no target available")
            return

        # Hard timeout -- protects against runaway drift in any direction.
        if now - self.state.phase_started_at > self.RTH_TIMEOUT_S:
            self._set_phase(Phase.LAND,
                            f"RTH: timeout after {self.RTH_TIMEOUT_S:.0f}s")
            return

        x_now, y_now = self.position.position_m
        dx = target[0] - x_now      # east error
        dy = target[1] - y_now      # north error
        dist = math.hypot(dx, dy)

        try:
            cur_yaw = float(tel.raw.get("yaw") or 0.0) if tel is not None else 0.0
        except (TypeError, ValueError):
            cur_yaw = 0.0

        # ----- Sub-stage 2: at home, now align yaw to takeoff heading -----
        if dist < self.RTH_ARRIVE_RADIUS_M:
            home_yaw = self.state.previous_mission_takeoff_yaw_deg
            if home_yaw is None:
                # No takeoff yaw recorded (shouldn't happen, but fall
                # back to landing in current orientation).
                self._set_phase(Phase.LAND, "RTH: arrived (no takeoff yaw)")
                return
            yaw_err_home = ((home_yaw - cur_yaw + 540.0) % 360.0) - 180.0
            if abs(yaw_err_home) < self.RTH_FINAL_YAW_TOLERANCE_DEG:
                self._set_phase(Phase.LAND,
                                f"RTH: arrived (dist={dist:.2f}m, "
                                f"yaw aligned to home {home_yaw:.0f}°)")
                return
            # Spin in place to reach home yaw. No fwd / lat motion --
            # we're already at the target position.
            if abs(yaw_err_home) < cfg.yaw_deadband_deg:
                u_yaw = 0.0
            else:
                u_yaw = self.pd_yaw.step(yaw_err_home, now)
            self._send_rc(lr=0, fb=0, ud=0, yaw=int(u_yaw))
            with self.state.lock:
                self.state.note = (f"RTH: at home, yawing to {home_yaw:+.0f}° "
                                   f"(err {yaw_err_home:+.0f}°)")
            return

        # ----- Sub-stage 1: navigate to home position -----
        # Bearing to target in compass (CW from north). atan2(east,
        # north) gives that directly: at (E=0, N=+) -> 0, at (E=+, N=0)
        # -> +90, at (E=0, N=-) -> +/-180.
        target_bearing_deg = math.degrees(math.atan2(dx, dy))
        # Shortest-arc yaw error in (-180, 180].
        yaw_err = ((target_bearing_deg - cur_yaw + 540.0) % 360.0) - 180.0

        # Yaw command: PD on yaw error, identical to other phases'
        # yaw control law. Saturates at yaw_rc_max.
        if abs(yaw_err) < cfg.yaw_deadband_deg:
            u_yaw = 0.0
        else:
            u_yaw = self.pd_yaw.step(yaw_err, now)

        # Forward command: only push forward once we're roughly facing
        # the target. Otherwise we'd fly sideways while yawing in.
        if abs(yaw_err) < self.RTH_YAW_GATE_DEG:
            u_fwd_raw = self.pd_fwd.step(dist, now)
        else:
            u_fwd_raw = 0.0
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        # Don't allow the damper to flip a positive-intent command into
        # backward during RTH -- distance is always positive so PD
        # always wants forward; backward here would be a damping
        # artefact, not a desired behaviour.
        if u_fwd_raw > 0:
            u_fwd = max(0.0, u_fwd)

        self._send_rc(lr=0, fb=int(u_fwd), ud=0, yaw=int(u_yaw))

        with self.state.lock:
            self.state.note = (f"RTH: dist={dist:.2f}m  "
                               f"bearing={target_bearing_deg:+.0f}°  "
                               f"yaw_err={yaw_err:+.0f}°")

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
                self._set_phase(Phase.DONE, "landed")
                return
        # Hard timeout
        if now - self.state.phase_started_at > 30.0:
            self._abort("timed out waiting for flying=False after land")

    # ---------------------------------------------------------------- helpers
    def _send_rc(self, lr: int, fb: int, ud: int, yaw: int,
                 dry_run: bool = False) -> None:
        cfg = self.cfg
        # Final clamp -- in case PD output and individual clips disagree.
        lr  = max(-cfg.lat_rc_max, min(cfg.lat_rc_max, int(round(lr))))
        fb  = max(-cfg.fwd_rc_max, min(cfg.fwd_rc_max, int(round(fb))))
        ud  = max(-cfg.ud_rc_max,  min(cfg.ud_rc_max,  int(round(ud))))
        yaw = max(-cfg.yaw_rc_max, min(cfg.yaw_rc_max, int(round(yaw))))
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
        floor_active = d < cfg.distance_floor_factor * cfg.target_distance_m

        e_yaw = yaw_to_marker
        e_fwd = d - cfg.target_distance_m
        e_hdg = ((cfg.target_relative_heading_deg - hdg) + 540.0) % 360.0 - 180.0

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
