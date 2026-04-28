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

    INIT -> TAKEOFF -> SEARCH -> APPROACH -> ORBIT -> HOLD -> LAND -> DONE
                          ^         |          |        |
                          |         v          v        v
                          \\---------+----------+--------/   (marker lost)

Phases:

* TAKEOFF   -- request takeoff and wait until ``flying`` becomes True.
* SEARCH    -- yaw in place. If the marker becomes visible, go to APPROACH.
               If a full sweep completes without a sighting, LAND.
* APPROACH  -- close yaw error to zero AND distance error to zero. The
               orbit-bearing error is intentionally NOT controlled here;
               we simply face the marker straight on first.
* ORBIT     -- maintain target distance while sliding around the marker
               until relative_heading hits the target. Yaw is also kept
               at zero (camera faces the marker) so that the final pose
               is yaw=0, heading=target.
* HOLD      -- hover for ``hold_time_s`` while keeping all errors zero.
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
    APPROACH  = "approach"
    ORBIT     = "orbit"
    HOLD      = "hold"
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
    """

    def __init__(self, alpha: float, max_age_s: float):
        self.alpha = float(alpha)
        self.max_age_s = float(max_age_s)
        self._d: Optional[float] = None
        self._yaw: Optional[float] = None
        self._hdg: Optional[float] = None
        self._last_seen: float = 0.0

    def reset(self) -> None:
        self._d = None
        self._yaw = None
        self._hdg = None
        self._last_seen = 0.0

    def update(self, pose: Optional[MarkerPose], now: float) -> None:
        if pose is None:
            return
        if self._d is None:
            self._d = pose.distance_m
            self._yaw = pose.yaw_deg
            self._hdg = pose.relative_heading_deg
        else:
            a = self.alpha
            self._d   = a * pose.distance_m            + (1 - a) * self._d
            self._yaw = a * pose.yaw_deg               + (1 - a) * self._yaw
            self._hdg = a * pose.relative_heading_deg  + (1 - a) * self._hdg
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

    abort_reason: str = ""
    note: str = ""                                            # informational

    lock: threading.Lock = field(default_factory=threading.Lock)

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

        self._stop = threading.Event()
        self._go = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
                elif phase == Phase.APPROACH:
                    self._step_approach(tel, now)
                elif phase == Phase.ORBIT:
                    self._step_orbit(tel, now)
                elif phase == Phase.HOLD:
                    self._step_hold(tel, now)
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
            self._set_phase(Phase.SEARCH, "airborne, beginning marker search")
            return
        # If takeoff hasn't completed in 15 s, abort.
        if now - self.state.phase_started_at > 15.0:
            self._abort("timed out waiting for flying=True after takeoff")

    # ---------------------------------------------------------------- search
    def _step_search(self, now: float) -> None:
        cfg = self.cfg
        # If we already see the marker, switch immediately.
        if self.smoother.get(now) is not None:
            self._set_phase(Phase.APPROACH, "marker acquired -- approaching")
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

    # -------------------------------------------------------------- approach
    def _step_approach(self, tel: Optional[TelemetrySnapshot],
                      now: float) -> None:
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._marker_lost(now); return
        d, yaw_to_marker, _ = meas

        # Hard distance floor: never command forward motion inside
        # distance_floor_factor * target. Last-line guard against pose noise.
        if d < cfg.distance_floor_factor * cfg.target_distance_m:
            self._send_rc(0, 0, 0, 0)
            with self.state.lock:
                self.state.note = (f"distance floor ({d:.2f} m < "
                                   f"{cfg.distance_floor_factor:.2f} * "
                                   f"{cfg.target_distance_m:.2f} m) -- "
                                   "refusing forward command")
                self.state.settle_began_at = None
            return

        # Errors
        e_yaw = yaw_to_marker                  # want yaw_to_marker -> 0
        e_fwd = d - cfg.target_distance_m      # positive: too far -> move forward

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

        # Lateral / vertical zero in approach -- we want to face the marker
        # head-on first, not orbit.
        self._send_rc(lr=0, fb=int(u_fwd), ud=0, yaw=int(u_yaw))

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
            self._set_phase(Phase.ORBIT, "approach settled")

    # ----------------------------------------------------------------- orbit
    def _step_orbit(self, tel: Optional[TelemetrySnapshot],
                   now: float) -> None:
        """Slide around the marker maintaining distance, until the heading
        target is reached.

        Strategy: command lateral motion proportional to the heading error
        AROUND the marker, plus forward/back motion to keep the distance,
        plus yaw correction to keep the marker dead ahead.
        """
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            self._marker_lost(now); return
        d, yaw_to_marker, hdg = meas

        # Hard distance floor: same guard as in approach.
        if d < cfg.distance_floor_factor * cfg.target_distance_m:
            self._send_rc(0, 0, 0, 0)
            with self.state.lock:
                self.state.note = (f"distance floor ({d:.2f} m) -- "
                                   "refusing motion in orbit")
                self.state.settle_began_at = None
            return

        e_yaw = yaw_to_marker
        e_fwd = d - cfg.target_distance_m
        # Heading error: difference, shortest-arc.
        e_hdg = ((cfg.target_relative_heading_deg - hdg) + 540.0) % 360.0 - 180.0

        # Deadbands
        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        # Lateral channel: slide the drone around the marker.
        # Sign reasoning (top-down view, drone facing marker):
        #
        #     +heading => drone is on the marker's RIGHT (looking outward
        #     from the marker), which is the camera's LEFT side of the
        #     scene. To move from heading=0 toward heading=+90, the drone
        #     must walk CW around the marker as seen from above. From the
        #     drone's own POV (facing the marker), walking CW around the
        #     marker means stepping to its OWN LEFT, which is a NEGATIVE
        #     `lr` command (lr>0 = right). So u_lat = -k * e_hdg.
        if abs(e_hdg) < cfg.heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            # Convert heading-error degrees into a metres-equivalent arc
            # length so the lateral PD stays sensibly tuned at any
            # distance: arc = d * theta. Negate per the comment above.
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)

        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw))

        in_band = (abs(e_yaw) < cfg.yaw_deadband_deg
                   and abs(e_fwd) < cfg.distance_deadband_m
                   and abs(e_hdg) < cfg.heading_deadband_deg)
        # Same lock-then-decide pattern as approach: don't call _set_phase
        # while holding state.lock (non-reentrant -> self-deadlock).
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
            self._set_phase(Phase.HOLD, "orbit reached target heading")

    # ------------------------------------------------------------------ hold
    def _step_hold(self, tel: Optional[TelemetrySnapshot],
                  now: float) -> None:
        cfg = self.cfg
        meas = self.smoother.get(now)
        if meas is None:
            # During hold, brief marker loss is tolerable -- just zero out.
            # If sustained loss, _marker_lost will eventually escalate.
            self._send_rc(0, 0, 0, 0)
            self._marker_lost(now, escalate=True); return
        d, yaw_to_marker, hdg = meas

        # Hard distance floor: same guard as in approach / orbit.
        if d < cfg.distance_floor_factor * cfg.target_distance_m:
            self._send_rc(0, 0, 0, 0)
            with self.state.lock:
                self.state.note = (f"distance floor ({d:.2f} m) -- "
                                   "refusing motion in hold")
            return

        # Same control law as orbit, just with the goal already met.
        e_yaw = yaw_to_marker
        e_fwd = d - cfg.target_distance_m
        e_hdg = ((cfg.target_relative_heading_deg - hdg) + 540.0) % 360.0 - 180.0
        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
        if abs(e_hdg) < cfg.heading_deadband_deg:
            u_lat_raw = 0.0
        else:
            # Same sign reasoning as in _step_orbit: u_lat = -k * e_hdg.
            arc_err_m = -d * math.radians(e_hdg)
            u_lat_raw = self.pd_lat.step(arc_err_m, now)
        u_lat = self._velocity_damp_lat(u_lat_raw, tel)
        self._send_rc(lr=int(u_lat), fb=int(u_fwd), ud=0, yaw=int(u_yaw))

        with self.state.lock:
            if self.state.hold_began_at is None:
                self.state.hold_began_at = now
            elapsed = now - self.state.hold_began_at
        if elapsed >= cfg.hold_time_s:
            self._set_phase(Phase.LAND, f"hold of {cfg.hold_time_s:.0f}s complete")

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
        if tel and not tel.flying:
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
        """Compute the RC commands an orbit-style controller WOULD send and
        write them to ``state.last_rc`` without sending to the drone.

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

        # Hard distance floor: same guard as in flight phases.
        if d < cfg.distance_floor_factor * cfg.target_distance_m:
            self._send_rc(0, 0, 0, 0, dry_run=True)
            with self.state.lock:
                self.state.note = (f"pre-flight dry-run: distance floor "
                                   f"({d:.2f} m < "
                                   f"{cfg.distance_floor_factor:.2f} * "
                                   f"{cfg.target_distance_m:.2f} m) "
                                   "-- would refuse motion")
            return

        e_yaw = yaw_to_marker
        e_fwd = d - cfg.target_distance_m
        e_hdg = ((cfg.target_relative_heading_deg - hdg) + 540.0) % 360.0 - 180.0

        u_yaw = 0.0 if abs(e_yaw) < cfg.yaw_deadband_deg else self.pd_yaw.step(e_yaw, now)
        u_fwd_raw = 0.0 if abs(e_fwd) < cfg.distance_deadband_m else self.pd_fwd.step(e_fwd, now)
        u_fwd = self._velocity_damp_fwd(u_fwd_raw, tel)
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
    @staticmethod
    def _telemetry_speed(tel: Optional[TelemetrySnapshot],
                        key: str) -> Optional[float]:
        """Return body-frame speed in cm/s from telemetry, or None.

        The Anafi server publishes ``vgx`` (body-right) and ``vgy``
        (body-forward) in cm/s under ``telemetry.raw``. Older firmwares /
        the simulator may omit them; in that case skip the damping rather
        than guessing.
        """
        if tel is None:
            return None
        v = tel.raw.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _velocity_damp_fwd(self, u_raw: float,
                          tel: Optional[TelemetrySnapshot]) -> float:
        """Subtract fwd_kv * vgy from the forward command, then clamp.

        Without this, the PD term saturates rc_fb at fwd_rc_max for the
        entire long-range approach (because at e_fwd > ~0.3 m the P term
        already exceeds the cap), so the drone accelerates uncontested
        until distance error closes -- way too late to brake.
        """
        cfg = self.cfg
        vgy = self._telemetry_speed(tel, "vgy")
        u = u_raw if vgy is None else u_raw - cfg.fwd_kv * vgy
        return max(-cfg.fwd_rc_max, min(cfg.fwd_rc_max, u))

    def _velocity_damp_lat(self, u_raw: float,
                          tel: Optional[TelemetrySnapshot]) -> float:
        """Mirror of :meth:`_velocity_damp_fwd` for the lateral channel."""
        cfg = self.cfg
        vgx = self._telemetry_speed(tel, "vgx")
        u = u_raw if vgx is None else u_raw - cfg.lat_kv * vgx
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
