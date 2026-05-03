# Mission language and flight-phase reference (detailed)

The companion to [`MISSION_REFERENCE.md`](MISSION_REFERENCE.md).

The plain-English version explains *what* each command and phase does.
This document explains *how*, with the relevant settle times,
deadbands, edge cases and recovery paths laid out so that a reader of
[`controller.py`](../controller.py) and [`mission_script.py`](../mission_script.py)
can navigate the code without rediscovering them.

We still keep PID maths out of scope — for the controller maths see the
`PIDController` class and the per-axis `kp / kd / ki / kv / i_clip /
out_clip` values in `config.py`.

---

## Table of contents

- [Top-level architecture](#top-level-architecture)
- [Operator-typed script](#operator-typed-script)
- [Mission state on the controller](#mission-state-on-the-controller)
- [Script commands (detailed)](#script-commands-detailed)
- [Flight phases (detailed)](#flight-phases-detailed)
- [Cross-cutting mechanics](#cross-cutting-mechanics)
- [Termination and recovery](#termination-and-recovery)
- [Cfg fields the runtime reads vs. writes](#cfg-fields-the-runtime-reads-vs-writes)
- [Arena world positioning](#arena-world-positioning)

---

## Top-level architecture

Three threads collaborate during flight:

1. **`vision_worker`** decodes the MJPEG stream, runs the ArUco detector,
   filters by the **active marker id** (read from
   `state.active_marker_id`), and pushes the latest pose into a holder.
   Annotates frames for the live feed and the recorder.
2. **`telemetry_worker`** polls `unified_api_server.py` for
   battery / yaw / position-velocity / `flying` / `connected`, ~4 Hz.
3. **`MissionController._run`** is the actual control loop. At
   `control_rate_hz` (default 10 Hz) it pulls the latest pose +
   telemetry, dispatches on the current `Phase`, and produces a single
   `(lr, fb, ud, yaw)` RC command via `_send_rc()`.

A separate `log_worker` writes a CSV row per tick to `flight_log.csv`.

The operator UI is a Flask app that reads `state.snapshot()` and the
latest annotated frame; it can mutate cfg via `/api/tune/*`, install a
script via `/api/start`, and trigger `controller.stop()`.

---

## Operator-typed script

The textarea on the camera page is parsed by
`mission_script.parse(text, defaults)` into a `List[Step]`. Defaults
are pulled from cfg at parse time (`defaults_from_cfg`) — once the
script has been parsed, the steps carry their own values and the cfg
defaults are never re-read for them.

### Parser behaviour

- One command per line. `#` starts a comment, blank lines ignored.
- Case-insensitive; `format()` re-emits the canonical form
  (uppercase command, normalised whitespace).
- Any malformed line raises `ScriptError` with a 1-based line number,
  surfaced in the UI as a red error message under the textarea.

### Persistence (priority chain)

When the camera page first loads, the textarea is populated by
`mission_script.load_priority_script()` walking three sources in order:

1. Active draft at `~/.marker_mission/active_mission_script.txt`.
   Written by debounced auto-save on every keystroke.
2. Newest `mission_script.txt` under `FLIGHTS_DIR`.
3. The hard-coded default `TAKEOFF / APPROACH / HOOVER / LAND`.

When `/api/start` succeeds, the executed script is also saved as
`<flight_dir>/mission_script.txt` (canonicalised via `format()`), next
to `cfg_start.json`.

### Named save / load

The "Saved scripts" disclosure on the camera page mirrors the tuning
snapshot pattern. Endpoints:

- `GET /api/mission/scripts` — list (name, mtime, size).
- `POST /api/mission/scripts/<name>` (body: `{text}`) — save.
- `POST /api/mission/scripts/<name>/load` — return text.
- `DELETE /api/mission/scripts/<name>` — delete.

Files live under `~/.marker_mission/mission_scripts/<name>.txt`.

Name validation: `^[A-Za-z0-9][A-Za-z0-9_\-\. ]{0,63}$`, mirrored from
the tuning-snapshot regex.

---

## Mission state on the controller

`MissionState` (in `controller.py`) holds everything the controller
mutates per tick. Key fields the script feature added:

| Field | Meaning |
|---|---|
| `mission_script: List[Step]` | The currently-installed script. |
| `mission_step_idx: int` | Index of the executing step (-1 before first). |
| `current_step_kind: Optional[str]` | The kind of `mission_script[mission_step_idx]`. |
| `last_completed_step_kind` | Set on every `_advance_script` call — used by `HOOVER` to decide HOLD vs IDLE. |
| `target_distance_m`, `target_relative_heading_deg`, `active_marker_id`, `hold_time_s` | The **runtime** versions of the cfg values. Phases read these, NOT the cfg fields, so an APPROACH step with explicit values never pollutes the cfg defaults consumed by the parser on the next mission. |
| `idle_until`, `dance_until`, `dance_origin_xy_m`, `dance_origin_height_m`, `dance_mode`, `height_target_m` | Per-step transient state used by IDLE / HEIGHT / DANCE. |

The UI's `/api/state` (and `/api/replay/<id>/state` during replay)
exposes a JSON snapshot of the relevant subset — including
`mission_script` (canonicalised lines), `mission_step_idx`, and
`current_step_kind` so the camera-page list can highlight the active
step.

---

## Script commands (detailed)

The mapping from a script `Step` to a starting `Phase` happens in
`MissionController._apply_step_to_phase(step, reason)`:

```
TAKEOFF   -> Phase.TAKEOFF        + api.takeoff()
APPROACH  -> Phase.SEARCH         + state.active_marker_id, target_distance_m, target_relative_heading_deg = 0
HOOVER    -> Phase.HOLD           + state.hold_time_s        (if previous step was APPROACH)
          -> Phase.IDLE           + state.idle_until = now + step.seconds  (otherwise)
HEIGHT    -> Phase.HEIGHT         + state.height_target_m (clamped to [min_height_m, max_height_m])
DANCE     -> Phase.DANCE          + state.dance_until / dance_mode / dance_origin_xy_m / dance_origin_height_m
LAND      -> Phase.LAND
```

When the **current** step's last phase finishes, the controller calls
`_advance_script(reason)`, which:

1. Sets `last_completed_step_kind = current_step_kind`.
2. Increments `mission_step_idx`.
3. If we're past the end of the script:
   - If the last completed step was `LAND` → `Phase.DONE`.
   - Otherwise → `Phase.LAND` (a safety LAND so the drone always
     lands cleanly).
4. Otherwise calls `_apply_step_to_phase(next_step, reason)`.

### `TAKEOFF`

| Arguments | none |
|---|---|
| Starting phase | `TAKEOFF` |
| Per-step state | none |
| Phase exit | `_step_takeoff` settles → `_advance_script("takeoff complete")` |

`_apply_step_to_phase` issues `api.takeoff()` synchronously. If the
HTTP call fails (`DroneApiError`), the controller goes straight to
`_abort` (Phase.ABORT) and never enters TAKEOFF for real.

### `APPROACH [<marker-id>] [<distance>]`

| Arguments | both optional; default to `cfg.target_marker_id` and `cfg.target_distance_m` at parse time |
|---|---|
| Starting phase | `SEARCH` |
| Per-step state | `state.active_marker_id`, `state.target_distance_m`, `state.target_relative_heading_deg = 0.0` |
| Sub-phases (within the same script step) | SEARCH → HEIGHT_ALIGN → ALIGN → APPROACH |
| Exit | `_step_approach` settles → `_advance_script("approach settled")` |

`relative_heading_deg = 0` is hard-coded for APPROACH (the operator
doesn't get to override it) because APPROACH must close head-on for
the detector to stay reliable. HOLD inherits this 0 setpoint via
`state.target_relative_heading_deg`.

If SEARCH does a full sweep without seeing the marker, the script is
truncated: `_terminate_script("no marker found in full sweep -- landing")`
empties `state.mission_script`, sets `current_step_kind = "LAND"`, and
calls `_set_phase(Phase.LAND)`. The post-LAND `_advance_script("landed")`
then sees an empty script with last-kind = "LAND" and routes to DONE.

### `HOOVER [<seconds>]`

| Arguments | optional; default to `cfg.hold_time_s` |
|---|---|
| Starting phase | `HOLD` if `last_completed_step_kind == "APPROACH"`, otherwise `IDLE` |
| Per-step state | `state.hold_time_s = step.seconds` (HOLD branch) or `state.idle_until = now + step.seconds` (IDLE branch) |
| Exit | timer expires → `_advance_script("hold complete" / "idle complete")` |

The HOLD timer uses `state.hold_began_at` (set in `_set_phase(HOLD)`)
plus `state.hold_time_s` so the timer can't be skewed by per-tick
clock drift. Marker-loss inside the HOLD timer is tolerated up to the
grace window (`search_marker_lost_grace_s`); beyond that it escalates
to SEARCH.

### `HEIGHT [<height>]`

| Arguments | optional; default to `cfg.default_height_m`. Clamped to `[min_height_m, max_height_m]` at apply time. |
|---|---|
| Starting phase | `HEIGHT` |
| Per-step state | `state.height_target_m` |
| Exit | height inside `height_deadband_m` for `height_settle_time_s` (default 1 s) → `_advance_script("height reached")` |

`_step_height` runs the height PID only — `lr / fb / yaw` are zero. If
height telemetry is unavailable, the step times out after 5 s and
advances anyway, on the assumption that something else (HOLD, LAND)
will save the operator.

### `DANCE [<seconds>] [<mode>]`

| Arguments | seconds defaults to `cfg.default_dance_seconds_s`; mode defaults to `wobble`, must be one of `{wobble, spin, random}` |
|---|---|
| Starting phase | `DANCE` |
| Per-step state | `state.dance_until`, `state.dance_mode`, `state.dance_origin_xy_m`, `state.dance_origin_height_m` |
| Exit | `now >= state.dance_until` → `_advance_script("dance complete")` |

Dance origin captures both the dead-reckoning XY position from
`PositionEstimator` and the current `tel.height_cm` so the drone can
stay within `dance_radius_m` (default 0.5 m) of where the dance
started. The position estimator is a small NED dead-reckoning
integrator settled empirically against marker-derived ground truth in
`tools/analyze_velocity_frame.py`.

Per tick, the base RC pattern is:

- `spin` — `yaw = cfg.yaw_rc_max`, others zero.
- `wobble` — `yaw = yaw_rc_max * sin(2π·0.5·t)`, `ud = (ud_rc_max/4) * sin(2π·1.0·t)`.
- `random` — all four channels with different frequencies and phases.

The radius bound then projects the current drift `(dx_east, dy_north)`
into the body frame using the live drone yaw and:

- Zeroes any commanded channel pushing further outward
  (`fb * inward_v_fwd < 0` → `fb = 0`; same for `lr`).
- Adds an inward correction sized by the excess past `dance_radius_m`.

If yaw telemetry is unavailable the body-frame projection isn't
possible, so `lr` and `fb` are conservatively zeroed when over-radius.
Vertical drift is bounded the same way: if `|drone_height - origin| >
dance_radius_m`, any `ud` pushing further out is inverted.

### `LAND`

| Arguments | none |
|---|---|
| Starting phase | `LAND` |
| Per-step state | none |
| Exit | drone touches down → `_advance_script("landed")` |

`_step_land` first sends `rc=(0,0,0,0)` for one tick, then issues
`api.land()` (only once per phase entry — `_set_phase(LAND)` resets
the `_land_requested` flag, which is important for scripts that have
multiple LAND steps).

The Anafi flips `flying=False` the moment it *starts* descending, not
when it touches down. We require both `flying=False` AND a near-ground
height (`< 10 cm`) — or a 5 s descent timeout — before transitioning,
so the recorder keeps capturing the last few seconds of flight.

After LAND, `_advance_script("landed")` decides:

- If `mission_script` is empty (terminated) and `last_completed_step_kind == "LAND"` → DONE.
- If there's another step (rare, e.g. `LAND` then `TAKEOFF`) → load it.

---

## Flight phases (detailed)

The dispatch loop in `_run` is a flat `if / elif` on
`state.phase`. Each tick:

1. `_refresh_inputs(now)` — pull latest telemetry + pose, update the
   pose smoother, feed the `PositionEstimator`.
2. Dispatch to the phase's `_step_*` method.
3. The `_step_*` method computes `(lr, fb, ud, yaw)` and calls
   `_send_rc(...)`. `_send_rc` clamps each channel to its `*_rc_max`
   and applies the `min_height_m` / `max_height_m` envelope (forces
   `ud >= 0` below floor, `ud <= 0` above ceiling) before the HTTP
   `api.rc(...)` call.

### `INIT`

The control thread is alive but parked, waiting on `self._go.wait()`.
Per-tick it still runs `_refresh_inputs` and a *dry-run* of the
APPROACH-style PID computation (`_step_preflight_dryrun`) so the
operator can walk the marker around and watch the would-be RC values
appear on the UI without anything being sent to the drone.

`controller.trigger()` (called from `/api/start`) sets `_go`. The
thread breaks out of the wait loop and calls
`_advance_script("mission start")` which loads the first script step.

If `controller.stop()` fires before `_go`, the thread just returns —
`_run` early-returns when `_stop` is set after `_go`.

### `TAKEOFF`

Two stages:

1. **Wait airborne.** Until `tel.flying = True`, send `rc=(0,0,0,0)`.
   Hard timeout: 15 s with `tel.flying = False` → `_abort`.
2. **Climb to default height.** Read `tel.raw["height_cm"]`; if
   `|default_height_m - drone_h| < height_deadband_m` advance. Else
   command `rc_ud = pd_height.step(error)`. Hard timeout: 30 s in the
   phase total → `_abort`.

If height telemetry is unavailable, the climb is skipped (advance via
`_advance_script("airborne (no height telemetry)")`).

### `SEARCH`

Smoother check first: if `smoother.get(now)` is not None, the marker
became visible — `_set_phase(HEIGHT_ALIGN)`. Otherwise yaw at
`cfg.search_yaw_rc` (default 25 ≈ 25 °/s CW).

Sweep tracking uses the `tel.yaw_deg` reading. The first time we have
yaw telemetry in this phase, latch it as `_search_start_yaw`.
Subsequent ticks accumulate `|delta|` in `_search_swept` (handling
±180° wrap). When `_search_swept >= search_total_deg` (default 360°)
without ever finding the marker, `_terminate_script` truncates and
goes to LAND.

If telemetry yaw is never reported, fall back to a 30 s phase timeout
and terminate.

### `HEIGHT_ALIGN`

Read smoothed pose (need it for yaw centring) and the raw last-pose
(need `tvec[1]` for the marker-altitude geometry). If pose is missing
→ `_marker_lost`.

Marker world altitude:

```
marker_height_m = drone_h - pose.tvec[1]
target_h        = clamp(marker_height_m, min_height_m, max_height_m)
e_h             = target_h - drone_h
```

Drive `rc_ud` from `pd_height` and `rc_yaw` from `pd_yaw` (to keep the
marker centred). `lr / fb` stay zero.

Settle when both `|e_h| < height_deadband_m` and `|e_yaw| <
yaw_deadband_deg` for `height_settle_time_s` (default 1 s) →
`_set_phase(ALIGN)`.

If height telemetry is unavailable, hand off straight to ALIGN with a
note in `state.note` so the operator can see the degraded mode.

### `ALIGN`

Holds the radius captured on phase entry (`state.align_distance_m =
state.smoothed[0]` at the moment of `_set_phase(ALIGN)`). The forward
PD targets that radius — distance is NOT closed in here, only after
ALIGN settles.

Lateral PID drives the **arc-length error**:

```
e_hdg     = wrap_signed(0.0 - relative_heading_deg)   # signed degrees
arc_err_m = -d * radians(e_hdg)
u_lat     = pid_lat.step(arc_err_m, now)
```

The minus sign reflects the orbit-direction convention (see the
top-level README). `e_hdg > 0` (drone CW of the normal) → drone walks
CCW around the marker → body-frame is `lr > 0` (step right). Confirm
the sign with the dry-run procedure in the README before flying.

Settle uses the **looser** `align_heading_deadband_deg` (default 30°)
because long-range pose is noisy. APPROACH then tightens it down —
see below.

Distance floor (`distance_floor_factor * state.target_distance_m`)
clamps `fb ≤ 0` if we drift inside, but `yaw` and `lr` keep running so
we can recover.

### `APPROACH`

Same four-channel control as ALIGN, but with two key differences:

1. **Distance is closed in.** `e_fwd = d - state.target_distance_m`
   instead of being pinned to the radius captured on entry.
2. **Tight heading deadband.** Uses
   `approach_heading_deadband_deg` (default 5°) instead of ALIGN's
   30°. Reason: ALIGN's loose band gave APPROACH 60° of "free space"
   inside which the lateral channel never fired — heading drift then
   accumulated unchecked through the long forward closure (flight
   `2026-04-29_21-42-17`: heading walked +15° → -45° over 6 s with
   `rc_lr = 0` the whole time). The tight band keeps the lateral
   channel actively fighting drift.
3. **Continuous height alignment.** Same geometry as `HEIGHT_ALIGN`,
   running every tick — reason: `HEIGHT_ALIGN`'s one-shot settle leaves
   the drone vulnerable to drift over the seconds-long approach (gimbal
   bias, wind, residual ud from prior phases) which can walk the marker
   out of frame. Implemented soft (no-op when pose / telemetry is
   missing) so the horizontal control keeps working in degraded modes.

Settle: `|e_yaw| < yaw_deadband_deg` AND `|e_fwd| <
distance_deadband_m` for `approach_settle_time_s` (default 1 s) →
`_advance_script("approach settled")`.

The `_velocity_damp_fwd` and `_velocity_damp_lat` helpers add an
active brake against measured body velocity (`fwd_kv`, `lat_kv`) with
a no-sign-flip guard so a wrong-frame velocity reading can't push the
drone in the wrong direction.

### `HOLD`

Identical to APPROACH except:

1. The forward setpoint stays at `state.target_distance_m` (already
   reached at the end of APPROACH) — there's no closing-in to do.
2. The lateral setpoint is `state.target_relative_heading_deg` (which
   APPROACH set to 0), so the drone holds head-on.
3. The hold timer (`state.hold_began_at + state.hold_time_s`) is
   checked **first**, before reading the pose. Otherwise a brief
   marker loss right at the end of HOLD would pre-empt the LAND
   transition via `_marker_lost` and escalate to SEARCH instead — the
   operator's eye sees "drone hovered then started spinning around
   again".

Continuous height alignment runs the same way as APPROACH.

### `IDLE`

`rc = (0, 0, 0, 0)`. The Anafi's onboard stabiliser does the rest.
Exit when `now >= state.idle_until`.

Used for `HOOVER` after a non-APPROACH step (e.g.
`TAKEOFF / HOOVER 5 / LAND`) where there's no marker to track.

### `HEIGHT`

`rc_ud = pd_height.step(target_h - drone_h)`, others zero. The
target was captured into `state.height_target_m` by
`_apply_step_to_phase` and clamped to the safe envelope.

Settle uses `height_deadband_m` and `height_settle_time_s`.
Without height telemetry, time out after 5 s and advance with a
status note.

### `DANCE`

See the script-command section above for the per-mode RC patterns
and the radius-bound logic. The phase reads:

- `state.dance_until` — timer end.
- `state.dance_mode` — pattern selector.
- `state.dance_origin_xy_m` — set at phase entry from
  `PositionEstimator.position_m`.
- `state.dance_origin_height_m` — set at phase entry from
  `tel.height_cm` (or `cfg.default_height_m` as fallback).

All four channels are mode-driven; the safety layer adjusts them
based on drift.

### `LAND`

See the script-command section. Notable internal state:

- `_land_requested` (bool): ensures `api.land()` is called once per
  phase entry. Reset in `_set_phase(LAND)` so a multi-LAND script
  works correctly.
- `_descent_started_at` (float): captured the first tick we see
  `flying=False`. Used for the 5 s timeout.

After touchdown, `_advance_script("landed")` either advances the
script (if more steps) or transitions to DONE.

### `DONE` / `ABORT`

Terminal. `_run` returns. The mission loop in `mission.py` then waits
for the thread to exit, finalises the recorder (close videos, write
`mission_meta.json`, run the H.264 mux), calls `controller.reset()` to
clear state, and re-arms for the next run.

---

## Cross-cutting mechanics

### Pose smoothing

`PoseSmoother` holds an EMA on `(distance, yaw_to_marker, relative_heading)`.
- `pose_smoothing_alpha` (default 0.6) — higher = more responsive,
  noisier.
- `pose_max_age_s` (default 0.8) — beyond this, `smoother.get(now)`
  returns `None` and the controller treats the marker as lost.

The relative-heading channel additionally has an outlier-rejection
guard for the IPPE planar-pose ambiguity:
single-frame jumps > 5° are held through up to 5 consecutive ticks; if
the "jump" persists for 5+ ticks it's accepted as real motion and the
filter resyncs.

### Marker loss

`_marker_lost(now, escalate=True)` is the standard handler:

1. Always commands `rc=(0,0,0,0)` for this tick. (Earlier code held
   the last command, which crashed flight `2026-04-26` because the
   drone coasted forward at ~1 m/s through the grace window with
   `rc_fb` pinned at +30.)
2. If `(now - last_marker_seen_at) < search_marker_lost_grace_s`
   (default 1.5 s): tolerate the loss this tick.
3. Otherwise, if not already in SEARCH and `escalate=True`:
   `_set_phase(SEARCH, "marker lost -- searching")`.

HOLD calls it with `escalate=True`, so prolonged HOLD-time loss
escalates to SEARCH (which can recover or eventually land).

### Distance floor

`distance_floor_factor` (default 0.7) × `state.target_distance_m` is
the minimum allowed distance. Inside the floor, `_step_align`,
`_step_approach`, `_step_hold`, and `_step_preflight_dryrun` clamp
`u_fwd ≤ 0`. Yaw and lateral PIDs keep running, so the drone can yaw
to recenter and walk to the side without being unable to back away.

The earlier "zero all channels at the floor" pattern froze the drone
in place if it ever crossed the floor (flight `2026-04-26_22-09-49`:
hovered at d=1.37 with hdg=22 for 8 s, never recovering).

### Altitude envelope

`_send_rc` always applies a final altitude clamp before sending: above
`max_height_m`, force `rc_ud ≤ 0`; below `min_height_m`, force
`rc_ud ≥ 0`. Belt-and-braces — phase-level controllers respect the
envelope through their PID setpoints, but a misbehaving control law
can't push past the limits.

### PID anti-windup

Every `PIDController` has `ki = 0` by default — disabled. When ki is
non-zero, two layers protect against integrator runaway:

1. Hard clamp: `|integral| ≤ i_clip` per axis.
2. Back-calculation: when the unclamped output exceeds `out_clip`,
   the integrator is unwound by the saturation excess
   `(u_unclipped - u_clipped) / ki` so it can't keep growing while the
   actuator is pinned.

Plus: `dt > 1 s` freezes integration (covers phase changes and
stalls); `reset()` zeroes the integrator (called on every
`_set_phase`).

### Velocity feedforward

`fwd_kv` and `lat_kv` produce an active brake against the measured
body-frame velocity. `vgx / vgy` are interpreted as world-NED
(`vN, vE`) — settled empirically by `tools/analyze_velocity_frame.py`
— and rotated through the live drone yaw to body-(forward, right).

Both dampers run a "no-sign-flip" guard: if damping inverts the sign
of the underlying PID command, the damping is discarded for that tick
and we fall back to PID-only. Worst case is a single tick of
PID-only output, never a damping-driven reverse command.

---

## Termination and recovery

### Normal termination

Script ends with LAND → `_advance_script("landed")` → DONE.

### Implicit safety LAND

Script ends without LAND (e.g. `TAKEOFF / HOOVER 5`) → after the last
step's `_advance_script` runs out of steps with
`last_completed_step_kind != "LAND"`, the controller appends a safety
LAND. Once that LAND completes, DONE.

### Mid-script termination (search failure)

SEARCH does a full sweep without finding the marker.
`_terminate_script(reason)` truncates `state.mission_script` to `[]`
and sets `state.current_step_kind = "LAND"`, then `_set_phase(LAND)`.

Why both? `_advance_script("landed")` after this LAND sees an empty
script with `last_completed_step_kind == "LAND"` → DONE. Without the
truncation, `_advance_script` would walk forward to the next step in
the original script and execute it as if APPROACH had succeeded.

### Operator stop

`/api/stop` calls `controller.stop()`, which sets `_stop` (and `_go`,
to unblock a parked thread). If we're still in INIT, `_run` early-returns.
If we're airborne, `_run`'s main loop exits; on the way out it calls
`_safe_shutdown` which:

1. `api.rc_zero()` — kill any commanded motion.
2. `api.land()` — request a landing.
3. Poll `api.telemetry()` for up to 15 s, waiting for `flying=False`.

`stop()`'s join timeout is sized to accommodate the worst case
(rc_zero ~2 s + land ~20 s + telemetry poll ~15 s).

### API errors

`DroneApiError` raised inside the dispatch loop is caught — print and
continue next tick. Unknown exceptions are caught and printed; the
mission keeps running (the controller never crashes the entire
process due to a transient HTTP issue).

`takeoff()` / `land()` errors during `_apply_step_to_phase(TAKEOFF)`
or `_step_land` are escalated to `_abort` — those are unrecoverable
within the current mission.

---

## Cfg fields the runtime reads vs. writes

The script feature was carefully designed to not pollute cfg defaults
across mission runs. The split is:

| Field | Runtime read source | When cfg version is updated |
|---|---|---|
| `target_marker_id` | `state.active_marker_id` | tune-page only |
| `target_distance_m` | `state.target_distance_m` | tune-page only |
| `target_relative_heading_deg` | `state.target_relative_heading_deg` | tune-page only |
| `hold_time_s` | `state.hold_time_s` | tune-page only |
| `default_height_m` | read directly from cfg | tune-page only |
| `default_dance_seconds_s` | read directly from cfg | tune-page only |
| PID gains (`*_kp/kd/ki/kv/i_clip`, `*_rc_max`) | mirrored to PID instances; resynced via `apply_config_changes()` on tune | tune-page only |
| Geometry / safety (`marker_size_m`, `min_height_m`, `max_height_m`, `distance_floor_factor`, deadbands, settle times) | read directly from cfg per tick | tune-page only |

Script execution writes to `state.*` only. `cfg.target_*` and
`cfg.hold_time_s` stay stable across runs, so the parser's
`defaults_from_cfg()` always returns the user's tuned defaults — not
the trailing values from the previous mission.

---

## Arena world positioning

Optional layer on top of the per-marker pose: combine all visible
reference markers into a single arena-frame camera position estimate.
Lives in [`arena.py`](../arena.py). Disabled by default (no
`active_arena_config.json` yet → the loader falls through to the 16-marker
default; vision_worker still runs the estimator).

### Coordinate convention (centred origin)

* Origin `(0, 0, 0)` is the centre of the arena floor.
* Looking down at the floor: `+x` to the right, `+y` toward the front
  wall (top of the top-down view), `+z` is up.
* Walls are at `x = ±width/2` (left / right) and `y = ±depth/2`
  (front / back). Markers sit on the **inside face** of each wall with
  their normal pointing toward the origin.

The `WALL_ROTATIONS` matrices in [`arena.py`](../arena.py) implement
the marker→arena rotation per wall. Each is a proper rotation
(det = +1), verified by hand. Per-wall geometry is unit-tested by
constructing a head-on pose for the centre marker on each wall and
confirming the inversion lands on `(0, 0, 1.5)` for a drone at the
arena centre.

### JSON schema

```json
{
  "marker_size_m": 0.18,
  "width_m": 10.0,
  "depth_m": 25.0,
  "top_z_m": 4.0,
  "bottom_z_m": 2.0,
  "markers": [
    {"id": 1, "label": "Front high #1", "wall": "front",
     "x": 0.0, "y": 12.5, "z": 4.0},
    ...
  ]
}
```

The metadata fields (`width_m`, `depth_m`, `top_z_m`, `bottom_z_m`,
`marker_size_m`) are optional in the JSON; `ArenaConfig.from_dict` fills
in sensible defaults when they're absent. The Arena tab uses them to
re-render the top-down view and validate marker positions on save.

### Persistence

* `~/.marker_mission/active_arena_config.json` — single active draft,
  written by the Arena tab's "Save active" button. Loaded at
  startup by `mission.py` via `arena.load_priority_arena()`.
* `~/.marker_mission/arenas/<name>.json` — named save/load list,
  same UI pattern as mission scripts and tuning snapshots.

Source priority at startup: `--arena-config <path>` CLI override →
active draft → hard-coded default 16-marker layout.

### Hot reload

`vision_worker` reads from `_ArenaHolder` (in [`mission.py`](../mission.py))
each frame, so a `POST /api/arena/active` from the Arena tab takes
effect immediately without a mission restart. The endpoint's handler
calls `arena_holder.set(new_arena)` after writing the file.

### Default 16-marker layout

`default_arena(width_m, depth_m, top_z_m, bottom_z_m)` produces:

* 1 marker on each of the front and back walls (centred horizontally).
* 3 evenly-spaced markers on each of the left and right walls (at
  `y = ±depth/4` and `y = 0`).

Repeated at `top_z_m` (IDs 1–8) and `bottom_z_m` (IDs 9–16). Clockwise
from the front wall: `front (1) → right (3) → back (1) → left (3)`.

### Migration from the old convention

The `aruco-position/controller-modular/arena_config-tobe-1to4.json`
config used a different convention (origin at the front wall, opposite
X handedness) and is **incompatible** with the current code. Loading
it will produce wrong positions. Use the Arena tab → **Reset to
default** → tweak as needed → **Save active** to build a fresh config
in the new convention.

### Per-marker votes and bias diagnosis

`estimate_position()` returns a `PositionEstimate` with
`per_marker_position_m` exposing each marker's vote *before* the
weighted average. The recorder logs them as
`arena_per_marker_world` (CSV column, format `id:x,y,z|...`). Inverse-distance weighting means closer markers dominate.

If per-marker votes disagree systematically, the most likely culprit is a
mismatch between the JSON layout and the physical arena (markers placed
on a different wall than the config says, or the wall length wrong).
Use the Arena tab's top-down view to spot-check.

---

## Quick lookup

| Phase | Active channels (per tick) | Settle / exit |
|---|---|---|
| INIT | none (pre-flight dry-run for UI) | `controller.trigger()` sets `_go` |
| TAKEOFF | ud (climb to default height) | wait airborne 15 s, climb 30 s; `_advance_script` |
| SEARCH | yaw (sweep) | marker visible / 360° sweep / 30 s no-yaw |
| HEIGHT_ALIGN | ud + yaw | both inside deadbands for `height_settle_time_s` |
| ALIGN | lr + fb (radius hold) + yaw | heading + yaw inside deadbands for `align_settle_time_s` |
| APPROACH | lr + fb + ud + yaw | distance + yaw inside deadbands for `approach_settle_time_s` |
| HOLD | lr + fb + ud + yaw | `state.hold_time_s` timer |
| IDLE | none (rc = 0,0,0,0) | `state.idle_until` timer |
| HEIGHT | ud only | inside `height_deadband_m` for `height_settle_time_s` |
| DANCE | mode-dependent + position-bound | `state.dance_until` timer |
| LAND | none (api.land + wait) | `flying=False` + height < 10 cm or 5 s descent timeout |
| DONE / ABORT | terminal | — |
