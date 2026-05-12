# Mission language and flight-phase reference

A plain-English reference for the operator-typed mission script and the
flight phases the controller walks through. Pairs with the code in
[`mission_script.py`](../mission_script.py) (the parser) and
[`controller.py`](../controller.py) (the state machine).

This file deliberately skips the PID / damping / pose-smoothing details
— see the source for those. Here we cover **what each command and phase
actually does**, in the same level of detail you would explain to a new
operator.

---

## How the script and the phases relate

The operator types a sequence of **script commands** (one per line) into
the textarea on the camera page:

```
TAKEOFF
APPROACH 4 1.0
HOOVER 5
LAND
```

The controller walks one command at a time. Each command picks a
**starting flight phase**, configures its target values, then hands
control to the per-tick state machine. Some commands are one phase from
start to finish (e.g. `LAND` is just the LAND phase). Others walk
through several internal sub-phases — most notably `APPROACH`, which is
SEARCH → HEIGHT_ALIGN → ALIGN → APPROACH.

When a command's last phase finishes, the controller advances to the
next command in the script. When the script runs out, the drone always
lands (a safety LAND is appended if you forgot to add one).

---

## Script commands

### `TAKEOFF`

> Lift off and climb to a known altitude.

The controller asks the drone to take off, waits for the airframe to
report `flying = True`, then climbs to `default_height_m` (default
1.5 m).

Done as soon as the drone is at that altitude (within the height
deadband).

---

### `APPROACH [<marker-id>] [<distance>]`

> Find the marker, get in front of it, fly toward it.

The richest command. Walks the drone through four sub-phases:

```
SEARCH        # spin in place looking for the marker
  └─→ HEIGHT_ALIGN   # climb/descend until your altitude matches the marker's
        └─→ ALIGN     # slide sideways around the marker until you face it head-on
              └─→ APPROACH   # fly forward until you are at the target distance
```

Arguments:
- `marker-id`: which ArUco marker to chase. Defaults to the cfg value.
- `distance`: standoff distance in metres. Defaults to the cfg value.

If the marker is never seen during the SEARCH sweep, the script is
truncated and the drone lands as a safety measure.

Done as soon as the APPROACH phase has settled (distance and rotation
both inside their deadbands for ~1 s).

---

### `HOOVER [<seconds>]`

> Hover for some time. *Behaviour depends on what came before.*

Three flavours:

- **After `APPROACH`** — station-keeping HOLD. The drone actively holds
  its position relative to the marker (distance, heading, altitude) for
  the configured time.
- **After `TO`** — station-keeping at the world target. The drone keeps
  driving toward the same `(x, y[, z])` for the duration. Same
  position-loss recovery as `TO` (yaw-search if the world fix goes
  stale, resume on first fresh fix).
- **Otherwise** — IDLE. The drone sends `rc = (0,0,0,0)` and lets the
  Anafi's onboard stabiliser keep it roughly in place. No marker
  tracking.

Arguments:
- `seconds`: hover duration. Defaults to `hold_time_s` from cfg.

Done when the timer expires.

---

### `AWAIT <marker-id> <timeout-seconds>`

> Hover (like `HOOVER`) but exit early as soon as a specific marker is
> seen.

Behaviour matches `HOOVER`:

- **After `APPROACH`** — station-keeping HOLD on the previously
  approached marker.
- **Otherwise** — IDLE.

The difference is the early-exit condition: as soon as `marker-id`
appears in the detector's visible-marker set, the script advances
immediately. If the marker is never seen, the step ends after
`timeout-seconds`.

Arguments:
- `marker-id`: id to watch for. Required.
- `timeout-seconds`: hard upper bound. Required.

Done when the awaited marker is seen *or* the timeout expires.

---

### `PAUSE <seconds>`

> Unconditional IDLE for the given seconds.

Sends `rc = (0,0,0,0)` for the whole duration regardless of what came
before. No HOLD station-keeping, no early exit. Useful for fixed
pauses — e.g. between two `APPROACH` steps when you want a clean stop
rather than carrying station-keeping state forward.

Arguments:
- `seconds`: pause duration. Required.

Done when the timer expires.

---

### `HEIGHT [<height>]`

> Drive the drone to a specific altitude.

Climb or descend until the drone is at `height` metres (clamped to
`[min_height_m, max_height_m]`) and hold there briefly.

Arguments:
- `height`: target altitude in metres. Defaults to `default_height_m`.

Done when the height has settled inside the deadband for the configured
settle time (~1 s).

---

### `TO <x> <y> [<z>] [<yaw>|auto]`

> Drive the drone to an arena-frame coordinate.

Coordinates use the arena frame: origin at the floor centre, `+x`
right, `+y` toward the front wall, `+z` up. With two arguments the
drone moves in the horizontal plane and the Anafi's onboard
stabiliser holds whatever altitude it's currently at; with three the
third is a target altitude (clamped to `[min_height_m, max_height_m]`).

The fourth (optional) argument controls the drone's arena-frame yaw
target in degrees (CW from the front wall). Use a numeric value to
pin a heading explicitly, or `auto` (the default when the argument
is omitted *and* a `z` was given) to let the controller pick a yaw
that maximises the number of well-placed reference markers visible
from the target — the drone arrives facing wherever the camera will
get the cleanest world-position fix.

The controller drives only when both the world-position estimate
*and* the derived arena yaw are fresh (≤ `pose_max_age_s`). If
either goes stale mid-step (e.g. all reference markers leave the
camera frame), the controller falls back to a yaw-search in place
until a fresh fix arrives — the drone never flies open-loop on a
dead estimate.

Arguments:
- `x`, `y`: target coordinates in metres. Required.
- `z`: target altitude in metres. Optional; omit to keep the current
  altitude.
- `yaw`: target arena-frame yaw in degrees, or `auto`. Default:
  `auto` (when the controller has an arena config; otherwise the
  drone keeps its current heading). The yaw arg requires `z` to be
  given (positional grammar).

Done when horizontal error is inside `distance_deadband_m`, height
error is inside `height_deadband_m` (when `z` was given), *and* yaw
error is inside `yaw_deadband_deg` (when a yaw target is in effect)
for the configured settle time.

---

### `DANCE [<seconds>] [<mode>]`

> Programmed RC routine for a fixed time, bounded inside a small radius.

Modes:
- `wobble` — yaw side-to-side plus a small up/down bob.
- `spin` — yaw only (continuous CW rotation).
- `random` — procedural lr/fb/ud/yaw oscillation.

A position estimator tracks the drone's drift since the dance started.
If it strays more than `dance_radius_m` (default 0.5 m) from the entry
position, the channel pushing it outward is zeroed and an inward
correction is added to bring it back.

Arguments:
- `seconds`: dance duration. Defaults to `default_dance_seconds_s`.
- `mode`: one of `wobble` (default), `spin`, `random`.

Done when the timer expires.

---

### `LAND`

> Land the drone, then end the script (or move to the next step).

The controller asks the drone to land and waits until telemetry reports
both `flying = False` and a near-ground height (or a 5 s descent
timeout).

If `LAND` is the last command, the script transitions to DONE. If you
forgot to put `LAND` at the end, a safety LAND is appended automatically.
A script with `LAND` followed by another command is legal — the drone
takes off again.

---

## Flight phases

Phases are the per-tick state machine the controller runs at 10 Hz.
Every tick it dispatches on the current phase, computes RC commands, and
sends them to the drone.

### INIT

> Park on the ground, wait for the operator to press *Start*.

The control thread is alive but holding zero RC. Telemetry and pose
detection are still running, so the camera page stays live during
pre-flight setup. Pressing *Start* releases the thread into the script's
first command.

---

### TAKEOFF

> Get airborne and reach a known altitude.

Two stages, in order:

1. Issue `api.takeoff()` and wait until `tel.flying = True`.
2. Climb to `default_height_m` while holding zero on the other channels.

Exits to whatever phase the next script command picks. Most commonly
SEARCH (start of an `APPROACH`), but could be HEIGHT, DANCE, etc.

---

### SEARCH

> We can't see the marker. Spin in place looking for it.

Yaw at a fixed slow rate (`search_yaw_rc`) clockwise. Each tick:

- If the marker becomes visible → exit to HEIGHT_ALIGN.
- If we've yawed a full 360° without ever seeing it → terminate the
  script and land. (The marker isn't here; subsequent commands wouldn't
  make sense.)

This phase is also re-entered if the marker is lost mid-flight (e.g.
during APPROACH or HOLD) for longer than the grace window — a recovery
path.

---

### HEIGHT_ALIGN

> Match the drone's altitude to the marker's altitude.

Once we can see the marker, we know its world altitude (drone height
minus the marker's offset in the camera frame). Drive the drone up or
down until they match — clamped to the safe altitude envelope. While
this happens, yaw still keeps the marker centred so it doesn't drift out
of frame.

Exits when both the height and the yaw have stayed inside their
deadbands for the settle time (~1 s).

> *Why bother?* Pose accuracy degrades sharply when the marker is far
> off-axis vertically. By matching altitudes first, the rest of the
> APPROACH is geometrically clean.

---

### ALIGN

> Slide sideways around the marker until we're facing it head-on.

Holds the current distance (orbit radius). The lateral channel walks
the drone tangentially around the marker until `relative_heading ≈ 0`
(drone directly in front of the marker's face). Yaw still actively
keeps the marker centred.

Exits when the heading and the centring have both settled (looser
deadband than APPROACH, since long-range pose is noisy).

> *Why bother?* The ArUco detector becomes unreliable at high tilt
> angles. Approaching head-on keeps the marker inside the detector's
> reliable range for the entire closure.

---

### APPROACH

> Close the gap to the target distance.

By the time we're here, the drone is already at the right altitude and
roughly head-on. The fly-forward channel does the actual work; the
others (yaw, lateral, height) keep their corrections going so we don't
drift on the way in.

Per tick, four things happen at once:

1. **Rotate** to keep the marker centred in the camera image.
2. **Fly forward** to close distance to `target_distance_m`.
3. **Slide sideways** to keep `relative_heading` near 0.
4. **Climb / descend** to keep tracking the marker's altitude.

A safety floor (default 70 % of target distance) clamps the forward
channel to ≤0 if the drone gets too close — the other channels keep
working so it can recover.

Exits when distance and rotation both settle for ~1 s.

---

### HOLD

> Station-keep on the marker.

Same four corrections as APPROACH (rotate, distance, sideways, altitude),
but the **distance setpoint is now frozen** at `target_distance_m`
instead of being closed-in to. The drone actively fights wind, drift,
and sensor noise to stay in place.

Exits when the configured hold timer expires. If the marker is briefly
lost, the controller tolerates it (zero RC for the grace window). If
the loss persists, it falls back to SEARCH.

---

### IDLE

> Hover with no marker tracking.

Sends `rc = (0,0,0,0)` and lets the Anafi's own stabiliser hold the
drone roughly in place. Used for `HOOVER` steps that don't follow an
`APPROACH` (so there's nothing to track).

Exits when the timer expires.

> Why this and not HOLD? Because HOLD requires a visible marker. If the
> operator says `TAKEOFF / HOOVER 5 / LAND`, there's no marker yet —
> IDLE is the honest behaviour.

---

### HEIGHT

> Drive the drone to a specific altitude and stop there.

The forward, lateral and yaw channels are zero. The up/down channel
runs a controller toward the target height (clamped to the safe
envelope). Once the drone is at altitude and has stayed there for the
settle time, we move on.

Exits when the height has been inside the deadband for ~1 s. Without
height telemetry, exits after a 5 s timeout so the script doesn't hang.

---

### DANCE

> Programmed RC routine, position-bounded.

Each tick computes a base RC command from the chosen mode and the
elapsed time:

- `spin` — constant yaw, all other channels zero.
- `wobble` — yaw oscillating, plus a small ud bob.
- `random` — all four channels oscillating at different rates.

Then a position-bound layer kicks in. A dead-reckoning estimator tracks
the drift in metres from the dance entry point. If that drift exceeds
`dance_radius_m`:

- Any RC channel pushing further outward is zeroed.
- An inward correction (proportional to how far past the radius we are)
  is added to bring the drone back.

Same logic applies vertically. The result is a routine that "stays
within roughly half a metre of where it started".

Exits when the timer expires.

---

### LAND

> Touch down and stop.

Sends zero RC for one tick (so the drone is stable), then issues
`api.land()`. Waits until telemetry reports `flying = False` AND a
near-ground height — the Anafi flips `flying=False` the moment it
*starts* descending, so we have to keep recording until it's actually
down.

A 5 s descent timeout safeguards against telemetry stalls.

If LAND was the last script command → DONE. Otherwise → next command
(scripts can take off again).

---

### DONE / ABORT

> Terminal. Mission is over.

DONE = nominal end. ABORT = something went wrong (timeout, API error,
no marker found in SEARCH, etc.). Both end the control thread; the
mission loop then resets state for the next run.

---

## Quick lookup table

| Phase | Active channels (per tick) | Exit condition |
|---|---|---|
| INIT | none (parked) | Operator presses *Start* |
| TAKEOFF | ud (climb to default height) | Drone at default height |
| SEARCH | yaw (sweep) | Marker found, or full 360° sweep → land |
| HEIGHT_ALIGN | ud + yaw | Height + yaw inside deadbands for ~1 s |
| ALIGN | lr + fb (radius hold) + yaw | Heading + yaw inside deadbands for ~1 s |
| APPROACH | lr + fb + ud + yaw | Distance + yaw inside deadbands for ~1 s |
| HOLD | lr + fb + ud + yaw | Hold timer expires |
| IDLE | none (rc = 0,0,0,0) | Idle timer expires |
| HEIGHT | ud only | Height inside deadband for ~1 s |
| DANCE | mode-dependent + position bounds | Dance timer expires |
| LAND | none (api.land + wait) | `flying=False` + height ≈ 0 |
| DONE / ABORT | terminal | — |

## How the script maps to phases

| Script command | Walks through these phases |
|---|---|
| `TAKEOFF` | TAKEOFF |
| `APPROACH` | SEARCH → HEIGHT_ALIGN → ALIGN → APPROACH |
| `HOOVER` (after APPROACH) | HOLD |
| `HOOVER` (after `TO`) | GOTO (station-keeping at the world target) |
| `HOOVER` (otherwise) | IDLE |
| `AWAIT` (after APPROACH) | HOLD with marker-id early-exit |
| `AWAIT` (otherwise) | IDLE with marker-id early-exit |
| `PAUSE` | IDLE (unconditional, no early exit) |
| `HEIGHT` | HEIGHT |
| `TO` | GOTO |
| `DANCE` | DANCE |
| `LAND` | LAND → DONE (or next command if any) |
