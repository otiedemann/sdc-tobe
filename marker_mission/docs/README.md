# Marker-mission control for Parrot Anafi

A small Python application that:

1. Connects to your existing `unified_api_server.py` over HTTP.
2. Streams the drone's camera and detects ArUco markers.
3. Recovers each marker's 6-DOF pose using OpenCV's `solvePnP` with a
   per-drone camera calibration.
4. Walks an **operator-typed mission script** through a phase-based
   state machine. The script language has six commands —
   `TAKEOFF`, `APPROACH`, `HOOVER`, `HEIGHT`, `DANCE`, `LAND` — and
   omitted arguments fall back to cfg defaults. The default script is
   `TAKEOFF / APPROACH / HOOVER / LAND`, equivalent to the original
   fixed flow.
5. Runs a five-screen operator UI: live camera, charts, replay
   browser, live tuning, calibration capture.
6. Writes the raw video, annotated video, the executed mission
   script, the cfg snapshot at takeoff and landing, a parameter-change
   log, and a per-tick CSV to disk for every flight.

## Mission-language and flight-phase reference

Two companion documents:

- [`MISSION_REFERENCE.md`](MISSION_REFERENCE.md) — plain-English overview
  of every script command and flight phase. Read this first.
- [`MISSION_REFERENCE_DETAILED.md`](MISSION_REFERENCE_DETAILED.md) — the
  same material with the deadbands, settle times, edge cases, recovery
  paths, and pointers into `controller.py` / `mission_script.py` filled
  in. The thing to keep open while reading the source.

## Conventions and sign rules

Everything below uses the **aerospace right-hand convention**: angles
about the vertical are signed with the right-hand rule about world-up,
which means **positive = clockwise when viewed from above** (= "nose
right" for yaw).

The Anafi's gimbal keeps the camera horizontal at all times, which we
exploit so that all reported angles are in the **world horizontal
plane** and are therefore invariant to:

* drone roll and pitch (the gimbal absorbs them),
* how the marker is rotated within its own plane on the wall (upright,
  sideways, upside down — all give the same `yaw_to_marker_deg` and
  `relative_heading_deg`).

### Coordinate frames

![Camera and world coordinate frames](figures/camera_and_world_frames.svg)

The detector receives images in the standard OpenCV camera frame (left
panel). Because the gimbal holds the camera level, the camera's
horizontal axes (`+x_cam`, `+z_cam`) coincide with the world horizontal
plane (right panel), and world-up is exactly `−y_cam`. All horizontal
angles below are computed in this plane.

### Yaw to marker

![Yaw to marker, top-down view](figures/yaw_to_marker.svg)

`yaw_to_marker_deg` is the signed angle between the drone's nose
direction and the line from the drone to the marker centre, measured
in the horizontal plane.

* `yaw_to_marker_deg = 0` ⇒ marker dead ahead in the camera image.
* `yaw_to_marker_deg > 0` ⇒ marker is on the camera's right ⇒ drone
  must yaw RIGHT (CW from above, positive aerospace yaw) to face it.
* The controller commands `yaw RC = +k · yaw_to_marker_deg`.

### Relative heading around the marker

![Relative heading, top-down view](figures/relative_heading.svg)

`relative_heading_deg` is the drone's compass bearing **at the marker**,
measured around world-up, with **0° = directly in front of the marker
face** and CW positive (from above).

The script's `APPROACH` command always pins this setpoint to **0°**
so the drone closes the gap head-on (the detector is most reliable at
small tilt angles). HOLD inherits the same 0° setpoint via
`state.target_relative_heading_deg`. If you want a non-zero HOLD
heading, tune `cfg.target_relative_heading_deg` *between* APPROACH and
HOOVER — currently no script syntax exposes this directly.

To slide CW around the marker (heading 0 → +30), the drone — while
keeping the marker centred in the camera (yaw = 0) — must step to its
**own LEFT** (`lr` RC negative). The controller takes care of this
sign automatically; you should not have to think about it unless you
modify the lateral law.

### RC channels (matches the unified API server)

| channel | + sign means | drone moves                              |
|---------|--------------|------------------------------------------|
| `lr`    | right        | strafe to the drone's right              |
| `fb`    | forward      | nose-forward translation                 |
| `ud`    | up           | climb                                    |
| `yaw`   | clockwise    | yaw CW from above (aerospace nose-right) |

### Marker frame and gimbal robustness

`solvePnP` returns the marker's pose in the camera frame. The marker's
intrinsic axes (its own `+x` = "to the viewer's right when looking at
its face", `+y` up, `+z` out of the face) **rotate with the marker** if
someone hangs the marker sideways or upside-down. We avoid this by
projecting the relevant vectors onto the camera's horizontal plane (=
world horizontal plane, thanks to the gimbal) and computing all
horizontal angles there. The diagnostic field `marker_inplane_rot_deg`
tells you how the marker is rotated within its own plane, but the
control outputs (`yaw_to_marker_deg`, `relative_heading_deg`) are
invariant to that rotation.

`marker_tilt_deg` reports how far the marker plane deviates from
vertical (0° = vertical wall, ±90° = floor or ceiling). A marker on
the floor or ceiling has no horizontal-plane normal, so heading
becomes geometrically undefined; the detector returns `0` and the
operator should notice via `marker_tilt_deg ≈ ±90°`.

### Mission state machine

The flight phases are: `INIT → TAKEOFF → SEARCH → HEIGHT_ALIGN → ALIGN
→ APPROACH → HOLD → IDLE → HEIGHT → DANCE → LAND → DONE / ABORT`.
Which phases run, and in what order, depends on the operator's script
— see [`MISSION_REFERENCE.md`](MISSION_REFERENCE.md) for the
command → phase chain. The marker-loss recovery path returns to
`SEARCH` from any APPROACH-style phase; if a full sweep doesn't
reacquire the marker the script is truncated and the drone lands.

## Layout

```
marker_mission/
├── config.py                            # tuning constants + tune-page schema
├── calibration_store.py                 # per-serial-number camera intrinsics
├── aruco_detector.py                    # detection + pose extraction
├── drone_api.py                         # HTTP client and MJPEG reader
├── controller.py                        # PID controllers + state machine + script driver
├── mission_script.py                    # operator-typed script parser
├── recorder.py                          # raw/annotated video + CSV log
├── replay.py                            # post-flight playback through the live UI
├── web_calibration.py                   # in-UI calibration capture
├── ui.py                                # five-screen Flask UI
├── mission.py                           # entry point (CLI)
├── README.md                            # stub pointing at docs/
├── docs/
│   ├── README.md                        # this file
│   ├── MISSION_REFERENCE.md             # plain-English command + phase reference
│   ├── MISSION_REFERENCE_DETAILED.md    # detailed reference with code pointers
│   └── figures/
│       ├── camera_and_world_frames.svg
│       ├── relative_heading.svg
│       ├── yaw_to_marker.svg
│       └── mission_state_machine.svg   # legacy 7-phase diagram (predates the script feature)
└── requirements.txt
```

## Install

```bash
python -m pip install -r marker_mission/requirements.txt
```

## Configure once

The first run creates `~/.marker_mission/config.json`. Edit it (or
override via `MM_*` env vars or CLI flags) to set:

* `target_marker_id`, `marker_size_m`
* `target_distance_m`, `target_relative_heading_deg`, `hold_time_s`
* PID gains, velocity feedforward (`kv`), and safety clamps (every
  tunable line is tagged `# TUNE` in `config.py`)
* `api_base_url` (the URL of `unified_api_server.py`)

Most parameters are also editable live via the **Tune** page in the UI
without restarting the mission. PD remains the default behaviour
because each axis defaults to `ki = 0`; bump the I gain via `/tune` to
opt the axis into PID control on the fly.

## Calibrate each drone (one-time per unit)

1. Print a 9 × 6 inner-corner checkerboard, glue it to a flat board,
   measure one square side precisely (e.g. 25 mm).
2. With the drone on the ground (motors off), record a video of the
   board from many angles and distances. Use the **same resolution and
   stabilisation settings** you fly with. About 30–40 seconds is enough.
3. Save the file as `calib.mp4` and run:

   ```bash
   python -m marker_mission.mission calibrate \
       --video calib.mp4 \
       --serial PI040421AA1234 \
       --resolution 720p \
       --pattern 9x6 \
       --square 0.025
   ```

   Use the actual serial returned by your API server's `/api/telemetry`
   (`serial_number` field). The calibration file is saved at
   `~/.marker_mission/calibrations/anafi_<serial>_<resolution>.npz`.
   Reprojection error should be **< 0.5 px** for a good calibration.

The mission script automatically loads the right file at startup based
on the drone's serial. If no calibration is found it falls back to the
published Anafi defaults and prints a warning — the mission will still
fly, but pose accuracy will be reduced (≈ 1° / 5 cm vs. 0.3° / 2 cm).

## Fly

```bash
python -m marker_mission.mission                         # uses config defaults
python -m marker_mission.mission --distance 1.5 --hold 30
python -m marker_mission.mission --marker-id 7 --heading -90
```

The operator UI starts on `http://<host>:8080/`:

* `/`           live camera with overlay, mission status table, and
                the **mission-script editor** (textarea in INIT,
                step-by-step list with the active step highlighted
                during flight). Editor autosaves on every keystroke;
                a "Saved scripts" disclosure exposes named save / load.
* `/charts`     live charts of distance, yaw, relative heading,
                telemetry (drone yaw / battery / height), and RC
                commands.
* `/replay`     browse recorded flights; per-flight playback view at
                `/replay/<flight_id>` plays the recorded video synced
                with the saved CSV and shows the executed script with
                the active step highlighted, just like the live page.
* `/tune`       live PID-gain editing with named snapshots
                (save / load / overwrite / delete). Changes apply
                instantly to the running controller and are appended
                to `<flight_dir>/parameter_changes.csv`.
* `/calibrate`  in-UI camera calibration capture for the connected
                drone. Records a chessboard video, runs the OpenCV
                solver, saves the per-serial NPZ.

### Pre-flight script

The textarea on `/` is parsed at Start. Examples:

```
# Default flow (same as the legacy fixed mission)
TAKEOFF
APPROACH 4 1.0
HOOVER 5
LAND
```

```
# Visit two markers in sequence, then dance, then land
TAKEOFF
APPROACH 4 1.0
HOOVER 3
APPROACH 7 0.8
DANCE 5 wobble
LAND
```

```
# Just an altitude scan, no marker tracking
TAKEOFF
HEIGHT 1.0
HEIGHT 2.0
LAND
```

See [`MISSION_REFERENCE.md`](MISSION_REFERENCE.md) for the full
command reference.

## Safety notes (READ BEFORE FLYING)

The PID gains in `config.py` are educated initial values, **not**
values verified on real hardware. The default `ki = 0` on every axis
keeps the controller behaving exactly like the older PD until you opt
an axis into integral control via `/tune`. Before any free flight:

1. Test with the propellers removed first. While the controller is in
   INIT phase it runs a *pre-flight dry-run* — the same APPROACH-style
   PIDs compute RC values you can watch on the camera page without
   anything being sent to the drone. Move the marker board around in
   front of the still drone and confirm the displayed RC values make
   sense.
2. Then test in a tethered or netted environment with conservative
   `*_rc_max` clamps.
3. Watch the search behaviour: if the drone yaws in the wrong
   direction for your marker layout, flip the sign of `search_yaw_rc`
   or yaw the drone manually first.
4. Verify the angle-sign convention with a quick test: with the drone
   stationary, present the marker slightly to the camera's right and
   confirm that `yaw_to_marker_deg` reads **positive** in the UI. (See
   the "Conventions and sign rules" section above for the full picture.)
5. Verify the orbit-direction sign: on the ground, with the drone
   stationary and the marker dead ahead, manually slide the *marker*
   to the marker's own right (drone's view: left). The displayed
   `relative_heading_deg` should swing **negative** (drone is now to
   the marker's left side from the marker's POV). With ALIGN's lateral
   law the drone slides to its own left to drive the heading back
   toward 0; if it slides the wrong way, flip the negation in
   `_step_align` / `_step_approach` / `_step_hold`'s
   `arc_err_m = -d * math.radians(e_hdg)` line.

## Per-flight artefacts

Every flight creates a directory **at TAKEOFF** (lazy creation — runs
that never fly leave the flights folder untouched):

```
~/.marker_mission/flights/<YYYY-mm-dd_HH-MM-SS>_<serial>/
├── raw.mp4                 # raw camera frames
├── annotated.mp4           # frames with marker overlay + HUD baked in
├── flight_log.csv          # per-tick state (~5 Hz)
├── mission_script.txt      # the operator-typed script that flew
├── cfg_start.json          # cfg snapshot at takeoff
├── cfg_end.json             # cfg snapshot at landing (may differ if /tune was used mid-flight)
├── parameter_changes.csv   # one row per /tune change during the flight
└── mission_meta.json       # calibration source + final phase + abort reason
```

`flight_log.csv` columns include the smoothed marker pose, the marker
id and `marker_seen` flag, the active mission-step index, the RC
commands sent, and the drone telemetry (`vgx / vgy / vgz / yaw / pitch
/ roll / battery / height_cm / flight_time_s`). The Replay browser
uses `mission_script.txt` + `mission_step_idx` to reconstruct the
step-by-step view during playback.

The runtime persists three other files outside the flight directories:

```
~/.marker_mission/
├── config.json                    # the cfg defaults (tune-page persistence)
├── active_mission_script.txt      # the textarea draft (debounced auto-save)
├── snapshots/<name>.json          # named tuning snapshots
├── mission_scripts/<name>.txt     # named mission scripts
├── calibrations/anafi_<serial>_<resolution>.npz
└── flights/...                    # see above
```
