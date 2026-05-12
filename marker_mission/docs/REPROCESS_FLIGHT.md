# Offline flight re-processor

`tools/reprocess_flight.py` re-runs the marker detector + arena
estimator over a recorded flight's `raw.mp4` and produces two
annotated comparison videos: an **OLD** pass (no magnetometer) and
a **NEW** pass (magnetometer-aided IPPE branch picker). It exists
primarily to diagnose IPPE branch-flip incidents — when the live
pipeline locked onto the wrong planar-pose candidate and the drone
tracked the geometric mirror of the intended trajectory — and to A/B
candidate magnetic-north offsets without flying again.

It does **not** modify the recorded flight: the live `flight_log.csv`,
`annotated.mp4` and `mission_meta.json` are untouched. New videos are
written next to them (or to `--output-dir`) as `reprocessed_old.mp4`
and `reprocessed_new.mp4`.

---

## What it reads

| File | Purpose |
|---|---|
| `<flight-dir>/raw.mp4` | source frames (the only thing decoded; annotated.mp4 is ignored) |
| `<flight-dir>/flight_log.csv` | maps frame index → telemetry (`tel_yaw`) for the magnetometer pick |
| `<flight-dir>/mission_meta.json` | calibration source (serial + resolution) and intrinsics fallback |
| `<flight-dir>/cfg_start.json` (via `mission_meta`) | `marker_size_m` + `aruco_dict` |
| `~/.marker_mission/calibrations/anafi_<serial>_<resolution>.npz` | preferred calibration (matches what the live mission used, including distortion) |
| `--arena <path>` or `~/.marker_mission/active_arena_config.json` | arena layout (markers + walls + magnetic-north offset) |

If the calibration store doesn't have a matching entry, the tool falls
back to `mission_meta.json`'s intrinsics with zero distortion
coefficients and prints a warning. Sub-pixel corner-position
differences from the missing distortion model can flip the branch on
borderline frames, so prefer the NPZ where available.

## What it writes

`<output-dir>/reprocessed_old.mp4`, `<output-dir>/reprocessed_new.mp4`
— both with the standard marker overlay plus the arena mini-map in
the upper-right corner. The mini-map shows:

- 1 m grid + wall colours
- arena marker positions
- the drone's computed world position as a yellow dot
- a yellow heading arrow whose direction is each pipeline's own
  derived arena yaw (so when the magnetometer pick swaps the branch
  in NEW but not OLD, the two arrows visibly disagree by 20–60°)
- a `OLD` / `NEW` title in the corner

`stdout` summary at the end:

```
[reprocess] DONE
  frames processed:  <N>
  OLD frames w/ fix: <count>
  NEW frames w/ fix: <count>
  NEW frames using ippe_mag_swap: <count>
  -> .../reprocessed_old.mp4
  -> .../reprocessed_new.mp4
```

`ippe_mag_swap` counts ticks where the NEW pipeline's magnetometer
pick replaced the chosen branch with the loser branch — the headline
metric for whether the magnetometer offset would have rescued the
flight.

## CLI

```
python -m marker_mission.tools.reprocess_flight \
    --flight-dir <path>           # required
    [--mag-offset DEG]            # override the arena's magnetic_north_arena_yaw_deg
    [--arena <path>]              # arena config JSON (default: ~/.marker_mission/active_arena_config.json)
    [--output-dir <path>]         # default: --flight-dir
```

| Flag | Effect |
|---|---|
| `--flight-dir` | The recorded flight directory (the one containing `raw.mp4`). |
| `--mag-offset DEG` | Inject this magnetic-north arena-yaw offset into the NEW pipeline, overriding whatever the arena config has. Useful for sweeping candidate offsets without editing the arena. |
| `--arena <path>` | Use this arena config JSON for both pipelines. Defaults to the active arena. |
| `--output-dir <path>` | Where to write the reprocessed videos. Defaults to writing them inside `--flight-dir`. |

### OLD vs NEW semantics

* **OLD**: arena copy with `magnetic_north_arena_yaw_deg = None` and
  `tel_yaw_deg=None` passed to `estimate_position`. Isolates the
  pre-magnetometer branch picker — it still benefits from the
  mirror-collapse / OOB-filter / alt-branch / aggregate-OOB layers
  that landed before the magnetometer feature, but no magnetometer
  pick.
* **NEW**: arena copy with the offset injected (CLI override, or the
  arena's own value), and `tel_yaw_deg = tel.yaw` from the closest
  CSV row. Full magnetometer logic runs.

When neither the arena nor `--mag-offset` provides an offset, NEW
falls back to the same logic as OLD and the tool prints a warning —
the two videos will be identical (modulo random tie-breaks).

### Per-stream prev-anchor

Each pipeline carries its own `(prev_position_m, prev_age_s)` so the
prev-anchor / sticky-position state doesn't leak between OLD and NEW.
Sticky carry-forward matches `vision_worker`'s behaviour: when
`estimate_position` returns `None`, the previous fix stays cached.

## Examples

Re-process the wall-incident flight with a hand-measured offset:

```bash
python -m marker_mission.tools.reprocess_flight \
    --flight-dir ~/.marker_mission/flights/2026-05-05_14-45-09_PI040416BA8G061033 \
    --mag-offset -37.0
```

Sweep candidate offsets on the same flight:

```bash
for off in -45 -40 -37 -33 -30; do
  python -m marker_mission.tools.reprocess_flight \
      --flight-dir ~/.marker_mission/flights/2026-05-05_14-45-09_PI040416BA8G061033 \
      --mag-offset $off \
      --output-dir /tmp/reproc_$off
done
```

Compare the live recording against the (already-calibrated) magnetometer
pipeline using the active arena's offset:

```bash
python -m marker_mission.tools.reprocess_flight \
    --flight-dir ~/.marker_mission/flights/<flight_id>
```

## Caveats

* The reprocessor decodes `raw.mp4`. If the post-flight re-encoder
  rewrote `raw.mp4` to H.264, that's still the right input — the
  re-encode preserves frame timing.
* CSV-row alignment uses `frame_index / fps` against the CSV's
  `monotonic` column. Flights where the camera dropped a long burst
  of frames mid-flight will see a small drift in `tel_yaw` lookup;
  in practice the magnetometer picker tolerates this (the offset
  threshold is ±12° vs sub-degree tracking error).
* The live `flight_log.csv` and the reprocessor's NEW pipeline can
  still diverge by a few percent on borderline frames — corner noise
  from MJPEG compression isn't reproducible across decoder runs, and
  the temporal cache evolves slightly differently. The point of the
  tool is the qualitative OLD-vs-NEW comparison, not bit-exact
  reproduction.
