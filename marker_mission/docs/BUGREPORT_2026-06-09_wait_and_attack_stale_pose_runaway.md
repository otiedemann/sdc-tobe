# Bug report — WAIT_AND_ATTACK runaway after marker occlusion (stale-pose drive)

**Status:** open · for analysis (FC / `marker_mission` controller)
**Severity:** high — uncommanded full-stick flight in the wrong direction; marker lost; near-runaway
**Component:** `marker_mission/controller.py` (approach / marker-lost / WAIT_AND_ATTACK path)
**Reporter:** C2 V2 replay analysis
**Date observed:** 2026-06-09

---

## TL;DR

During a `WAIT_AND_ATTACK 46` step the target box was **occluded by a person** for ~25 s.
When the marker flickered briefly into view, the controller entered the **APPROACH** phase
and drove the drone with **large, erratic, saturated RC** — including a **full-forward
`rc_fb = 100` for ~2.4 s** — while its **distance estimate stayed frozen at 9.15 m** even
though the drone **physically moved ~4–5 m**. `active_marker_id` was **`None` the whole time**.
The drone "shot off" ~4–5 m to the wrong part of the arena, then lost the marker and fell
back to SEARCH.

**It was *not* a frozen camera image** — the video pipeline was healthy
(`camera_restart_count = 0`, `telemetry_stalled = false`) and the *marker-last-seen age*
was **climbing** (0 → 2.6 s), which a frozen frame still showing the marker would keep ~0.
The thing that was "frozen" is the **marker position estimate inside the control loop**:
the APPROACH kept steering toward a **stale fix** instead of recognising the marker was gone.

---

## Environment

| | |
|---|---|
| Drone | `PI040416BA8G061033` (flightctrl3) |
| Replay | `http://flightctrl3:8080/replay/2026-06-09_15-52-59_PI040416BA8G061033` |
| Flight duration | 53.3 s (timeline ~5 Hz, 275 samples) |
| Related recent change | `369095d` "marker_mission: WAIT_AND_ATTACK no longer blocks marker-lost recovery" (Sascha) |

### Mission script

```
0  TAKEOFF
1  HEIGHT 1.3
2  WAIT_AND_ATTACK 46 1.5 0.2     <-- problem occurs here (step never completed)
3  FB_UD_IMU 1.5 0.5
4  YAW_IMU 180
5  GO_HOME 33 2 0.5
6  FB_UD_IMU 1.5 0.5
7  YAW_IMU 180
8  REPEAT
```

The flight never advanced past step 2: `mission_step_idx` stayed at 2 (`WAIT_AND_ATTACK`) for
the entire window analysed.

### Relevant config (default `MissionConfig`; the drone's tuned values may differ — see open questions)

```
pose_max_age_s            = 0.8     # smoother returns a held pose for up to this long
search_marker_lost_grace_s = 1.5    # _marker_lost zeroes RC for this long before SEARCH
fwd_kp = 30.0   fwd_kd = 30.0       # forward PD
fwd_rc_max = 3                       # PD output clamp (default; drone clearly higher — rc_fb hit 100)
```

---

## Symptom (operator's words)

> "During the WAIT_AND_ATTACK command it did not find the target (someone stood in front of it),
> the drone suddenly shot off in the wrong direction after executing a recovery attempt."

---

## Timeline (from the replay)

Marker-visibility transitions (target/any marker by distance feed):

| t (s) | event |
|------:|-------|
| 0.0 – 25.1 | target marker **never detected** — `WAIT_AND_ATTACK` SEARCH yaw-spinning (`rc_yaw≈20`). Person occluding box. |
| 25.1 | marker flickers into view at **5.5 m**, at frame edge (rel. heading ≈ −45°). |
| 38.7 | marker fully lost → phase returns to SEARCH. |

Per-frame trace through the event (seek-sampled from `/api/replay/<id>/state`; `world` = vision arena pose):

| t (s) | idx | phase | active_mk | dist (m) | mk_seen_age (s) | world (x,y) | rc_fb |
|------:|----:|-------|:---------:|---------:|----------------:|:-----------:|------:|
| 24.1 | 2 | search   | None | –    | –    | (−2.05, −6.24) |   0 |
| 25.1 | 2 | approach | None | 5.50 | 0.20 | (−2.08, −6.31) |   0 |
| 26.1 | 2 | approach | None | 4.48 | 0.40 | (−2.06, −6.39) | **−93** |
| 28.1 | 2 | approach | None | 8.27 | 0.00 | (−0.90, −3.70) |  36 |
| 30.1 | 2 | approach | None | 8.67 | 0.00 | (−0.09, −3.56) |  −8 |
| 32.1 | 2 | approach | None | 8.55 | 0.00 | ( 1.25, −4.32) |  −8 |
| 34.1 | 2 | approach | None | 8.99 | 0.00 | ( 1.68, −4.30) |  −8 |
| 36.1 | 2 | approach | None | **9.15** | 0.00 | (−2.18, −1.12) | **100** |
| 37.1 | 2 | approach | None | **9.15** | 0.60 | (−3.10, −0.19) | **98** |
| 38.1 | 2 | approach | None | **9.15** | 1.60 | (−3.62, −2.63) |  78 |
| 39.1 | 2 | search   | None | –    | 2.61 | (−3.34, −4.29) | −25 |
| 40.2 | 2 | search   | None | –    | 3.62 | (−2.94, −5.72) | −25 |
| 42.1 | 2 | search   | None | –    | 5.62 | (−2.44, −6.01) | −25 |

Raw RC (5 Hz timeline) highlights:

- `t=25.3–26.3`: `rc_fb = −99`, `rc_lr = −20` — **full backward-left lunge right after acquiring a target 5.5 m *ahead*.**
- `t=26.9–28.1`: forward `rc_fb = 49…66` but **distance grows 5.2 → 8.1 m** (moving *away* from the marker).
- `t=36.1–38.5`: **`rc_fb = 100 … 70` for ~2.4 s**, distance **frozen at 9.15 m**, then marker lost at 38.7.

---

## Root-cause analysis

### The drove-on-a-stale-fix mechanism

1. The box was occluded, so target marker `46` was only seen in brief flickers, caught **oblique /
   at the edge of frame** while the drone was mid-SEARCH — a poor-quality pose.
2. The APPROACH phase (`_step_approach`) steers from `e_fwd = d − standoff`. With
   `d ≈ 9.15 m` and `standoff = 1.5 m`, `e_fwd ≈ 7.65 m`; `u_fwd = fwd_kp · e_fwd = 30 × 7.65`
   **saturates the forward channel** → `rc_fb` pins near full stick (≈100 on this drone). A *bad*
   distance estimate therefore produces a **violent full-stick lunge**, not a gentle correction.
3. The drive direction was wrong because the **heading was never aligned** to the (stale/oblique)
   marker — the forward command pointed roughly *sideways* to the real target. The clearest proof:
   between `t=34.1` and `t=38.1` the drone **physically moved ~4–5 m** (world `(1.68,−4.30) →
   (−3.62,−2.63)`) while the **reported distance never changed from 9.15 m**. A live measurement
   would have moved by metres; a **frozen number means the controller was acting on a held fix**, not
   a live one.
4. After the held fix aged out, the marker went fully lost (`mk_seen_age` 0 → 2.6 s) and the step
   dropped back to SEARCH (`t=38.7+`).

### The strongest anomaly (please focus here)

**The APPROACH phase drove full-stick forward with `active_marker_id = None` and a frozen
distance for ~2.4 s — i.e. it kept driving with no live marker lock.** Expected behaviour is that a
missing/stale pose stops forward drive quickly (`_marker_lost` zeroes RC after the grace), yet here
the drive persisted well beyond `pose_max_age_s = 0.8 s`. Something in the
WAIT_AND_ATTACK / recovery / smoother interaction kept `_step_approach` driving on a stale estimate
instead of declaring the marker lost.

### What it is NOT — a frozen camera image

Ruled out by the live signals during the burst:

| t (s) | mk_seen_age | camera_restart_count | telemetry_stalled | visible_marker_ids |
|------:|------------:|---------------------:|:------------------:|:-------------------|
| 34.1 | 0.00 | 0 | false | [13] |
| 37.1 | 0.60 | 0 | false | [13] |
| 38.1 | 1.60 | 0 | false | [13, 33] |
| 39.1 | 2.61 | 0 | false | [13] |

A frozen frame *still showing the target* would keep `mk_seen_age ≈ 0` and the same
`visible_marker_ids`; instead the age **climbs** and the visible set changes (`[13]` → `[13,33]`),
so the camera was streaming live frames of an occluded scene. The freeze was in the **estimate**,
not the picture.

---

## Code references

- `marker_mission/controller.py`
  - `_step_approach` (~L2173): `meas = self.smoother.get(now)`; `e_fwd = d − tgt_d`; `u_fwd = pd_fwd.step(e_fwd)` — saturates to `fwd_rc_max` for large `e_fwd`.
  - `_marker_lost` (grace) — commands zero RC only after `search_marker_lost_grace_s`; docstring already records an earlier crash from "coasting forward … rc_fb pinned".
  - `PoseSmoother.get` (~L285) + `pose_max_age_s` — returns the last pose for up to `pose_max_age_s`.
  - `_recover_via_central_marker` (~L1693) — recovery hop to a central wall marker; sets `arrive_hdg_tol_deg=None`, `arrive_yaw_tol_deg=None`, `approach_positioning=True`.
  - `_wait_attack_pre_tick` (~L1463) + target/sibling swap (~L520-533).
- Recent related fix: `369095d` (WAIT_AND_ATTACK no longer blocks marker-lost recovery).

---

## Open questions (for further analysis)

1. **Why does `_step_approach` drive full-stick for ~2.4 s with `active_marker_id = None` and a frozen distance?** What keeps `smoother.get()` non-None (or otherwise keeps the drive alive) past `pose_max_age_s = 0.8 s`? Is the WAIT_AND_ATTACK target/sibling swap or a recovery anchor holding a stale `target_distance_m`?
2. **Why is `active_marker_id = None` throughout an active APPROACH drive?** Display artefact, or is the controller genuinely driving with no locked id?
3. **The `−99` backward lunge at `t=25.3`** (target 5.5 m *ahead*): is `e_fwd` going negative (smoother briefly reporting `d < standoff`), or is this `_velocity_damp_fwd` / a sibling-hold back-off?
4. **World-position jump** `(1.68,−4.30) → (−2.18,−1.12)` between `t=34.1` and `36.1` (~5 m in 2 s): real motion, or a localisation glitch from the occlusion/oblique markers? If a glitch, it could be the trigger for the saturated drive.
5. What were the drone's **actual tuned** `fwd_rc_max` / `pose_max_age_s` (config snapshot in the flight dir), vs the library defaults quoted above?

---

## Recommended fixes (FC-side; not applied here)

1. **Gate large forward drive on a live, recent, heading-aligned detection.** Don't allow a saturating `rc_fb` unless the pose is fresh (e.g. `mk_seen_age < ~0.3 s`) AND `|e_yaw|`/`|e_hdg|` are within an alignment band — so a fresh oblique/stale pose can't trigger a full-stick lunge.
2. **Cut forward drive the instant the pose goes stale during APPROACH** (don't ride `pose_max_age_s` at full stick); the existing `_marker_lost` zero-RC should pre-empt the drive, not follow it.
3. **Slew-rate-limit / ramp the forward channel after (re)acquisition** so distance-error saturation can't produce an instant 0→100 stick step.
4. Consider an APPROACH **sanity check**: if commanded forward but the measured distance isn't decreasing over N ticks (drone moving, distance frozen), treat the fix as stale → stop + re-search.

---

## How to reproduce / inspect

- Replay UI: `http://flightctrl3:8080/replay/2026-06-09_15-52-59_PI040416BA8G061033`
- Programmatic per-frame state: `POST /api/replay/<id>/pause`, `POST /api/replay/<id>/seek?t=<s>`, then `GET /api/replay/<id>/state` (the wrapped note reports the actual frame time; reads lag the seek by ~1 frame, so converge on the reported time).
- Full chart arrays: `GET /api/replay/<id>/timeline` (battery, d, drone_yaw, height, rc_lr/fb/ud/yaw, t).
- Authoritative raw log on the drone: `<flights_dir>/2026-06-09_15-52-59_PI040416BA8G061033/flight_log.csv` (+ `mission_script.txt`) — not exposed over HTTP; pull from the Pi for a source-level pose/note trace.
