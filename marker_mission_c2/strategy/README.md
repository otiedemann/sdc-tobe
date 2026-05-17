# `marker_mission_c2.strategy` — reactive strategy layer for the fleet C2

> **What's new**: the default planner is now
> :class:`RoleAssignmentPlanner` — every tick the strategy decides each
> drone's role (`attacker` / `defender` / `scout` / `idle`) from live
> state plus :class:`StrategySettings`, and dispatches a role-specific
> task that pushes a `marker_mission` mission-script to the FC.
> The previous `role` setting now means "pin this drone to this role
> (override the auto-assignment)"; default `idle` = let the strategy
> decide. See [Role-aware execution](#role-aware-execution) below.


Five small modules that sit *on top of* the existing
`marker_mission_c2` machinery (`fc_pool`, `fc_client`, `config`, etc.)
without modifying any of it. The strategy stack decides what each
drone in the swarm should do **right now**, given the live state the
pool is already collecting.

```
┌─────────────────────────────────┐
│ marker_mission_c2/              │   existing: fleet poll + HTTP fan-out
│   fc_pool.FCPool                │
└────────────┬────────────────────┘
             │ snapshot(), client(name)
┌────────────▼────────────────────┐
│ strategy/world_model.py         │   builds SwarmState per tick
│   SwarmWorldModel.observe()     │
└────────────┬────────────────────┘
             │ SwarmState
┌────────────▼────────────────────┐
│ strategy/planner.py             │   picks one DroneTask per FC
│   SwarmPlanner.decide(state)    │
└────────────┬────────────────────┘
             │ dict[fc_name → DroneTask]
┌────────────▼────────────────────┐
│ strategy/tasks.py               │   per-tick FCCommand
│   task.tick(state) → FCCommand  │
└────────────┬────────────────────┘
             │ FCCommand
┌────────────▼────────────────────┐
│ strategy/safety.py              │   veto / modify before dispatch
│   gate(cmd, state) → Verdict    │
└────────────┬────────────────────┘
             │ FCCommand (possibly different)
┌────────────▼────────────────────┐
│ strategy/runner.py              │   1 Hz loop, dispatches via FCPool
│   SwarmRunner                   │
└─────────────────────────────────┘
```

## What lives in each module

### `world_model.py`
- Reads `FCPool.snapshot()` once per tick, turns each FC's mutable
  poll-cache into a frozen `DroneObservation`.
- Returns an immutable `SwarmState` of `{fc_name: DroneObservation}`.
- **Does not** make any HTTP call, mutate pool state, or decide
  anything.

### `tasks.py`
- `DroneTask` base class plus concrete shippable tasks: `Idle`,
  `StartMission(script=…)`, `StopMission`, `ApplyTune(updates=…)`,
  `SetArena(arena=…)`.
- Each task is a pure function of `SwarmState`: `tick(state) →
  FCCommand`. No I/O, no globals.
- `FCCommand` is the typed envelope the runner dispatches. `IDLE`
  commands are skipped — they cost nothing.

### `planner.py`
- `SwarmPlanner` protocol: anything with `decide(state) → Mapping[fc,
  DroneTask]`.
- Two ready implementations:
  - `StaticAssignmentPlanner({fc: task})` — fixed roles. Tasks run to
    `done()`, then the FC falls through to `Idle`.
  - `UtilityPlanner(candidates)` — scores each candidate per FC per
    tick, greedy-assigns best, supports per-candidate fleet-wide
    limits.
- Roll your own planner by implementing the one method.

### `safety.py`
- `SafetyConfig` dataclass: stale-poll, battery-critical,
  fence-warning bounds, stop-cooldown.
- `SafetyGate.gate(cmd, state) → SafetyVerdict` checks rules in
  order: manual hold → offline / stale → drone link dropped →
  battery → geofence (warn).
- Override `hold_all(True)` from the operator UI for an emergency
  freeze on the next tick.

### `runner.py`
- `SwarmRunner` owns the asyncio loop. Steps: observe → decide →
  tick → gate → dispatch. Dispatches all FCs in parallel with
  `asyncio.gather` so one slow FC can't stall the others.
- Optional `on_tick(record)` hook for SSE / logging / replay.

## Role-aware execution

The planner shipped in `app.py`'s `build_planner()` is now
`RoleAssignmentPlanner`. Per tick:

1. **`roles.score_attacker / score_defender / score_scout`** rate every
   drone for every role. Each score factors in: battery (with per-role
   floor), live position, an optional operator pin
   (`settings.drones[fc].role`). Pin = `Role.IDLE` (the default) means
   "auto"; any other value forces that role with score `1000`.
2. **Greedy assignment** under caps (`max_attackers=2,
   max_defenders=1` by default — exactly the doctrine for the 10-pt
   strike + one home-zone watcher). Tied scores break by drone name.
3. **Per-drone task** built by the matching `roles.build_*_task(...)`
   factory:
   - `Role.ATTACKER` → `SyncAttackPair` (paired with the other
     attacker on a qualifying target pair from `settings.attack_pair_targets()`),
     or `HoldAboveTarget` if no partner.
   - `Role.DEFENDER` → `WaitInNeutral`, unless intrusion detected →
     `ReclaimOnIntrusion` (intrusion detection currently stubbed —
     pending an enemy-pose channel).
   - `Role.SCOUT` → `WaitInNeutral` at the drone's neutral-band slot.
   - `Role.IDLE` → `Idle` (no commands sent).

Each role task generates a marker_mission mission-script and pushes
it via `START_MISSION`. The FC's own mission player executes it and
talks to Olympe — the strategy stays at 1 Hz, flight control runs at
the FC's native rate.

### The 10-point sync-strike, end-to-end

```text
Tick N      planner picks {fc1: ATTACKER → target 41,  fc2: ATTACKER → target 42}
            build_attacker_task wires fc1 with partner=fc2, fc2 with partner=fc1
            both tasks emit:  TAKEOFF / TO <target_xy> <hover_alt> / HOOVER 9999
            FCs fly to staging.

Tick N+k    both attackers' poses ≤ ready_radius_m of their respective targets
            SyncAttackPair flips both tasks to PHASE_STRIKE on the SAME tick
            both tasks emit:  TO <target_xy> <strike_alt> / HOOVER 2 / LAND
            FCs descend together (within one tick = ≤1 s synchronisation).

Tick N+k+m  fc1.flying = False → PHASE_DONE → task.done=True → planner
            re-evaluates this slot.
```

Tested with a fake pool: pinning a drone to a role with insufficient
battery still returns `IDLE` (safety floor wins over operator intent);
when battery recovers, the pin takes effect immediately on the next
tick.

### Tuning knobs

All in `StrategySettings`:

| Knob | Effect on the role planner |
| --- | --- |
| `attack.hover_alt_m`         | stage altitude for `HoldAboveTarget` / `SyncAttackPair` (PHASE_HOLD) |
| `attack.strike_alt_m`        | dive altitude for `SyncAttackPair` (PHASE_STRIKE) |
| `attack.sync_window_s`       | unused yet — placeholder for the time-bounded sync abort |
| `attack.pair_targets_override` | which marker pairs qualify; `null` derives every pair of own_target_ids |
| `defender.intercept_radius_m` | how close enemy must be to trigger `ReclaimOnIntrusion` (also unused until enemy positions are wired) |
| `arena.safety_margin_m`      | edge clip; safety gate still warns when a pose breaches |
| `drones[fc].role`            | operator pin; `idle` (default) = auto |
| `drones[fc].altitude_m`      | cruise altitude per FC — used by `WaitInNeutral` slot calc |

## Wiring it up

The default runner is a separate process from `marker_mission_c2.server`:

```bash
# Run the fleet C2 dashboard
python -m marker_mission_c2 --config config.json

# In a second terminal, run the strategy against the same FCs
python -m marker_mission_c2.strategy.app --config config.json --tick-hz 1
```

Both consume the *same* `config.json`; the runner opens its own
`FCPool` against those FCs and the existing dashboard stays
untouched.

To merge them into one process (so the dashboard and the strategy
share a pool), import the runner from your own program — see the
docstring on `strategy.runner.SwarmRunner`. That's the **only**
integration that would touch existing C2 code.

## Designing a real planner

The shipped planners are scaffolding. For a real challenge,
override `strategy.app.build_planner` (or pass your own to
`SwarmRunner(...)`). Two recipes:

### Static role assignment

```python
from marker_mission_c2.strategy.planner import StaticAssignmentPlanner
from marker_mission_c2.strategy.tasks import StartMission

scout_script  = open("scripts/scout.txt").read()
attack_script = open("scripts/attack.txt").read()

def build_planner(cfg):
    return StaticAssignmentPlanner({
        "flightctrl1": StartMission("flightctrl1", script=scout_script),
        "flightctrl2": StartMission("flightctrl2", script=attack_script),
        # … rest stay Idle by default
    })
```

### Reactive utility scoring

```python
from marker_mission_c2.strategy.planner import UtilityPlanner
from marker_mission_c2.strategy.tasks    import StartMission, Idle

def score_scout(state, fc):
    obs = state.drones[fc]
    if obs.flying:                       return float("-inf")
    if (obs.battery_pct or 0) < 50:      return float("-inf")
    return 10.0 - state.t * 0.0          # everyone equally eligible

def build_planner(cfg):
    return UtilityPlanner(candidates=[
        ("scout",  lambda fc: StartMission(fc, script=open("scripts/scout.txt").read()),
                   score_scout, 1),       # only one scout at a time
        ("idle",   lambda fc: Idle(fc),
                   lambda s, fc: 0.0,     None),
    ])
```

## Operator web UI

The strategy app embeds a tiny operator UI on a background Flask thread.
Defaults to **`http://<host>:8091/`** (8090 is the main C2 dashboard —
keeps them apart). Starts automatically with `strategy.app`; pass
`--no-web` to disable, or `--web-port N` to move it.

| Page | What it does |
|------|---|
| `/`         | Live top-down arena view. SVG drawn from `SwarmWorldModel.observe()`, polls `/api/state` at 2 Hz. Shows zone bands (our home / neutral / enemy), safety-margin outline, each drone's pose with role + altitude badge. |
| `/settings` | Form bound to `StrategySettings`. Every section from the schema below is editable. **Save** validates via `_from_dict` (so a bad team string is rejected with a clear 400) and mutates the *live* instance — the running planner sees the new values on the next tick, no restart needed. **Reload from disk** re-reads `settings.json` so two operators editing in different tools stay in sync. |

JSON API (useful for scripted edits or a custom UI):

| route                          | method | use                                                                   |
| ------------------------------ | ------ | --------------------------------------------------------------------- |
| `/api/topology`                | GET    | FC list from the active C2 config (drives the per-FC table)            |
| `/api/settings`                | GET    | current settings as JSON                                              |
| `/api/settings`                | POST   | replace settings — body = same shape as `settings.example.json`        |
| `/api/settings/reload`         | POST   | re-read `settings.json` from disk                                     |
| `/api/state`                   | GET    | live SwarmState (drones + zones + targets) — what `/` polls            |

## Operator-tunable settings (`settings.py`)

Knobs that change match-to-match live in their own JSON, separate from
`marker_mission_c2/config.json` (which is fleet-topology stuff that
hardly ever changes).

### Top-level

| field                       | meaning                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `team_color`                | `"red"` (own ids 41–46) or `"blue"` (own ids 31–36)          |
| `own_target_ids_override`   | replace the team-derived own-target set (rare; null = use team) |
| `enemy_target_ids_override` | replace the team-derived enemy set (rare; null = use team)    |
| `live_targets_only`         | ignore xx4/xx5/xx6 spares (default true)                      |

### `arena.*`

Arena geometry + zone definitions. Red team owns the -X side, blue the
+X side — the helpers below pick "ours" / "theirs" automatically based
on `team_color`.

| field              | type           | default        | meaning                                |
| ------------------ | -------------- | -------------- | -------------------------------------- |
| `width_m`          | float          | 10             | arena X extent (x ∈ ±5)                |
| `depth_m`          | float          | 20             | arena Y extent (y ∈ ±10)               |
| `safety_margin_m`  | float          | 0.5            | clip-back margin from any wall          |
| `red_home_x_m`     | `[lo, hi]`     | `[-5, -2]`     | red's home band along X                 |
| `blue_home_x_m`    | `[lo, hi]`     | `[ 2,  5]`     | blue's home band along X                |
| `neutral_zone_x_m` | `[lo, hi]`     | `[-2,  2]`     | middle band where scouts park           |

### `drones.<fc_name>.*`

Per-FC role + cruise altitude. Layered altitudes are the primary
collision-avoidance strategy — give each drone its own slot.

| field         | type    | default | meaning                                          |
| ------------- | ------- | ------- | ------------------------------------------------ |
| `role`        | string  | `idle`  | `attacker` / `scout` / `defender` / `idle`       |
| `altitude_m`  | float   | 1.5     | cruise altitude in arena Z                       |

### `attack.*`

Parameters for the 10-point simultaneous-strike maneuver.

| field                      | type           | default | meaning                                          |
| -------------------------- | -------------- | ------- | ------------------------------------------------ |
| `hover_alt_m`              | float          | 3.5     | wait altitude above the assigned target          |
| `strike_alt_m`             | float          | 1.5     | sync-drop altitude (both attackers descend here) |
| `sync_window_s`            | float          | 1.5     | how long the first attacker waits for the partner|
| `pair_targets_override`    | `list[[a,b]]`  | null    | which marker pairs qualify; null = derive from `own_target_ids` (every 2-combination) |
| `home_zone_clear`          | bool           | true    | abort the strike if any non-attacker friendly is in our home zone |

### `defender.*`

| field                  | type   | default | meaning                                                 |
| ---------------------- | ------ | ------- | ------------------------------------------------------- |
| `intercept_radius_m`   | float  | 2.5     | distance enemy must close on one of our targets before defender breaks toward it |

Resolution: `--strategy-settings` flag → `$MARKER_MISSION_C2_STRATEGY_SETTINGS`
env → `./marker_mission_c2/strategy/settings.json` →
`./marker_mission_c2/strategy/settings.example.json` → defaults.

Run-time:

```bash
# Flip team for one run only
python -m marker_mission_c2.strategy.app --team blue

# Flip + persist (settings.json gets written)
python -m marker_mission_c2.strategy.app --team blue --save-settings
```

Programmatic use — a planner that prefers our own targets:

```python
from marker_mission_c2.strategy import (
    settings as s, UtilityPlanner, StartMission, Idle,
)
cfg = s.load()                     # picks up settings.json

def score_attack(state, fc):
    obs = state.drones[fc]
    if not obs.last_marker_id:                     return -1.0
    if cfg.is_own_target(obs.last_marker_id):       return  10.0
    if cfg.is_enemy_target(obs.last_marker_id):     return  -5.0
    return 0.0
```

Mutating `cfg.team_color = TeamColor.BLUE` at run time flips
`own_target_ids` / `enemy_target_ids` / `our_home_x_m` /
`enemy_home_x_m` / `attack_pair_targets()` on the *next* read — every
derived property is computed at access time, no recreation needed.

### Derived helpers worth knowing

| call                                          | use                                    |
| --------------------------------------------- | -------------------------------------- |
| `cfg.our_home_x_m` / `cfg.enemy_home_x_m`     | which band is ours / theirs (flips with `team_color`) |
| `cfg.is_in_home_zone(x)` / `is_in_enemy_zone(x)` / `is_in_neutral_zone(x)` | classify an X coord |
| `cfg.is_within_arena(x, y)`                   | True iff inside fence by `safety_margin_m` |
| `cfg.role_for(fc_name)`                       | `Role` enum for that FC (`IDLE` if not assigned) |
| `cfg.cruise_altitude_for(fc_name)`            | per-FC altitude (1.5 m default if not assigned) |
| `cfg.fcs_with_role(Role.ATTACKER)`            | list of FC names with that role |
| `cfg.attack_pair_targets()`                   | list of (a, b) marker pairs that qualify for the 10-pt move; auto-derived from `own_target_ids` unless `pair_targets_override` is set |

## Tradeoffs we accepted

- **Reactive only.** The strategy looks one tick ahead. No multi-step
  search, no STRIPS planner. For "should I claim Target 3 before the
  other drone does?" you compute a one-shot utility score; if you
  need real lookahead, hold that state in your own `SwarmPlanner`
  subclass.
- **1 Hz default tick.** FC poll runs at 5 Hz, so the strategy
  shouldn't move faster than that anyway. Bump `--tick-hz` if you
  have a reason.
- **Greedy assignment in `UtilityPlanner`.** Small swarms (<10 FCs)
  don't benefit from Hungarian-style global assignment, so we walk
  FCs in name order. Replace with a `scipy.optimize.linear_sum_assignment`
  call inside your own planner if you ever need it.
- **Safety doesn't enforce a hard fence.** The C2 talks scripts, not
  RC sticks, so the fence is informational here — the mission script
  on the FC has to honour it. Safety logs out-of-bounds for the
  operator and stops the mission on the more dangerous conditions
  (low battery, dropped link, stale poll).

## Extending the surface

| Want to …                       | Edit                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| add an action verb              | new `CmdKind` + handler in `runner.SwarmRunner._dispatch`         |
| add a sensor                    | extend `DroneObservation` + the matching extractor in `world_model._extract` |
| swarm-level constraint          | new `SwarmPlanner` subclass; or compose existing planners        |
| custom emergency conditions     | subclass `SafetyGate` and override `gate(...)`                   |
| stream decisions into the UI    | pass `on_tick` to `SwarmRunner`; push records onto your SSE bus  |
