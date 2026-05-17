# `marker_mission_c2.strategy` — reactive strategy layer for the fleet C2

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

## Operator-tunable settings (`settings.py`)

Knobs that change match-to-match live in their own JSON, separate from
`marker_mission_c2/config.json` (which is fleet-topology stuff that
hardly ever changes).

| field                       | meaning                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `team_color`                | `"red"` (own ids 41–46) or `"blue"` (own ids 31–36)          |
| `own_target_ids_override`   | replace the team-derived own-target set (rare; null = use team) |
| `enemy_target_ids_override` | replace the team-derived enemy set (rare; null = use team)    |
| `live_targets_only`         | ignore xx4/xx5/xx6 spares (default true)                      |

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
`own_target_ids` / `enemy_target_ids` on the *next* read — every
derived property is computed at access time, no recreation needed.

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
