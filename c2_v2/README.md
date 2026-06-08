# C2 V2 — swarm Ground Control Station (rebuild)

A clean strategy brain for the Swarm Drone Challenge that connects **directly to
the five real flight controllers** (`flightctrl1-5:8080`) and plays the v7
Capture-the-Flag game. It **reuses the battle-proven primitives** from
`marker_mission_c2` (FC HTTP client, live arena geometry, box-state tracker,
the proven capture geometry) and rebuilds the part that was tangled in V1: the
per-drone role state machines and a single swarm **coordinator** that actually
plays for the 5 / 10 / Special scoring tiers and **never lands the drones except
on the `?` emergency-land**. `marker_mission` (the FC) is never modified.

## Run

```bash
# from the repo root, in the C2 venv (has flask + httpx)
python -m c2_v2                       # connect to flightctrl1-5:8080
python -m c2_v2 --fcs flightctrl2     # subset, for bench testing
python -m c2_v2 --port 8100
```

Open `http://<host>:8100/`.

## Build phases (each shipped + verified separately)

1. **FC client + live per-drone world + read-only dashboard** ← *current*
2. Box-state tracker + arena + 6-box status panel + team / arena toggles
3. **Roles + the never-land mechanism + first commands** ← *current*
4. The coordinator — autonomous role assignment + strategy
5. **Tuning UI + C2-link-loss safety + verify** ← *current (rebuild complete)*

## Phase 5 (this commit) — tuning + safety

- `tunables.py` + `match.py` — the proven strategy/capture constants are now
  **live-editable** (capture standoff/tol/forward/rise, RTH standoff, scout +
  mover altitudes + step, scout/defender rotate speed, defender standoff, attack
  chain length, baseline defenders). Clamped to safe ranges, persisted, applied
  on the next (re)launch. `scripts.py` + the coordinator read them live.
- dashboard — a **Tuning** panel (collapsible) + a red **LINK-LOST banner** when
  an armed drone goes unreachable.
- **C2-link-loss safety (v7 §3.3):** the FC's own watchdog auto-lands a drone if
  it gets no `/api/` request for `REMOTE_TIMEOUT_S` (= **2.0 s**). C2 V2 polls
  `/api/state` at **5 Hz**, so **our polling is the heartbeat** — if the C2 link
  drops, the FC lands the drone within 2 s on its own. No FC change needed
  (constraint: `marker_mission` untouched).

## Safety model summary
- **`?` key / EMERGENCY LAND** → lands every drone now + disarms.
- **C2 link loss** → FC watchdog auto-lands (≤2 s) because polling stops.
- **FC↔drone radio loss** → `marker_mission`'s own safety lands the drone.
- **Disarmed** → C2 V2 observes only (drones keep flying their current script;
  use EMERGENCY LAND to bring them down).
- Restart always comes up **disarmed**.

## Phase 4 (this commit) — the autonomous brain (AUTO mode)

- `coordinator.py` — a pure decision function: from the live box holders + drone
  states it assigns each drone a role, biased to **win the game**:
    * 1 **scout** always (feeds box-state).
    * **defenders** = our threatened boxes + a baseline home presence (so the
      enemy can't walk into an undefended box).
    * **attackers** = the rest (≥1 always) — continuously re-capture the 3 enemy
      boxes (each capture+home = 1 pt; they flip back when we leave, so the loop
      scores all match).
    * **Special push**: at 5/6 boxes ours, go ALL-OUT to grab the last enemy box
      for the instant-win (hold all 6 for 5 s).
  Stable assignment — keeps drones in their current role where it fits the
  target counts, so it re-tasks the *minimum* number of drones.
- `runner.py` — in AUTO, applies the plan with **hold-time hysteresis** (a role
  must be wanted 12 s before it's adopted) and a per-drone **re-task cooldown**
  (25 s between stops), since re-tasking costs a landing. The same re-task path
  also applies MANUAL role changes (stop → land → relaunch the new role), and
  "stands down" (lands) a drone turned off / set idle.
- dashboard — **Go AUTO / MANUAL** toggle, the live coordinator strategy line,
  and role selectors disabled in AUTO.

> Why not precise 5-pt / 10-pt orchestration? With independent never-land
> scripts we can't reliably hit the sub-second 10-pt double-strike or the
> "all drones outside home at the instant of capture" 5-pt window. So the brain
> banks continuous 1-pt captures and drives toward the **Special instant-win** —
> the highest-value outcome actually achievable with this architecture.

## The never-land mechanism

The FC safety-LANDS whenever a mission script COMPLETES, and `/api/stop` lands
too (`rc_zero + land`). A new mission can only START from idle (landed). So to
keep drones airborne the whole match (only `?` lands them), every role runs an
**effectively-infinite script** that never completes:

- **scout** — centre, then one ~1 h `RC` yaw drive.
- **defender** — park at a side marker near our boxes, slow-rotate ~1 h (presence
  blocks enemy capture; rotation keeps it active, not a "dead duck").
- **attacker** — a 30-pass chain of capture legs over the 3 enemy boxes (they flip
  back when we leave, so re-attacking stays productive).

The runner launches each enabled drone once (when armed + FC idle); the infinite
script then runs for the whole match. Re-tasking is what costs a landing, so the
coordinator (Phase 4) re-tasks sparingly.

## Phase 3 (this commit) — first commands

- `scripts.py` — the never-land role scripts, with the operator's proven capture
  geometry (`APPROACH 1.50 0.20` / `FB_UD_IMU 1.50 0.50` / `YAW_IMU 180` /
  `GO_HOME .. 3.50 0.5 <hdg>`) + deconflicted altitudes (scout 1.0; movers 1.3,
  1.6, 1.9, … ≥30 cm apart) + RTH fan-out (±30°).
- `runner.py` — 1 Hz tick: launch each enabled drone's role script when armed +
  idle; computes mover altitude lanes + attacker fan-out live.
- `match.py` — per-drone role + enabled + arm state (persisted; arm always
  resets OFF on restart).
- dashboard — **Arm/Disarm**, **EMERGENCY LAND** button + the `?` key (lands all,
  disarms), per-drone **role** selector + on/off, and the runner's last action.

⚠️ Arming makes drones fly. The default fleet is scout / attacker×2 / defender×2.

## Phase 1 (this commit)

- `config.py` — the five FCs (flightctrl1-5:8080) + poll cadence.
- `world.py` — `DroneWorld`: per-drone read-model parsed defensively from `/api/state`
  (arena position, telemetry, battery, FC phase, what it's doing).
- `pool.py` — async pool: one task per FC polling `/api/state` (+ slow `/api/identity`,
  `/api/wlan`), reusing `marker_mission_c2.fc_client.AsyncFCClient`.
- `web.py` — read-only dashboard: per-drone cards + a top-down arena map with live
  ArUco positions (arena geometry from `arena_state`).
- `app.py` / `__main__.py` — asyncio pool thread + Flask dashboard.

No commands are sent in Phase 1 — connect and observe only.
