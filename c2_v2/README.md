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
4. The coordinator — unified assignment + 5/10/Special plays + recovery + deconfliction
5. C2-link-loss auto-land, tuning UI, end-to-end verify

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
