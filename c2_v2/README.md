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
3. Roles + the never-land chained-script mechanism
4. The coordinator — unified assignment + 5/10/Special plays + recovery + deconfliction
5. Arming, `?` kill-all-land, C2-link-loss safety, tuning UI, end-to-end verify

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
