# Onboarding — sdc-tobe (snapshot 2026-05-23)

Welcome. This file is a colleague-to-colleague handoff for someone joining
the project mid-stream. It is **not** the operator manual ([README.md](README.md))
and **not** the agentic-tool guide ([AGENTS.md](AGENTS.md)) — read those
first if you have not seen them. This file tells you *what has been
happening recently* and where to look in the code so a standup or PR review
makes sense on day one.

The two people pushing commits right now:

- **Sascha Dirbach** — operator-facing fixes in `marker_mission/` and the
  multi-FC overview in `marker_mission_c2/`. Closes the loop between what
  the operator sees and what the FC actually does.
- **Oliver Tiedemann (`otiedemann` on GitHub)** — heavy lift in
  `c2_strategy/`, the attacker controller, marker-based positioning
  (`marker_mission/arena.py`), arena geometry, and the AWS bootstrap. The
  bulk of May commits are his.

Co-authored commits with "Claude Opus" in the trailer mean we drove the
change through Claude Code; treat the human author as the owner.

---

## The system in 30 seconds

We fly a Parrot Anafi (real or simulated) end-to-end:

```
operator's browser
    │ Tailscale
    ▼
C2  :8070  ── controller/remote_web_controller.py
    │
    ▼ HTTP
FC  :8080  ── controller_unified/unified_api_server.py
    │           …or marker_mission/app.py (combined: FC + mission UI on :8080)
    ▼ Olympe / pdraw
Sphinx (sim)   or real Anafi over Wi-Fi
```

Two extra services live alongside:

- `:8090` **sphinx-control** — sim manager; spawns the UE4 world, the sim
  drone, and the FC subprocess.
- `:9091` **drone-detector-ui** — YOLO-based collision detector.

A **multi-FC overview** ("C2 of C2s") lives in
[marker_mission_c2/](marker_mission_c2/) — it polls several FCs at once and
gives the operator one screen for all of them. This is where Sascha's
recent UI work sits.

For the AWS sphinx host (`sphinx3.otconsulting.de`) the ports, systemd
units, and recovery steps are all in [README.md](README.md). Use it.

---

## What has changed in the last ~2 weeks (May 2026)

Grouped by area, not by commit, so you can map "this is what I'm working
on" to "this is what changed under my feet."

### 1. Marker-based positioning — [marker_mission/arena.py](marker_mission/arena.py)

The single biggest area of churn. Background: each ArUco wall marker gives
two valid IPPE pose branches (planar-marker ambiguity); the wrong branch
flips the drone's world position. Oliver attacked this from three axes:

- **Altimeter-aware IPPE branch picker** (`c764827`) — the two branches
  project to different drone heights; pick the one matching `tel.height_cm`.
  Extended (`d076386`) to also handle the single-branch case where IPPE
  collapsed to one solution.
- **Magnetometer cross-check** (`9afb450`) — works on the ground now,
  previously bailed when the drone wasn't airborne.
- **Hard gate** (`a83894a`) — a single-marker pose is only accepted as a
  world-position fix if **both** altimeter and magnetometer agree.
  Conservative: fewer false fixes, possibly less coverage when one sensor
  is noisy.
- **Cold-start safety** (`bb3bf16`) — at boot, world position needs 2+
  visible wall markers before it is published. Avoids accepting a single
  flipped fix as the initial position.
- **Reset on TAKEOFF** (`225ca09`) — stale estimator state across flights
  was a bug; we now reset every takeoff.
- **Silent ValueError fix** (`b2c6a98`) — per-marker votes were being
  silently dropped because the contribution tuple grew from 4 to 5 fields
  but one unpack site wasn't updated. Strict win.

If you are debugging "the drone reports a flipped position," start at
[marker_mission/arena.py](marker_mission/arena.py) and re-read the IPPE
picker — the comments there are now the canonical reference.

There is also a per-marker `size` field (`6d75781`) so different markers
can be different physical sizes in the same arena.

### 2. marker_mission DSL — new flight primitives

The mission DSL (the table in [README.md](README.md#80800mission--marker-mission-combined-app))
gained a few primitives this month:

- **`FB_BRAKE`** (`c3bab74`) — closed-loop forward-brake using
  marker-relative memory; cuts a long attack run from 174 s to 95 s, then
  to 45 s after combined climb+cruise tuning (`96f1407`).
- **`TO_HOME`** (`0fed8dc`) — return to the takeoff snapshot. The
  formatter fix in `5f9be7a` is what stopped `/api/state` from crashing
  when the home pose was rendered.
- **Two-stage approach speed** (`5df1e68`) — cruise speed until close, then
  slow zone. Big win for "fly to far marker" missions.
- **Decoupled deadbands** (`7a367de`) — `goto_deadband` (used by `TO`) is
  now separate from the marker-relative deadband used during APPROACH.
- **SEARCH retreat-then-yaw** (`6d75781`) — if SEARCH does not see a
  marker, back off then rotate, instead of rotating in place.

### 3. AUTO_ATTACK — [c2_strategy/strategy.py](c2_strategy/strategy.py) + attacker phase

Oliver built a reactive attacker controller end-to-end in May
(perception → decision → RC, `13e5804`). The interesting commits:

- `34d8fe7` — wall-safety guard + marker-memory `FB_BRAKE` + the fast
  choreography that landed us under 1 minute.
- `7e47d4a` — marker positions now come from the active arena config
  instead of being hard-coded.
- `2906ce1` — capture = rise + move over marker (the earlier version
  turned around too early and lost the marker mid-rotation).
- `3dad365` — return-home choreography after the drop, without landing.
- `f46a143` — manual variant for the operator: trigger the same attack
  maneuver from existing commands, without engaging AUTO_ATTACK mode.

Tuning parameters were exposed step-by-step (`ae98c44` altitude +
approach speed; `5d50645` ... `a061e4c` scout tuning).

### 4. Strategy module rewrite — [c2_strategy/](c2_strategy/)

`d6bb0ab` is a rewrite-from-scratch of the strategy server around
**per-drone roles** instead of the old global state machine. Followups
worth knowing about:

- `3bd3235` — targets are modeled as 6 slots; the "holder" is which face
  is currently visible.
- `0dc109b` — role changes apply synchronously so a freshly-assigned
  target sticks (previously raced with the next tick).
- `d4d6597` + `c514c21` — the strategy service now derives the C2 base URL
  from its config bind block; the `--c2` flag is gone.
- Scout loop (`dada65d` ... `bc3e5f3`) — drops the precision-TO step in
  favour of `HEIGHT + SCOUT`, chains N scout cycles per push to stop
  yo-yoing, then turns the whole thing into one continuous RC drive
  without brakes between rotations.

If you are touching the strategy server, **read `c2_strategy/README.md`
first** — the role-state model is the heart of the rewrite and easy to
misread.

### 5. Arena geometry — [c2_strategy/arena_config.py](c2_strategy/arena_config.py) + `arena_config*.json`

- `ed10ff7` — floor zones now run along the **long** axis; targets sit
  inside their own zone.
- `0a91b0b` — target boxes moved fully inside the 10 m playing field.
- `1b9c669` — target markers use ROLL only (matches the existing wall
  convention).
- `dbd9447` — added the sphinx3 drone entry + ArUco target-marker
  generator under `tools/`.

There is a stale-bounds mismatch you need to know about: see "Known
pitfalls" below.

### 6. Tests — [tests/](tests/)

`1105eaa` introduced an autonomous attacker test harness
(`tests/strategy_loop/`). `1a9552d` and `97cf7bd` fixed the f-string +
enemy-face math and tightened early-exit / added `error` and
`success_off_home` outcomes. This is the closest thing we have to an
end-to-end check; run it before pushing strategy changes.

### 7. Infra — AWS sphinx bootstrap

Two bugs in [infra/aws/](infra/aws/) bootstrap (`5f05ca6`, `e397681`):
unit names in `restart_firmwared_full` were wrong, and `spc_up` was used
before being defined inside the `HARD`-flag branch. Both are now fixed —
if the sim host fails to come up at boot, these are the recent suspects.

### 8. Operator UI / multi-FC overview (Sascha's recent work)

- **Per-FC drone-link dot** in [marker_mission_c2/ui_pages.py](marker_mission_c2/ui_pages.py)
  + [marker_mission_c2/ui_shared.py](marker_mission_c2/ui_shared.py)
  (`3cd66d7`). Each FC card now shows **two** status dots side by side:
  first = C2↔FC link, second = FC↔drone link. Disambiguates "is the FC
  up?" from "is the drone connected to that FC?" — previously you had to
  open the card to see "Drone: no link" in the stat grid.
- **Front-wall magnetic-north capture** in [marker_mission/ui.py](marker_mission/ui.py)
  (`a428e13`). Two operator-reported issues from a flightctrl4 session:
  (a) the arena XY guard was clamping legitimate APPROACH RC because its
  bounds (`y∈[0, 10.8]`) are stale relative to the centred
  marker_mission arena (`y∈[-10, +10]`) — **default flipped to OFF until
  bounds are regenerated**; (b) added a *"Capture (facing front wall)"*
  button that calibrates magnetic north from `tel.yaw` alone (no markers
  required), so the operator can calibrate on the ground. The existing
  marker-based button was renamed *"Capture (from markers)"* for clarity.

---

## Known pitfalls right now

A few traps that will bite a newcomer in the first week:

1. **Arena XY guard is OFF by default** until the bounds in
   [controller_unified/unified_api_server.py](controller_unified/unified_api_server.py)
   are regenerated to match the centred marker_mission arena. Re-enable
   per host with `ARENA_GUARD_ENABLED=1` or live via
   `POST /api/config/arena_safety {"enabled": true}` — but only after
   confirming the bounds match the arena you actually loaded.
2. **The "both altimeter and magnetometer agree" gate** means a flight
   without per-arena magnetic-offset calibration will reject many
   single-marker fixes. Use the new "Capture (facing front wall)" button
   on the Arena tab before serious flying.
3. **Never run a second Olympe client** against the same Anafi while the
   FC is connected — ARSDK is single-controller and the FC's session will
   be evicted. Talk to `:8080/api/state` instead.
4. **`marker_mission` and `unified_api_server` are mutually exclusive on
   :8080.** The Ansible playbook in `ansible/fc-deploy/` switches a host
   between the two modes — `Conflicts=` in the systemd units enforces it.
5. **Anafi battery floor:** ≤ 58 % battery this airframe loses climb
   authority. If "won't climb", check battery first.
6. **Two parallel solvePnP code paths** still exist: `marker_mission`'s own
   estimator and the unified positioning subsystem. Oliver's recent
   improvements are all on the `marker_mission` side. Keep them straight
   when reading positioning code.

---

## Suggested first reads, in order

1. [README.md](README.md) — operator's map, ports, systemd, recovery.
2. [AGENTS.md](AGENTS.md) — module-by-module orientation for agentic
   tooling; doubles as a layout overview.
3. [marker_mission/docs/MISSION_REFERENCE.md](marker_mission/docs/MISSION_REFERENCE.md)
   — the mission DSL.
4. [marker_mission/arena.py](marker_mission/arena.py) — the IPPE branch
   picker and validation gate are the canonical reference for how we
   resolve world position.
5. [c2_strategy/README.md](c2_strategy/README.md) and
   [c2_strategy/strategy.py](c2_strategy/strategy.py) — the per-drone-role
   model is the foundation of everything attacker-side.
6. [tests/strategy_loop/](tests/strategy_loop/) — closest thing we have
   to an end-to-end check; run before pushing strategy or attacker
   changes.

---

## Asking for help

- **Sascha** — marker_mission UI, multi-FC overview, operator workflow,
  AWS host issues.
- **Oliver** — strategy, attacker, positioning estimator, arena geometry,
  infra/aws bootstrap.

When in doubt, `git log -p --since="2026-05-15" -- <path>` tells you who
last touched a file and *why* (commit messages here are intentionally
verbose — read them).
