#!/usr/bin/env python3
"""Autonomous attacker test harness.

Runs a series of attack cycles against configured target slots, classifies
each outcome (success / crash / hang / timeout), and writes a JSON log.
Operator: stays out of the loop unless we exhaust the iteration budget.

Single test cycle:
  1. Health check: drone must be in INIT, on the ground, FC connected.
  2. Assign target slot via /api/strategy/target/<fc>.
  3. Arm runner via /api/strategy/arm.
  4. Poll telemetry every POLL_INTERVAL_S, recording (t, phase, world_pos,
     height, visible_marker_ids, strategy phase, last decision).
  5. Wait for completion: strategy role-state phase = "done" AND FC phase
     in (init, done) AND drone landed (flying=False).
  6. Disarm.
  7. Classify outcome from the recorded trace.
  8. Write a per-cycle JSON file with the trace + classification.

Run by hand against the live system:
    python tests/strategy_loop/attacker_loop.py --help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Arena bounds (from regulations + our setup).
ARENA_X_MIN, ARENA_X_MAX = -5.0, 5.0
ARENA_Y_MIN, ARENA_Y_MAX = -10.0, 10.0
ARENA_Z_MIN, ARENA_Z_MAX = 0.0, 6.0

POLL_INTERVAL_S = 0.5
MISSION_TIMEOUT_S = 90.0
SAME_POS_HANG_THRESHOLD_S = 20.0
SAME_POS_TOL_M = 0.10


@dataclass
class TraceSample:
    t: float
    fc_phase: str
    rs_phase: str
    flying: Optional[bool]
    height_m: Optional[float]
    world_x: Optional[float]
    world_y: Optional[float]
    world_z: Optional[float]
    visible: List[int]
    last_decision: str
    mission_step_idx: Optional[int]


@dataclass
class CycleResult:
    iteration: int
    fc_name: str
    target_slot: int
    expected_face_id: Optional[int]
    started_unix_s: float
    duration_s: float
    outcome: str            # success | crash | hang | timeout | aborted | error
    classification_reason: str
    final_world_pos: Optional[Tuple[float, float, float]]
    home_pos: Tuple[float, float, float]
    home_distance_m: Optional[float]
    peak_speed_mps: Optional[float]
    last_decision: str
    trace: List[TraceSample] = field(default_factory=list)


def _http_get_json(url: str, timeout: float = 4.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_post_json(url: str, body: Dict[str, Any], timeout: float = 4.0) -> Dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode() or "{}"
        try:
            return json.loads(body)
        except ValueError:
            return {"raw": body}


class StrategyAPI:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def state(self) -> Dict[str, Any]:
        return _http_get_json(self.base + "/api/state")

    def settings(self) -> Dict[str, Any]:
        return _http_get_json(self.base + "/api/settings")

    def arm(self) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/strategy/arm", {})

    def disarm(self) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/strategy/disarm", {})

    def assign_target(self, fc: str, slot: Optional[int]) -> Dict[str, Any]:
        return _http_post_json(
            self.base + f"/api/strategy/target/{fc}",
            {"slot": slot},
        )

    def patch_match(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/settings/match", patch)

    def patch_drone(self, fc: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        return _http_post_json(self.base + f"/api/settings/drone/{fc}", patch)


class FCAPI:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def state(self) -> Dict[str, Any]:
        return _http_get_json(self.base + "/api/state")

    def stop(self) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/stop", {})

    def tune(self) -> Dict[str, Any]:
        return _http_get_json(self.base + "/api/tune")

    def tune_apply(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/tune/apply", patch)

    def tune_save(self) -> Dict[str, Any]:
        return _http_post_json(self.base + "/api/tune/save", {})


def _sample(strategy: StrategyAPI, fc: FCAPI, fc_name: str, t0: float) -> TraceSample:
    s_state: Dict[str, Any] = {}
    fc_state: Dict[str, Any] = {}
    try:
        s_state = strategy.state()
    except Exception:
        pass
    try:
        fc_state = fc.state()
    except Exception:
        pass
    tel = (fc_state.get("telemetry") or {}) if isinstance(fc_state, dict) else {}
    wp = fc_state.get("world_position_m") if isinstance(fc_state, dict) else None
    if not isinstance(wp, (list, tuple)) or len(wp) < 3:
        wp = [None, None, None]
    rs = ((s_state.get("runner") or {}).get("drones") or {}).get(fc_name, {})
    h_cm = tel.get("height_cm")
    h_m = (h_cm / 100.0) if isinstance(h_cm, (int, float)) else None
    return TraceSample(
        t=time.time() - t0,
        fc_phase=str(fc_state.get("phase") or "?"),
        rs_phase=str(rs.get("phase") or "?"),
        flying=tel.get("flying"),
        height_m=h_m,
        world_x=wp[0], world_y=wp[1], world_z=wp[2],
        visible=list(fc_state.get("visible_marker_ids") or []),
        last_decision=str(rs.get("last_decision_reason") or ""),
        mission_step_idx=fc_state.get("mission_step_idx"),
    )


def _classify(result: CycleResult, trace: List[TraceSample]) -> None:
    if not trace:
        result.outcome = "error"
        result.classification_reason = "empty trace"
        return
    last = trace[-1]
    result.final_world_pos = (last.world_x, last.world_y, last.world_z)
    if all(v is not None for v in result.final_world_pos):
        dx = (last.world_x or 0.0) - result.home_pos[0]
        dy = (last.world_y or 0.0) - result.home_pos[1]
        result.home_distance_m = math.hypot(dx, dy)
    result.last_decision = last.last_decision

    # Out-of-arena check.
    pos_oob = False
    if last.world_x is not None and last.world_y is not None and last.world_z is not None:
        if not (ARENA_X_MIN - 0.5 <= last.world_x <= ARENA_X_MAX + 0.5):
            pos_oob = True
        if not (ARENA_Y_MIN - 0.5 <= last.world_y <= ARENA_Y_MAX + 0.5):
            pos_oob = True

    # Hang detection (last N samples within a tiny radius, mid-mission).
    hang = False
    if len(trace) >= 6:
        tail = trace[-8:]
        xs = [s.world_x for s in tail if s.world_x is not None]
        ys = [s.world_y for s in tail if s.world_y is not None]
        if len(xs) >= 4 and len(ys) >= 4:
            spread = max(xs) - min(xs) + max(ys) - min(ys)
            tail_span = tail[-1].t - tail[0].t
            if spread < SAME_POS_TOL_M and tail_span > SAME_POS_HANG_THRESHOLD_S:
                hang = True

    # Phase-based crash detection: emergency phase, or drone reports
    # height=0 mid-mission while strategy thinks it should be flying.
    emergency = any(s.fc_phase in ("emergency", "abort") for s in trace)
    abrupt_land = False
    flying_seen = False
    for s in trace:
        if s.flying is True:
            flying_seen = True
        elif s.flying is False and flying_seen and s.rs_phase not in ("done", "idle"):
            # Drone landed but strategy isn't done — likely a forced land or crash.
            abrupt_land = True
            break

    # Success criteria: ended at phase=init/done with drone landed within
    # 1.5 m of home and strategy role-state = done.
    landed = last.flying is False
    near_home = (result.home_distance_m is not None
                 and result.home_distance_m < 1.5)
    rs_done = last.rs_phase in ("done", "idle")
    fc_done = last.fc_phase in ("init", "done", "")

    if emergency:
        result.outcome = "crash"
        result.classification_reason = "FC entered emergency phase"
    elif pos_oob:
        result.outcome = "crash"
        result.classification_reason = (
            f"out-of-arena final position {result.final_world_pos}"
        )
    elif abrupt_land:
        result.outcome = "crash"
        result.classification_reason = (
            "drone landed mid-mission (flying flipped False before role-state done)"
        )
    elif hang:
        result.outcome = "hang"
        result.classification_reason = (
            f"position stuck under {SAME_POS_TOL_M}m for {SAME_POS_HANG_THRESHOLD_S}s+"
        )
    elif rs_done and fc_done and landed and near_home:
        result.outcome = "success"
        result.classification_reason = (
            f"landed at home (d={result.home_distance_m:.2f}m)"
        )
    elif result.duration_s >= MISSION_TIMEOUT_S - 1.0:
        result.outcome = "timeout"
        result.classification_reason = (
            f"hit timeout {MISSION_TIMEOUT_S}s before reaching done state"
        )
    elif rs_done and fc_done:
        result.outcome = "success"
        result.classification_reason = (
            f"role-state + fc done (home_dist={result.home_distance_m}, flying={last.flying})"
        )
    else:
        result.outcome = "aborted"
        result.classification_reason = (
            f"unclassified end state (rs_phase={last.rs_phase} "
            f"fc_phase={last.fc_phase} flying={last.flying})"
        )

    # Compute peak speed from successive samples.
    speeds: List[float] = []
    for a, b in zip(trace, trace[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        if (a.world_x is None or b.world_x is None or
                a.world_y is None or b.world_y is None):
            continue
        d = math.hypot(b.world_x - a.world_x, b.world_y - a.world_y)
        speeds.append(d / dt)
    result.peak_speed_mps = max(speeds) if speeds else None


def run_cycle(
    *,
    strategy: StrategyAPI,
    fc: FCAPI,
    fc_name: str,
    target_slot: int,
    iteration: int,
    out_dir: pathlib.Path,
) -> CycleResult:
    s = strategy.state()
    settings = s.get("settings") or {}
    drones = (settings.get("drones") or {})
    our_team = (settings.get("markers") or {}).get("our_team", "red")
    drone_cfg = drones.get(fc_name, {})
    home_alt = drone_cfg.get("home_alt_m") or 3.0
    match = settings.get("match") or {}
    home_block = match.get("home_red") if our_team == "red" else match.get("home_blue")
    home_pos = (
        float(home_block.get("x", 0.0)),
        float(home_block.get("y", 0.0)),
        float(home_alt),
    )
    enemy_face = (40 if our_team == "red" else 30) + target_slot

    print(f"\n========== iteration {iteration} — slot {target_slot} (expect APPROACH {enemy_face}) ==========")
    print(f"  our_team={our_team}  home={home_pos}")

    # Health: drone landed, FC reachable.
    try:
        fc_state = fc.state()
    except Exception as e:
        print(f"  FC not reachable: {e}")
        return CycleResult(
            iteration=iteration, fc_name=fc_name, target_slot=target_slot,
            expected_face_id=enemy_face,
            started_unix_s=time.time(), duration_s=0,
            outcome="error", classification_reason=f"FC unreachable: {e}",
            final_world_pos=None, home_pos=home_pos,
            home_distance_m=None, peak_speed_mps=None,
            last_decision="",
        )
    tel = fc_state.get("telemetry") or {}
    if tel.get("flying"):
        print("  drone reports flying=True at cycle start — stopping mission")
        try:
            fc.stop()
            time.sleep(4)
        except Exception as e:
            print(f"  stop failed: {e}")

    # Disarm first (clean slate).
    try:
        strategy.disarm()
    except Exception:
        pass

    # Assign slot.
    strategy.assign_target(fc_name, target_slot)
    time.sleep(0.2)

    started = time.time()
    cycle = CycleResult(
        iteration=iteration, fc_name=fc_name, target_slot=target_slot,
        expected_face_id=enemy_face,
        started_unix_s=started, duration_s=0,
        outcome="aborted", classification_reason="",
        final_world_pos=None, home_pos=home_pos,
        home_distance_m=None, peak_speed_mps=None,
        last_decision="",
    )

    # Arm and run.
    strategy.arm()
    print(f"  armed at t=0")

    trace: List[TraceSample] = []
    deadline = started + MISSION_TIMEOUT_S
    last_fc_phase = None
    last_rs_phase = None
    while time.time() < deadline:
        s = _sample(strategy, fc, fc_name, started)
        trace.append(s)
        if s.fc_phase != last_fc_phase or s.rs_phase != last_rs_phase:
            print(f"  t={s.t:5.1f}s  fc={s.fc_phase:10s} rs={s.rs_phase:10s}"
                  f" h={s.height_m if s.height_m is not None else '?'}"
                  f" pos=({s.world_x:.2f if s.world_x is not None else '?' },"
                  f" {s.world_y:.2f if s.world_y is not None else '?' })"
                  if isinstance(s.world_x, (int, float)) and isinstance(s.world_y, (int, float))
                  else f"  t={s.t:5.1f}s  fc={s.fc_phase} rs={s.rs_phase}")
            last_fc_phase = s.fc_phase
            last_rs_phase = s.rs_phase
        # Early exit: strategy reports done AND drone landed.
        if s.rs_phase in ("done", "idle") and s.flying is False and s.fc_phase in ("init", "done"):
            # confirm with a follow-up sample
            time.sleep(POLL_INTERVAL_S)
            s2 = _sample(strategy, fc, fc_name, started)
            trace.append(s2)
            if s2.rs_phase in ("done", "idle") and s2.flying is False:
                break
        time.sleep(POLL_INTERVAL_S)

    cycle.duration_s = time.time() - started
    cycle.trace = trace
    _classify(cycle, trace)

    # Safe disarm.
    try:
        strategy.disarm()
    except Exception:
        pass

    print(f"  -> {cycle.outcome}: {cycle.classification_reason}")
    print(f"  duration={cycle.duration_s:.1f}s  peak_speed={cycle.peak_speed_mps}"
          f"  final={cycle.final_world_pos}  home_dist={cycle.home_distance_m}")

    # Write per-cycle log.
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"iter{iteration:02d}_slot{target_slot}.json"
    fname.write_text(json.dumps({
        "iteration": cycle.iteration,
        "fc_name": cycle.fc_name,
        "target_slot": cycle.target_slot,
        "expected_face_id": cycle.expected_face_id,
        "started_unix_s": cycle.started_unix_s,
        "duration_s": cycle.duration_s,
        "outcome": cycle.outcome,
        "classification_reason": cycle.classification_reason,
        "final_world_pos": cycle.final_world_pos,
        "home_pos": cycle.home_pos,
        "home_distance_m": cycle.home_distance_m,
        "peak_speed_mps": cycle.peak_speed_mps,
        "last_decision": cycle.last_decision,
        "trace": [asdict(s) for s in trace],
    }, indent=2, default=str))
    return cycle


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="http://127.0.0.1:8091",
                   help="Strategy API base URL")
    p.add_argument("--fc", default="http://sphinx3.otconsulting.de:8080",
                   help="FC API base URL")
    p.add_argument("--fc-name", default="Sphinx3",
                   help="C2 name of the drone (matches strategy settings)")
    p.add_argument("--slots", default="1,2,3",
                   help="Comma-separated target slots to test")
    p.add_argument("--iterations", type=int, default=1,
                   help="Pass count over the slots list")
    p.add_argument("--out", default="tests/strategy_loop/runs",
                   help="Output directory for per-cycle JSON logs")
    args = p.parse_args(argv)

    strategy = StrategyAPI(args.strategy)
    fc = FCAPI(args.fc)
    slots = [int(s) for s in args.slots.split(",") if s.strip()]
    out_dir = pathlib.Path(args.out) / time.strftime("%Y%m%d_%H%M%S")

    print(f"strategy: {args.strategy}")
    print(f"fc:       {args.fc}  ({args.fc_name})")
    print(f"slots:    {slots}")
    print(f"iters:    {args.iterations}")
    print(f"out:      {out_dir}")

    results: List[CycleResult] = []
    iter_no = 0
    for pass_idx in range(args.iterations):
        for slot in slots:
            iter_no += 1
            try:
                r = run_cycle(
                    strategy=strategy, fc=fc, fc_name=args.fc_name,
                    target_slot=slot, iteration=iter_no, out_dir=out_dir,
                )
            except Exception as e:
                r = CycleResult(
                    iteration=iter_no, fc_name=args.fc_name, target_slot=slot,
                    expected_face_id=None,
                    started_unix_s=time.time(), duration_s=0,
                    outcome="error", classification_reason=f"exception: {e}",
                    final_world_pos=None, home_pos=(0,0,0),
                    home_distance_m=None, peak_speed_mps=None,
                    last_decision="",
                )
                print(f"  EXCEPTION: {e}")
            results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(results)} cycles)")
    print("=" * 70)
    succ = sum(1 for r in results if r.outcome == "success")
    crash = sum(1 for r in results if r.outcome == "crash")
    hang = sum(1 for r in results if r.outcome == "hang")
    tmo = sum(1 for r in results if r.outcome == "timeout")
    err = sum(1 for r in results if r.outcome == "error")
    abrt = sum(1 for r in results if r.outcome == "aborted")
    print(f"  success={succ}  crash={crash}  hang={hang}  timeout={tmo}"
          f"  error={err}  aborted={abrt}")
    for r in results:
        print(f"  iter {r.iteration:2d} slot {r.target_slot}: {r.outcome:8s}"
              f"  d={r.duration_s:5.1f}s  home_dist={r.home_distance_m}"
              f"  '{r.classification_reason}'")

    out_summary = out_dir / "summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({
        "n": len(results),
        "success": succ, "crash": crash, "hang": hang,
        "timeout": tmo, "error": err, "aborted": abrt,
        "results": [
            {
                "iteration": r.iteration, "slot": r.target_slot,
                "outcome": r.outcome, "duration_s": r.duration_s,
                "home_distance_m": r.home_distance_m,
                "peak_speed_mps": r.peak_speed_mps,
                "classification_reason": r.classification_reason,
            } for r in results
        ],
    }, indent=2, default=str))
    return 0 if (succ == len(results)) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
