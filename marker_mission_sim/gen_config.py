"""Generate matching sim + C2 configs for an arbitrary number of drones.

The local stack needs two configs that agree on the drone roster:
  * the sim config (``drones``: id/team/fc_port/spawn), and
  * the C2 config (``fcs``: name/host/port).

This helper builds both from a red/blue count so ``run_local.sh --drones N``
can spin up any number of drones end-to-end (sim -> C2 -> strategy -> hub).

Drones are named ``red1..redR`` / ``blue1..blueB``, assigned sequential FC
ports from ``--base-port`` (default 9101), spawned in their home zone
(red at -Y, blue at +Y) and spread across the arena width, each facing the
enemy side. Other settings are inherited from the example configs.

Usage:
    python -m marker_mission_sim.gen_config --red 3 --blue 3 \
        --out-sim /tmp/sim.json --out-c2 /tmp/c2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIM_BASE = REPO / "marker_mission_sim" / "sim_config.example.json"
C2_BASE = REPO / "marker_mission_c2" / "config.dev.json"


def _spread(n: int, lo: float = -4.0, hi: float = 4.0) -> list[float]:
    """n evenly spaced x positions across [lo, hi] (single drone -> centre)."""
    if n <= 1:
        return [0.0]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def build_rosters(red: int, blue: int, base_port: int,
                  fc_host: str = "127.0.0.1") -> tuple[list[dict], list[dict]]:
    """Return (sim_drones, c2_fcs) for the given per-team counts."""
    drones: list[dict] = []
    fcs: list[dict] = []
    port = base_port
    # (team, count, spawn_y, heading_deg facing the enemy side)
    for team, n, y, hdg in (("red", red, -9.0, 0.0), ("blue", blue, 9.0, 180.0)):
        xs = _spread(n)
        for i in range(n):
            name = f"{team}{i + 1}"
            drones.append({
                "id": name, "team": team, "fc_port": port,
                "spawn_x": round(xs[i], 2), "spawn_y": y,
                "spawn_heading_deg": hdg,
            })
            fcs.append({"name": name, "host": fc_host, "port": port})
            port += 1
    return drones, fcs


def _load(path: Path, fallback: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(fallback)


def generate(red: int, blue: int, out_sim: Path, out_c2: Path,
             base_port: int = 9101, fc_host: str = "127.0.0.1") -> list[dict]:
    drones, fcs = build_rosters(red, blue, base_port, fc_host)

    sim = _load(SIM_BASE, {})
    sim["drones"] = drones
    out_sim.write_text(json.dumps(sim, indent=2) + "\n")

    c2 = _load(C2_BASE, {"bind": {"host": "127.0.0.1", "port": 8090}})
    c2["fcs"] = fcs
    out_c2.write_text(json.dumps(c2, indent=2) + "\n")
    return drones


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marker_mission_sim.gen_config",
                                 description=__doc__)
    ap.add_argument("--red", type=int, default=5,
                    help="red drones (default 5 — SDC26 §1.3 max)")
    ap.add_argument("--blue", type=int, default=5,
                    help="blue drones (default 5 — SDC26 §1.3 max)")
    ap.add_argument("--base-port", type=int, default=9101)
    ap.add_argument("--fc-host", default="127.0.0.1")
    ap.add_argument("--out-sim", required=True, type=Path)
    ap.add_argument("--out-c2", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.red < 0 or a.blue < 0 or (a.red + a.blue) < 1:
        ap.error("need at least one drone")
    drones = generate(a.red, a.blue, a.out_sim, a.out_c2, a.base_port, a.fc_host)
    names = [d["id"] for d in drones]
    print(f"generated {len(drones)} drones: {names}")
    print(f"  ports {drones[0]['fc_port']}..{drones[-1]['fc_port']}")
    print(f"  sim config: {a.out_sim}")
    print(f"  c2 config : {a.out_c2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
