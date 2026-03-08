import argparse
import json
from pathlib import Path

from scenarios import SCENARIOS
from sim import run_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline", choices=SCENARIOS.keys())
    parser.add_argument("--out", default="sim_swarm/out")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    result, log = run_sim(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f"{args.scenario}_result.json"
    log_path = out_dir / f"{args.scenario}_timeline.json"

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Scenario: {args.scenario}")
    print(json.dumps(result, indent=2))
    print(f"Saved: {result_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
