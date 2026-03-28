#!/usr/bin/env python3
"""
Start the SDC26 C2 Strategy server together with the sim_swarm_API simulator.

Launches both servers in a single process and shuts them down cleanly on Ctrl+C.

Usage:
    python start_c2_sim.py

Servers:
    - sim_swarm_API  →  http://localhost:8080  (3D simulator at /sim/)
    - c2_strategy    →  http://localhost:9090  (strategy dashboard)
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SIM_API_DIR = ROOT / "sim_swarm_API"
SIM_API_SCRIPT = SIM_API_DIR / "sim_api_server.py"
SIM_API_PORT = int(os.environ.get("SIM_API_PORT", 8080))

C2_PORT = int(os.environ.get("C2_PORT", 9090))

PYTHON = sys.executable


def main():
    procs: list[subprocess.Popen] = []

    def cleanup(*_):
        print("\nShutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Starting sim_swarm_API on port {SIM_API_PORT}...")
    sim_proc = subprocess.Popen(
        [PYTHON, str(SIM_API_SCRIPT)],
        cwd=str(SIM_API_DIR),
        env={**os.environ, "SIM_API_PORT": str(SIM_API_PORT)},
    )
    procs.append(sim_proc)

    # Give the sim server a moment to bind its port
    time.sleep(1)

    print(f"Starting c2_strategy on port {C2_PORT}...")
    c2_proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "c2_strategy.strategy_server:app",
         "--host", "0.0.0.0", "--port", str(C2_PORT)],
        cwd=str(ROOT),
    )
    procs.append(c2_proc)

    print()
    print("=" * 60)
    print(f"  Simulator:  http://localhost:{SIM_API_PORT}/sim/")
    print(f"  C2 Dashboard:  http://localhost:{C2_PORT}/")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop both servers.")
    print()

    # Wait for either process to exit
    while True:
        for p in procs:
            ret = p.poll()
            if ret is not None:
                print(f"Process {p.args} exited with code {ret}")
                cleanup()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
