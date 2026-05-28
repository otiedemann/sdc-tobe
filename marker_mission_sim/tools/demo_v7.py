"""Drive the four v7 capture mechanics against the LIVE local sim.

Prereq: the stack is running (./run_local.sh --drones 4). This script then
pushes mission scripts through the C2 to the simulated drones and prints
the affected box states so you can see (and verify) each of the v7 rules
from §1.4.3:

    A.  basic enemy capture (>=2 s hover -> flip + 5 s lock)
    B.  5 s post-capture lock blocks an immediate recapture
    C.  defender presence (home-team drone in the box) blocks capture
    D.  home-zone recapture is instant (no 2 s hover) once the lock expires

Run:
    python -m marker_mission_sim.tools.demo_v7
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

SIM = "http://127.0.0.1:9100"     # sim UI (publishes /api/world)
C2 = "http://127.0.0.1:8090"      # C2 server (relays /api/c2/<fc>/start)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2) as r:
        return json.loads(r.read())


def _post_json(url: str, body: dict, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def boxes() -> dict[int, dict]:
    w = _get_json(f"{SIM}/api/world")
    return {b["slot"]: b for b in w["boxes"]}


def show(label: str) -> None:
    now = time.time()
    bs = boxes()
    parts = []
    for s in sorted(bs):
        b = bs[s]
        locked = "LOCK" if (b.get("lock_until", 0) > now) else "    "
        parts.append(f"{s}={b['holder'][0].upper()}/{b['current_face_id']} [{locked}]")
    print(f"  {label:<18} " + "  ".join(parts))


def push(fc: str, script: str) -> dict:
    return _post_json(f"{C2}/api/c2/{fc}/start", {"script": script})


def wait(secs: float, *, while_label: str | None = None) -> None:
    if while_label:
        print(f"  ... waiting {secs:.1f}s ({while_label})")
    else:
        print(f"  ... waiting {secs:.1f}s")
    time.sleep(secs)


def hold_at_home_script(team: str, alt: float = 1.6) -> str:
    home_y = -9.0 if team == "red" else 9.0
    return (f"TAKEOFF\nHEIGHT {alt}\nTO 0 {home_y:g}\nHOOVER 600\n")


def attack_script(box_x: float, box_y: float, *, hover_s: float = 3.5,
                  home_alt: float = 1.6, capture_alt: float = 1.5) -> str:
    # Take off, climb above box height, fly over the box, drop to capture
    # altitude, hover hover_s seconds, then go home and HOOVER forever.
    return (f"TAKEOFF\n"
            f"HEIGHT {capture_alt}\n"
            f"TO {box_x:g} {box_y:g}\n"
            f"HOOVER {hover_s:g}\n"
            f"HEIGHT {home_alt}\n"
            f"HOOVER 600\n")


def ensure_stack_up() -> None:
    try:
        bs = boxes()
        assert len(bs) == 6
    except Exception as e:
        print(f"!! sim not reachable at {SIM} ({e}); is ./run_local.sh running?",
              file=sys.stderr)
        sys.exit(2)
    try:
        _get_json(f"{C2}/api/c2/overview")
    except Exception as e:
        print(f"!! C2 not reachable at {C2} ({e})", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ensure_stack_up()
    print("Demo of SDC26 v7 capture mechanics (§1.4.3).")
    print("Box state per row:  slot=<H>/<face> [LOCK?]")
    show("start")

    # --- A. basic enemy capture --------------------------------------------
    # blue1 attacks RED box 1 at (-3, -7.5). 3.5 s hover > 2 s threshold.
    print("\n[A] enemy capture: blue1 -> red box 1 (expect 1 flips to blue, 5 s lock arms)")
    push("blue1", attack_script(-3.0, -7.5, hover_s=3.5))
    wait(20, while_label="blue1 flies + hovers + RTH")
    show("after A")

    # --- B. 5 s lock blocks recapture --------------------------------------
    # Immediately push red1 to slot 1; while the lock is open the box must
    # NOT flip. Then wait past the lock and the home-team drone (red1) will
    # instantly recap (case D).
    print("\n[B] lock test: red1 -> red box 1 right after capture "
          "(within 5-s lock -> NO flip)")
    push("red1", attack_script(-3.0, -7.5, hover_s=2.5))
    wait(3.0, while_label="red1 in flight; lock still active")
    show("during lock")

    print("\n[D] home recapture: lock expires -> red drone over its own box "
          "flips it back to red INSTANTLY (no 2 s hover, 0 pts)")
    wait(10, while_label="red1 reaches slot 1 after lock expires")
    show("after D")

    # --- C. defender blocks ------------------------------------------------
    # Park red2 at red box 2 first, then send blue2 over it. No flip.
    print("\n[C] defender block: red2 dwells at red box 2; blue2 attacks "
          "(should NOT flip — defender presence denies capture)")
    push("red2", attack_script(0.0, -7.5, hover_s=12.0))  # red2 sits over its own box ~12s
    wait(10, while_label="red2 arriving at slot 2")
    show("red2 present")
    push("blue2", attack_script(0.0, -7.5, hover_s=4.0))
    wait(22, while_label="blue2 tries to capture; red2 still defending")
    show("after C")

    print("\nDemo complete. The 3D view + the strategy dashboard show this "
          "live too. /api/world's `lock_until` field arms after every flip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
