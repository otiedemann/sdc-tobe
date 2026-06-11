#!/usr/bin/env python3
"""Direct-to-drone takeoff latency probe — BYPASSES the whole FC software stack
(unified_api_server, marker_mission, C2). Talks straight to the Anafi via Olympe
so you can A/B the drone's own press->rotor latency against the FC path.

IMPORTANT: the Anafi accepts ONE controller connection at a time. Stop the FC
service first so it releases the link:

    sudo systemctl stop sdc-fc.service

Run (use the venv that already has Olympe):

    /home/sdc/sdc-tobe/controller_unified/.venv/bin/python3 \
        /home/sdc/sdc-tobe/tools/direct_takeoff_test.py

Optional: pass the drone IP (default 192.168.42.1 = Anafi's own AP) and a
HOVER_S before auto-landing:

    DRONE_IP=192.168.42.1 HOVER_S=4 .../python3 .../direct_takeoff_test.py

It prints, with monotonic timestamps:
  - connect time
  - TakeOff() -> 'takingoff' (ROTOR START / liftoff)   <-- the number you want
  - TakeOff() -> 'hovering'  (physical takeoff complete)
then lands and disconnects.
"""
import os
import time

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

IP = os.getenv("DRONE_IP", "192.168.42.1")
HOVER_S = float(os.getenv("HOVER_S", "3.0"))


def main() -> None:
    print(f"[direct] connecting to {IP} ...")
    drone = olympe.Drone(IP)

    t = time.monotonic()
    if not drone.connect():
        print("[direct] CONNECT FAILED — is sdc-fc.service still holding the "
              "link? `sudo systemctl stop sdc-fc.service` and retry.")
        return
    print(f"[direct] connect: +{time.monotonic() - t:.3f}s")

    # Let FlyingStateChanged populate after connect (the FC path waits on this
    # too — see backend.takeoff()).
    time.sleep(0.5)
    fs = drone.get_state(FlyingStateChanged)
    print(f"[direct] initial FlyingStateChanged="
          f"{'None' if fs is None else fs.get('state')}")

    # Rotor start: TakeOff command -> firmware transitions to 'takingoff'.
    t = time.monotonic()
    ok_lift = drone(
        TakeOff() >> FlyingStateChanged(state="takingoff", _timeout=10)
    ).wait().success()
    t_lift = time.monotonic() - t
    print(f"[direct] TakeOff -> takingoff (ROTOR START): +{t_lift:.3f}s "
          f"ok={ok_lift}")

    # Physical takeoff complete: -> 'hovering'.
    ok_hover = drone(
        FlyingStateChanged(state="hovering", _timeout=10)
    ).wait().success()
    print(f"[direct] -> hovering (takeoff complete): "
          f"+{time.monotonic() - t:.3f}s ok={ok_hover}")

    time.sleep(HOVER_S)

    print("[direct] landing ...")
    drone(Landing()).wait(_timeout=10)
    drone.disconnect()
    print("[direct] done. Restart the FC when finished: "
          "`sudo systemctl start sdc-fc.service`")


if __name__ == "__main__":
    main()
