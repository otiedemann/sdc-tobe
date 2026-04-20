#!/usr/bin/env python3
"""
Terminal keyboard controller for the Parrot Anafi fleet.

Talks to a running unified_api_server.py over HTTP (not djitellopy), so it
works against the same /api/rc endpoint the C2 web controller uses. Run it
on any machine that can reach the drone's API (your laptop, the flight
controller itself, or a separate ops terminal).

USAGE
    python3 controller/anafi_keyboard_terminal.py
        [--api  http://flightctrl1:8080]    # the drone you want to pilot
        [--c2   http://localhost:8090]      # optional: C2 web controller URL,
                                            # used for the 0-key "land ALL"
        [--stick 60]                        # RC magnitude 0..100
        [--rate  20]                        # RC loop Hz
        [--tick-ms 250]                     # RC duration_ms per packet

KEYS (terminal must be focused)
    w / s    forward / back
    a / d    strafe left / right
    q / e    yaw left / right
    r / f    up / down
    x        stop RC (zero all axes)
    t        takeoff
    l        land
    0        LAND ALL drones (requires --c2, panics the whole fleet)
    ESC      quit (sends rc_stop on exit; only auto-lands if we took off)

MOVEMENT BEHAVIOUR
    Terminals don't emit key-up events in cbreak mode — they do emit auto-
    repeat characters while a key is held. So we treat a key as "held" if
    it was pressed within the last HOLD_WINDOW_MS milliseconds. Release
    happens automatically HOLD_WINDOW_MS after the last tap. Press X to
    force an immediate stop without waiting for the window to expire.

EXIT
    Ctrl+C and ESC both restore the terminal settings and send a single
    rc_stop (lr=fb=ud=yaw=0). If we launched the drone via the T key,
    we'll also send a land on exit; otherwise we leave it to the C2.
"""

from __future__ import annotations

import argparse
import signal
import select
import sys
import termios
import threading
import time
import tty
from typing import Optional

import requests

RC_HZ_DEFAULT = 20
STICK_DEFAULT = 60
TICK_MS_DEFAULT = 250     # duration_ms on each /api/rc packet
HOLD_WINDOW_MS = 180      # key considered "held" if pressed within this many ms

# Keys we actually care about (anything else is ignored)
MOVEMENT_KEYS = {"w", "s", "a", "d", "q", "e", "r", "f"}
ONESHOT_KEYS = {"t", "l", "x", "0", "esc"}


# ─── Shared state ───────────────────────────────────────────────────────────
_last_press: dict[str, float] = {}      # key → last-seen wall-clock time
_oneshot_pending: list[str] = []        # FIFO of one-shot keys to consume
_state_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _record_press(key: str):
    """Called by the reader thread for every keystroke."""
    key = key.lower()
    with _state_lock:
        if key in MOVEMENT_KEYS:
            _last_press[key] = time.time()
        elif key in ONESHOT_KEYS:
            _oneshot_pending.append(key)


def _is_held(key: str, window_ms: int = HOLD_WINDOW_MS) -> bool:
    with _state_lock:
        t = _last_press.get(key)
    if t is None:
        return False
    return (time.time() - t) * 1000 <= window_ms


def _pop_oneshot() -> Optional[str]:
    with _state_lock:
        return _oneshot_pending.pop(0) if _oneshot_pending else None


def _axis(pos_held: bool, neg_held: bool) -> int:
    return (1 if pos_held else 0) + (-1 if neg_held else 0)


# ─── Terminal reader thread ─────────────────────────────────────────────────

def key_reader(running_flag: list[bool]):
    """Puts terminal into cbreak mode and forwards every keystroke into the
    shared state. ESC becomes the literal string "esc". Restores termios on
    exit (even if the main thread crashed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while running_flag[0]:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _record_press("esc")
            elif ch == "\x03":   # Ctrl+C — let main handle via SIGINT
                running_flag[0] = False
                break
            else:
                _record_press(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─── HTTP helpers ───────────────────────────────────────────────────────────

class DroneClient:
    def __init__(self, api_base: str, timeout: float = 2.0):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def _post(self, path: str, body: Optional[dict] = None):
        try:
            r = self.s.post(f"{self.api_base}{path}", json=body or {},
                            timeout=self.timeout)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {"raw": r.text[:120]}
        except Exception as e:
            return 0, {"error": str(e)[:120]}

    def rc(self, lr: int, fb: int, ud: int, yaw: int, duration_ms: int):
        return self._post("/api/rc", {
            "lr": int(lr), "fb": int(fb),
            "ud": int(ud), "yaw": int(yaw),
            "duration_ms": int(duration_ms),
        })

    def takeoff(self): return self._post("/api/takeoff")
    def land(self):    return self._post("/api/land")
    def rc_stop(self): return self.rc(0, 0, 0, 0, 200)

    def heartbeat(self):
        try:
            r = self.s.get(f"{self.api_base}/api/heartbeat",
                           timeout=self.timeout)
            return r.status_code == 200, r.json() if r.ok else None
        except Exception:
            return False, None


class C2Client:
    def __init__(self, c2_base: Optional[str], timeout: float = 4.0):
        self.c2_base = c2_base.rstrip("/") if c2_base else None
        self.timeout = timeout
        self.s = requests.Session()

    def land_all(self):
        if not self.c2_base:
            return 0, {"error": "no --c2 URL configured"}
        try:
            r = self.s.post(f"{self.c2_base}/proxy/land_all",
                            timeout=self.timeout)
            return r.status_code, r.json() if r.ok else {"raw": r.text[:120]}
        except Exception as e:
            return 0, {"error": str(e)[:120]}


# ─── Main loop ──────────────────────────────────────────────────────────────

BANNER = """\
Anafi terminal controller
─────────────────────────────────────────────────────────────
  w/s forward/back    a/d strafe L/R    q/e yaw L/R    r/f up/dn
  t   takeoff         l   land          x   stop RC
  0   LAND ALL (fleet-wide, via C2)     ESC / Ctrl-C quit
─────────────────────────────────────────────────────────────
"""


def _status_line(lr: int, fb: int, ud: int, yaw: int, flying: bool,
                 connected: bool) -> str:
    conn = "\033[32mOK\033[0m" if connected else "\033[31mNO\033[0m"
    fly  = "\033[33mAIR\033[0m" if flying else "\033[90mgnd\033[0m"
    return (f"\r[{conn} {fly}]  "
            f"lr={lr:+4d}  fb={fb:+4d}  ud={ud:+4d}  yaw={yaw:+4d}   ")


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--api", default="http://flightctrl1:8080",
                   help="unified_api_server base URL (default: %(default)s)")
    p.add_argument("--c2", default=None,
                   help="C2 web controller URL, enables 0-key LAND ALL")
    p.add_argument("--stick", type=int, default=STICK_DEFAULT,
                   help=f"RC magnitude 0..100 (default: {STICK_DEFAULT})")
    p.add_argument("--rate", type=int, default=RC_HZ_DEFAULT,
                   help=f"RC loop Hz (default: {RC_HZ_DEFAULT})")
    p.add_argument("--tick-ms", type=int, default=TICK_MS_DEFAULT,
                   help=f"duration_ms per packet (default: {TICK_MS_DEFAULT})")
    args = p.parse_args()

    stick = max(0, min(100, args.stick))
    rate  = max(1, min(60, args.rate))
    tick_ms = max(50, min(2000, args.tick_ms))

    drone = DroneClient(args.api)
    c2    = C2Client(args.c2)

    # Initial heartbeat check
    print(BANNER)
    ok, hb = drone.heartbeat()
    if ok:
        print(f"  API:       {args.api}  (connected={hb.get('connected')}, "
              f"flying={hb.get('flying')}, type={hb.get('drone_type')})")
        flying = bool(hb.get("flying"))
    else:
        print(f"  API:       {args.api}  \033[31m(NOT REACHABLE — will keep "
              f"retrying)\033[0m")
        flying = False
    if args.c2:
        print(f"  C2:        {args.c2}  — press 0 for LAND ALL")
    else:
        print(f"  C2:        (not configured — pass --c2 to enable LAND ALL)")
    print(f"  Stick: {stick}%   Rate: {rate} Hz   Tick: {tick_ms} ms")
    print()

    running = [True]

    def _sigint(*_):
        running[0] = False
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    reader = threading.Thread(target=key_reader, args=(running,), daemon=True)
    reader.start()

    tick_s = 1.0 / rate
    took_off_here = False
    last_status_print = 0.0

    try:
        while running[0]:
            # ── one-shot keys ──
            key = _pop_oneshot()
            while key is not None:
                if key == "t":
                    print("\n[t] Takeoff →", end=" ", flush=True)
                    code, resp = drone.takeoff()
                    print(f"HTTP {code} {resp}", flush=True)
                    if resp.get("ok"):
                        flying = True
                        took_off_here = True
                elif key == "l":
                    print("\n[l] Land →", end=" ", flush=True)
                    code, resp = drone.land()
                    print(f"HTTP {code} {resp}", flush=True)
                    flying = False
                elif key == "x":
                    code, resp = drone.rc_stop()
                    # silent — status line will reflect zero RC
                elif key == "0":
                    print("\n[0] LAND ALL →", end=" ", flush=True)
                    code, resp = c2.land_all()
                    print(f"HTTP {code} {resp}", flush=True)
                    flying = False
                elif key == "esc":
                    running[0] = False
                    break
                key = _pop_oneshot()
            if not running[0]:
                break

            # ── continuous movement ──
            w = _is_held("w"); s = _is_held("s")
            a = _is_held("a"); d = _is_held("d")
            q = _is_held("q"); e = _is_held("e")
            r = _is_held("r"); f = _is_held("f")

            # 'x' being held forces stop for the entire window
            if _is_held("x"):
                lr = fb = ud = yaw = 0
            else:
                lr  = _axis(d, a) * stick
                fb  = _axis(w, s) * stick
                ud  = _axis(r, f) * stick
                yaw = _axis(e, q) * stick

            drone.rc(lr, fb, ud, yaw, tick_ms)

            # ── status line (throttled to ~5 Hz) ──
            tnow = time.time()
            if tnow - last_status_print >= 0.2:
                sys.stdout.write(_status_line(lr, fb, ud, yaw, flying, True))
                sys.stdout.flush()
                last_status_print = tnow

            time.sleep(tick_s)

    finally:
        running[0] = False
        # Final rc_stop so the drone doesn't continue drifting after we exit
        try:
            drone.rc_stop()
        except Exception:
            pass
        # Only auto-land if WE took off via this script — don't land a drone
        # that was already airborne when we started (might be mission-owned).
        if took_off_here:
            try:
                print("\n  auto-landing drone we launched…", flush=True)
                drone.land()
            except Exception:
                pass
        print("\n  exited cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
