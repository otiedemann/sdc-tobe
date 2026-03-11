import sys
import time
import tty
import termios
import select
import threading

from djitellopy import Tello

RC_HZ = 20
STICK = 60
PRINT_HZ = 2

pressed = set()
lock = threading.Lock()


def axis(pos: bool, neg: bool) -> int:
    return (1 if pos else 0) + (-1 if neg else 0)


def key_reader(running_flag):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while running_flag[0]:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)
            with lock:
                if ch == "\x1b":
                    pressed.add("esc")
                else:
                    pressed.add(ch.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def consume_key(k: str) -> bool:
    with lock:
        if k in pressed:
            pressed.discard(k)
            return True
        return False


def has_key(k: str) -> bool:
    with lock:
        return k in pressed


def main():
    print("Connecting to Tello...")
    t = Tello()
    t.connect()
    print("Battery:", t.get_battery())

    running = [True]
    flying = False
    last_print = 0.0

    reader = threading.Thread(target=key_reader, args=(running,), daemon=True)
    reader.start()

    print("Controls (Terminal must be focused):")
    print("  t takeoff | l land | x stop rc | ESC quit")
    print("  Hold movement keys by repeating taps quickly:")
    print("  w/s forward/back, a/d left/right, r/f up/down, q/e yaw")

    try:
        while running[0]:
            # one-shot keys
            if consume_key("t") and not flying:
                t.takeoff()
                flying = True
                print("Takeoff")
            if consume_key("l") and flying:
                t.land()
                flying = False
                print("Land")
            if consume_key("esc"):
                print("Quit")
                running[0] = False
                break

            # continuous-ish control from recently pressed keys
            # (each key press stays active for one loop cycle)
            lr = axis(consume_key("d"), consume_key("a")) * STICK
            fb = axis(consume_key("w"), consume_key("s")) * STICK
            ud = axis(consume_key("r"), consume_key("f")) * STICK
            yaw = axis(consume_key("e"), consume_key("q")) * STICK

            if consume_key("x"):
                lr = fb = ud = yaw = 0

            try:
                t.send_rc_control(lr, fb, ud, yaw)
            except Exception:
                pass

            now = time.time()
            if now - last_print >= 1.0 / PRINT_HZ:
                last_print = now
                print(f"RC lr={lr:4d} fb={fb:4d} ud={ud:4d} yaw={yaw:4d}")

            time.sleep(1.0 / RC_HZ)

    finally:
        running[0] = False
        try:
            t.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        if flying:
            try:
                t.land()
            except Exception:
                pass
        try:
            t.end()
        except Exception:
            pass
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
