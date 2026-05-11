"""
End-to-end perf probe for the sphinx sim video pipeline.

Pipeline (left = source, right = client):

    UE4 (T4 render) -> gz camera plugin -> shm -> pysphinx -> JPEG encode
                                              -> HTTP MJPEG -> network -> client

Each stage's frame rate is measured independently so the bottleneck is
obvious from a single run. The probe is read-only: it doesn't change
camera params, doesn't fly the drone, and doesn't restart any service.

Run on the AWS sphinx host (sphinx-control + FC must already be up):

    /opt/sdc-tobe/.venv/bin/python tools/perf_probe.py
    /opt/sdc-tobe/.venv/bin/python tools/perf_probe.py --camera horizontal_camera --duration 5

For a remote client view of the bottleneck (measured from your laptop),
add ``--client-url http://<host>:8080/api/video`` and the probe will
also report the rate the network actually delivers to the client.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager


PRETTY = {True: "OK", False: "FAIL"}


# ─── helpers ────────────────────────────────────────────────────────


@contextmanager
def section(title: str):
    print()
    print(f"━━━ {title} " + "━" * max(0, 60 - len(title)))
    yield
    sys.stdout.flush()


def sh(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command, return (rc, combined stdout+stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timed out after {timeout}s"


def http_get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ─── stage 1: host resource snapshot ───────────────────────────────


def host_snapshot() -> None:
    with section("host"):
        rc, out = sh(["nvidia-smi",
                      "--query-gpu=name,utilization.gpu,utilization.memory,"
                      "memory.used,memory.total,temperature.gpu,power.draw",
                      "--format=csv,noheader,nounits"])
        if rc == 0:
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    print(f"GPU: {parts[0]}  util={parts[1]}%  "
                          f"mem={parts[3]}/{parts[4]} MiB  "
                          f"temp={parts[5]}C  power={parts[6]} W")
        else:
            print("GPU: nvidia-smi unavailable")

        # CPU
        try:
            with open("/proc/loadavg") as f:
                la = f.read().split()[:3]
            print(f"CPU load (1/5/15 min): {' / '.join(la)}")
        except Exception:
            pass
        rc, out = sh(["nproc"])
        if rc == 0:
            print(f"CPU cores: {out.strip()}")

        # UE4 + gzserver process CPU%
        rc, out = sh(["ps", "-eo", "pid,%cpu,%mem,comm,args",
                      "--sort=-%cpu"], timeout=2)
        if rc == 0:
            for line in out.splitlines():
                if any(k in line for k in ("UnrealApp", "gzserver",
                                            "uvicorn server",
                                            "unified_api_server")):
                    print("  " + " ".join(line.split()[:4]) + " ...")


# ─── stage 2: UE4 render rate (best-effort) ────────────────────────


_FPS_RE = re.compile(r"(?:FPS|fps)[^0-9]{0,5}([0-9]+\.?[0-9]*)")


def ue4_fps_from_log(log_path: str) -> float | None:
    """Try to extract a recent FPS number from the UE4 Saved log.

    UE4 only writes FPS to disk on graceful shutdown unless ``stat fps``
    is enabled. Returns the last number found, or None.
    """
    try:
        with open(log_path) as f:
            last = None
            for line in f:
                m = _FPS_RE.search(line)
                if m:
                    try:
                        last = float(m.group(1))
                    except ValueError:
                        pass
            return last
    except FileNotFoundError:
        return None


def ue4_stage() -> None:
    with section("UE4 render"):
        # Path is fixed for the parrot-ue4-sphx-tests app on AWS.
        log = ("/opt/parrot-ue4-sphx-tests/SphxTests/Saved/Logs/"
               "SphxTests.log")
        if not os.path.exists(log):
            print(f"UE4 log not found at {log} — skipping")
            return
        fps = ue4_fps_from_log(log)
        if fps is None:
            print("UE4 log: no FPS line recorded (this is normal — UE4 "
                  "only logs FPS on shutdown or with `stat fps`).")
            print("Use `nvidia-smi dmon` or check gzserver/UE4 CPU% above "
                  "as a proxy.")
        else:
            print(f"UE4 last logged FPS: {fps:.1f}")


# ─── stage 3: shm read rate (raw pysphinx) ─────────────────────────


def shm_stage(camera: str, duration: float) -> tuple[float, int, int]:
    """Read raw frames from sphinx shm for `duration` seconds.

    Returns (fps, frames, bytes_total). Drops alpha, doesn't encode.
    """
    with section(f"pysphinx shm read ({camera})"):
        try:
            sys.path.insert(0, "/opt/parrot-sphinx/usr/lib/python/"
                               "site-packages")
            import pysphinx  # noqa: F401
            from pysphinx.sphinx import Sphinx
            from pysphinx.cameras import SphinxShmCamera
        except ImportError as e:
            print(f"pysphinx import failed: {e}")
            return 0.0, 0, 0

        # connect and find the machine name
        try:
            sphx = Sphinx()
            info = sphx.get_info()
        except Exception as e:
            print(f"sphinx JSON-RPC failed: {e}")
            return 0.0, 0, 0
        if not info:
            print("sphinx info empty — no drone running?")
            return 0.0, 0, 0
        # Pick first non-UE4 machine (the drone)
        machines = [k for k in info.keys() if not k.startswith("__")]
        if not machines:
            print("no machines in sphinx info")
            return 0.0, 0, 0
        machine = machines[0]
        print(f"machine: {machine}")
        try:
            cam = sphx.get_camera(machine, camera)
        except Exception as e:
            print(f"get_camera({machine!r},{camera!r}) failed: {e}")
            return 0.0, 0, 0
        if cam is None:
            print(f"camera {camera!r} not found on {machine!r}")
            return 0.0, 0, 0

        # warm up — first frame is sometimes the previous run's leftover
        try:
            cam.read()
        except Exception:
            pass

        frames = 0
        nbytes = 0
        t0 = time.time()
        deadline = t0 + duration
        last_seq = None
        unique = 0
        while time.time() < deadline:
            try:
                frame = cam.read(blocking=True, timeout=0.5)
            except Exception as e:
                print(f"read err: {e}")
                break
            if frame is None:
                continue
            buf = getattr(frame, "buffer", None)
            if buf is None:
                continue
            seq = getattr(frame, "frame_index", None) or \
                  getattr(frame, "seq", None) or id(buf)
            if seq != last_seq:
                unique += 1
                last_seq = seq
            frames += 1
            nbytes += buf.nbytes
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0
        ufps = unique / dt if dt > 0 else 0.0
        mb_s = nbytes / dt / 1e6 if dt > 0 else 0.0
        print(f"shm read: {fps:5.1f} fps  ({unique} unique → "
              f"{ufps:.1f} unique/s)")
        print(f"          {mb_s:5.1f} MB/s raw "
              f"({nbytes / max(1, frames) / 1024:.0f} kB/frame)")
        return ufps, unique, nbytes


# ─── stage 4: JPEG encode benchmark ─────────────────────────────────


def jpeg_stage(camera: str, n: int = 30) -> None:
    with section("JPEG encode"):
        try:
            sys.path.insert(0, "/opt/parrot-sphinx/usr/lib/python/"
                               "site-packages")
            from pysphinx.sphinx import Sphinx
            import cv2
            import numpy as np
        except ImportError as e:
            print(f"import failed: {e}")
            return
        try:
            sphx = Sphinx()
            info = sphx.get_info() or {}
            machines = [k for k in info if not k.startswith("__")]
            if not machines:
                print("no drone — skipping")
                return
            cam = sphx.get_camera(machines[0], camera)
        except Exception as e:
            print(f"setup failed: {e}")
            return
        if cam is None:
            print("camera not available")
            return
        frame = cam.read(blocking=True, timeout=2.0)
        if frame is None or frame.buffer is None:
            print("no frame available")
            return
        # Strip alpha (BGRA → BGR) like the production loop
        bgra = frame.buffer
        bgr = bgra[..., :3] if bgra.ndim == 3 else bgra
        sizes = []
        t0 = time.time()
        for _ in range(n):
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                sizes.append(jpg.nbytes)
        dt = time.time() - t0
        per = dt / n * 1000
        print(f"encode {n}x: avg {per:5.1f} ms/frame  "
              f"({n / dt:5.1f} encodes/s)")
        if sizes:
            avg_kb = sum(sizes) / len(sizes) / 1024
            print(f"JPEG size : {avg_kb:5.1f} kB/frame "
                  f"(at quality=80, {bgr.shape[1]}x{bgr.shape[0]})")


# ─── stage 5: FC server side rate ───────────────────────────────────


def fc_stage(fc_url: str, duration: float) -> None:
    with section(f"FC server-side rate ({fc_url})"):
        # /api/video/status reports frames_decoded — sample twice and
        # diff. That's the rate the FC's `_sim_video_loop` actually
        # produces (post-JPEG-encode), independent of any HTTP client.
        try:
            t0 = time.time()
            a = http_get_json(fc_url.rstrip("/") + "/api/video/status")
            time.sleep(duration)
            b = http_get_json(fc_url.rstrip("/") + "/api/video/status")
            dt = time.time() - t0 - 0  # the diff IS over duration+overhead
        except Exception as e:
            print(f"FC status fetch failed: {e}")
            return
        fa = a.get("frames_decoded", 0)
        fb = b.get("frames_decoded", 0)
        dt = max(1e-6, dt)
        fps = (fb - fa) / dt
        print(f"FC frames_decoded delta: {fb - fa}  over {dt:.1f}s  "
              f"→ {fps:5.1f} fps")
        print(f"  has_frame={b.get('has_frame')} "
              f"mode={b.get('mode')} cv2={b.get('cv2_available')}")


# ─── stage 6: client-perceived rate over the network ───────────────


def client_stage(client_url: str, duration: float) -> None:
    """Connect to the MJPEG stream as a real client and count boundary
    markers. Measures end-to-end including network latency.
    """
    with section(f"client MJPEG over network ({client_url})"):
        try:
            req = urllib.request.Request(client_url)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=5) as r:
                # Need the boundary from content-type to count frames
                ctype = r.headers.get("Content-Type", "")
                m = re.search(r"boundary=([^;]+)", ctype)
                if not m:
                    print(f"no MJPEG boundary in content-type: {ctype}")
                    return
                boundary = ("--" + m.group(1).strip().strip('"')).encode()
                deadline = t0 + duration
                buf = b""
                frames = 0
                nbytes = 0
                # Read in chunks; count boundary occurrences.
                while time.time() < deadline:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    nbytes += len(chunk)
                    buf += chunk
                    # Count boundaries that fully landed; keep tail.
                    parts = buf.split(boundary)
                    frames += max(0, len(parts) - 1)
                    buf = parts[-1]
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0.0
            mbps = nbytes * 8 / dt / 1e6 if dt > 0 else 0.0
            print(f"client received: {frames} frames in {dt:.1f}s "
                  f"→ {fps:5.1f} fps")
            print(f"                 {mbps:5.2f} Mbps over the wire")
        except Exception as e:
            print(f"client stream failed: {e}")


# ─── stage 7: network latency snapshot ─────────────────────────────


def network_stage(target: str | None) -> None:
    if not target:
        return
    with section(f"network latency to {target}"):
        rc, out = sh(["ping", "-c", "5", "-W", "2", target], timeout=10)
        if rc == 0:
            for line in out.splitlines():
                if "min/avg" in line or "packet loss" in line:
                    print(line.strip())
        else:
            print(f"ping failed: rc={rc}\n{out}")


# ─── main ───────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="horizontal_camera",
                   help="sphinx camera name "
                        "(horizontal_camera or vertical_camera)")
    p.add_argument("--duration", type=float, default=4.0,
                   help="seconds to measure each rate (default 4)")
    p.add_argument("--fc-url", default="http://127.0.0.1:8080",
                   help="flight controller base URL (default :8080)")
    p.add_argument("--client-url", default=None,
                   help="if set, also measure the rate a real MJPEG "
                        "client sees over the network "
                        "(e.g. http://host:8080/api/video)")
    p.add_argument("--ping", default=None,
                   help="optional host to ping for network RTT")
    p.add_argument("--skip-shm", action="store_true",
                   help="skip the pysphinx shm read benchmark")
    p.add_argument("--skip-jpeg", action="store_true",
                   help="skip the cv2.imencode benchmark")
    args = p.parse_args()

    print(f"perf_probe — camera={args.camera}  duration={args.duration}s")

    host_snapshot()
    ue4_stage()
    if not args.skip_shm:
        shm_stage(args.camera, args.duration)
    if not args.skip_jpeg:
        jpeg_stage(args.camera, n=30)
    fc_stage(args.fc_url, args.duration)
    if args.client_url:
        client_stage(args.client_url, args.duration)
    if args.ping:
        network_stage(args.ping)

    with section("how to read this"):
        print(textwrap_dedent("""
            If shm-read fps is low (<10): bottleneck is the GPU/UE4
                render path. Check GPU util in `host`; the T4 should
                be 50-90 %% during steady render. If it's 100 %%, scene
                is GPU-bound. If 20 %%, gz_camera_plugin or shm sync
                is the limit.
            If shm-read is fine (>15) but FC fps is low: bottleneck is
                JPEG encode or _sim_video_loop scheduling. Check
                jpeg-encode ms/frame; if >50 ms, drop JPEG quality from
                80 to 60 or downscale.
            If FC fps is fine but client fps is low: bottleneck is
                network. Compare client Mbps to expected JPEG-size *
                FC-fps; if much lower, network is saturated. Check
                tailscale relay path (`tailscale status`); a DERP relay
                ("fra", "fra2"...) adds 30-80 ms and caps throughput.
            If everything is fine here but the video FEELS choppy:
                browser side. Open the stream in `mpv` or `ffplay`
                and compare; choppy in mpv too -> source issue,
                smooth in mpv but jittery in browser -> browser
                MJPEG decoder.
        """).strip())


def textwrap_dedent(s: str) -> str:
    # tiny inline dedent so we don't depend on textwrap import order
    lines = s.splitlines()
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()),
                 default=0)
    return "\n".join(l[indent:] for l in lines)


if __name__ == "__main__":
    main()
