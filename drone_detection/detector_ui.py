"""
Web UI for the drone collision detector.

Provides:
  * Single-page dashboard with live MJPEG video and SVG-overlay of
    detected drones (bounding boxes drawn in the browser, scaled to
    whatever size the video element ends up at).
  * Editable config: FC api_base, model_path, conf_threshold, evasion
    bank magnitudes (per direction).
  * Real-time read-out of detections + the RC command the evasion law
    *would* send.
  * "Arm evasion" toggle that, when on, actually POSTs /api/rc to the
    FC each tick so the drone dodges intruders. Off by default — the
    dashboard is safe to leave open in a browser without anything
    flying away.

Run:
    source drone_detection/.venv/bin/activate
    pip install flask requests  # if you haven't already
    python drone_detection/detector_ui.py --port 9091

Then open http://<host>:9091/.

Architecture note: the existing DroneDetector class (drone_detector.py)
is reused unchanged — this module wraps it with a Flask service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# Make `drone_detection` importable when run directly
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from drone_detection.drone_detector import DroneDetector

PRETRAINED_ONNX = THIS_DIR / "pretrained" / "yolo11n_seraphim_e37.onnx"

# ─── defaults — overridable via env or via the UI's POST /api/config ──

DEFAULT_CONFIG = {
    "api_base":       os.environ.get("FC_BASE",  "http://localhost:8080"),
    "model_path":     os.environ.get("MODEL_PATH", str(PRETRAINED_ONNX)),
    "conf_threshold": 0.25,
    # evasion gains: peak RC magnitude (units: roll/pitch sticks, ±100)
    # applied when the intruder is at the screen edge. Scales linearly
    # toward 0 as the intruder approaches the centre.
    "bank_left":      50,    # roll command when intruder is on the RIGHT (push us left)
    "bank_right":     50,    # roll command when intruder is on the LEFT
    "bank_up":        30,    # pitch command (climb) when intruder is BELOW
    "bank_down":      30,    # pitch command (descend) when intruder is ABOVE
    # Only act on detections at least this confident
    "evade_conf_min": 0.45,
    # Only act on detections at least this size in pixels² (closer drones)
    "evade_bbox_min": 1500,
    # Send an RC tick at this rate when armed (Hz)
    "evade_rate_hz":  10,
}


# ─── state ────────────────────────────────────────────────────────────

class DetectorService:
    """Owns one DroneDetector + the evasion thread + the live config."""

    def __init__(self) -> None:
        self.cfg = dict(DEFAULT_CONFIG)
        self.detector: DroneDetector | None = None
        self.armed: bool = False
        self._lock = threading.Lock()
        self._evader: threading.Thread | None = None
        self._evader_stop = threading.Event()
        self._last_rc = {"roll": 0, "pitch": 0, "yaw": 0, "throttle": 0, "ts": 0.0}

    # ─── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self.detector is not None:
                self.detector.stop()
                self.detector = None
            self.detector = DroneDetector(
                api_base=self.cfg["api_base"],
                model_path=self.cfg["model_path"],
                conf_threshold=float(self.cfg["conf_threshold"]),
            )
            self.detector.start()

    def stop(self) -> None:
        self.disarm()
        with self._lock:
            if self.detector is not None:
                self.detector.stop()
                self.detector = None

    # ─── config ────────────────────────────────────────────────────

    def update_config(self, patch: dict) -> dict:
        """Merge patch into cfg. If api_base, model_path or conf
        changed AND detector is running, restart it."""
        restart_keys = {"api_base", "model_path", "conf_threshold"}
        changed = False
        for k, v in patch.items():
            if k not in self.cfg:
                continue
            # numeric coerce for known numeric fields
            if k in {"conf_threshold", "evade_conf_min"}:
                v = float(v)
            elif k in {"bank_left", "bank_right", "bank_up", "bank_down",
                       "evade_bbox_min", "evade_rate_hz"}:
                v = int(float(v))
            if self.cfg[k] != v:
                self.cfg[k] = v
                if k in restart_keys:
                    changed = True
        if changed and self.detector is not None:
            self.start()
        return self.cfg

    # ─── arm / disarm ──────────────────────────────────────────────

    def arm(self) -> None:
        if self.armed:
            return
        if self.detector is None:
            self.start()
        self._evader_stop.clear()
        self._evader = threading.Thread(target=self._evasion_loop, daemon=True,
                                        name="evader")
        self._evader.start()
        self.armed = True

    def disarm(self) -> None:
        if not self.armed:
            return
        self.armed = False
        self._evader_stop.set()
        if self._evader is not None and self._evader.is_alive():
            self._evader.join(timeout=2)
        self._evader = None
        # send a zero RC tick so the drone doesn't keep banking
        self._send_rc(0, 0, 0, 0)

    # ─── evasion law ───────────────────────────────────────────────

    def compute_evasion(self, detections: list[dict]) -> dict:
        """Decide the RC stick output based on the worst (closest+highest-
        confidence) detection. Returns {roll, pitch, yaw, throttle,
        intruder} — `intruder` is the detection used (or None).

        Roll: positive bank RIGHT, negative bank LEFT.
        Pitch: positive PITCH FORWARD (Anafi convention is +pitch = forward,
            -pitch = back; for "climb" we use throttle, not pitch. We'll
            use throttle for vertical evasion).
        We treat "bank up" as throttle UP, "bank down" as throttle DOWN.
        """
        worst = None
        for d in detections:
            if d["confidence"] < self.cfg["evade_conf_min"]:
                continue
            if d["bbox_area"] < self.cfg["evade_bbox_min"]:
                continue
            if worst is None or d["bbox_area"] > worst["bbox_area"]:
                worst = d

        if worst is None:
            return {"roll": 0, "pitch": 0, "yaw": 0, "throttle": 0,
                    "intruder": None, "reason": "no qualifying threats"}

        # intruder on the RIGHT (rel_x > 0)  → bank LEFT (negative roll)
        # intruder on the LEFT  (rel_x < 0)  → bank RIGHT (positive roll)
        rel_x = worst["rel_x"]
        rel_y = worst["rel_y"]
        if rel_x > 0:
            roll = -int(self.cfg["bank_left"]  * min(1.0, abs(rel_x)))
        else:
            roll =  int(self.cfg["bank_right"] * min(1.0, abs(rel_x)))
        # intruder ABOVE (rel_y < 0)  → descend (negative throttle)
        # intruder BELOW (rel_y > 0)  → climb   (positive throttle)
        if rel_y > 0:
            throttle =  int(self.cfg["bank_up"]   * min(1.0, abs(rel_y)))
        else:
            throttle = -int(self.cfg["bank_down"] * min(1.0, abs(rel_y)))
        return {
            "roll": roll, "pitch": 0, "yaw": 0, "throttle": throttle,
            "intruder": worst,
            "reason": f"largest-threat conf={worst['confidence']:.2f} "
                      f"size={worst['bbox_area']:.0f}px",
        }

    def _evasion_loop(self) -> None:
        period = 1.0 / max(1, int(self.cfg["evade_rate_hz"]))
        while not self._evader_stop.is_set():
            d = self.detector.get_detections() if self.detector else []
            rc = self.compute_evasion(d)
            self._send_rc(rc["roll"], rc["pitch"], rc["yaw"], rc["throttle"])
            time.sleep(period)

    def _send_rc(self, roll: int, pitch: int, yaw: int, throttle: int) -> None:
        self._last_rc = {"roll": roll, "pitch": pitch, "yaw": yaw,
                         "throttle": throttle, "ts": time.time()}
        try:
            requests.post(
                f"{self.cfg['api_base']}/api/rc",
                json={"roll": roll, "pitch": pitch, "yaw": yaw,
                      "throttle": throttle},
                timeout=0.5,
            )
        except Exception:
            # FC unreachable / dropped — just log via last_rc; don't crash
            pass

    # ─── snapshots for the UI ──────────────────────────────────────

    def state(self) -> dict:
        dets = self.detector.get_detections() if self.detector else []
        fw, fh = self.detector.get_frame_size() if self.detector else (1280, 720)
        rc = self.compute_evasion(dets)
        return {
            "running": self.detector is not None,
            "armed":   self.armed,
            "config":  self.cfg,
            "detections": dets,
            "frame_w": fw,
            "frame_h": fh,
            "rc":      rc,
            "last_rc_sent": self._last_rc,
        }


SERVICE = DetectorService()


# ─── Flask app ────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(THIS_DIR / "templates"))


@app.route("/")
def index():
    return render_template("detector.html",
                           defaults=DEFAULT_CONFIG)


@app.route("/api/state")
def api_state():
    return jsonify(SERVICE.state())


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        patch = request.get_json(force=True) or {}
        SERVICE.update_config(patch)
    return jsonify(SERVICE.cfg)


@app.route("/api/start", methods=["POST"])
def api_start():
    SERVICE.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    SERVICE.stop()
    return jsonify({"ok": True})


@app.route("/api/arm", methods=["POST"])
def api_arm():
    SERVICE.arm()
    return jsonify({"ok": True, "armed": True})


@app.route("/api/disarm", methods=["POST"])
def api_disarm():
    SERVICE.disarm()
    return jsonify({"ok": True, "armed": False})


@app.route("/api/video")
def api_video():
    """Proxy the FC's MJPEG stream so the browser sees it under our
    origin (avoids CORS + lets us swap upstream cleanly when the
    operator edits api_base)."""
    upstream = f"{SERVICE.cfg['api_base']}/api/video"
    try:
        r = requests.get(upstream, stream=True, timeout=10)
    except Exception as e:
        return Response(f"upstream unreachable: {e}", status=502)
    return Response(
        stream_with_context(r.iter_content(chunk_size=32 * 1024)),
        mimetype=r.headers.get("Content-Type",
                               "multipart/x-mixed-replace; boundary=frame"),
        headers={"Cache-Control": "no-cache, no-store"},
    )


# ─── main ─────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("DETECTOR_UI_PORT", "9091")))
    p.add_argument("--autostart", action="store_true",
                   help="Start the detector on launch (skip the Start button)")
    args = p.parse_args()

    if args.autostart:
        try:
            SERVICE.start()
            print(f"[detector-ui] DroneDetector started")
        except Exception as e:
            print(f"[detector-ui] autostart failed: {e}")

    print(f"[detector-ui] listening on http://{args.host}:{args.port}/")
    print(f"[detector-ui] FC api_base = {SERVICE.cfg['api_base']}")
    print(f"[detector-ui] model       = {SERVICE.cfg['model_path']}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
