"""Tiny stub of one marker_mission flight controller, for local C2 dev.

Listens on ``--port``, implements the minimum subset of marker_mission's
HTTP surface the C2 reads/writes. State is in-memory and per-process;
restart the stub to reset.

Run six of these on 9001..9006 and point a ``config.dev.json`` at them
to smoke-test the C2 without any drone hardware.

    python -m marker_mission_c2.tools.fake_fc --port 9001 --serial PIA1
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fake_fc")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--serial", default="FAKE0001",
                    help="Reported drone serial (used as identity).")
    ap.add_argument("--phase", default="init",
                    help="Initial mission phase (init, takeoff, search, "
                         "approach, hold, land, done, abort).")
    ap.add_argument("--data-dir", default=None,
                    help="Override the calib_dir / state dir. Default: "
                         "/tmp/fake_fc_<port>.")
    args = ap.parse_args(argv)

    data_dir = (Path(args.data_dir) if args.data_dir
                else Path("/tmp") / f"fake_fc_{args.port}")
    calib_dir = data_dir / "calibrations"
    calib_dir.mkdir(parents=True, exist_ok=True)

    state_lock = Lock()
    state = {
        "phase": args.phase,
        "phase_age_s": 0.0,
        "distance_m": None,
        "yaw_to_marker_deg": None,
        "relative_heading_deg": None,
        "world_position_m": [0.0, 0.0, 0.0],
        "mission_step_idx": None,
        "mission_script": [],
        "drone_connected": True,
        "telemetry": {
            "serial_number": args.serial,
            "battery": 92,
            "height_cm": 0,
            "yaw": 0.0,
            "flying": False,
        },
        "rc": {"lr": 0, "fb": 0, "ud": 0, "yaw": 0},
        "note": "stub FC",
    }
    persisted: dict = {
        "active_arena": {"width_m": 10, "depth_m": 25,
                         "top_z_m": 4, "bottom_z_m": 2,
                         "marker_size_m": 0.18,
                         "markers": [{"id": 4, "label": "stub",
                                       "wall": "front",
                                       "x": 0.0, "y": 3.0, "z": 1.98}]},
        "tune_current_values": {"fwd_rc_max": 10, "lat_rc_max": 10,
                                 "fwd_kp": 30, "lat_kp": 30,
                                 "fwd_kv": 0.4, "lat_kv": 0.4,
                                 "distance_floor_factor": 0.7,
                                 "target_distance_m": 2.0,
                                 "killswitch_key": "@",
                                 "target_marker_id": 4},
        "active_script": "TAKEOFF\nAPPROACH\nHOOVER\nLAND\n",
    }

    app = Flask(f"fake_fc_{args.port}")

    @app.get("/api/state")
    def api_state():
        with state_lock:
            return jsonify(dict(state))

    @app.get("/api/identity")
    def api_identity():
        return jsonify({
            "hostname": f"fake-fc-{args.port}",
            "drone_serial": args.serial,
            "drone_connected": True,
            "marker_mission_version": "stub",
        })

    @app.post("/api/start")
    def api_start():
        body = request.get_json(silent=True) or {}
        with state_lock:
            script = (body.get("script") or persisted["active_script"]
                      or "")
            lines = [l.strip() for l in script.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            state["mission_script"] = lines
            state["mission_step_idx"] = 0 if lines else None
            state["phase"] = "takeoff" if lines else "init"
            state["telemetry"]["flying"] = True
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def api_stop():
        with state_lock:
            phase = state["phase"]
            if phase in ("init", "done", "abort"):
                return jsonify({"ok": False,
                                "error": f"mission not running "
                                         f"(phase={phase})"}), 409
            state["phase"] = "abort"
            state["telemetry"]["flying"] = False
        return jsonify({"ok": True})

    @app.get("/api/arena/active")
    def api_arena_get():
        return jsonify(persisted["active_arena"])

    @app.post("/api/arena/active")
    def api_arena_set():
        body = request.get_json(silent=True) or {}
        persisted["active_arena"] = body
        return jsonify({"ok": True})

    @app.get("/api/tune")
    def api_tune_get():
        return jsonify({"current_values": persisted["tune_current_values"]})

    @app.post("/api/tune/apply")
    def api_tune_apply():
        body = request.get_json(silent=True) or {}
        for k, v in body.items():
            persisted["tune_current_values"][k] = v
        return jsonify({"ok": True})

    @app.post("/api/tune/save")
    def api_tune_save():
        return jsonify({"ok": True})

    @app.get("/api/mission/script")
    def api_script_get():
        return jsonify({"text": persisted["active_script"]})

    @app.post("/api/mission/script")
    def api_script_set():
        body = request.get_json(silent=True) or {}
        persisted["active_script"] = str(body.get("text") or "")
        return jsonify({"ok": True})

    # ---------------------------------------------------- calibration files
    @app.get("/api/calibrate/files")
    def api_calibrate_files_list():
        entries = []
        for path in sorted(calib_dir.glob("anafi_*.npz")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append({
                "name": path.name,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "serial": args.serial,
                "resolution": "720p",
                "rms_error": 0.5,
            })
        return jsonify({"files": entries})

    @app.get("/api/calibrate/files/<name>")
    def api_calibrate_files_get(name: str):
        path = calib_dir / name
        if not path.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        return send_file(str(path), mimetype="application/octet-stream")

    @app.post("/api/calibrate/files/<name>")
    def api_calibrate_files_put(name: str):
        blob = request.get_data(cache=False, as_text=False)
        if not blob:
            return jsonify({"ok": False, "error": "empty"}), 400
        (calib_dir / name).write_bytes(blob)
        return jsonify({"ok": True, "name": name, "size": len(blob)})

    @app.delete("/api/calibrate/files/<name>")
    def api_calibrate_files_delete(name: str):
        path = calib_dir / name
        if not path.exists():
            return jsonify({"ok": False, "error": "not found"}), 404
        path.unlink()
        return jsonify({"ok": True})

    # Trivial 1x1 mjpeg-y "video" so /calibrate iframes don't 404. Real
    # MJPEG isn't worth the complexity for the stub; the C2's overview
    # video toggle still wires <img src=…> correctly, the source is
    # just an empty white pixel served as a multipart stream of one
    # frame, which most browsers happily render.
    @app.get("/video.mjpg")
    def video_mjpg():
        from flask import Response
        # A 1x1 white JPEG.
        jpg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb0043"
            "00080606070605080707070909080a0c140d0c0b0b0c1912"
            "130f141d1a1f1e1d1a1c1c20242e2720222c231c1c283728"
            "2c30313434341f273a3d3832dc323434ffc0000b08000100"
            "01010110ffc4001f0000010501010101010100000000000000"
            "000102030405060708090a0bffc400b510000201030302040305"
            "0504040000017d01020300041105122131410613516107227114"
            "328191a1082342b1c11552d1f02433627282090a161718191a25"
            "262728292a3435363738393a434445464748494a535455565758"
            "595a636465666768696a737475767778797a838485868788898a"
            "92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9"
            "bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7"
            "e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbffd9"
        )

        def gen():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg + b"\r\n")
        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"[fake_fc] {args.host}:{args.port} serial={args.serial} "
          f"data_dir={data_dir}", flush=True)
    app.run(host=args.host, port=args.port,
            threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
