"""Per-drone flight-log writer. See FlightLogger.__doc__ for the format."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class FlightLogger:
    """Per-drone flight logger.

    Runs one background thread that polls /api/telemetry + /api/position +
    /proxy/aruco/state for every configured drone at FLIGHT_LOG_HZ. On the
    rising edge of ``flying`` (or airborne detected by height_cm > 30) a
    new JSONL file is opened; on the falling edge it's closed. Commands
    logged via ``log_command()`` are funnelled into the active files so
    every per-flight file is a complete, timestamped audit trail of:
      - telemetry (battery, attitude, velocity, ceiling state, ...)
      - fused arena position (x, y, z, dir, vel, stale)
      - visible ArUco markers (seen + reference lists)
      - every command sent (takeoff, land, rc, mission-start, pause, ...)

    Files land in ``<FLIGHT_LOG_DIR>/flight_<timestamp>_drone-<id>.jsonl``.
    """

    def __init__(self, drones: dict, session, log_dir: Path, hz: float = 5.0):
        self.drones  = drones
        self.session = session
        self.log_dir = log_dir
        self.period  = max(0.1, 1.0 / float(hz or 5.0))
        self._flights: dict[str, dict] = {}
        self._lock    = threading.Lock()
        self._running = False
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="flight-logger")
        t.start()
        print(f"[FLIGHT_LOG] started, dir={self.log_dir}, period={self.period:.2f}s")

    def stop(self):
        self._running = False
        with self._lock:
            for did in list(self._flights.keys()):
                self._close_unlocked(did, reason="shutdown")

    # Public — called from log_command
    def record_command(self, drone_id, event: str, payload: dict | None):
        """Append a command record to the active flight log(s). If
        drone_id is None we broadcast to every active flight (fleet-wide
        commands like PAUSE_ALL / LAND_ALL apply to all airborne drones)."""
        if not self._running:
            return
        did = str(drone_id) if drone_id else None
        with self._lock:
            targets = [did] if did and did in self._flights else list(self._flights.keys())
            for d in targets:
                flt = self._flights.get(d)
                if not flt:
                    continue
                self._write_unlocked(flt, {
                    "type":    "cmd",
                    "ts":      time.time(),
                    "drone_id": d,
                    "event":   event,
                    "payload": payload or {},
                })

    def list_files(self) -> list[dict]:
        """For /proxy/flight_logs — list all flight files with basic meta."""
        out = []
        try:
            for p in sorted(self.log_dir.glob("flight_*.jsonl"), reverse=True):
                try:
                    st = p.stat()
                    out.append({
                        "name": p.name,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def file_path(self, name: str) -> Path | None:
        """Resolve a filename to a path inside log_dir. Rejects path
        traversal attempts."""
        p = (self.log_dir / name).resolve()
        try:
            p.relative_to(self.log_dir)
        except ValueError:
            return None
        return p if p.exists() else None

    # ── internals ──
    def _loop(self):
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                print(f"[FLIGHT_LOG] tick error: {e}")
            dt = time.time() - t0
            time.sleep(max(0.02, self.period - dt))

    def _tick(self):
        for did, info in list(self.drones.items()):
            did  = str(did)
            base = (info or {}).get("base")
            if not base:
                continue
            # Skip HTTP polls entirely for drones whose WS is fully down —
            # otherwise each unreachable Pi burns 1.5 s per tick and the
            # whole fleet-logger thread falls behind at 5 Hz.
            cli = drone_ws.get(did) if 'drone_ws' in globals() else None
            if cli is not None:
                all_down = (not cli._ws_connected.get("telemetry") and
                            not cli._ws_connected.get("position") and
                            not cli._ws_connected.get("rc"))
                if all_down:
                    # Close any open flight for this drone (we can't log
                    # what we can't see) and move on.
                    with self._lock:
                        if did in self._flights:
                            self._close_unlocked(did, reason="unreachable")
                    continue
            try:
                tr  = self.session.get(f"{base.rstrip('/')}/api/telemetry", timeout=0.6)
                tel = tr.json() if tr.ok else {}
            except Exception:
                tel = {}
            try:
                pr  = self.session.get(f"{base.rstrip('/')}/api/position", timeout=0.3)
                pos = pr.json() if pr.ok else {}
            except Exception:
                pos = {}

            # Airborne detection: trust "flying" but fall back to height_cm
            flying   = bool(tel.get("flying"))
            height_cm = (tel.get("height_cm") or 0)
            airborne = flying or (height_cm and height_cm > 30)

            with self._lock:
                flt = self._flights.get(did)
                if airborne and flt is None:
                    self._open_unlocked(did, tel)
                    flt = self._flights.get(did)
                elif not airborne and flt is not None:
                    self._write_unlocked(flt, {
                        "type": "land", "ts": time.time(), "drone_id": did,
                        "telemetry": tel,
                    })
                    self._close_unlocked(did, reason="landed")
                    flt = None
                if flt is None:
                    continue
                # Active flight — emit a tick record.
                #
                # Visible markers: the ArUco observer exposes a `visible_ids`
                # list, but only when the observer thread is started (which
                # happens when ArUco Seek is armed or a mission is running).
                # During plain manual flight the observer is idle and that
                # list is empty. The per-drone position service on the Pi
                # runs its own detection pipeline and publishes `seen_markers`
                # in /api/position — use it as a fallback so the flight log
                # always reflects what the camera is actually seeing.
                vis_markers = []
                try:
                    obs = aruco_fleet.get(did)
                    if obs is not None:
                        st = obs.get_state()
                        vis_markers = st.get("visible_ids") or []
                except Exception:
                    pass
                if not vis_markers:
                    vis_markers = list(pos.get("seen_markers") or [])
                rec = {
                    "type":    "tick",
                    "ts":      time.time(),
                    "drone_id": did,
                    "telemetry": {k: tel.get(k) for k in (
                        "battery", "height_cm", "altitude_m", "flying",
                        "connected", "yaw", "pitch", "roll",
                        "vgx", "vgy", "vgz", "agx", "agy", "agz",
                        "ceiling_m", "ceiling_engaged", "ceiling_reason",
                        "state_age_s", "state_fresh",
                    ) if k in tel},
                    "position": pos.get("pos"),
                    "direction": pos.get("dir"),
                    "pos_vel": pos.get("vel"),
                    "pos_stale": pos.get("stale"),
                    "visible_markers": vis_markers,
                    "ref_markers":  pos.get("ref_markers") or [],
                    "seen_markers": pos.get("seen_markers") or [],
                }
                self._write_unlocked(flt, rec)

    def _open_unlocked(self, did: str, tel: dict):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        name = (self.drones.get(did, {}).get("name") or did).replace(" ", "_")
        stem = f"flight_{ts}_drone-{did}_{name}"
        path = self.log_dir / f"{stem}.jsonl"
        fh = path.open("w", encoding="utf-8", buffering=1)  # line-buffered
        # Kick off video recording on the FC with a matching basename so the
        # .mp4 and the .jsonl travel together. Annotated mode so the recording
        # has the detected-marker overlay for post-flight review.
        #
        # The FC's /api/video/record/start self-starts MJPEG if it's off, so
        # we intentionally do NOT ping /api/video/start separately here —
        # that endpoint always calls video_stop_all() + video_start_mjpeg()
        # and would restart the decoder on every takeoff (dropping frames
        # and freezing the position tracker for several seconds). No-op
        # when the FC is unreachable — video is nice-to-have, log is the
        # essential record.
        video_name = f"{stem}.mp4"
        video_started = False
        video_err = None
        base = (self.drones.get(did, {}) or {}).get("base")
        if base:
            try:
                r = self.session.post(
                    f"{base.rstrip('/')}/api/video/record/start",
                    json={"filename": video_name, "raw": False},
                    timeout=2.0,
                )
                if r.ok:
                    body = r.json() if r.content else {}
                    if body.get("ok") is True:
                        video_started = True
                    else:
                        video_err = body.get("error") or "record/start returned ok=false"
                else:
                    video_err = f"HTTP {r.status_code}"
            except Exception as e:
                video_err = str(e)
                print(f"[FLIGHT_LOG] video record start failed: {e}")

        # Fetch the FC's version info so the flight log header captures
        # BOTH the C2 git sha (self) and the FC code_version/git_sha.
        # Short 1 s timeout — if the FC is down we'd rather have a
        # no-fc-version log than no log at all.
        fc_version = {}
        if base:
            try:
                vr = self.session.get(f"{base.rstrip('/')}/api/version",
                                       timeout=1.0)
                if vr.ok:
                    vdata = vr.json() or {}
                    fc_version = {
                        "code_version": vdata.get("code_version"),
                        "git_revision": vdata.get("git_revision") or {},
                    }
            except Exception as ve:
                print(f"[FLIGHT_LOG] FC version fetch failed: {ve}")
        header = {
            "type": "takeoff", "ts": time.time(), "drone_id": did,
            "drone_name": self.drones.get(did, {}).get("name"),
            "git_revision": _GIT_REVISION,         # C2 side
            "fc_version":   fc_version,            # FC side
            "drone_base": base,
            "video_filename": video_name if video_started else None,
            "telemetry": tel,
        }
        fh.write(json.dumps(header, default=str) + "\n")
        self._flights[did] = {
            "fh": fh, "path": path, "opened_at": time.time(), "records": 1,
            "stem": stem, "video_name": video_name if video_started else None,
        }
        if video_started:
            print(f"[FLIGHT_LOG] takeoff → {path}  + video → {video_name}")
        else:
            print(f"[FLIGHT_LOG] takeoff → {path}  (no video: {video_err or 'FC unreachable'})")

    def _close_unlocked(self, did: str, reason: str = "landed"):
        flt = self._flights.pop(did, None)
        if flt is None:
            return
        # Stop the matching video recording on the Pi (no-op if not started).
        base = (self.drones.get(did, {}) or {}).get("base")
        video_frames = None
        if base and flt.get("video_name"):
            try:
                r = self.session.post(
                    f"{base.rstrip('/')}/api/video/record/stop",
                    json={}, timeout=1.5,
                )
                if r.ok:
                    j = r.json()
                    video_frames = j.get("frames")
            except Exception as e:
                print(f"[FLIGHT_LOG] video record stop failed: {e}")
        try:
            dur = time.time() - flt["opened_at"]
            flt["fh"].write(json.dumps({
                "type": "close", "ts": time.time(), "reason": reason,
                "duration_s": round(dur, 2), "records": flt["records"],
                "video_filename": flt.get("video_name"),
                "video_frames": video_frames,
            }) + "\n")
            flt["fh"].close()
        except Exception:
            pass
        print(f"[FLIGHT_LOG] closed {flt['path']} "
              f"(reason={reason}, records={flt['records']}, "
              f"duration={time.time() - flt['opened_at']:.1f}s"
              + (f", video={video_frames} frames" if video_frames is not None else "")
              + ")")

    def _write_unlocked(self, flt: dict, rec: dict):
        try:
            flt["fh"].write(json.dumps(rec, default=str) + "\n")
            flt["records"] += 1
        except Exception as e:
            print(f"[FLIGHT_LOG] write error: {e}")

