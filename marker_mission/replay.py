"""
Flight replay -- drive a recorded flight through the live web UI.

A :class:`FlightReplay` reads ``<flight_dir>/flight_log.csv`` and
``<flight_dir>/annotated.mp4`` and walks them in sync, populating a
:class:`MissionState` and a :class:`LatestFrame` so the existing UI
templates render the playback. Supports play / pause / seek / speed via
the methods on the instance.

Designed to coexist with a live mission: the live ``UiServer`` keeps
its own ``state`` / ``frame`` and never sees these.
"""

from __future__ import annotations

import bisect
import csv
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

# Force OpenCV's ffmpeg backend into single-threaded decoding. The
# multi-threaded H.264 decoder asserts on libavcodec/pthread_frame.c:174
# when a Python-driven loop interleaves cap.set(POS_MSEC) and cap.read()
# faster than its async-lock invariants tolerate. Single-threaded mode
# is plenty fast for our ~25 fps inputs and avoids the abort.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import cv2  # noqa: E402  -- after the env-var override above

from .controller import MissionState, Phase
from .drone_api import TelemetrySnapshot
from . import mission_script as _ms


class FlightReplay:
    """Reads one flight's recorded artefacts and exposes a paused
    playback driven by a daemon thread.

    State and frame are public attributes so the UI can mount its
    existing templates against them without any changes.
    """

    def __init__(self, flight_dir: Path):
        from .ui import LatestFrame  # local import: avoid circular
        self.dir = Path(flight_dir)
        self.flight_id = self.dir.name
        # Recover the drone serial from mission_meta.json or, failing
        # that, the flight_dir name suffix (make_flight_dir produces
        # ``<wall_ts>_<serial>`` so the serial is everything after the
        # second underscore separator).
        self.serial: Optional[str] = self._lookup_serial()

        # Load the per-tick CSV. Anything that fails to parse is fatal --
        # the operator should pick a different flight.
        rows: List[dict] = []
        with (self.dir / "flight_log.csv").open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if not rows:
            raise ValueError(f"flight_log.csv in {flight_dir} is empty")
        self.rows = rows
        t0 = float(rows[0]["monotonic"])
        self.timeline: List[float] = [float(r["monotonic"]) - t0
                                       for r in rows]
        self.duration_s = self.timeline[-1]

        # Pre-compute, per row, the log-relative time at which the
        # CURRENT phase started, and the log-relative time of the most
        # recent marker sighting. These let us populate phase_age_s /
        # marker_seen_age_s correctly in replay without using wall time.
        self.phase_start_t: List[float] = [0.0] * len(rows)
        self.marker_seen_t: List[Optional[float]] = [None] * len(rows)
        # Per-row index of the most recent row (<= i) that carried a
        # full world-position estimate. -1 until the first sighting.
        # Used by ``_apply_row`` so a row with empty world_x still
        # shows the last-known fix, mirroring the live behaviour
        # where state.world_position_m persists across marker loss.
        # ``world_pos_seen_t`` is the matching log-relative time for
        # the snapshot's world_position_age_s.
        self.world_pos_idx: List[int] = [-1] * len(rows)
        self.world_pos_seen_t: List[Optional[float]] = [None] * len(rows)
        last_phase = rows[0].get("phase", "")
        cur_phase_start = self.timeline[0]
        cur_marker_seen: Optional[float] = None
        cur_world_idx = -1
        cur_world_t: Optional[float] = None
        for i, r in enumerate(rows):
            ph = r.get("phase", "")
            if ph != last_phase:
                cur_phase_start = self.timeline[i]
                last_phase = ph
            if r.get("marker_seen") == "1":
                cur_marker_seen = self.timeline[i]
            if (r.get("world_x") or "").strip():
                cur_world_idx = i
                cur_world_t = self.timeline[i]
            self.phase_start_t[i] = cur_phase_start
            self.marker_seen_t[i] = cur_marker_seen
            self.world_pos_idx[i] = cur_world_idx
            self.world_pos_seen_t[i] = cur_world_t

        # Mission script. Saved by mission.py at TAKEOFF as the
        # canonicalised step list -- already has all defaults filled
        # in, so parse() never touches its defaults dict. Old flights
        # that pre-date the mission_script feature have no file, so
        # we run replay without a script panel.
        self.script_steps: List = []
        sp = self.dir / "mission_script.txt"
        if sp.is_file():
            try:
                self.script_steps = _ms.parse(sp.read_text(), {})
            except Exception as e:
                print(f"[replay] mission_script.txt parse failed: {e}")
                self.script_steps = []

        # Per-row mission_step_idx. Prefer the recorded column for
        # exactness; fall back to a phase-walk for flights recorded
        # before the column existed.
        self.step_idx_per_row: List[int] = self._build_step_idx_per_row(rows)

        # Per-flight arena. Recent flights save the active arena layout
        # at TAKEOFF as ``arena_config.json`` next to the CSV, pinning
        # the geometry that was used. Older flights don't have this
        # file -- fall back to whatever arena is currently active so
        # the position view still draws (with whatever marker layout
        # the operator has now). None disables the yaw arrow only;
        # the position dot still works either way.
        self.arena = None
        try:
            from .arena import ArenaConfig, load_priority_arena
            arena_path = self.dir / "arena_config.json"
            if arena_path.is_file():
                self.arena = ArenaConfig.load(arena_path)
            else:
                self.arena = load_priority_arena()
        except Exception as e:
            print(f"[replay] arena load failed: {e}; "
                  f"position view will skip the yaw arrow")
            self.arena = None

        # Annotated.mp4 already has the HUD baked in -- prefer it.
        # Fall back to raw.mp4 if annotated is missing for some reason.
        # NOTE: don't open the cv2.VideoCapture here. libavcodec's
        # threaded H.264 decoder asserts (pthread_frame.c:174) when a
        # capture opened in one thread is read from another at our
        # cadence. Defer opening to inside the worker thread.
        for name in ("annotated.mp4", "raw.mp4"):
            p = self.dir / name
            if p.exists() and p.stat().st_size > 0:
                self.video_path: Optional[Path] = p
                break
        else:
            self.video_path = None
        self.cap = None  # opened lazily inside _run

        # Public state for the UI to read.
        self.state = MissionState()
        # Install the recorded script so snapshot()'s mission_script
        # field gets populated. mission_step_idx is set per-tick in
        # _apply_row from the precomputed array above.
        with self.state.lock:
            self.state.mission_script = self.script_steps
        self.frame = LatestFrame()
        self._last_pos_ms = -1.0

        # Playback controls.
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._paused = True            # always start paused
        self._speed = 1.0
        self._playhead_idx = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"replay-{self.flight_id}")
        self._thread.start()

    # ------------------------------------------------------------ controls
    def play(self) -> None:
        with self._lock:
            self._paused = False
        self._wakeup.set()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self._wakeup.set()

    def toggle_play(self) -> None:
        with self._lock:
            self._paused = not self._paused
        self._wakeup.set()

    def seek(self, t_seconds: float) -> None:
        """Move the playhead to absolute time ``t_seconds`` (clamped)."""
        with self._lock:
            t = max(0.0, min(float(t_seconds), self.duration_s))
            new_idx = bisect.bisect_left(self.timeline, t)
            if new_idx >= len(self.timeline):
                new_idx = len(self.timeline) - 1
            self._playhead_idx = new_idx
        self._wakeup.set()

    def set_speed(self, rate: float) -> None:
        with self._lock:
            self._speed = max(0.1, min(10.0, float(rate)))
        self._wakeup.set()

    def timeline_arrays(self) -> dict:
        """Return parallel arrays for the full flight timeline. Used by
        the replay UI to pre-fill the charts in one shot. Values that
        are missing in a row come back as None so the chart drawing
        function can break the line cleanly."""
        def _col(key: str) -> List[Optional[float]]:
            return [self._fnum(r, key) for r in self.rows]
        return {
            "t": list(self.timeline),
            "d": _col("distance_m"),
            "y": _col("yaw_to_marker_deg"),
            "h": _col("relative_heading_deg"),
            "drone_yaw": _col("tel_yaw"),
            "battery":   _col("tel_battery"),
            "height":    _col("tel_height_cm"),
            "rc_lr":  _col("rc_lr"),
            "rc_fb":  _col("rc_fb"),
            "rc_ud":  _col("rc_ud"),
            "rc_yaw": _col("rc_yaw"),
            "duration_s": self.duration_s,
        }

    def snapshot(self) -> dict:
        """Return MissionState.snapshot() with replay-correct timing.
        phase_age_s and marker_seen_age_s are computed from log
        timestamps relative to the playhead -- the standard snapshot
        uses wall time, which is meaningless during replay (we set
        ``phase_started_at = time.monotonic()`` on every row apply, so
        the standard value is always near zero)."""
        snap = self.state.snapshot()
        with self._lock:
            idx = self._playhead_idx
        if 0 <= idx < len(self.rows):
            playhead_t = self.timeline[idx]
            snap["phase_age_s"] = playhead_t - self.phase_start_t[idx]
            seen_t = self.marker_seen_t[idx]
            snap["marker_seen_age_s"] = (
                (playhead_t - seen_t) if seen_t is not None else None)
            wp_seen_t = self.world_pos_seen_t[idx]
            snap["world_position_age_s"] = (
                (playhead_t - wp_seen_t) if wp_seen_t is not None else None)
            snap["uptime_s"] = playhead_t
        return snap

    @property
    def status(self) -> dict:
        with self._lock:
            idx = self._playhead_idx
            return {
                "flight_id": self.flight_id,
                "duration_s": round(self.duration_s, 3),
                "playhead_s": round(self.timeline[idx], 3)
                              if idx < len(self.timeline)
                              else round(self.duration_s, 3),
                "paused": self._paused,
                "speed": self._speed,
                "row_count": len(self.rows),
                "video_path": str(self.video_path)
                              if self.video_path else None,
            }

    def close(self) -> None:
        self._stop.set()
        self._wakeup.set()
        # Don't release self.cap here -- it was opened in the worker
        # thread and should be released there. The worker exits its
        # loop on _stop and releases its own cap.

    # ---------------------------------------------------- step-idx mapping
    def _build_step_idx_per_row(self, rows: List[dict]) -> List[int]:
        """Per CSV row, the script-step index that was active when the
        row was logged.

        Two paths:

        * ``mission_step_idx`` column present (newer flights) -- read
          it directly. The recorder writes -1 before the first step.
        * Column absent (older flights) -- walk the phase column and
          advance through ``self.script_steps`` whenever the phase
          enters a kind-specific entry phase. Each step kind has
          exactly one entry phase under the live controller's flow:

          * TAKEOFF -> ``takeoff``
          * APPROACH -> ``search`` (sub-phases never start the step)
          * HOOVER -> ``hold`` (after APPROACH) or ``idle`` (otherwise)
          * HEIGHT -> ``height``
          * DANCE -> ``dance``
          * LAND -> ``land``
        """
        n = len(rows)
        out = [-1] * n
        # Path 1: explicit column.
        col_present = any("mission_step_idx" in r and r["mission_step_idx"]
                          for r in rows)
        if col_present:
            cur = -1
            for i, r in enumerate(rows):
                v = r.get("mission_step_idx", "")
                if v != "":
                    try:
                        cur = int(v)
                    except ValueError:
                        pass
                out[i] = cur
            return out
        # Path 2: phase walk.
        if not self.script_steps:
            return out
        ENTRY = {
            "TAKEOFF":  {"takeoff"},
            "APPROACH": {"search"},
            "HOOVER":   {"hold", "idle"},
            "HEIGHT":   {"height"},
            "DANCE":    {"dance"},
            "LAND":     {"land"},
        }
        idx = -1
        prev_phase = None
        for i, r in enumerate(rows):
            phase = r.get("phase", "")
            if phase != prev_phase and idx + 1 < len(self.script_steps):
                next_kind = self.script_steps[idx + 1].kind
                if phase in ENTRY.get(next_kind, set()):
                    idx += 1
            out[i] = idx
            prev_phase = phase
        return out

    # ---------------------------------------------------- per-row dispatch
    def _lookup_serial(self) -> Optional[str]:
        # Prefer the persisted mission metadata if present.
        meta_p = self.dir / "mission_meta.json"
        if meta_p.exists():
            try:
                import json as _json
                blob = _json.loads(meta_p.read_text())
                outcome = blob.get("outcome") or blob
                s = outcome.get("serial") if isinstance(outcome, dict) else None
                if s:
                    return str(s)
            except Exception:
                pass
        # Fall back to the dir-name suffix. make_flight_dir uses
        # YYYY-MM-DD_HH-MM-SS_<serial>; serial is whatever follows the
        # second underscore. Avoid returning the literal "unknown"
        # placeholder used when no serial was available at flight time.
        parts = self.flight_id.split("_", 2)
        if len(parts) >= 3 and parts[2] and parts[2] != "unknown":
            return parts[2]
        return None

    @staticmethod
    def _fnum(row: dict, key: str) -> Optional[float]:
        v = row.get(key, "")
        if v in ("", "nan", "NaN", None):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _ensure_cap(self) -> None:
        """Open the VideoCapture in the calling thread the first time
        we need it. Used so the cap is owned by the playback worker
        rather than the constructor's thread."""
        if self.cap is None and self.video_path is not None:
            self.cap = cv2.VideoCapture(str(self.video_path))

    def _row_at_video_time(self, video_t_s: float) -> int:
        """Find the CSV row whose log time is closest to ``video_t_s``.
        Used to align the state side-panel with the actual video frame
        being displayed -- without this, state shows row[idx] values
        but the video may be one or two CSV ticks ahead, causing the
        numeric values to disagree with the on-frame annotation."""
        i = bisect.bisect_left(self.timeline, video_t_s)
        if i >= len(self.timeline):
            return len(self.timeline) - 1
        if i > 0 and (abs(self.timeline[i-1] - video_t_s)
                       < abs(self.timeline[i] - video_t_s)):
            return i - 1
        return i

    def _apply_row(self, idx: int) -> None:
        # Advance the video first so we know the ACTUAL displayed frame
        # time, then sync the state to whichever CSV row best matches
        # that time -- otherwise the side panel shows row[idx] values
        # while the video shows a slightly different moment.
        self._advance_video_to(self.timeline[idx])
        if self.cap is not None and self._last_pos_ms >= 0:
            idx = self._row_at_video_time(self._last_pos_ms / 1000.0)
        row = self.rows[idx]
        # Rebuild a TelemetrySnapshot from the recorded tel_* columns.
        tel_raw: dict = {}
        for src in ("battery", "yaw", "pitch", "roll", "height_cm",
                    "flight_time_s", "vgx", "vgy", "vgz"):
            v = self._fnum(row, f"tel_{src}")
            if v is not None:
                tel_raw[src] = v
        tel_raw["flying"] = (row.get("tel_flying") == "1")
        tel_raw["connected"] = (row.get("tel_connected") == "1")
        if self.serial:
            tel_raw["serial_number"] = self.serial
        tel_snap = TelemetrySnapshot(raw=tel_raw, received_at=time.time())

        # Pose tuple (or None if marker wasn't seen this row).
        d = self._fnum(row, "distance_m")
        y = self._fnum(row, "yaw_to_marker_deg")
        h = self._fnum(row, "relative_heading_deg")
        smoothed = (d, y, h) if d is not None else None

        try:
            phase = Phase(row.get("phase", "init"))
        except ValueError:
            phase = Phase.INIT

        # Active script step index for this row -- precomputed at
        # __init__. Out-of-range indices (e.g., past last step) clamp
        # to len(script_steps) so the UI shows "no step active".
        s_idx = self.step_idx_per_row[idx] if self.step_idx_per_row else -1
        if 0 <= s_idx < len(self.script_steps):
            cur_kind = self.script_steps[s_idx].kind
        else:
            cur_kind = None

        # Arena world-position estimate logged by the recorder. world_x/y/z
        # may be empty strings when no reference marker was visible at
        # that tick. We mirror the live behaviour and carry the most
        # recent fix forward (precomputed in ``world_pos_idx``); the
        # snapshot exposes the age via ``world_position_age_s`` so the
        # UI can colour-code staleness.
        src_idx = self.world_pos_idx[idx] if idx < len(self.world_pos_idx) else -1
        if src_idx >= 0:
            src_row = self.rows[src_idx]
            wx = self._fnum(src_row, "world_x")
            wy = self._fnum(src_row, "world_y")
            wz = self._fnum(src_row, "world_z")
        else:
            src_row = row
            wx = wy = wz = None
        if wx is not None and wy is not None and wz is not None:
            world_pos = (wx, wy, wz)
        else:
            world_pos = None
        wnu = self._fnum(src_row, "world_n_used")
        world_n_used_int = int(wnu) if wnu is not None else 0

        # Pose-method tags. CSV stores arena methods as "id:method|id:method".
        # target_pose_method describes the active controller marker
        # for THIS row (current marker visibility), the arena
        # used-markers / per-marker votes describe whichever row
        # produced the carried-forward world_pos.
        target_pose_method = (row.get("target_pose_method") or "").strip()
        arena_methods_blob = (src_row.get("arena_pose_methods") or "").strip()
        used_marker_ids: List[int] = []
        used_methods: List[str] = []
        if arena_methods_blob:
            for chunk in arena_methods_blob.split("|"):
                if ":" not in chunk:
                    continue
                mid_s, meth = chunk.split(":", 1)
                try:
                    used_marker_ids.append(int(mid_s))
                    used_methods.append(meth)
                except ValueError:
                    pass
        # Per-marker world-position votes ("id:x,y,z|id:x,y,z").
        per_marker_blob = (src_row.get("arena_per_marker_world") or "").strip()
        per_marker_world: List[tuple] = []
        if per_marker_blob:
            for chunk in per_marker_blob.split("|"):
                if ":" not in chunk:
                    continue
                _, xyz = chunk.split(":", 1)
                parts = xyz.split(",")
                if len(parts) != 3:
                    continue
                try:
                    per_marker_world.append(
                        (float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass

        with self.state.lock:
            self.state.phase = phase
            self.state.last_telemetry = tel_snap
            self.state.smoothed = smoothed
            self.state.target_distance_m = (
                self._fnum(row, "target_distance_m") or 1.0)
            self.state.target_relative_heading_deg = (
                self._fnum(row, "target_relative_heading_deg") or 0.0)
            self.state.mission_step_idx = s_idx
            self.state.current_step_kind = cur_kind
            self.state.world_position_m = world_pos
            # If the new arena_pose_methods column is present we know
            # which markers contributed; otherwise (older flights) fall
            # back to a placeholder list of the right length so the
            # status tile still shows the count.
            if used_marker_ids:
                self.state.world_position_used_markers = used_marker_ids
                self.state.world_position_pose_methods = used_methods
            else:
                self.state.world_position_used_markers = (
                    [-1] * world_n_used_int if world_n_used_int > 0 else [])
                self.state.world_position_pose_methods = []
            self.state.world_position_per_marker = per_marker_world
            self.state.target_pose_method = target_pose_method
            self.state.last_rc = (
                int(self._fnum(row, "rc_lr") or 0),
                int(self._fnum(row, "rc_fb") or 0),
                int(self._fnum(row, "rc_ud") or 0),
                int(self._fnum(row, "rc_yaw") or 0),
            )
            seen = (row.get("marker_seen") == "1")
            if seen:
                self.state.last_marker_seen_at = time.monotonic()
            self.state.note = (
                f"REPLAY {self.flight_id} -- t={self.timeline[idx]:.1f}s "
                f"/ {self.duration_s:.1f}s"
            )
            self.state.phase_started_at = time.monotonic()

            # Recompute arena_yaw_deg from CSV data so the position
            # view's yaw arrow renders during replay. We need:
            #   * the drone world position (just set above)
            #   * the active marker's id and arena position
            #   * the camera-frame yaw to that marker (from
            #     ``yaw_to_marker_deg`` column)
            # When any input is missing we leave arena_yaw_deg at None
            # (the position dot still draws; only the arrow is
            # skipped).
            self.state.arena_yaw_deg = None
            self.state.arena_yaw_updated_at = 0.0
            if (world_pos is not None
                    and seen
                    and self.arena is not None
                    and smoothed is not None):
                try:
                    mid = int(row.get("marker_id") or "")
                except (TypeError, ValueError):
                    mid = None
                marker = (self.arena.markers.get(mid)
                          if mid is not None else None)
                if marker is not None:
                    import math as _math
                    yaw_to_marker = float(smoothed[1])  # smoothed = (d, yaw, hdg)
                    dxm = float(marker.position_m[0]) - float(world_pos[0])
                    dym = float(marker.position_m[1]) - float(world_pos[1])
                    if dxm != 0.0 or dym != 0.0:
                        bearing = _math.degrees(_math.atan2(dxm, dym))
                        self.state.arena_yaw_deg = (
                            ((bearing - yaw_to_marker + 540.0) % 360.0)
                            - 180.0)
                        self.state.arena_yaw_updated_at = time.monotonic()

    def _advance_video_to(self, target_t_s: float) -> None:
        """Advance the video to ``target_t_s`` (relative to flight
        start). The CSV is logged at ~5 Hz but the video is at ~25 fps,
        so a single cap.read() per row would only consume one of every
        five frames -- playback would crawl at 1/5 speed. We read
        forward in the cap until _last_pos_ms catches up to
        target_t_ms, then publish the last frame we got. Sequential
        reads are far cheaper than explicit seeks; we only seek for
        big jumps (initial load, backward seek, forward leap larger
        than ~1.5 s)."""
        if self.cap is None:
            return
        target_t_ms = target_t_s * 1000.0
        big_jump = (self._last_pos_ms < 0
                    or target_t_ms < self._last_pos_ms - 200.0
                    or target_t_ms > self._last_pos_ms + 1500.0)
        if big_jump:
            try:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, target_t_ms)
            except Exception:
                return
            try:
                ok, frame = self.cap.read()
            except Exception:
                return
            if ok and frame is not None:
                self._last_pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.frame.set(frame)
            return
        # Small forward step -- chew through frames until caught up.
        last_frame = None
        for _ in range(64):  # safety cap; 64 frames ~ 2.5 s at 25fps
            if self._last_pos_ms >= target_t_ms:
                break
            try:
                ok, frame = self.cap.read()
            except Exception:
                break
            if not ok or frame is None:
                break
            last_frame = frame
            self._last_pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if last_frame is not None:
            self.frame.set(last_frame)

    # ---------------------------------------------------- playback loop
    def _wait(self, timeout: float) -> None:
        """Sleep up to ``timeout`` seconds; wake immediately on a state
        change or stop. Caller clears the event after waking."""
        self._wakeup.wait(timeout=timeout)
        self._wakeup.clear()

    def _run(self) -> None:
        # Open the VideoCapture in this thread to keep libavcodec's
        # internal lock invariants happy (see __init__ note).
        self._ensure_cap()
        # Show the first frame so the UI isn't blank while paused.
        if self.rows:
            self._apply_row(0)
            last_dispatched = 0
        else:
            return

        # Wall-clock anchor for advancing through the timeline.
        base_wall = time.monotonic()
        base_t = self.timeline[0]

        while not self._stop.is_set():
            with self._lock:
                idx = self._playhead_idx
                paused = self._paused
                speed = self._speed

            # Seek (or initial entry) -- redraw + reset the wall anchor.
            if idx != last_dispatched:
                self._apply_row(idx)
                last_dispatched = idx
                base_wall = time.monotonic()
                base_t = self.timeline[idx]

            # End of timeline -- auto-pause and wait.
            if idx >= len(self.rows) - 1:
                if not paused:
                    with self._lock:
                        self._paused = True
                self._wait(0.5)
                continue

            if paused:
                # Keep the anchor pinned to the current row so unpause
                # resumes at exactly the right wall time.
                base_wall = time.monotonic()
                base_t = self.timeline[idx]
                self._wait(0.5)
                continue

            # Advance to the new row whose timeline timestamp is
            # <= base_t + (wall_elapsed * speed).
            now = time.monotonic()
            target_t = base_t + (now - base_wall) * speed
            new_idx = idx
            while (new_idx + 1 < len(self.rows)
                   and self.timeline[new_idx + 1] <= target_t):
                new_idx += 1

            if new_idx > idx:
                with self._lock:
                    if self._playhead_idx == idx:
                        self._playhead_idx = new_idx
                # Apply outside the lock so cv2.VideoCapture I/O doesn't
                # serialize against the controls.
                self._apply_row(new_idx)
                last_dispatched = new_idx
            else:
                # Sleep until the next row's relative timestamp.
                next_dt_real = (self.timeline[idx + 1] - target_t) / max(speed, 0.001)
                self._wait(max(0.005, min(next_dt_real, 0.1)))

        # Loop exited (close() was called). Release the cap from the
        # same thread that opened it.
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
