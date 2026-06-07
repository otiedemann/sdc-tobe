"""
Per-flight artefacts.

For each flight we create one directory at::

    <flights_dir>/<YYYY-mm-dd_HH-MM-SS>_<serial>/

and write:

* ``raw.mp4``         -- the unmodified MJPEG frames as the drone delivered them
* ``annotated.mp4``   -- the same frames with the marker/HUD overlay
* ``flight_log.csv``  -- one row per control tick: pose, RC, telemetry, phase
* ``mission_meta.json`` -- config, calibration source, mission outcome

Both video files use the H.264 / mp4 container via cv2.VideoWriter. The
recorder runs in its own thread so writing files doesn't slow the
control loop.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np

from .aruco_detector import MarkerPose
from .calibration_store import Calibration
from .config import MissionConfig
from .controller import MissionState
from .drone_api import TelemetrySnapshot


_CSV_FIELDS = [
    "wall_time", "monotonic", "phase",
    "distance_m", "yaw_to_marker_deg", "relative_heading_deg",
    "marker_normal_bearing_deg", "marker_tilt_deg", "marker_inplane_rot_deg",
    "marker_id", "marker_seen",
    "target_distance_m", "target_relative_heading_deg",
    "mission_step_idx",
    "world_x", "world_y", "world_z", "world_n_used", "world_position_age_s",
    "world_position_confidence", "arena_yaw_deg", "arena_yaw_age_s",
    # Position Kalman filter velocity output (arena frame, m/s) when
    # cfg.enable_position_kalman is True. Empty when KF is disabled
    # or hasn't initialised yet. world_x/y/z above already carries
    # the KF position when the filter is on (no separate raw column
    # in the same flight; A/B by toggling the switch between flights).
    "world_vx_kf", "world_vy_kf", "world_vz_kf",
    "target_pose_method", "arena_pose_methods", "arena_per_marker_world",
    "rc_lr", "rc_fb", "rc_ud", "rc_yaw",
    "tel_battery", "tel_yaw", "tel_pitch", "tel_roll",
    "tel_height_cm", "tel_flight_time_s",
    "tel_vgx", "tel_vgy", "tel_vgz",
    "tel_flying", "tel_connected",
    # Two-stage stall flags. ``telemetry_stalled`` = "diagnostic"
    # level (age > detect_threshold ~ 200 ms), marks the CSV without
    # changing RC -- catches short blips for offline analysis. The
    # higher-threshold ``telemetry_rc_gated`` = "controller actually
    # zeroed RC" because the link was stalled long enough (age >
    # gate_threshold ~ 500 ms) to make pushing PD output dangerous.
    # Grep for telemetry_rc_gated=1 to find moments where the
    # recorded rc=(0,0,0,0) reflects a safety override, not PD intent.
    "telemetry_stalled", "telemetry_rc_gated",
    # Host resource indicators sampled at ~1 Hz by cpu_monitor.
    # ``host_cpu_pct`` is system-wide (all cores busy ratio),
    # ``host_proc_cpu_pct`` is THIS process expressed as % of one
    # core (>100 = using more than one core's worth). Lets us
    # correlate telemetry stalls / RC chatter with CPU pressure
    # offline: a stall coincident with cpu spike points at the host,
    # a stall with idle cpu points at the drone link.
    "host_cpu_pct", "host_proc_cpu_pct", "host_load_1m", "host_mem_pct",
    # Note + abort reason carry whatever the controller's state
    # machine has surfaced this tick. The killswitch / Stop button
    # write a non-empty abort_reason synchronously into state so
    # the precise tick where the emergency land was triggered is
    # always captured -- the operator can grep the CSV for the
    # row whose abort_reason transitions from "" to non-empty.
    "note", "abort_reason",
    # Pipe-joined list of every marker id the detector reported
    # this tick (e.g. "1|3|7"). Distinct from
    # ``arena_pose_methods`` (which only lists markers that
    # contributed to the position vote): a marker can be visible
    # but excluded from the average by the magnetometer / OOB
    # filters. Replay reads this column to ring all visible
    # markers in the position view, not just contributors.
    "visible_marker_ids",
]


class FlightRecorder:
    """Background thread that writes raw and annotated video plus a CSV log."""

    def __init__(self, flight_dir: Path, fps: int = 25):
        self.dir = Path(flight_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._ann_writer: Optional[cv2.VideoWriter] = None
        self._raw_size: Optional[tuple[int, int]] = None
        self._ann_size: Optional[tuple[int, int]] = None
        self._csv_path = self.dir / "flight_log.csv"
        self._csv_fp = self._csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=_CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_lock = threading.Lock()

        self._video_q: "Queue[tuple[str, np.ndarray]]" = Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="recorder")
        self._thread.start()
        self._frames_dropped = 0
        # Per-kind frame arrival tracking. ``self.fps`` is just the
        # value cv2.VideoWriter is opened with -- it ends up in the
        # mp4 header but doesn't reflect what the stream actually
        # delivered. We measure the real arrival rate here so
        # ``_reencode_for_sharing`` can stamp the right fps onto the
        # re-encoded H.264 output (without it, a 15 fps stream
        # encoded at header=25 plays back 1.67x too fast).
        self._first_frame_t: dict[str, float] = {}
        self._last_frame_t: dict[str, float] = {}
        self._frame_count: dict[str, int] = {"raw": 0, "ann": 0}

    # ------------------------------------------------------------------ life
    def stop(self, meta: Optional[dict] = None) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        with self._csv_lock:
            try:
                self._csv_fp.close()
            except Exception:
                pass
        if self._raw_writer:
            try: self._raw_writer.release()
            except Exception: pass
        if self._ann_writer:
            try: self._ann_writer.release()
            except Exception: pass
        # meta
        if meta is not None:
            (self.dir / "mission_meta.json").write_text(json.dumps(meta, indent=2,
                                                                   default=str))
        # Effective per-stream fps: how fast frames actually arrived,
        # not the cv2.VideoWriter header value. Used by the re-encode
        # pass to stamp correct timing onto the H.264 output.
        # Mapping is "stem in the on-disk file" -> measured fps.
        effective_fps: dict[str, float] = {}
        for kind, stem in (("raw", "raw"), ("ann", "annotated")):
            n = self._frame_count.get(kind, 0)
            t0 = self._first_frame_t.get(kind)
            t1 = self._last_frame_t.get(kind)
            if n >= 2 and t0 is not None and t1 is not None and t1 > t0:
                # n-1 intervals between n frames -> arrival rate
                effective_fps[stem] = (n - 1) / (t1 - t0)
        # OpenCV's mp4v output isn't accepted by WhatsApp (and several
        # other ingest pipelines); re-encode to H.264 + silent AAC so the
        # files are shareable. Run this in the background -- the next
        # mission can roll a fresh flight dir and start while the old
        # one is still re-encoding. The daemon thread is fire-and-forget;
        # the encoded file just appears when it's done.
        threading.Thread(target=self._reencode_for_sharing,
                         args=(effective_fps,),
                         name=f"reencode-{self.dir.name}",
                         daemon=True).start()

    # --------------------------------------------------------- post-process
    def _reencode_for_sharing(self,
                              effective_fps: Optional[dict] = None) -> None:
        """Re-encode raw and annotated videos to H.264 + dummy AAC track.

        OpenCV writes mp4v (MPEG-4 Part 2) which most modern apps refuse
        to ingest. We use ffmpeg with libx264 and a generated silent
        audio stream so the resulting file plays everywhere and can be
        sent over WhatsApp/Telegram/etc.

        ``effective_fps`` is an optional ``{stem: fps}`` map (e.g.
        ``{"raw": 14.93, "annotated": 14.93}``) measuring the real
        arrival rate of each stream. When supplied, ``-r FPS -i src``
        overrides the input file's claimed fps so the H.264 output is
        timed to match wall-clock reality. Without it (or when the
        measurement is implausible) we fall back to the source file's
        header fps -- which is correct only when the live stream
        actually delivered at ``cfg.record_fps``.

        Originals are replaced atomically on success and deleted if the
        OpenCV writer fell back to .avi. On failure (ffmpeg missing or
        encode error) the originals are left untouched so the operator
        still has the raw artefacts.
        """
        effective_fps = effective_fps or {}
        if shutil.which("ffmpeg") is None:
            print("[rec] ffmpeg not found -- keeping mp4v originals "
                  "(install ffmpeg for shareable H.264 output)")
            return
        encoder = _pick_h264_encoder()
        if encoder is None:
            print("[rec] ffmpeg has no H.264 encoder available -- "
                  "keeping mp4v originals")
            return
        # Only re-encode the annotated stream. raw.mp4 is the
        # untouched detector input and stays exactly as the OpenCV
        # writer left it: keeps the bytes lossless against a
        # possible re-encode pass, and avoids the 5-10 s ffmpeg cost
        # per flight on the only artefact that actually needs to be
        # share-friendly. Operators who want H.264 raw can run the
        # offline reprocessor or ffmpeg manually.
        for stem in ("annotated",):
            # The OpenCV writer falls back to .avi/MJPG if mp4v is
            # unavailable on the host -- handle either input.
            src: Optional[Path] = None
            for ext in (".mp4", ".avi"):
                p = self.dir / f"{stem}{ext}"
                if p.exists() and p.stat().st_size > 0:
                    src = p
                    break
            if src is None:
                continue
            final = self.dir / f"{stem}.mp4"
            tmp = self.dir / f"{stem}.h264.tmp.mp4"
            # Different H.264 encoders take different quality knobs.
            # Common to all: explicit BT.709 color metadata + Main
            # profile + yuv420p so the bitstream renders in VLC etc.
            # (libopenh264 defaults to constrained-baseline + missing
            # color tags; some VLC builds render that as a black frame.)
            common = ["-pix_fmt", "yuv420p",
                      "-color_primaries", "bt709",
                      "-color_trc", "bt709",
                      "-colorspace", "bt709",
                      "-profile:v", "main"]
            if encoder == "libx264":
                video_args = ["-c:v", "libx264", "-preset", "veryfast",
                              "-crf", "23", *common]
            else:
                # libopenh264 / nvenc / vaapi / qsv: bitrate-controlled.
                video_args = ["-c:v", encoder, "-b:v", "4M", *common]
            # Override the input's claimed fps when we measured a
            # plausible arrival rate. ``-r N`` BEFORE ``-i`` makes
            # ffmpeg synthesise per-frame timestamps assuming
            # constant rate N -- the output ends up correctly
            # timed regardless of what the source mp4 header
            # claimed. Clamp to [1, 60] so a freak measurement
            # (e.g. one stale frame at startup) doesn't give us
            # garbage timing.
            input_args: list[str] = []
            efps = effective_fps.get(stem)
            if efps is not None and 1.0 <= efps <= 60.0:
                input_args = ["-r", f"{efps:.4f}"]
                print(f"[rec] re-timing {src.name} at measured "
                      f"{efps:.2f} fps (vs cv2 header {self.fps})")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                *input_args,
                "-i", str(src),
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                *video_args,
                "-c:a", "aac", "-b:a", "96k",
                "-shortest", "-movflags", "+faststart",
                str(tmp),
            ]
            print(f"[rec] re-encoding {src.name} -> H.264 ({encoder}) "
                  f"+ silent AAC ...")
            t0 = time.monotonic()
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"[rec] ffmpeg failed for {src.name}: "
                      f"{(e.stderr or '').strip()}")
                try: tmp.unlink()
                except FileNotFoundError: pass
                continue
            except FileNotFoundError as e:
                print(f"[rec] could not run ffmpeg: {e}")
                return
            # Sanity-check the output. ffmpeg can return success while
            # producing a file that no player will accept; a quick
            # ffprobe + decode probe catches the worst cases before we
            # destroy the original.
            if not _probe_output_is_h264_playable(tmp):
                print(f"[rec] re-encoded {src.name} failed playback "
                      f"sanity check -- keeping original")
                try: tmp.unlink()
                except FileNotFoundError: pass
                continue
            # Path.replace is atomic on POSIX and overwrites the target,
            # so this works whether src == final (both .mp4) or not.
            tmp.replace(final)
            if src != final:
                try: src.unlink()
                except FileNotFoundError: pass
            size_mb = final.stat().st_size / (1024 * 1024)
            print(f"[rec]   -> {final.name} ({size_mb:.1f} MB in "
                  f"{time.monotonic() - t0:.1f}s)")

    # ----------------------------------------------------------- public write
    def push_raw_frame(self, frame_bgr: np.ndarray) -> None:
        self._enqueue("raw", frame_bgr)

    def push_annotated_frame(self, frame_bgr: np.ndarray) -> None:
        self._enqueue("ann", frame_bgr)

    def log_row(self, state: MissionState,
                pose: Optional[MarkerPose],
                tel: Optional[TelemetrySnapshot]) -> None:
        snap = state.snapshot()
        with self._csv_lock:
            row = {f: "" for f in _CSV_FIELDS}
            row["wall_time"] = _dt.datetime.now().isoformat(timespec="milliseconds")
            row["monotonic"] = f"{time.monotonic():.4f}"
            row["phase"] = snap["phase"]
            d = snap.get("distance_m"); y = snap.get("yaw_to_marker_deg"); h = snap.get("relative_heading_deg")
            row["distance_m"] = f"{d:.3f}" if d is not None else ""
            row["yaw_to_marker_deg"] = f"{y:.2f}" if y is not None else ""
            row["relative_heading_deg"] = f"{h:.2f}" if h is not None else ""
            row["marker_id"] = pose.marker_id if pose else ""
            row["marker_seen"] = "1" if pose else "0"
            if pose is not None:
                row["marker_normal_bearing_deg"] = f"{pose.marker_normal_bearing_deg:.2f}"
                row["marker_tilt_deg"] = f"{pose.marker_tilt_deg:.2f}"
                row["marker_inplane_rot_deg"] = f"{pose.marker_inplane_rot_deg:.2f}"
            row["target_distance_m"] = f"{snap['target_distance_m']:.3f}"
            row["target_relative_heading_deg"] = f"{snap['target_relative_heading_deg']:.2f}"
            # mission_step_idx (-1 means "before first step"). Replay
            # uses this to highlight the active step in the script
            # list without re-deriving it from phase transitions.
            mi = snap.get("mission_step_idx")
            row["mission_step_idx"] = "" if mi is None else str(mi)
            wp = snap.get("world_position_m")
            if wp is not None:
                row["world_x"] = f"{wp[0]:.3f}"
                row["world_y"] = f"{wp[1]:.3f}"
                row["world_z"] = f"{wp[2]:.3f}"
            wpu = snap.get("world_position_used_markers") or []
            row["world_n_used"] = str(len(wpu))
            wpa = snap.get("world_position_age_s")
            row["world_position_age_s"] = (
                f"{wpa:.3f}" if wpa is not None else "")
            wpc = snap.get("world_position_confidence")
            row["world_position_confidence"] = (
                f"{wpc:.3f}" if wpc is not None else "")
            ayd = snap.get("arena_yaw_deg")
            row["arena_yaw_deg"] = f"{ayd:.2f}" if ayd is not None else ""
            aya = snap.get("arena_yaw_age_s")
            row["arena_yaw_age_s"] = f"{aya:.3f}" if aya is not None else ""
            wvk = snap.get("world_velocity_m_kf")
            if wvk is not None:
                row["world_vx_kf"] = f"{wvk[0]:.4f}"
                row["world_vy_kf"] = f"{wvk[1]:.4f}"
                row["world_vz_kf"] = f"{wvk[2]:.4f}"
            row["target_pose_method"] = snap.get("target_pose_method", "") or ""
            wpm = snap.get("world_position_pose_methods") or []
            # Format as "id:method|id:method" so the CSV stays one
            # column but the per-marker breakdown is recoverable.
            row["arena_pose_methods"] = "|".join(
                f"{mid}:{meth}"
                for mid, meth in zip(wpu, wpm))
            wppm = snap.get("world_position_per_marker") or []
            # Format: "id:x,y,z|id:x,y,z" -- per-marker camera world
            # position vote BEFORE the weighted average, so per-marker
            # bias can be diagnosed from the log.
            row["arena_per_marker_world"] = "|".join(
                f"{mid}:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}"
                for mid, p in zip(wpu, wppm))
            rc = snap["rc"]
            row["rc_lr"] = rc["lr"]; row["rc_fb"] = rc["fb"]
            row["rc_ud"] = rc["ud"]; row["rc_yaw"] = rc["yaw"]
            tel_raw = snap.get("telemetry") or {}
            for src, dst in [("battery", "tel_battery"), ("yaw", "tel_yaw"),
                             ("pitch", "tel_pitch"), ("roll", "tel_roll"),
                             ("height_cm", "tel_height_cm"),
                             ("flight_time_s", "tel_flight_time_s"),
                             ("vgx", "tel_vgx"), ("vgy", "tel_vgy"),
                             ("vgz", "tel_vgz")]:
                v = tel_raw.get(src)
                row[dst] = v if v is not None else ""
            row["tel_flying"] = "1" if tel_raw.get("flying") else "0"
            row["tel_connected"] = "1" if tel_raw.get("connected") else "0"
            row["telemetry_stalled"] = ("1" if snap.get("telemetry_stalled")
                                        else "0")
            row["telemetry_rc_gated"] = ("1" if snap.get("telemetry_rc_gated")
                                         else "0")
            host = snap.get("host") or {}
            for src, dst in [("system_cpu_pct", "host_cpu_pct"),
                             ("process_cpu_pct", "host_proc_cpu_pct"),
                             ("load_1m", "host_load_1m"),
                             ("mem_used_pct", "host_mem_pct")]:
                v = host.get(src)
                row[dst] = "" if v is None else v
            row["note"] = (snap.get("note") or "")
            row["abort_reason"] = (snap.get("abort_reason") or "")
            vmids = snap.get("visible_marker_ids") or []
            row["visible_marker_ids"] = "|".join(str(int(m))
                                                  for m in vmids)
            try:
                self._csv_writer.writerow(row)
                # Flush periodically so we keep data on disk if we crash.
                self._csv_fp.flush()
            except Exception as e:
                print(f"[rec] csv write error: {e}")

    @property
    def stats(self) -> dict:
        return {
            "queue_depth": self._video_q.qsize(),
            "frames_dropped": self._frames_dropped,
            "csv_path": str(self._csv_path),
        }

    # ------------------------------------------------------------------ impl
    def _enqueue(self, kind: str, frame: np.ndarray) -> None:
        try:
            self._video_q.put_nowait((kind, frame))
        except Exception:
            self._frames_dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._video_q.empty():
            try:
                kind, frame = self._video_q.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._write(kind, frame)
            except Exception as e:
                print(f"[rec] write error: {e}")

    def _open_writer(self, path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(path), fourcc, self.fps, size)
        if not w.isOpened():
            # Fallback to AVI if mp4v isn't available.
            avi_path = path.with_suffix(".avi")
            w = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                self.fps, size)
        return w

    def _write(self, kind: str, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        size = (w, h)
        if kind == "raw":
            if self._raw_writer is None or self._raw_size != size:
                if self._raw_writer is not None:
                    self._raw_writer.release()
                self._raw_writer = self._open_writer(self.dir / "raw.mp4", size)
                self._raw_size = size
            self._raw_writer.write(frame)
        elif kind == "ann":
            if self._ann_writer is None or self._ann_size != size:
                if self._ann_writer is not None:
                    self._ann_writer.release()
                self._ann_writer = self._open_writer(self.dir / "annotated.mp4",
                                                     size)
                self._ann_size = size
            self._ann_writer.write(frame)
        # Record arrival timestamps for the effective-fps computation
        # at finalize. Done AFTER the write so a failing encoder
        # doesn't pollute the count.
        now = time.monotonic()
        if kind not in self._first_frame_t:
            self._first_frame_t[kind] = now
        self._last_frame_t[kind] = now
        self._frame_count[kind] = self._frame_count.get(kind, 0) + 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_output_is_h264_playable(path: Path) -> bool:
    """Quick sanity check: the file's first stream is H.264 AND ffmpeg
    can decode at least a few frames out of it.

    Catches the case where the encoder reports success but produces a
    bitstream that decodes as a black/empty image (occasionally seen
    with libopenh264 builds + certain players).
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, timeout=5.0,
        ).stdout.strip().splitlines()
    except Exception:
        return False
    if not out or out[0].strip() != "h264":
        return False
    # `nb_frames` may be missing for some containers -- fall back to a
    # short decode probe.
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
             "-frames:v", "5", "-f", "null", "-"],
            capture_output=True, text=True, check=True, timeout=10.0,
        )
    except Exception:
        return False
    return True


def _probe_h264_encoder(name: str) -> bool:
    """Try to actually encode one black frame with the given encoder.

    `ffmpeg -encoders` lists every encoder ffmpeg was compiled against,
    but listed != usable -- libopenh264.so might be missing at runtime,
    a VAAPI render node might not be accessible, an NVENC card might
    not be present. The only reliable test is to run a tiny encode.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi",
        "-i", "color=c=black:s=64x64:r=1:d=0.04",
        "-c:v", name, "-f", "null", "-",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=10.0)
        return True
    except Exception:
        return False


def _pick_h264_encoder() -> Optional[str]:
    """Return the name of an available ffmpeg H.264 encoder, or None.

    Preference order: libx264 (best quality + universal compatibility,
    but proprietary; not bundled in stock Fedora ffmpeg -- install
    `ffmpeg-libs` from RPMFusion-free if you want it), libopenh264
    (Cisco's free encoder, ships everywhere but earlier flights showed
    its output sometimes refuses to render in VLC), then common
    hardware backends.

    Each candidate is *probed* with a tiny test encode -- a listed
    encoder isn't necessarily a working encoder.
    """
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True,
                             check=True, timeout=5.0).stdout
    except Exception as e:
        print(f"[rec] ffmpeg -encoders failed: {e}")
        return None
    candidates = ("libx264", "libopenh264",
                  "h264_nvenc", "h264_vaapi", "h264_qsv")
    listed = [n for n in candidates if n in out]
    if not listed:
        snippet = "\n  ".join(line for line in out.splitlines()
                              if "h264" in line.lower())[:600]
        print(f"[rec] ffmpeg has no recognised H.264 encoder.")
        if snippet:
            print(f"[rec]   h264-related encoders ffmpeg knows about:\n  "
                  f"{snippet}")
        return None
    for name in listed:
        if _probe_h264_encoder(name):
            return name
        print(f"[rec] H.264 encoder '{name}' is listed but does not "
              f"actually run on this host -- trying next.")
    print(f"[rec] none of {listed} produced a working encode.")
    return None


def make_flight_dir(root: Path, serial: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_serial = "".join(c for c in (serial or "unknown") if c.isalnum() or c in "-_")
    out = Path(root) / f"{stamp}_{safe_serial}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tally_pose_methods(csv_path: Path) -> dict:
    """Count how often each IPPE branch-picker layer fired across a
    finished flight. Reads the recorded CSV, tallies the
    ``target_pose_method`` column (the active-marker pose path the
    controller actually used) and the per-marker ``arena_pose_methods``
    column (the world-position vote contributors). Result lands in
    ``mission_meta.outcome.pose_method_counts`` so the operator can
    answer "did the magnetometer rescue mostly do the work, or did
    the prev-anchor / cold-start fallbacks earn their keep?".

    Returns a dict like::

        {
            "target": {"ippe_temporal": 134, "ippe_mag_swap": 47, ...},
            "per_marker": {"ippe_lowerr": 200, "ippe_swapped": 13, ...},
            "rows": 247,
            "frames_with_marker": 218,
        }
    """
    out = {"target": {}, "per_marker": {}, "rows": 0,
           "frames_with_marker": 0}
    if not csv_path.is_file():
        return out
    try:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                out["rows"] += 1
                if row.get("marker_seen") == "1":
                    out["frames_with_marker"] += 1
                tgt = (row.get("target_pose_method") or "").strip()
                if tgt:
                    out["target"][tgt] = out["target"].get(tgt, 0) + 1
                blob = (row.get("arena_pose_methods") or "").strip()
                if blob:
                    for chunk in blob.split("|"):
                        if ":" not in chunk:
                            continue
                        _, meth = chunk.split(":", 1)
                        meth = meth.strip()
                        if not meth:
                            continue
                        out["per_marker"][meth] = (
                            out["per_marker"].get(meth, 0) + 1)
    except (OSError, ValueError) as e:
        print(f"[meta] pose_method tally failed: {e}")
    return out


def write_meta(flight_dir: Path, cfg: MissionConfig,
               calibration: Calibration,
               outcome: dict) -> None:
    # Tally the IPPE branch-picker layer firing counts so the operator
    # can compare what actually carried the position estimate over the
    # flight (magnetometer? prev-anchor swap? cold-start lower-err?).
    # Lands in outcome.pose_method_counts.
    counts = _tally_pose_methods(flight_dir / "flight_log.csv")
    outcome = dict(outcome)
    outcome["pose_method_counts"] = counts
    meta = {
        "config": asdict(cfg),
        "calibration": {
            "serial": calibration.serial,
            "resolution": calibration.resolution,
            "is_default": calibration.is_default,
            "rms_error": calibration.rms_error,
            "calibrated_at": calibration.calibrated_at,
            "fx": calibration.fx, "fy": calibration.fy,
            "cx": calibration.cx, "cy": calibration.cy,
            "image_size": list(calibration.image_size),
            "notes": calibration.notes,
        },
        "outcome": outcome,
    }
    (flight_dir / "mission_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))