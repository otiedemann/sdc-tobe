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
    "rc_lr", "rc_fb", "rc_ud", "rc_yaw",
    "tel_battery", "tel_yaw", "tel_pitch", "tel_roll",
    "tel_height_cm", "tel_flight_time_s",
    "tel_vgx", "tel_vgy", "tel_vgz",
    "tel_flying", "tel_connected",
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
        # OpenCV's mp4v output isn't accepted by WhatsApp (and several
        # other ingest pipelines); re-encode to H.264 + silent AAC so the
        # files are shareable. Run this in the background -- the next
        # mission can roll a fresh flight dir and start while the old
        # one is still re-encoding. The daemon thread is fire-and-forget;
        # the encoded file just appears when it's done.
        threading.Thread(target=self._reencode_for_sharing,
                         name=f"reencode-{self.dir.name}",
                         daemon=True).start()

    # --------------------------------------------------------- post-process
    def _reencode_for_sharing(self) -> None:
        """Re-encode raw and annotated videos to H.264 + dummy AAC track.

        OpenCV writes mp4v (MPEG-4 Part 2) which most modern apps refuse
        to ingest. We use ffmpeg with libx264 and a generated silent
        audio stream so the resulting file plays everywhere and can be
        sent over WhatsApp/Telegram/etc.

        Originals are replaced atomically on success and deleted if the
        OpenCV writer fell back to .avi. On failure (ffmpeg missing or
        encode error) the originals are left untouched so the operator
        still has the raw artefacts.
        """
        if shutil.which("ffmpeg") is None:
            print("[rec] ffmpeg not found -- keeping mp4v originals "
                  "(install ffmpeg for shareable H.264 output)")
            return
        encoder = _pick_h264_encoder()
        if encoder is None:
            print("[rec] ffmpeg has no H.264 encoder available -- "
                  "keeping mp4v originals")
            return
        for stem in ("raw", "annotated"):
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
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
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


def write_meta(flight_dir: Path, cfg: MissionConfig,
               calibration: Calibration,
               outcome: dict) -> None:
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
