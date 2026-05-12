"""
Per-drone camera calibration store.

Calibration files are stored as compressed NumPy archives at
``<calib_dir>/anafi_<serial>_<resolution>.npz`` so that you can keep a
small file per drone in version control alongside this code.

Each archive contains:

* ``camera_matrix``   -- shape (3, 3), the intrinsic K matrix
* ``dist_coeffs``     -- shape (5,) or (8,), Brown-Conrady distortion
* ``image_size``      -- shape (2,), [width, height] in pixels
* ``rms_error``       -- scalar, reprojection error from calibration
* ``serial``          -- string, the drone serial number
* ``resolution``      -- string, e.g. ``720p`` or ``1080p``
* ``calibrated_at``   -- ISO timestamp string
* ``notes``           -- free-form string (lens, EIS state, software ver.)

If a calibration file does not yet exist for a given drone, callers
get a sensible default (the published Anafi 720p intrinsics with zero
distortion); the script will warn loudly so the operator knows the
flight will run with reduced accuracy.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Default intrinsics for an uncalibrated Anafi at 720p w/ EIS-class crop.
# Recovered empirically from the test dataset (see earlier analysis):
#   f_eff ~ 970 px, principal point at image centre, no distortion model.
# ---------------------------------------------------------------------------

_DEFAULT_W = 1280
_DEFAULT_H = 720
_DEFAULT_F = 970.0


def _default_K(width: int = _DEFAULT_W, height: int = _DEFAULT_H,
               f_eff: float = _DEFAULT_F) -> np.ndarray:
    return np.array([[f_eff, 0.0,    width / 2.0],
                     [0.0,   f_eff,  height / 2.0],
                     [0.0,   0.0,    1.0         ]], dtype=np.float64)


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    serial: str
    resolution: str
    rms_error: float = float("nan")
    calibrated_at: str = ""
    notes: str = ""
    is_default: bool = False        # True if this is the fallback, not a real cal

    # ------------------------------------------------------------------ misc
    @property
    def fx(self) -> float: return float(self.camera_matrix[0, 0])
    @property
    def fy(self) -> float: return float(self.camera_matrix[1, 1])
    @property
    def cx(self) -> float: return float(self.camera_matrix[0, 2])
    @property
    def cy(self) -> float: return float(self.camera_matrix[1, 2])

    def short_summary(self) -> str:
        kind = "DEFAULT (uncalibrated)" if self.is_default else "calibrated"
        return (f"{kind}: serial={self.serial} resolution={self.resolution} "
                f"fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f} "
                f"rms={self.rms_error:.3f}")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CalibrationStore:
    """File-system-backed key/value store keyed by serial+resolution."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- key
    def _path(self, serial: str, resolution: str) -> Path:
        safe_serial = "".join(c for c in serial if c.isalnum() or c in "-_") or "unknown"
        safe_res = "".join(c for c in resolution if c.isalnum()) or "unknown"
        return self.root / f"anafi_{safe_serial}_{safe_res}.npz"

    # ----------------------------------------------------------------- query
    def has(self, serial: str, resolution: str = "720p") -> bool:
        return self._path(serial, resolution).exists()

    def load(self, serial: str, resolution: str = "720p",
             allow_default: bool = True) -> Calibration:
        """Load the calibration for ``serial`` at ``resolution``.

        If no file exists and ``allow_default`` is True, returns the
        published Anafi defaults with ``is_default=True`` so that the
        caller can warn the user. If ``allow_default`` is False, raises
        FileNotFoundError.
        """
        path = self._path(serial, resolution)
        if not path.exists():
            if not allow_default:
                raise FileNotFoundError(path)
            # Pick image size from the resolution label
            if resolution == "1080p":
                w, h, f = 1920, 1080, 1455.0  # scaled f_eff = 970 * (1920/1280)
            else:
                w, h, f = _DEFAULT_W, _DEFAULT_H, _DEFAULT_F
            return Calibration(
                camera_matrix=_default_K(w, h, f),
                dist_coeffs=np.zeros(5, dtype=np.float64),
                image_size=(w, h),
                serial=serial,
                resolution=resolution,
                notes=f"DEFAULT: no calibration file at {path}",
                is_default=True,
            )
        with np.load(path, allow_pickle=False) as npz:
            cm = np.asarray(npz["camera_matrix"], dtype=np.float64)
            dc = np.asarray(npz["dist_coeffs"], dtype=np.float64)
            sz = tuple(int(v) for v in npz["image_size"])
            rms = float(npz["rms_error"]) if "rms_error" in npz else float("nan")
            cal_at = str(npz["calibrated_at"]) if "calibrated_at" in npz else ""
            notes = str(npz["notes"]) if "notes" in npz else ""
        return Calibration(
            camera_matrix=cm, dist_coeffs=dc, image_size=sz,
            serial=serial, resolution=resolution,
            rms_error=rms, calibrated_at=cal_at, notes=notes,
            is_default=False,
        )

    # ----------------------------------------------------------------- write
    def save(self, cal: Calibration) -> Path:
        path = self._path(cal.serial, cal.resolution)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix=cal.camera_matrix.astype(np.float64),
            dist_coeffs=cal.dist_coeffs.astype(np.float64),
            image_size=np.asarray(cal.image_size, dtype=np.int32),
            rms_error=np.float64(cal.rms_error),
            calibrated_at=cal.calibrated_at or _dt.datetime.now().isoformat(timespec="seconds"),
            notes=cal.notes,
        )
        return path


# ---------------------------------------------------------------------------
# Calibration capture / solve
# ---------------------------------------------------------------------------

def calibrate_from_video(video_path: Path,
                         pattern_size: tuple[int, int] = (9, 6),
                         square_size_m: float = 0.025,
                         frame_stride: int = 10) -> Calibration:
    """Run intrinsic calibration on a checkerboard video.

    Returns a :class:`Calibration` with ``serial`` and ``resolution`` left
    blank -- the caller is responsible for filling those in before saving
    so that the file is keyed correctly. This keeps this function free
    of any drone-side dependencies (pure cv2 + numpy).
    """
    import cv2  # local import: calibration is rarely run

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: Optional[tuple[int, int]] = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    idx = 0
    accepted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                                                   | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
        accepted += 1
    cap.release()
    if accepted < 8:
        raise RuntimeError(f"only {accepted} usable frames -- need >= 8")
    rms, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size,
                                          None, None)
    return Calibration(
        camera_matrix=K.astype(np.float64),
        dist_coeffs=D.astype(np.float64).ravel(),
        image_size=img_size,
        serial="",
        resolution="",
        rms_error=float(rms),
        calibrated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        notes=f"video={video_path.name} n_frames={accepted}",
        is_default=False,
    )
