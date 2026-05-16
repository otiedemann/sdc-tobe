"""Central calibration library: pulls .npz files from every FC into
``library_dir/`` so they can later be pushed back to whatever FC the
drone has been physically moved to.

Each ``.npz`` is accompanied by a ``<stem>.json`` sidecar carrying the
source FC name + fetch time so the UI can show provenance without
re-querying.

Direction of travel:
  - FC → library: automatic, every ``calibration_sync_seconds``.
  - library → FC: manual only (via ``CalibrationLibrary.push``).
This avoids silently pulling a stale FC copy backwards.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from .config import C2Config
from .fc_pool import FCPool

_NAME_RE = re.compile(r"^anafi_[A-Za-z0-9_\-]+_[A-Za-z0-9]+\.npz$")


def is_valid_calibration_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


class CalibrationLibrary:
    def __init__(self, cfg: C2Config, pool: FCPool):
        self.cfg = cfg
        self.pool = pool
        self.library_dir: Path = cfg.library_path()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("c2.cal")
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_sync_monotonic: Optional[float] = None
        self._sync_count = 0
        self._added_since_start = 0

    # ------------------------------------------------------------------ paths
    def _path(self, name: str) -> Path:
        if not is_valid_calibration_name(name):
            raise ValueError(f"invalid calibration filename: {name!r}")
        return self.library_dir / name

    def _sidecar(self, name: str) -> Path:
        if not is_valid_calibration_name(name):
            raise ValueError(f"invalid calibration filename: {name!r}")
        return self.library_dir / (name[:-len(".npz")] + ".json")

    # ----------------------------------------------------------------- write
    def _write_atomic(self, target: Path, data: bytes) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, target)

    def _write_sidecar(self, name: str, source_fc: str,
                       source_name: str) -> None:
        sidecar_path = self._sidecar(name)
        payload = {
            "source_fc": source_fc,
            "source_filename": source_name,
            "fetched_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        self._write_atomic(sidecar_path,
                           json.dumps(payload, indent=2).encode("utf-8"))

    # ----------------------------------------------------------------- sync
    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="c2-cal-sync")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self) -> None:
        # Initial small delay so the FC pool has time to start polling.
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=2.0)
            return
        except asyncio.TimeoutError:
            pass
        while not self._stop.is_set():
            try:
                await self.sweep_once()
            except Exception as e:
                self.log.warning("sweep failed: %s", e)
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.cfg.calibration_sync_seconds,
                )
                return
            except asyncio.TimeoutError:
                continue

    async def sweep_once(self) -> dict[str, Any]:
        """One pull-from-every-FC pass. Returns a summary dict."""
        async with self._lock:
            added = 0
            errors: dict[str, str] = {}
            for client in self.pool.all_clients():
                fc_name = client.spec.name
                ok, payload = await client.list_calibrations()
                if not ok or not isinstance(payload, dict):
                    errors[fc_name] = (
                        payload if isinstance(payload, str) else str(payload))
                    continue
                entries = payload.get("files") or []
                if not isinstance(entries, list):
                    errors[fc_name] = "files[] missing"
                    continue
                # Update the pool's per-FC calibrations cache so the UI
                # /calibrate iframe and the library table stay in sync.
                await self.pool.set_calibrations(fc_name, entries)
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    if (not isinstance(name, str)
                            or not is_valid_calibration_name(name)):
                        continue
                    fc_mtime = float(entry.get("mtime") or 0.0)
                    local_path = self._path(name)
                    pull = False
                    if not local_path.exists():
                        pull = True
                    else:
                        # Pull only if FC has a strictly newer copy. Never
                        # overwrite a local copy that is newer — that would
                        # silently roll back a calibration the operator
                        # just pushed.
                        local_mtime = local_path.stat().st_mtime
                        if fc_mtime > local_mtime + 0.5:
                            pull = True
                    if pull:
                        ok2, blob = await client.fetch_calibration(name)
                        if not ok2 or not isinstance(blob, (bytes,
                                                            bytearray)):
                            errors[f"{fc_name}/{name}"] = (
                                blob if isinstance(blob, str)
                                else "fetch failed")
                            continue
                        try:
                            self._write_atomic(local_path, bytes(blob))
                            self._write_sidecar(name, fc_name, name)
                            added += 1
                            self._added_since_start += 1
                            self.log.info("pulled %s from %s (%d bytes)",
                                          name, fc_name, len(blob))
                        except OSError as e:
                            errors[f"{fc_name}/{name}"] = str(e)
            self._last_sync_monotonic = time.monotonic()
            self._sync_count += 1
            return {"added": added, "errors": errors,
                    "total_since_start": self._added_since_start}

    # ----------------------------------------------------------- read library
    def list_entries(self) -> list[dict]:
        """Scan ``library_dir`` and return one entry per ``.npz`` with its
        sidecar metadata, mtime, and the scalars stored in the npz."""
        out: list[dict] = []
        if not self.library_dir.exists():
            return out
        try:
            import numpy as np  # local — keep startup cheap
        except ImportError:
            np = None
        for path in sorted(self.library_dir.glob("anafi_*.npz")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entry: dict[str, Any] = {
                "name": path.name,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
            sidecar = self._sidecar(path.name)
            if sidecar.exists():
                try:
                    side = json.loads(sidecar.read_text())
                    entry["source_fc"] = side.get("source_fc")
                    entry["fetched_at"] = side.get("fetched_at")
                except (OSError, ValueError):
                    pass
            if np is not None:
                try:
                    with np.load(path, allow_pickle=False) as npz:
                        entry["serial"] = (str(npz["serial"])
                                           if "serial" in npz else None)
                        entry["resolution"] = (str(npz["resolution"])
                                               if "resolution" in npz
                                               else None)
                        entry["rms_error"] = (float(npz["rms_error"])
                                              if "rms_error" in npz
                                              else None)
                        entry["calibrated_at"] = (
                            str(npz["calibrated_at"])
                            if "calibrated_at" in npz else None
                        )
                        entry["image_size"] = (
                            [int(v) for v in npz["image_size"]]
                            if "image_size" in npz else None
                        )
                except Exception as e:
                    entry["read_error"] = str(e)
            out.append(entry)
        return out

    def read_blob(self, name: str) -> Optional[bytes]:
        path = self._path(name)
        if not path.exists():
            return None
        return path.read_bytes()

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        sidecar = self._sidecar(name)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        return True

    @property
    def last_sync_age_s(self) -> Optional[float]:
        if self._last_sync_monotonic is None:
            return None
        return time.monotonic() - self._last_sync_monotonic

    @property
    def sync_count(self) -> int:
        return self._sync_count

    @property
    def added_since_start(self) -> int:
        return self._added_since_start

    # ---------------------------------------------------------------- push
    async def push(self, name: str, fc_name: str) -> dict[str, Any]:
        """Push a local .npz to an FC. Returns ``{ok, warning?, error?}``."""
        client = self.pool.client(fc_name)
        if client is None:
            return {"ok": False, "error": f"unknown FC: {fc_name}"}
        blob = self.read_blob(name)
        if blob is None:
            return {"ok": False, "error": f"library missing: {name}"}
        # Best-effort serial-mismatch warning. Operator pre-staging a
        # calibration before connecting the matching drone is a valid
        # workflow, so this is a soft warning, not a refusal.
        warning: Optional[str] = None
        try:
            import numpy as np
            with np.load(self.library_path() if False else self._path(name),
                         allow_pickle=False) as npz:
                file_serial = (str(npz["serial"])
                               if "serial" in npz else "")
        except Exception:
            file_serial = ""
        fc_state = self.pool.states.get(fc_name)
        fc_serial = fc_state.drone_serial if fc_state else None
        if file_serial and fc_serial and file_serial != fc_serial:
            warning = (f"calibration serial {file_serial!r} does not match "
                       f"FC's current drone serial {fc_serial!r}")
        ok, payload = await client.push_calibration(name, blob)
        if not ok:
            return {"ok": False,
                    "error": (payload if isinstance(payload, str)
                              else str(payload)),
                    **({"warning": warning} if warning else {})}
        out: dict[str, Any] = {"ok": True, "payload": payload}
        if warning:
            out["warning"] = warning
        return out
