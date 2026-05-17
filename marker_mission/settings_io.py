"""Local-settings import / export for marker_mission.

Bundles everything the operator can edit through the UI (live tune
state, live mission, live arena, plus every saved snapshot of each)
into a single zip file that can be moved between hosts. On import
the destination's current state is backed up first, then wiped and
replaced — the import never merges, so a confusing "half new, half
old" state is impossible.

In scope
--------
Top-level files under ``~/.marker_mission/`` (or ``$MM_DATA_DIR``):

* ``config.json``                  live tune state
* ``defaults.json``                pinned-default pointers (which
                                   snapshot/script/arena counts as
                                   the per-subsystem default)
* ``active_mission_script.txt``    live mission script draft
* ``active_arena_config.json``     live arena config

Subdirectories:

* ``snapshots/*.json``             saved tune snapshots
* ``mission_scripts/*.txt``        saved mission scripts
* ``arenas/*.json``                saved arena configs

Out of scope (deliberately): ``calibrations/*.npz`` (per-drone
camera intrinsics — host-specific, would mismatch on import),
``camera_configs/*.json`` (per-drone camera tuning — same reason)
and ``flights/`` (large per-flight artefacts).

Backups & revert
----------------
Every import creates a backup at
``~/.marker_mission/backups/backup-<ts>.zip`` containing the state
that the import is about to overwrite. Same for revert ("pre-revert"
backup of the state we're reverting away from). The UI exposes the
backup list as a history; restoring from a backup is itself another
import, so an accidental revert is itself revertible.
"""
from __future__ import annotations

import io
import json
import re
import socket
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from . import config as cfg


# Top-level files copied verbatim.
EXPORT_FILES: List[str] = [
    "config.json",
    "defaults.json",
    "active_mission_script.txt",
    "active_arena_config.json",
]

# Subdirectories whose direct file children are copied. Recursion is
# deliberately one-level — anything nested deeper isn't part of the
# advertised contract.
EXPORT_DIRS: List[str] = [
    "snapshots",
    "mission_scripts",
    "arenas",
]

MANIFEST_FILENAME = "MANIFEST.json"
SCHEMA_VERSION = 1

BACKUPS_DIR = cfg.DEFAULT_DATA_DIR / "backups"
BACKUP_PREFIX = "backup-"
BACKUP_SUFFIX = ".zip"


# ---------------------------------------------------------------------------
# Hostname → short id used in export filenames
# ---------------------------------------------------------------------------

_HOST_ID_RE = re.compile(r"(?:sdc-)?(?:flightctrl|fc)(\d+)", re.IGNORECASE)


def _host_short_id() -> str:
    """``flightctrl3`` → ``fc3``; anything else → first 12 chars of
    hostname with dots replaced by dashes so it survives in a filename.
    Used to build the export filename per the operator's request
    (``sdc-fc<N>-settings-<ts>.zip``)."""
    h = socket.gethostname()
    m = _HOST_ID_RE.match(h)
    if m:
        return f"fc{m.group(1)}"
    return h[:12].replace(".", "-").replace("/", "-") or "host"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _build_manifest(origin: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "origin": origin,                     # "export" | "backup" | "pre-revert"
        "hostname": socket.gethostname(),
        "host_id": _host_short_id(),
        "created_at_unix": time.time(),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "files": [],                          # filled in as we write
    }


def _add_to_zip(z: zipfile.ZipFile, manifest: dict) -> None:
    """Write all in-scope files into ``z`` and append their arcnames to
    ``manifest['files']``. Missing files are silently skipped — a host
    that has never used a particular subsystem just exports without
    that file present."""
    for fn in EXPORT_FILES:
        p = cfg.DEFAULT_DATA_DIR / fn
        if p.is_file():
            z.write(p, arcname=fn)
            manifest["files"].append(fn)
    for d in EXPORT_DIRS:
        dp = cfg.DEFAULT_DATA_DIR / d
        if not dp.is_dir():
            continue
        for fp in sorted(dp.iterdir()):
            if not fp.is_file():
                continue
            arc = f"{d}/{fp.name}"
            z.write(fp, arcname=arc)
            manifest["files"].append(arc)


def export_zip() -> Tuple[bytes, str]:
    """Return ``(zip_bytes, suggested_filename)`` ready for HTTP
    download. Filename matches the operator-requested format
    ``sdc-fc<N>-settings-<ts>.zip``."""
    buf = io.BytesIO()
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"sdc-{_host_short_id()}-settings-{ts}.zip"
    manifest = _build_manifest(origin="export")
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        _add_to_zip(z, manifest)
        z.writestr(MANIFEST_FILENAME, json.dumps(manifest, indent=2))
    return buf.getvalue(), fname


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def _build_backup_zip(origin: str) -> Path:
    """Snapshot current state into ``BACKUPS_DIR``. ``origin`` is
    stored in the manifest so the history UI can distinguish a
    pre-import backup from a pre-revert one."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = BACKUPS_DIR / f"{BACKUP_PREFIX}{ts}{BACKUP_SUFFIX}"
    # Uniqueness — if the operator triggers two imports inside one
    # second the suffix-counter keeps both backups instead of
    # silently clobbering one.
    n = 1
    while path.exists():
        path = BACKUPS_DIR / f"{BACKUP_PREFIX}{ts}-{n}{BACKUP_SUFFIX}"
        n += 1
    manifest = _build_manifest(origin=origin)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        _add_to_zip(z, manifest)
        z.writestr(MANIFEST_FILENAME, json.dumps(manifest, indent=2))
    return path


def list_backups() -> List[dict]:
    """Newest backup first. Reads each backup's MANIFEST.json so the
    UI can show ``origin`` and the captured file count without the
    operator having to download the zip."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[dict] = []
    for p in sorted(BACKUPS_DIR.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
                    reverse=True):
        try:
            stat = p.stat()
        except OSError:
            continue
        manifest: dict = {}
        try:
            with zipfile.ZipFile(p, "r") as z:
                manifest = json.loads(z.read(MANIFEST_FILENAME).decode())
        except Exception:
            pass
        out.append({
            "name": p.name,
            "size_bytes": int(stat.st_size),
            "modified_at_unix": stat.st_mtime,
            "modified_at_utc": (datetime.utcfromtimestamp(stat.st_mtime)
                                .isoformat(timespec="seconds") + "Z"),
            "origin": manifest.get("origin"),
            "host_id": manifest.get("host_id"),
            "file_count": len(manifest.get("files", [])),
        })
    return out


def delete_backup(name: str) -> bool:
    """Operator-triggered cleanup. ``name`` must match the
    ``backup-<ts>[-N].zip`` pattern so this can never delete an
    unrelated file in ``BACKUPS_DIR`` (defensive — that dir is
    ours, but path-traversal protection is cheap)."""
    if not _is_backup_name(name):
        raise ValueError(f"invalid backup name: {name!r}")
    p = BACKUPS_DIR / name
    if not p.is_file():
        return False
    p.unlink()
    return True


def _is_backup_name(name: str) -> bool:
    return (isinstance(name, str)
            and "/" not in name
            and "\\" not in name
            and name.startswith(BACKUP_PREFIX)
            and name.endswith(BACKUP_SUFFIX))


# ---------------------------------------------------------------------------
# Restore (used by both import and revert)
# ---------------------------------------------------------------------------

def _clear_target_paths() -> None:
    """Remove all in-scope files so the restore lands on a clean
    slate. ``flights/``, ``calibrations/``, ``camera_configs/`` and
    ``backups/`` are NEVER touched here."""
    for fn in EXPORT_FILES:
        p = cfg.DEFAULT_DATA_DIR / fn
        if p.is_file():
            p.unlink()
    for d in EXPORT_DIRS:
        dp = cfg.DEFAULT_DATA_DIR / d
        if not dp.is_dir():
            continue
        for fp in dp.iterdir():
            if fp.is_file():
                fp.unlink()


def _extract_zip_bytes(zip_bytes: bytes) -> dict:
    """Extract a settings zip into ``DEFAULT_DATA_DIR``. Whitelists
    every entry against the allowed top-level files and subdirectories
    so a malicious or corrupted archive can't write outside
    ``~/.marker_mission`` (zip-slip)."""
    allowed_files = set(EXPORT_FILES)
    allowed_dirs = set(EXPORT_DIRS)
    cfg.DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    skipped: List[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        try:
            manifest = json.loads(z.read(MANIFEST_FILENAME).decode())
        except KeyError:
            raise ValueError(
                "zip is missing MANIFEST.json — not a settings export"
            )
        for info in z.infolist():
            name = info.filename
            if name == MANIFEST_FILENAME:
                continue
            if info.is_dir():
                continue
            # Reject any traversal attempt up-front. zipfile already
            # normalises forward-slashes; we explicitly reject
            # back-slashes and ``..`` in case a Windows-built zip ever
            # slips through.
            if (".." in Path(name).parts
                    or name.startswith("/")
                    or "\\" in name):
                skipped.append(name)
                continue
            if "/" in name:
                parts = name.split("/")
                if len(parts) != 2 or parts[0] not in allowed_dirs:
                    skipped.append(name)
                    continue
                target = cfg.DEFAULT_DATA_DIR / parts[0] / parts[1]
                target.parent.mkdir(parents=True, exist_ok=True)
            elif name in allowed_files:
                target = cfg.DEFAULT_DATA_DIR / name
            else:
                skipped.append(name)
                continue
            with z.open(info) as src:
                target.write_bytes(src.read())
    if skipped:
        manifest = dict(manifest)
        manifest["skipped"] = skipped
        print(f"[settings_io] skipped {len(skipped)} out-of-scope "
              f"entries: {skipped[:5]}{'…' if len(skipped) > 5 else ''}")
    return manifest


def import_zip(zip_bytes: bytes) -> dict:
    """Wipe-and-restore from operator-uploaded zip. Returns the
    backup id (filename) created from the pre-import state plus the
    imported zip's manifest, so the UI can render the result."""
    backup_path = _build_backup_zip(origin="pre-import")
    _clear_target_paths()
    manifest = _extract_zip_bytes(zip_bytes)
    return {
        "backup_id": backup_path.name,
        "manifest": manifest,
    }


def restore_backup(name: str) -> dict:
    """Revert: snapshot current state into a fresh backup, then
    overwrite with the named backup. The pre-revert backup is itself
    listed in the history, so an accidental revert is recoverable."""
    if not _is_backup_name(name):
        raise ValueError(f"invalid backup name: {name!r}")
    src = BACKUPS_DIR / name
    if not src.is_file():
        raise FileNotFoundError(f"backup not found: {name}")
    pre_revert = _build_backup_zip(origin="pre-revert")
    _clear_target_paths()
    manifest = _extract_zip_bytes(src.read_bytes())
    return {
        "backup_id": pre_revert.name,
        "manifest": manifest,
        "restored_from": name,
    }


def read_backup_bytes(name: str) -> bytes:
    """Return the raw bytes of a named backup — for the UI's "Download"
    button. Defensive about the name."""
    if not _is_backup_name(name):
        raise ValueError(f"invalid backup name: {name!r}")
    p = BACKUPS_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"backup not found: {name}")
    return p.read_bytes()
