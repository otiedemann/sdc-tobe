"""Background settings sync to peer FCs.

Reads ~/.marker_mission/fleet.json for the peer list, exports the local
settings ZIP and POSTs it to every peer in a daemon thread. Peers that
receive a sync request do NOT push back (?peer_sync=1 flag), breaking
the loop.

fleet.json format (same file on every FC — each FC filters itself out):
  {
    "peers": [
      "http://flightctrl1:8080",
      "http://flightctrl2:8080",
      ...
    ]
  }

If fleet.json is absent or contains no peers, sync is silently skipped.
"""
from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import List

from .config import DEFAULT_DATA_DIR

FLEET_PATH = DEFAULT_DATA_DIR / "fleet.json"

_sync_lock = threading.Lock()
_sync_active = False


def load_peers() -> List[str]:
    """Return peer URLs, excluding any that match this host's hostname."""
    try:
        data = json.loads(FLEET_PATH.read_text())
        all_peers = [str(u).rstrip("/") for u in data.get("peers", []) if u]
    except Exception:
        return []
    # Filter out self so the FC never pushes to itself.
    hostname = socket.gethostname().lower()
    return [u for u in all_peers if hostname not in u.lower()]


def push_to_peers() -> None:
    """Export local settings and push to all peers in a background thread.

    Fire-and-forget: errors are logged but never raised to the caller.
    A second call while a push is already in-flight is silently dropped
    (one sync per save is enough).
    """
    global _sync_active

    peers = load_peers()
    if not peers:
        return

    with _sync_lock:
        if _sync_active:
            return
        _sync_active = True

    def _run():
        global _sync_active
        try:
            from .settings_io import export_zip
            import urllib.request
            import urllib.error

            try:
                zip_bytes, _ = export_zip()
            except Exception as e:
                print(f"[fleet_sync] export failed: {e}")
                return

            boundary = b"PeerSyncBoundary"
            body = (
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="file";'
                b' filename="peer-sync.zip"\r\n'
                b"Content-Type: application/zip\r\n\r\n"
                + zip_bytes
                + b"\r\n--" + boundary + b"--\r\n"
            )
            ct = f"multipart/form-data; boundary={boundary.decode()}"

            for peer in peers:
                url = f"{peer}/api/mission/settings/import?peer_sync=1"
                try:
                    req = urllib.request.Request(
                        url, data=body,
                        headers={"Content-Type": ct},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        result = json.loads(resp.read())
                        print(f"[fleet_sync] → {peer}: ok={result.get('ok')}")
                except urllib.error.URLError as e:
                    print(f"[fleet_sync] → {peer}: FAILED ({e})")
                except Exception as e:
                    print(f"[fleet_sync] → {peer}: ERROR ({e})")
        finally:
            with _sync_lock:
                _sync_active = False

    threading.Thread(target=_run, daemon=True, name="fleet-sync").start()
