"""Async HTTP client for one drone's ``unified_api_server`` (the FC).

Every drone in the fleet is fronted by its own
``controller_unified/unified_api_server.py`` Flask instance. ``DroneFC``
wraps that HTTP surface in a small, exception-safe async client so the
commander never has to think about sockets, timeouts or JSON shapes.

Design contract (relied on by world_model / commander / server):

* One ``DroneFC`` per drone, constructed with the drone's base URL.
* Backed by a single persistent ``httpx.AsyncClient`` (keep-alive, so a
  4 Hz command loop doesn't pay TCP setup every tick).
* EVERY method is exception-safe. Network errors never propagate:
    - dict-returning methods return ``{"ok": False, "error": str(e)}``
    - :meth:`disable_arena_guard` returns ``False``
  and a warning is logged to ``logging.getLogger("c2.fc")``.
* Methods are thin: they POST/GET the documented endpoint and hand the
  decoded JSON straight back. Higher layers interpret it.

FC API reference (see ``controller_unified/unified_api_server.py``):
    POST /api/takeoff   POST /api/land   POST /api/emergency   (empty body)
    POST /api/rc                     body {lr,fb,ud,yaw} each in [-100,100]
    GET  /api/position               {ok,pos,vel,dir,targets,stale,...}
    GET  /api/telemetry              {connected,flying,battery,height_cm,...}
    POST /api/config/arena_safety    body {"enabled": false}

Note on the /api/rc body: the live server (``api_rc``) reads the short
keys ``lr/fb/ud/yaw``; the C2 contract documents the long keys
``left_right/forward_back/up_down/yaw``. We send BOTH so the request is
correct against the real endpoint and self-documenting per the contract.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("c2.fc")


class DroneFC:
    """Exception-safe async HTTP client for a single drone FC."""

    def __init__(self, drone_id: str, base_url: str, timeout: float = 4.0) -> None:
        self.drone_id = drone_id
        # Normalise: no trailing slash so f"{base}/api/..." is always clean.
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        # One persistent client per drone (connection pooling for the
        # 4 Hz command loop). Created eagerly; closed via close().
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ low-level
    async def _post(self, path: str, json: dict | None = None) -> dict:
        """POST ``path`` and return the decoded JSON dict.

        On any network/decode error return ``{"ok": False, "error": ...}``
        and log a warning. Never raises. If the server returns a non-dict
        JSON body (unexpected) it is wrapped so callers always get a dict.
        """
        try:
            resp = await self._client.post(path, json=(json or {}))
            return self._decode(resp, path)
        except Exception as e:  # noqa: BLE001 - must never propagate
            log.warning("[%s] POST %s failed: %s", self.drone_id, path, e)
            return {"ok": False, "error": str(e)}

    async def _get(self, path: str) -> dict:
        """GET ``path`` and return the decoded JSON dict. Never raises."""
        try:
            resp = await self._client.get(path)
            return self._decode(resp, path)
        except Exception as e:  # noqa: BLE001 - must never propagate
            log.warning("[%s] GET %s failed: %s", self.drone_id, path, e)
            return {"ok": False, "error": str(e)}

    def _decode(self, resp: "httpx.Response", path: str) -> dict:
        """Turn an httpx response into a plain dict.

        A non-2xx status is surfaced as ``{"ok": False, "error": ...}``
        but still carries the parsed body under ``"body"`` when present,
        so a caller can inspect a structured 4xx/5xx if it wants to.
        """
        body: Any = None
        try:
            body = resp.json()
        except Exception:  # body wasn't JSON (HTML error page, empty, ...)
            body = (resp.text or "").strip() or None

        if resp.status_code >= 400:
            log.warning(
                "[%s] %s -> HTTP %s", self.drone_id, path, resp.status_code
            )
            out = {"ok": False, "error": f"http {resp.status_code}"}
            if body is not None:
                out["body"] = body
            return out

        if isinstance(body, dict):
            return body
        # 2xx but non-dict JSON (or empty body): normalise to a dict.
        return {"ok": True, "body": body}

    # ------------------------------------------------------------------ config
    async def disable_arena_guard(self) -> bool:
        """Disable the Pi-side arena boundary guard (the C2 owns bounds).

        POST /api/config/arena_safety {"enabled": false}. Returns True
        only if the server confirms the guard is now off; False on any
        error or unexpected response.
        """
        try:
            resp = await self._client.post(
                "/api/config/arena_safety", json={"enabled": False}
            )
            data = self._decode(resp, "/api/config/arena_safety")
        except Exception as e:  # noqa: BLE001 - never propagate
            log.warning("[%s] disable_arena_guard failed: %s", self.drone_id, e)
            return False

        if not data.get("ok", False):
            log.warning(
                "[%s] disable_arena_guard not ok: %r", self.drone_id, data
            )
            return False
        # Server echoes the live guard state. Treat an explicit
        # enabled=False as success; if the field is absent fall back to
        # the ok flag (older servers may not echo it).
        enabled = data.get("enabled", False)
        success = (enabled is False)
        if not success:
            log.warning(
                "[%s] arena guard still enabled after disable: %r",
                self.drone_id, data,
            )
        return success

    # ------------------------------------------------------------------ flight
    async def takeoff(self) -> dict:
        """POST /api/takeoff (empty body)."""
        return await self._post("/api/takeoff")

    async def land(self) -> dict:
        """POST /api/land (empty body)."""
        return await self._post("/api/land")

    async def emergency(self) -> dict:
        """POST /api/emergency (empty body) -- motor cut, last resort."""
        return await self._post("/api/emergency")

    async def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> dict:
        """POST /api/rc with the four stick axes.

        Axes follow the FC convention (each in [-100, 100]):
          +left_right   = right
          +forward_back = forward
          +up_down      = up
          +yaw          = rotate clockwise

        We send both the long-form contract keys and the short keys the
        live ``api_rc`` endpoint actually reads (see module docstring).
        Values are coerced to int defensively.
        """
        lr_i, fb_i, ud_i, yaw_i = int(lr), int(fb), int(ud), int(yaw)
        body = {
            # documented C2 contract keys
            "left_right": lr_i,
            "forward_back": fb_i,
            "up_down": ud_i,
            "yaw": yaw_i,
            # short keys the live unified_api_server reads
            "lr": lr_i,
            "fb": fb_i,
            "ud": ud_i,
        }
        return await self._post("/api/rc", body)

    # ------------------------------------------------------------------ telemetry
    async def get_position(self) -> dict:
        """GET /api/position -> arena-frame pose + visible target slots."""
        return await self._get("/api/position")

    async def get_telemetry(self) -> dict:
        """GET /api/telemetry -> connection / flying / battery / height."""
        return await self._get("/api/telemetry")

    # ------------------------------------------------------------------ lifecycle
    async def close(self) -> None:
        """Close the underlying HTTP client. Idempotent; never raises."""
        try:
            await self._client.aclose()
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] close failed: %s", self.drone_id, e)

    # Convenient async-context-manager support (optional for callers).
    async def __aenter__(self) -> "DroneFC":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DroneFC(drone_id={self.drone_id!r}, base_url={self.base_url!r})"
