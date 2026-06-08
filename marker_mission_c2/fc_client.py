"""Per-FC async HTTP client built on a single shared ``httpx.AsyncClient``.

Every method returns ``(ok, payload)`` so callers never need ``try/except``.
Errors are logged at WARNING with the FC name prefix; the C2 keeps polling.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import httpx

from .config import FCSpec

Result = Tuple[bool, Any]


def make_client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s, connect=timeout_s),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=24),
        follow_redirects=False,
    )


class AsyncFCClient:
    def __init__(self, spec: FCSpec, http: httpx.AsyncClient,
                 default_timeout_s: float = 2.0):
        self.spec = spec
        self.http = http
        self.default_timeout_s = default_timeout_s
        self.log = logging.getLogger(f"c2.fc.{spec.name}")

    # ------------------------------------------------------------------ utils
    @property
    def base(self) -> str:
        return self.spec.base_url

    async def _get(self, path: str, *, timeout: Optional[float] = None,
                   raw: bool = False) -> Result:
        url = f"{self.base}{path}"
        try:
            r = await self.http.get(url, timeout=timeout
                                    or self.default_timeout_s)
        except httpx.HTTPError as e:
            self.log.warning("GET %s failed: %s", path, e)
            return False, str(e)
        if r.status_code >= 400:
            text = r.text[:200]
            self.log.warning("GET %s -> %d (%s)", path, r.status_code, text)
            return False, {"status": r.status_code, "body": text}
        if raw:
            return True, r.content
        try:
            return True, r.json()
        except ValueError:
            return True, r.text

    async def _post(self, path: str, *,
                    json_body: Any = None,
                    raw_body: Optional[bytes] = None,
                    content_type: Optional[str] = None,
                    timeout: Optional[float] = None,
                    ok_status: tuple[int, ...] = (200, 201, 204)) -> Result:
        url = f"{self.base}{path}"
        headers = {}
        kwargs: dict = {"timeout": timeout or self.default_timeout_s}
        if raw_body is not None:
            kwargs["content"] = raw_body
            headers["Content-Type"] = content_type or "application/octet-stream"
        elif json_body is not None:
            kwargs["json"] = json_body
        else:
            kwargs["json"] = {}
        if headers:
            kwargs["headers"] = headers
        try:
            r = await self.http.post(url, **kwargs)
        except httpx.HTTPError as e:
            self.log.warning("POST %s failed: %s", path, e)
            return False, str(e)
        if r.status_code in ok_status:
            try:
                return True, r.json()
            except ValueError:
                return True, r.text
        # Surface the body so callers (e.g. emergency-land) can decide whether
        # a 409 means "already stopped" rather than a real failure.
        try:
            body = r.json()
        except ValueError:
            body = {"body": r.text[:200]}
        self.log.warning("POST %s -> %d (%s)", path, r.status_code, body)
        return False, {"status": r.status_code, "body": body}

    async def _delete(self, path: str, *,
                      timeout: Optional[float] = None) -> Result:
        url = f"{self.base}{path}"
        try:
            r = await self.http.delete(url, timeout=timeout
                                       or self.default_timeout_s)
        except httpx.HTTPError as e:
            self.log.warning("DELETE %s failed: %s", path, e)
            return False, str(e)
        if r.status_code in (200, 204):
            try:
                return True, r.json()
            except ValueError:
                return True, {}
        try:
            body = r.json()
        except ValueError:
            body = {"body": r.text[:200]}
        self.log.warning("DELETE %s -> %d (%s)", path, r.status_code, body)
        return False, {"status": r.status_code, "body": body}

    # --------------------------------------------------------- telemetry &c.
    async def get_state(self) -> Result:
        return await self._get("/api/state")

    async def get_identity(self) -> Result:
        return await self._get("/api/identity")

    async def get_wlan(self) -> Result:
        """The FC host's WLAN interfaces (iwconfig): {iface: {ssid, signal_dbm,
        band, ...}}. The wl* interface's SSID is the connected drone's WiFi
        name. Polled at a slow cadence by the pool."""
        return await self._get("/api/wlan")

    # --------------------------------------------------------- mission ctrl
    async def start_mission(self, script: Optional[str] = None) -> Result:
        body: dict = {}
        if script is not None:
            body["script"] = script
        return await self._post("/api/start", json_body=body, timeout=5.0)

    async def stop_mission(self) -> Result:
        # marker_mission's /api/stop returns 409 when the mission is not
        # running. The caller treats 409 as a no-op so the C2's
        # emergency-land-all dashboard reports green for FCs that were
        # already idle.
        return await self._post(
            "/api/stop", timeout=5.0, ok_status=(200, 409))

    # ------------------------------------------------------------- arena
    async def get_active_arena(self) -> Result:
        return await self._get("/api/arena/active")

    async def set_active_arena(self, payload: dict) -> Result:
        return await self._post("/api/arena/active", json_body=payload,
                                timeout=4.0)

    # ------------------------------------------------------------- tune
    async def get_tune(self) -> Result:
        return await self._get("/api/tune")

    async def apply_tune(self, updates: dict) -> Result:
        return await self._post("/api/tune/apply", json_body=updates,
                                timeout=4.0)

    async def save_tune(self) -> Result:
        return await self._post("/api/tune/save", timeout=3.0)

    # ------------------------------------------------------- mission script
    async def get_mission_script(self) -> Result:
        return await self._get("/api/mission/script")

    async def set_mission_script(self, text: str) -> Result:
        return await self._post("/api/mission/script",
                                json_body={"text": text}, timeout=3.0)

    async def splice_mission(self, script: str) -> Result:
        """Live-replace the FC's REMAINING mission steps WITHOUT landing
        (re-target an airborne drone). The FC keeps the current step running
        and walks into the new tail. A non-200 (e.g. 409 'not airborne' on an
        older FC without the splice endpoint) returns ok=False so the caller
        falls back to stop->relaunch."""
        return await self._post("/api/mission/splice",
                                json_body={"script": script}, timeout=5.0)

    async def list_mission_scripts(self) -> Result:
        """List the FC's named mission scripts (its
        ``~/.marker_mission/mission_scripts/*.txt`` directory)."""
        return await self._get("/api/mission/scripts", timeout=4.0)

    async def load_mission_script_named(self, name: str) -> Result:
        """Fetch a specific named mission script from the FC. The FC's
        /load endpoint both reads the file and applies it as the
        active draft; from the C2's POV we just want the text so we
        can show / re-broadcast it."""
        return await self._post(
            f"/api/mission/scripts/{name}/load", timeout=4.0)

    # ------------------------------------------------------- calibration
    async def list_calibrations(self) -> Result:
        return await self._get("/api/calibrate/files", timeout=4.0)

    async def fetch_calibration(self, name: str) -> Result:
        return await self._get(f"/api/calibrate/files/{name}",
                               timeout=8.0, raw=True)

    async def push_calibration(self, name: str, blob: bytes) -> Result:
        return await self._post(f"/api/calibrate/files/{name}",
                                raw_body=blob, timeout=8.0)

    async def delete_calibration(self, name: str) -> Result:
        return await self._delete(f"/api/calibrate/files/{name}",
                                  timeout=4.0)
