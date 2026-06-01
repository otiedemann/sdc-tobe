"""Thin async HTTP client for the marker_mission_c2 server.

Only the endpoints the strategy actually needs. Every call returns a
``(ok, payload)`` tuple to mirror the C2 server's own response shape and
avoid surprises in the runner.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class C2Client:
    """Async client. One instance per app run; uses a long-lived AsyncClient."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8090",
        timeout_s: float = 2.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout_s),
        )

    @property
    def base_url(self) -> str:
        return self._base

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def overview(self) -> Tuple[bool, Dict[str, Any]]:
        """``GET /api/c2/overview`` — returns {fc_name: {...state...}}."""
        try:
            r = await self._client.get("/api/c2/overview")
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("c2 overview failed: %s", e)
            return False, {}
        if not isinstance(data, dict):
            return False, {}
        return True, data

    async def fc_names(self) -> List[str]:
        """Convenience: list of fc_names known to the C2."""
        ok, ov = await self.overview()
        if not ok:
            return []
        return sorted(ov.keys())

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def push_script(self, fc_name: str, text: str) -> Tuple[bool, Any]:
        """``POST /api/c2/<fc>/script`` — set script without starting."""
        return await self._post_json(
            f"/api/c2/{fc_name}/script", {"text": text}
        )

    async def start(self, fc_name: str, script: Optional[str] = None) -> Tuple[bool, Any]:
        """``POST /api/c2/<fc>/start`` — start mission (with optional script)."""
        body: Dict[str, Any] = {}
        if script is not None:
            body["script"] = script
        return await self._post_json(f"/api/c2/{fc_name}/start", body)

    async def stop(self, fc_name: str) -> Tuple[bool, Any]:
        return await self._post_json(f"/api/c2/{fc_name}/stop", {})

    async def set_fc_endpoint(
        self, fc_name: str, host: Optional[str] = None,
        port: Optional[int] = None, reset: bool = False,
    ) -> Tuple[bool, Any]:
        """``POST /api/c2/<fc>/endpoint`` — re-point an FC at a real drone's IP
        (or back to the sim with ``reset=True``)."""
        if reset:
            body: Dict[str, Any] = {"reset": True}
        else:
            body = {"host": host}
            if port is not None:
                body["port"] = port
        return await self._post_json(f"/api/c2/{fc_name}/endpoint", body)

    async def emergency_land_all(self) -> Tuple[bool, Any]:
        return await self._post_json("/api/c2/emergency-land-all", {})

    async def push_to_many(
        self, fc_names: Iterable[str], text: str
    ) -> Tuple[bool, Any]:
        return await self._post_json(
            "/api/c2/scripts/push",
            {"fcs": list(fc_names), "text": text},
        )

    async def start_many(
        self, fc_names: Iterable[str], text: str
    ) -> Tuple[bool, Any]:
        return await self._post_json(
            "/api/c2/scripts/start",
            {"fcs": list(fc_names), "text": text},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_json(self, path: str, body: Dict[str, Any]) -> Tuple[bool, Any]:
        try:
            r = await self._client.post(path, json=body)
            r.raise_for_status()
            payload = r.json() if r.content else None
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("c2 POST %s failed: %s", path, e)
            return False, None
        # C2 server returns either {ok, payload, ...} or per-fc dicts.
        if isinstance(payload, dict) and "ok" in payload:
            return bool(payload.get("ok")), payload
        return True, payload
