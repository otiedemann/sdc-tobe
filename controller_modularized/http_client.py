"""HTTP plumbing shared by every blueprint:
- ``_http_session`` — connection-pooled requests session (TCP keep-alive)
- ``pi_get``/``pi_post`` — call the active flight controller's API
- ``log_command`` — append every C2-side command to the per-flight logger
  + the optional debug command-log file
- ``_record_pi_call`` — track slow Pi calls for /proxy/diagnostics
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from controller_modularized import state
from controller_modularized.config import TIMEOUT_CMD


# ── Connection-pooled HTTP session ───────────────────────────────
_http_session = requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=16,
    pool_maxsize=64,
    max_retries=0,
)
_http_session.mount("http://", _adapter)
_http_session.mount("https://", _adapter)

# Thread pool for parallel heartbeats / fan-out requests
_heartbeat_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hb")


# ── Optional debug command log file ──────────────────────────────
command_log_enabled: bool = os.getenv("REMOTE_COMMAND_LOG", "0") in {"1", "true", "True"}
command_log_path: Path = Path(os.getenv("REMOTE_COMMAND_LOG_PATH",
                                         "remote_command_log.jsonl"))
command_log_last: dict[str, float] = {}


# ── Slow-call ring buffer (for /proxy/diagnostics) ───────────────
_slow_calls_lock = threading.Lock()
_slow_calls: list = []
_SLOW_CALL_THRESHOLD_S: float = 0.5


# Late-bound — the entrypoint creates the FlightLogger instance and
# assigns it here so log_command() can feed it without importing the
# entrypoint (which would cause a cycle).
flight_logger = None


def log_command(event: str, payload: dict | None = None) -> None:
    """Funnel a C2-side command into the per-flight logger and the
    debug command-log file. Cheap no-op when no drone is flying."""
    try:
        if flight_logger is not None:
            did = None
            if payload is not None:
                did = payload.get("id") or payload.get("drone_id")
            flight_logger.record_command(did, event, payload)
    except Exception:
        pass

    if not command_log_enabled:
        return
    try:
        now = time.time()
        key = event
        throttle_s = 0.0
        if event in {"key_down", "key_up"}:
            k = str((payload or {}).get("key", ""))
            key = f"{event}:{k}"
            throttle_s = 0.5
        last = command_log_last.get(key, 0.0)
        if throttle_s > 0 and (now - last) < throttle_s:
            return
        command_log_last[key] = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {"ts": ts, "event": event, "payload": payload or {}}
        command_log_path.parent.mkdir(parents=True, exist_ok=True)
        with command_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[REMOTE CMD] {ts} {event} payload={payload or {}}")
    except Exception:
        pass


def _record_pi_call(method: str, path: str, dt_s: float,
                    status: int | None, err: str | None = None) -> None:
    if dt_s < _SLOW_CALL_THRESHOLD_S and err is None:
        return
    rec = {
        "ts":       time.time(),
        "method":   method,
        "path":     path,
        "dt_ms":    int(dt_s * 1000),
        "status":   status,
        "error":    err,
        "drone_id": state.active_drone_id,
    }
    with _slow_calls_lock:
        _slow_calls.append(rec)
        if len(_slow_calls) > 200:
            del _slow_calls[:len(_slow_calls) - 200]
    print(f"[PI SLOW] {method} {path} {dt_s:.2f}s "
          f"status={status} err={err or '-'}")


def pi_post(path: str, body: dict | None = None,
            timeout: float | None = None):
    t0 = time.time()
    status = None
    err = None
    try:
        r = _http_session.post(f"{state.PI_BASE}{path}",
                               json=body or {},
                               timeout=TIMEOUT_CMD if timeout is None else timeout)
        status = r.status_code
        return r
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _record_pi_call("POST", path, time.time() - t0, status, err)


def pi_get(path: str, timeout: float | None = None):
    t0 = time.time()
    status = None
    err = None
    try:
        r = _http_session.get(f"{state.PI_BASE}{path}",
                              timeout=TIMEOUT_CMD if timeout is None else timeout)
        status = r.status_code
        return r
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _record_pi_call("GET", path, time.time() - t0, status, err)
