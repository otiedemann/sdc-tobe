"""SQLite-backed state for the Sphinx Control service.

Persists drone metadata across management-service restarts. The actual
Sphinx subprocesses are NOT persisted — they're respawned on demand if
the management service was the one that started them. Stored rows
become "stale" entries the launcher reconciles at startup (see
``Launcher.reconcile_on_startup``).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


_SCHEMA = """
CREATE TABLE IF NOT EXISTS environments (
    env_id      TEXT PRIMARY KEY,
    world_name  TEXT NOT NULL,
    world_app   TEXT NOT NULL,
    config_file TEXT,
    ue4_pid     INTEGER,
    status      TEXT NOT NULL DEFAULT 'unknown',
    started_at  REAL,
    stopped_at  REAL,
    last_error  TEXT
);
CREATE TABLE IF NOT EXISTS drones (
    drone_id     TEXT PRIMARY KEY,
    instance_id  INTEGER NOT NULL UNIQUE,
    drone_type   TEXT NOT NULL,
    descriptor   TEXT NOT NULL,
    world_app    TEXT NOT NULL,
    config_file  TEXT,
    firmware_url TEXT,
    drone_ip     TEXT NOT NULL,
    drone_port   INTEGER,
    sphinx_pid   INTEGER,
    ue4_pid      INTEGER,
    status       TEXT NOT NULL DEFAULT 'unknown',
    started_at   REAL,
    stopped_at   REAL,
    last_error   TEXT
);
CREATE INDEX IF NOT EXISTS idx_drones_status ON drones(status);
CREATE INDEX IF NOT EXISTS idx_envs_status ON environments(status);
"""


@dataclass
class EnvironmentRecord:
    """A running UE4 world (drone-agnostic). Singleton on Sphinx 2.x —
    ``firmwared`` and the Gz↔UE4 bridge claim fixed system paths so
    only one environment can exist at a time on this host."""

    env_id: str
    world_name: str
    world_app: str
    config_file: str | None = None
    ue4_pid: int | None = None
    status: str = "unknown"   # unknown | starting | running | stopped | error
    started_at: float | None = None
    stopped_at: float | None = None
    last_error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uptime_s"] = (
            time.time() - self.started_at
            if self.status == "running" and self.started_at
            else None
        )
        return d


@dataclass
class DroneRecord:
    drone_id: str
    instance_id: int
    drone_type: str
    descriptor: str
    world_app: str
    drone_ip: str
    config_file: str | None = None
    firmware_url: str | None = None
    drone_port: int | None = None
    sphinx_pid: int | None = None
    ue4_pid: int | None = None
    status: str = "unknown"  # unknown | spawning | running | stopped | error
    started_at: float | None = None
    stopped_at: float | None = None
    last_error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uptime_s"] = (
            time.time() - self.started_at
            if self.status == "running" and self.started_at
            else None
        )
        return d


class StateStore:
    """Thread-safe SQLite wrapper. Caller-supplied path; in-memory if
    empty/`:memory:`. All public methods acquire the lock — never call
    one from inside another, just inline the logic."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ─── drones table ────────────────────────────────────────────

    def upsert(self, rec: DroneRecord) -> None:
        with self._cursor() as c:
            c.execute(
                """
                INSERT INTO drones(
                    drone_id, instance_id, drone_type, descriptor,
                    world_app, config_file, firmware_url, drone_ip,
                    drone_port, sphinx_pid, ue4_pid, status,
                    started_at, stopped_at, last_error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(drone_id) DO UPDATE SET
                    instance_id  = excluded.instance_id,
                    drone_type   = excluded.drone_type,
                    descriptor   = excluded.descriptor,
                    world_app    = excluded.world_app,
                    config_file  = excluded.config_file,
                    firmware_url = excluded.firmware_url,
                    drone_ip     = excluded.drone_ip,
                    drone_port   = excluded.drone_port,
                    sphinx_pid   = excluded.sphinx_pid,
                    ue4_pid      = excluded.ue4_pid,
                    status       = excluded.status,
                    started_at   = excluded.started_at,
                    stopped_at   = excluded.stopped_at,
                    last_error   = excluded.last_error
                """,
                (
                    rec.drone_id, rec.instance_id, rec.drone_type, rec.descriptor,
                    rec.world_app, rec.config_file, rec.firmware_url, rec.drone_ip,
                    rec.drone_port, rec.sphinx_pid, rec.ue4_pid, rec.status,
                    rec.started_at, rec.stopped_at, rec.last_error,
                ),
            )

    def update_status(
        self,
        drone_id: str,
        status: str,
        sphinx_pid: int | None = None,
        ue4_pid: int | None = None,
        last_error: str | None = None,
    ) -> None:
        now = time.time()
        with self._cursor() as c:
            updates: list[str] = ["status = ?"]
            params: list[Any] = [status]
            if status == "running":
                updates.append("started_at = ?")
                params.append(now)
                updates.append("stopped_at = NULL")
            elif status == "stopped":
                updates.append("stopped_at = ?")
                params.append(now)
            if sphinx_pid is not None:
                updates.append("sphinx_pid = ?")
                params.append(sphinx_pid)
            if ue4_pid is not None:
                updates.append("ue4_pid = ?")
                params.append(ue4_pid)
            if last_error is not None:
                updates.append("last_error = ?")
                params.append(last_error)
            params.append(drone_id)
            c.execute(
                f"UPDATE drones SET {', '.join(updates)} WHERE drone_id = ?",
                params,
            )

    def get(self, drone_id: str) -> DroneRecord | None:
        with self._cursor() as c:
            row = c.execute(
                "SELECT * FROM drones WHERE drone_id = ?", (drone_id,)
            ).fetchone()
            return _row_to_record(row) if row else None

    def list_all(self, status: str | None = None) -> list[DroneRecord]:
        sql = "SELECT * FROM drones"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY instance_id"
        with self._cursor() as c:
            rows = c.execute(sql, params).fetchall()
            return [_row_to_record(r) for r in rows]

    def delete(self, drone_id: str) -> None:
        with self._cursor() as c:
            c.execute("DELETE FROM drones WHERE drone_id = ?", (drone_id,))

    def delete_stopped_at_instance_id(self, instance_id: int) -> int:
        """Drop any stopped rows holding ``instance_id``. Used right
        before inserting a new drone so the UNIQUE(instance_id) constraint
        doesn't reject the insert because a previous run's stopped row
        is still parked at that slot. Returns the number of rows
        deleted. Never touches running rows."""
        with self._cursor() as c:
            c.execute(
                "DELETE FROM drones "
                "WHERE instance_id = ? AND status = 'stopped'",
                (instance_id,),
            )
            return c.rowcount

    def used_instance_ids(self) -> set[int]:
        with self._cursor() as c:
            rows = c.execute(
                "SELECT instance_id FROM drones WHERE status != 'stopped'"
            ).fetchall()
            return {int(r["instance_id"]) for r in rows}

    # ─── environments table ─────────────────────────────────────

    def upsert_env(self, rec: EnvironmentRecord) -> None:
        with self._cursor() as c:
            c.execute(
                """
                INSERT INTO environments(
                    env_id, world_name, world_app, config_file,
                    ue4_pid, status, started_at, stopped_at, last_error
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(env_id) DO UPDATE SET
                    world_name  = excluded.world_name,
                    world_app   = excluded.world_app,
                    config_file = excluded.config_file,
                    ue4_pid     = excluded.ue4_pid,
                    status      = excluded.status,
                    started_at  = excluded.started_at,
                    stopped_at  = excluded.stopped_at,
                    last_error  = excluded.last_error
                """,
                (
                    rec.env_id, rec.world_name, rec.world_app, rec.config_file,
                    rec.ue4_pid, rec.status, rec.started_at, rec.stopped_at,
                    rec.last_error,
                ),
            )

    def update_env_status(
        self,
        env_id: str,
        status: str,
        ue4_pid: int | None = None,
        last_error: str | None = None,
    ) -> None:
        now = time.time()
        with self._cursor() as c:
            updates = ["status = ?"]
            params: list[Any] = [status]
            if status == "running":
                updates.append("started_at = ?")
                params.append(now)
                updates.append("stopped_at = NULL")
            elif status == "stopped":
                updates.append("stopped_at = ?")
                params.append(now)
            if ue4_pid is not None:
                updates.append("ue4_pid = ?")
                params.append(ue4_pid)
            if last_error is not None:
                updates.append("last_error = ?")
                params.append(last_error)
            params.append(env_id)
            c.execute(
                f"UPDATE environments SET {', '.join(updates)} WHERE env_id = ?",
                params,
            )

    def get_env(self, env_id: str) -> EnvironmentRecord | None:
        with self._cursor() as c:
            row = c.execute(
                "SELECT * FROM environments WHERE env_id = ?", (env_id,)
            ).fetchone()
            return _row_to_env(row) if row else None

    def current_env(self) -> EnvironmentRecord | None:
        """The currently-running environment, or None. Singleton on
        Sphinx 2.x; if multiple rows are running (shouldn't happen),
        returns the most recently started."""
        with self._cursor() as c:
            row = c.execute(
                "SELECT * FROM environments WHERE status = 'running' "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return _row_to_env(row) if row else None

    def list_envs(self) -> list[EnvironmentRecord]:
        with self._cursor() as c:
            rows = c.execute(
                "SELECT * FROM environments ORDER BY started_at DESC"
            ).fetchall()
            return [_row_to_env(r) for r in rows]

    def delete_env(self, env_id: str) -> None:
        with self._cursor() as c:
            c.execute("DELETE FROM environments WHERE env_id = ?", (env_id,))


def _row_to_record(row: sqlite3.Row) -> DroneRecord:
    d = dict(row)
    return DroneRecord(**d)


def _row_to_env(row: sqlite3.Row) -> EnvironmentRecord:
    d = dict(row)
    return EnvironmentRecord(**d)
