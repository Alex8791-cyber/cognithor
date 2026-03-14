"""SQLite Database Backend.

Wrapped das bestehende sqlite3-Verhalten transparent.
Async-Methoden nutzen asyncio.to_thread um den Event Loop nicht zu blockieren.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS

logger = logging.getLogger("jarvis.db.sqlite")


class SQLiteBackend:
    """SQLite-Backend mit Row-Factory."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        encryption_key: str | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 0.1,
    ) -> None:
        self._db_path = str(db_path)
        self._encryption_key = encryption_key
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._conn: sqlite3.Connection | None = None
        self._ensure_connection()

    # ── Retry helper ────────────────────────────────────────────

    def _retry_on_locked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* with retry-on-locked backoff."""
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc):
                    raise
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt) * random.uniform(0.5, 1.0)
                    logger.warning(
                        "database is locked – retry %d/%d in %.3fs (%s)",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        self._db_path,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ── Connection ──────────────────────────────────────────────

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

            from jarvis.db.encryption import open_sqlite

            self._conn = open_sqlite(self._db_path, self._encryption_key)
            self._conn.row_factory = sqlite3.Row

            def _run_pragmas() -> None:
                assert self._conn is not None
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
                self._conn.execute("PRAGMA foreign_keys=ON")

            self._retry_on_locked(_run_pragmas)
            logger.info("SQLite-Verbindung hergestellt: %s", self._db_path)
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Direkter Zugriff auf die Verbindung (fuer Legacy-Code)."""
        return self._ensure_connection()

    # ── Sync-Hilfsmethoden (für asyncio.to_thread) ──────────────

    def _execute_sync(self, query: str, params: Sequence[Any] = ()) -> Any:
        conn = self._ensure_connection()

        def _do() -> Any:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor

        return self._retry_on_locked(_do)

    def _executemany_sync(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        conn = self._ensure_connection()

        def _do() -> None:
            conn.executemany(query, params_seq)
            conn.commit()

        self._retry_on_locked(_do)

    def _executescript_sync(self, script: str) -> None:
        conn = self._ensure_connection()
        self._retry_on_locked(conn.executescript, script)

    def _fetchone_sync(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        conn = self._ensure_connection()

        def _do() -> dict[str, Any] | None:
            row = conn.execute(query, params).fetchone()
            if row is None:
                return None
            return dict(row)

        return self._retry_on_locked(_do)

    def _fetchall_sync(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._ensure_connection()

        def _do() -> list[dict[str, Any]]:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

        return self._retry_on_locked(_do)

    def _commit_sync(self) -> None:
        if self._conn:
            self._conn.commit()

    def _close_sync(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite-Verbindung geschlossen: %s", self._db_path)

    # ── Async-Methoden (wrappen sync via to_thread) ─────────────

    async def execute(self, query: str, params: Sequence[Any] = ()) -> Any:
        return await asyncio.to_thread(self._execute_sync, query, params)

    async def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        await asyncio.to_thread(self._executemany_sync, query, params_seq)

    async def executescript(self, script: str) -> None:
        await asyncio.to_thread(self._executescript_sync, script)

    async def fetchone(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._fetchone_sync, query, params)

    async def fetchall(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetchall_sync, query, params)

    async def commit(self) -> None:
        await asyncio.to_thread(self._commit_sync)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    @property
    def placeholder(self) -> str:
        return "?"

    @property
    def backend_type(self) -> str:
        return "sqlite"
