"""SQLite Database Backend.

Wrapped das bestehende sqlite3-Verhalten transparent.
Async-Methoden nutzen asyncio.to_thread um den Event Loop nicht zu blockieren.
"""
from __future__ import annotations

import asyncio
import sqlite3
import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("jarvis.db.sqlite")


class SQLiteBackend:
    """SQLite-Backend mit Row-Factory."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_connection()

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            logger.info("SQLite-Verbindung hergestellt: %s", self._db_path)
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Direkter Zugriff auf die Verbindung (fuer Legacy-Code)."""
        return self._ensure_connection()

    # ── Sync-Hilfsmethoden (für asyncio.to_thread) ──────────────

    def _execute_sync(self, query: str, params: Sequence[Any] = ()) -> Any:
        conn = self._ensure_connection()
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor

    def _executemany_sync(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        conn = self._ensure_connection()
        conn.executemany(query, params_seq)
        conn.commit()

    def _executescript_sync(self, script: str) -> None:
        conn = self._ensure_connection()
        conn.executescript(script)

    def _fetchone_sync(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        conn = self._ensure_connection()
        row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return dict(row)

    def _fetchall_sync(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._ensure_connection()
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

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
