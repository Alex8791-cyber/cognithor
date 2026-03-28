"""Multi-session manager with cross-session Core Memory.

Provides persistent session management with a ``sessions`` SQLite table.
Core Memory (a 6th tier above the existing 5) persists across sessions:
user preferences, agent personas, long-term goals. Max 2048 tokens,
never auto-trimmed.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    core_memory_snapshot TEXT DEFAULT '{}',
    platform TEXT NOT NULL DEFAULT ''
);
"""

CORE_MEMORY_MAX_TOKENS = 2048


@dataclass
class SessionInfo:
    """Metadata for a saved session."""

    id: str
    name: str
    created_at: str
    last_active: str
    platform: str = ""


class SessionManager:
    """Manages persistent multi-session state in SQLite.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = encrypted_connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SESSIONS_SCHEMA)
        self._conn.commit()

    def create_session(self, name: str, platform: str = "") -> SessionInfo:
        """Create a new session."""
        import sys

        session_id = uuid.uuid4().hex[:16]
        now = datetime.now(UTC).isoformat()
        plat = platform or sys.platform

        self._conn.execute(
            "INSERT INTO sessions (id, name, created_at, last_active, platform) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, name, now, now, plat),
        )
        self._conn.commit()
        log.info("session_created", session_id=session_id[:8], name=name)
        return SessionInfo(id=session_id, name=name, created_at=now, last_active=now, platform=plat)

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        """List sessions ordered by last active (most recent first)."""
        rows = self._conn.execute(
            "SELECT id, name, created_at, last_active, platform "
            "FROM sessions ORDER BY last_active DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            SessionInfo(
                id=r["id"],
                name=r["name"],
                created_at=r["created_at"],
                last_active=r["last_active"],
                platform=r["platform"] or "",
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get a specific session by ID."""
        row = self._conn.execute(
            "SELECT id, name, created_at, last_active, platform FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SessionInfo(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            platform=row["platform"] or "",
        )

    def touch_session(self, session_id: str) -> None:
        """Update last_active timestamp for a session."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?",
            (now, session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def save_core_memory(self, session_id: str, core_memory: dict[str, Any]) -> None:
        """Save the cross-session Core Memory snapshot for a session."""
        snapshot = json.dumps(core_memory, ensure_ascii=False)
        self._conn.execute(
            "UPDATE sessions SET core_memory_snapshot = ? WHERE id = ?",
            (snapshot, session_id),
        )
        self._conn.commit()

    def load_core_memory(self, session_id: str) -> dict[str, Any]:
        """Load the Core Memory snapshot for a session."""
        row = self._conn.execute(
            "SELECT core_memory_snapshot FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None or not row["core_memory_snapshot"]:
            return {}
        try:
            return json.loads(row["core_memory_snapshot"])
        except (json.JSONDecodeError, TypeError):
            return {}

    def session_count(self) -> int:
        """Return total number of sessions."""
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
