"""Long-term episodic storage with SQLite FTS5."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.models import EpisodicEntry
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


class EpisodicStore:
    """SQLite-backed episodic database with FTS5 full-text search."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.RLock()
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = encrypted_connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        return self._conn

    def _ensure_schema(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT '',
                tool_sequence TEXT NOT NULL DEFAULT '[]',
                success_score REAL NOT NULL DEFAULT 0.0,
                tags TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_episodes_score ON episodes(success_score);

            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                topic, content, tags, content='episodes', content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, topic, content, tags)
                VALUES (new.rowid, new.topic, new.content, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, topic, content, tags)
                VALUES ('delete', old.rowid, old.topic, old.content, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, topic, content, tags)
                VALUES ('delete', old.rowid, old.topic, old.content, old.tags);
                INSERT INTO episodes_fts(rowid, topic, content, tags)
                VALUES (new.rowid, new.topic, new.content, new.tags);
            END;

            CREATE TABLE IF NOT EXISTS episode_summaries (
                id TEXT PRIMARY KEY,
                period TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_learnings TEXT NOT NULL DEFAULT '[]'
            );
            """)
            conn.commit()

    def store_episode(
        self,
        session_id: str,
        topic: str,
        content: str,
        outcome: str = "",
        tool_sequence: list[str] | None = None,
        success_score: float = 0.0,
        tags: list[str] | None = None,
        episode_id: str | None = None,
    ) -> str:
        """Store an episode."""
        from jarvis.models import _new_id

        eid = episode_id or _new_id()
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO episodes (id, session_id, timestamp, topic, content, outcome,
                   tool_sequence, success_score, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eid,
                    session_id,
                    datetime.now(UTC).isoformat(),
                    topic,
                    content,
                    outcome,
                    json.dumps(tool_sequence or []),
                    success_score,
                    json.dumps(tags or []),
                ),
            )
            conn.commit()
        return eid

    def list_episodes(
        self,
        date_range: tuple[str, str] | None = None,
        limit: int = 100,
    ) -> list[EpisodicEntry]:
        """List episodes by date range without full-text search."""
        conditions = []
        params: list[Any] = []
        if date_range:
            conditions.append("timestamp >= ? AND timestamp <= ?")
            params.extend(date_range)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        try:
            with self._write_lock:
                conn = self._get_conn()
                cursor = conn.execute(
                    f"SELECT * FROM episodes {where} ORDER BY timestamp DESC LIMIT ?",
                    params,
                )
                return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as exc:
            log.warning("list_episodes_error", error=str(exc))
            return []

    def search_episodes(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        date_range: tuple[str, str] | None = None,
    ) -> list[EpisodicEntry]:
        """FTS5 full-text search across episodes."""
        # Sanitize FTS5 query: strip operators/special chars to prevent query injection
        import re

        _fts_clean = re.compile(r'["\(\)\*\:\^]')
        words = [_fts_clean.sub("", w) for w in query.strip().split()]
        words = [w for w in words if w and w.upper() not in ("AND", "OR", "NOT", "NEAR")]
        if not words:
            return []
        safe_query = " OR ".join(f'"{w}"' for w in words)

        # Build query
        conditions = ["episodes_fts MATCH ?"]
        params: list[Any] = [safe_query]

        if min_score > 0:
            conditions.append("e.success_score >= ?")
            params.append(min_score)

        if date_range:
            conditions.append("e.timestamp >= ? AND e.timestamp <= ?")
            params.extend(date_range)

        where = " AND ".join(conditions)
        params.append(limit)

        try:
            with self._write_lock:
                conn = self._get_conn()
                rows = conn.execute(
                    f"""SELECT e.* FROM episodes e
                        JOIN episodes_fts ON episodes_fts.rowid = e.rowid
                        WHERE {where}
                        ORDER BY rank
                        LIMIT ?""",
                    params,
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error - return empty
            return []

        return [self._row_to_entry(r) for r in rows]

    def get_similar_episodes(
        self,
        tool_sequence: list[str],
        limit: int = 5,
    ) -> list[EpisodicEntry]:
        """Find episodes with similar tool sequences."""
        with self._write_lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM episodes WHERE success_score > 0 ORDER BY success_score DESC"
            ).fetchall()

        # Score by Jaccard similarity of tool sequences
        target_set = set(tool_sequence)
        if not target_set:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            seq = set(json.loads(row["tool_sequence"]))
            if not seq:
                continue
            intersection = len(target_set & seq)
            union = len(target_set | seq)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > 0:
                scored.append((jaccard, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_entry(r) for _, r in scored[:limit]]

    def get_session_episodes(self, session_id: str) -> list[EpisodicEntry]:
        """All episodes of a session."""
        with self._write_lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM episodes WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_episode_count(self) -> int:
        """Total number of episodes."""
        with self._write_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
        return row["cnt"] if row else 0

    def store_summary(
        self,
        period: str,
        start_date: str,
        end_date: str,
        summary: str,
        key_learnings: list[str] | None = None,
    ) -> str:
        """Store a summary."""
        from jarvis.models import _new_id

        sid = _new_id()
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO episode_summaries
                   (id, period, start_date, end_date,
                    summary, key_learnings)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, period, start_date, end_date, summary, json.dumps(key_learnings or [])),
            )
            conn.commit()
        return sid

    def get_summaries(self, period: str | None = None) -> list[dict[str, Any]]:
        """Get summaries, optionally filtered by period."""
        with self._write_lock:
            conn = self._get_conn()
            if period:
                rows = conn.execute(
                    "SELECT * FROM episode_summaries WHERE period = ? ORDER BY start_date DESC",
                    (period,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM episode_summaries ORDER BY start_date DESC"
                ).fetchall()

        return [
            {
                "id": r["id"],
                "period": r["period"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "summary": r["summary"],
                "key_learnings": json.loads(r["key_learnings"]),
            }
            for r in rows
        ]

    def _row_to_entry(self, row: sqlite3.Row) -> EpisodicEntry:
        """Convert a DB row to an EpisodicEntry."""
        return EpisodicEntry(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            topic=row["topic"],
            content=row["content"],
            outcome=row["outcome"],
            tool_sequence=json.loads(row["tool_sequence"]),
            success_score=row["success_score"],
            tags=json.loads(row["tags"]),
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> EpisodicStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
