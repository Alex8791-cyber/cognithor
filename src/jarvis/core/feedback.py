"""User Feedback System -- thumbs up/down with optional comment.

Stores feedback in SQLite, provides aggregation for self-improvement.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["FeedbackStore"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT DEFAULT '',
    agent_name TEXT DEFAULT 'jarvis',
    channel TEXT DEFAULT '',
    user_message TEXT DEFAULT '',
    assistant_response TEXT DEFAULT '',
    tool_calls TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_agent ON feedback(agent_name);
"""


class FeedbackStore:
    """SQLite-backed feedback storage."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def submit(
        self,
        session_id: str,
        message_id: str,
        rating: int,
        *,
        comment: str = "",
        agent_name: str = "jarvis",
        channel: str = "",
        user_message: str = "",
        assistant_response: str = "",
        tool_calls: str = "",
    ) -> str:
        """Submit feedback. rating: 1 (thumbs up) or -1 (thumbs down)."""
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback (id, session_id, message_id, rating, comment, "
                "agent_name, channel, user_message, assistant_response, tool_calls, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feedback_id,
                    session_id,
                    message_id,
                    rating,
                    comment,
                    agent_name,
                    channel,
                    user_message[:2000],
                    assistant_response[:2000],
                    tool_calls[:1000],
                    time.time(),
                ),
            )
        log.info("feedback_submitted", id=feedback_id, rating=rating, agent=agent_name)
        return feedback_id

    def add_comment(self, feedback_id: str, comment: str) -> bool:
        """Add or update comment on existing feedback."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE feedback SET comment = ? WHERE id = ?",
                (comment[:2000], feedback_id),
            )
            return cursor.rowcount > 0

    def get_stats(self, agent_name: str = "", hours: int = 0) -> dict[str, Any]:
        """Get feedback statistics."""
        with self._conn() as conn:
            conditions: list[str] = []
            params: list[Any] = []
            if agent_name:
                conditions.append("agent_name = ?")
                params.append(agent_name)
            if hours > 0:
                conditions.append("created_at > ?")
                params.append(time.time() - hours * 3600)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            total = conn.execute(
                f"SELECT COUNT(*) FROM feedback {where}",  # noqa: S608
                params,
            ).fetchone()[0]

            pos_where = f"{where} {'AND' if where else 'WHERE'} rating > 0"
            positive = conn.execute(
                f"SELECT COUNT(*) FROM feedback {pos_where}",  # noqa: S608
                params,
            ).fetchone()[0]

            neg_where = f"{where} {'AND' if where else 'WHERE'} rating < 0"
            negative = conn.execute(
                f"SELECT COUNT(*) FROM feedback {neg_where}",  # noqa: S608
                params,
            ).fetchone()[0]

            com_where = f"{where} {'AND' if where else 'WHERE'} comment != ''"
            with_comment = conn.execute(
                f"SELECT COUNT(*) FROM feedback {com_where}",  # noqa: S608
                params,
            ).fetchone()[0]

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / total * 100, 1) if total > 0 else 0,
            "with_comment": with_comment,
        }

    def get_negative_feedback(self, limit: int = 20, agent_name: str = "") -> list[dict[str, Any]]:
        """Get recent negative feedback for learning."""
        with self._conn() as conn:
            if agent_name:
                rows = conn.execute(
                    "SELECT * FROM feedback WHERE rating < 0 AND agent_name = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (agent_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feedback WHERE rating < 0 ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent feedback entries."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
