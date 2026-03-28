"""Question-Answer knowledge base with confidence tracking."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QAPair:
    """A single question-answer pair with metadata."""

    id: str
    question: str
    answer: str
    topic: str = ""
    confidence: float = 0.5
    source: str = ""
    entity_id: str = ""
    created_at: str = ""
    last_verified: str = ""
    verification_count: int = 0


class KnowledgeQAStore:
    """SQLite-backed Q&A knowledge base."""

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".jarvis" / "memory" / "knowledge_qa.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with encrypted_connect(self._db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS qa_pairs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    topic TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    source TEXT DEFAULT '',
                    entity_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_verified TEXT DEFAULT '',
                    verification_count INTEGER DEFAULT 0
                )""",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_topic ON qa_pairs(topic)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_confidence ON qa_pairs(confidence)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_entity ON qa_pairs(entity_id)",
            )

    def add(
        self,
        question: str,
        answer: str,
        *,
        topic: str = "",
        confidence: float = 0.5,
        source: str = "",
        entity_id: str = "",
    ) -> QAPair:
        """Insert a new Q&A pair."""
        qa = QAPair(
            id=str(uuid4()),
            question=question,
            answer=answer,
            topic=topic,
            confidence=confidence,
            source=source,
            entity_id=entity_id,
            created_at=datetime.now(UTC).isoformat(),
        )
        with encrypted_connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO qa_pairs "
                "(id, question, answer, topic, confidence, "
                "source, entity_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    qa.id,
                    qa.question,
                    qa.answer,
                    qa.topic,
                    qa.confidence,
                    qa.source,
                    qa.entity_id,
                    qa.created_at,
                ),
            )
        return qa

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[QAPair]:
        """Search Q&A pairs by question, answer, or topic."""
        pattern = f"%{query}%"
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_pairs "
                "WHERE question LIKE ? "
                "OR answer LIKE ? "
                "OR topic LIKE ? "
                "ORDER BY confidence DESC LIMIT ?",
                (pattern, pattern, pattern, limit),
            ).fetchall()
            return [self._row_to_qa(r) for r in rows]

    def get_by_topic(
        self,
        topic: str,
        limit: int = 50,
    ) -> list[QAPair]:
        """Get all Q&A pairs for a topic."""
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_pairs WHERE topic = ? ORDER BY confidence DESC LIMIT ?",
                (topic, limit),
            ).fetchall()
            return [self._row_to_qa(r) for r in rows]

    def get_by_entity(
        self,
        entity_id: str,
    ) -> list[QAPair]:
        """Get all Q&A pairs linked to an entity."""
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_pairs WHERE entity_id = ? ORDER BY confidence DESC",
                (entity_id,),
            ).fetchall()
            return [self._row_to_qa(r) for r in rows]

    def update_confidence(
        self,
        qa_id: str,
        new_confidence: float,
    ) -> bool:
        """Update the confidence score for a Q&A pair."""
        with encrypted_connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE qa_pairs SET confidence = ? WHERE id = ?",
                (new_confidence, qa_id),
            )
            return cur.rowcount > 0

    def verify(self, qa_id: str) -> bool:
        """Mark a Q&A pair as verified, boosting confidence."""
        now = datetime.now(UTC).isoformat()
        with encrypted_connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE qa_pairs "
                "SET last_verified = ?, "
                "verification_count = verification_count + 1, "
                "confidence = MIN(1.0, confidence + 0.1) "
                "WHERE id = ?",
                (now, qa_id),
            )
            return cur.rowcount > 0

    def delete(self, qa_id: str) -> bool:
        """Delete a Q&A pair."""
        with encrypted_connect(self._db_path) as conn:
            cur = conn.execute(
                "DELETE FROM qa_pairs WHERE id = ?",
                (qa_id,),
            )
            return cur.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        with encrypted_connect(self._db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM qa_pairs",
            ).fetchone()[0]
            avg_conf = conn.execute(
                "SELECT COALESCE(AVG(confidence), 0) FROM qa_pairs",
            ).fetchone()[0]
            topics = conn.execute(
                "SELECT COUNT(DISTINCT topic) FROM qa_pairs WHERE topic != ''",
            ).fetchone()[0]
            return {
                "total_pairs": total,
                "avg_confidence": round(avg_conf, 3),
                "topics": topics,
            }

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QAPair]:
        """List Q&A pairs with pagination."""
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_pairs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_qa(r) for r in rows]

    @staticmethod
    def _row_to_qa(row: sqlite3.Row) -> QAPair:
        return QAPair(**{k: row[k] for k in row.keys()})  # noqa: SIM118
