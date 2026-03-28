"""Knowledge lineage tracking -- provenance for every learned fact."""

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
class LineageEntry:
    """A single provenance record for an entity change."""

    id: str
    entity_id: str
    source_type: str  # file, web, conversation, feedback, exploration
    source_path: str  # file path, URL, session ID
    action: str  # created, updated, confidence_changed, verified, decayed
    old_value: str = ""
    new_value: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    timestamp: str = ""


class KnowledgeLineageTracker:
    """SQLite-backed lineage tracker for knowledge provenance."""

    def __init__(
        self,
        db_path: str | Path | None = None,
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".jarvis" / "memory" / "knowledge_lineage.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with encrypted_connect(self._db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS lineage (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_path TEXT DEFAULT '',
                    action TEXT NOT NULL,
                    old_value TEXT DEFAULT '',
                    new_value TEXT DEFAULT '',
                    confidence_before REAL DEFAULT 0.0,
                    confidence_after REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL
                )""",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_entity ON lineage(entity_id)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_action ON lineage(action)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_time ON lineage(timestamp)",
            )

    def record(
        self,
        entity_id: str,
        source_type: str,
        action: str,
        **kwargs: Any,
    ) -> LineageEntry:
        """Record a lineage entry for an entity change."""
        entry = LineageEntry(
            id=str(uuid4()),
            entity_id=entity_id,
            source_type=source_type,
            action=action,
            source_path=kwargs.get("source_path", ""),
            old_value=kwargs.get("old_value", ""),
            new_value=kwargs.get("new_value", ""),
            confidence_before=kwargs.get(
                "confidence_before",
                0.0,
            ),
            confidence_after=kwargs.get(
                "confidence_after",
                0.0,
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )
        with encrypted_connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO lineage "
                "(id, entity_id, source_type, source_path, "
                "action, old_value, new_value, "
                "confidence_before, confidence_after, "
                "timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.entity_id,
                    entry.source_type,
                    entry.source_path,
                    entry.action,
                    entry.old_value,
                    entry.new_value,
                    entry.confidence_before,
                    entry.confidence_after,
                    entry.timestamp,
                ),
            )
        return entry

    def get_entity_lineage(
        self,
        entity_id: str,
        limit: int = 50,
    ) -> list[LineageEntry]:
        """Get all lineage entries for an entity."""
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM lineage WHERE entity_id = ? ORDER BY timestamp DESC LIMIT ?",
                (entity_id, limit),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def get_recent(
        self,
        limit: int = 100,
    ) -> list[LineageEntry]:
        """Get most recent lineage entries."""
        with encrypted_connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM lineage ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return lineage statistics."""
        with encrypted_connect(self._db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM lineage",
            ).fetchone()[0]
            entities = conn.execute(
                "SELECT COUNT(DISTINCT entity_id) FROM lineage",
            ).fetchone()[0]
            by_action: dict[str, int] = {}
            for row in conn.execute(
                "SELECT action, COUNT(*) FROM lineage GROUP BY action",
            ).fetchall():
                by_action[row[0]] = row[1]
            by_source: dict[str, int] = {}
            for row in conn.execute(
                "SELECT source_type, COUNT(*) FROM lineage GROUP BY source_type",
            ).fetchall():
                by_source[row[0]] = row[1]
            return {
                "total_entries": total,
                "entities_tracked": entities,
                "by_action": by_action,
                "by_source": by_source,
            }

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LineageEntry:
        return LineageEntry(
            **{k: row[k] for k in row.keys()},  # noqa: SIM118
        )
