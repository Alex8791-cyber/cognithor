"""GoalScopedIndex -- per-goal isolated vector + entity storage with global fallback."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["GoalScopedIndex"]


@dataclass
class IndexedChunk:
    """A chunk stored in the goal-scoped index."""

    id: str = ""
    text: str = ""
    source_url: str = ""
    goal_slug: str = ""
    created_at: str = ""


class GoalScopedIndex:
    """Per-goal isolated storage for chunks and entities.

    Each goal gets:
    - SQLite DB for chunks (full-text searchable)
    - SQLite DB for entities + relations (scoped)

    Global memory system receives copies for cross-domain discovery.
    """

    def __init__(self, goal_slug: str, base_dir: Path | str) -> None:
        self._goal_slug = goal_slug
        self._base_dir = Path(base_dir) / goal_slug
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Per-goal chunk DB
        self._chunk_db_path = self._base_dir / "chunks.db"
        self._chunk_conn = encrypted_connect(
            str(self._chunk_db_path), check_same_thread=False
        )
        self._chunk_conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_chunk_schema()

        # Per-goal entity DB
        self._entity_db_path = self._base_dir / "entities.db"
        self._entity_conn = encrypted_connect(
            str(self._entity_db_path), check_same_thread=False
        )
        self._entity_conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_entity_schema()

    def _ensure_chunk_schema(self) -> None:
        self._chunk_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                goal_slug TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            )
        """
        )
        self._chunk_conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, content=chunks, content_rowid=rowid)
        """
        )
        self._chunk_conn.commit()

    def _ensure_entity_schema(self) -> None:
        self._entity_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT DEFAULT '',
                attributes TEXT DEFAULT '{}',
                source_url TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            )
        """
        )
        self._entity_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_name TEXT NOT NULL,
                attributes TEXT DEFAULT '{}',
                created_at TEXT DEFAULT ''
            )
        """
        )
        self._entity_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_name)
        """
        )
        self._entity_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_name)
        """
        )
        self._entity_conn.commit()

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _new_id(self) -> str:
        import uuid

        return uuid.uuid4().hex[:16]

    # -- Chunk operations ------------------------------------------------

    def add_chunk(self, text: str, source_url: str = "") -> str:
        """Add a text chunk to the goal-scoped index. Returns chunk ID."""
        chunk_id = self._new_id()
        now = self._now()
        self._chunk_conn.execute(
            "INSERT INTO chunks (id, text, source_url, goal_slug, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, text, source_url, self._goal_slug, now),
        )
        # Update FTS index
        self._chunk_conn.execute(
            "INSERT INTO chunks_fts (rowid, text) "
            "VALUES (last_insert_rowid(), ?)",
            (text,),
        )
        self._chunk_conn.commit()
        return chunk_id

    def search_chunks(self, query: str, limit: int = 10) -> list[IndexedChunk]:
        """Full-text search within this goal's chunks."""
        try:
            rows = self._chunk_conn.execute(
                """SELECT c.id, c.text, c.source_url, c.goal_slug, c.created_at
                   FROM chunks c
                   JOIN chunks_fts f ON c.rowid = f.rowid
                   WHERE chunks_fts MATCH ?
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
        except Exception:
            # Fallback: LIKE search if FTS fails
            rows = self._chunk_conn.execute(
                "SELECT id, text, source_url, goal_slug, created_at "
                "FROM chunks WHERE text LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [
            IndexedChunk(
                id=r[0],
                text=r[1],
                source_url=r[2],
                goal_slug=r[3],
                created_at=r[4],
            )
            for r in rows
        ]

    def chunk_count(self) -> int:
        row = self._chunk_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    # -- Entity operations -----------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: dict | None = None,
        source_url: str = "",
    ) -> None:
        """Add or update an entity in the goal-scoped index."""
        attrs_json = json.dumps(attributes or {}, ensure_ascii=False)
        self._entity_conn.execute(
            """INSERT OR REPLACE INTO entities
               (name, entity_type, attributes, source_url, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (name, entity_type, attrs_json, source_url, self._now()),
        )
        self._entity_conn.commit()

    def add_relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        attributes: dict | None = None,
    ) -> None:
        """Add a relation to the goal-scoped index."""
        # Avoid exact duplicates
        existing = self._entity_conn.execute(
            "SELECT id FROM relations "
            "WHERE source_name=? AND relation_type=? AND target_name=? LIMIT 1",
            (source_name, relation_type, target_name),
        ).fetchone()
        if existing:
            return
        attrs_json = json.dumps(attributes or {}, ensure_ascii=False)
        self._entity_conn.execute(
            "INSERT INTO relations "
            "(source_name, relation_type, target_name, attributes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_name, relation_type, target_name, attrs_json, self._now()),
        )
        self._entity_conn.commit()

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get an entity by name from the goal-scoped index."""
        row = self._entity_conn.execute(
            "SELECT name, entity_type, attributes, source_url "
            "FROM entities WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "type": row[1],
            "attributes": json.loads(row[2] or "{}"),
            "source_url": row[3],
        }

    def entity_count(self) -> int:
        row = self._entity_conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()
        return row[0] if row else 0

    def relation_count(self) -> int:
        row = self._entity_conn.execute(
            "SELECT COUNT(*) FROM relations"
        ).fetchone()
        return row[0] if row else 0

    def get_entity_relations(self, name: str) -> list[dict[str, str]]:
        """Get all relations involving an entity."""
        rows = self._entity_conn.execute(
            "SELECT source_name, relation_type, target_name "
            "FROM relations WHERE source_name=? OR target_name=?",
            (name, name),
        ).fetchall()
        return [
            {"source": r[0], "relation": r[1], "target": r[2]} for r in rows
        ]

    # -- Stats -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "goal_slug": self._goal_slug,
            "chunks": self.chunk_count(),
            "entities": self.entity_count(),
            "relations": self.relation_count(),
            "path": str(self._base_dir),
        }

    def close(self) -> None:
        self._chunk_conn.close()
        self._entity_conn.close()
