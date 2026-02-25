"""SQLite Index-Management · FTS5 + Entitäten + Relationen + Vektoren. [B§4.7]

Zentrale Datenbank für alle Memory-Indizes.
Alles ableitbar aus den Markdown Source-of-Truth Dateien.

Thread-Safety:
SQLite im WAL-Modus unterstützt gleichzeitige Reads, aber nur einen
Writer. Bei Multi-Channel-Betrieb (z.B. Telegram + CLI + Cron) können
gleichzeitige Writes zu "database is locked" Fehlern führen.

Lösung: Alle Write-Operationen werden über ein threading.RLock
serialisiert. Reads benötigen kein Lock (WAL erlaubt concurrent reads).
RLock statt Lock, da manche Write-Methoden intern andere Write-Methoden
aufrufen (z.B. delete_entity → DELETE relations + DELETE entities).
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.models import Chunk, Entity, MemoryTier, Relation


def _serialize_vector(vec: list[float]) -> bytes:
    """Serialisiert einen Float-Vektor zu Bytes (für sqlite-vec Kompatibilität)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(data: bytes) -> list[float]:
    """Deserialisiert Bytes zurück zu Float-Vektor."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class MemoryIndex:
    """SQLite-basierter Index für das Memory-System.

    Schema:
    - chunks: Alle Text-Chunks mit Metadaten
    - chunks_fts: FTS5 Volltextsuche (BM25)
    - embeddings: Vektor-Embeddings (content_hash → vector)
    - entities: Wissens-Graph Entitäten
    - relations: Beziehungen zwischen Entitäten
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialisiert den Memory-Index mit SQLite-Datenbank."""
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.RLock()

    @property
    def db_path(self) -> Path:
        """Gibt den Pfad zur SQLite-Datenbank zurück."""
        return self._db_path

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection. Erstellt DB beim ersten Zugriff.

        Thread-safe: Connection-Erstellung ist über das Write-Lock geschützt,
        da _init_schema() Schreiboperationen ausführt.
        """
        if self._conn is None:
            with self._write_lock:
                # Double-check nach Lock-Erwerb (ein anderer Thread
                # könnte die Connection inzwischen erstellt haben)
                if self._conn is None:
                    self._db_path.parent.mkdir(parents=True, exist_ok=True)
                    self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA busy_timeout=5000")
                    self._conn.execute("PRAGMA foreign_keys=ON")
                    self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Erstellt alle Tabellen und Indizes."""
        c = self.conn
        c.executescript(
            """
            -- Chunks (alle Memory-Tiers)
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_path TEXT NOT NULL,
                line_start INTEGER DEFAULT 0,
                line_end INTEGER DEFAULT 0,
                content_hash TEXT NOT NULL,
                memory_tier TEXT NOT NULL,
                timestamp REAL,
                token_count INTEGER DEFAULT 0,
                entities_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_tier ON chunks(memory_tier);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_tier_created ON chunks(memory_tier, created_at);

            -- FTS5 Volltextsuche (BM25)
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );

            -- Trigger für FTS5 Sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES('delete', old.rowid, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES('delete', old.rowid, old.text);
                INSERT INTO chunks_fts(rowid, text)
                VALUES (new.rowid, new.text);
            END;

            -- Embeddings (content_hash → Vektor)
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at REAL NOT NULL
            );

            -- Entitäten (Wissens-Graph)
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                attributes_json TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_file);

            -- Relationen
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity TEXT NOT NULL REFERENCES entities(id),
                relation_type TEXT NOT NULL,
                target_entity TEXT NOT NULL REFERENCES entities(id),
                attributes_json TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '',
                created_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity);
        """
        )

    # ── Chunk Operations ─────────────────────────────────────────

    def upsert_chunk(self, chunk: Chunk) -> None:
        """Fügt einen Chunk ein oder aktualisiert ihn.

        NOTE: This method does NOT commit. Callers should use upsert_chunks()
        for batch operations, or call self.conn.commit() explicitly after
        one or more upsert_chunk() calls.
        """
        now = datetime.now().timestamp()
        ts = chunk.timestamp.timestamp() if chunk.timestamp else None

        self.conn.execute(
            """
            INSERT INTO chunks (id, text, source_path, line_start, line_end,
                              content_hash, memory_tier, timestamp, token_count,
                              entities_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text,
                source_path=excluded.source_path,
                line_start=excluded.line_start,
                line_end=excluded.line_end,
                content_hash=excluded.content_hash,
                memory_tier=excluded.memory_tier,
                timestamp=excluded.timestamp,
                token_count=excluded.token_count,
                entities_json=excluded.entities_json
            """,
            (
                chunk.id,
                chunk.text,
                chunk.source_path,
                chunk.line_start,
                chunk.line_end,
                chunk.content_hash,
                chunk.memory_tier.value,
                ts,
                chunk.token_count,
                json.dumps(chunk.entities),
                now,
            ),
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Batch-Insert von Chunks. Returns Anzahl eingefügt.

        Thread-safe: Geschützt über Write-Lock.
        """
        now = datetime.now().timestamp()
        rows = []
        for c in chunks:
            ts = c.timestamp.timestamp() if c.timestamp else None
            rows.append(
                (
                    c.id,
                    c.text,
                    c.source_path,
                    c.line_start,
                    c.line_end,
                    c.content_hash,
                    c.memory_tier.value,
                    ts,
                    c.token_count,
                    json.dumps(c.entities),
                    now,
                )
            )

        with self._write_lock:
            self.conn.executemany(
                """
                INSERT INTO chunks (id, text, source_path, line_start, line_end,
                                  content_hash, memory_tier, timestamp, token_count,
                                  entities_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    content_hash=excluded.content_hash,
                    memory_tier=excluded.memory_tier,
                    timestamp=excluded.timestamp,
                    token_count=excluded.token_count,
                    entities_json=excluded.entities_json
                """,
                rows,
            )
            self.conn.commit()
        return len(rows)

    def delete_chunks_by_source(self, source_path: str) -> int:
        """Löscht alle Chunks einer Quelldatei. Returns Anzahl gelöscht."""
        with self._write_lock:
            cursor = self.conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
            self.conn.commit()
            return cursor.rowcount

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Lädt einen Chunk anhand seiner ID."""
        row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Lädt mehrere Chunks anhand ihrer IDs in einem Batch.

        Nutzt SELECT ... WHERE id IN (...) statt einzelner Queries (N+1 vermeiden).
        Beachtet SQLite's Parameter-Limit von 999 und splittet bei Bedarf.

        Args:
            chunk_ids: Liste von Chunk-IDs.

        Returns:
            Dict mapping chunk_id → Chunk. Fehlende IDs werden ausgelassen.
        """
        if not chunk_ids:
            return {}

        result: dict[str, Chunk] = {}
        # SQLite hat ein Standard-Limit von 999 Parametern pro Query
        batch_size = 999

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                result[row["id"]] = self._row_to_chunk(row)

        return result

    def get_chunks_by_source(self, source_path: str) -> list[Chunk]:
        """Alle Chunks einer Quelldatei."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE source_path = ? ORDER BY line_start",
            (source_path,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_all_content_hashes(self) -> set[str]:
        """Alle content_hashes im Index (für Embedding-Cache)."""
        rows = self.conn.execute("SELECT DISTINCT content_hash FROM chunks").fetchall()
        return {r["content_hash"] for r in rows}

    def count_chunks(self, tier: MemoryTier | None = None) -> int:
        """Zählt Chunks, optional gefiltert nach Tier."""
        if tier:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM chunks WHERE memory_tier = ?",
                (tier.value,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    # ── BM25 Search ──────────────────────────────────────────────

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """FTS5 BM25 Suche.

        Args:
            query: Suchbegriffe.
            top_k: Maximale Ergebnisse.

        Returns:
            Liste von (chunk_id, bm25_score) Tupeln.
            Score ist negativ (FTS5 Konvention), wird hier positiv gemacht.
        """
        if not query.strip():
            return []

        # FTS5 Query: Wörter mit OR verbinden, Prefix-Match für deutsche Komposita
        words = query.strip().split()
        # Jedes Wort bekommt Prefix-Match (*) für bessere Treffer bei Komposita
        fts_query = " OR ".join(f"{w}*" for w in words)

        try:
            rows = self.conn.execute(
                """
                SELECT c.id, rank
                FROM chunks_fts fts
                JOIN chunks c ON c.rowid = fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback bei ungültiger FTS-Query
            return []

        # FTS5 rank ist negativ (kleiner = besser), wir machen ihn positiv
        return [(r["id"], -r["rank"]) for r in rows]

    # ── Embedding Operations ─────────────────────────────────────

    def store_embedding(
        self,
        content_hash: str,
        vector: list[float],
        model_name: str = "nomic-embed-text",
    ) -> None:
        """Speichert ein Embedding. Thread-safe."""
        now = datetime.now().timestamp()
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO embeddings (content_hash, vector, model_name, dimensions, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    vector=excluded.vector,
                    model_name=excluded.model_name,
                    dimensions=excluded.dimensions
                """,
                (content_hash, _serialize_vector(vector), model_name, len(vector), now),
            )
            self.conn.commit()

    def get_embedding(self, content_hash: str) -> list[float] | None:
        """Lädt ein Embedding anhand des Content-Hash."""
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_vector(row["vector"])

    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Lädt alle Embeddings. Returns {content_hash: vector}."""
        rows = self.conn.execute("SELECT content_hash, vector FROM embeddings").fetchall()
        return {r["content_hash"]: _deserialize_vector(r["vector"]) for r in rows}

    def count_embeddings(self) -> int:
        """Anzahl gespeicherter Embeddings."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM embeddings").fetchone()
        return row["cnt"] if row else 0

    # ── Entity Operations ────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        """Fügt eine Entität ein oder aktualisiert sie. Thread-safe."""
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO entities (id, type, name, attributes_json, source_file,
                                    created_at, updated_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    type=excluded.type,
                    name=excluded.name,
                    attributes_json=excluded.attributes_json,
                    source_file=excluded.source_file,
                    updated_at=excluded.updated_at,
                    confidence=excluded.confidence
                """,
                (
                    entity.id,
                    entity.type,
                    entity.name,
                    json.dumps(entity.attributes, default=str),
                    entity.source_file,
                    entity.created_at.timestamp(),
                    entity.updated_at.timestamp(),
                    entity.confidence,
                ),
            )
            self.conn.commit()

    def get_entity_by_id(self, entity_id: str) -> Entity | None:
        """Lädt eine Entität."""
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def search_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Sucht Entitäten nach Name und/oder Typ."""
        conditions: list[str] = []
        params: list[Any] = []

        if name:
            # Escape LIKE wildcards to prevent injection
            escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("name LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(
            f"SELECT * FROM entities WHERE {where} ORDER BY name", params
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def delete_entity(self, entity_id: str) -> bool:
        """Löscht eine Entität und ihre Relationen. Thread-safe."""
        with self._write_lock:
            self.conn.execute(
                "DELETE FROM relations WHERE source_entity = ? OR target_entity = ?",
                (entity_id, entity_id),
            )
            cursor = self.conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def count_entities(self) -> int:
        """Zählt die Anzahl der Entitäten im Index."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()
        return row["cnt"] if row else 0

    # ── Relation Operations ──────────────────────────────────────

    def upsert_relation(self, relation: Relation) -> None:
        """Fügt eine Relation ein oder aktualisiert sie. Thread-safe."""
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO relations (id, source_entity, relation_type, target_entity,
                                     attributes_json, source_file, created_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    relation_type=excluded.relation_type,
                    attributes_json=excluded.attributes_json,
                    confidence=excluded.confidence
                """,
                (
                    relation.id,
                    relation.source_entity,
                    relation.relation_type,
                    relation.target_entity,
                    json.dumps(relation.attributes, default=str),
                    relation.source_file,
                    relation.created_at.timestamp(),
                    relation.confidence,
                ),
            )
            self.conn.commit()

    def get_relations_for_entity(
        self,
        entity_id: str,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Alle Relationen einer Entität (als Quelle oder Ziel)."""
        if relation_type:
            rows = self.conn.execute(
                """SELECT * FROM relations
                   WHERE (source_entity = ? OR target_entity = ?)
                   AND relation_type = ?""",
                (entity_id, entity_id, relation_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relations WHERE source_entity = ? OR target_entity = ?",
                (entity_id, entity_id),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def graph_traverse(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> list[Entity]:
        """Traversiert den Wissens-Graph ab einer Entität.

        Args:
            entity_id: Start-Entität.
            max_depth: Maximale Tiefe.

        Returns:
            Alle erreichbaren Entitäten.
        """
        rows = self.conn.execute(
            """
            WITH RECURSIVE reachable AS (
                SELECT target_entity AS entity_id, 1 AS depth
                FROM relations WHERE source_entity = ?
                UNION ALL
                SELECT source_entity AS entity_id, 1 AS depth
                FROM relations WHERE target_entity = ?
                UNION ALL
                SELECT CASE
                    WHEN r.source_entity = rc.entity_id THEN r.target_entity
                    ELSE r.source_entity
                END, rc.depth + 1
                FROM relations r
                JOIN reachable rc ON (
                    r.source_entity = rc.entity_id
                    OR r.target_entity = rc.entity_id
                )
                WHERE rc.depth < ?
            )
            SELECT DISTINCT e.* FROM reachable rc
            JOIN entities e ON rc.entity_id = e.id
            WHERE e.id != ?
            """,
            (entity_id, entity_id, max_depth, entity_id),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def count_relations(self) -> int:
        """Zählt die Anzahl der Relationen im Index."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM relations").fetchone()
        return row["cnt"] if row else 0

    def get_chunks_with_entity_overlap(
        self,
        entity_ids: set[str],
    ) -> list[tuple[str, list[str]]]:
        """Find chunks whose entities_json contains any of the given entity IDs.

        Uses SQL LIKE filtering to avoid loading all chunks into Python.

        Args:
            entity_ids: Set of entity IDs to search for.

        Returns:
            List of (chunk_id, entity_list) tuples for chunks that have
            at least one entity in *entity_ids*.
        """
        if not entity_ids:
            return []

        # Build a WHERE clause using LIKE for each entity ID.
        # entities_json is a JSON array of strings, e.g. '["e1","e2"]'.
        # We use LIKE with the quoted entity ID to filter at the DB level.
        conditions = []
        params: list[str] = []
        for eid in entity_ids:
            # Escape LIKE wildcards in entity IDs
            escaped = eid.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("entities_json LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")

        where_clause = " OR ".join(conditions)
        query = (
            f"SELECT id, entities_json FROM chunks "
            f"WHERE entities_json != '[]' AND ({where_clause})"
        )
        rows = self.conn.execute(query, params).fetchall()

        results: list[tuple[str, list[str]]] = []
        for row in rows:
            chunk_entities = json.loads(row["entities_json"]) if row["entities_json"] else []
            results.append((row["id"], chunk_entities))
        return results

    # ── Maintenance ──────────────────────────────────────────────

    def rebuild_fts(self) -> None:
        """Baut den FTS5 Index komplett neu auf. Thread-safe."""
        with self._write_lock:
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self.conn.commit()

    def vacuum(self) -> None:
        """Komprimiert die Datenbank. Thread-safe."""
        with self._write_lock:
            self.conn.execute("VACUUM")

    def stats(self) -> dict[str, int]:
        """Statistiken über den Index."""
        return {
            "chunks": self.count_chunks(),
            "embeddings": self.count_embeddings(),
            "entities": self.count_entities(),
            "relations": self.count_relations(),
        }

    def close(self) -> None:
        """Schließt die Datenbankverbindung. Thread-safe."""
        with self._write_lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Row Conversion ───────────────────────────────────────────

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        """Konvertiert eine SQLite-Row in ein Chunk-Objekt."""
        ts = datetime.fromtimestamp(row["timestamp"]) if row["timestamp"] is not None else None
        entities = json.loads(row["entities_json"]) if row["entities_json"] else []
        return Chunk(
            id=row["id"],
            text=row["text"],
            source_path=row["source_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            content_hash=row["content_hash"],
            memory_tier=MemoryTier(row["memory_tier"]),
            timestamp=ts,
            token_count=row["token_count"],
            entities=entities,
        )

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        """Konvertiert eine SQLite-Row in ein Entity-Objekt."""
        attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
        return Entity(
            id=row["id"],
            type=row["type"],
            name=row["name"],
            attributes=attrs,
            source_file=row["source_file"],
            created_at=datetime.fromtimestamp(row["created_at"]),
            updated_at=datetime.fromtimestamp(row["updated_at"]),
            confidence=row["confidence"],
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        """Konvertiert eine SQLite-Row in ein Relation-Objekt."""
        attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
        return Relation(
            id=row["id"],
            source_entity=row["source_entity"],
            relation_type=row["relation_type"],
            target_entity=row["target_entity"],
            attributes=attrs,
            source_file=row["source_file"],
            created_at=datetime.fromtimestamp(row["created_at"]),
            confidence=row["confidence"],
        )
