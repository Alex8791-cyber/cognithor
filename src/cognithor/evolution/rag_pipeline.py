"""EvolutionRAG -- lightweight RAG pipeline for evolution knowledge retrieval.

Uses SQLite for chunk storage and simple TF-IDF-style keyword matching
for retrieval (no external embedding dependencies).  Documents are split
into paragraph-based chunks at ingest time.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.evolution.autonomous_learner import LearningTask

log = get_logger(__name__)

__all__ = [
    "EvolutionRAG",
    "RAGChunk",
    "RAGDocument",
    "RAGResult",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class RAGDocument:
    title: str
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    ingested_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "content": self.content[:500],
            "metadata": self.metadata,
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RAGDocument:
        return cls(
            id=d.get("id", _new_id()),
            title=d["title"],
            source=d.get("source", ""),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            ingested_at=d.get("ingested_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class RAGChunk:
    document_id: str
    text: str
    chunk_index: int = 0
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text[:500],
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RAGChunk:
        return cls(
            id=d.get("id", _new_id()),
            document_id=d["document_id"],
            text=d["text"],
            chunk_index=d.get("chunk_index", 0),
        )


@dataclass
class RAGResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    title: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text[:500],
            "score": self.score,
            "title": self.title,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RAGResult:
        return cls(
            chunk_id=d["chunk_id"],
            document_id=d["document_id"],
            text=d["text"],
            score=d.get("score", 0.0),
            title=d.get("title", ""),
            source=d.get("source", ""),
        )


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_MIN_CHUNK_CHARS = 80
_MAX_CHUNK_CHARS = 2000


def _chunk_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraph-based chunks.

    Adjacent short paragraphs are merged up to *_MAX_CHUNK_CHARS*.
    Paragraphs shorter than *_MIN_CHUNK_CHARS* are joined with the next.
    """
    paragraphs = _PARAGRAPH_SPLIT.split(text.strip())
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 1 > _MAX_CHUNK_CHARS and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}".strip() if buf else para
    if buf and len(buf) >= _MIN_CHUNK_CHARS:
        chunks.append(buf)
    elif buf and chunks:
        # Merge tiny trailing chunk into the last one
        chunks[-1] = f"{chunks[-1]}\n{buf}"
    elif buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Simple keyword scoring (TF overlap)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "und",
        "oder",
        "aber",
        "in",
        "von",
        "zu",
        "mit",
        "auf",
        "fuer",
        "den",
        "dem",
        "ist",
        "es",
        "nicht",
        "als",
        "auch",
        "an",
        "werden",
        "nach",
        "aus",
        "bei",
        "ueber",
        "wie",
        "am",
        "er",
        "sie",
        "the",
        "a",
        "and",
        "or",
        "but",
        "of",
        "to",
        "with",
        "on",
        "for",
        "is",
        "it",
        "not",
        "as",
        "also",
        "at",
        "be",
        "from",
        "by",
        "over",
        "how",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split, remove stop words and short tokens."""
    words = re.findall(r"[a-zäöüß]{3,}", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _score_overlap(query_tokens: list[str], chunk_tokens: set[str]) -> float:
    """Fraction of query tokens found in the chunk token set."""
    if not query_tokens:
        return 0.0
    hits = sum(1 for qt in query_tokens if qt in chunk_tokens)
    return hits / len(query_tokens)


# ---------------------------------------------------------------------------
# EvolutionRAG
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""


class EvolutionRAG:
    """Lightweight RAG pipeline backed by SQLite.

    Storage lives at ``~/.cognithor/data/evolution_rag.db`` by default.
    Chunking is paragraph-based; retrieval uses simple keyword overlap
    (no external embedding model required).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".cognithor" / "data" / "evolution_rag.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> RAGDocument:
        """Read a file, chunk it, and store in the RAG database.

        *path* can be a file path or a URL-like string (in which case
        the content must be passed via *metadata["content"]*).
        """
        metadata = metadata or {}
        content = metadata.pop("content", None)
        title = metadata.pop("title", os.path.basename(path))

        if content is None:
            resolved = Path(path)
            if resolved.is_file():
                content = resolved.read_text(encoding="utf-8", errors="replace")
            else:
                log.warning("rag_ingest_no_file", path=path)
                content = ""

        doc = RAGDocument(
            title=title,
            source=path,
            content=content,
            metadata=metadata,
        )

        # Persist document
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, source, metadata, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                doc.id,
                doc.title,
                doc.source,
                json.dumps(doc.metadata, ensure_ascii=False),
                doc.ingested_at,
            ),
        )

        # Chunk and persist
        chunks = _chunk_paragraphs(content)
        for idx, chunk_text in enumerate(chunks):
            chunk = RAGChunk(
                document_id=doc.id,
                text=chunk_text,
                chunk_index=idx,
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks (id, document_id, text, chunk_index) "
                "VALUES (?, ?, ?, ?)",
                (chunk.id, chunk.document_id, chunk.text, chunk.chunk_index),
            )

        self._conn.commit()
        log.info(
            "rag_document_ingested",
            doc_id=doc.id[:8],
            title=title[:60],
            chunks=len(chunks),
        )
        return doc

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[RAGResult]:
        """Retrieve the *top_k* most relevant chunks for *question*."""
        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        cursor = self._conn.execute(
            "SELECT c.id, c.document_id, c.text, d.title, d.source "
            "FROM chunks c JOIN documents d ON c.document_id = d.id"
        )

        scored: list[tuple[float, RAGResult]] = []
        for row in cursor:
            chunk_id, doc_id, text, title, source = row
            chunk_tokens = set(_tokenize(text))
            score = _score_overlap(q_tokens, chunk_tokens)
            if score > 0:
                scored.append(
                    (
                        score,
                        RAGResult(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            text=text,
                            score=score,
                            title=title,
                            source=source,
                        ),
                    )
                )

        scored.sort(key=lambda x: -x[0])
        results = [r for _, r in scored[:top_k]]
        log.info("rag_query", question=question[:60], results=len(results))
        return results

    # ------------------------------------------------------------------
    # Context builder for learning tasks
    # ------------------------------------------------------------------

    async def get_context_for_task(self, task: LearningTask) -> str:
        """Build a context string from RAG results for *task*."""
        results = await self.query(task.query, top_k=5)
        if not results:
            return ""
        parts: list[str] = []
        for r in results:
            parts.append(f"[{r.title}] (score={r.score:.2f})\n{r.text[:500]}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def document_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()
