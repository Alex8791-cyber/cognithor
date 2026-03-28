"""Durable Message Queue - SQLite-based persistent message queue.

Ensures that incoming messages are not lost, even when
the gateway is busy or crashes. Supports priorities,
automatic retries and a dead-letter queue (DLQ).

Tabellen:
  message_queue  -- Alle eingehenden Nachrichten mit Status und Metadaten

Bibel-Referenz: §9.1 (Gateway), §15.5 (Resilienz)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums / Datenklassen
# ---------------------------------------------------------------------------


class MessagePriority(IntEnum):
    """Message priority. Higher values = higher priority."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class QueuedMessage:
    """Eine in der Queue gepufferte Nachricht mit Metadaten."""

    id: str
    message_json: str  # Serialisiertes IncomingMessage-JSON
    priority: int
    status: str  # "pending", "processing", "completed", "failed", "dead"
    retry_count: int
    max_retries: int
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None

    @property
    def message_data(self) -> dict[str, Any]:
        """Deserialisiert die Nachricht als Dict."""
        return json.loads(self.message_json)


# ---------------------------------------------------------------------------
# SQL-Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS message_queue (
    id           TEXT PRIMARY KEY,
    message_json TEXT NOT NULL,
    priority     INTEGER NOT NULL DEFAULT 5,
    status       TEXT NOT NULL DEFAULT 'pending',
    retry_count  INTEGER NOT NULL DEFAULT 0,
    max_retries  INTEGER NOT NULL DEFAULT 3,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_mq_status_priority
    ON message_queue(status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_mq_created
    ON message_queue(created_at);
"""


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> float:
    """datetime → Unix-Timestamp."""
    return dt.timestamp()


def _from_ts(ts: float) -> datetime:
    """Unix-Timestamp → datetime (UTC)."""
    return datetime.fromtimestamp(ts, tz=UTC)


def _utc_now() -> datetime:
    """Aktuelle UTC-Zeit."""
    return datetime.now(UTC)


def _row_to_queued(row: sqlite3.Row) -> QueuedMessage:
    """Konvertiert eine DB-Zeile in ein QueuedMessage-Objekt."""
    return QueuedMessage(
        id=row["id"],
        message_json=row["message_json"],
        priority=row["priority"],
        status=row["status"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        created_at=_from_ts(row["created_at"]),
        updated_at=_from_ts(row["updated_at"]),
        error_message=row["error_message"],
    )


# ---------------------------------------------------------------------------
# DurableMessageQueue
# ---------------------------------------------------------------------------


class DurableMessageQueue:
    """SQLite-based persistent message queue with priorities, retry and DLQ.

    Verwendet WAL-Modus und check_same_thread=False (gleiche Muster wie
    session_store.py). Alle oeffentlichen Methoden sind async, da sie
    via asyncio.to_thread ausgefuehrt werden, um den Event-Loop nicht
    zu blockieren.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        max_size: int = 10_000,
        max_retries: int = 3,
        ttl_hours: int = 24,
    ) -> None:
        self._db_path = Path(db_path)
        self._max_size = max_size
        self._max_retries = max_retries
        self._ttl_hours = ttl_hours
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection (lazy init, gleich wie session_store.py)
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-initialisiert die DB-Verbindung und Schema."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = encrypted_connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
        return self._conn

    # ------------------------------------------------------------------
    # Core-Operationen
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        message: Any,
        *,
        priority: int = MessagePriority.NORMAL,
    ) -> str:
        """Insert a message into the queue.

        Args:
            message: Ein IncomingMessage (Pydantic-Model) oder ein Dict.
            priority: Message priority (1-10).

        Returns:
            Die UUID der eingereihten Nachricht.

        Raises:
            RuntimeError: Wenn die Queue voll ist (max_size erreicht).
        """
        async with self._lock:
            return await asyncio.to_thread(self._enqueue_sync, message, priority)

    def _enqueue_sync(self, message: Any, priority: int) -> str:
        """Synchrone Enqueue-Implementierung."""
        # Check queue size
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM message_queue WHERE status IN ('pending', 'processing')"
        ).fetchone()
        current_size = row["cnt"] if row else 0
        if current_size >= self._max_size:
            raise RuntimeError(f"Message-Queue voll: {current_size}/{self._max_size}")

        # Nachricht serialisieren
        if hasattr(message, "model_dump_json"):
            # Pydantic v2
            message_json = message.model_dump_json()
        elif hasattr(message, "json"):
            # Pydantic v1 Fallback
            message_json = message.json()
        elif isinstance(message, dict):
            message_json = json.dumps(message, default=str)
        elif isinstance(message, str):
            message_json = message
        else:
            message_json = json.dumps({"raw": str(message)}, default=str)

        msg_id = uuid.uuid4().hex
        now = _ts(_utc_now())

        self.conn.execute(
            """
            INSERT INTO message_queue
                (id, message_json, priority, status, retry_count,
                 max_retries, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (msg_id, message_json, priority, self._max_retries, now, now),
        )
        self.conn.commit()

        logger.debug(
            "message_enqueued id=%s priority=%d queue_depth=%d",
            msg_id,
            priority,
            current_size + 1,
        )
        return msg_id

    async def dequeue(self) -> QueuedMessage | None:
        """Get the next pending message (highest priority, oldest first).

        Sets the status to 'processing'. Returns None if the queue is empty.
        """
        async with self._lock:
            return await asyncio.to_thread(self._dequeue_sync)

    def _dequeue_sync(self) -> QueuedMessage | None:
        """Synchrone Dequeue-Implementierung."""
        row = self.conn.execute(
            """
            SELECT * FROM message_queue
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            return None

        now = _ts(_utc_now())
        self.conn.execute(
            "UPDATE message_queue SET status = 'processing', updated_at = ? WHERE id = ?",
            (now, row["id"]),
        )
        self.conn.commit()

        msg = _row_to_queued(row)
        msg.status = "processing"
        msg.updated_at = _from_ts(now)

        logger.debug("message_dequeued id=%s priority=%d", msg.id, msg.priority)
        return msg

    async def complete(self, message_id: str) -> None:
        """Markiert eine Nachricht als erfolgreich verarbeitet."""
        async with self._lock:
            await asyncio.to_thread(self._complete_sync, message_id)

    def _complete_sync(self, message_id: str) -> None:
        """Synchrone Complete-Implementierung."""
        now = _ts(_utc_now())
        self.conn.execute(
            "UPDATE message_queue SET status = 'completed', updated_at = ? WHERE id = ?",
            (now, message_id),
        )
        self.conn.commit()
        logger.debug("message_completed id=%s", message_id)

    async def fail(self, message_id: str, error: str) -> None:
        """Markiert eine Nachricht als fehlgeschlagen.

        Wenn noch Retries uebrig sind, wird die Nachricht wieder auf 'pending'
        gesetzt. Ansonsten wird sie in die Dead-Letter-Queue verschoben
        (status = 'dead').
        """
        async with self._lock:
            await asyncio.to_thread(self._fail_sync, message_id, error)

    def _fail_sync(self, message_id: str, error: str) -> None:
        """Synchrone Fail-Implementierung."""
        row = self.conn.execute(
            "SELECT retry_count, max_retries FROM message_queue WHERE id = ?",
            (message_id,),
        ).fetchone()

        if row is None:
            logger.warning("message_not_found id=%s", message_id)
            return

        now = _ts(_utc_now())
        new_retry_count = row["retry_count"] + 1

        if new_retry_count < row["max_retries"]:
            # Back into the queue
            self.conn.execute(
                """
                UPDATE message_queue
                SET status = 'pending', retry_count = ?, updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (new_retry_count, now, error, message_id),
            )
            logger.info(
                "message_requeued id=%s retry=%d/%d",
                message_id,
                new_retry_count,
                row["max_retries"],
            )
        else:
            # Dead Letter
            self.conn.execute(
                """
                UPDATE message_queue
                SET status = 'dead', retry_count = ?, updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (new_retry_count, now, error, message_id),
            )
            logger.warning(
                "message_dead_lettered id=%s retries_exhausted=%d error=%s",
                message_id,
                new_retry_count,
                error,
            )

        self.conn.commit()

    async def get_dead_letters(self, limit: int = 100) -> list[QueuedMessage]:
        """Return messages from the dead-letter queue."""
        async with self._lock:
            return await asyncio.to_thread(self._get_dead_letters_sync, limit)

    def _get_dead_letters_sync(self, limit: int) -> list[QueuedMessage]:
        """Synchrone Dead-Letter-Abfrage."""
        rows = self.conn.execute(
            """
            SELECT * FROM message_queue
            WHERE status = 'dead'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_queued(r) for r in rows]

    async def cleanup(self) -> int:
        """Entfernt abgeschlossene und abgelaufene Nachrichten.

        Returns:
            Anzahl der entfernten Nachrichten.
        """
        async with self._lock:
            return await asyncio.to_thread(self._cleanup_sync)

    def _cleanup_sync(self) -> int:
        """Synchrone Cleanup-Implementierung."""
        cutoff = _ts(_utc_now()) - (self._ttl_hours * 3600)

        # Abgeschlossene Nachrichten entfernen
        cursor = self.conn.execute("DELETE FROM message_queue WHERE status = 'completed'")
        completed_count = cursor.rowcount

        # Remove expired messages (older than TTL)
        cursor = self.conn.execute(
            "DELETE FROM message_queue WHERE created_at < ? AND status IN ('dead', 'failed')",
            (cutoff,),
        )
        expired_count = cursor.rowcount

        self.conn.commit()
        total = completed_count + expired_count

        if total > 0:
            logger.info(
                "queue_cleanup completed=%d expired=%d total_removed=%d",
                completed_count,
                expired_count,
                total,
            )
        return total

    async def get_depth(self) -> int:
        """Aktuelle Anzahl ausstehender Nachrichten."""
        async with self._lock:
            return await asyncio.to_thread(self._get_depth_sync)

    def _get_depth_sync(self) -> int:
        """Synchrone Depth-Abfrage."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM message_queue WHERE status = 'pending'"
        ).fetchone()
        return row["cnt"] if row else 0

    async def get_stats(self) -> dict[str, int]:
        """Queue-Statistiken: pending, processing, completed, failed, dead, total."""
        async with self._lock:
            return await asyncio.to_thread(self._get_stats_sync)

    def _get_stats_sync(self) -> dict[str, int]:
        """Synchrone Stats-Abfrage."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM message_queue GROUP BY status"
        ).fetchall()

        stats: dict[str, int] = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "dead": 0,
        }
        total = 0
        for row in rows:
            stats[row["status"]] = row["cnt"]
            total += row["cnt"]
        stats["total"] = total
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("message_queue_closed db=%s", self._db_path)
