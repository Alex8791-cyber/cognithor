"""Session Store · SQLite-basierte Session-Persistenz. [B§9.1]

Sessions überleben Gateway-Neustarts. Speichert SessionContext
und Working-Memory-Chat-History in SQLite.

Tabellen:
  sessions      -- SessionContext-Felder
  chat_history   -- Messages pro Session (für Working Memory)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from jarvis.models import (
    Message,
    MessageRole,
    SessionContext,
)

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    channel       TEXT NOT NULL,
    agent_id      TEXT NOT NULL DEFAULT 'jarvis',
    started_at    REAL NOT NULL,
    last_activity REAL NOT NULL,
    message_count INTEGER DEFAULT 0,
    active        INTEGER DEFAULT 1,
    max_iterations INTEGER DEFAULT 10
);

CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    channel     TEXT DEFAULT '',
    timestamp   REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_session
    ON chat_history(session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_sessions_user_channel
    ON sessions(user_id, channel);
"""

# Schema-Migrationen (idempotent, Reihenfolge wichtig)
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'jarvis';",
    "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id, user_id, channel);",
    # Migration 3: Channel-Mappings für persistente Session→Chat-ID Zuordnungen
    """\
    CREATE TABLE IF NOT EXISTS channel_mappings (
        channel       TEXT NOT NULL,
        mapping_key   TEXT NOT NULL,
        mapping_value TEXT NOT NULL,
        updated_at    REAL NOT NULL,
        PRIMARY KEY (channel, mapping_key)
    );
    """,
]


def _ts(dt: datetime) -> float:
    """datetime → Unix-Timestamp."""
    return dt.timestamp()


def _from_ts(ts: float) -> datetime:
    """Unix-Timestamp → datetime (UTC)."""
    return datetime.fromtimestamp(ts, tz=UTC)


class SessionStore:
    """SQLite-basierte Session-Persistenz.

    Idempotent -- kann beliebig oft instanziiert werden.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialisiert den SessionStore mit SQLite-Pfad."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-initialisiert die DB-Verbindung und Schema."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            # Migrationen ausführen (idempotent)
            for migration in _MIGRATIONS:
                try:
                    self._conn.execute(migration)
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass  # Spalte/Index existiert bereits
        return self._conn

    def save_session(self, session: SessionContext) -> None:
        """Speichert oder aktualisiert eine Session."""
        agent_id = getattr(session, "agent_name", "jarvis") or "jarvis"
        self.conn.execute(
            """
            INSERT INTO sessions
                (session_id, user_id, channel, agent_id, started_at,
                 last_activity, message_count, active, max_iterations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_activity=excluded.last_activity,
                message_count=excluded.message_count,
                active=excluded.active
            """,
            (
                session.session_id,
                session.user_id,
                session.channel,
                agent_id,
                _ts(session.started_at),
                _ts(session.last_activity),
                session.message_count,
                int(session.active),
                session.max_iterations,
            ),
        )
        self.conn.commit()

    def load_session(
        self,
        channel: str,
        user_id: str,
        agent_id: str = "jarvis",
    ) -> SessionContext | None:
        """Lädt die letzte aktive Session für Channel+User+Agent."""
        row = self.conn.execute(
            """
            SELECT * FROM sessions
            WHERE user_id = ? AND channel = ? AND agent_id = ? AND active = 1
            ORDER BY last_activity DESC
            LIMIT 1
            """,
            (user_id, channel, agent_id),
        ).fetchone()

        if row is None:
            return None

        session = SessionContext(
            session_id=row["session_id"],
            user_id=row["user_id"],
            channel=row["channel"],
            agent_name=agent_id,
            started_at=_from_ts(row["started_at"]),
            last_activity=_from_ts(row["last_activity"]),
            message_count=row["message_count"],
            active=bool(row["active"]),
            max_iterations=row["max_iterations"],
        )
        return session

    def deactivate_session(self, session_id: str) -> None:
        """Markiert eine Session als inaktiv."""
        self.conn.execute(
            "UPDATE sessions SET active = 0 WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()

    def save_chat_history(
        self,
        session_id: str,
        messages: list[Message],
    ) -> int:
        """Speichert die Chat-History einer Session.

        Löscht vorherige History und schreibt alles neu
        (einfach + idempotent).

        Returns:
            Anzahl gespeicherter Messages.
        """
        self.conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?",
            (session_id,),
        )
        for msg in messages:
            ts = msg.timestamp.timestamp() if msg.timestamp else datetime.now(tz=UTC).timestamp()
            self.conn.execute(
                """
                INSERT INTO chat_history
                    (session_id, role, content, channel, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    msg.role.value,
                    msg.content,
                    msg.channel or "",
                    ts,
                ),
            )
        self.conn.commit()
        return len(messages)

    def load_chat_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Lädt die Chat-History einer Session.

        Args:
            session_id: Session-ID
            limit: Maximale Anzahl Messages (neueste zuerst, dann umkehren)

        Returns:
            Chronologisch sortierte Messages.
        """
        rows = self.conn.execute(
            """
            SELECT role, content, channel, timestamp
            FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

        messages = []
        for row in reversed(rows):  # Chronologische Reihenfolge
            messages.append(
                Message(
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    channel=row["channel"] or None,
                    timestamp=_from_ts(row["timestamp"]),
                )
            )
        return messages

    def list_sessions(
        self,
        user_id: str | None = None,
        active_only: bool = True,
        limit: int = 20,
    ) -> list[dict[str, str | int | float]]:
        """Listet Sessions auf.

        Returns:
            Liste von Session-Infos als Dicts.
        """
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[str | int] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if active_only:
            query += " AND active = 1"

        query += " ORDER BY last_activity DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_sessions(self, active_only: bool = True) -> int:
        """Anzahl Sessions."""
        if active_only:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE active = 1"
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
        return row["cnt"] if row else 0

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Deaktiviert Sessions die älter als max_age_days sind.

        Returns:
            Anzahl deaktivierter Sessions.
        """
        cutoff = datetime.now(tz=UTC).timestamp() - (max_age_days * 86400)
        cursor = self.conn.execute(
            """
            UPDATE sessions SET active = 0
            WHERE active = 1 AND last_activity < ?
            """,
            (cutoff,),
        )
        self.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(
                "Alte Sessions deaktiviert: %d (älter als %d Tage)",
                count,
                max_age_days,
            )
        return count

    # ========================================================================
    # Channel-Mappings (persistente Session→Chat-ID Zuordnungen)
    # ========================================================================

    def save_channel_mapping(self, channel: str, key: str, value: str) -> None:
        """Speichert ein Channel-Mapping (z.B. session_id → chat_id).

        Args:
            channel: Channel-Namespace (z.B. 'telegram_session', 'discord_user').
            key: Mapping-Key (z.B. Session-ID).
            value: Mapping-Value (z.B. Chat-ID als String).
        """
        now = datetime.now(tz=UTC).timestamp()
        self.conn.execute(
            """
            INSERT INTO channel_mappings (channel, mapping_key, mapping_value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel, mapping_key) DO UPDATE SET
                mapping_value = excluded.mapping_value,
                updated_at = excluded.updated_at
            """,
            (channel, key, value, now),
        )
        self.conn.commit()

    def load_channel_mapping(self, channel: str, key: str) -> str | None:
        """Lädt ein einzelnes Channel-Mapping.

        Returns:
            Mapping-Value oder None wenn nicht vorhanden.
        """
        row = self.conn.execute(
            "SELECT mapping_value FROM channel_mappings WHERE channel = ? AND mapping_key = ?",
            (channel, key),
        ).fetchone()
        return row["mapping_value"] if row else None

    def load_all_channel_mappings(self, channel: str) -> dict[str, str]:
        """Lädt alle Mappings für einen Channel-Namespace.

        Returns:
            Dict von key → value.
        """
        rows = self.conn.execute(
            "SELECT mapping_key, mapping_value FROM channel_mappings WHERE channel = ?",
            (channel,),
        ).fetchall()
        return {row["mapping_key"]: row["mapping_value"] for row in rows}

    def cleanup_channel_mappings(self, max_age_days: int = 30) -> int:
        """Löscht veraltete Channel-Mappings.

        Returns:
            Anzahl gelöschter Einträge.
        """
        cutoff = datetime.now(tz=UTC).timestamp() - (max_age_days * 86400)
        cursor = self.conn.execute(
            "DELETE FROM channel_mappings WHERE updated_at < ?",
            (cutoff,),
        )
        self.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(
                "Alte Channel-Mappings gelöscht: %d (älter als %d Tage)",
                count,
                max_age_days,
            )
        return count

    def close(self) -> None:
        """Schließt die DB-Verbindung."""
        if self._conn:
            self._conn.close()
            self._conn = None
