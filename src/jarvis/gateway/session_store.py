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
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
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

# Schema migrations (idempotent, order matters)
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'jarvis';",
    "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id, user_id, channel);",
    # Migration 3: Channel mappings for persistent session-to-chat-ID assignments
    """\
    CREATE TABLE IF NOT EXISTS channel_mappings (
        channel       TEXT NOT NULL,
        mapping_key   TEXT NOT NULL,
        mapping_value TEXT NOT NULL,
        updated_at    REAL NOT NULL,
        PRIMARY KEY (channel, mapping_key)
    );
    """,
    # Migration 4: Title column for chat history sidebar
    "ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT '';",
    # Migration 5: Folder column for folder/project system
    "ALTER TABLE sessions ADD COLUMN folder TEXT DEFAULT '';",
    # Migration 6: Incognito mode
    "ALTER TABLE sessions ADD COLUMN incognito INTEGER DEFAULT 0;",
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
        self._conn_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-initialisiert die DB-Verbindung und Schema (thread-safe)."""
        if self._conn is not None:
            return self._conn
        with self._conn_lock:
            # Double-check after acquiring lock
            if self._conn is not None:
                return self._conn
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            # Run migrations (idempotent)
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
                 last_activity, message_count, active, max_iterations, incognito)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(getattr(session, "incognito", False)),
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
        try:
            session.incognito = bool(row["incognito"])
        except (KeyError, IndexError):
            session.incognito = False
        return session

    def deactivate_session(self, session_id: str) -> None:
        """Markiert eine Session als inaktiv."""
        self.conn.execute(
            "UPDATE sessions SET active = 0 WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()

    # Roles that represent user-visible chat messages (not internal context)
    _VISIBLE_ROLES = {"user", "assistant"}

    def save_chat_history(
        self,
        session_id: str,
        messages: list[Message],
    ) -> int:
        """Speichert die Chat-History einer Session.

        Löscht vorherige History und schreibt alles neu
        (einfach + idempotent).

        Only saves user and assistant messages — system messages
        (context pipeline, identity, tool results) are internal
        and must not be persisted as chat history.

        Returns:
            Anzahl gespeicherter Messages.
        """
        self.conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?",
            (session_id,),
        )
        saved = 0
        for msg in messages:
            # Skip system/internal messages — only persist user-visible chat
            if msg.role.value not in self._VISIBLE_ROLES:
                continue
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
            saved += 1
        self.conn.commit()
        return saved

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
    # Chat history API (for WebUI sidebar)
    # ========================================================================

    def list_sessions_for_channel(
        self,
        channel: str = "webui",
        user_id: str = "web_user",
        limit: int = 50,
    ) -> list[dict[str, str | int | float]]:
        """Listet Sessions sortiert nach last_activity DESC.

        Returns:
            Liste mit id, title, message_count, started_at, last_activity.
        """
        rows = self.conn.execute(
            """
            SELECT session_id, title, message_count, started_at, last_activity,
                   folder, incognito
            FROM sessions
            WHERE channel = ? AND user_id = ? AND active = 1
            ORDER BY last_activity DESC
            LIMIT ?
            """,
            (channel, user_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row["session_id"],
                    "title": row["title"] or "",
                    "message_count": row["message_count"],
                    "started_at": row["started_at"],
                    "last_activity": row["last_activity"],
                    "folder": row["folder"] or "",
                    "incognito": bool(row["incognito"]) if "incognito" in row.keys() else False,
                }
            )
        return result

    def get_session_history(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, str | float]]:
        """Lädt Chat-Messages für eine Session als einfache Dicts.

        Only returns user and assistant messages — system messages
        are filtered out even if they were persisted by older code.

        Returns:
            Liste mit role, content, timestamp (chronologisch).
        """
        rows = self.conn.execute(
            """
            SELECT role, content, timestamp
            FROM chat_history
            WHERE session_id = ?
              AND role IN ('user', 'assistant')
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in reversed(rows)
        ]

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Setzt einen Display-Titel für eine Session.

        Returns:
            True wenn eine Zeile aktualisiert wurde.
        """
        cursor = self.conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_session_folder(self, session_id: str, folder: str) -> bool:
        """Setzt den Ordner für eine Session.

        Returns:
            True wenn eine Zeile aktualisiert wurde.
        """
        cursor = self.conn.execute(
            "UPDATE sessions SET folder = ? WHERE session_id = ?",
            (folder, session_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def list_folders(
        self,
        channel: str = "webui",
        user_id: str = "web_user",
    ) -> list[str]:
        """Gibt alle eindeutigen Ordnernamen für einen Channel/User zurück.

        Returns:
            Sortierte Liste von Ordnernamen (ohne Leerstring).
        """
        rows = self.conn.execute(
            """
            SELECT DISTINCT folder FROM sessions
            WHERE channel = ? AND user_id = ? AND active = 1
              AND folder IS NOT NULL AND folder != ''
            ORDER BY folder
            """,
            (channel, user_id),
        ).fetchall()
        return [row["folder"] for row in rows]

    def list_sessions_by_folder(
        self,
        folder: str,
        channel: str = "webui",
        user_id: str = "web_user",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List sessions filtered by project/folder."""
        rows = self.conn.execute(
            """
            SELECT session_id, title, message_count, started_at,
                   last_activity, folder
            FROM sessions
            WHERE channel = ? AND user_id = ? AND folder = ? AND active = 1
            ORDER BY last_activity DESC
            LIMIT ?
            """,
            (channel, user_id, folder, limit),
        ).fetchall()
        return [
            {
                "id": r["session_id"],
                "title": r["title"] or "",
                "message_count": r["message_count"],
                "started_at": r["started_at"],
                "last_activity": r["last_activity"],
                "folder": r["folder"] or "",
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Soft-Delete: Setzt active=0.

        Returns:
            True wenn eine Zeile aktualisiert wurde.
        """
        cursor = self.conn.execute(
            "UPDATE sessions SET active = 0 WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def auto_title(self, session_id: str) -> str:
        """Generiert einen Titel aus der ersten User-Message (max 60 Zeichen).

        Setzt den Titel nur, wenn noch keiner existiert.

        Returns:
            Der gesetzte (oder existierende) Titel.
        """
        # Check if a title already exists
        row = self.conn.execute(
            "SELECT title FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row and row["title"]:
            return row["title"]

        # Find first user message
        msg_row = self.conn.execute(
            """
            SELECT content FROM chat_history
            WHERE session_id = ? AND role = 'user'
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if not msg_row:
            return ""

        title = msg_row["content"].strip()
        # Remove newlines, truncate to 60 characters
        title = title.replace("\n", " ").replace("\r", "")
        if len(title) > 60:
            title = title[:57] + "..."

        self.update_session_title(session_id, title)
        return title

    def search_chat_history(
        self,
        query: str,
        channel: str = "webui",
        user_id: str = "web_user",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search across all chat messages."""
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """
            SELECT ch.session_id, ch.role, ch.content, ch.timestamp,
                   s.title, s.folder
            FROM chat_history ch
            JOIN sessions s ON ch.session_id = s.session_id
            WHERE ch.content LIKE ?
              AND ch.role IN ('user', 'assistant')
              AND s.active = 1
              AND s.channel = ?
              AND s.user_id = ?
            ORDER BY ch.timestamp DESC
            LIMIT ?
            """,
            (pattern, channel, user_id, limit),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "session_title": r["title"] or "",
                "folder": r["folder"] or "",
            }
            for r in rows
        ]

    # ========================================================================
    # Channel mappings (persistent session-to-chat-ID assignments)
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

    def should_create_new_session(
        self,
        channel: str,
        user_id: str,
        inactivity_timeout_minutes: int = 30,
        agent_id: str = "jarvis",
    ) -> bool:
        """Check if the most recent session is too old to resume."""
        row = self.conn.execute(
            """
            SELECT last_activity FROM sessions
            WHERE channel = ? AND user_id = ? AND agent_id = ? AND active = 1
            ORDER BY last_activity DESC LIMIT 1
            """,
            (channel, user_id, agent_id),
        ).fetchone()
        if row is None:
            return True  # No session exists
        last = datetime.fromtimestamp(row["last_activity"], tz=UTC)
        age = datetime.now(tz=UTC) - last
        return age.total_seconds() > inactivity_timeout_minutes * 60

    def export_session(self, session_id: str) -> dict[str, Any]:
        """Export a session with metadata and all messages as JSON-ready dict."""
        session_row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not session_row:
            return {"error": "Session not found"}

        messages = self.get_session_history(session_id, limit=10000)

        return {
            "session_id": session_id,
            "title": session_row["title"] or "",
            "folder": session_row["folder"] or "",
            "started_at": session_row["started_at"],
            "last_activity": session_row["last_activity"],
            "message_count": len(messages),
            "messages": messages,
            "exported_at": datetime.now(tz=UTC).isoformat(),
        }

    def close(self) -> None:
        """Schließt die DB-Verbindung."""
        if self._conn:
            self._conn.close()
            self._conn = None
