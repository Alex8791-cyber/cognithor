"""User Preference Store: SQLite-based per-user preferences.

Tracks user preferences (verbosity, greeting name, formality) and
auto-learns from interaction patterns (average message length → verbosity).
Uses the same DB as SessionStore (~/.jarvis/sessions.db).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS

log = logging.getLogger(__name__)


class UserPreference(BaseModel):
    """User preference model."""

    user_id: str
    verbosity: Literal["terse", "normal", "verbose"] = "normal"
    greeting_name: str = ""  # Preferred name for greetings
    formality: Literal["informal", "formal"] = "informal"
    avg_message_length: float = 0.0  # Auto-learned
    interaction_count: int = 0

    @property
    def verbosity_hint(self) -> str:
        """Returns a system-prompt hint based on verbosity preference."""
        if self.verbosity == "terse":
            return (
                "HINWEIS: Dieser User bevorzugt kurze, knappe Antworten. "
                "Fasse dich kurz und komm direkt zum Punkt."
            )
        elif self.verbosity == "verbose":
            return (
                "HINWEIS: Dieser User bevorzugt ausführliche Antworten. "
                "Erkläre detailliert und gib Hintergrund-Informationen."
            )
        return ""


class UserPreferenceStore:
    """SQLite-based user preference store.

    Uses the same database as SessionStore (~/.jarvis/sessions.db)
    but a separate table.
    """

    _CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id      TEXT PRIMARY KEY,
        verbosity    TEXT NOT NULL DEFAULT 'normal',
        greeting_name TEXT NOT NULL DEFAULT '',
        formality    TEXT NOT NULL DEFAULT 'informal',
        avg_message_length REAL NOT NULL DEFAULT 0.0,
        interaction_count INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".jarvis" / "sessions.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Persistente Connection statt pro-Aufruf connect (Performance)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Creates the user_preferences table if it doesn't exist."""
        try:
            self._conn.execute(self._CREATE_TABLE)
            self._conn.commit()
        except Exception as exc:
            log.warning("user_preferences_table_creation_failed", exc_info=exc)

    def close(self) -> None:
        """Schließt die DB-Verbindung."""
        try:
            self._conn.close()
        except Exception as exc:
            log.debug("preferences_db_close_error: %s", exc)  # Cleanup — failure is non-critical
        self._conn = None  # type: ignore[assignment]

    def get_or_create(self, user_id: str) -> UserPreference:
        """Gets or creates a user preference record."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ?",
                    (user_id,),
                ).fetchone()

                if row:
                    return UserPreference(
                        user_id=row["user_id"],
                        verbosity=row["verbosity"],
                        greeting_name=row["greeting_name"],
                        formality=row["formality"],
                        avg_message_length=row["avg_message_length"],
                        interaction_count=row["interaction_count"],
                    )

                # Create new
                pref = UserPreference(user_id=user_id)
                self._conn.execute(
                    "INSERT INTO user_preferences (user_id) VALUES (?)",
                    (user_id,),
                )
                self._conn.commit()
            return pref
        except Exception as exc:
            log.warning("user_preferences_get_failed", exc_info=exc)
            return UserPreference(user_id=user_id)

    def update(self, pref: UserPreference) -> None:
        """Updates a user preference record."""
        try:
            with self._lock:
                self._conn.execute(
                    """\
                    INSERT INTO user_preferences
                        (user_id, verbosity, greeting_name, formality,
                         avg_message_length, interaction_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        verbosity = excluded.verbosity,
                        greeting_name = excluded.greeting_name,
                        formality = excluded.formality,
                        avg_message_length = excluded.avg_message_length,
                        interaction_count = excluded.interaction_count
                    """,
                    (
                        pref.user_id,
                        pref.verbosity,
                        pref.greeting_name,
                        pref.formality,
                        pref.avg_message_length,
                        pref.interaction_count,
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            log.warning("user_preferences_update_failed", exc_info=exc)

    def record_interaction(self, user_id: str, msg_length: int) -> UserPreference:
        """Records an interaction and auto-learns verbosity from message length.

        Uses exponential moving average (alpha=0.1) to smooth the learned
        average message length. Then derives verbosity preference:
          - avg < 30 chars → terse
          - avg > 200 chars → verbose
          - otherwise → normal
        """
        pref = self.get_or_create(user_id)
        pref.interaction_count += 1

        # Exponential moving average
        alpha = 0.1
        if pref.avg_message_length == 0.0:
            pref.avg_message_length = float(msg_length)
        else:
            pref.avg_message_length = alpha * msg_length + (1 - alpha) * pref.avg_message_length

        # Auto-derive verbosity after enough interactions
        if pref.interaction_count >= 5:
            if pref.avg_message_length < 30:
                pref.verbosity = "terse"
            elif pref.avg_message_length > 200:
                pref.verbosity = "verbose"
            else:
                pref.verbosity = "normal"

        self.update(pref)
        return pref
