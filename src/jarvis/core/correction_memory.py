"""Correction Memory — learns from user corrections to avoid repeating mistakes.

Stores corrections in SQLite, matches similar situations by keyword overlap,
and provides reminders for the Planner context.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["CorrectionMemory"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    correction_text TEXT NOT NULL,
    original_plan TEXT DEFAULT '',
    corrected_plan TEXT DEFAULT '',
    keywords TEXT DEFAULT '',
    times_triggered INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    last_triggered_at REAL
);
CREATE INDEX IF NOT EXISTS idx_corr_keywords ON corrections(keywords);
"""


class CorrectionMemory:
    """SQLite-backed correction store with keyword matching."""

    def __init__(self, db_path: Path | str, proactive_threshold: int = 3) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._proactive_threshold = proactive_threshold
        with encrypted_connect(str(self._db_path)) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = encrypted_connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def store(
        self,
        user_message: str,
        correction_text: str,
        original_plan: str = "",
        corrected_plan: str = "",
        keywords: list[str] | None = None,
    ) -> str:
        """Store a correction. Returns correction ID."""
        corr_id = f"corr_{uuid.uuid4().hex[:12]}"
        if keywords is None:
            keywords = self._extract_keywords(user_message + " " + correction_text)
        kw_str = ",".join(keywords)

        # Check for similar existing correction (merge)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id, times_triggered FROM corrections WHERE keywords LIKE ? LIMIT 1",
                (f"%{keywords[0]}%" if keywords else "",),
            ).fetchone()

            if existing and self._text_overlap(correction_text, existing) > 0.5:
                conn.execute(
                    "UPDATE corrections SET times_triggered = times_triggered + 1, "
                    "last_triggered_at = ? WHERE id = ?",
                    (time.time(), existing["id"]),
                )
                return existing["id"]

            conn.execute(
                "INSERT INTO corrections (id, user_message, correction_text, "
                "original_plan, corrected_plan, keywords, created_at, last_triggered_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    corr_id,
                    user_message[:500],
                    correction_text[:500],
                    original_plan[:500],
                    corrected_plan[:500],
                    kw_str,
                    time.time(),
                    time.time(),
                ),
            )

        log.info("correction_stored", id=corr_id, keywords=kw_str[:60])
        return corr_id

    def find_similar(self, user_message: str, limit: int = 3) -> list[dict[str, Any]]:
        """Find corrections similar to the current user message."""
        keywords = self._extract_keywords(user_message)
        if not keywords:
            return []

        with self._conn() as conn:
            results = []
            for kw in keywords:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE keywords LIKE ? "
                    "ORDER BY times_triggered DESC, last_triggered_at DESC "
                    "LIMIT ?",
                    (f"%{kw}%", limit),
                ).fetchall()
                for r in rows:
                    d = dict(r)
                    if d not in results:
                        results.append(d)

        return results[:limit]

    def should_ask_proactively(self, query: str, keywords: list[str] | None = None) -> bool:
        """Check if we should proactively ask before acting (threshold reached)."""
        if keywords is None:
            keywords = self._extract_keywords(query)
        if not keywords:
            return False

        with self._conn() as conn:
            for kw in keywords:
                row = conn.execute(
                    "SELECT SUM(times_triggered) as total FROM corrections WHERE keywords LIKE ?",
                    (f"%{kw}%",),
                ).fetchone()
                if row and (row["total"] or 0) >= self._proactive_threshold:
                    return True
        return False

    def get_reminder(self, user_message: str) -> str | None:
        """Get a reminder string for the Planner context, or None."""
        matches = self.find_similar(user_message, limit=2)
        if not matches:
            return None

        reminders = []
        for m in matches:
            reminders.append(
                f'- Bei "{m["user_message"][:80]}" hat der User korrigiert: '
                f'"{m["correction_text"][:120]}"'
            )

        if self.should_ask_proactively(user_message):
            return (
                "WICHTIG — Der User hat bei aehnlichen Anfragen mehrfach korrigiert. "
                "Frage ZUERST ob dein Ansatz passt, bevor du handelst:\n" + "\n".join(reminders)
            )

        return (
            "ERINNERUNG — Der User hat bei aehnlichen Anfragen korrigiert:\n"
            + "\n".join(reminders)
            + "\nBeruecksichtige das in deinem Plan."
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        import re

        stopwords = {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "ist",
            "sind",
            "hat",
            "haben",
            "wird",
            "werden",
            "mit",
            "von",
            "fuer",
            "auf",
            "den",
            "dem",
            "des",
            "im",
            "in",
            "an",
            "zu",
            "nicht",
            "nein",
            "ja",
            "bitte",
            "mal",
            "noch",
            "auch",
            "nur",
            "mir",
            "mich",
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "not",
        }
        words = re.findall(
            r"\b[a-zA-Z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]{3,}\b", text.lower()
        )
        return [w for w in words if w not in stopwords][:10]

    @staticmethod
    def _text_overlap(text: str, row: Any) -> float:
        """Simple word overlap score between correction text and existing row."""
        try:
            existing_text = row["correction_text"] if hasattr(row, "__getitem__") else ""
            words_a = set(text.lower().split())
            words_b = set(existing_text.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / max(len(words_a), len(words_b))
        except Exception:
            return 0.0
