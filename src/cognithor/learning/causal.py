"""Causal-Learning-Layer: Korreliert Tool-Reihenfolgen mit Erfolgsscores.

Subsequence-Matching: Extrahiert alle 2er/3er Subsequenzen aus Tool-Sequenzen
und aggregiert deren Erfolgsraten.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cognithor.db import SQLITE_BUSY_TIMEOUT_MS
from cognithor.models import SequenceScore
from cognithor.security.encrypted_db import encrypted_connect

try:
    from cognithor.security.encrypted_db import compatible_row_factory
except ImportError:

    def compatible_row_factory() -> Any:
        return sqlite3.Row


from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = get_logger(__name__)


class CausalAnalyzer:
    """Analysiert kausale Zusammenhaenge zwischen Tool-Sequenzen und Erfolg."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        audit_emit_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Operational-Trust PR-A: ``audit_emit_callback`` is the
        Reflector's ``_emit_reflection_audit_event`` helper, bound here
        so ``record_sequence`` can emit a structured audit event in the
        same atomic-transaction window as the DB INSERT. ``None`` keeps
        the legacy behaviour (no audit, debug-log only on empty skip).
        """
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._audit_emit_callback: Callable[[str, dict[str, Any]], None] | None = (
            audit_emit_callback
        )
        self._ensure_schema()

    def set_audit_emit_callback(
        self,
        callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        """Late-bind the audit-emit callback.

        Used by the gateway boot path where the ``CausalAnalyzer`` is
        constructed before the ``Reflector`` exists, so the helper
        cannot be passed at ``__init__`` time.
        """
        self._audit_emit_callback = callback

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = encrypted_connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS causal_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_sequence TEXT NOT NULL,
                success_score REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_causal_session
                ON causal_sequences(session_id);
            CREATE INDEX IF NOT EXISTS idx_causal_timestamp
                ON causal_sequences(timestamp);
        """)
        conn.commit()

        # Schema-Migration: model_used Spalte hinzufuegen (fuer Cognithor-Learning)
        try:
            conn.execute("ALTER TABLE causal_sequences ADD COLUMN model_used TEXT DEFAULT ''")
            conn.commit()
        except Exception:
            # Spalte existiert bereits -- ignorieren (catches both sqlite3 and sqlcipher3)
            pass

    def record_sequence(
        self,
        session_id: str,
        tool_sequence: list[str],
        success_score: float,
        model_used: str = "",
    ) -> None:
        """Zeichnet eine Tool-Sequenz mit ihrem Erfolgs-Score auf.

        Operational-Trust PR-A: empty sequences are no longer silently
        dropped — an audit event ``causal_skipped_empty_sequence`` is
        emitted before the early return when an audit callback is
        wired. The DB INSERT and audit emit on the success path are
        atomic via ``with conn:`` (sqlite3 commits on block exit, rolls
        back on any exception inside). ``tool_sequence`` is serialised
        canonically (``sort_keys=True``, default separators,
        ``ensure_ascii=False``) so the on-disk row content is
        bit-identical for identical inputs and matches the canonical-
        form convention used by ``AuditLogger._last_hash_for_file``.
        """
        if not tool_sequence:
            if self._audit_emit_callback is not None:
                self._audit_emit_callback(
                    "causal_skipped_empty_sequence",
                    {
                        "session_id": session_id,
                        "success_score": success_score,
                        "model_used": model_used,
                    },
                )
            else:
                log.debug(
                    "skipped_empty_sequence",
                    session_id=session_id,
                )
            return

        conn = self._get_conn()
        timestamp = datetime.now(UTC).isoformat()
        tool_sequence_canonical = json.dumps(
            tool_sequence,
            sort_keys=True,
            ensure_ascii=False,
        )
        with conn:
            conn.execute(
                """INSERT INTO causal_sequences
                   (session_id, timestamp, tool_sequence,
                    success_score, model_used)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    timestamp,
                    tool_sequence_canonical,
                    success_score,
                    model_used,
                ),
            )
            if self._audit_emit_callback is not None:
                self._audit_emit_callback(
                    "causal_sequence_recorded",
                    {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "tool_sequence": tool_sequence,
                        "success_score": success_score,
                        "model_used": model_used,
                    },
                )

    def get_sequence_scores(
        self,
        min_occurrences: int = 3,
        max_subseq_len: int = 3,
    ) -> list[SequenceScore]:
        """Berechnet Scores fuer alle Subsequenzen.

        Extrahiert 2er und 3er Subsequenzen, aggregiert Erfolgsraten.

        Args:
            min_occurrences: Mindestanzahl Vorkommen fuer Bewertung.
            max_subseq_len: Maximale Laenge der Subsequenzen (2 oder 3).

        Returns:
            Liste von SequenceScore, sortiert nach avg_score absteigend.
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT tool_sequence, success_score FROM causal_sequences").fetchall()

        if not rows:
            return []

        total_sequences = len(rows)

        # Sammle Scores pro Subsequenz
        subseq_scores: dict[tuple[str, ...], list[float]] = defaultdict(list)

        for row in rows:
            seq = json.loads(row["tool_sequence"])
            score = row["success_score"]

            # 2er und 3er Subsequenzen (Reihenfolge erhalten)
            for length in range(2, min(max_subseq_len + 1, len(seq) + 1)):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    subseq_scores[subseq].append(score)

        # Aggregieren
        results: list[SequenceScore] = []
        for subseq, scores in subseq_scores.items():
            if len(scores) < min_occurrences:
                continue
            results.append(
                SequenceScore(
                    subsequence=subseq,
                    avg_score=sum(scores) / len(scores),
                    occurrence_count=len(scores),
                    confidence=len(scores) / total_sequences,
                )
            )

        results.sort(key=lambda s: s.avg_score, reverse=True)
        return results

    def suggest_tools(
        self,
        current_sequence: list[str],
        goal: str = "",
        top_n: int = 3,
    ) -> list[str]:
        """Empfiehlt naechste Tool-Schritte basierend auf historischen Daten.

        Schaut welche Tools nach der aktuellen Sequenz am erfolgreichsten waren.

        Args:
            current_sequence: Bisherige Tool-Sequenz in dieser Session.
            goal: Optionale Zielbeschreibung (derzeit unused, fuer Erweiterung).
            top_n: Anzahl Empfehlungen.

        Returns:
            Liste empfohlener Tool-Namen.
        """
        if not current_sequence:
            return []

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tool_sequence, success_score FROM causal_sequences WHERE success_score >= 0.5"
        ).fetchall()

        if not rows:
            return []

        # Finde Sequenzen die mit current_sequence beginnen oder sie enthalten
        current_tuple = tuple(current_sequence)
        next_tool_scores: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            seq = json.loads(row["tool_sequence"])
            score = row["success_score"]

            # Suche current_sequence als Subsequenz
            for i in range(len(seq) - len(current_tuple) + 1):
                if tuple(seq[i : i + len(current_tuple)]) == current_tuple:
                    # Naechstes Tool nach dem Match
                    next_idx = i + len(current_tuple)
                    if next_idx < len(seq):
                        next_tool_scores[seq[next_idx]].append(score)

        if not next_tool_scores:
            # Fallback: Schau nur aufs letzte Tool
            last_tool = current_sequence[-1]
            for row in rows:
                seq = json.loads(row["tool_sequence"])
                score = row["success_score"]
                for i, tool in enumerate(seq[:-1]):
                    if tool == last_tool:
                        next_tool_scores[seq[i + 1]].append(score)

        # Sortiere nach durchschnittlichem Score
        ranked = sorted(
            next_tool_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )

        return [tool for tool, _ in ranked[:top_n]]

    def get_total_sequences(self) -> int:
        """Gesamtzahl aufgezeichneter Sequenzen."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM causal_sequences").fetchone()
        return row["cnt"] if row else 0

    def get_model_performance(self, min_records: int = 5) -> dict[str, dict[str, Any]]:
        """Gibt Erfolgsstatistiken pro Modell zurueck.

        Cognithor kann diese Daten nutzen um die Modellwahl zu optimieren.

        Args:
            min_records: Mindestanzahl Datensaetze pro Modell fuer Auswertung.

        Returns:
            {"gpt-5.3-codex-spark": {"avg_score": 0.85, "count": 42}, ...}
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT model_used, AVG(success_score) as avg_score, COUNT(*) as cnt
               FROM causal_sequences
               WHERE model_used != ''
               GROUP BY model_used
               HAVING cnt >= ?""",
            (min_records,),
        ).fetchall()

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[row["model_used"]] = {
                "avg_score": round(row["avg_score"], 4),
                "count": row["cnt"],
            }
        return result

    def close(self) -> None:
        """Schliesst die DB-Verbindung."""
        if self._conn:
            self._conn.close()
            self._conn = None
