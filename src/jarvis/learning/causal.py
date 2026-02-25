"""Causal-Learning-Layer: Korreliert Tool-Reihenfolgen mit Erfolgsscores.

Subsequence-Matching: Extrahiert alle 2er/3er Subsequenzen aus Tool-Sequenzen
und aggregiert deren Erfolgsraten.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.models import SequenceScore, ToolSequenceRecord
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CausalAnalyzer:
    """Analysiert kausale Zusammenhaenge zwischen Tool-Sequenzen und Erfolg."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
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

    def record_sequence(
        self,
        session_id: str,
        tool_sequence: list[str],
        success_score: float,
    ) -> None:
        """Zeichnet eine Tool-Sequenz mit ihrem Erfolgs-Score auf."""
        if not tool_sequence:
            return

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO causal_sequences (session_id, timestamp, tool_sequence, success_score)
               VALUES (?, ?, ?, ?)""",
            (
                session_id,
                datetime.now(UTC).isoformat(),
                json.dumps(tool_sequence),
                success_score,
            ),
        )
        conn.commit()

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
        rows = conn.execute(
            "SELECT tool_sequence, success_score FROM causal_sequences"
        ).fetchall()

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

    def close(self) -> None:
        """Schliesst die DB-Verbindung."""
        if self._conn:
            self._conn.close()
            self._conn = None
