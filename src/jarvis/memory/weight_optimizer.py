"""Performance-basierte Weight-Anpassung fuer HybridSearch.

Nutzt Exponential Moving Average (EMA) ueber Kanal-Nuetzlichkeit,
gemessen an User-Zufriedenheit (Reflector-Score).

EMA-Formel: w_new = alpha * observed + (1-alpha) * w_old
Constraints: Jedes Gewicht min 0.05, Summe = 1.0
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class SearchWeightOptimizer:
    """Optimiert HybridSearch-Gewichte basierend auf Sucherfolg."""

    # Minimum weight per channel
    MIN_WEIGHT = 0.05
    # EMA smoothing factor
    DEFAULT_ALPHA = 0.1

    def __init__(
        self,
        db_path: str | Path | None = None,
        alpha: float = DEFAULT_ALPHA,
        initial_weights: tuple[float, float, float] | None = None,
    ) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._alpha = alpha
        self._conn: sqlite3.Connection | None = None

        # Current weights (vector, bm25, graph)
        if initial_weights:
            self._w_vector, self._w_bm25, self._w_graph = initial_weights
        else:
            self._w_vector = 0.50
            self._w_bm25 = 0.30
            self._w_graph = 0.20

        self._ensure_schema()
        self._load_weights()

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
            CREATE TABLE IF NOT EXISTS search_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                w_vector_contribution REAL NOT NULL DEFAULT 0.0,
                w_bm25_contribution REAL NOT NULL DEFAULT 0.0,
                w_graph_contribution REAL NOT NULL DEFAULT 0.0,
                feedback_score REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
                ON search_outcomes(timestamp);

            CREATE TABLE IF NOT EXISTS weight_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                w_vector REAL NOT NULL,
                w_bm25 REAL NOT NULL,
                w_graph REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.commit()

    def _load_weights(self) -> None:
        """Laedt gespeicherte Gewichte aus der DB."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM weight_state WHERE id = 1").fetchone()
        if row:
            self._w_vector = row["w_vector"]
            self._w_bm25 = row["w_bm25"]
            self._w_graph = row["w_graph"]

    def _save_weights(self) -> None:
        """Speichert aktuelle Gewichte in die DB."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO weight_state (id, w_vector, w_bm25, w_graph, updated_at)
               VALUES (1, ?, ?, ?, ?)""",
            (self._w_vector, self._w_bm25, self._w_graph, now),
        )
        conn.commit()

    @staticmethod
    def _normalize_weights(
        w_vector: float, w_bm25: float, w_graph: float, min_w: float = 0.05,
    ) -> tuple[float, float, float]:
        """Normalisiert Gewichte: min min_w je Kanal, Summe = 1.0.

        Approach: Clamp all weights to min_w, then distribute
        remaining budget (1.0 - 3*min_w) proportionally.
        """
        weights = [max(w_vector, min_w), max(w_bm25, min_w), max(w_graph, min_w)]
        total = sum(weights)
        if total == 0:
            return (1 / 3, 1 / 3, 1 / 3)

        # Normalize to sum 1.0 while keeping minimum constraint
        # Reserve min_w for each channel, distribute rest proportionally
        reserved = 3 * min_w
        if reserved >= 1.0:
            return (1 / 3, 1 / 3, 1 / 3)

        remaining = 1.0 - reserved
        excess = [w - min_w for w in weights]
        excess_total = sum(excess)

        if excess_total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)

        result = [min_w + (e / excess_total) * remaining for e in excess]
        return (result[0], result[1], result[2])

    def record_outcome(
        self,
        query: str,
        channel_contributions: dict[str, float],
        feedback_score: float,
    ) -> None:
        """Zeichnet ein Suchergebnis auf und aktualisiert Gewichte via EMA.

        Args:
            query: Die Suchanfrage.
            channel_contributions: {"vector": 0-1, "bm25": 0-1, "graph": 0-1}
            feedback_score: Nuetzlichkeit des Ergebnisses (0-1, z.B. Reflector-Score).
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        v_contrib = channel_contributions.get("vector", 0.0)
        b_contrib = channel_contributions.get("bm25", 0.0)
        g_contrib = channel_contributions.get("graph", 0.0)

        # Persist outcome
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO search_outcomes
               (timestamp, query_hash, w_vector_contribution, w_bm25_contribution,
                w_graph_contribution, feedback_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                query_hash,
                v_contrib,
                b_contrib,
                g_contrib,
                feedback_score,
            ),
        )
        conn.commit()

        # EMA update: Scale contributions by feedback score
        # Higher feedback = this channel mix was good
        if feedback_score > 0:
            total_contrib = v_contrib + b_contrib + g_contrib
            if total_contrib > 0:
                observed_v = v_contrib / total_contrib
                observed_b = b_contrib / total_contrib
                observed_g = g_contrib / total_contrib

                # Weight by feedback_score: strong feedback → stronger update
                effective_alpha = self._alpha * feedback_score

                self._w_vector = (
                    effective_alpha * observed_v + (1 - effective_alpha) * self._w_vector
                )
                self._w_bm25 = (
                    effective_alpha * observed_b + (1 - effective_alpha) * self._w_bm25
                )
                self._w_graph = (
                    effective_alpha * observed_g + (1 - effective_alpha) * self._w_graph
                )

                # Normalize
                self._w_vector, self._w_bm25, self._w_graph = self._normalize_weights(
                    self._w_vector, self._w_bm25, self._w_graph, self.MIN_WEIGHT,
                )
                self._save_weights()

    def get_optimized_weights(self) -> tuple[float, float, float]:
        """Gibt aktuelle optimierte Gewichte zurueck: (w_vector, w_bm25, w_graph)."""
        return (self._w_vector, self._w_bm25, self._w_graph)

    def report(self) -> dict[str, Any]:
        """Aktuelle Gewichte + Statistiken."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(feedback_score) as avg_score FROM search_outcomes"
        ).fetchone()

        return {
            "weights": {
                "vector": round(self._w_vector, 4),
                "bm25": round(self._w_bm25, 4),
                "graph": round(self._w_graph, 4),
            },
            "total_outcomes": row["cnt"] if row else 0,
            "avg_feedback_score": round(row["avg_score"], 4) if row and row["avg_score"] else 0.0,
            "alpha": self._alpha,
        }

    def close(self) -> None:
        """Schliesst die DB-Verbindung."""
        if self._conn:
            self._conn.close()
            self._conn = None
