"""Performance-based weight adjustment for HybridSearch.

Uses Exponential Moving Average (EMA) over channel usefulness,
measured by user satisfaction (reflector score).

EMA-Formel: w_new = alpha * observed + (1-alpha) * w_old
Constraints: Jedes Gewicht min 0.05, Summe = 1.0

Operational-Trust PR-B (2026-05-05): every successful ``record_outcome``
call now produces a Fernet-encrypted snapshot of the active weight
vector under ``<snapshot_dir>/<weight_sha256>.fernet`` plus a plaintext
``<weight_sha256>.meta.json`` sidecar. Snapshots are content-addressed
by the SHA-256 of the canonical-NFC-JSON of the plaintext weights, so
identical weights produce identical hashes and the file is written at
most once. The snapshot SHA-256 is forwarded to the Reflector's audit
channel (PR-A) so TRUST-1 receipt verifiers can chain-verify which
weight vector was active at run time without needing the encryption
key.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cognithor.db import SQLITE_BUSY_TIMEOUT_MS
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

    from cognithor.security.encrypted_file import EncryptedFileIO

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
        *,
        encrypted_file_io: EncryptedFileIO | None = None,
        snapshot_dir: Path | None = None,
        audit_emit_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Operational-Trust PR-B: snapshot-related kwargs.

        ``encrypted_file_io`` is an :class:`EncryptedFileIO` instance
        wired to the same OS-keyring chain as the rest of the storage
        stack; it abstracts Fernet encryption + key loading. Snapshots
        require both ``encrypted_file_io`` AND ``snapshot_dir`` to be
        non-None — otherwise the snapshot step is skipped (with a
        ``log.debug("weight_snapshot_skipped_no_io")``) and the
        optimizer behaves exactly like before.

        ``audit_emit_callback`` is the Reflector's
        ``_emit_reflection_audit_event`` helper. When set, every
        persisted snapshot fires a ``weight_snapshot_persisted`` event
        with ``weight_sha256`` + ``session_id`` + ``snapshot_bytes`` so
        the TRUST-1 receipt can cross-verify the active weight vector
        for a run. ``None`` keeps the legacy debug-only behaviour.
        """
        self._db_path = str(db_path) if db_path else ":memory:"
        self._alpha = alpha
        self._conn: sqlite3.Connection | None = None
        self._encrypted_file_io: EncryptedFileIO | None = encrypted_file_io
        self._snapshot_dir: Path | None = snapshot_dir
        self._audit_emit_callback: Callable[[str, dict[str, Any]], None] | None = (
            audit_emit_callback
        )

        # Current weights (vector, bm25, graph)
        if initial_weights:
            self._w_vector, self._w_bm25, self._w_graph = initial_weights
        else:
            self._w_vector = 0.50
            self._w_bm25 = 0.30
            self._w_graph = 0.20

        self._ensure_schema()
        self._load_weights()

    def set_audit_emit_callback(
        self,
        callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        """Late-bind the audit-emit callback (Operational-Trust PR-B).

        Mirrors :meth:`CausalAnalyzer.set_audit_emit_callback` from PR-A.
        Used by the gateway boot path: the optimizer is constructed
        inside :class:`MemoryManager` before the :class:`Reflector`
        exists, so the helper cannot be passed at ``__init__`` time.
        """
        self._audit_emit_callback = callback

    def set_snapshot_io(
        self,
        encrypted_file_io: EncryptedFileIO | None,
        snapshot_dir: Path | None,
    ) -> None:
        """Late-bind the snapshot encryption + directory pair.

        Production wiring: the gateway constructs ``EncryptedFileIO``
        after the :class:`MemoryManager` (which constructs this
        optimizer). The pair must be set together — either both
        non-None to enable snapshots, or both None to disable them.
        """
        self._encrypted_file_io = encrypted_file_io
        self._snapshot_dir = snapshot_dir

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
        w_vector: float,
        w_bm25: float,
        w_graph: float,
        min_w: float = 0.05,
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
        *,
        session_id: str = "",
    ) -> None:
        """Zeichnet ein Suchergebnis auf und aktualisiert Gewichte via EMA.

        Args:
            query: Die Suchanfrage.
            channel_contributions: {"vector": 0-1, "bm25": 0-1, "graph": 0-1}
            feedback_score: Nuetzlichkeit des Ergebnisses (0-1, z.B. Reflector-Score).
            session_id: Operational-Trust PR-B correlation key. Forwarded to
                the snapshot meta + audit event so a TRUST-1 receipt can
                cross-link the weight vector active during this run.
                Empty string when called outside a Plan→Gate→Execute scope.
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
                self._w_bm25 = effective_alpha * observed_b + (1 - effective_alpha) * self._w_bm25
                self._w_graph = effective_alpha * observed_g + (1 - effective_alpha) * self._w_graph

                # Normalize
                self._w_vector, self._w_bm25, self._w_graph = self._normalize_weights(
                    self._w_vector,
                    self._w_bm25,
                    self._w_graph,
                    self.MIN_WEIGHT,
                )
                self._save_weights()

        # Operational-Trust PR-B: persist a content-addressed Fernet
        # snapshot of the (possibly updated) weight vector + sidecar.
        # Best-effort: a snapshot failure must not break the EMA update.
        self._persist_snapshot(session_id=session_id)

    # ------------------------------------------------------------------
    # Operational-Trust PR-B: snapshot machinery
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_weight_bytes(weights: dict[str, float]) -> bytes:
        """Return canonical-NFC-JSON bytes of a weight dict.

        Uses the SAME canonical recipe as PR-A's
        ``Reflector._emit_reflection_audit_event``: NFC normalisation +
        ``json.dumps(sort_keys=True, ensure_ascii=False)``. Bit-identical
        across processes, enabling content-addressed deduplication.
        """
        canonical = unicodedata.normalize(
            "NFC",
            json.dumps(weights, sort_keys=True, ensure_ascii=False),
        )
        return canonical.encode("utf-8")

    def current_weight_vector(self) -> dict[str, float]:
        """Return the active weight vector as a plain dict.

        Stable key order is enforced by callers via the canonical-form
        recipe; this getter just exposes the live values.
        """
        return {
            "vector": self._w_vector,
            "bm25": self._w_bm25,
            "graph": self._w_graph,
        }

    def _persist_snapshot(self, *, session_id: str = "") -> str | None:
        """Persist a Fernet-encrypted snapshot + plaintext sidecar.

        Returns the ``weight_sha256`` hash on success (whether or not
        the file was newly written — content-addressed dedup means
        identical weights produce identical hashes). Returns ``None``
        when snapshots are disabled or the encryption layer is
        unavailable.

        Layout under ``self._snapshot_dir``:

        * ``<weight_sha256>.fernet`` — Fernet-encrypted JSON of the
          weight dict (via ``EncryptedFileIO.write``)
        * ``<weight_sha256>.meta.json`` — plaintext sidecar with
          ``weight_sha256`` + ``fernet_file`` + ``created_at`` +
          ``session_id`` + ``snapshot_bytes``. Lets receipt verifiers
          observe existence + chain link without key access.

        On the success path with an audit callback wired, emits a
        ``weight_snapshot_persisted`` reflection event carrying the
        same triple (``weight_sha256``, ``session_id``,
        ``snapshot_bytes``) so the TRUST-1 receipt can cross-link.
        """
        if self._encrypted_file_io is None or self._snapshot_dir is None:
            log.debug("weight_snapshot_skipped_no_io")
            return None

        # ``EncryptedFileIO.is_available`` triggers lazy-init + key
        # lookup. False = no key in env / keyring / credential store —
        # the file would be written as plaintext, which would silently
        # break the encrypted-at-rest contract. Skip with a debug log.
        try:
            if not self._encrypted_file_io.is_available:
                log.debug("weight_snapshot_skipped_no_key")
                return None
        except Exception as exc:
            log.debug("weight_snapshot_io_probe_failed", error=str(exc))
            return None

        weights = self.current_weight_vector()
        try:
            canonical = self._canonical_weight_bytes(weights)
            weight_sha256 = hashlib.sha256(canonical).hexdigest()
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            fernet_path = self._snapshot_dir / f"{weight_sha256}.fernet"
            meta_path = self._snapshot_dir / f"{weight_sha256}.meta.json"

            snapshot_bytes = len(canonical)
            # Content-addressed dedup: same weights → same hash → same
            # filename. Skip rewrite when the encrypted file already
            # exists (the meta sidecar may legitimately be re-emitted
            # because callers want a per-run created_at + session_id).
            if not fernet_path.exists():
                # ``EncryptedFileIO.write`` accepts ``str`` content; the
                # plaintext we pass is the canonical-NFC-JSON string —
                # the bytes the SHA-256 was computed over. Decryption
                # by an operator returns exactly the same bytes.
                self._encrypted_file_io.write(fernet_path, canonical.decode("utf-8"))

            meta_payload = {
                "weight_sha256": weight_sha256,
                "fernet_file": fernet_path.name,
                "created_at": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "snapshot_bytes": snapshot_bytes,
            }
            meta_path.write_text(
                json.dumps(meta_payload, sort_keys=True, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("weight_snapshot_persist_failed", error=str(exc))
            return None

        if self._audit_emit_callback is not None:
            try:
                self._audit_emit_callback(
                    "weight_snapshot_persisted",
                    {
                        "weight_sha256": weight_sha256,
                        "session_id": session_id,
                        "snapshot_bytes": snapshot_bytes,
                    },
                )
            except Exception as exc:
                log.warning("weight_snapshot_audit_emit_failed", error=str(exc))

        return weight_sha256

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
        """Close the DB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
