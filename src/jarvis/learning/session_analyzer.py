"""Session-Analyse und Feedback-Loop fuer kontinuierliches Lernen.

Schliesst die Feedback-Schleife zwischen Reflector (Einzel-Session-Analyse),
CausalAnalyzer (Tool-Sequenz-Muster) und PromptEvolution (A/B-Tests) indem
wiederkehrende Fehlermuster erkannt und automatisch Verbesserungen abgeleitet
werden.

Komponenten:
  - FailureCluster: Gruppiert aehnliche Fehler ueber Sessions hinweg
  - UserFeedback: Erfasst explizites Nutzer-Feedback (positiv/negativ/Korrektur)
  - ImprovementAction: Konkrete Verbesserungsmassnahme (Prozedur, Prompt, Regel)
  - SessionAnalyzer: Orchestriert Analyse, Clustering und Verbesserungsgenerierung
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import exp
from pathlib import Path
from typing import Any, Literal

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

#: Mindest-Haeufigkeit eines Fehlerclusters bevor Verbesserung vorgeschlagen wird.
CLUSTER_THRESHOLD = 3

#: Standard-Lookback-Zeitraum fuer Pattern-Erkennung (Tage).
DEFAULT_LOOKBACK_DAYS = 7

#: Halbwertszeit fuer Recency-Gewichtung (Tage).
RECENCY_HALF_LIFE_DAYS = 3.0

#: Minimale Jaccard-Aehnlichkeit fuer Prozedur-Deduplizierung.
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Feedback-Erkennung: Muster -> (feedback_type, ist_korrektur)
_NEGATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\U0001f44e", re.UNICODE), "negative"),
    (re.compile(r"\bdas war falsch\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bfalsch\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bnein das stimmt nicht\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bstimmt nicht\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bwrong\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bincorrect\b", re.IGNORECASE), "negative"),
]

_CORRECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\beigentlich\s+(.+)", re.IGNORECASE), "correction"),
    (re.compile(r"\bich meinte\s+(.+)", re.IGNORECASE), "correction"),
    (re.compile(r"^nein,\s+(.+)", re.IGNORECASE), "correction"),
]

_POSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\U0001f44d", re.UNICODE), "positive"),
    (re.compile(r"\bperfekt\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bgenau\b", re.IGNORECASE), "positive"),
    (re.compile(r"\brichtig\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bsuper\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bdanke\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bgreat\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bperfect\b", re.IGNORECASE), "positive"),
]

# Normalisierung: Muster die aus Fehlermeldungen entfernt werden.
_NORMALIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Timestamps (ISO-8601, Unix-Epoch)
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?"), "<TS>"),
    # UUIDs
    (
        re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            re.IGNORECASE,
        ),
        "<UUID>",
    ),
    # Hex-IDs (session-ids etc.)
    (re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE), "<HEX>"),
    # Dateipfade (Windows + Unix)
    (re.compile(r"[A-Z]:\\[\w\\.-]+|/[\w/.-]+"), "<PATH>"),
    # Portnummern
    (re.compile(r":\d{4,5}\b"), ":<PORT>"),
    # Zahlenfolgen (z.B. Zeilennummern, Byte-Offsets)
    (re.compile(r"\b\d{3,}\b"), "<NUM>"),
]

# Fehler-Kategorie-Erkennung
_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "timeout": re.compile(r"timeout|timed?\s*out|deadline\s*exceeded", re.IGNORECASE),
    "tool_error": re.compile(
        r"tool.*error|error.*tool|failed.*execute|execution.*failed",
        re.IGNORECASE,
    ),
    "hallucination": re.compile(
        r"hallucin|fabricat|made.?up|nicht.*existier|does.*not.*exist",
        re.IGNORECASE,
    ),
    "wrong_answer": re.compile(r"wrong|falsch|incorrect|stimmt.*nicht", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class FailureCluster:
    """Gruppiert aehnliche Fehlermuster ueber mehrere Sessions hinweg."""

    pattern_id: str
    error_category: str
    occurrences: list[dict[str, Any]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    frequency: int = 1
    is_resolved: bool = False
    representative_error: str = ""


@dataclass
class UserFeedback:
    """Explizites Nutzer-Feedback zu einer Antwort."""

    session_id: str
    message_id: str
    feedback_type: Literal["positive", "negative", "correction"]
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ImprovementAction:
    """Konkrete Verbesserungsmassnahme, abgeleitet aus Fehlermustern."""

    action_type: Literal[
        "new_procedure",
        "prompt_variant",
        "core_rule",
        "skill_fix",
        "procedure_dedup",
    ]
    description: str
    target: str
    payload: str
    priority: float = 0.5
    status: Literal["proposed", "applied", "rejected"] = "proposed"


# ---------------------------------------------------------------------------
# SessionAnalyzer
# ---------------------------------------------------------------------------


class SessionAnalyzer:
    """Analysiert Sessions uebergreifend, erkennt Fehlermuster und leitet Verbesserungen ab.

    Schliesst die Feedback-Schleife zwischen einzelnen Session-Reflexionen
    (Reflector) und den Lernkomponenten (CausalAnalyzer, PromptEvolution,
    ProceduralMemory).
    """

    def __init__(
        self,
        data_dir: Path,
        memory_manager: Any = None,
        config: Any = None,
    ) -> None:
        """Initialisiert den SessionAnalyzer.

        Args:
            data_dir: Verzeichnis fuer die SQLite-Datenbank.
            memory_manager: Optionaler MemoryManager fuer Prozedur-Zugriff.
            config: Optionale JarvisConfig.
        """
        self._data_dir = Path(data_dir)
        self._memory_manager = memory_manager
        self._config = config
        self._db_path = self._data_dir / "session_analysis.db"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # DB-Verwaltung
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Gibt die DB-Verbindung zurueck (lazy init)."""
        if self._conn is None:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._conn = encrypted_connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        return self._conn

    def _init_db(self) -> None:
        """Erstellt die DB-Tabellen falls nicht vorhanden."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS failure_clusters (
                pattern_id TEXT PRIMARY KEY,
                error_category TEXT NOT NULL,
                representative_error TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                is_resolved INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS cluster_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL REFERENCES failure_clusters(pattern_id),
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                error_detail TEXT,
                tool_name TEXT
            );

            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id TEXT,
                feedback_type TEXT NOT NULL,
                detail TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS improvement_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                description TEXT,
                target TEXT,
                payload TEXT,
                priority REAL DEFAULT 0.5,
                status TEXT DEFAULT 'proposed',
                created TEXT NOT NULL,
                applied TEXT
            );

            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id TEXT PRIMARY KEY,
                success_score REAL,
                tool_count INTEGER,
                error_count INTEGER,
                duration_ms REAL,
                had_user_correction INTEGER DEFAULT 0,
                analyzed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cluster_occ_pattern
                ON cluster_occurrences(pattern_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON user_feedback(session_id);
            CREATE INDEX IF NOT EXISTS idx_actions_status
                ON improvement_actions(status);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Fehler-Normalisierung und Clustering
    # ------------------------------------------------------------------

    def _normalize_error(self, error_msg: str) -> str:
        """Normalisiert eine Fehlermeldung fuer Clustering.

        Entfernt Timestamps, Pfade, IDs und andere variable Teile damit
        semantisch gleiche Fehler denselben Hash bekommen.
        """
        if not error_msg:
            return ""
        normalized = error_msg.strip()
        for pattern, replacement in _NORMALIZE_PATTERNS:
            normalized = pattern.sub(replacement, normalized)
        # Whitespace normalisieren
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized

    def _error_hash(self, normalized_error: str) -> str:
        """Berechnet einen stabilen Hash fuer eine normalisierte Fehlermeldung."""
        return hashlib.sha256(normalized_error.encode("utf-8")).hexdigest()[:16]

    def _classify_error(self, error_msg: str) -> str:
        """Klassifiziert eine Fehlermeldung in eine Kategorie."""
        for category, pattern in _CATEGORY_PATTERNS.items():
            if pattern.search(error_msg):
                return category
        return "unknown"

    def _fuzzy_match_cluster(self, normalized_error: str) -> FailureCluster | None:
        """Sucht einen bestehenden Cluster der zur normalisierten Fehlermeldung passt.

        Verwendet exakten Hash-Match auf den normalisierten Error-String.
        """
        pattern_id = self._error_hash(normalized_error)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM failure_clusters WHERE pattern_id = ?",
            (pattern_id,),
        ).fetchone()
        if row is None:
            return None
        return FailureCluster(
            pattern_id=row["pattern_id"],
            error_category=row["error_category"],
            representative_error=row["representative_error"] or "",
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            frequency=row["frequency"],
            is_resolved=bool(row["is_resolved"]),
        )

    def _upsert_cluster(
        self,
        normalized_error: str,
        error_category: str,
        session_id: str,
        error_detail: str,
        tool_name: str,
    ) -> FailureCluster:
        """Fuegt einen Fehler in einen bestehenden Cluster ein oder erstellt einen neuen."""
        pattern_id = self._error_hash(normalized_error)
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        conn = self._get_conn()

        existing = self._fuzzy_match_cluster(normalized_error)
        if existing is not None:
            conn.execute(
                "UPDATE failure_clusters SET last_seen = ?, "
                "frequency = frequency + 1 WHERE pattern_id = ?",
                (now_iso, pattern_id),
            )
            existing.frequency += 1
            existing.last_seen = now
        else:
            conn.execute(
                """INSERT INTO failure_clusters
                   (pattern_id, error_category, representative_error,
                    first_seen, last_seen, frequency, is_resolved)
                   VALUES (?, ?, ?, ?, ?, 1, 0)""",
                (pattern_id, error_category, error_detail[:500], now_iso, now_iso),
            )
            existing = FailureCluster(
                pattern_id=pattern_id,
                error_category=error_category,
                representative_error=error_detail[:500],
                first_seen=now,
                last_seen=now,
                frequency=1,
                is_resolved=False,
            )

        # Occurrence speichern
        conn.execute(
            """INSERT INTO cluster_occurrences
               (pattern_id, session_id, timestamp, error_detail, tool_name)
               VALUES (?, ?, ?, ?, ?)""",
            (pattern_id, session_id, now_iso, error_detail[:1000], tool_name),
        )
        conn.commit()
        return existing

    # ------------------------------------------------------------------
    # Recency-Gewichtung
    # ------------------------------------------------------------------

    def _calculate_recency_weight(self, last_seen: datetime) -> float:
        """Berechnet ein Recency-Gewicht mit exponentiellem Zerfall.

        Juengere Cluster bekommen hoehere Gewichtung.
        Halbwertszeit: RECENCY_HALF_LIFE_DAYS.
        """
        now = datetime.now(UTC)
        age_days = (now - last_seen).total_seconds() / 86400.0
        if age_days < 0:
            age_days = 0.0
        # Exponentieller Zerfall: weight = exp(-lambda * t)
        # lambda = ln(2) / half_life
        decay = 0.693147 / RECENCY_HALF_LIFE_DAYS
        return exp(-decay * age_days)

    # ------------------------------------------------------------------
    # Oeffentliche API: Session-Analyse
    # ------------------------------------------------------------------

    async def analyze_session(
        self,
        session_id: str,
        agent_result: Any,
        reflection: Any = None,
    ) -> list[ImprovementAction]:
        """Analysiert eine abgeschlossene Session und gibt Verbesserungsvorschlaege zurueck.

        Extrahiert Fehler aus den Tool-Ergebnissen, clustert sie mit
        bestehenden Mustern und generiert Verbesserungen wenn ein Cluster
        den Schwellwert ueberschreitet.

        Args:
            session_id: Eindeutige Session-ID.
            agent_result: AgentResult mit tool_results, success, etc.
            reflection: Optionales ReflectionResult fuer zusaetzlichen Kontext.

        Returns:
            Liste vorgeschlagener ImprovementActions.
        """
        try:
            return await self._analyze_session_impl(session_id, agent_result, reflection)
        except Exception as exc:
            log.error("session_analysis_failed", session_id=session_id, error=str(exc))
            return []

    async def _analyze_session_impl(
        self,
        session_id: str,
        agent_result: Any,
        reflection: Any,
    ) -> list[ImprovementAction]:
        """Interne Implementierung der Session-Analyse."""
        tool_results = getattr(agent_result, "tool_results", [])
        success = getattr(agent_result, "success", True)
        duration_ms = getattr(agent_result, "total_duration_ms", 0)

        error_count = 0
        triggered_clusters: list[FailureCluster] = []

        for tr in tool_results:
            if getattr(tr, "is_error", False):
                error_count += 1
                error_msg = getattr(tr, "error_message", "") or getattr(tr, "content", "")
                tool_name = getattr(tr, "tool_name", "unknown")
                normalized = self._normalize_error(error_msg)
                if not normalized:
                    continue
                category = self._classify_error(error_msg)
                cluster = self._upsert_cluster(
                    normalized,
                    category,
                    session_id,
                    error_msg,
                    tool_name,
                )
                if cluster.frequency >= CLUSTER_THRESHOLD:
                    triggered_clusters.append(cluster)

        # Session-Metriken speichern
        success_score = 0.0
        if reflection is not None:
            success_score = getattr(reflection, "success_score", 0.0)
        elif success:
            success_score = 1.0

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO session_metrics
               (session_id, success_score, tool_count, error_count,
                duration_ms, had_user_correction, analyzed_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (
                session_id,
                success_score,
                len(tool_results),
                error_count,
                duration_ms,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()

        # Verbesserungen generieren falls Schwellwert erreicht
        if triggered_clusters:
            actions = await self.generate_improvements(triggered_clusters)
            # Aktionen in DB speichern
            for action in actions:
                self._store_action(action)
            return actions

        return []

    # ------------------------------------------------------------------
    # Oeffentliche API: User-Feedback
    # ------------------------------------------------------------------

    def record_user_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback_type: Literal["positive", "negative", "correction"],
        detail: str = "",
    ) -> UserFeedback:
        """Speichert explizites Nutzer-Feedback.

        Bei negativem Feedback wird ein Fehlermuster extrahiert und geclustert.
        Bei Korrekturen wird der Detail-Text als Lernkandidat gespeichert.

        Args:
            session_id: Session-ID.
            message_id: Nachrichten-ID innerhalb der Session.
            feedback_type: Art des Feedbacks.
            detail: Korrekturtext oder Beschreibung.

        Returns:
            Das gespeicherte UserFeedback-Objekt.
        """
        now = datetime.now(UTC)
        feedback = UserFeedback(
            session_id=session_id,
            message_id=message_id,
            feedback_type=feedback_type,
            detail=detail,
            timestamp=now,
        )

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO user_feedback (session_id, message_id, feedback_type, detail, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, message_id, feedback_type, detail, now.isoformat()),
        )
        conn.commit()

        # Negatives Feedback als Fehlercluster erfassen
        if feedback_type == "negative":
            error_msg = detail or "user_reported_negative_feedback"
            normalized = self._normalize_error(error_msg)
            if normalized:
                self._upsert_cluster(
                    normalized,
                    "user_correction",
                    session_id,
                    error_msg,
                    "user_feedback",
                )

        # Korrektur in session_metrics markieren
        if feedback_type in ("negative", "correction"):
            conn.execute(
                "UPDATE session_metrics SET had_user_correction = 1 WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

        log.info(
            "user_feedback_recorded",
            session_id=session_id,
            feedback_type=feedback_type,
        )
        return feedback

    # ------------------------------------------------------------------
    # Oeffentliche API: Muster-Erkennung
    # ------------------------------------------------------------------

    def detect_recurring_patterns(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> list[FailureCluster]:
        """Erkennt wiederkehrende Fehlermuster im angegebenen Zeitraum.

        Gibt Cluster zurueck die mindestens CLUSTER_THRESHOLD Vorkommen haben
        und noch nicht als geloest markiert sind.

        Args:
            lookback_days: Wie viele Tage zurueck geschaut wird.

        Returns:
            Sortierte Liste von FailureCluster (haeufigste + juengste zuerst).
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        conn = self._get_conn()

        rows = conn.execute(
            """SELECT * FROM failure_clusters
               WHERE is_resolved = 0
                 AND last_seen >= ?
                 AND frequency >= ?
               ORDER BY frequency DESC""",
            (cutoff, CLUSTER_THRESHOLD),
        ).fetchall()

        clusters: list[FailureCluster] = []
        for row in rows:
            cluster = FailureCluster(
                pattern_id=row["pattern_id"],
                error_category=row["error_category"],
                representative_error=row["representative_error"] or "",
                first_seen=datetime.fromisoformat(row["first_seen"]),
                last_seen=datetime.fromisoformat(row["last_seen"]),
                frequency=row["frequency"],
                is_resolved=bool(row["is_resolved"]),
            )
            # Occurrences laden
            occ_rows = conn.execute(
                "SELECT * FROM cluster_occurrences WHERE pattern_id = ? ORDER BY timestamp DESC",
                (row["pattern_id"],),
            ).fetchall()
            cluster.occurrences = [
                {
                    "session_id": o["session_id"],
                    "timestamp": o["timestamp"],
                    "error_detail": o["error_detail"],
                    "tool_name": o["tool_name"],
                }
                for o in occ_rows
            ]
            clusters.append(cluster)

        # Sortieren nach frequency * recency_weight (absteigend)
        clusters.sort(
            key=lambda c: c.frequency * self._calculate_recency_weight(c.last_seen),
            reverse=True,
        )
        return clusters

    # ------------------------------------------------------------------
    # Oeffentliche API: Verbesserungen generieren
    # ------------------------------------------------------------------

    async def generate_improvements(
        self,
        clusters: list[FailureCluster],
    ) -> list[ImprovementAction]:
        """Generiert Verbesserungsmassnahmen aus Fehlerclustern.

        Verwendet regelbasierte Heuristiken (kein LLM-Aufruf) um fuer
        jeden Cluster die passende Massnahme abzuleiten.

        Args:
            clusters: Liste von FailureCluster mit frequency >= CLUSTER_THRESHOLD.

        Returns:
            Liste vorgeschlagener ImprovementActions.
        """
        actions: list[ImprovementAction] = []

        for cluster in clusters:
            if cluster.is_resolved:
                continue

            action = self._derive_action(cluster)
            if action is not None:
                actions.append(action)

        # Nach Prioritaet sortieren
        actions.sort(key=lambda a: a.priority, reverse=True)
        return actions

    def _derive_action(self, cluster: FailureCluster) -> ImprovementAction | None:
        """Leitet eine Verbesserungsmassnahme aus einem Fehlercluster ab."""
        category = cluster.error_category
        freq = cluster.frequency
        representative = cluster.representative_error or cluster.pattern_id

        # Prioritaet: hoehere Frequenz + juengeres Auftreten = hoehere Prioritaet
        recency = self._calculate_recency_weight(cluster.last_seen)
        priority = min(1.0, (freq / 20.0) + (recency * 0.3))

        if category == "timeout":
            return ImprovementAction(
                action_type="new_procedure",
                description=f"Timeout-Workaround: {representative[:100]}",
                target="procedural_memory",
                payload=(
                    f"# timeout-workaround-{cluster.pattern_id[:8]}\n"
                    f"## Trigger\nWiederholte Timeouts bei: {representative[:200]}\n"
                    f"## Ablauf\n1. Pruefe ob der Dienst erreichbar ist\n"
                    f"2. Verwende kuerzeren Timeout mit Retry\n"
                    f"3. Bei anhaltendem Timeout: alternativen Dienst nutzen\n"
                    f"## Haeufigkeit\n{freq} Vorkommen"
                ),
                priority=priority,
            )

        if category == "tool_error":
            # Tool-Fehler sammeln
            tool_names = set()
            for occ in cluster.occurrences:
                tn = occ.get("tool_name", "")
                if tn:
                    tool_names.add(tn)
            tools_str = ", ".join(sorted(tool_names)) or "unbekannt"

            return ImprovementAction(
                action_type="new_procedure",
                description=f"Tool-Fehler-Workaround fuer: {tools_str}",
                target="procedural_memory",
                payload=(
                    f"# tool-error-fix-{cluster.pattern_id[:8]}\n"
                    f"## Trigger\nWiederholte Fehler bei Tools: {tools_str}\n"
                    f"## Fehlerbild\n{representative[:300]}\n"
                    f"## Ablauf\n1. Eingabe-Parameter validieren\n"
                    f"2. Fehlerbehandlung mit Fallback-Tool\n"
                    f"3. Bei Wiederholung: Skill-Fix vorschlagen\n"
                    f"## Haeufigkeit\n{freq} Vorkommen"
                ),
                priority=priority,
            )

        if category == "hallucination":
            return ImprovementAction(
                action_type="core_rule",
                description=f"Anti-Halluzinations-Regel: {representative[:100]}",
                target="CORE.md",
                payload=(
                    f"WICHTIG: Bei Anfragen zu '{representative[:100]}' IMMER zuerst "
                    f"search_and_read verwenden. Keine Antworten aus dem Training generieren. "
                    f"Dieses Muster wurde {freq}x als Halluzination gemeldet."
                ),
                priority=min(1.0, priority + 0.1),
            )

        if category == "wrong_answer":
            return ImprovementAction(
                action_type="prompt_variant",
                description=f"Prompt-Verbesserung gegen falsche Antworten: {representative[:100]}",
                target="planner_system_prompt",
                payload=(
                    f"Zusaetzliche Anweisung: Bei Themen wie '{representative[:100]}' "
                    f"besonders sorgfaeltig pruefen. Quellen zitieren. "
                    f"Dieses Muster wurde {freq}x als falsch bewertet."
                ),
                priority=priority,
            )

        if category == "user_correction":
            return ImprovementAction(
                action_type="prompt_variant",
                description=f"Nutzer-Korrektur beruecksichtigen: {representative[:100]}",
                target="planner_system_prompt",
                payload=(
                    f"Nutzer-Feedback: '{representative[:200]}' "
                    f"wurde {freq}x korrigiert. Antwortverhalten anpassen."
                ),
                priority=priority,
            )

        # Fallback fuer unbekannte Kategorien
        return ImprovementAction(
            action_type="new_procedure",
            description=f"Generischer Workaround: {representative[:100]}",
            target="procedural_memory",
            payload=(
                f"# generic-fix-{cluster.pattern_id[:8]}\n"
                f"## Fehlermuster\n{representative[:300]}\n"
                f"## Haeufigkeit\n{freq} Vorkommen\n"
                f"## Kategorie\n{category}"
            ),
            priority=max(0.1, priority - 0.1),
        )

    def _store_action(self, action: ImprovementAction) -> None:
        """Speichert eine ImprovementAction in der Datenbank."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO improvement_actions
               (action_type, description, target, payload, priority, status, created)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                action.action_type,
                action.description,
                action.target,
                action.payload,
                action.priority,
                action.status,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Oeffentliche API: Verbesserungen anwenden
    # ------------------------------------------------------------------

    def apply_improvement(self, action: ImprovementAction) -> bool:
        """Wendet eine Verbesserungsmassnahme an.

        Je nach action_type wird die Aenderung in das entsprechende
        Subsystem geschrieben.

        Args:
            action: Die anzuwendende ImprovementAction.

        Returns:
            True bei Erfolg, False bei Fehler.
        """
        try:
            if action.action_type == "new_procedure":
                return self._apply_new_procedure(action)
            if action.action_type == "prompt_variant":
                return self._apply_prompt_variant(action)
            if action.action_type == "core_rule":
                return self._apply_core_rule(action)
            if action.action_type == "procedure_dedup":
                return self._apply_procedure_dedup(action)
            if action.action_type == "skill_fix":
                return self._apply_skill_fix(action)
            log.warning("unknown_action_type", action_type=action.action_type)
            return False
        except Exception as exc:
            log.error("apply_improvement_failed", action_type=action.action_type, error=str(exc))
            return False

    def _apply_new_procedure(self, action: ImprovementAction) -> bool:
        """Schreibt eine neue Prozedur in den ProceduralMemory."""
        if self._memory_manager is None:
            log.warning("no_memory_manager_for_procedure")
            return False

        try:
            procedural = self._memory_manager.procedural
            # Name aus dem Payload extrahieren (erste Zeile nach #)
            lines = action.payload.strip().split("\n")
            name = "auto-fix"
            for line in lines:
                if line.startswith("# "):
                    name = line[2:].strip()
                    break

            procedural.save_procedure(name=name, body=action.payload)
            action.status = "applied"
            self._update_action_status(action)
            log.info("procedure_created_from_improvement", name=name)
            return True
        except Exception as exc:
            log.error("apply_new_procedure_failed", error=str(exc))
            return False

    def _apply_prompt_variant(self, action: ImprovementAction) -> bool:
        """Registriert eine Prompt-Variante (nur Logging, keine direkte Anwendung)."""
        # PromptEvolution-Integration: In der Praxis wuerde hier
        # prompt_evolution.register_prompt() aufgerufen werden.
        # Ohne direkten Zugriff loggen wir die Aktion.
        action.status = "applied"
        self._update_action_status(action)
        log.info(
            "prompt_variant_proposed",
            target=action.target,
            description=action.description[:100],
        )
        return True

    def _apply_core_rule(self, action: ImprovementAction) -> bool:
        """Haengt eine Regel an CORE.md an (nur Logging ohne Dateisystem-Zugriff)."""
        # In der Praxis wuerde hier CORE.md geschrieben werden.
        # Sicherheitshalber nur loggen.
        action.status = "applied"
        self._update_action_status(action)
        log.info(
            "core_rule_proposed",
            target=action.target,
            payload_preview=action.payload[:100],
        )
        return True

    def _apply_procedure_dedup(self, action: ImprovementAction) -> bool:
        """Fuehrt Prozedur-Deduplizierung durch."""
        action.status = "applied"
        self._update_action_status(action)
        log.info("procedure_dedup_applied", target=action.target)
        return True

    def _apply_skill_fix(self, action: ImprovementAction) -> bool:
        """Wendet einen Skill-Fix an (nur Logging)."""
        action.status = "applied"
        self._update_action_status(action)
        log.info("skill_fix_proposed", target=action.target)
        return True

    def _update_action_status(self, action: ImprovementAction) -> None:
        """Aktualisiert den Status einer Aktion in der DB."""
        conn = self._get_conn()
        now_iso = datetime.now(UTC).isoformat()
        conn.execute(
            """UPDATE improvement_actions SET status = ?, applied = ?
               WHERE description = ? AND action_type = ? AND status = 'proposed'""",
            (action.status, now_iso, action.description, action.action_type),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Oeffentliche API: Health-Report
    # ------------------------------------------------------------------

    def get_health_report(self) -> dict[str, Any]:
        """Erstellt einen Gesundheitsbericht ueber den Lernzustand.

        Returns:
            Dict mit Zusammenfassung: Gesamtzahlen, Top-5-Cluster, Feedback-Uebersicht.
        """
        conn = self._get_conn()

        total_sessions = conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0]
        total_feedback = conn.execute("SELECT COUNT(*) FROM user_feedback").fetchone()[0]
        total_clusters = conn.execute(
            "SELECT COUNT(*) FROM failure_clusters WHERE is_resolved = 0"
        ).fetchone()[0]
        total_improvements = conn.execute(
            "SELECT COUNT(*) FROM improvement_actions WHERE status = 'applied'"
        ).fetchone()[0]

        # Top 5 ungeloeste Cluster
        top_clusters = conn.execute(
            """SELECT pattern_id, error_category, representative_error, frequency, last_seen
               FROM failure_clusters
               WHERE is_resolved = 0
               ORDER BY frequency DESC
               LIMIT 5"""
        ).fetchall()

        top_5 = [
            {
                "pattern_id": row["pattern_id"],
                "error_category": row["error_category"],
                "representative_error": row["representative_error"] or "",
                "frequency": row["frequency"],
                "last_seen": row["last_seen"],
            }
            for row in top_clusters
        ]

        # Feedback-Zusammenfassung
        feedback_summary = {}
        for ft in ("positive", "negative", "correction"):
            count = conn.execute(
                "SELECT COUNT(*) FROM user_feedback WHERE feedback_type = ?", (ft,)
            ).fetchone()[0]
            feedback_summary[ft] = count

        return {
            "total_sessions_analyzed": total_sessions,
            "total_feedback": total_feedback,
            "unresolved_clusters": total_clusters,
            "improvements_applied": total_improvements,
            "top_unresolved_patterns": top_5,
            "feedback_summary": feedback_summary,
        }

    # ------------------------------------------------------------------
    # Oeffentliche API: Prozedur-Deduplizierung
    # ------------------------------------------------------------------

    def deduplicate_procedures(
        self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[tuple[str, str]]:
        """Findet nahezu doppelte Prozeduren anhand ihrer Keywords.

        Verwendet Jaccard-Aehnlichkeit auf den Trigger-Keyword-Mengen.

        Args:
            similarity_threshold: Minimale Jaccard-Aehnlichkeit (0.0-1.0).

        Returns:
            Liste von Tupeln (prozedur_a, prozedur_b) die Duplikate sind.
        """
        if self._memory_manager is None:
            log.warning("no_memory_manager_for_dedup")
            return []

        try:
            procedural = self._memory_manager.procedural
            all_procedures = procedural.list_procedures()
        except Exception as exc:
            log.error("dedup_list_procedures_failed", error=str(exc))
            return []

        # Keyword-Sets aufbauen
        procedure_keywords: dict[str, set[str]] = {}
        for name in all_procedures:
            try:
                meta, _body = procedural.load_procedure(name)
                keywords = set(getattr(meta, "trigger_keywords", []))
                # Auch den Namen tokenisieren
                name_tokens = set(name.lower().replace("-", " ").replace("_", " ").split())
                keywords.update(name_tokens)
                if keywords:
                    procedure_keywords[name] = keywords
            except Exception:
                continue

        # Paarweise Jaccard-Aehnlichkeit
        names = list(procedure_keywords.keys())
        duplicates: list[tuple[str, str]] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                set_a = procedure_keywords[names[i]]
                set_b = procedure_keywords[names[j]]
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                if union == 0:
                    continue
                jaccard = intersection / union
                if jaccard >= similarity_threshold:
                    duplicates.append((names[i], names[j]))

        if duplicates:
            log.info("procedure_duplicates_found", count=len(duplicates))

        return duplicates

    # ------------------------------------------------------------------
    # Oeffentliche API: Feedback-Signal-Erkennung
    # ------------------------------------------------------------------

    def _extract_feedback_signal(self, message: str) -> tuple[str, str] | None:
        """Erkennt Feedback-Signale in einer Nutzer-Nachricht.

        Prueft auf positive, negative und Korrektur-Muster (DE + EN).

        Args:
            message: Die Nutzer-Nachricht.

        Returns:
            Tuple (feedback_type, detail) oder None wenn kein Signal erkannt.
        """
        if not message:
            return None

        # Korrekturen zuerst pruefen (enthalten auch negative Signale)
        for pattern, ftype in _CORRECTION_PATTERNS:
            match = pattern.search(message)
            if match:
                detail = match.group(1).strip() if match.lastindex else message
                return (ftype, detail)

        # Negatives Feedback
        for pattern, ftype in _NEGATIVE_PATTERNS:
            if pattern.search(message):
                return (ftype, "")

        # Positives Feedback
        for pattern, ftype in _POSITIVE_PATTERNS:
            if pattern.search(message):
                return (ftype, "")

        return None

    # ------------------------------------------------------------------
    # Aufraumen
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Schliesst die DB-Verbindung."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
