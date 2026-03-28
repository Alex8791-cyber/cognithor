"""PromptEvolutionEngine: A/B-test-based prompt optimization.

Tracks prompt versions, runs deterministic A/B splits per session,
records reward scores, and (optionally) generates improved variants
via a meta-LLM call.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time as _time
from datetime import UTC, datetime
from typing import Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

logger = get_logger(__name__)


class PromptVersionStore:
    """SQLite-backed store for prompt versions, sessions, and A/B tests."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = encrypted_connect(db_path, check_same_thread=False)
        self._conn.row_factory = compatible_row_factory()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id TEXT PRIMARY KEY,
                template_name TEXT NOT NULL,
                template_text TEXT NOT NULL,
                parent_id TEXT,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS prompt_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt_version_id TEXT NOT NULL,
                reward_score REAL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (prompt_version_id) REFERENCES prompt_versions(id)
            );

            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_name TEXT NOT NULL,
                version_a_id TEXT NOT NULL,
                version_b_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                winner_id TEXT,
                sessions_a INTEGER DEFAULT 0,
                sessions_b INTEGER DEFAULT 0,
                avg_reward_a REAL DEFAULT 0,
                avg_reward_b REAL DEFAULT 0,
                status TEXT DEFAULT 'running'
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()


def _version_id(template_name: str, template_text: str) -> str:
    """Generate a deterministic version ID (first 16 hex chars of SHA-256)."""
    h = hashlib.sha256(f"{template_name}:{template_text}".encode()).hexdigest()
    return h[:16]


class PromptEvolutionEngine:
    """Engine for evolving prompts through A/B testing.

    Workflow:
    1. Register prompt templates (original versions).
    2. During inference, get_active_version() returns the version to use,
       respecting any running A/B test (deterministic split via session hash).
    3. After inference, record_session() stores the reward for the version used.
    4. Periodically, maybe_evolve() evaluates completed tests and optionally
       generates new variants via a meta-LLM.
    """

    MIN_SESSIONS_PER_ARM = 20
    SIGNIFICANCE_THRESHOLD = 0.05

    def __init__(
        self,
        db_path: str,
        llm_client: Any = None,
        min_sessions_per_arm: int | None = None,
        significance_threshold: float | None = None,
        max_concurrent_tests: int = 1,
    ) -> None:
        self._store = PromptVersionStore(db_path)
        self._conn = self._store._conn
        self._llm_client = llm_client
        self._max_concurrent_tests = max_concurrent_tests
        self._last_evolution_at: float = 0.0
        self._evolution_interval_seconds: float = 6 * 3600

        if min_sessions_per_arm is not None:
            self.MIN_SESSIONS_PER_ARM = min_sessions_per_arm
        if significance_threshold is not None:
            self.SIGNIFICANCE_THRESHOLD = significance_threshold

    def set_evolution_interval_hours(self, hours: int) -> None:
        """Configure the minimum interval between evolution attempts."""
        self._evolution_interval_seconds = hours * 3600

    def register_prompt(self, template_name: str, template_text: str) -> str:
        """Register a prompt template. Returns version_id.

        If the exact same template_name+text already exists, returns existing ID.
        The first registered version for a template_name becomes active.
        """
        vid = _version_id(template_name, template_text)

        existing = self._conn.execute(
            "SELECT id FROM prompt_versions WHERE id = ?", (vid,)
        ).fetchone()
        if existing:
            return vid

        now = datetime.now(UTC).isoformat()

        # Check if this is the first version for the template
        has_active = self._conn.execute(
            "SELECT 1 FROM prompt_versions WHERE template_name = ? AND is_active = 1",
            (template_name,),
        ).fetchone()
        is_active = 0 if has_active else 1

        self._conn.execute(
            """
            INSERT INTO prompt_versions (id, template_name, template_text, parent_id,
                                         created_at, is_active, metadata_json)
            VALUES (?, ?, ?, NULL, ?, ?, '{}')
            """,
            (vid, template_name, template_text, now, is_active),
        )
        self._conn.commit()
        logger.info(
            "prompt_version_registered", template=template_name, version=vid, active=bool(is_active)
        )
        return vid

    def get_active_version(
        self, template_name: str, session_id: str = "default"
    ) -> tuple[str, str]:
        """Get the prompt version to use for this session.

        If an A/B test is running, deterministically assigns the session to
        arm A or B via hash(session_id) % 2.

        Returns:
            (version_id, template_text)
        """
        # Check for running A/B test
        test = self._conn.execute(
            "SELECT * FROM ab_tests WHERE template_name = ? AND status = 'running' "
            "ORDER BY id DESC LIMIT 1",
            (template_name,),
        ).fetchone()

        if test is not None:
            import hashlib

            arm = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 2
            version_id = test["version_a_id"] if arm == 0 else test["version_b_id"]
            row = self._conn.execute(
                "SELECT template_text FROM prompt_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if row:
                return version_id, row["template_text"]

        # No test running: return active version
        row = self._conn.execute(
            "SELECT id, template_text FROM prompt_versions "
            "WHERE template_name = ? AND is_active = 1 LIMIT 1",
            (template_name,),
        ).fetchone()
        if row:
            return row["id"], row["template_text"]

        raise ValueError(f"No registered prompt for template '{template_name}'")

    def record_session(
        self,
        session_id: str,
        prompt_version_id: str,
        reward: float,
    ) -> None:
        """Record the reward score for a session that used a specific prompt version."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO prompt_sessions "
            "(session_id, prompt_version_id, "
            "reward_score, recorded_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, prompt_version_id, reward, now),
        )

        # Update A/B test stats if this version is part of a running test
        test = self._conn.execute(
            "SELECT * FROM ab_tests WHERE status = 'running' AND "
            "(version_a_id = ? OR version_b_id = ?)",
            (prompt_version_id, prompt_version_id),
        ).fetchone()

        if test is not None:
            if prompt_version_id == test["version_a_id"]:
                new_count = test["sessions_a"] + 1
                new_avg = (test["avg_reward_a"] * test["sessions_a"] + reward) / new_count
                self._conn.execute(
                    "UPDATE ab_tests SET sessions_a = ?, avg_reward_a = ? WHERE id = ?",
                    (new_count, new_avg, test["id"]),
                )
            else:
                new_count = test["sessions_b"] + 1
                new_avg = (test["avg_reward_b"] * test["sessions_b"] + reward) / new_count
                self._conn.execute(
                    "UPDATE ab_tests SET sessions_b = ?, avg_reward_b = ? WHERE id = ?",
                    (new_count, new_avg, test["id"]),
                )

        self._conn.commit()

    def start_ab_test(self, template_name: str, version_a_id: str, version_b_id: str) -> int:
        """Start a new A/B test between two versions.

        Returns the test ID.
        """
        running_count = self._conn.execute(
            "SELECT COUNT(*) FROM ab_tests WHERE status = 'running'",
        ).fetchone()[0]
        if running_count >= self._max_concurrent_tests:
            raise ValueError(f"Max concurrent tests ({self._max_concurrent_tests}) reached")

        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO ab_tests (template_name, version_a_id, version_b_id, started_at) "
            "VALUES (?, ?, ?, ?)",
            (template_name, version_a_id, version_b_id, now),
        )
        self._conn.commit()
        test_id = cursor.lastrowid
        logger.info(
            "ab_test_started",
            test_id=test_id,
            template=template_name,
            a=version_a_id,
            b=version_b_id,
        )
        return test_id

    def evaluate_test(self, test_id: int) -> str | None:
        """Evaluate an A/B test and determine the winner.

        Returns the winner version_id, or None if not enough data or no
        significant difference.
        """
        test = self._conn.execute("SELECT * FROM ab_tests WHERE id = ?", (test_id,)).fetchone()
        if test is None:
            raise ValueError(f"A/B test {test_id} not found")

        if (
            test["sessions_a"] < self.MIN_SESSIONS_PER_ARM
            or test["sessions_b"] < self.MIN_SESSIONS_PER_ARM
        ):
            return None  # Not enough data

        diff = abs(test["avg_reward_a"] - test["avg_reward_b"])
        if diff < self.SIGNIFICANCE_THRESHOLD:
            return None  # No significant difference

        winner_id = (
            test["version_a_id"]
            if test["avg_reward_a"] >= test["avg_reward_b"]
            else test["version_b_id"]
        )

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE ab_tests SET status = 'completed', ended_at = ?, winner_id = ? WHERE id = ?",
            (now, winner_id, test_id),
        )
        # Make winner the active version
        template_name = test["template_name"]
        self._conn.execute(
            "UPDATE prompt_versions SET is_active = 0 WHERE template_name = ?",
            (template_name,),
        )
        self._conn.execute(
            "UPDATE prompt_versions SET is_active = 1 WHERE id = ?",
            (winner_id,),
        )
        self._conn.commit()
        self._last_evolution_at = _time.monotonic()
        logger.info("ab_test_completed", test_id=test_id, winner=winner_id, diff=round(diff, 4))
        return winner_id

    async def maybe_evolve(self, template_name: str) -> str | None:
        """Check if evolution is possible and generate a new variant if so.

        1. Evaluate any running A/B test for this template.
        2. If we have a winner and an LLM client, generate a new variant.
        3. Start a new A/B test with the winner vs. the new variant.

        Returns:
            New version_id if a variant was created, None otherwise.
        """
        # Interval enforcement: skip if too soon since last evolution
        if self._last_evolution_at > 0:
            elapsed = _time.monotonic() - self._last_evolution_at
            if elapsed < self._evolution_interval_seconds:
                return None

        # Find running test for this template
        test = self._conn.execute(
            "SELECT * FROM ab_tests WHERE template_name = ? AND status = 'running' "
            "ORDER BY id DESC LIMIT 1",
            (template_name,),
        ).fetchone()

        if test is None:
            return None

        # Try to evaluate
        winner_id = self.evaluate_test(test["id"])
        if winner_id is None:
            return None

        # No LLM client -> can't generate new variant
        if self._llm_client is None:
            return winner_id

        # Get the winner template text
        winner_row = self._conn.execute(
            "SELECT template_text FROM prompt_versions WHERE id = ?", (winner_id,)
        ).fetchone()
        if winner_row is None:
            return winner_id

        winner_text = winner_row["template_text"]

        # Get recent session feedback for context
        sessions = self._conn.execute(
            "SELECT reward_score FROM prompt_sessions "
            "WHERE prompt_version_id = ? ORDER BY id DESC LIMIT 50",
            (winner_id,),
        ).fetchall()
        avg_reward = sum(s["reward_score"] for s in sessions) / len(sessions) if sessions else 0.0

        # Meta-prompt for variant generation
        meta_prompt = (
            "Du bist ein Prompt-Optimierer. Analysiere diesen "
            "System-Prompt und die Session-Daten.\n"
            "Erstelle eine verbesserte Version die:\n"
            "- Gleiche Struktur und Platzhalter behaelt "
            "({tools_section}, {context_section}, etc.)\n"
            "- Klarere Anweisungen fuer schwache Bereiche gibt\n"
            "- Keine neuen Platzhalter einfuehrt\n"
            "- WICHTIG: Die Tool-Section ({tools_section}) darf NICHT veraendert werden. "
            "Der Platzhalter {tools_section} muss exakt erhalten bleiben.\n"
            "Antworte NUR mit dem verbesserten Prompt, ohne Erklaerung.\n\n"
            f"Aktueller Prompt (avg reward: {avg_reward:.3f}):\n"
            f"---\n{winner_text}\n---"
        )

        try:
            new_text = await self._llm_client(meta_prompt)
            if not new_text or new_text.strip() == winner_text.strip():
                return None

            new_text = new_text.strip()

            # Post-evolution validation: reject if {tools_section} placeholder was removed
            if "{tools_section}" in winner_text and "{tools_section}" not in new_text:
                logger.warning(
                    "prompt_evolution_rejected",
                    reason="tools_section_placeholder_removed",
                    template=template_name,
                )
                return None
            new_vid = _version_id(template_name, new_text)

            # Register the new variant
            now = datetime.now(UTC).isoformat()
            self._conn.execute(
                """
                INSERT OR IGNORE INTO prompt_versions
                    (id, template_name, template_text,
                     parent_id, created_at, is_active,
                     metadata_json)
                VALUES (?, ?, ?, ?, ?, 0, '{}')
                """,
                (new_vid, template_name, new_text, winner_id, now),
            )
            self._conn.commit()

            # Start new A/B test
            self.start_ab_test(template_name, winner_id, new_vid)
            self._last_evolution_at = _time.monotonic()
            logger.info("prompt_evolved", template=template_name, new_version=new_vid)
            return new_vid

        except Exception:
            logger.debug("prompt_evolution_llm_error", exc_info=True)
            return None

    def get_stats(self, template_name: str) -> dict[str, Any]:
        """Get statistics for a template (for dashboard/monitoring)."""
        versions = self._conn.execute(
            "SELECT COUNT(*) FROM prompt_versions WHERE template_name = ?",
            (template_name,),
        ).fetchone()[0]

        active = self._conn.execute(
            "SELECT id FROM prompt_versions WHERE template_name = ? AND is_active = 1",
            (template_name,),
        ).fetchone()

        total_sessions = self._conn.execute(
            "SELECT COUNT(*) FROM prompt_sessions ps "
            "JOIN prompt_versions pv ON ps.prompt_version_id = pv.id "
            "WHERE pv.template_name = ?",
            (template_name,),
        ).fetchone()[0]

        running_tests = self._conn.execute(
            "SELECT COUNT(*) FROM ab_tests WHERE template_name = ? AND status = 'running'",
            (template_name,),
        ).fetchone()[0]

        completed_tests = self._conn.execute(
            "SELECT COUNT(*) FROM ab_tests WHERE template_name = ? AND status = 'completed'",
            (template_name,),
        ).fetchone()[0]

        return {
            "template_name": template_name,
            "version_count": versions,
            "active_version_id": active["id"] if active else None,
            "total_sessions": total_sessions,
            "running_tests": running_tests,
            "completed_tests": completed_tests,
        }

    def close(self) -> None:
        self._store.close()
