"""
cognitio/working_memory.py

Short-term memory buffer — the "RAM" layer.

Works like the hippocampus in the human brain:
- Intra-day interactions accumulate here (fast writes, ~1ms)
- Periodically consolidated into long-term memory
- Data is never lost even if the system crashes (SQLite WAL)

Checkpoint Triggers:
    - After every 5 user messages
    - If 10 minutes have passed since the last checkpoint
    - When the user issues the /save command
    - On graceful session shutdown
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

from jarvis.security.encrypted_db import encrypted_connect

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    SQLite WAL-based short-term memory buffer.

    Parameters:
        db_path: SQLite database file path
        checkpoint_every_n: Trigger a checkpoint after this many messages
        checkpoint_interval_minutes: Checkpoint interval in minutes
    """

    def __init__(
        self,
        db_path: str = "data/working_memory.db",
        checkpoint_every_n: int = 5,
        checkpoint_interval_minutes: int = 10,
    ) -> None:
        self.db_path = db_path
        self.checkpoint_every_n = checkpoint_every_n
        self.checkpoint_interval = timedelta(minutes=checkpoint_interval_minutes)

        self._session_id = str(uuid.uuid4())
        self._message_count = 0
        self._last_checkpoint: datetime | None = None

        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._initialize_db()

        logger.info("WorkingMemory initialized.")

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection with WAL mode."""
        conn = encrypted_connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        """Create tables."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    emotional_tone REAL DEFAULT 0.0,
                    checkpointed INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS pending_memories (
                    id                  TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    summary             TEXT NOT NULL,
                    memory_type         TEXT NOT NULL,
                    emotional_intensity REAL DEFAULT 0.0,
                    emotional_valence   TEXT DEFAULT 'neutral',
                    tags                TEXT DEFAULT '[]',
                    source_type         TEXT DEFAULT 'user_stated',
                    status              TEXT DEFAULT 'pending'
                );

                CREATE INDEX IF NOT EXISTS idx_interactions_session
                    ON interactions(session_id);
                CREATE INDEX IF NOT EXISTS idx_interactions_checkpointed
                    ON interactions(checkpointed);
                CREATE INDEX IF NOT EXISTS idx_pending_status
                    ON pending_memories(status);
            """)
        logger.debug("WorkingMemory tables ready")

    def add_interaction(
        self,
        role: str,
        content: str,
        emotional_tone: float = 0.0,
    ) -> str:
        """
        Save a single message.

        Very fast — ~1ms with SQLite WAL.

        Parameters:
            role: Message owner ('user' or 'assistant')
            content: Message content
            emotional_tone: Emotional tone (-1.0 to 1.0)

        Returns:
            str: Generated interaction ID
        """
        interaction_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO interactions
                   (id, session_id, timestamp, role, content, emotional_tone)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (interaction_id, self._session_id, now, role, content, emotional_tone),
            )

        if role == "user":
            self._message_count += 1

        return interaction_id

    def get_current_session(self) -> list[dict]:
        """
        Retrieve all messages of the current session.

        Returns:
            list[dict]: Message list in chronological order
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE session_id = ?
                   ORDER BY timestamp ASC""",
                (self._session_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_recent(self, minutes: int = 30) -> list[dict]:
        """
        Retrieve messages from the last N minutes.

        Parameters:
            minutes: How many minutes to look back

        Returns:
            list[dict]: Message list in chronological order
        """
        since = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE timestamp > ?
                   ORDER BY timestamp ASC""",
                (since,),
            ).fetchall()

        return [dict(row) for row in rows]

    def should_checkpoint(self) -> bool:
        """
        Should a checkpoint be triggered?

        Conditions:
        - Enough messages have accumulated
        - Enough time has passed since the last checkpoint

        Returns:
            bool: Whether a checkpoint is needed
        """
        # Has enough messages accumulated?
        if self._message_count >= self.checkpoint_every_n:
            return True

        # Has enough time passed?
        if self._last_checkpoint is None:
            return self._message_count > 0

        elapsed = datetime.now(UTC) - self._last_checkpoint
        return elapsed >= self.checkpoint_interval and self._message_count > 0

    def checkpoint(self, llm_summarizer=None) -> list[dict]:
        """
        Consolidation: summarize pending interactions and write to
        pending_memories for transfer to long-term memory.

        Parameters:
            llm_summarizer: LLM function to use for summarization
                           (falls back to simple rule-based summarization if None)

        Returns:
            list[dict]: Generated pending memories
        """
        # Get un-checkpointed interactions
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE session_id = ? AND checkpointed = 0
                   ORDER BY timestamp ASC""",
                (self._session_id,),
            ).fetchall()

        if not rows:
            return []

        interactions = [dict(row) for row in rows]
        interaction_ids = [r["id"] for r in interactions]

        # Generate summaries
        pending_memories = self._create_pending_memories(interactions, llm_summarizer)

        # Write to pending_memories
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            for pm in pending_memories:
                conn.execute(
                    """INSERT INTO pending_memories
                       (id, session_id, created_at, summary, memory_type,
                        emotional_intensity, emotional_valence, tags, source_type, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (
                        pm["id"],
                        self._session_id,
                        now,
                        pm["summary"],
                        pm["memory_type"],
                        pm["emotional_intensity"],
                        pm["emotional_valence"],
                        json.dumps(pm["tags"]),
                        pm["source_type"],
                    ),
                )

            # Mark interactions as checkpointed
            placeholders = ",".join("?" * len(interaction_ids))
            conn.execute(
                f"UPDATE interactions SET checkpointed = 1 WHERE id IN ({placeholders})",
                interaction_ids,
            )

        # Reset counter
        self._message_count = 0
        self._last_checkpoint = datetime.now(UTC)

        logger.info(
            f"Checkpoint complete: {len(interactions)} interactions → "
            f"{len(pending_memories)} pending memories created"
        )
        return pending_memories

    def _create_pending_memories(
        self,
        interactions: list[dict],
        llm_summarizer=None,
    ) -> list[dict]:
        """
        Create pending memories from interactions.

        Falls back to rule-based summarization if no LLM is available.

        Parameters:
            interactions: Interactions to summarize
            llm_summarizer: LLM summarization function (optional)

        Returns:
            list[dict]: List of pending memories
        """
        if not interactions:
            return []

        # Concatenate conversation text
        conversation_text = "\n".join(
            [f"{i['role'].upper()}: {i['content']}" for i in interactions]
        )

        # Compute emotional tone
        tones = [i.get("emotional_tone", 0.0) for i in interactions]
        avg_tone = sum(tones) / len(tones) if tones else 0.0
        intensity = abs(avg_tone)
        valence = "positive" if avg_tone > 0.1 else ("negative" if avg_tone < -0.1 else "neutral")

        if llm_summarizer is not None:
            try:
                summary_data = llm_summarizer(conversation_text)
                return [
                    {
                        "id": str(uuid.uuid4()),
                        "summary": summary_data.get("summary", conversation_text[:500]),
                        "memory_type": summary_data.get("memory_type", "episodic"),
                        "emotional_intensity": summary_data.get("emotional_intensity", intensity),
                        "emotional_valence": summary_data.get("emotional_valence", valence),
                        "tags": summary_data.get("tags", []),
                        "source_type": "user_stated",
                    }
                ]
            except Exception as e:
                logger.warning(f"LLM summarization failed, using rule-based fallback: {e}")

        # Rule-based fallback summarization
        summary = conversation_text[:500] + ("..." if len(conversation_text) > 500 else "")
        return [
            {
                "id": str(uuid.uuid4()),
                "summary": summary,
                "memory_type": "episodic",
                "emotional_intensity": intensity,
                "emotional_valence": valence,
                "tags": ["session", "checkpoint"],
                "source_type": "user_stated",
            }
        ]

    def flush_to_long_term(self) -> list[dict]:
        """
        Return records from pending_memories for transfer to CognitioEngine.
        Mark transferred records as 'flushed'.

        Returns:
            list[dict]: List of pending memories to transfer
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM pending_memories
                   WHERE status = 'pending'
                   ORDER BY created_at ASC"""
            ).fetchall()

            pending = [dict(row) for row in rows]

            if pending:
                ids = [r["id"] for r in pending]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE pending_memories SET status = 'flushed' WHERE id IN ({placeholders})",
                    ids,
                )

        # JSON string → list conversion
        for pm in pending:
            if isinstance(pm.get("tags"), str):
                try:
                    pm["tags"] = json.loads(pm["tags"])
                except Exception:
                    pm["tags"] = []

        logger.debug(f"flush_to_long_term: transferring {len(pending)} pending memories")
        return pending

    def get_context_window(self, max_chars: int = 8000) -> str:
        """
        Build the short-term context text to pass to the LLM.

        Recent messages + pending memories summary.

        Parameters:
            max_chars: Maximum character count

        Returns:
            str: Context text
        """
        # Get recent messages (at most 20)
        recent = self.get_current_session()[-20:]

        lines = []
        for msg in recent:
            line = f"{msg['role'].upper()}: {msg['content']}"
            lines.append(line)

        context = "\n".join(lines)

        if len(context) > max_chars:
            context = context[-max_chars:]  # Keep last N characters

        return context

    def cleanup(self, older_than_days: int = 7) -> int:
        """
        Clean up old, flushed records.

        Parameters:
            older_than_days: Delete records older than this many days

        Returns:
            int: Number of records deleted
        """
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()

        with self._get_connection() as conn:
            # Delete old flushed pending memories
            result = conn.execute(
                """DELETE FROM pending_memories
                   WHERE status = 'flushed' AND created_at < ?""",
                (cutoff,),
            )
            pm_deleted = result.rowcount

            # Delete old checkpointed interactions
            result = conn.execute(
                """DELETE FROM interactions
                   WHERE checkpointed = 1 AND timestamp < ?""",
                (cutoff,),
            )
            int_deleted = result.rowcount

        total = pm_deleted + int_deleted
        logger.info(f"WorkingMemory cleanup: {total} records deleted (>{older_than_days} days)")
        return total

    def clear_session(self) -> None:
        """
        Delete all interactions and pending memories of the current session.

        Called by soft_reset(). Clears all rows in the SQLite database,
        resets counters, and assigns a new session ID.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM interactions")
            conn.execute("DELETE FROM pending_memories")

        import uuid

        self._session_id = str(uuid.uuid4())
        self._message_count = 0
        logger.info("WorkingMemory cleared, new session_id=%s", self._session_id[:8])

    def force_checkpoint_save(self, llm_summarizer=None) -> list[dict]:
        """Called when the user issues /save or on session shutdown."""
        self._message_count = self.checkpoint_every_n  # Exceed the threshold
        return self.checkpoint(llm_summarizer)

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def message_count(self) -> int:
        """Message count in this checkpoint period."""
        return self._message_count
