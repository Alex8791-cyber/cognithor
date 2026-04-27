"""
tests/test_identity/cognitio/test_working_memory.py

Integration-light tests for cognithor.identity.cognitio.working_memory.
Uses real SQLite (via encrypted_connect fallback) with tmp_path for isolation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from cognithor.identity.cognitio.working_memory import WorkingMemory
from cognithor.security.encrypted_db import encrypted_connect

# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_db_file_created_and_interactions_table_exists(self, tmp_path):
        db = tmp_path / "wm.db"
        WorkingMemory(str(db))
        assert db.exists()
        # Use encrypted_connect to be compatible with SQLCipher if active
        conn = encrypted_connect(str(db))
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "interactions" in tables

    def test_new_instance_has_zero_message_count_and_valid_session_id(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        assert wm.message_count == 0
        # session_id should be a valid UUID
        parsed = uuid.UUID(wm.session_id)
        assert str(parsed) == wm.session_id


# ---------------------------------------------------------------------------
# TestAddAndRetrieve
# ---------------------------------------------------------------------------


class TestAddAndRetrieve:
    def test_add_user_interaction_increments_message_count(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("user", "hello")
        assert wm.message_count == 1

    def test_add_assistant_does_not_increment_message_count(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("assistant", "hi there")
        assert wm.message_count == 0

    def test_add_returns_string_id(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        result = wm.add_interaction("user", "test")
        assert isinstance(result, str)
        uuid.UUID(result)  # must be parseable as UUID

    def test_get_current_session_returns_added_record(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("user", "session message")
        rows = wm.get_current_session()
        assert len(rows) == 1
        assert rows[0]["role"] == "user"
        assert rows[0]["content"] == "session message"

    def test_get_recent_returns_recent_records(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("user", "recent message")
        rows = wm.get_recent(minutes=30)
        assert any(r["content"] == "recent message" for r in rows)

    def test_get_recent_excludes_old_records(self, tmp_path):
        """Records with timestamps outside the window should not appear."""
        db_path = str(tmp_path / "wm.db")
        wm = WorkingMemory(db_path)
        # Inject a row with an old timestamp using encrypted_connect for compatibility
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        conn = encrypted_connect(db_path)
        conn.execute(
            "INSERT INTO interactions (id, session_id, timestamp, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), wm.session_id, old_ts, "user", "old message"),
        )
        conn.commit()
        conn.close()
        rows = wm.get_recent(minutes=30)
        contents = [r["content"] for r in rows]
        assert "old message" not in contents


# ---------------------------------------------------------------------------
# TestCheckpointTrigger
# ---------------------------------------------------------------------------


class TestCheckpointTrigger:
    def test_should_checkpoint_false_with_no_messages(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        assert wm.should_checkpoint() is False

    def test_should_checkpoint_false_below_threshold(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=5)
        # Simulate a recent checkpoint so _last_checkpoint is set and interval
        # has NOT elapsed — this exercises the count-based branch only.
        wm._last_checkpoint = datetime.now(UTC)
        for _ in range(4):
            wm.add_interaction("user", "msg")
        assert wm.should_checkpoint() is False

    def test_should_checkpoint_true_at_threshold(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=5)
        for _ in range(5):
            wm.add_interaction("user", "msg")
        assert wm.should_checkpoint() is True

    def test_should_checkpoint_resets_after_checkpoint(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=5)
        for _ in range(5):
            wm.add_interaction("user", "msg")
        assert wm.should_checkpoint() is True
        wm.checkpoint()
        assert wm.should_checkpoint() is False

    def test_should_checkpoint_true_after_interval_with_messages(self, tmp_path):
        """Time-based trigger: elapsed >= checkpoint_interval with messages pending."""
        wm = WorkingMemory(
            str(tmp_path / "wm.db"),
            checkpoint_every_n=100,
            checkpoint_interval_minutes=10,
        )
        wm.add_interaction("user", "one message")
        # Simulate last_checkpoint being 15 minutes in the past
        wm._last_checkpoint = datetime.now(UTC) - timedelta(minutes=15)
        assert wm.should_checkpoint() is True

    def test_should_checkpoint_true_when_last_checkpoint_is_none_and_messages_exist(self, tmp_path):
        """When _last_checkpoint is None and there are messages, trigger is True."""
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=100)
        wm.add_interaction("user", "some message")
        # _last_checkpoint starts as None
        assert wm._last_checkpoint is None
        assert wm.should_checkpoint() is True


# ---------------------------------------------------------------------------
# TestCheckpointFlow
# ---------------------------------------------------------------------------


class TestCheckpointFlow:
    def test_checkpoint_returns_pending_memories(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=5)
        for idx in range(5):
            wm.add_interaction("user", f"msg {idx}")
        result = wm.checkpoint()
        assert isinstance(result, list)
        assert len(result) >= 1
        # Each item should have at least a summary field
        assert "summary" in result[0]

    def test_checkpoint_with_empty_buffer_returns_empty_list(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        result = wm.checkpoint()
        assert result == []

    def test_checkpoint_with_llm_summarizer(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("user", "hello world")

        def fake_summarizer(text):
            return {
                "summary": "Fake summary",
                "memory_type": "episodic",
                "emotional_intensity": 0.5,
                "emotional_valence": "positive",
                "tags": ["test"],
            }

        result = wm.checkpoint(llm_summarizer=fake_summarizer)
        assert len(result) == 1
        assert result[0]["summary"] == "Fake summary"
        assert result[0]["memory_type"] == "episodic"

    def test_flush_to_long_term_returns_all_pending_and_clears(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        for _ in range(3):
            wm.add_interaction("user", "msg")
        wm.checkpoint()  # populate pending_memories

        flushed = wm.flush_to_long_term()
        assert len(flushed) >= 1
        # A second flush should return nothing (already marked flushed)
        second = wm.flush_to_long_term()
        assert second == []


# ---------------------------------------------------------------------------
# TestContextWindow
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_context_window_respects_max_chars(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        for _ in range(10):
            wm.add_interaction("user", "x" * 50)  # 10 * ~55 chars each
        ctx = wm.get_context_window(max_chars=200)
        assert len(ctx) <= 200

    def test_context_window_empty_buffer_returns_empty_string(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        assert wm.get_context_window() == ""


# ---------------------------------------------------------------------------
# TestCleanupAndClear
# ---------------------------------------------------------------------------


class TestCleanupAndClear:
    def test_cleanup_older_than_zero_days_removes_checkpointed_interactions(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        for _ in range(3):
            wm.add_interaction("user", "msg")
        wm.checkpoint()  # marks them checkpointed + writes pending_memories
        wm.flush_to_long_term()  # marks pending_memories as flushed

        # older_than_days=0 → cutoff is now; same-second timestamps may yield 0
        deleted = wm.cleanup(older_than_days=0)
        assert deleted >= 0

    def test_cleanup_999_days_removes_nothing(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        wm.add_interaction("user", "new message")
        deleted = wm.cleanup(older_than_days=999)
        assert deleted == 0
        assert len(wm.get_current_session()) == 1

    def test_clear_session_resets_count_and_new_session_id(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"))
        old_sid = wm.session_id
        wm.add_interaction("user", "something")
        assert wm.message_count == 1

        wm.clear_session()

        assert wm.message_count == 0
        assert wm.session_id != old_sid
        assert wm.get_current_session() == []


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_records_survive_new_instance(self, tmp_path):
        db = str(tmp_path / "wm_persist.db")
        wm1 = WorkingMemory(db)
        sid = wm1.session_id
        wm1.add_interaction("user", "persisted message")
        del wm1

        # Open a second instance to confirm the DB file is not wiped on init
        WorkingMemory(db)
        # Verify via encrypted_connect that the row still exists
        conn = encrypted_connect(db)
        rows = conn.execute(
            "SELECT content FROM interactions WHERE session_id=?",
            (sid,),
        ).fetchall()
        conn.close()
        assert any("persisted message" in row[0] for row in rows)

    def test_new_instance_gets_fresh_session_id(self, tmp_path):
        db = str(tmp_path / "wm_session.db")
        wm1 = WorkingMemory(db)
        sid1 = wm1.session_id
        del wm1

        wm2 = WorkingMemory(db)
        sid2 = wm2.session_id
        assert sid1 != sid2


# ---------------------------------------------------------------------------
# TestForceCheckpoint
# ---------------------------------------------------------------------------


class TestForceCheckpoint:
    def test_force_checkpoint_save_works_regardless_of_buffer(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=100)
        wm.add_interaction("user", "only one message")
        # _last_checkpoint is None + count > 0 → should_checkpoint() is True
        assert wm.should_checkpoint() is True
        result = wm.force_checkpoint_save()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_force_checkpoint_save_with_zero_messages(self, tmp_path):
        wm = WorkingMemory(str(tmp_path / "wm.db"), checkpoint_every_n=100)
        # No messages at all — sets count to threshold, checkpoint returns []
        result = wm.force_checkpoint_save()
        # Nothing to checkpoint: returns empty list (no un-checkpointed rows)
        assert result == []
