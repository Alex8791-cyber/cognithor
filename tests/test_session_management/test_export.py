"""Tests for session export."""
from __future__ import annotations

from datetime import UTC, datetime

from jarvis.gateway.session_store import SessionStore
from jarvis.models import Message, MessageRole, SessionContext


def test_export_session_json(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    s = SessionContext(
        session_id="export000000001",
        user_id="u",
        channel="webui",
        agent_name="jarvis",
    )
    store.save_session(s)
    store.save_chat_history(
        "export000000001",
        [
            Message(role=MessageRole.USER, content="Hallo", timestamp=datetime.now(tz=UTC)),
            Message(role=MessageRole.ASSISTANT, content="Hi!", timestamp=datetime.now(tz=UTC)),
        ],
    )

    export = store.export_session("export000000001")
    assert export["session_id"] == "export000000001"
    assert len(export["messages"]) == 2
    assert export["messages"][0]["role"] == "user"
    assert "exported_at" in export


def test_export_nonexistent_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    export = store.export_session("nonexistent00001")
    assert "error" in export
