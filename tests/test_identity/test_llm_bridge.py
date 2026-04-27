"""Tests for CognithorLLMBridge (llm_bridge.py) — Session 2."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.identity.llm_bridge import CognithorLLMBridge

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_bridge():
    """CognithorLLMBridge backed by a MagicMock UnifiedLLMClient.

    loop=None forces _run_async to use asyncio.run (no running loop in tests).
    """
    client = MagicMock()
    client.chat = AsyncMock(return_value={"content": "hello"})
    bridge = CognithorLLMBridge(client, model="test-model", loop=None)
    return bridge


# ---------------------------------------------------------------------------
# TestParseJsonSafeAdditional
# ---------------------------------------------------------------------------


class TestParseJsonSafeAdditional:
    """Additional _parse_json_safe paths not covered in Session 1."""

    def test_embedded_json_in_prose(self):
        """JSON embedded in surrounding prose is found via the find('{') path."""
        result = CognithorLLMBridge._parse_json_safe('prose before {"k": 1} prose after')
        assert result == {"k": 1}

    def test_missing_expected_keys_filled_as_none(self):
        """When expected_keys contains a key absent from valid JSON, it becomes None."""
        result = CognithorLLMBridge._parse_json_safe('{"a": 1}', expected_keys=["a", "b"])
        assert result["a"] == 1
        assert result["b"] is None

    def test_large_text_skips_embedded_block(self):
        """Texts longer than 8192 chars skip the embedded-block { search path."""
        long_prefix = "x" * 8193
        text = long_prefix + '{"k": 99}'
        result = CognithorLLMBridge._parse_json_safe(text, expected_keys=["k"])
        # The find("{") path is guarded by len(text) <= 8192; it is skipped.
        # Direct parse also fails (whole blob is not valid JSON), so result is defaults.
        assert result == {"k": None}


# ---------------------------------------------------------------------------
# TestRunAsyncFallback
# ---------------------------------------------------------------------------


class TestRunAsyncFallback:
    """_run_async fallback to asyncio.run when no running loop exists."""

    def test_falls_back_to_asyncio_run(self, mock_bridge):
        """loop=None + no running loop → asyncio.run() path returns value."""

        async def simple():
            return 42

        result = mock_bridge._run_async(simple())
        assert result == 42


# ---------------------------------------------------------------------------
# TestComplete
# ---------------------------------------------------------------------------


class TestComplete:
    """complete() extracts content from various LLM response shapes."""

    def test_response_dict_with_content_key(self, mock_bridge):
        mock_bridge._llm.chat = AsyncMock(return_value={"content": "hello"})
        result = mock_bridge.complete("hi")
        assert result == "hello"

    def test_response_nested_message_content(self, mock_bridge):
        mock_bridge._llm.chat = AsyncMock(return_value={"message": {"content": "nested"}})
        result = mock_bridge.complete("hi")
        assert result == "nested"

    def test_response_plain_string(self, mock_bridge):
        mock_bridge._llm.chat = AsyncMock(return_value="plain")
        result = mock_bridge.complete("hi")
        assert result == "plain"

    def test_system_prompt_prepended(self, mock_bridge):
        """system_prompt must appear as first message in the call."""
        mock_bridge._llm.chat = AsyncMock(return_value={"content": "ok"})
        mock_bridge.complete("prompt text", system_prompt="be helpful")
        call_args = mock_bridge._llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "be helpful"


# ---------------------------------------------------------------------------
# TestChat
# ---------------------------------------------------------------------------


class TestChat:
    """chat() forwards messages and prepends system message when given."""

    def test_forwards_messages(self, mock_bridge):
        msgs = [{"role": "user", "content": "hi"}]
        mock_bridge._llm.chat = AsyncMock(return_value={"content": "pong"})
        result = mock_bridge.chat(msgs)
        # result must be a string (the extracted content)
        assert result == "pong"

    def test_system_prompt_prepends(self, mock_bridge):
        msgs = [{"role": "user", "content": "hi"}]
        mock_bridge._llm.chat = AsyncMock(return_value={"content": "ok"})
        mock_bridge.chat(msgs, system_prompt="sys")
        call_args = mock_bridge._llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# TestCompleteJson
# ---------------------------------------------------------------------------


class TestCompleteJson:
    """complete_json() happy path and exception path."""

    def test_happy_path_parses_json(self, mock_bridge, monkeypatch):
        monkeypatch.setattr(mock_bridge, "complete", lambda *a, **kw: '{"a": 1}')
        result = mock_bridge.complete_json("prompt", expected_keys=["a"])
        assert result == {"a": 1}

    def test_exception_returns_defaults(self, mock_bridge, monkeypatch):
        def _raise(*a, **kw):
            raise RuntimeError("llm down")

        monkeypatch.setattr(mock_bridge, "complete", _raise)
        result = mock_bridge.complete_json("prompt", expected_keys=["a"])
        assert result == {"a": None}


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """health_check() delegates to complete()."""

    def test_returns_true_when_complete_returns_string(self, mock_bridge, monkeypatch):
        monkeypatch.setattr(mock_bridge, "complete", lambda *a, **kw: "ok")
        assert mock_bridge.health_check() is True

    def test_returns_false_when_complete_raises(self, mock_bridge, monkeypatch):
        def _raise(*a, **kw):
            raise ConnectionError("down")

        monkeypatch.setattr(mock_bridge, "complete", _raise)
        assert mock_bridge.health_check() is False
