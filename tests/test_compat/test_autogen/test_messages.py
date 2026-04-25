"""Message classes — match AutoGen field surface."""

from __future__ import annotations

from cognithor.compat.autogen.messages import (
    HandoffMessage,
    StructuredMessage,
    TextMessage,
    ToolCallSummaryMessage,
)


def test_text_message_required_fields() -> None:
    m = TextMessage(content="hello", source="agent-1")
    assert m.content == "hello"
    assert m.source == "agent-1"
    assert m.type == "TextMessage"
    assert m.metadata == {}


def test_text_message_models_usage_optional() -> None:
    m = TextMessage(content="x", source="a", models_usage={"total_tokens": 5})
    assert m.models_usage == {"total_tokens": 5}


def test_tool_call_summary_message_records_tool_name() -> None:
    m = ToolCallSummaryMessage(content="result-text", source="agent-1", tool_name="search")
    assert m.tool_name == "search"
    assert m.type == "ToolCallSummaryMessage"


def test_handoff_message_target() -> None:
    m = HandoffMessage(content="passing the ball", source="agent-1", target="agent-2")
    assert m.target == "agent-2"
    assert m.type == "HandoffMessage"


def test_structured_message_holds_arbitrary_payload() -> None:
    payload = {"key": "value"}
    m = StructuredMessage(content=payload, source="agent-1")
    assert m.content == payload
    assert m.type == "StructuredMessage"


def test_text_message_str_uses_content() -> None:
    m = TextMessage(content="hello world", source="a")
    assert str(m) == "hello world"
