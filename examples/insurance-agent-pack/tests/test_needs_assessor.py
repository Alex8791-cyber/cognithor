"""NeedsAssessor — turns interview answers into structured need profile."""

from __future__ import annotations

from insurance_agent_pack.agents.needs_assessor import build_needs_assessor


def test_needs_assessor_role_label() -> None:
    a = build_needs_assessor(model="ollama/qwen3:8b")
    assert a.role == "needs-assessor"


def test_needs_assessor_uses_no_tools() -> None:
    """Assessor is pure conversational reasoning; no external tool calls."""
    a = build_needs_assessor(model="ollama/qwen3:8b")
    assert a.tools == []


def test_needs_assessor_memory_enabled() -> None:
    """Memory must be on so the Assessor can refer back to earlier answers."""
    a = build_needs_assessor(model="ollama/qwen3:8b")
    assert a.memory is True
