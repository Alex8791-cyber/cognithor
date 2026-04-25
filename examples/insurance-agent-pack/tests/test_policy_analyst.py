"""PolicyAnalyst — declarative CrewAgent with PDF tool-use."""

from __future__ import annotations

from insurance_agent_pack.agents.policy_analyst import build_policy_analyst


def test_policy_analyst_role_label() -> None:
    a = build_policy_analyst(model="ollama/qwen3:8b")
    assert a.role == "policy-analyst"


def test_policy_analyst_has_pdf_extract_tool() -> None:
    a = build_policy_analyst(model="ollama/qwen3:8b")
    assert "pdf_extract_text" in a.tools


def test_policy_analyst_loads_prompt_text_into_backstory() -> None:
    a = build_policy_analyst(model="ollama/qwen3:8b")
    assert "Versicherung" in a.backstory or "Police" in a.backstory


def test_policy_analyst_disallows_delegation() -> None:
    """PGE-Trinity: PolicyAnalyst is an Executor, not a Planner — no delegation."""
    a = build_policy_analyst(model="ollama/qwen3:8b")
    assert a.allow_delegation is False
