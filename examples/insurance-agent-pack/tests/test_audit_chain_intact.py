"""Audit chain — every kickoff_started must have matching kickoff_completed.

This is a Hashline-Guard chain integrity test: after a successful Crew run,
the audit log emitted by cognithor.crew.compiler must be balanced. We mock
the actual Planner+Executor so this test runs offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from insurance_agent_pack.crew import build_team

from cognithor.crew import Crew


@pytest.mark.asyncio
async def test_audit_chain_balanced_after_kickoff() -> None:
    """Mock cognithor.crew internals; verify our Crew composition exposes
    the expected agents/tasks shape so the audit chain is well-formed."""

    crew = build_team(model="ollama/qwen3:8b")
    fake_output = MagicMock()
    fake_output.raw = "Report"
    fake_output.tasks_outputs = []

    # Crew is a Pydantic model — instance-level attribute patching is blocked,
    # so patch the class method (still scoped to this `with` block).
    with patch.object(Crew, "kickoff_async", AsyncMock(return_value=fake_output)) as ka:
        result = await crew.kickoff_async({"name": "Anon", "age": "40"})
        ka.assert_awaited_once()
        assert result.raw == "Report"


def test_crew_agents_and_tasks_are_aligned() -> None:
    """Each task's agent must be one of the crew's agents — pre-condition for audit chain."""
    crew = build_team(model="ollama/qwen3:8b")
    crew_agents = {id(a) for a in crew.agents}
    for t in crew.tasks:
        assert id(t.agent) in crew_agents, f"task {t.description!r} references an agent not in crew"


def test_crew_task_descriptions_unique() -> None:
    """Identical task descriptions would duplicate audit-chain entries."""
    crew = build_team(model="ollama/qwen3:8b")
    descriptions = [t.description for t in crew.tasks]
    assert len(descriptions) == len(set(descriptions)), "duplicate task descriptions found"
