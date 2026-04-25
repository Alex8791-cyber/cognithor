"""Team composition — verify the 4-agent Crew is built correctly."""

from __future__ import annotations

from insurance_agent_pack.crew import build_team

from cognithor.crew import Crew, CrewProcess


def test_build_team_returns_crew() -> None:
    crew = build_team(model="ollama/qwen3:8b")
    assert isinstance(crew, Crew)


def test_team_has_four_agents() -> None:
    crew = build_team(model="ollama/qwen3:8b")
    assert len(crew.agents) == 4
    roles = {a.role for a in crew.agents}
    assert roles == {
        "policy-analyst",
        "needs-assessor",
        "compliance-gatekeeper",
        "report-generator",
    }


def test_team_uses_sequential_process() -> None:
    crew = build_team(model="ollama/qwen3:8b")
    assert crew.process == CrewProcess.SEQUENTIAL


def test_team_task_count_matches_agents() -> None:
    """Each agent has at least one task; sequence is meaningful."""
    crew = build_team(model="ollama/qwen3:8b")
    assert len(crew.tasks) >= 4
    # First task assigned to needs-assessor (interview), last to report-generator
    assert crew.tasks[0].agent.role == "needs-assessor"
    assert crew.tasks[-1].agent.role == "report-generator"
