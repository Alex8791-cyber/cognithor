import pytest
from pydantic import ValidationError

from cognithor.crew.agent import CrewAgent


class TestCrewAgent:
    def test_minimal_construction(self):
        a = CrewAgent(role="writer", goal="produce drafts")
        assert a.role == "writer"
        assert a.goal == "produce drafts"
        assert a.backstory == ""
        assert a.tools == []
        assert a.llm is None
        assert a.allow_delegation is False
        assert a.max_iter == 20
        assert a.memory is True
        assert a.verbose is False

    def test_full_construction(self):
        a = CrewAgent(
            role="analyst",
            goal="analyze tarifs",
            backstory="veteran broker",
            tools=["web_search", "pdf_reader"],
            llm="ollama/qwen3:32b",
            allow_delegation=True,
            max_iter=5,
            memory=False,
            verbose=True,
        )
        assert a.tools == ["web_search", "pdf_reader"]
        assert a.llm == "ollama/qwen3:32b"
        assert a.max_iter == 5

    def test_role_and_goal_required(self):
        with pytest.raises(ValidationError):
            CrewAgent(goal="x")  # role missing
        with pytest.raises(ValidationError):
            CrewAgent(role="x")  # goal missing

    def test_max_iter_positive(self):
        with pytest.raises(ValidationError):
            CrewAgent(role="x", goal="y", max_iter=0)

    def test_tools_must_be_strings(self):
        with pytest.raises(ValidationError):
            CrewAgent(role="x", goal="y", tools=[123])  # type: ignore[list-item]

    def test_frozen(self):
        a = CrewAgent(role="x", goal="y")
        with pytest.raises(ValidationError):
            a.role = "z"  # type: ignore[misc]

    def test_metadata_preserves_arbitrary_keys(self):
        a = CrewAgent(
            role="x",
            goal="y",
            metadata={"pack_id": "cognithor.packs.research", "owner": "alice"},
        )
        assert a.metadata == {"pack_id": "cognithor.packs.research", "owner": "alice"}
