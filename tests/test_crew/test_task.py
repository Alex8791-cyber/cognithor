import pytest
from pydantic import BaseModel, ValidationError

from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


@pytest.fixture
def agent() -> CrewAgent:
    return CrewAgent(role="writer", goal="draft")


class TestCrewTask:
    def test_minimal(self, agent: CrewAgent):
        t = CrewTask(description="Write something", expected_output="A sentence.", agent=agent)
        assert t.description == "Write something"
        assert t.agent.role == "writer"
        assert t.context == []
        assert t.tools == []
        assert t.guardrail is None
        assert t.async_execution is False

    def test_context_accepts_other_tasks(self, agent: CrewAgent):
        t1 = CrewTask(description="research", expected_output="facts", agent=agent)
        t2 = CrewTask(description="write", expected_output="text", agent=agent, context=[t1])
        assert len(t2.context) == 1
        # Value equality — Pydantic v2 doesn't guarantee object identity for
        # nested models in list fields; use __eq__ (frozen → hashable) not `is`.
        assert t2.context[0] == t1
        assert t2.context[0].task_id == t1.task_id

    def test_max_retries_bounds(self, agent: CrewAgent):
        with pytest.raises(ValidationError):
            CrewTask(description="x", expected_output="y", agent=agent, max_retries=-1)
        with pytest.raises(ValidationError):
            CrewTask(description="x", expected_output="y", agent=agent, max_retries=11)

    def test_guardrail_callable_accepted(self, agent: CrewAgent):
        t = CrewTask(
            description="x",
            expected_output="y",
            agent=agent,
            guardrail=lambda out: (True, out),
        )
        assert t.guardrail is not None

    def test_guardrail_string_accepted(self, agent: CrewAgent):
        t = CrewTask(
            description="x",
            expected_output="y",
            agent=agent,
            guardrail="Output must be one sentence",
        )
        assert isinstance(t.guardrail, str)

    def test_output_json_must_be_pydantic_model(self, agent: CrewAgent):
        class Schema(BaseModel):
            name: str

        t = CrewTask(description="x", expected_output="y", agent=agent, output_json=Schema)
        assert t.output_json is Schema

    def test_description_required(self, agent: CrewAgent):
        with pytest.raises(ValidationError):
            CrewTask(expected_output="y", agent=agent)  # type: ignore[call-arg]

    def test_frozen(self, agent: CrewAgent):
        t = CrewTask(description="x", expected_output="y", agent=agent)
        with pytest.raises(ValidationError):
            t.description = "mutated"  # type: ignore[misc]
