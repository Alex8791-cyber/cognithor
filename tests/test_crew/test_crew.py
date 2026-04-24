import pytest
from pydantic import ValidationError

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


@pytest.fixture
def agent() -> CrewAgent:
    return CrewAgent(role="writer", goal="draft")


@pytest.fixture
def task(agent: CrewAgent) -> CrewTask:
    return CrewTask(description="x", expected_output="y", agent=agent)


class TestCrewConstruction:
    def test_minimal(self, agent: CrewAgent, task: CrewTask):
        c = Crew(agents=[agent], tasks=[task])
        assert len(c.agents) == 1
        assert c.process is CrewProcess.SEQUENTIAL
        assert c.verbose is False
        assert c.planning is False
        assert c.manager_llm is None

    def test_full(self, agent: CrewAgent, task: CrewTask):
        c = Crew(
            agents=[agent],
            tasks=[task],
            process=CrewProcess.HIERARCHICAL,
            verbose=True,
            planning=True,
            manager_llm="ollama/qwen3:32b",
        )
        assert c.process is CrewProcess.HIERARCHICAL
        assert c.manager_llm == "ollama/qwen3:32b"

    def test_rejects_empty_agents(self, task: CrewTask):
        with pytest.raises(ValidationError):
            Crew(agents=[], tasks=[task])

    def test_rejects_empty_tasks(self, agent: CrewAgent):
        with pytest.raises(ValidationError):
            Crew(agents=[agent], tasks=[])

    def test_hierarchical_without_manager_llm_warns(self, agent: CrewAgent, task: CrewTask):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Crew(agents=[agent], tasks=[task], process=CrewProcess.HIERARCHICAL)
        assert any("manager_llm" in str(w.message) for w in caught)
