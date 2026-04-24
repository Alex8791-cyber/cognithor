"""Task 17 — class-based Crew decorators.

Concept inspired by CrewAI's @agent/@task/@crew pattern; implementation is
Apache 2.0 (no source-level borrow). Each decorator caches its result
per instance so repeated calls within the same Crew return the same
Pydantic model (important for context-graph identity).
"""

from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew import decorators as crew_dec


def test_agent_decorator_binds_kwargs():
    class Host:
        @crew_dec.agent
        def analyst(self) -> CrewAgent:
            return CrewAgent(role="analyst", goal="x")

    host = Host()
    a = host.analyst()
    assert isinstance(a, CrewAgent)
    assert a.role == "analyst"


def test_agent_decorator_caches_per_instance():
    class Host:
        @crew_dec.agent
        def analyst(self) -> CrewAgent:
            return CrewAgent(role="analyst", goal="x")

    host = Host()
    first = host.analyst()
    second = host.analyst()
    assert first is second  # same object, cached


def test_task_decorator():
    class Host:
        @crew_dec.agent
        def writer(self) -> CrewAgent:
            return CrewAgent(role="writer", goal="w")

        @crew_dec.task
        def draft(self) -> CrewTask:
            return CrewTask(description="d", expected_output="e", agent=self.writer())

    host = Host()
    t = host.draft()
    assert isinstance(t, CrewTask)


def test_crew_decorator_assembles_from_declared_agents_and_tasks():
    class PKVCrew:
        @crew_dec.agent
        def analyst(self) -> CrewAgent:
            return CrewAgent(role="analyst", goal="analyze")

        @crew_dec.task
        def research(self) -> CrewTask:
            return CrewTask(description="r", expected_output="facts", agent=self.analyst())

        @crew_dec.crew
        def assemble(self) -> Crew:
            return Crew(agents=[self.analyst()], tasks=[self.research()])

    c = PKVCrew().assemble()
    assert isinstance(c, Crew)
    assert len(c.agents) == 1
