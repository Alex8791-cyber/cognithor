from unittest.mock import AsyncMock, patch

import pytest

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.crew.output import TaskOutput


@pytest.fixture
def researcher() -> CrewAgent:
    return CrewAgent(role="researcher", goal="research", llm="ollama/qwen3:8b")


@pytest.fixture
def writer() -> CrewAgent:
    return CrewAgent(role="writer", goal="write", llm="ollama/qwen3:8b")


class TestSequentialKickoff:
    def test_two_tasks_run_in_order(self, researcher: CrewAgent, writer: CrewAgent):
        t1 = CrewTask(description="research topic", expected_output="facts", agent=researcher)
        t2 = CrewTask(
            description="write report", expected_output="report", agent=writer, context=[t1]
        )
        crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process=CrewProcess.SEQUENTIAL)

        fake_outputs = [
            TaskOutput(task_id=t1.task_id, agent_role="researcher", raw="FACTS ABOUT TOPIC"),
            TaskOutput(task_id=t2.task_id, agent_role="writer", raw="REPORT DRAFT"),
        ]

        # Task 11: Crew.kickoff now trampolines through asyncio.run(kickoff_async),
        # so the actual per-task call site is execute_task_async. Patch that path
        # rather than the sync wrapper (which is no longer reached via kickoff).
        with patch(
            "cognithor.crew.compiler.execute_task_async",
            new=AsyncMock(side_effect=fake_outputs),
        ) as mocked:
            result = crew.kickoff()

        assert result.raw == "REPORT DRAFT"
        assert len(result.tasks_output) == 2
        assert result.trace_id
        # Sequential ordering: first call is t1, second is t2
        assert mocked.call_args_list[0].args[0].task_id == t1.task_id
        assert mocked.call_args_list[1].args[0].task_id == t2.task_id

    def test_inputs_threaded_into_first_task(self, researcher: CrewAgent):
        t1 = CrewTask(description="research {topic}", expected_output="facts", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[t1])

        captured: list = []

        async def spy(task, *, context, inputs, registry, planner=None, trace_id=None):
            captured.append(inputs)
            return TaskOutput(task_id=task.task_id, agent_role=task.agent.role, raw="OK")

        with patch("cognithor.crew.compiler.execute_task_async", side_effect=spy):
            crew.kickoff(inputs={"topic": "PKV tariffs"})

        assert captured[0] == {"topic": "PKV tariffs"}
