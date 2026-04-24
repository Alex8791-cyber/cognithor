from unittest.mock import AsyncMock, patch

from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.output import TaskOutput


async def test_kickoff_async_returns_same_as_sync():
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    fake = TaskOutput(task_id=task.task_id, agent_role="x", raw="DONE")

    with patch("cognithor.crew.compiler.execute_task_async", new=AsyncMock(return_value=fake)):
        result = await crew.kickoff_async()

    assert result.raw == "DONE"
    assert len(result.tasks_output) == 1


async def test_async_tasks_run_concurrently_when_no_dependency():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="a", expected_output="b", agent=agent, async_execution=True)
    t2 = CrewTask(description="c", expected_output="d", agent=agent, async_execution=True)
    crew = Crew(agents=[agent], tasks=[t1, t2])

    import asyncio

    call_times: list[float] = []

    async def timed(task, context, inputs, registry, planner=None, trace_id=None):
        call_times.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.05)
        return TaskOutput(task_id=task.task_id, agent_role="x", raw="OK")

    with patch("cognithor.crew.compiler.execute_task_async", side_effect=timed):
        await crew.kickoff_async()

    # Concurrent execution: both tasks start before either finishes its 50 ms
    # sleep, so the start-time delta is pure event-loop overhead (<< 30 ms).
    # A sequential execution would require ≥50 ms between starts. Using 30 ms
    # gives headroom over Windows' ~15 ms system-timer resolution while still
    # distinguishing concurrent from sequential.
    assert abs(call_times[0] - call_times[1]) < 0.03
