"""Task 13 - context=[t1] on t2 must flow t1's output into t2's Planner call.

Task 11's ``execute_task_async`` synthesizes prior TaskOutputs as
``ToolResult`` entries in the ``results`` list passed to
``Planner.formulate_response``. This test locks that behaviour in: if a
future change stops plumbing prior outputs, this test fails.
"""

from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask


async def test_task2_receives_task1_output():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="phase 1", expected_output="res1", agent=agent)
    t2 = CrewTask(description="phase 2", expected_output="res2", agent=agent, context=[t1])

    captured_results: list = []
    captured_user_msgs: list = []

    async def capture(user_message, results, working_memory):
        captured_user_msgs.append(user_message)
        captured_results.append(list(results))
        n = len(captured_user_msgs)
        return ResponseEnvelope(
            content="PHASE1_RESULT" if n == 1 else "PHASE2_RESULT",
            directive=None,
        )

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=capture)

    crew = Crew(agents=[agent], tasks=[t1, t2], planner=mock_planner)
    result = await crew.kickoff_async()

    assert result.tasks_output[1].raw == "PHASE2_RESULT"
    # The 2nd call (for t2) must carry t1's output in `results`
    t2_results = captured_results[1]
    assert any("PHASE1_RESULT" in r.content for r in t2_results)
