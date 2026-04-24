"""Task 11 — real PGE-Trinity integration.

Verifies that ``execute_task_async`` actually routes through
``Planner.formulate_response`` (not a bypass), propagates prior task
outputs as ``ToolResult``s, and surfaces per-task token usage read from
the planner's ``CostTracker.last_call()`` accessor.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import CrewAgent, CrewTask
from cognithor.crew.compiler import execute_task_async


async def test_execute_task_routes_through_planner():
    """Spec §1.6: Crew-Layer must NOT bypass the Planner. Every task must
    end up calling ``Planner.formulate_response`` with a user message that
    reflects the task description."""
    agent = CrewAgent(role="writer", goal="write", llm="ollama/qwen3:8b")
    task = CrewTask(description="Write a haiku", expected_output="three lines", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="First line / Second line / Third line",
            directive=None,
        )
    )

    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    out = await execute_task_async(
        task,
        context=[],
        inputs=None,
        registry=mock_registry,
        planner=mock_planner,
    )
    assert out.task_id == task.task_id
    assert out.agent_role == "writer"
    assert out.raw == "First line / Second line / Third line"

    call = mock_planner.formulate_response.call_args
    args = (
        call.args
        if call.args
        else (
            call.kwargs.get("user_message"),
            call.kwargs.get("results"),
            call.kwargs.get("working_memory"),
        )
    )
    # The task description is folded into the user message.
    assert "Write a haiku" in args[0]
    # The results arg is a list (prior_results).
    assert isinstance(args[1], list)


async def test_execute_task_passes_context_as_prior_tool_results():
    """Prior task outputs must be injected as ``ToolResult``s so the Planner
    sees the research findings before drafting the downstream response."""
    agent = CrewAgent(role="writer", goal="write")
    t1 = CrewTask(description="research", expected_output="facts", agent=agent)
    t2 = CrewTask(description="write report", expected_output="text", agent=agent, context=[t1])

    from cognithor.crew.output import TaskOutput

    prior = [TaskOutput(task_id=t1.task_id, agent_role="writer", raw="FACTS_HERE")]

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="REPORT", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    await execute_task_async(
        t2,
        context=prior,
        inputs=None,
        registry=mock_registry,
        planner=mock_planner,
    )

    call = mock_planner.formulate_response.call_args
    args = (
        call.args
        if call.args
        else (
            call.kwargs["user_message"],
            call.kwargs["results"],
            call.kwargs["working_memory"],
        )
    )
    results = args[1]
    assert any("FACTS_HERE" in r.content for r in results)


async def test_execute_task_token_usage_from_cost_tracker():
    """``TaskOutput.token_usage`` must be populated from the planner's
    ``CostTracker.last_call()`` accessor so the Crew-level aggregate is
    meaningful."""
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(description="x", expected_output="y", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )
    from types import SimpleNamespace

    record = SimpleNamespace(input_tokens=42, output_tokens=7)
    tracker = MagicMock()
    tracker.last_call.return_value = record
    mock_planner._cost_tracker = tracker
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    out = await execute_task_async(
        task,
        context=[],
        inputs=None,
        registry=mock_registry,
        planner=mock_planner,
    )
    assert out.token_usage == {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
    }


def test_crew_model_copy_preserves_injected_planner():
    """Task 11 quality-review fix: ``Crew.model_copy`` must re-attach the
    ``_planner`` set by ``Crew(planner=...)``. Pydantic v2's default
    implementation only forwards declared fields, silently dropping the
    private ``_planner`` state and forcing the copy to fall back to
    :func:`get_default_planner` at kickoff time — a subtle production
    footgun.
    """
    from cognithor.crew import Crew

    live_planner = MagicMock(name="LivePlanner")
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(description="x", expected_output="y", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], planner=live_planner)

    copied = crew.model_copy(update={"verbose": True})

    assert copied.verbose is True  # update applied
    assert getattr(copied, "_planner", None) is live_planner  # planner preserved
