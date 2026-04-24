"""Task 14 - Hashline-Guard audit chain integration.

Every kickoff must emit lifecycle events (kickoff_started, task_started,
task_completed, kickoff_completed) with the kickoff's trace_id as session_id.
"""

from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask


async def test_kickoff_emits_audit_event_with_trace_id(monkeypatch):
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    events: list = []

    def spy(event_name, **fields):
        events.append((event_name, fields))

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="OK", directive=None),
    )

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    monkeypatch.setattr("cognithor.crew.compiler.append_audit", spy)
    result = await crew.kickoff_async()

    # At least one crew_* audit event was emitted
    crew_events = [e for e in events if "crew" in e[0]]
    assert crew_events
    # And at least one carries our trace_id
    assert any(fields.get("trace_id") == result.trace_id for _name, fields in crew_events)


async def test_kickoff_emits_lifecycle_sequence(monkeypatch):
    """All four lifecycle events fire in order for a 1-task crew."""
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    events: list = []

    def spy(event_name, **fields):
        events.append(event_name)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="OK", directive=None),
    )
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    monkeypatch.setattr("cognithor.crew.compiler.append_audit", spy)
    await crew.kickoff_async()

    assert events == [
        "crew_kickoff_started",
        "crew_task_started",
        "crew_task_completed",
        "crew_kickoff_completed",
    ]


def test_audit_events_are_pii_scrubbed():
    """R4-I8: audit fields containing PII must be redacted before persisting."""
    from cognithor.crew.compiler import _scrub_audit_fields

    cleaned = _scrub_audit_fields(
        {
            "task_id": "t1",
            "feedback": "Email user at test@example.com after the call",
            "duration_ms": 123.4,
        }
    )
    assert "test@example.com" not in cleaned["feedback"]
    assert "[REDACTED:email]" in cleaned["feedback"]
    assert cleaned["task_id"] == "t1"  # non-PII strings pass through
    assert cleaned["duration_ms"] == 123.4  # non-string values pass through
