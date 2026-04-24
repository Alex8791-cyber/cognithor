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


async def test_task_failure_emits_failed_events(monkeypatch):
    """Quality-review fix: a task failure must emit crew_task_failed +
    crew_kickoff_failed so every started event has a matching terminal event
    (no orphaned open entries in the Hashline-Guard chain).
    """
    import pytest

    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    events: list = []
    monkeypatch.setattr(
        "cognithor.crew.compiler.append_audit",
        lambda name, **fields: events.append(name),
    )

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        side_effect=RuntimeError("planner blew up"),
    )
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.raises(RuntimeError, match="planner blew up"):
        await crew.kickoff_async()

    # Every started event has a matching terminal event.
    assert "crew_task_started" in events
    assert "crew_task_failed" in events
    assert "crew_kickoff_failed" in events
    # crew_task_completed / crew_kickoff_completed must NOT appear on failure.
    assert "crew_task_completed" not in events
    assert "crew_kickoff_completed" not in events


def test_get_audit_trail_caches_failed_init(monkeypatch):
    """Quality-review fix: a failed AuditTrail init must be cached so
    subsequent calls don't re-run load_config() on every audit event.
    """
    import cognithor.crew.compiler as compiler_mod

    # Reset the module-level sentinel so the test starts clean.
    monkeypatch.setattr(
        compiler_mod,
        "_audit_trail",
        compiler_mod._AUDIT_TRAIL_UNINITIALIZED,
    )

    call_count = {"load_config": 0}

    def fake_load_config():
        call_count["load_config"] += 1
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("cognithor.config.load_config", fake_load_config)

    # First call triggers the init attempt.
    result_1 = compiler_mod._get_audit_trail()
    # Second and third calls MUST short-circuit on the cached None.
    result_2 = compiler_mod._get_audit_trail()
    result_3 = compiler_mod._get_audit_trail()

    assert result_1 is None
    assert result_2 is None
    assert result_3 is None
    assert call_count["load_config"] == 1  # NOT 3
