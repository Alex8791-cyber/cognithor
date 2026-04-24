"""Task 15 — idempotent kickoff cache + distributed lock.

``_kickoff_id`` in inputs enables deterministic replay: calling
``kickoff_async`` twice with the same id returns the first result without
re-running any tasks. Concurrent same-id kickoffs serialize via the
distributed-lock singleton.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask


async def test_same_kickoff_id_returns_cached_output():
    """Second call with the same _kickoff_id returns the cached CrewOutput
    without re-running the Planner."""
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    call_count = {"n": 0}

    async def fake_resp(user_message, results, working_memory):
        call_count["n"] += 1
        return ResponseEnvelope(content=f"RUN-{call_count['n']}", directive=None)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake_resp)

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)
    out1 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})
    out2 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})

    assert out1.raw == out2.raw, "Same kickoff_id must return identical output"
    assert call_count["n"] == 1, "Planner must be called only once for same kickoff_id"


async def test_kickoff_id_removed_non_destructively():
    """Caller's inputs dict must not be mutated by the kickoff-id strip."""
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    inputs = {"_kickoff_id": "keep-me", "topic": "PKV"}
    await crew.kickoff_async(inputs=inputs)

    assert inputs == {"_kickoff_id": "keep-me", "topic": "PKV"}


async def test_concurrent_same_id_serializes_under_local_backend(monkeypatch):
    """Two concurrent kickoffs with same _kickoff_id must serialize.

    Regression for R3-NC2: constructing a new DistributedLock per
    kickoff_async call made the "double-check cache inside the lock" pattern
    useless under LocalLockBackend — each call saw a fresh dict.
    The singleton _get_distributed_lock() fixes it.
    """
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    in_flight = {"n": 0, "max_seen": 0}

    async def fake_resp(user_message, results, working_memory):
        in_flight["n"] += 1
        in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["n"])
        await asyncio.sleep(0.05)
        in_flight["n"] -= 1
        return ResponseEnvelope(content="RUN", directive=None)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake_resp)
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    # Reset the singleton so the test gets a fresh LocalLockBackend
    import cognithor.crew.crew as crew_mod

    monkeypatch.setattr(crew_mod, "_lock_singleton", None, raising=False)

    out1, out2 = await asyncio.gather(
        crew.kickoff_async(inputs={"_kickoff_id": "serialize-me"}),
        crew.kickoff_async(inputs={"_kickoff_id": "serialize-me"}),
    )

    assert out1.raw == out2.raw
    assert in_flight["max_seen"] == 1, (
        "Concurrent kickoffs with same id must serialize via the singleton "
        "distributed lock, not run in parallel."
    )
