"""Shared fixtures for the Crew-Layer test suite.

The default ``get_default_tool_registry()`` and ``get_default_planner()``
both build heavy singletons (real ``ToolRegistryDB`` / real ``Planner``
with Ollama + router wiring) — autouse fixtures patch them so the kickoff
tests never hit the filesystem, the singletons don't leak across tests,
and we don't emit spurious ``RuntimeWarning``\\s about ``~/.cognithor/``
being missing on clean CI runners.

Tests that need a specific registry/planner mock can override the fixture
in the test file (or re-monkeypatch after the autouse fixture applies).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patched_tool_registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace `get_default_tool_registry` with a MagicMock for every crew test.

    Returns the mock so individual tests can inspect call arguments if needed.
    """
    mock_registry = MagicMock(name="MockToolRegistry")
    # Registry stubs for resolve_tools: empty tool list => any resolve_tools([])
    # trivially returns []. Still support the "all tools" sentinel for future
    # tests that exercise non-empty CrewAgent.tools lists.
    mock_registry.get_tools_for_role.return_value = []
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_tool_registry",
        lambda: mock_registry,
    )
    # Also reset the module-level singleton so no stale instance leaks between
    # test modules (e.g. if another suite imported and exercised it first).
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_registry_singleton", None)
    return mock_registry


@pytest.fixture(autouse=True)
def _clear_kickoff_cache() -> None:
    """Flush the module-level ``_KICKOFF_CACHE`` before + after every test.

    Without this, a test that wrote to the cache (e.g. idempotent-kickoff
    tests using ``_kickoff_id``) leaks state into any later test that reuses
    the same id, silently returning a stale ``CrewOutput`` from the wrong
    mock planner. Clearing both sides keeps tests hermetic regardless of
    execution order.
    """
    import cognithor.crew.crew as crew_mod

    crew_mod._KICKOFF_CACHE.clear()
    yield
    crew_mod._KICKOFF_CACHE.clear()


@pytest.fixture(autouse=True)
def _patched_default_planner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``get_default_planner`` with a MagicMock Planner.

    Crew.kickoff_async() now falls back to ``get_default_planner()`` when the
    caller doesn't pass ``planner=``; without this patch the real factory
    would try to load ``~/.cognithor/`` config, wire Ollama + router + cost
    tracker, and hang clean CI runners. Tests that want to assert on Planner
    calls should still construct their own MagicMock and either pass it via
    ``Crew(planner=...)`` or patch ``compiler.execute_task_async`` directly.
    """
    mock_planner = MagicMock(name="MockDefaultPlanner")
    # formulate_response is awaited inside execute_task_async, so default to
    # AsyncMock with a safe ResponseEnvelope-compatible return. Most existing
    # tests patch execute_task / execute_task_async wholesale so this path is
    # only exercised if a future test lets the real compiler run.
    from cognithor.core.observer import ResponseEnvelope

    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="", directive=None)
    )
    mock_planner._cost_tracker = None
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_planner",
        lambda: mock_planner,
    )
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_planner_singleton", None)
    return mock_planner
