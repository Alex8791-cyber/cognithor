"""Shared fixtures for the Crew-Layer test suite.

The default `get_default_tool_registry()` builds (and caches) a real
`ToolRegistryDB` — an autouse fixture patches it so the kickoff tests
never hit the filesystem, the singleton doesn't leak across tests, and
we don't emit a spurious `RuntimeWarning` about `~/.cognithor/` missing
on clean CI runners.

Tests that need a specific registry mock can override the fixture in the
test file.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patched_tool_registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace `get_default_tool_registry` with a MagicMock for every crew test.

    Returns the mock so individual tests can inspect call arguments if needed.
    """
    mock_registry = MagicMock(name="MockToolRegistry")
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_tool_registry",
        lambda: mock_registry,
    )
    # Also reset the module-level singleton so no stale instance leaks between
    # test modules (e.g. if another suite imported and exercised it first).
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_registry_singleton", None)
    return mock_registry
