# tests/test_compat/test_autogen/conftest.py
"""Test fixtures for cognithor.compat.autogen shim tests.

Provides:
- `requires_autogen` marker — skip if autogen-agentchat is not installed.
- `mock_model_client` fixture — deterministic model output.
"""

from __future__ import annotations

from typing import Any

import pytest


def pytest_collection_modifyitems(config, items) -> None:
    """Skip @requires_autogen tests when autogen-agentchat is not installed."""
    try:
        import autogen_agentchat  # noqa: F401

        autogen_available = True
    except ImportError:
        autogen_available = False

    skip_marker = pytest.mark.skip(
        reason="autogen-agentchat not installed — install with `pip install cognithor[autogen]`",
    )
    for item in items:
        if "requires_autogen" in item.keywords and not autogen_available:
            item.add_marker(skip_marker)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_autogen: mark test as requiring `pip install cognithor[autogen]`",
    )


class _MockModelClient:
    """Deterministic mock — returns whatever was set via `set_response`."""

    def __init__(self, response: str = "") -> None:
        self._response = response

    def set_response(self, response: str) -> None:
        self._response = response

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        class _R:
            content = self._response
            usage = {"prompt_tokens": 0, "completion_tokens": 0}

        return _R()


@pytest.fixture
def mock_model_client() -> _MockModelClient:
    return _MockModelClient(response="OK")
