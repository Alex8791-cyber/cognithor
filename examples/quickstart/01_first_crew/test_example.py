"""Smoke test for 01_first_crew/main.py — no Ollama required.

Patches `get_default_planner` + `get_default_tool_registry` before importing
`main` so the example runs against a mocked PGE pipeline. Imports via
importlib.util so the test works regardless of the example's sys.path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.observer import ResponseEnvelope


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "_first_crew_main",
        Path(__file__).parent / "main.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_build_crew_and_kickoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the singletons *before* build_crew() / kickoff_async() touch them.
    mock_planner = MagicMock(name="MockPlanner")
    mock_planner._cost_tracker = None
    mock_planner.formulate_response = AsyncMock(
        side_effect=[
            ResponseEnvelope(content="- Trend A\n- Trend B\n- Trend C", directive=None),
            ResponseEnvelope(content="# Report\n\nDetailed analysis ...", directive=None),
        ]
    )
    mock_registry = MagicMock(name="MockToolRegistry")
    mock_registry.get_tools_for_role.return_value = []

    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_planner",
        lambda: mock_planner,
    )
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_tool_registry",
        lambda: mock_registry,
    )
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_planner_singleton", None)
    monkeypatch.setattr(runtime, "_registry_singleton", None)

    main_mod = _load_main()
    crew = main_mod.build_crew()
    result = await crew.kickoff_async()

    assert result.raw  # non-empty final output
    assert len(result.tasks_output) == 2
    assert "Report" in result.raw
