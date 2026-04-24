"""Smoke test for 05_pkv_report/main.py — spec §1.4 end-to-end.

Mirrors the acceptance test at tests/test_crew/test_pkv_example.py but loads
the example's own `main.py` via importlib so the file in the example folder
stays the canonical copy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "_pkv_report_main",
        Path(__file__).parent / "main.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_pkv_report_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch singletons BEFORE main is imported / build_crew runs.
    mock_registry = MagicMock(name="MockToolRegistry")
    mock_registry.get_tools_for_role.return_value = []
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_tool_registry",
        lambda: mock_registry,
    )
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_registry_singleton", None)

    main_mod = _load_main()

    tracker = MagicMock()
    tracker.last_call = MagicMock(
        side_effect=[
            SimpleNamespace(input_tokens=500, output_tokens=100),
            SimpleNamespace(input_tokens=800, output_tokens=600),
        ]
    )

    mock_planner = MagicMock()
    mock_planner._cost_tracker = tracker
    mock_planner.formulate_response = AsyncMock(
        side_effect=[
            ResponseEnvelope(
                content=(
                    "| Tarif | Beitrag | Leistungen |\n|---|---|---|\n| A | 450€ | Stationär |"
                ),
                directive=None,
            ),
            ResponseEnvelope(
                content="# PKV-Empfehlung\nBasierend auf der Analyse empfehlen wir...",
                directive=None,
            ),
        ]
    )

    # Rebuild the crew with our mock planner — build_crew() uses the default.
    base_crew = main_mod.build_crew()
    crew = Crew(
        agents=list(base_crew.agents),
        tasks=list(base_crew.tasks),
        process=base_crew.process,
        verbose=True,
        planner=mock_planner,
    )

    result = await crew.kickoff_async()

    assert "PKV-Empfehlung" in result.raw
    assert len(result.tasks_output) == 2
    assert result.trace_id
    # Aggregate token usage: sum of per-task totals. Task 1: 600, Task 2: 1400.
    assert result.token_usage["total_tokens"] == 2000
