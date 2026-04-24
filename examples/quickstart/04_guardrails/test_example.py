"""Smoke test for 04_guardrails/main.py.

Exercises both retry-success and retry-exhausted paths with a mocked Planner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew
from cognithor.crew.errors import GuardrailFailure


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "_guardrails_main",
        Path(__file__).parent / "main.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_registry = MagicMock(name="MockToolRegistry")
    mock_registry.get_tools_for_role.return_value = []
    monkeypatch.setattr(
        "cognithor.crew.runtime.get_default_tool_registry",
        lambda: mock_registry,
    )
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_registry_singleton", None)


@pytest.mark.asyncio
async def test_guardrail_passes_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_registry(monkeypatch)
    main_mod = _load_main()

    # First response violates word_count(max_words=10), second fits.
    outputs = iter(
        [
            # 11 words — blows the max_words=10 guard
            ResponseEnvelope(
                content=(
                    "Cognithor ist ein lokal-first Agent OS "
                    "mit PGE Trinity Architektur für Automation"
                ),
                directive=None,
            ),
            # 8 words — passes
            ResponseEnvelope(
                content="Cognithor ist ein lokales Agenten-Betriebssystem.",
                directive=None,
            ),
        ]
    )

    async def fake_formulate(user_message, results, working_memory):
        return next(outputs)

    mock_planner = MagicMock(name="MockPlanner")
    mock_planner._cost_tracker = None
    mock_planner.formulate_response = AsyncMock(side_effect=fake_formulate)

    agent = main_mod.build_crew().agents[0]
    task = main_mod.build_strict_task(agent)
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    result = await crew.kickoff_async()
    assert result.tasks_output[0].guardrail_verdict == "pass"
    assert "Agenten" in result.raw


@pytest.mark.asyncio
async def test_guardrail_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch)
    main_mod = _load_main()

    # Every attempt is 20 words — always fails word_count(max_words=10).
    async def fake_formulate(user_message, results, working_memory):
        return ResponseEnvelope(
            content=(
                "Cognithor ist ein lokal-first autonomes Agenten-Betriebssystem "
                "mit umfangreicher MCP-Tool-Palette und PGE-Trinity Architektur für "
                "komplexe Aufgaben"
            ),
            directive=None,
        )

    mock_planner = MagicMock(name="MockPlanner")
    mock_planner._cost_tracker = None
    mock_planner.formulate_response = AsyncMock(side_effect=fake_formulate)

    agent = main_mod.build_crew().agents[0]
    task = main_mod.build_strict_task(agent)
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.raises(GuardrailFailure) as exc:
        await crew.kickoff_async()

    # max_retries=2 → 1 initial + 2 retries = 3 attempts
    assert exc.value.attempts == 3
