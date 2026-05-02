# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 PR-12 — LLM prompt enrichment with FrameAnalyzer summary.

The :class:`LLMActionDecoder` now accepts an optional
:class:`FrameAnalyzer`. When wired, every prompt sent to the LLM
includes a one-line summary of the per-action movement signatures
the analyser has learned so far — a critical signal for keyboard-
controlled games where the upstream API doesn't announce what each
``ACTIONx`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    LLMActionDecoder,
    summarise_action_effects,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str = "NOT_FINISHED"


@dataclass
class _StubAction:
    name: str
    value: int
    reasoning: str = ""
    _data: dict[str, Any] = field(default_factory=dict)
    _is_simple: bool = True

    def is_simple(self) -> bool:
        return self._is_simple

    def is_complex(self) -> bool:
        return not self._is_simple

    def set_data(self, data: dict[str, Any]) -> None:
        self._data = dict(data)


@dataclass
class _StubFrame:
    game_id: str = "smoke"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubAction] = field(default_factory=list)


def _frame(grid: np.ndarray) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
        _StubAction(name="ACTION3", value=3),
    ]
    return _StubFrame(frame=[grid], available_actions=actions)


class TestSummariseActionEffects:
    def test_empty_analyzer_returns_placeholder(self) -> None:
        fa = FrameAnalyzer()
        out = summarise_action_effects(fa)
        assert "no action effects" in out

    def test_summary_contains_action_names(self) -> None:
        fa = FrameAnalyzer()
        # Train the analyzer on a clear "DOWN" pattern.
        fa.analyze(np.array([[1, 0], [0, 0]], dtype=np.int8))
        fa.analyze(np.array([[0, 0], [1, 0]], dtype=np.int8), action="DOWN")
        out = summarise_action_effects(fa)
        assert "DOWN" in out
        assert "n=" in out

    def test_summary_caps_at_max_actions(self) -> None:
        fa = FrameAnalyzer()
        # Inject 10 distinct actions.
        prev = np.zeros((4, 4), dtype=np.int8)
        fa.analyze(prev)
        for i in range(10):
            new = prev.copy()
            new[i % 4, i % 4] = (i % 7) + 1
            fa.analyze(new, action=f"ACT{i}")
            prev = new
        out = summarise_action_effects(fa, max_actions=3)
        # Only 3 actions in the summary string.
        assert out.count(";") == 2  # 3 items separated by 2 semicolons


class TestFrameContextActionEffects:
    def test_default_is_empty(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1"],
            history_summary="",
            levels_completed=0,
            win_levels=1,
        )
        assert ctx.action_effects_summary == ""

    def test_explicit_value_persists(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1"],
            history_summary="",
            levels_completed=0,
            win_levels=1,
            action_effects_summary="ACTION1: row+1, col+0 (n=2)",
        )
        assert "row+1" in ctx.action_effects_summary


class TestLLMActionDecoderWithAnalyzer:
    def test_no_analyzer_summary_empty(self) -> None:
        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_capture,
        )
        f = _frame(np.array([[1, 0]], dtype=np.int8))
        decoder.decode([f], f)
        assert captured[0].action_effects_summary == ""

    def test_with_analyzer_summary_populated(self) -> None:
        fa = FrameAnalyzer()
        # Pre-train the analyzer.
        fa.analyze(np.array([[1, 0]], dtype=np.int8))
        fa.analyze(np.array([[0, 1]], dtype=np.int8), action="RIGHT")

        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_capture,
            frame_analyzer=fa,
        )
        f = _frame(np.array([[1, 0]], dtype=np.int8))
        decoder.decode([f], f)
        assert captured[0].action_effects_summary  # non-empty
        assert "RIGHT" in captured[0].action_effects_summary
