# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-2 — ActionDecoder + UniformActionDecoder tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import (
    ActionDecoder,
    UniformActionDecoder,
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
    game_id: str = "ls20"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubAction] = field(default_factory=list)


def _simple_actions() -> list[_StubAction]:
    return [
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
        _StubAction(name="RESET", value=0),
    ]


def _complex_action() -> _StubAction:
    return _StubAction(name="ACTION6", value=6, _is_simple=False)


class TestUniformActionDecoder:
    def test_picks_first_available(self) -> None:
        decoder = UniformActionDecoder()
        actions = _simple_actions()
        frame = _StubFrame(available_actions=actions)
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "ACTION1"

    def test_sets_reasoning(self) -> None:
        decoder = UniformActionDecoder()
        actions = _simple_actions()
        frame = _StubFrame(available_actions=actions)
        chosen = decoder.decode([frame], frame)
        assert "UniformActionDecoder" in chosen.reasoning

    def test_raises_on_empty_available_actions(self) -> None:
        decoder = UniformActionDecoder()
        frame = _StubFrame(available_actions=[])
        with pytest.raises(RuntimeError, match="no available_actions"):
            decoder.decode([frame], frame)


class TestComplexActionDataWiring:
    def test_default_complex_data_is_origin(self) -> None:
        decoder = UniformActionDecoder()
        complex_first = [_complex_action(), _StubAction(name="ACTION1", value=1)]
        frame = _StubFrame(available_actions=complex_first)
        chosen = decoder.decode([frame], frame)
        assert chosen.is_complex() is True
        # set_data was called with default origin coordinates.
        assert chosen._data == {"x": 0, "y": 0}

    def test_simple_action_does_not_get_data(self) -> None:
        decoder = UniformActionDecoder()
        actions = _simple_actions()
        frame = _StubFrame(available_actions=actions)
        chosen = decoder.decode([frame], frame)
        assert chosen.is_simple() is True
        assert chosen._data == {}


class TestSubclassContract:
    def test_subclass_returns_action_outside_whitelist_raises(self) -> None:
        """A subclass that returns an action not in `available_actions`
        gets caught by the base class's validation, preventing the
        upstream undefined-behaviour from leaking through.
        """

        class _BadDecoder(ActionDecoder):
            def pick_action(
                self,
                frames: list[Any],
                latest_frame: Any,
                available_actions: list[Any],
            ) -> tuple[Any, str]:
                # Fabricate an action that isn't in the whitelist.
                return _StubAction(name="ACTION999", value=999), "fabricated"

        decoder = _BadDecoder()
        frame = _StubFrame(available_actions=_simple_actions())
        with pytest.raises(RuntimeError, match="not in available_actions"):
            decoder.decode([frame], frame)

    def test_custom_subclass_can_choose_specific_action(self) -> None:
        class _PickResetDecoder(ActionDecoder):
            def pick_action(
                self,
                frames: list[Any],
                latest_frame: Any,
                available_actions: list[Any],
            ) -> tuple[Any, str]:
                reset = next(a for a in available_actions if a.name == "RESET")
                return reset, "test: always reset"

        decoder = _PickResetDecoder()
        frame = _StubFrame(available_actions=_simple_actions())
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "RESET"
        assert chosen.reasoning == "test: always reset"

    def test_default_reasoning_when_subclass_returns_empty(self) -> None:
        class _NoReasoningDecoder(ActionDecoder):
            def pick_action(
                self,
                frames: list[Any],
                latest_frame: Any,
                available_actions: list[Any],
            ) -> tuple[Any, str]:
                return available_actions[0], ""

        decoder = _NoReasoningDecoder()
        frame = _StubFrame(available_actions=_simple_actions())
        chosen = decoder.decode([frame], frame)
        # Falls back to the class default when the subclass returns "".
        assert chosen.reasoning == ActionDecoder.DEFAULT_REASONING
