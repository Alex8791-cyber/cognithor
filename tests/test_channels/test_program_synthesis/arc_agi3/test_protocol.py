# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-1 — Protocol-conformance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognithor.channels.program_synthesis.arc_agi3.protocol import (
    FrameDataProtocol,
    GameActionProtocol,
    GameStateProtocol,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str


@dataclass
class _StubGameAction:
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
class _StubFrameData:
    game_id: str
    state: _StubGameState
    levels_completed: int
    win_levels: int
    guid: str
    full_reset: bool
    frame: list[Any]
    available_actions: list[_StubGameAction]


class TestProtocolConformance:
    def test_stub_state_satisfies_protocol(self) -> None:
        s = _StubGameState(name="WIN")
        assert isinstance(s, GameStateProtocol)
        assert s.name == "WIN"

    def test_stub_action_satisfies_protocol(self) -> None:
        a = _StubGameAction(name="ACTION1", value=1, reasoning="test")
        assert isinstance(a, GameActionProtocol)
        assert a.is_simple() is True
        assert a.is_complex() is False
        a.set_data({"x": 1})
        assert a._data == {"x": 1}

    def test_stub_frame_satisfies_protocol(self) -> None:
        f = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            levels_completed=0,
            win_levels=3,
            guid="test-guid",
            full_reset=False,
            frame=[[[0, 1], [2, 3]]],
            available_actions=[
                _StubGameAction(name="ACTION1", value=1),
                _StubGameAction(name="RESET", value=0),
            ],
        )
        assert isinstance(f, FrameDataProtocol)
        assert f.game_id == "ls20"
        assert f.state.name == "NOT_FINISHED"
        assert len(f.available_actions) == 2

    def test_complex_action_set_data_idempotent(self) -> None:
        a = _StubGameAction(name="ACTION6", value=6, _is_simple=False)
        a.set_data({"x": 5, "y": 7})
        assert a.is_complex() is True
        a.set_data({"x": 0, "y": 0})
        assert a._data == {"x": 0, "y": 0}
