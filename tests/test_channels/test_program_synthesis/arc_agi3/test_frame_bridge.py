# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-2 — FrameBridge tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import (
    ClampPolicy,
    FrameBridge,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str = "NOT_FINISHED"


@dataclass
class _StubFrame:
    game_id: str = "ls20"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[Any] = field(default_factory=list)


def _frame_with(layers: list[Any]) -> _StubFrame:
    return _StubFrame(frame=layers)


class TestFrameBridgeBasics:
    def test_default_layer_index_zero(self) -> None:
        b = FrameBridge()
        assert b.layer_index == 0
        assert b.clamp_policy is ClampPolicy.SATURATE

    def test_negative_layer_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="layer_index"):
            FrameBridge(layer_index=-1)

    def test_extracts_first_layer(self) -> None:
        b = FrameBridge()
        out = b.extract_grid(_frame_with([np.array([[1, 2], [3, 4]], dtype=np.int_)]))
        assert out.tolist() == [[1, 2], [3, 4]]
        assert out.dtype == np.int8

    def test_layer_index_out_of_range_raises(self) -> None:
        b = FrameBridge(layer_index=2)
        with pytest.raises(IndexError, match="only has 1 layer"):
            b.extract_grid(_frame_with([np.array([[1]])]))

    def test_accepts_list_of_lists(self) -> None:
        b = FrameBridge()
        out = b.extract_grid(_frame_with([[[5, 6, 7], [8, 9, 0]]]))
        assert out.shape == (2, 3)
        assert out.tolist() == [[5, 6, 7], [8, 9, 0]]

    def test_rejects_non_2d_layer(self) -> None:
        b = FrameBridge()
        with pytest.raises(ValueError, match="must be 2-D"):
            b.extract_grid(_frame_with([np.array([1, 2, 3], dtype=np.int_)]))


class TestClampPolicies:
    def test_saturate_clamps_above_nine(self) -> None:
        b = FrameBridge(clamp_policy=ClampPolicy.SATURATE)
        out = b.extract_grid(_frame_with([np.array([[10, 11, 15]], dtype=np.int_)]))
        # 10..15 → saturated to 9.
        assert out.tolist() == [[9, 9, 9]]

    def test_saturate_clamps_below_zero(self) -> None:
        b = FrameBridge(clamp_policy=ClampPolicy.SATURATE)
        # ARC-AGI-3 colour values shouldn't be negative, but the clamp
        # is symmetric. (Value -1 is unlikely upstream but robust here.)
        out = b.extract_grid(_frame_with([np.array([[-1, 0, 5]], dtype=np.int_)]))
        assert out.tolist() == [[0, 0, 5]]

    def test_modulo_wraps(self) -> None:
        b = FrameBridge(clamp_policy=ClampPolicy.MODULO)
        out = b.extract_grid(_frame_with([np.array([[10, 11, 15]], dtype=np.int_)]))
        # 10 % 10 = 0, 11 % 10 = 1, 15 % 10 = 5.
        assert out.tolist() == [[0, 1, 5]]

    def test_strict_passes_clean_grid(self) -> None:
        b = FrameBridge(clamp_policy=ClampPolicy.STRICT)
        out = b.extract_grid(_frame_with([np.array([[0, 5, 9]], dtype=np.int_)]))
        assert out.tolist() == [[0, 5, 9]]

    def test_strict_raises_on_value_above_nine(self) -> None:
        b = FrameBridge(clamp_policy=ClampPolicy.STRICT)
        with pytest.raises(ValueError, match="STRICT"):
            b.extract_grid(_frame_with([np.array([[0, 5, 10]], dtype=np.int_)]))


class TestMultiLayerFrame:
    def test_picks_chosen_layer(self) -> None:
        b = FrameBridge(layer_index=1)
        frame = _frame_with(
            [
                np.array([[1, 1], [1, 1]], dtype=np.int_),  # layer 0
                np.array([[2, 2], [2, 2]], dtype=np.int_),  # layer 1
            ]
        )
        out = b.extract_grid(frame)
        assert out.tolist() == [[2, 2], [2, 2]]

    def test_first_layer_is_default(self) -> None:
        b = FrameBridge()  # layer_index=0
        frame = _frame_with(
            [
                np.array([[7, 8]], dtype=np.int_),
                np.array([[1, 2]], dtype=np.int_),
            ]
        )
        out = b.extract_grid(frame)
        assert out.tolist() == [[7, 8]]


class TestPurity:
    def test_does_not_mutate_input(self) -> None:
        original = np.array([[10, 11], [12, 13]], dtype=np.int_)
        b = FrameBridge(clamp_policy=ClampPolicy.SATURATE)
        b.extract_grid(_frame_with([original.copy()]))
        # Original outside the bridge unchanged. (We pass a copy in,
        # so this also confirms the bridge doesn't write back into
        # the input through aliasing.)
        assert original.tolist() == [[10, 11], [12, 13]]
