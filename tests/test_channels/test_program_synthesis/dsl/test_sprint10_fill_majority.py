# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Wave-1 PR-3 — ``fill_with_most_common_color`` primitive.

Single new high-impact primitive: fill the entire grid with its
most-frequent colour. Solves ARC task ``5582e5ca`` where the rule is
"collapse the input grid to its dominant colour".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import fill_with_most_common_color
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class TestFillWithMostCommonColor:
    def test_basic_dominant_fill(self) -> None:
        out = fill_with_most_common_color(_g([[4, 4, 8], [6, 4, 3], [6, 3, 0]]))
        # Most common is 4 (count 3). Output is 3×3 of 4s.
        assert out.tolist() == [[4, 4, 4], [4, 4, 4], [4, 4, 4]]

    def test_preserves_shape(self) -> None:
        out = fill_with_most_common_color(_g([[1, 2, 3, 4, 5]]))
        assert out.shape == (1, 5)

    def test_preserves_int8_dtype(self) -> None:
        out = fill_with_most_common_color(_g([[1, 1]]))
        assert out.dtype == np.int8

    def test_zero_is_dominant_yields_all_zero(self) -> None:
        out = fill_with_most_common_color(_g([[0, 0, 0], [0, 1, 0]]))
        assert out.tolist() == [[0, 0, 0], [0, 0, 0]]

    def test_tie_broken_by_lowest_index(self) -> None:
        # Colours 1 and 2 each appear twice; 1 wins on lowest-index tie-break.
        out = fill_with_most_common_color(_g([[1, 2], [1, 2]]))
        assert out.tolist() == [[1, 1], [1, 1]]


class TestRegistryIntegration:
    def test_registered_in_global_registry(self) -> None:
        assert "fill_with_most_common_color" in REGISTRY.names()
        spec = REGISTRY.get("fill_with_most_common_color")
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"


class TestSolves5582e5ca:
    """Existence proof on real fchollet/ARC-AGI training task 5582e5ca."""

    @staticmethod
    def _load() -> dict:
        path = REAL_CORPUS_ROOT / "tasks" / "training" / "5582e5ca.json"
        if not path.exists():
            pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_train_demos_solved(self) -> None:
        task = self._load()
        for i, demo in enumerate(task["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            np.testing.assert_array_equal(
                fill_with_most_common_color(inp),
                expected,
                err_msg=f"5582e5ca demo {i}",
            )

    def test_test_demo_solved(self) -> None:
        task = self._load()
        demo = task["test"][0]
        inp = np.array(demo["input"], dtype=np.int8)
        expected = np.array(demo["output"], dtype=np.int8)
        np.testing.assert_array_equal(fill_with_most_common_color(inp), expected)
