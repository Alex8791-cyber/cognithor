# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Wave-2 PR-4 — ``crop_largest_component`` primitive.

Single new high-impact primitive: extract the largest 4-connected
non-zero component and return its bounding-box subgrid. Solves real
ARC tasks ``1f85a75f`` and ``be94b721`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import crop_largest_component
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class TestCropLargestComponentUnit:
    def test_basic_two_components_keeps_largest(self) -> None:
        # Two single-cell components (size 1 each) — first found wins.
        # Replace with a clear two-component grid: a 2-cell column vs 1-cell.
        out = crop_largest_component(_g([[0, 1, 0, 2], [0, 1, 0, 0]]))
        # Largest = the 2-cell column of 1s. Its bbox is 2×1, content [[1],[1]].
        assert out.tolist() == [[1], [1]]

    def test_single_component_returns_its_bbox(self) -> None:
        # Single L-shaped component → bbox is 2×2.
        out = crop_largest_component(_g([[0, 0, 0], [0, 5, 5], [0, 5, 0]]))
        assert out.tolist() == [[5, 5], [5, 0]]

    def test_all_zero_input_returns_input_copy(self) -> None:
        inp = _g([[0, 0], [0, 0]])
        out = crop_largest_component(inp)
        assert out.tolist() == inp.tolist()

    def test_separate_components_different_colors(self) -> None:
        # 1×1 of color 3 and 2×1 of color 7 → return the 2×1.
        out = crop_largest_component(_g([[3, 0], [0, 7], [0, 7]]))
        assert out.tolist() == [[7], [7]]

    def test_preserves_int8_dtype(self) -> None:
        out = crop_largest_component(_g([[1, 0], [1, 0]]))
        assert out.dtype == np.int8


class TestRegistryIntegration:
    def test_registered(self) -> None:
        assert "crop_largest_component" in REGISTRY.names()
        spec = REGISTRY.get("crop_largest_component")
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"


class TestRealARCExistenceProofs:
    @pytest.mark.parametrize("task_id", ["1f85a75f", "be94b721"])
    def test_solves(self, task_id: str) -> None:
        path = REAL_CORPUS_ROOT / "tasks" / "training" / f"{task_id}.json"
        if not path.exists():
            pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, demo in enumerate(data["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            np.testing.assert_array_equal(
                crop_largest_component(inp),
                expected,
                err_msg=f"{task_id} demo {i}",
            )
