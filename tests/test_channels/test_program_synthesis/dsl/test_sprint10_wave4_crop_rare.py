# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Wave-4 — ``crop_to_least_common_color_cells``.

Single new high-impact primitive: find the rarest non-zero colour and
return the bounding box of its cells. Solves ARC tasks 0b148d64 and
c909285e.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import (
    crop_to_least_common_color_cells,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class TestUnit:
    def test_basic_single_marker(self) -> None:
        # 8 cells of color 2, 1 cell of color 1 → rarest = 1 → bbox 1×1.
        out = crop_to_least_common_color_cells(_g([[2, 2, 2], [2, 1, 2], [2, 2, 2]]))
        assert out.tolist() == [[1]]

    def test_marker_block(self) -> None:
        # 8 cells of color 2, 2 cells of color 5 → rarest = 5 → bbox of 5s.
        out = crop_to_least_common_color_cells(_g([[2, 2, 2], [5, 2, 5], [2, 2, 2]]))
        # 5s at (1,0) and (1,2) → bbox rows=[1..2), cols=[0..3) → grid[1:2, 0:3].
        assert out.tolist() == [[5, 2, 5]]

    def test_all_zero_returns_input_copy(self) -> None:
        inp = _g([[0, 0], [0, 0]])
        out = crop_to_least_common_color_cells(inp)
        assert out.tolist() == inp.tolist()

    def test_preserves_int8_dtype(self) -> None:
        out = crop_to_least_common_color_cells(_g([[2, 2, 1]]))
        assert out.dtype == np.int8


class TestRegistry:
    def test_registered(self) -> None:
        assert "crop_to_least_common_color_cells" in REGISTRY.names()


class TestRealARC:
    @pytest.mark.parametrize("task_id", ["0b148d64", "c909285e"])
    def test_solves(self, task_id: str) -> None:
        path = REAL_CORPUS_ROOT / "tasks" / "training" / f"{task_id}.json"
        if not path.exists():
            pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, demo in enumerate(data["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            np.testing.assert_array_equal(
                crop_to_least_common_color_cells(inp),
                expected,
                err_msg=f"{task_id} demo {i}",
            )
