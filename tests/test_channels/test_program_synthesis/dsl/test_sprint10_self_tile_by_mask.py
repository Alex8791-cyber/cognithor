# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 — ``self_tile_by_mask`` primitive.

Validates the new fractal self-tile primitive end-to-end:
- Correct output shape and stamp/blank semantics on small grids.
- Solves real ARC task ``007bbfb7`` (3×3 → 9×9, fchollet/ARC-AGI training).
- Registered in the global :data:`REGISTRY` and increments the catalog
  past Sprint-8's 65 primitives.

The 007bbfb7 fixture is loaded from the real ARC corpus committed in
PR #273 (``cognithor_bench/arc_agi3_real/``). All five training demos
must reproduce exactly when the input is fed through
``self_tile_by_mask`` — that is the primitive's existence proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import self_tile_by_mask
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Unit semantics
# ---------------------------------------------------------------------------


class TestSelfTileByMaskUnit:
    def test_output_shape_is_squared(self) -> None:
        out = self_tile_by_mask(_g([[0, 1], [1, 0]]))
        # 2×2 input → 4×4 output (R*R, C*C).
        assert out.shape == (4, 4)

    def test_non_square_input_output_shape(self) -> None:
        # 2×3 input → 4×9 output.
        out = self_tile_by_mask(_g([[0, 1, 1], [1, 0, 1]]))
        assert out.shape == (4, 9)

    def test_zero_cell_yields_zero_block(self) -> None:
        out = self_tile_by_mask(_g([[0, 1], [1, 0]]))
        # Block (0,0) corresponds to grid[0,0] = 0 → all zeros.
        assert out[0:2, 0:2].tolist() == [[0, 0], [0, 0]]
        # Block (1,1) corresponds to grid[1,1] = 0 → all zeros.
        assert out[2:4, 2:4].tolist() == [[0, 0], [0, 0]]

    def test_nonzero_cell_yields_input_stamp(self) -> None:
        inp = _g([[0, 1], [1, 0]])
        out = self_tile_by_mask(inp)
        # Block (0,1) corresponds to grid[0,1] = 1 → stamp = input itself.
        assert out[0:2, 2:4].tolist() == inp.tolist()
        # Block (1,0) likewise.
        assert out[2:4, 0:2].tolist() == inp.tolist()

    def test_full_2x2_pattern(self) -> None:
        out = self_tile_by_mask(_g([[0, 1], [1, 0]]))
        # Documented exemplar from the primitive's docstring.
        assert out.tolist() == [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]

    def test_preserves_int8_dtype(self) -> None:
        out = self_tile_by_mask(_g([[0, 1], [1, 0]]))
        assert out.dtype == np.int8

    def test_all_zero_input_yields_all_zero_output(self) -> None:
        out = self_tile_by_mask(_g([[0, 0], [0, 0]]))
        assert out.shape == (4, 4)
        assert out.tolist() == [[0] * 4 for _ in range(4)]

    def test_all_nonzero_input_equals_tile_3x_for_3x3(self) -> None:
        # For an all-nonzero 3×3 grid every block is the input → equivalent
        # to the existing ``tile_3x`` semantics. This is the "no-mask" case.
        from cognithor.channels.program_synthesis.dsl.primitives import tile_3x

        inp = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(self_tile_by_mask(inp), tile_3x(inp))


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestSelfTileByMaskRegistry:
    def test_registered_in_global_registry(self) -> None:
        assert "self_tile_by_mask" in REGISTRY.names()
        spec = REGISTRY.get("self_tile_by_mask")
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"

    def test_total_primitive_count_at_least_66(self) -> None:
        # Sprint-8 brought the catalog to 65; Sprint-10 first wave adds
        # ``self_tile_by_mask`` → at least 66.
        assert len(REGISTRY.names()) >= 66


# ---------------------------------------------------------------------------
# Real ARC task 007bbfb7 — existence proof
# ---------------------------------------------------------------------------


class TestSolves007bbfb7:
    """The existence proof: this primitive's reason for being is to
    solve ARC task 007bbfb7 (the alphabetically first training task in
    the canonical fchollet/ARC-AGI corpus, 3×3 → 9×9 fractal).
    """

    @staticmethod
    def _load_007bbfb7() -> dict:
        path = REAL_CORPUS_ROOT / "tasks" / "training" / "007bbfb7.json"
        if not path.exists():
            pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_first_training_demo_solved(self) -> None:
        task = self._load_007bbfb7()
        demo = task["train"][0]
        inp = np.array(demo["input"], dtype=np.int8)
        expected = np.array(demo["output"], dtype=np.int8)
        np.testing.assert_array_equal(self_tile_by_mask(inp), expected)

    def test_all_five_training_demos_solved(self) -> None:
        task = self._load_007bbfb7()
        for i, demo in enumerate(task["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            actual = self_tile_by_mask(inp)
            assert actual.shape == expected.shape, f"demo {i} shape mismatch"
            np.testing.assert_array_equal(actual, expected, err_msg=f"007bbfb7 demo {i} mismatch")

    def test_test_demo_solved(self) -> None:
        task = self._load_007bbfb7()
        demo = task["test"][0]
        inp = np.array(demo["input"], dtype=np.int8)
        expected = np.array(demo["output"], dtype=np.int8)
        np.testing.assert_array_equal(self_tile_by_mask(inp), expected)
