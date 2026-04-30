# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 partial-pixel-match tests (plan task 8 slice, spec §7.2)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    average_partial_pixel_match,
    partial_pixel_match,
)

# ---------------------------------------------------------------------------
# partial_pixel_match
# ---------------------------------------------------------------------------


class TestPartialPixelMatch:
    def test_identical_grids_return_one(self) -> None:
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[1, 2], [3, 4]], dtype=np.int8)
        assert partial_pixel_match(a, b) == 1.0

    def test_completely_different_returns_zero(self) -> None:
        a = np.array([[0, 0], [0, 0]], dtype=np.int8)
        b = np.array([[1, 1], [1, 1]], dtype=np.int8)
        assert partial_pixel_match(a, b) == 0.0

    def test_half_match_returns_half(self) -> None:
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[1, 9], [3, 9]], dtype=np.int8)
        # 2 / 4 cells match = 0.5.
        assert partial_pixel_match(a, b) == 0.5

    def test_one_quarter_match(self) -> None:
        a = np.array([[1, 2, 3, 4]], dtype=np.int8)
        b = np.array([[1, 0, 0, 0]], dtype=np.int8)
        assert partial_pixel_match(a, b) == 0.25

    def test_shape_mismatch_returns_zero(self) -> None:
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[1, 2], [3, 4]], dtype=np.int8)
        assert partial_pixel_match(a, b) == 0.0

    def test_empty_grids_return_zero(self) -> None:
        a = np.zeros((0, 0), dtype=np.int8)
        b = np.zeros((0, 0), dtype=np.int8)
        assert partial_pixel_match(a, b) == 0.0

    def test_non_ndarray_input_raises(self) -> None:
        a = np.array([[1]], dtype=np.int8)
        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            partial_pixel_match([[1]], a)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            partial_pixel_match(a, [[1]])  # type: ignore[arg-type]

    def test_non_2d_input_raises(self) -> None:
        a = np.array([1, 2, 3], dtype=np.int8)
        b = np.array([1, 2, 3], dtype=np.int8)
        with pytest.raises(TypeError, match="must be 2-D"):
            partial_pixel_match(a, b)


# ---------------------------------------------------------------------------
# average_partial_pixel_match
# ---------------------------------------------------------------------------


class TestAveragePartialPixelMatch:
    def test_all_perfect_returns_one(self) -> None:
        actuals = [
            np.array([[1, 2]], dtype=np.int8),
            np.array([[3, 4]], dtype=np.int8),
        ]
        assert average_partial_pixel_match(actuals, list(actuals)) == 1.0

    def test_mixed_pairs_average(self) -> None:
        # First pair: 1.0 match. Second pair: 0.5 match. Average = 0.75.
        actuals = [
            np.array([[1, 2], [3, 4]], dtype=np.int8),
            np.array([[1, 2], [3, 4]], dtype=np.int8),
        ]
        expecteds = [
            np.array([[1, 2], [3, 4]], dtype=np.int8),
            np.array([[1, 9], [3, 9]], dtype=np.int8),
        ]
        assert average_partial_pixel_match(actuals, expecteds) == 0.75

    def test_empty_list_returns_zero(self) -> None:
        assert average_partial_pixel_match([], []) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            average_partial_pixel_match(
                [np.array([[1]], dtype=np.int8)],
                [
                    np.array([[1]], dtype=np.int8),
                    np.array([[2]], dtype=np.int8),
                ],
            )

    def test_one_size_mismatch_pulls_average_down(self) -> None:
        actuals = [
            np.array([[1, 2]], dtype=np.int8),
            np.array([[1]], dtype=np.int8),  # shape mismatch with expected
        ]
        expecteds = [
            np.array([[1, 2]], dtype=np.int8),
            np.array([[1, 1]], dtype=np.int8),  # different shape
        ]
        # Pair 1: 1.0; pair 2: 0.0 (shape mismatch). Average = 0.5.
        assert average_partial_pixel_match(actuals, expecteds) == 0.5
