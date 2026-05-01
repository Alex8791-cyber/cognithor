# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Wave-1 PR-2 — symmetry-completion primitives.

Four new high-impact primitives that fill in a partial grid so it
becomes symmetric across one of the four ARC-relevant axes:

- ``complete_symmetry_h`` — vertical axis (left-right mirror)
- ``complete_symmetry_v`` — horizontal axis (top-bottom mirror)
- ``complete_symmetry_d`` — main diagonal (square grids only)
- ``complete_symmetry_antidiag`` — anti-diagonal (square grids only)

Existence-proof tasks loaded from the real ARC training corpus
(committed in PR #273): each primitive must reproduce all train demos
of at least one canonical task family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import (
    complete_symmetry_antidiag,
    complete_symmetry_d,
    complete_symmetry_h,
    complete_symmetry_v,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _load_train_demos(task_id: str) -> list[tuple[np.ndarray, np.ndarray]]:
    path = REAL_CORPUS_ROOT / "tasks" / "training" / f"{task_id}.json"
    if not path.exists():
        pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        (np.array(d["input"], dtype=np.int8), np.array(d["output"], dtype=np.int8))
        for d in data["train"]
    ]


# ---------------------------------------------------------------------------
# complete_symmetry_h — horizontal (left-right) axis
# ---------------------------------------------------------------------------


class TestCompleteSymmetryH:
    def test_basic_left_right_fill(self) -> None:
        out = complete_symmetry_h(_g([[1, 0, 0], [0, 2, 0]]))
        # Right column gets filled from left.
        assert out.tolist() == [[1, 0, 1], [0, 2, 0]]

    def test_already_symmetric_grid_is_unchanged(self) -> None:
        sym = _g([[1, 2, 1], [3, 4, 3]])
        out = complete_symmetry_h(sym)
        np.testing.assert_array_equal(out, sym)

    def test_all_zero_input_yields_all_zero(self) -> None:
        out = complete_symmetry_h(_g([[0, 0], [0, 0]]))
        assert out.tolist() == [[0, 0], [0, 0]]

    def test_conflict_cells_keep_input_value(self) -> None:
        # Both (0,0)=1 and (0,2)=2 are non-zero and conflict — neither
        # is zero, so neither gets overwritten by the other.
        inp = _g([[1, 0, 2]])
        out = complete_symmetry_h(inp)
        assert out.tolist() == [[1, 0, 2]]

    def test_preserves_int8_dtype(self) -> None:
        out = complete_symmetry_h(_g([[1, 0]]))
        assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# complete_symmetry_v — vertical (top-bottom) axis
# ---------------------------------------------------------------------------


class TestCompleteSymmetryV:
    def test_basic_top_bottom_fill(self) -> None:
        out = complete_symmetry_v(_g([[1, 2], [0, 0]]))
        # Bottom row gets filled from top.
        assert out.tolist() == [[1, 2], [1, 2]]

    def test_already_symmetric_grid_is_unchanged(self) -> None:
        sym = _g([[1, 2], [3, 4], [1, 2]])
        out = complete_symmetry_v(sym)
        np.testing.assert_array_equal(out, sym)

    def test_three_row_with_partial_top_and_bottom(self) -> None:
        # Top row has [1,2], middle is centre, bottom is empty.
        # Vertical partner of (0, c) is (2, c); of (2, c) is (0, c).
        inp = _g([[1, 2], [3, 4], [0, 0]])
        out = complete_symmetry_v(inp)
        assert out.tolist() == [[1, 2], [3, 4], [1, 2]]


# ---------------------------------------------------------------------------
# complete_symmetry_d — main diagonal (square only)
# ---------------------------------------------------------------------------


class TestCompleteSymmetryD:
    def test_basic_main_diagonal_fill(self) -> None:
        out = complete_symmetry_d(_g([[1, 2, 3], [0, 5, 6], [0, 0, 9]]))
        # Lower-triangle gets filled from upper.
        assert out.tolist() == [[1, 2, 3], [2, 5, 6], [3, 6, 9]]

    def test_non_square_returns_input_unchanged(self) -> None:
        inp = _g([[1, 2, 3], [4, 5, 6]])
        out = complete_symmetry_d(inp)
        np.testing.assert_array_equal(out, inp)

    def test_already_symmetric_unchanged(self) -> None:
        sym = _g([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        out = complete_symmetry_d(sym)
        np.testing.assert_array_equal(out, sym)


# ---------------------------------------------------------------------------
# complete_symmetry_antidiag — anti-diagonal (square only)
# ---------------------------------------------------------------------------


class TestCompleteSymmetryAntidiag:
    def test_basic_antidiagonal_fill(self) -> None:
        # Anti-diagonal partner of (r, c) is (n-1-c, n-1-r). In a 3×3
        # grid, (1, 2) ↔ (0, 1). Input[0,1]=2 fills input[1,2]=0.
        inp = _g([[1, 2, 0], [0, 5, 0], [7, 0, 9]])
        out = complete_symmetry_antidiag(inp)
        # Verified by hand against the partner formula above.
        assert out.tolist() == [[1, 2, 0], [0, 5, 2], [7, 0, 9]]

    def test_non_square_returns_input_unchanged(self) -> None:
        inp = _g([[1, 2, 3], [4, 5, 6]])
        out = complete_symmetry_antidiag(inp)
        np.testing.assert_array_equal(out, inp)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    @pytest.mark.parametrize(
        "name",
        [
            "complete_symmetry_h",
            "complete_symmetry_v",
            "complete_symmetry_d",
            "complete_symmetry_antidiag",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert name in REGISTRY.names()
        spec = REGISTRY.get(name)
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"


# ---------------------------------------------------------------------------
# Real ARC existence proofs
# ---------------------------------------------------------------------------


class TestRealARCExistenceProofs:
    """Existence proofs against the real fchollet/ARC-AGI training set.

    Task IDs were determined empirically by running the actual
    primitive on every training input and comparing to the expected
    output. Only ``complete_symmetry_v`` produces an exact match on
    any current real task (496994bd, f25ffba3) — the other axes are
    still shipped because Phase-1 search will reach them via
    ``rotate90 ∘ complete_symmetry_v ∘ rotate270`` etc., and so
    enrich the compositional search space even without standalone
    existence proofs.
    """

    @pytest.mark.parametrize("task_id", ["496994bd", "f25ffba3"])
    def test_complete_symmetry_v_solves(self, task_id: str) -> None:
        for inp, expected in _load_train_demos(task_id):
            np.testing.assert_array_equal(
                complete_symmetry_v(inp), expected, err_msg=f"{task_id} demo"
            )
