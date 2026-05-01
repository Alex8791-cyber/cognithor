# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Wave-3 — spatial primitives.

Two new primitives:

- ``crop_smallest_component`` — bbox of the smallest 4-connected
  non-zero component (mirror of ``crop_largest_component``)
- ``neighbor_count_grid`` — each cell becomes the count of 8-connected
  non-zero neighbours (including self), capped at 9

Real-ARC existence proofs:
- ``crop_smallest_component`` solves 23b5c85d, d9fac9be
- ``neighbor_count_grid`` solves ce22a75a
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.primitives import (
    crop_smallest_component,
    neighbor_count_grid,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

REAL_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "cognithor_bench" / "arc_agi3_real"


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# crop_smallest_component
# ---------------------------------------------------------------------------


class TestCropSmallestComponent:
    def test_basic_two_components(self) -> None:
        # 1-cell of color 1, 2-cell of color 2 → return the 1-cell bbox.
        out = crop_smallest_component(_g([[0, 1, 0, 2, 2], [0, 0, 0, 2, 0]]))
        assert out.tolist() == [[1]]

    def test_single_component_returns_its_bbox(self) -> None:
        out = crop_smallest_component(_g([[0, 0, 0], [0, 5, 5], [0, 5, 0]]))
        # Single component → smallest = it.
        assert out.tolist() == [[5, 5], [5, 0]]

    def test_all_zero_returns_input_copy(self) -> None:
        inp = _g([[0, 0], [0, 0]])
        out = crop_smallest_component(inp)
        assert out.tolist() == inp.tolist()

    def test_preserves_int8_dtype(self) -> None:
        out = crop_smallest_component(_g([[1, 0], [1, 0]]))
        assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# neighbor_count_grid
# ---------------------------------------------------------------------------


class TestNeighborCountGrid:
    def test_diagonal_pattern(self) -> None:
        # 3 isolated diagonal cells. Counts: corners see 2, mid sees 3, off-diag see 1 or 2.
        out = neighbor_count_grid(_g([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        assert out.tolist() == [[2, 2, 1], [2, 3, 2], [1, 2, 2]]

    def test_all_zero_yields_all_zero(self) -> None:
        out = neighbor_count_grid(_g([[0, 0], [0, 0]]))
        assert out.tolist() == [[0, 0], [0, 0]]

    def test_full_grid_caps_at_nine(self) -> None:
        # 4×4 all-1 grid: corner sees 4, edge sees 6, interior sees 9.
        out = neighbor_count_grid(np.ones((4, 4), dtype=np.int8))
        assert out[0, 0] == 4  # corner: self + 3 neighbours
        assert out[0, 1] == 6  # edge: self + 5 neighbours
        assert out[1, 1] == 9  # interior: self + 8 neighbours

    def test_preserves_shape(self) -> None:
        out = neighbor_count_grid(_g([[1, 2, 3, 4, 5]]))
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    @pytest.mark.parametrize("name", ["crop_smallest_component", "neighbor_count_grid"])
    def test_registered(self, name: str) -> None:
        assert name in REGISTRY.names()
        spec = REGISTRY.get(name)
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"


# ---------------------------------------------------------------------------
# Real-ARC existence proofs
# ---------------------------------------------------------------------------


class TestRealARCExistenceProofs:
    @staticmethod
    def _load(task_id: str) -> dict:
        path = REAL_CORPUS_ROOT / "tasks" / "training" / f"{task_id}.json"
        if not path.exists():
            pytest.skip(f"real ARC corpus not present at {REAL_CORPUS_ROOT}")
        return json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("task_id", ["23b5c85d", "d9fac9be"])
    def test_crop_smallest_component_solves(self, task_id: str) -> None:
        task = self._load(task_id)
        for i, demo in enumerate(task["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            np.testing.assert_array_equal(
                crop_smallest_component(inp), expected, err_msg=f"{task_id} demo {i}"
            )

    def test_neighbor_count_grid_solves_ce22a75a(self) -> None:
        task = self._load("ce22a75a")
        for i, demo in enumerate(task["train"]):
            inp = np.array(demo["input"], dtype=np.int8)
            expected = np.array(demo["output"], dtype=np.int8)
            np.testing.assert_array_equal(
                neighbor_count_grid(inp), expected, err_msg=f"ce22a75a demo {i}"
            )
