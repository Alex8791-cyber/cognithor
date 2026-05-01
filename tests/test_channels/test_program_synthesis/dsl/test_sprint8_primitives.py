# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-8 — new DSL primitives (object-level + scalar-as-grid)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.dsl.primitives import (
    count_components,
    recolor_by_component_size,
    remove_singletons,
    tile_3x,
    unique_colors_diagonal,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# tile_3x
# ---------------------------------------------------------------------------


class TestTile3x:
    def test_doubles_each_dimension_three_times(self) -> None:
        out = tile_3x(_g([[1, 2]]))
        # Input 1x2 → output 3x6.
        assert out.shape == (3, 6)
        # Every row is [1, 2, 1, 2, 1, 2].
        for r in range(3):
            assert out[r].tolist() == [1, 2, 1, 2, 1, 2]

    def test_preserves_dtype(self) -> None:
        out = tile_3x(_g([[5]]))
        assert out.dtype == np.int8

    def test_registered_in_registry(self) -> None:
        assert "tile_3x" in REGISTRY.names()
        spec = REGISTRY.get("tile_3x")
        assert spec.signature.arity == 1


# ---------------------------------------------------------------------------
# remove_singletons
# ---------------------------------------------------------------------------


class TestRemoveSingletons:
    def test_keeps_pair_drops_singleton(self) -> None:
        # 1@(0,0), 1@(0,1) form a pair; 2@(0,3) and 3@(2,2) are singletons.
        inp = _g([[1, 1, 0, 2], [1, 0, 0, 0], [0, 0, 3, 0]])
        expected = _g([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        # Hmm — (1,0) is also 1, neighbour to (0,0). So three 1s form a
        # connected cluster and stay. Let me just assert against the
        # actual algorithmic output.
        out = remove_singletons(inp)
        # Cell (0,3)=2 has no same-colour neighbour → drop.
        assert out[0, 3] == 0
        # Cell (2,2)=3 has no same-colour neighbour → drop.
        assert out[2, 2] == 0
        # Pair-or-larger 1s stay.
        assert out[0, 0] == 1
        assert out[0, 1] == 1
        assert out[1, 0] == 1
        # Compare full grid via the actual algorithm.
        assert out.tolist() == expected.tolist()

    def test_preserves_zero_background(self) -> None:
        inp = _g([[0, 0], [0, 0]])
        out = remove_singletons(inp)
        assert out.tolist() == [[0, 0], [0, 0]]

    def test_registered(self) -> None:
        assert "remove_singletons" in REGISTRY.names()


# ---------------------------------------------------------------------------
# count_components
# ---------------------------------------------------------------------------


class TestCountComponents:
    def test_four_isolated_cells_count_four(self) -> None:
        out = count_components(_g([[1, 0, 2], [0, 0, 0], [3, 0, 4]]))
        assert out.shape == (1, 1)
        assert int(out[0, 0]) == 4

    def test_one_connected_cluster(self) -> None:
        out = count_components(_g([[1, 1, 1], [1, 1, 1]]))
        assert int(out[0, 0]) == 1

    def test_empty_grid_returns_zero(self) -> None:
        out = count_components(_g([[0, 0], [0, 0]]))
        assert int(out[0, 0]) == 0

    def test_saturates_at_nine(self) -> None:
        # A 3×4 checkerboard has up to 6 isolated cells → count saturates fine.
        # Build a grid with > 9 components manually.
        big = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            big[0, i] = 1  # All same colour — but 10 cells form a single
            # connected component, not 10 separate.
        # Use isolated cells instead: every other cell different colour.
        iso = np.zeros((4, 5), dtype=np.int8)
        # 10 isolated cells at every other position.
        positions = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (2, 0), (2, 2), (2, 4), (3, 1), (3, 3)]
        for p in positions:
            iso[p] = 1  # All same colour but isolated.
        out = count_components(iso)
        # 10 isolated cells → count saturates at 9.
        assert int(out[0, 0]) == 9


# ---------------------------------------------------------------------------
# recolor_by_component_size
# ---------------------------------------------------------------------------


class TestRecolorByComponentSize:
    def test_size_one_becomes_color_one(self) -> None:
        out = recolor_by_component_size(_g([[5, 0, 0], [0, 0, 0]]))
        assert int(out[0, 0]) == 1

    def test_size_two_becomes_color_two(self) -> None:
        out = recolor_by_component_size(_g([[1, 1, 0], [0, 0, 0]]))
        assert int(out[0, 0]) == 2
        assert int(out[0, 1]) == 2

    def test_mixed_sizes(self) -> None:
        out = recolor_by_component_size(_g([[1, 0, 1], [1, 0, 0], [0, 0, 1]]))
        # Components: {(0,0),(1,0)} size 2, {(0,2)} size 1, {(2,2)} size 1.
        assert int(out[0, 0]) == 2
        assert int(out[1, 0]) == 2
        assert int(out[0, 2]) == 1
        assert int(out[2, 2]) == 1

    def test_preserves_zero_background(self) -> None:
        out = recolor_by_component_size(_g([[0, 0], [0, 0]]))
        assert out.tolist() == [[0, 0], [0, 0]]


# ---------------------------------------------------------------------------
# unique_colors_diagonal
# ---------------------------------------------------------------------------


class TestUniqueColorsDiagonal:
    def test_three_colors_yield_3x3(self) -> None:
        out = unique_colors_diagonal(_g([[3, 0, 1], [0, 2, 0]]))
        assert out.shape == (3, 3)
        # Diagonal: sorted {1, 2, 3}.
        assert int(out[0, 0]) == 1
        assert int(out[1, 1]) == 2
        assert int(out[2, 2]) == 3
        # Off-diagonal: zeros.
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert int(out[i, j]) == 0

    def test_two_colors(self) -> None:
        out = unique_colors_diagonal(_g([[5, 0, 7]]))
        assert out.shape == (2, 2)
        assert int(out[0, 0]) == 5
        assert int(out[1, 1]) == 7

    def test_empty_grid_returns_1x1_zero(self) -> None:
        out = unique_colors_diagonal(_g([[0, 0], [0, 0]]))
        assert out.shape == (1, 1)
        assert int(out[0, 0]) == 0


# ---------------------------------------------------------------------------
# Registry — all 5 new primitives auto-registered
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_all_five_primitives_registered(self) -> None:
        for name in (
            "tile_3x",
            "remove_singletons",
            "count_components",
            "recolor_by_component_size",
            "unique_colors_diagonal",
        ):
            assert name in REGISTRY.names(), f"{name} not registered"

    def test_total_primitive_count(self) -> None:
        # Sprint-7 had 60; Sprint-8 adds 5 → 65.
        assert len(REGISTRY.names()) >= 65
