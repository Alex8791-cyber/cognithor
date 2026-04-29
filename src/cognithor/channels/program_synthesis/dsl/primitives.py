# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-DSL primitive implementations (spec §7.2).

Each primitive is a pure function returning a fresh array (no in-place
mutation, no shared state). All inputs are np.int8 grids in the
ARC-conventional value range 0..9.

Primitives are registered in the module-level :data:`REGISTRY` at import
time via the :func:`primitive` decorator. The catalog is the
single source of truth for both the search engine and the public DSL
reference.

This file currently holds the **geometric** and **color** groups
(15 primitives). Subsequent PRs add size/scale, spatial, object,
mask/logic, construction, and constant groups for a Phase 1 total of 56.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature

_Grid = NDArray[np.int8]


def _check_grid(g: object, name: str) -> _Grid:
    """Validate *g* is a 2-D int8 numpy grid in the ARC value range.

    Raises :class:`TypeMismatchError` for any deviation. The check is
    cheap (shape + dtype only); value-range is verified once on input
    boundaries, not per primitive call, to avoid quadratic overhead in
    deep search.
    """
    if not isinstance(g, np.ndarray):
        raise TypeMismatchError(f"{name}: expected ndarray, got {type(g).__name__}")
    if g.ndim != 2:
        raise TypeMismatchError(f"{name}: expected 2-D grid, got {g.ndim}-D")
    if g.dtype != np.int8:
        raise TypeMismatchError(f"{name}: expected int8 dtype, got {g.dtype}")
    return g


def _check_color(c: object, name: str) -> int:
    if not isinstance(c, int) or isinstance(c, bool):
        raise TypeMismatchError(f"{name}: expected int color, got {type(c).__name__}")
    if not 0 <= c <= 9:
        raise TypeMismatchError(f"{name}: color {c} out of ARC range 0..9")
    return c


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------


@primitive(
    name="identity",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=0.1,
    description="Return the grid unchanged. Cheap building block for branches.",
    examples=(("[[1,2],[3,4]]", "[[1,2],[3,4]]"),),
)
def identity(grid: _Grid) -> _Grid:
    _check_grid(grid, "identity")
    return grid.copy()


# ---------------------------------------------------------------------------
# 2-9. Geometric transforms
# ---------------------------------------------------------------------------


@primitive(
    name="rotate90",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Rotate the grid 90° clockwise.",
    examples=(("[[1,2],[3,4]]", "[[3,1],[4,2]]"),),
)
def rotate90(grid: _Grid) -> _Grid:
    _check_grid(grid, "rotate90")
    return np.rot90(grid, k=-1).copy()


@primitive(
    name="rotate180",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Rotate the grid 180°.",
    examples=(("[[1,2],[3,4]]", "[[4,3],[2,1]]"),),
)
def rotate180(grid: _Grid) -> _Grid:
    _check_grid(grid, "rotate180")
    return np.rot90(grid, k=2).copy()


@primitive(
    name="rotate270",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Rotate the grid 270° clockwise (= 90° counter-clockwise).",
    examples=(("[[1,2],[3,4]]", "[[2,4],[1,3]]"),),
)
def rotate270(grid: _Grid) -> _Grid:
    _check_grid(grid, "rotate270")
    return np.rot90(grid, k=1).copy()


@primitive(
    name="mirror_horizontal",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Flip the grid left-to-right (mirror across the vertical axis).",
    examples=(("[[1,2],[3,4]]", "[[2,1],[4,3]]"),),
)
def mirror_horizontal(grid: _Grid) -> _Grid:
    _check_grid(grid, "mirror_horizontal")
    return np.fliplr(grid).copy()


@primitive(
    name="mirror_vertical",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Flip the grid top-to-bottom (mirror across the horizontal axis).",
    examples=(("[[1,2],[3,4]]", "[[3,4],[1,2]]"),),
)
def mirror_vertical(grid: _Grid) -> _Grid:
    _check_grid(grid, "mirror_vertical")
    return np.flipud(grid).copy()


@primitive(
    name="transpose",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Transpose: swap rows and columns (flip across main diagonal).",
    examples=(("[[1,2],[3,4]]", "[[1,3],[2,4]]"),),
)
def transpose(grid: _Grid) -> _Grid:
    _check_grid(grid, "transpose")
    return grid.T.copy()


@primitive(
    name="mirror_diagonal",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.2,
    description="Mirror across the main diagonal. Equivalent to transpose for square grids.",
    examples=(("[[1,2],[3,4]]", "[[1,3],[2,4]]"),),
)
def mirror_diagonal(grid: _Grid) -> _Grid:
    _check_grid(grid, "mirror_diagonal")
    return grid.T.copy()


@primitive(
    name="mirror_antidiagonal",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.2,
    description="Mirror across the anti-diagonal (top-right to bottom-left).",
    examples=(("[[1,2],[3,4]]", "[[4,2],[3,1]]"),),
)
def mirror_antidiagonal(grid: _Grid) -> _Grid:
    _check_grid(grid, "mirror_antidiagonal")
    # Flip both axes then transpose — equivalent to flipping across the
    # anti-diagonal. np.flip(np.flip(g, 0), 1).T == g[::-1, ::-1].T
    return grid[::-1, ::-1].T.copy()


# ---------------------------------------------------------------------------
# 10-15. Color
# ---------------------------------------------------------------------------


@primitive(
    name="recolor",
    signature=Signature(inputs=("Grid", "Color", "Color"), output="Grid"),
    cost=1.5,
    description="Replace every occurrence of color *src* with color *dst*.",
    examples=(("[[1,2],[1,3]] src=1 dst=4", "[[4,2],[4,3]]"),),
)
def recolor(grid: _Grid, src: int, dst: int) -> _Grid:
    _check_grid(grid, "recolor")
    _check_color(src, "recolor.src")
    _check_color(dst, "recolor.dst")
    out = grid.copy()
    out[out == src] = dst
    return out


@primitive(
    name="swap_colors",
    signature=Signature(inputs=("Grid", "Color", "Color"), output="Grid"),
    cost=1.5,
    description="Swap two colors throughout the grid.",
    examples=(("[[1,2],[2,1]] a=1 b=2", "[[2,1],[1,2]]"),),
)
def swap_colors(grid: _Grid, a: int, b: int) -> _Grid:
    _check_grid(grid, "swap_colors")
    _check_color(a, "swap_colors.a")
    _check_color(b, "swap_colors.b")
    out = grid.copy()
    mask_a = grid == a
    mask_b = grid == b
    out[mask_a] = b
    out[mask_b] = a
    return out


@primitive(
    name="most_common_color",
    signature=Signature(inputs=("Grid",), output="Color"),
    cost=1.0,
    description="Return the most-frequent color in the grid (ties broken by lowest index).",
    examples=(("[[1,2,2],[3,3,3]]", "3"),),
)
def most_common_color(grid: _Grid) -> int:
    _check_grid(grid, "most_common_color")
    counts = np.bincount(grid.ravel(), minlength=10)
    # argmax returns the lowest index on ties — stable for cache hashing.
    return int(np.argmax(counts))


@primitive(
    name="least_common_color",
    signature=Signature(inputs=("Grid",), output="Color"),
    cost=1.0,
    description=(
        "Return the least-frequent color present in the grid. "
        "Colors with zero occurrence are ignored; ties broken by lowest index."
    ),
    examples=(("[[1,2,2],[3,3,3]]", "1"),),
)
def least_common_color(grid: _Grid) -> int:
    _check_grid(grid, "least_common_color")
    counts = np.bincount(grid.ravel(), minlength=10)
    # Replace zero-counts with a sentinel so they're never picked.
    masked = np.where(counts == 0, np.iinfo(np.int64).max, counts)
    return int(np.argmin(masked))


@primitive(
    name="color_count",
    signature=Signature(inputs=("Grid",), output="Int"),
    cost=1.0,
    description="Number of distinct colors present in the grid (0..10).",
    examples=(("[[1,2,2],[3,3,3]]", "3"),),
)
def color_count(grid: _Grid) -> int:
    _check_grid(grid, "color_count")
    return int(np.unique(grid).size)


@primitive(
    name="replace_background",
    signature=Signature(inputs=("Grid", "Color"), output="Grid"),
    cost=1.5,
    description=(
        "Replace the background (most-common color) with the given color. "
        "Equivalent to ``recolor(grid, most_common_color(grid), new)``."
    ),
    examples=(("[[1,2,2],[3,3,3]] new=0", "[[1,2,2],[0,0,0]]"),),
)
def replace_background(grid: _Grid, new: int) -> _Grid:
    _check_grid(grid, "replace_background")
    _check_color(new, "replace_background.new")
    bg = most_common_color(grid)
    return recolor(grid, bg, new)
