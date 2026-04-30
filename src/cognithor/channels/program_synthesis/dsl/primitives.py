# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-DSL primitive implementations (spec §7.2).

Each primitive is a pure function returning a fresh array (no in-place
mutation, no shared state). All inputs are np.int8 grids in the
ARC-conventional value range 0..9.

Primitives are registered in the module-level :data:`REGISTRY` at import
time via the :func:`primitive` decorator. The catalog is the
single source of truth for both the search engine and the public DSL
reference.

This file holds the full Phase-1 base catalog (56 primitives):
geometric, color, size/scale, spatial, object-detection, mask/logic,
construction, and color constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet

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


def _check_int(n: object, name: str, *, min_val: int | None = None) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeMismatchError(f"{name}: expected int, got {type(n).__name__}")
    if min_val is not None and n < min_val:
        raise TypeMismatchError(f"{name}: {n} < {min_val}")
    return n


# ---------------------------------------------------------------------------
# 16-21. Size / Scale
# ---------------------------------------------------------------------------


@primitive(
    name="scale_up_2x",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Scale the grid up by 2× (each pixel becomes a 2×2 block).",
    examples=(("[[1,2]]", "[[1,1,2,2],[1,1,2,2]]"),),
)
def scale_up_2x(grid: _Grid) -> _Grid:
    _check_grid(grid, "scale_up_2x")
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1).copy()


@primitive(
    name="scale_up_3x",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Scale the grid up by 3× (each pixel becomes a 3×3 block).",
    examples=(("[[1]]", "[[1,1,1],[1,1,1],[1,1,1]]"),),
)
def scale_up_3x(grid: _Grid) -> _Grid:
    _check_grid(grid, "scale_up_3x")
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1).copy()


@primitive(
    name="scale_down_2x",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description=(
        "Scale the grid down by 2× by sampling the top-left pixel of each 2×2 block. "
        "Odd dimensions are truncated. Only valid for grids with shape ≥ 2×2."
    ),
    examples=(("[[1,1,2,2],[1,1,2,2]]", "[[1,2]]"),),
)
def scale_down_2x(grid: _Grid) -> _Grid:
    _check_grid(grid, "scale_down_2x")
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        raise TypeMismatchError(f"scale_down_2x: grid {grid.shape} too small to halve")
    return grid[::2, ::2].copy()


@primitive(
    name="tile_2x",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Tile the grid in a 2×2 pattern (output dimensions = input × 2).",
    examples=(("[[1,2]]", "[[1,2,1,2],[1,2,1,2]]"),),
)
def tile_2x(grid: _Grid) -> _Grid:
    _check_grid(grid, "tile_2x")
    return np.tile(grid, (2, 2)).copy()


@primitive(
    name="crop_bbox",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.5,
    description=(
        "Crop to the bounding box of all non-background pixels (background = "
        "most-common color). Returns a 1×1 grid containing the background color "
        "if the grid is uniformly background."
    ),
    examples=(("[[0,0,0],[0,5,0],[0,0,0]]", "[[5]]"),),
)
def crop_bbox(grid: _Grid) -> _Grid:
    _check_grid(grid, "crop_bbox")
    bg = most_common_color(grid)
    mask = grid != bg
    if not mask.any():
        # Uniformly background — return a single-cell grid with that color.
        return np.array([[bg]], dtype=np.int8)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return grid[r0:r1, c0:c1].copy()


@primitive(
    name="pad_with",
    signature=Signature(inputs=("Grid", "Color", "Int"), output="Grid"),
    cost=1.8,
    description=(
        "Pad the grid on all four sides with *width* pixels of *color*. Width must be ≥ 0."
    ),
    examples=(("[[1]] color=0 width=1", "[[0,0,0],[0,1,0],[0,0,0]]"),),
)
def pad_with(grid: _Grid, color: int, width: int) -> _Grid:
    _check_grid(grid, "pad_with")
    _check_color(color, "pad_with.color")
    _check_int(width, "pad_with.width", min_val=0)
    if width == 0:
        return grid.copy()
    return np.pad(grid, pad_width=width, mode="constant", constant_values=color).copy()


# ---------------------------------------------------------------------------
# 22-27. Spatial (gravity / shift)
# ---------------------------------------------------------------------------


def _gravity(grid: _Grid, axis: int, direction: int, name: str) -> _Grid:
    """Pull non-background pixels toward one edge along *axis*.

    ``direction`` = +1 means toward the high-index edge (down / right);
    ``direction`` = -1 means toward the low-index edge (up / left).
    Background = most-common color.
    """
    _check_grid(grid, name)
    bg = most_common_color(grid)
    out = np.full_like(grid, bg)
    if axis == 0:
        for c in range(grid.shape[1]):
            col = grid[:, c]
            non_bg = col[col != bg]
            if direction == 1:
                out[grid.shape[0] - len(non_bg) :, c] = non_bg
            else:
                out[: len(non_bg), c] = non_bg
    else:
        for r in range(grid.shape[0]):
            row = grid[r, :]
            non_bg = row[row != bg]
            if direction == 1:
                out[r, grid.shape[1] - len(non_bg) :] = non_bg
            else:
                out[r, : len(non_bg)] = non_bg
    return out


@primitive(
    name="gravity_down",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Pull all non-background pixels in each column toward the bottom edge.",
    examples=(("[[1,0],[0,2],[0,0]]", "[[0,0],[0,0],[1,2]]"),),
)
def gravity_down(grid: _Grid) -> _Grid:
    return _gravity(grid, axis=0, direction=1, name="gravity_down")


@primitive(
    name="gravity_up",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Pull all non-background pixels in each column toward the top edge.",
    examples=(("[[0,0],[1,0],[0,2]]", "[[1,2],[0,0],[0,0]]"),),
)
def gravity_up(grid: _Grid) -> _Grid:
    return _gravity(grid, axis=0, direction=-1, name="gravity_up")


@primitive(
    name="gravity_left",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Pull all non-background pixels in each row toward the left edge.",
    examples=(("[[0,1,0,2]]", "[[1,2,0,0]]"),),
)
def gravity_left(grid: _Grid) -> _Grid:
    return _gravity(grid, axis=1, direction=-1, name="gravity_left")


@primitive(
    name="gravity_right",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=2.0,
    description="Pull all non-background pixels in each row toward the right edge.",
    examples=(("[[1,0,2,0]]", "[[0,0,1,2]]"),),
)
def gravity_right(grid: _Grid) -> _Grid:
    return _gravity(grid, axis=1, direction=1, name="gravity_right")


@primitive(
    name="shift",
    signature=Signature(inputs=("Grid", "Int", "Int"), output="Grid"),
    cost=2.0,
    description=(
        "Shift the grid by (dy, dx). Pixels that fall off the edge are dropped, "
        "exposed cells are filled with the background (most-common color). "
        "Range is unrestricted; large shifts collapse the output to all-background."
    ),
    examples=(("[[1,2],[3,4]] dy=1 dx=0", "[[bg,bg],[1,2]]"),),
)
def shift(grid: _Grid, dy: int, dx: int) -> _Grid:
    _check_grid(grid, "shift")
    _check_int(dy, "shift.dy")
    _check_int(dx, "shift.dx")
    bg = most_common_color(grid)
    h, w = grid.shape
    out = np.full_like(grid, bg)

    # Source slice in the input, destination slice in the output.
    src_r0 = max(0, -dy)
    src_r1 = min(h, h - dy)
    dst_r0 = max(0, dy)
    dst_r1 = dst_r0 + (src_r1 - src_r0)

    src_c0 = max(0, -dx)
    src_c1 = min(w, w - dx)
    dst_c0 = max(0, dx)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return out


@primitive(
    name="wrap_shift",
    signature=Signature(inputs=("Grid", "Int", "Int"), output="Grid"),
    cost=2.2,
    description="Shift the grid by (dy, dx) with toroidal wrap-around (numpy.roll).",
    examples=(("[[1,2],[3,4]] dy=1 dx=0", "[[3,4],[1,2]]"),),
)
def wrap_shift(grid: _Grid, dy: int, dx: int) -> _Grid:
    _check_grid(grid, "wrap_shift")
    _check_int(dy, "wrap_shift.dy")
    _check_int(dx, "wrap_shift.dx")
    return np.roll(grid, shift=(dy, dx), axis=(0, 1)).copy()


def _check_object(o: object, name: str) -> Object:
    if not isinstance(o, Object):
        raise TypeMismatchError(f"{name}: expected Object, got {type(o).__name__}")
    return o


def _check_object_set(s: object, name: str) -> ObjectSet:
    if not isinstance(s, ObjectSet):
        raise TypeMismatchError(f"{name}: expected ObjectSet, got {type(s).__name__}")
    return s


# ---------------------------------------------------------------------------
# 28-29. Connected components
# ---------------------------------------------------------------------------


def _connected_components(grid: _Grid, connectivity: int) -> ObjectSet:
    """Compute connected components via flood-fill (raster-scan order).

    ``connectivity`` is 4 (orthogonal) or 8 (orthogonal + diagonal).
    Background pixels (most-common color) are *excluded* — they form no
    objects. Each non-background color produces its own component(s).

    No scipy dependency: a stack-based flood-fill keeps the implementation
    sandbox-friendly and avoids pulling in a heavy import for a single
    primitive group.
    """
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components: list[Object] = []

    offsets: tuple[tuple[int, int], ...]
    if connectivity == 4:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:  # 8
        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack: list[tuple[int, int]] = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if (
                    rr < 0
                    or rr >= h
                    or cc < 0
                    or cc >= w
                    or visited[rr, cc]
                    or int(grid[rr, cc]) != color
                ):
                    continue
                visited[rr, cc] = True
                cells.append((rr, cc))
                for dr, dc in offsets:
                    stack.append((rr + dr, cc + dc))
            components.append(Object(color=color, cells=tuple(cells)))

    return ObjectSet(objects=tuple(components))


@primitive(
    name="connected_components_4",
    signature=Signature(inputs=("Grid",), output="ObjectSet"),
    cost=2.5,
    description=(
        "4-connectivity flood-fill of all non-background pixels. "
        "Background = most-common color (excluded from output)."
    ),
    examples=(("[[0,1,0],[0,1,0]]", "ObjectSet([Object(color=1, size=2)])"),),
)
def connected_components_4(grid: _Grid) -> ObjectSet:
    _check_grid(grid, "connected_components_4")
    return _connected_components(grid, connectivity=4)


@primitive(
    name="connected_components_8",
    signature=Signature(inputs=("Grid",), output="ObjectSet"),
    cost=2.5,
    description=(
        "8-connectivity flood-fill of all non-background pixels. "
        "Diagonal neighbours count; otherwise identical to "
        "``connected_components_4``."
    ),
    examples=(("[[1,0],[0,1]]", "ObjectSet([Object(color=1, size=2)])"),),
)
def connected_components_8(grid: _Grid) -> ObjectSet:
    _check_grid(grid, "connected_components_8")
    return _connected_components(grid, connectivity=8)


# ---------------------------------------------------------------------------
# 30. objects_of_color
# ---------------------------------------------------------------------------


@primitive(
    name="objects_of_color",
    signature=Signature(inputs=("Grid", "Color"), output="ObjectSet"),
    cost=2.0,
    description=(
        "Return the 4-connected components whose color matches the argument. "
        "Treats the requested color as foreground regardless of background."
    ),
    examples=(("[[1,0,2],[1,0,0]] color=1", "ObjectSet of one 2-cell object"),),
)
def objects_of_color(grid: _Grid, color: int) -> ObjectSet:
    _check_grid(grid, "objects_of_color")
    _check_color(color, "objects_of_color.color")
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components: list[Object] = []
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if (
                    rr < 0
                    or rr >= h
                    or cc < 0
                    or cc >= w
                    or visited[rr, cc]
                    or int(grid[rr, cc]) != color
                ):
                    continue
                visited[rr, cc] = True
                cells.append((rr, cc))
                for dr, dc in offsets:
                    stack.append((rr + dr, cc + dc))
            components.append(Object(color=color, cells=tuple(cells)))
    return ObjectSet(objects=tuple(components))


# ---------------------------------------------------------------------------
# 31-32. largest / smallest
# ---------------------------------------------------------------------------


@primitive(
    name="largest_object",
    signature=Signature(inputs=("ObjectSet",), output="Object"),
    cost=1.5,
    description=(
        "Object with the largest pixel count in the set. "
        "Ties broken by discovery order (first occurrence wins)."
    ),
    examples=(("ObjectSet of [{size=2}, {size=5}]", "Object size=5"),),
)
def largest_object(objects: ObjectSet) -> Object:
    _check_object_set(objects, "largest_object")
    if objects.is_empty():
        raise TypeMismatchError("largest_object: empty ObjectSet")
    best = objects.objects[0]
    for o in objects.objects[1:]:
        if o.size > best.size:
            best = o
    return best


@primitive(
    name="smallest_object",
    signature=Signature(inputs=("ObjectSet",), output="Object"),
    cost=1.5,
    description=(
        "Object with the smallest pixel count in the set. "
        "Ties broken by discovery order (first occurrence wins)."
    ),
    examples=(("ObjectSet of [{size=2}, {size=5}]", "Object size=2"),),
)
def smallest_object(objects: ObjectSet) -> Object:
    _check_object_set(objects, "smallest_object")
    if objects.is_empty():
        raise TypeMismatchError("smallest_object: empty ObjectSet")
    best = objects.objects[0]
    for o in objects.objects[1:]:
        if o.size < best.size:
            best = o
    return best


# ---------------------------------------------------------------------------
# 33. bounding_box
# ---------------------------------------------------------------------------


@primitive(
    name="bounding_box",
    signature=Signature(inputs=("Object",), output="Grid"),
    cost=1.5,
    description=(
        "Render the object as a tight grid of size = bbox dimensions. "
        "Pixels inside the object get its color, pixels outside get 0."
    ),
    examples=(("Object(color=5, cells=[(0,0),(0,1),(1,1)])", "[[5,5],[0,5]]"),),
)
def bounding_box(obj: Object) -> _Grid:
    _check_object(obj, "bounding_box")
    if not obj.cells:
        return np.array([[0]], dtype=np.int8)
    r0, r1, c0, c1 = obj.bbox
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.int8)
    for r, c in obj.cells:
        out[r - r0, c - c0] = obj.color
    return out


# ---------------------------------------------------------------------------
# 34. object_count
# ---------------------------------------------------------------------------


@primitive(
    name="object_count",
    signature=Signature(inputs=("ObjectSet",), output="Int"),
    cost=1.0,
    description="Number of objects in the set (≥ 0).",
    examples=(("ObjectSet of 3", "3"),),
)
def object_count(objects: ObjectSet) -> int:
    _check_object_set(objects, "object_count")
    return len(objects)


# ---------------------------------------------------------------------------
# 35. render_objects
# ---------------------------------------------------------------------------


@primitive(
    name="render_objects",
    signature=Signature(inputs=("ObjectSet", "Grid"), output="Grid"),
    cost=2.0,
    description=(
        "Paint every object in the set onto a copy of *base*. "
        "Cells outside the grid are silently dropped (clip-to-edge). "
        "Later objects overwrite earlier ones at overlapping cells."
    ),
    examples=(("ObjectSet of one (color=2)] onto [[0,0],[0,0]]", "[[2,0],[0,0]]"),),
)
def render_objects(objects: ObjectSet, base: _Grid) -> _Grid:
    _check_object_set(objects, "render_objects")
    _check_grid(base, "render_objects.base")
    out = base.copy()
    h, w = out.shape
    for obj in objects.objects:
        for r, c in obj.cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = obj.color
    return out


_Mask = NDArray[np.bool_]


def _check_mask(m: object, name: str) -> _Mask:
    if not isinstance(m, np.ndarray):
        raise TypeMismatchError(f"{name}: expected ndarray, got {type(m).__name__}")
    if m.ndim != 2:
        raise TypeMismatchError(f"{name}: expected 2-D mask, got {m.ndim}-D")
    if m.dtype != np.bool_:
        raise TypeMismatchError(f"{name}: expected bool dtype, got {m.dtype}")
    return m


# ---------------------------------------------------------------------------
# 36-42. Mask / Logic
# ---------------------------------------------------------------------------


@primitive(
    name="mask_eq",
    signature=Signature(inputs=("Grid", "Color"), output="Mask"),
    cost=1.5,
    description="Return a boolean mask: True where the grid equals *color*.",
    examples=(("[[1,2],[1,3]] color=1", "[[True,False],[True,False]]"),),
)
def mask_eq(grid: _Grid, color: int) -> _Mask:
    _check_grid(grid, "mask_eq")
    _check_color(color, "mask_eq.color")
    return cast("_Mask", (grid == color).copy())


@primitive(
    name="mask_ne",
    signature=Signature(inputs=("Grid", "Color"), output="Mask"),
    cost=1.5,
    description="Return a boolean mask: True where the grid is *not* color.",
    examples=(("[[1,2],[1,3]] color=1", "[[False,True],[False,True]]"),),
)
def mask_ne(grid: _Grid, color: int) -> _Mask:
    _check_grid(grid, "mask_ne")
    _check_color(color, "mask_ne.color")
    return cast("_Mask", (grid != color).copy())


@primitive(
    name="mask_apply",
    signature=Signature(inputs=("Grid", "Mask", "Color"), output="Grid"),
    cost=2.0,
    description=(
        "Set every cell of the grid where *mask* is True to *color*. "
        "Mask shape must match the grid shape exactly."
    ),
    examples=(("[[1,2]] mask=[[T,F]] color=9", "[[9,2]]"),),
)
def mask_apply(grid: _Grid, mask: _Mask, color: int) -> _Grid:
    _check_grid(grid, "mask_apply")
    _check_mask(mask, "mask_apply.mask")
    _check_color(color, "mask_apply.color")
    if mask.shape != grid.shape:
        raise TypeMismatchError(f"mask_apply: mask shape {mask.shape} != grid shape {grid.shape}")
    out = grid.copy()
    out[mask] = color
    return out


def _check_same_shape(a: _Mask, b: _Mask, name: str) -> None:
    if a.shape != b.shape:
        raise TypeMismatchError(f"{name}: shape mismatch {a.shape} vs {b.shape}")


@primitive(
    name="mask_and",
    signature=Signature(inputs=("Mask", "Mask"), output="Mask"),
    cost=1.5,
    description="Pixel-wise logical AND of two masks of equal shape.",
    examples=(("[[T,F]] AND [[T,T]]", "[[T,F]]"),),
)
def mask_and(a: _Mask, b: _Mask) -> _Mask:
    _check_mask(a, "mask_and.a")
    _check_mask(b, "mask_and.b")
    _check_same_shape(a, b, "mask_and")
    return np.logical_and(a, b).copy()


@primitive(
    name="mask_or",
    signature=Signature(inputs=("Mask", "Mask"), output="Mask"),
    cost=1.5,
    description="Pixel-wise logical OR of two masks of equal shape.",
    examples=(("[[T,F]] OR [[F,T]]", "[[T,T]]"),),
)
def mask_or(a: _Mask, b: _Mask) -> _Mask:
    _check_mask(a, "mask_or.a")
    _check_mask(b, "mask_or.b")
    _check_same_shape(a, b, "mask_or")
    return np.logical_or(a, b).copy()


@primitive(
    name="mask_xor",
    signature=Signature(inputs=("Mask", "Mask"), output="Mask"),
    cost=1.5,
    description="Pixel-wise logical XOR of two masks of equal shape.",
    examples=(("[[T,F]] XOR [[T,T]]", "[[F,T]]"),),
)
def mask_xor(a: _Mask, b: _Mask) -> _Mask:
    _check_mask(a, "mask_xor.a")
    _check_mask(b, "mask_xor.b")
    _check_same_shape(a, b, "mask_xor")
    return np.logical_xor(a, b).copy()


@primitive(
    name="mask_not",
    signature=Signature(inputs=("Mask",), output="Mask"),
    cost=1.2,
    description="Pixel-wise logical NOT (involution: mask_not(mask_not(x)) == x).",
    examples=(("[[T,F]]", "[[F,T]]"),),
)
def mask_not(a: _Mask) -> _Mask:
    _check_mask(a, "mask_not")
    return np.logical_not(a).copy()


# ---------------------------------------------------------------------------
# 43-46. Construction / Composition
# ---------------------------------------------------------------------------


@primitive(
    name="stack_horizontal",
    signature=Signature(inputs=("Grid", "Grid"), output="Grid"),
    cost=2.0,
    description=(
        "Stack two grids side-by-side (left-to-right). "
        "Row counts must match; output cols = left.cols + right.cols."
    ),
    examples=(("[[1]]||[[2]]", "[[1,2]]"),),
)
def stack_horizontal(left: _Grid, right: _Grid) -> _Grid:
    _check_grid(left, "stack_horizontal.left")
    _check_grid(right, "stack_horizontal.right")
    if left.shape[0] != right.shape[0]:
        raise TypeMismatchError(
            f"stack_horizontal: row mismatch {left.shape[0]} vs {right.shape[0]}"
        )
    return np.concatenate([left, right], axis=1).copy()


@primitive(
    name="stack_vertical",
    signature=Signature(inputs=("Grid", "Grid"), output="Grid"),
    cost=2.0,
    description=(
        "Stack two grids top-to-bottom. "
        "Column counts must match; output rows = top.rows + bottom.rows."
    ),
    examples=(("[[1,2]]==[[3,4]]", "[[1,2],[3,4]]"),),
)
def stack_vertical(top: _Grid, bottom: _Grid) -> _Grid:
    _check_grid(top, "stack_vertical.top")
    _check_grid(bottom, "stack_vertical.bottom")
    if top.shape[1] != bottom.shape[1]:
        raise TypeMismatchError(f"stack_vertical: col mismatch {top.shape[1]} vs {bottom.shape[1]}")
    return np.concatenate([top, bottom], axis=0).copy()


@primitive(
    name="overlay",
    signature=Signature(inputs=("Grid", "Grid", "Color"), output="Grid"),
    cost=2.5,
    description=(
        "Overlay *top* onto *base*: cells of *top* equal to *transparent_color* "
        "are skipped, all other cells overwrite *base*. Both grids must have "
        "the same shape."
    ),
    examples=(("base=[[1,1]] top=[[0,2]] transparent=0", "[[1,2]]"),),
)
def overlay(base: _Grid, top: _Grid, transparent_color: int) -> _Grid:
    _check_grid(base, "overlay.base")
    _check_grid(top, "overlay.top")
    _check_color(transparent_color, "overlay.transparent_color")
    if base.shape != top.shape:
        raise TypeMismatchError(f"overlay: shape mismatch {base.shape} vs {top.shape}")
    out = base.copy()
    mask = top != transparent_color
    out[mask] = top[mask]
    return out


@primitive(
    name="frame",
    signature=Signature(inputs=("Grid", "Color"), output="Grid"),
    cost=1.8,
    description=(
        "Draw a 1-pixel border of *color* around the grid edge, "
        "leaving the interior unchanged. Grid must be at least 1×1."
    ),
    examples=(("[[1,2],[3,4]] color=0", "[[0,0],[0,0]] (2x2 fully framed → all border)"),),
)
def frame(grid: _Grid, color: int) -> _Grid:
    _check_grid(grid, "frame")
    _check_color(color, "frame.color")
    out = grid.copy()
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    return out


# ---------------------------------------------------------------------------
# 47-56. Color constants
# ---------------------------------------------------------------------------


def _make_color_const(c: int) -> None:
    """Register a zero-arity primitive that returns the literal color *c*.

    Each is its own primitive in the catalog so the search engine can
    enumerate them as leaves; the cost is intentionally low (0.5) so they
    don't dominate Occam ranking.
    """

    @primitive(
        name=f"const_color_{c}",
        signature=Signature(inputs=(), output="Color"),
        cost=0.5,
        description=f"Constant color {c}.",
        examples=(("(no input)", str(c)),),
    )
    def _const() -> int:
        return c

    # Suppress "function defined but never used" — Python keeps the
    # registry reference alive via the decorator.
    _ = _const


for _c in range(10):
    _make_color_const(_c)


# ---------------------------------------------------------------------------
# Phase 1.5: Higher-order primitives (H1, H2). H3-H5 follow in subsequent PRs.
# ---------------------------------------------------------------------------


from cognithor.channels.program_synthesis.dsl.lambdas import (
    Lambda,
    evaluate_lambda,
)
from cognithor.channels.program_synthesis.dsl.predicates import (
    Predicate,
    PredicateContext,
    evaluate_predicate,
)


def _check_lambda(fn: object, name: str) -> Lambda:
    if not isinstance(fn, Lambda):
        raise TypeMismatchError(f"{name}: expected Lambda, got {type(fn).__name__}")
    return fn


def _check_predicate(p: object, name: str) -> Predicate:
    if not isinstance(p, Predicate):
        raise TypeMismatchError(f"{name}: expected Predicate, got {type(p).__name__}")
    return p


@primitive(
    name="map_objects",
    signature=Signature(inputs=("ObjectSet", "Lambda"), output="ObjectSet"),
    cost=3.0,
    description=(
        "Apply *fn* to every object in the set; return the resulting "
        "ObjectSet in the same order. Pure, no in-place mutation."
    ),
    examples=(("ObjectSet of 3, recolor_lambda(5)", "ObjectSet of 3, all color=5"),),
)
def map_objects(objects: ObjectSet, fn: Lambda) -> ObjectSet:
    _check_object_set(objects, "map_objects")
    _check_lambda(fn, "map_objects.fn")
    transformed = tuple(evaluate_lambda(fn, o) for o in objects.objects)
    return ObjectSet(objects=transformed)


@primitive(
    name="filter_objects",
    signature=Signature(inputs=("ObjectSet", "Predicate"), output="ObjectSet"),
    cost=2.5,
    description=(
        "Keep only objects for which *pred* is True. The predicate's "
        "is_largest_in / is_smallest_in receive the original ObjectSet "
        "as context so 'largest' refers to the input set, not the "
        "filtered output."
    ),
    examples=(("ObjectSet of 3 (colors 1,2,3), color_eq(2)", "ObjectSet of 1 (color=2)"),),
)
def filter_objects(objects: ObjectSet, pred: Predicate) -> ObjectSet:
    _check_object_set(objects, "filter_objects")
    _check_predicate(pred, "filter_objects.pred")
    ctx = PredicateContext(object_set=objects)
    kept = tuple(o for o in objects.objects if evaluate_predicate(pred, o, ctx))
    return ObjectSet(objects=kept)


# ---------------------------------------------------------------------------
# Phase 1.5: H3 align_to + AlignMode enum
# ---------------------------------------------------------------------------


from enum import Enum


class AlignMode(str, Enum):
    """Where to anchor object A relative to object B's bounding box."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


def _check_align_mode(m: object, name: str) -> AlignMode:
    if isinstance(m, AlignMode):
        return m
    if isinstance(m, str):
        try:
            return AlignMode(m)
        except ValueError as exc:
            raise TypeMismatchError(
                f"{name}: unknown AlignMode {m!r}; allowed: {[mode.value for mode in AlignMode]}"
            ) from exc
    raise TypeMismatchError(f"{name}: expected AlignMode or str, got {type(m).__name__}")


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Integer center of a half-open bbox (r0, r1, c0, c1).

    Uses floor-division so the result is always a valid cell position.
    """
    r0, r1, c0, c1 = bbox
    return ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)


def _align_delta(
    a_bbox: tuple[int, int, int, int],
    b_bbox: tuple[int, int, int, int],
    mode: AlignMode,
) -> tuple[int, int]:
    """How much to shift A so its bbox aligns with B's per *mode*."""
    ar0, ar1, ac0, ac1 = a_bbox
    br0, br1, bc0, bc1 = b_bbox
    a_cy, a_cx = _bbox_center(a_bbox)
    b_cy, b_cx = _bbox_center(b_bbox)

    # Default: centre-aligned on both axes.
    dy = b_cy - a_cy
    dx = b_cx - a_cx

    if mode in (AlignMode.LEFT, AlignMode.TOP_LEFT, AlignMode.BOTTOM_LEFT):
        dx = bc0 - ac0
    if mode in (AlignMode.RIGHT, AlignMode.TOP_RIGHT, AlignMode.BOTTOM_RIGHT):
        # bbox is half-open; right edge cell is r1-1.
        dx = (bc1 - 1) - (ac1 - 1)
    if mode in (AlignMode.TOP, AlignMode.TOP_LEFT, AlignMode.TOP_RIGHT):
        dy = br0 - ar0
    if mode in (AlignMode.BOTTOM, AlignMode.BOTTOM_LEFT, AlignMode.BOTTOM_RIGHT):
        dy = (br1 - 1) - (ar1 - 1)
    return (dy, dx)


@primitive(
    name="align_to",
    signature=Signature(inputs=("Object", "Object", "AlignMode"), output="Object"),
    cost=3.0,
    description=(
        "Translate object A so its bounding box aligns with B's per *mode*. "
        "CENTER aligns both axes; the four edges align that axis and "
        "centre the other; corners align both axes simultaneously."
    ),
    examples=(("a, b, CENTER", "a translated so its centre matches b's centre"),),
)
def align_to(a: Object, b: Object, mode: AlignMode) -> Object:
    _check_object(a, "align_to.a")
    _check_object(b, "align_to.b")
    m = _check_align_mode(mode, "align_to.mode")
    if not a.cells or not b.cells:
        return a
    dy, dx = _align_delta(a.bbox, b.bbox, m)
    if dy == 0 and dx == 0:
        return a
    return Object(
        color=a.color,
        cells=tuple((r + dy, c + dx) for r, c in a.cells),
    )


# ---------------------------------------------------------------------------
# Phase 1.5: H4 sort_objects + SortKey enum
# ---------------------------------------------------------------------------


class SortKey(str, Enum):
    """Enumerated sort keys for ``sort_objects`` (spec §7.5)."""

    SIZE_ASC = "size_asc"
    SIZE_DESC = "size_desc"
    COLOR_ASC = "color_asc"
    COLOR_DESC = "color_desc"
    POSITION_ROW = "position_row"
    POSITION_COL = "position_col"
    DISTANCE_FROM_CENTER = "distance_from_center"


def _check_sort_key(k: object, name: str) -> SortKey:
    if isinstance(k, SortKey):
        return k
    if isinstance(k, str):
        try:
            return SortKey(k)
        except ValueError as exc:
            raise TypeMismatchError(
                f"{name}: unknown SortKey {k!r}; allowed: {[mode.value for mode in SortKey]}"
            ) from exc
    raise TypeMismatchError(f"{name}: expected SortKey or str, got {type(k).__name__}")


def _object_top_left(obj: Object) -> tuple[int, int]:
    if not obj.cells:
        return (0, 0)
    r0, _, c0, _ = obj.bbox
    return (r0, c0)


def _grid_center_distance_squared(obj: Object) -> int:
    """Squared Euclidean distance from object's bbox-centre to (0, 0).

    Square distance is monotonic in the actual distance and avoids
    floating-point — important for cache-stable ordering across
    machines.
    """
    if not obj.cells:
        return 0
    r0, r1, c0, c1 = obj.bbox
    cy = (r0 + r1 - 1) // 2
    cx = (c0 + c1 - 1) // 2
    return cy * cy + cx * cx


def _sort_keyfn(key: SortKey) -> Callable[[tuple[int, Object]], object]:
    """Return a key-function for ``sorted(..., key=...)`` over (idx, obj) tuples.

    Ties always break by the discovery index ``idx`` so the output is
    reproducible across runs (cache-stable).
    """
    if key == SortKey.SIZE_ASC:
        return lambda io: (io[1].size, io[0])
    if key == SortKey.SIZE_DESC:
        return lambda io: (-io[1].size, io[0])
    if key == SortKey.COLOR_ASC:
        return lambda io: (io[1].color, io[0])
    if key == SortKey.COLOR_DESC:
        return lambda io: (-io[1].color, io[0])
    if key == SortKey.POSITION_ROW:
        return lambda io: (_object_top_left(io[1]), io[0])
    if key == SortKey.POSITION_COL:
        return lambda io: (
            (_object_top_left(io[1])[1], _object_top_left(io[1])[0]),
            io[0],
        )
    if key == SortKey.DISTANCE_FROM_CENTER:
        return lambda io: (_grid_center_distance_squared(io[1]), io[0])
    raise ValueError(f"_sort_keyfn: unhandled SortKey {key!r}")


@primitive(
    name="sort_objects",
    signature=Signature(inputs=("ObjectSet", "SortKey"), output="ObjectSet"),
    cost=2.5,
    description=(
        "Stable-sort the set by *key*. Ties break by discovery order so "
        "the result is reproducible across runs (cache-stable)."
    ),
    examples=(("ObjectSet of 3 (sizes 1,3,2), SIZE_ASC", "Order: size 1, 2, 3"),),
)
def sort_objects(objects: ObjectSet, key: SortKey) -> ObjectSet:
    _check_object_set(objects, "sort_objects")
    k = _check_sort_key(key, "sort_objects.key")
    if len(objects) <= 1:
        return objects
    indexed = list(enumerate(objects.objects))
    indexed.sort(key=cast("Callable[[tuple[int, Object]], Any]", _sort_keyfn(k)))
    return ObjectSet(objects=tuple(o for _, o in indexed))


# ---------------------------------------------------------------------------
# Phase 1.5: H5 branch — conditional-Lambda combinator (spec v1.2 §7.5)
# ---------------------------------------------------------------------------


from cognithor.channels.program_synthesis.dsl.lambdas import (
    branch_lambda as _branch_lambda,
)


@primitive(
    name="branch",
    signature=Signature(inputs=("Predicate", "Lambda", "Lambda"), output="Lambda"),
    cost=3.5,
    description=(
        "Build a conditional Lambda: ``λobj. then_fn(obj) if pred(obj) "
        "else else_fn(obj)``. Sub-tiefe ≤ 1 — nested ``branch`` "
        "forbidden in Phase 1 (spec §7.5)."
    ),
    examples=(
        (
            "branch(size_gt(5), recolor_lambda(2), identity_lambda())",
            "Lambda that recolours large objects red, leaves others alone.",
        ),
    ),
)
def branch(predicate: Predicate, then_fn: Lambda, else_fn: Lambda) -> Lambda:
    _check_predicate(predicate, "branch.predicate")
    _check_lambda(then_fn, "branch.then_fn")
    _check_lambda(else_fn, "branch.else_fn")
    if then_fn.constructor == "branch_lambda" or else_fn.constructor == "branch_lambda":
        raise TypeMismatchError("branch: nested branches forbidden in Phase 1 (sub-tiefe limit 1)")
    return _branch_lambda(predicate, then_fn, else_fn)
