"""Image V2 primitive catalog (Sprint-26.4).

12+ pixel-grid primitives targeting ARC-AGI training tasks the
Sprint-10 catalog couldn't solve. Grids are tuples-of-tuples-of-int
so they hash into Sprint-22's MCTS cache cleanly.

Categories:

* **Symmetry**: ``mirror_h``, ``mirror_v``, ``rotate_180``, ``transpose``,
  ``complete_symmetric_h``
* **Anchors**: ``find_anchor``, ``align_to_anchor``
* **Conditional fill**: ``fill_if_color``, ``flood_fill_protected``
* **Pattern**: ``find_period_h``, ``tile_pattern_h``,
  ``self_tile_by_mask``
* **Object**: ``connected_components``, ``bounding_box``

The catalog is intentionally small + documented; primitive-count
floor for Sprint-26.4 was "12+", we ship 14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# A pixel-grid is a tuple of row tuples for hashability.
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ImageV2Primitive:
    name: str
    fn: Callable[..., Any]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid Image-v2 primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"Image-v2 primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class ImageV2Catalog:
    def __init__(self) -> None:
        self._entries: dict[str, ImageV2Primitive] = {}

    def add(self, primitive: ImageV2Primitive) -> None:
        if primitive.name in self._entries:
            msg = f"Image-v2 primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> ImageV2Primitive:
        if name not in self._entries:
            msg = f"Unknown Image-v2 primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


IMAGE_V2_PRIMITIVE_NAMES: tuple[str, ...] = (
    # Symmetry
    "mirror_h",
    "mirror_v",
    "rotate_180",
    "transpose",
    "complete_symmetric_h",
    # Anchors
    "find_anchor",
    "align_to_anchor",
    # Conditional fill
    "fill_if_color",
    "flood_fill_protected",
    # Pattern
    "find_period_h",
    "tile_pattern_h",
    "self_tile_by_mask",
    # Object
    "connected_components",
    "bounding_box",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tuple_grid(grid: Any) -> Grid:
    if not grid:
        return ()
    return tuple(tuple(int(c) for c in row) for row in grid)


def _grid_dims(grid: Grid) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def _mirror_h(grid: Any) -> Grid:
    g = _to_tuple_grid(grid)
    return tuple(row[::-1] for row in g)


def _mirror_v(grid: Any) -> Grid:
    g = _to_tuple_grid(grid)
    return tuple(reversed(g))


def _rotate_180(grid: Any) -> Grid:
    g = _to_tuple_grid(grid)
    return tuple(row[::-1] for row in reversed(g))


def _transpose(grid: Any) -> Grid:
    g = _to_tuple_grid(grid)
    if not g:
        return ()
    rows, cols = _grid_dims(g)
    return tuple(tuple(g[r][c] for r in range(rows)) for c in range(cols))


def _complete_symmetric_h(grid: Any, *, fill_color: int = 0) -> Grid:
    """Fill in cells so the result is horizontal-mirror symmetric.

    Cells equal to ``fill_color`` are treated as "missing" and replaced
    by their mirror partner when the mirror has a non-fill value.
    """
    g = _to_tuple_grid(grid)
    if not g:
        return ()
    rows, cols = _grid_dims(g)
    out: list[list[int]] = [list(row) for row in g]
    for r in range(rows):
        for c in range(cols):
            mirror_c = cols - 1 - c
            if out[r][c] == fill_color and out[r][mirror_c] != fill_color:
                out[r][c] = out[r][mirror_c]
    return tuple(tuple(row) for row in out)


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def _find_anchor(grid: Any, color: int) -> tuple[int, int] | None:
    """Top-left coordinate of the first cell matching ``color``."""
    g = _to_tuple_grid(grid)
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == color:
                return r, c
    return None


def _align_to_anchor(
    grid: Any,
    anchor: tuple[int, int],
    *,
    target: tuple[int, int] = (0, 0),
    fill: int = 0,
) -> Grid:
    """Translate ``grid`` so ``anchor`` lands on ``target``.

    Cells shifted out of bounds are dropped; new cells revealed at
    the edges are filled with ``fill``.
    """
    g = _to_tuple_grid(grid)
    if not g:
        return ()
    rows, cols = _grid_dims(g)
    dr = target[0] - anchor[0]
    dc = target[1] - anchor[1]
    out = [[fill] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            new_r = r + dr
            new_c = c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                out[new_r][new_c] = g[r][c]
    return tuple(tuple(row) for row in out)


# ---------------------------------------------------------------------------
# Conditional fill
# ---------------------------------------------------------------------------


def _fill_if_color(grid: Any, *, target: int, replacement: int) -> Grid:
    g = _to_tuple_grid(grid)
    return tuple(tuple(replacement if v == target else v for v in row) for row in g)


def _flood_fill_protected(
    grid: Any,
    start: tuple[int, int],
    *,
    new_color: int,
    barrier: int,
) -> Grid:
    """4-connectivity flood from ``start``, stopping at ``barrier`` cells."""
    g = [list(row) for row in _to_tuple_grid(grid)]
    if not g:
        return ()
    rows = len(g)
    cols = len(g[0])
    sr, sc = start
    if not (0 <= sr < rows and 0 <= sc < cols):
        return tuple(tuple(row) for row in g)
    origin = g[sr][sc]
    if origin in (barrier, new_color):
        return tuple(tuple(row) for row in g)
    stack: list[tuple[int, int]] = [start]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if g[r][c] in (new_color, barrier):
            continue
        if g[r][c] != origin:
            continue
        g[r][c] = new_color
        stack.extend(((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))
    return tuple(tuple(row) for row in g)


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


def _find_period_h(grid: Any) -> int:
    """Smallest horizontal period that explains every row.

    Returns the grid width when no smaller period is detected.
    """
    g = _to_tuple_grid(grid)
    if not g:
        return 0
    rows, cols = _grid_dims(g)
    for period in range(1, cols + 1):
        if cols % period != 0:
            continue
        if all(g[r][c] == g[r][c % period] for r in range(rows) for c in range(cols)):
            return period
    return cols


def _tile_pattern_h(grid: Any, *, target_cols: int) -> Grid:
    """Tile each row horizontally so width = ``target_cols``.

    Rows shorter than ``target_cols`` get repeated until they reach the
    target. Existing rows already at ``target_cols`` are passed through.
    """
    g = _to_tuple_grid(grid)
    out: list[tuple[int, ...]] = []
    for row in g:
        if not row:
            out.append(tuple([0] * target_cols))
            continue
        repeats = (target_cols + len(row) - 1) // len(row)
        tiled = (row * repeats)[:target_cols]
        out.append(tiled)
    return tuple(out)


def _self_tile_by_mask(
    grid: Any,
    mask: Any,
    *,
    background: int = 0,
) -> Grid:
    """Use ``grid`` as the tile, ``mask`` as the layout (cell ≠ background ⇒
    place a copy of ``grid`` there).

    Output dimensions = mask_rows × tile_rows by mask_cols × tile_cols.
    """
    tile = _to_tuple_grid(grid)
    layout = _to_tuple_grid(mask)
    if not tile or not layout:
        return ()
    t_rows, t_cols = _grid_dims(tile)
    m_rows, m_cols = _grid_dims(layout)
    out_rows = m_rows * t_rows
    out_cols = m_cols * t_cols
    out = [[background] * out_cols for _ in range(out_rows)]
    for mr in range(m_rows):
        for mc in range(m_cols):
            if layout[mr][mc] == background:
                continue
            for tr in range(t_rows):
                for tc in range(t_cols):
                    out[mr * t_rows + tr][mc * t_cols + tc] = tile[tr][tc]
    return tuple(tuple(row) for row in out)


# ---------------------------------------------------------------------------
# Object
# ---------------------------------------------------------------------------


def _connected_components(grid: Any, *, background: int = 0) -> list[list[tuple[int, int]]]:
    """Return components of non-background cells via 4-connectivity."""
    g = _to_tuple_grid(grid)
    if not g:
        return []
    rows, cols = _grid_dims(g)
    seen = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if seen[r][c] or g[r][c] == background:
                continue
            origin = g[r][c]
            cells: list[tuple[int, int]] = []
            stack = [(r, c)]
            while stack:
                sr, sc = stack.pop()
                if not (0 <= sr < rows and 0 <= sc < cols):
                    continue
                if seen[sr][sc] or g[sr][sc] != origin:
                    continue
                seen[sr][sc] = True
                cells.append((sr, sc))
                stack.extend(((sr + 1, sc), (sr - 1, sc), (sr, sc + 1), (sr, sc - 1)))
            components.append(cells)
    return components


def _bounding_box(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_row, min_col, max_row, max_col) of a cell list."""
    if not cells:
        return 0, 0, -1, -1
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_image_v2_catalog() -> ImageV2Catalog:
    cat = ImageV2Catalog()

    def add(name: str, fn: Callable[..., Any], cost: float, desc: str) -> None:
        cat.add(ImageV2Primitive(name=name, fn=fn, cost=cost, description=desc))

    # Symmetry
    add("mirror_h", _mirror_h, 0.2, "Horizontal mirror")
    add("mirror_v", _mirror_v, 0.2, "Vertical mirror")
    add("rotate_180", _rotate_180, 0.2, "180° rotation")
    add("transpose", _transpose, 0.3, "Matrix transpose")
    add(
        "complete_symmetric_h",
        _complete_symmetric_h,
        0.5,
        "Fill missing cells using horizontal mirror partner",
    )

    # Anchors
    add("find_anchor", _find_anchor, 0.2, "Top-left of first cell with given color")
    add(
        "align_to_anchor",
        _align_to_anchor,
        0.4,
        "Translate grid so anchor lands on target coord",
    )

    # Conditional fill
    add("fill_if_color", _fill_if_color, 0.3, "Replace target color with new")
    add(
        "flood_fill_protected",
        _flood_fill_protected,
        0.5,
        "4-connectivity flood, stopping at barrier",
    )

    # Pattern
    add("find_period_h", _find_period_h, 0.5, "Smallest horizontal period")
    add("tile_pattern_h", _tile_pattern_h, 0.4, "Tile rows horizontally to width")
    add(
        "self_tile_by_mask",
        _self_tile_by_mask,
        0.7,
        "Use grid as tile, mask as layout",
    )

    # Object
    add(
        "connected_components",
        _connected_components,
        0.5,
        "4-connectivity components excluding background",
    )
    add("bounding_box", _bounding_box, 0.2, "(min_r, min_c, max_r, max_c)")

    return cat
