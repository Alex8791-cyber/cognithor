"""Visual State Encoder — converts 64×64 color-index grids into compact LLM descriptions."""

from __future__ import annotations

__all__ = ["VisualStateEncoder"]

import numpy as np


class VisualStateEncoder:
    """Encode ARC-AGI grid states as compact text for LLM context."""

    def __init__(self) -> None:
        self.color_names: dict[int, str] = {
            0: "schwarz",
            1: "blau",
            2: "rot",
            3: "gruen",
            4: "gelb",
            5: "grau",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "braun",
            10: "weiss",
            11: "hellblau",
            12: "dunkelgruen",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_for_llm(self, grid: np.ndarray, diff: np.ndarray | None = None) -> str:
        """Return a human-readable German description of *grid* for LLM context.

        Parameters
        ----------
        grid:
            2-D (or 1×H×W) int8 array of color indices 0-12.
        diff:
            Optional boolean mask of the same spatial shape indicating changed pixels.
        """
        grid_2d = self._to_2d(grid)
        lines: list[str] = []

        # 1. Color histogram
        lines.append("Farbverteilung:")
        total_pixels = grid_2d.size
        unique, counts = np.unique(grid_2d, return_counts=True)
        order = np.argsort(counts)[::-1]
        for _rank, idx in enumerate(order[:5]):
            color_idx = int(unique[idx])
            pct = counts[idx] / total_pixels * 100
            name = self.color_names.get(color_idx, str(color_idx))
            lines.append(f"  {name} ({pct:.1f}%)")

        # 2. Region detection
        boxes = self._find_bounding_boxes(grid_2d, background=0, min_size=4)
        # Group by color; keep at most 5 distinct non-background colors
        seen_colors: list[int] = []
        color_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        for color, x1, y1, x2, y2 in boxes:
            if color not in seen_colors:
                if len(seen_colors) >= 5:
                    break
                seen_colors.append(color)
                color_boxes[color] = []
            color_boxes[color].append((x1, y1, x2, y2))

        if color_boxes:
            lines.append("Regionen:")
            for color in seen_colors:
                name = self.color_names.get(color, str(color))
                for x1, y1, x2, y2 in color_boxes[color]:
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    lines.append(f"  {name}: ({x1},{y1})-({x2},{y2}) {w}x{h}")

        # 3. Diff info
        if diff is not None:
            diff_2d = self._to_2d(diff) if diff.ndim == 3 else diff
            changed = int(np.sum(diff_2d))
            lines.append(f"Aenderungen: {changed} Pixel veraendert")
            if changed > 0:
                ys, xs = np.where(diff_2d)
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                bx1, bx2 = int(xs.min()), int(xs.max())
                by1, by2 = int(ys.min()), int(ys.max())
                lines.append(f"  Zentrum: ({cx},{cy}), BBox: ({bx1},{by1})-({bx2},{by2})")

        return "\n".join(lines)

    def encode_compact(self, grid: np.ndarray) -> str:
        """Return a minimal one-line color summary: ``[color1:count, ...]``."""
        grid_2d = self._to_2d(grid)
        unique, counts = np.unique(grid_2d, return_counts=True)
        order = np.argsort(counts)[::-1]
        parts: list[str] = []
        for idx in order[:3]:
            color_idx = int(unique[idx])
            name = self.color_names.get(color_idx, str(color_idx))
            parts.append(f"{name}:{int(counts[idx])}")
        return "[" + ", ".join(parts) + "]"

    def _find_bounding_boxes(
        self,
        grid_2d: np.ndarray,
        background: int = 0,
        min_size: int = 4,
    ) -> list[tuple[int, int, int, int, int]]:
        """Return ``(color, x1, y1, x2, y2)`` tuples for each non-background region.

        Only regions with at least *min_size* pixels are included.
        """
        results: list[tuple[int, int, int, int, int]] = []
        unique_colors = np.unique(grid_2d)
        for color in unique_colors:
            if color == background:
                continue
            mask = grid_2d == color
            pixel_count = int(np.sum(mask))
            if pixel_count < min_size:
                continue
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            results.append((int(color), x1, y1, x2, y2))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        """Squeeze a leading size-1 batch dimension if present."""
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                return arr[0]
            # Fallback: take first slice
            return arr[0]
        return arr
