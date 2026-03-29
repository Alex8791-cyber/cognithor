"""Tests for VisualStateEncoder (Task 3 — ARC-AGI-3)."""

from __future__ import annotations

import numpy as np

from jarvis.arc.visual_encoder import VisualStateEncoder


class TestEncodeForLLM:
    def test_returns_string(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        result = enc.encode_for_llm(grid)
        assert isinstance(result, str)
        assert "Farbverteilung" in result

    def test_shows_dominant_color(self):
        enc = VisualStateEncoder()
        grid = np.full((64, 64), 2, dtype=np.int8)  # All red
        result = enc.encode_for_llm(grid)
        assert "rot" in result

    def test_shows_regions(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:20, 10:20] = 1  # Blue 10x10 block
        result = enc.encode_for_llm(grid)
        assert "blau" in result

    def test_diff_info(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        diff = np.zeros((64, 64), dtype=bool)
        diff[30:35, 30:35] = True  # 25 changed pixels
        result = enc.encode_for_llm(grid, diff=diff)
        assert "25" in result or "Pixel" in result.lower() or "nderung" in result

    def test_3d_grid_handled(self):
        enc = VisualStateEncoder()
        grid = np.zeros((1, 64, 64), dtype=np.int8)
        result = enc.encode_for_llm(grid)
        assert isinstance(result, str)


class TestEncodeCompact:
    def test_format(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        result = enc.encode_compact(grid)
        assert result.startswith("[")
        assert result.endswith("]")

    def test_contains_color(self):
        enc = VisualStateEncoder()
        grid = np.full((64, 64), 4, dtype=np.int8)
        result = enc.encode_compact(grid)
        assert "gelb" in result


class TestBoundingBoxes:
    def test_single_region(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[5:15, 10:20] = 3  # Green block
        boxes = enc._find_bounding_boxes(grid, background=0)
        assert len(boxes) >= 1
        color, x1, y1, x2, y2 = boxes[0]
        assert color == 3

    def test_ignores_small_regions(self):
        enc = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, 0] = 5  # Single pixel
        boxes = enc._find_bounding_boxes(grid, background=0, min_size=4)
        assert len(boxes) == 0
