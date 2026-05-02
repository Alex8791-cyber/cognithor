# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-14 PR-2 — vision_render tests."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.vision_render import (
    ARC_PALETTE,
    render_grid_data_uri,
    render_grid_image,
    render_grid_png_bytes,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestPaletteShape:
    def test_palette_has_16_colors(self) -> None:
        assert len(ARC_PALETTE) == 16

    def test_each_entry_is_rgb_triple(self) -> None:
        for color in ARC_PALETTE:
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255


class TestRenderGridImage:
    def test_2d_input_produces_correct_size(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int8)
        img = render_grid_image(grid, scale=2)
        assert img.size == (8, 8)

    def test_default_scale_64x64_produces_512x512(self) -> None:
        grid = np.zeros((64, 64), dtype=np.int8)
        img = render_grid_image(grid)
        assert img.size == (512, 512)

    def test_rejects_1d_grid(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            render_grid_image(np.array([1, 2, 3], dtype=np.int8))

    def test_pixel_color_matches_palette(self) -> None:
        grid = np.zeros((2, 2), dtype=np.int8)
        grid[0, 0] = 2  # red
        img = render_grid_image(grid, scale=4)
        # Top-left 4x4 block should be ARC_PALETTE[2] = (255, 65, 54).
        assert img.getpixel((0, 0)) == ARC_PALETTE[2]
        # Bottom-right block should be ARC_PALETTE[0] = (0, 0, 0).
        assert img.getpixel((7, 7)) == ARC_PALETTE[0]

    def test_out_of_range_clamps_to_zero(self) -> None:
        grid = np.zeros((1, 1), dtype=np.int32)
        grid[0, 0] = 99  # out of palette
        img = render_grid_image(grid, scale=2)
        assert img.getpixel((0, 0)) == ARC_PALETTE[0]


class TestPngBytes:
    def test_returns_valid_png(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int8)
        data = render_grid_png_bytes(grid)
        # PNG magic header.
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


class TestDataUri:
    def test_returns_base64_data_uri(self) -> None:
        grid = np.zeros((2, 2), dtype=np.int8)
        uri = render_grid_data_uri(grid)
        assert uri.startswith("data:image/png;base64,")
        # The b64 payload decodes to a valid PNG.
        b64 = uri.removeprefix("data:image/png;base64,")
        decoded = base64.b64decode(b64)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_grid_with_colors_produces_unique_uri(self) -> None:
        grid_a = np.zeros((4, 4), dtype=np.int8)
        grid_b = np.zeros((4, 4), dtype=np.int8)
        grid_b[1, 1] = 5
        assert render_grid_data_uri(grid_a) != render_grid_data_uri(grid_b)
