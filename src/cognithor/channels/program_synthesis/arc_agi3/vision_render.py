# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-14 PR-2 — vision rendering for multimodal LLM prompts.

Qwen3.6-27B is a vision model. Feeding it a 64x64 grid as 4096 ASCII
characters wastes its actual capability and costs 2-3K input tokens.
Rendering the same grid as a small PNG (with the ARC-AGI-3 16-color
palette) is dramatically cheaper AND lets the LLM "see" shapes,
spatial layouts, and colour-cluster boundaries directly.

This module ships:

* :func:`render_grid_image` — int8 grid → PIL Image (upscaled to a
  human/LLM-friendly resolution; default 8 px/cell so 64x64 → 512x512).
* :func:`render_grid_data_uri` — same image as a base64 data URI for
  use in chat-with-image content blocks.
* :func:`render_grid_png_bytes` — same image as raw PNG bytes for
  callers that prefer a buffer.

The colour palette mirrors ARC-AGI-3's official 16-colour scheme so
the LLM sees the same visualisation a human would.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ARC_PALETTE",
    "render_grid_data_uri",
    "render_grid_image",
    "render_grid_png_bytes",
]


# ARC-AGI-3 official 16-colour palette (RGB tuples).
# Indices 0..9 match the classic ARC-AGI-1 palette; 10..15 are the
# Sprint-12 palette extension for the wider ARC-AGI-3 colour space.
ARC_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0  black
    (0, 116, 217),  # 1  blue
    (255, 65, 54),  # 2  red
    (46, 204, 64),  # 3  green
    (255, 220, 0),  # 4  yellow
    (170, 170, 170),  # 5  grey
    (240, 18, 190),  # 6  magenta
    (255, 133, 27),  # 7  orange
    (127, 219, 255),  # 8  light-blue
    (135, 12, 37),  # 9  dark-red
    (200, 200, 200),  # 10 light-grey
    (139, 69, 19),  # 11 brown
    (100, 50, 200),  # 12 purple
    (50, 200, 200),  # 13 cyan
    (200, 100, 0),  # 14 dark-orange
    (255, 255, 255),  # 15 white
)


def render_grid_image(grid: np.ndarray[Any, Any], *, scale: int = 8) -> Any:
    """Render an ``int8`` grid as an upscaled PIL Image.

    Each cell becomes a ``scale × scale`` block of solid colour from
    :data:`ARC_PALETTE`. Out-of-range values are clamped to 0 (black).
    Default ``scale=8`` produces 512x512 images for 64x64 grids — a
    sweet spot between LLM-readable and token-efficient.
    """
    from PIL import Image

    if grid.ndim != 2:
        raise ValueError(f"render_grid_image: expected 2-D grid, got {grid.ndim}-D")
    h, w = grid.shape
    img = Image.new("RGB", (w * scale, h * scale))
    pixels = img.load()
    if pixels is None:  # pragma: no cover — defensive, PIL always returns a buffer
        raise RuntimeError("PIL.Image.new(RGB).load() unexpectedly returned None")
    palette_max = len(ARC_PALETTE) - 1
    for y in range(h):
        for x in range(w):
            color_idx = int(grid[y, x])
            if color_idx < 0 or color_idx > palette_max:
                color_idx = 0
            color = ARC_PALETTE[color_idx]
            for dy in range(scale):
                for dx in range(scale):
                    pixels[x * scale + dx, y * scale + dy] = color
    return img


def render_grid_png_bytes(grid: np.ndarray[Any, Any], *, scale: int = 8) -> bytes:
    """Return the rendered grid as raw PNG bytes."""
    img = render_grid_image(grid, scale=scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_grid_data_uri(grid: np.ndarray[Any, Any], *, scale: int = 8) -> str:
    """Return the rendered grid as a ``data:image/png;base64,...`` URI.

    Suitable for direct insertion into chat content blocks of the
    form ``{"type": "image_url", "image_url": {"url": "data:..."}}``
    (OpenAI / vLLM multimodal convention).
    """
    payload = base64.b64encode(render_grid_png_bytes(grid, scale=scale)).decode("ascii")
    return f"data:image/png;base64,{payload}"
