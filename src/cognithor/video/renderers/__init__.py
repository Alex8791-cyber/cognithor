"""Concrete renderers + dispatch registry (HF-2).

The MCP tools shipped in HF-3 dispatch to a renderer by string
name, so we keep a single source-of-truth registry here. Future
renderers (Remotion, homegrown, cloud) add a single line to
:data:`renderer_registry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from cognithor.video.renderers.hyperframes import HyperFramesRenderer

if TYPE_CHECKING:
    from collections.abc import Callable

    from cognithor.video.renderer_base import RendererABC

#: Registry of available renderer factories, keyed by short
#: identifier. Stays a plain dict so callers can introspect /
#: extend it for tests without touching production code.
renderer_registry: Final[dict[str, Callable[[], RendererABC]]] = {
    HyperFramesRenderer.NAME: HyperFramesRenderer,
}

__all__ = ["HyperFramesRenderer", "renderer_registry"]
