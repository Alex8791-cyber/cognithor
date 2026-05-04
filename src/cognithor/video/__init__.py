"""Cognithor video-rendering surface (Sprint-27 HF track).

Owner-decision Option C (see
``docs/superpowers/spikes/2026-05-04-hyperframes-spike.md``):
ship a thin :class:`RendererABC` abstraction with HyperFrames as
the default backend, so future renderers (Remotion, homegrown,
cloud) can be swapped in without touching the MCP-tool layer.

Public surface:

* :class:`RendererABC` — base contract every renderer implements.
* :class:`RenderRequest` / :class:`RenderResult` / :class:`RenderError`
  — the wire types between the MCP tools (HF-3) and a renderer.
* :class:`HyperFramesRenderer` — default Apache-2.0 backend
  (HF-2 ships this).
* :class:`OutputFormat` / :class:`FrameAdapter` — taxonomies the
  Gatekeeper + skill layer (HF-5) consume.
* :data:`renderer_registry` — ``{name: factory}`` lookup the
  ``video_render`` MCP tool dispatches against.
"""

from __future__ import annotations

from cognithor.video.renderer_base import (
    DEFAULT_ALLOWED_ADAPTERS,
    FrameAdapter,
    OutputFormat,
    RendererABC,
    RenderError,
    RenderRequest,
    RenderResult,
)
from cognithor.video.renderers import HyperFramesRenderer, renderer_registry

__all__ = [
    "DEFAULT_ALLOWED_ADAPTERS",
    "FrameAdapter",
    "HyperFramesRenderer",
    "OutputFormat",
    "RenderError",
    "RenderRequest",
    "RenderResult",
    "RendererABC",
    "renderer_registry",
]
