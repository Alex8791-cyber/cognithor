"""Renderer ABC + wire types for the Cognithor video surface (HF-2).

Decision per ``docs/superpowers/spikes/2026-05-04-hyperframes-spike.md``:
the rendering layer is renderer-agnostic. Every backend (HyperFrames
default, future Remotion, future homegrown) implements
:class:`RendererABC`. The MCP tools (``video_compose`` / ``video_render``)
shipped in HF-3 talk only to this contract — never to a concrete
renderer — so a swap is a one-file change.

Frozen dataclass + ``StrEnum`` taxonomy mirrors the rest of the
TRUST stack (see ``cognithor.security.permission_scope``,
``cognithor.security.cost_ledger``). All wire types are
JSON-serialisable so they can ride the MCP transport without
custom encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from pathlib import Path


class OutputFormat(StrEnum):
    """Supported render-output container formats."""

    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"


class FrameAdapter(StrEnum):
    """Animation runtimes a renderer may load when rendering a composition.

    Mirrors the HyperFrames adapter taxonomy (CSS, WAAPI, Anime.js,
    Lottie, Three.js, GSAP) so other renderers can opt into the same
    name space. Not every renderer supports every adapter — the
    ``RendererABC.supported_adapters()`` introspection lets callers
    discover what is available.
    """

    CSS = "css"
    WAAPI = "waapi"
    ANIME = "anime"
    LOTTIE = "lottie"
    THREE = "three"
    GSAP = "gsap"


# Default adapter allowlist for paid-pack render paths (HF-1 spike
# doc — GreenSock license risk excludes GSAP from the default).
# Free-pack / core paths can opt in to GSAP via an explicit flag.
DEFAULT_ALLOWED_ADAPTERS: Final[frozenset[FrameAdapter]] = frozenset(
    {
        FrameAdapter.CSS,
        FrameAdapter.WAAPI,
        FrameAdapter.ANIME,
        FrameAdapter.LOTTIE,
        FrameAdapter.THREE,
    },
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderRequest:
    """A renderer-agnostic render request.

    Either ``html_path`` (a path to a self-contained composition
    HTML file) OR ``html_text`` (the inline composition string)
    must be set, never both. The renderer-agnostic Gatekeeper
    rule in HF-3 enforces this invariant — repeated here as a
    construction-time validation so accidental misuse is caught
    before it reaches the renderer.

    ``run_id`` is the same identifier used by the streaming
    EventEmitter (PR-A) and TRUST-1 receipt API, so a single video
    render can be cross-referenced from the agent run that
    produced it.
    """

    run_id: str
    html_path: Path | None = None
    html_text: str | None = None
    output_format: OutputFormat = OutputFormat.MP4
    output_dir: Path | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration_seconds: float | None = None
    allowed_adapters: frozenset[FrameAdapter] = DEFAULT_ALLOWED_ADAPTERS
    timeout_seconds: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            msg = "RenderRequest.run_id must be a non-empty string"
            raise ValueError(msg)
        if self.html_path is None and self.html_text is None:
            msg = "RenderRequest needs either html_path or html_text"
            raise ValueError(msg)
        if self.html_path is not None and self.html_text is not None:
            msg = "RenderRequest cannot carry both html_path and html_text"
            raise ValueError(msg)
        if self.width < 16 or self.height < 16:
            msg = "RenderRequest width / height must be >= 16"
            raise ValueError(msg)
        if self.fps < 1 or self.fps > 240:
            msg = "RenderRequest fps must be in [1, 240]"
            raise ValueError(msg)
        if self.timeout_seconds <= 0:
            msg = "RenderRequest timeout_seconds must be > 0"
            raise ValueError(msg)
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            msg = "RenderRequest duration_seconds must be > 0 when set"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderResult:
    """Successful render output."""

    run_id: str
    output_path: Path
    output_format: OutputFormat
    duration_ms: int
    width: int
    height: int
    fps: int
    bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable wire shape for MCP responses + audit entries."""

        return {
            "run_id": self.run_id,
            "output_path": str(self.output_path),
            "output_format": self.output_format.value,
            "duration_ms": self.duration_ms,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bytes_written": self.bytes_written,
        }


class RenderError(RuntimeError):
    """Raised by every renderer when a render cannot complete.

    Carries the same ``run_id`` the request did so the caller can
    correlate the failure with the originating agent run / TRUST-1
    receipt, plus an optional ``stderr_excerpt`` (truncated to 500
    chars) for the audit trail.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        stderr_excerpt: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.stderr_excerpt = (stderr_excerpt or "")[:500] or None

    def __str__(self) -> str:
        base = super().__str__()
        if self.stderr_excerpt:
            return f"{base}\n--- stderr (truncated) ---\n{self.stderr_excerpt}"
        return base


class RendererABC(ABC):
    """Base class every renderer (HyperFrames, future Remotion, ...) implements.

    Subclasses override :meth:`render` to do the actual work and
    :meth:`supported_adapters` / :meth:`name` for introspection.
    They MUST NOT mutate the input :class:`RenderRequest` — any
    transformation is the renderer's own internal state.
    """

    #: Short-and-stable identifier — used in the MCP tools'
    #: ``--renderer NAME`` flag and in the registry dict below.
    NAME: str = "abstract"

    @property
    def name(self) -> str:
        """Convenience accessor for the class-level ``NAME``."""

        return self.NAME

    @abstractmethod
    async def is_available(self) -> bool:
        """``True`` when this renderer can satisfy requests right now.

        Implementations probe whatever they need (e.g. ``npx`` on
        PATH, a writable temp dir, a running browser pool) and
        return a bool. Must not raise.
        """

    @abstractmethod
    def supported_adapters(self) -> frozenset[FrameAdapter]:
        """Adapters this renderer can load. Static — describes the binary."""

    @abstractmethod
    async def render(self, request: RenderRequest) -> RenderResult:
        """Run the render. Raises :class:`RenderError` on any failure."""

    def reject_disallowed_adapters(self, request: RenderRequest) -> None:
        """Validate the request's adapter allowlist against the renderer's support set.

        Called by concrete renderers at the top of :meth:`render` so
        every backend uses the same enforcement. Mismatches raise
        :class:`RenderError` so the Planner sees a single typed
        failure rather than a renderer-specific shell error.
        """

        unsupported = request.allowed_adapters - self.supported_adapters()
        if unsupported:
            names = sorted(a.value for a in unsupported)
            msg = f"renderer {self.name!r} does not support adapter(s): " + ", ".join(names)
            raise RenderError(msg, run_id=request.run_id)
