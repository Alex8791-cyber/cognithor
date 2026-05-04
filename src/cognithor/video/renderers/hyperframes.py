"""HyperFrames default renderer (HF-2).

Apache-2.0 (per HF-1 spike). Spawns ``npx hyperframes render``
against a sandboxed temp directory under
``~/.cognithor/render/<run_id>/``. The Python side is intentionally
thin — composition / scene authoring lives in the skill layer
(HF-5); this class only takes a finished HTML and produces a
finished MP4.

Default adapter allowlist (per :data:`DEFAULT_ALLOWED_ADAPTERS`) is
MIT-clean. GSAP is NOT in the support set unless the caller
explicitly opts in via the request — protects paid-pack render
paths from the GreenSock paid-tier trigger.

The subprocess contract:

* ``npx --yes hyperframes render <composition.html> --out <output.<fmt>>``
* cwd = the sandbox dir
* env = inherit
* timeout from ``RenderRequest.timeout_seconds`` (default 300 s)
* stdout / stderr captured; stderr (truncated) is folded into
  :class:`RenderError` on non-zero exit.

This module imports nothing renderer-specific from elsewhere in
Cognithor — the only coupling is the ``cognithor.video.renderer_base``
contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from cognithor.video.renderer_base import (
    FrameAdapter,
    RendererABC,
    RenderError,
    RenderResult,
)

if TYPE_CHECKING:
    from cognithor.video.renderer_base import RenderRequest

log = logging.getLogger(__name__)


# Where renders write under the user-home. Overridden in tests via
# the ``COGNITHOR_HOME`` env var (the same convention the rest of
# the codebase uses).
def _render_root() -> Path:
    """Return the canonical render-output root directory."""

    home_override = os.environ.get("COGNITHOR_HOME")
    base = Path(home_override).expanduser() if home_override else Path.home() / ".cognithor"
    return base / "render"


# Set of adapters this renderer is willing to load. GSAP is in the
# *support* set (HyperFrames bundles it) but not in
# :data:`DEFAULT_ALLOWED_ADAPTERS` — the caller still has to opt in.
_HYPERFRAMES_SUPPORTED: Final[frozenset[FrameAdapter]] = frozenset(
    {
        FrameAdapter.CSS,
        FrameAdapter.WAAPI,
        FrameAdapter.ANIME,
        FrameAdapter.LOTTIE,
        FrameAdapter.THREE,
        FrameAdapter.GSAP,
    },
)


class HyperFramesRenderer(RendererABC):
    """Default renderer — HeyGen HyperFrames via ``npx``."""

    NAME = "hyperframes"

    def __init__(
        self,
        *,
        npx_command: str = "npx",
        node_command: str = "node",
    ) -> None:
        self._npx_command = npx_command
        self._node_command = node_command

    # ------------------------------------------------------------------
    # Renderer contract
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Probe Node 22+ + npx. Returns ``False`` on any failure."""

        # First — `npx` and `node` must be on PATH.
        if shutil.which(self._npx_command) is None:
            return False
        if shutil.which(self._node_command) is None:
            return False

        # Second — Node major must be 22+ (HF spec).
        try:
            proc = await asyncio.create_subprocess_exec(
                self._node_command,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except (OSError, TimeoutError):
            return False

        version = stdout_b.decode("utf-8", errors="replace").strip().lstrip("v")
        major_part = version.split(".", 1)[0]
        try:
            major = int(major_part)
        except ValueError:
            return False
        return major >= 22

    def supported_adapters(self) -> frozenset[FrameAdapter]:
        return _HYPERFRAMES_SUPPORTED

    async def render(self, request: RenderRequest) -> RenderResult:
        # Adapter-allowlist gate (default disables GSAP for paid
        # paths; caller-supplied allowlist still has to be a subset
        # of what HyperFrames itself supports).
        self.reject_disallowed_adapters(request)

        sandbox = self._prepare_sandbox(request.run_id)
        try:
            html_path = self._materialize_html(request, sandbox)
            output_path = self._resolve_output_path(request, sandbox)
            started = time.monotonic()
            await self._spawn_render(
                html_path=html_path,
                output_path=output_path,
                request=request,
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            return RenderResult(
                run_id=request.run_id,
                output_path=output_path,
                output_format=request.output_format,
                duration_ms=duration_ms,
                width=request.width,
                height=request.height,
                fps=request.fps,
                bytes_written=output_path.stat().st_size,
            )
        except RenderError:
            raise
        except (OSError, TimeoutError) as exc:
            msg = f"hyperframes render: {type(exc).__name__}: {exc}"
            raise RenderError(msg, run_id=request.run_id) from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_sandbox(self, run_id: str) -> Path:
        """Create + return ``~/.cognithor/render/<run_id>/`` cleanly."""

        sandbox = _render_root() / run_id
        sandbox.mkdir(parents=True, exist_ok=True)
        return sandbox

    def _materialize_html(self, request: RenderRequest, sandbox: Path) -> Path:
        """Return a path to the composition HTML inside the sandbox.

        When ``html_path`` is supplied, it is copied into the
        sandbox so the renderer's cwd contains a stable filename
        the operator can find later. When ``html_text`` is supplied,
        it is written to ``composition.html``.
        """

        target = sandbox / "composition.html"
        if request.html_path is not None:
            target.write_text(
                Path(request.html_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            assert request.html_text is not None  # invariant from RenderRequest
            target.write_text(request.html_text, encoding="utf-8")
        return target

    def _resolve_output_path(self, request: RenderRequest, sandbox: Path) -> Path:
        """Where the renderer should write the MP4 / MOV / WebM."""

        out_dir = request.output_dir or sandbox
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"render.{request.output_format.value}"

    async def _spawn_render(
        self,
        *,
        html_path: Path,
        output_path: Path,
        request: RenderRequest,
    ) -> None:
        """Run ``npx hyperframes render``. Raises :class:`RenderError` on failure."""

        argv = [
            self._npx_command,
            "--yes",
            "hyperframes",
            "render",
            str(html_path),
            "--out",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=html_path.parent,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=request.timeout_seconds,
            )
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            msg = f"hyperframes render timed out after {request.timeout_seconds}s"
            raise RenderError(msg, run_id=request.run_id) from exc

        if proc.returncode != 0:
            stderr_text = stderr_b.decode("utf-8", errors="replace")
            msg = f"hyperframes render exited with code {proc.returncode}"
            raise RenderError(
                msg,
                run_id=request.run_id,
                stderr_excerpt=stderr_text,
            )

        if not output_path.exists():
            stdout_text = stdout_b.decode("utf-8", errors="replace")
            msg = f"hyperframes render claimed success but {output_path} is missing"
            raise RenderError(
                msg,
                run_id=request.run_id,
                stderr_excerpt=stdout_text,
            )
