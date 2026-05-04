"""Tests for the default :class:`HyperFramesRenderer`.

The real ``npx hyperframes render`` is not invoked in CI. These
tests pin the subprocess contract via a tiny shim binary written
into ``tmp_path`` and used as a stand-in for ``npx``. The shim
simulates HyperFrames by writing a fake MP4 byte sequence to the
``--out`` path, then exits 0. Failure paths swap in a shim that
exits non-zero.
"""

from __future__ import annotations

import platform
import stat
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from cognithor.video.renderer_base import (
    FrameAdapter,
    OutputFormat,
    RenderError,
    RenderRequest,
)
from cognithor.video.renderers.hyperframes import HyperFramesRenderer

if TYPE_CHECKING:
    from pathlib import Path


def _write_fake_npx_shim(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    write_output: bool = True,
    stderr_text: str = "",
) -> Path:
    """Write a minimal Python script + thin wrapper that mimics ``npx``.

    The shim parses ``hyperframes render <html> --out <output>``
    from sys.argv, writes a tiny fake MP4 to the output path (or
    skips it to test the missing-output failure mode), and exits
    with the requested code.
    """

    py_path = tmp_path / "fake_npx.py"
    py_path.write_text(
        textwrap.dedent(
            f"""
            import sys, pathlib
            argv = sys.argv[1:]
            try:
                out_idx = argv.index("--out")
                out_path = pathlib.Path(argv[out_idx + 1])
            except (ValueError, IndexError):
                sys.stderr.write("fake_npx: missing --out\\n")
                sys.exit(2)
            if {write_output!r}:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"fake-mp4-bytes")
            if {stderr_text!r}:
                sys.stderr.write({stderr_text!r})
            sys.exit({exit_code})
            """,
        ).strip(),
        encoding="utf-8",
    )
    if platform.system() == "Windows":
        wrapper = tmp_path / "fake_npx.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{py_path}" %*\r\n',
            encoding="utf-8",
        )
        return wrapper
    wrapper = tmp_path / "fake_npx"
    wrapper.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{py_path}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


@pytest.fixture
def cognithor_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sandbox the render-output root under tmp_path."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COGNITHOR_HOME", str(home))
    return home


# ---------------------------------------------------------------------------
# render() happy-path
# ---------------------------------------------------------------------------


class TestRenderHappy:
    @pytest.mark.asyncio
    async def test_inline_html_renders_to_mp4(
        self,
        tmp_path: Path,
        cognithor_home: Path,
    ) -> None:
        shim = _write_fake_npx_shim(tmp_path)
        renderer = HyperFramesRenderer(npx_command=str(shim))

        req = RenderRequest(
            run_id="run-A",
            html_text='<div data-composition-id="x"></div>',
            output_format=OutputFormat.MP4,
            timeout_seconds=10.0,
        )
        result = await renderer.render(req)

        assert result.run_id == "run-A"
        assert result.output_format == OutputFormat.MP4
        assert result.bytes_written == len(b"fake-mp4-bytes")
        assert result.output_path.exists()
        # Sandbox lives under ~/.cognithor/render/<run_id>/
        assert "run-A" in str(result.output_path)
        assert result.output_path.parent.is_relative_to(cognithor_home)

    @pytest.mark.asyncio
    async def test_html_path_input_is_copied_into_sandbox(
        self,
        tmp_path: Path,
        cognithor_home: Path,
    ) -> None:
        shim = _write_fake_npx_shim(tmp_path)
        external_html = tmp_path / "external.html"
        external_html.write_text('<div data-composition-id="ext"></div>', encoding="utf-8")

        renderer = HyperFramesRenderer(npx_command=str(shim))
        req = RenderRequest(
            run_id="run-B",
            html_path=external_html,
            timeout_seconds=10.0,
        )
        result = await renderer.render(req)
        # Sandboxed composition.html must exist with the original content.
        sandbox = cognithor_home / "render" / "run-B"
        assert (sandbox / "composition.html").exists()
        assert (sandbox / "composition.html").read_text(
            encoding="utf-8"
        ) == '<div data-composition-id="ext"></div>'
        assert result.output_path.exists()


# ---------------------------------------------------------------------------
# render() failure paths
# ---------------------------------------------------------------------------


class TestRenderFailure:
    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_render_error(
        self,
        tmp_path: Path,
        cognithor_home: Path,
    ) -> None:
        shim = _write_fake_npx_shim(
            tmp_path,
            exit_code=3,
            write_output=False,
            stderr_text="boom from fake_npx\n",
        )
        renderer = HyperFramesRenderer(npx_command=str(shim))
        req = RenderRequest(run_id="run-C", html_text="<div/>", timeout_seconds=10.0)
        with pytest.raises(RenderError) as exc_info:
            await renderer.render(req)
        err = exc_info.value
        assert err.run_id == "run-C"
        assert err.stderr_excerpt is not None
        assert "boom" in err.stderr_excerpt

    @pytest.mark.asyncio
    async def test_missing_output_raises_even_on_zero_exit(
        self,
        tmp_path: Path,
        cognithor_home: Path,
    ) -> None:
        shim = _write_fake_npx_shim(tmp_path, exit_code=0, write_output=False)
        renderer = HyperFramesRenderer(npx_command=str(shim))
        req = RenderRequest(run_id="run-D", html_text="<div/>", timeout_seconds=10.0)
        with pytest.raises(RenderError, match="missing"):
            await renderer.render(req)

    @pytest.mark.asyncio
    async def test_disallowed_gsap_adapter_rejected(
        self,
        tmp_path: Path,
        cognithor_home: Path,
    ) -> None:
        shim = _write_fake_npx_shim(tmp_path)
        renderer = HyperFramesRenderer(npx_command=str(shim))
        # Caller asks for an adapter HyperFrames *can* support
        # (GSAP) but it's not in the request's allowlist — should
        # still pass since the request is the source-of-truth for
        # allowlist. To trigger rejection, ask for something the
        # *renderer* cannot support.
        # Inject a non-existent adapter to simulate that case.
        # FrameAdapter is closed StrEnum; use an existing adapter
        # not in the renderer's support set by patching support.
        renderer._supported_override = frozenset({FrameAdapter.CSS})  # type: ignore[attr-defined]

        # Replace supported_adapters via subclass-style monkey:
        original = HyperFramesRenderer.supported_adapters

        def restricted(self: HyperFramesRenderer) -> frozenset[FrameAdapter]:
            return frozenset({FrameAdapter.CSS})

        HyperFramesRenderer.supported_adapters = restricted  # type: ignore[method-assign]
        try:
            req = RenderRequest(
                run_id="run-E",
                html_text="<div/>",
                timeout_seconds=10.0,
                allowed_adapters=frozenset({FrameAdapter.CSS, FrameAdapter.LOTTIE}),
            )
            with pytest.raises(RenderError, match="lottie"):
                await renderer.render(req)
        finally:
            HyperFramesRenderer.supported_adapters = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# is_available probing
# ---------------------------------------------------------------------------


class TestAvailability:
    @pytest.mark.asyncio
    async def test_npx_missing_returns_false(self) -> None:
        renderer = HyperFramesRenderer(
            npx_command="this-binary-does-not-exist-zxqv",
            node_command="this-binary-does-not-exist-zxqv",
        )
        assert (await renderer.is_available()) is False

    @pytest.mark.asyncio
    async def test_node_below_22_returns_false(
        self,
        tmp_path: Path,
    ) -> None:
        """Fake `node --version` outputting v20.x → treated as unavailable."""

        # Simple shim that prints v20.10.0 and exits 0.
        py_path = tmp_path / "fake_node.py"
        py_path.write_text('print("v20.10.0")\n', encoding="utf-8")
        if platform.system() == "Windows":
            wrap = tmp_path / "fake_node.cmd"
            wrap.write_text(
                f'@echo off\r\n"{sys.executable}" "{py_path}" %*\r\n',
                encoding="utf-8",
            )
        else:
            wrap = tmp_path / "fake_node"
            wrap.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{py_path}"\n',
                encoding="utf-8",
            )
            wrap.chmod(
                wrap.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
            )

        # We also need an existing `npx` for the check to even reach
        # the node-version branch — reuse the same fake_node as a stand-in
        # since `which` only looks for existence.
        renderer = HyperFramesRenderer(
            npx_command=str(wrap),
            node_command=str(wrap),
        )
        assert (await renderer.is_available()) is False
