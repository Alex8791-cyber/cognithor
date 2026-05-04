"""Tests for `cognithor.video.renderer_base` — wire-types + ABC contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.video.renderer_base import (
    DEFAULT_ALLOWED_ADAPTERS,
    FrameAdapter,
    OutputFormat,
    RendererABC,
    RenderError,
    RenderRequest,
    RenderResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# RenderRequest validation
# ---------------------------------------------------------------------------


class TestRenderRequest:
    def test_html_path_only_is_valid(self, tmp_path: Path) -> None:
        html = tmp_path / "comp.html"
        html.write_text("<div></div>", encoding="utf-8")
        req = RenderRequest(run_id="r1", html_path=html)
        assert req.html_path == html
        assert req.html_text is None
        assert req.output_format == OutputFormat.MP4
        assert req.allowed_adapters == DEFAULT_ALLOWED_ADAPTERS

    def test_html_text_only_is_valid(self) -> None:
        req = RenderRequest(run_id="r1", html_text="<div></div>")
        assert req.html_text == "<div></div>"

    def test_run_id_required(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            RenderRequest(run_id="", html_text="<div></div>")

    def test_must_have_html_source(self) -> None:
        with pytest.raises(ValueError, match="html_path or html_text"):
            RenderRequest(run_id="r1")

    def test_cannot_have_both_html_sources(self, tmp_path: Path) -> None:
        html = tmp_path / "comp.html"
        html.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot carry both"):
            RenderRequest(run_id="r1", html_path=html, html_text="<div/>")

    def test_dimensions_floor(self) -> None:
        with pytest.raises(ValueError, match=">= 16"):
            RenderRequest(run_id="r1", html_text="<div/>", width=4, height=4)

    def test_fps_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 240\]"):
            RenderRequest(run_id="r1", html_text="<div/>", fps=0)
        with pytest.raises(ValueError, match=r"\[1, 240\]"):
            RenderRequest(run_id="r1", html_text="<div/>", fps=300)

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            RenderRequest(run_id="r1", html_text="<div/>", timeout_seconds=0)

    def test_duration_when_set_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds"):
            RenderRequest(
                run_id="r1",
                html_text="<div/>",
                duration_seconds=-1.0,
            )

    def test_default_adapters_excludes_gsap(self) -> None:
        # HF-1 spike decision — GSAP off by default.
        assert FrameAdapter.GSAP not in DEFAULT_ALLOWED_ADAPTERS
        # MIT adapters are on by default.
        for adapter in (
            FrameAdapter.CSS,
            FrameAdapter.WAAPI,
            FrameAdapter.ANIME,
            FrameAdapter.LOTTIE,
            FrameAdapter.THREE,
        ):
            assert adapter in DEFAULT_ALLOWED_ADAPTERS


# ---------------------------------------------------------------------------
# RenderResult.to_dict round-trip
# ---------------------------------------------------------------------------


class TestRenderResult:
    def test_to_dict_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "render.mp4"
        out.write_bytes(b"fake")
        result = RenderResult(
            run_id="r1",
            output_path=out,
            output_format=OutputFormat.MP4,
            duration_ms=4321,
            width=1920,
            height=1080,
            fps=30,
            bytes_written=4,
        )
        d = result.to_dict()
        assert d == {
            "run_id": "r1",
            "output_path": str(out),
            "output_format": "mp4",
            "duration_ms": 4321,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "bytes_written": 4,
        }


# ---------------------------------------------------------------------------
# RenderError carries run_id + truncates stderr
# ---------------------------------------------------------------------------


class TestRenderError:
    def test_str_without_stderr(self) -> None:
        err = RenderError("boom", run_id="r1")
        s = str(err)
        assert "boom" in s
        assert "stderr" not in s

    def test_str_with_stderr(self) -> None:
        err = RenderError("boom", run_id="r1", stderr_excerpt="oh no")
        assert "stderr" in str(err)
        assert "oh no" in str(err)

    def test_stderr_truncated_to_500(self) -> None:
        long = "x" * 1000
        err = RenderError("boom", run_id="r1", stderr_excerpt=long)
        assert err.stderr_excerpt is not None
        assert len(err.stderr_excerpt) == 500

    def test_empty_stderr_becomes_none(self) -> None:
        err = RenderError("boom", run_id="r1", stderr_excerpt="")
        assert err.stderr_excerpt is None


# ---------------------------------------------------------------------------
# RendererABC.reject_disallowed_adapters helper
# ---------------------------------------------------------------------------


class _StubRenderer(RendererABC):
    NAME = "stub"

    def __init__(self, supported: frozenset[FrameAdapter]) -> None:
        self._supported = supported

    async def is_available(self) -> bool:
        return True

    def supported_adapters(self) -> frozenset[FrameAdapter]:
        return self._supported

    async def render(self, request: RenderRequest) -> RenderResult:  # pragma: no cover
        msg = "stub does not actually render"
        raise NotImplementedError(msg)


class TestAdapterEnforcement:
    def test_allowed_subset_passes(self) -> None:
        renderer = _StubRenderer(supported=DEFAULT_ALLOWED_ADAPTERS)
        req = RenderRequest(
            run_id="r1",
            html_text="<div/>",
            allowed_adapters=DEFAULT_ALLOWED_ADAPTERS,
        )
        # Should NOT raise.
        renderer.reject_disallowed_adapters(req)

    def test_disallowed_adapter_raises(self) -> None:
        renderer = _StubRenderer(
            supported=frozenset({FrameAdapter.CSS}),
        )
        req = RenderRequest(
            run_id="r1",
            html_text="<div/>",
            allowed_adapters=frozenset({FrameAdapter.CSS, FrameAdapter.GSAP}),
        )
        with pytest.raises(RenderError, match="gsap"):
            renderer.reject_disallowed_adapters(req)

    def test_render_error_carries_run_id(self) -> None:
        renderer = _StubRenderer(supported=frozenset())
        req = RenderRequest(
            run_id="some-run-id",
            html_text="<div/>",
            allowed_adapters=frozenset({FrameAdapter.CSS}),
        )
        with pytest.raises(RenderError) as exc_info:
            renderer.reject_disallowed_adapters(req)
        assert exc_info.value.run_id == "some-run-id"
