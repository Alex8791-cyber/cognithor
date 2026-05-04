"""Sprint-27 HF-3 — `video_compose` + `video_render` MCP-tool handlers."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.mcp.video_tools import (
    _coerce_adapters,
    _coerce_format,
    _sanitize_text,
    _video_compose_handler,
    _video_render_handler,
    compose_html,
    register_video_tools,
)
from cognithor.video.renderer_base import (
    DEFAULT_ALLOWED_ADAPTERS,
    FrameAdapter,
    OutputFormat,
)

# ---------------------------------------------------------------------------
# compose_html — pure function tests
# ---------------------------------------------------------------------------


class TestComposeHTML:
    def test_minimal_spec_renders_shell(self) -> None:
        html = compose_html({"title": "Test", "scenes": []})
        assert "<!doctype html>" in html
        assert "<title>Test</title>" in html
        assert 'data-composition-id="comp"' in html  # default sanitised id

    def test_scenes_emitted_with_track_index(self) -> None:
        html = compose_html(
            {
                "title": "Multi",
                "scenes": [
                    {"start": 0, "duration": 3, "caption": "scene one"},
                    {"start": 3, "duration": 2, "caption": "scene two"},
                ],
            },
        )
        assert 'data-track-index="0"' in html
        assert 'data-track-index="1"' in html
        assert "scene one" in html
        assert "scene two" in html
        # Start + duration baked in
        assert 'data-start="0"' in html
        assert 'data-duration="3"' in html

    def test_html_text_escapes_dangerous_chars(self) -> None:
        html = compose_html(
            {
                "title": "<script>alert(1)</script>",
                "scenes": [{"start": 0, "duration": 1, "caption": "</script>"}],
            },
        )
        # The <title> field should NOT contain raw <script>.
        assert "<script>" not in html
        # Caption escaping
        assert "</script>" not in html

    def test_dimensions_clamped(self) -> None:
        html = compose_html({"title": "Big", "width": 99999, "height": 99999})
        # Clamped to 7680 / 4320
        assert 'data-width="7680"' in html
        assert 'data-height="4320"' in html

    def test_invalid_scene_dropped(self) -> None:
        html = compose_html(
            {
                "scenes": [
                    "not a dict",
                    {"start": 0, "duration": -1},  # bad duration
                    {"start": 0, "duration": 1, "caption": "kept"},
                ],
            },
        )
        assert "kept" in html
        # Only the valid scene should land — track-index 2 (the third item).
        assert 'data-track-index="2"' in html
        assert 'data-track-index="0"' not in html
        assert 'data-track-index="1"' not in html

    def test_remote_image_url_stripped(self) -> None:
        html = compose_html(
            {
                "scenes": [
                    {
                        "start": 0,
                        "duration": 1,
                        "image_url": "https://evil.example.com/x.png",
                    },
                ],
            },
        )
        assert "evil.example.com" not in html

    def test_local_image_url_kept(self) -> None:
        html = compose_html(
            {
                "scenes": [
                    {"start": 0, "duration": 1, "image_url": "./assets/hero.png"},
                ],
            },
        )
        assert "./assets/hero.png" in html


# ---------------------------------------------------------------------------
# _sanitize_text + _coerce_* helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_sanitize_keeps_unicode_letters(self) -> None:
        assert _sanitize_text("Schöne Grüße!") == "Schöne Grüße!"

    def test_sanitize_drops_html_brackets(self) -> None:
        out = _sanitize_text("<b>X</b>")
        assert "<" not in out
        assert ">" not in out

    def test_sanitize_truncates_long_input(self) -> None:
        long = "x" * 1000
        assert len(_sanitize_text(long)) == 512

    def test_coerce_adapters_falls_back_to_default(self) -> None:
        assert _coerce_adapters(None) == DEFAULT_ALLOWED_ADAPTERS
        assert _coerce_adapters("nope") == DEFAULT_ALLOWED_ADAPTERS  # type: ignore[arg-type]

    def test_coerce_adapters_typed(self) -> None:
        out = _coerce_adapters(["css", "lottie"])
        assert out == frozenset({FrameAdapter.CSS, FrameAdapter.LOTTIE})

    def test_coerce_adapters_drops_unknown(self) -> None:
        out = _coerce_adapters(["css", "fake-thing"])
        assert out == frozenset({FrameAdapter.CSS})

    def test_coerce_format_default(self) -> None:
        assert _coerce_format(None) == OutputFormat.MP4
        assert _coerce_format("WEBM") == OutputFormat.WEBM
        assert _coerce_format("nope") == OutputFormat.MP4


# ---------------------------------------------------------------------------
# video_compose handler
# ---------------------------------------------------------------------------


class TestVideoComposeHandler:
    @pytest.mark.asyncio
    async def test_returns_html_on_valid_spec(self) -> None:
        result = await _video_compose_handler(
            spec={"title": "Demo", "scenes": [{"start": 0, "duration": 2}]},
            run_id="run-1",
        )
        assert result["ok"] is True
        assert "<!doctype html>" in result["html"]
        assert result["byte_size"] > 0
        assert result["run_id"] == "run-1"

    @pytest.mark.asyncio
    async def test_rejects_non_object_spec(self) -> None:
        result = await _video_compose_handler(spec="not a dict")  # type: ignore[arg-type]
        assert result["ok"] is False
        assert "object" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_oversize_html(self) -> None:
        # Pump in 200 scenes with long captions to exceed 64 KB cap.
        scenes = [{"start": i, "duration": 1, "caption": "x" * 500} for i in range(200)]
        result = await _video_compose_handler(spec={"scenes": scenes})
        assert result["ok"] is False
        assert "cap" in result["error"]


# ---------------------------------------------------------------------------
# video_render handler — exercises the registry dispatch + error paths
# ---------------------------------------------------------------------------


class TestVideoRenderHandler:
    @pytest.mark.asyncio
    async def test_unknown_renderer_returns_error(self) -> None:
        result = await _video_render_handler(
            run_id="r1",
            html_text="<div/>",
            renderer="does-not-exist",
        )
        assert result["ok"] is False
        assert "unknown renderer" in result["error"]
        assert "available" in result

    @pytest.mark.asyncio
    async def test_missing_run_id_rejected(self) -> None:
        result = await _video_render_handler(run_id="", html_text="<div/>")
        assert result["ok"] is False
        assert "run_id" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_request_caught(self) -> None:
        # Empty html_text + html_path → RenderRequest validation fails.
        result = await _video_render_handler(run_id="r1")
        assert result["ok"] is False
        assert "render request" in result["error"]


# ---------------------------------------------------------------------------
# register_video_tools — schema + handler wired correctly
# ---------------------------------------------------------------------------


class _StubServer:
    """Captures registered MCPToolDef instances."""

    def __init__(self) -> None:
        self.tools: list[Any] = []

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)


class TestRegistration:
    def test_both_tools_registered(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        names = [t.name for t in srv.tools]
        assert "video_compose" in names
        assert "video_render" in names

    def test_video_compose_is_green(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        compose = next(t for t in srv.tools if t.name == "video_compose")
        assert compose.annotations["risk_level"] == "green"
        assert compose.annotations["category"] == "video"

    def test_video_render_is_orange(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        render = next(t for t in srv.tools if t.name == "video_render")
        assert render.annotations["risk_level"] == "orange"

    def test_compose_schema_requires_spec(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        compose = next(t for t in srv.tools if t.name == "video_compose")
        assert "spec" in compose.input_schema["required"]

    def test_render_schema_requires_run_id(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        render = next(t for t in srv.tools if t.name == "video_render")
        assert "run_id" in render.input_schema["required"]
