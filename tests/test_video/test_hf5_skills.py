"""Sprint-27 HF-5 — video-composition skills layer tests.

Covers the three skill MCP tools + their underlying pure functions:

* ``compose_explainer`` / ``video_compose_explainer``
* ``compose_social_cut`` / ``video_compose_social_cut``
* ``caption_overlay`` / ``video_caption_overlay``

Each skill is GREEN-risk: pure function over ``compose_html``, no
subprocess, no filesystem write.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from cognithor.mcp.video_tools import (
    _video_caption_overlay_handler,
    _video_compose_explainer_handler,
    _video_compose_social_cut_handler,
    register_video_tools,
)
from cognithor.video.skills import (
    DEFAULT_SOCIAL_HEIGHT,
    DEFAULT_SOCIAL_WIDTH,
    MAX_BEATS,
    MAX_SECTIONS,
    caption_overlay,
    compose_explainer,
    compose_social_cut,
)

# ---------------------------------------------------------------------------
# compose_explainer — pure function tests
# ---------------------------------------------------------------------------


class TestComposeExplainer:
    def test_minimal_title_only_emits_one_scene(self) -> None:
        spec = compose_explainer(title="Hello")
        assert spec["title"] == "Hello"
        assert spec["composition_id"] == "explainer"
        scenes = spec["scenes"]
        assert len(scenes) == 1
        assert scenes[0]["title"] == "Hello"
        assert scenes[0]["start"] == 0.0
        assert scenes[0]["duration"] > 0

    def test_sections_become_body_scenes(self) -> None:
        spec = compose_explainer(
            title="Intro",
            sections=["First point", "Second point", "Third point"],
        )
        # 1 title + 3 body
        assert len(spec["scenes"]) == 4
        body = spec["scenes"][1:]
        assert [s["caption"] for s in body] == [
            "First point",
            "Second point",
            "Third point",
        ]

    def test_section_objects_with_image_url(self) -> None:
        spec = compose_explainer(
            title="With assets",
            sections=[
                {"caption": "frame one", "image_url": "./assets/a.png"},
                {"image_url": "./assets/b.png"},
            ],
        )
        body = spec["scenes"][1:]
        assert body[0]["caption"] == "frame one"
        assert body[0]["image_url"] == "./assets/a.png"
        assert "caption" not in body[1]
        assert body[1]["image_url"] == "./assets/b.png"

    def test_cta_emitted_as_final_scene(self) -> None:
        spec = compose_explainer(
            title="Open",
            sections=["body"],
            cta="Try it now",
        )
        assert spec["scenes"][-1]["title"] == "Try it now"
        # title + body + CTA = 3 scenes
        assert len(spec["scenes"]) == 3

    def test_scene_starts_are_monotonic(self) -> None:
        spec = compose_explainer(
            title="X",
            sections=["a", "b", "c"],
            cta="Done",
        )
        starts = [s["start"] for s in spec["scenes"]]
        assert starts == sorted(starts)
        assert all(b > a for a, b in itertools.pairwise(starts))

    def test_sections_clamped_to_max(self) -> None:
        spec = compose_explainer(
            title="cap",
            sections=[f"section {i}" for i in range(MAX_SECTIONS + 5)],
        )
        # title + MAX_SECTIONS body
        assert len(spec["scenes"]) == 1 + MAX_SECTIONS

    def test_invalid_section_dropped(self) -> None:
        spec = compose_explainer(
            title="filter",
            sections=["valid", 42, None, {"caption": "kept"}, ""],
        )
        body = spec["scenes"][1:]
        captions = [s.get("caption") for s in body]
        assert "valid" in captions
        assert "kept" in captions
        assert len(body) == 2

    def test_empty_title_still_returns_spec(self) -> None:
        spec = compose_explainer(title="", sections=["body"])
        # Title scene dropped (empty), but body remains.
        assert len(spec["scenes"]) == 1
        assert spec["scenes"][0]["caption"] == "body"
        # Spec falls back to default headline string for the <title> tag.
        assert spec["title"] == "Cognithor Explainer"


# ---------------------------------------------------------------------------
# compose_social_cut — pure function tests
# ---------------------------------------------------------------------------


class TestComposeSocialCut:
    def test_default_aspect_is_vertical(self) -> None:
        spec = compose_social_cut(hook="hook")
        assert spec["width"] == DEFAULT_SOCIAL_WIDTH
        assert spec["height"] == DEFAULT_SOCIAL_HEIGHT
        assert spec["height"] > spec["width"]

    def test_hook_beats_outro_full_layout(self) -> None:
        spec = compose_social_cut(
            hook="Watch this",
            beats=["beat1", "beat2", "beat3"],
            outro="Follow!",
        )
        # hook + 3 beats + outro = 5 scenes
        assert len(spec["scenes"]) == 5
        assert spec["scenes"][0]["title"] == "Watch this"
        assert spec["scenes"][-1]["title"] == "Follow!"
        assert spec["scenes"][1]["caption"] == "beat1"

    def test_beats_clamped_to_max(self) -> None:
        spec = compose_social_cut(
            hook="h",
            beats=[f"beat {i}" for i in range(MAX_BEATS + 3)],
        )
        # hook + MAX_BEATS scenes
        assert len(spec["scenes"]) == 1 + MAX_BEATS

    def test_beats_with_image_url(self) -> None:
        spec = compose_social_cut(
            hook="h",
            beats=[
                {"caption": "with image", "image_url": "./b1.png"},
                {"image_url": "./b2.png"},
            ],
        )
        beats = spec["scenes"][1:]
        assert beats[0]["image_url"] == "./b1.png"
        assert beats[1]["image_url"] == "./b2.png"

    def test_empty_hook_drops_hook_scene(self) -> None:
        spec = compose_social_cut(hook="", beats=["b1", "b2"])
        # Only the beats remain.
        assert len(spec["scenes"]) == 2

    def test_overrideable_dimensions(self) -> None:
        spec = compose_social_cut(hook="h", width=720, height=1280)
        assert spec["width"] == 720
        assert spec["height"] == 1280


# ---------------------------------------------------------------------------
# caption_overlay — pure function tests
# ---------------------------------------------------------------------------


class TestCaptionOverlay:
    def _base(self) -> dict[str, Any]:
        return {
            "title": "Base",
            "scenes": [
                {"start": 0.0, "duration": 2.0, "caption": "old1"},
                {"start": 2.0, "duration": 2.0, "caption": "old2"},
                {"start": 4.0, "duration": 2.0},
            ],
        }

    def test_replaces_existing_captions(self) -> None:
        out = caption_overlay(
            base_spec=self._base(),
            captions=["new1", "new2", "added3"],
        )
        captions = [s.get("caption") for s in out["scenes"]]
        assert captions == ["new1", "new2", "added3"]

    def test_empty_caption_leaves_existing(self) -> None:
        out = caption_overlay(
            base_spec=self._base(),
            captions=["", "  ", "kept"],
        )
        captions = [s.get("caption") for s in out["scenes"]]
        # Indexes 0 and 1 keep their old caption since new is empty.
        assert captions[0] == "old1"
        assert captions[1] == "old2"
        assert captions[2] == "kept"

    def test_extra_captions_dropped(self) -> None:
        out = caption_overlay(
            base_spec=self._base(),
            captions=["a", "b", "c", "d", "e"],
        )
        # 3 scenes only; no scene 4 created.
        assert len(out["scenes"]) == 3

    def test_does_not_mutate_input(self) -> None:
        base = self._base()
        original = base["scenes"][0]["caption"]
        caption_overlay(base_spec=base, captions=["mutated"])
        assert base["scenes"][0]["caption"] == original

    def test_preserves_title_and_dimensions(self) -> None:
        base = {
            "title": "Keep",
            "width": 1280,
            "height": 720,
            "scenes": [{"start": 0, "duration": 1}],
        }
        out = caption_overlay(base_spec=base, captions=["x"])
        assert out["title"] == "Keep"
        assert out["width"] == 1280
        assert out["height"] == 720

    def test_non_dict_base_returns_empty(self) -> None:
        out = caption_overlay(base_spec="nope", captions=["x"])  # type: ignore[arg-type]
        assert out == {"scenes": []}

    def test_non_list_captions_returns_copy(self) -> None:
        base = self._base()
        out = caption_overlay(base_spec=base, captions="bad")  # type: ignore[arg-type]
        # Spec returned untouched (functional copy).
        assert out["scenes"][0]["caption"] == "old1"


# ---------------------------------------------------------------------------
# MCP-tool handlers
# ---------------------------------------------------------------------------


class TestExplainerHandler:
    @pytest.mark.asyncio
    async def test_returns_html_on_valid_call(self) -> None:
        result = await _video_compose_explainer_handler(
            title="My Explainer",
            sections=["one", "two"],
            cta="CTA",
            run_id="r1",
        )
        assert result["ok"] is True
        assert "<!doctype html>" in result["html"]
        assert result["byte_size"] > 0
        assert result["run_id"] == "r1"
        # Spec returned for downstream piping.
        assert len(result["spec"]["scenes"]) == 4

    @pytest.mark.asyncio
    async def test_rejects_empty_title(self) -> None:
        result = await _video_compose_explainer_handler(title="")
        assert result["ok"] is False
        assert "title" in result["error"]


class TestSocialCutHandler:
    @pytest.mark.asyncio
    async def test_returns_html_on_valid_call(self) -> None:
        result = await _video_compose_social_cut_handler(
            hook="Hook!",
            beats=["b1", "b2"],
            outro="bye",
        )
        assert result["ok"] is True
        assert "<!doctype html>" in result["html"]
        # Default vertical aspect baked into HTML
        assert 'data-width="1080"' in result["html"]
        assert 'data-height="1920"' in result["html"]

    @pytest.mark.asyncio
    async def test_rejects_empty_hook(self) -> None:
        result = await _video_compose_social_cut_handler(hook="")
        assert result["ok"] is False
        assert "hook" in result["error"]


class TestCaptionOverlayHandler:
    @pytest.mark.asyncio
    async def test_glues_captions_into_html(self) -> None:
        base = {
            "title": "B",
            "scenes": [
                {"start": 0, "duration": 2},
                {"start": 2, "duration": 2},
            ],
        }
        result = await _video_caption_overlay_handler(
            base_spec=base,
            captions=["captionA", "captionB"],
        )
        assert result["ok"] is True
        assert "captionA" in result["html"]
        assert "captionB" in result["html"]

    @pytest.mark.asyncio
    async def test_rejects_non_dict_base(self) -> None:
        result = await _video_caption_overlay_handler(
            base_spec="nope",  # type: ignore[arg-type]
            captions=["x"],
        )
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_rejects_non_list_captions(self) -> None:
        result = await _video_caption_overlay_handler(
            base_spec={"scenes": []},
            captions="not a list",  # type: ignore[arg-type]
        )
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Registration — three new tools land on the MCP server
# ---------------------------------------------------------------------------


class _StubServer:
    def __init__(self) -> None:
        self.tools: list[Any] = []

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)


class TestSkillRegistration:
    def test_all_skill_tools_registered(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        names = [t.name for t in srv.tools]
        assert "video_compose_explainer" in names
        assert "video_compose_social_cut" in names
        assert "video_caption_overlay" in names

    def test_skills_are_green_risk(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        for name in (
            "video_compose_explainer",
            "video_compose_social_cut",
            "video_caption_overlay",
        ):
            tool = next(t for t in srv.tools if t.name == name)
            assert tool.annotations["risk_level"] == "green"
            assert tool.annotations["category"] == "video"

    def test_explainer_schema_requires_title(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        tool = next(t for t in srv.tools if t.name == "video_compose_explainer")
        assert "title" in tool.input_schema["required"]

    def test_social_cut_schema_requires_hook(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        tool = next(t for t in srv.tools if t.name == "video_compose_social_cut")
        assert "hook" in tool.input_schema["required"]

    def test_caption_overlay_schema_requires_base_and_captions(self) -> None:
        srv = _StubServer()
        register_video_tools(srv)  # type: ignore[arg-type]
        tool = next(t for t in srv.tools if t.name == "video_caption_overlay")
        required = tool.input_schema["required"]
        assert "base_spec" in required
        assert "captions" in required
