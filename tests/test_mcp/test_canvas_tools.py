"""Tests für MCP Canvas Tools."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from jarvis.channels.canvas import CanvasManager
from jarvis.mcp.canvas_tools import CanvasTools


@pytest.fixture
def canvas_manager() -> CanvasManager:
    return CanvasManager()


@pytest.fixture
def canvas_tools(canvas_manager: CanvasManager) -> CanvasTools:
    return CanvasTools(canvas_manager)


class TestCanvasToolDefinitions:
    """Tests für Tool-Definitionen."""

    def test_has_four_tools(self, canvas_tools: CanvasTools) -> None:
        defs = canvas_tools.tool_definitions
        assert len(defs) == 4

    def test_tool_names(self, canvas_tools: CanvasTools) -> None:
        names = {d["name"] for d in canvas_tools.tool_definitions}
        assert names == {"canvas_push", "canvas_reset", "canvas_snapshot", "canvas_eval"}

    def test_all_have_schema(self, canvas_tools: CanvasTools) -> None:
        for d in canvas_tools.tool_definitions:
            assert "inputSchema" in d
            assert d["inputSchema"]["type"] == "object"

    def test_push_requires_html(self, canvas_tools: CanvasTools) -> None:
        push_def = next(d for d in canvas_tools.tool_definitions if d["name"] == "canvas_push")
        assert "html" in push_def["inputSchema"]["required"]

    def test_eval_requires_js(self, canvas_tools: CanvasTools) -> None:
        eval_def = next(d for d in canvas_tools.tool_definitions if d["name"] == "canvas_eval")
        assert "js" in eval_def["inputSchema"]["required"]


class TestCanvasToolPush:
    """Tests für canvas_push Tool."""

    @pytest.mark.asyncio
    async def test_push_success(self, canvas_tools: CanvasTools) -> None:
        result = await canvas_tools.handle_tool_call(
            "canvas_push",
            {"html": "<h1>Hello</h1>", "title": "Test"},
            "session_1",
        )
        assert result["success"] is True
        assert "aktualisiert" in result["message"]

    @pytest.mark.asyncio
    async def test_push_empty_html(self, canvas_tools: CanvasTools) -> None:
        result = await canvas_tools.handle_tool_call(
            "canvas_push",
            {"html": ""},
            "session_1",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_push_updates_canvas(
        self, canvas_tools: CanvasTools, canvas_manager: CanvasManager,
    ) -> None:
        await canvas_tools.handle_tool_call(
            "canvas_push",
            {"html": "<p>Content</p>", "title": "My Title"},
            "session_1",
        )
        html = await canvas_manager.snapshot("session_1")
        assert html == "<p>Content</p>"
        assert canvas_manager.get_title("session_1") == "My Title"


class TestCanvasToolReset:
    """Tests für canvas_reset Tool."""

    @pytest.mark.asyncio
    async def test_reset_success(self, canvas_tools: CanvasTools) -> None:
        await canvas_tools.handle_tool_call(
            "canvas_push", {"html": "<p>Content</p>"}, "session_1",
        )
        result = await canvas_tools.handle_tool_call(
            "canvas_reset", {}, "session_1",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_reset_clears_canvas(
        self, canvas_tools: CanvasTools, canvas_manager: CanvasManager,
    ) -> None:
        await canvas_tools.handle_tool_call(
            "canvas_push", {"html": "<p>Content</p>"}, "session_1",
        )
        await canvas_tools.handle_tool_call("canvas_reset", {}, "session_1")
        html = await canvas_manager.snapshot("session_1")
        assert html == ""


class TestCanvasToolSnapshot:
    """Tests für canvas_snapshot Tool."""

    @pytest.mark.asyncio
    async def test_snapshot_empty(self, canvas_tools: CanvasTools) -> None:
        result = await canvas_tools.handle_tool_call(
            "canvas_snapshot", {}, "session_1",
        )
        assert result["success"] is True
        assert result["html"] == ""

    @pytest.mark.asyncio
    async def test_snapshot_with_content(self, canvas_tools: CanvasTools) -> None:
        await canvas_tools.handle_tool_call(
            "canvas_push", {"html": "<div>Test</div>"}, "session_1",
        )
        result = await canvas_tools.handle_tool_call(
            "canvas_snapshot", {}, "session_1",
        )
        assert result["success"] is True
        assert result["html"] == "<div>Test</div>"
        assert result["length"] == len("<div>Test</div>")


class TestCanvasToolEval:
    """Tests für canvas_eval Tool."""

    @pytest.mark.asyncio
    async def test_eval_success(self) -> None:
        broadcaster = AsyncMock()
        cm = CanvasManager(broadcaster=broadcaster)
        tools = CanvasTools(cm)

        result = await tools.handle_tool_call(
            "canvas_eval",
            {"js": "document.title = 'New Title'"},
            "session_1",
        )
        assert result["success"] is True
        broadcaster.assert_called_once()

    @pytest.mark.asyncio
    async def test_eval_empty_js(self, canvas_tools: CanvasTools) -> None:
        result = await canvas_tools.handle_tool_call(
            "canvas_eval", {"js": ""}, "session_1",
        )
        assert "error" in result


class TestCanvasToolUnknown:
    """Tests für unbekannte Tools."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self, canvas_tools: CanvasTools) -> None:
        result = await canvas_tools.handle_tool_call(
            "canvas_unknown", {}, "session_1",
        )
        assert "error" in result
