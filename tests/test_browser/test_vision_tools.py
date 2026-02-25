"""Tests für browser/tools.py — Vision MCP-Tools."""

from __future__ import annotations

import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from jarvis.browser.tools import register_browser_use_tools
from jarvis.browser.types import BrowserConfig


# ============================================================================
# Tool-Registrierung
# ============================================================================


class TestVisionToolRegistration:
    def test_vision_tools_registered_with_analyzer(self) -> None:
        """3 Vision-Tools werden registriert wenn vision_analyzer gesetzt ist."""
        mcp_mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            registered_tools[name] = kwargs

        mcp_mock.register_tool = mock_register

        mock_vision = MagicMock()
        agent = register_browser_use_tools(mcp_mock, vision_analyzer=mock_vision)

        assert "browser_vision_analyze" in registered_tools
        assert "browser_vision_find" in registered_tools
        assert "browser_vision_screenshot" in registered_tools

    def test_vision_tools_registered_without_analyzer(self) -> None:
        """Vision-Tools werden auch ohne Analyzer registriert (geben Fehler zurück)."""
        mcp_mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            registered_tools[name] = kwargs

        mcp_mock.register_tool = mock_register

        agent = register_browser_use_tools(mcp_mock)

        # Tools sind trotzdem registriert (graceful degradation)
        assert "browser_vision_analyze" in registered_tools
        assert "browser_vision_find" in registered_tools
        assert "browser_vision_screenshot" in registered_tools

    def test_base_tools_still_registered(self) -> None:
        """Die 9 Basis-Tools sind weiterhin vorhanden."""
        mcp_mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            registered_tools[name] = kwargs

        mcp_mock.register_tool = mock_register

        register_browser_use_tools(mcp_mock)

        base_tools = [
            "browser_navigate", "browser_click", "browser_fill",
            "browser_fill_form", "browser_screenshot", "browser_extract",
            "browser_analyze", "browser_execute_js", "browser_tab",
        ]
        for name in base_tools:
            assert name in registered_tools, f"Missing: {name}"

    def test_total_tool_count(self) -> None:
        """13 Tools insgesamt mit Vision-Analyzer (10 base + 3 vision)."""
        mcp_mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            registered_tools[name] = kwargs

        mcp_mock.register_tool = mock_register

        register_browser_use_tools(mcp_mock, vision_analyzer=MagicMock())
        assert len(registered_tools) == 13


# ============================================================================
# Vision-Tools — Deaktiviert
# ============================================================================


class TestVisionToolsDisabled:
    def _get_handlers(self) -> dict[str, Any]:
        mcp_mock = MagicMock()
        handlers: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            handlers[name] = kwargs.get("handler")

        mcp_mock.register_tool = mock_register
        register_browser_use_tools(mcp_mock)
        return handlers

    @pytest.mark.asyncio
    async def test_vision_analyze_disabled(self) -> None:
        handlers = self._get_handlers()
        handler = handlers["browser_vision_analyze"]

        # Agent not running and no vision → error
        result = json.loads(await handler({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_vision_find_no_description(self) -> None:
        handlers = self._get_handlers()
        handler = handlers["browser_vision_find"]

        result = json.loads(await handler({}))
        assert "error" in result
        assert "description" in result["error"]


# ============================================================================
# Vision-Tools — Handler-Aufrufe
# ============================================================================


class TestVisionAnalyzeTool:
    @pytest.mark.asyncio
    async def test_handler_calls_analyze_page_with_vision(self) -> None:
        mcp_mock = MagicMock()
        handlers: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            handlers[name] = kwargs.get("handler")

        mcp_mock.register_tool = mock_register

        mock_vision = MagicMock()
        mock_vision.is_enabled = True
        agent = register_browser_use_tools(mcp_mock, vision_analyzer=mock_vision)

        # Simulate running agent
        agent._running = True
        agent.analyze_page_with_vision = AsyncMock(return_value={
            "dom": "DOM summary",
            "vision": "Vision desc",
            "combined": "Combined",
        })

        handler = handlers["browser_vision_analyze"]
        result = json.loads(await handler({"prompt": "Test"}))
        assert result["dom"] == "DOM summary"
        assert result["vision"] == "Vision desc"


class TestVisionFindTool:
    @pytest.mark.asyncio
    async def test_handler_calls_find_and_click_with_vision(self) -> None:
        mcp_mock = MagicMock()
        handlers: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            handlers[name] = kwargs.get("handler")

        mcp_mock.register_tool = mock_register

        mock_vision = MagicMock()
        mock_vision.is_enabled = True
        agent = register_browser_use_tools(mcp_mock, vision_analyzer=mock_vision)

        from jarvis.browser.types import ActionResult
        agent._running = True
        agent.find_and_click_with_vision = AsyncMock(return_value=ActionResult(
            action_id="v1", success=True, data={"clicked": True},
        ))

        handler = handlers["browser_vision_find"]
        result = json.loads(await handler({"description": "Login-Button"}))
        assert result["success"] is True


class TestVisionScreenshotTool:
    @pytest.mark.asyncio
    async def test_handler_returns_description(self) -> None:
        mcp_mock = MagicMock()
        handlers: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            handlers[name] = kwargs.get("handler")

        mcp_mock.register_tool = mock_register

        from jarvis.browser.vision import VisionAnalysisResult

        mock_vision = MagicMock()
        mock_vision.is_enabled = True
        mock_vision.analyze_screenshot = AsyncMock(return_value=VisionAnalysisResult(
            success=True, description="Login-Seite mit Formular"
        ))

        agent = register_browser_use_tools(mcp_mock, vision_analyzer=mock_vision)

        from jarvis.browser.types import ActionResult
        agent._running = True
        agent.screenshot = AsyncMock(return_value=ActionResult(
            action_id="ss1", success=True, data={}, screenshot_b64="aGVsbG8="
        ))

        handler = handlers["browser_vision_screenshot"]
        result = json.loads(await handler({}))
        assert result["success"] is True
        assert "Login-Seite" in result["description"]
