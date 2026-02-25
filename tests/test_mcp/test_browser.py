"""Tests für das Browser-Tool (Playwright MCP)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.browser import (
    BROWSER_TOOL_SCHEMAS,
    BrowserResult,
    BrowserTool,
    MAX_TEXT_LENGTH,
)


class TestBrowserResult:
    def test_default_success(self) -> None:
        r = BrowserResult()
        assert r.success is True
        assert r.text == ""
        assert r.error is None

    def test_error_result(self) -> None:
        r = BrowserResult(success=False, error="Timeout")
        assert r.success is False
        assert r.error == "Timeout"


class TestBrowserToolInit:
    def test_default_config(self) -> None:
        tool = BrowserTool()
        assert tool._headless is True
        assert tool._initialized is False

    def test_custom_config(self, tmp_path: Path) -> None:
        tool = BrowserTool(workspace_dir=tmp_path, headless=False, timeout_ms=5000)
        assert tool._workspace_dir == tmp_path
        assert tool._headless is False
        assert tool._timeout_ms == 5000


class TestBrowserToolNotInitialized:
    """Tests für Aktionen ohne vorherige Initialisierung."""

    @pytest.mark.asyncio
    async def test_navigate_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.navigate("https://example.com")
        assert result.success is False
        assert "nicht initialisiert" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.screenshot()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_click_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.click("#btn")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_fill_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.fill("#input", "test")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_js_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.execute_js("document.title")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_page_info_not_initialized(self) -> None:
        tool = BrowserTool()
        result = await tool.get_page_info()
        assert result.success is False


class TestBrowserToolWithMocks:
    """Tests mit gemocktem Playwright."""

    def _make_initialized_tool(self, tmp_path: Path) -> BrowserTool:
        tool = BrowserTool(workspace_dir=tmp_path)
        tool._initialized = True

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.inner_text = AsyncMock(return_value="Hello World")
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.click = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="result")
        mock_page.screenshot = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.set_default_timeout = MagicMock()

        tool._page = mock_page
        return tool

    @pytest.mark.asyncio
    async def test_navigate_success(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)

        result = await tool.navigate("https://example.com")

        assert result.success is True
        assert result.text == "Hello World"
        assert result.title == "Example"
        assert result.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_without_text_extraction(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)

        result = await tool.navigate("https://example.com", extract_text=False)

        assert result.success is True
        assert result.text == ""
        tool._page.inner_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_navigate_truncates_long_text(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        long_text = "A" * (MAX_TEXT_LENGTH + 500)
        tool._page.inner_text = AsyncMock(return_value=long_text)

        result = await tool.navigate("https://example.com")

        assert result.success is True
        assert len(result.text) < len(long_text)
        assert "gekürzt" in result.text

    @pytest.mark.asyncio
    async def test_navigate_error(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        tool._page.goto = AsyncMock(side_effect=Exception("Network error"))

        result = await tool.navigate("https://bad-url.xyz")

        assert result.success is False
        assert "Navigation fehlgeschlagen" in result.error

    @pytest.mark.asyncio
    async def test_screenshot(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        out_path = str(tmp_path / "test.png")

        result = await tool.screenshot(path=out_path)

        assert result.success is True
        assert result.screenshot_path == out_path
        tool._page.screenshot.assert_awaited_once_with(path=out_path, full_page=False)

    @pytest.mark.asyncio
    async def test_screenshot_auto_path(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)

        result = await tool.screenshot()

        assert result.success is True
        assert result.screenshot_path is not None
        assert "screenshot-" in result.screenshot_path

    @pytest.mark.asyncio
    async def test_click(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)

        result = await tool.click("#submit-btn")

        assert result.success is True
        tool._page.click.assert_awaited_once_with("#submit-btn")

    @pytest.mark.asyncio
    async def test_click_error(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        tool._page.click = AsyncMock(side_effect=Exception("Element not found"))

        result = await tool.click("#missing")

        assert result.success is False
        assert "fehlgeschlagen" in result.error

    @pytest.mark.asyncio
    async def test_fill(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)

        result = await tool.fill("#email", "test@example.com")

        assert result.success is True
        tool._page.fill.assert_awaited_once_with("#email", "test@example.com")

    @pytest.mark.asyncio
    async def test_execute_js(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        tool._page.evaluate = AsyncMock(return_value=42)

        result = await tool.execute_js("1 + 41")

        assert result.success is True
        assert "42" in result.text

    @pytest.mark.asyncio
    async def test_execute_js_truncates_long_result(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        tool._page.evaluate = AsyncMock(return_value="X" * (MAX_TEXT_LENGTH + 100))

        result = await tool.execute_js("long()")

        assert result.success is True
        assert "gekürzt" in result.text

    @pytest.mark.asyncio
    async def test_execute_js_none_result(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        tool._page.evaluate = AsyncMock(return_value=None)

        result = await tool.execute_js("void 0")

        assert result.success is True
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_get_page_info(self, tmp_path: Path) -> None:
        tool = self._make_initialized_tool(tmp_path)
        # Erste evaluate → links, zweite → inputs
        tool._page.evaluate = AsyncMock(
            side_effect=[
                [{"text": "Home", "href": "https://example.com/"}],
                [{"tag": "input", "type": "text", "name": "q", "id": "search", "text": ""}],
            ]
        )

        result = await tool.get_page_info()

        assert result.success is True
        assert "Example" in result.text
        assert "Home" in result.text
        assert "search" in result.text


class TestBrowserToolClose:
    @pytest.mark.asyncio
    async def test_close_not_initialized(self) -> None:
        tool = BrowserTool()
        # Should not raise
        await tool.close()
        assert tool._initialized is False

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        tool = BrowserTool()
        tool._initialized = True
        tool._page = AsyncMock()
        tool._context = AsyncMock()
        tool._browser = AsyncMock()
        tool._playwright = AsyncMock()

        await tool.close()

        assert tool._initialized is False
        assert tool._page is None
        assert tool._browser is None


class TestBrowserToolInitialization:
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self) -> None:
        tool = BrowserTool()
        tool._initialized = True
        assert await tool.initialize() is True

    @pytest.mark.asyncio
    async def test_initialize_returns_false_when_playwright_missing(self) -> None:
        """Testet dass initialize() False zurückgibt wenn Playwright fehlt."""
        tool = BrowserTool()
        # Direkt den Import-Pfad mocken statt builtins.__import__
        original_init = BrowserTool.initialize

        async def patched_init(self: BrowserTool) -> bool:
            # Simuliere ImportError
            return False

        with patch.object(BrowserTool, "initialize", patched_init):
            result = await tool.initialize()
            assert result is False


class TestBrowserToolSchemas:
    def test_all_schemas_present(self) -> None:
        expected = [
            "browse_url",
            "browse_screenshot",
            "browse_click",
            "browse_fill",
            "browse_execute_js",
            "browse_page_info",
        ]
        for name in expected:
            assert name in BROWSER_TOOL_SCHEMAS

    def test_schemas_have_description(self) -> None:
        for name, schema in BROWSER_TOOL_SCHEMAS.items():
            assert "description" in schema, f"{name} hat keine description"
            assert "inputSchema" in schema, f"{name} hat kein inputSchema"

    def test_browse_url_requires_url(self) -> None:
        schema = BROWSER_TOOL_SCHEMAS["browse_url"]
        assert "url" in schema["inputSchema"]["required"]

    def test_browse_click_requires_selector(self) -> None:
        schema = BROWSER_TOOL_SCHEMAS["browse_click"]
        assert "selector" in schema["inputSchema"]["required"]

    def test_browse_fill_requires_selector_and_value(self) -> None:
        schema = BROWSER_TOOL_SCHEMAS["browse_fill"]
        assert "selector" in schema["inputSchema"]["required"]
        assert "value" in schema["inputSchema"]["required"]
