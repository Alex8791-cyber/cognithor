"""Coverage-Tests fuer browser.py -- fehlende Pfade abdecken."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.browser import (
    BROWSER_TOOL_SCHEMAS,
    BrowserResult,
    BrowserTool,
    BrowserToolError,
    register_browser_tools,
)


@pytest.fixture
def tool(tmp_path: Path) -> BrowserTool:
    return BrowserTool(workspace_dir=tmp_path)


class TestBrowserToolInit:
    def test_defaults(self) -> None:
        t = BrowserTool()
        assert t._headless is True
        assert not t._initialized

    def test_with_config(self) -> None:
        config = MagicMock()
        config.browser = MagicMock()
        config.browser.max_text_length = 5000
        config.browser.max_js_length = 10_000
        config.browser.default_timeout_ms = 60_000
        config.browser.default_viewport_width = 1920
        config.browser.default_viewport_height = 1080
        t = BrowserTool(config=config)
        assert t._max_text_length == 5000
        assert t._max_js_length == 10_000
        assert t._timeout_ms == 60_000
        assert t._viewport == {"width": 1920, "height": 1080}

    def test_explicit_timeout_overrides_config(self) -> None:
        config = MagicMock()
        config.browser = MagicMock()
        config.browser.default_timeout_ms = 60_000
        config.browser.max_text_length = 8000
        config.browser.max_js_length = 50_000
        config.browser.default_viewport_width = 1280
        config.browser.default_viewport_height = 720
        t = BrowserTool(config=config, timeout_ms=15_000)
        assert t._timeout_ms == 15_000


class TestValidateUrl:
    def test_valid_https(self) -> None:
        assert BrowserTool._validate_url("https://example.com") is None

    def test_valid_http(self) -> None:
        assert BrowserTool._validate_url("http://example.com") is None

    def test_ftp_blocked(self) -> None:
        result = BrowserTool._validate_url("ftp://example.com")
        assert result is not None
        assert "HTTP/HTTPS" in result

    def test_localhost_blocked(self) -> None:
        result = BrowserTool._validate_url("http://localhost/admin")
        assert result is not None
        assert "blockiert" in result

    def test_private_10_blocked(self) -> None:
        result = BrowserTool._validate_url("http://10.0.0.1/")
        assert result is not None
        assert "private" in result

    def test_private_172_blocked(self) -> None:
        result = BrowserTool._validate_url("http://172.16.0.1/")
        assert result is not None

    def test_private_192_blocked(self) -> None:
        result = BrowserTool._validate_url("http://192.168.1.1/")
        assert result is not None

    def test_ipv6_private(self) -> None:
        result = BrowserTool._validate_url("http://fc00::1/")
        assert result is not None

    def test_empty_domain(self) -> None:
        result = BrowserTool._validate_url("https://")
        assert result is not None


class TestNavigate:
    @pytest.mark.asyncio
    async def test_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.navigate("https://example.com")
        assert not result.success
        assert "nicht initialisiert" in result.error

    @pytest.mark.asyncio
    async def test_ssrf_blocked(self, tool: BrowserTool) -> None:
        tool._initialized = True
        result = await tool.navigate("http://localhost/admin")
        assert not result.success
        assert "blockiert" in result.error

    @pytest.mark.asyncio
    async def test_success(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.url = "https://example.com"
        mock_page.inner_text = AsyncMock(return_value="Page content")
        tool._page = mock_page
        result = await tool.navigate("https://example.com")
        assert result.success
        assert result.title == "Test Page"

    @pytest.mark.asyncio
    async def test_exception(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=RuntimeError("timeout"))
        tool._page = mock_page
        result = await tool.navigate("https://example.com")
        assert not result.success
        assert "fehlgeschlagen" in result.error


class TestScreenshot:
    @pytest.mark.asyncio
    async def test_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.screenshot()
        assert not result.success

    @pytest.mark.asyncio
    async def test_success(self, tool: BrowserTool, tmp_path: Path) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.title = AsyncMock(return_value="Page")
        mock_page.url = "https://example.com"
        tool._page = mock_page
        result = await tool.screenshot(path=str(tmp_path / "test.png"))
        assert result.success
        assert result.screenshot_path is not None

    @pytest.mark.asyncio
    async def test_exception(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=RuntimeError("fail"))
        tool._page = mock_page
        result = await tool.screenshot(path=str(Path(tempfile.gettempdir()) / "test.png"))
        assert not result.success


class TestClickAndFill:
    @pytest.mark.asyncio
    async def test_click_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.click("#btn")
        assert not result.success

    @pytest.mark.asyncio
    async def test_click_success(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.title = AsyncMock(return_value="P")
        mock_page.url = "https://example.com"
        tool._page = mock_page
        result = await tool.click("#btn")
        assert result.success

    @pytest.mark.asyncio
    async def test_click_exception(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.click = AsyncMock(side_effect=RuntimeError("not found"))
        tool._page = mock_page
        result = await tool.click("#nonexistent")
        assert not result.success

    @pytest.mark.asyncio
    async def test_fill_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.fill("#input", "value")
        assert not result.success

    @pytest.mark.asyncio
    async def test_fill_success(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.url = "https://example.com"
        tool._page = mock_page
        result = await tool.fill("#input", "test")
        assert result.success

    @pytest.mark.asyncio
    async def test_fill_exception(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock(side_effect=RuntimeError("fail"))
        tool._page = mock_page
        result = await tool.fill("#x", "y")
        assert not result.success


class TestExecuteJs:
    @pytest.mark.asyncio
    async def test_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.execute_js("1+1")
        assert not result.success

    @pytest.mark.asyncio
    async def test_script_too_long(self, tool: BrowserTool) -> None:
        tool._initialized = True
        tool._max_js_length = 10
        result = await tool.execute_js("x" * 100)
        assert not result.success
        assert "zu lang" in result.error

    @pytest.mark.asyncio
    async def test_success(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=42)
        mock_page.url = "https://example.com"
        tool._page = mock_page
        result = await tool.execute_js("1+1")
        assert result.success
        assert "42" in result.text

    @pytest.mark.asyncio
    async def test_none_result(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)
        mock_page.url = "https://example.com"
        tool._page = mock_page
        result = await tool.execute_js("void(0)")
        assert result.success
        assert result.text == ""


class TestGetPageInfo:
    @pytest.mark.asyncio
    async def test_not_initialized(self, tool: BrowserTool) -> None:
        result = await tool.get_page_info()
        assert not result.success

    @pytest.mark.asyncio
    async def test_success(self, tool: BrowserTool) -> None:
        tool._initialized = True
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(
            side_effect=[
                [{"text": "Home", "href": "https://example.com/"}],
                [{"tag": "input", "type": "text", "name": "q", "id": "search", "text": ""}],
            ]
        )
        tool._page = mock_page
        result = await tool.get_page_info()
        assert result.success
        assert "Titel: Test" in result.text


class TestInitializeAndClose:
    @pytest.mark.asyncio
    async def test_initialize_playwright_not_installed(self, tool: BrowserTool) -> None:
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            result = await tool.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_already_initialized(self, tool: BrowserTool) -> None:
        tool._initialized = True
        result = await tool.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self, tool: BrowserTool) -> None:
        tool._initialized = True
        tool._page = AsyncMock()
        tool._context = AsyncMock()
        tool._browser = AsyncMock()
        tool._playwright = AsyncMock()
        await tool.close()
        assert not tool._initialized


class TestRegisterBrowserTools:
    def test_registers_all_tools(self) -> None:
        mock_client = MagicMock()
        tool = register_browser_tools(mock_client)
        assert isinstance(tool, BrowserTool)
        assert mock_client.register_builtin_handler.call_count == len(BROWSER_TOOL_SCHEMAS)

    def test_tool_names(self) -> None:
        mock_client = MagicMock()
        register_browser_tools(mock_client)
        registered = [call.args[0] for call in mock_client.register_builtin_handler.call_args_list]
        assert "browse_url" in registered
        assert "browse_screenshot" in registered
        assert "browse_click" in registered
        assert "browse_fill" in registered
        assert "browse_execute_js" in registered
        assert "browse_page_info" in registered
