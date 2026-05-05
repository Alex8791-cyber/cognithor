"""Deep coverage for ``cognithor.mcp.browser``.

The browser tool is security-sensitive: its public surface includes the SSRF
guards that decide whether the agent's outbound HTTP requests reach an
internal address. PR #479 (PASS-4 SEC-CRIT-1) introduced the DNS-resolution
layer (`_validate_resolved_host`) on top of the existing hostname-string
check (`_validate_url`). This file exercises both layers exhaustively
plus the navigation, screenshot, click, fill, JS, page-info, and
registration paths.

Continuation of Wave-1/2 backend deep coverage (PRs #486, #488).
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.mcp.browser import (
    BROWSER_TOOL_SCHEMAS,
    BrowserResult,
    BrowserTool,
    BrowserToolError,
    register_browser_tools,
)

if TYPE_CHECKING:
    from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_page(
    *,
    url: str = "https://example.com",
    title: str = "Example Domain",
    body: str = "Hello, world.",
) -> Any:
    """Build an AsyncMock playwright page that returns the given values."""
    page = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.inner_text = AsyncMock(return_value=body)
    page.goto = AsyncMock(return_value=MagicMock(status=200))
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    page.screenshot = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.set_default_timeout = MagicMock()
    return page


def _initialized_tool(tmp_path: Path, page: Any | None = None) -> BrowserTool:
    """Build a BrowserTool with `_initialized=True` and a mock page."""
    tool = BrowserTool(workspace_dir=tmp_path)
    tool._initialized = True
    tool._page = page or _mock_page()
    return tool


def _addrinfo(ip: str, family: int = socket.AF_INET) -> list[tuple[Any, ...]]:
    """Build a fake `socket.getaddrinfo` reply for the given IP."""
    if family == socket.AF_INET:
        return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]
    # IPv6
    return [(family, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]


# ─────────────────────────────────────────────────────────────────────────────
# Module surface / dataclass / exception
# ─────────────────────────────────────────────────────────────────────────────


class TestBrowserResultDataclass:
    def test_default_is_success(self) -> None:
        r = BrowserResult()
        assert r.success is True
        assert r.text == ""
        assert r.url == ""
        assert r.title == ""
        assert r.screenshot_path is None
        assert r.error is None

    def test_failure_carries_error(self) -> None:
        r = BrowserResult(success=False, url="x", error="boom")
        assert r.success is False
        assert r.error == "boom"

    def test_browser_tool_error_is_exception(self) -> None:
        # Sanity: explicit exception export.
        with pytest.raises(BrowserToolError):
            raise BrowserToolError("synthetic")


# ─────────────────────────────────────────────────────────────────────────────
# _validate_url — hostname-string SSRF guard
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateUrlSchemes:
    @pytest.mark.parametrize(
        "url",
        ["http://example.com/", "https://example.com/", "https://sub.example.com/path?q=1"],
    )
    def test_http_https_accepted(self, url: str) -> None:
        assert BrowserTool._validate_url(url) is None

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/file",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "gopher://example.com/",
            "ws://example.com/socket",
            "wss://example.com/socket",
        ],
    )
    def test_non_http_schemes_refused(self, url: str) -> None:
        err = BrowserTool._validate_url(url)
        assert err is not None


class TestValidateUrlHostnameBlocks:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1/",
            "http://0.0.0.0/",
            "http://169.254.169.254/latest/meta-data/",  # AWS IMDS
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://[::1]/",
        ],
    )
    def test_blocked_hostnames(self, url: str) -> None:
        err = BrowserTool._validate_url(url)
        assert err is not None

    @pytest.mark.parametrize(
        "url,family",
        [
            ("http://10.0.0.1/", "10/8"),
            ("http://10.255.255.255/", "10/8"),
            ("http://172.16.0.1/", "172.16/12"),
            ("http://172.31.255.255/", "172.16/12"),
            ("http://192.168.0.1/", "192.168/16"),
            ("http://192.168.255.255/", "192.168/16"),
        ],
    )
    def test_rfc1918_ranges_refused(self, url: str, family: str) -> None:
        err = BrowserTool._validate_url(url)
        assert err is not None, f"{url} ({family}) should be blocked"

    @pytest.mark.parametrize(
        "url",
        [
            "http://[fc00::1]/",
            "http://[fd12:3456:789a::1]/",
            "http://[fe80::1]/",
        ],
    )
    def test_ipv6_private_refused(self, url: str) -> None:
        err = BrowserTool._validate_url(url)
        assert err is not None

    def test_no_hostname_refused(self) -> None:
        err = BrowserTool._validate_url("https:///nopath")
        assert err is not None

    def test_172_15_is_not_private(self) -> None:
        # Boundary: 172.15/16 is NOT in RFC1918 range (172.16-31).
        # The string-only check must let it through (DNS layer may still block).
        assert BrowserTool._validate_url("http://172.15.0.1/") is None

    def test_172_32_is_not_private(self) -> None:
        # Other side of the boundary.
        assert BrowserTool._validate_url("http://172.32.0.1/") is None

    def test_hostname_case_insensitive(self) -> None:
        # ``LOCALHOST`` should still be caught.
        err = BrowserTool._validate_url("http://LOCALHOST/")
        assert err is not None


# ─────────────────────────────────────────────────────────────────────────────
# _validate_resolved_host — DNS-layer SSRF guard (PASS-4 SEC-CRIT-1)
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateResolvedHost:
    @pytest.mark.asyncio
    async def test_literal_ip_skipped(self) -> None:
        # Literal IPs already pass through ``_validate_url``; resolution is
        # short-circuited so no socket call needed.
        assert await BrowserTool._validate_resolved_host("http://93.184.216.34/") is None

    @pytest.mark.asyncio
    async def test_no_hostname_refused(self) -> None:
        err = await BrowserTool._validate_resolved_host("https:///")
        assert err is not None

    @pytest.mark.asyncio
    async def test_resolves_to_loopback_blocked(self) -> None:
        # DNS-rebinding: hostname is innocuous but resolves to 127.x.
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("127.0.0.1")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://inner.evil.com/")
        assert err is not None

    @pytest.mark.asyncio
    async def test_resolves_to_rfc1918_blocked(self) -> None:
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("10.0.0.1")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://intranet.example/")
        assert err is not None

    @pytest.mark.asyncio
    async def test_resolves_to_link_local_blocked(self) -> None:
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("169.254.1.1")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://aws-meta.evil/")
        assert err is not None

    @pytest.mark.asyncio
    async def test_resolves_to_ipv6_loopback_blocked(self) -> None:
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("::1", family=socket.AF_INET6)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://v6loop.evil/")
        assert err is not None

    @pytest.mark.asyncio
    async def test_ipv6_zone_id_stripped(self) -> None:
        # `fe80::1%eth0` style addresses with zone IDs: the impl strips the
        # zone before parsing so the ``ipaddress`` module accepts it.
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fe80::1%eth0", 0, 0, 0))]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://link-local.evil/")
        assert err is not None

    @pytest.mark.asyncio
    async def test_resolution_failure_is_passed_through(self) -> None:
        # NXDOMAIN must NOT be reported as an SSRF block — the navigate call
        # surfaces the real error so users can debug.
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            raise OSError("nodename nor servname provided")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://nx.example.invalid/")
        assert err is None

    @pytest.mark.asyncio
    async def test_public_ip_passes(self) -> None:
        # 93.184.216.34 (example.com) is global — must pass.
        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("93.184.216.34")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            err = await BrowserTool._validate_resolved_host("http://example.com/")
        assert err is None


# ─────────────────────────────────────────────────────────────────────────────
# navigate() — SSRF + happy path + truncation + error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestNavigate:
    @pytest.mark.asyncio
    async def test_uninitialised_returns_error(self, tmp_path: Path) -> None:
        tool = BrowserTool(workspace_dir=tmp_path)
        result = await tool.navigate("https://example.com/")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "http://localhost/",
            "http://127.0.0.1/",
            "http://169.254.169.254/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
        ],
    )
    async def test_string_layer_ssrf_blocked(self, tmp_path: Path, url: str) -> None:
        tool = _initialized_tool(tmp_path)
        result = await tool.navigate(url)
        assert result.success is False
        # The page mock should never have been touched.
        tool._page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_dns_layer_ssrf_blocked(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)

        async def fake_getaddrinfo(*_: Any, **__: Any) -> list[tuple[Any, ...]]:
            return _addrinfo("127.0.0.1")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(side_effect=fake_getaddrinfo)
            result = await tool.navigate("http://rebind.evil.test/")

        assert result.success is False
        tool._page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_extracts_text(self, tmp_path: Path) -> None:
        page = _mock_page(body="The quick brown fox.")
        tool = _initialized_tool(tmp_path, page)
        result = await tool.navigate("https://example.com/")
        assert result.success is True
        assert "fox" in result.text
        assert result.title == "Example Domain"

    @pytest.mark.asyncio
    async def test_extract_text_false_skips_inner_text(self, tmp_path: Path) -> None:
        page = _mock_page()
        tool = _initialized_tool(tmp_path, page)
        result = await tool.navigate("https://example.com/", extract_text=False)
        assert result.success is True
        assert result.text == ""
        page.inner_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_text_truncated(self, tmp_path: Path) -> None:
        # max_text_length defaults to 8000; produce double that.
        long_body = "A" * 16_000
        page = _mock_page(body=long_body)
        tool = _initialized_tool(tmp_path, page)
        result = await tool.navigate("https://example.com/")
        assert result.success is True
        assert len(result.text) < len(long_body)

    @pytest.mark.asyncio
    async def test_goto_exception_returns_error(self, tmp_path: Path) -> None:
        page = _mock_page()
        page.goto = AsyncMock(side_effect=RuntimeError("connection reset"))
        tool = _initialized_tool(tmp_path, page)
        result = await tool.navigate("https://example.com/")
        assert result.success is False
        # The exception type leaks into the error string but the actual
        # message ("connection reset") must NOT — we don't want to leak
        # internal stack info to the LLM.
        assert "connection reset" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_goto_returns_none_response(self, tmp_path: Path) -> None:
        # ``goto`` may return ``None`` for about:blank etc; module reads
        # ``response.status`` only when truthy.
        page = _mock_page()
        page.goto = AsyncMock(return_value=None)
        tool = _initialized_tool(tmp_path, page)
        result = await tool.navigate("https://example.com/")
        assert result.success is True


# ─────────────────────────────────────────────────────────────────────────────
# screenshot() / click() / fill() / execute_js()
# ─────────────────────────────────────────────────────────────────────────────


class TestScreenshot:
    @pytest.mark.asyncio
    async def test_default_path_uses_workspace(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)
        result = await tool.screenshot()
        assert result.success is True
        assert result.screenshot_path is not None
        # Auto-generated path must live in the workspace dir.
        assert str(tmp_path) in (result.screenshot_path or "")

    @pytest.mark.asyncio
    async def test_explicit_path_used(self, tmp_path: Path) -> None:
        out = tmp_path / "shot.png"
        tool = _initialized_tool(tmp_path)
        result = await tool.screenshot(path=str(out))
        assert result.success is True
        assert result.screenshot_path == str(out)
        tool._page.screenshot.assert_awaited_once_with(path=str(out), full_page=False)

    @pytest.mark.asyncio
    async def test_full_page_passed_through(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)
        await tool.screenshot(path=str(tmp_path / "full.png"), full_page=True)
        tool._page.screenshot.assert_awaited_once_with(
            path=str(tmp_path / "full.png"), full_page=True
        )


class TestExecuteJs:
    @pytest.mark.asyncio
    async def test_long_script_refused(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)
        tool._max_js_length = 50
        result = await tool.execute_js("a" * 200)
        assert result.success is False
        assert tool._page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_long_result_truncated(self, tmp_path: Path) -> None:
        page = _mock_page()
        # Produce a string longer than the configured limit.
        page.evaluate = AsyncMock(return_value="X" * 20_000)
        tool = _initialized_tool(tmp_path, page)
        tool._max_text_length = 100
        result = await tool.execute_js("longString()")
        assert result.success is True
        # Truncated payload + suffix marker — definitely shorter than 20k.
        assert len(result.text) < 1_000

    @pytest.mark.asyncio
    async def test_js_exception_surfaces(self, tmp_path: Path) -> None:
        page = _mock_page()
        page.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        tool = _initialized_tool(tmp_path, page)
        result = await tool.execute_js("1+1")
        assert result.success is False
        # Error type leaks but raw message must not.
        assert "boom" not in (result.error or "")


class TestClickFillPageInfo:
    @pytest.mark.asyncio
    async def test_click_passes_selector(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)
        result = await tool.click("#submit")
        assert result.success is True
        tool._page.click.assert_awaited_once_with("#submit")

    @pytest.mark.asyncio
    async def test_fill_passes_selector_and_value(self, tmp_path: Path) -> None:
        tool = _initialized_tool(tmp_path)
        result = await tool.fill("#email", "user@example.com")
        assert result.success is True
        tool._page.fill.assert_awaited_once_with("#email", "user@example.com")

    @pytest.mark.asyncio
    async def test_page_info_collects_links_and_inputs(self, tmp_path: Path) -> None:
        page = _mock_page(title="Search Page")
        page.evaluate = AsyncMock(
            side_effect=[
                [{"text": "Home", "href": "https://example.com/"}],
                [
                    {
                        "tag": "input",
                        "type": "text",
                        "name": "q",
                        "id": "search",
                        "text": "",
                    }
                ],
            ]
        )
        tool = _initialized_tool(tmp_path, page)
        result = await tool.get_page_info()
        assert result.success is True
        assert "Home" in result.text
        assert "search" in result.text


# ─────────────────────────────────────────────────────────────────────────────
# initialize() / close() lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_idempotent_when_initialized(self, tmp_path: Path) -> None:
        tool = BrowserTool(workspace_dir=tmp_path)
        tool._initialized = True
        assert await tool.initialize() is True

    @pytest.mark.asyncio
    async def test_initialize_returns_false_when_playwright_missing(self, tmp_path: Path) -> None:
        tool = BrowserTool(workspace_dir=tmp_path)
        # Force the late `from playwright.async_api import async_playwright`
        # to fail — register `None` shim modules which Python imports as
        # missing-attr ImportError.
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            assert await tool.initialize() is False
        assert tool._initialized is False

    @pytest.mark.asyncio
    async def test_close_when_uninitialised_is_safe(self) -> None:
        tool = BrowserTool()
        await tool.close()  # must not raise
        assert tool._initialized is False

    @pytest.mark.asyncio
    async def test_close_resets_handles(self, tmp_path: Path) -> None:
        tool = BrowserTool(workspace_dir=tmp_path)
        tool._initialized = True
        tool._page = AsyncMock()
        tool._context = AsyncMock()
        tool._browser = AsyncMock()
        tool._playwright = AsyncMock()
        await tool.close()
        assert tool._initialized is False
        assert tool._page is None
        assert tool._context is None
        assert tool._browser is None


# ─────────────────────────────────────────────────────────────────────────────
# register_browser_tools — wires up BROWSER_TOOL_SCHEMAS to a client
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_returns_browsertool(self) -> None:
        client = MagicMock()
        tool = register_browser_tools(client)
        assert isinstance(tool, BrowserTool)

    def test_register_calls_client_for_each_schema(self) -> None:
        client = MagicMock()
        register_browser_tools(client)
        assert client.register_builtin_handler.call_count == len(BROWSER_TOOL_SCHEMAS)
        registered = {call.args[0] for call in client.register_builtin_handler.call_args_list}
        assert registered == set(BROWSER_TOOL_SCHEMAS.keys())

    def test_schemas_cover_all_public_handlers(self) -> None:
        # Every schema must declare a description and an inputSchema.
        for name, schema in BROWSER_TOOL_SCHEMAS.items():
            assert schema["description"], f"{name} missing description"
            assert "inputSchema" in schema, f"{name} missing inputSchema"
            assert schema["inputSchema"]["type"] == "object"
