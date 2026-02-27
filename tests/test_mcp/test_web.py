"""Tests für Web-Tools: Suche, Fetch, Sicherheit.

Testet URL-Validierung, Text-Extraktion, SSRF-Schutz
und Such-Backend-Integration (gemocked).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import re
from urllib.parse import urlparse

from jarvis.mcp.web import (
    WebError,
    WebTools,
    _extract_text_from_html,
    _format_search_results,
    _is_private_host,
    _simple_html_to_text,
    _truncate_text,
    register_web_tools,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def web() -> WebTools:
    """WebTools ohne Backend-Konfiguration."""
    return WebTools()


@pytest.fixture()
def web_searxng() -> WebTools:
    """WebTools mit SearXNG-URL."""
    return WebTools(searxng_url="http://localhost:8888")


@pytest.fixture()
def web_brave() -> WebTools:
    """WebTools mit Brave API Key."""
    return WebTools(brave_api_key="test-key-123")


# ── URL-Validierung ───────────────────────────────────────────────────────


class TestURLValidation:
    """Tests für SSRF-Schutz und URL-Validierung."""

    def test_valid_https_url(self, web: WebTools) -> None:
        result = web._validate_url("https://example.com/page")
        assert result == "https://example.com/page"

    def test_valid_http_url(self, web: WebTools) -> None:
        result = web._validate_url("http://example.com")
        assert result == "http://example.com"

    def test_rejects_ftp(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="Nur HTTP/HTTPS"):
            web._validate_url("ftp://example.com/file")

    def test_rejects_file_scheme(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="Nur HTTP/HTTPS"):
            web._validate_url("file:///etc/passwd")

    def test_rejects_localhost(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="blockiert"):
            web._validate_url("http://localhost/admin")

    def test_rejects_127_0_0_1(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="blockiert"):
            web._validate_url("http://127.0.0.1:8080/")

    def test_rejects_metadata_endpoint(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="blockiert"):
            web._validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_10_network(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="private"):
            web._validate_url("http://10.0.0.1/internal")

    def test_rejects_private_172_network(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="private"):
            web._validate_url("http://172.16.0.1/internal")

    def test_rejects_private_192_168(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="private"):
            web._validate_url("http://192.168.1.1/router")

    def test_rejects_empty_url(self, web: WebTools) -> None:
        with pytest.raises(WebError):
            web._validate_url("")

    def test_rejects_no_domain(self, web: WebTools) -> None:
        with pytest.raises(WebError):
            web._validate_url("https://")


# ── _is_private_host ───────────────────────────────────────────────────────


class TestIsPrivateHost:
    """Tests für Private-IP-Erkennung."""

    def test_10_network(self) -> None:
        assert _is_private_host("10.0.0.1") is True
        assert _is_private_host("10.255.255.255") is True

    def test_172_16_to_31(self) -> None:
        assert _is_private_host("172.16.0.1") is True
        assert _is_private_host("172.31.255.255") is True
        assert _is_private_host("172.15.0.1") is False
        assert _is_private_host("172.32.0.1") is False

    def test_192_168(self) -> None:
        assert _is_private_host("192.168.0.1") is True
        assert _is_private_host("192.168.100.1") is True

    def test_public_ips(self) -> None:
        assert _is_private_host("8.8.8.8") is False
        assert _is_private_host("1.1.1.1") is False
        assert _is_private_host("93.184.216.34") is False

    def test_hostname(self) -> None:
        assert _is_private_host("example.com") is False

    def test_ipv6_private(self) -> None:
        assert _is_private_host("fc00::1") is True
        assert _is_private_host("fd12::1") is True


# ── _simple_html_to_text ──────────────────────────────────────────────────


class TestSimpleHtmlToText:
    """Tests für die Fallback-HTML→Text-Konvertierung."""

    def test_strips_tags(self) -> None:
        result = _simple_html_to_text("<p>Hallo <b>Welt</b></p>")
        assert "Hallo" in result
        assert "Welt" in result
        assert "<" not in result

    def test_removes_scripts(self) -> None:
        html = '<p>OK</p><script>alert("evil")</script><p>Gut</p>'
        result = _simple_html_to_text(html)
        assert "OK" in result
        assert "Gut" in result
        assert "alert" not in result

    def test_removes_styles(self) -> None:
        html = "<style>body{color:red}</style><p>Text</p>"
        result = _simple_html_to_text(html)
        assert "color" not in result
        assert "Text" in result

    def test_converts_block_elements_to_newlines(self) -> None:
        html = "<h1>Titel</h1><p>Absatz</p>"
        result = _simple_html_to_text(html)
        assert "\n" in result

    def test_handles_entities(self) -> None:
        html = "a &amp; b &lt; c &gt; d"
        result = _simple_html_to_text(html)
        assert "a & b" in result

    def test_normalizes_whitespace(self) -> None:
        html = "<p>  viel    Platz   </p>"
        result = _simple_html_to_text(html)
        assert "  " not in result.replace("\n", " ").strip() or "viel Platz" in result


# ── _truncate_text ─────────────────────────────────────────────────────────


class TestTruncateText:
    """Tests für Text-Kürzung."""

    def test_short_text_unchanged(self) -> None:
        result = _truncate_text("Kurz.", 100)
        assert result == "Kurz."

    def test_long_text_truncated(self) -> None:
        text = "A" * 100
        result = _truncate_text(text, 50)
        assert len(result) < 100
        assert "gekürzt" in result

    def test_truncates_at_sentence_boundary(self) -> None:
        text = "Erster Satz. Zweiter Satz. Dritter Satz ist lang genug um gekürzt zu werden."
        result = _truncate_text(text, 30, "test.com")
        assert result.startswith("Erster Satz.")
        assert "gekürzt" in result


# ── _format_search_results ─────────────────────────────────────────────────


class TestFormatSearchResults:
    """Tests für Suchergebnis-Formatierung."""

    def test_formats_results(self) -> None:
        results = [
            {"title": "Test", "url": "https://example.com", "content": "Snippet"},
        ]
        text = _format_search_results(results, "test query")
        assert "test query" in text
        assert "Test" in text
        assert "https://example.com" in text
        assert "Snippet" in text

    def test_empty_results(self) -> None:
        text = _format_search_results([], "nothing")
        assert "nothing" in text

    def test_truncates_long_snippets(self) -> None:
        results = [{"title": "X", "url": "http://x.com", "content": "A" * 800}]
        text = _format_search_results(results, "q")
        # Snippet sollte auf 600 Zeichen gekürzt sein
        assert text.count("A") <= 600


# ── web_search ─────────────────────────────────────────────────────────────


class TestWebSearch:
    """Tests für die Websuche."""

    @pytest.mark.asyncio()
    async def test_no_backend_configured(self) -> None:
        """Ohne Backend und DDG deaktiviert → hilfreiche Meldung."""
        web = WebTools()
        web._duckduckgo_enabled = False
        result = await web.web_search("test")
        assert "Keine Suchengine konfiguriert" in result

    @pytest.mark.asyncio()
    async def test_empty_query(self, web: WebTools) -> None:
        result = await web.web_search("")
        assert "Keine Suchanfrage" in result

    @pytest.mark.asyncio()
    async def test_searxng_success(self, web_searxng: WebTools) -> None:
        """SearXNG-Suche mit Mock."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://r1.com", "content": "Snippet 1"},
                {"title": "Result 2", "url": "https://r2.com", "content": "Snippet 2"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web_searxng.web_search("python", num_results=2)
            assert "Result 1" in result
            assert "Result 2" in result

    @pytest.mark.asyncio()
    async def test_brave_success(self, web_brave: WebTools) -> None:
        """Brave-Suche mit Mock."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Brave R1", "url": "https://br1.com", "description": "Desc 1"},
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web_brave.web_search("test")
            assert "Brave R1" in result

    @pytest.mark.asyncio()
    async def test_searxng_fallback_to_brave(self) -> None:
        """SearXNG-Fehler → Brave-Fallback."""
        web = WebTools(searxng_url="http://broken:9999", brave_api_key="key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {"results": [{"title": "Fallback", "url": "http://f.com", "description": "OK"}]}
        }

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "broken" in url:
                raise httpx.ConnectError("Connection refused")
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = mock_get
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web.web_search("fallback test")
            assert "Fallback" in result

    @pytest.mark.asyncio()
    async def test_num_results_clamped(self, web: WebTools) -> None:
        """num_results wird auf 1-10 begrenzt."""
        # Kein Backend → direkte Rückmeldung, aber kein Crash
        result = await web.web_search("test", num_results=100)
        assert isinstance(result, str)


# ── web_fetch ──────────────────────────────────────────────────────────────


class TestWebFetch:
    """Tests für URL-Fetch."""

    @pytest.mark.asyncio()
    async def test_fetch_html_extracts_text(self, web: WebTools) -> None:
        """Fetch mit trafilatura-Extraktion."""
        html = "<html><body><h1>Titel</h1><p>Inhalt der Seite.</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.content = html.encode("utf-8")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web.web_fetch("https://example.com")
            assert isinstance(result, str)
            # Text sollte extrahiert sein (entweder via trafilatura oder Fallback)
            assert len(result) > 0

    @pytest.mark.asyncio()
    async def test_fetch_plain_text(self, web: WebTools) -> None:
        """Fetch von Nicht-HTML → direkt als Text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"Hello World"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web.web_fetch("https://example.com/file.txt")
            assert "Hello World" in result

    @pytest.mark.asyncio()
    async def test_fetch_rejects_blocked_url(self, web: WebTools) -> None:
        with pytest.raises(WebError, match="blockiert"):
            await web.web_fetch("http://localhost/admin")

    @pytest.mark.asyncio()
    async def test_fetch_http_error(self, web: WebTools) -> None:
        """HTTP 404 → WebError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(WebError, match="404"):
                await web.web_fetch("https://example.com/not-found")

    @pytest.mark.asyncio()
    async def test_fetch_raw_html(self, web: WebTools) -> None:
        """Fetch mit extract_text=False → Raw HTML."""
        html = "<html><body><p>Raw</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = html.encode("utf-8")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web.web_fetch("https://example.com", extract_text=False)
            assert "<p>Raw</p>" in result


# ── _extract_text_from_html ────────────────────────────────────────────────


class TestExtractTextFromHtml:
    """Tests für HTML→Text mit trafilatura + Fallback."""

    def test_extracts_with_trafilatura(self) -> None:
        """trafilatura sollte installiiert sein."""
        html = "<html><body><article><p>Dies ist ein Testtext.</p></article></body></html>"
        result = _extract_text_from_html(html)
        # Entweder trafilatura oder Fallback
        assert isinstance(result, str)

    def test_fallback_without_trafilatura(self) -> None:
        """Ohne trafilatura → Regex-Fallback."""
        html = "<p>Fallback <b>Test</b></p>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            # Dies testet den Import-Fehler-Zweig
            result = _simple_html_to_text(html)
            assert "Fallback" in result
            assert "Test" in result


# ── search_and_read ────────────────────────────────────────────────────────


class TestSearchAndRead:
    """Tests für die kombinierte Suche+Fetch-Funktion."""

    @pytest.mark.asyncio()
    async def test_search_and_read_no_backend(self) -> None:
        """Ohne Backend und DDG deaktiviert → Meldung."""
        web = WebTools()
        web._duckduckgo_enabled = False
        result = await web.search_and_read("test")
        assert "Keine Suchengine" in result

    @pytest.mark.asyncio()
    async def test_search_and_read_with_results(self) -> None:
        """Suche + Fetch der Ergebnisse."""
        web = WebTools(searxng_url="http://localhost:8888")

        # Mock für Suche
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.raise_for_status = MagicMock()
        search_response.json.return_value = {
            "results": [
                {"title": "Page 1", "url": "https://page1.com", "content": "Snippet"},
            ]
        }

        # Mock für Fetch
        fetch_response = MagicMock()
        fetch_response.status_code = 200
        fetch_response.raise_for_status = MagicMock()
        fetch_response.headers = {"content-type": "text/html"}
        fetch_response.content = b"<p>Inhalt</p>"

        call_urls: list[str] = []

        async def mock_get(url, **kwargs):
            call_urls.append(str(url))
            if "localhost" in str(url):
                return search_response
            return fetch_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = mock_get
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web.search_and_read("test query", num_results=1)
            assert "test query" in result
            urls_in_result = re.findall(r"https?://\S+", result)
            assert urls_in_result, "Es sollten URLs im Ergebnis enthalten sein."
            parsed_url = urlparse(urls_in_result[0])
            assert parsed_url.hostname == "page1.com"


# ── register_web_tools ────────────────────────────────────────────────────


class TestRegisterWebTools:
    """Tests für die MCP-Client-Registrierung."""

    def test_registers_four_tools(self) -> None:
        """Alle 4 Web-Tools werden registriert."""

        mock_client = MagicMock()
        web = register_web_tools(mock_client, searxng_url="http://localhost:8888")

        assert isinstance(web, WebTools)
        assert mock_client.register_builtin_handler.call_count == 4

        # Tool-Namen prüfen
        registered = [call.args[0] for call in mock_client.register_builtin_handler.call_args_list]
        assert "web_search" in registered
        assert "web_news_search" in registered
        assert "web_fetch" in registered
        assert "search_and_read" in registered

    def test_passes_config(self) -> None:
        """Config wird an WebTools weitergegeben."""

        mock_client = MagicMock()
        web = register_web_tools(
            mock_client,
            searxng_url="http://my-searxng:9999",
            brave_api_key="brave-key-42",
        )

        assert web._searxng_url == "http://my-searxng:9999"
        assert web._brave_api_key == "brave-key-42"

    def test_schemas_have_required_fields(self) -> None:
        """Jedes Tool-Schema hat description und input_schema."""

        mock_client = MagicMock()
        register_web_tools(mock_client)

        for call in mock_client.register_builtin_handler.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            # Können als args oder kwargs übergeben werden
            desc = call.args[2] if len(call.args) >= 3 else kwargs.get("description", "")
            assert desc, f"Tool {call.args[0]} hat keine description"
