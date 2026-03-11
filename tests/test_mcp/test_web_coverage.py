"""Coverage-Tests fuer web.py -- fehlende Pfade abdecken.

Schwerpunkt: DuckDuckGo-Suche, News-Suche, Google CSE, Jina Reader,
DDG-Cache, domain blocklist/allowlist, DNS-Cache, config-basierte Init,
_fetch_via_jina, search_and_read mit cross_check, _format_news_results,
_is_private_host edge-cases, register_web_tools mit config.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jarvis.mcp.web import (
    WebError,
    WebTools,
    _extract_text_from_html,
    _format_news_results,
    _format_search_results,
    _is_private_host,
    _simple_html_to_text,
    _truncate_text,
    register_web_tools,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def web() -> WebTools:
    return WebTools()


@pytest.fixture
def web_with_cache(tmp_path: Path) -> WebTools:
    w = WebTools()
    w._ddg_cache_dir = tmp_path / "cache"
    return w


# ============================================================================
# WebTools config init
# ============================================================================


class TestWebToolsConfigInit:
    def test_init_with_full_config(self, tmp_path: Path) -> None:
        config = MagicMock()
        web_cfg = MagicMock()
        web_cfg.searxng_url = "http://searx:8888"
        web_cfg.brave_api_key = "brave-key"
        web_cfg.google_cse_api_key = "gkey"
        web_cfg.google_cse_cx = "gcx"
        web_cfg.jina_api_key = "jina-key"
        web_cfg.duckduckgo_enabled = False
        web_cfg.domain_blocklist = ["evil.com"]
        web_cfg.domain_allowlist = ["good.com"]
        web_cfg.max_fetch_bytes = 100_000
        web_cfg.max_text_chars = 5000
        web_cfg.fetch_timeout_seconds = 30
        web_cfg.search_timeout_seconds = 20
        web_cfg.max_search_results = 8
        web_cfg.ddg_min_delay_seconds = 3.0
        web_cfg.ddg_ratelimit_wait_seconds = 60
        web_cfg.ddg_cache_ttl_seconds = 7200
        web_cfg.search_and_read_max_chars = 3000
        config.web = web_cfg
        config.jarvis_home = str(tmp_path)

        w = WebTools(config=config)
        assert w._searxng_url == "http://searx:8888"
        assert w._brave_api_key == "brave-key"
        assert w._google_cse_api_key == "gkey"
        assert w._google_cse_cx == "gcx"
        assert w._jina_api_key == "jina-key"
        assert w._duckduckgo_enabled is False
        assert w._domain_blocklist == ["evil.com"]
        assert w._domain_allowlist == ["good.com"]
        assert w._max_fetch_bytes == 100_000
        assert w._max_text_chars == 5000
        assert w._fetch_timeout == 30
        assert w._search_timeout == 20
        assert w._ddg_cache_dir == Path(str(tmp_path)) / "cache" / "web_search"

    def test_init_config_no_web_section(self) -> None:
        config = MagicMock()
        config.web = None
        config.jarvis_home = None
        w = WebTools(config=config)
        assert w._duckduckgo_enabled is True

    def test_init_explicit_overrides_config(self) -> None:
        config = MagicMock()
        web_cfg = MagicMock()
        web_cfg.searxng_url = "http://config-searx:8888"
        web_cfg.brave_api_key = "config-key"
        web_cfg.google_cse_api_key = ""
        web_cfg.google_cse_cx = ""
        web_cfg.jina_api_key = ""
        web_cfg.duckduckgo_enabled = True
        web_cfg.domain_blocklist = []
        web_cfg.domain_allowlist = []
        web_cfg.max_fetch_bytes = 500_000
        web_cfg.max_text_chars = 20_000
        web_cfg.fetch_timeout_seconds = 15
        web_cfg.search_timeout_seconds = 10
        web_cfg.max_search_results = 10
        web_cfg.ddg_min_delay_seconds = 2.0
        web_cfg.ddg_ratelimit_wait_seconds = 30
        web_cfg.ddg_cache_ttl_seconds = 3600
        web_cfg.search_and_read_max_chars = 5000
        config.web = web_cfg
        config.jarvis_home = None

        w = WebTools(config=config, searxng_url="http://explicit:9999")
        assert w._searxng_url == "http://explicit:9999"


# ============================================================================
# Domain allowlist / blocklist
# ============================================================================


class TestDomainFiltering:
    def test_allowlist_blocks_not_allowed(self) -> None:
        w = WebTools()
        w._domain_allowlist = ["example.com"]
        with pytest.raises(WebError, match="nicht in der Allowlist"):
            w._check_domain_allowed("evil.com")

    def test_allowlist_allows_exact_match(self) -> None:
        w = WebTools()
        w._domain_allowlist = ["example.com"]
        w._check_domain_allowed("example.com")  # should not raise

    def test_allowlist_allows_subdomain(self) -> None:
        w = WebTools()
        w._domain_allowlist = ["example.com"]
        w._check_domain_allowed("sub.example.com")  # should not raise

    def test_blocklist_blocks_domain(self) -> None:
        w = WebTools()
        w._domain_blocklist = ["evil.com"]
        with pytest.raises(WebError, match="blockiert"):
            w._check_domain_allowed("evil.com")

    def test_blocklist_blocks_subdomain(self) -> None:
        w = WebTools()
        w._domain_blocklist = ["evil.com"]
        with pytest.raises(WebError, match="blockiert"):
            w._check_domain_allowed("sub.evil.com")

    def test_blocklist_allows_other(self) -> None:
        w = WebTools()
        w._domain_blocklist = ["evil.com"]
        w._check_domain_allowed("good.com")  # should not raise

    def test_empty_hostname(self) -> None:
        w = WebTools()
        w._domain_allowlist = ["example.com"]
        w._check_domain_allowed("")  # should not raise (early return)


# ============================================================================
# _validate_url -- DNS cache and resolution paths
# ============================================================================


class TestValidateUrlDns:
    async def test_dns_cache_hit(self, web: WebTools) -> None:
        web._dns_cache.set("example.com", ["93.184.216.34"])
        result = await web._validate_url("https://example.com/page")
        assert result == "https://example.com/page"

    async def test_dns_cache_hit_with_blocked_ip_invalidates(self, web: WebTools) -> None:
        import socket

        web._dns_cache.set("tricky.com", ["127.0.0.1"])
        # After cache invalidation, it re-resolves via DNS -> also returns blocked
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            ],
        ):
            with pytest.raises(WebError, match="blockierte Adresse"):
                await web._validate_url("https://tricky.com/page")

    async def test_dns_resolution_failure(self, web: WebTools) -> None:
        import socket

        with patch("socket.getaddrinfo", side_effect=socket.gaierror("No such host")):
            with pytest.raises(WebError, match="DNS-Aufloesung fehlgeschlagen"):
                await web._validate_url("https://nonexistent.example.invalid/page")

    async def test_dns_resolves_to_private_ip(self, web: WebTools) -> None:
        import socket

        with patch(
            "socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            ],
        ):
            with pytest.raises(WebError, match="blockierte Adresse"):
                await web._validate_url("https://sneaky.com/page")


# ============================================================================
# _is_private_host -- additional edge cases
# ============================================================================


class TestIsPrivateHostAdditional:
    def test_127_x(self) -> None:
        assert _is_private_host("127.0.0.1") is True
        assert _is_private_host("127.255.255.255") is True

    def test_zero_prefix(self) -> None:
        assert _is_private_host("0.0.0.0") is True

    def test_link_local_169_254(self) -> None:
        assert _is_private_host("169.254.1.1") is True

    def test_ipv6_link_local(self) -> None:
        assert _is_private_host("fe80::1") is True

    def test_ipv6_loopback_full(self) -> None:
        assert _is_private_host("0:0:0:0:0:0:0:1") is True
        assert _is_private_host("0:0:0:0:0:0:0:0") is True

    def test_ipv4_mapped_ipv6_private(self) -> None:
        assert _is_private_host("::ffff:10.0.0.1") is True

    def test_ipv4_mapped_ipv6_public(self) -> None:
        assert _is_private_host("::ffff:8.8.8.8") is False

    def test_brackets_stripped(self) -> None:
        assert _is_private_host("[fc00::1]") is True


# ============================================================================
# DDG cache
# ============================================================================


class TestDDGCache:
    def test_cache_put_and_get(self, web_with_cache: WebTools) -> None:
        web_with_cache._ddg_cache_put("test query", "de-de", None, 5, [{"title": "R1"}])
        result = web_with_cache._ddg_cache_get("test query", "de-de", None, 5)
        assert result is not None
        assert result[0]["title"] == "R1"

    def test_cache_miss(self, web_with_cache: WebTools) -> None:
        result = web_with_cache._ddg_cache_get("missing", "de-de", None, 5)
        assert result is None

    def test_cache_expired(self, web_with_cache: WebTools) -> None:
        web_with_cache._ddg_cache_ttl = 0  # immediate expiry
        web_with_cache._ddg_cache_put("expired", "de-de", None, 5, [{"title": "Old"}])
        result = web_with_cache._ddg_cache_get("expired", "de-de", None, 5)
        assert result is None

    def test_cache_none_dir(self) -> None:
        w = WebTools()
        w._ddg_cache_dir = None
        w._ddg_cache_put("q", "r", None, 5, [{"title": "X"}])
        assert w._ddg_cache_get("q", "r", None, 5) is None

    def test_cache_empty_results_not_stored(self, web_with_cache: WebTools) -> None:
        web_with_cache._ddg_cache_put("empty", "de-de", None, 5, [])
        result = web_with_cache._ddg_cache_get("empty", "de-de", None, 5)
        assert result is None

    def test_cache_key_deterministic(self, web_with_cache: WebTools) -> None:
        key1 = web_with_cache._ddg_cache_key("test", "de-de", None, 5)
        key2 = web_with_cache._ddg_cache_key("test", "de-de", None, 5)
        assert key1 == key2
        key3 = web_with_cache._ddg_cache_key("other", "de-de", None, 5)
        assert key1 != key3


# ============================================================================
# DuckDuckGo search with fallback
# ============================================================================


class TestDDGSearchWithFallback:
    def test_ddgs_not_installed(self, web: WebTools) -> None:
        with patch.dict("sys.modules", {"ddgs": None, "duckduckgo_search": None}):
            with pytest.raises(WebError, match="ddgs nicht installiert"):
                web._ddg_search_with_fallback("test", "de-de", None, 5)

    def test_ddgs_success_first_backend(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": "R1", "href": "https://r1.com", "body": "Snippet 1"},
        ]
        mock_ddgs_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            results = web._ddg_search_with_fallback("test", "de-de", None, 5)
            assert len(results) >= 1
            assert results[0]["title"] == "R1"

    def test_ddgs_ratelimit_fallback(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()

        def mock_text(*args, **kwargs):
            backend = kwargs.get("backend", "")
            if backend in ("duckduckgo", "bing"):
                raise Exception("RateLimit error 429")
            return [{"title": "Fallback", "href": "https://f.com", "body": "OK"}]

        mock_instance = MagicMock()
        mock_instance.text = mock_text
        mock_ddgs_cls.return_value = mock_instance

        web._ddg_ratelimit_wait = 0  # Skip sleep in test

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            results = web._ddg_search_with_fallback("test", "de-de", None, 5)
            assert len(results) >= 1

    def test_ddgs_all_backends_fail(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.text.side_effect = Exception("Backend error")
        mock_ddgs_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            with pytest.raises(WebError, match="alle Backends"):
                web._ddg_search_with_fallback("test", "de-de", None, 5)

    def test_ddgs_empty_results_try_next(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()

        def mock_text(*args, **kwargs):
            backend = kwargs.get("backend", "")
            if backend == "duckduckgo":
                return []  # empty
            return [{"title": "Found", "href": "https://f.com", "body": "OK"}]

        mock_instance = MagicMock()
        mock_instance.text = mock_text
        mock_ddgs_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            results = web._ddg_search_with_fallback("test", "de-de", None, 5)
            assert len(results) >= 1
            assert results[0]["title"] == "Found"


# ============================================================================
# _search_duckduckgo (async)
# ============================================================================


class TestSearchDuckDuckGoAsync:
    @pytest.mark.asyncio
    async def test_search_duckduckgo_cache_hit(self, web_with_cache: WebTools) -> None:
        web_with_cache._ddg_cache_put(
            "cached query",
            "de-de",
            None,
            5,
            [
                {"title": "Cached", "url": "https://c.com", "content": "Cached snippet"},
            ],
        )
        result = await web_with_cache._search_duckduckgo("cached query", 5, "de", "")
        assert "Cached" in result

    @pytest.mark.asyncio
    async def test_search_duckduckgo_rate_limit_wait(self, web: WebTools) -> None:
        web._ddg_last_search = time.monotonic()  # just searched
        web._ddg_min_delay = 0.01  # very small delay for test

        with patch.object(
            web,
            "_ddg_search_with_fallback",
            return_value=[
                {"title": "R", "url": "https://r.com", "content": "S"},
            ],
        ):
            result = await web._search_duckduckgo("test", 5, "de", "")
            assert "R" in result

    @pytest.mark.asyncio
    async def test_search_duckduckgo_no_results(self, web: WebTools) -> None:
        with patch.object(web, "_ddg_search_with_fallback", return_value=[]):
            result = await web._search_duckduckgo("noresults", 5, "de", "")
            assert "Keine Ergebnisse" in result

    @pytest.mark.asyncio
    async def test_search_duckduckgo_with_timelimit(self, web: WebTools) -> None:
        with patch.object(
            web,
            "_ddg_search_with_fallback",
            return_value=[
                {"title": "T", "url": "https://t.com", "content": "S"},
            ],
        ) as mock:
            result = await web._search_duckduckgo("test", 5, "en", "w")
            assert "T" in result


# ============================================================================
# Google CSE
# ============================================================================


class TestGoogleCSESearch:
    @pytest.mark.asyncio
    async def test_google_cse_success(self) -> None:
        w = WebTools()
        w._google_cse_api_key = "test-key"
        w._google_cse_cx = "test-cx"
        w._duckduckgo_enabled = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {"title": "CSE Result", "link": "https://cse.com", "snippet": "CSE snippet"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w._search_google_cse("test", 5, "de")
            assert "CSE Result" in result

    @pytest.mark.asyncio
    async def test_google_cse_no_results(self) -> None:
        w = WebTools()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"items": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w._search_google_cse("empty", 5, "de")
            assert "Keine Ergebnisse" in result

    @pytest.mark.asyncio
    async def test_google_cse_in_fallback_chain(self) -> None:
        """Google CSE is used in web_search fallback after SearXNG and Brave fail."""
        w = WebTools()
        w._google_cse_api_key = "gkey"
        w._google_cse_cx = "gcx"
        w._duckduckgo_enabled = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {"title": "CSE Fallback", "link": "https://cse.com", "snippet": "OK"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w.web_search("test")
            assert "CSE Fallback" in result


# ============================================================================
# web_news_search
# ============================================================================


class TestWebNewsSearch:
    @pytest.mark.asyncio
    async def test_news_search_empty_query(self, web: WebTools) -> None:
        result = await web.web_news_search("")
        assert "Keine Suchanfrage" in result

    @pytest.mark.asyncio
    async def test_news_search_success(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.news.return_value = [
            {
                "title": "News 1",
                "url": "https://n1.com",
                "body": "Body 1",
                "source": "CNN",
                "date": "2026-03-01",
            },
        ]
        mock_ddgs_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            result = await web.web_news_search("test news", num_results=5)
            assert "News 1" in result or "Nachrichten" in result

    @pytest.mark.asyncio
    async def test_news_search_no_results(self, web: WebTools) -> None:
        mock_ddgs_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.news.return_value = []
        mock_ddgs_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls
        with patch.dict("sys.modules", {"ddgs": mock_module}):
            result = await web.web_news_search("nonexistent topic")
            assert "Keine Nachrichten" in result

    @pytest.mark.asyncio
    async def test_news_search_ddgs_not_installed(self, web: WebTools) -> None:
        with patch.dict("sys.modules", {"ddgs": None, "duckduckgo_search": None}):
            with pytest.raises((WebError, ImportError)):
                await web.web_news_search("test")


# ============================================================================
# _format_news_results
# ============================================================================


class TestFormatNewsResults:
    def test_full_news_format(self) -> None:
        results = [
            {
                "title": "News A",
                "url": "https://a.com",
                "content": "Snippet A",
                "source": "Reuters",
                "date": "2026-03-01",
            },
            {
                "title": "News B",
                "url": "https://b.com",
                "content": "Snippet B",
                "source": "",
                "date": "",
            },
        ]
        text = _format_news_results(results, "test query")
        assert "Nachrichtenergebnisse" in text
        assert "News A" in text
        assert "Reuters" in text
        assert "Datum: 2026-03-01" in text
        assert "News B" in text

    def test_empty_news(self) -> None:
        text = _format_news_results([], "nothing")
        assert "Nachrichtenergebnisse" in text


# ============================================================================
# _fetch_via_jina
# ============================================================================


class TestFetchViaJina:
    @pytest.mark.asyncio
    async def test_jina_success(self, web: WebTools) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "Extracted content from Jina"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await web._fetch_via_jina("https://example.com")
            assert "Extracted content" in result

    @pytest.mark.asyncio
    async def test_jina_with_api_key(self) -> None:
        w = WebTools()
        w._jina_api_key = "test-jina-key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "Content"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w._fetch_via_jina("https://example.com")
            assert result == "Content"

    @pytest.mark.asyncio
    async def test_jina_empty_response(self, web: WebTools) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "   "

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(WebError, match="Leere Antwort"):
                await web._fetch_via_jina("https://example.com")

    @pytest.mark.asyncio
    async def test_jina_http_error(self, web: WebTools) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(WebError, match="Jina Reader HTTP"):
                await web._fetch_via_jina("https://example.com")

    @pytest.mark.asyncio
    async def test_jina_connection_error(self, web: WebTools) -> None:
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(WebError, match="Verbindungsfehler"):
                await web._fetch_via_jina("https://example.com")


# ============================================================================
# web_fetch -- additional paths
# ============================================================================


class TestWebFetchAdditional:
    @pytest.mark.asyncio
    async def test_fetch_jina_mode(self, web: WebTools) -> None:
        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch.object(web, "_fetch_via_jina", return_value="Jina content here"):
                result = await web.web_fetch("https://example.com", reader_mode="jina")
                assert "Jina content" in result

    @pytest.mark.asyncio
    async def test_fetch_auto_fallback_to_jina_on_error(self, web: WebTools) -> None:
        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                with patch.object(web, "_fetch_via_jina", return_value="Jina fallback content"):
                    result = await web.web_fetch("https://example.com", reader_mode="auto")
                    assert "Jina fallback" in result

    @pytest.mark.asyncio
    async def test_fetch_trafilatura_mode_error_raises(self, web: WebTools) -> None:
        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                with pytest.raises(WebError, match="Fetch fehlgeschlagen"):
                    await web.web_fetch("https://example.com", reader_mode="trafilatura")

    @pytest.mark.asyncio
    async def test_fetch_domain_blocked(self) -> None:
        w = WebTools()
        w._domain_blocklist = ["blocked.com"]
        # Need DNS to pass first
        w._dns_cache.set("blocked.com", ["93.184.216.34"])
        with pytest.raises(WebError, match="blockiert"):
            await w.web_fetch("https://blocked.com/page")

    @pytest.mark.asyncio
    async def test_fetch_large_response_truncated(self, web: WebTools) -> None:
        web._max_fetch_bytes = 50
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"A" * 200

        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get = AsyncMock(return_value=mock_response)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                result = await web.web_fetch("https://example.com/file.txt")
                # Should be truncated, not full 200 chars
                assert len(result) <= 100  # much less than 200

    @pytest.mark.asyncio
    async def test_fetch_auto_short_trafilatura_jina_fallback(self, web: WebTools) -> None:
        """Auto mode: trafilatura returns short text -> try Jina."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><p>Short</p></body></html>"

        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get = AsyncMock(return_value=mock_response)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                with patch("jarvis.mcp.web._extract_text_from_html", return_value="Short"):
                    with patch.object(
                        web,
                        "_fetch_via_jina",
                        return_value="Much longer Jina content that is better",
                    ):
                        result = await web.web_fetch("https://example.com", reader_mode="auto")
                        assert "Jina content" in result

    @pytest.mark.asyncio
    async def test_fetch_auto_jina_fallback_fails(self, web: WebTools) -> None:
        """Auto mode: trafilatura short, Jina also fails -> use trafilatura."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><p>Short</p></body></html>"

        with patch.object(web, "_validate_url", return_value="https://example.com"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get = AsyncMock(return_value=mock_response)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                with patch("jarvis.mcp.web._extract_text_from_html", return_value="Short"):
                    with patch.object(web, "_fetch_via_jina", side_effect=WebError("Jina down")):
                        result = await web.web_fetch("https://example.com", reader_mode="auto")
                        assert "Short" in result


# ============================================================================
# web_search -- all-providers-fail path
# ============================================================================


class TestWebSearchAllFail:
    @pytest.mark.asyncio
    async def test_all_providers_fail(self) -> None:
        w = WebTools(searxng_url="http://broken:9999")
        w._duckduckgo_enabled = False

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w.web_search("test")
            assert "fehlgeschlagen" in result
            assert "SearXNG" in result

    @pytest.mark.asyncio
    async def test_searxng_no_results(self) -> None:
        w = WebTools(searxng_url="http://searx:8888")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w.web_search("nothing")
            assert "Keine Ergebnisse" in result

    @pytest.mark.asyncio
    async def test_brave_no_results(self) -> None:
        w = WebTools(brave_api_key="key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await w.web_search("empty")
            assert "Keine Ergebnisse" in result


# ============================================================================
# search_and_read -- cross_check path
# ============================================================================


class TestSearchAndReadCrossCheck:
    @pytest.mark.asyncio
    async def test_cross_check_enabled(self) -> None:
        w = WebTools(searxng_url="http://searx:8888")

        search_response = MagicMock()
        search_response.status_code = 200
        search_response.raise_for_status = MagicMock()
        search_response.json.return_value = {
            "results": [
                {"title": "Page A", "url": "https://pagea.com", "content": "Snippet A"},
                {"title": "Page B", "url": "https://pageb.com", "content": "Snippet B"},
            ],
        }

        fetch_response = MagicMock()
        fetch_response.status_code = 200
        fetch_response.raise_for_status = MagicMock()
        fetch_response.headers = {"content-type": "text/html"}
        fetch_response.content = b"<p>Page content here is reasonably long enough to not trigger jina fallback with more text added here.</p>"

        async def mock_get(url, **kwargs):
            if "searx" in str(url):
                return search_response
            return fetch_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = mock_get
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            # Need DNS to pass
            w._dns_cache.set("pagea.com", ["93.184.216.34"])
            w._dns_cache.set("pageb.com", ["93.184.216.35"])

            result = await w.search_and_read("test", num_results=2, cross_check=True)
            assert "Quellenvergleich" in result
            assert "Quelle 1" in result or "Quelle 2" in result

    @pytest.mark.asyncio
    async def test_search_and_read_fetch_error_handled(self) -> None:
        w = WebTools(searxng_url="http://searx:8888")

        search_response = MagicMock()
        search_response.status_code = 200
        search_response.raise_for_status = MagicMock()
        search_response.json.return_value = {
            "results": [
                {"title": "P", "url": "https://err.com", "content": "S"},
            ],
        }

        async def mock_get(url, **kwargs):
            if "searx" in str(url):
                return search_response
            raise httpx.ConnectError("fail")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = mock_get
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            # The fetch itself will try to validate URL, which needs DNS
            with patch.object(w, "web_fetch", side_effect=WebError("Fetch failed")):
                result = await w.search_and_read("test", num_results=1)
                assert "Fehler" in result


# ============================================================================
# _extract_text_from_html edge cases
# ============================================================================


class TestExtractTextFromHtmlEdgeCases:
    def test_trafilatura_returns_none(self) -> None:
        with patch("trafilatura.extract", return_value=None):
            result = _extract_text_from_html("<p>Fallback</p>")
            assert "Fallback" in result

    def test_trafilatura_exception(self) -> None:
        with patch("trafilatura.extract", side_effect=RuntimeError("crash")):
            result = _extract_text_from_html("<p>Also fallback</p>")
            assert "Also fallback" in result


# ============================================================================
# register_web_tools with config
# ============================================================================


class TestRegisterWebToolsConfig:
    def test_register_with_config(self) -> None:
        mock_client = MagicMock()
        config = MagicMock()
        web_cfg = MagicMock()
        web_cfg.searxng_url = "http://myhost:8888"
        web_cfg.brave_api_key = ""
        web_cfg.google_cse_api_key = ""
        web_cfg.google_cse_cx = ""
        web_cfg.jina_api_key = ""
        web_cfg.duckduckgo_enabled = True
        web_cfg.domain_blocklist = []
        web_cfg.domain_allowlist = []
        web_cfg.max_fetch_bytes = 500_000
        web_cfg.max_text_chars = 20_000
        web_cfg.fetch_timeout_seconds = 15
        web_cfg.search_timeout_seconds = 10
        web_cfg.max_search_results = 10
        web_cfg.ddg_min_delay_seconds = 2.0
        web_cfg.ddg_ratelimit_wait_seconds = 30
        web_cfg.ddg_cache_ttl_seconds = 3600
        web_cfg.search_and_read_max_chars = 5000
        web_cfg.http_request_max_body_bytes = 1_048_576
        web_cfg.http_request_timeout_seconds = 30
        web_cfg.http_request_rate_limit_seconds = 1.0
        config.web = web_cfg
        config.jarvis_home = None

        web = register_web_tools(mock_client, config=config)
        assert isinstance(web, WebTools)
        assert web._searxng_url == "http://myhost:8888"
        assert mock_client.register_builtin_handler.call_count == 5
