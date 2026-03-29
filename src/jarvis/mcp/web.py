"""Web-Tools fuer Jarvis: Suche und URL-Fetch.

Ermoeglicht dem Agenten Webrecherche und Seiteninhalt-Extraktion.

Tools:
  - web_search: Websuche ueber SearXNG, Brave oder DuckDuckGo (Multi-Backend)
  - web_news_search: Nachrichtensuche via DuckDuckGo News
  - web_fetch: URL abrufen und Text extrahieren (via trafilatura)
  - search_and_read: Kombinierte Suche + Fetch

DuckDuckGo-Optimierungen:
  - Multi-Backend-Fallback (ddgs: duckduckgo → bing → google → brave)
  - Lokaler Cache mit konfigurierbarem TTL (Standard: 1h)
  - Rate-Limiting (2s Mindestabstand zwischen Suchen)
  - Region/Sprache/Zeitfilter-Support

Bibel-Referenz: §5.3 (jarvis-web Server)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from jarvis.utils.logging import get_logger
from jarvis.utils.ttl_dict import TTLDict

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_DEFAULT_MAX_FETCH_BYTES = 500_000  # 500 KB maximaler Fetch
_DEFAULT_MAX_TEXT_CHARS = 20_000  # 20K Zeichen extrahierter Text
_DEFAULT_FETCH_TIMEOUT = 15  # Sekunden
_DEFAULT_SEARCH_TIMEOUT = 10  # Sekunden
_DEFAULT_MAX_SEARCH_RESULTS = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# DuckDuckGo optimization: backends in fallback order
DDG_BACKENDS = ("duckduckgo", "bing", "google", "brave")
# Minimum delay between DuckDuckGo searches (seconds)
_DEFAULT_DDG_MIN_DELAY = 2.0
# Wait time on rate-limiting (seconds)
_DEFAULT_DDG_RATELIMIT_WAIT = 30
# Cache TTL (seconds) — default: 1 hour
_DEFAULT_DDG_CACHE_TTL = 3600
# search_and_read: maximum characters per page
_DEFAULT_SEARCH_AND_READ_MAX_CHARS = 5000

# Blocked domains (security)
BLOCKED_DOMAINS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::",  # IPv6 unspecified (urlparse strips brackets)
        "::1",  # IPv6 loopback (urlparse strips brackets)
        "metadata.google.internal",
        "169.254.169.254",  # AWS metadata
    }
)


__all__ = [
    "WebError",
    "WebTools",
    "register_web_tools",
]


class WebError(Exception):
    """Fehler bei Web-Operationen."""


class WebTools:
    """Web-Recherche und URL-Fetch-Tools. [B§5.3]

    Unterstuetzt vier Such-Backends (Fallback-Kette):
      1. SearXNG (self-hosted, bevorzugt)
      2. Brave Search API
      3. Google Custom Search Engine (100 Anfragen/Tag kostenlos)
      4. DuckDuckGo via ddgs (kostenlos, Multi-Backend, mit Cache)

    DuckDuckGo-Optimierungen:
      - Multi-Backend-Fallback (duckduckgo → bing → google → brave)
      - Lokaler Ergebnis-Cache (1h TTL)
      - Rate-Limiting (2s Mindestabstand)
      - timelimit-Support (d/w/m/y)

    Attributes:
        searxng_url: URL der SearXNG-Instanz.
        brave_api_key: Brave Search API Key.
    """

    def __init__(
        self,
        config: JarvisConfig | None = None,
        searxng_url: str | None = None,
        brave_api_key: str | None = None,
    ) -> None:
        """Initialisiert WebTools.

        Args:
            config: Jarvis-Konfiguration.
            searxng_url: SearXNG Base-URL (z.B. "http://localhost:8888").
            brave_api_key: Brave Search API Key.
        """
        self._searxng_url = searxng_url
        self._brave_api_key = brave_api_key
        self._duckduckgo_enabled = True

        # Google CSE credentials
        self._google_cse_api_key = ""
        self._google_cse_cx = ""

        # Jina AI Reader API key
        self._jina_api_key = ""

        # Domain filtering
        self._domain_blocklist: list[str] = []
        self._domain_allowlist: list[str] = []

        # DuckDuckGo optimization: state
        self._ddg_last_search: float = 0.0
        self._ddg_cache_dir: Path | None = None

        # Configurable constants (defaults from module-level constants)
        self._max_fetch_bytes: int = _DEFAULT_MAX_FETCH_BYTES
        self._max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS
        self._fetch_timeout: int = _DEFAULT_FETCH_TIMEOUT
        self._search_timeout: int = _DEFAULT_SEARCH_TIMEOUT
        self._max_search_results: int = _DEFAULT_MAX_SEARCH_RESULTS
        self._ddg_min_delay: float = _DEFAULT_DDG_MIN_DELAY
        self._ddg_ratelimit_wait: int = _DEFAULT_DDG_RATELIMIT_WAIT
        self._ddg_cache_ttl: int = _DEFAULT_DDG_CACHE_TTL
        self._search_and_read_max_chars: int = _DEFAULT_SEARCH_AND_READ_MAX_CHARS

        # HTTP request tool
        self._http_request_max_body: int = 1_048_576  # 1 MB
        self._http_request_timeout: int = 30
        self._http_request_rate_limit: float = 1.0
        self._http_request_last_call: float = 0.0

        # Load from config if available
        if config is not None:
            web_cfg = getattr(config, "web", None)
            if web_cfg is not None:
                self._searxng_url = self._searxng_url or getattr(web_cfg, "searxng_url", None) or ""
                self._brave_api_key = (
                    self._brave_api_key or getattr(web_cfg, "brave_api_key", None) or ""
                )
                self._google_cse_api_key = getattr(web_cfg, "google_cse_api_key", "") or ""
                self._google_cse_cx = getattr(web_cfg, "google_cse_cx", "") or ""
                self._jina_api_key = getattr(web_cfg, "jina_api_key", "") or ""
                self._duckduckgo_enabled = getattr(web_cfg, "duckduckgo_enabled", True)
                self._domain_blocklist = list(getattr(web_cfg, "domain_blocklist", []))
                self._domain_allowlist = list(getattr(web_cfg, "domain_allowlist", []))

                # Apply configurable constants from web config
                self._max_fetch_bytes = getattr(web_cfg, "max_fetch_bytes", self._max_fetch_bytes)
                self._max_text_chars = getattr(web_cfg, "max_text_chars", self._max_text_chars)
                self._fetch_timeout = getattr(web_cfg, "fetch_timeout_seconds", self._fetch_timeout)
                self._search_timeout = getattr(
                    web_cfg, "search_timeout_seconds", self._search_timeout
                )
                self._max_search_results = getattr(
                    web_cfg, "max_search_results", self._max_search_results
                )
                self._ddg_min_delay = getattr(web_cfg, "ddg_min_delay_seconds", self._ddg_min_delay)
                self._ddg_ratelimit_wait = getattr(
                    web_cfg, "ddg_ratelimit_wait_seconds", self._ddg_ratelimit_wait
                )
                self._ddg_cache_ttl = getattr(web_cfg, "ddg_cache_ttl_seconds", self._ddg_cache_ttl)
                self._search_and_read_max_chars = getattr(
                    web_cfg, "search_and_read_max_chars", self._search_and_read_max_chars
                )

                # HTTP request tool
                self._http_request_max_body = getattr(
                    web_cfg, "http_request_max_body_bytes", self._http_request_max_body
                )
                self._http_request_timeout = getattr(
                    web_cfg, "http_request_timeout_seconds", self._http_request_timeout
                )
                self._http_request_rate_limit = getattr(
                    web_cfg, "http_request_rate_limit_seconds", self._http_request_rate_limit
                )

            # Cache directory: ~/.jarvis/cache/web_search/
            jarvis_home = getattr(config, "jarvis_home", None)
            if jarvis_home:
                self._ddg_cache_dir = Path(jarvis_home) / "cache" / "web_search"

        if self._ddg_cache_dir is None:
            self._ddg_cache_dir = Path.home() / ".jarvis" / "cache" / "web_search"

        # Reference to config for live-reload
        self._config = config

        # DNS cache: avoids repeated DNS resolution per request
        self._dns_cache: TTLDict[str, list[str]] = TTLDict(
            max_size=1000,
            ttl_seconds=300,
            cleanup_interval=60,
        )

    def reload_config(self, config: JarvisConfig) -> None:
        """Aktualisiert WebTools-Parameter aus neuer Config (Live-Reload).

        Wird vom Gateway aufgerufen wenn der User Einstellungen im UI aendert.
        API-Keys, Domain-Listen und Limits werden sofort aktualisiert.
        """
        self._config = config
        web_cfg = getattr(config, "web", None)
        if web_cfg is None:
            return

        self._searxng_url = getattr(web_cfg, "searxng_url", "") or ""
        self._brave_api_key = getattr(web_cfg, "brave_api_key", "") or ""
        self._google_cse_api_key = getattr(web_cfg, "google_cse_api_key", "") or ""
        self._google_cse_cx = getattr(web_cfg, "google_cse_cx", "") or ""
        self._jina_api_key = getattr(web_cfg, "jina_api_key", "") or ""
        self._duckduckgo_enabled = getattr(web_cfg, "duckduckgo_enabled", True)
        self._domain_blocklist = list(getattr(web_cfg, "domain_blocklist", []))
        self._domain_allowlist = list(getattr(web_cfg, "domain_allowlist", []))

        self._max_fetch_bytes = getattr(web_cfg, "max_fetch_bytes", self._max_fetch_bytes)
        self._max_text_chars = getattr(web_cfg, "max_text_chars", self._max_text_chars)
        self._fetch_timeout = getattr(web_cfg, "fetch_timeout_seconds", self._fetch_timeout)
        self._search_timeout = getattr(web_cfg, "search_timeout_seconds", self._search_timeout)
        self._max_search_results = getattr(web_cfg, "max_search_results", self._max_search_results)
        self._ddg_min_delay = getattr(web_cfg, "ddg_min_delay_seconds", self._ddg_min_delay)
        self._ddg_ratelimit_wait = getattr(
            web_cfg, "ddg_ratelimit_wait_seconds", self._ddg_ratelimit_wait
        )
        self._ddg_cache_ttl = getattr(web_cfg, "ddg_cache_ttl_seconds", self._ddg_cache_ttl)
        self._search_and_read_max_chars = getattr(
            web_cfg, "search_and_read_max_chars", self._search_and_read_max_chars
        )

        self._http_request_max_body = getattr(
            web_cfg, "http_request_max_body_bytes", self._http_request_max_body
        )
        self._http_request_timeout = getattr(
            web_cfg, "http_request_timeout_seconds", self._http_request_timeout
        )
        self._http_request_rate_limit = getattr(
            web_cfg, "http_request_rate_limit_seconds", self._http_request_rate_limit
        )

        log.info("web_tools_config_reloaded")

    async def _validate_url(self, url: str) -> str:
        """Validiert eine URL gegen SSRF-Angriffe.

        Args:
            url: Die zu validierende URL.

        Returns:
            Validierte URL.

        Raises:
            WebError: Bei ungueltiger oder blockierter URL.
        """
        try:
            parsed = urlparse(url)
        except ValueError as exc:
            raise WebError(f"Ungültige URL: {url}") from exc

        if parsed.scheme not in ("http", "https"):
            raise WebError(f"Nur HTTP/HTTPS erlaubt, nicht '{parsed.scheme}'")

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise WebError(f"Keine gültige Domain: {url}")

        if hostname in BLOCKED_DOMAINS:
            raise WebError(f"Zugriff auf {hostname} blockiert (Sicherheit)")

        # Block private IP ranges
        if _is_private_host(hostname):
            raise WebError(f"Zugriff auf private Adressen blockiert: {hostname}")

        # Check DNS resolution to prevent DNS-rebinding/bypass
        import socket

        cached_ips = self._dns_cache.get(hostname)
        if cached_ips is not None:
            # Cache-Hit: IPs erneut validieren (Paranoia-Check)
            for ip in cached_ips:
                if ip in BLOCKED_DOMAINS or _is_private_host(ip):
                    # Invalid IP in cache → delete and re-resolve
                    del self._dns_cache[hostname]
                    cached_ips = None
                    break

        if cached_ips is None:
            try:
                loop = asyncio.get_running_loop()
                resolved = await loop.run_in_executor(
                    None,
                    lambda: socket.getaddrinfo(
                        hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
                    ),
                )
                ips: list[str] = []
                for _family, _type, _proto, _canonname, sockaddr in resolved:
                    ip = sockaddr[0]
                    if ip in BLOCKED_DOMAINS or _is_private_host(ip):
                        raise WebError(f"DNS für {hostname} zeigt auf blockierte Adresse: {ip}")
                    ips.append(ip)
                self._dns_cache.set(hostname, ips)
            except socket.gaierror:
                raise WebError(f"DNS-Aufloesung fehlgeschlagen fuer {hostname}") from None

        return url

    def _check_domain_allowed(self, hostname: str) -> None:
        """Prueft ob eine Domain durch die Blocklist/Allowlist erlaubt ist.

        Args:
            hostname: Der zu pruefende Hostname.

        Raises:
            WebError: Wenn die Domain blockiert ist.
        """
        hostname = hostname.lower().strip()
        if not hostname:
            return

        # Allowlist mode: if set, ONLY allow these domains
        if self._domain_allowlist:
            allowed = any(
                hostname == d.lower() or hostname.endswith("." + d.lower())
                for d in self._domain_allowlist
            )
            if not allowed:
                raise WebError(
                    f"Domain {hostname} nicht in der Allowlist. "
                    f"Erlaubte Domains: {', '.join(self._domain_allowlist)}"
                )
            return

        # Blocklist: block these domains
        if self._domain_blocklist:
            blocked = any(
                hostname == d.lower() or hostname.endswith("." + d.lower())
                for d in self._domain_blocklist
            )
            if blocked:
                raise WebError(f"Domain {hostname} ist blockiert (Domain-Blocklist).")

    # ── web_search ─────────────────────────────────────────────────────────

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "de",
        timelimit: str = "",
    ) -> str:
        """Fuehrt eine Websuche durch.

        Hybrid-Modus: Alle verfuegbaren Engines werden parallel abgefragt.
        Ergebnisse werden nach Cross-Engine-Score gerankt (URL die in mehreren
        Engines auftaucht = hoehere Relevanz). Fallback auf Einzelergebnis
        wenn nur eine Engine verfuegbar.

        Args:
            query: Suchanfrage.
            num_results: Anzahl gewuenschter Ergebnisse (1-10).
            language: Sprache fuer Suchergebnisse.
            timelimit: Zeitfilter ('d'=Tag, 'w'=Woche, 'm'=Monat, 'y'=Jahr, ''=alle).

        Returns:
            Formatierte Suchergebnisse als Text.
        """
        if not query.strip():
            return "Keine Suchanfrage angegeben."

        num_results = min(max(num_results, 1), self._max_search_results)

        # Collect all available search backends
        import asyncio as _aio

        tasks: dict[str, _aio.Task] = {}
        if self._searxng_url:
            tasks["searxng"] = _aio.create_task(
                self._search_raw_searxng(query, num_results, language)
            )
        if self._brave_api_key:
            tasks["brave"] = _aio.create_task(self._search_raw_brave(query, num_results, language))
        if self._google_cse_api_key and self._google_cse_cx:
            tasks["google"] = _aio.create_task(
                self._search_raw_google_cse(query, num_results, language)
            )
        if self._duckduckgo_enabled:
            tasks["ddg"] = _aio.create_task(
                self._search_raw_duckduckgo(query, num_results, language, timelimit)
            )

        if not tasks:
            return (
                "Keine Suchengine konfiguriert.\n"
                "Setze `searxng_url` oder `brave_api_key` in der Konfiguration,\n"
                "oder aktiviere `duckduckgo_enabled: true` (Standard)."
            )

        # Wait for all (with individual error handling)
        all_results: dict[str, list[dict[str, str]]] = {}
        provider_errors: list[str] = []

        for name, task in tasks.items():
            try:
                results = await task
                if results:
                    all_results[name] = results
                    log.info(
                        "hybrid_search_ok", backend=name, results=len(results), query=query[:40]
                    )
            except Exception as exc:
                provider_errors.append(f"{name}: {exc}")
                log.warning("hybrid_search_failed", backend=name, error=str(exc)[:80])

        if not all_results:
            if provider_errors:
                error_details = "\n".join(f"  - {e}" for e in provider_errors)
                return (
                    f"Websuche fuer '{query}' fehlgeschlagen.\n"
                    f"Alle Such-Provider haben Fehler gemeldet:\n{error_details}"
                )
            return f"Keine Ergebnisse fuer: {query}"

        # Single engine → return directly (no merge needed)
        if len(all_results) == 1:
            engine_name = next(iter(all_results))
            return _format_search_results(all_results[engine_name][:num_results], query)

        # Hybrid merge: rank by cross-engine agreement
        merged = self._merge_hybrid_results(all_results, num_results)
        engines_used = ", ".join(sorted(all_results.keys()))
        log.info("hybrid_search_merged", engines=engines_used, merged=len(merged), query=query[:40])
        return _format_search_results(merged, query)

    def _merge_hybrid_results(
        self,
        engine_results: dict[str, list[dict[str, str]]],
        num_results: int,
    ) -> list[dict[str, str]]:
        """Merge results from multiple search engines, ranked by cross-engine score.

        URLs that appear in multiple engines get a higher score.
        """
        from urllib.parse import urlparse

        url_data: dict[str, dict[str, Any]] = {}  # url → {result, engines, score}
        total_engines = len(engine_results)

        for engine_name, results in engine_results.items():
            for rank, result in enumerate(results):
                url = result.get("url", "").rstrip("/")
                if not url:
                    continue
                domain = urlparse(url).netloc

                if url not in url_data:
                    url_data[url] = {
                        "result": result,
                        "engines": set(),
                        "rank_sum": 0,
                        "domain": domain,
                    }
                url_data[url]["engines"].add(engine_name)
                url_data[url]["rank_sum"] += rank

        # Score: cross-engine agreement (0.0-1.0) minus average rank penalty
        scored: list[tuple[float, dict[str, str]]] = []
        for url, data in url_data.items():
            cross_score = len(data["engines"]) / total_engines
            rank_penalty = data["rank_sum"] / (len(data["engines"]) * 10)
            final_score = cross_score - rank_penalty

            # Annotate result with engine info
            result = dict(data["result"])
            engine_list = ", ".join(sorted(data["engines"]))
            result["content"] = (
                f"[{len(data['engines'])}/{total_engines} engines: {engine_list}] "
                + result.get("content", "")
            )
            scored.append((final_score, result))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:num_results]]

    # -- Raw search methods (return list[dict] instead of formatted str) --

    async def _search_raw_searxng(
        self, query: str, num_results: int, language: str
    ) -> list[dict[str, str]]:
        """SearXNG search returning raw result dicts."""
        url = f"{self._searxng_url}/search"
        params = {"q": query, "format": "json", "language": language, "categories": "general"}
        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in data.get("results", [])[:num_results]
        ]

    async def _search_raw_brave(
        self, query: str, num_results: int, language: str
    ) -> list[dict[str, str]]:
        """Brave Search API returning raw result dicts."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_api_key or "",
        }
        params = {"q": query, "count": str(num_results), "search_lang": language, "country": "DE"}
        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("description", ""),
            }
            for r in data.get("web", {}).get("results", [])[:num_results]
        ]

    async def _search_raw_google_cse(
        self, query: str, num_results: int, language: str
    ) -> list[dict[str, str]]:
        """Google CSE returning raw result dicts."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self._google_cse_api_key,
            "cx": self._google_cse_cx,
            "q": query,
            "num": str(min(num_results, 10)),
            "lr": f"lang_{language}",
        }
        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("link", ""), "content": r.get("snippet", "")}
            for r in data.get("items", [])[:num_results]
        ]

    async def _search_raw_duckduckgo(
        self, query: str, num_results: int, language: str, timelimit: str = ""
    ) -> list[dict[str, str]]:
        """DuckDuckGo search returning raw result dicts."""
        # Use the existing DDG method but parse the formatted output back
        # This is a wrapper — the real DDG logic has caching + rate limiting
        formatted = await self._search_duckduckgo(query, num_results, language, timelimit)
        # Parse back to list of dicts (rough extraction from formatted text)
        results: list[dict[str, str]] = []
        import re as _re

        for match in _re.finditer(
            r"\[(\d+)\] (.+?)\n\s+(.+?)\n\s+(.+?)(?=\n\[|\n\n|$)", formatted, _re.DOTALL
        ):
            results.append(
                {
                    "title": match.group(2).strip(),
                    "url": match.group(3).strip(),
                    "content": match.group(4).strip(),
                }
            )
        if not results and "Ergebnisse" not in formatted:
            # DDG returned something but we couldn't parse → return as single result
            return [{"title": query, "url": "", "content": formatted[:500]}]
        return results

    async def _search_searxng(
        self,
        query: str,
        num_results: int,
        language: str,
    ) -> str:
        """Suche ueber SearXNG-Instanz."""
        url = f"{self._searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "categories": "general",
        }

        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])[:num_results]
        if not results:
            return f"Keine Ergebnisse für: {query}"

        return _format_search_results(results, query)

    async def _search_brave(
        self,
        query: str,
        num_results: int,
        language: str,
    ) -> str:
        """Suche ueber Brave Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        # Set API key directly as header, never log it
        _token = self._brave_api_key or ""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": _token,
        }
        params = {
            "q": query,
            "count": str(num_results),
            "search_lang": language,
            "country": "DE",
        }

        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        web_results = data.get("web", {}).get("results", [])[:num_results]
        if not web_results:
            return f"Keine Ergebnisse für: {query}"

        # Convert Brave format → unified format
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("description", ""),
            }
            for r in web_results
        ]
        return _format_search_results(results, query)

    async def _search_google_cse(
        self,
        query: str,
        num_results: int,
        language: str,
    ) -> str:
        """Suche ueber Google Custom Search Engine API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self._google_cse_api_key,
            "cx": self._google_cse_cx,
            "q": query,
            "num": str(min(num_results, 10)),
            "lr": f"lang_{language}",
        }

        async with httpx.AsyncClient(timeout=self._search_timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])[:num_results]
        if not items:
            return f"Keine Ergebnisse für: {query}"

        # Convert Google CSE format → unified format
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", ""),
            }
            for r in items
        ]
        return _format_search_results(results, query)

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        language: str,
        timelimit: str = "",
    ) -> str:
        """Optimierte DuckDuckGo-Suche mit Cache, Rate-Limiting und Multi-Backend.

        Features:
          - Lokaler Cache (1h TTL) — identische Queries werden nicht doppelt gesucht
          - Rate-Limiting (2s Mindestabstand) — verhindert HTTP 429
          - Multi-Backend-Fallback — bei Blocking automatisch auf bing/google wechseln
          - Region/Zeitfilter-Support

        Args:
            query: Suchanfrage.
            num_results: Anzahl gewuenschter Ergebnisse.
            language: Sprache (ISO-Code).
            timelimit: Zeitfilter ('d', 'w', 'm', 'y', '' fuer alle).
        """
        import anyio

        # Region mapping
        region_map = {
            "de": "de-de",
            "en": "us-en",
            "fr": "fr-fr",
            "es": "es-es",
            "it": "it-it",
            "pt": "pt-pt",
            "nl": "nl-nl",
            "ja": "jp-jp",
            "zh": "cn-zh",
            "ru": "ru-ru",
            "ko": "kr-ko",
            "pl": "pl-pl",
        }
        region = region_map.get(language, "wt-wt")
        tl = timelimit if timelimit in ("d", "w", "m", "y") else None

        # 1. Check cache
        cached = self._ddg_cache_get(query, region, tl, num_results)
        if cached is not None:
            log.info("ddg_cache_hit", query=query[:60])
            return _format_search_results(cached, query)

        # 2. Rate-limiting: enforce minimum delay
        now = time.monotonic()
        elapsed = now - self._ddg_last_search
        if elapsed < self._ddg_min_delay:
            wait = self._ddg_min_delay - elapsed
            log.debug("ddg_rate_limit_wait", wait_s=round(wait, 1))
            await anyio.sleep(wait)

        # 3. Multi-backend search with fallback
        results = await anyio.to_thread.run_sync(
            lambda: self._ddg_search_with_fallback(query, region, tl, num_results)
        )
        self._ddg_last_search = time.monotonic()

        if not results:
            return f"Keine Ergebnisse für: {query}"

        # 4. Save to cache
        self._ddg_cache_put(query, region, tl, num_results, results)

        return _format_search_results(results, query)

    def _ddg_search_with_fallback(
        self,
        query: str,
        region: str,
        timelimit: str | None,
        num_results: int,
    ) -> list[dict[str, Any]]:
        """Synchrone DuckDuckGo-Suche mit Backend-Fallback-Kette.

        Probiert nacheinander: duckduckgo → bing → google → brave.
        Bei RateLimit: 30s warten, dann naechstes Backend.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore[no-redef]
            except ImportError:
                raise WebError(
                    "ddgs nicht installiert. Installiere mit: pip install ddgs"
                ) from None

        last_error: Exception | None = None

        for backend in DDG_BACKENDS:
            try:
                log.debug("ddg_backend_try", backend=backend, query=query[:60])
                raw = list(
                    DDGS(timeout=self._search_timeout).text(
                        query,
                        region=region,
                        safesearch="moderate",
                        timelimit=timelimit,
                        max_results=num_results,
                        backend=backend,
                    )
                )
                if raw:
                    log.info(
                        "ddg_search_ok",
                        backend=backend,
                        results=len(raw),
                        query=query[:60],
                    )
                    return [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "content": r.get("body", ""),
                        }
                        for r in raw
                    ]
                # Empty results → try next backend
                log.debug("ddg_backend_empty", backend=backend)

            except Exception as exc:
                last_error = exc
                exc_str = str(exc).lower()
                is_ratelimit = "ratelimit" in exc_str or "429" in exc_str or "202" in exc_str
                if is_ratelimit:
                    log.warning(
                        "ddg_ratelimit",
                        backend=backend,
                        query=query[:60],
                    )
                    # Wait briefly on rate-limit before trying next backend
                    time.sleep(min(self._ddg_ratelimit_wait, 5))
                else:
                    log.warning(
                        "ddg_backend_error",
                        backend=backend,
                        error=str(exc)[:200],
                    )

        # All backends failed
        if last_error:
            raise WebError(
                f"DuckDuckGo-Suche fehlgeschlagen (alle Backends): {last_error}"
            ) from last_error
        return []

    # ── DuckDuckGo News-Suche ──────────────────────────────────────────────

    async def web_news_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "de",
        timelimit: str = "w",
    ) -> str:
        """Sucht aktuelle Nachrichten ueber DuckDuckGo News.

        Args:
            query: Suchanfrage.
            num_results: Anzahl Ergebnisse (1-10).
            language: Sprache.
            timelimit: Zeitfilter ('d'=Tag, 'w'=Woche, 'm'=Monat).

        Returns:
            Formatierte Nachrichtenergebnisse.
        """
        if not query.strip():
            return "Keine Suchanfrage angegeben."

        num_results = min(max(num_results, 1), self._max_search_results)

        import anyio

        region_map = {
            "de": "de-de",
            "en": "us-en",
            "fr": "fr-fr",
            "es": "es-es",
        }
        region = region_map.get(language, "wt-wt")
        tl = timelimit if timelimit in ("d", "w", "m") else "w"

        # Rate-Limiting
        now = time.monotonic()
        elapsed = now - self._ddg_last_search
        if elapsed < self._ddg_min_delay:
            await anyio.sleep(self._ddg_min_delay - elapsed)

        search_timeout = self._search_timeout

        def _sync_news() -> list[dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # type: ignore[no-redef]
                except ImportError:
                    raise WebError("ddgs nicht installiert. pip install ddgs") from None

            raw = list(
                DDGS(timeout=search_timeout).news(
                    query,
                    region=region,
                    safesearch="moderate",
                    timelimit=tl,
                    max_results=num_results,
                )
            )
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("body", ""),
                    "source": r.get("source", ""),
                    "date": r.get("date", ""),
                }
                for r in raw
            ]

        results = await anyio.to_thread.run_sync(_sync_news)
        self._ddg_last_search = time.monotonic()

        if not results:
            return f"Keine Nachrichten für: {query}"

        return _format_news_results(results, query)

    # ── DuckDuckGo Cache ───────────────────────────────────────────────────

    def _ddg_cache_key(
        self,
        query: str,
        region: str,
        timelimit: str | None,
        num_results: int,
    ) -> str:
        """Berechnet einen deterministischen Cache-Key."""
        data = json.dumps(
            {"q": query, "r": region, "t": timelimit or "", "n": num_results},
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    def _ddg_cache_get(
        self,
        query: str,
        region: str,
        timelimit: str | None,
        num_results: int,
    ) -> list[dict[str, Any]] | None:
        """Liest Suchergebnisse aus dem lokalen Cache.

        Returns:
            Gecachte Ergebnisse oder None bei Cache-Miss/Expired.
        """
        if self._ddg_cache_dir is None:
            return None

        key = self._ddg_cache_key(query, region, timelimit, num_results)
        cache_file = self._ddg_cache_dir / f"{key}.json"

        try:
            if not cache_file.exists():
                return None
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - data.get("ts", 0) >= self._ddg_cache_ttl:
                # Expired → delete
                cache_file.unlink(missing_ok=True)
                return None
            return data.get("results")
        except Exception:
            return None

    def _ddg_cache_put(
        self,
        query: str,
        region: str,
        timelimit: str | None,
        num_results: int,
        results: list[dict[str, Any]],
    ) -> None:
        """Speichert Suchergebnisse im lokalen Cache."""
        if self._ddg_cache_dir is None or not results:
            return

        try:
            self._ddg_cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._ddg_cache_key(query, region, timelimit, num_results)
            cache_file = self._ddg_cache_dir / f"{key}.json"
            cache_file.write_text(
                json.dumps(
                    {"ts": time.time(), "q": query, "results": results},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("ddg_cache_write_failed", error=str(exc))

    # ── web_fetch ──────────────────────────────────────────────────────────

    async def web_fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_chars: int | None = None,
        reader_mode: str = "auto",
    ) -> str:
        """Ruft eine URL ab und extrahiert den Text.

        Args:
            url: Die abzurufende URL.
            extract_text: Text extrahieren (True) oder Raw-HTML (False).
            max_chars: Maximale Zeichenanzahl (Default: self._max_text_chars).
            reader_mode: Extraktions-Modus:
                'auto' = trafilatura, bei <200 Zeichen Fallback auf Jina
                'trafilatura' = nur trafilatura
                'jina' = nur Jina AI Reader

        Returns:
            Extrahierter Text oder HTML-Inhalt.
        """
        validated = await self._validate_url(url)
        max_chars = max_chars or self._max_text_chars

        # Check domain filter
        parsed = urlparse(validated)
        hostname = (parsed.hostname or "").lower()
        self._check_domain_allowed(hostname)

        # Jina-only mode
        if reader_mode == "jina":
            text = await self._fetch_via_jina(validated)
            return _truncate_text(text, max_chars, url)

        # Standard fetch
        fetch_failed = False
        try:
            async with httpx.AsyncClient(
                timeout=self._fetch_timeout,
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                resp = await client.get(validated)
                resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if reader_mode == "trafilatura":
                raise WebError(f"Fetch fehlgeschlagen für {url}: {exc}") from exc
            # auto mode: try Jina as fallback on error
            log.warning("web_fetch_failed_trying_jina", url=url, error=str(exc))
            fetch_failed = True

        if fetch_failed:
            text = await self._fetch_via_jina(validated)
            return _truncate_text(text, max_chars, url)

        content_type = resp.headers.get("content-type", "")
        raw = resp.content

        if len(raw) > self._max_fetch_bytes:
            raw = raw[: self._max_fetch_bytes]

        # Non-HTML → return as plaintext
        if "text/html" not in content_type and extract_text:
            text = raw.decode("utf-8", errors="replace")
            return _truncate_text(text, max_chars, url)

        html = raw.decode("utf-8", errors="replace")

        if not extract_text:
            return _truncate_text(html, max_chars, url)

        # Text extraction with trafilatura
        text = _extract_text_from_html(html, url)

        # Auto mode: Jina fallback when content is short (<200 chars)
        if reader_mode == "auto" and len(text.strip()) < 200:
            log.info("trafilatura_short_trying_jina", url=url, chars=len(text.strip()))
            try:
                jina_text = await self._fetch_via_jina(validated)
                if len(jina_text.strip()) > len(text.strip()):
                    text = jina_text
            except Exception as jina_exc:
                log.debug("jina_fallback_failed", url=url, error=str(jina_exc))

        return _truncate_text(text, max_chars, url)

    # ── http_request ────────────────────────────────────────────────────────

    async def http_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Fuehrt einen HTTP-Request aus (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS).

        Args:
            url: Ziel-URL.
            method: HTTP-Methode.
            headers: Optionale HTTP-Headers.
            body: Request-Body (fuer POST/PUT/PATCH).
            timeout_seconds: Timeout in Sekunden (1-120).

        Returns:
            Formatierte Response mit Status-Code und Body.
        """
        allowed_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
        method = method.upper()
        if method not in allowed_methods:
            raise WebError(
                f"Ungültige HTTP-Methode: {method}. Erlaubt: {', '.join(sorted(allowed_methods))}"
            )

        validated = await self._validate_url(url)

        # Check domain filter
        parsed = urlparse(validated)
        hostname = (parsed.hostname or "").lower()
        self._check_domain_allowed(hostname)

        # SSRF protection
        if _is_private_host(hostname):
            raise WebError(f"Zugriff auf private Adresse blockiert: {hostname}")

        # Limit body size
        max_body = self._http_request_max_body
        if body and len(body) > max_body:
            raise WebError(f"Request-Body zu groß: {len(body)} Bytes (max {max_body} Bytes)")

        # Timeout: Default aus Config, clampen auf 1-120
        if timeout_seconds == 30:
            timeout_seconds = self._http_request_timeout
        timeout_seconds = min(max(timeout_seconds, 1), 120)

        # Rate-Limiting
        if self._http_request_rate_limit > 0:
            import anyio

            now = time.monotonic()
            elapsed = now - self._http_request_last_call
            if elapsed < self._http_request_rate_limit:
                wait = self._http_request_rate_limit - elapsed
                log.debug("http_request_rate_limit_wait", wait_s=round(wait, 2))
                await anyio.sleep(wait)
        self._http_request_last_call = time.monotonic()

        max_chars = self._max_text_chars

        try:
            async with httpx.AsyncClient(
                timeout=float(timeout_seconds),
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                resp = await client.request(
                    method,
                    validated,
                    headers=headers,
                    content=body,
                )
        except httpx.TimeoutException as exc:
            raise WebError(f"Timeout nach {timeout_seconds}s für {url}") from exc
        except httpx.RequestError as exc:
            raise WebError(f"Request fehlgeschlagen für {url}: {exc}") from exc

        ct = resp.headers.get("content-type", "")
        body_text = resp.text[:max_chars] if resp.text else ""

        return f"HTTP {resp.status_code}\nContent-Type: {ct}\n\n{body_text}"

    async def _fetch_via_jina(self, url: str) -> str:
        """Fetcht eine URL ueber den Jina AI Reader Service.

        Jina Reader extrahiert sauber formatierten Inhalt, besonders
        gut fuer JS-heavy/SPA-Seiten wo trafilatura versagt.

        Args:
            url: Die abzurufende URL.

        Returns:
            Extrahierter Text.
        """
        jina_url = f"https://r.jina.ai/{url}"
        headers: dict[str, str] = {
            "Accept": "text/plain",
        }
        if self._jina_api_key:
            headers["Authorization"] = f"Bearer {self._jina_api_key}"

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(jina_url, headers=headers)
                resp.raise_for_status()
                text = resp.text
                if text.strip():
                    log.info("jina_fetch_ok", url=url, chars=len(text))
                    return text
                raise WebError(f"Jina Reader: Leere Antwort für {url}")
            except httpx.HTTPStatusError as exc:
                raise WebError(f"Jina Reader HTTP {exc.response.status_code} für {url}") from exc
            except httpx.RequestError as exc:
                raise WebError(f"Jina Reader Verbindungsfehler für {url}: {exc}") from exc

    # ── Combination: Search + Fetch ─────────────────────────────────────────

    async def search_and_read(
        self,
        query: str,
        num_results: int = 3,
        language: str = "de",
        cross_check: bool = False,
    ) -> str:
        """Sucht im Web und liest die Top-Ergebnisse.

        Kombiniert web_search + web_fetch fuer tiefere Recherche.

        Args:
            query: Suchanfrage.
            num_results: Anzahl der zu lesenden Seiten.
            language: Suchsprache.
            cross_check: Wenn True, wird ein Quellenvergleich angehaengt.

        Returns:
            Zusammengefasste Inhalte der Top-Ergebnisse.
        """
        search_results = await self.web_search(query, num_results, language)

        # URLs aus den Suchergebnissen extrahieren (begrenzt auf num_results)
        urls = re.findall(r"URL: (https?://[^\s]+)", search_results)[:num_results]
        if not urls:
            return search_results

        parts = [f"## Suchergebnisse für: {query}\n"]
        fetched_contents: list[dict[str, str]] = []

        for i, url in enumerate(urls[:num_results], 1):
            try:
                content = await self.web_fetch(url, max_chars=self._search_and_read_max_chars)
                parts.append(f"\n### [{i}] {url}\n{content}\n")
                fetched_contents.append({"url": url, "content": content})
            except WebError as exc:
                parts.append(f"\n### [{i}] {url}\nFehler: {exc}\n")

        # Append source comparison
        if cross_check and len(fetched_contents) >= 2:
            parts.append("\n---\n## Quellenvergleich\n")
            parts.append(
                "Die folgenden Quellen wurden abgerufen. "
                "Übereinstimmungen und Widersprüche sollten beachtet werden:\n"
            )
            for i, fc in enumerate(fetched_contents, 1):
                domain = urlparse(fc["url"]).hostname or fc["url"]
                # Erste 200 Zeichen als Kurzfassung
                preview = fc["content"][:200].replace("\n", " ").strip()
                parts.append(f"- **Quelle {i}** ({domain}): {preview}...")
            parts.append(
                "\n*Hinweis: Automatischer Quellenvergleich. "
                "Bei widersprüchlichen Angaben die Primärquelle bevorzugen.*"
            )

        return "\n".join(parts)


# ── Helper functions ────────────────────────────────────────────────────────


def _extract_text_from_html(html: str, url: str = "") -> str:
    """Extrahiert lesbaren Text aus HTML.

    Versucht trafilatura, Fallback auf einfache Regex-Extraktion.

    Args:
        html: HTML-Inhalt.
        url: Original-URL (fuer trafilatura-Kontext).

    Returns:
        Extrahierter Text.
    """
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
        if text:
            return text
    except ImportError:
        log.debug("trafilatura nicht installiert, nutze Fallback")
    except Exception:
        log.debug("trafilatura-Extraktion fehlgeschlagen, nutze Fallback")

    # Fallback: simple regex extraction
    return _simple_html_to_text(html)


class _TextExtractor(HTMLParser):
    """Einfache HTML-Parser-Klasse zum Extrahieren von Text.

    Ignoriert Inhalt von <script> und <style> und fuegt fuer bestimmte
    Block-Elemente Zeilenumbrueche ein.
    """

    _BLOCK_TAGS = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._texts: list[str] = []
        self._in_script_or_style = False

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._in_script_or_style = True
            return
        if tag_lower in self._BLOCK_TAGS:
            # Treat block elements as line breaks
            self._texts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._in_script_or_style = False
            return
        if tag_lower in self._BLOCK_TAGS:
            self._texts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if not self._in_script_or_style:
            self._texts.append(data)

    def get_text(self) -> str:
        return "".join(self._texts)


def _simple_html_to_text(html: str) -> str:
    """Einfache HTML→Text-Konvertierung als Fallback.

    Entfernt Tags, Scripts, Styles und normalisiert Whitespace.
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = parser.get_text()
    # HTML entities (in addition to convert_charrefs)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_search_results(results: list[dict[str, Any]], query: str) -> str:
    """Formatiert Suchergebnisse einheitlich.

    Args:
        results: Liste von Ergebnis-Dicts (title, url, content).
        query: Original-Suchanfrage.

    Returns:
        Formatierter Text.
    """
    lines = [f"Suchergebnisse für: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Kein Titel")
        url = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))
        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {url}")
        if snippet:
            lines.append(f"    {snippet[:600]}")
        lines.append("")
    return "\n".join(lines)


def _format_news_results(results: list[dict[str, Any]], query: str) -> str:
    """Formatiert Nachrichtenergebnisse mit Quelle und Datum.

    Args:
        results: Liste von News-Dicts (title, url, content, source, date).
        query: Original-Suchanfrage.

    Returns:
        Formatierter Text.
    """
    lines = [f"Nachrichtenergebnisse für: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Kein Titel")
        url = r.get("url", "")
        snippet = r.get("content", "")
        source = r.get("source", "")
        date = r.get("date", "")
        header = f"[{i}] {title}"
        if source:
            header += f"  ({source})"
        lines.append(header)
        if date:
            lines.append(f"    Datum: {date}")
        lines.append(f"    URL: {url}")
        if snippet:
            lines.append(f"    {snippet[:600]}")
        lines.append("")
    return "\n".join(lines)


def _truncate_text(text: str, max_chars: int, url: str = "") -> str:
    """Kuerzt Text auf maximale Zeichenanzahl.

    Args:
        text: Der zu kuerzende Text.
        max_chars: Maximale Zeichenanzahl.
        url: Quell-URL fuer Hinweis.

    Returns:
        Gekuerzter Text mit Hinweis.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Truncate at last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        truncated = truncated[: last_period + 1]

    return truncated + f"\n\n[... gekürzt: {len(truncated)}/{len(text)} Zeichen, Quelle: {url}]"


def _is_private_host(hostname: str) -> bool:
    """Prueft ob ein Hostname auf eine private Adresse zeigt.

    Blockiert: 10.x.x.x, 172.16-31.x.x, 192.168.x.x, 127.x.x.x,
    fc00::/7, fe80::/10, ::1, ::, 0.0.0.0

    Args:
        hostname: Der zu pruefende Hostname.

    Returns:
        True wenn privat.
    """
    # Strip IPv6 brackets if present
    h = hostname.strip("[]").lower()

    # IPv6 checks (before IPv4 to handle mapped addresses)
    if ":" in h:
        # fc00::/7 (unique local)
        if h.startswith(("fc", "fd")):
            return True
        # fe80::/10 (link-local)
        if h.startswith("fe80"):
            return True
        # Loopback and unspecified
        if h in ("::", "::1", "0:0:0:0:0:0:0:0", "0:0:0:0:0:0:0:1"):
            return True
        # IPv4-mapped IPv6 (::ffff:10.0.0.1)
        if h.startswith("::ffff:"):
            ipv4_part = h[7:]
            if "." in ipv4_part:
                return _is_private_host(ipv4_part)
        return False

    # Direct IPv4 check
    parts = h.split(".")
    if len(parts) == 4:
        try:
            octets = [int(p) for p in parts]
            if octets[0] == 10:
                return True
            if octets[0] == 172 and 16 <= octets[1] <= 31:
                return True
            if octets[0] == 192 and octets[1] == 168:
                return True
            if octets[0] == 127:
                return True
            if octets[0] == 0:
                return True
            # Link-local
            if octets[0] == 169 and octets[1] == 254:
                return True
        except ValueError:
            pass

    return False


# ── MCP client registration ──────────────────────────────────────────────


def register_web_tools(
    mcp_client: Any,
    config: Any | None = None,
    searxng_url: str | None = None,
    brave_api_key: str | None = None,
) -> WebTools:
    """Registriert Web-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: JarvisConfig (optional).
        searxng_url: SearXNG Base-URL (optional, ueberschreibt Config).
        brave_api_key: Brave Search API Key (optional, ueberschreibt Config).

    Returns:
        WebTools-Instanz.
    """
    web = WebTools(
        config=config,
        searxng_url=searxng_url,
        brave_api_key=brave_api_key,
    )

    mcp_client.register_builtin_handler(
        "web_search",
        web.web_search,
        description=(
            "Websuche durchführen. Gibt formatierte Suchergebnisse "
            "mit Titel, URL und Snippet zurück. "
            "Unterstützt Zeitfilter für aktuelle Ergebnisse."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Anzahl Ergebnisse (1-10, Default: 5)",
                    "default": 5,
                },
                "language": {
                    "type": "string",
                    "description": "Sprachcode (Default: de)",
                    "default": "de",
                },
                "timelimit": {
                    "type": "string",
                    "enum": ["", "d", "w", "m", "y"],
                    "description": "Zeitfilter: d=Tag, w=Woche, m=Monat, y=Jahr, leer=alle",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "web_fetch",
        web.web_fetch,
        description=(
            "URL abrufen und Haupttext extrahieren. "
            "Nutzt trafilatura für saubere Text-Extraktion, mit automatischem "
            "Jina AI Reader Fallback für JS-heavy Seiten."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Die abzurufende URL (http/https)",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Text extrahieren (True) oder Raw-HTML (False)",
                    "default": True,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximale Zeichenanzahl (Default: 20000)",
                    "default": 20000,
                },
                "reader_mode": {
                    "type": "string",
                    "enum": ["auto", "trafilatura", "jina"],
                    "description": (
                        "Extraktions-Modus: auto (Standard, "
                        "Jina-Fallback bei wenig Inhalt), "
                        "trafilatura, jina"
                    ),
                    "default": "auto",
                },
            },
            "required": ["url"],
        },
    )

    mcp_client.register_builtin_handler(
        "search_and_read",
        web.search_and_read,
        description=(
            "Kombinierte Websuche + Fetch: Sucht im Web und liest "
            "die Top-Ergebnisse. Ideal für tiefere Recherche. "
            "Mit cross_check=true werden Quellen verglichen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Anzahl zu lesender Seiten (1-5, Default: 3)",
                    "default": 3,
                },
                "language": {
                    "type": "string",
                    "description": "Sprachcode (Default: de)",
                    "default": "de",
                },
                "cross_check": {
                    "type": "boolean",
                    "description": "Quellenvergleich anhängen (Default: false)",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "web_news_search",
        web.web_news_search,
        description=(
            "Nachrichtensuche durchführen. Gibt aktuelle Nachrichten "
            "mit Titel, Quelle, Datum und Snippet zurück. "
            "Ideal für aktuelle Ereignisse und Neuigkeiten."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Anzahl Ergebnisse (1-10, Default: 5)",
                    "default": 5,
                },
                "language": {
                    "type": "string",
                    "description": "Sprachcode (Default: de)",
                    "default": "de",
                },
                "timelimit": {
                    "type": "string",
                    "enum": ["d", "w", "m"],
                    "description": "Zeitfilter: d=Tag, w=Woche, m=Monat (Default: w)",
                    "default": "w",
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "http_request",
        web.http_request,
        description=(
            "HTTP-Request ausführen (GET/POST/PUT/PATCH/DELETE). "
            "Für API-Aufrufe, Webhooks und REST-Interaktionen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Ziel-URL",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                    "description": "HTTP-Methode (Default: GET)",
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP-Headers als Key-Value-Paare",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "Request-Body (für POST/PUT/PATCH)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in Sekunden (1-120, Default: 30)",
                    "default": 30,
                },
            },
            "required": ["url"],
        },
    )

    log.info(
        "web_tools_registered",
        tools=["web_search", "web_news_search", "web_fetch", "search_and_read", "http_request"],
    )
    return web
