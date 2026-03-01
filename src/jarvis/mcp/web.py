"""Web-Tools für Jarvis: Suche und URL-Fetch.

Ermöglicht dem Agenten Webrecherche und Seiteninhalt-Extraktion.

Tools:
  - web_search: Websuche über SearXNG, Brave oder DuckDuckGo (Multi-Backend)
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

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

MAX_FETCH_BYTES = 500_000  # 500 KB maximaler Fetch
MAX_TEXT_CHARS = 20_000  # 20K Zeichen extrahierter Text
FETCH_TIMEOUT = 15  # Sekunden
SEARCH_TIMEOUT = 10  # Sekunden
MAX_SEARCH_RESULTS = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# DuckDuckGo-Optimierung: Backends in Fallback-Reihenfolge
DDG_BACKENDS = ("duckduckgo", "bing", "google", "brave")
# Mindestabstand zwischen DuckDuckGo-Suchen (Sekunden)
DDG_MIN_DELAY = 2.0
# Wartezeit bei Rate-Limiting (Sekunden)
DDG_RATELIMIT_WAIT = 30
# Cache-TTL (Sekunden) — Standard: 1 Stunde
DDG_CACHE_TTL = 3600

# Blocked Domains (Sicherheit)
BLOCKED_DOMAINS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::",           # IPv6 unspecified (urlparse strips brackets)
        "::1",          # IPv6 loopback (urlparse strips brackets)
        "metadata.google.internal",
        "169.254.169.254",  # AWS metadata
    }
)


__all__ = [
    "WebTools",
    "WebError",
    "register_web_tools",
]


class WebError(Exception):
    """Fehler bei Web-Operationen."""


class WebTools:
    """Web-Recherche und URL-Fetch-Tools. [B§5.3]

    Unterstützt vier Such-Backends (Fallback-Kette):
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

        # Google CSE
        self._google_cse_api_key = ""
        self._google_cse_cx = ""

        # Jina AI Reader
        self._jina_api_key = ""

        # Domain-Filtering
        self._domain_blocklist: list[str] = []
        self._domain_allowlist: list[str] = []

        # DuckDuckGo-Optimierung: State
        self._ddg_last_search: float = 0.0
        self._ddg_cache_dir: Path | None = None

        # Aus Config laden falls vorhanden
        if config is not None:
            web_cfg = getattr(config, "web", None)
            if web_cfg is not None:
                self._searxng_url = self._searxng_url or getattr(web_cfg, "searxng_url", None) or ""
                self._brave_api_key = self._brave_api_key or getattr(web_cfg, "brave_api_key", None) or ""
                self._google_cse_api_key = getattr(web_cfg, "google_cse_api_key", "") or ""
                self._google_cse_cx = getattr(web_cfg, "google_cse_cx", "") or ""
                self._jina_api_key = getattr(web_cfg, "jina_api_key", "") or ""
                self._duckduckgo_enabled = getattr(web_cfg, "duckduckgo_enabled", True)
                self._domain_blocklist = list(getattr(web_cfg, "domain_blocklist", []))
                self._domain_allowlist = list(getattr(web_cfg, "domain_allowlist", []))

            # Cache-Verzeichnis: ~/.jarvis/cache/web_search/
            jarvis_home = getattr(config, "jarvis_home", None)
            if jarvis_home:
                self._ddg_cache_dir = Path(jarvis_home) / "cache" / "web_search"

        if self._ddg_cache_dir is None:
            self._ddg_cache_dir = Path.home() / ".jarvis" / "cache" / "web_search"

    def _validate_url(self, url: str) -> str:
        """Validiert eine URL gegen SSRF-Angriffe.

        Args:
            url: Die zu validierende URL.

        Returns:
            Validierte URL.

        Raises:
            WebError: Bei ungültiger oder blockierter URL.
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

        # Private IP-Bereiche blockieren
        if _is_private_host(hostname):
            raise WebError(f"Zugriff auf private Adressen blockiert: {hostname}")

        # DNS-Resolution prüfen um DNS-Rebinding/Bypass zu verhindern
        import socket
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, _type, _proto, _canonname, sockaddr in resolved:
                ip = sockaddr[0]
                if ip in BLOCKED_DOMAINS or _is_private_host(ip):
                    raise WebError(f"DNS für {hostname} zeigt auf blockierte Adresse: {ip}")
        except socket.gaierror:
            raise WebError(f"DNS-Aufloesung fehlgeschlagen fuer {hostname}") from None

        return url

    def _check_domain_allowed(self, hostname: str) -> None:
        """Prüft ob eine Domain durch die Blocklist/Allowlist erlaubt ist.

        Args:
            hostname: Der zu prüfende Hostname.

        Raises:
            WebError: Wenn die Domain blockiert ist.
        """
        hostname = hostname.lower().strip()
        if not hostname:
            return

        # Allowlist-Modus: Wenn gesetzt, NUR diese Domains erlauben
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

        # Blocklist: Diese Domains blockieren
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
        """Führt eine Websuche durch.

        Fallback-Kette: SearXNG → Brave → DuckDuckGo (Multi-Backend).

        Args:
            query: Suchanfrage.
            num_results: Anzahl gewünschter Ergebnisse (1-10).
            language: Sprache für Suchergebnisse.
            timelimit: Zeitfilter ('d'=Tag, 'w'=Woche, 'm'=Monat, 'y'=Jahr, ''=alle).

        Returns:
            Formatierte Suchergebnisse als Text.
        """
        if not query.strip():
            return "Keine Suchanfrage angegeben."

        num_results = min(max(num_results, 1), MAX_SEARCH_RESULTS)
        provider_errors: list[str] = []

        # SearXNG versuchen
        if self._searxng_url:
            try:
                return await self._search_searxng(query, num_results, language)
            except Exception as exc:
                provider_errors.append(f"SearXNG: {exc}")
                log.warning("SearXNG-Suche fehlgeschlagen: %s", exc)

        # Brave Search versuchen
        if self._brave_api_key:
            try:
                return await self._search_brave(query, num_results, language)
            except Exception as exc:
                provider_errors.append(f"Brave: {exc}")
                log.warning("Brave-Suche fehlgeschlagen: %s", exc)

        # Google Custom Search Engine versuchen
        if self._google_cse_api_key and self._google_cse_cx:
            try:
                return await self._search_google_cse(query, num_results, language)
            except Exception as exc:
                provider_errors.append(f"Google CSE: {exc}")
                log.warning("Google-CSE-Suche fehlgeschlagen: %s", exc)

        # DuckDuckGo mit Multi-Backend-Fallback, Cache und Rate-Limiting
        if self._duckduckgo_enabled:
            try:
                return await self._search_duckduckgo(query, num_results, language, timelimit)
            except Exception as exc:
                provider_errors.append(f"DuckDuckGo: {exc}")
                log.warning("DuckDuckGo-Suche fehlgeschlagen: %s", exc)

        # Alle Provider fehlgeschlagen → detaillierte Fehlermeldung
        if provider_errors:
            error_details = "\n".join(f"  • {e}" for e in provider_errors)
            return (
                f"Websuche für '{query}' fehlgeschlagen. "
                f"Alle Such-Provider haben Fehler gemeldet:\n{error_details}\n\n"
                "Mögliche Ursachen: Netzwerkprobleme, ungültige API-Keys, Rate-Limits."
            )

        return (
            "Keine Suchengine konfiguriert.\n"
            "Setze `searxng_url` oder `brave_api_key` in der Konfiguration,\n"
            "oder aktiviere `duckduckgo_enabled: true` (Standard)."
        )

    async def _search_searxng(
        self,
        query: str,
        num_results: int,
        language: str,
    ) -> str:
        """Suche über SearXNG-Instanz."""
        url = f"{self._searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "categories": "general",
        }

        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
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
        """Suche über Brave Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        # API-Key direkt als Header setzen, nie loggen
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

        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        web_results = data.get("web", {}).get("results", [])[:num_results]
        if not web_results:
            return f"Keine Ergebnisse für: {query}"

        # Brave-Format → einheitliches Format konvertieren
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
        """Suche über Google Custom Search Engine API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self._google_cse_api_key,
            "cx": self._google_cse_cx,
            "q": query,
            "num": str(min(num_results, 10)),
            "lr": f"lang_{language}",
        }

        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])[:num_results]
        if not items:
            return f"Keine Ergebnisse für: {query}"

        # Google-CSE-Format → einheitliches Format konvertieren
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
            num_results: Anzahl gewünschter Ergebnisse.
            language: Sprache (ISO-Code).
            timelimit: Zeitfilter ('d', 'w', 'm', 'y', '' für alle).
        """
        import anyio

        # Region-Mapping
        region_map = {
            "de": "de-de", "en": "us-en", "fr": "fr-fr", "es": "es-es",
            "it": "it-it", "pt": "pt-pt", "nl": "nl-nl", "ja": "jp-jp",
            "zh": "cn-zh", "ru": "ru-ru", "ko": "kr-ko", "pl": "pl-pl",
        }
        region = region_map.get(language, "wt-wt")
        tl = timelimit if timelimit in ("d", "w", "m", "y") else None

        # 1. Cache prüfen
        cached = self._ddg_cache_get(query, region, tl, num_results)
        if cached is not None:
            log.info("ddg_cache_hit", query=query[:60])
            return _format_search_results(cached, query)

        # 2. Rate-Limiting: Mindestabstand einhalten
        now = time.monotonic()
        elapsed = now - self._ddg_last_search
        if elapsed < DDG_MIN_DELAY:
            wait = DDG_MIN_DELAY - elapsed
            log.debug("ddg_rate_limit_wait", wait_s=round(wait, 1))
            await anyio.sleep(wait)

        # 3. Multi-Backend-Suche mit Fallback
        results = await anyio.to_thread.run_sync(
            lambda: self._ddg_search_with_fallback(query, region, tl, num_results)
        )
        self._ddg_last_search = time.monotonic()

        if not results:
            return f"Keine Ergebnisse für: {query}"

        # 4. Cache speichern
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
        Bei RateLimit: 30s warten, dann nächstes Backend.
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
                    DDGS(timeout=SEARCH_TIMEOUT).text(
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
                # Leere Ergebnisse → nächstes Backend probieren
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
                    # Bei Rate-Limit kurz warten bevor nächstes Backend versucht wird
                    time.sleep(min(DDG_RATELIMIT_WAIT, 5))
                else:
                    log.warning(
                        "ddg_backend_error",
                        backend=backend,
                        error=str(exc)[:200],
                    )

        # Alle Backends gescheitert
        if last_error:
            raise WebError(f"DuckDuckGo-Suche fehlgeschlagen (alle Backends): {last_error}") from last_error
        return []

    # ── DuckDuckGo News-Suche ──────────────────────────────────────────────

    async def web_news_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "de",
        timelimit: str = "w",
    ) -> str:
        """Sucht aktuelle Nachrichten über DuckDuckGo News.

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

        num_results = min(max(num_results, 1), MAX_SEARCH_RESULTS)

        import anyio

        region_map = {
            "de": "de-de", "en": "us-en", "fr": "fr-fr", "es": "es-es",
        }
        region = region_map.get(language, "wt-wt")
        tl = timelimit if timelimit in ("d", "w", "m") else "w"

        # Rate-Limiting
        now = time.monotonic()
        elapsed = now - self._ddg_last_search
        if elapsed < DDG_MIN_DELAY:
            await anyio.sleep(DDG_MIN_DELAY - elapsed)

        def _sync_news() -> list[dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # type: ignore[no-redef]
                except ImportError:
                    raise WebError("ddgs nicht installiert. pip install ddgs") from None

            raw = list(
                DDGS(timeout=SEARCH_TIMEOUT).news(
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
            if time.time() - data.get("ts", 0) > DDG_CACHE_TTL:
                # Abgelaufen → löschen
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
            max_chars: Maximale Zeichenanzahl (Default: MAX_TEXT_CHARS).
            reader_mode: Extraktions-Modus:
                'auto' = trafilatura, bei <200 Zeichen Fallback auf Jina
                'trafilatura' = nur trafilatura
                'jina' = nur Jina AI Reader

        Returns:
            Extrahierter Text oder HTML-Inhalt.
        """
        validated = self._validate_url(url)
        max_chars = max_chars or MAX_TEXT_CHARS

        # Domain-Filter prüfen
        parsed = urlparse(validated)
        hostname = (parsed.hostname or "").lower()
        self._check_domain_allowed(hostname)

        # Jina-only Modus
        if reader_mode == "jina":
            text = await self._fetch_via_jina(validated)
            return _truncate_text(text, max_chars, url)

        # Standard-Fetch
        fetch_failed = False
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                resp = await client.get(validated)
                resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if reader_mode == "trafilatura":
                raise WebError(f"Fetch fehlgeschlagen für {url}: {exc}") from exc
            # auto-Modus: Bei Fehler Jina als Fallback versuchen
            log.warning("web_fetch_failed_trying_jina", url=url, error=str(exc))
            fetch_failed = True

        if fetch_failed:
            text = await self._fetch_via_jina(validated)
            return _truncate_text(text, max_chars, url)

        content_type = resp.headers.get("content-type", "")
        raw = resp.content

        if len(raw) > MAX_FETCH_BYTES:
            raw = raw[:MAX_FETCH_BYTES]

        # Nicht-HTML → als Plaintext zurückgeben
        if "text/html" not in content_type and extract_text:
            text = raw.decode("utf-8", errors="replace")
            return _truncate_text(text, max_chars, url)

        html = raw.decode("utf-8", errors="replace")

        if not extract_text:
            return _truncate_text(html, max_chars, url)

        # Text-Extraktion mit trafilatura
        text = _extract_text_from_html(html, url)

        # Auto-Modus: Bei wenig Inhalt (<200 Zeichen) Jina-Fallback
        if reader_mode == "auto" and len(text.strip()) < 200:
            log.info("trafilatura_short_trying_jina", url=url, chars=len(text.strip()))
            try:
                jina_text = await self._fetch_via_jina(validated)
                if len(jina_text.strip()) > len(text.strip()):
                    text = jina_text
            except Exception as jina_exc:
                log.debug("jina_fallback_failed", url=url, error=str(jina_exc))

        return _truncate_text(text, max_chars, url)

    async def _fetch_via_jina(self, url: str) -> str:
        """Fetcht eine URL über den Jina AI Reader Service.

        Jina Reader extrahiert sauber formatierten Inhalt, besonders
        gut für JS-heavy/SPA-Seiten wo trafilatura versagt.

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

    # ── Kombination: Suche + Fetch ─────────────────────────────────────────

    async def search_and_read(
        self,
        query: str,
        num_results: int = 3,
        language: str = "de",
        cross_check: bool = False,
    ) -> str:
        """Sucht im Web und liest die Top-Ergebnisse.

        Kombiniert web_search + web_fetch für tiefere Recherche.

        Args:
            query: Suchanfrage.
            num_results: Anzahl der zu lesenden Seiten.
            language: Suchsprache.
            cross_check: Wenn True, wird ein Quellenvergleich angehängt.

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
                content = await self.web_fetch(url, max_chars=5000)
                parts.append(f"\n### [{i}] {url}\n{content}\n")
                fetched_contents.append({"url": url, "content": content})
            except WebError as exc:
                parts.append(f"\n### [{i}] {url}\nFehler: {exc}\n")

        # Quellenvergleich anhängen
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


# ── Hilfsfunktionen ────────────────────────────────────────────────────────


def _extract_text_from_html(html: str, url: str = "") -> str:
    """Extrahiert lesbaren Text aus HTML.

    Versucht trafilatura, Fallback auf einfache Regex-Extraktion.

    Args:
        html: HTML-Inhalt.
        url: Original-URL (für trafilatura-Kontext).

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

    # Fallback: einfache Regex-Extraktion
    return _simple_html_to_text(html)


class _TextExtractor(HTMLParser):
    """Einfache HTML-Parser-Klasse zum Extrahieren von Text.

    Ignoriert Inhalt von <script> und <style> und fügt für bestimmte
    Block-Elemente Zeilenumbrüche ein.
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
            # Block-Elemente als Zeilenumbruch behandeln
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
    # HTML-Entities (zusätzlich zu convert_charrefs)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Whitespace normalisieren
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
    """Kürzt Text auf maximale Zeichenanzahl.

    Args:
        text: Der zu kürzende Text.
        max_chars: Maximale Zeichenanzahl.
        url: Quell-URL für Hinweis.

    Returns:
        Gekürzter Text mit Hinweis.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Am letzten Satzende kürzen
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        truncated = truncated[: last_period + 1]

    return truncated + f"\n\n[... gekürzt, Quelle: {url}]"


def _is_private_host(hostname: str) -> bool:
    """Prüft ob ein Hostname auf eine private Adresse zeigt.

    Blockiert: 10.x.x.x, 172.16-31.x.x, 192.168.x.x, 127.x.x.x,
    fc00::/7, fe80::/10, ::1, ::, 0.0.0.0

    Args:
        hostname: Der zu prüfende Hostname.

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

    # Direkte IPv4-Prüfung
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


# ── MCP-Client-Registrierung ──────────────────────────────────────────────


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
        searxng_url: SearXNG Base-URL (optional, überschreibt Config).
        brave_api_key: Brave Search API Key (optional, überschreibt Config).

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
                    "description": "Extraktions-Modus: auto (Standard, Jina-Fallback bei wenig Inhalt), trafilatura, jina",
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

    log.info("web_tools_registered", tools=["web_search", "web_news_search", "web_fetch", "search_and_read"])
    return web
