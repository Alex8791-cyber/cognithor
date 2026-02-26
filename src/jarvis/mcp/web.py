"""Web-Tools für Jarvis: Suche und URL-Fetch.

Ermöglicht dem Agenten Webrecherche und Seiteninhalt-Extraktion.

Tools:
  - web_search: Websuche über SearXNG oder Brave Search API
  - web_fetch: URL abrufen und Text extrahieren (via trafilatura)

Bibel-Referenz: §5.3 (jarvis-web Server)
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
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

    Unterstützt zwei Such-Backends:
      1. SearXNG (self-hosted, bevorzugt)
      2. Brave Search API (Fallback)

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

        # Aus Config laden falls vorhanden
        if config is not None:
            web_cfg = getattr(config, "web", None)
            if web_cfg is not None:
                self._searxng_url = self._searxng_url or getattr(web_cfg, "searxng_url", None) or ""
                self._brave_api_key = self._brave_api_key or getattr(web_cfg, "brave_api_key", None) or ""
                self._duckduckgo_enabled = getattr(web_cfg, "duckduckgo_enabled", True)

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

    # ── web_search ─────────────────────────────────────────────────────────

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "de",
    ) -> str:
        """Führt eine Websuche durch.

        Versucht zuerst SearXNG, dann Brave Search als Fallback.

        Args:
            query: Suchanfrage.
            num_results: Anzahl gewünschter Ergebnisse (1-10).
            language: Sprache für Suchergebnisse.

        Returns:
            Formatierte Suchergebnisse als Text.
        """
        if not query.strip():
            return "Keine Suchanfrage angegeben."

        num_results = min(max(num_results, 1), MAX_SEARCH_RESULTS)

        # SearXNG versuchen
        if self._searxng_url:
            try:
                return await self._search_searxng(query, num_results, language)
            except Exception as exc:
                log.warning("SearXNG-Suche fehlgeschlagen: %s", exc)

        # Brave Search versuchen
        if self._brave_api_key:
            try:
                return await self._search_brave(query, num_results, language)
            except Exception as exc:
                log.warning("Brave-Suche fehlgeschlagen: %s", exc)

        # DuckDuckGo als kostenloser Fallback (kein API-Key nötig)
        if self._duckduckgo_enabled:
            try:
                return await self._search_duckduckgo(query, num_results, language)
            except Exception as exc:
                log.warning("DuckDuckGo-Suche fehlgeschlagen: %s", exc)

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

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        language: str,
    ) -> str:
        """Suche über DuckDuckGo (kein API-Key nötig).

        Nutzt die duckduckgo-search Bibliothek für zuverlässige Ergebnisse.
        Kostenlos und ohne Registrierung nutzbar.
        """
        import anyio

        def _sync_search() -> list[dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    raise WebError(
                        "ddgs nicht installiert. "
                        "Installiere mit: pip install ddgs"
                    )

            # Region-Mapping: Sprachcode -> DuckDuckGo Region
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
            }
            region = region_map.get(language, "wt-wt")

            raw = list(DDGS().text(query, region=region, max_results=num_results))

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                }
                for r in raw
            ]

        results = await anyio.to_thread.run_sync(_sync_search)

        if not results:
            return f"Keine Ergebnisse für: {query}"

        return _format_search_results(results, query)

    # ── web_fetch ──────────────────────────────────────────────────────────

    async def web_fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_chars: int | None = None,
    ) -> str:
        """Ruft eine URL ab und extrahiert den Text.

        Args:
            url: Die abzurufende URL.
            extract_text: Text extrahieren (True) oder Raw-HTML (False).
            max_chars: Maximale Zeichenanzahl (Default: MAX_TEXT_CHARS).

        Returns:
            Extrahierter Text oder HTML-Inhalt.
        """
        validated = self._validate_url(url)
        max_chars = max_chars or MAX_TEXT_CHARS

        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            try:
                resp = await client.get(validated)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise WebError(f"HTTP {exc.response.status_code} für {url}") from exc
            except httpx.RequestError as exc:
                raise WebError(f"Verbindungsfehler für {url}: {exc}") from exc

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
        return _truncate_text(text, max_chars, url)

    # ── Kombination: Suche + Fetch ─────────────────────────────────────────

    async def search_and_read(
        self,
        query: str,
        num_results: int = 3,
        language: str = "de",
    ) -> str:
        """Sucht im Web und liest die Top-Ergebnisse.

        Kombiniert web_search + web_fetch für tiefere Recherche.

        Args:
            query: Suchanfrage.
            num_results: Anzahl der zu lesenden Seiten.
            language: Suchsprache.

        Returns:
            Zusammengefasste Inhalte der Top-Ergebnisse.
        """
        search_results = await self.web_search(query, num_results, language)

        # URLs aus den Suchergebnissen extrahieren (begrenzt auf num_results)
        urls = re.findall(r"URL: (https?://[^\s]+)", search_results)[:num_results]
        if not urls:
            return search_results

        parts = [f"## Suchergebnisse für: {query}\n"]

        for i, url in enumerate(urls[:num_results], 1):
            try:
                content = await self.web_fetch(url, max_chars=5000)
                parts.append(f"\n### [{i}] {url}\n{content}\n")
            except WebError as exc:
                parts.append(f"\n### [{i}] {url}\nFehler: {exc}\n")

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
            lines.append(f"    {snippet[:300]}")
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
            "mit Titel, URL und Snippet zurück."
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
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "web_fetch",
        web.web_fetch,
        description=(
            "URL abrufen und Haupttext extrahieren. "
            "Nutzt trafilatura für saubere Text-Extraktion aus HTML."
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
            },
            "required": ["url"],
        },
    )

    mcp_client.register_builtin_handler(
        "search_and_read",
        web.search_and_read,
        description=(
            "Kombinierte Websuche + Fetch: Sucht im Web und liest "
            "die Top-Ergebnisse. Ideal für tiefere Recherche."
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
            },
            "required": ["query"],
        },
    )

    log.info("web_tools_registered", tools=["web_search", "web_fetch", "search_and_read"])
    return web
