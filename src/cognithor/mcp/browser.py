"""Browser-Tool: Web-Automatisierung via Playwright.

Ermoeglicht Jarvis das Navigieren, Lesen, Klicken und Ausfuellen
von Webseiten -- headless und lokal, ohne Cloud-Dienste.

Features:
  - Seiten laden und Text extrahieren
  - Screenshots erstellen
  - Formulare ausfuellen und Buttons klicken
  - JavaScript ausfuehren
  - Cookie- und Session-Management
  - Konfigurierbare Timeouts und Viewport

Benoetigt: pip install playwright && playwright install chromium

MCP-Tool-Registrierung:
  - browse_url: Seite laden und Text/HTML extrahieren
  - browse_screenshot: Screenshot einer Seite erstellen
  - browse_click: Element anklicken
  - browse_fill: Formularfeld ausfuellen
  - browse_execute_js: JavaScript auf der Seite ausfuehren
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Maximale Textlaenge die ans LLM zurueckgegeben wird
_DEFAULT_MAX_TEXT_LENGTH = 8000

# Maximale JS-Script-Laenge (Zeichen)
_DEFAULT_MAX_JS_LENGTH = 50_000

# Default-Timeouts
_DEFAULT_TIMEOUT_MS = 30_000
_DEFAULT_VIEWPORT = {"width": 1280, "height": 720}

# Backward-compatible aliases
MAX_TEXT_LENGTH = _DEFAULT_MAX_TEXT_LENGTH
MAX_JS_LENGTH = _DEFAULT_MAX_JS_LENGTH
DEFAULT_TIMEOUT_MS = _DEFAULT_TIMEOUT_MS
DEFAULT_VIEWPORT = _DEFAULT_VIEWPORT

__all__ = [
    "BROWSER_TOOL_SCHEMAS",
    "BrowserResult",
    "BrowserTool",
    "BrowserToolError",
    "register_browser_tools",
]


@dataclass
class BrowserResult:
    """Ergebnis einer Browser-Aktion."""

    success: bool = True
    text: str = ""
    url: str = ""
    title: str = ""
    screenshot_path: str | None = None
    error: str | None = None


class BrowserToolError(Exception):
    """Fehler im Browser-Tool."""


class BrowserTool:
    """Headless-Browser via Playwright fuer Web-Automatisierung.

    Verwaltet eine einzelne Browser-Instanz mit einer aktiven Seite.
    Alle Aktionen laufen headless -- kein GUI erforderlich.

    Typische Nutzung:
        tool = BrowserTool(workspace_dir=Path("~/.cognithor/workspace"))
        await tool.initialize()
        result = await tool.navigate("https://example.com")
        await tool.close()
    """

    def __init__(
        self,
        workspace_dir: Path | None = None,
        headless: bool = True,
        timeout_ms: int | None = None,
        config: Any = None,
    ) -> None:
        self._workspace_dir = workspace_dir or Path.home() / ".cognithor" / "workspace"
        self._headless = headless
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._initialized = False

        # Read browser config values, falling back to module-level defaults
        browser_cfg = getattr(config, "browser", None) if config else None
        self._max_text_length: int = getattr(
            browser_cfg, "max_text_length", _DEFAULT_MAX_TEXT_LENGTH
        )
        self._max_js_length: int = getattr(browser_cfg, "max_js_length", _DEFAULT_MAX_JS_LENGTH)
        self._timeout_ms: int = (
            timeout_ms
            if timeout_ms is not None
            else getattr(browser_cfg, "default_timeout_ms", _DEFAULT_TIMEOUT_MS)
        )
        vp_width: int = getattr(browser_cfg, "default_viewport_width", _DEFAULT_VIEWPORT["width"])
        vp_height: int = getattr(
            browser_cfg, "default_viewport_height", _DEFAULT_VIEWPORT["height"]
        )
        self._viewport: dict[str, int] = {"width": vp_width, "height": vp_height}

    async def initialize(self) -> bool:
        """Startet den Browser. Gibt False zurueck wenn Playwright nicht installiert."""
        if self._initialized:
            return True

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            self._context = await self._browser.new_context(
                viewport=self._viewport,
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self._timeout_ms)
            self._initialized = True
            log.info("browser_initialized", headless=self._headless)
            return True

        except ImportError:
            log.error(
                "playwright_not_installed",
                hint="pip install playwright && playwright install chromium",
            )
            return False
        except Exception as exc:
            log.error("browser_init_failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Browser sauber herunterfahren."""
        with contextlib.suppress(Exception):
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if hasattr(self, "_playwright") and self._playwright:
                await self._playwright.stop()
        self._initialized = False
        self._page = None
        self._context = None
        self._browser = None
        log.info("browser_closed")

    @staticmethod
    def _validate_url(url: str) -> str | None:
        """Validates a URL against SSRF. Returns error message or None."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except ValueError:
            return t("browser.invalid_url", url=url)
        if parsed.scheme not in ("http", "https"):
            return t("browser.scheme_not_allowed", scheme=parsed.scheme)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return t("browser.no_valid_domain", url=url)
        _blocked = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::",
            "::1",
            "metadata.google.internal",
            "169.254.169.254",
        }
        if hostname in _blocked:
            return t("browser.host_blocked", hostname=hostname)
        # Private IPv4 ranges
        parts = hostname.split(".")
        if len(parts) == 4:
            try:
                octets = [int(p) for p in parts]
                if octets[0] == 10:
                    return t("browser.private_address_blocked", hostname=hostname)
                if octets[0] == 172 and 16 <= octets[1] <= 31:
                    return t("browser.private_address_blocked", hostname=hostname)
                if octets[0] == 192 and octets[1] == 168:
                    return t("browser.private_address_blocked", hostname=hostname)
            except ValueError:
                pass  # Not a numeric IP address, skip private-range check
        # IPv6 private
        if hostname.startswith(("fc", "fd", "fe80")):
            return t("browser.private_address_blocked", hostname=hostname)
        return None

    async def navigate(self, url: str, *, extract_text: bool = True) -> BrowserResult:
        """Navigiert zu einer URL und extrahiert optional den Text.

        Args:
            url: Ziel-URL.
            extract_text: Ob der sichtbare Text extrahiert werden soll.

        Returns:
            BrowserResult mit Seitentext, URL und Titel.
        """
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        # SSRF-Schutz: URL validieren
        if err := self._validate_url(url):
            return BrowserResult(success=False, url=url, error=err)

        try:
            response = await self._page.goto(url, wait_until="domcontentloaded")

            title = await self._page.title()
            current_url = self._page.url

            text = ""
            if extract_text:
                text = await self._page.inner_text("body")
                if len(text) > self._max_text_length:
                    text = text[: self._max_text_length] + t(
                        "browser.text_truncated", total=len(text)
                    )

            status = response.status if response else 0
            log.info("browser_navigate", url=url, status=status, title=title)

            return BrowserResult(
                success=True,
                text=text,
                url=current_url,
                title=title,
            )
        except Exception as exc:
            log.error("browser_navigate_failed", url=url, error=str(exc))
            return BrowserResult(
                success=False,
                url=url,
                error=t("browser.navigate_failed", exc_type=type(exc).__name__),
            )

    async def screenshot(
        self, path: str | None = None, *, full_page: bool = False
    ) -> BrowserResult:
        """Erstellt einen Screenshot der aktuellen Seite.

        Args:
            path: Speicherpfad. Auto-generiert wenn None.
            full_page: Ob die gesamte Seite oder nur der Viewport erfasst wird.

        Returns:
            BrowserResult mit Pfad zum Screenshot.
        """
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        try:
            if path is None:
                import time

                self._workspace_dir.mkdir(parents=True, exist_ok=True)
                path = str(self._workspace_dir / f"screenshot-{int(time.time())}.png")

            await self._page.screenshot(path=path, full_page=full_page)
            title = await self._page.title()

            log.info("browser_screenshot", path=path)
            return BrowserResult(
                success=True,
                text=t("browser.screenshot_saved", path=path),
                url=self._page.url,
                title=title,
                screenshot_path=path,
            )
        except Exception as exc:
            log.error("browser_screenshot_failed", error=str(exc))
            return BrowserResult(
                success=False, error=t("browser.screenshot_failed", exc_type=type(exc).__name__)
            )

    async def click(self, selector: str) -> BrowserResult:
        """Klickt auf ein Element.

        Args:
            selector: CSS-Selektor oder Text-Selektor (z.B. 'text=Anmelden').

        Returns:
            BrowserResult mit Erfolgsstatus.
        """
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        try:
            await self._page.click(selector)
            await self._page.wait_for_load_state("domcontentloaded")
            title = await self._page.title()

            log.info("browser_click", selector=selector)
            return BrowserResult(
                success=True,
                text=t("browser.click_success", selector=selector),
                url=self._page.url,
                title=title,
            )
        except Exception as exc:
            log.error("browser_click_failed", selector=selector, error=str(exc))
            return BrowserResult(
                success=False, error=t("browser.click_failed", exc_type=type(exc).__name__)
            )

    async def fill(self, selector: str, value: str) -> BrowserResult:
        """Fuellt ein Formularfeld aus.

        Args:
            selector: CSS-Selektor des Input-Feldes.
            value: Einzugebender Text.

        Returns:
            BrowserResult mit Erfolgsstatus.
        """
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        try:
            await self._page.fill(selector, value)

            log.info("browser_fill", selector=selector)
            return BrowserResult(
                success=True,
                text=t("browser.fill_success", selector=selector),
                url=self._page.url,
            )
        except Exception as exc:
            log.error("browser_fill_failed", selector=selector, error=str(exc))
            return BrowserResult(
                success=False, error=t("browser.fill_failed", exc_type=type(exc).__name__)
            )

    async def execute_js(self, script: str) -> BrowserResult:
        """Fuehrt JavaScript auf der aktuellen Seite aus.

        Args:
            script: JavaScript-Code.

        Returns:
            BrowserResult mit dem Rueckgabewert des Scripts.
        """
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        if len(script) > self._max_js_length:
            return BrowserResult(
                success=False,
                error=t("browser.script_too_long", count=len(script), max=self._max_js_length),
            )

        try:
            result = await self._page.evaluate(script)
            result_str = str(result) if result is not None else ""

            if len(result_str) > self._max_text_length:
                result_str = result_str[: self._max_text_length] + t("browser.result_truncated")

            log.info("browser_js_executed", script_length=len(script))
            return BrowserResult(
                success=True,
                text=result_str,
                url=self._page.url,
            )
        except Exception as exc:
            log.error("browser_js_failed", error=str(exc))
            return BrowserResult(
                success=False, error=t("browser.js_failed", exc_type=type(exc).__name__)
            )

    async def get_page_info(self) -> BrowserResult:
        """Gibt Informationen zur aktuellen Seite zurueck."""
        if not self._initialized:
            return BrowserResult(success=False, error=t("browser.not_initialized"))

        try:
            title = await self._page.title()
            url = self._page.url

            # Links und Formulare sammeln
            links = await self._page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .slice(0, 20)
                    .map(a => ({text: a.textContent.trim().slice(0, 50), href: a.href}))
            """)
            inputs = await self._page.evaluate("""
                () => Array.from(document.querySelectorAll('input, textarea, select, button'))
                    .slice(0, 15)
                    .map(el => ({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        name: el.name || '',
                        id: el.id || '',
                        text: (el.textContent || el.value
                          || el.placeholder || '').trim().slice(0, 40)
                    }))
            """)

            info_parts = [t("browser.page_title", title=title), t("browser.page_url", url=url), ""]
            if links:
                info_parts.append(t("browser.links_header"))
                for link in links:
                    if link["text"]:
                        info_parts.append(f"  - [{link['text']}]({link['href']})")
            if inputs:
                info_parts.append(t("browser.form_elements_header"))
                for inp in inputs:
                    desc = f"  - <{inp['tag']}"
                    if inp["type"]:
                        desc += f" type={inp['type']}"
                    if inp["name"]:
                        desc += f" name={inp['name']}"
                    if inp["id"]:
                        desc += f" id={inp['id']}"
                    if inp["text"]:
                        desc += f"> {inp['text']}"
                    else:
                        desc += ">"
                    info_parts.append(desc)

            return BrowserResult(
                success=True,
                text="\n".join(info_parts),
                url=url,
                title=title,
            )
        except Exception as exc:
            log.error("browser_page_info_failed", error=str(exc))
            return BrowserResult(
                success=False, error=t("browser.page_info_failed", exc_type=type(exc).__name__)
            )


# ============================================================================
# MCP-Tool-Registrierung
# ============================================================================

# Tool-Schemas fuer die MCP-Registrierung
BROWSER_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "browse_url": {
        "description": (
            "Navigiert zu einer URL und extrahiert den sichtbaren Text. "
            "Nützlich für Recherche, Preisvergleiche, Nachrichtenlesen."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Ziel-URL (mit https://)"},
                "extract_text": {
                    "type": "boolean",
                    "description": "Seitentext extrahieren (default: true)",
                    "default": True,
                },
            },
            "required": ["url"],
        },
    },
    "browse_screenshot": {
        "description": "Erstellt einen Screenshot der aktuellen Browser-Seite.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "full_page": {
                    "type": "boolean",
                    "description": "Gesamte Seite oder nur sichtbaren Bereich (default: false)",
                    "default": False,
                },
            },
        },
    },
    "browse_click": {
        "description": (
            "Klickt auf ein Element der aktuellen Seite. "
            "Selektor kann CSS sein (z.B. '#submit-btn') oder Text (z.B. 'text=Anmelden')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS- oder Text-Selektor"},
            },
            "required": ["selector"],
        },
    },
    "browse_fill": {
        "description": "Füllt ein Formularfeld auf der aktuellen Seite aus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS-Selektor des Eingabefeldes"},
                "value": {"type": "string", "description": "Einzugebender Text"},
            },
            "required": ["selector", "value"],
        },
    },
    "browse_execute_js": {
        "description": "Führt JavaScript auf der aktuellen Seite aus und gibt das Ergebnis zurück.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "JavaScript-Code"},
            },
            "required": ["script"],
        },
    },
    "browse_page_info": {
        "description": (
            "Gibt eine Übersicht der aktuellen Seite zurück: "
            "Titel, URL, Links und Formular-Elemente."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def register_browser_tools(mcp_client: Any, config: Any = None) -> BrowserTool:
    """Registriert Browser-MCP-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: Optionale Konfiguration mit ``config.browser.*`` Werten.

    Returns:
        BrowserTool-Instanz.
    """
    from typing import Any as _Any

    tool = BrowserTool(config=config)

    async def _ensure_initialized() -> str | None:
        """Stellt sicher, dass der Browser initialisiert ist."""
        if not tool._initialized:
            ok = await tool.initialize()
            if not ok:
                return t("browser.init_failed_playwright")
        return None

    async def _browse_url(url: str, extract_text: bool = True, **_: _Any) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.navigate(url, extract_text=extract_text)
        if not result.success:
            return t("browser.error_prefix", error=result.error)
        parts = [t("browser.page_title", title=result.title), t("browser.page_url", url=result.url)]
        if result.text:
            parts.append(f"\n{result.text}")
        return "\n".join(parts)

    async def _browse_screenshot(
        path: str | None = None, full_page: bool = False, **_: _Any
    ) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.screenshot(path=path, full_page=full_page)
        if not result.success:
            return t("browser.error_prefix", error=result.error)
        return t("browser.screenshot_saved", path=result.screenshot_path)

    async def _browse_click(selector: str, **_: _Any) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.click(selector)
        return result.text if result.success else t("browser.error_prefix", error=result.error)

    async def _browse_fill(selector: str, value: str, **_: _Any) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.fill(selector, value)
        return result.text if result.success else t("browser.error_prefix", error=result.error)

    async def _browse_execute_js(script: str, **_: _Any) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.execute_js(script)
        return result.text if result.success else t("browser.error_prefix", error=result.error)

    async def _browse_page_info(**_: _Any) -> str:
        if err := await _ensure_initialized():
            return err
        result = await tool.get_page_info()
        return result.text if result.success else t("browser.error_prefix", error=result.error)

    handlers = {
        "browse_url": _browse_url,
        "browse_screenshot": _browse_screenshot,
        "browse_click": _browse_click,
        "browse_fill": _browse_fill,
        "browse_execute_js": _browse_execute_js,
        "browse_page_info": _browse_page_info,
    }

    for name, schema in BROWSER_TOOL_SCHEMAS.items():
        mcp_client.register_builtin_handler(
            name,
            handlers[name],
            description=schema["description"],
            input_schema=schema["inputSchema"],
        )

    log.info("browser_tools_registered", tools=list(BROWSER_TOOL_SCHEMAS.keys()))
    return tool
