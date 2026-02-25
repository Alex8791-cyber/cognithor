"""Page Analyzer -- Intelligente Seiten-Analyse für Browser-Use v17.

Extrahiert aus einer geladenen Seite:
  - Interaktive Elemente (Buttons, Links, Inputs)
  - Formulare mit Feld-Erkennung
  - Tabellen als strukturierte Daten
  - Seitenstruktur und Navigation
  - Cookie-Banner-Erkennung

Arbeitet über JavaScript-Injection -- kein Parsing von HTML nötig.
Funktioniert mit jedem Playwright-Page-Objekt.
"""

from __future__ import annotations

import json
from typing import Any

from jarvis.browser.types import (
    ElementInfo,
    ElementType,
    FormField,
    FormInfo,
    PageState,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── JavaScript Snippets für Element-Extraktion ───────────────────

JS_EXTRACT_LINKS = """
() => {
    return Array.from(document.querySelectorAll('a[href]')).slice(0, 200).map((a, i) => ({
        selector: `a[href="${a.getAttribute('href')}"]`,
        text: (a.innerText || a.textContent || '').trim().substring(0, 200),
        href: a.href,
        visible: a.offsetParent !== null,
        ariaLabel: a.getAttribute('aria-label') || '',
    }));
}
"""

JS_EXTRACT_BUTTONS = """
() => {
    const btns = [
        ...document.querySelectorAll('button'),
        ...document.querySelectorAll('input[type="submit"]'),
        ...document.querySelectorAll('input[type="button"]'),
        ...document.querySelectorAll('[role="button"]'),
    ];
    return btns.slice(0, 100).map((b, i) => ({
        selector: b.id ? `#${b.id}` : `button:nth-of-type(${i+1})`,
        text: (b.innerText || b.value || b.textContent || '').trim().substring(0, 200),
        visible: b.offsetParent !== null,
        enabled: !b.disabled,
        ariaLabel: b.getAttribute('aria-label') || '',
        type: b.type || 'button',
    }));
}
"""

JS_EXTRACT_INPUTS = """
() => {
    const inputs = [
        ...document.querySelectorAll('input:not([type="hidden"]):not([type="submit"]):not([type="button"])'),
        ...document.querySelectorAll('textarea'),
        ...document.querySelectorAll('select'),
    ];
    return inputs.slice(0, 100).map((el, i) => {
        const tag = el.tagName.toLowerCase();
        const type = el.type || tag;
        let label = '';
        if (el.id) {
            const lbl = document.querySelector(`label[for="${el.id}"]`);
            if (lbl) label = lbl.textContent.trim();
        }
        if (!label && el.closest('label')) {
            label = el.closest('label').textContent.trim();
        }
        let options = [];
        if (tag === 'select') {
            options = Array.from(el.options).map(o => o.text || o.value);
        }
        return {
            selector: el.id ? `#${el.id}` : (el.name ? `[name="${el.name}"]` : `${tag}:nth-of-type(${i+1})`),
            name: el.name || '',
            type: type,
            value: el.value || '',
            placeholder: el.placeholder || '',
            label: label.substring(0, 200),
            required: el.required || false,
            visible: el.offsetParent !== null,
            enabled: !el.disabled,
            ariaLabel: el.getAttribute('aria-label') || '',
            options: options.slice(0, 50),
        };
    });
}
"""

JS_EXTRACT_FORMS = """
() => {
    return Array.from(document.querySelectorAll('form')).slice(0, 10).map((form, fi) => {
        const fields = Array.from(form.querySelectorAll(
            'input:not([type="hidden"]):not([type="submit"]), textarea, select'
        )).map((el, i) => {
            const tag = el.tagName.toLowerCase();
            let label = '';
            if (el.id) {
                const lbl = document.querySelector(`label[for="${el.id}"]`);
                if (lbl) label = lbl.textContent.trim();
            }
            let options = [];
            if (tag === 'select') {
                options = Array.from(el.options).map(o => o.text || o.value);
            }
            return {
                name: el.name || el.id || `field_${i}`,
                type: el.type || tag,
                label: label.substring(0, 200),
                value: el.value || '',
                placeholder: el.placeholder || '',
                required: el.required || false,
                options: options.slice(0, 30),
                selector: el.id ? `#${el.id}` : `[name="${el.name}"]`,
            };
        });
        const submit = form.querySelector('button[type="submit"], input[type="submit"]');
        return {
            action: form.action || '',
            method: (form.method || 'GET').toUpperCase(),
            name: form.name || form.id || `form_${fi}`,
            fields: fields,
            submitSelector: submit ? (submit.id ? `#${submit.id}` : 'button[type="submit"]') : '',
            selector: form.id ? `#${form.id}` : `form:nth-of-type(${fi+1})`,
        };
    });
}
"""

JS_EXTRACT_TABLES = """
() => {
    return Array.from(document.querySelectorAll('table')).slice(0, 5).map(table => {
        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
        const rows = Array.from(table.querySelectorAll('tbody tr, tr')).slice(0, 100).map(tr => {
            return Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim());
        }).filter(r => r.length > 0);
        return { headers, rows, rowCount: rows.length, colCount: headers.length || (rows[0] || []).length };
    });
}
"""

JS_DETECT_COOKIE_BANNER = """
() => {
    const selectors = [
        '[class*="cookie"]', '[id*="cookie"]',
        '[class*="consent"]', '[id*="consent"]',
        '[class*="gdpr"]', '[id*="gdpr"]',
        '[class*="privacy"]', '[id*="privacy-banner"]',
        '[class*="cc-"]', '.cc-banner',
    ];
    for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el && el.offsetParent !== null) {
            const accept = el.querySelector(
                'button[class*="accept"], button[class*="agree"], button[class*="ok"], ' +
                '[class*="accept"], [id*="accept"]'
            );
            return {
                found: true,
                selector: sel,
                acceptSelector: accept ? (accept.id ? `#${accept.id}` : `${sel} button`) : '',
                text: el.textContent.trim().substring(0, 300),
            };
        }
    }
    return { found: false };
}
"""


class PageAnalyzer:
    """Analysiert den aktuellen Zustand einer Browser-Seite."""

    def __init__(self) -> None:
        self._analysis_count = 0

    async def analyze(self, page: Any, *, extract_text: bool = True,
                      max_text_length: int = 5000) -> PageState:
        """Vollständige Seiten-Analyse."""
        import time as _time
        start = _time.monotonic()
        state = PageState()

        try:
            state.url = page.url
            state.title = await page.title()
            state.is_loaded = True

            # Text-Content
            if extract_text:
                try:
                    text = await page.evaluate("() => document.body?.innerText || ''")
                    state.text_content = (text or "")[:max_text_length]
                except Exception:
                    state.text_content = ""

            # HTML-Länge
            try:
                state.html_length = await page.evaluate("() => document.documentElement.outerHTML.length")
            except Exception:
                pass

            # Interaktive Elemente
            state.links = await self._extract_links(page)
            state.buttons = await self._extract_buttons(page)
            state.inputs = await self._extract_inputs(page)
            state.forms = await self._extract_forms(page)
            state.tables = await self._extract_tables(page)

        except Exception as exc:
            state.errors.append(f"Analysis error: {exc}")
            log.warning("page_analysis_error", url=state.url, error=str(exc))

        elapsed = int((_time.monotonic() - start) * 1000)
        state.load_time_ms = elapsed
        self._analysis_count += 1
        return state

    async def detect_cookie_banner(self, page: Any) -> dict[str, Any]:
        """Erkennt Cookie-Banner und gibt Accept-Selector zurück."""
        try:
            return await page.evaluate(JS_DETECT_COOKIE_BANNER)
        except Exception:
            return {"found": False}

    async def find_element(self, page: Any, description: str) -> ElementInfo | None:
        """Findet ein Element anhand einer natürlichsprachlichen Beschreibung.

        Sucht nach: Text-Match, aria-label, placeholder, name-Attribut.
        """
        desc_lower = description.lower().strip()

        # 1. Buttons
        buttons = await self._extract_buttons(page)
        for btn in buttons:
            if self._fuzzy_match(desc_lower, btn.text, btn.aria_label, btn.name):
                return btn

        # 2. Links
        links = await self._extract_links(page)
        for link in links:
            if self._fuzzy_match(desc_lower, link.text, link.aria_label, link.href):
                return link

        # 3. Inputs
        inputs = await self._extract_inputs(page)
        for inp in inputs:
            if self._fuzzy_match(desc_lower, inp.name, inp.placeholder, inp.aria_label):
                return inp

        return None

    def _fuzzy_match(self, query: str, *candidates: str) -> bool:
        """Einfacher Fuzzy-Match: Query muss in einem Kandidaten enthalten sein."""
        for c in candidates:
            if c and query in c.lower():
                return True
            if c and c.lower() in query:
                return True
        return False

    # ── Element Extraction ───────────────────────────────────────

    async def _extract_links(self, page: Any) -> list[ElementInfo]:
        try:
            raw = await page.evaluate(JS_EXTRACT_LINKS)
            return [
                ElementInfo(
                    selector=r.get("selector", ""), element_type=ElementType.LINK,
                    text=r.get("text", ""), href=r.get("href", ""),
                    is_visible=r.get("visible", True),
                    aria_label=r.get("ariaLabel", ""),
                )
                for r in (raw or [])
            ]
        except Exception:
            return []

    async def _extract_buttons(self, page: Any) -> list[ElementInfo]:
        try:
            raw = await page.evaluate(JS_EXTRACT_BUTTONS)
            return [
                ElementInfo(
                    selector=r.get("selector", ""), element_type=ElementType.BUTTON,
                    text=r.get("text", ""), is_visible=r.get("visible", True),
                    is_enabled=r.get("enabled", True),
                    aria_label=r.get("ariaLabel", ""),
                )
                for r in (raw or [])
            ]
        except Exception:
            return []

    async def _extract_inputs(self, page: Any) -> list[ElementInfo]:
        try:
            raw = await page.evaluate(JS_EXTRACT_INPUTS)
            results: list[ElementInfo] = []
            for r in (raw or []):
                rtype = r.get("type", "text")
                if rtype == "checkbox":
                    etype = ElementType.CHECKBOX
                elif rtype == "radio":
                    etype = ElementType.RADIO
                elif rtype == "file":
                    etype = ElementType.FILE_INPUT
                elif rtype in ("select", "select-one", "select-multiple"):
                    etype = ElementType.SELECT
                elif rtype == "textarea":
                    etype = ElementType.TEXTAREA
                else:
                    etype = ElementType.INPUT

                results.append(ElementInfo(
                    selector=r.get("selector", ""), element_type=etype,
                    name=r.get("name", ""), value=r.get("value", ""),
                    placeholder=r.get("placeholder", ""),
                    aria_label=r.get("ariaLabel", ""),
                    is_visible=r.get("visible", True),
                    is_enabled=r.get("enabled", True),
                    is_required=r.get("required", False),
                ))
            return results
        except Exception:
            return []

    async def _extract_forms(self, page: Any) -> list[FormInfo]:
        try:
            raw = await page.evaluate(JS_EXTRACT_FORMS)
            forms: list[FormInfo] = []
            for r in (raw or []):
                fields = [
                    FormField(
                        name=f.get("name", ""), field_type=f.get("type", "text"),
                        label=f.get("label", ""), value=f.get("value", ""),
                        placeholder=f.get("placeholder", ""),
                        required=f.get("required", False),
                        options=f.get("options", []),
                        selector=f.get("selector", ""),
                    )
                    for f in r.get("fields", [])
                ]
                forms.append(FormInfo(
                    action=r.get("action", ""), method=r.get("method", "GET"),
                    fields=fields, submit_selector=r.get("submitSelector", ""),
                    selector=r.get("selector", ""), name=r.get("name", ""),
                ))
            return forms
        except Exception:
            return []

    async def _extract_tables(self, page: Any) -> list[dict[str, Any]]:
        try:
            return await page.evaluate(JS_EXTRACT_TABLES) or []
        except Exception:
            return []

    def stats(self) -> dict[str, Any]:
        return {"analysis_count": self._analysis_count}
