"""Browser MCP Tools -- Registriert Browser-Use als MCP-Tools (v17).

Stellt dem LLM folgende High-Level-Tools bereit:
  - browser_navigate:      Seite laden + analysieren
  - browser_click:         Element anklicken (CSS oder Beschreibung)
  - browser_fill:          Formularfeld ausfüllen
  - browser_fill_form:     Ganzes Formular ausfüllen
  - browser_screenshot:    Screenshot erstellen
  - browser_extract:       Text/Tabellen/Links extrahieren
  - browser_analyze:       Seitenstruktur analysieren
  - browser_execute_js:    JavaScript ausführen
  - browser_tab:           Tab-Management
  - browser_workflow:      Multi-Step-Workflow ausführen

Jedes Tool gibt ein strukturiertes Ergebnis zurück das
der LLM für Folge-Aktionen nutzen kann.
"""

from __future__ import annotations

import json
from typing import Any

from jarvis.browser.agent import BrowserAgent
from jarvis.browser.types import (
    ActionType,
    BrowserAction,
    BrowserConfig,
    BrowserWorkflow,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def register_browser_use_tools(
    mcp_client: Any,
    config: BrowserConfig | None = None,
    vision_analyzer: Any | None = None,
) -> BrowserAgent:
    """Registriert alle Browser-Use-Tools im MCP-Client.

    Args:
        mcp_client: MCP-Client für Tool-Registrierung.
        config: Optionale BrowserConfig.
        vision_analyzer: Optionaler VisionAnalyzer für Vision-Tools.

    Returns:
        BrowserAgent-Instanz (muss mit await agent.start() gestartet werden)
    """
    agent = BrowserAgent(config=config, vision_analyzer=vision_analyzer)

    # ── browser_navigate ─────────────────────────────────────────

    async def _navigate(params: dict[str, Any]) -> str:
        url = params.get("url", "")
        if not url:
            return json.dumps({"error": "URL required"})

        if not agent.is_running:
            started = await agent.start()
            if not started:
                return json.dumps({"error": "Browser not available. Install playwright."})

        state = await agent.navigate(url)
        return json.dumps(state.to_dict(), ensure_ascii=False)

    mcp_client.register_tool(
        name="browser_navigate",
        description="Navigiert zu einer URL und analysiert die Seite. "
                    "Gibt Seitenstruktur zurück: Titel, Links, Buttons, Inputs, Formulare.",
        parameters={"url": {"type": "string", "description": "Die zu ladende URL"}},
        handler=_navigate,
    )

    # ── browser_click ────────────────────────────────────────────

    async def _click(params: dict[str, Any]) -> str:
        selector = params.get("selector", "")
        description = params.get("description", "")

        if description and not selector:
            result = await agent.find_and_click(description)
        elif selector:
            result = await agent.click(selector)
        else:
            return json.dumps({"error": "selector or description required"})

        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_click",
        description="Klickt auf ein Element. Entweder per CSS-Selector oder Beschreibung.",
        parameters={
            "selector": {"type": "string", "description": "CSS-Selector (optional)"},
            "description": {"type": "string", "description": "Element-Beschreibung (z.B. 'Login-Button')"},
        },
        handler=_click,
    )

    # ── browser_fill ─────────────────────────────────────────────

    async def _fill(params: dict[str, Any]) -> str:
        selector = params.get("selector", "")
        value = params.get("value", "")
        if not selector:
            return json.dumps({"error": "selector required"})
        result = await agent.fill(selector, value)
        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_fill",
        description="Füllt ein Formularfeld aus.",
        parameters={
            "selector": {"type": "string", "description": "CSS-Selector des Feldes"},
            "value": {"type": "string", "description": "Einzugebender Wert"},
        },
        handler=_fill,
    )

    # ── browser_fill_form ────────────────────────────────────────

    async def _fill_form(params: dict[str, Any]) -> str:
        data = params.get("data", {})
        submit = params.get("submit", False)
        if not data:
            return json.dumps({"error": "data required (field_name: value)"})
        results = await agent.fill_form(data, submit=submit)
        return json.dumps({
            "filled": len(results),
            "results": [r.to_dict() for r in results],
        })

    mcp_client.register_tool(
        name="browser_fill_form",
        description="Füllt ein ganzes Formular mit mehreren Feldern aus.",
        parameters={
            "data": {"type": "object", "description": "Mapping Feldname → Wert"},
            "submit": {"type": "boolean", "description": "Formular absenden?"},
        },
        handler=_fill_form,
    )

    # ── browser_screenshot ───────────────────────────────────────

    async def _screenshot(params: dict[str, Any]) -> str:
        full = params.get("full_page", False)
        result = await agent.screenshot(full_page=full)
        if result.success:
            return json.dumps({
                "success": True,
                "screenshot_b64": result.screenshot_b64[:100] + "...",
                "full_data_length": len(result.screenshot_b64),
            })
        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_screenshot",
        description="Erstellt einen Screenshot der aktuellen Seite.",
        parameters={
            "full_page": {"type": "boolean", "description": "Ganze Seite oder nur Viewport?"},
        },
        handler=_screenshot,
    )

    # ── browser_extract ──────────────────────────────────────────

    async def _extract(params: dict[str, Any]) -> str:
        mode = params.get("mode", "text")
        selector = params.get("selector", "body")

        if mode == "text":
            text = await agent.extract_text(selector)
            return json.dumps({"text": text[:5000]})
        elif mode == "tables":
            tables = await agent.extract_tables()
            return json.dumps({"tables": tables})
        elif mode == "links":
            links = await agent.extract_links()
            return json.dumps({"links": links[:100]})
        else:
            return json.dumps({"error": f"Unknown mode: {mode}"})

    mcp_client.register_tool(
        name="browser_extract",
        description="Extrahiert Inhalte von der Seite: Text, Tabellen oder Links.",
        parameters={
            "mode": {"type": "string", "description": "text|tables|links"},
            "selector": {"type": "string", "description": "CSS-Selector (nur für text)"},
        },
        handler=_extract,
    )

    # ── browser_analyze ──────────────────────────────────────────

    async def _analyze(params: dict[str, Any]) -> str:
        state = await agent.analyze_page()
        return json.dumps({
            "summary": state.to_summary(max_text=2000),
            "stats": state.to_dict(),
        }, ensure_ascii=False)

    mcp_client.register_tool(
        name="browser_analyze",
        description="Analysiert die aktuelle Seite: Formulare, Buttons, Links, Inputs.",
        parameters={},
        handler=_analyze,
    )

    # ── browser_execute_js ───────────────────────────────────────

    _MAX_JS_LENGTH = 50_000
    _JS_BLOCKLIST = [
        "eval(", "Function(", "require(", "import(",
        "fetch(", "XMLHttpRequest", "child_process",
        "fs.", "process.exit",
    ]

    async def _execute_js(params: dict[str, Any]) -> str:
        script = params.get("script", "")
        if not script:
            return json.dumps({"error": "script required"})
        if len(script) > _MAX_JS_LENGTH:
            return json.dumps({"error": f"Script too long (max {_MAX_JS_LENGTH} chars)"})
        for blocked in _JS_BLOCKLIST:
            if blocked in script:
                return json.dumps({"error": f"Blocked API: '{blocked}' is not allowed"})
        result = await agent.execute_js(script)
        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_execute_js",
        description="Führt JavaScript auf der Seite aus.",
        parameters={"script": {"type": "string", "description": "JavaScript-Code"}},
        handler=_execute_js,
    )

    # ── browser_tab ──────────────────────────────────────────────

    async def _tab(params: dict[str, Any]) -> str:
        action = params.get("action", "list")
        if action == "new":
            result = await agent.new_tab(params.get("url", ""))
            return json.dumps(result.to_dict())
        elif action == "close":
            result = await agent.close_tab(params.get("index", -1))
            return json.dumps(result.to_dict())
        elif action == "switch":
            result = agent.switch_tab(params.get("index", 0))
            return json.dumps(result.to_dict())
        elif action == "list":
            return json.dumps({
                "tab_count": agent.page_count,
                "active_tab": agent._active_page_idx,
            })
        return json.dumps({"error": f"Unknown tab action: {action}"})

    mcp_client.register_tool(
        name="browser_tab",
        description="Tab-Management: new, close, switch, list.",
        parameters={
            "action": {"type": "string", "description": "new|close|switch|list"},
            "url": {"type": "string", "description": "URL für neuen Tab"},
            "index": {"type": "integer", "description": "Tab-Index"},
        },
        handler=_tab,
    )

    # ── browser_key ──────────────────────────────────────────────

    async def _key(params: dict[str, Any]) -> str:
        key = params.get("key", "Enter")
        result = await agent.press_key(key)
        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_key",
        description="Drückt eine Taste (Enter, Tab, Escape, ArrowDown, etc.).",
        parameters={"key": {"type": "string", "description": "Taste (z.B. Enter, Tab)"}},
        handler=_key,
    )

    # ── browser_vision_analyze ──────────────────────────────────

    async def _vision_analyze(params: dict[str, Any]) -> str:
        prompt = params.get("prompt", "")

        if not agent.is_running:
            started = await agent.start()
            if not started:
                return json.dumps({"error": "Browser not available."})

        if agent._vision is None or not getattr(agent._vision, "is_enabled", False):
            return json.dumps({"error": "Vision nicht aktiviert. Setze vision_model in der Config."})

        result = await agent.analyze_page_with_vision(prompt)
        return json.dumps(result, ensure_ascii=False)

    mcp_client.register_tool(
        name="browser_vision_analyze",
        description="Screenshot der Seite + KI-Vision-Analyse. "
                    "Kombiniert DOM-Analyse mit visueller Beschreibung.",
        parameters={
            "prompt": {"type": "string", "description": "Optionaler Analyse-Prompt"},
        },
        handler=_vision_analyze,
    )

    # ── browser_vision_find ──────────────────────────────────────

    async def _vision_find(params: dict[str, Any]) -> str:
        description = params.get("description", "")
        if not description:
            return json.dumps({"error": "description required"})

        if not agent.is_running:
            started = await agent.start()
            if not started:
                return json.dumps({"error": "Browser not available."})

        if agent._vision is None or not getattr(agent._vision, "is_enabled", False):
            return json.dumps({"error": "Vision nicht aktiviert. Setze vision_model in der Config."})

        result = await agent.find_and_click_with_vision(description)
        return json.dumps(result.to_dict())

    mcp_client.register_tool(
        name="browser_vision_find",
        description="Element per Beschreibung finden (Text-Match + Vision-Fallback). "
                    "Klickt das Element wenn gefunden.",
        parameters={
            "description": {"type": "string", "description": "Beschreibung des Elements (z.B. 'blauer Login-Button')"},
        },
        handler=_vision_find,
    )

    # ── browser_vision_screenshot ────────────────────────────────

    async def _vision_screenshot(params: dict[str, Any]) -> str:
        full = params.get("full_page", False)

        if not agent.is_running:
            started = await agent.start()
            if not started:
                return json.dumps({"error": "Browser not available."})

        result = await agent.screenshot(full_page=full)
        if not result.success:
            return json.dumps(result.to_dict())

        description = ""
        if agent._vision is not None and getattr(agent._vision, "is_enabled", False):
            try:
                page_content = await agent._extract_page_content()
                vision_result = await agent._vision.analyze_screenshot(
                    result.screenshot_b64, page_content=page_content
                )
                if vision_result.success:
                    description = vision_result.description
            except Exception:
                pass

        return json.dumps({
            "success": True,
            "description": description or "(Vision nicht verfügbar)",
            "screenshot_b64_length": len(result.screenshot_b64),
        }, ensure_ascii=False)

    mcp_client.register_tool(
        name="browser_vision_screenshot",
        description="Screenshot mit KI-Beschreibung statt rohem Base64. "
                    "Gibt eine textuelle Beschreibung des Screenshots zurück.",
        parameters={
            "full_page": {"type": "boolean", "description": "Ganze Seite oder nur Viewport?"},
        },
        handler=_vision_screenshot,
    )

    tool_count = 13 if vision_analyzer else 10
    log.info("browser_use_tools_registered", tool_count=tool_count)
    return agent
