"""MCP-Tools fuer das Live Canvas.

Stellt MCP-kompatible Tools bereit, die der Agent nutzen kann um
HTML/CSS/JS-Inhalte in das Canvas-Panel des Clients zu pushen.

Tools:
  - canvas_push: HTML/CSS/JS ans Canvas pushen
  - canvas_reset: Canvas leeren
  - canvas_snapshot: Aktuellen Canvas-Inhalt lesen
  - canvas_eval: JavaScript im Canvas ausfuehren
"""

from __future__ import annotations

import logging
from typing import Any

from cognithor.i18n import t

logger = logging.getLogger(__name__)


class CanvasTools:
    """MCP-Tool-Definitionen fuer das Live Canvas.

    Wird vom MCP-Server registriert und stellt dem Agent
    Canvas-Operationen als Tools zur Verfuegung.
    """

    def __init__(self, canvas_manager: Any) -> None:
        """Initialisiert die Canvas-Tools.

        Args:
            canvas_manager: CanvasManager-Instanz fuer Canvas-Operationen.
        """
        self._canvas = canvas_manager

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Gibt die MCP-Tool-Definitionen zurueck."""
        return [
            {
                "name": "canvas_push",
                "description": (
                    "Pusht HTML/CSS/JS-Inhalt in das Canvas-Panel des Clients. "
                    "Kann für Visualisierungen, Dashboards, Formulare und "
                    "interaktive Inhalte verwendet werden. Der Inhalt wird in "
                    "einem sandboxed iframe dargestellt."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "html": {
                            "type": "string",
                            "description": "HTML/CSS/JS-Inhalt für das Canvas",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optionaler Titel für das Canvas-Panel",
                            "default": "",
                        },
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "canvas_reset",
                "description": "Leert das Canvas und entfernt allen Inhalt.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "canvas_snapshot",
                "description": (
                    "Liest den aktuellen HTML-Inhalt des Canvas. "
                    "Nützlich um den aktuellen Zustand zu inspizieren "
                    "bevor Änderungen vorgenommen werden."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "canvas_eval",
                "description": (
                    "Führt JavaScript-Code im Canvas-iframe aus. "
                    "Kann verwendet werden um bestehende Canvas-Inhalte "
                    "dynamisch zu aktualisieren ohne den gesamten HTML "
                    "neu zu pushen."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "js": {
                            "type": "string",
                            "description": "JavaScript-Code zur Ausführung im Canvas",
                        },
                    },
                    "required": ["js"],
                },
            },
        ]

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Verarbeitet einen MCP-Tool-Aufruf.

        Args:
            tool_name: Name des aufgerufenen Tools.
            arguments: Tool-Argumente.
            session_id: Aktive Session-ID.

        Returns:
            Tool-Ergebnis als Dict.
        """
        if tool_name == "canvas_push":
            return await self._handle_push(arguments, session_id)
        elif tool_name == "canvas_reset":
            return await self._handle_reset(session_id)
        elif tool_name == "canvas_snapshot":
            return await self._handle_snapshot(session_id)
        elif tool_name == "canvas_eval":
            return await self._handle_eval(arguments, session_id)
        else:
            return {"error": f"Unbekanntes Canvas-Tool: {tool_name}"}

    async def _handle_push(
        self,
        arguments: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Verarbeitet canvas_push."""
        html = arguments.get("html", "")
        title = arguments.get("title", "")

        if not html:
            return {"error": t("tools.canvas_no_html")}

        await self._canvas.push(session_id, html, title)
        return {
            "success": True,
            "message": f"Canvas aktualisiert (title={title!r}, {len(html)} Zeichen)",
        }

    async def _handle_reset(self, session_id: str) -> dict[str, Any]:
        """Verarbeitet canvas_reset."""
        await self._canvas.reset(session_id)
        return {"success": True, "message": "Canvas geleert"}

    async def _handle_snapshot(self, session_id: str) -> dict[str, Any]:
        """Verarbeitet canvas_snapshot."""
        html = await self._canvas.snapshot(session_id)
        if html:
            return {"success": True, "html": html, "length": len(html)}
        return {"success": True, "html": "", "message": "Canvas ist leer"}

    async def _handle_eval(
        self,
        arguments: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Verarbeitet canvas_eval."""
        js = arguments.get("js", "")
        if not js:
            return {"error": t("tools.canvas_no_js")}

        await self._canvas.eval_js(session_id, js)
        return {
            "success": True,
            "message": f"JavaScript ausgeführt ({len(js)} Zeichen)",
        }


def register_canvas_tools(mcp_client: Any, canvas_manager: Any) -> CanvasTools:
    """Registriert die 4 Canvas-Tools beim MCP-Client.

    Wird im Gateway-Init aufgerufen, sobald ein `CanvasManager` mit aktivem
    Broadcaster (typischerweise an `WebUIChannel.send_canvas_event` gebunden)
    bereitsteht. Die Tools `canvas_push`, `canvas_reset`, `canvas_snapshot`
    und `canvas_eval` werden dem Planner als reguläre MCP-Tools angeboten.

    Args:
        mcp_client: `JarvisMCPClient`-Instanz mit `register_builtin_handler`.
        canvas_manager: `CanvasManager`-Instanz mit verbundenem Broadcaster.

    Returns:
        Die `CanvasTools`-Instanz fuer optionalen direkten Zugriff (z.B.
        Tests). Im Normalbetrieb braucht der Aufrufer das nicht.
    """
    ct = CanvasTools(canvas_manager)

    for tool_def in ct.tool_definitions:
        name = tool_def["name"]

        async def _handler(
            session_id: str = "default",
            tool_name: str = name,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return await ct.handle_tool_call(tool_name, kwargs, session_id)

        mcp_client.register_builtin_handler(
            name,
            _handler,
            description=tool_def.get("description", ""),
            input_schema=tool_def.get("inputSchema"),
        )

    return ct
