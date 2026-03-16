"""MCP Identity Tools — Cognithor's cognitive identity interface.

Provides 4 tools for the Planner to interact with the Identity Layer:
- identity_recall: Search long-term cognitive memory
- identity_state: Get current cognitive state
- identity_reflect: Trigger self-reflection
- identity_dream: Trigger dream cycle (memory consolidation)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig
    from jarvis.identity import IdentityLayer
    from jarvis.mcp.client import JarvisMCPClient

log = logging.getLogger("jarvis.mcp.identity_tools")


def register_identity_tools(
    mcp_client: JarvisMCPClient,
    identity_layer: IdentityLayer,
    config: JarvisConfig | None = None,
) -> None:
    """Register identity MCP tools.

    Args:
        mcp_client: MCP client for tool registration.
        identity_layer: The IdentityLayer instance.
        config: Optional config.
    """

    async def _handle_identity_recall(**kwargs: Any) -> str:
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 5)
        if not query:
            return "Keine Suchanfrage angegeben."
        results = identity_layer.recall_for_cognithor(query, top_k=top_k)
        if not results:
            return "Keine Erinnerungen zu diesem Thema gefunden."
        parts = []
        for i, mem in enumerate(results, 1):
            parts.append(f"{i}. [{mem['type']}] (Score: {mem['score']:.2f}) {mem['content'][:200]}")
        return "\n".join(parts)

    async def _handle_identity_state(**kwargs: Any) -> str:
        state = identity_layer.get_state_summary()
        if not state.get("available"):
            return "Identity Layer nicht verfuegbar."
        import json

        return json.dumps(state, indent=2, default=str, ensure_ascii=False)

    async def _handle_identity_reflect(**kwargs: Any) -> str:
        topic = kwargs.get("topic", "general reflection")
        identity_layer.reflect(
            session_summary=f"Self-initiated reflection on: {topic}",
            success_score=0.7,
        )
        return f"Reflexion zu '{topic}' durchgefuehrt. Erinnerungen konsolidiert."

    async def _handle_identity_dream(**kwargs: Any) -> str:
        if not identity_layer.available:
            return "Identity Layer nicht verfuegbar."
        try:
            stats = identity_layer._engine.dream.run(identity_layer._engine)
            return f"Traumzyklus abgeschlossen: {stats}"
        except Exception as exc:
            return f"Traumzyklus fehlgeschlagen: {exc}"

    # Register tools
    mcp_client.register_builtin_handler(
        tool_name="identity_recall",
        handler=_handle_identity_recall,
        description=(
            "Erinnere dich an etwas aus deinem kognitiven Langzeitgedaechtnis. "
            "Nutze dies wenn der User nach vergangenen Gespraechen oder deiner "
            "Meinung fragt."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Wonach suchst du in deiner Erinnerung?",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Anzahl der Erinnerungen (1-20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        tool_name="identity_state",
        handler=_handle_identity_state,
        description=(
            "Pruefe deinen aktuellen kognitiven Zustand: Energie, Emotionen, "
            "Zeitwahrnehmung, Unsicherheiten."
        ),
        input_schema={"type": "object", "properties": {}},
    )

    mcp_client.register_builtin_handler(
        tool_name="identity_reflect",
        handler=_handle_identity_reflect,
        description=(
            "Loese eine Selbstreflexion aus. Nutze dies nach wichtigen "
            "Gespraechen oder Entscheidungen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Worueber reflektierst du?",
                },
            },
            "required": ["topic"],
        },
    )

    mcp_client.register_builtin_handler(
        tool_name="identity_dream",
        handler=_handle_identity_dream,
        description=(
            "Loese einen Traumzyklus aus — konsolidiert Erinnerungen, "
            "findet verborgene Verbindungen, reguliert Emotionen."
        ),
        input_schema={"type": "object", "properties": {}},
    )

    log.info("identity_tools_registered tools=%d", 4)
