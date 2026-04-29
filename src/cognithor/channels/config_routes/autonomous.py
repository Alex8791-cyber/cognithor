"""Cognithor · Autonomous + Feedback routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Bundle aus
zwei kleineren Helfern:

  - `_register_autonomous_routes()` — Autonomous-Task-Orchestration.
  - `_register_feedback_routes()` — Thumbs-up/down + Chat-Tree-Vote
    Feedback-Endpoints.
"""

from __future__ import annotations

from typing import Any

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ======================================================================
# Autonomous Task Orchestration routes
# ======================================================================


def _register_autonomous_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Endpoints for querying autonomous task execution status."""

    @app.get("/api/v1/autonomous/tasks", dependencies=deps)
    async def list_autonomous_tasks() -> dict[str, Any]:
        """List active autonomous tasks."""
        if not hasattr(gateway, "_autonomous_orchestrator"):
            return {"tasks": []}
        return {"tasks": gateway._autonomous_orchestrator.get_active_tasks()}


# ======================================================================
# Feedback routes (thumbs up/down)
# ======================================================================


def _register_feedback_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for user feedback (thumbs up/down)."""

    @app.post("/api/v1/feedback", dependencies=deps)
    async def submit_feedback(request: Request) -> dict[str, Any]:
        """Submit thumbs up/down feedback for a message."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        rating = body.get("rating", 0)
        if rating not in (1, -1):
            return {"error": "rating must be 1 (thumbs up) or -1 (thumbs down)"}

        feedback_id = feedback_store.submit(
            session_id=body.get("session_id", ""),
            message_id=body.get("message_id", ""),
            rating=rating,
            comment=body.get("comment", ""),
            agent_name=body.get("agent_name", "jarvis"),
            channel=body.get("channel", "webui"),
            user_message=body.get("user_message", ""),
            assistant_response=body.get("assistant_response", ""),
            tool_calls=body.get("tool_calls", ""),
        )
        return {"status": "ok", "feedback_id": feedback_id}

    @app.patch("/api/v1/feedback/{feedback_id}", dependencies=deps)
    async def update_feedback_comment(feedback_id: str, request: Request) -> dict[str, Any]:
        """Add comment to existing feedback (after follow-up question)."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        ok = feedback_store.add_comment(feedback_id, body.get("comment", ""))
        return {"status": "ok" if ok else "not_found"}

    @app.get("/api/v1/feedback/stats", dependencies=deps)
    async def feedback_stats(agent_name: str = "", hours: int = 0) -> dict[str, Any]:
        """Get feedback statistics."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0}
        return feedback_store.get_stats(agent_name=agent_name, hours=hours)

    @app.get("/api/v1/feedback/recent", dependencies=deps)
    async def recent_feedback(limit: int = 50) -> dict[str, Any]:
        """Get recent feedback entries."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"entries": []}
        return {"entries": feedback_store.get_recent(limit=limit)}

    # ── Chat Tree / Branching ────────────────────────────────────────

    @app.get("/api/v1/chat/tree/latest", dependencies=deps)
    async def get_latest_chat_tree(session_id: str = "") -> dict[str, Any]:
        """Get the most recent conversation tree.

        If session_id is provided, look up the session's persisted
        conversation_id first so the correct tree is returned.
        """
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"nodes": [], "conversation_id": None}

        conv_id = None

        # Try session-specific conversation first
        if session_id:
            store = getattr(gateway, "_session_store", None)
            if store:
                session = store.load_session_by_id(session_id)
                if session and getattr(session, "conversation_id", ""):
                    conv_id = session.conversation_id

        # Fallback: most recent conversation
        if not conv_id:
            with tree._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                if row:
                    conv_id = row["id"]

        if not conv_id:
            return {"nodes": [], "conversation_id": None}
        return tree.get_tree_structure(conv_id)

    @app.get("/api/v1/chat/tree/{conversation_id}", dependencies=deps)
    async def get_chat_tree(conversation_id: str) -> dict[str, Any]:
        """Get full conversation tree structure."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        return tree.get_tree_structure(conversation_id)

    @app.get("/api/v1/chat/path/{conversation_id}/{leaf_id}", dependencies=deps)
    async def get_chat_path(conversation_id: str, leaf_id: str) -> dict[str, Any]:
        """Get path from root to a specific leaf."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        path = tree.get_path_to_root(leaf_id)
        return {"path": path, "count": len(path)}

    @app.post("/api/v1/chat/branch", dependencies=deps)
    async def create_chat_branch(request: Request) -> dict[str, Any]:
        """Create a branch at a specific node."""
        body = await request.json()
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        conv_id = body.get("conversation_id", "")
        parent_id = body.get("parent_id", "")
        text = body.get("text", "")
        role = body.get("role", "user")
        if not conv_id or not text:
            return {"error": "conversation_id and text required"}
        node_id = tree.add_node(conv_id, role=role, text=text, parent_id=parent_id or None)
        return {
            "node_id": node_id,
            "branch_index": tree.get_branch_index(node_id),
        }
