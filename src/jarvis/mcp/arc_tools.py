"""ARC-AGI-3 MCP Tools for Cognithor.

Exposes three MCP tools for controlling the ARC-AGI-3 benchmark agent:
  - arc_play    : Start or continue a game session
  - arc_status  : Query the current state of a running game session
  - arc_replay  : Replay a completed game session from recorded audit trail
"""

from __future__ import annotations

from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "register_arc_tools",
]

# ---------------------------------------------------------------------------
# In-memory session store (game_id -> result dict)
# ---------------------------------------------------------------------------
_active_sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


async def handle_arc_play(**kwargs: Any) -> str:
    """Start or resume an ARC-AGI-3 game session."""
    game_id: str = kwargs.get("game_id", "").strip()
    if not game_id:
        return "Error: 'game_id' is required."

    use_llm: bool = kwargs.get("use_llm", True)
    if isinstance(use_llm, str):
        use_llm = use_llm.lower() not in ("false", "0", "no")

    max_steps: int = int(kwargs.get("max_steps", 500))

    try:
        from jarvis.arc.agent import CognithorArcAgent
    except ImportError as exc:
        return f"Error: ARC-AGI-3 module not available ({exc}). Install jarvis[arc] dependencies."

    log.info("arc_play.start", game_id=game_id, use_llm=use_llm, max_steps=max_steps)

    import asyncio

    loop = asyncio.get_running_loop()

    def _run() -> dict[str, Any]:
        agent = CognithorArcAgent(
            game_id=game_id,
            use_llm_planner=use_llm,
            max_steps_per_level=max_steps,
        )
        return agent.run()

    try:
        result: dict[str, Any] = await loop.run_in_executor(None, _run)
    except Exception as exc:
        log.error("arc_play.failed", game_id=game_id, error=str(exc))
        return f"Error: Game run failed for '{game_id}': {exc}"

    _active_sessions[game_id] = result

    score = result.get("score", 0.0)
    levels = result.get("levels_completed", 0)
    steps = result.get("total_steps", 0)
    return f"Game '{game_id}' completed. Score: {score:.4f} | Levels: {levels} | Steps: {steps}"


async def handle_arc_status(**kwargs: Any) -> str:
    """Query the status of a completed or active ARC game session."""
    game_id: str = kwargs.get("game_id", "").strip()

    if game_id:
        if game_id not in _active_sessions:
            return f"No session found for game_id='{game_id}'. Use arc_play to start one."
        result = _active_sessions[game_id]
        score = result.get("score", 0.0)
        levels = result.get("levels_completed", 0)
        steps = result.get("total_steps", 0)
        resets = result.get("total_resets", 0)
        return (
            f"Session '{game_id}': score={score:.4f} levels={levels} steps={steps} resets={resets}"
        )

    # List all sessions
    if not _active_sessions:
        return "No active ARC sessions. Use arc_play to start a game."

    lines = ["Active ARC sessions:"]
    for gid, res in _active_sessions.items():
        score = res.get("score", 0.0)
        levels = res.get("levels_completed", 0)
        lines.append(f"  {gid}: score={score:.4f} levels={levels}")
    return "\n".join(lines)


async def handle_arc_replay(**kwargs: Any) -> str:
    """Replay a completed ARC game session from its audit trail."""
    game_id: str = kwargs.get("game_id", "").strip()
    if not game_id:
        return "Error: 'game_id' is required."

    verbose: bool = kwargs.get("verbose", False)
    if isinstance(verbose, str):
        verbose = verbose.lower() in ("true", "1", "yes")

    try:
        from jarvis.arc.audit import ArcAuditTrail
    except ImportError as exc:
        return f"Error: ARC audit module not available ({exc})."

    try:
        trail = ArcAuditTrail(game_id)
        events = trail.load_events()
    except Exception as exc:
        return f"Error: Could not load audit trail for '{game_id}': {exc}"

    if not events:
        return f"No audit trail found for game_id='{game_id}'."

    total = len(events)
    summary_lines = [
        f"Replay of '{game_id}': {total} recorded event(s).",
    ]

    if verbose:
        for i, evt in enumerate(events[:20]):  # cap at 20 for output safety
            summary_lines.append(f"  [{i + 1:>4}] {evt}")
        if total > 20:
            summary_lines.append(f"  ... ({total - 20} more events)")

    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def register_arc_tools(mcp_client: Any) -> None:
    """Register ARC-AGI-3 MCP tools with the handler registry.

    Args:
        mcp_client: JarvisMCPClient instance (provides register_builtin_handler).
    """
    # -- arc_play -----------------------------------------------------------
    mcp_client.register_builtin_handler(
        "arc_play",
        handle_arc_play,
        description=(
            "Start or run an ARC-AGI-3 game session. Provide a 'game_id' to identify "
            "the environment. Optionally set 'use_llm' (default true) and 'max_steps'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "description": "ARC-AGI-3 environment/game identifier",
                },
                "use_llm": {
                    "type": "boolean",
                    "description": "Enable LLM planner (default: true)",
                    "default": True,
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum steps per level (default: 500)",
                    "default": 500,
                },
            },
            "required": ["game_id"],
        },
    )

    # -- arc_status ---------------------------------------------------------
    mcp_client.register_builtin_handler(
        "arc_status",
        handle_arc_status,
        description=(
            "Query the status of an ARC-AGI-3 session. "
            "Provide 'game_id' for a specific session, or omit to list all sessions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "description": "Game/session ID to query (optional — omit to list all)",
                },
            },
        },
    )

    # -- arc_replay ---------------------------------------------------------
    mcp_client.register_builtin_handler(
        "arc_replay",
        handle_arc_replay,
        description=(
            "Replay a completed ARC-AGI-3 session from its recorded audit trail. "
            "Set 'verbose' to true to see individual event details."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "description": "Game ID whose audit trail should be replayed",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show individual events (default: false)",
                    "default": False,
                },
            },
            "required": ["game_id"],
        },
    )

    log.info(
        "arc_tools_registered",
        tools=["arc_play", "arc_status", "arc_replay"],
    )
