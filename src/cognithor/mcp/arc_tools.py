"""ARC-AGI-3 MCP Tools for Cognithor.

Exposes three MCP tools for controlling the ARC-AGI-3 benchmark agent:
  - arc_play    : Start or continue a game session
  - arc_status  : Query the current state of a running game session
  - arc_replay  : Replay a completed game session from recorded audit trail
"""

from __future__ import annotations

from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "register_arc_tools",
]

# ---------------------------------------------------------------------------
# In-memory session store (game_id -> result dict)
# ---------------------------------------------------------------------------
_active_sessions: dict[str, dict[str, Any]] = {}


def _resolve_arc_choice_fn(
    *,
    build_vllm_choice_fn: Any,
    build_inprocess_vllm_choice_fn: Any,
) -> Any:
    """Sprint-21: pick the LLM choice-fn for the ARC agent.

    The PSE channel historically called the in-process vLLM factory
    directly, which only works on Linux + WSL2 with vLLM installed
    locally. Cognithor itself ships the central :class:`VLLMBackend`
    that talks to a vLLM HTTP endpoint over OpenAI-compatible REST —
    works cross-platform (Windows host can hit a WSL-running vLLM,
    or any remote vLLM the user configured).

    Resolution order:

    1. Central HTTP backend at the configured ``vllm_base_url`` if
       reachable. Wraps a fresh :class:`VLLMBackend` via
       ``build_vllm_choice_fn`` — keeps existing prompt/parsing
       logic, only swaps the transport.
    2. Linux-only in-process factory as fallback. Same call shape as
       before so an existing Linux/WSL setup keeps working.
    3. ``None`` → caller falls back to the heuristic DSL agent.
    """
    base_url: str | None = None
    try:
        from cognithor.config import load_config

        cfg = load_config()
        base_url = getattr(cfg, "vllm_base_url", None)
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("arc_play.config_unavailable", error=str(exc))

    if base_url:
        try:
            from cognithor.core.vllm_backend import VLLMBackend

            backend = VLLMBackend(base_url=base_url)
            log.info("arc_play.using_central_vllm_http", base_url=base_url)
            return build_vllm_choice_fn(backend=backend)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "arc_play.central_vllm_unavailable",
                base_url=base_url,
                error=str(exc),
            )

    try:
        return build_inprocess_vllm_choice_fn()
    except RuntimeError as exc:
        log.warning("arc_play.inprocess_vllm_unavailable", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


async def handle_arc_play(**kwargs: Any) -> str:
    """Start or resume an ARC-AGI-3 game session.

    Sprint-12: rewired to the new ``program_synthesis.arc_agi3`` stack
    via :class:`EpisodeRunner`. ``use_llm=True`` selects
    :class:`LLMReasoningAgent` (vLLM/qwen3.6:27b in-process); ``use_llm=False``
    selects :class:`Sprint10DSLAgent` (heuristic, no LLM dependency).
    """
    game_id: str = kwargs.get("game_id", "").strip()
    if not game_id:
        return "Error: 'game_id' is required."

    use_llm: bool = kwargs.get("use_llm", True)
    if isinstance(use_llm, str):
        use_llm = use_llm.lower() not in ("false", "0", "no")

    max_steps: int = int(kwargs.get("max_steps", 80))

    try:
        from cognithor.channels.program_synthesis.arc_agi3 import (
            ArcAuditTrail,
            EpisodeRunner,
            GameProfile,
            LLMReasoningAgent,
            Sprint10DSLAgent,
            build_inprocess_vllm_choice_fn,
            build_vllm_choice_fn,
        )
    except ImportError as exc:
        return f"Error: PSE arc_agi3 module not available ({exc})."

    log.info("arc_play.start", game_id=game_id, use_llm=use_llm, max_steps=max_steps)

    import asyncio

    loop = asyncio.get_running_loop()

    def _run() -> dict[str, Any]:
        # Cross-episode profile + per-run audit trail.
        profile = GameProfile.load(game_id) or GameProfile(
            game_id=game_id,
            game_type="mixed",
            available_actions=[],
            click_zones=[],
            target_colors=[],
            movement_effects={},
            win_condition="",
            vision_description="",
            vision_strategy="",
            strategy_metrics={},
        )
        trail = ArcAuditTrail(game_id=game_id)

        if use_llm:
            choice_fn = _resolve_arc_choice_fn(
                build_vllm_choice_fn=build_vllm_choice_fn,
                build_inprocess_vllm_choice_fn=build_inprocess_vllm_choice_fn,
            )
            use_llm_local = choice_fn is not None
        else:
            use_llm_local = False

        if use_llm_local:
            agent = LLMReasoningAgent(  # type: ignore[call-arg]
                choice_fn=choice_fn,
                audit_trail=trail,
                game_profile=profile,
                strategy_name="llm_reasoning",
                fast_path_enabled=True,
            )
        else:
            agent = Sprint10DSLAgent(
                audit_trail=trail,
                game_profile=profile,
                strategy_name="dsl_full",
                fast_path_enabled=True,
            )

        result = EpisodeRunner(agent=agent, game_id=game_id, max_steps=max_steps).run()
        # Persist the updated profile so the next run inherits the metrics.
        try:
            profile.save()
        except Exception as save_exc:  # pragma: no cover — defensive
            log.warning("arc_play.profile_save_failed", error=str(save_exc))

        return {
            "score": result.score,
            "levels_completed": result.levels_completed,
            "total_steps": result.total_steps,
            "final_state": result.final_state,
            "won": result.won,
            "error": result.error,
        }

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
    """Replay a completed ARC game session from its audit trail JSONL.

    Sprint-12: reads the JSONL exported by the new
    :class:`ArcAuditTrail.export_jsonl`. By default looks for
    ``~/.cognithor/arc/audits/<game_id>.jsonl``; override via
    ``audit_path``.
    """
    import json
    from pathlib import Path

    game_id: str = kwargs.get("game_id", "").strip()
    if not game_id:
        return "Error: 'game_id' is required."

    verbose: bool = kwargs.get("verbose", False)
    if isinstance(verbose, str):
        verbose = verbose.lower() in ("true", "1", "yes")

    audit_path_arg = kwargs.get("audit_path")
    if audit_path_arg:
        audit_path = Path(str(audit_path_arg))
    else:
        audit_path = Path.home() / ".cognithor" / "arc" / "audits" / f"{game_id}.jsonl"

    if not audit_path.exists():
        return (
            f"No audit trail found for game_id='{game_id}' at {audit_path}. "
            f"Pass 'audit_path' to override."
        )

    try:
        events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return f"Error: Could not parse audit trail for '{game_id}': {exc}"

    if not events:
        return f"Audit trail for '{game_id}' is empty."

    total = len(events)
    summary_lines = [
        f"Replay of '{game_id}': {total} recorded event(s) from {audit_path}.",
    ]

    if verbose:
        for i, evt in enumerate(events[:20]):
            summary_lines.append(
                f"  [{i + 1:>4}] {evt.get('event_type', '?')} "
                f"step={evt.get('step', '?')} action={evt.get('action', '-')}"
            )
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
