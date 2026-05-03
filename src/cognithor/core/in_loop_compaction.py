"""In-loop compaction — LLM-based mid-session summarization.

Compresses the *middle* portion of a conversation history so that the
agent loop can stay within the model's context window while preserving
the most important context at both ends of the transcript.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

PROTECT_HEAD_MESSAGES = 4
PROTECT_TAIL_MESSAGES = 8
SUMMARY_LABEL = "[In-loop compaction summary]"


@dataclass
class InLoopCompactionResult:
    """Return value of :func:`compact_in_loop`."""

    history: list[dict[str, Any]]
    changed: bool
    compacted_messages: int
    summary_source: str  # "llm" | "heuristic" | "none"


async def compact_in_loop(
    history: list[dict[str, Any]],
    summarize_fn: Callable[[list[dict[str, Any]], int], Awaitable[str]] | None = None,
    context_window_tokens: int = 128_000,
) -> InLoopCompactionResult:
    """Compact the middle of *history* into a single summary message.

    Parameters
    ----------
    history:
        Full conversation history (list of ``{"role": ..., "content": ...}``).
    summarize_fn:
        Optional async callable ``(messages, max_tokens) -> str`` used to
        produce an LLM-based summary.  When *None* or when it raises, a
        deterministic heuristic fallback is used instead.
    context_window_tokens:
        Size of the model context window — used to cap summary length.
    """
    # --- determine compaction region ---
    leading_system = 0
    while leading_system < len(history) and history[leading_system].get("role") == "system":
        leading_system += 1

    body = history[leading_system:]
    if len(body) <= PROTECT_HEAD_MESSAGES + PROTECT_TAIL_MESSAGES + 1:
        return InLoopCompactionResult(history, False, 0, "none")

    head = history[:leading_system] + body[:PROTECT_HEAD_MESSAGES]
    middle = body[PROTECT_HEAD_MESSAGES:-PROTECT_TAIL_MESSAGES]
    tail = body[-PROTECT_TAIL_MESSAGES:]

    if not middle:
        return InLoopCompactionResult(history, False, 0, "none")

    # --- summarize ---
    max_summary_tokens = max(256, min(1024, int(context_window_tokens * 0.08)))
    summary_source = "llm"

    if summarize_fn is not None:
        try:
            summary = await summarize_fn(
                _build_summary_prompt(middle),
                max_summary_tokens,
            )
        except Exception:
            summary_source = "heuristic"
            summary = _build_heuristic_summary(middle)
    else:
        summary_source = "heuristic"
        summary = _build_heuristic_summary(middle)

    if not summary or not summary.strip():
        summary_source = "heuristic"
        summary = _build_heuristic_summary(middle)

    # --- cap summary length ---
    max_chars = max(1200, min(6000, int(context_window_tokens * 0.08)))
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n[summary truncated]"

    summary_msg = {"role": "assistant", "content": f"{SUMMARY_LABEL}\n{summary}"}

    return InLoopCompactionResult(
        history=[*head, summary_msg, *tail],
        changed=True,
        compacted_messages=len(middle),
        summary_source=summary_source,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_summary_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an LLM prompt that asks for a concise compaction summary."""
    transcript = "\n\n".join(
        f"---\nrole={m['role']}\n{str(m.get('content', ''))[:2000]}" for m in messages
    )
    return [
        {
            "role": "system",
            "content": (
                "You are compacting earlier turns from an active tool-using agent loop. "
                "Summarize so the agent can continue without losing state. "
                "Preserve: user goal, active plan, tool outputs that matter, file paths, "
                "commands, URLs, errors, decisions, unresolved follow-ups. "
                "Drop filler. Return plain markdown."
            ),
        },
        {
            "role": "user",
            "content": f"Compacted region:\n{transcript}\n\nWrite a concise summary.",
        },
    ]


def _build_heuristic_summary(messages: list[dict[str, Any]]) -> str:
    """Deterministic fallback when LLM summarization is unavailable."""
    roles = Counter(m.get("role", "unknown") for m in messages)
    role_str = ", ".join(f"{r}: {c}" for r, c in roles.items())
    highlights = "\n".join(
        f"- {m['role']}: {str(m.get('content', ''))[:280]}" for m in messages[-6:]
    )
    return (
        f"Compacted {len(messages)} messages to stay within context window.\n"
        f"Roles: {role_str}.\n\n"
        f"Recent highlights:\n{highlights}"
    )
