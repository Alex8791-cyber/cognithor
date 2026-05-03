"""3-Tier Context Guard — In-flight Context-Window Schutz.

Verhindert dass Tool-Ergebnisse das Context-Window sprengen:
  Tier 1: Einzelne zu grosse Tool-Results per Head/Tail truncaten
  Tier 2: Alte Tool-Results durch Placeholder ersetzen
  Tier 3: Overflow-Flag setzen → In-Loop Compaction triggern

Laeuft VOR jedem Model-Call im Agent-Loop.

Bibel-Referenz: Phase 3, Verbesserung 3 (HybridClaw).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextGuardConfig:
    enabled: bool = True
    per_result_share: float = 0.5  # Max 50% des Context-Windows pro Result
    compaction_ratio: float = 0.75  # Bei 75%: alte Results compacten
    overflow_ratio: float = 0.9  # Bei 90%: Overflow-Flag
    max_retries: int = 3


@dataclass
class ContextGuardResult:
    total_tokens_after: int = 0
    overflow_budget_tokens: int = 0
    truncated_tool_results: int = 0
    compacted_tool_results: int = 0
    tier3_triggered: bool = False


COMPACTED_PLACEHOLDER = "[Historical tool result compacted to preserve context budget.]"
TRUNCATION_MARKER = "\n\n...[tool result truncated by context guard]...\n\n"


def truncate_head_tail(
    text: str,
    max_chars: int,
    marker: str = TRUNCATION_MARKER,
    head_ratio: float = 0.7,
    tail_ratio: float = 0.2,
) -> str:
    """Behaelt 70% vom Anfang und 20% vom Ende, Marker in der Mitte."""
    if len(text) <= max_chars:
        return text
    available = max_chars - len(marker)
    if available <= 0:
        return text[:max_chars]
    head_chars = int(available * head_ratio)
    tail_chars = int(available * tail_ratio)
    if head_chars + tail_chars > available:
        tail_chars = max(0, available - head_chars)
    else:
        head_chars += available - (head_chars + tail_chars)
    if tail_chars <= 0:
        return text[:head_chars] + marker
    return text[:head_chars] + marker + text[-tail_chars:]


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Grobe Token-Schaetzung fuer eine Message-Liste."""
    total = 2  # Overhead
    for msg in messages:
        total += 4  # Message-Overhead
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if msg.get("role") == "tool":
            total += len(content) // 2  # Tool-Results: dichter
        else:
            total += len(content) // 4  # Normaler Text
    return total


def apply_context_guard(
    messages: list[dict[str, Any]],
    context_window_tokens: int = 128_000,
    config: ContextGuardConfig | None = None,
) -> ContextGuardResult:
    """Wendet den 3-Tier Context Guard auf eine Message-Liste an.

    Modifiziert messages IN-PLACE (truncation, compaction).

    Args:
        messages: Chat-History (wird in-place modifiziert).
        context_window_tokens: Groesse des Context-Windows in Tokens.
        config: Guard-Konfiguration.

    Returns:
        ContextGuardResult mit Metriken.
    """
    if config is None:
        config = ContextGuardConfig()

    if not config.enabled or not messages:
        return ContextGuardResult()

    per_result_limit = int(context_window_tokens * config.per_result_share)
    compaction_budget = int(context_window_tokens * config.compaction_ratio)
    overflow_budget = int(context_window_tokens * config.overflow_ratio)

    total_tokens = estimate_messages_tokens(messages)
    truncated = 0
    compacted = 0

    # TIER 1: Einzelne zu grosse Tool-Results truncaten
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        result_tokens = len(content) // 2
        if result_tokens <= per_result_limit:
            continue
        max_chars = per_result_limit * 2
        truncated_content = truncate_head_tail(content, max_chars)
        if truncated_content != content:
            msg["content"] = truncated_content
            truncated += 1

    total_tokens = estimate_messages_tokens(messages)

    # TIER 2: Alte Tool-Results durch Placeholder ersetzen
    if total_tokens > compaction_budget:
        compacted_ids: set[int] = set()
        for msg in messages:
            if total_tokens <= compaction_budget:
                break
            if msg.get("role") != "tool":
                continue
            if id(msg) in compacted_ids:
                continue
            old_tokens = len(msg.get("content", "")) // 2
            msg["content"] = COMPACTED_PLACEHOLDER
            new_tokens = len(COMPACTED_PLACEHOLDER) // 2
            total_tokens -= old_tokens - new_tokens
            compacted_ids.add(id(msg))
            compacted += 1

    total_tokens = estimate_messages_tokens(messages)

    return ContextGuardResult(
        total_tokens_after=total_tokens,
        overflow_budget_tokens=overflow_budget,
        truncated_tool_results=truncated,
        compacted_tool_results=compacted,
        tier3_triggered=total_tokens > overflow_budget,
    )
