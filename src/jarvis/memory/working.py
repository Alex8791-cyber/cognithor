"""Working Memory · Tier 5 — Session-Kontext & Context-Budget. [B§4.6]

RAM-only (außer Pre-Compaction Flush).
Verwaltet das Token-Budget und entscheidet wann komprimiert wird.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jarvis.config import MemoryConfig
from jarvis.models import (
    ActionPlan,
    MemorySearchResult,
    Message,
    ToolResult,
    WorkingMemory,
)

logger = logging.getLogger("jarvis.memory.working")

# Token-Budget Verteilung [B§4.6]
BUDGET_CORE_MEMORY = 500
BUDGET_SYSTEM_PROMPT = 800
BUDGET_PROCEDURES = 600
BUDGET_INJECTED_MEMORIES = 1500
BUDGET_TOOL_DESCRIPTIONS = 1200
BUDGET_RESPONSE_RESERVE = 3000

# Statische Budgets (immer reserviert)
STATIC_BUDGET = (
    BUDGET_CORE_MEMORY
    + BUDGET_SYSTEM_PROMPT
    + BUDGET_PROCEDURES
    + BUDGET_INJECTED_MEMORIES
    + BUDGET_TOOL_DESCRIPTIONS
    + BUDGET_RESPONSE_RESERVE
)


@dataclass
class CompactionResult:
    """Ergebnis eines Pre-Compaction Flush."""

    messages_removed: int = 0
    tokens_freed: int = 0
    facts_extracted: list[str] = field(default_factory=list)
    summary: str = ""


class WorkingMemoryManager:
    """Verwaltet den aktiven Session-Kontext.

    Responsibilities:
    - Token-Budget Tracking
    - Entscheidung wann Compaction nötig ist
    - FIFO-Entfernung alter Chat-History
    - Injection von relevanten Memories
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        max_tokens: int = 32768,
    ) -> None:
        """Initialisiert den WorkingMemoryManager mit Konfiguration und Token-Budget."""
        self._config = config or MemoryConfig()
        self._max_tokens = max_tokens
        self._memory = WorkingMemory(max_tokens=max_tokens)

    @property
    def memory(self) -> WorkingMemory:
        """Zugriff auf das aktuelle WorkingMemory-Objekt."""
        return self._memory

    @property
    def max_tokens(self) -> int:
        """Maximales Token-Budget für die Working Memory."""
        return self._max_tokens

    @property
    def available_chat_tokens(self) -> int:
        """Verfügbare Tokens für Chat-History."""
        return max(0, self._max_tokens - STATIC_BUDGET)

    @property
    def current_chat_tokens(self) -> int:
        """Geschätzte Tokens in der aktuellen Chat-History."""
        return sum(self._estimate_message_tokens(m) for m in self._memory.chat_history)

    @property
    def usage_ratio(self) -> float:
        """Auslastung des Chat-Budgets (0-1)."""
        avail = self.available_chat_tokens
        if avail <= 0:
            return 1.0
        return min(1.0, self.current_chat_tokens / avail)

    @property
    def needs_compaction(self) -> bool:
        """True wenn Pre-Compaction Flush nötig ist."""
        return self.usage_ratio > self._config.compaction_threshold

    # ── Session Management ───────────────────────────────────────

    def new_session(self, session_id: str = "") -> WorkingMemory:
        """Startet eine neue Session."""
        self._memory = WorkingMemory(
            max_tokens=self._max_tokens,
        )
        if session_id:
            self._memory.session_id = session_id
        return self._memory

    def add_message(self, message: Message) -> None:
        """Fügt eine Nachricht zur Chat-History hinzu."""
        self._memory.add_message(message)
        self._update_token_count()

    def add_tool_result(self, result: ToolResult) -> None:
        """Fügt ein Tool-Ergebnis hinzu."""
        self._memory.add_tool_result(result)

    def set_plan(self, plan: ActionPlan | None) -> None:
        """Setzt den aktiven Plan."""
        self._memory.active_plan = plan

    def inject_memories(self, results: list[MemorySearchResult]) -> None:
        """Injiziert Memory-Suchergebnisse in den Kontext."""
        self._memory.injected_memories = results

    def inject_procedures(self, procedure_texts: list[str]) -> None:
        """Injiziert relevante Prozedur-Texte."""
        self._memory.injected_procedures = procedure_texts[:2]  # Max 2 [B§4.6]

    def set_core_memory(self, text: str) -> None:
        """Setzt den Core-Memory Text."""
        self._memory.core_memory_text = text

    # ── Compaction ───────────────────────────────────────────────

    def compact(self) -> CompactionResult:
        """Pre-Compaction: Entfernt älteste Chat-History Einträge.

        Behält die letzten N Nachrichten (aus Config).
        Extrahiert keine Fakten — das macht der Reflector separat.

        Returns:
            CompactionResult mit Infos über was entfernt wurde.
        """
        keep_n = self._config.compaction_keep_last_n
        history = self._memory.chat_history

        if len(history) <= keep_n:
            return CompactionResult()

        # Zu entfernende Nachrichten
        to_remove = history[:-keep_n]
        tokens_before = self.current_chat_tokens

        # Entfernen
        self._memory.chat_history = history[-keep_n:]
        self._update_token_count()

        tokens_after = self.current_chat_tokens
        freed = tokens_before - tokens_after

        result = CompactionResult(
            messages_removed=len(to_remove),
            tokens_freed=freed,
        )

        logger.info(
            "Compaction: %d Nachrichten entfernt, ~%d Tokens befreit (%.1f%% → %.1f%%)",
            result.messages_removed,
            result.tokens_freed,
            (tokens_before / self.available_chat_tokens * 100)
            if self.available_chat_tokens > 0
            else 0,
            self.usage_ratio * 100,
        )

        return result

    def get_removable_messages(self) -> list[Message]:
        """Gibt die Nachrichten zurück die bei Compaction entfernt werden.

        Nützlich für den Reflector, der vorher Fakten extrahieren kann.
        """
        keep_n = self._config.compaction_keep_last_n
        history = self._memory.chat_history
        if len(history) <= keep_n:
            return []
        return history[:-keep_n]

    # ── Context Building ─────────────────────────────────────────

    def build_context_parts(self) -> dict[str, str]:
        """Baut die einzelnen Kontext-Teile für den LLM-Prompt.

        Returns:
            Dict mit benannten Kontext-Teilen.
        """
        parts: dict[str, str] = {}

        if self._memory.core_memory_text:
            parts["core_memory"] = self._memory.core_memory_text

        if self._memory.injected_procedures:
            parts["procedures"] = "\n\n---\n\n".join(self._memory.injected_procedures)

        if self._memory.injected_memories:
            memory_texts = []
            for mr in self._memory.injected_memories:
                source = (
                    mr.chunk.source_path.split("/")[-1]
                    if "/" in mr.chunk.source_path
                    else mr.chunk.source_path
                )
                memory_texts.append(f"[{source} | Score: {mr.score:.2f}]\n{mr.chunk.text}")
            parts["memories"] = "\n\n---\n\n".join(memory_texts)

        return parts

    def build_budget_report(self) -> str:
        """Erstellt einen lesbaren Budget-Bericht."""
        chat_tokens = self.current_chat_tokens
        avail = self.available_chat_tokens

        lines = [
            f"Token-Budget ({self._max_tokens} gesamt):",
            f"  Core Memory:       ~{BUDGET_CORE_MEMORY}",
            f"  System-Prompt:     ~{BUDGET_SYSTEM_PROMPT}",
            f"  Procedures:        ~{BUDGET_PROCEDURES}",
            f"  Injected Memories: ~{BUDGET_INJECTED_MEMORIES}",
            f"  Tool-Beschreibungen: ~{BUDGET_TOOL_DESCRIPTIONS}",
            f"  Antwort-Reserve:   ~{BUDGET_RESPONSE_RESERVE}",
            "  ──────────────────────",
            f"  Statisch:          ~{STATIC_BUDGET}",
            f"  Chat verfügbar:    ~{avail}",
            f"  Chat genutzt:      ~{chat_tokens} ({self.usage_ratio:.0%})",
            f"  Nachrichten:       {len(self._memory.chat_history)}",
            f"  Compaction nötig:  {'JA' if self.needs_compaction else 'Nein'}",
        ]
        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_message_tokens(msg: Message) -> int:
        """Schätzt Token-Anzahl einer Nachricht."""
        text = msg.content or ""
        # ~4 Zeichen pro Token + Overhead für Rolle etc.
        return len(text) // 4 + 10

    def _update_token_count(self) -> None:
        """Aktualisiert den Token-Counter."""
        chat_tokens = self.current_chat_tokens
        self._memory.token_count = STATIC_BUDGET + chat_tokens
