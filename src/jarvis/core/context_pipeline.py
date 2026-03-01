"""Adaptive Context Pipeline — Automatische Kontext-Anreicherung.

Sammelt vor jedem Planner-Aufruf relevanten Kontext aus:
- Memory (BM25-only, sync, ~5-20ms)
- Vault (Volltextsuche, async, ~10-50ms)
- Episoden (letzte Tage, sync, ~1-5ms)

Das Ergebnis wird in WorkingMemory.injected_memories und
injected_procedures injiziert, sodass der Planner automatisch
über relevantes Wissen verfügt.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.config import ContextPipelineConfig

if TYPE_CHECKING:
    from jarvis.models import MemorySearchResult, WorkingMemory

logger = logging.getLogger("jarvis.core.context_pipeline")


@dataclass
class ContextResult:
    """Ergebnis der Kontext-Anreicherung."""

    memory_results: list[Any] = field(default_factory=list)  # MemorySearchResult
    vault_snippets: list[str] = field(default_factory=list)
    episode_snippets: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


class ContextPipeline:
    """Sammelt automatisch relevanten Kontext vor dem Planner-Aufruf.

    Dependency Injection: Memory und Vault werden nach der Initialisierung
    über set_memory_manager() / set_vault_tools() gesetzt (gleicher Pattern
    wie Synthesis).
    """

    def __init__(self, config: ContextPipelineConfig) -> None:
        self._config = config
        self._memory_manager: Any | None = None  # MemoryManager (sync BM25)
        self._vault_tools: Any | None = None  # VaultTools (async search)

    # ── Dependency Injection ──────────────────────────────────────

    def set_memory_manager(self, mm: Any) -> None:
        """Setzt den MemoryManager für BM25-Suche und Episoden."""
        self._memory_manager = mm

    def set_vault_tools(self, vt: Any) -> None:
        """Setzt die VaultTools für Volltextsuche."""
        self._vault_tools = vt

    # ── Hauptmethode ──────────────────────────────────────────────

    async def enrich(self, user_message: str, wm: WorkingMemory) -> ContextResult:
        """Sammelt relevanten Kontext und injiziert ihn in WorkingMemory.

        Args:
            user_message: Die aktuelle User-Nachricht.
            wm: Die aktive WorkingMemory-Instanz.

        Returns:
            ContextResult mit den gesammelten Daten und Metriken.
        """
        if not self._config.enabled:
            return ContextResult(skipped=True, skip_reason="disabled")

        t0 = time.perf_counter()

        # Smalltalk/Short-Message Check
        if self._is_smalltalk(user_message):
            return ContextResult(
                skipped=True,
                skip_reason="smalltalk",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        # Parallel sammeln
        memory_task = asyncio.get_event_loop().run_in_executor(
            None, self._search_memory, user_message,
        )
        vault_task = self._search_vault(user_message)
        episode_task = asyncio.get_event_loop().run_in_executor(
            None, self._get_episodes,
        )

        memory_results, vault_snippets, episode_snippets = await asyncio.gather(
            memory_task, vault_task, episode_task,
            return_exceptions=True,
        )

        # Exceptions graceful behandeln
        if isinstance(memory_results, BaseException):
            logger.debug("context_memory_gather_failed", exc_info=memory_results)
            memory_results = []
        if isinstance(vault_snippets, BaseException):
            logger.debug("context_vault_gather_failed", exc_info=vault_snippets)
            vault_snippets = []
        if isinstance(episode_snippets, BaseException):
            logger.debug("context_episode_gather_failed", exc_info=episode_snippets)
            episode_snippets = []

        # Injizieren: Memory-Ergebnisse → wm.injected_memories
        if memory_results:
            wm.injected_memories = list(memory_results)

        # Vault + Episoden → wm.injected_procedures (max 1 Slot)
        supplementary = self._format_supplementary_context(vault_snippets, episode_snippets)
        if supplementary and len(wm.injected_procedures) < 2:
            # Budget kürzen
            if len(supplementary) > self._config.max_context_chars:
                supplementary = supplementary[: self._config.max_context_chars] + "\n[...]"
            wm.injected_procedures.insert(0, supplementary)

        duration_ms = (time.perf_counter() - t0) * 1000

        return ContextResult(
            memory_results=list(memory_results) if memory_results else [],
            vault_snippets=list(vault_snippets) if vault_snippets else [],
            episode_snippets=list(episode_snippets) if episode_snippets else [],
            duration_ms=duration_ms,
        )

    # ── Hilfsmethoden ─────────────────────────────────────────────

    def _is_smalltalk(self, text: str) -> bool:
        """Prüft ob Nachricht Smalltalk ist (keine Suche nötig)."""
        normalized = text.strip().lower().rstrip("!?.,")
        if len(normalized) < self._config.min_query_length:
            return True
        return normalized in self._config.smalltalk_patterns

    def _search_memory(self, query: str) -> list[MemorySearchResult]:
        """BM25-only Suche via MemoryManager.search_memory_sync() — sync, ~5-20ms."""
        if not self._memory_manager:
            return []
        try:
            return self._memory_manager.search_memory_sync(
                query=query, top_k=self._config.memory_top_k,
            )
        except Exception:
            logger.debug("context_memory_search_failed", exc_info=True)
            return []

    async def _search_vault(self, query: str) -> list[str]:
        """Vault-Volltextsuche — async, ~10-50ms."""
        if not self._vault_tools:
            return []
        try:
            result = await self._vault_tools.vault_search(
                query=query, limit=self._config.vault_top_k,
            )
            # result ist ein String mit formatierten Treffern
            if result and "Keine Treffer" not in result:
                return [result]
            return []
        except Exception:
            logger.debug("context_vault_search_failed", exc_info=True)
            return []

    def _get_episodes(self) -> list[str]:
        """Letzte Episoden — sync, ~1-5ms."""
        if not self._memory_manager:
            return []
        try:
            episodic = getattr(self._memory_manager, "episodic", None)
            if episodic is None:
                return []
            recent = episodic.get_recent(days=self._config.episode_days)
            return [
                f"[{d.isoformat()}] {text[:500]}"
                for d, text in recent
                if text.strip()
            ]
        except Exception:
            logger.debug("context_episode_fetch_failed", exc_info=True)
            return []

    def _format_supplementary_context(
        self,
        vault_snippets: list[str],
        episode_snippets: list[str],
    ) -> str:
        """Formatiert Vault+Episoden als einen kompakten Kontext-String."""
        parts: list[str] = []
        if vault_snippets:
            parts.append("**Vault-Notizen:**\n" + "\n".join(vault_snippets[:3]))
        if episode_snippets:
            parts.append("**Letzte Aktivitäten:**\n" + "\n".join(episode_snippets[:3]))
        return "\n\n".join(parts)
