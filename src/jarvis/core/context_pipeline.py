"""Adaptive Context Pipeline — Automatische Kontext-Anreicherung.

Sammelt vor jedem Planner-Aufruf relevanten Kontext aus:
- Wave 1 (parallel): Memory (BM25-only), Vault (Volltextsuche), Episoden
- Checkpoint: merge and deduplicate
- Wave 2 (parallel): Skill injection, User preference lookup

Das Ergebnis wird in WorkingMemory.injected_memories und
injected_procedures injiziert, sodass der Planner automatisch
über relevantes Wissen verfügt.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import ContextPipelineConfig
    from jarvis.models import MemorySearchResult, WorkingMemory

log = get_logger(__name__)


@dataclass
class ContextResult:
    """Ergebnis der Kontext-Anreicherung."""

    memory_results: list[Any] = field(default_factory=list)  # MemorySearchResult
    vault_snippets: list[str] = field(default_factory=list)
    episode_snippets: list[str] = field(default_factory=list)
    skill_context: str = ""
    user_pref_hint: str = ""
    duration_ms: float = 0.0
    wave1_ms: float = 0.0
    wave2_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


class ContextPipeline:
    """Sammelt automatisch relevanten Kontext vor dem Planner-Aufruf.

    Two-wave parallel execution:
      Wave 1: memory search, vault search, episode retrieval (parallel)
      Checkpoint: merge and deduplicate results
      Wave 2: skill injection, user preference lookup (parallel)

    Dependency Injection: Memory und Vault werden nach der Initialisierung
    ueber set_memory_manager() / set_vault_tools() gesetzt (gleicher Pattern
    wie Synthesis).
    """

    def __init__(self, config: ContextPipelineConfig) -> None:
        self._config = config
        self._memory_manager: Any | None = None  # MemoryManager (sync BM25)
        self._vault_tools: Any | None = None  # VaultTools (async search)
        self._skill_registry: Any | None = None  # SkillRegistry
        self._user_pref_store: Any | None = None  # UserPreferenceStore

    # ── Dependency Injection ──────────────────────────────────────

    def set_memory_manager(self, mm: Any) -> None:
        """Setzt den MemoryManager fuer BM25-Suche und Episoden."""
        self._memory_manager = mm

    def set_vault_tools(self, vt: Any) -> None:
        """Setzt die VaultTools fuer Volltextsuche."""
        self._vault_tools = vt

    def set_skill_registry(self, sr: Any) -> None:
        """Setzt die SkillRegistry fuer Skill-Kontext-Injection."""
        self._skill_registry = sr

    def set_user_pref_store(self, ups: Any) -> None:
        """Setzt den UserPreferenceStore fuer Preference-Lookup."""
        self._user_pref_store = ups

    # ── Hauptmethode ──────────────────────────────────────────────

    async def enrich(
        self,
        user_message: str,
        wm: WorkingMemory,
        *,
        user_id: str = "",
    ) -> ContextResult:
        """Sammelt relevanten Kontext und injiziert ihn in WorkingMemory.

        Uses two-wave parallel execution:
          Wave 1: memory, vault, episodes (parallel via asyncio.gather)
          Checkpoint: merge and deduplicate
          Wave 2: skill injection, user preferences (parallel)

        Args:
            user_message: Die aktuelle User-Nachricht.
            wm: Die aktive WorkingMemory-Instanz.
            user_id: Optional user ID for preference lookup.

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

        # ── Wave 1: Memory, Vault, Episodes (parallel) ──────────
        w1_start = time.perf_counter()

        memory_task = asyncio.get_running_loop().run_in_executor(
            None,
            self._search_memory,
            user_message,
        )
        vault_task = self._search_vault(user_message)
        episode_task = asyncio.get_running_loop().run_in_executor(
            None,
            self._get_episodes,
        )

        memory_results, vault_snippets, episode_snippets = await asyncio.gather(
            memory_task,
            vault_task,
            episode_task,
            return_exceptions=True,
        )

        # Exceptions graceful behandeln
        if isinstance(memory_results, BaseException):
            log.debug("context_memory_gather_failed", exc_info=memory_results)
            memory_results = []
        if isinstance(vault_snippets, BaseException):
            log.debug("context_vault_gather_failed", exc_info=vault_snippets)
            vault_snippets = []
        if isinstance(episode_snippets, BaseException):
            log.debug("context_episode_gather_failed", exc_info=episode_snippets)
            episode_snippets = []

        wave1_ms = (time.perf_counter() - w1_start) * 1000

        # ── Checkpoint: merge and deduplicate ────────────────────
        memory_results, vault_snippets, episode_snippets = self._deduplicate(
            list(memory_results) if memory_results else [],
            list(vault_snippets) if vault_snippets else [],
            list(episode_snippets) if episode_snippets else [],
        )

        # ── Wave 2: Skill injection, User preferences (parallel) ─
        w2_start = time.perf_counter()

        skill_task = asyncio.get_running_loop().run_in_executor(
            None,
            self._get_skill_context,
            user_message,
        )
        pref_task = asyncio.get_running_loop().run_in_executor(
            None,
            self._get_user_pref_hint,
            user_id,
        )

        skill_context, user_pref_hint = await asyncio.gather(
            skill_task,
            pref_task,
            return_exceptions=True,
        )

        if isinstance(skill_context, BaseException):
            log.debug("context_skill_gather_failed", exc_info=skill_context)
            skill_context = ""
        if isinstance(user_pref_hint, BaseException):
            log.debug("context_pref_gather_failed", exc_info=user_pref_hint)
            user_pref_hint = ""

        wave2_ms = (time.perf_counter() - w2_start) * 1000

        # ── Inject into WorkingMemory ────────────────────────────
        if memory_results:
            wm.injected_memories = list(memory_results)

        # Vault + Episoden -> wm.injected_procedures (max 1 Slot)
        supplementary = self._format_supplementary_context(vault_snippets, episode_snippets)
        if supplementary and len(wm.injected_procedures) < 2:
            # Budget kuerzen
            if len(supplementary) > self._config.max_context_chars:
                supplementary = supplementary[: self._config.max_context_chars] + "\n[...]"
            wm.injected_procedures.insert(0, supplementary)

        duration_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "context_pipeline_complete",
            wave1_ms=round(wave1_ms, 1),
            wave2_ms=round(wave2_ms, 1),
            total_ms=round(duration_ms, 1),
        )

        return ContextResult(
            memory_results=list(memory_results) if memory_results else [],
            vault_snippets=list(vault_snippets) if vault_snippets else [],
            episode_snippets=list(episode_snippets) if episode_snippets else [],
            skill_context=skill_context or "",
            user_pref_hint=user_pref_hint or "",
            duration_ms=duration_ms,
            wave1_ms=wave1_ms,
            wave2_ms=wave2_ms,
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
                query=query,
                top_k=self._config.memory_top_k,
            )
        except Exception:
            log.debug("context_memory_search_failed", exc_info=True)
            return []

    async def _search_vault(self, query: str) -> list[str]:
        """Vault-Volltextsuche — async, ~10-50ms."""
        if not self._vault_tools:
            return []
        try:
            result = await self._vault_tools.vault_search(
                query=query,
                limit=self._config.vault_top_k,
            )
            # result ist ein String mit formatierten Treffern
            if result and "Keine Treffer" not in result:
                return [result]
            return []
        except Exception:
            log.debug("context_vault_search_failed", exc_info=True)
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
            return [f"[{d.isoformat()}] {text[:500]}" for d, text in recent if text.strip()]
        except Exception:
            log.debug("context_episode_fetch_failed", exc_info=True)
            return []

    def _deduplicate(
        self,
        memory_results: list[Any],
        vault_snippets: list[str],
        episode_snippets: list[str],
    ) -> tuple[list[Any], list[str], list[str]]:
        """Merge and deduplicate results from Wave 1.

        Removes duplicate vault snippets and episode entries that overlap
        with memory results (by text content).
        """
        # Collect memory text fingerprints for dedup
        memory_texts: set[str] = set()
        for mr in memory_results:
            chunk = getattr(mr, "chunk", None)
            if chunk:
                text = getattr(chunk, "text", "")
                if text:
                    memory_texts.add(text.strip().lower()[:200])

        # Deduplicate vault snippets against memory
        deduped_vault: list[str] = []
        seen_vault: set[str] = set()
        for snippet in vault_snippets:
            key = snippet.strip().lower()[:200]
            if key not in memory_texts and key not in seen_vault:
                seen_vault.add(key)
                deduped_vault.append(snippet)

        # Deduplicate episodes
        deduped_episodes: list[str] = []
        seen_episodes: set[str] = set()
        for ep in episode_snippets:
            key = ep.strip().lower()[:200]
            if key not in seen_episodes:
                seen_episodes.add(key)
                deduped_episodes.append(ep)

        return memory_results, deduped_vault, deduped_episodes

    def _get_skill_context(self, query: str) -> str:
        """Look up relevant skill context from SkillRegistry."""
        if not self._skill_registry:
            return ""
        try:
            # Try to find matching skills by keywords
            match_fn = getattr(self._skill_registry, "find_matching_skills", None)
            if match_fn is None:
                return ""
            matches = match_fn(query)
            if not matches:
                return ""
            # Format top match as context hint
            top = matches[0] if isinstance(matches, list) else matches
            name = getattr(top, "name", str(top))
            return f"Relevant skill: {name}"
        except Exception:
            log.debug("context_skill_lookup_failed", exc_info=True)
            return ""

    def _get_user_pref_hint(self, user_id: str) -> str:
        """Look up user preference hint from UserPreferenceStore."""
        if not self._user_pref_store or not user_id:
            return ""
        try:
            get_fn = getattr(self._user_pref_store, "get_preference", None)
            if get_fn is None:
                return ""
            pref = get_fn(user_id)
            if pref is None:
                return ""
            hint = getattr(pref, "verbosity_hint", "")
            return hint or ""
        except Exception:
            log.debug("context_pref_lookup_failed", exc_info=True)
            return ""

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
            parts.append("**Letzte Aktivit\u00e4ten:**\n" + "\n".join(episode_snippets[:3]))
        return "\n\n".join(parts)
