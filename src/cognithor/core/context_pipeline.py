"""Adaptive Context Pipeline — Automatic context enrichment.

Collects relevant context before each planner call from:
- Wave 1 (parallel): Memory (BM25-only), Vault (full-text search), Episodes
- Checkpoint: merge and deduplicate
- Wave 2 (parallel): Skill injection, User preference lookup

The result is injected into WorkingMemory.injected_memories and
injected_procedures so the planner automatically has access
to relevant knowledge.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import ContextPipelineConfig
    from cognithor.models import MemorySearchResult, WorkingMemory

log = get_logger(__name__)


@dataclass
class ContextResult:
    """Result of context enrichment."""

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
    # Sprint-24: PSE Auto-Switch — which profile the heuristic picked
    # for this request and why. Populated only when
    # ``ContextPipelineConfig.auto_switch_context_profile`` is True and
    # a ModelRouter is wired in. Empty string means the auto-switch did
    # not run for this request.
    selected_profile: str = ""
    profile_reason: str = ""


class ContextPipeline:
    """Automatically collect relevant context before the planner call.

    Two-wave parallel execution:
      Wave 1: memory search, vault search, episode retrieval (parallel)
      Checkpoint: merge and deduplicate results
      Wave 2: skill injection, user preference lookup (parallel)

    Dependency Injection: Memory and Vault are set after initialization
    via set_memory_manager() / set_vault_tools() (same pattern
    as Synthesis).
    """

    def __init__(self, config: ContextPipelineConfig) -> None:
        self._config = config
        self._memory_manager: Any | None = None  # MemoryManager (sync BM25)
        self._vault_tools: Any | None = None  # VaultTools (async search)
        self._skill_registry: Any | None = None  # SkillRegistry
        self._user_pref_store: Any | None = None  # UserPreferenceStore
        # Sprint-24: PSE Auto-Switch — wired by the gateway after
        # ``ModelRouter`` is initialised. Set via ``set_model_router``.
        # When None, the pipeline still runs but the auto-switch step
        # is a no-op.
        self._model_router: Any | None = None

    # ── Dependency Injection ──────────────────────────────────────

    def set_memory_manager(self, mm: Any) -> None:
        """Set the MemoryManager for BM25 search and episodes."""
        self._memory_manager = mm

    def set_vault_tools(self, vt: Any) -> None:
        """Set the VaultTools for full-text search."""
        self._vault_tools = vt

    def set_skill_registry(self, sr: Any) -> None:
        """Set the SkillRegistry for skill context injection."""
        self._skill_registry = sr

    def set_model_router(self, mr: Any) -> None:
        """Sprint-24: wire the ModelRouter for auto-switch profile activation.

        When set and ``config.auto_switch_context_profile`` is True,
        :meth:`enrich` calls ``recommend_context_profile(...)`` and
        applies the result via ``ModelRouter.set_context_profile(...)``.
        ContextVar isolation makes the change request-scoped — concurrent
        requests do not interfere.
        """
        self._model_router = mr

    def set_correction_memory(self, cm: Any) -> None:
        """Set the CorrectionMemory for correction reminders."""
        self._correction_memory = cm

    def set_user_pref_store(self, ups: Any) -> None:
        """Set the UserPreferenceStore for preference lookup."""
        self._user_pref_store = ups

    # ── Main method ──────────────────────────────────────────────

    async def enrich(
        self,
        user_message: str,
        wm: WorkingMemory,
        *,
        user_id: str = "",
        channel_kind: str | None = None,
    ) -> ContextResult:
        """Collect relevant context and inject it into WorkingMemory.

        Uses two-wave parallel execution:
          Wave 1: memory, vault, episodes (parallel via asyncio.gather)
          Checkpoint: merge and deduplicate
          Wave 2: skill injection, user preferences (parallel)

        Args:
            user_message: The current user message.
            wm: The active WorkingMemory instance.
            user_id: Optional user ID for preference lookup.
            channel_kind: Optional channel identifier (``"cli"``,
                ``"telegram"``, ``"arc_agi3"``, ...). Sprint-24 uses
                this to bias the auto-switch heuristic toward heavy
                channels (game-loop wants the 128 k window).

        Returns:
            ContextResult with collected data, metrics, and the
            selected context profile (Sprint-24 auto-switch).
        """
        if not self._config.enabled:
            return ContextResult(skipped=True, skip_reason="disabled")

        t0 = time.perf_counter()

        # Sprint-24: PSE Auto-Switch — runs *before* the smalltalk
        # short-circuit so latency-tight smalltalk responses still get
        # the right (small) window. Returns the selected profile + reason
        # so callers / tests can verify.
        selected_profile, profile_reason = self._maybe_apply_context_profile(
            user_message=user_message,
            wm=wm,
            channel_kind=channel_kind,
        )

        # Smalltalk/Short-Message Check
        if self._is_smalltalk(user_message):
            return ContextResult(
                skipped=True,
                skip_reason="smalltalk",
                duration_ms=(time.perf_counter() - t0) * 1000,
                selected_profile=selected_profile,
                profile_reason=profile_reason,
            )

        # ── Wave 1+2: All enrichment in parallel ──────────────────
        # Wave 1 (memory, vault, episodes) and Wave 2 (skill, prefs) are
        # independent — run them all concurrently for lower latency.
        w1_start = time.perf_counter()
        _loop = asyncio.get_running_loop()

        memory_task = self._search_memory_async(user_message)
        vault_task = self._search_vault(user_message)
        episode_task = _loop.run_in_executor(None, self._get_episodes)
        skill_task = _loop.run_in_executor(None, self._get_skill_context, user_message)
        pref_task = _loop.run_in_executor(None, self._get_user_pref_hint, user_id)

        (
            memory_results,
            vault_snippets,
            episode_snippets,
            skill_context,
            user_pref_hint,
        ) = await asyncio.gather(
            memory_task,
            vault_task,
            episode_task,
            skill_task,
            pref_task,
            return_exceptions=True,
        )

        # Handle exceptions gracefully
        if isinstance(memory_results, BaseException):
            log.debug("context_memory_gather_failed", exc_info=memory_results)
            memory_results = []
        if isinstance(vault_snippets, BaseException):
            log.debug("context_vault_gather_failed", exc_info=vault_snippets)
            vault_snippets = []
        if isinstance(episode_snippets, BaseException):
            log.debug("context_episode_gather_failed", exc_info=episode_snippets)
            episode_snippets = []
        if isinstance(skill_context, BaseException):
            log.debug("context_skill_gather_failed", exc_info=skill_context)
            skill_context = ""
        if isinstance(user_pref_hint, BaseException):
            log.debug("context_pref_gather_failed", exc_info=user_pref_hint)
            user_pref_hint = ""

        wave1_ms = (time.perf_counter() - w1_start) * 1000

        # ── Checkpoint: merge and deduplicate ────────────────────
        memory_results, vault_snippets, episode_snippets = self._deduplicate(
            list(memory_results) if memory_results else [],
            list(vault_snippets) if vault_snippets else [],
            list(episode_snippets) if episode_snippets else [],
        )

        wave2_ms = 0.0  # Wave 2 now runs in parallel with Wave 1

        # ── Inject into WorkingMemory ────────────────────────────
        if memory_results:
            wm.injected_memories = list(memory_results)

        # Vault + episodes -> wm.injected_procedures (max 1 slot)
        supplementary = self._format_supplementary_context(vault_snippets, episode_snippets)
        if supplementary and len(wm.injected_procedures) < 2:
            # Truncate budget
            if len(supplementary) > self._config.max_context_chars:
                supplementary = supplementary[: self._config.max_context_chars] + "\n[...]"
            wm.injected_procedures.insert(0, supplementary)

        # ── Correction Reminders (Smart Recovery) ────────────────
        if hasattr(self, "_correction_memory") and self._correction_memory:
            try:
                reminder = self._correction_memory.get_reminder(user_message)
                if reminder and len(wm.injected_procedures) < 3:
                    wm.injected_procedures.append(reminder)
                    log.debug("correction_reminder_injected", length=len(reminder))
            except Exception:
                log.debug("correction_reminder_failed", exc_info=True)

        # Wave 3: Tactical Memory insights
        tactical = getattr(self._memory_manager, "tactical", None)
        if tactical is not None:
            try:
                _budget = 400
                _tcfg = getattr(self._config, "tactical_memory", None)
                if _tcfg and hasattr(_tcfg, "budget_tokens"):
                    _budget = _tcfg.budget_tokens
                tactical_text = tactical.get_insights_for_llm(user_message, max_chars=_budget)
                if tactical_text:
                    wm.injected_tactical = tactical_text
            except Exception:
                log.debug("context_pipeline_tactical_failed", exc_info=True)

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
            selected_profile=selected_profile,
            profile_reason=profile_reason,
        )

    # ── Helper methods ─────────────────────────────────────────────

    def _maybe_apply_context_profile(
        self,
        *,
        user_message: str,
        wm: WorkingMemory,
        channel_kind: str | None,
    ) -> tuple[str, str]:
        """Sprint-24: pick + activate the recommended context profile.

        Reads the auto-switch flag, runs ``recommend_context_profile``,
        and applies the result to the wired ``ModelRouter``. Returns
        ``(profile_name, reason)`` so callers can log / assert.

        No-op when the flag is False, the router isn't wired, or the
        active model_router rejects the recommendation (we never let
        a profile mismatch break enrichment).
        """
        if not getattr(self._config, "auto_switch_context_profile", False):
            return "", ""
        if self._model_router is None:
            return "", "no model_router wired"

        # Lazy import to avoid the circular reference flagged in
        # ``context_profile_selector``'s docstring (model_router →
        # config → core → ... ).
        try:
            from cognithor.core.context_profile_selector import (
                recommend_context_profile,
            )
        except Exception:
            log.debug("context_profile_selector_import_failed", exc_info=True)
            return "", "selector import failed"

        # Detect attachments straight off the WorkingMemory — Sprint-23's
        # heuristic uses this to nudge medium prompts up one notch.
        has_attachments = bool(
            getattr(wm, "image_attachments", None) or getattr(wm, "video_attachment", None)
        )

        try:
            recommendation = recommend_context_profile(
                prompt_chars=len(user_message or ""),
                channel_kind=channel_kind,
                has_attachments=has_attachments,
            )
        except Exception:
            log.debug("context_profile_recommend_failed", exc_info=True)
            return "", "recommend failed"

        # Apply via the request-scoped ContextVar so concurrent requests
        # do not bleed into each other. ``set_context_profile`` raises
        # ValueError on unknown names — we swallow + log to avoid
        # breaking enrichment for an unrelated reason.
        try:
            self._model_router.set_context_profile(recommendation.profile)
        except Exception:
            log.warning(
                "context_profile_set_failed",
                profile=recommendation.profile,
                reason=recommendation.reason,
                exc_info=True,
            )
            return "", "set failed"

        log.info(
            "context_profile_auto_switch",
            profile=recommendation.profile,
            reason=recommendation.reason,
            channel=channel_kind or "",
            attachments=has_attachments,
        )
        return recommendation.profile, recommendation.reason

    def _is_smalltalk(self, text: str) -> bool:
        """Check whether message is smalltalk (no search needed)."""
        normalized = text.strip().lower().rstrip("!?.,")
        if len(normalized) < self._config.min_query_length:
            return True
        return normalized in self._config.smalltalk_patterns

    async def _search_memory_async(self, query: str) -> list[MemorySearchResult]:
        """Full hybrid search (BM25 + Vector + Graph) via MemoryManager."""
        if not self._memory_manager:
            return []
        try:
            if hasattr(self._memory_manager, "search_memory"):
                results: list[MemorySearchResult] = await self._memory_manager.search_memory(
                    query=query,
                    top_k=self._config.memory_top_k,
                    enhanced=True,
                )
                return results
            # Fallback: sync BM25-only (legacy)
            sync_results: list[MemorySearchResult] = self._memory_manager.search_memory_sync(
                query=query,
                top_k=self._config.memory_top_k,
            )
            return sync_results
        except Exception:
            log.debug("context_memory_search_failed", exc_info=True)
            return []

    def _search_memory(self, query: str) -> list[MemorySearchResult]:
        """BM25-only search — sync fallback, kept for backward compatibility."""
        if not self._memory_manager:
            return []
        try:
            results: list[MemorySearchResult] = self._memory_manager.search_memory_sync(
                query=query,
                top_k=self._config.memory_top_k,
            )
            return results
        except Exception:
            log.debug("context_memory_search_failed", exc_info=True)
            return []

    async def _search_vault(self, query: str) -> list[str]:
        """Vault full-text search -- async, ~10-50ms."""
        if not self._vault_tools:
            return []
        try:
            result = await self._vault_tools.vault_search(
                query=query,
                limit=self._config.vault_top_k,
            )
            # result is a string with formatted hits
            if result and "Keine Treffer" not in result:
                return [result]
            return []
        except Exception:
            log.debug("context_vault_search_failed", exc_info=True)
            return []

    def _get_episodes(self) -> list[str]:
        """Recent episodes -- sync, ~1-5ms."""
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
            matches = self._skill_registry.match(query, top_k=3)
            if not matches:
                return ""
            lines = []
            for m in matches[:3]:
                s = m.skill
                kw = ", ".join(s.trigger_keywords[:5]) if s.trigger_keywords else ""
                lines.append(f"- {s.name}: {s.description} (Keywords: {kw})")
            return "Verfuegbare Skills:\n" + "\n".join(lines)
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
        """Format vault+episodes as a compact context string."""
        parts: list[str] = []
        if vault_snippets:
            parts.append("**Vault-Notizen:**\n" + "\n".join(vault_snippets[:3]))
        if episode_snippets:
            parts.append("**Letzte Aktivit\u00e4ten:**\n" + "\n".join(episode_snippets[:3]))
        return "\n\n".join(parts)
