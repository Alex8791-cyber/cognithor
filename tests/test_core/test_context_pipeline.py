"""
Tests für jarvis.core.context_pipeline — Adaptive Context Pipeline.

Testet:
  - Smalltalk-Erkennung (skip)
  - Kurze Nachrichten (skip)
  - Memory-Injection in WorkingMemory
  - Vault-Injection in injected_procedures
  - Episoden-Injection
  - Graceful degradation ohne Memory/Vault
  - Pipeline deaktiviert
  - Procedures-Slot wird nicht überschrieben
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import ContextPipelineConfig
from jarvis.core.context_pipeline import ContextPipeline, ContextResult
from jarvis.models import Chunk, MemorySearchResult, WorkingMemory


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def pipeline_config() -> ContextPipelineConfig:
    """Standard-Config für Tests."""
    return ContextPipelineConfig()


@pytest.fixture
def pipeline(pipeline_config: ContextPipelineConfig) -> ContextPipeline:
    """ContextPipeline mit Standard-Config."""
    return ContextPipeline(pipeline_config)


@pytest.fixture
def wm() -> WorkingMemory:
    """Leere WorkingMemory."""
    return WorkingMemory()


def _make_search_result(text: str, score: float = 0.5) -> MemorySearchResult:
    """Erzeugt ein MemorySearchResult für Tests."""
    return MemorySearchResult(
        chunk=Chunk(text=text, source_path="test.md"),
        score=score,
        bm25_score=score,
    )


def _make_memory_manager(
    search_results: list[MemorySearchResult] | None = None,
    episodes: list[tuple[date, str]] | None = None,
) -> MagicMock:
    """Erzeugt einen Mock-MemoryManager."""
    mm = MagicMock()
    mm.search_memory_sync.return_value = search_results or []
    episodic = MagicMock()
    episodic.get_recent.return_value = episodes or []
    mm.episodic = episodic
    return mm


def _make_vault_tools(search_result: str = "") -> AsyncMock:
    """Erzeugt Mock-VaultTools."""
    vt = AsyncMock()
    vt.vault_search = AsyncMock(return_value=search_result)
    return vt


# ── Tests ─────────────────────────────────────────────────────────


class TestSmalltalkSkip:
    """Smalltalk und kurze Nachrichten werden übersprungen."""

    @pytest.mark.asyncio
    async def test_hallo_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("Hallo", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_hi_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("Hi", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_danke_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("Danke!", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_ok_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("ok", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_guten_morgen_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("Guten Morgen!", wm)
        assert result.skipped is True


class TestShortMessageSkip:
    """Nachrichten unter min_query_length werden übersprungen."""

    @pytest.mark.asyncio
    async def test_short_skip(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        result = await pipeline.enrich("ja", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_exact_threshold(self, wm: WorkingMemory) -> None:
        config = ContextPipelineConfig(min_query_length=5)
        pipeline = ContextPipeline(config)
        # "abcd" = 4 chars < 5 → skip
        result = await pipeline.enrich("abcd", wm)
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_above_threshold_not_skipped(self, wm: WorkingMemory) -> None:
        config = ContextPipelineConfig(min_query_length=5)
        pipeline = ContextPipeline(config)
        # "abcde" = 5 chars → NOT skipped (no memory/vault, but not skipped)
        result = await pipeline.enrich("abcde", wm)
        assert result.skipped is False


class TestMemoryInjection:
    """Memory-Ergebnisse landen in wm.injected_memories."""

    @pytest.mark.asyncio
    async def test_memory_results_injected(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        results = [
            _make_search_result("Projekt Alpha hat Deadline am 15.03", 0.8),
            _make_search_result("Alpha Budget: 50k EUR", 0.6),
        ]
        mm = _make_memory_manager(search_results=results)
        pipeline.set_memory_manager(mm)

        ctx = await pipeline.enrich("Was ist der Stand bei Projekt Alpha?", wm)

        assert ctx.skipped is False
        assert len(ctx.memory_results) == 2
        assert len(wm.injected_memories) == 2
        assert wm.injected_memories[0].chunk.text == "Projekt Alpha hat Deadline am 15.03"

    @pytest.mark.asyncio
    async def test_memory_search_called_with_correct_params(
        self, pipeline: ContextPipeline, wm: WorkingMemory,
    ) -> None:
        mm = _make_memory_manager()
        pipeline.set_memory_manager(mm)

        await pipeline.enrich("Wie funktioniert das Backup-System?", wm)

        mm.search_memory_sync.assert_called_once_with(
            query="Wie funktioniert das Backup-System?",
            top_k=4,  # default memory_top_k
        )


class TestVaultInjection:
    """Vault-Snippets landen in wm.injected_procedures."""

    @pytest.mark.asyncio
    async def test_vault_results_injected(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        vt = _make_vault_tools("## Tesla Q4 Ergebnisse\nUmsatz: 25 Mrd USD")
        pipeline.set_vault_tools(vt)

        ctx = await pipeline.enrich("Was wissen wir über Tesla?", wm)

        assert len(ctx.vault_snippets) == 1
        assert len(wm.injected_procedures) == 1
        assert "Vault-Notizen" in wm.injected_procedures[0]
        assert "Tesla Q4" in wm.injected_procedures[0]

    @pytest.mark.asyncio
    async def test_vault_keine_treffer_ignored(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        vt = _make_vault_tools("Keine Treffer für 'xyz' gefunden.")
        pipeline.set_vault_tools(vt)

        ctx = await pipeline.enrich("Was wissen wir über xyz?", wm)

        assert len(ctx.vault_snippets) == 0
        assert len(wm.injected_procedures) == 0


class TestEpisodeInjection:
    """Episoden-Texte im Kontext."""

    @pytest.mark.asyncio
    async def test_episodes_injected(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        episodes = [
            (date(2026, 3, 1), "## 14:30 · Meeting mit Team\nStatus-Update besprochen"),
            (date(2026, 2, 28), "## 10:00 · Code-Review\nBug in Auth gefunden"),
        ]
        mm = _make_memory_manager(episodes=episodes)
        pipeline.set_memory_manager(mm)

        ctx = await pipeline.enrich("Was haben wir gestern gemacht?", wm)

        assert len(ctx.episode_snippets) == 2
        assert len(wm.injected_procedures) == 1
        assert "Letzte Aktivitäten" in wm.injected_procedures[0]
        assert "2026-03-01" in wm.injected_procedures[0]


class TestNoMemoryManager:
    """Graceful degradation ohne MemoryManager."""

    @pytest.mark.asyncio
    async def test_no_memory_manager(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        # Kein set_memory_manager aufgerufen
        ctx = await pipeline.enrich("Was ist der Stand bei Projekt Alpha?", wm)

        assert ctx.skipped is False
        assert len(ctx.memory_results) == 0
        assert len(wm.injected_memories) == 0


class TestNoVaultTools:
    """Graceful degradation ohne VaultTools."""

    @pytest.mark.asyncio
    async def test_no_vault_tools(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        mm = _make_memory_manager()
        pipeline.set_memory_manager(mm)
        # Kein set_vault_tools aufgerufen

        ctx = await pipeline.enrich("Was steht im Vault über Tesla?", wm)

        assert ctx.skipped is False
        assert len(ctx.vault_snippets) == 0


class TestDisabled:
    """Pipeline deaktiviert → nichts passiert."""

    @pytest.mark.asyncio
    async def test_disabled_skips(self, wm: WorkingMemory) -> None:
        config = ContextPipelineConfig(enabled=False)
        pipeline = ContextPipeline(config)
        mm = _make_memory_manager(
            search_results=[_make_search_result("should not appear")],
        )
        pipeline.set_memory_manager(mm)

        ctx = await pipeline.enrich("Was ist Projekt Alpha?", wm)

        assert ctx.skipped is True
        assert ctx.skip_reason == "disabled"
        assert len(wm.injected_memories) == 0
        mm.search_memory_sync.assert_not_called()


class TestProceduresSlotPreserved:
    """Skill-Procedure wird NICHT überschrieben."""

    @pytest.mark.asyncio
    async def test_existing_procedures_preserved(
        self, pipeline: ContextPipeline, wm: WorkingMemory,
    ) -> None:
        # Simuliere: Skill hat bereits 2 Procedures injiziert
        wm.injected_procedures = ["Skill Step 1", "Skill Step 2"]

        vt = _make_vault_tools("Vault Treffer XYZ")
        pipeline.set_vault_tools(vt)

        episodes = [(date(2026, 3, 1), "Episode Text")]
        mm = _make_memory_manager(episodes=episodes)
        pipeline.set_memory_manager(mm)

        await pipeline.enrich("Was wissen wir über Projekt Alpha?", wm)

        # Procedures dürfen NICHT verändert werden (≥2 Einträge)
        assert len(wm.injected_procedures) == 2
        assert wm.injected_procedures[0] == "Skill Step 1"
        assert wm.injected_procedures[1] == "Skill Step 2"

    @pytest.mark.asyncio
    async def test_one_procedure_allows_insert(
        self, pipeline: ContextPipeline, wm: WorkingMemory,
    ) -> None:
        # Nur 1 Skill-Procedure → Slot frei für Kontext
        wm.injected_procedures = ["Skill Step 1"]

        vt = _make_vault_tools("Vault Treffer XYZ")
        pipeline.set_vault_tools(vt)

        await pipeline.enrich("Was wissen wir über Projekt Alpha?", wm)

        # Kontext wird VOR dem Skill eingefügt
        assert len(wm.injected_procedures) == 2
        assert "Vault-Notizen" in wm.injected_procedures[0]
        assert wm.injected_procedures[1] == "Skill Step 1"


class TestContextResult:
    """ContextResult Dataclass."""

    def test_default_values(self) -> None:
        cr = ContextResult()
        assert cr.memory_results == []
        assert cr.vault_snippets == []
        assert cr.episode_snippets == []
        assert cr.duration_ms == 0.0
        assert cr.skipped is False
        assert cr.skip_reason == ""


class TestMaxContextChars:
    """Budget-Kürzung bei max_context_chars."""

    @pytest.mark.asyncio
    async def test_long_context_truncated(self, wm: WorkingMemory) -> None:
        config = ContextPipelineConfig(max_context_chars=50)
        pipeline = ContextPipeline(config)

        vt = _make_vault_tools("A" * 200)
        pipeline.set_vault_tools(vt)

        await pipeline.enrich("Was wissen wir über dieses lange Thema?", wm)

        assert len(wm.injected_procedures) == 1
        # Should be truncated to max_context_chars + "[...]"
        assert len(wm.injected_procedures[0]) < 200
        assert wm.injected_procedures[0].endswith("[...]")


class TestDurationTracking:
    """Pipeline misst die Ausführungsdauer."""

    @pytest.mark.asyncio
    async def test_duration_measured(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        ctx = await pipeline.enrich("Was ist der Stand bei Projekt Alpha?", wm)
        assert ctx.duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_smalltalk_duration_measured(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        ctx = await pipeline.enrich("Hallo", wm)
        assert ctx.skipped is True
        assert ctx.duration_ms >= 0.0


class TestMemorySearchError:
    """Memory-Suche Fehler wird graceful behandelt."""

    @pytest.mark.asyncio
    async def test_memory_exception_returns_empty(
        self, pipeline: ContextPipeline, wm: WorkingMemory,
    ) -> None:
        mm = MagicMock()
        mm.search_memory_sync.side_effect = RuntimeError("BM25 index corrupt")
        mm.episodic = MagicMock()
        mm.episodic.get_recent.return_value = []
        pipeline.set_memory_manager(mm)

        ctx = await pipeline.enrich("Was ist Projekt Alpha?", wm)

        assert ctx.skipped is False
        assert len(ctx.memory_results) == 0
        assert len(wm.injected_memories) == 0


class TestVaultSearchError:
    """Vault-Suche Fehler wird graceful behandelt."""

    @pytest.mark.asyncio
    async def test_vault_exception_returns_empty(
        self, pipeline: ContextPipeline, wm: WorkingMemory,
    ) -> None:
        vt = AsyncMock()
        vt.vault_search = AsyncMock(side_effect=RuntimeError("Vault not found"))
        pipeline.set_vault_tools(vt)

        ctx = await pipeline.enrich("Was steht im Vault?", wm)

        assert len(ctx.vault_snippets) == 0


class TestCombinedInjection:
    """Memory + Vault + Episoden zusammen."""

    @pytest.mark.asyncio
    async def test_all_sources_combined(self, pipeline: ContextPipeline, wm: WorkingMemory) -> None:
        memory_results = [_make_search_result("Memory Fakt 1", 0.9)]
        episodes = [(date(2026, 3, 1), "Episode Heute")]
        mm = _make_memory_manager(search_results=memory_results, episodes=episodes)
        pipeline.set_memory_manager(mm)

        vt = _make_vault_tools("Vault Notiz ABC")
        pipeline.set_vault_tools(vt)

        ctx = await pipeline.enrich("Was ist der aktuelle Stand?", wm)

        assert len(ctx.memory_results) == 1
        assert len(ctx.vault_snippets) == 1
        assert len(ctx.episode_snippets) == 1
        assert len(wm.injected_memories) == 1
        assert len(wm.injected_procedures) == 1
        assert "Vault-Notizen" in wm.injected_procedures[0]
        assert "Letzte Aktivitäten" in wm.injected_procedures[0]
