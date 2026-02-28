"""Tests für Enhanced Retrieval.

Testet:
  - QueryDecomposer: Rule-based Dekomposition
  - Reciprocal Rank Fusion: Multi-Query Merge
  - CorrectiveRAG: Relevanz-Prüfung, Alternative Queries
  - FrequencyTracker: Zugriffs-Boost
  - EpisodicCompressor: Heuristische und zeitbasierte Kompression
  - EnhancedSearchPipeline: End-to-End Orchestrierung
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jarvis.memory.enhanced_retrieval import (
    CompressedEpisode,
    CorrectiveRAG,
    DecomposedQuery,
    EnhancedSearchPipeline,
    EpisodicCompressor,
    FrequencyTracker,
    QueryDecomposer,
    RelevanceVerdict,
    reciprocal_rank_fusion,
)
from jarvis.models import Chunk, MemorySearchResult, MemoryTier


# ============================================================================
# Helpers
# ============================================================================


def _make_result(
    chunk_id: str = "c1",
    text: str = "Test chunk",
    score: float = 0.5,
    bm25: float = 0.0,
    vector: float = 0.0,
    graph: float = 0.0,
) -> MemorySearchResult:
    """Erstellt ein MemorySearchResult für Tests."""
    return MemorySearchResult(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            source_path="test.md",
            memory_tier=MemoryTier.SEMANTIC,
        ),
        score=score,
        bm25_score=bm25,
        vector_score=vector,
        graph_score=graph,
    )


# ============================================================================
# QueryDecomposer
# ============================================================================


class TestQueryDecomposer:
    """Rule-based Query-Dekomposition."""

    def setup_method(self) -> None:
        self.decomposer = QueryDecomposer()

    def test_simple_query(self) -> None:
        result = self.decomposer.decompose("Was ist Jarvis?")
        assert result.query_type == "simple"
        assert result.sub_queries == ["Was ist Jarvis?"]
        assert result.original == "Was ist Jarvis?"

    def test_empty_query(self) -> None:
        result = self.decomposer.decompose("")
        assert result.sub_queries == [""]

    def test_comparison_query(self) -> None:
        result = self.decomposer.decompose(
            "Unterschied zwischen BU-Versicherung und Risikolebensversicherung",
        )
        assert result.query_type == "compound"
        assert len(result.sub_queries) == 3  # Original + 2 Teile
        assert result.original in result.sub_queries

    def test_vs_comparison(self) -> None:
        result = self.decomposer.decompose("WWK vs Allianz BU-Tarife")
        assert result.query_type == "compound"
        assert len(result.sub_queries) == 3

    def test_versus_comparison(self) -> None:
        result = self.decomposer.decompose("React versus Vue für Frontends")
        assert result.query_type == "compound"

    def test_pros_cons_aspects(self) -> None:
        result = self.decomposer.decompose("Vor- und Nachteile von Remote-Arbeit")
        assert result.query_type == "multi_aspect"
        assert any("Vorteile" in sq for sq in result.sub_queries)
        assert any("Nachteile" in sq for sq in result.sub_queries)

    def test_conjunction_split(self) -> None:
        result = self.decomposer.decompose(
            "Erkläre Jarvis Memory System und die Sandbox-Architektur",
        )
        assert result.query_type == "compound"
        assert len(result.sub_queries) >= 2

    def test_short_conjunction_not_split(self) -> None:
        """Zu kurze Teile werden nicht gesplittet."""
        result = self.decomposer.decompose("A und B")
        assert result.query_type == "simple"  # Zu kurz zum Splitten

    @pytest.mark.asyncio
    async def test_llm_fallback_without_llm(self) -> None:
        """Ohne LLM fällt decompose_with_llm auf rule-based zurück."""
        decomposer = QueryDecomposer(llm_fn=None)
        result = await decomposer.decompose_with_llm("Test Query")
        assert len(result.sub_queries) >= 1

    @pytest.mark.asyncio
    async def test_llm_decomposition(self) -> None:
        """Mit LLM werden Sub-Queries generiert."""
        async def fake_llm(prompt: str) -> str:
            return "Sub-Query 1\nSub-Query 2"

        decomposer = QueryDecomposer(llm_fn=fake_llm)
        result = await decomposer.decompose_with_llm("Komplexe Frage")
        assert result.query_type == "llm_decomposed"
        assert len(result.sub_queries) >= 3  # Original + 2 LLM

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self) -> None:
        """Bei LLM-Fehler fällt es auf rule-based zurück."""
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        decomposer = QueryDecomposer(llm_fn=failing_llm)
        result = await decomposer.decompose_with_llm("Test")
        assert len(result.sub_queries) >= 1  # Rule-based Fallback


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================


class TestRRF:
    """Reciprocal Rank Fusion Tests."""

    def test_single_list(self) -> None:
        results = [
            _make_result("c1", score=0.9),
            _make_result("c2", score=0.5),
        ]
        merged = reciprocal_rank_fusion([results])
        assert len(merged) == 2
        assert merged[0].chunk.id == "c1"

    def test_two_lists_overlap(self) -> None:
        """Chunks die in beiden Listen vorkommen erhalten höheren RRF-Score."""
        list1 = [
            _make_result("c1", score=0.9),
            _make_result("c2", score=0.7),
            _make_result("c3", score=0.3),
        ]
        list2 = [
            _make_result("c2", score=0.8),
            _make_result("c1", score=0.6),
            _make_result("c4", score=0.5),
        ]
        merged = reciprocal_rank_fusion([list1, list2])

        # c1 und c2 kommen in beiden Listen vor → höchste RRF-Scores
        ids = [r.chunk.id for r in merged]
        # c1 und c2 sollten oben sein
        top_2 = set(ids[:2])
        assert "c1" in top_2 or "c2" in top_2

    def test_empty_lists(self) -> None:
        merged = reciprocal_rank_fusion([[], []])
        assert merged == []

    def test_top_n_limit(self) -> None:
        results = [_make_result(f"c{i}", score=0.5) for i in range(10)]
        merged = reciprocal_rank_fusion([results], top_n=3)
        assert len(merged) == 3

    def test_rrf_deterministic(self) -> None:
        """Gleiche Eingabe → gleiches Ergebnis."""
        list1 = [_make_result("a", score=0.9), _make_result("b", score=0.5)]
        list2 = [_make_result("b", score=0.8), _make_result("a", score=0.3)]

        result_a = reciprocal_rank_fusion([list1, list2])
        result_b = reciprocal_rank_fusion([list1, list2])

        assert [r.chunk.id for r in result_a] == [r.chunk.id for r in result_b]

    def test_unique_chunks_both_included(self) -> None:
        """Chunks die nur in einer Liste vorkommen werden auch aufgenommen."""
        list1 = [_make_result("only_in_1", score=0.9)]
        list2 = [_make_result("only_in_2", score=0.8)]

        merged = reciprocal_rank_fusion([list1, list2])
        ids = {r.chunk.id for r in merged}
        assert "only_in_1" in ids
        assert "only_in_2" in ids

    def test_k_parameter_effect(self) -> None:
        """Höheres k → flachere RRF-Scores (weniger Positionseffekt)."""
        results = [_make_result("c1", score=0.9), _make_result("c2", score=0.1)]

        low_k = reciprocal_rank_fusion([results], k=1)
        high_k = reciprocal_rank_fusion([results], k=1000)

        # Bei k=1 ist der Unterschied zwischen Rank 1 und 2 größer
        diff_low = low_k[0].score - low_k[1].score
        diff_high = high_k[0].score - high_k[1].score
        assert diff_low > diff_high


# ============================================================================
# Corrective RAG
# ============================================================================


class TestCorrectiveRAG:
    """Relevanz-Prüfung und Alternative Queries."""

    def setup_method(self) -> None:
        self.crag = CorrectiveRAG(min_score_threshold=0.15, min_relevant_count=2)

    def test_all_relevant(self) -> None:
        results = [
            _make_result("c1", text="Versicherung BU Tarif", score=0.8),
            _make_result("c2", text="BU Vergleich Anbieter", score=0.6),
        ]
        verdict = self.crag.evaluate_relevance_heuristic("BU Tarif Versicherung", results)
        assert len(verdict.relevant_results) == 2
        assert verdict.needs_retry is False

    def test_low_score_irrelevant(self) -> None:
        results = [
            _make_result("c1", text="Unrelated stuff", score=0.05),
            _make_result("c2", text="Also unrelated", score=0.03),
        ]
        verdict = self.crag.evaluate_relevance_heuristic("BU Tarif", results)
        assert len(verdict.relevant_results) == 0
        assert len(verdict.irrelevant_results) == 2
        assert verdict.needs_retry is True

    def test_mixed_relevance(self) -> None:
        results = [
            _make_result("good", text="BU Tarif Vergleich", score=0.7),
            _make_result("bad", text="Kochen Rezepte Pasta", score=0.01),
        ]
        verdict = self.crag.evaluate_relevance_heuristic("BU Tarif", results)
        assert len(verdict.relevant_results) >= 1
        assert any(r.chunk.id == "good" for r in verdict.relevant_results)

    def test_empty_results(self) -> None:
        verdict = self.crag.evaluate_relevance_heuristic("Test", [])
        assert verdict.needs_retry is False  # Kein Retry bei leeren Ergebnissen
        assert verdict.confidence == 0.0

    def test_generate_alternatives_keywords_only(self) -> None:
        alts = self.crag.generate_alternative_queries(
            "Was ist der Unterschied zwischen BU und Risikolebensversicherung",
        )
        assert len(alts) >= 1
        # Sollte kürzere Variante enthalten
        assert any(len(a.split()) < 8 for a in alts)

    def test_generate_alternatives_short_query(self) -> None:
        alts = self.crag.generate_alternative_queries("BU Tarif")
        # Kurze Queries → weniger Alternativen
        assert isinstance(alts, list)

    def test_high_overlap_counts_as_relevant(self) -> None:
        """Hoher Wort-Overlap zählt auch bei niedrigerem Score."""
        results = [
            _make_result(
                "overlap",
                text="Jarvis Memory System Hybrid Search BM25 Vektor",
                score=0.2,
            ),
        ]
        verdict = self.crag.evaluate_relevance_heuristic(
            "Jarvis Memory Hybrid Search",
            results,
        )
        assert len(verdict.relevant_results) >= 1


# ============================================================================
# FrequencyTracker
# ============================================================================


class TestFrequencyTracker:
    """Zugriffs-Tracking und Frequency-Boost."""

    def setup_method(self) -> None:
        self.tracker = FrequencyTracker(frequency_weight=0.1)

    def test_initial_state(self) -> None:
        assert self.tracker.total_accesses == 0
        assert self.tracker.get_count("any") == 0
        assert self.tracker.boost_factor("any") == 1.0

    def test_record_single_access(self) -> None:
        self.tracker.record_access("c1")
        assert self.tracker.get_count("c1") == 1
        assert self.tracker.total_accesses == 1

    def test_record_multiple(self) -> None:
        self.tracker.record_accesses(["c1", "c2", "c1"])
        assert self.tracker.get_count("c1") == 2
        assert self.tracker.get_count("c2") == 1

    def test_boost_increases_with_frequency(self) -> None:
        boost_0 = self.tracker.boost_factor("c1")  # 1.0
        self.tracker.record_access("c1")
        boost_1 = self.tracker.boost_factor("c1")
        self.tracker.record_access("c1")
        self.tracker.record_access("c1")
        boost_3 = self.tracker.boost_factor("c1")

        assert boost_0 == 1.0
        assert boost_1 > boost_0
        assert boost_3 > boost_1

    def test_boost_is_logarithmic(self) -> None:
        """Boost wächst logarithmisch, nicht linear."""
        for _ in range(10):
            self.tracker.record_access("c1")
        boost_10 = self.tracker.boost_factor("c1")

        for _ in range(90):
            self.tracker.record_access("c1")
        boost_100 = self.tracker.boost_factor("c1")

        # 100x Zugriffe → nicht 10x Boost, sondern ~2x (logarithmisch)
        assert boost_100 < boost_10 * 3

    def test_apply_boost_reorders(self) -> None:
        """Frequency-Boost ändert die Reihenfolge."""
        # c2 hat niedrigeren Score aber wurde öfter abgerufen
        for _ in range(20):
            self.tracker.record_access("c2")

        results = [
            _make_result("c1", score=0.6),
            _make_result("c2", score=0.5),
        ]
        boosted = self.tracker.apply_boost(results)

        # c2 sollte jetzt oben sein (0.5 × boost > 0.6 × 1.0)
        assert boosted[0].chunk.id == "c2"

    def test_apply_boost_preserves_unboosted(self) -> None:
        """Chunks ohne Zugriffe behalten ihren Score."""
        results = [_make_result("new", score=0.7)]
        boosted = self.tracker.apply_boost(results)
        assert boosted[0].score == 0.7

    def test_top_accessed(self) -> None:
        self.tracker.record_accesses(["a"] * 10 + ["b"] * 5 + ["c"] * 1)
        top = self.tracker.top_accessed(2)
        assert top[0] == ("a", 10)
        assert top[1] == ("b", 5)

    def test_clear(self) -> None:
        self.tracker.record_access("c1")
        self.tracker.clear()
        assert self.tracker.total_accesses == 0

    def test_stats(self) -> None:
        self.tracker.record_accesses(["a", "b", "a"])
        stats = self.tracker.stats()
        assert stats["tracked_chunks"] == 2
        assert stats["total_accesses"] == 3


# ============================================================================
# EpisodicCompressor
# ============================================================================


class TestEpisodicCompressor:
    """Episodenkompression."""

    def setup_method(self) -> None:
        self.compressor = EpisodicCompressor(retention_days=30)

    def test_identify_compressible(self) -> None:
        today = date(2026, 2, 22)
        dates = [
            date(2026, 1, 1),   # 52 Tage alt → komprimierbar
            date(2026, 1, 20),  # 33 Tage alt → komprimierbar
            date(2026, 2, 15),  # 7 Tage alt → behalten
            date(2026, 2, 22),  # heute → behalten
        ]
        compressible = self.compressor.identify_compressible(dates, reference_date=today)
        assert len(compressible) == 2
        assert date(2026, 1, 1) in compressible
        assert date(2026, 1, 20) in compressible

    def test_none_compressible(self) -> None:
        today = date(2026, 2, 22)
        dates = [date(2026, 2, 21), date(2026, 2, 22)]
        assert self.compressor.identify_compressible(dates, reference_date=today) == []

    def test_group_into_weeks(self) -> None:
        dates = [
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
            # Gap
            date(2026, 1, 20),
            date(2026, 1, 21),
        ]
        weeks = self.compressor.group_into_weeks(dates)
        assert len(weeks) == 2
        assert weeks[0] == (date(2026, 1, 5), date(2026, 1, 7))
        assert weeks[1] == (date(2026, 1, 20), date(2026, 1, 21))

    def test_group_empty(self) -> None:
        assert self.compressor.group_into_weeks([]) == []

    def test_group_single_date(self) -> None:
        weeks = self.compressor.group_into_weeks([date(2026, 1, 1)])
        assert len(weeks) == 1
        assert weeks[0] == (date(2026, 1, 1), date(2026, 1, 1))

    def test_compress_heuristic(self) -> None:
        entries = [
            "Alexander hat mit WWK über die neuen BU-Tarife gesprochen.",
            "Die Preise sind um 5% gestiegen im Vergleich zu 2025.",
            "Kunde Max Müller hat eine Police abgeschlossen.",
            "Das Wetter war schön.",  # Sollte niedriger ranken
        ]
        compressed = self.compressor.compress_heuristic(
            entries,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 7),
        )

        assert compressed.start_date == date(2026, 1, 1)
        assert compressed.end_date == date(2026, 1, 7)
        assert compressed.summary  # Nicht leer
        assert compressed.original_entry_count == 4
        assert compressed.days_covered == 7
        assert len(compressed.entities_mentioned) > 0  # WWK, Alexander etc.
        assert len(compressed.key_facts) > 0

    def test_compress_empty_entries(self) -> None:
        compressed = self.compressor.compress_heuristic(
            [],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 1),
        )
        assert compressed.original_entry_count == 0

    def test_compress_favors_entities_and_numbers(self) -> None:
        """Sätze mit Entitäten und Zahlen werden bevorzugt."""
        entries = [
            "Es war ein normaler Tag.",  # Wenig Info
            "WWK hat den BU-Tarif Premium um 12% angehoben.",  # Entity + Zahl
        ]
        compressed = self.compressor.compress_heuristic(
            entries,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 1),
            max_sentences=1,
        )
        # Der informative Satz sollte bevorzugt werden
        assert "12" in compressed.summary or "WWK" in compressed.summary

    @pytest.mark.asyncio
    async def test_compress_with_llm_fallback(self) -> None:
        """Ohne LLM fällt auf heuristic zurück."""
        compressor = EpisodicCompressor(llm_fn=None)
        compressed = await compressor.compress_with_llm(
            ["Eintrag 1", "Eintrag 2"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 7),
        )
        assert compressed.original_entry_count == 2

    @pytest.mark.asyncio
    async def test_compress_with_llm(self) -> None:
        async def fake_llm(prompt: str) -> str:
            return "Alexander hat KW1 an BU-Tarifen gearbeitet. WWK war Fokus."

        compressor = EpisodicCompressor(llm_fn=fake_llm)
        compressed = await compressor.compress_with_llm(
            ["Eintrag 1"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 7),
        )
        assert "Alexander" in compressed.summary or "WWK" in compressed.summary

    def test_date_range_property(self) -> None:
        ep = CompressedEpisode(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 7),
            summary="Test",
        )
        assert ep.date_range == "2026-01-01 – 2026-01-07"


# ============================================================================
# Enhanced Search Pipeline (Integration)
# ============================================================================


class TestEnhancedSearchPipeline:
    """End-to-End Pipeline Tests."""

    @pytest.fixture
    def mock_search(self) -> AsyncMock:
        """Mock HybridSearch."""
        search = AsyncMock()
        search.search = AsyncMock(return_value=[
            _make_result("c1", text="BU Tarif Info", score=0.8),
            _make_result("c2", text="Versicherung Details", score=0.5),
            _make_result("c3", text="Andere Info", score=0.2),
        ])
        return search

    @pytest.mark.asyncio
    async def test_simple_query(self, mock_search: AsyncMock) -> None:
        pipeline = EnhancedSearchPipeline(mock_search)
        results = await pipeline.search("Was ist Jarvis?")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_compound_query_multi_search(self, mock_search: AsyncMock) -> None:
        """Zusammengesetzte Query führt zu mehreren Suchen."""
        pipeline = EnhancedSearchPipeline(mock_search)
        await pipeline.search("Unterschied zwischen BU und Risikoleben")

        # Decomposition erzeugt 3 Sub-Queries → 3 Suchaufrufe
        assert mock_search.search.call_count >= 2

    @pytest.mark.asyncio
    async def test_decomposition_disabled(self, mock_search: AsyncMock) -> None:
        pipeline = EnhancedSearchPipeline(
            mock_search,
            enable_decomposition=False,
        )
        await pipeline.search("Unterschied zwischen A und B")
        # Ohne Decomposition: genau 1 Suchaufruf
        assert mock_search.search.call_count == 1

    @pytest.mark.asyncio
    async def test_frequency_boost_applied(self, mock_search: AsyncMock) -> None:
        pipeline = EnhancedSearchPipeline(mock_search)

        # Mehrere Suchen → c1 wird öfter gefunden
        await pipeline.search("Test 1")
        await pipeline.search("Test 2")

        # Frequency-Tracker sollte Zugriffe haben
        assert pipeline.frequency_tracker.total_accesses > 0

    @pytest.mark.asyncio
    async def test_top_k_respected(self, mock_search: AsyncMock) -> None:
        pipeline = EnhancedSearchPipeline(mock_search)
        results = await pipeline.search("Test", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_correction_triggers_retry(self) -> None:
        """Wenn Ergebnisse irrelevant sind, wird erneut gesucht."""
        call_count = 0

        async def search_fn(*args: Any, **kwargs: Any) -> list[MemorySearchResult]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Erste Suche: irrelevante Ergebnisse
                return [_make_result("bad", text="Pasta Rezepte Kochen", score=0.05)]
            # Retry: bessere Ergebnisse
            return [_make_result("good", text="BU Tarif Versicherung", score=0.7)]

        mock = AsyncMock()
        mock.search = search_fn

        pipeline = EnhancedSearchPipeline(
            mock,
            enable_correction=True,
            enable_decomposition=False,
        )
        results = await pipeline.search("BU Tarif Info")
        # Mindestens 2 Suchaufrufe (original + retry)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_all_features_disabled(self, mock_search: AsyncMock) -> None:
        """Pipeline funktioniert auch mit allen Features deaktiviert."""
        pipeline = EnhancedSearchPipeline(
            mock_search,
            enable_decomposition=False,
            enable_correction=False,
            enable_frequency_boost=False,
        )
        results = await pipeline.search("Test")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_stats(self, mock_search: AsyncMock) -> None:
        pipeline = EnhancedSearchPipeline(mock_search)
        stats = pipeline.stats()
        assert "decomposition_enabled" in stats
        assert "frequency" in stats
