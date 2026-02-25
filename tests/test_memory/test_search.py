"""Tests für memory/search.py · Hybrid-Suche."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from jarvis.memory.embeddings import EmbeddingClient
from jarvis.memory.indexer import MemoryIndex
from jarvis.memory.search import HybridSearch, recency_decay
from jarvis.models import Chunk, MemoryTier

if TYPE_CHECKING:
    from pathlib import Path


class TestRecencyDecay:
    def test_today(self):
        assert recency_decay(date.today()) == 1.0

    def test_none_returns_one(self):
        assert recency_decay(None) == 1.0

    def test_future_returns_one(self):
        future = date.today() + timedelta(days=5)
        assert recency_decay(future) == 1.0

    def test_old_decays(self):
        old = date.today() - timedelta(days=60)
        result = recency_decay(old, half_life_days=30)
        assert 0.0 < result < 0.5  # Should be decayed significantly

    def test_half_life(self):
        half_life = 30
        d = date.today() - timedelta(days=half_life)
        result = recency_decay(d, half_life_days=half_life)
        # At exactly half_life, decay should be ~0.368 (1/e)
        assert 0.3 < result < 0.4

    def test_datetime_input(self):
        dt = datetime.now() - timedelta(days=10)
        result = recency_decay(dt)
        assert 0.0 < result < 1.0

    def test_very_old(self):
        old = date.today() - timedelta(days=365)
        result = recency_decay(old, half_life_days=30)
        assert result < 0.01


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    idx = MemoryIndex(tmp_path / "test.db")
    _ = idx.conn
    return idx


@pytest.fixture
def embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@pytest.fixture
def search(index: MemoryIndex, embedding_client: EmbeddingClient) -> HybridSearch:
    return HybridSearch(index, embedding_client)


class TestBM25OnlySearch:
    def test_basic_search(self, search: HybridSearch, index: MemoryIndex):
        chunks = [
            Chunk(text="Projektplanung Recherche Zusammenfassung", source_path="a.md", content_hash="h1"),
            Chunk(text="Haftpflichtversicherung privat", source_path="b.md", content_hash="h2"),
        ]
        index.upsert_chunks(chunks)

        results = search.search_bm25_only("Projektplanung", top_k=5)
        assert len(results) >= 1
        assert results[0].chunk.text == chunks[0].text
        assert results[0].bm25_score > 0

    def test_empty_query(self, search: HybridSearch):
        results = search.search_bm25_only("")
        assert results == []

    def test_no_results(self, search: HybridSearch, index: MemoryIndex):
        index.upsert_chunks([Chunk(text="Hello", source_path="a.md", content_hash="h1")])
        results = search.search_bm25_only("xyznonexistent")
        assert results == []

    def test_recency_decay_applied(self, search: HybridSearch, index: MemoryIndex):
        old_chunk = Chunk(
            text="Altes Dokument versicherung",
            source_path="a.md",
            content_hash="h1",
            timestamp=datetime(2025, 1, 1),
        )
        new_chunk = Chunk(
            text="Neues Dokument versicherung",
            source_path="b.md",
            content_hash="h2",
            timestamp=datetime.now(),
        )
        index.upsert_chunks([old_chunk, new_chunk])

        results = search.search_bm25_only("versicherung")
        assert len(results) == 2
        # Results sorted by final score: newer should be first
        assert results[0].score >= results[1].score
        assert results[0].recency_factor > results[1].recency_factor

    def test_core_no_decay(self, search: HybridSearch, index: MemoryIndex):
        core_chunk = Chunk(
            text="Core Memory Identität",
            source_path="CORE.md",
            content_hash="h1",
            memory_tier=MemoryTier.CORE,
            timestamp=datetime(2020, 1, 1),  # Very old
        )
        index.upsert_chunks([core_chunk])

        results = search.search_bm25_only("Identität")
        assert len(results) == 1
        assert results[0].recency_factor == 1.0  # No decay for core
