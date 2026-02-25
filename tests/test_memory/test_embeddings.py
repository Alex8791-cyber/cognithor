"""Tests für memory/embeddings.py · Embedding Client."""

from __future__ import annotations

import pytest

from jarvis.memory.embeddings import EmbeddingClient, EmbeddingResult, cosine_similarity


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_similar(self):
        sim = cosine_similarity([1, 1, 0], [1, 0.9, 0.1])
        assert sim > 0.9

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_empty(self):
        assert cosine_similarity([], []) == 0.0


class TestEmbeddingClient:
    def test_cache_operations(self):
        client = EmbeddingClient()
        count = client.load_cache({"h1": [1.0, 2.0], "h2": [3.0, 4.0]})
        assert count == 2
        assert client.get_cached("h1") == [1.0, 2.0]
        assert client.get_cached("h2") == [3.0, 4.0]
        assert client.get_cached("h3") is None

    def test_model_property(self):
        client = EmbeddingClient(model="test-model")
        assert client.model == "test-model"

    def test_dimensions_property(self):
        client = EmbeddingClient(dimensions=384)
        assert client.dimensions == 384

    def test_stats_initial(self):
        client = EmbeddingClient()
        assert client.stats.total_requests == 0
        assert client.stats.cache_hits == 0
        assert client.stats.cache_hit_rate == 0.0


class TestEmbeddingResult:
    def test_creation(self):
        result = EmbeddingResult(
            vector=[1.0, 2.0, 3.0],
            model="test",
            dimensions=3,
            cached=True,
        )
        assert len(result.vector) == 3
        assert result.model == "test"
        assert result.cached is True
