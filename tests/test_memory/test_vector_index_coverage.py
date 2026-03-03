"""Coverage-Tests fuer vector_index.py -- fehlende Pfade abdecken.

Schwerpunkt: BruteForceIndex (add, search, remove, rebuild, size),
FAISSIndex (mock-basiert), create_vector_index factory,
_l2_normalize_np, Tombstone-basiertes Delete, Auto-Rebuild.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.memory.vector_index import (
    BruteForceIndex,
    _l2_normalize_np,
    create_vector_index,
)


# ============================================================================
# _l2_normalize_np
# ============================================================================


class TestL2Normalize:
    def test_normal_vector(self) -> None:
        vec = np.array([3.0, 4.0], dtype=np.float32)
        result = _l2_normalize_np(vec)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_zero_vector(self) -> None:
        vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = _l2_normalize_np(vec)
        assert np.allclose(result, vec)

    def test_already_normalized(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _l2_normalize_np(vec)
        assert np.allclose(result, vec)


# ============================================================================
# BruteForceIndex
# ============================================================================


class TestBruteForceIndex:
    def test_empty_search(self) -> None:
        idx = BruteForceIndex()
        results = idx.search([1.0, 0.0, 0.0])
        assert results == []

    def test_add_and_search(self) -> None:
        idx = BruteForceIndex()
        idx.add("a", [1.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0])
        idx.add("c", [0.9, 0.1, 0.0])

        results = idx.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # "a" should be the most similar
        assert results[0][0] == "a"
        # Score should be close to 1.0
        assert results[0][1] > 0.9

    def test_add_replaces_existing(self) -> None:
        idx = BruteForceIndex()
        idx.add("a", [1.0, 0.0, 0.0])
        idx.add("a", [0.0, 1.0, 0.0])  # replace
        assert idx.size == 1
        results = idx.search([0.0, 1.0, 0.0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] > 0.9

    def test_remove_existing(self) -> None:
        idx = BruteForceIndex()
        idx.add("a", [1.0, 0.0])
        assert idx.remove("a") is True
        assert idx.size == 0

    def test_remove_nonexistent(self) -> None:
        idx = BruteForceIndex()
        assert idx.remove("nonexistent") is False

    def test_rebuild_noop(self) -> None:
        idx = BruteForceIndex()
        idx.add("a", [1.0, 0.0])
        idx.rebuild()  # should be no-op
        assert idx.size == 1

    def test_size(self) -> None:
        idx = BruteForceIndex()
        assert idx.size == 0
        idx.add("a", [1.0])
        assert idx.size == 1
        idx.add("b", [0.0])
        assert idx.size == 2

    def test_top_k_larger_than_index(self) -> None:
        idx = BruteForceIndex()
        idx.add("a", [1.0, 0.0])
        results = idx.search([1.0, 0.0], top_k=100)
        assert len(results) == 1


# ============================================================================
# FAISSIndex (mocked)
# ============================================================================


class TestFAISSIndex:
    def _make_mock_faiss(self) -> MagicMock:
        """Creates a mock faiss module."""
        mock_faiss = MagicMock()
        mock_faiss.METRIC_INNER_PRODUCT = 0

        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.hnsw = MagicMock()

        def mock_add(vectors):
            mock_index.ntotal += len(vectors)

        mock_index.add = mock_add

        def mock_search(query, k):
            # Return empty results
            return (
                np.array([[-1.0] * k], dtype=np.float32),
                np.array([[-1] * k], dtype=np.int64),
            )

        mock_index.search = mock_search

        mock_faiss.IndexHNSWFlat.return_value = mock_index
        return mock_faiss

    def test_faiss_init(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=128, m=16, ef_construction=100, ef_search=32)
            assert idx.size == 0

    def test_faiss_add(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx.add("a", [1.0, 0.0, 0.0])
            assert idx.size == 1

    def test_faiss_add_wrong_dimension(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                idx.add("a", [1.0, 0.0])

    def test_faiss_add_replaces_existing(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx.add("a", [1.0, 0.0, 0.0])
            idx.add("a", [0.0, 1.0, 0.0])  # replace
            assert idx.size == 1
            assert 0 in idx._deleted  # old position marked as deleted

    def test_faiss_remove(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx.add("a", [1.0, 0.0, 0.0])
            assert idx.remove("a") is True
            assert idx.size == 0
            assert idx.remove("a") is False

    def test_faiss_search_empty(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            results = idx.search([1.0, 0.0, 0.0])
            assert results == []

    def test_faiss_search_with_results(self) -> None:
        mock_faiss = self._make_mock_faiss()

        # Configure the mock index to return results
        mock_index = mock_faiss.IndexHNSWFlat.return_value
        mock_index.ntotal = 2

        def mock_search(query, k):
            return (
                np.array([[0.95, 0.8]], dtype=np.float32),
                np.array([[0, 1]], dtype=np.int64),
            )

        mock_index.search = mock_search

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            # Manually set up internal state
            idx._key_to_pos = {"a": 0, "b": 1}
            idx._pos_to_key = {0: "a", 1: "b"}
            idx._vectors = {
                "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
            idx._index = mock_index
            idx._next_pos = 2

            results = idx.search([1.0, 0.0, 0.0], top_k=2)
            assert len(results) == 2
            assert results[0][0] == "a"
            assert results[0][1] == pytest.approx(0.95)

    def test_faiss_search_with_deleted(self) -> None:
        mock_faiss = self._make_mock_faiss()

        mock_index = mock_faiss.IndexHNSWFlat.return_value
        mock_index.ntotal = 3

        def mock_search(query, k):
            return (
                np.array([[0.95, 0.9, 0.8]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int64),
            )

        mock_index.search = mock_search

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx._key_to_pos = {"a": 0, "c": 2}
            idx._pos_to_key = {0: "a", 2: "c"}
            idx._deleted = {1}  # position 1 is deleted
            idx._vectors = {
                "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "c": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
            idx._index = mock_index
            idx._next_pos = 3

            results = idx.search([1.0, 0.0, 0.0], top_k=2)
            # Position 1 should be skipped (deleted)
            assert len(results) == 2
            keys = [r[0] for r in results]
            assert "a" in keys
            assert "c" in keys

    def test_faiss_rebuild_empty(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx.rebuild()
            assert idx.size == 0

    def test_faiss_rebuild_with_vectors(self) -> None:
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            idx._vectors = {
                "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
            idx._deleted = {0}
            idx._key_to_pos = {"a": 0, "b": 1}
            idx._pos_to_key = {0: "a", 1: "b"}

            idx.rebuild()
            assert idx.size == 2
            assert len(idx._deleted) == 0

    def test_faiss_auto_rebuild_on_add(self) -> None:
        """Auto-rebuild when too many tombstones and enough keys."""
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            # Simulate many keys and tombstones
            for i in range(150):
                key = f"key_{i}"
                idx._key_to_pos[key] = i
                idx._pos_to_key[i] = key
                idx._vectors[key] = np.array([float(i), 0.0, 0.0], dtype=np.float32)
            idx._next_pos = 150
            # Add >30% tombstones
            for i in range(60):
                idx._deleted.add(i + 200)

            with patch.object(idx, "rebuild") as mock_rebuild:
                idx.add("new_key", [1.0, 0.0, 0.0])
                mock_rebuild.assert_called_once()

    def test_faiss_auto_rebuild_on_remove(self) -> None:
        """Auto-rebuild on remove when too many tombstones."""
        mock_faiss = self._make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = FAISSIndex(dimension=3)
            # Simulate many keys and tombstones (>100 keys needed)
            for i in range(150):
                key = f"key_{i}"
                idx._key_to_pos[key] = i
                idx._pos_to_key[i] = key
                idx._vectors[key] = np.array([float(i), 0.0, 0.0], dtype=np.float32)
            idx._next_pos = 150
            # Already close to threshold
            for i in range(45):
                idx._deleted.add(i + 200)

            with patch.object(idx, "rebuild") as mock_rebuild:
                idx.remove("key_0")
                mock_rebuild.assert_called_once()


# ============================================================================
# create_vector_index factory
# ============================================================================


class TestCreateVectorIndex:
    def test_auto_with_faiss_available(self) -> None:
        mock_faiss = MagicMock()
        mock_faiss.METRIC_INNER_PRODUCT = 0
        mock_index = MagicMock()
        mock_index.hnsw = MagicMock()
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex
            idx = create_vector_index(backend="auto", dimension=128)
            assert isinstance(idx, FAISSIndex)

    def test_auto_without_faiss(self) -> None:
        with patch.dict("sys.modules", {"faiss": None}):
            idx = create_vector_index(backend="auto", dimension=128)
            assert isinstance(idx, BruteForceIndex)

    def test_brute_force_explicit(self) -> None:
        idx = create_vector_index(backend="brute_force", dimension=128)
        assert isinstance(idx, BruteForceIndex)

    def test_faiss_explicit_not_installed(self) -> None:
        with patch.dict("sys.modules", {"faiss": None}):
            with pytest.raises(ImportError, match="faiss-cpu"):
                create_vector_index(backend="faiss", dimension=128)

    def test_invalid_dimension_type(self) -> None:
        idx = create_vector_index(backend="brute_force", dimension="not_a_number")
        assert isinstance(idx, BruteForceIndex)
        # Should fallback to 768

    def test_none_dimension(self) -> None:
        idx = create_vector_index(backend="brute_force", dimension=None)
        assert isinstance(idx, BruteForceIndex)


# ============================================================================
# VectorIndex Protocol
# ============================================================================


class TestVectorIndexProtocol:
    def test_brute_force_is_vector_index(self) -> None:
        from jarvis.memory.vector_index import VectorIndex
        idx = BruteForceIndex()
        assert isinstance(idx, VectorIndex)

    def test_faiss_is_vector_index(self) -> None:
        mock_faiss = MagicMock()
        mock_faiss.METRIC_INNER_PRODUCT = 0
        mock_index = MagicMock()
        mock_index.hnsw = MagicMock()
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from jarvis.memory.vector_index import FAISSIndex, VectorIndex
            idx = FAISSIndex(dimension=3)
            assert isinstance(idx, VectorIndex)
