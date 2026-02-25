"""Vector-Index Abstraktion mit FAISS ANN-Support.

Bietet O(log N) statt O(N×D) fuer Vector-Similarity-Search.

Design:
  - VectorIndex Protocol: add(), search(), rebuild(), remove()
  - BruteForceIndex: Pure Python Fallback (aktuelles Verhalten)
  - FAISSIndex: HNSW mit faiss-cpu, L2-normalisiert fuer Cosine-Similarity
  - create_vector_index(backend="auto"): Factory mit Auto-Detection

FAISS-Konfiguration:
  - Index: IndexHNSWFlat mit METRIC_INNER_PRODUCT
  - M=32, efSearch=64, efConstruction=200
  - Vektoren L2-normalisiert → Inner Product = Cosine Similarity
  - Inkrementelle Adds ohne Rebuild
  - Key→Position Mapping fuer Deduplikation

Architektur-Bibel: §4.7 (Vector Search)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("jarvis.memory.vector_index")


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol fuer Vector-Index Implementierungen."""

    def add(self, key: str, vector: list[float]) -> None:
        """Fuegt einen Vektor hinzu oder aktualisiert einen bestehenden."""
        ...

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Sucht die aehnlichsten Vektoren.

        Returns:
            Liste von (key, similarity_score) Tupeln, absteigend sortiert.
        """
        ...

    def remove(self, key: str) -> bool:
        """Entfernt einen Vektor. Returns True wenn gefunden."""
        ...

    def rebuild(self) -> None:
        """Baut den Index komplett neu auf (z.B. nach vielen Deletes)."""
        ...

    @property
    def size(self) -> int:
        """Anzahl der Vektoren im Index."""
        ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-Normalisierung eines Vektors."""
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


class BruteForceIndex:
    """Pure-Python Brute-Force Vector Index (Fallback).

    O(N×D) pro Suche. Funktioniert immer, keine Dependencies.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, key: str, vector: list[float]) -> None:
        self._vectors[key] = _l2_normalize(np.array(vector, dtype=np.float32))

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        if not self._vectors:
            return []

        query = _l2_normalize(np.array(query_vector, dtype=np.float32))
        scores: list[tuple[str, float]] = []

        for key, vec in self._vectors.items():
            sim = float(np.dot(query, vec))
            scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def remove(self, key: str) -> bool:
        return self._vectors.pop(key, None) is not None

    def rebuild(self) -> None:
        pass  # Nothing to rebuild

    @property
    def size(self) -> int:
        return len(self._vectors)


class FAISSIndex:
    """FAISS HNSW Vector Index fuer O(log N) Approximate Nearest Neighbor.

    Konfiguration:
      - IndexHNSWFlat mit METRIC_INNER_PRODUCT
      - M=32, efSearch=64, efConstruction=200
      - Vektoren L2-normalisiert → IP = Cosine Similarity
      - Inkrementelle Adds ohne Rebuild
      - Key→Position Mapping fuer Deduplikation
    """

    def __init__(
        self,
        dimension: int = 768,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        import faiss  # type: ignore

        self._dimension = dimension
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        # HNSW Index mit Inner Product (= Cosine nach L2-Norm)
        self._index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search

        # Key → Position Mapping
        self._key_to_pos: dict[str, int] = {}
        self._pos_to_key: dict[int, str] = {}
        self._next_pos: int = 0

        # Geloeschte Positionen (Tombstones)
        self._deleted: set[int] = set()

        # Backup fuer Rebuild
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, key: str, vector: list[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, got {len(vector)}"
            )
        vec = _l2_normalize(np.array(vector, dtype=np.float32))

        # Deduplikation: wenn Key existiert, als geloescht markieren
        if key in self._key_to_pos:
            old_pos = self._key_to_pos[key]
            self._deleted.add(old_pos)
            del self._pos_to_key[old_pos]

        # Neuen Vektor hinzufuegen
        pos = self._next_pos
        self._next_pos += 1

        self._key_to_pos[key] = pos
        self._pos_to_key[pos] = key
        self._vectors[key] = vec

        # Zu FAISS hinzufuegen (erwartet 2D-Array)
        self._index.add(vec.reshape(1, -1))

        # Auto-Rebuild wenn zu viele Tombstones (auch bei Deduplikation)
        if len(self._deleted) > len(self._key_to_pos) * 0.3 and len(self._key_to_pos) > 100:
            self.rebuild()

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []

        query = _l2_normalize(np.array(query_vector, dtype=np.float32))
        live_count = len(self._key_to_pos)

        # Mehr Ergebnisse holen um geloeschte rauszufiltern
        fetch_k = min(top_k + len(self._deleted) + 10, self._index.ntotal)

        results = self._search_inner(query, fetch_k, top_k)

        # Retry mit groesserem fetch_k wenn zu wenige Ergebnisse
        if len(results) < top_k and live_count >= top_k and fetch_k < self._index.ntotal:
            results = self._search_inner(query, self._index.ntotal, top_k)

        return results

    def _search_inner(
        self, query: "np.ndarray", fetch_k: int, top_k: int,
    ) -> list[tuple[str, float]]:
        distances, indices = self._index.search(query.reshape(1, -1), fetch_k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if idx in self._deleted:
                continue
            key = self._pos_to_key.get(idx)
            if key is None:
                continue
            results.append((key, float(dist)))
            if len(results) >= top_k:
                break

        return results

    def remove(self, key: str) -> bool:
        pos = self._key_to_pos.pop(key, None)
        if pos is None:
            return False
        self._deleted.add(pos)
        self._pos_to_key.pop(pos, None)
        self._vectors.pop(key, None)

        # Auto-Rebuild wenn zu viele Tombstones
        if len(self._deleted) > len(self._key_to_pos) * 0.3 and len(self._key_to_pos) > 100:
            self.rebuild()

        return True

    def rebuild(self) -> None:
        """Baut den FAISS-Index komplett neu auf."""
        import faiss  # type: ignore

        if not self._vectors:
            self._index = faiss.IndexHNSWFlat(
                self._dimension, self._m, faiss.METRIC_INNER_PRODUCT
            )
            self._index.hnsw.efConstruction = self._ef_construction
            self._index.hnsw.efSearch = self._ef_search
            self._key_to_pos.clear()
            self._pos_to_key.clear()
            self._deleted.clear()
            self._next_pos = 0
            return

        # Neuen Index erstellen
        new_index = faiss.IndexHNSWFlat(
            self._dimension, self._m, faiss.METRIC_INNER_PRODUCT
        )
        new_index.hnsw.efConstruction = self._ef_construction
        new_index.hnsw.efSearch = self._ef_search

        new_key_to_pos: dict[str, int] = {}
        new_pos_to_key: dict[int, str] = {}

        # Alle aktiven Vektoren einfuegen
        vectors_list = list(self._vectors.items())
        if vectors_list:
            matrix = np.vstack([v for _, v in vectors_list])
            new_index.add(matrix)
            for i, (key, _) in enumerate(vectors_list):
                new_key_to_pos[key] = i
                new_pos_to_key[i] = key

        self._index = new_index
        self._key_to_pos = new_key_to_pos
        self._pos_to_key = new_pos_to_key
        self._deleted.clear()
        self._next_pos = len(vectors_list)

        logger.info("FAISS-Index rebuilt: %d Vektoren", len(vectors_list))

    @property
    def size(self) -> int:
        return len(self._key_to_pos)


def create_vector_index(backend: str = "auto", dimension: int = 768) -> VectorIndex:
    """Factory: Erstellt den besten verfuegbaren Vector-Index.

    Args:
        backend: "auto", "faiss", oder "brute_force"
        dimension: Embedding-Dimension (default: 768 fuer nomic-embed-text)

    Returns:
        VectorIndex Instanz.
    """
    # Sicherstellen, dass dimension ein int ist (z.B. bei Mock-Config)
    try:
        dimension = int(dimension)
    except (TypeError, ValueError):
        dimension = 768
    if backend == "faiss" or backend == "auto":
        try:
            import faiss  # type: ignore  # noqa: F401

            result: VectorIndex = FAISSIndex(dimension=dimension)
            logger.info("FAISSIndex erstellt (HNSW, dim=%d)", dimension)
            return result
        except ImportError:
            if backend == "faiss":
                raise ImportError(
                    "faiss-cpu nicht installiert. "
                    "Installation: pip install faiss-cpu"
                ) from None
            logger.info("faiss-cpu nicht verfuegbar, verwende BruteForceIndex")

    result = BruteForceIndex()
    logger.info("BruteForceIndex erstellt (Fallback)")
    return result
