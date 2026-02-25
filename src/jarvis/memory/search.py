“”“3-Kanal Hybrid-Suche · BM25 + Vektor + Graph. [B§4.7]

Merge:
final_score = (
w_vector × vector_score +
w_bm25   × bm25_score +
w_graph  × graph_score
) × recency_decay

Defaults: w_vector=0.50, w_bm25=0.30, w_graph=0.20
“””

from **future** import annotations

import asyncio
import logging
import math
from collections import OrderedDict
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from jarvis.config import MemoryConfig
from jarvis.memory.embeddings import EmbeddingClient, cosine_similarity
from jarvis.memory.vector_index import VectorIndex, create_vector_index
from jarvis.models import MemorySearchResult, MemoryTier

if TYPE_CHECKING:
from jarvis.memory.indexer import MemoryIndex

logger = logging.getLogger(“jarvis.memory.search”)

def recency_decay(
source_date: date | datetime | None,
half_life_days: int = 30,
) -> float:
“”“Exponentieller Recency-Decay mit korrekter Halbwertszeit. [B§4.7]

```
Formel: decay = 2^(-age / half_life) = exp(-age * ln(2) / half_life)

Bei half_life_days=30 ergibt sich:
  - Nach  0 Tagen: 1.000
  - Nach 30 Tagen: 0.500 (exakte Halbierung)
  - Nach 60 Tagen: 0.250
  - Nach 90 Tagen: 0.125

CORE.md und Einträge ohne Datum bekommen immer 1.0.

Args:
    source_date: Datum des Eintrags.
    half_life_days: Halbwertszeit in Tagen.

Returns:
    Decay-Faktor zwischen 0.0 und 1.0.
"""
if source_date is None:
    return 1.0

if isinstance(source_date, datetime):
    source_date = source_date.date()

age_days = (date.today() - source_date).days
if age_days <= 0:
    return 1.0

# Korrekte Halbwertszeit: nach half_life_days ist der Wert exakt 0.5
return math.exp(-age_days * math.log(2) / half_life_days)
```

class HybridSearch:
“”“3-Kanal Hybrid-Suche über den Memory-Index.

```
Kanäle:
1. BM25 (FTS5) — lexikalische Treffer
2. Vektor (Cosine Similarity) — semantische Ähnlichkeit
3. Graph-Traversal — Beziehungsnähe

Ergebnisse werden per gewichtetem Score gemerged.
"""

# Max entries to keep in the graph search result cache
_GRAPH_CACHE_MAX_SIZE: int = 256

def __init__(
    self,
    index: MemoryIndex,
    embedding_client: EmbeddingClient,
    config: MemoryConfig | None = None,
    vector_index: VectorIndex | None = None,
    weight_optimizer: Any = None,
) -> None:
    """Initialisiert die Hybrid-Suche mit Index, Embeddings und Konfiguration."""
    self._index = index
    self._embeddings = embedding_client
    self._config = config or MemoryConfig()
    self._vector_index: VectorIndex = vector_index or create_vector_index(
        backend="auto", dimension=embedding_client.dimensions
    )
    self._weight_optimizer = weight_optimizer
    # Gecachtes Mapping content_hash → [chunk_ids] (wird lazy aufgebaut)
    self._chunk_hash_map: dict[str, list[str]] | None = None
    self._chunk_hash_map_version: int = 0
    # LRU cache for graph search results keyed by frozenset of query words
    self._graph_search_cache: OrderedDict[frozenset[str], dict[str, float]] = OrderedDict()

@property
def index(self) -> MemoryIndex:
    """Zugriff auf den Memory-Index."""
    return self._index

@property
def vector_index(self) -> VectorIndex:
    """Zugriff auf den Vector-Index."""
    return self._vector_index

def notify_embedding_added(self, key: str, vector: list[float]) -> None:
    """Benachrichtigt den Vector-Index ueber ein neues/aktualisiertes Embedding.

    Wird vom MemoryManager aufgerufen wenn ein Embedding gespeichert wird.

    Args:
        key: Content-Hash des Embeddings.
        vector: Der Embedding-Vektor.
    """
    self._vector_index.add(key, vector)
    # Chunk-Hash-Map und Graph-Cache invalidieren, da sich Zuordnungen geaendert haben koennten
    self._chunk_hash_map = None
    self._graph_search_cache.clear()

def invalidate_chunk_hash_map(self) -> None:
    """Invalidiert den gecachten Chunk-Hash-Map und den Graph-Search-Cache."""
    self._chunk_hash_map = None
    self._graph_search_cache.clear()

async def search(
    self,
    query: str,
    *,
    top_k: int | None = None,
    tier_filter: MemoryTier | None = None,
    enable_bm25: bool = True,
    enable_vector: bool = True,
    enable_graph: bool = True,
) -> list[MemorySearchResult]:
    """Führt eine Hybrid-Suche durch.

    Args:
        query: Suchtext.
        top_k: Max Ergebnisse (default aus Config).
        tier_filter: Optional nur bestimmten Tier durchsuchen.
        enable_bm25: BM25-Kanal aktivieren.
        enable_vector: Vektor-Kanal aktivieren.
        enable_graph: Graph-Kanal aktivieren.

    Returns:
        Sortierte Liste von MemorySearchResult (höchster Score zuerst).
    """
    if not query.strip():
        return []

    if top_k is None:
        top_k = self._config.search_top_k

    # Sammle Scores pro Chunk-ID aus allen Kanälen
    scores: dict[str, dict[str, float]] = {}  # chunk_id → {bm25, vector, graph}
    fetch_k = top_k * 3  # Mehr holen als nötig für besseres Merging

    # ── Kanal 1: BM25 ────────────────────────────────────────
    if enable_bm25 and self._config.weight_bm25 > 0:
        bm25_results = self._index.search_bm25(query, top_k=fetch_k)
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results) or 1.0
            for chunk_id, raw_score in bm25_results:
                scores.setdefault(chunk_id, {"bm25": 0, "vector": 0, "graph": 0})
                scores[chunk_id]["bm25"] = raw_score / max_bm25  # Normalisiert auf 0-1

    # ── Kanal 2: Vektor (via VectorIndex — O(log N) mit FAISS) ──
    if enable_vector and self._config.weight_vector > 0:
        try:
            query_emb = await self._embeddings.embed_text(query)

            # Gecachte Chunk-Hash-Map verwenden (via Executor um Event-Loop nicht zu blockieren)
            if self._chunk_hash_map is None:
                loop = asyncio.get_running_loop()
                self._chunk_hash_map = await loop.run_in_executor(
                    None, self._build_chunk_hash_map
                )

            # VectorIndex fuer ANN-Suche nutzen
            if self._vector_index.size > 0:
                ann_results = self._vector_index.search(
                    query_emb.vector, top_k=fetch_k
                )
                for content_hash, sim in ann_results:
                    chunk_ids = self._chunk_hash_map.get(content_hash, [])
                    for cid in chunk_ids:
                        scores.setdefault(
                            cid, {"bm25": 0, "vector": 0, "graph": 0}
                        )
                        scores[cid]["vector"] = max(
                            scores[cid]["vector"], max(0.0, sim)
                        )
            else:
                # Fallback: Brute-Force ueber alle Embeddings aus DB
                all_embeddings = self._index.get_all_embeddings()
                vector_scores: list[tuple[str, float]] = []
                for content_hash, stored_vec in all_embeddings.items():
                    sim = cosine_similarity(query_emb.vector, stored_vec)
                    chunk_ids = self._chunk_hash_map.get(content_hash, [])
                    for cid in chunk_ids:
                        vector_scores.append((cid, sim))

                vector_scores.sort(key=lambda x: x[1], reverse=True)
                for chunk_id, sim in vector_scores[:fetch_k]:
                    scores.setdefault(
                        chunk_id, {"bm25": 0, "vector": 0, "graph": 0}
                    )
                    scores[chunk_id]["vector"] = max(0.0, sim)

        except Exception as e:
            logger.warning("Vektor-Suche fehlgeschlagen: %s", e)

    # ── Kanal 3: Graph ───────────────────────────────────────
    if enable_graph and self._config.weight_graph > 0:
        graph_chunk_scores = self._graph_search(query)
        for chunk_id, g_score in graph_chunk_scores.items():
            scores.setdefault(chunk_id, {"bm25": 0, "vector": 0, "graph": 0})
            scores[chunk_id]["graph"] = g_score

    # ── Merge ────────────────────────────────────────────────
    # Gewichtung der Suchkanäle:
    # 1. Statische Config-Werte (Baseline)
    # 2. Dynamische Gewichtung nach Query-Länge (Heuristik-Baseline)
    # 3. Performance-basierte Optimizer-Gewichte (überschreiben Heuristik)
    w_vector = self._config.weight_vector
    w_bm25 = self._config.weight_bm25
    w_graph = self._config.weight_graph

    # Dynamische Gewichtung nach Query-Länge (Heuristik-Baseline)
    if getattr(self._config, "dynamic_weighting", False):
        try:
            num_tokens = len(query.split())
            if num_tokens <= 2:
                w_bm25, w_graph, w_vector = 0.4, 0.4, 0.2
            elif num_tokens <= 5:
                w_bm25, w_graph, w_vector = 0.33, 0.33, 0.34
            elif num_tokens <= 20:
                w_bm25, w_graph, w_vector = 0.25, 0.25, 0.5
            else:
                w_bm25, w_graph, w_vector = 0.2, 0.2, 0.6
        except Exception as e:
            logger.debug("Dynamische Gewichtung fehlgeschlagen: %s", e)

    # Performance-basierte Optimizer-Gewichte (überschreiben Heuristik)
    if self._weight_optimizer is not None:
        try:
            opt_weights = self._weight_optimizer.get_optimized_weights()
            if isinstance(opt_weights, dict):
                w_vector = opt_weights.get("vector", w_vector)
                w_bm25 = opt_weights.get("bm25", w_bm25)
                w_graph = opt_weights.get("graph", w_graph)
            else:
                w_vector, w_bm25, w_graph = opt_weights
        except Exception:
            pass  # Fallback auf vorherige Gewichte

    # Batch-Fetch aller Chunks (1 Query statt N+1)
    all_chunk_ids = list(scores.keys())
    chunks_by_id = self._index.get_chunks_by_ids(all_chunk_ids)

    results: list[MemorySearchResult] = []
    for chunk_id, channel_scores in scores.items():
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        # Tier-Filter
        if tier_filter and chunk.memory_tier != tier_filter:
            continue

        # Core Memory bekommt keinen Decay
        if chunk.memory_tier == MemoryTier.CORE:
            decay = 1.0
        else:
            decay = recency_decay(
                chunk.timestamp,
                self._config.recency_half_life_days,
            )

        bm25_s = channel_scores["bm25"]
        vector_s = channel_scores["vector"]
        graph_s = channel_scores["graph"]

        final_score = (
            w_vector * vector_s
            + w_bm25 * bm25_s
            + w_graph * graph_s
        ) * decay

        results.append(
            MemorySearchResult(
                chunk=chunk,
                score=final_score,
                bm25_score=bm25_s,
                vector_score=vector_s,
                graph_score=graph_s,
                recency_factor=decay,
            )
        )

    # Sortieren nach Score (absteigend)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]

def search_bm25_only(self, query: str, top_k: int = 10) -> list[MemorySearchResult]:
    """Synchrone BM25-only Suche (kein Embedding nötig).

    Nützlich für schnelle lexikalische Lookups.
    """
    bm25_results = self._index.search_bm25(query, top_k=top_k)
    if not bm25_results:
        return []

    max_score = max(s for _, s in bm25_results) or 1.0
    results: list[MemorySearchResult] = []

    # Batch-Fetch aller Chunks (1 Query statt N+1)
    bm25_chunk_ids = [cid for cid, _ in bm25_results]
    chunks_by_id = self._index.get_chunks_by_ids(bm25_chunk_ids)

    for chunk_id, raw_score in bm25_results:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        norm_score = raw_score / max_score
        decay = (
            1.0
            if chunk.memory_tier == MemoryTier.CORE
            else recency_decay(chunk.timestamp, self._config.recency_half_life_days)
        )

        results.append(
            MemorySearchResult(
                chunk=chunk,
                score=norm_score * decay,
                bm25_score=norm_score,
                vector_score=0.0,
                graph_score=0.0,
                recency_factor=decay,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results

def _build_chunk_hash_map(self) -> dict[str, list[str]]:
    """Baut ein Mapping content_hash → [chunk_ids].

    Wird für Vektor-Suche gebraucht um von Embedding
    zurück zum Chunk zu kommen.

    Wird lazy gecacht und via run_in_executor aufgerufen,
    um den Event-Loop nicht zu blockieren.
    """
    rows = self._index.conn.execute("SELECT id, content_hash FROM chunks").fetchall()
    mapping: dict[str, list[str]] = {}
    for r in rows:
        mapping.setdefault(r["content_hash"], []).append(r["id"])
    return mapping

def _graph_search(self, query: str) -> dict[str, float]:
    """Graph-basierte Suche: Findet Entitäten die zum Query passen,
    dann Chunks die mit diesen Entitäten verknüpft sind.

    Uses SQL-level filtering via MemoryIndex.get_chunks_with_entity_overlap()
    instead of loading all chunks and parsing JSON in Python (eliminates
    full table scan).  Results are cached per query-word set.

    Returns:
        {chunk_id: graph_score} Dict.
    """
    # Cache key: frozenset of lowered query words
    words = query.lower().split()
    cache_key = frozenset(words)

    if cache_key in self._graph_search_cache:
        self._graph_search_cache.move_to_end(cache_key)
        return self._graph_search_cache[cache_key]

    # Schritt 1: Entitäten finden die zum Query passen (SQL LIKE pro Wort)
    matching_entity_ids: set[str] = set()

    for word in words:
        for entity in self._index.search_entities(name=word):
            matching_entity_ids.add(entity.id)

    if not matching_entity_ids:
        self._graph_search_cache[cache_key] = {}
        return {}

    # Schritt 2: Verwandte Entitäten finden (1-Hop)
    related_ids: set[str] = set(matching_entity_ids)
    for eid in matching_entity_ids:
        neighbors = self._index.graph_traverse(eid, max_depth=1)
        for n in neighbors:
            related_ids.add(n.id)

    # Schritt 3: Chunks finden die diese Entitäten referenzieren
    #   Uses SQL LIKE filtering instead of a full table scan.
    chunk_scores: dict[str, float] = {}
    chunk_rows = self._index.get_chunks_with_entity_overlap(related_ids)

    for chunk_id, chunk_entities in chunk_rows:
        chunk_entity_set = set(chunk_entities)
        # Score basierend auf Overlap
        overlap = len(chunk_entity_set & related_ids)
        if overlap > 0:
            # Direkte Treffer zählen mehr als indirekte
            direct = len(chunk_entity_set & matching_entity_ids)
            indirect = overlap - direct
            score = (direct * 1.0 + indirect * 0.5) / max(len(related_ids), 1)
            chunk_scores[chunk_id] = min(score, 1.0)

    # Store in cache, evict oldest if full
    self._graph_search_cache[cache_key] = chunk_scores
    while len(self._graph_search_cache) > self._GRAPH_CACHE_MAX_SIZE:
        self._graph_search_cache.popitem(last=False)

    return chunk_scores
```