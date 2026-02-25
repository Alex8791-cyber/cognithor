"""Memory Manager · Central API for the 5-tier memory system. [B§4]

Orchestrates all memory tiers:
- Tier 1: Core Memory (CORE.md)
- Tier 2: Episodic Memory (daily log)
- Tier 3: Semantic Memory (knowledge graph)
- Tier 4: Procedural Memory (skills)
- Tier 5: Working Memory (session context)

Provides a unified API for search, indexing, and lifecycle management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anyio

from jarvis.config import JarvisConfig
from jarvis.memory.chunker import chunk_file, chunk_text
from jarvis.memory.core_memory import CoreMemory
from jarvis.memory.embeddings import EmbeddingClient
from jarvis.memory.enhanced_retrieval import (
    EnhancedSearchPipeline,
    EpisodicCompressor,
    FrequencyTracker,
    QueryDecomposer,
)
from jarvis.memory.episodic import EpisodicMemory
from jarvis.memory.graph_ranking import GraphRanking
from jarvis.memory.indexer import MemoryIndex
from jarvis.memory.multimodal import MultimodalMemory
from jarvis.memory.procedural import ProceduralMemory
from jarvis.memory.search import HybridSearch
from jarvis.memory.semantic import SemanticMemory
from jarvis.memory.vector_index import VectorIndex, create_vector_index
from jarvis.memory.working import WorkingMemoryManager
from jarvis.models import MemorySearchResult, MemoryTier

if TYPE_CHECKING:
    from pathlib import Path
    from jarvis.memory.episodic_store import EpisodicStore
    from jarvis.memory.weight_optimizer import SearchWeightOptimizer

logger = logging.getLogger("jarvis.memory.manager")


class MemoryManager:
    """Central interface to the memory system.

    Initializes and coordinates all 5 tiers.
    """

    def __init__(self, config: JarvisConfig | None = None, audit_logger: Any | None = None) -> None:
        """Initialisiert den MemoryManager und alle 5 Tiers."""
        self._config = config or JarvisConfig()
        self._mc = self._config.memory
        self._audit_logger = audit_logger

        # Tier 1: Core Memory
        self._core = CoreMemory(self._config.core_memory_path)

        # Tier 2: Episodic Memory
        self._episodic = EpisodicMemory(self._config.episodes_dir)

        # Index (SQLite)
        self._index = MemoryIndex(self._config.db_path)

        # Tier 3: Semantic Memory
        self._semantic = SemanticMemory(self._config.knowledge_dir, self._index)

        # Tier 4: Procedural Memory
        self._procedural = ProceduralMemory(self._config.procedures_dir)

        # Embedding Client
        self._embeddings = EmbeddingClient(
            model=self._config.models.embedding.name,
            base_url=self._config.ollama.base_url,
            dimensions=self._config.models.embedding.embedding_dimensions,
        )

        # Episodic Store (SQLite-basiert, optional)
        self._episodic_store: EpisodicStore | None = None
        try:
            from jarvis.memory.episodic_store import EpisodicStore as _ES
            db_path = str(self._config.db_path.with_name("memory_episodic.db"))
            self._episodic_store = _ES(db_path)
        except Exception:
            logger.debug("EpisodicStore init skipped")

        # Search Weight Optimizer (EMA-basiert, optional) -- VOR HybridSearch
        self._weight_optimizer: SearchWeightOptimizer | None = None
        try:
            from jarvis.memory.weight_optimizer import SearchWeightOptimizer as _SWO
            opt_db = str(self._config.db_path.with_name("memory_weights.db"))
            self._weight_optimizer = _SWO(opt_db)
        except Exception:
            logger.debug("SearchWeightOptimizer init skipped")

        # Vector Index (FAISS HNSW oder BruteForce Fallback)
        self._vector_index: VectorIndex = create_vector_index(
            backend="auto", dimension=self._embeddings.dimensions
        )

        # Hybrid Search (mit Vector-Index + Weight-Optimizer)
        self._search = HybridSearch(
            self._index, self._embeddings, self._mc,
            vector_index=self._vector_index,
            weight_optimizer=self._weight_optimizer,
        )

        # Graph Ranking (PageRank + Staleness)
        self._graph_ranking = GraphRanking(
            self._index,
            staleness_half_life_days=90,
        )

        # Enhanced Search Pipeline (Query-Decomposition + RRF + CRAG + Frequency)
        self._frequency_tracker = FrequencyTracker(frequency_weight=0.1)
        self._enhanced_search = EnhancedSearchPipeline(
            self._search,
            decomposer=QueryDecomposer(),
            frequency_tracker=self._frequency_tracker,
        )

        # Multimodal Memory
        self._multimodal = MultimodalMemory(
            memory_manager=self,
            media_pipeline=None,  # Wird via set_media_pipeline() gesetzt
        )

        # Episodic Compressor
        self._compressor = EpisodicCompressor(
            retention_days=self._mc.recency_half_life_days,
        )

        # Tier 5: Working Memory
        self._working = WorkingMemoryManager(
            config=self._mc,
            max_tokens=self._mc.compaction_keep_last_n * 4000,  # Grobe Schätzung
        )

        self._initialized = False

    # ── Properties ───────────────────────────────────────────────

    @property
    def core(self) -> CoreMemory:
        """Zugriff auf Tier 1: Core Memory."""
        return self._core

    @property
    def episodic(self) -> EpisodicMemory:
        """Zugriff auf Tier 2: Episodic Memory."""
        return self._episodic

    @property
    def semantic(self) -> SemanticMemory:
        """Zugriff auf Tier 3: Semantic Memory."""
        return self._semantic

    @property
    def procedural(self) -> ProceduralMemory:
        """Zugriff auf Tier 4: Procedural Memory."""
        return self._procedural

    @property
    def working(self) -> WorkingMemoryManager:
        """Zugriff auf Tier 5: Working Memory."""
        return self._working

    @property
    def index(self) -> MemoryIndex:
        """Zugriff auf den SQLite Memory-Index."""
        return self._index

    @property
    def search(self) -> HybridSearch:
        """Zugriff auf die Hybrid-Suche."""
        return self._search

    @property
    def enhanced_search(self) -> EnhancedSearchPipeline:
        """Zugriff auf die Enhanced Search Pipeline (Query-Decomp + RRF + CRAG)."""
        return self._enhanced_search

    @property
    def graph_ranking(self) -> GraphRanking:
        """Zugriff auf Graph-Ranking (PageRank + Staleness)."""
        return self._graph_ranking

    @property
    def multimodal(self) -> MultimodalMemory:
        """Zugriff auf Multimodal Memory (Bilder/Audio/Dokumente)."""
        return self._multimodal

    @property
    def compressor(self) -> EpisodicCompressor:
        """Zugriff auf den Episodic Compressor."""
        return self._compressor

    @property
    def frequency_tracker(self) -> FrequencyTracker:
        """Zugriff auf den Frequency Tracker."""
        return self._frequency_tracker

    @property
    def episodic_store(self) -> EpisodicStore | None:
        """Zugriff auf den SQLite-basierten Episodic Store (optional)."""
        return self._episodic_store

    @property
    def weight_optimizer(self) -> SearchWeightOptimizer | None:
        """Zugriff auf den Search Weight Optimizer (optional)."""
        return self._weight_optimizer

    @property
    def vector_index(self) -> VectorIndex:
        """Zugriff auf den Vector-Index (FAISS oder BruteForce)."""
        return self._vector_index

    @property
    def embeddings(self) -> EmbeddingClient:
        """Zugriff auf den Embedding-Client."""
        return self._embeddings

    # ── Initialization ───────────────────────────────────────────

    def _initialize_sync(self) -> dict[str, Any]:
        """Synchronous heavy initialization (blocking I/O).

        Creates directories, loads core memory, initializes the index,
        and populates the embedding cache and vector index.

        Returns:
            Status-Dict with info about the current state.
        """
        # Verzeichnisse erstellen
        self._config.memory_dir.mkdir(parents=True, exist_ok=True)
        self._config.index_dir.mkdir(parents=True, exist_ok=True)
        self._episodic.ensure_directory()
        self._semantic.ensure_directory()
        self._procedural.ensure_directory()
        self._config.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Core Memory laden (oder Default erstellen)
        core_text = self._core.load()
        if not core_text:
            core_text = self._core.create_default()
            logger.info("Default CORE.md erstellt")

        # Working Memory mit Core Memory initialisieren
        self._working.set_core_memory(core_text)

        # Episodic Memory aufräumen (alte Episoden löschen)
        try:
            retention_days = getattr(self._mc, "episodic_retention_days", None)
            if isinstance(retention_days, int) and retention_days > 0:
                deleted = self._episodic.prune_old(retention_days)
                if deleted:
                    logger.info(
                        "episodic_pruned",
                        deleted=deleted,
                        retention_days=retention_days,
                    )
        except Exception as exc:
            logger.warning("episodic_prune_failed", error=str(exc))

        # Index-Schema sicherstellen (passiert lazy beim ersten Zugriff)
        _ = self._index.conn

        # Embedding-Cache aus DB laden
        cached_embeddings = self._index.get_all_embeddings()
        if cached_embeddings:
            self._embeddings.load_cache(cached_embeddings)
            logger.info("Embedding-Cache geladen: %d Einträge", len(cached_embeddings))

            # Vector-Index mit bestehenden Embeddings populieren
            for content_hash, vector in cached_embeddings.items():
                self._vector_index.add(content_hash, vector)
            logger.info(
                "Vector-Index populiert: %d Vektoren (Backend: %s)",
                self._vector_index.size,
                type(self._vector_index).__name__,
            )

        self._initialized = True

        stats = self.stats()
        logger.info(
            "Memory-System initialisiert: %d Chunks, %d Entities, %d Procedures",
            stats["chunks"],
            stats["entities"],
            stats["procedures"],
        )
        return stats

    async def initialize(self) -> dict[str, Any]:
        """Initialisiert das gesamte Memory-System (async).

        Delegates blocking I/O (mkdir, file reads, DB operations) to a
        worker thread via anyio.to_thread.run_sync so the event loop
        is not blocked.

        Returns:
            Status-Dict mit Infos über den Zustand.
        """
        return await anyio.to_thread.run_sync(self._initialize_sync)

    def initialize_sync(self) -> dict[str, Any]:
        """Synchronous initialization entry point.

        Convenience wrapper around _initialize_sync() for callers that
        are not running inside an async event loop (e.g. tests, CLI tools).

        Returns:
            Status-Dict mit Infos über den Zustand.
        """
        return self._initialize_sync()

    # ── Unified Search ───────────────────────────────────────────

    async def search_memory(
        self,
        query: str,
        *,
        top_k: int | None = None,
        tier: MemoryTier | None = None,
        enhanced: bool = True,
    ) -> list[MemorySearchResult]:
        """Durchsucht das gesamte Memory-System.

        Args:
            query: Suchtext.
            top_k: Max Ergebnisse.
            tier: Optional nur bestimmten Tier durchsuchen.
            enhanced: True=Enhanced Pipeline (RRF+CRAG), False=nur HybridSearch.

        Returns:
            Sortierte Suchergebnisse.
        """
        k = top_k or self._mc.search_top_k

        if enhanced:
            results = await self._enhanced_search.search(query, top_k=k, tier_filter=tier)
        else:
            results = await self._search.search(query, top_k=k, tier_filter=tier)

        # Graph-Ranking Boost (wenn PageRank berechnet wurde)
        if self._graph_ranking.ranks:
            results = self._graph_ranking.boost_graph_scores(results)
            # Re-sort und limit nach boost
            results = sorted(results, key=lambda r: r.score, reverse=True)[:k]

        # Audit: Memory-Suche protokollieren
        if self._audit_logger:
            self._audit_logger.log_memory_op(
                f"search: {query[:80]}",
                result=f"{len(results)} Ergebnisse",
            )

        return results

    def search_memory_sync(
        self,
        query: str,
        *,
        top_k: int = 6,
    ) -> list[MemorySearchResult]:
        """Synchrone BM25-only Suche (kein Embedding nötig).

        Schneller Fallback wenn kein Embedding-Server verfügbar.
        """
        return self._search.search_bm25_only(query, top_k=top_k)

    # ── Indexing ─────────────────────────────────────────────────

    def index_file(self, file_path: str | Path, tier: MemoryTier | None = None) -> int:
        """Indexiert eine Markdown-Datei.

        1. Chunking
        2. Chunks in DB speichern
        3. FTS5 automatisch aktualisiert

        Args:
            file_path: Pfad zur Datei.
            tier: Expliziter Memory-Tier.

        Returns:
            Anzahl indexierter Chunks.
        """
        path_str = str(file_path)

        # Alte Chunks für diese Datei entfernen
        self._index.delete_chunks_by_source(path_str)

        # Neu chunken
        chunks = chunk_file(path_str, config=self._mc, tier=tier)
        if not chunks:
            return 0

        # In DB speichern
        count = self._index.upsert_chunks(chunks)
        logger.debug("Indexiert: %s → %d Chunks", path_str, count)
        return count

    def index_text(
        self,
        text: str,
        source_path: str,
        tier: MemoryTier | None = None,
    ) -> int:
        """Indexiert einen Text direkt (ohne Datei).

        Args:
            text: Der zu indexierende Text.
            source_path: Virtueller Quellpfad.
            tier: Memory-Tier.

        Returns:
            Anzahl indexierter Chunks.
        """
        self._index.delete_chunks_by_source(source_path)

        chunks = chunk_text(
            text,
            source_path,
            chunk_size_tokens=self._mc.chunk_size_tokens,
            chunk_overlap_tokens=self._mc.chunk_overlap_tokens,
            tier=tier,
        )
        if not chunks:
            return 0

        return self._index.upsert_chunks(chunks)

    async def index_with_embeddings(
        self,
        file_path: str | Path,
        tier: MemoryTier | None = None,
    ) -> int:
        """Indexiert eine Datei mit Embeddings.

        1. Chunking
        2. Chunks in DB
        3. Embeddings generieren und speichern

        Returns:
            Anzahl indexierter Chunks.
        """
        path_str = str(file_path)
        self._index.delete_chunks_by_source(path_str)

        chunks = chunk_file(path_str, config=self._mc, tier=tier)
        if not chunks:
            return 0

        # Chunks speichern
        self._index.upsert_chunks(chunks)

        # Embeddings generieren
        texts = [c.text for c in chunks]
        hashes = [c.content_hash for c in chunks]

        results = await self._embeddings.embed_batch(texts, hashes)
        if len(results) != len(chunks):
            logger.warning(
                "Embedding-Mismatch: %d Chunks, %d Ergebnisse -- überschüssige ignoriert",
                len(chunks),
                len(results),
            )
        for chunk, emb_result in zip(chunks, results, strict=False):
            if emb_result is None:
                logger.debug("Embedding fehlgeschlagen fuer chunk %s -- uebersprungen", chunk.content_hash[:8])
                continue
            if not emb_result.cached:
                self._index.store_embedding(
                    chunk.content_hash,
                    emb_result.vector,
                    emb_result.model,
                )
            # Immer den Vector-Index aktualisieren (inkrementell)
            self._search.notify_embedding_added(chunk.content_hash, emb_result.vector)

        logger.info(
            "Indexiert mit Embeddings: %s → %d Chunks (%d neue Embeddings)",
            path_str,
            len(chunks),
            sum(1 for r in results if r is not None and not r.cached),
        )

        # Audit: Indexierung protokollieren
        if self._audit_logger:
            self._audit_logger.log_memory_op(
                f"index: {path_str}",
                result=f"{len(chunks)} Chunks, {sum(1 for r in results if r is not None and not r.cached)} neue Embeddings",
            )

        return len(chunks)

    def reindex_all(self) -> dict[str, int]:
        """Re-indexiert alle Memory-Dateien.

        Scannt die gesamte Memory-Struktur und indexiert alles neu.

        Returns:
            {tier: chunk_count} Dict.
        """
        counts: dict[str, int] = {}

        # Core Memory
        if self._config.core_memory_path.exists():
            n = self.index_file(self._config.core_memory_path, MemoryTier.CORE)
            counts["core"] = n

        # Episodes
        ep_count = 0
        if self._config.episodes_dir.exists():
            for f in self._config.episodes_dir.glob("*.md"):
                ep_count += self.index_file(f, MemoryTier.EPISODIC)
        counts["episodic"] = ep_count

        # Knowledge (Semantic)
        sem_count = 0
        if self._config.knowledge_dir.exists():
            for f in self._config.knowledge_dir.rglob("*.md"):
                sem_count += self.index_file(f, MemoryTier.SEMANTIC)
        counts["semantic"] = sem_count

        # Procedures
        proc_count = 0
        if self._config.procedures_dir.exists():
            for f in self._config.procedures_dir.glob("*.md"):
                proc_count += self.index_file(f, MemoryTier.PROCEDURAL)
        counts["procedural"] = proc_count

        total = sum(counts.values())
        logger.info("Re-Index komplett: %d Chunks gesamt %s", total, counts)
        return counts

    # ── Session Lifecycle ────────────────────────────────────────

    def start_session(self, session_id: str = "") -> str:
        """Startet eine neue Memory-Session.

        1. Neue Working Memory
        2. Core Memory laden
        3. Heutigen Episodic Log laden

        Returns:
            Session-ID.
        """
        wm = self._working.new_session(session_id)

        # Core Memory immer laden
        core_text = self._core.load()
        self._working.set_core_memory(core_text)

        logger.info("Session gestartet: %s", wm.session_id)
        return wm.session_id

    def end_session(self, summary: str = "") -> None:
        """Beendet die aktuelle Session.

        Schreibt Session-Zusammenfassung in Episodic Memory.

        Args:
            summary: Zusammenfassung der Session.
        """
        if summary:
            self._episodic.append_entry(
                topic="Session-Ende",
                content=summary,
            )
            logger.info("Session beendet, Zusammenfassung gespeichert")

    # ── Stats ────────────────────────────────────────────────────

    def set_media_pipeline(self, pipeline: Any) -> None:
        """Setzt die MediaPipeline für Multimodal Memory.

        Wird nach Initialisierung aufgerufen wenn MediaPipeline verfügbar.
        """
        self._multimodal._pipeline = pipeline
        logger.info("Media-Pipeline für Multimodal Memory gesetzt")

    def stats(self) -> dict[str, Any]:
        """Gesamtstatistiken des Memory-Systems."""
        index_stats = self._index.stats()
        proc_stats = self._procedural.stats()
        emb_stats = self._embeddings.stats

        return {
            "chunks": index_stats["chunks"],
            "embeddings": index_stats["embeddings"],
            "entities": index_stats["entities"],
            "relations": index_stats["relations"],
            "procedures": proc_stats["total"],
            "procedures_reliable": proc_stats["reliable"],
            "embedding_cache_hits": emb_stats.cache_hits,
            "embedding_api_calls": emb_stats.api_calls,
            "core_memory_loaded": bool(self._core.content),
            "episode_dates": len(self._episodic.list_dates()),
            "initialized": self._initialized,
            "multimodal_assets": self._multimodal.asset_count,
            "graph_ranking_computed": self._graph_ranking.last_computed is not None,
            "frequency_tracked_chunks": self._frequency_tracker.total_accesses,
        }

    # ── Cleanup ──────────────────────────────────────────────────

    async def close(self) -> None:
        """Schließt alle Ressourcen."""
        self._index.close()
        if self._episodic_store:
            self._episodic_store.close()
        if self._weight_optimizer:
            self._weight_optimizer.close()
        await self._embeddings.close()
        logger.info("Memory-System geschlossen")

    def close_sync(self) -> None:
        """Synchrones Close (ohne Embedding-Client)."""
        self._index.close()
        if self._episodic_store:
            self._episodic_store.close()
        if self._weight_optimizer:
            self._weight_optimizer.close()
