"""Graph Ranking: PageRank, Staleness-Detection und Auto-Pruning.

Erweitert den bestehenden Wissensgraphen um:

  1. PageRank: Berechnet die Wichtigkeit jeder Entität basierend
     auf ihrer Vernetzung im Graphen. Stark vernetzte Entitäten
     (z.B. "WWK") erhalten höheren Score als isolierte.

  2. Staleness-Detection: Erkennt veraltete Entitäten anhand
     von updated_at, und senkt deren Score automatisch.

  3. Auto-Pruning: Entfernt Entitäten mit niedrigem Confidence-Score
     UND hoher Staleness (konfigurierbar).

  4. Graph-Score-Boost: Nutzt PageRank-Werte um die Graph-Komponente
     der HybridSearch zu verbessern.

Algorithmus:
  PageRank iteriert: PR(A) = (1-d)/N + d × Σ PR(T)/C(T)
  wobei d=0.85 (Damping), N=Anzahl Entitäten, C(T)=Ausgangsgrad von T.

Integration:
  - GraphRanking.compute_pagerank() → {entity_id: rank}
  - GraphRanking.boost_graph_scores() → Multipliziert Graph-Scores
  - GraphRanking.prune_stale() → Entfernt veraltete Entitäten

Bibel-Referenz: §4.4 (Wissens-Graph), §4.10 (Graph Ranking)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from jarvis.models import Entity, MemorySearchResult, Relation

logger = logging.getLogger("jarvis.memory.graph_ranking")


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass
class EntityRank:
    """PageRank-Ergebnis für eine Entität."""

    entity_id: str
    entity_name: str
    pagerank: float  # 0.0 – 1.0 (normalisiert)
    degree: int  # Anzahl Verbindungen (ein+ausgehend)
    staleness: float  # 0.0 (frisch) – 1.0 (veraltet)
    combined_score: float = 0.0  # pagerank × (1 - staleness)

    @property
    def is_stale(self) -> bool:
        """Entität gilt als veraltet wenn staleness > 0.7."""
        return self.staleness > 0.7


@dataclass
class PruneResult:
    """Ergebnis einer Pruning-Operation."""

    pruned_entities: list[str] = field(default_factory=list)
    pruned_relations: int = 0
    total_before: int = 0
    total_after: int = 0


# ============================================================================
# PageRank
# ============================================================================


class GraphRanking:
    """PageRank und Graph-Analyse für den Wissensgraphen.

    Args:
        index: MemoryIndex für Graph-Zugriff.
        damping: PageRank Damping-Faktor (Standard: 0.85).
        staleness_half_life: Halbwertszeit für Staleness in Tagen.
    """

    def __init__(
        self,
        index: Any,  # MemoryIndex
        *,
        damping: float = 0.85,
        staleness_half_life_days: int = 90,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
    ) -> None:
        self._index = index
        self._damping = damping
        self._staleness_half_life = staleness_half_life_days
        self._max_iterations = max_iterations
        self._convergence = convergence_threshold
        self._ranks: dict[str, EntityRank] = {}  # Cache
        self._last_computed: datetime | None = None

    @property
    def ranks(self) -> dict[str, EntityRank]:
        """Aktuell berechnete Ranks (leer wenn noch nicht berechnet)."""
        return dict(self._ranks)

    @property
    def last_computed(self) -> datetime | None:
        return self._last_computed

    # ========================================================================
    # PageRank Berechnung
    # ========================================================================

    def compute_pagerank(self) -> dict[str, EntityRank]:
        """Berechnet PageRank für alle Entitäten im Graphen.

        Algorithmus:
          1. Adjacency-Liste aus Relationen bauen
          2. Alle Ranks auf 1/N initialisieren
          3. Iterieren bis Konvergenz (oder max_iterations)
          4. Normalisieren auf 0-1
          5. Staleness berechnen
          6. Combined Score: pagerank × (1 - staleness × 0.5)

        Returns:
            {entity_id: EntityRank} Dict.
        """
        entities = self._index.search_entities()
        if not entities:
            self._ranks = {}
            self._last_computed = datetime.now(timezone.utc)
            return {}

        entity_map = {e.id: e for e in entities}
        n = len(entities)

        # Adjacency-Liste bauen
        outgoing: dict[str, set[str]] = defaultdict(set)
        incoming: dict[str, set[str]] = defaultdict(set)
        degree: dict[str, int] = defaultdict(int)

        seen_relations: set[tuple[str, str, str]] = set()
        for entity in entities:
            relations = self._index.get_relations_for_entity(entity.id)
            for rel in relations:
                rel_key = (rel.source_entity, rel.relation_type, rel.target_entity)
                if rel_key in seen_relations:
                    continue
                seen_relations.add(rel_key)
                src = rel.source_entity
                tgt = rel.target_entity
                if src in entity_map and tgt in entity_map:
                    outgoing[src].add(tgt)
                    incoming[tgt].add(src)
                    degree[src] = degree.get(src, 0) + 1
                    degree[tgt] = degree.get(tgt, 0) + 1

        # PageRank initialisieren
        pr: dict[str, float] = {e.id: 1.0 / n for e in entities}

        # Iterieren
        d = self._damping
        for iteration in range(self._max_iterations):
            new_pr: dict[str, float] = {}
            diff = 0.0

            for entity in entities:
                eid = entity.id
                rank_sum = 0.0
                for src_id in incoming.get(eid, set()):
                    out_count = len(outgoing.get(src_id, set()))
                    if out_count > 0:
                        rank_sum += pr[src_id] / out_count

                new_pr[eid] = (1.0 - d) / n + d * rank_sum
                diff += abs(new_pr[eid] - pr[eid])

            pr = new_pr

            if diff < self._convergence:
                logger.debug(
                    "pagerank_converged: iteration=%d, diff=%.8f",
                    iteration + 1, diff,
                )
                break

        # Normalisieren auf 0-1
        max_pr = max(pr.values()) if pr else 1.0
        if max_pr > 0:
            for eid in pr:
                pr[eid] /= max_pr

        # EntityRank-Objekte erstellen
        ranks: dict[str, EntityRank] = {}
        for entity in entities:
            eid = entity.id
            stale = self._compute_staleness(entity)
            pagerank = pr.get(eid, 0.0)

            # Combined: PageRank mit Staleness-Abzug
            combined = pagerank * (1.0 - stale * 0.5)

            ranks[eid] = EntityRank(
                entity_id=eid,
                entity_name=entity.name,
                pagerank=pagerank,
                degree=degree.get(eid, 0),
                staleness=stale,
                combined_score=combined,
            )

        self._ranks = ranks
        self._last_computed = datetime.now(timezone.utc)

        logger.info(
            "pagerank_computed: entities=%d, iterations<=%d",
            len(ranks), self._max_iterations,
        )
        return ranks

    def get_rank(self, entity_id: str) -> EntityRank | None:
        """Gibt den Rank einer einzelnen Entität zurück."""
        return self._ranks.get(entity_id)

    def top_entities(self, n: int = 10) -> list[EntityRank]:
        """Die N wichtigsten Entitäten nach Combined Score.

        Returns:
            Sortierte Liste (höchster Score zuerst).
        """
        ranked = sorted(
            self._ranks.values(),
            key=lambda r: r.combined_score,
            reverse=True,
        )
        return ranked[:n]

    def stale_entities(self, threshold: float = 0.7) -> list[EntityRank]:
        """Entitäten mit hoher Staleness.

        Args:
            threshold: Staleness-Schwellwert (0.0-1.0).

        Returns:
            Liste veralteter Entitäten.
        """
        return [r for r in self._ranks.values() if r.staleness > threshold]

    # ========================================================================
    # Staleness-Berechnung
    # ========================================================================

    def _compute_staleness(
        self,
        entity: Entity,
        *,
        reference_date: date | None = None,
    ) -> float:
        """Berechnet die Staleness einer Entität.

        Exponentieller Decay basierend auf dem letzten Update.

        Returns:
            Staleness 0.0 (frisch) bis 1.0 (veraltet).
        """
        ref = reference_date or date.today()
        updated = entity.updated_at

        if isinstance(updated, datetime):
            age_days = (ref - updated.date()).days
        elif isinstance(updated, date):
            age_days = (ref - updated).days
        else:
            # Kein Update-Datum → maximal stale
            return 1.0

        if age_days <= 0:
            return 0.0

        # Korrekte Halbwertszeit: nach half_life Tagen ist staleness = 0.5
        return 1.0 - 0.5 ** (age_days / self._staleness_half_life)

    # ========================================================================
    # Graph-Score Boost für HybridSearch
    # ========================================================================

    def boost_graph_scores(
        self,
        results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Boosted Graph-Scores der Suchergebnisse mit PageRank.

        Chunks die Entitäten mit hohem PageRank referenzieren
        erhalten einen Score-Boost.

        Args:
            results: Originale Suchergebnisse.

        Returns:
            Ergebnisse mit adjustierten Scores, neu sortiert.
        """
        if not self._ranks:
            return results

        boosted: list[MemorySearchResult] = []
        for result in results:
            boost = self._compute_chunk_boost(result.chunk)
            new_score = result.score * boost

            boosted.append(
                MemorySearchResult(
                    chunk=result.chunk,
                    score=new_score,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    graph_score=result.graph_score * boost,
                    recency_factor=result.recency_factor,
                ),
            )

        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted

    def _compute_chunk_boost(self, chunk: Any) -> float:
        """Berechnet den PageRank-Boost für einen Chunk.

        Basiert auf den Entitäten die im Chunk referenziert werden.
        """
        entity_ids = getattr(chunk, "entities", []) or []
        if not entity_ids:
            return 1.0

        total_rank = 0.0
        count = 0
        for eid in entity_ids:
            rank = self._ranks.get(eid)
            if rank:
                total_rank += rank.combined_score
                count += 1

        if count == 0:
            return 1.0

        avg_rank = total_rank / count
        # Boost: 1.0 (kein Boost) bis 1.5 (maximaler Boost)
        return 1.0 + avg_rank * 0.5

    # ========================================================================
    # Auto-Pruning
    # ========================================================================

    def prune_stale(
        self,
        *,
        staleness_threshold: float = 0.8,
        min_confidence: float = 0.3,
        min_degree: int = 0,
        dry_run: bool = False,
    ) -> PruneResult:
        """Entfernt veraltete, niedrig-vertrauenswürdige Entitäten.

        Kriterien (alle müssen zutreffen):
          - Staleness > threshold
          - Confidence < min_confidence
          - Degree <= min_degree (isolierte Entitäten bevorzugt)

        Args:
            staleness_threshold: Mindest-Staleness für Pruning.
            min_confidence: Maximaler Confidence-Wert für Pruning.
            min_degree: Maximaler Vernetzungsgrad für Pruning.
            dry_run: Wenn True, nur simulieren.

        Returns:
            PruneResult mit Details.
        """
        if not self._ranks:
            self.compute_pagerank()

        entities = self._index.search_entities()
        total_before = len(entities)

        to_prune: list[str] = []
        for entity in entities:
            rank = self._ranks.get(entity.id)
            if not rank:
                continue

            if (
                rank.staleness > staleness_threshold
                and entity.confidence < min_confidence
                and rank.degree <= min_degree
            ):
                to_prune.append(entity.id)

        pruned_relations = 0
        if not dry_run:
            for eid in to_prune:
                # Relationen werden durch delete_entity mitgelöscht
                relations = self._index.get_relations_for_entity(eid)
                pruned_relations += len(relations)
                self._index.delete_entity(eid)

            # Ranks aktualisieren
            if to_prune:
                for eid in to_prune:
                    self._ranks.pop(eid, None)

                logger.info(
                    "graph_pruned: %d entities, %d relations",
                    len(to_prune), pruned_relations,
                )

        return PruneResult(
            pruned_entities=to_prune,
            pruned_relations=pruned_relations,
            total_before=total_before,
            total_after=total_before - len(to_prune),
        )

    # ========================================================================
    # Graph Update
    # ========================================================================

    def update_entity_confidence(
        self,
        entity_id: str,
        delta: float,
    ) -> float | None:
        """Passt den Confidence-Wert einer Entität an.

        Positive delta → Entität wird vertrauenswürdiger.
        Negative delta → Entität wird weniger vertrauenswürdig.

        Args:
            entity_id: Entity-ID.
            delta: Änderung (-1.0 bis +1.0).

        Returns:
            Neuer Confidence-Wert oder None wenn nicht gefunden.
        """
        entity = self._index.get_entity_by_id(entity_id)
        if entity is None:
            return None

        new_conf = max(0.0, min(1.0, entity.confidence + delta))
        entity.confidence = new_conf
        entity.updated_at = datetime.now(timezone.utc)
        self._index.upsert_entity(entity)

        return new_conf

    def touch_entity(self, entity_id: str) -> bool:
        """Aktualisiert den Zeitstempel einer Entität (Reset Staleness).

        Args:
            entity_id: Entity-ID.

        Returns:
            True wenn erfolgreich.
        """
        entity = self._index.get_entity_by_id(entity_id)
        if entity is None:
            return False

        entity.updated_at = datetime.now(timezone.utc)
        self._index.upsert_entity(entity)
        return True

    # ========================================================================
    # Analyse & Statistiken
    # ========================================================================

    def graph_summary(self) -> dict[str, Any]:
        """Zusammenfassung des Graphen mit Ranking-Infos."""
        if not self._ranks:
            return {"status": "not_computed", "entities": 0}

        ranks = list(self._ranks.values())
        stale_count = sum(1 for r in ranks if r.is_stale)

        avg_pr = sum(r.pagerank for r in ranks) / max(len(ranks), 1)
        avg_degree = sum(r.degree for r in ranks) / max(len(ranks), 1)

        return {
            "total_entities": len(ranks),
            "stale_entities": stale_count,
            "fresh_entities": len(ranks) - stale_count,
            "avg_pagerank": round(avg_pr, 4),
            "avg_degree": round(avg_degree, 2),
            "top_3": [
                {"name": r.entity_name, "rank": round(r.combined_score, 4)}
                for r in self.top_entities(3)
            ],
            "last_computed": (
                self._last_computed.isoformat() if self._last_computed else None
            ),
        }

    def find_isolated_entities(self) -> list[EntityRank]:
        """Findet Entitäten ohne Verbindungen (degree=0)."""
        return [r for r in self._ranks.values() if r.degree == 0]

    def find_hub_entities(self, min_degree: int = 5) -> list[EntityRank]:
        """Findet stark vernetzte Hub-Entitäten."""
        hubs = [r for r in self._ranks.values() if r.degree >= min_degree]
        hubs.sort(key=lambda r: r.degree, reverse=True)
        return hubs
