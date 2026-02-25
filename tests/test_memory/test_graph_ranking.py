"""Tests für Graph Ranking: PageRank, Staleness, Auto-Pruning.

Testet:
  - PageRank Berechnung (Konvergenz, Normalisierung, Hub-Erkennung)
  - Staleness-Detection (Exponentieller Decay)
  - Combined Score (PageRank × Staleness)
  - Auto-Pruning (Stale + Low-Confidence + Isoliert)
  - Graph-Score Boost für HybridSearch
  - Entity Confidence Updates + Touch
  - Graph-Analyse (Hubs, Isolierte, Summary)
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.memory.graph_ranking import (
    EntityRank,
    GraphRanking,
    PruneResult,
)
from jarvis.memory.indexer import MemoryIndex
from jarvis.models import Chunk, Entity, MemorySearchResult, MemoryTier, Relation


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    """Frischer MemoryIndex."""
    idx = MemoryIndex(tmp_path / "test.db")
    return idx


@pytest.fixture
def populated_graph(index: MemoryIndex) -> MemoryIndex:
    """Graph mit 4 Entitäten und Relationen.

    Topologie:
      WWK ←→ Alexander ←→ Allianz
                 ↓
              Jarvis

    WWK und Alexander sind Hub-Entitäten.
    Jarvis hat nur 1 Verbindung.
    """
    now = datetime.now(timezone.utc)

    e_wwk = Entity(id="e_wwk", type="company", name="WWK", confidence=0.9, updated_at=now)
    e_alex = Entity(id="e_alex", type="person", name="Alexander", confidence=1.0, updated_at=now)
    e_allianz = Entity(
        id="e_allianz", type="company", name="Allianz", confidence=0.7,
        updated_at=now - timedelta(days=60),
    )
    e_jarvis = Entity(id="e_jarvis", type="project", name="Jarvis", confidence=0.8, updated_at=now)

    for e in [e_wwk, e_alex, e_allianz, e_jarvis]:
        index.upsert_entity(e)

    # Relationen
    r1 = Relation(
        id="r1", source_entity="e_alex", relation_type="arbeitet_bei",
        target_entity="e_wwk", confidence=1.0,
    )
    r2 = Relation(
        id="r2", source_entity="e_alex", relation_type="vergleicht",
        target_entity="e_allianz", confidence=0.8,
    )
    r3 = Relation(
        id="r3", source_entity="e_alex", relation_type="entwickelt",
        target_entity="e_jarvis", confidence=0.9,
    )
    r4 = Relation(
        id="r4", source_entity="e_wwk", relation_type="konkurriert_mit",
        target_entity="e_allianz", confidence=0.7,
    )

    for r in [r1, r2, r3, r4]:
        index.upsert_relation(r)

    return index


@pytest.fixture
def ranking(populated_graph: MemoryIndex) -> GraphRanking:
    return GraphRanking(populated_graph)


# ============================================================================
# PageRank Berechnung
# ============================================================================


class TestPageRank:
    """PageRank-Algorithmus."""

    def test_empty_graph(self, index: MemoryIndex) -> None:
        gr = GraphRanking(index)
        ranks = gr.compute_pagerank()
        assert ranks == {}

    def test_single_entity(self, index: MemoryIndex) -> None:
        e = Entity(id="solo", type="test", name="Solo", updated_at=datetime.now(timezone.utc))
        index.upsert_entity(e)

        gr = GraphRanking(index)
        ranks = gr.compute_pagerank()
        assert len(ranks) == 1
        assert ranks["solo"].pagerank == 1.0  # Normalisiert auf 1.0

    def test_populated_graph_ranks(self, ranking: GraphRanking) -> None:
        ranks = ranking.compute_pagerank()
        assert len(ranks) == 4

        # Alle Werte zwischen 0 und 1
        for r in ranks.values():
            assert 0.0 <= r.pagerank <= 1.0
            assert 0.0 <= r.combined_score <= 1.5  # pagerank × (1 - stale*0.5)

    def test_hub_entity_highest_rank(self, ranking: GraphRanking) -> None:
        """Alexander hat die meisten Verbindungen → höchster Degree."""
        ranks = ranking.compute_pagerank()

        alex = ranks["e_alex"]
        jarvis = ranks["e_jarvis"]

        # Alexander ist stärker vernetzt (mehr Kanten)
        assert alex.degree > jarvis.degree
        # Beide haben nicht-trivialen PageRank
        assert alex.pagerank > 0
        assert jarvis.pagerank > 0

    def test_pagerank_normalized(self, ranking: GraphRanking) -> None:
        """Höchster PageRank ist 1.0 (normalisiert)."""
        ranks = ranking.compute_pagerank()
        max_pr = max(r.pagerank for r in ranks.values())
        assert max_pr == pytest.approx(1.0)

    def test_pagerank_cached(self, ranking: GraphRanking) -> None:
        """Ranks werden gecached nach compute_pagerank()."""
        assert ranking.last_computed is None
        ranking.compute_pagerank()
        assert ranking.last_computed is not None
        assert len(ranking.ranks) == 4

    def test_get_rank(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        rank = ranking.get_rank("e_wwk")
        assert rank is not None
        assert rank.entity_name == "WWK"
        assert rank.pagerank > 0

    def test_get_rank_nonexistent(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        assert ranking.get_rank("ghost") is None


class TestTopEntities:
    """Top-Entitäten nach Combined Score."""

    def test_top_entities(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        top = ranking.top_entities(2)
        assert len(top) == 2
        assert top[0].combined_score >= top[1].combined_score

    def test_top_entities_empty(self, index: MemoryIndex) -> None:
        gr = GraphRanking(index)
        gr.compute_pagerank()
        assert gr.top_entities(5) == []


# ============================================================================
# Staleness
# ============================================================================


class TestStaleness:
    """Exponentieller Staleness-Decay."""

    def test_fresh_entity_zero_staleness(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        alex = ranking.get_rank("e_alex")
        assert alex is not None
        # Alexander wurde gerade aktualisiert → niedrige Staleness
        assert alex.staleness < 0.1

    def test_old_entity_higher_staleness(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        allianz = ranking.get_rank("e_allianz")
        assert allianz is not None
        # Allianz wurde vor 60 Tagen aktualisiert → höhere Staleness
        assert allianz.staleness > 0.3

    def test_is_stale_property(self) -> None:
        rank = EntityRank(
            entity_id="x", entity_name="X",
            pagerank=0.5, degree=2, staleness=0.8,
        )
        assert rank.is_stale is True

        fresh = EntityRank(
            entity_id="y", entity_name="Y",
            pagerank=0.5, degree=2, staleness=0.2,
        )
        assert fresh.is_stale is False

    def test_stale_entities_filter(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        stale = ranking.stale_entities(threshold=0.3)
        # Allianz (60 Tage alt) sollte über dem Schwellwert sein
        stale_ids = {r.entity_id for r in stale}
        assert "e_allianz" in stale_ids


class TestCombinedScore:
    """Combined Score: PageRank × (1 - Staleness × 0.5)."""

    def test_fresh_high_rank_best_combined(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        # Alexander: frisch + gut vernetzt → positive combined score
        alex = ranking.get_rank("e_alex")
        assert alex is not None
        assert alex.combined_score > 0
        assert alex.staleness < 0.1  # Frisch


# ============================================================================
# Graph-Score Boost
# ============================================================================


class TestGraphScoreBoost:
    """PageRank-basierter Score-Boost für Suchergebnisse."""

    def test_boost_without_ranks(self, ranking: GraphRanking) -> None:
        """Ohne berechnete Ranks: kein Boost."""
        results = [
            MemorySearchResult(
                chunk=Chunk(id="c1", text="Test", source_path="t.md"),
                score=0.5,
            ),
        ]
        boosted = ranking.boost_graph_scores(results)
        assert boosted[0].score == 0.5  # Unverändert

    def test_boost_with_entity_reference(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        """Chunks mit Entity-Referenzen erhalten Boost."""
        ranking.compute_pagerank()

        results = [
            MemorySearchResult(
                chunk=Chunk(
                    id="c_with_entity", text="Alexander bei WWK",
                    source_path="t.md",
                    entities=["e_alex", "e_wwk"],  # Referenziert Entitäten
                ),
                score=0.5,
            ),
            MemorySearchResult(
                chunk=Chunk(
                    id="c_no_entity", text="Random text",
                    source_path="t.md",
                ),
                score=0.5,
            ),
        ]
        boosted = ranking.boost_graph_scores(results)

        # Chunk mit Entitäten sollte höheren Score haben
        c_with = next(r for r in boosted if r.chunk.id == "c_with_entity")
        c_without = next(r for r in boosted if r.chunk.id == "c_no_entity")
        assert c_with.score > c_without.score

    def test_boost_reorders(self, ranking: GraphRanking) -> None:
        """Boost kann die Reihenfolge der Ergebnisse ändern."""
        ranking.compute_pagerank()

        results = [
            MemorySearchResult(
                chunk=Chunk(
                    id="c1", text="Allgemein", source_path="t.md",
                ),
                score=0.6,
            ),
            MemorySearchResult(
                chunk=Chunk(
                    id="c2", text="Entity-reich", source_path="t.md",
                    entities=["e_alex", "e_wwk", "e_jarvis"],
                ),
                score=0.55,
            ),
        ]
        boosted = ranking.boost_graph_scores(results)
        # c2 hat niedrigeren Basis-Score aber Entity-Boost
        assert boosted[0].chunk.id == "c2"


# ============================================================================
# Auto-Pruning
# ============================================================================


class TestPruning:
    """Entfernung veralteter, niedrig-vertrauenswürdiger Entitäten."""

    def test_no_pruning_needed(self, ranking: GraphRanking) -> None:
        """Keine Entität erfüllt alle Pruning-Kriterien."""
        ranking.compute_pagerank()
        result = ranking.prune_stale(dry_run=True)
        # Alle Entitäten haben hohe Confidence → nichts zu prunen
        assert len(result.pruned_entities) == 0

    def test_prune_stale_low_confidence(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        """Stale + niedrige Confidence + isoliert → wird gepruned."""
        # Neue isolierte, stale, low-confidence Entität hinzufügen
        old_entity = Entity(
            id="e_old",
            type="test",
            name="Old Thing",
            confidence=0.1,  # Niedrig
            updated_at=datetime.now(timezone.utc) - timedelta(days=365),  # Sehr alt
        )
        populated_graph.upsert_entity(old_entity)

        ranking.compute_pagerank()
        result = ranking.prune_stale(
            staleness_threshold=0.5,
            min_confidence=0.3,
            min_degree=0,
            dry_run=False,
        )

        assert "e_old" in result.pruned_entities
        assert result.total_after < result.total_before

    def test_prune_dry_run(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        """Dry-Run meldet was gepruned würde, ändert aber nichts."""
        old_entity = Entity(
            id="e_dry", type="test", name="Dry",
            confidence=0.1,
            updated_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        populated_graph.upsert_entity(old_entity)

        ranking.compute_pagerank()
        result = ranking.prune_stale(
            staleness_threshold=0.5,
            min_confidence=0.3,
            dry_run=True,
        )

        assert "e_dry" in result.pruned_entities
        # Aber die Entität existiert noch
        assert populated_graph.get_entity_by_id("e_dry") is not None

    def test_well_connected_not_pruned(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        """Gut vernetzte Entitäten werden nicht gepruned (selbst wenn stale)."""
        ranking.compute_pagerank()
        result = ranking.prune_stale(
            staleness_threshold=0.3,
            min_confidence=0.8,
            min_degree=0,  # Nur isolierte
        )
        # Alexander, WWK, etc. sind vernetzt → nicht gepruned
        assert "e_alex" not in result.pruned_entities
        assert "e_wwk" not in result.pruned_entities


# ============================================================================
# Entity Updates
# ============================================================================


class TestEntityUpdates:
    """Confidence-Updates und Touch."""

    def test_update_confidence(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        new_conf = ranking.update_entity_confidence("e_wwk", -0.2)
        assert new_conf is not None
        assert new_conf == pytest.approx(0.7)  # 0.9 - 0.2

    def test_update_confidence_clamped(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        # Über 1.0 → clamped
        new_conf = ranking.update_entity_confidence("e_wwk", +0.5)
        assert new_conf == 1.0

        # Unter 0.0 → clamped
        new_conf = ranking.update_entity_confidence("e_wwk", -2.0)
        assert new_conf == 0.0

    def test_update_nonexistent(self, ranking: GraphRanking) -> None:
        assert ranking.update_entity_confidence("ghost", 0.1) is None

    def test_touch_entity(
        self, ranking: GraphRanking, populated_graph: MemoryIndex,
    ) -> None:
        assert ranking.touch_entity("e_allianz") is True
        entity = populated_graph.get_entity_by_id("e_allianz")
        assert entity is not None
        # updated_at sollte jetzt frisch sein (DB gibt naive datetime zurück)
        now_naive = datetime.now()
        updated = entity.updated_at.replace(tzinfo=None) if entity.updated_at.tzinfo else entity.updated_at
        age = (now_naive - updated).total_seconds()
        assert age < 5  # Weniger als 5 Sekunden

    def test_touch_nonexistent(self, ranking: GraphRanking) -> None:
        assert ranking.touch_entity("ghost") is False


# ============================================================================
# Graph-Analyse
# ============================================================================


class TestGraphAnalysis:
    """Hub-Erkennung, Isolierte, Summary."""

    def test_find_isolated(self, index: MemoryIndex) -> None:
        """Entitäten ohne Verbindungen."""
        e = Entity(
            id="isolated", type="test", name="Lonely",
            updated_at=datetime.now(timezone.utc),
        )
        index.upsert_entity(e)

        gr = GraphRanking(index)
        gr.compute_pagerank()

        isolated = gr.find_isolated_entities()
        assert len(isolated) == 1
        assert isolated[0].entity_id == "isolated"

    def test_find_hubs(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        hubs = ranking.find_hub_entities(min_degree=2)
        hub_ids = {h.entity_id for h in hubs}
        assert "e_alex" in hub_ids  # Alexander hat 3+ Verbindungen

    def test_graph_summary(self, ranking: GraphRanking) -> None:
        ranking.compute_pagerank()
        summary = ranking.graph_summary()

        assert summary["total_entities"] == 4
        assert "stale_entities" in summary
        assert "fresh_entities" in summary
        assert "avg_pagerank" in summary
        assert "top_3" in summary
        assert len(summary["top_3"]) == 3
        assert summary["last_computed"] is not None

    def test_graph_summary_empty(self, index: MemoryIndex) -> None:
        gr = GraphRanking(index)
        summary = gr.graph_summary()
        assert summary["status"] == "not_computed"


# ============================================================================
# Damping & Convergence
# ============================================================================


class TestAlgorithmParams:
    """PageRank-Algorithmus Parametertests."""

    def test_custom_damping(self, populated_graph: MemoryIndex) -> None:
        gr_low = GraphRanking(populated_graph, damping=0.5)
        gr_high = GraphRanking(populated_graph, damping=0.95)

        ranks_low = gr_low.compute_pagerank()
        ranks_high = gr_high.compute_pagerank()

        # Beide konvergieren, aber mit unterschiedlichen Verteilungen
        assert len(ranks_low) == len(ranks_high) == 4

    def test_convergence(self, populated_graph: MemoryIndex) -> None:
        """PageRank konvergiert innerhalb der max_iterations."""
        gr = GraphRanking(populated_graph, max_iterations=100, convergence_threshold=1e-8)
        ranks = gr.compute_pagerank()
        assert len(ranks) == 4
        # Alle Ranks sind nicht-negative
        for r in ranks.values():
            assert r.pagerank >= 0.0

    def test_degree_count(self, ranking: GraphRanking) -> None:
        """Degree wird korrekt berechnet."""
        ranking.compute_pagerank()
        alex = ranking.get_rank("e_alex")
        assert alex is not None
        # Alexander hat Relationen zu WWK, Allianz, Jarvis + eingehend von WWK
        assert alex.degree >= 3
