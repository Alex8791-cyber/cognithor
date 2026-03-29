"""Tests for GoalScopedIndex — per-goal isolated vector + entity storage."""

from __future__ import annotations

import pytest

from jarvis.evolution.goal_index import GoalScopedIndex, IndexedChunk


class TestGoalScopedIndex:
    """Unit tests for GoalScopedIndex."""

    def test_add_and_search_chunks(self, tmp_path):
        """Add 3 chunks, search finds matching ones."""
        idx = GoalScopedIndex(goal_slug="german-tax", base_dir=tmp_path)
        idx.add_chunk("Einkommensteuer ist eine direkte Steuer auf das Einkommen")
        idx.add_chunk("Umsatzsteuer wird auf den Verkauf von Waren erhoben")
        idx.add_chunk("Gewerbesteuer betrifft gewerbliche Unternehmen")

        results = idx.search_chunks("Steuer Einkommen")
        assert len(results) >= 1
        assert all(isinstance(r, IndexedChunk) for r in results)
        # At least one result should mention Einkommensteuer
        texts = [r.text for r in results]
        assert any("Einkommensteuer" in t for t in texts)
        idx.close()

    def test_search_no_results(self, tmp_path):
        """Search for non-existent text returns empty list."""
        idx = GoalScopedIndex(goal_slug="quantum", base_dir=tmp_path)
        idx.add_chunk("Photons are particles of light")
        results = idx.search_chunks("zzznonexistentzzz")
        assert results == []
        idx.close()

    def test_chunk_count(self, tmp_path):
        """Add 5 chunks, count returns 5."""
        idx = GoalScopedIndex(goal_slug="math", base_dir=tmp_path)
        for i in range(5):
            idx.add_chunk(f"Chunk number {i} about mathematics")
        assert idx.chunk_count() == 5
        idx.close()

    def test_add_entity(self, tmp_path):
        """Add entity, get_entity returns it with correct fields."""
        idx = GoalScopedIndex(goal_slug="law", base_dir=tmp_path)
        idx.add_entity(
            "BGB", "law", attributes={"section": "276"}, source_url="https://example.com"
        )
        entity = idx.get_entity("BGB")
        assert entity is not None
        assert entity["name"] == "BGB"
        assert entity["type"] == "law"
        assert entity["attributes"]["section"] == "276"
        assert entity["source_url"] == "https://example.com"
        idx.close()

    def test_add_entity_upsert(self, tmp_path):
        """Add same entity name twice updates rather than duplicates."""
        idx = GoalScopedIndex(goal_slug="law", base_dir=tmp_path)
        idx.add_entity("BGB", "law", attributes={"version": "old"})
        idx.add_entity("BGB", "law", attributes={"version": "new"})
        assert idx.entity_count() == 1
        entity = idx.get_entity("BGB")
        assert entity["attributes"]["version"] == "new"
        idx.close()

    def test_add_relation(self, tmp_path):
        """Add relation, get_entity_relations returns it."""
        idx = GoalScopedIndex(goal_slug="graph", base_dir=tmp_path)
        idx.add_entity("Python", "language")
        idx.add_entity("Flask", "framework")
        idx.add_relation("Flask", "uses", "Python")
        rels = idx.get_entity_relations("Flask")
        assert len(rels) == 1
        assert rels[0]["source"] == "Flask"
        assert rels[0]["relation"] == "uses"
        assert rels[0]["target"] == "Python"
        idx.close()

    def test_add_relation_dedup(self, tmp_path):
        """Add same relation twice, only one stored."""
        idx = GoalScopedIndex(goal_slug="graph", base_dir=tmp_path)
        idx.add_relation("A", "links_to", "B")
        idx.add_relation("A", "links_to", "B")
        assert idx.relation_count() == 1
        idx.close()

    def test_entity_count(self, tmp_path):
        """Add 3 entities, count returns 3."""
        idx = GoalScopedIndex(goal_slug="bio", base_dir=tmp_path)
        idx.add_entity("DNA", "concept")
        idx.add_entity("RNA", "concept")
        idx.add_entity("Protein", "concept")
        assert idx.entity_count() == 3
        idx.close()

    def test_stats(self, tmp_path):
        """Stats returns correct counts and goal_slug."""
        idx = GoalScopedIndex(goal_slug="test-stats", base_dir=tmp_path)
        idx.add_chunk("chunk one")
        idx.add_chunk("chunk two")
        idx.add_entity("Entity1", "concept")
        idx.add_relation("Entity1", "related_to", "Entity2")
        s = idx.stats()
        assert s["goal_slug"] == "test-stats"
        assert s["chunks"] == 2
        assert s["entities"] == 1
        assert s["relations"] == 1
        assert "path" in s
        idx.close()

    def test_persistence(self, tmp_path):
        """Add data, close, reopen — data still there."""
        idx = GoalScopedIndex(goal_slug="persist", base_dir=tmp_path)
        idx.add_chunk("persistent chunk about databases")
        idx.add_entity("SQLite", "technology", attributes={"type": "embedded"})
        idx.add_relation("SQLite", "is_a", "database")
        idx.close()

        # Reopen
        idx2 = GoalScopedIndex(goal_slug="persist", base_dir=tmp_path)
        assert idx2.chunk_count() == 1
        assert idx2.entity_count() == 1
        assert idx2.relation_count() == 1
        entity = idx2.get_entity("SQLite")
        assert entity is not None
        assert entity["attributes"]["type"] == "embedded"
        results = idx2.search_chunks("databases")
        assert len(results) >= 1
        idx2.close()
