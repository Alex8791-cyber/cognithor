"""Tests für memory/semantic.py · Tier 3 Wissens-Graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.memory.indexer import MemoryIndex
from jarvis.memory.semantic import SemanticMemory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    idx = MemoryIndex(tmp_path / "test.db")
    _ = idx.conn
    return idx


@pytest.fixture
def sem(tmp_path: Path, index: MemoryIndex) -> SemanticMemory:
    return SemanticMemory(tmp_path / "knowledge", index)


class TestSemanticMemory:
    def test_add_entity(self, sem: SemanticMemory):
        entity = sem.add_entity("Hans Müller", "person", attributes={"beruf": "Ingenieur"})
        assert entity.name == "Hans Müller"
        assert entity.type == "person"
        assert entity.attributes["beruf"] == "Ingenieur"

    def test_get_entity(self, sem: SemanticMemory):
        e = sem.add_entity("Test", "person")
        loaded = sem.get_entity(e.id)
        assert loaded is not None
        assert loaded.name == "Test"

    def test_find_by_name(self, sem: SemanticMemory):
        sem.add_entity("Hans Müller", "person")
        sem.add_entity("Anna Schmidt", "person")
        results = sem.find_entities(name="Müller")
        assert len(results) == 1
        assert results[0].name == "Hans Müller"

    def test_find_by_type(self, sem: SemanticMemory):
        sem.add_entity("Müller", "person")
        sem.add_entity("TechCorp", "company")
        persons = sem.find_entities(entity_type="person")
        assert len(persons) == 1

    def test_update_entity(self, sem: SemanticMemory):
        e = sem.add_entity("Test", "person")
        updated = sem.update_entity(e.id, name="Neuer Name", attributes={"key": "val"})
        assert updated is not None
        assert updated.name == "Neuer Name"
        assert updated.attributes["key"] == "val"

    def test_update_nonexistent(self, sem: SemanticMemory):
        assert sem.update_entity("fake-id", name="X") is None

    def test_delete_entity(self, sem: SemanticMemory):
        e = sem.add_entity("Test", "person")
        assert sem.delete_entity(e.id)
        assert sem.get_entity(e.id) is None

    def test_add_relation(self, sem: SemanticMemory):
        e1 = sem.add_entity("Müller", "person")
        e2 = sem.add_entity("Cloud-Lizenz", "product")
        rel = sem.add_relation(e1.id, "hat_police", e2.id)
        assert rel is not None
        assert rel.relation_type == "hat_police"

    def test_add_relation_invalid_entity(self, sem: SemanticMemory):
        e1 = sem.add_entity("Müller", "person")
        rel = sem.add_relation(e1.id, "hat", "fake-id")
        assert rel is None

    def test_get_relations(self, sem: SemanticMemory):
        e1 = sem.add_entity("A", "person")
        e2 = sem.add_entity("B", "product")
        sem.add_relation(e1.id, "hat", e2.id)
        rels = sem.get_relations(e1.id)
        assert len(rels) == 1

    def test_get_neighbors(self, sem: SemanticMemory):
        e1 = sem.add_entity("A", "person")
        e2 = sem.add_entity("B", "person")
        sem.add_relation(e1.id, "kennt", e2.id)
        neighbors = sem.get_neighbors(e1.id)
        assert len(neighbors) >= 1
        assert neighbors[0].name == "B"

    def test_entity_with_relations(self, sem: SemanticMemory):
        e1 = sem.add_entity("Müller", "person")
        e2 = sem.add_entity("BU", "product")
        sem.add_relation(e1.id, "hat", e2.id)

        entity, connected = sem.get_entity_with_relations(e1.id)
        assert entity is not None
        assert len(connected) == 1
        assert connected[0][1].name == "BU"

    def test_entity_with_relations_nonexistent(self, sem: SemanticMemory):
        entity, connected = sem.get_entity_with_relations("fake-id")
        assert entity is None
        assert connected == []

    def test_export_graph_summary(self, sem: SemanticMemory):
        sem.add_entity("Müller", "person")
        sem.add_entity("TechCorp", "company")
        summary = sem.export_graph_summary()
        assert "Müller" in summary
        assert "TechCorp" in summary
        assert "Wissens-Graph" in summary

    def test_export_empty(self, sem: SemanticMemory):
        summary = sem.export_graph_summary()
        assert "Keine Entitäten" in summary

    def test_stats(self, sem: SemanticMemory):
        s = sem.stats()
        assert s["entities"] == 0
        assert s["relations"] == 0
        sem.add_entity("X", "person")
        s = sem.stats()
        assert s["entities"] == 1

    def test_ensure_directory(self, sem: SemanticMemory):
        sem.ensure_directory()
        assert (sem.directory / "kunden").exists()
        assert (sem.directory / "produkte").exists()
        assert (sem.directory / "projekte").exists()
