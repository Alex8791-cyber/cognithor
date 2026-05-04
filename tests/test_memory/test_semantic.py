"""Tests für memory/semantic.py · Tier 3 Wissens-Graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.memory.indexer import MemoryIndex
from cognithor.memory.semantic import SemanticMemory

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


class TestSemanticMemoryProvenance:
    """TRUST-9 wiring: passing ``provenance_source_type`` +
    ``provenance_source_id`` to ``add_entity`` writes a tag to the
    canonical PROVENANCE_LEDGER keyed by the entity's id.
    """

    def test_add_entity_without_provenance_does_not_tag(self, sem: SemanticMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            entity = sem.add_entity("ProbeNoTag", "person")
            assert entity.id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_add_entity_with_provenance_tags_ledger(self, sem: SemanticMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            entity = sem.add_entity(
                "ProbeWithTag",
                "person",
                confidence=0.85,
                provenance_source_type="tool_output",
                provenance_source_id="audit-42",
                provenance_notes="extracted by NER",
            )
            tag = isolated.current(entity.id)
            assert tag is not None
            assert tag.source_type == SourceType.TOOL_OUTPUT
            assert tag.source_id == "audit-42"
            assert tag.confidence == 0.85
            assert tag.notes == "extracted by NER"
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_unknown_source_type_falls_back_to_unknown(self, sem: SemanticMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            entity = sem.add_entity(
                "ProbeUnknown",
                "person",
                provenance_source_type="not_a_real_source",
                provenance_source_id="x",
            )
            tag = isolated.current(entity.id)
            assert tag is not None
            assert tag.source_type == SourceType.UNKNOWN
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_partial_provenance_args_do_not_tag(self, sem: SemanticMemory) -> None:
        # Both source_type AND source_id required — passing only one
        # silently skips tagging.
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            e1 = sem.add_entity("OnlyType", "person", provenance_source_type="tool_output")
            e2 = sem.add_entity("OnlyId", "person", provenance_source_id="audit-7")
            assert e1.id not in isolated
            assert e2.id not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]
