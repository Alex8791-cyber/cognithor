"""Tests für memory/indexer.py · SQLite Index."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from jarvis.memory.indexer import MemoryIndex, _deserialize_vector, _serialize_vector
from jarvis.models import Chunk, Entity, MemoryTier, Relation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
def index(db_path: Path) -> MemoryIndex:
    idx = MemoryIndex(db_path)
    _ = idx.conn  # Trigger schema creation
    return idx


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        text="Der Kontakt Müller nutzt eine Cloud-Lizenz",
        source_path="knowledge/kunden/mueller.md",
        line_start=0,
        line_end=1,
        content_hash="abc123",
        memory_tier=MemoryTier.SEMANTIC,
        timestamp=datetime(2026, 2, 21),
        token_count=10,
    )


@pytest.fixture
def sample_entity() -> Entity:
    return Entity(
        type="person",
        name="Hans Müller",
        attributes={"beruf": "Ingenieur", "alter": 45},
        source_file="kunden/mueller.md",
    )


class TestVectorSerialization:
    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        data = _serialize_vector(vec)
        result = _deserialize_vector(data)
        assert len(result) == len(vec)
        for a, b in zip(vec, result, strict=False):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        data = _serialize_vector([])
        result = _deserialize_vector(data)
        assert result == []


class TestMemoryIndexChunks:
    def test_upsert_chunk(self, index: MemoryIndex, sample_chunk: Chunk):
        index.upsert_chunk(sample_chunk)
        loaded = index.get_chunk_by_id(sample_chunk.id)
        assert loaded is not None
        assert loaded.text == sample_chunk.text
        assert loaded.memory_tier == MemoryTier.SEMANTIC

    def test_upsert_chunks_batch(self, index: MemoryIndex):
        chunks = [
            Chunk(text=f"Chunk {i}", source_path="test.md", content_hash=f"h{i}") for i in range(5)
        ]
        count = index.upsert_chunks(chunks)
        assert count == 5
        assert index.count_chunks() == 5

    def test_delete_chunks_by_source(self, index: MemoryIndex):
        chunks = [
            Chunk(text="A", source_path="file1.md", content_hash="h1"),
            Chunk(text="B", source_path="file1.md", content_hash="h2"),
            Chunk(text="C", source_path="file2.md", content_hash="h3"),
        ]
        index.upsert_chunks(chunks)
        deleted = index.delete_chunks_by_source("file1.md")
        assert deleted == 2
        assert index.count_chunks() == 1

    def test_get_chunks_by_source(self, index: MemoryIndex):
        chunks = [
            Chunk(text="A", source_path="same.md", content_hash="h1", line_start=0),
            Chunk(text="B", source_path="same.md", content_hash="h2", line_start=5),
            Chunk(text="C", source_path="other.md", content_hash="h3"),
        ]
        index.upsert_chunks(chunks)
        result = index.get_chunks_by_source("same.md")
        assert len(result) == 2
        assert result[0].line_start < result[1].line_start  # Sorted

    def test_count_chunks_by_tier(self, index: MemoryIndex):
        chunks = [
            Chunk(text="A", source_path="a.md", content_hash="h1", memory_tier=MemoryTier.CORE),
            Chunk(text="B", source_path="b.md", content_hash="h2", memory_tier=MemoryTier.EPISODIC),
            Chunk(text="C", source_path="c.md", content_hash="h3", memory_tier=MemoryTier.EPISODIC),
        ]
        index.upsert_chunks(chunks)
        assert index.count_chunks(MemoryTier.CORE) == 1
        assert index.count_chunks(MemoryTier.EPISODIC) == 2
        assert index.count_chunks() == 3

    def test_get_all_content_hashes(self, index: MemoryIndex):
        chunks = [
            Chunk(text="A", source_path="a.md", content_hash="h1"),
            Chunk(text="B", source_path="b.md", content_hash="h2"),
        ]
        index.upsert_chunks(chunks)
        hashes = index.get_all_content_hashes()
        assert hashes == {"h1", "h2"}


class TestBM25Search:
    def test_search_finds_match(self, index: MemoryIndex):
        chunks = [
            Chunk(
                text="Projektmanagement für Entwicklerteams",
                source_path="a.md",
                content_hash="h1",
            ),
            Chunk(
                text="Haftpflichtversicherung für Familien", source_path="b.md", content_hash="h2"
            ),
        ]
        index.upsert_chunks(chunks)
        results = index.search_bm25("Projektmanagement")
        assert len(results) >= 1
        assert results[0][0] == chunks[0].id

    def test_search_empty_query(self, index: MemoryIndex):
        assert index.search_bm25("") == []
        assert index.search_bm25("   ") == []

    def test_search_no_match(self, index: MemoryIndex):
        index.upsert_chunks([Chunk(text="Hello", source_path="a.md", content_hash="h1")])
        results = index.search_bm25("xyznonsense")
        assert len(results) == 0

    def test_search_multiple_words(self, index: MemoryIndex):
        chunks = [
            Chunk(text="Müller hat eine BU Police", source_path="a.md", content_hash="h1"),
            Chunk(text="Schmidt hat Haftpflicht", source_path="b.md", content_hash="h2"),
        ]
        index.upsert_chunks(chunks)
        results = index.search_bm25("Müller BU")
        assert len(results) >= 1


class TestEmbeddings:
    def test_store_and_get(self, index: MemoryIndex):
        vec = [0.1, 0.2, 0.3]
        index.store_embedding("hash1", vec, "test-model")
        loaded = index.get_embedding("hash1")
        assert loaded is not None
        assert len(loaded) == 3
        assert abs(loaded[0] - 0.1) < 1e-6

    def test_get_nonexistent(self, index: MemoryIndex):
        assert index.get_embedding("nope") is None

    def test_get_all(self, index: MemoryIndex):
        index.store_embedding("h1", [1.0, 2.0], "m")
        index.store_embedding("h2", [3.0, 4.0], "m")
        all_emb = index.get_all_embeddings()
        assert len(all_emb) == 2
        assert "h1" in all_emb
        assert "h2" in all_emb

    def test_count(self, index: MemoryIndex):
        assert index.count_embeddings() == 0
        index.store_embedding("h1", [1.0], "m")
        assert index.count_embeddings() == 1


class TestEntities:
    def test_upsert_entity(self, index: MemoryIndex, sample_entity: Entity):
        index.upsert_entity(sample_entity)
        loaded = index.get_entity_by_id(sample_entity.id)
        assert loaded is not None
        assert loaded.name == "Hans Müller"
        assert loaded.type == "person"
        assert loaded.attributes["beruf"] == "Ingenieur"

    def test_search_by_name(self, index: MemoryIndex, sample_entity: Entity):
        index.upsert_entity(sample_entity)
        results = index.search_entities(name="Müller")
        assert len(results) == 1
        assert results[0].name == "Hans Müller"

    def test_search_by_type(self, index: MemoryIndex):
        e1 = Entity(type="person", name="Müller")
        e2 = Entity(type="company", name="TechCorp")
        index.upsert_entity(e1)
        index.upsert_entity(e2)
        persons = index.search_entities(entity_type="person")
        assert len(persons) == 1
        assert persons[0].name == "Müller"

    def test_delete_entity(self, index: MemoryIndex, sample_entity: Entity):
        index.upsert_entity(sample_entity)
        assert index.delete_entity(sample_entity.id)
        assert index.get_entity_by_id(sample_entity.id) is None

    def test_delete_cascades_relations(self, index: MemoryIndex):
        e1 = Entity(type="person", name="A")
        e2 = Entity(type="product", name="B")
        index.upsert_entity(e1)
        index.upsert_entity(e2)
        rel = Relation(source_entity=e1.id, relation_type="hat", target_entity=e2.id)
        index.upsert_relation(rel)
        assert index.count_relations() == 1
        index.delete_entity(e1.id)
        assert index.count_relations() == 0


class TestRelations:
    def test_upsert_relation(self, index: MemoryIndex):
        e1 = Entity(type="person", name="Müller")
        e2 = Entity(type="product", name="Cloud-Lizenz")
        index.upsert_entity(e1)
        index.upsert_entity(e2)

        rel = Relation(
            source_entity=e1.id,
            relation_type="hat_police",
            target_entity=e2.id,
            attributes={"lizenznummer": "LIZ-123"},
        )
        index.upsert_relation(rel)

        rels = index.get_relations_for_entity(e1.id)
        assert len(rels) == 1
        assert rels[0].relation_type == "hat_police"

    def test_get_relations_filtered(self, index: MemoryIndex):
        e1 = Entity(type="person", name="A")
        e2 = Entity(type="product", name="B")
        e3 = Entity(type="company", name="C")
        for e in [e1, e2, e3]:
            index.upsert_entity(e)

        r1 = Relation(source_entity=e1.id, relation_type="hat", target_entity=e2.id)
        r2 = Relation(source_entity=e1.id, relation_type="arbeitet_bei", target_entity=e3.id)
        index.upsert_relation(r1)
        index.upsert_relation(r2)

        hat_rels = index.get_relations_for_entity(e1.id, "hat")
        assert len(hat_rels) == 1

    def test_graph_traverse(self, index: MemoryIndex):
        # A -> B -> C
        e1 = Entity(type="person", name="A")
        e2 = Entity(type="person", name="B")
        e3 = Entity(type="person", name="C")
        for e in [e1, e2, e3]:
            index.upsert_entity(e)

        index.upsert_relation(
            Relation(source_entity=e1.id, relation_type="kennt", target_entity=e2.id)
        )
        index.upsert_relation(
            Relation(source_entity=e2.id, relation_type="kennt", target_entity=e3.id)
        )

        # Depth 1: Only B
        neighbors_1 = index.graph_traverse(e1.id, max_depth=1)
        names_1 = {e.name for e in neighbors_1}
        assert "B" in names_1

        # Depth 2: B and C
        neighbors_2 = index.graph_traverse(e1.id, max_depth=2)
        names_2 = {e.name for e in neighbors_2}
        assert "B" in names_2
        assert "C" in names_2


class TestMaintenance:
    def test_stats(self, index: MemoryIndex):
        s = index.stats()
        assert s["chunks"] == 0
        assert s["embeddings"] == 0
        assert s["entities"] == 0
        assert s["relations"] == 0

    def test_rebuild_fts(self, index: MemoryIndex):
        index.upsert_chunks([Chunk(text="Test", source_path="a.md", content_hash="h1")])
        index.rebuild_fts()  # Should not raise

    def test_close_and_reopen(self, db_path: Path):
        idx = MemoryIndex(db_path)
        idx.upsert_chunks([Chunk(text="Test", source_path="a.md", content_hash="h1")])
        idx.close()

        idx2 = MemoryIndex(db_path)
        assert idx2.count_chunks() == 1
        idx2.close()
