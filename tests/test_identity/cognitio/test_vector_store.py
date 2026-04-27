"""
tests/test_identity/cognitio/test_vector_store.py

Integration-light tests for cognithor.identity.cognitio.vector_store.
Uses a real ChromaDB PersistentClient with tmp_path for full isolation.

Skipped when chromadb is not installed — it lives in the `[identity]`
optional-dependency group of pyproject.toml, so CI's default-deps test legs
won't have it. Run `pip install -e ".[identity]"` (or just `pip install
chromadb`) to enable these tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from cognithor.identity.cognitio.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMB_X = [1.0, 0.0, 0.0]
EMB_Y = [0.0, 1.0, 0.0]
EMB_Z = [0.0, 0.0, 1.0]
EMB_NEAR_X = [1.0, 0.0, 0.1]  # very close to EMB_X under cosine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vs(tmp_path):
    """Fresh VectorStore backed by a tmp_path directory."""
    return VectorStore(str(tmp_path / "vs"))


# ---------------------------------------------------------------------------
# TestVectorStoreInit
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    def test_constructs_and_count_is_zero(self, tmp_path):
        store = VectorStore(str(tmp_path / "vs_init"))
        assert store.count() == 0

    def test_persist_dir_created_on_disk(self, tmp_path):
        d = tmp_path / "vs_disk"
        VectorStore(str(d))
        assert d.exists()

    def test_default_collection_name_is_memories(self, tmp_path):
        store = VectorStore(str(tmp_path / "vs_default"))
        assert store.collection_name == "memories"

    def test_custom_collection_name_accepted(self, tmp_path):
        store = VectorStore(str(tmp_path / "vs_custom"), collection_name="test_col")
        assert store.collection_name == "test_col"
        assert store.count() == 0


# ---------------------------------------------------------------------------
# TestAddAndQuery
# ---------------------------------------------------------------------------


class TestAddAndQuery:
    def test_add_increments_count(self, vs):
        vs.add("mem-1", EMB_X, {"tag": "a"})
        assert vs.count() == 1

    def test_add_same_id_twice_updates_not_duplicate(self, vs):
        vs.add("mem-dup", EMB_X, {"tag": "first"})
        vs.add("mem-dup", EMB_Y, {"tag": "second"})
        assert vs.count() == 1

    def test_query_returns_closest_first(self, vs):
        vs.add("far", EMB_Y, {"k": "v"})
        vs.add("near", EMB_NEAR_X, {"k": "v"})
        results = vs.query(EMB_X, n_results=10)
        assert results[0] == "near"

    def test_query_empty_store_returns_empty_list(self, tmp_path):
        store = VectorStore(str(tmp_path / "vs_empty"))
        assert store.query(EMB_X, n_results=5) == []

    def test_query_with_where_filter(self, vs):
        vs.add("episodic-1", EMB_X, {"memory_type": "episodic"})
        vs.add("semantic-1", EMB_Y, {"memory_type": "semantic"})
        results = vs.query(EMB_X, n_results=10, where={"memory_type": "episodic"})
        assert results == ["episodic-1"]

    def test_query_n_results_larger_than_total(self, vs):
        vs.add("only-one", EMB_X, {"k": "v"})
        results = vs.query(EMB_X, n_results=999)
        assert len(results) == 1
        assert results[0] == "only-one"


# ---------------------------------------------------------------------------
# TestUpdateMetadata
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    def test_update_metadata_merges_fields(self, vs):
        vs.add("mem-upd", EMB_X, {"field_a": "original", "field_b": 42})
        vs.update_metadata("mem-upd", {"field_a": "updated", "field_c": True})

        raw = vs._collection.get(ids=["mem-upd"], include=["metadatas"])
        meta = raw["metadatas"][0]
        assert meta["field_a"] == "updated"  # overwritten
        assert meta["field_b"] == 42  # preserved
        assert meta["field_c"] is True  # new field added

    def test_update_metadata_missing_id_is_noop(self, vs):
        # Should not raise
        vs.update_metadata("does-not-exist", {"some": "data"})
        assert vs.count() == 0


# ---------------------------------------------------------------------------
# TestDeleteAndExists
# ---------------------------------------------------------------------------


class TestDeleteAndExists:
    def test_delete_removes_record(self, vs):
        vs.add("to-delete", EMB_X, {"k": "v"})
        assert vs.exists("to-delete")
        vs.delete("to-delete")
        assert not vs.exists("to-delete")
        assert vs.count() == 0

    def test_delete_nonexistent_is_noop(self, vs):
        vs.delete("ghost-id")  # must not raise
        assert vs.count() == 0

    def test_exists_returns_false_for_unknown_id(self, vs):
        assert not vs.exists("never-added")


# ---------------------------------------------------------------------------
# TestGetAllIdsAndClear
# ---------------------------------------------------------------------------


class TestGetAllIdsAndClear:
    def test_get_all_ids_returns_all_added(self, vs):
        ids = {"alpha", "beta", "gamma"}
        for id_ in ids:
            vs.add(id_, EMB_X, {"tag": id_})
        assert set(vs.get_all_ids()) == ids

    def test_clear_empties_collection(self, vs):
        vs.add("x", EMB_X, {"tag": "x"})
        vs.add("y", EMB_Y, {"tag": "y"})
        vs.clear()
        assert vs.count() == 0
        assert vs.get_all_ids() == []


# ---------------------------------------------------------------------------
# TestCleanMetadata  (static — no ChromaDB needed)
# ---------------------------------------------------------------------------


class TestCleanMetadata:
    def test_primitive_types_pass_through(self):
        data = {"a": 1, "b": "hello", "c": True, "d": 3.14}
        result = VectorStore._clean_metadata(data)
        assert result == {"a": 1, "b": "hello", "c": True, "d": 3.14}

    def test_long_string_truncated_to_1024(self):
        long_str = "x" * 2000
        result = VectorStore._clean_metadata({"k": long_str})
        assert len(result["k"]) == 1024

    def test_list_comma_joined(self):
        result = VectorStore._clean_metadata({"tags": ["a", "b", "c"]})
        assert result["tags"] == "a,b,c"

    def test_list_over_100_items_truncated(self):
        big_list = list(range(150))
        result = VectorStore._clean_metadata({"nums": big_list})
        parts = result["nums"].split(",")
        assert len(parts) == 100
        assert parts[0] == "0"
        assert parts[-1] == "99"

    def test_none_becomes_empty_string(self):
        result = VectorStore._clean_metadata({"empty": None})
        assert result["empty"] == ""

    def test_arbitrary_object_coerced_to_str(self):
        class Obj:
            def __str__(self):
                return "my-object"

        result = VectorStore._clean_metadata({"obj": Obj()})
        assert result["obj"] == "my-object"
