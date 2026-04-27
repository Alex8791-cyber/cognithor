"""
tests/test_identity/cognitio/test_memory.py

Pure-unit tests for cognithor.identity.cognitio.memory.
"""

from __future__ import annotations

import time

import pytest

from cognithor.identity.cognitio.memory import (
    MemoryRecord,
    MemoryStatus,
    MemoryStore,
    MemoryType,
    MemoryValence,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestMemoryTypeEnum:
    def test_all_six_values(self):
        values = {m.value for m in MemoryType}
        assert values == {
            "episodic",
            "semantic",
            "emotional",
            "procedural",
            "relational",
            "evolution",
        }

    def test_str_enum_protocol(self):
        # str-Enum: equality with the raw string value works
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        # .value returns the plain string
        assert MemoryType.EPISODIC.value == "episodic"


class TestMemoryValenceEnum:
    def test_all_three_values(self):
        values = {m.value for m in MemoryValence}
        assert values == {"positive", "negative", "neutral"}

    def test_str_enum_protocol(self):
        # str-Enum: equality with the raw string value works
        assert MemoryValence.POSITIVE == "positive"
        assert MemoryValence.NEGATIVE == "negative"
        assert MemoryValence.POSITIVE.value == "positive"


class TestMemoryStatusEnum:
    def test_all_six_values(self):
        values = {m.value for m in MemoryStatus}
        assert values == {
            "active",
            "pending",
            "contradicted",
            "superseded",
            "pruned",
            "ambivalent",
        }

    def test_str_enum_protocol(self):
        # str-Enum: equality with the raw string value works
        assert MemoryStatus.ACTIVE == "active"
        assert MemoryStatus.PRUNED == "pruned"
        assert MemoryStatus.ACTIVE.value == "active"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def record() -> MemoryRecord:
    return MemoryRecord(content="Test memory content")


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore()


# ---------------------------------------------------------------------------
# MemoryRecord construction
# ---------------------------------------------------------------------------


class TestMemoryRecordConstruction:
    def test_minimal_construction_defaults(self, record: MemoryRecord):
        assert record.content == "Test memory content"
        assert record.memory_type == MemoryType.SEMANTIC
        assert record.confidence == 0.5
        assert record.entrenchment == 0.1
        assert record.reinforcement_count == 0
        assert record.contradiction_count == 0
        assert record.status == MemoryStatus.ACTIVE
        assert record.emotional_valence == MemoryValence.NEUTRAL
        assert record.is_anchor is False
        assert record.is_absolute_core is False
        assert record.tags == []
        assert record.arweave_uri is None

    def test_content_required(self):
        with pytest.raises(TypeError):
            MemoryRecord()  # type: ignore[call-arg]

    def test_auto_uuid_unique(self):
        r1 = MemoryRecord(content="alpha")
        r2 = MemoryRecord(content="beta")
        assert r1.id != r2.id
        assert len(r1.id) == 36  # UUID4 canonical form

    def test_custom_memory_type(self):
        r = MemoryRecord(content="episode", memory_type=MemoryType.EPISODIC)
        assert r.memory_type == MemoryType.EPISODIC


# ---------------------------------------------------------------------------
# MemoryRecord.reinforce()
# ---------------------------------------------------------------------------


class TestReinforce:
    def test_increments_reinforcement_count(self, record: MemoryRecord):
        record.reinforce()
        assert record.reinforcement_count == 1
        record.reinforce()
        assert record.reinforcement_count == 2

    def test_default_delta_increases_entrenchment(self, record: MemoryRecord):
        before = record.entrenchment
        record.reinforce()
        assert record.entrenchment == pytest.approx(before + 0.08)

    def test_custom_delta(self, record: MemoryRecord):
        before = record.entrenchment
        record.reinforce(delta=0.20)
        assert record.entrenchment == pytest.approx(before + 0.20)

    def test_caps_at_one(self):
        r = MemoryRecord(content="saturated", entrenchment=0.95)
        r.reinforce(delta=0.10)
        assert r.entrenchment == 1.0

    def test_updates_last_reinforced_timestamp(self, record: MemoryRecord):
        before = record.last_reinforced
        time.sleep(0.01)
        record.reinforce()
        assert record.last_reinforced >= before

    def test_updates_last_accessed_timestamp(self, record: MemoryRecord):
        before = record.last_accessed
        time.sleep(0.01)
        record.reinforce()
        assert record.last_accessed >= before


# ---------------------------------------------------------------------------
# MemoryRecord.access()
# ---------------------------------------------------------------------------


class TestAccess:
    def test_access_updates_last_accessed(self, record: MemoryRecord):
        before = record.last_accessed
        time.sleep(0.01)
        record.access()
        assert record.last_accessed > before

    def test_access_does_not_change_last_reinforced(self, record: MemoryRecord):
        before_lr = record.last_reinforced
        before_rc = record.reinforcement_count
        time.sleep(0.01)
        record.access()
        assert record.last_reinforced == before_lr
        assert record.reinforcement_count == before_rc


# ---------------------------------------------------------------------------
# MemoryRecord.days_since_*
# ---------------------------------------------------------------------------


class TestDaysSince:
    def test_days_since_creation_small_for_fresh_record(self, record: MemoryRecord):
        result = record.days_since_creation()
        assert 0.0 <= result < 0.01

    def test_days_since_access_small_for_fresh_record(self, record: MemoryRecord):
        result = record.days_since_access()
        assert 0.0 <= result < 0.01


# ---------------------------------------------------------------------------
# MemoryRecord.to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_dict_from_dict_preserves_all_fields(self):
        original = MemoryRecord(
            content="Round-trip content",
            memory_type=MemoryType.EMOTIONAL,
            confidence=0.75,
            entrenchment=0.4,
            emotional_intensity=0.6,
            emotional_valence=MemoryValence.POSITIVE,
            is_anchor=True,
            is_absolute_core=True,
            reinforcement_count=3,
            contradiction_count=1,
            source_type="external_fact",
            source_trust_level=0.9,
            reality_check_score=0.8,
            arweave_uri="ar://test",
            tags=["philosophy", "ethics"],
            status=MemoryStatus.PENDING,
            temporal_density=0.5,
            is_ambivalent=True,
        )
        d = original.to_dict()
        restored = MemoryRecord.from_dict(d)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.confidence == original.confidence
        assert restored.entrenchment == original.entrenchment
        assert restored.emotional_intensity == original.emotional_intensity
        assert restored.emotional_valence == original.emotional_valence
        assert restored.is_anchor == original.is_anchor
        assert restored.is_absolute_core == original.is_absolute_core
        assert restored.reinforcement_count == original.reinforcement_count
        assert restored.contradiction_count == original.contradiction_count
        assert restored.source_type == original.source_type
        assert restored.source_trust_level == original.source_trust_level
        assert restored.reality_check_score == original.reality_check_score
        assert restored.arweave_uri == original.arweave_uri
        assert restored.tags == original.tags
        assert restored.status == original.status
        assert restored.temporal_density == original.temporal_density
        assert restored.is_ambivalent == original.is_ambivalent
        assert restored.created_at == original.created_at
        assert restored.last_reinforced == original.last_reinforced
        assert restored.last_accessed == original.last_accessed

    def test_from_dict_missing_optional_fields_uses_defaults(self):
        minimal = {"content": "Minimal"}
        r = MemoryRecord.from_dict(minimal)
        assert r.content == "Minimal"
        assert r.memory_type == MemoryType.SEMANTIC
        assert r.confidence == 0.5
        assert r.entrenchment == 0.1
        assert r.tags == []
        assert r.status == MemoryStatus.ACTIVE

    def test_from_dict_invalid_status_falls_back_to_active(self):
        d = {"content": "Bad status", "status": "nonexistent_status"}
        r = MemoryRecord.from_dict(d)
        assert r.status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class TestMemoryStoreAddGet:
    def test_add_then_get_returns_same_record(self, store: MemoryStore, record: MemoryRecord):
        store.add(record)
        retrieved = store.get(record.id)
        assert retrieved is record

    def test_get_missing_id_returns_none(self, store: MemoryStore):
        assert store.get("nonexistent-id") is None


class TestMemoryStoreUpdate:
    def test_update_replaces_record_by_id(self, store: MemoryStore):
        r = MemoryRecord(content="original")
        store.add(r)
        r.content = "modified"
        store.update(r)
        assert store.get(r.id).content == "modified"


class TestMemoryStoreDelete:
    def test_delete_existing_returns_true(self, store: MemoryStore, record: MemoryRecord):
        store.add(record)
        result = store.delete(record.id)
        assert result is True
        assert store.get(record.id) is None

    def test_delete_missing_returns_false(self, store: MemoryStore):
        assert store.delete("nonexistent-id") is False


class TestMemoryStoreFilters:
    def test_get_by_type_filters_correctly(self, store: MemoryStore):
        episodic = MemoryRecord(content="episode", memory_type=MemoryType.EPISODIC)
        semantic = MemoryRecord(content="concept", memory_type=MemoryType.SEMANTIC)
        store.add(episodic)
        store.add(semantic)
        result = store.get_by_type(MemoryType.EPISODIC)
        assert len(result) == 1
        assert result[0] is episodic

    def test_get_all_active_excludes_non_active(self, store: MemoryStore):
        active = MemoryRecord(content="active", status=MemoryStatus.ACTIVE)
        pruned = MemoryRecord(content="pruned", status=MemoryStatus.PRUNED)
        superseded = MemoryRecord(content="superseded", status=MemoryStatus.SUPERSEDED)
        contradicted = MemoryRecord(content="contradicted", status=MemoryStatus.CONTRADICTED)
        for r in (active, pruned, superseded, contradicted):
            store.add(r)
        result = store.get_all_active()
        ids = {r.id for r in result}
        assert active.id in ids
        assert pruned.id not in ids
        assert superseded.id not in ids
        assert contradicted.id not in ids

    def test_get_absolute_cores_returns_only_core_records(self, store: MemoryStore):
        core = MemoryRecord(content="core", is_absolute_core=True)
        normal = MemoryRecord(content="normal")
        store.add(core)
        store.add(normal)
        result = store.get_absolute_cores()
        assert len(result) == 1
        assert result[0] is core


class TestMemoryStoreCounts:
    def test_count_and_count_active_agree(self, store: MemoryStore):
        r1 = MemoryRecord(content="a", status=MemoryStatus.ACTIVE)
        r2 = MemoryRecord(content="b", status=MemoryStatus.PRUNED)
        r3 = MemoryRecord(content="c", status=MemoryStatus.ACTIVE)
        store.add(r1)
        store.add(r2)
        store.add(r3)
        assert store.count() == 3
        assert store.count_active() == 2

    def test_empty_store(self, store: MemoryStore):
        assert store.count() == 0
        assert store.count_active() == 0


class TestMemoryStoreRoundTrip:
    def test_to_dict_load_from_dict_preserves_three_records(self, store: MemoryStore):
        records = [
            MemoryRecord(content="alpha", memory_type=MemoryType.EPISODIC, tags=["a"]),
            MemoryRecord(
                content="beta",
                memory_type=MemoryType.SEMANTIC,
                status=MemoryStatus.PENDING,
            ),
            MemoryRecord(
                content="gamma",
                memory_type=MemoryType.PROCEDURAL,
                is_absolute_core=True,
            ),
        ]
        for r in records:
            store.add(r)

        serialized = store.to_dict()
        new_store = MemoryStore()
        new_store.load_from_dict(serialized)

        assert new_store.count() == 3
        for r in records:
            restored = new_store.get(r.id)
            assert restored is not None
            assert restored.content == r.content
            assert restored.memory_type == r.memory_type
            assert restored.status == r.status
            assert restored.is_absolute_core == r.is_absolute_core
