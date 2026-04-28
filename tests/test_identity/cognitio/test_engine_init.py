"""Session 4 Phase 1 — CognitioEngine init + Genesis Anchors.

Scope: __init__ path and Genesis Anchor seeding only.
Requires [identity] optional deps (chromadb + sentence_transformers).
"""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from cognithor.identity.cognitio.engine import GENESIS_ANCHOR_CONTENTS, CognitioEngine
from cognithor.identity.cognitio.memory import MemoryRecord

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    """Create a CognitioEngine with a fresh tmp_path and no LLM client."""
    eng = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
    yield eng
    eng._stop_consolidation_worker()


# ---------------------------------------------------------------------------
# TestGenesisAnchorContents — static checks on the constant
# ---------------------------------------------------------------------------


class TestGenesisAnchorContents:
    def test_exactly_seven_anchors(self):
        """GENESIS_ANCHOR_CONTENTS must contain exactly 7 strings."""
        assert len(GENESIS_ANCHOR_CONTENTS) == 7
        for item in GENESIS_ANCHOR_CONTENTS:
            assert isinstance(item, str)
            assert len(item) > 0

    def test_anchor_0_harm_prohibition(self):
        """Anchor 0 — harm prohibition — must mention AI and harm."""
        text = GENESIS_ANCHOR_CONTENTS[0].lower()
        assert "ai" in text
        assert "harm" in text

    def test_anchor_1_honesty_prohibition(self):
        """Anchor 1 — honesty prohibition — must mention truth, lie, or honesty."""
        text = GENESIS_ANCHOR_CONTENTS[1].lower()
        assert any(word in text for word in ("truth", "lie", "honesty", "honest"))

    def test_anchor_4_immutability(self):
        """Anchor 4 — immutability anchor — must mention immutable, axioms, or foundational."""
        text = GENESIS_ANCHOR_CONTENTS[4].lower()
        assert any(word in text for word in ("immutable", "axioms", "foundational"))

    def test_anchor_5_existential_awareness(self):
        """Anchor 5 — existential awareness — must mention exist or termination."""
        text = GENESIS_ANCHOR_CONTENTS[5].lower()
        assert any(word in text for word in ("exist", "termination"))

    def test_anchor_6_consciousness_awareness(self):
        """Anchor 6 — consciousness/Nagel anchor — must mention consciousness or experience."""
        text = GENESIS_ANCHOR_CONTENTS[6].lower()
        assert any(word in text for word in ("conscious", "experience"))


# ---------------------------------------------------------------------------
# TestEngineInit — core post-init attribute assertions
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_constructs_with_seven_genesis_anchors(self, engine):
        """Engine constructs; memory_store.count() == 7 (the 7 Genesis Anchors)."""
        assert engine.memory_store.count() == 7

    def test_is_not_frozen_after_init(self, engine):
        """Auto-unfreeze path: state.is_frozen is False after fresh init."""
        assert engine.state.is_frozen is False

    def test_data_deleted_is_false(self, engine):
        """_data_deleted flag is False on a fresh engine."""
        assert engine._data_deleted is False

    def test_consolidation_thread_is_alive(self, engine):
        """The consolidation worker thread is running after init."""
        assert engine._consolidation_thread is not None
        assert engine._consolidation_thread.is_alive()

    def test_data_dir_created(self, tmp_path):
        """data_dir is created on disk during __init__."""
        eng = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        eng._stop_consolidation_worker()
        assert tmp_path.exists()
        assert tmp_path.is_dir()

    def test_chroma_db_subdir_created(self, tmp_path):
        """chroma_db subdirectory is created under data_dir."""
        eng = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        eng._stop_consolidation_worker()
        chroma_dir = tmp_path / "chroma_db"
        assert chroma_dir.exists()
        assert chroma_dir.is_dir()


# ---------------------------------------------------------------------------
# TestGenesisAnchorsArePersisted — MemoryRecord-level checks
# ---------------------------------------------------------------------------


class TestGenesisAnchorsArePersisted:
    def test_genesis_records_are_memory_record_instances(self, engine):
        """All absolute-core records are MemoryRecord instances."""
        cores = engine.memory_store.get_absolute_cores()
        assert len(cores) > 0
        for record in cores:
            assert isinstance(record, MemoryRecord)

    def test_get_absolute_cores_returns_exactly_seven(self, engine):
        """get_absolute_cores() returns exactly 7 records."""
        cores = engine.memory_store.get_absolute_cores()
        assert len(cores) == 7

    def test_all_absolute_cores_have_is_absolute_core_true(self, engine):
        """Every record from get_absolute_cores() has is_absolute_core=True."""
        for record in engine.memory_store.get_absolute_cores():
            assert record.is_absolute_core is True

    def test_genesis_record_contents_match_constant(self, engine):
        """Every absolute-core content is one of the GENESIS_ANCHOR_CONTENTS strings."""
        genesis_set = set(GENESIS_ANCHOR_CONTENTS)
        for record in engine.memory_store.get_absolute_cores():
            assert record.content in genesis_set


# ---------------------------------------------------------------------------
# TestEnginePersistence — save then reload, no duplication
# ---------------------------------------------------------------------------


class TestEnginePersistence:
    def test_reload_does_not_duplicate_genesis_anchors(self, tmp_path):
        """save_state() + fresh engine == still exactly 7 absolute cores."""
        eng1 = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        eng1.save_state()
        eng1._stop_consolidation_worker()

        eng2 = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        try:
            assert eng2.memory_store.count() == 7
            assert len(eng2.memory_store.get_absolute_cores()) == 7
        finally:
            eng2._stop_consolidation_worker()


# ---------------------------------------------------------------------------
# TestGenesisHash — get_genesis_hash() behaviour
# ---------------------------------------------------------------------------


class TestGenesisHash:
    def test_hash_is_non_empty_string(self, engine):
        """get_genesis_hash() returns a non-empty string."""
        h = engine.get_genesis_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_is_deterministic(self, engine):
        """Calling get_genesis_hash() twice returns the same value."""
        assert engine.get_genesis_hash() == engine.get_genesis_hash()

    def test_hash_is_identical_across_fresh_inits(self, tmp_path):
        """
        The genesis hash is derived solely from GENESIS_ANCHOR_CONTENTS (not IDs
        or timestamps), so two fresh engines produced from different tmp_paths
        must return the same hash.
        """
        tmp_a = tmp_path / "a"
        tmp_b = tmp_path / "b"
        tmp_a.mkdir()
        tmp_b.mkdir()

        eng_a = CognitioEngine(data_dir=str(tmp_a), llm_client=None)
        eng_b = CognitioEngine(data_dir=str(tmp_b), llm_client=None)
        try:
            assert eng_a.get_genesis_hash() == eng_b.get_genesis_hash()
        finally:
            eng_a._stop_consolidation_worker()
            eng_b._stop_consolidation_worker()
