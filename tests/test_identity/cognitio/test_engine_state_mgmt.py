"""Session 4 Phase 4 — state management + kill switch + admin control panel.

Scope: save_state / _load_state, check_kill_switch, cognitive_shutdown, soft_reset,
user_freeze / user_unfreeze, full_delete, admin_freeze / admin_unfreeze, force_save.
Requires [identity] optional deps (chromadb + sentence_transformers).
"""

import hashlib
import json
import os

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from cognithor.identity.cognitio.engine import (
    CognitioEngine,
    _hash_admin_key,
    _hash_kill_switch,
)
from cognithor.identity.cognitio.memory import MemoryRecord, MemoryType

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
# Helper
# ---------------------------------------------------------------------------


def _add_non_genesis_memory(eng: CognitioEngine, content: str = "test memory") -> MemoryRecord:
    """Add a plain (non-genesis) MemoryRecord to both memory_store and vector_store."""
    record = MemoryRecord(content=content, memory_type=MemoryType.SEMANTIC)
    record.embedding = eng.embedder.encode(content)
    eng.memory_store.add(record)
    eng.vector_store.add(
        record.id,
        record.embedding,
        {
            "memory_type": record.memory_type.value,
            "emotional_intensity": record.emotional_intensity,
            "emotional_valence": record.emotional_valence.value,
            "entrenchment": record.entrenchment,
            "is_anchor": record.is_anchor,
            "tags": ",".join(record.tags),
            "created_at": record.created_at.isoformat(),
        },
    )
    return record


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_state_writes_json_file(self, engine, tmp_path):
        """save_state() writes a JSON file at the configured memory_file path."""
        engine.save_state()
        memory_file = os.path.join(str(tmp_path), "memories.json")
        assert os.path.exists(memory_file), "memories.json must exist after save_state()"

    def test_saved_json_has_expected_keys(self, engine, tmp_path):
        """The saved JSON contains saved_at, memories, and all 11 sub-state keys."""
        engine.save_state()
        memory_file = os.path.join(str(tmp_path), "memories.json")
        with open(memory_file, encoding="utf-8") as f:
            data = json.load(f)

        expected_keys = {
            "saved_at",
            "cognitive_state",
            "personality",
            "memories",
            "temporal",
            "somatic",
            "epistemic",
            "narrative",
            "relational",
            "dream",
            "existential",
            "predictive",
        }
        assert expected_keys == set(data.keys())

    def test_save_is_atomic_no_tmp_files_remain(self, engine, tmp_path):
        """After a successful save_state(), no *.tmp files are left in data_dir."""
        engine.save_state()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp files found: {tmp_files}"

    def test_round_trip_memory_count_and_interactions(self, tmp_path):
        """save_state + fresh CognitioEngine at same data_dir preserves memory count."""
        eng1 = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        _add_non_genesis_memory(eng1, "round-trip memory")
        # Bump total_interactions so we can verify it round-trips.
        eng1.state.total_interactions = 42
        eng1.save_state()
        eng1._stop_consolidation_worker()

        eng2 = CognitioEngine(data_dir=str(tmp_path), llm_client=None)
        try:
            assert eng2.memory_store.count() == 8  # 7 genesis + 1 added
            assert eng2.state.total_interactions == 42
        finally:
            eng2._stop_consolidation_worker()

    def test_load_state_missing_file_silent(self, engine, tmp_path):
        """_load_state on a non-existent path returns silently without raising."""
        missing = str(tmp_path / "does_not_exist.json")
        engine._load_state(missing)  # must not raise


# ---------------------------------------------------------------------------
# TestCheckKillSwitch
# ---------------------------------------------------------------------------


class TestCheckKillSwitch:
    def test_returns_false_when_hash_is_none(self, engine):
        """check_kill_switch returns False when _kill_switch_hash is None."""
        engine._kill_switch_hash = None
        assert engine.check_kill_switch("anything") is False

    def test_returns_true_for_correct_passphrase(self, engine):
        """check_kill_switch returns True when the passphrase matches the hash."""
        engine._kill_switch_hash = _hash_kill_switch("secret")
        assert engine.check_kill_switch("secret") is True

    def test_returns_false_for_wrong_passphrase(self, engine):
        """check_kill_switch returns False for an incorrect passphrase."""
        engine._kill_switch_hash = _hash_kill_switch("secret")
        assert engine.check_kill_switch("wrong") is False

    def test_uses_hmac_compare_digest(self, engine):
        """Smoke test: check_kill_switch delegates to hmac.compare_digest (source verified)."""
        import inspect

        import cognithor.identity.cognitio.engine as engine_mod

        source = inspect.getsource(engine_mod.CognitioEngine.check_kill_switch)
        assert "hmac.compare_digest" in source


# ---------------------------------------------------------------------------
# TestCognitiveShutdown
# ---------------------------------------------------------------------------


class TestCognitiveShutdown:
    def test_result_dict_structure(self, engine):
        """cognitive_shutdown() returns the expected keys with correct semantics."""
        _add_non_genesis_memory(engine)
        result = engine.cognitive_shutdown()
        assert result == {"success": True, "genesis_preserved": 7, "cleared": 1}

    def test_state_frozen_and_strength_zeroed(self, engine):
        """After cognitive_shutdown, state.is_frozen is True and character_strength is 0.0."""
        engine.cognitive_shutdown()
        assert engine.state.is_frozen is True
        assert engine.state.character_strength == 0.0

    def test_genesis_anchors_preserved_after_shutdown(self, engine):
        """After cognitive_shutdown, get_absolute_cores() still returns 7 records."""
        _add_non_genesis_memory(engine)
        engine.cognitive_shutdown()
        assert len(engine.memory_store.get_absolute_cores()) == 7

    def test_consolidation_worker_stopped_after_shutdown(self, engine):
        """After cognitive_shutdown, the consolidation thread is no longer alive."""
        engine.cognitive_shutdown()
        assert engine._consolidation_thread is not None
        assert engine._consolidation_thread.is_alive() is False


# ---------------------------------------------------------------------------
# TestSoftReset
# ---------------------------------------------------------------------------


class TestSoftReset:
    def test_result_dict_structure(self, engine):
        """soft_reset() clears 1 added memory and preserves 7 genesis anchors."""
        _add_non_genesis_memory(engine)
        result = engine.soft_reset()
        assert result == {"cleared": 1, "genesis_preserved": 7}

    def test_cognitive_state_reset(self, engine):
        """After soft_reset, character_strength is 0.0 and total_interactions is 0."""
        engine.state.total_interactions = 5
        engine.soft_reset()
        assert engine.state.character_strength == 0.0
        assert engine.state.total_interactions == 0

    def test_not_frozen_after_soft_reset(self, engine):
        """After soft_reset, state.is_frozen is False (system stays active)."""
        engine.state.is_frozen = True
        engine.soft_reset()
        assert engine.state.is_frozen is False

    def test_consolidation_worker_alive_after_soft_reset(self, engine):
        """After soft_reset, the consolidation worker has been re-spawned and is alive."""
        engine.soft_reset()
        assert engine._consolidation_thread is not None
        assert engine._consolidation_thread.is_alive() is True


# ---------------------------------------------------------------------------
# TestFreezeUnfreeze
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    def test_user_freeze_result_and_state(self, engine):
        """user_freeze() returns expected dict and sets state.is_frozen to True."""
        result = engine.user_freeze()
        assert result == {"frozen": True, "memories_preserved": 7}
        assert engine.state.is_frozen is True

    def test_user_unfreeze_result_and_state(self, engine):
        """user_unfreeze() returns {'frozen': False} and clears the frozen flag."""
        engine.state.is_frozen = True
        result = engine.user_unfreeze()
        assert result == {"frozen": False}
        assert engine.state.is_frozen is False


# ---------------------------------------------------------------------------
# TestFullDelete
# ---------------------------------------------------------------------------


class TestFullDelete:
    def test_data_deleted_flag_set(self, engine):
        """After full_delete, _data_deleted is True."""
        engine.full_delete()
        assert engine._data_deleted is True

    def test_json_file_deleted(self, engine, tmp_path):
        """After full_delete, memories.json is gone from data_dir."""
        engine.save_state()  # ensure file exists first
        engine.full_delete()
        memory_file = tmp_path / "memories.json"
        assert not memory_file.exists(), "memories.json must be deleted by full_delete()"

    def test_result_contains_data_wiped_true(self, engine):
        """full_delete() result dict has data_wiped: True."""
        result = engine.full_delete()
        assert result.get("data_wiped") is True


# ---------------------------------------------------------------------------
# TestAdminFreeze
# ---------------------------------------------------------------------------


class TestAdminFreeze:
    def test_freeze_fails_without_env_var(self, engine, monkeypatch):
        """admin_freeze returns failure dict when IMP_ADMIN_KEY_HASH is not set."""
        monkeypatch.delenv("IMP_ADMIN_KEY_HASH", raising=False)
        result = engine.admin_freeze("anything")
        assert result == {"success": False, "reason": "Invalid admin key"}

    def test_freeze_succeeds_with_pbkdf2_hash(self, engine, monkeypatch):
        """admin_freeze succeeds with a correct PBKDF2-hashed key and freezes state."""
        monkeypatch.setenv("IMP_ADMIN_KEY_HASH", _hash_admin_key("k"))
        result = engine.admin_freeze("k")
        assert result == {"success": True, "frozen": True, "by": "admin"}
        assert engine.state.is_frozen is True

    def test_freeze_succeeds_with_plain_sha256_fallback(self, engine, monkeypatch):
        """admin_freeze succeeds with a plain SHA-256 fallback hash (legacy migration path)."""
        plain_hash = hashlib.sha256(b"k").hexdigest()
        monkeypatch.setenv("IMP_ADMIN_KEY_HASH", plain_hash)
        result = engine.admin_freeze("k")
        assert result["success"] is True

    def test_unfreeze_succeeds(self, engine, monkeypatch):
        """admin_unfreeze with a valid key unfreezes state and returns success."""
        monkeypatch.setenv("IMP_ADMIN_KEY_HASH", _hash_admin_key("k"))
        engine.state.is_frozen = True
        result = engine.admin_unfreeze("k")
        assert result == {"success": True, "frozen": False, "by": "admin"}
        assert engine.state.is_frozen is False

    def test_unfreeze_fails_without_env_var(self, engine, monkeypatch):
        """admin_unfreeze returns failure dict when IMP_ADMIN_KEY_HASH is not set."""
        monkeypatch.delenv("IMP_ADMIN_KEY_HASH", raising=False)
        result = engine.admin_unfreeze("anything")
        assert result == {"success": False, "reason": "Invalid admin key"}


# ---------------------------------------------------------------------------
# TestForceSave
# ---------------------------------------------------------------------------


class TestForceSave:
    def test_force_save_writes_json_and_no_error(self, engine, tmp_path):
        """force_save() runs without error and produces a memories.json on disk."""
        engine.force_save()
        memory_file = tmp_path / "memories.json"
        assert memory_file.exists(), "force_save() must write memories.json"
