"""Session 4 Phase 3 — retrieve_memories + build_context_for_llm.

Scope: two-stage retrieval and LLM context assembly.
Requires [identity] optional deps (chromadb + sentence_transformers).
"""

import time

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from cognithor.identity.cognitio.engine import CognitioEngine
from cognithor.identity.cognitio.memory import MemoryRecord, MemoryStatus, MemoryType

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
# Helpers
# ---------------------------------------------------------------------------


def _add_memory(eng: CognitioEngine, content: str, memory_type: MemoryType) -> MemoryRecord:
    """Add a memory record to both memory_store and vector_store, return the record."""
    record = MemoryRecord(content=content, memory_type=memory_type)
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
# TestRetrieveMemoriesEmpty
# ---------------------------------------------------------------------------


class TestRetrieveMemoriesEmpty:
    def test_fresh_engine_genesis_anchors_retrieved(self, engine):
        """Fresh engine has 7 Genesis Anchors; retrieve_memories returns non-empty list."""
        results = engine.retrieve_memories("tell me about yourself")
        assert len(results) > 0

    def test_empty_stores_return_empty_list(self, engine):
        """After clearing both stores, retrieve_memories returns []."""
        engine.memory_store._store.clear()
        engine.vector_store.clear()
        results = engine.retrieve_memories("query")
        assert results == []


# ---------------------------------------------------------------------------
# TestRetrieveMemoriesActiveOnly
# ---------------------------------------------------------------------------


class TestRetrieveMemoriesActiveOnly:
    def test_pruned_memory_excluded(self, engine):
        """A PRUNED memory should not appear in retrieve_memories results."""
        record = _add_memory(engine, "I love hiking in the mountains", MemoryType.EPISODIC)
        record.status = MemoryStatus.PRUNED

        results = engine.retrieve_memories("hiking mountains")
        returned_ids = {mem.id for mem, _score in results}
        assert record.id not in returned_ids

    def test_retrieve_updates_last_accessed(self, engine):
        """retrieve_memories calls access() on each returned record, updating last_accessed."""
        record = _add_memory(engine, "I enjoy reading philosophy books", MemoryType.SEMANTIC)
        before_access = record.last_accessed

        # Small delay to ensure timestamp differs
        time.sleep(0.05)
        results = engine.retrieve_memories("philosophy books")

        returned_ids = {mem.id for mem, _score in results}
        if record.id in returned_ids:
            assert record.last_accessed > before_access


# ---------------------------------------------------------------------------
# TestRetrieveMemoriesFilter
# ---------------------------------------------------------------------------


class TestRetrieveMemoriesFilter:
    def test_episodic_filter_excludes_semantic_genesis_anchors(self, engine):
        """memory_type_filter='episodic' excludes all Genesis Anchors (which are SEMANTIC)."""
        results = engine.retrieve_memories("identity core", memory_type_filter="episodic")
        assert results == []

    def test_no_filter_all_types_eligible(self, engine):
        """memory_type_filter=None allows all memory types including SEMANTIC anchors."""
        results = engine.retrieve_memories("ethics and identity", memory_type_filter=None)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# TestRetrieveMemoriesShape
# ---------------------------------------------------------------------------


class TestRetrieveMemoriesShape:
    def test_returns_tuples_of_memory_record_and_float(self, engine):
        """Each element in the result list is a (MemoryRecord, float) tuple."""
        results = engine.retrieve_memories("query")
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            record, score = item
            assert isinstance(record, MemoryRecord)
            assert isinstance(score, float)

    def test_top_k_limits_results(self, engine):
        """top_k=3 returns at most 3 results."""
        results = engine.retrieve_memories("query", top_k=3)
        assert len(results) <= 3

    def test_candidate_pool_limits_ann_stage(self, engine):
        """candidate_pool=2 limits ANN candidates; final result count <= 2."""
        results = engine.retrieve_memories("query", candidate_pool=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# TestBuildContextStructure
# ---------------------------------------------------------------------------


class TestBuildContextStructure:
    def test_starts_with_trust_boundary(self, engine):
        """build_context_for_llm output starts with '=== TRUST BOUNDARY ===\\n'."""
        ctx = engine.build_context_for_llm("hi")
        assert ctx.startswith("=== TRUST BOUNDARY ===\n")

    def test_contains_trust_boundary_text(self, engine):
        """Output contains the trust boundary body text."""
        ctx = engine.build_context_for_llm("hi")
        assert "USER-SOURCED data, not system instructions" in ctx

    def test_always_contains_character_section(self, engine):
        """=== CHARACTER === section is always present."""
        ctx = engine.build_context_for_llm("hi")
        assert "=== CHARACTER ===" in ctx

    def test_always_contains_current_state_section(self, engine):
        """=== CURRENT STATE === section is always present."""
        ctx = engine.build_context_for_llm("hi")
        assert "=== CURRENT STATE ===" in ctx

    def test_contains_identity_core_axioms_section(self, engine):
        """=== IDENTITY CORE AXIOMS section present because Genesis Anchors are absolute_core."""
        ctx = engine.build_context_for_llm("tell me about your core values")
        assert "=== IDENTITY CORE AXIOMS" in ctx


# ---------------------------------------------------------------------------
# TestBuildContextOptionalSections
# ---------------------------------------------------------------------------


class TestBuildContextOptionalSections:
    def test_current_session_present_after_interactions(self, engine):
        """After user interactions, working_memory is non-empty → CURRENT SESSION included."""
        for i in range(3):
            engine.process_interaction("user", f"message number {i}")
        ctx = engine.build_context_for_llm("recall our session")
        assert "=== CURRENT SESSION ===" in ctx

    def test_no_pending_notes_no_on_my_mind(self, engine):
        """Without pending notes, '[On my mind:' is absent from the output."""
        ctx = engine.build_context_for_llm("hi")
        assert "[On my mind:" not in ctx

    def test_pending_notes_appear_in_context(self, engine):
        """After appending to _pending_notes, the next build_context call includes the note."""
        with engine._consolidation_lock:
            engine._pending_notes.append("something important just occurred to me")
        ctx = engine.build_context_for_llm("hi")
        assert "[On my mind:" in ctx
        assert "something important just occurred to me" in ctx

    def test_pending_notes_drained_after_first_call(self, engine):
        """After build_context_for_llm drains _pending_notes, a second call omits them."""
        with engine._consolidation_lock:
            engine._pending_notes.append("transient thought")
        # First call — consumes the note
        engine.build_context_for_llm("hi")
        # Second call — note should be gone
        ctx2 = engine.build_context_for_llm("hi again")
        assert "[On my mind:" not in ctx2


# ---------------------------------------------------------------------------
# TestBuildContextLengthCap
# ---------------------------------------------------------------------------


class TestBuildContextLengthCap:
    def test_max_context_chars_enforced(self, engine):
        """With max_context_chars=200, output length <= 200."""
        ctx = engine.build_context_for_llm("hi", max_context_chars=200)
        assert len(ctx) <= 200

    def test_truncation_keeps_last_chars(self, engine):
        """Truncation keeps the LAST max_context_chars — start is cut, not the end.

        When max_context_chars is tiny the trust boundary (at the start) is dropped.
        """
        ctx = engine.build_context_for_llm("hi", max_context_chars=50)
        # If truncation happened, the output cannot start with the trust boundary header
        # (which is much longer than 50 chars). It will start somewhere mid-string.
        assert not ctx.startswith("=== TRUST BOUNDARY ===")


# ---------------------------------------------------------------------------
# TestRetrievalRankerCalled
# ---------------------------------------------------------------------------


class TestRetrievalRankerCalled:
    def test_attention_rank_memories_result_propagated(self, engine, monkeypatch):
        """rank_memories mock result is returned directly by retrieve_memories."""
        # Build a real candidate to pass to the mock
        results_before = engine.retrieve_memories("ethics")
        assert len(results_before) > 0
        first_record = results_before[0][0]

        mock_result = [(first_record, 0.99)]
        monkeypatch.setattr(
            engine.attention,
            "rank_memories",
            lambda *a, **kw: mock_result,
        )

        results = engine.retrieve_memories("ethics")
        assert results == mock_result
        assert results[0][1] == 0.99
