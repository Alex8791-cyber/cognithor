"""
tests/test_identity/cognitio/test_epistemic.py

Pure-unit tests for cognithor.identity.cognitio.epistemic.
"""

from __future__ import annotations

import pytest

from cognithor.identity.cognitio.epistemic import EpistemicMap
from cognithor.identity.cognitio.memory import MemoryRecord, MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def em() -> EpistemicMap:
    return EpistemicMap()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_new_map_has_zero_topics(self, em: EpistemicMap):
        assert em.topic_count() == 0

    def test_unknown_topic_returns_default_confidence(self, em: EpistemicMap):
        assert em.get_confidence("anything") == pytest.approx(0.5)

    def test_custom_default_confidence(self):
        em2 = EpistemicMap(default_confidence=0.7)
        assert em2.get_confidence("unknown") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# update() — outcome deltas
# ---------------------------------------------------------------------------


class TestUpdateOutcomes:
    def test_added_raises_confidence(self, em: EpistemicMap):
        em.update("foo", "added")
        assert em.get_confidence("foo") == pytest.approx(0.55)

    def test_reinforced_raises_confidence(self, em: EpistemicMap):
        em.update("foo", "reinforced")
        assert em.get_confidence("foo") == pytest.approx(0.58)

    def test_contradicted_lowers_confidence(self, em: EpistemicMap):
        em.update("foo", "contradicted")
        assert em.get_confidence("foo") == pytest.approx(0.40)

    def test_ambivalent_lowers_slightly(self, em: EpistemicMap):
        em.update("foo", "ambivalent")
        assert em.get_confidence("foo") == pytest.approx(0.47)

    def test_unknown_outcome_no_change(self, em: EpistemicMap):
        em.update("foo", "nonsense_outcome")
        assert em.get_confidence("foo") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestClamping:
    def test_clamps_to_max_one(self, em: EpistemicMap):
        for _ in range(20):
            em.update("topic", "reinforced")
        assert em.get_confidence("topic") <= 1.0

    def test_clamps_to_min_zero(self, em: EpistemicMap):
        for _ in range(20):
            em.update("topic", "contradicted")
        assert em.get_confidence("topic") >= 0.0


# ---------------------------------------------------------------------------
# Empty / whitespace topics
# ---------------------------------------------------------------------------


class TestEdgeCaseTopics:
    def test_empty_topic_silently_ignored(self, em: EpistemicMap):
        em.update("", "added")
        assert em.topic_count() == 0

    def test_whitespace_topic_silently_ignored(self, em: EpistemicMap):
        em.update("   ", "added")
        assert em.topic_count() == 0


# ---------------------------------------------------------------------------
# Case-insensitive normalisation
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    def test_foo_and_Foo_same_topic(self, em: EpistemicMap):
        em.update("Foo", "added")
        em.update("foo", "added")
        assert em.topic_count() == 1
        assert em.get_confidence("FOO") == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# update_from_memory
# ---------------------------------------------------------------------------


class TestUpdateFromMemory:
    def test_update_from_memory_updates_tags_and_memory_type(self, em: EpistemicMap):
        rec = MemoryRecord(
            content="memory content",
            memory_type=MemoryType.EPISODIC,
            tags=["alpha", "beta"],
        )
        em.update_from_memory(rec, "added")
        # Tags should be updated
        assert em.get_confidence("alpha") == pytest.approx(0.55)
        assert em.get_confidence("beta") == pytest.approx(0.55)
        # memory_type.value should also be updated
        assert em.get_confidence("episodic") == pytest.approx(0.55)

    def test_update_from_memory_no_tags_updates_type_only(self, em: EpistemicMap):
        rec = MemoryRecord(content="no tags", memory_type=MemoryType.SEMANTIC, tags=[])
        em.update_from_memory(rec, "reinforced")
        assert em.topic_count() == 1
        assert em.get_confidence("semantic") == pytest.approx(0.58)


# ---------------------------------------------------------------------------
# get_uncertain_topics / get_confident_topics
# ---------------------------------------------------------------------------


class TestTopicQueries:
    def _populate(self, em: EpistemicMap):
        """Build a map with varied confidence levels."""
        # low (uncertain)
        for _ in range(3):
            em.update("low_topic", "contradicted")  # starts 0.5, -0.3 → 0.2
        # high (confident)
        for _ in range(6):
            em.update("high_topic", "reinforced")  # starts 0.5, +0.48 → 0.98 → 1.0

    def test_get_uncertain_topics_returns_lowest_first(self, em: EpistemicMap):
        self._populate(em)
        em.update("medium_topic", "added")  # 0.55
        uncertain = em.get_uncertain_topics()
        # low_topic should be in uncertain (below 0.35)
        assert "low_topic" in uncertain
        # medium and high should NOT be in the default threshold list
        assert "high_topic" not in uncertain

    def test_get_uncertain_topics_sorted_lowest_first(self, em: EpistemicMap):
        em.update("a", "contradicted")  # 0.40
        em.update("a", "contradicted")  # 0.30
        em.update("b", "contradicted")  # 0.40
        topics = em.get_uncertain_topics()
        # a has lower confidence than b
        assert topics[0] == "a"

    def test_get_confident_topics_returns_highest_first(self, em: EpistemicMap):
        self._populate(em)
        confident = em.get_confident_topics()
        assert "high_topic" in confident
        # Verify sorted descending
        if len(confident) >= 2:
            scores = [em.get_confidence(t) for t in confident]
            assert scores == sorted(scores, reverse=True)

    def test_get_uncertain_topics_custom_threshold(self, em: EpistemicMap):
        em.update("foo", "added")  # 0.55 — in threshold if threshold=0.6
        em.update("bar", "reinforced")  # 0.58 — in threshold if threshold=0.6
        uncertain_wide = em.get_uncertain_topics(threshold=0.6)
        assert "foo" in uncertain_wide
        assert "bar" in uncertain_wide


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_summary_empty_when_no_uncertain_topics(self, em: EpistemicMap):
        # No topics at all → empty
        assert em.get_summary() == ""

    def test_summary_empty_when_all_topics_above_threshold(self, em: EpistemicMap):
        for _ in range(10):
            em.update("confident", "reinforced")
        assert em.get_summary() == ""

    def test_summary_includes_uncertain_topics_with_scores(self, em: EpistemicMap):
        # Drive a topic below uncertain threshold
        for _ in range(3):
            em.update("quantum", "contradicted")  # 0.5 - 0.3 = 0.2 — below 0.35
        summary = em.get_summary()
        assert "quantum" in summary
        assert "confidence" in summary.lower()


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_preserves_confidence_and_evidence_count(self, em: EpistemicMap):
        em.update("physics", "reinforced")
        em.update("physics", "reinforced")
        em.update("history", "contradicted")
        d = em.to_dict()
        restored = EpistemicMap.from_dict(d)
        assert restored.get_confidence("physics") == pytest.approx(em.get_confidence("physics"))
        assert restored.get_confidence("history") == pytest.approx(em.get_confidence("history"))
        assert restored._evidence_count["physics"] == 2
        assert restored._evidence_count["history"] == 1

    def test_round_trip_preserves_default_confidence(self):
        em = EpistemicMap(default_confidence=0.3)
        d = em.to_dict()
        restored = EpistemicMap.from_dict(d)
        assert restored.get_confidence("unknown_topic") == pytest.approx(0.3)

    def test_from_dict_empty_data_uses_defaults(self):
        em = EpistemicMap.from_dict({})
        assert em.topic_count() == 0
        assert em.get_confidence("x") == pytest.approx(0.5)
