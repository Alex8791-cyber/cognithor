"""
tests/test_identity/cognitio/test_attention.py

Pure-unit tests for cognithor.identity.cognitio.attention.
Covers HeadWeights, and all four computation heads + rank_memories on
MultiHeadAttention using stub MemoryRecord objects and MagicMock engines.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from cognithor.identity.cognitio.attention import HeadWeights, MultiHeadAttention
from cognithor.identity.cognitio.memory import MemoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem(
    *,
    embedding: list[float] | None = None,
    emotional_intensity: float = 0.0,
    entrenchment: float = 0.0,
    is_anchor: bool = False,
    days_since_access: float = 0.0,
) -> MemoryRecord:
    """Construct a minimal MemoryRecord with controlled attributes."""
    m = MemoryRecord(content="test memory")
    m.embedding = embedding
    m.emotional_intensity = emotional_intensity
    m.entrenchment = entrenchment
    m.is_anchor = is_anchor
    # Monkey-patch days_since_access so tests don't depend on wall-clock time
    m.days_since_access = lambda: days_since_access  # type: ignore[method-assign]
    return m


# ---------------------------------------------------------------------------
# HeadWeights
# ---------------------------------------------------------------------------


class TestHeadWeights:
    @pytest.mark.parametrize(
        "cs, expected",
        [
            (0.0, HeadWeights.YOUNG),
            (4.99, HeadWeights.YOUNG),
            (5.0, HeadWeights.BALANCED),
            (14.99, HeadWeights.BALANCED),
            (15.0, HeadWeights.MATURE),
            (100.0, HeadWeights.MATURE),
        ],
    )
    def test_for_character_strength_returns_correct_tuple(self, cs, expected):
        assert HeadWeights.for_character_strength(cs) == expected

    def test_weights_sum_to_one(self):
        for weights in (HeadWeights.YOUNG, HeadWeights.BALANCED, HeadWeights.MATURE):
            assert abs(sum(weights) - 1.0) < 1e-9

    def test_young_identity_weight_is_smallest(self):
        w = HeadWeights.YOUNG
        assert w[3] < w[0]  # identity < semantic for young characters

    def test_mature_identity_weight_is_largest(self):
        w = HeadWeights.MATURE
        assert w[3] > w[1]  # identity > temporal for mature characters


# ---------------------------------------------------------------------------
# MultiHeadAttention — character_strength / weight update
# ---------------------------------------------------------------------------


class TestMultiHeadAttentionInit:
    def test_default_strength_gives_young_weights(self):
        mha = MultiHeadAttention(character_strength=0.0)
        assert mha.weights == HeadWeights.YOUNG

    def test_update_character_strength_changes_weights(self):
        mha = MultiHeadAttention(character_strength=0.0)
        mha.update_character_strength(20.0)
        assert mha.weights == HeadWeights.MATURE

    def test_strength_transition_balanced(self):
        mha = MultiHeadAttention(character_strength=10.0)
        assert mha.weights == HeadWeights.BALANCED


# ---------------------------------------------------------------------------
# Head 1 — semantic
# ---------------------------------------------------------------------------


class TestHeadSemantic:
    def test_no_embedding_returns_neutral(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=None)
        score = mha._head_semantic(m, [1.0, 0.0], None)
        assert score == 0.5

    def test_parallel_vectors_gives_one(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0, 0.0])
        score = mha._head_semantic(m, [1.0, 0.0], None)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_gives_zero(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0, 0.0])
        score = mha._head_semantic(m, [0.0, 1.0], None)
        assert abs(score) < 1e-6

    def test_opposite_vectors_clamped_to_zero(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0, 0.0])
        score = mha._head_semantic(m, [-1.0, 0.0], None)
        assert score == 0.0

    def test_uses_embedding_engine_when_provided(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0, 0.0])
        eng = MagicMock()
        eng.cosine_similarity.return_value = 0.77
        score = mha._head_semantic(m, [0.5, 0.5], eng)
        assert score == 0.77
        eng.cosine_similarity.assert_called_once_with([1.0, 0.0], [0.5, 0.5])

    def test_negative_cosine_from_engine_clamped_to_zero(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0])
        eng = MagicMock()
        eng.cosine_similarity.return_value = -0.5
        score = mha._head_semantic(m, [1.0], eng)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Head 2 — temporal
# ---------------------------------------------------------------------------


class TestHeadTemporal:
    def test_zero_days_gives_score_near_one(self):
        mha = MultiHeadAttention()
        m = _mem(days_since_access=0.0)
        score = mha._head_temporal(m, None)
        assert abs(score - math.exp(0.0)) < 1e-9

    @pytest.mark.parametrize("days", [1.0, 10.0, 50.0, 100.0])
    def test_decay_formula(self, days):
        mha = MultiHeadAttention()
        m = _mem(days_since_access=days)
        score = mha._head_temporal(m, None)
        expected = math.exp(-0.007 * days)
        assert abs(score - expected) < 1e-9

    def test_bias_engine_recency_score_used(self):
        mha = MultiHeadAttention()
        m = _mem(days_since_access=10.0)
        eng = MagicMock()
        eng.recency_score.return_value = 0.99
        score = mha._head_temporal(m, eng)
        assert score == 0.99
        eng.recency_score.assert_called_once_with(m)


# ---------------------------------------------------------------------------
# Head 3 — emotional
# ---------------------------------------------------------------------------


class TestHeadEmotional:
    def test_zero_intensity_gives_zero(self):
        mha = MultiHeadAttention()
        m = _mem(emotional_intensity=0.0)
        assert mha._head_emotional(m, 0.0, None) == 0.0

    def test_max_intensity_same_context_caps_at_max(self):
        mha = MultiHeadAttention()
        m = _mem(emotional_intensity=1.0)
        score = mha._head_emotional(m, 1.0, None)
        assert score <= MultiHeadAttention.MAX_EMOTIONAL_SCORE

    def test_mismatch_reduces_score(self):
        mha = MultiHeadAttention()
        m_match = _mem(emotional_intensity=0.8)
        m_mismatch = _mem(emotional_intensity=0.0)
        ctx = 0.8
        score_match = mha._head_emotional(m_match, ctx, None)
        score_mismatch = mha._head_emotional(m_mismatch, ctx, None)
        assert score_match > score_mismatch

    def test_bias_engine_emotional_weight_used(self):
        mha = MultiHeadAttention()
        m = _mem(emotional_intensity=0.5)
        eng = MagicMock()
        eng.emotional_weight.return_value = 2.0
        score = mha._head_emotional(m, 0.5, eng)
        # emotional_match = 1 - |0.5 - 0.5| = 1.0; score = 0.5 * 1.0 * 2.0 = 1.0 capped
        assert score == MultiHeadAttention.MAX_EMOTIONAL_SCORE


# ---------------------------------------------------------------------------
# Head 4 — identity
# ---------------------------------------------------------------------------


class TestHeadIdentity:
    def test_no_anchor_no_entrenchment(self):
        mha = MultiHeadAttention()
        m = _mem(entrenchment=0.0, is_anchor=False)
        score = mha._head_identity(m, None)
        assert score == 0.0

    def test_anchor_bonus_applied(self):
        mha = MultiHeadAttention()
        m_anchor = _mem(entrenchment=0.0, is_anchor=True)
        m_plain = _mem(entrenchment=0.0, is_anchor=False)
        assert mha._head_identity(m_anchor, None) > mha._head_identity(m_plain, None)

    @pytest.mark.parametrize("e", [0.0, 0.5, 1.0])
    def test_entrenchment_linear(self, e):
        mha = MultiHeadAttention()
        m = _mem(entrenchment=e, is_anchor=False)
        expected = e * 0.7
        assert abs(mha._head_identity(m, None) - expected) < 1e-9

    def test_bias_engine_identity_score_used(self):
        mha = MultiHeadAttention()
        m = _mem(entrenchment=0.5, is_anchor=False)
        eng = MagicMock()
        eng.identity_score.return_value = 0.88
        score = mha._head_identity(m, eng)
        assert score == 0.88


# ---------------------------------------------------------------------------
# compute_salience — integration
# ---------------------------------------------------------------------------


class TestComputeSalience:
    def test_returns_float(self):
        mha = MultiHeadAttention()
        m = _mem(embedding=[1.0, 0.0], emotional_intensity=0.0, entrenchment=0.5)
        s = mha.compute_salience(m, [1.0, 0.0])
        assert isinstance(s, float)

    def test_higher_entrenchment_raises_salience(self):
        mha = MultiHeadAttention()
        m_low = _mem(entrenchment=0.1)
        m_high = _mem(entrenchment=0.9)
        s_low = mha.compute_salience(m_low, [])
        s_high = mha.compute_salience(m_high, [])
        assert s_high > s_low

    def test_weighted_sum_formula_no_bias_engine(self):
        """Verify the formula manually for a controlled memory."""
        mha = MultiHeadAttention(character_strength=0.0)  # YOUNG weights
        w1, w2, w3, w4 = HeadWeights.YOUNG
        days = 0.0
        m = _mem(
            embedding=[1.0, 0.0],
            emotional_intensity=0.0,
            entrenchment=0.5,
            is_anchor=False,
            days_since_access=days,
        )
        h1 = 1.0  # parallel vectors → cosine = 1.0
        h2 = math.exp(-0.007 * days)  # = 1.0
        h3 = 0.0  # emotional_intensity = 0
        h4 = 0.5 * 0.7  # entrenchment * 0.7
        expected = w1 * h1 + w2 * h2 + w3 * h3 + w4 * h4
        result = mha.compute_salience(m, [1.0, 0.0])
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# rank_memories
# ---------------------------------------------------------------------------


class TestRankMemories:
    def test_returns_descending_order(self):
        mha = MultiHeadAttention()
        memories = [
            _mem(entrenchment=0.1),
            _mem(entrenchment=0.9),
            _mem(entrenchment=0.5),
        ]
        ranked = mha.rank_memories(memories, [])
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        mha = MultiHeadAttention()
        memories = [_mem(entrenchment=float(i) / 10) for i in range(10)]
        ranked = mha.rank_memories(memories, [], top_k=3)
        assert len(ranked) == 3

    def test_top_k_larger_than_input_returns_all(self):
        mha = MultiHeadAttention()
        memories = [_mem(), _mem()]
        ranked = mha.rank_memories(memories, [], top_k=100)
        assert len(ranked) == 2

    def test_empty_list_returns_empty(self):
        mha = MultiHeadAttention()
        assert mha.rank_memories([], []) == []

    def test_returns_tuples_of_record_and_float(self):
        mha = MultiHeadAttention()
        m = _mem()
        ranked = mha.rank_memories([m], [])
        assert len(ranked) == 1
        rec, score = ranked[0]
        assert rec is m
        assert isinstance(score, float)
