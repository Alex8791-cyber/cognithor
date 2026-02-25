"""Tests fuer SearchWeightOptimizer."""

import pytest

from jarvis.memory.weight_optimizer import SearchWeightOptimizer


class TestSearchWeightOptimizer:
    def setup_method(self):
        self.optimizer = SearchWeightOptimizer()  # in-memory

    def teardown_method(self):
        self.optimizer.close()

    def test_initial_weights(self):
        w_v, w_b, w_g = self.optimizer.get_optimized_weights()
        assert abs(w_v - 0.50) < 0.01
        assert abs(w_b - 0.30) < 0.01
        assert abs(w_g - 0.20) < 0.01

    def test_custom_initial_weights(self):
        opt = SearchWeightOptimizer(initial_weights=(0.4, 0.4, 0.2))
        w_v, w_b, w_g = opt.get_optimized_weights()
        assert abs(w_v - 0.4) < 0.01
        assert abs(w_b - 0.4) < 0.01
        assert abs(w_g - 0.2) < 0.01
        opt.close()

    def test_weights_sum_to_one(self):
        self.optimizer.record_outcome(
            "test query",
            {"vector": 0.8, "bm25": 0.1, "graph": 0.1},
            feedback_score=0.9,
        )
        w_v, w_b, w_g = self.optimizer.get_optimized_weights()
        assert abs((w_v + w_b + w_g) - 1.0) < 0.001

    def test_weights_shift_toward_contributing_channel(self):
        initial_v, _, _ = self.optimizer.get_optimized_weights()

        # Record many outcomes where vector dominates
        for _ in range(20):
            self.optimizer.record_outcome(
                "vector query",
                {"vector": 0.9, "bm25": 0.05, "graph": 0.05},
                feedback_score=1.0,
            )

        new_v, _, _ = self.optimizer.get_optimized_weights()
        assert new_v > initial_v  # Vector weight should increase

    def test_minimum_weight_constraint(self):
        # Push one channel heavily
        for _ in range(100):
            self.optimizer.record_outcome(
                "q", {"vector": 1.0, "bm25": 0.0, "graph": 0.0},
                feedback_score=1.0,
            )

        w_v, w_b, w_g = self.optimizer.get_optimized_weights()
        assert w_b >= SearchWeightOptimizer.MIN_WEIGHT
        assert w_g >= SearchWeightOptimizer.MIN_WEIGHT

    def test_zero_feedback_no_update(self):
        initial = self.optimizer.get_optimized_weights()
        self.optimizer.record_outcome(
            "q", {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
            feedback_score=0.0,
        )
        after = self.optimizer.get_optimized_weights()
        assert initial == after

    def test_report(self):
        self.optimizer.record_outcome(
            "q", {"vector": 0.5, "bm25": 0.3, "graph": 0.2},
            feedback_score=0.8,
        )
        report = self.optimizer.report()
        assert "weights" in report
        assert report["total_outcomes"] == 1
        assert report["avg_feedback_score"] > 0

    def test_report_empty(self):
        report = self.optimizer.report()
        assert report["total_outcomes"] == 0
        assert report["avg_feedback_score"] == 0.0

    def test_persistence(self):
        """Weights survive re-initialization with same DB."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            opt1 = SearchWeightOptimizer(db_path=db_path)
            for _ in range(10):
                opt1.record_outcome(
                    "q", {"vector": 0.8, "bm25": 0.1, "graph": 0.1},
                    feedback_score=0.9,
                )
            weights1 = opt1.get_optimized_weights()
            opt1.close()

            # Re-open same DB
            opt2 = SearchWeightOptimizer(db_path=db_path)
            weights2 = opt2.get_optimized_weights()
            opt2.close()

            assert abs(weights1[0] - weights2[0]) < 0.001
            assert abs(weights1[1] - weights2[1]) < 0.001
            assert abs(weights1[2] - weights2[2]) < 0.001
        finally:
            os.unlink(db_path)

    def test_normalize_weights_static(self):
        w = SearchWeightOptimizer._normalize_weights(0.0, 0.0, 0.0)
        assert abs(sum(w) - 1.0) < 0.001

    def test_normalize_weights_with_min(self):
        w = SearchWeightOptimizer._normalize_weights(0.01, 0.01, 0.98)
        assert all(wi >= 0.05 for wi in w)
        assert abs(sum(w) - 1.0) < 0.001
