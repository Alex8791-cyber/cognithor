"""Tests fuer RewardCalculator."""

import pytest

from jarvis.learning.reward import RewardCalculator


class TestRewardCalculator:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_perfect_score(self):
        reward = self.calc.calculate_reward(
            success_score=1.0,
            total_tools=5,
            failed_tools=0,
            unique_tools=5,
            total_tool_calls=5,
            duration_seconds=0.0,
        )
        assert reward == 1.0

    def test_worst_score(self):
        reward = self.calc.calculate_reward(
            success_score=0.0,
            total_tools=5,
            failed_tools=5,
            unique_tools=1,
            total_tool_calls=5,
            duration_seconds=300.0,
        )
        # Should be very low
        assert reward < 0.2

    def test_no_tools(self):
        reward = self.calc.calculate_reward(
            success_score=0.8,
            total_tools=0,
            failed_tools=0,
            unique_tools=0,
            total_tool_calls=0,
            duration_seconds=10.0,
        )
        # success * 0.4 + error_comp(1.0) * 0.2 + efficiency(1.0) * 0.2 + speed * 0.2
        assert 0.5 < reward < 1.0

    def test_high_error_ratio(self):
        reward_good = self.calc.calculate_reward(
            success_score=0.5, total_tools=10, failed_tools=0,
            unique_tools=5, total_tool_calls=10,
        )
        reward_bad = self.calc.calculate_reward(
            success_score=0.5, total_tools=10, failed_tools=8,
            unique_tools=5, total_tool_calls=10,
        )
        assert reward_good > reward_bad

    def test_efficiency_matters(self):
        # Efficient: 5 unique tools in 5 calls
        reward_efficient = self.calc.calculate_reward(
            success_score=0.7, total_tools=5, failed_tools=0,
            unique_tools=5, total_tool_calls=5,
        )
        # Inefficient: 2 unique tools in 10 calls (lots of repetition)
        reward_inefficient = self.calc.calculate_reward(
            success_score=0.7, total_tools=10, failed_tools=0,
            unique_tools=2, total_tool_calls=10,
        )
        assert reward_efficient > reward_inefficient

    def test_speed_matters(self):
        reward_fast = self.calc.calculate_reward(
            success_score=0.7, total_tools=3, failed_tools=0,
            unique_tools=3, total_tool_calls=3,
            duration_seconds=10.0,
        )
        reward_slow = self.calc.calculate_reward(
            success_score=0.7, total_tools=3, failed_tools=0,
            unique_tools=3, total_tool_calls=3,
            duration_seconds=290.0,
        )
        assert reward_fast > reward_slow

    def test_reward_clamped_0_1(self):
        reward = self.calc.calculate_reward(success_score=2.0)
        assert 0.0 <= reward <= 1.0

        reward = self.calc.calculate_reward(success_score=-1.0)
        assert 0.0 <= reward <= 1.0

    def test_from_context(self):
        context = {
            "success_score": 0.8,
            "total_tools": 5,
            "failed_tools": 1,
            "unique_tools": 4,
            "total_tool_calls": 5,
            "duration_seconds": 30.0,
        }
        reward = self.calc.calculate_from_context(context)
        assert 0.5 < reward < 1.0

    def test_from_empty_context(self):
        reward = self.calc.calculate_from_context({})
        assert 0.0 <= reward <= 1.0

    def test_custom_max_duration(self):
        calc = RewardCalculator(max_duration=60.0)
        # 30s out of 60s = speed = 0.5
        reward = calc.calculate_reward(
            success_score=1.0, duration_seconds=30.0,
        )
        expected_speed = 0.5
        # success(1.0)*0.4 + error(1.0)*0.2 + efficiency(1.0)*0.2 + speed(0.5)*0.2
        expected = 0.4 + 0.2 + 0.2 + 0.1
        assert abs(reward - expected) < 0.01

    def test_weights_sum_to_one(self):
        total = (
            RewardCalculator.W_SUCCESS
            + RewardCalculator.W_ERROR
            + RewardCalculator.W_EFFICIENCY
            + RewardCalculator.W_SPEED
        )
        assert abs(total - 1.0) < 0.001
