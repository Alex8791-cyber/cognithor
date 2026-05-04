"""Tests for ``DomainCostTracker``."""

from __future__ import annotations

import time

import pytest

from cognithor.channels.program_synthesis.domains.cost_tracker import (
    DEFAULT_PRICING,
    DomainCostTracker,
    ModelPricing,
)


class TestDomainCostTracker:
    def test_initial_snapshot_empty(self) -> None:
        t = DomainCostTracker()
        assert t.snapshot() == {}

    def test_record_basic(self) -> None:
        t = DomainCostTracker()
        rec = t.record(
            domain="sql",
            tokens_in=1000,
            tokens_out=500,
            wall_ms=42.0,
            model="gpt-4o",
        )
        assert rec.call_count == 1
        assert rec.tokens_in == 1000
        assert rec.tokens_out == 500
        assert rec.wall_ms_total == 42.0
        # GPT-4o: 1000/1000 * 0.005 + 500/1000 * 0.015 = 0.005 + 0.0075
        assert rec.cost_usd_local == pytest.approx(0.0125)
        assert rec.cost_usd_cloud_reference == pytest.approx(0.0125)
        assert rec.by_model == {"gpt-4o": 1}

    def test_local_qwen_costs_zero(self) -> None:
        t = DomainCostTracker()
        rec = t.record(
            domain="sql",
            tokens_in=10_000,
            tokens_out=5_000,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        assert rec.cost_usd_local == 0.0
        # Cloud reference still computed against gpt-4o
        assert rec.cost_usd_cloud_reference > 0.0

    def test_savings_calculation(self) -> None:
        t = DomainCostTracker()
        t.record(
            domain="sql",
            tokens_in=10_000,
            tokens_out=5_000,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        snap = t.snapshot()["sql"]
        assert snap["savings_usd_vs_cloud"] == pytest.approx(0.125)

    def test_aggregates_across_calls(self) -> None:
        t = DomainCostTracker()
        for _ in range(3):
            t.record(
                domain="sql",
                tokens_in=100,
                tokens_out=50,
                wall_ms=10.0,
                model="qwen3:27b",
            )
        rec = t.get("sql")
        assert rec is not None
        assert rec.call_count == 3
        assert rec.tokens_in == 300
        assert rec.tokens_out == 150
        assert rec.wall_ms_total == 30.0

    def test_escalation_count(self) -> None:
        t = DomainCostTracker()
        t.record(
            domain="sql",
            tokens_in=100,
            tokens_out=50,
            wall_ms=1.0,
            model="qwen3:27b",
            escalated=True,
        )
        t.record(
            domain="sql",
            tokens_in=100,
            tokens_out=50,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        rec = t.get("sql")
        assert rec is not None
        assert rec.escalation_count == 1

    def test_isolated_domains(self) -> None:
        t = DomainCostTracker()
        t.record(
            domain="sql",
            tokens_in=100,
            tokens_out=0,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        t.record(
            domain="json",
            tokens_in=200,
            tokens_out=0,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        snap = t.snapshot()
        assert set(snap) == {"sql", "json"}
        assert snap["sql"]["tokens_in"] == 100
        assert snap["json"]["tokens_in"] == 200

    def test_unknown_model_gets_zero_local(self) -> None:
        t = DomainCostTracker()
        t.record(
            domain="sql",
            tokens_in=1000,
            tokens_out=500,
            wall_ms=1.0,
            model="unknown-model",
        )
        rec = t.get("sql")
        assert rec is not None
        assert rec.cost_usd_local == 0.0
        # cloud reference still applied against gpt-4o (default)
        assert rec.cost_usd_cloud_reference == pytest.approx(0.0125)

    def test_negative_tokens_rejected(self) -> None:
        t = DomainCostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            t.record(
                domain="x",
                tokens_in=-1,
                tokens_out=0,
                wall_ms=0.0,
                model="qwen3:27b",
            )

    def test_negative_wall_ms_rejected(self) -> None:
        t = DomainCostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            t.record(
                domain="x",
                tokens_in=0,
                tokens_out=0,
                wall_ms=-1.0,
                model="qwen3:27b",
            )

    def test_reset(self) -> None:
        t = DomainCostTracker()
        t.record(
            domain="sql",
            tokens_in=10,
            tokens_out=10,
            wall_ms=1.0,
            model="qwen3:27b",
        )
        t.reset()
        assert t.snapshot() == {}

    def test_custom_pricing(self) -> None:
        custom = {
            "tiny": ModelPricing(input_per_1k_usd=0.001, output_per_1k_usd=0.002),
        }
        t = DomainCostTracker(pricing=custom, cloud_reference_model="tiny")
        t.record(
            domain="x",
            tokens_in=2000,
            tokens_out=1000,
            wall_ms=1.0,
            model="tiny",
        )
        rec = t.get("x")
        assert rec is not None
        assert rec.cost_usd_local == pytest.approx(0.004)
        assert rec.cost_usd_cloud_reference == pytest.approx(0.004)

    def test_default_pricing_includes_qwen_local(self) -> None:
        assert "qwen3:27b" in DEFAULT_PRICING
        assert DEFAULT_PRICING["qwen3:27b"].input_per_1k_usd == 0.0


class TestTimingContext:
    def test_records_wall_ms(self) -> None:
        t = DomainCostTracker()
        with t.time("sql", model="qwen3:27b") as scope:
            scope.set_tokens(in_=100, out_=50)
            scope.set_escalated(False)
            time.sleep(0.005)  # 5 ms

        rec = t.get("sql")
        assert rec is not None
        assert rec.tokens_in == 100
        assert rec.tokens_out == 50
        assert rec.wall_ms_total >= 4.0  # 5 ms slept, allow ±1 ms slack

    def test_propagates_exception_but_still_records(self) -> None:
        t = DomainCostTracker()
        with pytest.raises(RuntimeError):
            with t.time("sql", model="qwen3:27b") as scope:
                scope.set_tokens(in_=10, out_=5)
                raise RuntimeError("synthesis failed")

        rec = t.get("sql")
        assert rec is not None
        assert rec.call_count == 1
        assert rec.tokens_in == 10
