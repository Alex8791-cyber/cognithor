"""Tests fuer CostTracker (Feature 2)."""

from __future__ import annotations

from datetime import date

import pytest

from jarvis.telemetry.cost_tracker import CostTracker


@pytest.fixture()
def tracker(tmp_path):
    """CostTracker mit temp DB."""
    db = str(tmp_path / "costs.db")
    ct = CostTracker(db, daily_budget=1.0, monthly_budget=10.0)
    yield ct
    ct.close()


@pytest.fixture()
def tracker_no_budget(tmp_path):
    """CostTracker ohne Budget-Limits."""
    db = str(tmp_path / "costs_nb.db")
    ct = CostTracker(db)
    yield ct
    ct.close()


class TestCostRecording:
    def test_record_llm_call_calculates_cost(self, tracker):
        """GPT-4o 1000 tokens → korrekte USD."""
        record = tracker.record_llm_call(
            model="gpt-4o", input_tokens=1000, output_tokens=500, session_id="s1",
        )
        # gpt-4o: input=2.50/1M, output=10.00/1M
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert abs(record.cost_usd - expected) < 1e-9
        assert record.model == "gpt-4o"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500

    def test_ollama_is_free(self, tracker):
        """Ollama-Modelle → $0.00."""
        record = tracker.record_llm_call(
            model="qwen3:32b", input_tokens=5000, output_tokens=2000,
        )
        assert record.cost_usd == 0.0

    def test_unknown_model_default_pricing(self, tracker):
        """Unbekanntes Modell → Fallback-Preis."""
        record = tracker.record_llm_call(
            model="some-future-model", input_tokens=1000, output_tokens=1000,
        )
        # Fallback: input=5.00/1M, output=15.00/1M
        expected = 1000 * 5.00 / 1_000_000 + 1000 * 15.00 / 1_000_000
        assert abs(record.cost_usd - expected) < 1e-9

    def test_prefix_match(self, tracker):
        """Model mit Version-Suffix → Prefix-Match."""
        record = tracker.record_llm_call(
            model="gpt-4o-2024-11-20", input_tokens=1000, output_tokens=0,
        )
        # Sollte gpt-4o pricing matchen
        expected = 1000 * 2.50 / 1_000_000
        assert abs(record.cost_usd - expected) < 1e-9


class TestBudgetEnforcement:
    def test_daily_budget_enforcement(self, tracker):
        """Tageslimit ueberschritten → budget.ok = False."""
        # Record enough to exceed $1.00 daily budget
        # gpt-4-turbo: input=10.00/1M → 100k tokens = $1.00
        tracker.record_llm_call(model="gpt-4-turbo", input_tokens=100_000, output_tokens=0)
        budget = tracker.check_budget()
        assert not budget.ok
        assert "Tageslimit" in budget.warning

    def test_monthly_budget_enforcement(self, tracker):
        """Monatslimit ueberschritten → budget.ok = False."""
        # $10 monthly budget, record $10+ worth
        for _ in range(11):
            tracker.record_llm_call(model="gpt-4-turbo", input_tokens=100_000, output_tokens=0)
        budget = tracker.check_budget()
        assert not budget.ok
        assert "Monatslimit" in budget.warning

    def test_no_budget_unlimited(self, tracker_no_budget):
        """budget=0 → immer ok."""
        tracker_no_budget.record_llm_call(model="gpt-4-turbo", input_tokens=1_000_000, output_tokens=1_000_000)
        budget = tracker_no_budget.check_budget()
        assert budget.ok
        assert budget.daily_remaining == -1.0
        assert budget.monthly_remaining == -1.0

    def test_budget_warning_threshold(self, tracker):
        """Budget > 80% → Warning aber noch ok."""
        # Record 85% of $1.00 daily budget
        # gpt-4-turbo: 85k input tokens = $0.85
        tracker.record_llm_call(model="gpt-4-turbo", input_tokens=85_000, output_tokens=0)
        budget = tracker.check_budget()
        assert budget.ok  # Noch nicht ueberschritten
        assert "fast erreicht" in budget.warning


class TestSessionCosts:
    def test_get_session_cost(self, tracker):
        """Kosten pro Session korrekt aggregiert."""
        tracker.record_llm_call(model="gpt-4o", input_tokens=1000, output_tokens=500, session_id="s1")
        tracker.record_llm_call(model="gpt-4o", input_tokens=2000, output_tokens=1000, session_id="s1")
        tracker.record_llm_call(model="gpt-4o", input_tokens=500, output_tokens=200, session_id="s2")

        s1_cost = tracker.get_session_cost("s1")
        s2_cost = tracker.get_session_cost("s2")

        assert s1_cost > s2_cost
        assert s1_cost > 0
        assert s2_cost > 0


class TestCostReport:
    def test_get_cost_report(self, tracker):
        """Report mit cost_by_model, cost_by_day."""
        tracker.record_llm_call(model="gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record_llm_call(model="gpt-4o-mini", input_tokens=2000, output_tokens=1000)

        report = tracker.get_cost_report()
        assert report.total_calls == 2
        assert report.total_cost_usd > 0
        assert "gpt-4o" in report.cost_by_model
        assert "gpt-4o-mini" in report.cost_by_model
        assert len(report.cost_by_day) >= 1
        assert report.avg_cost_per_call > 0


class TestPersistence:
    def test_close_and_persistence(self, tmp_path):
        """Daten ueberleben close/reopen."""
        db = str(tmp_path / "persist.db")
        ct1 = CostTracker(db)
        ct1.record_llm_call(model="gpt-4o", input_tokens=1000, output_tokens=500, session_id="s1")
        cost_before = ct1.get_session_cost("s1")
        ct1.close()

        ct2 = CostTracker(db)
        cost_after = ct2.get_session_cost("s1")
        ct2.close()

        assert cost_before > 0
        assert abs(cost_before - cost_after) < 1e-9
