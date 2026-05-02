# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — MTP speculative-decoding stats tests."""

from __future__ import annotations

from typing import Any

from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import (
    MTPSnapshot,
    MTPStats,
    extract_per_request_acceptance,
    poll_engine_mtp_metrics,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestSnapshot:
    def test_acceptance_rate_basic(self) -> None:
        s = MTPSnapshot(
            drafts_proposed=100,
            drafts_accepted=70,
            tokens_emitted=120,
            num_speculative_tokens=3,
        )
        assert s.acceptance_rate == 0.7

    def test_acceptance_rate_none_when_no_drafts(self) -> None:
        s = MTPSnapshot(0, 0, 0, 3)
        assert s.acceptance_rate is None

    def test_efficiency_at_100_percent_acceptance(self) -> None:
        # 10 passes × 3 drafts each, all accepted → 40 tokens (10 base + 30 accepted)
        s = MTPSnapshot(
            drafts_proposed=30, drafts_accepted=30, tokens_emitted=40, num_speculative_tokens=3
        )
        # tokens_emitted / passes (= base tokens = emitted - accepted = 10) = 4.0
        assert s.spec_token_efficiency == 4.0

    def test_efficiency_with_no_acceptance(self) -> None:
        # 10 passes, 0 drafts accepted → tokens_emitted == 10 (just base)
        s = MTPSnapshot(
            drafts_proposed=30, drafts_accepted=0, tokens_emitted=10, num_speculative_tokens=3
        )
        assert s.spec_token_efficiency == 1.0


class TestExtractPerRequest:
    def test_no_metrics_returns_none(self) -> None:
        class _R:
            pass

        assert extract_per_request_acceptance(_R()) is None

    def test_no_acceptance_field_returns_none(self) -> None:
        class _Metrics:
            spec_token_acceptance_counts = None

        class _R:
            metrics = _Metrics()

        assert extract_per_request_acceptance(_R()) is None

    def test_aggregates_acceptance_counts(self) -> None:
        # 12 passes accepted 0 drafts, 8 accepted 1, 5 accepted 2, 3 accepted 3.
        # num_spec = 4 - 1 = 3.
        class _Metrics:
            spec_token_acceptance_counts = (12, 8, 5, 3)

        class _R:
            metrics = _Metrics()

        snap = extract_per_request_acceptance(_R())
        assert snap is not None
        assert snap.num_speculative_tokens == 3
        # passes = 28, drafts/pass = 3 → drafts_proposed = 84
        assert snap.drafts_proposed == 84
        # accepted: 0*12 + 1*8 + 2*5 + 3*3 = 27
        assert snap.drafts_accepted == 27
        # tokens_emitted = 28 base + 27 accepted = 55
        assert snap.tokens_emitted == 55


class TestPollEngine:
    def test_no_engine_returns_none(self) -> None:
        class _LLM:
            pass

        assert poll_engine_mtp_metrics(_LLM()) is None

    def test_modern_get_metrics_path(self) -> None:
        class _Metric:
            def __init__(self, name: str, value: int) -> None:
                self.name = name
                self.value = value

        class _Engine:
            def get_metrics(self) -> list[Any]:
                return [
                    _Metric("vllm:spec_decode_num_drafts_total", 100),
                    _Metric("vllm:spec_decode_num_accepted_tokens_total", 70),
                    _Metric("vllm:spec_decode_num_emitted_tokens_total", 120),
                ]

        class _LLM:
            llm_engine = _Engine()

        snap = poll_engine_mtp_metrics(_LLM())
        assert snap is not None
        assert snap.drafts_proposed == 100
        assert snap.drafts_accepted == 70
        assert snap.tokens_emitted == 120
        assert snap.acceptance_rate == 0.7

    def test_legacy_stat_logger_fallback(self) -> None:
        class _Spec:
            num_drafts = 50
            num_accepted_tokens = 35
            num_emitted_tokens = 60

        class _StatLogger:
            spec_decode_stats = _Spec()

        class _Engine:
            def get_metrics(self) -> list[Any]:
                return []  # modern path empty → fallback kicks in

            stat_logger = _StatLogger()

        class _LLM:
            llm_engine = _Engine()

        snap = poll_engine_mtp_metrics(_LLM())
        assert snap is not None
        assert snap.drafts_proposed == 50
        assert snap.drafts_accepted == 35
        assert snap.acceptance_rate == 0.7


class TestStatsAggregator:
    def test_acceptance_rate_aggregates_snapshots(self) -> None:
        stats = MTPStats()
        stats.snapshots.append(MTPSnapshot(100, 70, 120, 3))
        stats.snapshots.append(MTPSnapshot(50, 40, 65, 3))
        # combined: drafts 150, accepted 110 → 0.7333…
        assert abs(stats.acceptance_rate - 110 / 150) < 1e-9

    def test_summary_shape(self) -> None:
        stats = MTPStats()
        stats.snapshots.append(MTPSnapshot(100, 70, 120, 3))
        s = stats.summary()
        assert s["snapshots"] == 1
        assert s["total_drafts_proposed"] == 100
        assert s["total_drafts_accepted"] == 70
        assert s["acceptance_rate"] == 0.7
        assert s["spec_token_efficiency"] is not None

    def test_add_request_skips_none(self) -> None:
        stats = MTPStats()

        class _NoMetrics:
            pass

        result = stats.add_request(_NoMetrics())
        assert result is None
        assert stats.snapshots == []
