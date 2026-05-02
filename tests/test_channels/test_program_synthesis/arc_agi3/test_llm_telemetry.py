# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — LLM telemetry tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import (
    LLMTelemetry,
    estimate_token_count,
    extract_think_tokens,
    record_vllm_request_output,
    wrap_planning_choice_fn,
    wrap_text_choice_fn,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _ctx() -> FrameContext:
    return FrameContext(
        grid=np.zeros((4, 4), dtype=np.int8),
        available_action_names=["RESET", "ACTION1", "ACTION2"],
        history_summary="(no actions yet)",
        levels_completed=0,
        win_levels=1,
    )


class TestTokenEstimate:
    def test_zero_for_empty(self) -> None:
        assert estimate_token_count("") == 0

    def test_grows_with_length(self) -> None:
        assert estimate_token_count("abcd") < estimate_token_count("abcd" * 10)

    def test_minimum_one_for_nonempty(self) -> None:
        assert estimate_token_count("a") == 1


class TestThinkExtraction:
    def test_zero_without_think_block(self) -> None:
        assert extract_think_tokens('{"action": "RESET"}') == 0

    def test_counts_think_content(self) -> None:
        # The "think" block is "Long reasoning here" → ~5 tokens estimate.
        text = "<think>" + ("Long reasoning here. " * 5) + "</think>{json}"
        assert extract_think_tokens(text) > 5

    def test_counts_only_inside_block(self) -> None:
        a = "<think>x</think>after"
        b = "<think>x" + "y" * 200 + "</think>after"
        assert extract_think_tokens(b) > extract_think_tokens(a)


class TestTextWrapper:
    def test_records_one_call(self) -> None:
        tele = LLMTelemetry()

        def _stub(ctx: FrameContext) -> tuple[str, str]:
            return "ACTION1", "stub reasoning"

        wrapped = wrap_text_choice_fn(_stub, tele)
        out = wrapped(_ctx())
        assert out == ("ACTION1", "stub reasoning")
        assert len(tele) == 1
        rec = tele.records[0]
        assert rec.finish_reason == "stop"
        assert rec.input_tokens > 0
        assert rec.output_tokens > 0
        assert rec.wall_clock_s >= 0.0

    def test_records_abort_on_exception(self) -> None:
        tele = LLMTelemetry()

        def _bad(ctx: FrameContext) -> tuple[str, str]:
            raise RuntimeError("boom")

        wrapped = wrap_text_choice_fn(_bad, tele)
        with pytest.raises(RuntimeError):
            wrapped(_ctx())
        assert len(tele) == 1
        assert tele.records[0].finish_reason == "abort"
        assert tele.records[0].output_tokens == 0


class TestPlanningWrapper:
    def test_records_planning_call(self) -> None:
        tele = LLMTelemetry()

        def _stub(ctx: FrameContext) -> tuple[Any, str]:
            return ([{"action": "ACTION1"}], "two-step plan")

        wrapped = wrap_planning_choice_fn(_stub, tele)
        plan, reasoning = wrapped(_ctx())
        assert plan == [{"action": "ACTION1"}]
        assert reasoning == "two-step plan"
        assert len(tele) == 1


class TestVllmRequestOutput:
    def test_records_real_finish_reason(self) -> None:
        # Mock a vLLM RequestOutput-shaped object.
        class _Out:
            text = "<think>some reasoning</think>{json}"
            token_ids = list(range(120))
            finish_reason = "length"

        class _Req:
            prompt_token_ids = list(range(450))
            outputs = [_Out()]

        tele = LLMTelemetry()
        record_vllm_request_output(tele, _Req(), wall_clock_s=12.5)
        rec = tele.records[0]
        assert rec.finish_reason == "length"
        assert rec.input_tokens == 450
        assert rec.output_tokens == 120
        assert rec.think_tokens > 0
        assert rec.wall_clock_s == 12.5

    def test_handles_tool_calls_finish_reason(self) -> None:
        class _Out:
            text = '{"action": "call_tool"}'
            token_ids = [1, 2, 3]
            finish_reason = "tool_calls"

        class _Req:
            prompt_token_ids = [0] * 200
            outputs = [_Out()]

        tele = LLMTelemetry()
        record_vllm_request_output(tele, _Req(), wall_clock_s=1.0)
        s = tele.summary()
        assert s["finish_reason_dist"]["tool_calls"] == 1
        # tool_calls is NOT length → truncation rate stays 0.
        assert s["length_truncation_rate"] == 0.0


class TestSummary:
    def test_empty_summary(self) -> None:
        s = LLMTelemetry().summary()
        assert s == {"calls": 0}

    def test_aggregates_multiple_calls(self) -> None:
        tele = LLMTelemetry()

        def _stub(ctx: FrameContext) -> tuple[str, str]:
            return "X", "y" * 100

        wrapped = wrap_text_choice_fn(_stub, tele)
        for _ in range(3):
            wrapped(_ctx())
        s = tele.summary()
        assert s["calls"] == 3
        assert s["finish_reason_dist"] == {"stop": 3}
        assert s["length_truncation_rate"] == 0.0
        assert s["input_tokens_total"] > 0
        assert s["output_tokens_avg"] > 0

    def test_truncation_rate_picks_up_aborts(self) -> None:
        # Mix one abort with two normal calls → finish_reason_dist
        # tracks both, length_truncation_rate stays 0 (abort != length).
        tele = LLMTelemetry()
        good_count = [0]

        def _flaky(ctx: FrameContext) -> tuple[str, str]:
            if good_count[0] < 2:
                good_count[0] += 1
                return "ACTION1", "ok"
            raise RuntimeError("late failure")

        wrapped = wrap_text_choice_fn(_flaky, tele)
        wrapped(_ctx())
        wrapped(_ctx())
        with pytest.raises(RuntimeError):
            wrapped(_ctx())
        s = tele.summary()
        assert s["calls"] == 3
        assert s["finish_reason_dist"]["abort"] == 1
        assert s["finish_reason_dist"]["stop"] == 2
