# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""LLM-Repair Two-Stage tests (Sprint-1 plan task 9 slice, spec §6.5.2 Zone-1).

Exercises the prompt schema, JSON parser, retry-once rule, and
suggestion sorting against a stub :class:`LLMBackend`. No vLLM has
to be running.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import Phase2Config
from cognithor.channels.program_synthesis.refiner.diff_analyzer import analyze_diff
from cognithor.channels.program_synthesis.refiner.llm_repair_two_stage import (
    LLMRepairError,
    LLMRepairResult,
    LLMRepairSuggestion,
    LLMRepairTwoStageClient,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)
from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendType,
)

# ---------------------------------------------------------------------------
# Stub backend
# ---------------------------------------------------------------------------


@dataclass
class _StubBackend(LLMBackend):
    queued: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.VLLM

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
        images: list[str] | None = None,
        video: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "format_json": format_json,
            }
        )
        if not self.queued:
            raise AssertionError("stub backend ran out of queued responses")
        content = self.queued.pop(0)
        return ChatResponse(content=content, model=model)

    async def chat_stream(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError

    async def embed(self, *_: Any, **__: Any) -> EmbedResponse:
        raise NotImplementedError

    async def is_available(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return ["Qwen/Qwen3.6-27B-Instruct"]

    async def close(self) -> None:
        pass


def _failing_program() -> Program:
    # rotate90(input) — wrong for the demo we'll feed.
    return Program(
        primitive="rotate90",
        children=(InputRef(),),
        output_type="Grid",
    )


def _failing_demos() -> list[tuple[Any, Any, Any]]:
    inp = np.array([[1, 2], [3, 4]], dtype=np.int8)
    expected = np.array([[2, 4], [1, 3]], dtype=np.int8)  # actually rotate270
    actual = np.rot90(inp, k=-1)  # what rotate90 produced
    return [(inp, expected, actual)]


# ---------------------------------------------------------------------------
# Two-stage call sequence
# ---------------------------------------------------------------------------


class TestTwoStageHappyPath:
    @pytest.mark.asyncio
    async def test_returns_sorted_suggestions(self) -> None:
        backend = _StubBackend(
            queued=[
                "The program rotated the grid the wrong direction.",
                json.dumps(
                    {
                        "suggestions": [
                            {"replacement_source": "rotate270(input)", "confidence": 0.9},
                            {
                                "replacement_source": "rotate180(rotate90(input))",
                                "confidence": 0.4,
                                "reasoning": "two flips compose",
                            },
                            {"replacement_source": "rotate90(rotate90(input))", "confidence": 0.6},
                        ]
                    }
                ),
            ]
        )
        client = LLMRepairTwoStageClient(
            backend,
            primitive_whitelist=["rotate90", "rotate180", "rotate270"],
        )
        result = await client.repair(_failing_program(), _failing_demos())
        assert isinstance(result, LLMRepairResult)
        # Stage-1 prose preserved.
        assert "wrong direction" in result.stage1_reasoning
        # Three suggestions parsed, sorted by descending confidence.
        confidences = [s.confidence for s in result.suggestions]
        assert confidences == sorted(confidences, reverse=True)
        # Top suggestion is the 0.9 one.
        assert result.suggestions[0].confidence == 0.9
        assert result.suggestions[0].replacement_source == "rotate270(input)"
        # The 0.4 entry preserves its reasoning string.
        the_lowest = next(s for s in result.suggestions if s.confidence == 0.4)
        assert the_lowest.reasoning == "two flips compose"

    @pytest.mark.asyncio
    async def test_stage1_and_stage2_temperatures(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {"suggestions": [{"replacement_source": "rotate270(input)", "confidence": 0.5}]}
                ),
            ]
        )
        config = Phase2Config(llm_temperature_stage1=0.42, llm_temperature_stage2=0.07)
        client = LLMRepairTwoStageClient(
            backend,
            primitive_whitelist=["rotate270"],
            config=config,
        )
        await client.repair(_failing_program(), _failing_demos())
        assert backend.calls[0]["temperature"] == 0.42
        assert backend.calls[1]["temperature"] == 0.07
        assert backend.calls[0]["format_json"] is False
        assert backend.calls[1]["format_json"] is True

    @pytest.mark.asyncio
    async def test_uses_configured_model_name(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {"suggestions": [{"replacement_source": "rotate270(input)", "confidence": 0.5}]}
                ),
            ]
        )
        config = Phase2Config(llm_model_name="Qwen/Qwen3.6-27B-AWQ")
        client = LLMRepairTwoStageClient(
            backend,
            primitive_whitelist=["rotate270"],
            config=config,
        )
        await client.repair(_failing_program(), _failing_demos())
        assert all(call["model"] == "Qwen/Qwen3.6-27B-AWQ" for call in backend.calls)


# ---------------------------------------------------------------------------
# Diff is included in prompt when supplied
# ---------------------------------------------------------------------------


class TestDiffInPrompt:
    @pytest.mark.asyncio
    async def test_diff_summary_appears_in_user_messages(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {"suggestions": [{"replacement_source": "rotate270(input)", "confidence": 0.5}]}
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate270"])
        inp = np.array([[1, 2], [3, 4]], dtype=np.int8)
        expected = np.array([[2, 4], [1, 3]], dtype=np.int8)
        actual = np.rot90(inp, k=-1)
        diff = analyze_diff(actual, expected)
        await client.repair(_failing_program(), [(inp, expected, actual)], diff=diff)
        # User message of the stage-1 call must mention the diff summary.
        stage1_user = backend.calls[0]["messages"][1]["content"]
        assert "shape_mismatch=" in stage1_user
        # Stage-2 user must too.
        stage2_user = backend.calls[1]["messages"][1]["content"]
        assert "shape_mismatch=" in stage2_user
        # And the whitelist must appear in stage-2 (substring is enough).
        assert "rotate270" in stage2_user

    @pytest.mark.asyncio
    async def test_no_diff_uses_placeholder(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {"suggestions": [{"replacement_source": "rotate270(input)", "confidence": 0.5}]}
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate270"])
        await client.repair(_failing_program(), _failing_demos())
        stage1_user = backend.calls[0]["messages"][1]["content"]
        assert "No structured diff supplied." in stage1_user


# ---------------------------------------------------------------------------
# JSON parser robustness
# ---------------------------------------------------------------------------


class TestJsonParser:
    @pytest.mark.asyncio
    async def test_drops_entries_with_missing_source(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {
                        "suggestions": [
                            {"replacement_source": "rotate270(input)", "confidence": 0.7},
                            {"confidence": 0.9},  # no source — drop
                            {"replacement_source": "", "confidence": 0.5},  # empty — drop
                            {"replacement_source": "   ", "confidence": 0.5},  # whitespace — drop
                        ]
                    }
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate270"])
        result = await client.repair(_failing_program(), _failing_demos())
        assert len(result.suggestions) == 1
        assert result.suggestions[0].replacement_source == "rotate270(input)"

    @pytest.mark.asyncio
    async def test_clamps_confidence_to_unit_range(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {
                        "suggestions": [
                            {"replacement_source": "a", "confidence": 1.7},
                            {"replacement_source": "b", "confidence": -0.3},
                        ]
                    }
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate90"])
        result = await client.repair(_failing_program(), _failing_demos())
        # Sorted by confidence descending: clamped 1.0 first, clamped 0.0 last.
        assert [s.confidence for s in result.suggestions] == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_non_finite_confidence_replaced_with_default(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {
                        "suggestions": [
                            {"replacement_source": "a", "confidence": "nope"},
                        ]
                    }
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate90"])
        result = await client.repair(_failing_program(), _failing_demos())
        # Default 0.5 used.
        assert result.suggestions[0].confidence == 0.5

    @pytest.mark.asyncio
    async def test_missing_suggestions_key_raises(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps({"foo": "bar"}),  # no `suggestions`
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate90"])
        with pytest.raises(LLMRepairError):
            await client.repair(_failing_program(), _failing_demos())


# ---------------------------------------------------------------------------
# Retry-once on parse failure
# ---------------------------------------------------------------------------


class TestRetryOnceRule:
    @pytest.mark.asyncio
    async def test_invalid_then_valid_retries_and_returns(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                "{not actually json",
                json.dumps(
                    {"suggestions": [{"replacement_source": "rotate270(input)", "confidence": 0.5}]}
                ),
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate270"])
        result = await client.repair(_failing_program(), _failing_demos())
        assert len(result.suggestions) == 1
        # Stage-1 + 2 stage-2 calls = 3 total.
        assert len(backend.calls) == 3

    @pytest.mark.asyncio
    async def test_two_consecutive_failures_raise(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                "{not json",
                "still not json",
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate90"])
        with pytest.raises(LLMRepairError):
            await client.repair(_failing_program(), _failing_demos())

    @pytest.mark.asyncio
    async def test_non_object_json_treated_as_failure(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                "[1, 2, 3]",  # array, not object
                "[4, 5]",  # also array
            ]
        )
        client = LLMRepairTwoStageClient(backend, primitive_whitelist=["rotate90"])
        with pytest.raises(LLMRepairError):
            await client.repair(_failing_program(), _failing_demos())


# ---------------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------------


class TestLLMRepairSuggestionDataclass:
    def test_is_frozen_and_hashable(self) -> None:
        s = LLMRepairSuggestion(
            replacement_source="rotate90(input)",
            confidence=0.5,
            reasoning="meh",
        )
        # Frozen dataclass → hashable.
        assert hash(s) == hash(s)
        # And immutable.
        with pytest.raises(FrozenInstanceError):
            s.confidence = 0.9  # type: ignore[misc]

    def test_default_reasoning_empty(self) -> None:
        s = LLMRepairSuggestion(replacement_source="x", confidence=0.5)
        assert s.reasoning == ""
