# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module A — LLM-Prior over vLLM tests (spec v1.4 §4 Sprint-1).

These exercise the prompt schema + parser + retry logic against a
stub :class:`LLMBackend`. No vLLM has to be running — the backend is
fully dependency-injected. The production wiring (
:class:`VLLMBackend` against a Qwen 3.6 27B served on RTX 5090) is
covered by ``tests/test_vllm_*`` already, separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    LLMPrior,
    LLMPriorClient,
    LLMPriorError,
    Phase2Config,
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
    """Minimal LLMBackend stub that yields a queued list of responses."""

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


def _examples() -> list[tuple[Any, Any]]:
    return [
        ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
        ([[1]], [[0]]),
    ]


# ---------------------------------------------------------------------------
# Two-stage call sequence
# ---------------------------------------------------------------------------


class TestTwoStageHappyPath:
    @pytest.mark.asyncio
    async def test_returns_normalised_distribution(self) -> None:
        backend = _StubBackend(
            queued=[
                "The output flips colors via a global recolor.",
                json.dumps(
                    {
                        "rotate90": 0.1,
                        "recolor": 0.7,
                        "tile": 0.2,
                        "alpha_entropy_hint": 0.65,
                    }
                ),
            ]
        )
        client = LLMPriorClient(
            backend,
            primitive_whitelist=["rotate90", "recolor", "tile", "mirror"],
        )
        prior = await client.get_prior(_examples())

        assert isinstance(prior, LLMPrior)
        # Scores sum to ~1.0 (normalised).
        assert abs(sum(prior.primitive_scores.values()) - 1.0) < 1e-9
        # Keys preserved + filtered to whitelist.
        assert set(prior.primitive_scores) == {"rotate90", "recolor", "tile"}
        # Stage-1 reasoning preserved verbatim.
        assert "global recolor" in prior.stage1_reasoning

    @pytest.mark.asyncio
    async def test_stage1_uses_stage1_temperature(self) -> None:
        backend = _StubBackend(
            queued=["reasoning", json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.5})]
        )
        config = Phase2Config(llm_temperature_stage1=0.42, llm_temperature_stage2=0.07)
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"], config=config)
        await client.get_prior(_examples())
        assert backend.calls[0]["temperature"] == 0.42
        assert backend.calls[1]["temperature"] == 0.07

    @pytest.mark.asyncio
    async def test_stage2_requests_json_format(self) -> None:
        backend = _StubBackend(
            queued=["reasoning", json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.5})]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        await client.get_prior(_examples())
        assert backend.calls[0]["format_json"] is False
        assert backend.calls[1]["format_json"] is True

    @pytest.mark.asyncio
    async def test_uses_configured_model_name(self) -> None:
        backend = _StubBackend(
            queued=["reasoning", json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.5})]
        )
        config = Phase2Config(llm_model_name="Qwen/Qwen3.6-27B-Instruct-AWQ")
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"], config=config)
        await client.get_prior(_examples())
        assert all(call["model"] == "Qwen/Qwen3.6-27B-Instruct-AWQ" for call in backend.calls)


# ---------------------------------------------------------------------------
# JSON parser robustness
# ---------------------------------------------------------------------------


class TestJsonParser:
    @pytest.mark.asyncio
    async def test_filters_hallucinated_primitives(self) -> None:
        # The LLM returns a name not in the whitelist — quietly dropped.
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {
                        "recolor": 0.5,
                        "fictional_primitive_xyz": 0.5,
                        "alpha_entropy_hint": 0.6,
                    }
                ),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor", "rotate90"])
        prior = await client.get_prior(_examples())
        assert "fictional_primitive_xyz" not in prior.primitive_scores
        assert prior.primitive_scores == {"recolor": 1.0}

    @pytest.mark.asyncio
    async def test_negative_or_nan_scores_dropped(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps(
                    {
                        "recolor": -0.5,
                        "rotate90": 0.4,
                        "tile": "not a number",
                        "alpha_entropy_hint": 0.5,
                    }
                ),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor", "rotate90", "tile"])
        prior = await client.get_prior(_examples())
        assert prior.primitive_scores == {"rotate90": 1.0}

    @pytest.mark.asyncio
    async def test_retry_once_on_parse_failure(self) -> None:
        # Stage-1 returns reasoning. Stage-2 first attempt is invalid JSON;
        # retry returns valid JSON. Result should succeed.
        backend = _StubBackend(
            queued=[
                "reasoning",
                "{ not json",
                json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.5}),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        prior = await client.get_prior(_examples())
        assert prior.primitive_scores == {"recolor": 1.0}

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self) -> None:
        backend = _StubBackend(queued=["reasoning", "{ broken", "{ still broken"])
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        with pytest.raises(LLMPriorError, match="JSON parse failed"):
            await client.get_prior(_examples())

    @pytest.mark.asyncio
    async def test_empty_filtered_set_raises(self) -> None:
        # Every score gets dropped (all not in whitelist).
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps({"unknown1": 0.4, "unknown2": 0.6, "alpha_entropy_hint": 0.5}),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        with pytest.raises(LLMPriorError, match="no usable primitive scores"):
            await client.get_prior(_examples())

    @pytest.mark.asyncio
    async def test_non_object_json_treated_as_parse_failure(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps([1, 2, 3]),
                json.dumps([4, 5]),
            ]
        )
        config = Phase2Config(llm_json_max_retries=1)
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"], config=config)
        with pytest.raises(LLMPriorError, match="non-object"):
            await client.get_prior(_examples())


# ---------------------------------------------------------------------------
# Alpha-entropy hint clamping
# ---------------------------------------------------------------------------


class TestAlphaEntropyHint:
    @pytest.mark.asyncio
    async def test_value_within_band_passes_through(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.7}),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        prior = await client.get_prior(_examples())
        assert prior.alpha_entropy_hint == 0.7

    @pytest.mark.asyncio
    async def test_above_band_clamps_to_upper_bound(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps({"recolor": 1.0, "alpha_entropy_hint": 1.5}),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        prior = await client.get_prior(_examples())
        # Default upper bound is 0.85.
        assert prior.alpha_entropy_hint == 0.85

    @pytest.mark.asyncio
    async def test_below_band_clamps_to_lower_bound(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                json.dumps({"recolor": 1.0, "alpha_entropy_hint": 0.0}),
            ]
        )
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        prior = await client.get_prior(_examples())
        assert prior.alpha_entropy_hint == 0.5

    @pytest.mark.asyncio
    async def test_missing_hint_falls_back_to_neutral(self) -> None:
        backend = _StubBackend(queued=["reasoning", json.dumps({"recolor": 1.0})])
        client = LLMPriorClient(backend, primitive_whitelist=["recolor"])
        prior = await client.get_prior(_examples())
        # Neutral 0.5 is inside the band so it doesn't get clamped.
        assert prior.alpha_entropy_hint == 0.5


# ---------------------------------------------------------------------------
# Phase2Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_empty_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_model_name"):
            Phase2Config(llm_model_name="")

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_temperature_stage1"):
            Phase2Config(llm_temperature_stage1=-0.1)
        with pytest.raises(ValueError, match="llm_temperature_stage2"):
            Phase2Config(llm_temperature_stage2=-0.1)

    def test_too_high_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_temperature_stage1"):
            Phase2Config(llm_temperature_stage1=2.1)

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_json_max_retries"):
            Phase2Config(llm_json_max_retries=-1)

    def test_zero_top_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_top_k_default"):
            Phase2Config(llm_top_k_default=0)

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_call_timeout_seconds"):
            Phase2Config(llm_call_timeout_seconds=0.0)

    def test_default_model_is_qwen_3_6_27b(self) -> None:
        # Spec anchors the default model.
        from cognithor.channels.program_synthesis.phase2 import (
            DEFAULT_PHASE2_CONFIG,
        )

        assert DEFAULT_PHASE2_CONFIG.llm_model_name == "Qwen/Qwen3.6-27B-Instruct"
        assert DEFAULT_PHASE2_CONFIG.llm_fallback_model_name == "Qwen/Qwen3.6-27B-Instruct-AWQ"
        assert DEFAULT_PHASE2_CONFIG.llm_base_url == "http://localhost:8000/v1"
