# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module A — Dual-Prior mixer tests (spec v1.4 §4 + §4.4.4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DualPriorMixer,
    DualPriorResult,
    LLMPriorClient,
    Phase2Config,
    SymbolicPrior,
    SymbolicPriorResult,
    UniformSymbolicPrior,
)
from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendType,
)

# ---------------------------------------------------------------------------
# Stub backend (mirrors the LLMPrior tests' shape)
# ---------------------------------------------------------------------------


@dataclass
class _StubBackend(LLMBackend):
    queued: list[str] = field(default_factory=list)

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
        if not self.queued:
            raise AssertionError("stub backend ran out of queued responses")
        return ChatResponse(content=self.queued.pop(0), model=model)

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
        ([[1, 0]], [[0, 1]]),
        ([[2]], [[3]]),
        ([[5, 5]], [[6, 6]]),
        ([[7]], [[8]]),
    ]


def _llm_response(scores: dict[str, float], hint: float) -> str:
    return json.dumps({**scores, "alpha_entropy_hint": hint})


# ---------------------------------------------------------------------------
# UniformSymbolicPrior
# ---------------------------------------------------------------------------


class TestUniformSymbolicPrior:
    def test_returns_flat_distribution_over_whitelist(self) -> None:
        prior = UniformSymbolicPrior(primitive_whitelist=["a", "b", "c", "d"])
        result = prior.get_prior(_examples())
        assert set(result.primitive_scores) == {"a", "b", "c", "d"}
        assert all(abs(v - 0.25) < 1e-9 for v in result.primitive_scores.values())

    def test_effective_confidence_dampened_by_sample_size(self) -> None:
        prior = UniformSymbolicPrior(primitive_whitelist=["a"])
        # Default n0 = 4, n_samples = 4 → dampened to 0.5.
        result = prior.get_prior(_examples())
        assert result.effective_confidence == 0.5

    def test_zero_examples_yields_zero_confidence(self) -> None:
        prior = UniformSymbolicPrior(primitive_whitelist=["a"])
        result = prior.get_prior([])
        assert result.effective_confidence == 0.0
        # Distribution still flat over the whitelist.
        assert result.primitive_scores == {"a": 1.0}

    def test_empty_whitelist_raises(self) -> None:
        prior = UniformSymbolicPrior(primitive_whitelist=[])
        with pytest.raises(ValueError, match="empty primitive whitelist"):
            prior.get_prior(_examples())


# ---------------------------------------------------------------------------
# DualPriorMixer
# ---------------------------------------------------------------------------


class _FakeSymbolic(SymbolicPrior):
    """Returns a fixed result regardless of input — for deterministic mixer tests."""

    def __init__(self, scores: dict[str, float], confidence: float) -> None:
        self._scores = scores
        self._confidence = confidence

    def get_prior(self, examples: Any) -> SymbolicPriorResult:
        return SymbolicPriorResult(
            primitive_scores=dict(self._scores),
            effective_confidence=self._confidence,
        )


class TestDualPriorMixer:
    @pytest.mark.asyncio
    async def test_combines_with_alpha_in_band(self) -> None:
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 0.8, "b": 0.2}, hint=0.7),
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a", "b"])
        sym = _FakeSymbolic(scores={"a": 0.4, "b": 0.6}, confidence=0.8)
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples())

        assert isinstance(result, DualPriorResult)
        # α = clamp(0.7) * clamp(0.8) = 0.56
        assert abs(result.alpha - 0.56) < 1e-9
        # Combined sums to 1.0.
        assert abs(sum(result.primitive_scores.values()) - 1.0) < 1e-9
        # Per-side data echoed for telemetry.
        assert result.llm_prior.alpha_entropy_hint == 0.7
        assert result.symbolic_prior.effective_confidence == 0.8

    @pytest.mark.asyncio
    async def test_alpha_clamped_at_band_floor(self) -> None:
        # α_entropy = 0.5 (floor), α_performance = 0.5 → α = 0.25.
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 1.0}, hint=0.0),  # below band → clamps to 0.5
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a"])
        sym = _FakeSymbolic(scores={"a": 1.0}, confidence=0.0)  # clamps to 0.5
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples())
        assert result.alpha == 0.25

    @pytest.mark.asyncio
    async def test_disjoint_keys_unioned_in_output(self) -> None:
        # LLM has only {a, b}; symbolic only {b, c}. Output should
        # contain {a, b, c} (b weighted on both sides).
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 0.5, "b": 0.5}, hint=0.7),
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a", "b", "c"])
        sym = _FakeSymbolic(scores={"b": 0.5, "c": 0.5}, confidence=0.8)
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples())
        assert set(result.primitive_scores) == {"a", "b", "c"}
        # b is weighted from both sides → must outrank a and c.
        assert result.primitive_scores["b"] > result.primitive_scores["a"]
        assert result.primitive_scores["b"] > result.primitive_scores["c"]

    @pytest.mark.asyncio
    async def test_uniform_symbolic_yields_llm_dominance(self) -> None:
        # When the symbolic side is uniform over the same whitelist
        # and the LLM is sharply peaked, the combined distribution
        # should still preserve the LLM's argmax (the keystone test
        # for the mixer being well-behaved at low symbolic confidence).
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 0.9, "b": 0.05, "c": 0.05}, hint=0.8),
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a", "b", "c"])
        sym = UniformSymbolicPrior(primitive_whitelist=["a", "b", "c"])
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples())
        argmax = max(result.primitive_scores, key=result.primitive_scores.__getitem__)
        assert argmax == "a"

    @pytest.mark.asyncio
    async def test_config_override_propagates_to_alpha(self) -> None:
        # Tighter α band (e.g. [0.6, 0.9] · [0.6, 1.0]) yields a higher
        # α floor than the spec default of 0.25.
        config = Phase2Config(
            alpha_entropy_lower=0.6,
            alpha_entropy_upper=0.9,
            alpha_performance_lower=0.6,
            alpha_performance_upper=1.0,
        )
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 1.0}, hint=0.0),  # → clamps to 0.6
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a"], config=config)
        sym = _FakeSymbolic(scores={"a": 1.0}, confidence=0.0)  # → 0.6
        mixer = DualPriorMixer(llm, sym, config=config)
        result = await mixer.get_prior(_examples())
        # α = 0.6 · 0.6 = 0.36, well above the spec-default floor 0.25.
        assert abs(result.alpha - 0.36) < 1e-9
