# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-10 Track B — LLM-Prior wiring smoke tests.

Validates that the benchmark runner constructs the production
LLM-Prior stack (vLLM → LLMPriorClient → DualPriorMixer) without
requiring a running vLLM server. Runtime correctness against an
actual qwen3.6:27b instance is out of scope for unit tests — this
module proves the wiring is *constructable* and that the CLI flags
parse and reach the engine.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.benchmark_runner import (
    _build_dual_prior_stack,
    _build_phase2_engine,
    _parse_args,
)


class TestDualPriorStackConstruction:
    def test_constructs_with_default_qwen_endpoint(self) -> None:
        # Default vLLM endpoint + qwen3.6:27b. No actual call is made.
        mixer = _build_dual_prior_stack(
            base_url="http://localhost:8000/v1",
            model_name="Qwen/Qwen3.6-27B-Instruct",
        )
        # Mixer wraps an LLMPriorClient + UniformSymbolicPrior; both
        # private but we can check it's the right class.
        from cognithor.channels.program_synthesis.phase2 import DualPriorMixer

        assert isinstance(mixer, DualPriorMixer)

    def test_constructs_with_alternate_endpoint(self) -> None:
        mixer = _build_dual_prior_stack(
            base_url="http://gpu-host:9000/v1",
            model_name="Qwen/Qwen3.6-27B-Instruct-AWQ",
        )
        from cognithor.channels.program_synthesis.phase2 import DualPriorMixer

        assert isinstance(mixer, DualPriorMixer)


class TestPhase2EngineAcceptsDualPrior:
    def test_engine_built_with_dual_prior(self) -> None:
        mixer = _build_dual_prior_stack(
            base_url="http://localhost:8000/v1",
            model_name="Qwen/Qwen3.6-27B-Instruct",
        )
        engine = _build_phase2_engine(dual_prior=mixer)
        # WiredPhase2Engine — exact class assertion via private attr.
        assert engine._dual_prior is mixer  # type: ignore[attr-defined]

    def test_engine_built_without_dual_prior_defaults_to_none(self) -> None:
        engine = _build_phase2_engine()
        assert engine._dual_prior is None  # type: ignore[attr-defined]


class TestCLIFlags:
    def test_default_flags(self) -> None:
        args = _parse_args(["--output", "out.json"])
        assert args.llm_prior is False
        assert args.llm_base_url == "http://localhost:8000/v1"
        assert args.llm_model == "Qwen/Qwen3.6-27B-Instruct"

    def test_llm_prior_flag(self) -> None:
        args = _parse_args(
            [
                "--output",
                "out.json",
                "--phase2",
                "--llm-prior",
            ]
        )
        assert args.llm_prior is True
        assert args.phase2 is True

    def test_llm_endpoint_override(self) -> None:
        args = _parse_args(
            [
                "--output",
                "out.json",
                "--llm-base-url",
                "http://gpu-host:9000/v1",
                "--llm-model",
                "Qwen/Qwen3.6-27B-Instruct-AWQ",
            ]
        )
        assert args.llm_base_url == "http://gpu-host:9000/v1"
        assert args.llm_model == "Qwen/Qwen3.6-27B-Instruct-AWQ"
