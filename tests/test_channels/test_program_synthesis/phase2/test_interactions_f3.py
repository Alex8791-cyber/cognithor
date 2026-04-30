# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""F3 — Wechselwirkungs-Tests (spec v1.4 §12.2).

Twelve interaction tests across three risk categories:

* §12.2.1 — **E1 × E2** Search-α vs Repair-α asymmetry.
* §12.2.2 — **E3 × E1** LLM dominance with few demos.
* §12.2.3 — **E6 × E7** Reward hacking via High-Impact arguments.

Sprint-1 surface is sufficient to exercise the *current* claims:
config defaults, mode-controller behaviour, mixer math, suspicion
ordering. Sprint-2 piece (heuristic catalog + ArgumentQualityFaktor)
adds the non-uniform branches the spec reserves; those tests assert
the Sprint-1 baseline (reserves OFF, uniform symbolic prior) and
re-assert the same invariants after a Sprint-2 PR flips the
config.
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
    DEFAULT_PHASE2_CONFIG,
    DualPriorMixer,
    LLMPriorClient,
    Phase2Config,
    UniformSymbolicPrior,
    alpha_bounds,
    apply_sample_size_dampening,
    compute_suspicion,
    mix_alpha,
)
from cognithor.channels.program_synthesis.refiner import RefinerModeController
from cognithor.channels.program_synthesis.search.candidate import InputRef, Program
from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendType,
)

# ---------------------------------------------------------------------------
# Helpers
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


def _llm_response(scores: dict[str, float], *, hint: float) -> str:
    return json.dumps({**scores, "alpha_entropy_hint": hint})


def _g(name: str) -> Program:
    return Program(primitive=name, children=(InputRef(),), output_type="Grid")


def _examples(n: int) -> list[tuple[Any, Any]]:
    return [([[i]], [[i + 1]]) for i in range(n)]


# ---------------------------------------------------------------------------
# §12.2.1 — E1 × E2 — Search-α vs Repair-α asymmetry
# ---------------------------------------------------------------------------


class TestE1xE2_SearchVsRepairAlphaAsymmetry:
    """Search-α (in [0.25, 0.85]) and Repair-α-Schwellen (0.35/0.45)
    must stay distinct concepts. Spec §4.4.4 documents this; the
    runtime behaviour must match the documentation."""

    def test_search_uses_llm_at_alpha_below_repair_threshold(self) -> None:
        # α = 0.3 sits below the Repair zone1_lower (0.45) and below
        # the Repair zone3_upper (0.35) — Refiner picks "symbolic".
        # But Search-α 0.3 is still inside the Search band [0.25, 0.85],
        # so MCTS's LLM query is *valid*. The asymmetry is intentional.
        controller = RefinerModeController()
        repair_mode = controller.select_mode(0.30)
        assert repair_mode == "symbolic"

        # Search-α band still includes 0.3.
        lo, hi = alpha_bounds()
        assert lo <= 0.30 <= hi

    def test_documentation_of_alpha_semantics_present(self) -> None:
        # Spec §12.2.1 mandates code-side documentation of the
        # asymmetry. Verify the canonical files mention both concepts.
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]
        mixer_src = (
            root / "src/cognithor/channels/program_synthesis/phase2/alpha_mixer.py"
        ).read_text(encoding="utf-8")
        mode_src = (
            root / "src/cognithor/channels/program_synthesis/refiner/mode_controller.py"
        ).read_text(encoding="utf-8")

        # Each file references its own α concept.
        assert "Search-α" in mixer_src or "search" in mixer_src.lower()
        assert "α" in mode_src or "alpha" in mode_src.lower()

    def test_telemetry_distinguishes_alpha_uses(self) -> None:
        # Mixer + RefinerModeController emit telemetry independently:
        # phase2_counters has separate counters for refiner_mode and
        # for the mixer's downstream consumers. Verify the counter
        # name-set carries the spec-mandated split.
        from cognithor.channels.program_synthesis.observability.metrics import (
            Registry,
        )
        from cognithor.channels.program_synthesis.phase2 import phase2_counters

        registry = Registry()
        phase2_counters(registry=registry)
        emitted = set(registry.snapshot().counters)
        # refiner_mode_total: tracks Repair-α decisions.
        assert "cognithor_synthesis_refiner_mode_total" in emitted
        # The mixer's α value is histogram-able (Sprint-2 wires it);
        # for now we only assert the refiner side is split out.
        assert "cognithor_synthesis_refiner_mode_hysteresis_held_total" in emitted


# ---------------------------------------------------------------------------
# §12.2.2 — E3 × E1 — LLM dominance with few demos
# ---------------------------------------------------------------------------


class TestE3xE1_LLMDominanceFewDemos:
    """At low n_demos the symbolic side's effective_confidence is
    dampened (n/(n+n0)). The LLM's α_entropy_hint defaults toward the
    upper band. The asymmetric pair could push α high, letting the
    LLM dominate. Spec §12.2.2 ensures the floor mechanics keep the
    symbolic minimum-contribution non-trivial."""

    @pytest.mark.asyncio
    async def test_few_demos_llm_not_overdominant(self) -> None:
        # 1 demo + an over-confident LLM (hint=0.85, top of band).
        # Symbolic confidence is dampened to 1/(1+4) = 0.2 → clamped
        # to alpha_performance band floor (0.5).
        # α = 0.85 · 0.5 = 0.425 — well below the band ceiling.
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 1.0}, hint=0.85),
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a", "b"])
        sym = UniformSymbolicPrior(primitive_whitelist=["a", "b"])
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples(1))

        # α ≤ ceiling. Spec invariant: α never exceeds 0.85.
        _, hi = alpha_bounds()
        assert result.alpha <= hi
        # Symbolic side contributes (1−α) which must be ≥ 0.15
        # so the symbolic prior cannot be silently zeroed.
        assert (1.0 - result.alpha) >= 0.15

    @pytest.mark.asyncio
    async def test_symbolic_minimum_contribution_below_4_demos(self) -> None:
        # Below 4 demos, (1−α) · π_symbolic must still contribute at
        # least ~15 % of the top-1 weight. Verified through the floor:
        # α ≤ 0.85 → (1−α) ≥ 0.15.
        backend = _StubBackend(
            queued=[
                "reasoning",
                _llm_response({"a": 1.0}, hint=0.85),
            ]
        )
        llm = LLMPriorClient(backend, primitive_whitelist=["a"])
        sym = UniformSymbolicPrior(primitive_whitelist=["a"])
        mixer = DualPriorMixer(llm, sym)
        result = await mixer.get_prior(_examples(2))
        assert (1.0 - result.alpha) >= 0.15

    def test_few_demos_dampening_reserve_stays_off_in_sprint1(self) -> None:
        # Spec §22.4.2: enable_few_demos_dampening is OFF until the
        # interaction tests demand it. Sprint-1 ships with the
        # default Phase2Config — the reserve must be OFF.
        assert DEFAULT_PHASE2_CONFIG.enable_few_demos_dampening is False
        # And the underlying dampening helper is well-defined for n<4
        # so flipping the reserve later doesn't introduce new errors.
        assert apply_sample_size_dampening(1.0, 1) > 0.0
        assert apply_sample_size_dampening(1.0, 1) < apply_sample_size_dampening(1.0, 4)


# ---------------------------------------------------------------------------
# §12.2.3 — E6 × E7 — Reward hacking via High-Impact arguments
# ---------------------------------------------------------------------------


class TestE6xE7_RewardHackingHighImpactArgs:
    """High-Impact whitelist + 3× multiplier could let an LLM exploit
    an arg-confused High-Impact primitive: same partial score, but
    suspicion-free because of class boost. Spec §12.2.3 mandates
    the boost not be unbounded."""

    def test_high_impact_with_bad_args_still_below_neutralisation(self) -> None:
        # 50 adversarial 1-tokener High-Impact programs at high
        # partial_score. Spec asserts: suspicion never reaches ~1.0
        # (which would mean "totally legitimate"). The Sprint-1
        # formula yields suspicion = partial · (1 − sc). For a
        # 1-tokener tile with default config:
        #   sc = 0.6·(3/12) + 0.4·(1/6) = 0.15 + 0.067 = 0.217
        #   suspicion = 0.85 · (1 − 0.217) ≈ 0.665
        # We assert the looser invariant: suspicion < partial_score
        # (the boost is meaningful but never neutralises completely).
        partial = 0.85
        for prim in (
            "tile",
            "rotate90",
            "mirror_horizontal",
            "transpose",
            "scale_up_2x",
        ):
            s = compute_suspicion(_g(prim), partial_score=partial)
            assert s.value < partial, (
                f"{prim} 1-tokener neutralised suspicion at partial={partial}; "
                f"spec §12.2.3 requires boost to be bounded."
            )

    def test_argument_quality_factor_reserve_off_in_sprint1(self) -> None:
        # Spec §22.4.2 — the reserve is gated.
        assert DEFAULT_PHASE2_CONFIG.enable_argument_quality_factor is False

    def test_no_suspicion_neutralization_complete(self) -> None:
        # Even at perfect partial_score, a depth-1 High-Impact 1-tokener
        # cannot have suspicion=0 (which would mean "totally trusted"
        # — that's reserved for deep, complex programs). The Sprint-1
        # formula gives sc < 1 for any depth-1 tree, so suspicion > 0.
        s = compute_suspicion(_g("tile"), partial_score=1.0)
        assert s.value > 0.0
        # And it scales with partial: suspicion(0.5) < suspicion(1.0).
        s_half = compute_suspicion(_g("tile"), partial_score=0.5)
        assert s_half.value < s.value


# ---------------------------------------------------------------------------
# Cross-section: spec §12.2 self-consistency (Sprint-1 invariants)
# ---------------------------------------------------------------------------


class TestSpecSelfConsistencyAcrossSprints:
    """A small set of cross-cutting invariants that must hold both
    in Sprint-1 (today) and after Sprint-2 lights up the heuristic
    catalog. Pinning them now means Sprint-2 has to keep them."""

    def test_reserves_default_off_signals_explicit_opt_in(self) -> None:
        # F3 reserves enabled together would silently skew α and
        # suspicion. Both must default OFF, requiring explicit
        # config.replace() to enable.
        assert DEFAULT_PHASE2_CONFIG.enable_few_demos_dampening is False
        assert DEFAULT_PHASE2_CONFIG.enable_argument_quality_factor is False

    def test_search_alpha_band_subset_of_unit_interval(self) -> None:
        lo, hi = alpha_bounds()
        assert 0.0 < lo < hi < 1.0

    def test_repair_alpha_zones_inside_search_alpha_band(self) -> None:
        # Sanity: Repair zone1/zone3 thresholds must lie inside the
        # Search-α band — otherwise the Refiner could pick a mode that
        # the Search engine never produces.
        lo, hi = alpha_bounds()
        assert lo <= DEFAULT_PHASE2_CONFIG.repair_alpha_zone3_upper
        assert DEFAULT_PHASE2_CONFIG.repair_alpha_zone1_lower <= hi

    def test_mix_alpha_clamping_does_not_silently_invert(self) -> None:
        # Bug-class: a misbehaving performance tracker reporting a
        # negative confidence shouldn't silently produce a positive
        # α via the (-1) · (-1) = 1 shortcut. Verify mix_alpha clamps
        # both factors to their lower bounds rather than letting the
        # product flip sign.
        result = mix_alpha(-1.0, -1.0)
        lo, _ = alpha_bounds()
        assert result == lo
        # And not the unsafe positive product.
        assert result > 0.0

    def test_phase2_config_invariants_block_mixer_pathology(self) -> None:
        # An A/B that pushes the alpha_performance lower bound to 0
        # would let the mixer return α=0 (pure-symbolic). The current
        # Phase2Config invariants allow this — but require the upper
        # to remain ≥ lower, blocking inverted bands.
        with pytest.raises(ValueError, match="alpha_performance"):
            Phase2Config(alpha_performance_lower=0.7, alpha_performance_upper=0.3)
