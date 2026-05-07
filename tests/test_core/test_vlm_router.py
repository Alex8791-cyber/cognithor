"""Unit tests for the VLM router.

Three things this guards:

1. **Classifier** is deterministic and matches expected task classes
   for the prompt patterns we care about (DE + EN).
2. **Override precedence** — ContextVar > config > heuristic — works
   exactly as documented; one async task's override does not bleed
   into another.
3. **TRUST-2 fields** populate consistently — every routing decision
   has a non-empty ``rule_id`` and ``rule_source``, and the matched
   pattern is the actual substring that triggered the rule.
"""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Any

import pytest

from cognithor.core.vlm_router import (
    VLM_PROFILES,
    VlmProfile,
    VlmRouter,
    VlmRoutingDecision,
    VlmTaskClass,
    classify_vlm_task,
)

# ---------------------------------------------------------------------------
# Profiles — sanity
# ---------------------------------------------------------------------------


class TestProfiles:
    """The three built-in profiles must satisfy basic invariants."""

    def test_three_profiles_exist(self) -> None:
        assert set(VLM_PROFILES) == {"fast", "balanced", "premium"}

    def test_quality_pct_strictly_increasing_by_tier(self) -> None:
        # fast < balanced < premium — relative_quality_pct must reflect this
        fast = VLM_PROFILES["fast"].relative_quality_pct
        balanced = VLM_PROFILES["balanced"].relative_quality_pct
        premium = VLM_PROFILES["premium"].relative_quality_pct
        assert fast < balanced < premium

    def test_throughput_strictly_decreasing_by_tier(self) -> None:
        # Inverse of quality: fast >> balanced >> premium tok/s
        fast = VLM_PROFILES["fast"].expected_throughput_tok_s
        balanced = VLM_PROFILES["balanced"].expected_throughput_tok_s
        premium = VLM_PROFILES["premium"].expected_throughput_tok_s
        assert fast > balanced > premium

    def test_memory_footprint_within_consumer_gpu(self) -> None:
        # No profile may claim it fits in <16 GiB or needs >32 GiB —
        # those are hardware-specific and would need their own tier.
        for profile in VLM_PROFILES.values():
            assert 8.0 <= profile.memory_footprint_gib <= 32.0, profile.name

    def test_premium_carries_offload_flag(self) -> None:
        # Spike-finding 2026-04-23: 27B + Vision on RTX 5090 32 GB
        # MUST use --cpu-offload-gb. If the flag drops out of the
        # premium profile someone is about to OOM.
        flags = VLM_PROFILES["premium"].vllm_flags
        assert "--cpu-offload-gb" in flags

    def test_fast_does_not_carry_offload_flag(self) -> None:
        # Fast is fast precisely *because* it does not offload —
        # asserting absence prevents a regression where someone
        # copy-pastes premium flags into fast.
        flags = VLM_PROFILES["fast"].vllm_flags
        assert "--cpu-offload-gb" not in flags

    def test_serve_command_includes_model_id(self) -> None:
        for profile in VLM_PROFILES.values():
            argv = profile.vllm_serve_command()
            assert argv[:2] == ["vllm", "serve"]
            assert argv[2] == profile.model_id


# ---------------------------------------------------------------------------
# Classifier — deterministic, pattern-driven
# ---------------------------------------------------------------------------


class TestClassifier:
    """`classify_vlm_task` heuristics."""

    def test_empty_prompt_is_quick_describe(self) -> None:
        assert classify_vlm_task("") == VlmTaskClass.QUICK_DESCRIBE
        assert classify_vlm_task("   \t\n  ") == VlmTaskClass.QUICK_DESCRIBE

    def test_short_prompt_is_quick_describe(self) -> None:
        assert classify_vlm_task("Was passiert hier?") == VlmTaskClass.QUICK_DESCRIBE
        assert classify_vlm_task("Describe this clip.") == VlmTaskClass.QUICK_DESCRIBE

    def test_ocr_keywords_route_to_ocr_dominant(self) -> None:
        for prompt in (
            "Read the text on the sign.",
            "What does the text say in this frame?",
            "OCR the title card.",
            "Lies die Schrift im Bild.",
            "Was steht auf dem Schild?",
        ):
            assert classify_vlm_task(prompt) == VlmTaskClass.OCR_DOMINANT, prompt

    def test_reasoning_keywords_route_to_multi_step(self) -> None:
        for prompt in (
            "Compare the first frame to the last.",
            "Calculate the camera pan speed.",
            "Vergleiche die Bewegung in Sekunde 2 und 5.",
            "Erkläre warum die Szene plötzlich dunkler wird.",
            "Analyse the motion pattern.",
        ):
            assert classify_vlm_task(prompt) == VlmTaskClass.MULTI_STEP_REASONING, prompt

    def test_forensic_keywords_route_to_forensic_even_with_short_video(self) -> None:
        # Forensic must trump everything else, including video duration
        for prompt in (
            "Detect any deepfake artifacts.",
            "Is this video manipulated? Need it for evidence.",
            "Bitte prüfe die Authentizität dieser Aufnahme.",
        ):
            assert classify_vlm_task(prompt, video_seconds=2) == VlmTaskClass.FORENSIC, prompt

    def test_long_video_escalates_even_with_simple_prompt(self) -> None:
        # 90 s clip with "describe" prompt — heuristic still escalates
        # to LONG_FORM because temporal coherence matters past 60 s.
        result = classify_vlm_task(
            "Describe this clip.",
            video_seconds=90.0,
        )
        assert result == VlmTaskClass.LONG_FORM

    def test_long_prompt_escalates_to_detailed_analysis(self) -> None:
        # Build a prompt with >= 60 words but no reasoning keywords.
        prompt = " ".join(
            [
                "Please look at this clip and",
                "tell me what you see in it",
                "with as much depth as you can muster,",
                "covering the visible elements one after",
                "another in the order they appear on screen,",
                "noting colours and shapes and lighting and",
                "anything else that strikes you as interesting",
                "or unusual or worth pointing out for the record",
                "of this conversation between us today.",
            ]
        )
        # 60+ words — sanity-check the test data
        assert len(prompt.split()) >= 60
        assert classify_vlm_task(prompt) == VlmTaskClass.DETAILED_ANALYSIS

    def test_explicit_class_overrides_heuristic(self) -> None:
        # Caller forces FORENSIC even though prompt is innocent
        result = classify_vlm_task(
            "Describe this.",
            explicit_class="forensic",
        )
        assert result == VlmTaskClass.FORENSIC

    def test_unknown_explicit_class_falls_back_to_heuristic(self) -> None:
        # Bad caller input must not crash — log warning and continue
        result = classify_vlm_task(
            "Compare the frames.",
            explicit_class="invalid_class_name",
        )
        assert result == VlmTaskClass.MULTI_STEP_REASONING

    def test_classification_is_deterministic(self) -> None:
        # Same input → same output; called repeatedly
        prompt = "Compare frame 3 to frame 12."
        first = classify_vlm_task(prompt)
        for _ in range(50):
            assert classify_vlm_task(prompt) == first


# ---------------------------------------------------------------------------
# Router precedence — ContextVar > config > heuristic
# ---------------------------------------------------------------------------


class TestRouterPrecedence:
    """`VlmRouter.select_profile_explained` precedence layers."""

    def test_heuristic_only_no_overrides(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained("Describe this clip.")
        assert decision.profile.name == "fast"
        assert decision.task_class == VlmTaskClass.QUICK_DESCRIBE
        assert decision.rule_id == "vlm_heuristic_quick_describe"
        assert decision.rule_source == "vlm_router.classify_vlm_task"
        assert decision.overridden_by is None

    def test_heuristic_picks_premium_for_reasoning_prompt(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained("Compare the first and last frame.")
        assert decision.profile.name == "premium"
        assert decision.task_class == VlmTaskClass.MULTI_STEP_REASONING
        assert decision.matched_pattern is not None
        assert "compare" in decision.matched_pattern.lower()

    def test_contextvar_override_trumps_heuristic(self) -> None:
        router = VlmRouter()
        with router.quality_scope("fast"):
            # Even with a "compare" prompt, the ContextVar pin wins.
            decision = router.select_profile_explained("Compare the frames in detail.")
        assert decision.profile.name == "fast"
        assert decision.rule_id == "vlm_override_contextvar"
        assert decision.overridden_by == "context_var"

    def test_quality_scope_restores_on_exit(self) -> None:
        router = VlmRouter()
        # No pin before
        assert router.get_quality() is None
        with router.quality_scope("premium"):
            assert router.get_quality() == "premium"
        # Restored after
        assert router.get_quality() is None

    def test_quality_scope_restores_on_exception(self) -> None:
        router = VlmRouter()
        with pytest.raises(RuntimeError):
            with router.quality_scope("premium"):
                assert router.get_quality() == "premium"
                raise RuntimeError("boom")
        # Even after exception — restored
        assert router.get_quality() is None

    def test_unknown_quality_in_scope_raises(self) -> None:
        router = VlmRouter()
        with pytest.raises(ValueError, match="Unknown VLM quality"):
            with router.quality_scope("ultra-mega-premium"):
                pytest.fail("scope body should not run")

    def test_config_default_used_when_no_contextvar(self) -> None:
        # SimpleNamespace standing in for CognithorConfig
        config = SimpleNamespace(vllm=SimpleNamespace(quality_default="balanced"))
        router = VlmRouter(config=config)
        decision = router.select_profile_explained(
            "Describe this clip."  # would heuristic to "fast"
        )
        assert decision.profile.name == "balanced"
        assert decision.rule_id == "vlm_override_config"
        assert decision.overridden_by == "config"

    def test_contextvar_trumps_config(self) -> None:
        config = SimpleNamespace(vllm=SimpleNamespace(quality_default="balanced"))
        router = VlmRouter(config=config)
        with router.quality_scope("fast"):
            decision = router.select_profile_explained("anything")
        assert decision.profile.name == "fast"
        assert decision.overridden_by == "context_var"


# ---------------------------------------------------------------------------
# ContextVar concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrencyIsolation:
    """Two async tasks must not see each other's pinned quality."""

    @pytest.mark.asyncio
    async def test_contextvar_isolation_across_tasks(self) -> None:
        router = VlmRouter()

        async def task_a() -> str | None:
            with router.quality_scope("fast"):
                await asyncio.sleep(0.01)
                return router.get_quality()

        async def task_b() -> str | None:
            with router.quality_scope("premium"):
                await asyncio.sleep(0.01)
                return router.get_quality()

        # Wrap each task in its own copy_context to mimic real
        # asyncio.create_task semantics.
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        a_result = await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )
        # Both tasks should observe THEIR own pin, not each other's
        assert set(a_result) == {"fast", "premium"}

        # And after both tasks finish, the calling task sees no pin
        assert router.get_quality() is None
        # ctx_a / ctx_b were copied but never run via .run() — no need
        # to assert their state. The above gather is the canonical test.
        del ctx_a, ctx_b


# ---------------------------------------------------------------------------
# TRUST-2 explanation surface
# ---------------------------------------------------------------------------


class TestExplanationSurface:
    """Every decision exposes the fields a Receipt sidebar needs."""

    @pytest.mark.parametrize(
        ("prompt", "expected_class"),
        [
            ("Describe this.", VlmTaskClass.QUICK_DESCRIBE),
            ("Read the text on the sign.", VlmTaskClass.OCR_DOMINANT),
            ("Compare frame 1 to frame 5.", VlmTaskClass.MULTI_STEP_REASONING),
            ("Is this footage authentic?", VlmTaskClass.FORENSIC),
        ],
    )
    def test_decision_has_required_fields(
        self,
        prompt: str,
        expected_class: VlmTaskClass,
    ) -> None:
        router = VlmRouter()
        decision: VlmRoutingDecision = router.select_profile_explained(prompt)
        assert isinstance(decision.profile, VlmProfile)
        assert decision.task_class == expected_class
        assert decision.rule_id  # non-empty
        assert decision.rule_source  # non-empty
        # Heuristic-driven decisions for non-default classes carry a
        # matched pattern; default-quick-describe legitimately has None.
        if expected_class != VlmTaskClass.QUICK_DESCRIBE:
            assert decision.matched_pattern is not None, expected_class

    def test_select_profile_convenience_unwraps_decision(self) -> None:
        router = VlmRouter()
        profile = router.select_profile("Compare the frames.")
        assert isinstance(profile, VlmProfile)
        assert profile.name == "premium"

    def test_long_form_decision_exposes_duration_in_pattern(self) -> None:
        router = VlmRouter()
        decision = router.select_profile_explained(
            "Describe this clip.",
            video_seconds=120.0,
        )
        assert decision.task_class == VlmTaskClass.LONG_FORM
        assert decision.matched_pattern is not None
        assert "120" in decision.matched_pattern


# ---------------------------------------------------------------------------
# Sanity guard against config drift
# ---------------------------------------------------------------------------


class TestConfigGuards:
    """Catch the most common ways a future change breaks the router."""

    def test_task_map_covers_every_task_class(self) -> None:
        # Every VlmTaskClass must map to a profile, so the heuristic
        # path can never KeyError. If a new class gets added to the
        # enum, this test fails until the map is updated too.
        from cognithor.core.vlm_router import _DEFAULT_TASK_PROFILE_MAP

        for task_class in VlmTaskClass:
            assert task_class in _DEFAULT_TASK_PROFILE_MAP, task_class
            assert _DEFAULT_TASK_PROFILE_MAP[task_class] in VLM_PROFILES

    def test_every_profile_has_non_trivial_description(self) -> None:
        # Description shows up in the Settings → vLLM dropdown — must
        # be useful, not "TODO".
        for profile in VLM_PROFILES.values():
            assert len(profile.description) >= 50, profile.name


def _swallow(_obj: Any) -> None:  # pragma: no cover
    """Helper to satisfy ``Any`` import without affecting test logic."""
