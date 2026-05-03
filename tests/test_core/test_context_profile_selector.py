# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tests for Sprint-23 PR#A — pure heuristic context-profile selector."""

from __future__ import annotations

from cognithor.core.context_profile_selector import (
    PROFILE_ARC_AGI3,
    PROFILE_DEEP,
    PROFILE_DEFAULT,
    PROFILE_QUICK,
    ProfileRecommendation,
    estimate_prompt_tokens,
    recommend_context_profile,
)
from cognithor.core.model_router import CONTEXT_PROFILES

# ---------------------------------------------------------------------------
# Cross-module pin: selector profile names match the router's registry keys.
# Without this, the selector would silently route to a profile name the
# router refuses, and ``set_context_profile`` would raise at runtime.
# ---------------------------------------------------------------------------


class TestProfileNameRegistryAlignment:
    def test_all_constants_map_to_registry_keys(self) -> None:
        for name in (PROFILE_QUICK, PROFILE_DEFAULT, PROFILE_DEEP, PROFILE_ARC_AGI3):
            assert name in CONTEXT_PROFILES, (
                f"Selector advertises profile {name!r} that is not in the "
                f"model_router CONTEXT_PROFILES registry"
            )


# ---------------------------------------------------------------------------
# estimate_prompt_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_zero_chars_zero_tokens(self) -> None:
        assert estimate_prompt_tokens(0) == 0

    def test_negative_clamped_to_zero(self) -> None:
        assert estimate_prompt_tokens(-100) == 0

    def test_one_token_per_four_chars(self) -> None:
        assert estimate_prompt_tokens(4) == 1
        assert estimate_prompt_tokens(40) == 10
        assert estimate_prompt_tokens(4_000) == 1_000

    def test_floor_division(self) -> None:
        # 7 chars -> 1 token, not 1.75.
        assert estimate_prompt_tokens(7) == 1


# ---------------------------------------------------------------------------
# Channel-kind override rule (priority 1)
# ---------------------------------------------------------------------------


class TestChannelKindOverride:
    def test_arc_agi3_channel_forces_arc_profile_on_short_prompt(self) -> None:
        rec = recommend_context_profile(prompt_chars=10, channel_kind="arc_agi3")
        assert rec.profile == PROFILE_ARC_AGI3
        assert "game-loop" in rec.reason

    def test_arc_channel_alias_forces_arc_profile(self) -> None:
        rec = recommend_context_profile(prompt_chars=100, channel_kind="arc_channel")
        assert rec.profile == PROFILE_ARC_AGI3

    def test_game_loop_kind_forces_arc_profile(self) -> None:
        rec = recommend_context_profile(prompt_chars=100, channel_kind="game_loop")
        assert rec.profile == PROFILE_ARC_AGI3

    def test_channel_kind_is_case_insensitive(self) -> None:
        rec = recommend_context_profile(prompt_chars=10, channel_kind="ARC_AGI3")
        assert rec.profile == PROFILE_ARC_AGI3


# ---------------------------------------------------------------------------
# Long-prompt escalation
# ---------------------------------------------------------------------------


class TestLongPromptEscalation:
    def test_very_long_prompt_routes_to_arc(self) -> None:
        # 50 000 tokens worth of chars (200 000 chars) → above deep
        # ceiling, below arc ceiling — but tokens >= deep_ceiling
        # pushes straight to arc_agi3.
        rec = recommend_context_profile(prompt_tokens=50_000)
        assert rec.profile == PROFILE_ARC_AGI3

    def test_arc_threshold_with_attachments(self) -> None:
        # At exactly the arc ceiling with attachments → arc_agi3.
        rec = recommend_context_profile(prompt_tokens=96_000, has_attachments=True)
        assert rec.profile == PROFILE_ARC_AGI3

    def test_long_prompt_routes_to_deep(self) -> None:
        rec = recommend_context_profile(prompt_tokens=30_000)
        assert rec.profile == PROFILE_DEEP


# ---------------------------------------------------------------------------
# Attachments tilt
# ---------------------------------------------------------------------------


class TestAttachmentsTilt:
    def test_medium_prompt_with_attachments_escalates_to_deep(self) -> None:
        # 10 000 tokens (medium) + attachments → deep, not default.
        rec = recommend_context_profile(prompt_tokens=10_000, has_attachments=True)
        assert rec.profile == PROFILE_DEEP

    def test_short_prompt_with_attachments_stays_default(self) -> None:
        # Below the quick ceiling, attachments alone shouldn't push
        # all the way to deep — default is enough headroom.
        rec = recommend_context_profile(
            prompt_tokens=2_000,
            channel_kind="webui",
            has_attachments=True,
        )
        assert rec.profile == PROFILE_DEFAULT


# ---------------------------------------------------------------------------
# Quick path
# ---------------------------------------------------------------------------


class TestQuickPath:
    def test_short_prompt_simple_channel_routes_to_quick(self) -> None:
        rec = recommend_context_profile(prompt_chars=200, channel_kind="cli")
        assert rec.profile == PROFILE_QUICK

    def test_short_prompt_telegram_routes_to_quick(self) -> None:
        rec = recommend_context_profile(prompt_chars=500, channel_kind="telegram")
        assert rec.profile == PROFILE_QUICK

    def test_short_prompt_no_channel_falls_back_to_default(self) -> None:
        # Without a channel hint, we default to ``default`` rather than
        # ``quick`` — a missing channel is uninformative, and ``quick``
        # is only safe for genuinely simple chat surfaces.
        rec = recommend_context_profile(prompt_chars=200)
        assert rec.profile == PROFILE_DEFAULT

    def test_short_prompt_unknown_channel_falls_back_to_default(self) -> None:
        rec = recommend_context_profile(prompt_chars=200, channel_kind="unknown_kind")
        assert rec.profile == PROFILE_DEFAULT


# ---------------------------------------------------------------------------
# Default middle
# ---------------------------------------------------------------------------


class TestDefaultPath:
    def test_medium_prompt_routes_to_default(self) -> None:
        rec = recommend_context_profile(prompt_tokens=10_000)
        assert rec.profile == PROFILE_DEFAULT

    def test_medium_prompt_simple_channel_still_default(self) -> None:
        # Simple-channel quick path only fires below the quick ceiling;
        # 10 000 tokens is medium so default wins even on telegram.
        rec = recommend_context_profile(prompt_tokens=10_000, channel_kind="telegram")
        assert rec.profile == PROFILE_DEFAULT


# ---------------------------------------------------------------------------
# Output dataclass shape
# ---------------------------------------------------------------------------


class TestRecommendationDataclass:
    def test_recommendation_is_frozen(self) -> None:
        rec = ProfileRecommendation(profile="quick", reason="x")
        try:
            rec.profile = "default"  # type: ignore[misc]
        except (AttributeError, Exception):
            return
        raise AssertionError("ProfileRecommendation should be frozen")

    def test_reason_is_non_empty_for_every_branch(self) -> None:
        # Walk every routing branch and assert each carries a useful
        # reason string. A blank reason in a log line is a debugging
        # nightmare, so this is a hard contract.
        cases: list[tuple[dict[str, object], str]] = [
            ({"channel_kind": "arc_agi3"}, PROFILE_ARC_AGI3),
            ({"prompt_tokens": 50_000}, PROFILE_ARC_AGI3),
            ({"prompt_tokens": 30_000}, PROFILE_DEEP),
            (
                {"prompt_tokens": 10_000, "has_attachments": True},
                PROFILE_DEEP,
            ),
            ({"prompt_chars": 200, "channel_kind": "cli"}, PROFILE_QUICK),
            ({"prompt_tokens": 10_000}, PROFILE_DEFAULT),
        ]
        for kwargs, expected in cases:
            rec = recommend_context_profile(**kwargs)  # type: ignore[arg-type]
            assert rec.profile == expected, (kwargs, rec)
            assert rec.reason and rec.reason.strip(), kwargs
