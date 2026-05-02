# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — game_prompts registry tests."""

from __future__ import annotations

from cognithor.channels.program_synthesis.arc_agi3.game_prompts import (
    CLICK_FAMILY_HINT,
    GAME_PROMPTS,
    GENERIC_CONTEXT,
    LS20_LOCKSMITH_RULES,
    build_system_prompt,
    game_prefix,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestGamePrefix:
    def test_with_dash(self) -> None:
        assert game_prefix("ls20-0a0ad940") == "ls20"

    def test_without_dash(self) -> None:
        assert game_prefix("ls20") == "ls20"

    def test_short_id(self) -> None:
        assert game_prefix("ft09") == "ft09"


class TestBuildSystemPrompt:
    def test_known_game_includes_rules(self) -> None:
        prompt = build_system_prompt("ls20-0a0ad940", "RESET, ACTION1, ACTION2")
        assert "LockSmith" in prompt
        assert "INT<10>" in prompt  # walls colour
        assert "ACTION1: move up" in prompt
        assert "RESET, ACTION1, ACTION2" in prompt  # action whitelist

    def test_unknown_game_falls_back_to_generic(self) -> None:
        prompt = build_system_prompt("xx99-deadbeef", "RESET, ACTION1")
        assert "INT<0,15>" in prompt  # generic context
        assert "LockSmith" not in prompt  # no game-specific rules
        assert "RESET, ACTION1" in prompt  # whitelist still present

    def test_ft09_includes_click_hint(self) -> None:
        prompt = build_system_prompt("ft09-abc123", "RESET, ACTION1")
        assert "ft09" in prompt or "click" in prompt.lower()

    def test_click_family_games_get_hint(self) -> None:
        for prefix in ["bp35", "cn04", "sk48", "ar25", "lp85"]:
            prompt = build_system_prompt(f"{prefix}-test", "RESET, ACTION6")
            assert "ACTION6" in prompt

    def test_action_whitelist_in_output_schema(self) -> None:
        prompt = build_system_prompt("ls20", "ACTION1, ACTION2, RESET")
        # The output-schema section must reference the actual whitelist.
        assert "[ACTION1, ACTION2, RESET]" in prompt

    def test_behavioural_guidelines_always_present(self) -> None:
        prompt = build_system_prompt("ls20", "ACTION1")
        assert "Explore" in prompt
        assert "If an action repeatedly does nothing" in prompt


class TestRegistryContents:
    def test_ls20_rules_verbatim_quality(self) -> None:
        # Verbatim from upstream — these specific phrases MUST be present.
        assert "INT<10>" in LS20_LOCKSMITH_RULES
        assert "energy pills" in LS20_LOCKSMITH_RULES
        assert "scaled down 2X" in LS20_LOCKSMITH_RULES
        assert "rotator" in LS20_LOCKSMITH_RULES

    def test_click_family_hint_mentions_action6(self) -> None:
        assert "ACTION6" in CLICK_FAMILY_HINT

    def test_generic_context_short_and_focused(self) -> None:
        # Should be the upstream verbatim block — under 500 chars.
        assert len(GENERIC_CONTEXT) < 500
        assert "WIN" in GENERIC_CONTEXT
        assert "GAME_OVER" in GENERIC_CONTEXT

    def test_at_least_15_games_registered(self) -> None:
        # Mirrors the 25 official games we saw in get_environments().
        assert len(GAME_PROMPTS) >= 15
