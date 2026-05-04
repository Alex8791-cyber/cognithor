"""Tests for the TRUST-5 PermissionScope foundation."""

from __future__ import annotations

import dataclasses

import pytest

from cognithor.models import RiskLevel
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
    ScopeViolation,
)


class TestPermissionScope:
    def test_minimal_scope(self) -> None:
        scope = PermissionScope(axis=ScopeAxis.CHANNEL, identity="telegram")
        assert scope.key == ("channel", "telegram")
        assert scope.max_risk == RiskLevel.RED
        assert scope.tool_allowlist == frozenset()
        assert scope.tool_denylist == frozenset()

    def test_empty_identity_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            PermissionScope(axis=ScopeAxis.CHANNEL, identity="")

    def test_overlap_allowlist_denylist_rejected(self) -> None:
        with pytest.raises(ValueError, match="both"):
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset({"web_fetch"}),
                tool_denylist=frozenset({"web_fetch"}),
            )

    def test_frozen_via_dataclass(self) -> None:
        scope = PermissionScope(axis=ScopeAxis.USER, identity="alex")
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.identity = "bob"  # type: ignore[misc]


class TestScopeRegistryBasic:
    def test_empty_registry_passes_everything(self) -> None:
        reg = ScopeRegistry()
        verdict = reg.evaluate([], "any_tool", RiskLevel.GREEN)
        assert verdict.allowed
        assert verdict.reasons == ()

    def test_register_get_remove(self) -> None:
        reg = ScopeRegistry()
        scope = PermissionScope(axis=ScopeAxis.USER, identity="alex")
        reg.register(scope)
        assert reg.get(ScopeAxis.USER, "alex") is scope
        assert len(reg) == 1
        assert reg.remove(ScopeAxis.USER, "alex") is True
        assert reg.get(ScopeAxis.USER, "alex") is None
        assert reg.remove(ScopeAxis.USER, "alex") is False

    def test_register_replaces_existing(self) -> None:
        reg = ScopeRegistry()
        reg.register(PermissionScope(axis=ScopeAxis.USER, identity="alex"))
        replacement = PermissionScope(
            axis=ScopeAxis.USER,
            identity="alex",
            tool_denylist=frozenset({"shell"}),
        )
        reg.register(replacement)
        assert reg.get(ScopeAxis.USER, "alex") is replacement

    def test_list_scopes_sorted(self) -> None:
        reg = ScopeRegistry()
        for axis, ident in (
            (ScopeAxis.USER, "bob"),
            (ScopeAxis.CHANNEL, "slack"),
            (ScopeAxis.USER, "alex"),
        ):
            reg.register(PermissionScope(axis=axis, identity=ident))
        keys = [s.key for s in reg.list_scopes()]
        assert keys == [("channel", "slack"), ("user", "alex"), ("user", "bob")]

    def test_clear(self) -> None:
        reg = ScopeRegistry()
        reg.register(PermissionScope(axis=ScopeAxis.USER, identity="x"))
        reg.clear()
        assert len(reg) == 0


class TestScopeEvaluation:
    def test_denylist_blocks(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="telegram",
                tool_denylist=frozenset({"shell"}),
            )
        )
        verdict = reg.evaluate([(ScopeAxis.CHANNEL, "telegram")], "shell", RiskLevel.GREEN)
        assert verdict.denied
        assert "denylist" in verdict.reasons[0]

    def test_allowlist_outside_blocks(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset({"web_fetch", "memory_search"}),
            )
        )
        verdict = reg.evaluate([(ScopeAxis.USER, "alex")], "shell", RiskLevel.GREEN)
        assert verdict.denied
        assert "not in allowlist" in verdict.reasons[0]

    def test_allowlist_member_passes(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset({"web_fetch"}),
            )
        )
        verdict = reg.evaluate([(ScopeAxis.USER, "alex")], "web_fetch", RiskLevel.GREEN)
        assert verdict.allowed

    def test_max_risk_blocks_higher(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="cron",
                max_risk=RiskLevel.YELLOW,
            )
        )
        verdict = reg.evaluate([(ScopeAxis.CHANNEL, "cron")], "shell", RiskLevel.RED)
        assert verdict.denied
        assert "max_risk" in verdict.reasons[0]

    def test_max_risk_passes_equal_or_lower(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="cron",
                max_risk=RiskLevel.YELLOW,
            )
        )
        # GREEN is below ceiling
        v1 = reg.evaluate([(ScopeAxis.CHANNEL, "cron")], "memory_search", RiskLevel.GREEN)
        assert v1.allowed
        # YELLOW equals ceiling — allowed
        v2 = reg.evaluate([(ScopeAxis.CHANNEL, "cron")], "create_file", RiskLevel.YELLOW)
        assert v2.allowed

    def test_unknown_scope_key_skipped(self) -> None:
        reg = ScopeRegistry()
        verdict = reg.evaluate([(ScopeAxis.USER, "stranger")], "any_tool", RiskLevel.GREEN)
        assert verdict.allowed  # no scope ⇒ no restriction

    def test_most_restrictive_wins_across_axes(self) -> None:
        reg = ScopeRegistry()
        # User allows web_fetch
        reg.register(
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset({"web_fetch"}),
            )
        )
        # Channel denylists web_fetch
        reg.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="cron",
                tool_denylist=frozenset({"web_fetch"}),
            )
        )
        # Cron-channel + alex-user — cron's denylist wins
        verdict = reg.evaluate(
            [(ScopeAxis.CHANNEL, "cron"), (ScopeAxis.USER, "alex")],
            "web_fetch",
            RiskLevel.GREEN,
        )
        assert verdict.denied

    def test_denylist_beats_allowlist(self) -> None:
        # Same axis: a single scope can't have a tool in both lists
        # (constructor rejects), but two scopes on different axes can
        # disagree. Denylist always wins.
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.WORKFLOW,
                identity="morning_brief",
                tool_denylist=frozenset({"shell"}),
            )
        )
        reg.register(
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset({"shell"}),
            )
        )
        verdict = reg.evaluate(
            [
                (ScopeAxis.WORKFLOW, "morning_brief"),
                (ScopeAxis.USER, "alex"),
            ],
            "shell",
            RiskLevel.GREEN,
        )
        assert verdict.denied
        assert "denylist" in verdict.reasons[0]


class TestAssertAllowed:
    def test_passes_silently_when_allowed(self) -> None:
        reg = ScopeRegistry()
        # Empty registry → always allowed
        reg.assert_allowed([(ScopeAxis.CHANNEL, "telegram")], "any_tool", RiskLevel.GREEN)

    def test_raises_violation_on_deny(self) -> None:
        reg = ScopeRegistry()
        reg.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="cron",
                max_risk=RiskLevel.YELLOW,
            )
        )
        with pytest.raises(ScopeViolation) as exc_info:
            reg.assert_allowed([(ScopeAxis.CHANNEL, "cron")], "shell", RiskLevel.RED)
        violation = exc_info.value
        assert violation.tool == "shell"
        assert violation.axis == "channel"
        assert violation.identity == "cron"
        assert "max_risk" in violation.reason


class TestScopeViolation:
    def test_message_includes_context(self) -> None:
        v = ScopeViolation(
            axis="channel",
            identity="cron",
            tool="shell",
            reason="max_risk exceeded",
        )
        msg = str(v)
        assert "cron" in msg
        assert "shell" in msg
        assert "max_risk" in msg
