"""Tests for ``ScopeRegistry.from_config`` (TRUST-5 production wiring)."""

from __future__ import annotations

from cognithor.models import RiskLevel
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
)


class TestFromConfigEmptyShapes:
    def test_none_returns_empty_registry(self) -> None:
        reg = ScopeRegistry.from_config(None)
        assert len(reg) == 0

    def test_empty_list_returns_empty_registry(self) -> None:
        reg = ScopeRegistry.from_config([])
        assert len(reg) == 0

    def test_dict_with_empty_scopes_key(self) -> None:
        reg = ScopeRegistry.from_config({"scopes": []})
        assert len(reg) == 0

    def test_invalid_shape_returns_empty(self) -> None:
        # Strings, ints, etc. are not list/dict — registry stays empty.
        reg = ScopeRegistry.from_config("not a list")  # type: ignore[arg-type]
        assert len(reg) == 0


class TestFromConfigList:
    def test_minimal_scope(self) -> None:
        reg = ScopeRegistry.from_config([{"axis": "channel", "identity": "telegram"}])
        assert len(reg) == 1
        scope = reg.get(ScopeAxis.CHANNEL, "telegram")
        assert scope is not None
        assert scope.max_risk == RiskLevel.RED  # default
        assert scope.tool_allowlist == frozenset()

    def test_full_scope(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                {
                    "axis": "channel",
                    "identity": "cron",
                    "tool_allowlist": ["web_fetch", "memory_search"],
                    "tool_denylist": ["shell"],
                    "max_risk": "yellow",
                }
            ]
        )
        scope = reg.get(ScopeAxis.CHANNEL, "cron")
        assert scope is not None
        assert "web_fetch" in scope.tool_allowlist
        assert "shell" in scope.tool_denylist
        assert scope.max_risk == RiskLevel.YELLOW

    def test_dict_wrapper_with_scopes_key(self) -> None:
        reg = ScopeRegistry.from_config(
            {
                "scopes": [
                    {"axis": "user", "identity": "alex"},
                    {"axis": "user", "identity": "bob"},
                ]
            }
        )
        assert len(reg) == 2

    def test_skips_non_dict_entry(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                "not a scope",  # type: ignore[list-item]
                {"axis": "user", "identity": "alex"},
            ]
        )
        assert len(reg) == 1

    def test_skips_missing_required_fields(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                {"axis": "channel"},  # missing identity
                {"identity": "alex"},  # missing axis
                {"axis": "user", "identity": ""},  # empty identity
                {"axis": "channel", "identity": "telegram"},
            ]
        )
        assert len(reg) == 1
        assert reg.get(ScopeAxis.CHANNEL, "telegram") is not None

    def test_skips_invalid_axis(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                {"axis": "not_a_real_axis", "identity": "x"},
                {"axis": "user", "identity": "alex"},
            ]
        )
        assert len(reg) == 1

    def test_skips_invalid_max_risk(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                {
                    "axis": "user",
                    "identity": "alex",
                    "max_risk": "purple",
                }
            ]
        )
        assert len(reg) == 0

    def test_skips_non_list_tool_lists(self) -> None:
        reg = ScopeRegistry.from_config(
            [
                {
                    "axis": "user",
                    "identity": "alex",
                    "tool_allowlist": "not a list",
                }
            ]
        )
        assert len(reg) == 0

    def test_skips_overlapping_allow_and_denylist(self) -> None:
        # Construction-time validation prevents this — the registry
        # logs and skips the bad row.
        reg = ScopeRegistry.from_config(
            [
                {
                    "axis": "user",
                    "identity": "alex",
                    "tool_allowlist": ["x"],
                    "tool_denylist": ["x"],
                }
            ]
        )
        assert len(reg) == 0


class TestEndToEnd:
    def test_yaml_like_payload_round_trip(self) -> None:
        # Simulate the shape of a parsed config.security.scopes block.
        payload = [
            {
                "axis": "channel",
                "identity": "telegram",
                "tool_allowlist": ["web_search", "memory_search"],
                "max_risk": "yellow",
            },
            {
                "axis": "channel",
                "identity": "cron",
                "tool_denylist": ["shell", "exec_command"],
                "max_risk": "yellow",
            },
            {
                "axis": "user",
                "identity": "alex",
                "max_risk": "red",
            },
        ]
        reg = ScopeRegistry.from_config(payload)
        assert len(reg) == 3
        assert isinstance(reg.get(ScopeAxis.CHANNEL, "cron"), PermissionScope)

        # And the registry is fully evaluable.
        verdict = reg.evaluate([(ScopeAxis.CHANNEL, "cron")], "shell", RiskLevel.GREEN)
        assert verdict.denied
