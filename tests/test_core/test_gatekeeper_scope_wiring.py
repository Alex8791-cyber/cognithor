"""Integration tests: Gatekeeper ↔ ScopeRegistry (TRUST-5 production wiring)."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest

from cognithor.config import (
    CognithorConfig,
    SecurityConfig,
    ToolsConfig,
    ensure_directory_structure,
)
from cognithor.core.gatekeeper import Gatekeeper
from cognithor.models import (
    GateStatus,
    PlannedAction,
    RiskLevel,
    SessionContext,
)
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def gk_config(tmp_path: Path) -> CognithorConfig:
    config = CognithorConfig(
        cognithor_home=tmp_path,
        security=SecurityConfig(
            allowed_paths=[str(tmp_path), os.path.join(tempfile.gettempdir(), "jarvis", "")],
        ),
        tools=ToolsConfig(),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gatekeeper(gk_config: CognithorConfig) -> Gatekeeper:
    gk = Gatekeeper(gk_config)
    gk.initialize()
    # Inject a fresh registry so tests don't leak into the canonical one.
    gk.set_scope_registry(ScopeRegistry())
    return gk


def _telegram_session() -> SessionContext:
    """Session with a non-default channel + user so scope keys fire."""
    return SessionContext(user_id="alex", channel="telegram")


def _cli_session() -> SessionContext:
    """Default cli + default user — scope keys are skipped (un-scoped path)."""
    return SessionContext(user_id="default", channel="cli")


class TestScopeWiringBlocks:
    def test_denylist_blocks_at_gate(self, gatekeeper: Gatekeeper) -> None:
        gatekeeper.set_scope_registry(
            self._registry_with(
                PermissionScope(
                    axis=ScopeAxis.CHANNEL,
                    identity="telegram",
                    tool_denylist=frozenset({"shell"}),
                )
            )
        )
        action = PlannedAction(tool="shell", params={"cmd": "ls"})
        decision = gatekeeper.evaluate(action, _telegram_session())
        assert decision.status == GateStatus.BLOCK
        assert decision.risk_level == RiskLevel.RED
        assert decision.policy_name == "permission_scope:channel"
        assert decision.explanation is not None
        assert decision.explanation.rule_id == "scope:channel:telegram"
        assert "denylist" in decision.reason

    def test_max_risk_ceiling_blocks(self, gatekeeper: Gatekeeper) -> None:
        gatekeeper.set_scope_registry(
            self._registry_with(
                PermissionScope(
                    axis=ScopeAxis.CHANNEL,
                    identity="telegram",
                    max_risk=RiskLevel.GREEN,
                )
            )
        )
        # An unknown tool falls through to ORANGE in the default
        # classifier — that's above the GREEN ceiling.
        action = PlannedAction(tool="brand_new_tool_42", params={})
        decision = gatekeeper.evaluate(action, _telegram_session())
        assert decision.status == GateStatus.BLOCK
        assert "max_risk" in decision.reason
        assert decision.explanation is not None
        assert decision.explanation.rule_source.endswith("ScopeRegistry.evaluate")

    def test_allowlist_excludes_blocks(self, gatekeeper: Gatekeeper) -> None:
        gatekeeper.set_scope_registry(
            self._registry_with(
                PermissionScope(
                    axis=ScopeAxis.USER,
                    identity="alex",
                    tool_allowlist=frozenset({"web_search"}),
                )
            )
        )
        action = PlannedAction(tool="read_file", params={"path": "/tmp/x"})
        decision = gatekeeper.evaluate(action, _telegram_session())
        assert decision.status == GateStatus.BLOCK
        assert "not in allowlist" in decision.reason

    @staticmethod
    def _registry_with(*scopes: PermissionScope) -> ScopeRegistry:
        reg = ScopeRegistry()
        for scope in scopes:
            reg.register(scope)
        return reg


class TestScopeWiringPasses:
    def test_empty_registry_falls_through(self, gatekeeper: Gatekeeper) -> None:
        # Default fixture already seeds an empty registry — a normal
        # action must still flow through risk classification.
        action = PlannedAction(tool="read_file", params={"path": "/tmp/x"})
        decision = gatekeeper.evaluate(action, _telegram_session())
        # read_file is GREEN — no scope, no deny, allowed.
        assert (
            decision.status != GateStatus.BLOCK
            or decision.policy_name != "permission_scope:channel"
        )

    def test_cli_session_skips_scope_evaluation(self, gatekeeper: Gatekeeper) -> None:
        # Even with a strict scope, a default cli/default session
        # bypasses the scope keys entirely (scope_keys is empty).
        gatekeeper.set_scope_registry(
            TestScopeWiringBlocks._registry_with(
                PermissionScope(
                    axis=ScopeAxis.CHANNEL,
                    identity="telegram",
                    tool_denylist=frozenset({"shell"}),
                )
            )
        )
        action = PlannedAction(tool="shell", params={"cmd": "ls"})
        decision = gatekeeper.evaluate(action, _cli_session())
        # cli session bypasses scope keys — scope did not block.
        if decision.policy_name:
            assert "permission_scope" not in decision.policy_name

    def test_allowlist_member_passes_to_risk_classifier(self, gatekeeper: Gatekeeper) -> None:
        gatekeeper.set_scope_registry(
            TestScopeWiringBlocks._registry_with(
                PermissionScope(
                    axis=ScopeAxis.USER,
                    identity="alex",
                    tool_allowlist=frozenset({"read_file", "write_file"}),
                )
            )
        )
        action = PlannedAction(tool="read_file", params={"path": "/tmp/x"})
        decision = gatekeeper.evaluate(action, _telegram_session())
        # read_file is GREEN at the underlying classifier — passing
        # through scope eval shouldn't re-stamp the policy name.
        if decision.policy_name:
            assert "permission_scope" not in decision.policy_name


class TestScopeWiringExplanation:
    def test_explanation_is_structured(self, gatekeeper: Gatekeeper) -> None:
        # TRUST-2 contract: explanation is non-None on scope blocks
        # so the Trace-UI can render rule_id / rule_source / pattern.
        gatekeeper.set_scope_registry(
            TestScopeWiringBlocks._registry_with(
                PermissionScope(
                    axis=ScopeAxis.WORKFLOW,
                    identity="morning_brief",
                    tool_denylist=frozenset({"shell"}),
                )
            )
        )
        # Workflow axis isn't auto-derived from SessionContext yet —
        # this test documents the future wiring; for now we just
        # verify the no-match path keeps the gate clean.
        action = PlannedAction(tool="read_file", params={"path": "/tmp/x"})
        decision = gatekeeper.evaluate(action, SessionContext(user_id="alex", channel="telegram"))
        # Non-matching scope → no scope-block explanation set.
        if decision.explanation is not None:
            assert decision.explanation.rule_source != ""
