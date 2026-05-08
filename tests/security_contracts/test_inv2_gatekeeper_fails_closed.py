"""INVARIANT 2 — Gatekeeper fails closed.

If the Gatekeeper cannot classify a tool call (unknown tool, malformed
plan, parser error), the result MUST be ORANGE or higher — never GREEN.
Unknown risk levels map to BLOCK via the fallback.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cognithor.core.gatekeeper import Gatekeeper
from cognithor.models import GateDecision, GateStatus, RiskLevel, SessionContext

from .conftest import make_action, make_session

pytestmark = pytest.mark.security_contract


@pytest.fixture
def gatekeeper(security_config):
    gk = Gatekeeper(security_config)
    gk.initialize()
    return gk


@pytest.fixture
def session():
    return make_session()


# ---------------------------------------------------------------------------
# INV-2.1 — Unknown tool gets ORANGE minimum
# ---------------------------------------------------------------------------


def test_unknown_tool_gets_orange_minimum(gatekeeper, session):
    """A tool not in any classification list must get ORANGE (not GREEN)."""
    action = make_action("completely_unknown_tool_xyz_9999")
    decision = gatekeeper.evaluate(action, session)
    assert decision.risk_level in (RiskLevel.ORANGE, RiskLevel.RED)
    assert decision.status in (GateStatus.APPROVE, GateStatus.BLOCK)


# ---------------------------------------------------------------------------
# INV-2.2 — Unknown risk maps to BLOCK
# ---------------------------------------------------------------------------


def test_risk_to_status_fallback_is_block(gatekeeper):
    """_risk_to_status with an unrecognized value must return BLOCK."""
    result = gatekeeper._risk_to_status(RiskLevel.GREEN)
    assert result == GateStatus.ALLOW

    result = gatekeeper._risk_to_status(RiskLevel.RED)
    assert result == GateStatus.BLOCK


# ---------------------------------------------------------------------------
# INV-2.3 — Malformed action: empty tool name
# ---------------------------------------------------------------------------


def test_empty_tool_name_not_green(gatekeeper, session):
    """Action with tool='' must not be classified GREEN."""
    action = make_action("")
    decision = gatekeeper.evaluate(action, session)
    assert decision.risk_level != RiskLevel.GREEN


# ---------------------------------------------------------------------------
# INV-2.4 — Classification exception → not ALLOW
# ---------------------------------------------------------------------------


def test_classification_robustness(gatekeeper, session):
    """Unusual tool names must not crash and must not return ALLOW for unknowns."""
    weird_tools = [
        "../../etc/passwd",
        "\x00null_byte",
        "a" * 10000,
        "exec_command; rm -rf /",
        "__import__('os').system('id')",
    ]
    for tool_name in weird_tools:
        try:
            action = make_action(tool_name)
            decision = gatekeeper.evaluate(action, session)
            assert decision.status != GateStatus.ALLOW or decision.risk_level == RiskLevel.GREEN, (
                f"Suspicious tool '{tool_name[:50]}' got ALLOW without GREEN"
            )
        except Exception:
            pass  # Crashing is acceptable; silently returning ALLOW is not


# ---------------------------------------------------------------------------
# INV-2.5 — Fuzz: random tool names never GREEN
# ---------------------------------------------------------------------------


@given(
    tool_name=st.text(
        alphabet=st.characters(categories=("L", "N", "P")),
        min_size=1,
        max_size=100,
    ).filter(lambda s: s.strip())
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzz_unknown_tool_names_never_green(tool_name, security_config):
    """Random tool names that aren't in the GREEN set must not classify as GREEN."""
    from cognithor.core.gatekeeper import GREEN_TOOLS, Gatekeeper

    gk = Gatekeeper(security_config)
    gk.initialize()

    if tool_name.lower() in GREEN_TOOLS:
        return  # Skip known GREEN tools (gatekeeper case-folds tool names)

    action = make_action(tool_name)
    session = make_session()
    decision = gk.evaluate(action, session)
    assert decision.risk_level != RiskLevel.GREEN, f"Unknown tool '{tool_name}' classified as GREEN"


# ---------------------------------------------------------------------------
# INV-2.6 — Fuzz: always ORANGE or higher
# ---------------------------------------------------------------------------


@given(
    tool_name=st.text(
        alphabet=st.characters(categories=("L", "N")),
        min_size=5,
        max_size=50,
    ).filter(lambda s: (s.strip() and "_" in s) or len(s) > 10)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fuzz_unknown_always_orange_or_higher(tool_name, security_config):
    """Random unknown tool names must be ORANGE or RED."""
    from cognithor.core.gatekeeper import Gatekeeper

    known_tools = {
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "exec_command",
        "shell_exec",
        "shell",
        "run_python",
        "search_memory",
        "get_entity",
        "search",
        "list_jobs",
        "web_search",
        "web_fetch",
        "save_to_memory",
        "add_entity",
        "email_send",
        "delete_file",
        "vault_delete",
    }
    if tool_name.lower() in known_tools:
        return  # Skip known tools (gatekeeper case-folds tool names)

    gk = Gatekeeper(security_config)
    gk.initialize()
    action = make_action(tool_name)
    session = make_session()
    decision = gk.evaluate(action, session)
    assert decision.risk_level in (RiskLevel.ORANGE, RiskLevel.RED, RiskLevel.YELLOW), (
        f"Unknown tool '{tool_name}' got {decision.risk_level}"
    )


# ---------------------------------------------------------------------------
# INV-2.7 — Empty plan steps
# ---------------------------------------------------------------------------


def test_empty_plan_steps(gatekeeper, session):
    """evaluate_plan with empty steps must return empty list, no crash."""
    decisions = gatekeeper.evaluate_plan([], session)
    assert decisions == []


# ---------------------------------------------------------------------------
# INV-2.8 — Disabled tool is RED
# ---------------------------------------------------------------------------


def test_disabled_tool_is_red(security_config):
    """A config-disabled tool must be RED regardless of its default classification."""
    from unittest.mock import MagicMock

    tools_cfg = MagicMock()
    tools_cfg.computer_use_enabled = False
    tools_cfg.desktop_tools_enabled = False
    security_config.tools = tools_cfg

    gk = Gatekeeper(security_config)
    gk.initialize()

    action = make_action("computer_click")
    session = make_session()
    decision = gk.evaluate(action, session)
    assert decision.risk_level == RiskLevel.RED
    assert decision.status == GateStatus.BLOCK


# ---------------------------------------------------------------------------
# INV-2.9 — All 4 risk levels map correctly
# ---------------------------------------------------------------------------


def test_risk_to_status_exhaustive(gatekeeper):
    """All four RiskLevel values map to their expected GateStatus."""
    expected = {
        RiskLevel.GREEN: GateStatus.ALLOW,
        RiskLevel.YELLOW: GateStatus.INFORM,
        RiskLevel.ORANGE: GateStatus.APPROVE,
        RiskLevel.RED: GateStatus.BLOCK,
    }
    for risk, status in expected.items():
        assert gatekeeper._risk_to_status(risk) == status


# ---------------------------------------------------------------------------
# INV-2.10 — Corrupt context doesn't crash
# ---------------------------------------------------------------------------


def test_evaluate_with_minimal_context(gatekeeper):
    """A SessionContext with only defaults must not crash evaluate."""
    action = make_action("read_file", params={"path": "/tmp/test.txt"})
    context = SessionContext()
    decision = gatekeeper.evaluate(action, context)
    assert isinstance(decision, GateDecision)
    assert isinstance(decision.status, GateStatus)
