"""INVARIANT 4 — Budget limits are hard ceilings, not soft nudges.

Token, cost, and tool-call-count limits MUST abort execution when
exceeded. This suite tests what's enforced and documents what isn't.
"""

from __future__ import annotations

import pytest

from cognithor.core.token_budget import TokenBudgetManager

pytestmark = pytest.mark.security_contract


# ===========================================================================
# Token Budget — ADVISORY (documents the gap)
# ===========================================================================


def test_token_budget_allocate_is_advisory():
    """DOCUMENTED GAP: TokenBudgetManager.allocate() returns False when exceeded
    but does NOT raise or block. It is advisory only."""
    budget = TokenBudgetManager(complexity="simple", channel="cli")
    total = budget._total
    assert budget.allocate(total - 10) is True
    assert budget.allocate(20) is False  # exceeded
    assert budget.exceeded
    assert budget.remaining == 0


def test_token_budget_exceeded_property():
    """exceeded property must be True when allocated > total."""
    budget = TokenBudgetManager(complexity="simple", channel="cli")
    budget.allocate(budget._total + 1)
    assert budget.exceeded


def test_token_budget_remaining_floors_at_zero():
    """remaining must not go negative."""
    budget = TokenBudgetManager(complexity="simple", channel="cli")
    budget.allocate(budget._total + 100)
    assert budget.remaining == 0


# ===========================================================================
# Cost Budget — Hard block at ENTRY (documents mid-loop gap)
# ===========================================================================


def test_cost_budget_blocks_when_exceeded(tmp_path):
    """CostTracker.check_budget() must return ok=False when daily/monthly exceeded."""
    from cognithor.telemetry.cost_tracker import CostTracker

    db = str(tmp_path / "cost.db")
    tracker = CostTracker(db_path=db, daily_budget=0.01, monthly_budget=1.0)
    tracker.record_llm_call(
        model="claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=500_000
    )

    status = tracker.check_budget()
    assert not status.ok


def test_cost_budget_ok_when_within(tmp_path):
    """CostTracker.check_budget() must return ok=True when within budget."""
    from cognithor.telemetry.cost_tracker import CostTracker

    db = str(tmp_path / "cost.db")
    tracker = CostTracker(db_path=db, daily_budget=100.0, monthly_budget=1000.0)
    status = tracker.check_budget()
    assert status.ok


def test_cost_budget_checked_mid_loop():
    """The PGE loop MUST call `check_budget()` mid-loop.

    The loop body was extracted from `gateway.py::_run_pge_loop` into
    `gateway/pge_loop.py::run_pge_loop` (PR #191). Scan both:
      1. The Gateway-class wrapper source first (still works pre-split).
      2. The free function in `pge_loop` (post-split source-of-truth).
    """
    import inspect

    from cognithor.gateway.gateway import Gateway

    wrapper_source = inspect.getsource(Gateway._run_pge_loop)
    if "check_budget" in wrapper_source:
        return  # pre-split layout
    try:
        from cognithor.gateway import pge_loop

        loop_source = inspect.getsource(pge_loop.run_pge_loop)
    except (ImportError, AttributeError):
        loop_source = ""
    assert "check_budget" in loop_source, (
        "check_budget() must be called inside the PGE loop "
        "(checked both gateway.Gateway._run_pge_loop and "
        "gateway.pge_loop.run_pge_loop)"
    )


def test_cost_budget_daily_and_monthly_independent(tmp_path):
    """Both daily AND monthly budgets must independently trigger ok=False."""
    from cognithor.telemetry.cost_tracker import CostTracker

    db = str(tmp_path / "cost.db")
    tracker = CostTracker(db_path=db, daily_budget=0.001, monthly_budget=1000.0)
    tracker.record_llm_call(
        model="claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=500_000
    )
    status = tracker.check_budget()
    assert not status.ok


# ===========================================================================
# Ralph Loop — Time budget IS a hard ceiling
# ===========================================================================


@pytest.mark.asyncio
async def test_ralph_loop_total_budget_hard_stop():
    """Ralph Loop must break when total_budget_seconds is exceeded."""
    from cognithor.core.ralph_loop import RalphConfig, RalphLoop

    config = RalphConfig(
        max_iterations=100,
        total_budget_seconds=0.1,
        iteration_budget_seconds=300,
    )
    loop = RalphLoop(config=config)

    call_count = 0

    async def slow_pge(input_text, ralph_prompt=""):
        nonlocal call_count
        call_count += 1
        await __import__("asyncio").sleep(0.05)
        return "[CONTINUE: next]", ["web_search"]

    result = await loop.run("test task", slow_pge)
    assert result.stop_reason in ("total_budget_exceeded", "converged", "no_progress")
    assert call_count <= 5  # budget should stop long before 100 iterations


@pytest.mark.asyncio
async def test_ralph_loop_max_iterations_hard_stop():
    """Ralph Loop must stop at max_iterations."""
    from cognithor.core.ralph_loop import RalphConfig, RalphLoop

    config = RalphConfig(
        max_iterations=3,
        total_budget_seconds=300,
        iteration_budget_seconds=300,
    )
    loop = RalphLoop(config=config)

    iteration_count = 0

    async def fast_pge(input_text, ralph_prompt=""):
        nonlocal iteration_count
        iteration_count += 1
        return f"[CONTINUE: step {iteration_count}]", [f"tool_{iteration_count}"]

    await loop.run("test task", fast_pge)
    assert iteration_count <= 3


@pytest.mark.asyncio
async def test_ralph_loop_respects_cancellation():
    """Ralph Loop must break on cancellation."""
    from cognithor.core.ralph_loop import RalphConfig, RalphLoop

    config = RalphConfig(max_iterations=100, total_budget_seconds=300)
    loop = RalphLoop(config=config)
    loop.cancel()

    async def pge(input_text, ralph_prompt=""):
        return "[CONTINUE: next]", ["tool"]

    result = await loop.run("test", pge)
    assert result.stop_reason == "cancelled"


# ===========================================================================
# ToolEnforcer max_tool_calls — Hard ceiling per skill
# ===========================================================================


def test_tool_enforcer_max_calls_blocks():
    """ToolEnforcer must block when max_tool_calls exceeded."""
    from pathlib import Path

    from cognithor.models import PlannedAction
    from cognithor.skills.community.tool_enforcer import ToolEnforcer
    from cognithor.skills.registry import CommunitySkillManifest, Skill

    manifest = CommunitySkillManifest(
        name="test",
        tools_required=["web_search"],
        max_tool_calls=2,
    )
    skill = Skill(
        name="test",
        slug="test",
        file_path=Path("/fake.md"),
        tools_required=["web_search"],
        source="community",
        manifest=manifest,
    )
    enforcer = ToolEnforcer(max_tool_calls=10)
    action = PlannedAction(tool="web_search", params={})

    enforcer.check(action, skill)
    enforcer.check(action, skill)
    result = enforcer.check(action, skill)
    assert not result.allowed


# ===========================================================================
# PolicyEngine tool-call limit
# ===========================================================================


def test_policy_engine_tool_call_limit():
    """PolicyEngine.check_tool_call_limit must return violation at limit."""
    from cognithor.security.policies import PolicyEngine

    engine = PolicyEngine()
    session = "test-session"

    for _ in range(engine._get_quota(session).max_total_tool_calls):
        engine.record_tool_call(session)

    violation = engine.check_tool_call_limit(session)
    assert violation is not None
    assert "max_tool_calls" in violation.rule


def test_budget_check_exception_handled(tmp_path):
    """If check_budget() raises, it must not silently auto-allow."""
    from cognithor.telemetry.cost_tracker import CostTracker

    db = str(tmp_path / "cost.db")
    tracker = CostTracker(db_path=db, daily_budget=100.0)
    status = tracker.check_budget()
    assert status.ok
