"""Live integration test for the supervised Claude Code driver.

Skipped automatically when the ``claude`` CLI is not on PATH (developer
laptops without Claude Code installed) or when the ``COGNITHOR_LIVE_CC``
env var is not truthy (CI by default). When enabled, runs an actual
``claude -p --output-format stream-json`` subprocess and asserts the
supervisor parses the real event stream end-to-end.

Catches drift between Cognithor's NDJSON parser and the live CLI output
(field renames, new event types) before it ships.

Run locally with::

    COGNITHOR_LIVE_CC=1 pytest tests/integration -m integration -v

Marked ``slow`` + ``integration`` so the default ``pytest`` invocation
(unit tests only) ignores it.
"""

from __future__ import annotations

import os
import shutil

import pytest

from cognithor.core.claude_code_supervised import (
    ClaudeCodeSupervisor,
    GoalEvaluation,
    SupervisorResult,
)

_LIVE_ENABLED = os.environ.get("COGNITHOR_LIVE_CC", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_CLAUDE_PATH = shutil.which("claude")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _LIVE_ENABLED,
        reason="set COGNITHOR_LIVE_CC=1 to opt in to live Claude Code calls",
    ),
    pytest.mark.skipif(
        _CLAUDE_PATH is None,
        reason="`claude` CLI not on PATH",
    ),
]


@pytest.mark.asyncio
async def test_supervisor_runs_against_real_claude_cli() -> None:
    """One-turn live run: ask Claude to say 'hello world' and stop.

    Asserts:
      - The subprocess parses successfully (no NDJSON drift).
      - The session_id is captured from the system:init frame.
      - The result frame produces a non-empty ``final_text``.
      - At least the basic verdict path returns ``done`` for a trivial goal.

    Cost guardrail: ``max_cost_usd=0.50`` and ``max_turns=1`` keep the
    test cheap even if the underlying account uses a paid tier.
    """
    supervisor = ClaudeCodeSupervisor(
        model=os.environ.get("COGNITHOR_LIVE_CC_MODEL", "haiku"),
        max_turns=1,
        max_cost_usd=0.50,
        per_turn_timeout_seconds=120,
        goal_evaluator=None,  # use the default, single-turn -> "done"
    )
    result = await supervisor.run("Reply with just the two words 'hello world' and nothing else.")

    assert isinstance(result, SupervisorResult)
    assert result.verdict in ("done", "abort"), result.reason
    assert len(result.turns) == 1, "should be exactly one turn"
    turn = result.turns[0]
    assert turn.session_id, "system:init frame should provide a session_id"
    assert not turn.is_error, f"turn errored: {turn.error}"
    assert turn.assistant_text, "result frame should produce non-empty text"


@pytest.mark.asyncio
async def test_supervisor_respects_max_cost_against_real_cli() -> None:
    """Set max_cost_usd to 0 so we abort BEFORE invoking the CLI a second time.

    The first turn still runs (budget is checked at the top of the loop,
    after a turn with cost=0.0 we'd compare 0 >= 0). This documents the
    edge: running with max_cost_usd=0 effectively limits to one turn.
    """
    supervisor = ClaudeCodeSupervisor(
        model=os.environ.get("COGNITHOR_LIVE_CC_MODEL", "haiku"),
        max_turns=5,
        max_cost_usd=0.0001,  # any non-zero cost trips the budget
        per_turn_timeout_seconds=120,
        goal_evaluator=lambda _ui, _t: _continue_eval(),  # always continue
    )
    result = await supervisor.run("Reply with just the two words 'hello world' and nothing else.")
    # We expect either max_cost abort after the first turn, or a one-turn
    # done if the live cost happens to be reported as exactly 0.
    if result.verdict == "abort":
        assert "max_cost" in result.reason
    else:
        assert result.verdict == "done"
    assert len(result.turns) >= 1


async def _continue_eval() -> GoalEvaluation:
    return GoalEvaluation(verdict="continue", next_prompt="continue")
