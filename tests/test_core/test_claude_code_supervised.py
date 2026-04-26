"""Tests for the ClaudeCodeSupervisor autonomous loop driver.

Mocks ``asyncio.create_subprocess_exec`` to return a fake process whose
stdout yields a scripted NDJSON event stream. We do not actually spawn the
real ``claude`` CLI.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from cognithor.core.claude_code_supervised import (
    ClaudeCodeSupervisedBackend,
    ClaudeCodeSupervisor,
    GoalEvaluation,
    PlannerGoalEvaluator,
    SupervisorResult,
    TurnResult,
)
from cognithor.core.llm_backend import ChatResponse, LLMBackendError, LLMBackendType

if TYPE_CHECKING:
    from cognithor.models import ToolResult

# ─────────────────────────────────────────────────────────────────────────────
# Fake subprocess plumbing
# ─────────────────────────────────────────────────────────────────────────────


class _FakeStdout:
    """Async stream that yields scripted NDJSON lines, then EOF."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        for event in events:
            self._queue.put_nowait((json.dumps(event) + "\n").encode("utf-8"))
        self._queue.put_nowait(b"")  # EOF

    async def readline(self) -> bytes:
        line = await self._queue.get()
        return line


class _FakeStdin:
    def __init__(self) -> None:
        self.buffer: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        self.buffer.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True


class _FakeProc:
    def __init__(self, events: list[dict[str, Any]], *, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(events)
        self.stderr = _FakeStdout([])  # never produces output in tests
        self.returncode = returncode
        self._wait = asyncio.Event()
        self._wait.set()

    async def wait(self) -> int:
        await self._wait.wait()
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9
        self._wait.set()


def _script_events(
    *, session_id: str = "sess-xyz", text: str = "done", cost: float = 0.01
) -> list[dict[str, Any]]:
    return [
        {"type": "system", "subtype": "init", "session_id": session_id, "cwd": "/w"},
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "total_cost_usd": cost,
            "result": text,
            "is_error": False,
        },
    ]


def _script_events_with_tool(
    *, session_id: str = "sess-tool", text: str = "finished"
) -> list[dict[str, Any]]:
    return [
        {"type": "system", "subtype": "init", "session_id": session_id, "cwd": "/w"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "file1\nfile2"},
                ],
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "total_cost_usd": 0.02,
            "result": text,
            "is_error": False,
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleTurn:
    @pytest.mark.asyncio
    async def test_single_turn_success(self):
        proc = _FakeProc(_script_events(text="hello"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=AsyncMock(return_value=GoalEvaluation(verdict="done")),
            )
            result = await sup.run("do the thing")

        assert isinstance(result, SupervisorResult)
        assert result.verdict == "done"
        assert result.final_text == "hello"
        assert len(result.turns) == 1
        turn = result.turns[0]
        assert turn.session_id == "sess-xyz"
        assert turn.cost_usd == pytest.approx(0.01)
        assert turn.is_error is False
        # Stdin should have received one user frame.
        assert len(proc.stdin.buffer) == 1
        frame = json.loads(proc.stdin.buffer[0].decode())
        assert frame["type"] == "user"
        assert frame["message"]["content"] == "do the thing"

    @pytest.mark.asyncio
    async def test_tool_use_and_result_are_paired(self):
        proc = _FakeProc(_script_events_with_tool(text="finished"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=AsyncMock(return_value=GoalEvaluation(verdict="done")),
            )
            result = await sup.run("ls the dir")

        assert result.verdict == "done"
        turn = result.turns[0]
        assert len(turn.tool_results) == 1
        tr = turn.tool_results[0]
        assert tr.tool_name == "Bash"
        assert "file1" in tr.content
        assert tr.is_error is False


class TestBudgets:
    @pytest.mark.asyncio
    async def test_max_turns_enforced(self):
        proc_factory = AsyncMock(side_effect=lambda *a, **kw: _FakeProc(_script_events()))
        with patch("asyncio.create_subprocess_exec", proc_factory):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                max_turns=2,
                goal_evaluator=AsyncMock(
                    return_value=GoalEvaluation(verdict="continue", next_prompt="again")
                ),
            )
            result = await sup.run("never-ending")

        assert result.verdict == "abort"
        assert "max_turns" in result.reason
        assert len(result.turns) == 2

    @pytest.mark.asyncio
    async def test_max_cost_enforced(self):
        proc_factory = AsyncMock(side_effect=lambda *a, **kw: _FakeProc(_script_events(cost=10.0)))
        with patch("asyncio.create_subprocess_exec", proc_factory):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                max_turns=10,
                max_cost_usd=1.0,
                goal_evaluator=AsyncMock(
                    return_value=GoalEvaluation(verdict="continue", next_prompt="again")
                ),
            )
            result = await sup.run("pricey")

        assert result.verdict == "abort"
        assert "max_cost" in result.reason


class TestEvaluatorVerdicts:
    @pytest.mark.asyncio
    async def test_done_stops_immediately(self):
        proc = _FakeProc(_script_events(text="first"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            evaluator = AsyncMock(return_value=GoalEvaluation(verdict="done", reason="satisfied"))
            sup = ClaudeCodeSupervisor(claude_path="claude", goal_evaluator=evaluator, max_turns=5)
            result = await sup.run("query")

        assert result.verdict == "done"
        assert result.reason == "satisfied"
        assert len(result.turns) == 1
        evaluator.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_abort_stops_and_propagates_reason(self):
        proc = _FakeProc(_script_events())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=AsyncMock(
                    return_value=GoalEvaluation(verdict="abort", reason="policy failure")
                ),
            )
            result = await sup.run("q")

        assert result.verdict == "abort"
        assert result.reason == "policy failure"


class TestClaudeMissing:
    @pytest.mark.asyncio
    async def test_file_not_found_returns_error_turn(self):
        with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError())):
            sup = ClaudeCodeSupervisor(
                claude_path="claude-nope",
                goal_evaluator=AsyncMock(return_value=GoalEvaluation(verdict="done")),
            )
            result = await sup.run("q")

        assert result.verdict == "abort"
        assert len(result.turns) == 1
        assert result.turns[0].is_error is True
        assert "claude CLI not found" in (result.turns[0].error or "")


class TestParallelToolPairing:
    @pytest.mark.asyncio
    async def test_parallel_tool_use_paired_by_id(self):
        """Two tool_use blocks issued in parallel must each receive their
        own tool_result by id, not by ordering heuristic."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "s", "cwd": "/w"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_a",
                            "name": "Read",
                            "input": {"path": "a.txt"},
                        },
                        {
                            "type": "tool_use",
                            "id": "tu_b",
                            "name": "Read",
                            "input": {"path": "b.txt"},
                        },
                    ],
                },
            },
            # Results come back out-of-order on purpose: tu_b before tu_a.
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_b", "content": "B-content"},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_a", "content": "A-content"},
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "both read"}]},
            },
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.0,
                "result": "both read",
                "is_error": False,
            },
        ]
        proc = _FakeProc(events)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=AsyncMock(return_value=GoalEvaluation(verdict="done")),
            )
            result = await sup.run("read both")

        assert result.verdict == "done"
        results = result.turns[0].tool_results
        assert len(results) == 2
        # Order preserved from tool_use emission (tu_a first), but content
        # is paired by id, not by arrival order of tool_result blocks.
        assert results[0].tool_name == "Read" and results[0].content == "A-content"
        assert results[1].tool_name == "Read" and results[1].content == "B-content"

    @pytest.mark.asyncio
    async def test_tool_result_without_matching_id_appended_best_effort(self):
        events = [
            {"type": "system", "subtype": "init", "session_id": "s", "cwd": "/w"},
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "orphan", "content": "lost"},
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            },
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.0,
                "result": "ok",
                "is_error": False,
            },
        ]
        proc = _FakeProc(events)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=AsyncMock(return_value=GoalEvaluation(verdict="done")),
            )
            result = await sup.run("orphan")
        results = result.turns[0].tool_results
        assert len(results) == 1
        assert results[0].tool_name == "orphan"
        assert results[0].content == "lost"


# ─────────────────────────────────────────────────────────────────────────────
# ClaudeCodeSupervisedBackend (LLMBackend wrapper)
# ─────────────────────────────────────────────────────────────────────────────


class TestSupervisedBackend:
    def test_backend_type(self):
        b = ClaudeCodeSupervisedBackend()
        assert b.backend_type == LLMBackendType.CLAUDE_CODE_SUPERVISED

    @pytest.mark.asyncio
    async def test_list_models(self):
        b = ClaudeCodeSupervisedBackend()
        assert "sonnet" in await b.list_models()

    @pytest.mark.asyncio
    async def test_chat_runs_one_supervised_turn(self):
        proc = _FakeProc(_script_events(text="hi-from-supervisor", cost=0.005))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            b = ClaudeCodeSupervisedBackend(claude_path="claude")
            resp = await b.chat("sonnet", [{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "hi-from-supervisor"
        assert resp.model == "sonnet"
        assert resp.raw is not None
        assert resp.raw["verdict"] == "done"
        assert resp.raw["total_cost_usd"] == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_chat_flattens_system_and_assistant_messages(self):
        proc = _FakeProc(_script_events(text="ok"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            b = ClaudeCodeSupervisedBackend(claude_path="claude")
            await b.chat(
                "sonnet",
                [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yes?"},
                    {"role": "user", "content": "do x"},
                ],
            )
        # The flattened prompt should have appeared in the user-frame written to stdin.
        sent = b"".join(proc.stdin.buffer).decode()
        assert "[Context]: be terse" in sent
        assert "[Previous response]: yes?" in sent
        assert "do x" in sent

    @pytest.mark.asyncio
    async def test_chat_aborted_without_text_raises_backend_error(self):
        with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError())):
            b = ClaudeCodeSupervisedBackend(claude_path="missing")
            with pytest.raises(LLMBackendError):
                await b.chat("sonnet", [{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_embed_raises(self):
        b = ClaudeCodeSupervisedBackend()
        with pytest.raises(LLMBackendError):
            await b.embed("any", "text")


# ─────────────────────────────────────────────────────────────────────────────
# PlannerGoalEvaluator
# ─────────────────────────────────────────────────────────────────────────────


def _turn(
    *,
    turn: int = 1,
    text: str = "ok",
    tools: list[ToolResult] | None = None,
    cost: float = 0.01,
    is_error: bool = False,
    error: str | None = None,
) -> TurnResult:
    return TurnResult(
        turn=turn,
        prompt="(prompt)",
        assistant_text=text,
        tool_results=list(tools or []),
        cost_usd=cost,
        duration_ms=10,
        is_error=is_error,
        error=error,
        session_id="sess",
    )


class _FakeLLM:
    """Minimal LLM stub matching the .chat(model, messages, **kwargs) shape."""

    def __init__(self, content: str, *, raise_on_call: Exception | None = None) -> None:
        self.content = content
        self.calls: list[dict] = []
        self._raise = raise_on_call

    async def chat(self, model: str, messages: list[dict], **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if self._raise is not None:
            raise self._raise
        return ChatResponse(content=self.content, model=model)


class TestPlannerGoalEvaluator:
    @pytest.mark.asyncio
    async def test_done_verdict_parsed(self):
        llm = _FakeLLM(content='{"verdict": "done", "next_prompt": "", "reason": "goal met"}')
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("write hello.py", [_turn(text="DONE — wrote hello.py")])
        assert result.verdict == "done"
        assert result.reason == "goal met"
        assert llm.calls and llm.calls[0]["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_continue_verdict_carries_next_prompt(self):
        llm = _FakeLLM(
            content=(
                '{"verdict": "continue", "next_prompt": "Now add tests for it.", '
                '"reason": "code written but no tests"}'
            )
        )
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("write hello.py with tests", [_turn(text="wrote hello.py")])
        assert result.verdict == "continue"
        assert result.next_prompt == "Now add tests for it."

    @pytest.mark.asyncio
    async def test_abort_verdict_parsed(self):
        llm = _FakeLLM(content='{"verdict": "abort", "next_prompt": "", "reason": "stuck"}')
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev(
            "do impossible thing",
            [_turn(text="failed again", is_error=True, error="loop detected")],
        )
        assert result.verdict == "abort"
        assert result.reason == "stuck"

    @pytest.mark.asyncio
    async def test_json_with_code_fences_is_parsed(self):
        llm = _FakeLLM(
            content=('```json\n{"verdict": "done", "next_prompt": "", "reason": "ok"}\n```\n')
        )
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [_turn()])
        assert result.verdict == "done"

    @pytest.mark.asyncio
    async def test_json_with_prose_preamble_is_parsed(self):
        llm = _FakeLLM(
            content=(
                "Here is my verdict:\n"
                '{"verdict": "continue", "next_prompt": "do step 2", "reason": "step 1 done"}'
            )
        )
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [_turn()])
        assert result.verdict == "continue"
        assert result.next_prompt == "do step 2"

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_failure_default(self):
        llm = _FakeLLM(content="Not JSON at all, just words.")
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [_turn()])
        assert result.verdict == "continue"
        assert result.reason == "judge_unavailable"

    @pytest.mark.asyncio
    async def test_invalid_verdict_value_returns_failure_default(self):
        llm = _FakeLLM(content='{"verdict": "maybe", "next_prompt": "", "reason": "?"}')
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [_turn()])
        assert result.verdict == "continue"
        assert result.reason == "judge_unavailable"

    @pytest.mark.asyncio
    async def test_llm_raise_returns_failure_default(self):
        llm = _FakeLLM(content="", raise_on_call=RuntimeError("boom"))
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [_turn()])
        assert result.verdict == "continue"
        assert "judge could not produce" in result.next_prompt

    @pytest.mark.asyncio
    async def test_no_turns_yields_continue(self):
        llm = _FakeLLM(content='{"verdict": "done"}')
        ev = PlannerGoalEvaluator(llm=llm, model="sonnet")
        result = await ev("goal", [])
        assert result.verdict == "continue"
        # LLM must not be called when there's nothing to judge.
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_evaluator_plugged_into_supervisor(self):
        llm = _FakeLLM(content='{"verdict": "done", "next_prompt": "", "reason": "fits"}')
        evaluator = PlannerGoalEvaluator(llm=llm, model="sonnet")
        proc = _FakeProc(_script_events(text="wrote it"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            sup = ClaudeCodeSupervisor(
                claude_path="claude",
                goal_evaluator=evaluator,
            )
            result = await sup.run("write hello.py")
        assert result.verdict == "done"
        # The supervisor must have passed the user_intent through the new
        # two-arg evaluator signature.
        assert llm.calls, "evaluator should have been invoked"
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "write hello.py" in user_msg
