"""Deep coverage for cognithor.gateway.post_processing.

The module is the post-PGE pipeline: reflection, skill-tracking,
telemetry, profiler, run-recording, reflexion, evolution, session
analysis, pattern documentation, and key-tool persistence.

The functions take the live ``Gateway`` instance and read internal
state through it, so most tests instantiate ``Gateway.__new__(Gateway)``
and inject only the attributes the function under test reads.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.gateway import post_processing
from cognithor.gateway.gateway import Gateway
from cognithor.gateway.post_processing import (
    maybe_record_pattern,
    persist_key_tool_results,
    persist_session,
    run_post_processing,
)
from cognithor.models import (
    AgentResult,
    Message,
    MessageRole,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bare_gateway() -> Gateway:
    """A Gateway shell without running __init__.

    Sets only the attributes that post_processing functions touch
    when they probe via ``hasattr(gw, "_x")`` / ``getattr(gw, "_x", None)``.
    """
    gw = Gateway.__new__(Gateway)
    # Defaults used everywhere — None means "feature off"
    gw._reflector = None  # type: ignore[attr-defined]
    gw._memory_manager = None  # type: ignore[attr-defined]
    gw._skill_registry = None  # type: ignore[attr-defined]
    gw._skill_generator = None  # type: ignore[attr-defined]
    gw._task_telemetry = None  # type: ignore[attr-defined]
    gw._task_profiler = None  # type: ignore[attr-defined]
    gw._run_recorder = None  # type: ignore[attr-defined]
    gw._strategy_memory = None  # type: ignore[attr-defined]
    gw._reflexion_memory = None  # type: ignore[attr-defined]
    gw._session_analyzer = None  # type: ignore[attr-defined]
    gw._evolution_orchestrator = None  # type: ignore[attr-defined]
    gw._prompt_evolution = None  # type: ignore[attr-defined]
    gw._trace_store = None  # type: ignore[attr-defined]
    gw._planner = None  # type: ignore[attr-defined]
    gw._session_store = None  # type: ignore[attr-defined]
    gw._deep_learner = None  # type: ignore[attr-defined]
    gw._evolution_loop = None  # type: ignore[attr-defined]
    gw._config = None  # type: ignore[attr-defined]
    gw._pattern_record_timestamps = []  # type: ignore[attr-defined]
    return gw


def _session(channel: str = "cli") -> SessionContext:
    return SessionContext(session_id="sess-12345678", channel=channel)


def _wm_with_user(text: str) -> WorkingMemory:
    wm = WorkingMemory()
    wm.add_message(Message(role=MessageRole.USER, content=text))
    return wm


def _ok_tool(tool_name: str = "web_search", content: str = "out") -> ToolResult:
    return ToolResult(tool_name=tool_name, content=content, is_error=False)


def _err_tool(tool_name: str = "read_file", error_type: str = "FileNotFound") -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        content="boom",
        is_error=True,
        error_type=error_type,
        error_message="boom-msg",
    )


def _agent_result(
    *,
    success: bool = True,
    tool_results: list[ToolResult] | None = None,
    response: str = "Done",
) -> AgentResult:
    return AgentResult(
        response=response,
        success=success,
        tool_results=tool_results or [],
        total_duration_ms=100,
        model_used="qwen3:32b",
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_post_processing — branch coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPostProcessing:
    @pytest.mark.asyncio
    async def test_no_subsystems_is_noop(self) -> None:
        gw = _bare_gateway()
        await run_post_processing(
            gw,
            _session(),
            _wm_with_user("hello"),
            _agent_result(),
            None,
            None,
        )

    @pytest.mark.asyncio
    async def test_reflector_runs_when_should_reflect_true(self) -> None:
        gw = _bare_gateway()
        reflection = MagicMock()
        reflection.success_score = 0.9
        gw._reflector = MagicMock()  # type: ignore[attr-defined]
        gw._reflector.should_reflect.return_value = True
        gw._reflector.reflect = AsyncMock(return_value=reflection)
        result = _agent_result()
        await run_post_processing(gw, _session(), _wm_with_user("q"), result, None, None)
        gw._reflector.reflect.assert_awaited_once()
        assert result.reflection is reflection

    @pytest.mark.asyncio
    async def test_reflector_skipped_when_should_reflect_false(self) -> None:
        gw = _bare_gateway()
        gw._reflector = MagicMock()  # type: ignore[attr-defined]
        gw._reflector.should_reflect.return_value = False
        gw._reflector.reflect = AsyncMock()
        await run_post_processing(gw, _session(), _wm_with_user("q"), _agent_result(), None, None)
        gw._reflector.reflect.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflector_exception_is_swallowed(self) -> None:
        gw = _bare_gateway()
        gw._reflector = MagicMock()  # type: ignore[attr-defined]
        gw._reflector.should_reflect.return_value = True
        gw._reflector.reflect = AsyncMock(side_effect=RuntimeError("boom"))
        # MUST NOT raise — reflection failure can never break post-processing
        await run_post_processing(gw, _session(), _wm_with_user("q"), _agent_result(), None, None)

    @pytest.mark.asyncio
    async def test_low_score_triggers_evolution_gap_injection(self) -> None:
        gw = _bare_gateway()
        reflection = MagicMock()
        reflection.success_score = 0.1  # below 0.5 threshold
        gw._reflector = MagicMock()  # type: ignore[attr-defined]
        gw._reflector.should_reflect.return_value = True
        gw._reflector.reflect = AsyncMock(return_value=reflection)
        gw._reflector.apply = AsyncMock(return_value={"episodic": 1})
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        gw._deep_learner = MagicMock()  # type: ignore[attr-defined]
        gw._evolution_loop = MagicMock()  # type: ignore[attr-defined]
        cfg = MagicMock()
        cfg.evolution.learning_goals = []
        gw._config = cfg  # type: ignore[attr-defined]

        wm = _wm_with_user("how do I X")
        await run_post_processing(gw, _session(), wm, _agent_result(), None, None)
        # A gap goal was appended
        assert len(cfg.evolution.learning_goals) == 1
        assert "Schwache Antwort" in cfg.evolution.learning_goals[0]

    @pytest.mark.asyncio
    async def test_strategy_memory_records_when_tools_used(self) -> None:
        gw = _bare_gateway()
        gw._strategy_memory = MagicMock()  # type: ignore[attr-defined]
        result = _agent_result(tool_results=[_ok_tool("web_search"), _ok_tool("web_fetch")])
        await run_post_processing(gw, _session(), _wm_with_user("q"), result, None, None)
        gw._strategy_memory.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_memory_skipped_when_no_tools(self) -> None:
        gw = _bare_gateway()
        gw._strategy_memory = MagicMock()  # type: ignore[attr-defined]
        await run_post_processing(gw, _session(), _wm_with_user("q"), _agent_result(), None, None)
        gw._strategy_memory.record.assert_not_called()

    @pytest.mark.asyncio
    async def test_skill_registry_records_usage_for_active_skill(self) -> None:
        gw = _bare_gateway()
        gw._skill_registry = MagicMock()  # type: ignore[attr-defined]
        active_skill = MagicMock()
        active_skill.skill.slug = "my-skill"
        active_skill.procedure_name = "proc-1"
        await run_post_processing(
            gw, _session(), _wm_with_user("q"), _agent_result(), active_skill, None
        )
        gw._skill_registry.record_usage.assert_called_once()
        call_kwargs = gw._skill_registry.record_usage.call_args.kwargs
        assert call_kwargs["success"] is True
        # No reflection → falls back to 0.8 for success
        assert call_kwargs["score"] == 0.8

    @pytest.mark.asyncio
    async def test_task_telemetry_extracts_first_error(self) -> None:
        gw = _bare_gateway()
        gw._task_telemetry = MagicMock()  # type: ignore[attr-defined]
        result = _agent_result(
            success=False,
            tool_results=[_ok_tool("a"), _err_tool("b", "TimeoutError")],
        )
        await run_post_processing(gw, _session(), _wm_with_user("q"), result, None, None)
        kwargs = gw._task_telemetry.record_task.call_args.kwargs
        assert kwargs["error_type"] == "TimeoutError"
        assert kwargs["success"] is False
        assert kwargs["tool_calls"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_task_profiler_finishes_with_score(self) -> None:
        gw = _bare_gateway()
        gw._task_profiler = MagicMock()  # type: ignore[attr-defined]
        await run_post_processing(
            gw, _session(), _wm_with_user("q"), _agent_result(success=False), None, None
        )
        kwargs = gw._task_profiler.finish_task.call_args.kwargs
        assert kwargs["session_id"] == "sess-12345678"
        # No reflection + failure → 0.3
        assert kwargs["success_score"] == 0.3

    @pytest.mark.asyncio
    async def test_run_recorder_only_called_when_run_id_present(self) -> None:
        gw = _bare_gateway()
        gw._run_recorder = MagicMock()  # type: ignore[attr-defined]
        await run_post_processing(gw, _session(), _wm_with_user("q"), _agent_result(), None, None)
        gw._run_recorder.finish_run.assert_not_called()
        await run_post_processing(
            gw, _session(), _wm_with_user("q"), _agent_result(), None, "RUN-1"
        )
        gw._run_recorder.finish_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_truncated_to_500_chars_in_run_recorder(self) -> None:
        gw = _bare_gateway()
        gw._run_recorder = MagicMock()  # type: ignore[attr-defined]
        big = "x" * 1000
        await run_post_processing(
            gw,
            _session(),
            _wm_with_user("q"),
            _agent_result(response=big),
            None,
            "RUN-1",
        )
        kwargs = gw._run_recorder.finish_run.call_args.kwargs
        assert len(kwargs["final_response"]) == 500

    @pytest.mark.asyncio
    async def test_reflexion_records_unknown_error(self) -> None:
        gw = _bare_gateway()
        rm = MagicMock()
        rm.get_solution.return_value = None
        gw._reflexion_memory = rm  # type: ignore[attr-defined]
        result = _agent_result(success=False, tool_results=[_err_tool("read_file")])
        await run_post_processing(
            gw, _session(), _wm_with_user("read /etc/passwd"), result, None, None
        )
        rm.record_error.assert_called_once()
        assert rm.get_solution.called

    @pytest.mark.asyncio
    async def test_reflexion_logs_known_solution_without_recording(self) -> None:
        gw = _bare_gateway()
        rm = MagicMock()
        rm.get_solution.return_value = MagicMock(prevention_rule="check-existence")
        gw._reflexion_memory = rm  # type: ignore[attr-defined]
        result = _agent_result(success=False, tool_results=[_err_tool("read_file")])
        await run_post_processing(gw, _session(), _wm_with_user("q"), result, None, None)
        rm.record_error.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# maybe_record_pattern (branch coverage beyond test_pattern_documentation.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestMaybeRecordPattern:
    def test_no_user_message_skips_recording(self) -> None:
        gw = _bare_gateway()
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        procedural = MagicMock()
        procedural.search_procedures.return_value = ""
        gw._memory_manager.procedural = procedural
        wm = WorkingMemory()  # no user message
        maybe_record_pattern(
            gw,
            _session(),
            wm,
            _agent_result(tool_results=[_ok_tool("web_search")]),
        )
        procedural.save_procedure.assert_not_called()

    def test_only_stopwords_skips_recording(self) -> None:
        gw = _bare_gateway()
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        procedural = MagicMock()
        procedural.search_procedures.return_value = ""
        gw._memory_manager.procedural = procedural
        # All words ≤3 chars or in stopword list
        maybe_record_pattern(
            gw,
            _session(),
            _wm_with_user("the and for"),
            _agent_result(tool_results=[_ok_tool("web_search")]),
        )
        procedural.save_procedure.assert_not_called()

    def test_pattern_recorded_with_keywords_and_tools(self) -> None:
        gw = _bare_gateway()
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        procedural = MagicMock()
        procedural.search_procedures.return_value = ""
        gw._memory_manager.procedural = procedural
        gw._pattern_record_timestamps = []  # type: ignore[attr-defined]

        maybe_record_pattern(
            gw,
            _session(),
            _wm_with_user("Search Python tutorials online today"),
            _agent_result(tool_results=[_ok_tool("web_search"), _ok_tool("search_and_read")]),
        )
        procedural.save_procedure.assert_called_once()
        kwargs = procedural.save_procedure.call_args.kwargs
        body: str = kwargs["body"]
        assert "web_search" in body
        assert "search_and_read" in body
        # Timestamp pushed for rate limiting
        assert len(gw._pattern_record_timestamps) == 1

    def test_no_memory_manager_is_noop(self) -> None:
        gw = _bare_gateway()
        # Must not raise
        maybe_record_pattern(
            gw,
            _session(),
            _wm_with_user("Search Python tutorials online today"),
            _agent_result(tool_results=[_ok_tool("web_search")]),
        )

    def test_internal_exception_is_swallowed(self) -> None:
        gw = _bare_gateway()
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        procedural = MagicMock()
        procedural.search_procedures.side_effect = RuntimeError("blow up")
        gw._memory_manager.procedural = procedural
        # Must not raise even though search_procedures throws
        maybe_record_pattern(
            gw,
            _session(),
            _wm_with_user("Search Python tutorials online today"),
            _agent_result(tool_results=[_ok_tool("web_search")]),
        )

    def test_tool_result_without_tool_name_filtered(self) -> None:
        gw = _bare_gateway()
        gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
        procedural = MagicMock()
        procedural.search_procedures.return_value = ""
        gw._memory_manager.procedural = procedural
        gw._pattern_record_timestamps = []  # type: ignore[attr-defined]
        # ToolResult with empty tool_name → filtered out
        empty = ToolResult(tool_name="", content="x", is_error=False)
        maybe_record_pattern(
            gw,
            _session(),
            _wm_with_user("Search Python tutorials online today"),
            _agent_result(tool_results=[empty]),
        )
        procedural.save_procedure.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# persist_session
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistSession:
    @pytest.mark.asyncio
    async def test_no_session_store_is_noop(self) -> None:
        gw = _bare_gateway()
        await persist_session(gw, _session(), _wm_with_user("q"))
        # No exception → fine

    @pytest.mark.asyncio
    async def test_normal_session_persists_session_and_history(self) -> None:
        gw = _bare_gateway()
        gw._session_store = MagicMock()  # type: ignore[attr-defined]
        # Disable auto_title attribute completely so the second branch
        # doesn't fire
        del gw._session_store.auto_title
        wm = _wm_with_user("hello world")
        await persist_session(gw, _session(), wm)
        gw._session_store.save_session.assert_called_once()
        gw._session_store.save_chat_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_incognito_only_saves_metadata(self) -> None:
        gw = _bare_gateway()
        gw._session_store = MagicMock()  # type: ignore[attr-defined]
        sess = SessionContext(session_id="x", channel="cli", incognito=True)
        await persist_session(gw, sess, _wm_with_user("q"))
        gw._session_store.save_session.assert_called_once()
        # Chat history is NEVER persisted in incognito mode
        gw._session_store.save_chat_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_store_save_exception_is_swallowed(self) -> None:
        gw = _bare_gateway()
        gw._session_store = MagicMock()  # type: ignore[attr-defined]
        gw._session_store.save_session.side_effect = RuntimeError("disk full")
        # Must not raise
        await persist_session(gw, _session(), _wm_with_user("q"))

    @pytest.mark.asyncio
    async def test_auto_title_called_when_available(self) -> None:
        gw = _bare_gateway()
        store = MagicMock()
        store.auto_title = MagicMock()
        gw._session_store = store  # type: ignore[attr-defined]
        await persist_session(gw, _session(), _wm_with_user("q"))
        store.auto_title.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_title_exception_is_swallowed(self) -> None:
        gw = _bare_gateway()
        store = MagicMock()
        store.auto_title.side_effect = RuntimeError("boom")
        gw._session_store = store  # type: ignore[attr-defined]
        # Must not raise
        await persist_session(gw, _session(), _wm_with_user("q"))


# ─────────────────────────────────────────────────────────────────────────────
# persist_key_tool_results
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistKeyToolResults:
    def test_empty_results_is_noop(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(gw, wm, [])
        assert wm.chat_history == []

    def test_failed_result_skipped(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(gw, wm, [_err_tool("web_search")])
        assert wm.chat_history == []

    def test_non_context_tool_skipped(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        # "list_files" not in CONTEXT_TOOLS
        persist_key_tool_results(gw, wm, [_ok_tool("list_files", "blob")])
        assert wm.chat_history == []

    def test_empty_content_skipped(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(gw, wm, [_ok_tool("web_search", "   ")])
        assert wm.chat_history == []

    def test_context_tool_persisted_as_tool_message(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(gw, wm, [_ok_tool("web_search", "found 3 results")])
        assert len(wm.chat_history) == 1
        msg = wm.chat_history[0]
        assert msg.role == MessageRole.TOOL
        assert msg.name == "web_search"
        assert msg.content == "found 3 results"

    def test_oversized_content_truncated_with_marker(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        # Force a tiny limit so we don't have to build a 4001-char fixture
        gw._CONTEXT_RESULT_LIMIT = 10  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(gw, wm, [_ok_tool("web_search", "0123456789ABCDEF")])
        msg = wm.chat_history[0]
        assert msg.content.startswith("0123456789")
        assert "[... gekürzt]" in msg.content

    def test_multiple_results_all_persisted_in_order(self) -> None:
        gw = _bare_gateway()
        gw._CONTEXT_TOOLS = Gateway._CONTEXT_TOOLS  # type: ignore[attr-defined]
        gw._CONTEXT_RESULT_LIMIT = Gateway._CONTEXT_RESULT_LIMIT  # type: ignore[attr-defined]
        wm = WorkingMemory()
        persist_key_tool_results(
            gw,
            wm,
            [
                _ok_tool("web_search", "first"),
                _ok_tool("web_fetch", "second"),
                _err_tool("web_search"),  # skipped
                _ok_tool("not_in_context", "skipped"),  # skipped
                _ok_tool("media_extract_text", "third"),
            ],
        )
        assert [m.name for m in wm.chat_history] == [
            "web_search",
            "web_fetch",
            "media_extract_text",
        ]
        assert [m.content for m in wm.chat_history] == ["first", "second", "third"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration touchpoint: _maybe_record_pattern delegation
# ─────────────────────────────────────────────────────────────────────────────


def test_gateway_method_delegates_to_post_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Gateway._maybe_record_pattern`` is a thin wrapper around
    ``post_processing.maybe_record_pattern`` — verify it forwards args."""
    captured: dict[str, Any] = {}

    def fake_mrp(gw: Any, session: Any, wm: Any, agent_result: Any) -> None:
        captured["gw"] = gw
        captured["session"] = session
        captured["wm"] = wm
        captured["ar"] = agent_result

    monkeypatch.setattr(post_processing, "maybe_record_pattern", fake_mrp)
    gw = _bare_gateway()
    sess = _session()
    wm = _wm_with_user("q")
    ar = _agent_result()
    gw._maybe_record_pattern(sess, wm, ar)
    assert captured == {"gw": gw, "session": sess, "wm": wm, "ar": ar}


def test_pattern_rate_limit_prunes_old_timestamps_in_place() -> None:
    """Old timestamps (>1h) should be pruned from
    ``gw._pattern_record_timestamps`` even when the rest of the
    function bails out before recording."""
    gw = _bare_gateway()
    gw._memory_manager = MagicMock()  # type: ignore[attr-defined]
    procedural = MagicMock()
    procedural.search_procedures.return_value = ""
    gw._memory_manager.procedural = procedural
    now = time.monotonic()
    # Two old + one recent
    gw._pattern_record_timestamps = [  # type: ignore[attr-defined]
        now - 7200,
        now - 7300,
        now - 10,
    ]
    maybe_record_pattern(
        gw,
        _session(),
        _wm_with_user("Find latest research articles online"),
        _agent_result(tool_results=[_ok_tool("web_search")]),
    )
    # Old ones gone; the recent one stays plus the freshly-appended record.
    assert all(now - ts < 3600 for ts in gw._pattern_record_timestamps)
