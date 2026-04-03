"""Tests for CUAgentExecutor — closed-loop desktop automation agent."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.cu_agent import (
    CUAgentConfig,
    CUAgentExecutor,
    CUAgentResult,
    CUSubTask,
    CUTaskDecomposer,
    CUTaskPlan,
)
from jarvis.models import ActionPlan, PlannedAction


class TestCUAgentConfig:
    def test_defaults(self):
        cfg = CUAgentConfig()
        assert cfg.max_iterations == 30
        assert cfg.max_duration_seconds == 480
        assert cfg.vision_model == "qwen3-vl:32b"
        assert cfg.stuck_detection_threshold == 3

    def test_custom(self):
        cfg = CUAgentConfig(max_iterations=10, max_duration_seconds=120)
        assert cfg.max_iterations == 10
        assert cfg.max_duration_seconds == 120


class TestCUAgentResult:
    def test_defaults(self):
        r = CUAgentResult()
        assert r.success is False
        assert r.iterations == 0
        assert r.duration_ms == 0
        assert r.tool_results == []
        assert r.abort_reason == ""
        assert r.extracted_content == ""
        assert r.action_history == []


class TestCUAgentAbort:
    def _make_agent(self, **config_overrides) -> CUAgentExecutor:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        mcp = MagicMock()
        mcp._builtin_handlers = {}
        return CUAgentExecutor(
            planner, mcp, MagicMock(), MagicMock(), {}, CUAgentConfig(**config_overrides)
        )

    def test_check_abort_max_iterations(self):
        agent = self._make_agent(max_iterations=5)
        result = CUAgentResult(iterations=5)
        assert agent._check_abort(result, time.monotonic(), None) == "max_iterations"

    def test_check_abort_timeout(self):
        agent = self._make_agent(max_duration_seconds=1)
        result = CUAgentResult(iterations=1)
        start = time.monotonic() - 2
        assert agent._check_abort(result, start, None) == "timeout"

    def test_check_abort_user_cancel(self):
        agent = self._make_agent()
        result = CUAgentResult(iterations=1)
        assert agent._check_abort(result, time.monotonic(), lambda: True) == "user_cancel"

    def test_check_abort_stuck_loop(self):
        agent = self._make_agent(stuck_detection_threshold=3)
        agent._recent_actions = ["click:x=100,y=200"] * 3
        result = CUAgentResult(iterations=3)
        assert agent._check_abort(result, time.monotonic(), None) == "stuck_loop"

    def test_check_abort_no_abort(self):
        agent = self._make_agent()
        result = CUAgentResult(iterations=1)
        assert agent._check_abort(result, time.monotonic(), None) == ""


class TestCUDecisionParsing:
    def _make_agent(self) -> CUAgentExecutor:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        mcp = MagicMock()
        mcp._builtin_handlers = {}
        return CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})

    def test_parse_tool_json(self):
        agent = self._make_agent()
        result = agent._parse_tool_decision(
            '{"tool": "computer_click", "params": {"x": 200, "y": 300}, "rationale": "click"}'
        )
        assert result is not None
        assert result["tool"] == "computer_click"
        assert result["params"]["x"] == 200

    def test_parse_json_in_markdown(self):
        agent = self._make_agent()
        raw = '```json\n{"tool": "computer_type", "params": {"text": "hello"}, "rationale": "type"}\n```'
        result = agent._parse_tool_decision(raw)
        assert result is not None
        assert result["tool"] == "computer_type"

    def test_parse_garbage_returns_none(self):
        agent = self._make_agent()
        assert agent._parse_tool_decision("This is just text.") is None

    def test_parse_done_not_a_tool(self):
        agent = self._make_agent()
        assert agent._parse_tool_decision("DONE: Rechner zeigt 459") is None


class TestCUToolExecution:
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        planner = MagicMock()
        mcp = MagicMock()
        handler = AsyncMock(return_value={"success": True, "action": "click"})
        mcp._builtin_handlers = {"computer_click": handler}

        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})
        result = await agent._execute_tool("computer_click", {"x": 100, "y": 200})

        assert result.success is True
        assert "click" in result.content
        handler.assert_awaited_once_with(x=100, y=200)

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        planner = MagicMock()
        mcp = MagicMock()
        mcp._builtin_handlers = {}

        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})
        result = await agent._execute_tool("nonexistent", {})

        assert result.is_error is True
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        planner = MagicMock()
        mcp = MagicMock()
        handler = AsyncMock(side_effect=RuntimeError("pyautogui crash"))
        mcp._builtin_handlers = {"computer_click": handler}

        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})
        result = await agent._execute_tool("computer_click", {"x": 0, "y": 0})

        assert result.is_error is True
        assert "pyautogui crash" in result.content


class TestCUAgentExecuteLoop:
    """Tests for the full execute() loop."""

    @pytest.mark.asyncio
    async def test_happy_path_done_in_3_iterations(self):
        """Agent: exec → screenshot(sees window) → decide(DONE)."""
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(
            side_effect=[
                {
                    "message": {
                        "content": '{"tool": "computer_click", "params": {"x": 200, "y": 300}, "rationale": "click window"}'
                    }
                },
                {"message": {"content": "DONE: Taschenrechner zeigt 459"}},
            ]
        )

        mcp = MagicMock()
        screenshot_handler = AsyncMock(
            return_value={
                "success": True,
                "width": 1920,
                "height": 1080,
                "description": "Rechner sichtbar",
                "elements": [
                    {"name": "Rechner", "type": "window", "x": 200, "y": 300, "clickable": True}
                ],
            }
        )
        click_handler = AsyncMock(return_value={"success": True})
        exec_handler = AsyncMock(return_value="OK")
        mcp._builtin_handlers = {
            "computer_screenshot": screenshot_handler,
            "computer_click": click_handler,
            "exec_command": exec_handler,
        }

        initial_plan = ActionPlan(
            goal="Rechner oeffnen",
            steps=[
                PlannedAction(
                    tool="exec_command", params={"command": "start calc.exe"}, rationale="start"
                )
            ],
        )

        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})
        result = await agent.execute(goal="Rechner oeffnen", initial_plan=initial_plan)

        assert result.success is True
        assert result.abort_reason == "done"
        assert result.iterations >= 1
        assert len(result.action_history) >= 2
        assert "459" in str(result.action_history)

    @pytest.mark.asyncio
    async def test_abort_on_max_iterations(self):
        """Agent stops after max_iterations."""
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(
            return_value={
                "message": {
                    "content": '{"tool": "computer_click", "params": {"x": 100, "y": 100}}'
                },
            }
        )

        mcp = MagicMock()
        mcp._builtin_handlers = {
            "computer_screenshot": AsyncMock(
                return_value={
                    "success": True,
                    "description": "screen",
                    "elements": [],
                }
            ),
            "computer_click": AsyncMock(return_value={"success": True}),
        }

        config = CUAgentConfig(max_iterations=3)
        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {}, config)

        initial_plan = ActionPlan(goal="test", steps=[])
        result = await agent.execute(goal="test", initial_plan=initial_plan)

        assert result.success is False
        assert result.abort_reason == "max_iterations"
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_abort_on_user_cancel(self):
        """Agent stops when cancel_check returns True."""
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(
            return_value={
                "message": {"content": '{"tool": "computer_click", "params": {"x": 1, "y": 1}}'},
            }
        )

        mcp = MagicMock()
        mcp._builtin_handlers = {
            "computer_screenshot": AsyncMock(
                return_value={
                    "success": True,
                    "description": "screen",
                    "elements": [],
                }
            ),
            "computer_click": AsyncMock(return_value={"success": True}),
        }

        agent = CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})
        result = await agent.execute(
            goal="test",
            initial_plan=ActionPlan(goal="test", steps=[]),
            cancel_check=lambda: True,
        )

        assert result.abort_reason == "user_cancel"
        assert result.iterations <= 1


class TestCUSubTask:
    def test_defaults(self):
        st = CUSubTask(name="open_app", goal="Oeffne Reddit", completion_hint="Reddit sichtbar")
        assert st.name == "open_app"
        assert st.goal == "Oeffne Reddit"
        assert st.completion_hint == "Reddit sichtbar"
        assert st.max_iterations == 10
        assert st.available_tools == []
        assert st.extract_content is False
        assert st.content_key == ""
        assert st.output_file == ""
        assert st.status == "pending"

    def test_custom(self):
        st = CUSubTask(
            name="scroll",
            goal="Scrolle Posts",
            completion_hint="10 Posts gelesen",
            max_iterations=15,
            available_tools=["computer_scroll", "extract_text"],
            extract_content=True,
            content_key="posts",
        )
        assert st.max_iterations == 15
        assert st.extract_content is True
        assert st.content_key == "posts"
        assert "computer_scroll" in st.available_tools


class TestCUTaskPlan:
    def test_defaults(self):
        plan = CUTaskPlan(original_goal="test", sub_tasks=[])
        assert plan.original_goal == "test"
        assert plan.sub_tasks == []
        assert plan.output_filename == ""
        assert plan.variables == {}

    def test_with_sub_tasks(self):
        st1 = CUSubTask(name="a", goal="g1", completion_hint="h1")
        st2 = CUSubTask(name="b", goal="g2", completion_hint="h2")
        plan = CUTaskPlan(
            original_goal="do stuff",
            sub_tasks=[st1, st2],
            output_filename="out.txt",
            variables={"date": "20260403"},
        )
        assert len(plan.sub_tasks) == 2
        assert plan.variables["date"] == "20260403"


class TestCUAgentResultExtended:
    def test_new_fields_default(self):
        r = CUAgentResult()
        assert r.output_files == []
        assert r.task_summary == ""

    def test_new_fields_populated(self):
        r = CUAgentResult(
            output_files=["/home/user/docs/out.txt"],
            task_summary="3/3 Phasen abgeschlossen.",
        )
        assert len(r.output_files) == 1
        assert "3/3" in r.task_summary


class TestCompletionHintMatching:
    def _make_agent(self) -> CUAgentExecutor:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        mcp = MagicMock()
        mcp._builtin_handlers = {}
        return CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})

    def test_hint_matches_when_keywords_present(self):
        agent = self._make_agent()
        assert agent._check_completion_hint(
            "locallama erscheint in URL oder Titel",
            "Browser zeigt reddit.com/r/locallama im Titel",
        ) is True

    def test_hint_no_match_when_keywords_missing(self):
        agent = self._make_agent()
        assert agent._check_completion_hint(
            "locallama erscheint in URL oder Titel",
            "Desktop mit verschiedenen Icons sichtbar",
        ) is False

    def test_hint_empty_returns_false(self):
        agent = self._make_agent()
        assert agent._check_completion_hint("", "something on screen") is False

    def test_hint_short_words_ignored(self):
        agent = self._make_agent()
        assert agent._check_completion_hint(
            "in URL Titel locallama erscheint",
            "locallama erscheint Titel",
        ) is True

    def test_hint_partial_match_below_threshold(self):
        agent = self._make_agent()
        assert agent._check_completion_hint(
            "Rechner Fenster zeigt Ergebnis sichtbar",
            "Rechner Fenster ist im Hintergrund",
        ) is False

    def test_hint_60_percent_threshold(self):
        agent = self._make_agent()
        assert agent._check_completion_hint(
            "Reddit Seite zeigt locallama Ergebnisse",
            "Reddit zeigt locallama und andere Dinge Ergebnisse",
        ) is True


class TestScreenshotSimilarity:
    def _make_agent(self) -> CUAgentExecutor:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        mcp = MagicMock()
        mcp._builtin_handlers = {}
        return CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {})

    def test_identical_descriptions(self):
        agent = self._make_agent()
        assert agent._screenshot_similarity("Desktop mit Icons", "Desktop mit Icons") == 1.0

    def test_completely_different(self):
        agent = self._make_agent()
        sim = agent._screenshot_similarity("Rechner zeigt Ergebnis", "Browser offen leer")
        assert sim < 0.2

    def test_empty_strings(self):
        agent = self._make_agent()
        assert agent._screenshot_similarity("", "something") == 0.0
        assert agent._screenshot_similarity("something", "") == 0.0
        assert agent._screenshot_similarity("", "") == 0.0

    def test_high_overlap(self):
        agent = self._make_agent()
        sim = agent._screenshot_similarity(
            "Reddit Seite mit locallama Posts sichtbar",
            "Reddit Seite mit locallama Posts und Kommentare sichtbar",
        )
        assert sim > 0.7


class TestCUTaskDecomposerVariables:
    def _make_decomposer(self) -> CUTaskDecomposer:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        return CUTaskDecomposer(planner, CUAgentConfig())

    def test_resolve_variables_has_date(self):
        d = self._make_decomposer()
        v = d._resolve_variables("some goal")
        assert "date" in v
        assert len(v["date"]) == 8
        assert v["date"].isdigit()

    def test_resolve_variables_has_documents(self):
        d = self._make_decomposer()
        v = d._resolve_variables("some goal")
        assert "documents" in v
        assert "Documents" in v["documents"] or "documents" in v["documents"].lower()

    def test_resolve_variables_has_date_formats(self):
        d = self._make_decomposer()
        v = d._resolve_variables("some goal")
        assert "date_dots" in v
        assert "date_iso" in v

    def test_resolve_output_path_simple(self):
        d = self._make_decomposer()
        variables = {"date": "20260403", "documents": "C:\\Users\\Test\\Documents"}
        path = d._resolve_output_path("Reddit_fetch_{date}.txt", variables)
        assert path == "C:\\Users\\Test\\Documents\\Reddit_fetch_20260403.txt"

    def test_resolve_output_path_no_variables(self):
        d = self._make_decomposer()
        variables = {"date": "20260403", "documents": "C:\\Users\\Test\\Documents"}
        path = d._resolve_output_path("static_name.txt", variables)
        assert path == "C:\\Users\\Test\\Documents\\static_name.txt"


class TestCUTaskDecomposerParsing:
    def _make_decomposer(self) -> CUTaskDecomposer:
        planner = MagicMock()
        planner._ollama = AsyncMock()
        return CUTaskDecomposer(planner, CUAgentConfig())

    def test_parse_subtasks_valid_json(self):
        d = self._make_decomposer()
        raw = json.dumps([
            {"name": "open_app", "goal": "Oeffne Reddit", "completion_hint": "Reddit sichtbar",
             "max_iterations": 8, "tools": ["computer_click"], "extract_content": False,
             "content_key": "", "output_file": ""},
            {"name": "search", "goal": "Suche locallama", "completion_hint": "locallama in URL",
             "max_iterations": 6, "tools": ["computer_type", "computer_click"],
             "extract_content": False, "content_key": "", "output_file": ""},
        ])
        tasks = d._parse_subtasks(raw)
        assert len(tasks) == 2
        assert tasks[0].name == "open_app"
        assert tasks[0].max_iterations == 8
        assert tasks[1].available_tools == ["computer_type", "computer_click"]

    def test_parse_subtasks_markdown_block(self):
        d = self._make_decomposer()
        raw = (
            "Hier ist der Plan:\n```json\n"
            + json.dumps([{"name": "step1", "goal": "Do thing", "completion_hint": "done",
                          "max_iterations": 5, "tools": [], "extract_content": False,
                          "content_key": "", "output_file": ""}])
            + "\n```\nDas war der Plan."
        )
        tasks = d._parse_subtasks(raw)
        assert len(tasks) == 1
        assert tasks[0].name == "step1"

    def test_parse_subtasks_garbage_returns_empty(self):
        d = self._make_decomposer()
        tasks = d._parse_subtasks("This is not JSON at all, just rambling text.")
        assert tasks == []

    def test_parse_subtasks_partial_fields_uses_defaults(self):
        d = self._make_decomposer()
        raw = json.dumps([{"name": "x", "goal": "y", "completion_hint": "z"}])
        tasks = d._parse_subtasks(raw)
        assert len(tasks) == 1
        assert tasks[0].max_iterations == 10
        assert tasks[0].extract_content is False
        assert tasks[0].available_tools == []

    def test_parse_subtasks_tools_mapped_to_available_tools(self):
        d = self._make_decomposer()
        raw = json.dumps([{"name": "a", "goal": "b", "completion_hint": "c",
                          "tools": ["computer_click", "extract_text"]}])
        tasks = d._parse_subtasks(raw)
        assert tasks[0].available_tools == ["computer_click", "extract_text"]


class TestCUTaskDecomposerDecompose:
    @pytest.mark.asyncio
    async def test_decompose_happy_path(self):
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": json.dumps([
                {"name": "open_app", "goal": "Oeffne Reddit", "completion_hint": "Reddit sichtbar",
                 "max_iterations": 8, "tools": ["computer_click"], "output_file": ""},
                {"name": "write_result", "goal": "Schreibe Datei", "completion_hint": "Datei geschrieben",
                 "max_iterations": 5, "tools": ["write_file"], "output_file": "result_{date}.txt"},
            ])}
        })
        d = CUTaskDecomposer(planner, CUAgentConfig())
        plan = await d.decompose("Oeffne Reddit und speichere")
        assert len(plan.sub_tasks) == 2
        assert plan.sub_tasks[0].name == "open_app"
        assert "result_" in plan.sub_tasks[1].output_file
        assert "Documents" in plan.sub_tasks[1].output_file
        assert plan.output_filename == plan.sub_tasks[1].output_file

    @pytest.mark.asyncio
    async def test_decompose_llm_failure_degrades_to_single_task(self):
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
        d = CUTaskDecomposer(planner, CUAgentConfig())
        plan = await d.decompose("Mach etwas")
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].name == "full_task"
        assert plan.sub_tasks[0].goal == "Mach etwas"

    @pytest.mark.asyncio
    async def test_decompose_garbage_response_degrades(self):
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": "Ich bin ein Sprachmodell und kann keine Phasen erzeugen."}
        })
        d = CUTaskDecomposer(planner, CUAgentConfig())
        plan = await d.decompose("Irgendwas")
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].name == "full_task"

    @pytest.mark.asyncio
    async def test_decompose_think_tags_stripped(self):
        planner = MagicMock()
        planner._ollama = AsyncMock()
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": (
                "<think>Let me think about this...</think>"
                + json.dumps([{"name": "step1", "goal": "do it", "completion_hint": "done"}])
            )}
        })
        d = CUTaskDecomposer(planner, CUAgentConfig())
        plan = await d.decompose("Test")
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].name == "step1"
