"""Tests for CUAgentExecutor — closed-loop desktop automation agent."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.cu_agent import CUAgentConfig, CUAgentExecutor, CUAgentResult
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
        return CUAgentExecutor(planner, mcp, MagicMock(), MagicMock(), {}, CUAgentConfig(**config_overrides))

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
        planner._ollama.chat = AsyncMock(side_effect=[
            {"message": {"content": '{"tool": "computer_click", "params": {"x": 200, "y": 300}, "rationale": "click window"}'}},
            {"message": {"content": "DONE: Taschenrechner zeigt 459"}},
        ])

        mcp = MagicMock()
        screenshot_handler = AsyncMock(return_value={
            "success": True, "width": 1920, "height": 1080,
            "description": "Rechner sichtbar",
            "elements": [{"name": "Rechner", "type": "window", "x": 200, "y": 300, "clickable": True}],
        })
        click_handler = AsyncMock(return_value={"success": True})
        exec_handler = AsyncMock(return_value="OK")
        mcp._builtin_handlers = {
            "computer_screenshot": screenshot_handler,
            "computer_click": click_handler,
            "exec_command": exec_handler,
        }

        initial_plan = ActionPlan(
            goal="Rechner oeffnen",
            steps=[PlannedAction(tool="exec_command", params={"command": "start calc.exe"}, rationale="start")],
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
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": '{"tool": "computer_click", "params": {"x": 100, "y": 100}}'},
        })

        mcp = MagicMock()
        mcp._builtin_handlers = {
            "computer_screenshot": AsyncMock(return_value={
                "success": True, "description": "screen", "elements": [],
            }),
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
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": '{"tool": "computer_click", "params": {"x": 1, "y": 1}}'},
        })

        mcp = MagicMock()
        mcp._builtin_handlers = {
            "computer_screenshot": AsyncMock(return_value={
                "success": True, "description": "screen", "elements": [],
            }),
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
