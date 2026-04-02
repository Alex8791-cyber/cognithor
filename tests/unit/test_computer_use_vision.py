"""Tests for computer_screenshot with VisionAnalyzer integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from jarvis.browser.vision import VisionAnalysisResult
from jarvis.mcp.computer_use import ComputerUseTools


class TestComputerScreenshotWithVision:
    @pytest.mark.asyncio
    async def test_elements_in_result(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(
            return_value=VisionAnalysisResult(
                success=True,
                description="Desktop mit Rechner",
                elements=[
                    {
                        "name": "Rechner",
                        "type": "window",
                        "x": 200,
                        "y": 300,
                        "w": 400,
                        "h": 500,
                        "text": "",
                        "clickable": True,
                    }
                ],
            )
        )

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch(
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080)
        ):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert len(result["elements"]) == 1
        assert result["elements"][0]["name"] == "Rechner"
        assert "Rechner" in result["description"]
        mock_vision.analyze_desktop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_vision_returns_empty_elements(self):
        tools = ComputerUseTools(vision_analyzer=None)

        with patch(
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080)
        ):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert result["elements"] == []
        assert "No vision" in result["description"]

    @pytest.mark.asyncio
    async def test_vision_error_returns_empty_elements(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(
            return_value=VisionAnalysisResult(
                success=False,
                error="GPU timeout",
            )
        )

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch(
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080)
        ):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert result["elements"] == []
        assert "GPU timeout" in result["description"]

    @pytest.mark.asyncio
    async def test_task_context_passed_through(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(
            return_value=VisionAnalysisResult(
                success=True,
                description="OK",
                elements=[],
            )
        )

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch(
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080)
        ):
            await tools.computer_screenshot(task_context="Reddit oeffnen")

        call_args = mock_vision.analyze_desktop.call_args
        assert "Reddit" in str(call_args)


class TestGatekeeperCUClassification:
    """Verify security classification hasn't regressed."""

    def test_screenshot_green_actions_yellow(self):
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.config import JarvisConfig, ToolsConfig
        from jarvis.models import PlannedAction

        config = JarvisConfig(tools=ToolsConfig(computer_use_enabled=True))
        gk = Gatekeeper(config)

        # Screenshot is GREEN (read-only)
        action = PlannedAction(tool="computer_screenshot", params={}, rationale="test")
        assert gk._classify_risk(action).name == "GREEN"

        # Active actions are YELLOW (not GREEN!)
        for tool in ["computer_click", "computer_type", "computer_hotkey"]:
            action = PlannedAction(tool=tool, params={}, rationale="test")
            assert gk._classify_risk(action).name == "YELLOW", f"{tool} must be YELLOW"


class TestGatewayCUDetection:
    """Tests for _is_cu_plan detection."""

    def test_cu_plan_with_computer_click(self):
        from jarvis.gateway.gateway import Gateway
        from jarvis.models import ActionPlan, PlannedAction

        plan = ActionPlan(
            goal="test",
            steps=[
                PlannedAction(tool="exec_command", params={"command": "calc.exe"}, rationale="start"),
                PlannedAction(tool="computer_click", params={"x": 100, "y": 200}, rationale="click"),
            ],
        )
        assert Gateway._is_cu_plan(plan) is True

    def test_non_cu_plan(self):
        from jarvis.gateway.gateway import Gateway
        from jarvis.models import ActionPlan, PlannedAction

        plan = ActionPlan(
            goal="test",
            steps=[PlannedAction(tool="web_search", params={"query": "test"}, rationale="search")],
        )
        assert Gateway._is_cu_plan(plan) is False

    def test_direct_response_not_cu(self):
        from jarvis.gateway.gateway import Gateway
        from jarvis.models import ActionPlan

        plan = ActionPlan(goal="test", direct_response="Hello!")
        assert Gateway._is_cu_plan(plan) is False
