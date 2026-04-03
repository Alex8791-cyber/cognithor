"""Tests for computer_screenshot with VisionAnalyzer integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080, 1.0)
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
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080, 1.0)
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
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080, 1.0)
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
            "jarvis.mcp.computer_use._take_screenshot_b64", return_value=("base64", 1920, 1080, 1.0)
        ):
            await tools.computer_screenshot(task_context="Reddit oeffnen")

        call_args = mock_vision.analyze_desktop.call_args
        assert "Reddit" in str(call_args)


class TestGatekeeperCUClassification:
    """Verify security classification hasn't regressed."""

    def test_screenshot_green_actions_yellow(self):
        from jarvis.config import JarvisConfig, ToolsConfig
        from jarvis.core.gatekeeper import Gatekeeper
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
                PlannedAction(
                    tool="exec_command", params={"command": "calc.exe"}, rationale="start"
                ),
                PlannedAction(
                    tool="computer_click", params={"x": 100, "y": 200}, rationale="click"
                ),
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


class TestCoordinateScaling:
    @pytest.mark.asyncio
    async def test_click_scales_coordinates(self):
        tools = ComputerUseTools()
        tools._last_scale_factor = 0.5

        with patch("jarvis.mcp.computer_use._get_pyautogui") as mock_gui:
            mock_pag = MagicMock()
            mock_gui.return_value = mock_pag
            await tools.computer_click(x=100, y=200)
            call_args = mock_pag.click.call_args
            # 100 / 0.5 = 200, 200 / 0.5 = 400
            assert call_args.kwargs.get("x", call_args[1].get("x")) == 200

    @pytest.mark.asyncio
    async def test_click_no_scaling_when_factor_is_1(self):
        tools = ComputerUseTools()
        tools._last_scale_factor = 1.0

        with patch("jarvis.mcp.computer_use._get_pyautogui") as mock_gui:
            mock_pag = MagicMock()
            mock_gui.return_value = mock_pag
            await tools.computer_click(x=100, y=200)
            call_args = mock_pag.click.call_args
            assert call_args.kwargs.get("x", call_args[1].get("x")) == 100

    @pytest.mark.asyncio
    async def test_screenshot_stores_scale_factor(self):
        tools = ComputerUseTools()
        assert tools._last_scale_factor == 1.0

        with patch("jarvis.mcp.computer_use._take_screenshot_b64") as mock_ss:
            mock_ss.return_value = ("base64data", 2560, 1440, 0.667)
            await tools.computer_screenshot()
            assert tools._last_scale_factor == 0.667

    @pytest.mark.asyncio
    async def test_drag_scales_all_coordinates(self):
        tools = ComputerUseTools()
        tools._last_scale_factor = 0.5

        with patch("jarvis.mcp.computer_use._get_pyautogui") as mock_gui:
            mock_pag = MagicMock()
            mock_gui.return_value = mock_pag
            await tools.computer_drag(start_x=100, start_y=200, end_x=300, end_y=400)
            # start_x / 0.5 = 200, start_y / 0.5 = 400
            move_args = mock_pag.moveTo.call_args
            assert move_args[0][0] == 200  # start_x scaled
            assert move_args[0][1] == 400  # start_y scaled


class TestWaitForStableScreen:
    @pytest.mark.asyncio
    async def test_returns_on_stable_screen(self):
        tools = ComputerUseTools()

        with patch("jarvis.mcp.computer_use._take_screenshot_b64") as mock_ss:
            mock_ss.return_value = ("same_image_data", 1920, 1080, 1.0)
            await tools._wait_for_stable_screen(
                min_delay_ms=10, poll_interval_ms=10, timeout_ms=5000
            )
            # Should complete quickly without hitting timeout

    @pytest.mark.asyncio
    async def test_timeout_on_changing_screen(self):
        tools = ComputerUseTools()
        counter = {"n": 0}

        def changing_screenshot(monitor_index=0):
            counter["n"] += 1
            return (f"different_image_{counter['n']}", 1920, 1080, 1.0)

        with patch("jarvis.mcp.computer_use._take_screenshot_b64", side_effect=changing_screenshot):
            import time as _time

            start = _time.monotonic()
            await tools._wait_for_stable_screen(
                min_delay_ms=10, poll_interval_ms=10, timeout_ms=200
            )
            elapsed = (_time.monotonic() - start) * 1000
            assert elapsed >= 150  # at least close to timeout
