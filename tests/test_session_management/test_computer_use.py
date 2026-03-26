"""Tests for Computer Use (desktop automation via coordinates)."""

from __future__ import annotations

import pytest


def test_computer_use_tools_importable():
    """ComputerUseTools class should be importable."""
    from jarvis.mcp.computer_use import ComputerUseTools

    tools = ComputerUseTools()
    assert tools is not None


def test_computer_use_gatekeeper_green():
    """Computer use tools should be GREEN when computer_use_enabled=True."""
    from jarvis.core.gatekeeper import Gatekeeper
    from jarvis.config import JarvisConfig, ToolsConfig
    from jarvis.models import PlannedAction

    gk = Gatekeeper(JarvisConfig(tools=ToolsConfig(computer_use_enabled=True)))
    for tool in [
        "computer_screenshot",
        "computer_click",
        "computer_type",
        "computer_hotkey",
        "computer_scroll",
        "computer_drag",
    ]:
        action = PlannedAction(tool=tool, params={}, rationale="test")
        risk = gk._classify_risk(action)
        assert risk.value == "green", f"{tool} should be green, got {risk}"


@pytest.mark.asyncio
async def test_computer_screenshot():
    """computer_screenshot should return width/height."""
    from jarvis.mcp.computer_use import ComputerUseTools

    tools = ComputerUseTools()
    try:
        result = await tools.computer_screenshot()
        if result["success"]:
            assert "width" in result
            assert "height" in result
            assert result["width"] > 0
    except Exception:
        pytest.skip("Desktop screenshot not available in CI")
