"""Smoke test for 03_first_skill/main.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "_first_skill_main",
        Path(__file__).parent / "main.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_gcd_tool_callable() -> None:
    main_mod = _load_main()
    result = await main_mod.calculate_gcd(48, 18)
    assert result == 6


def test_math_helper_skill_registered() -> None:
    _load_main()
    from cognithor.sdk.decorators import get_registry

    defn = get_registry().get_agent("math_helper")
    assert defn is not None
    assert defn.name == "math_helper"
    assert "calculate_gcd" in defn.tools
    assert "ggT" in defn.trigger_keywords
    assert "Mathematik-Assistent" in defn.system_prompt


@pytest.mark.asyncio
async def test_math_helper_on_message() -> None:
    main_mod = _load_main()
    skill = main_mod.MathHelperAgent()
    reply = await skill.on_message("ggT 12 8")
    assert "Math-Helper" in reply
