"""Smoke test for 02_first_tool/main.py.

Verifies the `@tool`-decorated function is callable, registered in the SDK
registry, and returns the expected schema.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "_first_tool_main",
        Path(__file__).parent / "main.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_word_count_tool_callable() -> None:
    main_mod = _load_main()
    count = await main_mod.word_count("Hallo Cognithor Welt")
    assert count == 3


def test_word_count_tool_registered() -> None:
    # Importing main.py registers the tool as a side effect of @tool execution.
    _load_main()
    from cognithor.sdk.decorators import get_registry

    defn = get_registry().get_tool("word_count")
    assert defn is not None
    assert defn.name == "word_count"
    schema = defn.input_schema
    assert schema["type"] == "object"
    assert "text" in schema["properties"]
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["required"] == ["text"]
