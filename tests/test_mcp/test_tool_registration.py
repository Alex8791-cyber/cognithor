"""Tests für die Tool-Registrierungsfunktionen (Browser + Media)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.mcp.browser import BROWSER_TOOL_SCHEMAS, register_browser_tools
from jarvis.mcp.media import MEDIA_TOOL_SCHEMAS, register_media_tools


class MockMCPClient:
    """Simpler Mock für den JarvisMCPClient."""

    def __init__(self) -> None:
        self.registered: dict[str, dict] = {}

    def register_builtin_handler(
        self,
        name: str,
        handler: object,
        *,
        description: str = "",
        input_schema: dict | None = None,
    ) -> None:
        self.registered[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema,
        }


class TestRegisterBrowserTools:
    def test_all_tools_registered(self) -> None:
        client = MockMCPClient()
        tool = register_browser_tools(client)

        assert tool is not None
        for name in BROWSER_TOOL_SCHEMAS:
            assert name in client.registered, f"Browser-Tool '{name}' nicht registriert"

    def test_handlers_are_callable(self) -> None:
        client = MockMCPClient()
        register_browser_tools(client)

        for name, entry in client.registered.items():
            assert callable(entry["handler"]), f"Handler für '{name}' nicht aufrufbar"

    def test_descriptions_non_empty(self) -> None:
        client = MockMCPClient()
        register_browser_tools(client)

        for name, entry in client.registered.items():
            assert entry["description"], f"Description für '{name}' ist leer"

    def test_schemas_present(self) -> None:
        client = MockMCPClient()
        register_browser_tools(client)

        for name, entry in client.registered.items():
            assert entry["input_schema"] is not None, f"Schema für '{name}' fehlt"


class TestRegisterMediaTools:
    def test_all_tools_registered(self) -> None:
        client = MockMCPClient()
        pipeline = register_media_tools(client)

        assert pipeline is not None
        for name in MEDIA_TOOL_SCHEMAS:
            assert name in client.registered, f"Media-Tool '{name}' nicht registriert"

    def test_handlers_are_callable(self) -> None:
        client = MockMCPClient()
        register_media_tools(client)

        for name, entry in client.registered.items():
            assert callable(entry["handler"]), f"Handler für '{name}' nicht aufrufbar"

    def test_descriptions_non_empty(self) -> None:
        client = MockMCPClient()
        register_media_tools(client)

        for name, entry in client.registered.items():
            assert entry["description"], f"Description für '{name}' ist leer"

    def test_expected_tool_count(self) -> None:
        client = MockMCPClient()
        register_media_tools(client)
        assert len(client.registered) == 6
