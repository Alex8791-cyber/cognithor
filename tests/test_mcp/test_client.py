"""Tests für JarvisMCPClient – Multi-Server MCP-Client.

Testet:
  - Builtin-Handler: Registrierung, Aufruf (sync + async), Fehler
  - Tool-Dispatching: call_tool an richtigen Handler
  - Schemas: get_tool_schemas, get_tool_list
  - Server-Config: YAML laden, fehlende Datei, ungültiges YAML
  - Disconnect-Verhalten
  - Tool nicht gefunden
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.mcp.client import JarvisMCPClient, ToolCallResult

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(tmp_path: Path) -> JarvisConfig:
    cfg = JarvisConfig(
        jarvis_home=tmp_path / ".jarvis",
        security=SecurityConfig(allowed_paths=[str(tmp_path)]),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def client(config: JarvisConfig) -> JarvisMCPClient:
    return JarvisMCPClient(config)


# =============================================================================
# ToolCallResult
# =============================================================================


class TestToolCallResult:
    def test_success_result(self) -> None:
        r = ToolCallResult(content="OK", is_error=False)
        assert r.content == "OK"
        assert r.is_error is False

    def test_error_result(self) -> None:
        r = ToolCallResult(content="Fehler", is_error=True)
        assert r.is_error is True

    def test_default_values(self) -> None:
        r = ToolCallResult()
        assert r.content == ""
        assert r.is_error is False


# =============================================================================
# Builtin-Handler Registrierung
# =============================================================================


class TestBuiltinRegistration:
    def test_register_sync_handler(self, client: JarvisMCPClient) -> None:
        """Synchrone Handler werden registriert."""

        def greet(name: str) -> str:
            return f"Hallo {name}"

        client.register_builtin_handler(
            "greet",
            greet,
            description="Begrüßung",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert "greet" in client.get_tool_list()
        schemas = client.get_tool_schemas()
        assert schemas["greet"]["description"] == "Begrüßung"

    def test_register_async_handler(self, client: JarvisMCPClient) -> None:
        """Asynchrone Handler werden registriert."""

        async def async_greet(name: str) -> str:
            return f"Hallo {name}"

        client.register_builtin_handler("async_greet", async_greet, description="Async Begrüßung")
        assert "async_greet" in client.get_tool_list()

    def test_register_multiple_handlers(self, client: JarvisMCPClient) -> None:
        """Mehrere Handler können registriert werden."""
        client.register_builtin_handler("tool_a", lambda: "a")
        client.register_builtin_handler("tool_b", lambda: "b")
        client.register_builtin_handler("tool_c", lambda: "c")

        tools = client.get_tool_list()
        assert tools == ["tool_a", "tool_b", "tool_c"]  # Alphabetisch sortiert
        assert client.tool_count == 3


# =============================================================================
# Builtin-Handler Aufruf
# =============================================================================


class TestBuiltinCall:
    @pytest.mark.asyncio
    async def test_call_sync_handler(self, client: JarvisMCPClient) -> None:
        """Synchroner Handler wird korrekt aufgerufen."""

        def add(a: int, b: int) -> str:
            return str(a + b)

        client.register_builtin_handler("add", add)
        result = await client.call_tool("add", {"a": 3, "b": 4})
        assert result.content == "7"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_call_async_handler(self, client: JarvisMCPClient) -> None:
        """Asynchroner Handler wird korrekt aufgerufen."""

        async def async_add(a: int, b: int) -> str:
            await asyncio.sleep(0)  # Simulate async work
            return str(a + b)

        client.register_builtin_handler("async_add", async_add)
        result = await client.call_tool("async_add", {"a": 10, "b": 20})
        assert result.content == "30"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_call_handler_with_error(self, client: JarvisMCPClient) -> None:
        """Handler-Fehler werden als ToolCallResult mit is_error=True zurückgegeben."""

        def broken(**kwargs: Any) -> str:
            raise ValueError("Kaputt!")

        client.register_builtin_handler("broken", broken)
        result = await client.call_tool("broken", {})
        assert result.is_error is True
        assert "Kaputt!" in result.content

    @pytest.mark.asyncio
    async def test_call_handler_returns_non_string(self, client: JarvisMCPClient) -> None:
        """Nicht-String-Rückgabe wird zu String konvertiert."""

        def returns_dict() -> dict:
            return {"status": "ok", "count": 42}

        client.register_builtin_handler("returns_dict", returns_dict)
        result = await client.call_tool("returns_dict", {})
        assert "ok" in result.content
        assert "42" in result.content

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, client: JarvisMCPClient) -> None:
        """Nicht vorhandenes Tool gibt is_error=True zurück."""
        result = await client.call_tool("gibts_nicht", {})
        assert result.is_error is True
        assert "nicht gefunden" in result.content


# =============================================================================
# Tool-Schemas
# =============================================================================


class TestToolSchemas:
    def test_get_tool_schemas_empty(self, client: JarvisMCPClient) -> None:
        """Ohne registrierte Tools ist Schema leer."""
        assert client.get_tool_schemas() == {}

    def test_get_tool_schemas_with_tools(self, client: JarvisMCPClient) -> None:
        """Schemas enthalten alle registrierten Tools."""
        client.register_builtin_handler(
            "test_tool",
            lambda: "x",
            description="Test-Beschreibung",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        schemas = client.get_tool_schemas()
        assert "test_tool" in schemas
        assert schemas["test_tool"]["name"] == "test_tool"
        assert schemas["test_tool"]["description"] == "Test-Beschreibung"
        assert "properties" in schemas["test_tool"]["inputSchema"]

    def test_get_tool_list_sorted(self, client: JarvisMCPClient) -> None:
        """Tool-Liste ist alphabetisch sortiert."""
        client.register_builtin_handler("zebra", lambda: "z")
        client.register_builtin_handler("alpha", lambda: "a")
        client.register_builtin_handler("mitte", lambda: "m")

        assert client.get_tool_list() == ["alpha", "mitte", "zebra"]

    def test_tool_count(self, client: JarvisMCPClient) -> None:
        assert client.tool_count == 0
        client.register_builtin_handler("t1", lambda: "")
        assert client.tool_count == 1
        client.register_builtin_handler("t2", lambda: "")
        assert client.tool_count == 2


# =============================================================================
# Server-Config Laden
# =============================================================================


class TestServerConfigLoading:
    def test_no_config_file(self, client: JarvisMCPClient) -> None:
        """Ohne config.yaml gibt es keine Server-Configs."""
        configs = client._load_server_configs()
        assert configs == {}

    def test_load_valid_config(self, config: JarvisConfig) -> None:
        """Gültige YAML-Config wird geladen."""
        mcp_config = {
            "servers": {
                "my-server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "my_server"],
                    "enabled": True,
                }
            }
        }
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text(yaml.dump(mcp_config))

        client = JarvisMCPClient(config)
        configs = client._load_server_configs()
        assert "my-server" in configs
        assert configs["my-server"].transport == "stdio"
        assert configs["my-server"].command == "python"

    def test_load_disabled_server(self, config: JarvisConfig) -> None:
        """Deaktivierte Server werden geladen aber nicht verbunden."""
        mcp_config = {
            "servers": {
                "disabled-server": {
                    "transport": "stdio",
                    "command": "python",
                    "enabled": False,
                }
            }
        }
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text(yaml.dump(mcp_config))

        client = JarvisMCPClient(config)
        configs = client._load_server_configs()
        assert configs["disabled-server"].enabled is False

    def test_load_invalid_server_entry(self, config: JarvisConfig) -> None:
        """Ungültige Server-Einträge werden übersprungen."""
        mcp_config = {
            "servers": {
                "bad-server": "not_a_dict",
                "good-server": {
                    "transport": "stdio",
                    "command": "python",
                },
            }
        }
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text(yaml.dump(mcp_config))

        client = JarvisMCPClient(config)
        configs = client._load_server_configs()
        assert "good-server" in configs
        assert "bad-server" not in configs

    def test_load_empty_yaml(self, config: JarvisConfig) -> None:
        """Leere YAML-Datei gibt keine Server zurück."""
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text("")

        client = JarvisMCPClient(config)
        configs = client._load_server_configs()
        assert configs == {}

    def test_load_yaml_without_servers_key(self, config: JarvisConfig) -> None:
        """YAML ohne 'servers' Key gibt keine Server zurück."""
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text(yaml.dump({"other_key": "value"}))

        client = JarvisMCPClient(config)
        configs = client._load_server_configs()
        assert configs == {}


# =============================================================================
# Disconnect
# =============================================================================


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_clears_registries(self, client: JarvisMCPClient) -> None:
        """disconnect_all leert alle Registries."""
        client.register_builtin_handler("test", lambda: "x")
        assert client.tool_count == 1

        await client.disconnect_all()
        assert client.tool_count == 0
        assert client.server_count == 0
        assert client.get_tool_list() == []

    @pytest.mark.asyncio
    async def test_disconnect_empty_client(self, client: JarvisMCPClient) -> None:
        """disconnect_all auf leeren Client ist safe."""
        await client.disconnect_all()  # Kein Fehler


# =============================================================================
# Integration: FileSystem + Shell über MCP-Client
# =============================================================================


class TestBuiltinIntegration:
    @pytest.mark.asyncio
    async def test_filesystem_via_client(self, config: JarvisConfig) -> None:
        """FileSystem-Tools über den MCP-Client aufrufen."""
        from jarvis.mcp.filesystem import register_fs_tools

        client = JarvisMCPClient(config)
        register_fs_tools(client, config)

        # Datei schreiben
        result = await client.call_tool(
            "write_file",
            {
                "path": str(config.workspace_dir / "test.txt"),
                "content": "Jarvis Test",
            },
        )
        assert result.is_error is False
        assert "geschrieben" in result.content

        # Datei lesen
        result = await client.call_tool(
            "read_file",
            {
                "path": str(config.workspace_dir / "test.txt"),
            },
        )
        assert result.is_error is False
        assert "Jarvis Test" in result.content

        # Datei editieren
        result = await client.call_tool(
            "edit_file",
            {
                "path": str(config.workspace_dir / "test.txt"),
                "old_text": "Test",
                "new_text": "Welt",
            },
        )
        assert result.is_error is False
        assert "bearbeitet" in result.content

        # Ergebnis prüfen
        result = await client.call_tool(
            "read_file",
            {
                "path": str(config.workspace_dir / "test.txt"),
            },
        )
        assert "Jarvis Welt" in result.content

    @pytest.mark.asyncio
    async def test_shell_via_client(self, config: JarvisConfig) -> None:
        """Shell-Tool über den MCP-Client aufrufen."""
        from jarvis.mcp.shell import register_shell_tools

        client = JarvisMCPClient(config)
        register_shell_tools(client, config)

        result = await client.call_tool("exec_command", {"command": "echo Hallo von Shell"})
        assert result.is_error is False
        assert "Hallo von Shell" in result.content

    @pytest.mark.asyncio
    async def test_all_tools_combined(self, config: JarvisConfig) -> None:
        """Alle Tools zusammen registriert und nutzbar."""
        from jarvis.mcp.filesystem import register_fs_tools
        from jarvis.mcp.shell import register_shell_tools

        client = JarvisMCPClient(config)
        register_fs_tools(client, config)
        register_shell_tools(client, config)

        tools = client.get_tool_list()
        assert len(tools) == 5  # read, write, edit, list_dir, exec_command
        assert "read_file" in tools
        assert "exec_command" in tools

        # Beide Typen funktionieren
        r1 = await client.call_tool(
            "write_file",
            {
                "path": str(config.workspace_dir / "combined.txt"),
                "content": "test",
            },
        )
        assert r1.is_error is False

        r2 = await client.call_tool("exec_command", {"command": "echo works"})
        assert r2.is_error is False
        assert "works" in r2.content
