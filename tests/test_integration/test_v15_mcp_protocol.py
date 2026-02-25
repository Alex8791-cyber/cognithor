"""Tests für v15: MCP Protocol Upgrade.

Testet alle neuen Module:
  - MCPServer: JSON-RPC Dispatch, Tool/Resource/Prompt-Handling
  - Resources: Memory, Config, Workspace Resources
  - Prompts: Alle 8 Prompt-Templates
  - Discovery: Agent Card, Health, Skill-Ableitung
  - Bridge: Tool-Konvertierung, Annotations, Setup/Lifecycle

73 Tests, alle ohne externe Abhängigkeiten.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock

import pytest


# ============================================================================
# MCP Server Tests
# ============================================================================


class TestMCPToolDef:
    """Tests für MCPToolDef und Annotations."""

    def test_tool_def_basic(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        tool = MCPToolDef(
            name="test_tool",
            description="Ein Test-Tool",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=lambda **kw: "ok",
        )
        assert tool.name == "test_tool"
        assert tool.description == "Ein Test-Tool"

    def test_tool_def_to_schema(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        tool = MCPToolDef(
            name="read_file",
            description="Liest eine Datei",
            input_schema={"type": "object"},
            handler=lambda **kw: "content",
            annotations={"readOnlyHint": True, "idempotentHint": True},
        )
        schema = tool.to_mcp_schema()
        assert schema["name"] == "read_file"
        assert schema["annotations"]["readOnlyHint"] is True
        assert schema["annotations"]["idempotentHint"] is True

    def test_tool_def_no_annotations(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        tool = MCPToolDef(
            name="test",
            description="",
            input_schema={},
            handler=lambda **kw: "",
        )
        schema = tool.to_mcp_schema()
        assert "annotations" not in schema


class TestMCPResource:
    def test_resource_basic(self) -> None:
        from jarvis.mcp.server import MCPResource

        r = MCPResource(
            uri="jarvis://memory/core",
            name="Core Memory",
            description="Kern-Erinnerungen",
            mime_type="text/markdown",
        )
        schema = r.to_mcp_schema()
        assert schema["uri"] == "jarvis://memory/core"
        assert schema["mimeType"] == "text/markdown"

    def test_resource_template(self) -> None:
        from jarvis.mcp.server import MCPResourceTemplate

        t = MCPResourceTemplate(
            uri_template="jarvis://memory/entity/{entity_id}",
            name="Entity",
        )
        schema = t.to_mcp_schema()
        assert "uriTemplate" in schema
        assert "{entity_id}" in schema["uriTemplate"]


class TestMCPPromptDef:
    def test_prompt_basic(self) -> None:
        from jarvis.mcp.server import MCPPrompt, MCPPromptArgument

        p = MCPPrompt(
            name="summarize",
            description="Zusammenfassung",
            arguments=[
                MCPPromptArgument(name="content", required=True),
                MCPPromptArgument(name="length", required=False),
            ],
        )
        schema = p.to_mcp_schema()
        assert schema["name"] == "summarize"
        assert len(schema["arguments"]) == 2
        assert schema["arguments"][0]["required"] is True


class TestMCPServerConfig:
    def test_default_disabled(self) -> None:
        from jarvis.mcp.server import MCPServerConfig as Cfg, MCPServerMode

        cfg = Cfg()
        assert cfg.mode == MCPServerMode.DISABLED
        assert cfg.http_port == 3001
        assert cfg.require_auth is False

    def test_http_mode(self) -> None:
        from jarvis.mcp.server import MCPServerConfig as Cfg, MCPServerMode

        cfg = Cfg(mode=MCPServerMode.HTTP, http_port=8080)
        assert cfg.mode == MCPServerMode.HTTP
        assert cfg.http_port == 8080


class TestJarvisMCPServer:
    """Tests für den MCP-Server-Kern."""

    def _make_server(self) -> "JarvisMCPServer":
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode

        cfg = MCPServerConfig(mode=MCPServerMode.HTTP)
        return JarvisMCPServer(cfg)

    def test_register_tool(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="test", description="", input_schema={}, handler=lambda **kw: "ok",
        ))
        assert "test" in server._tools

    def test_register_resource(self) -> None:
        from jarvis.mcp.server import MCPResource

        server = self._make_server()
        server.register_resource(MCPResource(
            uri="jarvis://test", name="Test",
        ))
        assert "jarvis://test" in server._resources

    def test_register_prompt(self) -> None:
        from jarvis.mcp.server import MCPPrompt

        server = self._make_server()
        server.register_prompt(MCPPrompt(name="test", description="Test"))
        assert "test" in server._prompts

    @pytest.mark.asyncio
    async def test_handle_initialize(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="t", description="", input_schema={}, handler=lambda **kw: "",
        ))
        result = await server.handle_initialize({})
        assert result["protocolVersion"] == "2025-11-25"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "jarvis"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="read_file", description="Liest Datei",
            input_schema={"type": "object"}, handler=lambda **kw: "content",
        ))
        result = await server.handle_tools_list()
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_handle_tools_call_success(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="greet", description="Begrüßung",
            input_schema={}, handler=lambda **kw: f"Hallo {kw.get('name', 'Welt')}",
        ))
        result = await server.handle_tools_call("greet", {"name": "Alexander"})
        assert result["isError"] is False
        assert "Alexander" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_async_handler(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        async def async_handler(**kw: str) -> str:
            return "async result"

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="async_tool", description="", input_schema={},
            handler=async_handler,
        ))
        result = await server.handle_tools_call("async_tool")
        assert "async result" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_not_found(self) -> None:
        server = self._make_server()
        result = await server.handle_tools_call("nonexistent")
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_tools_call_error(self) -> None:
        from jarvis.mcp.server import MCPToolDef

        def failing_handler(**kw: str) -> str:
            raise ValueError("Boom!")

        server = self._make_server()
        server.register_tool(MCPToolDef(
            name="fail", description="", input_schema={},
            handler=failing_handler,
        ))
        result = await server.handle_tools_call("fail")
        assert result["isError"] is True
        assert "Tool-Fehler" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_resources_list(self) -> None:
        from jarvis.mcp.server import MCPResource

        server = self._make_server()
        server.register_resource(MCPResource(
            uri="jarvis://test", name="Test", mime_type="text/plain",
        ))
        result = await server.handle_resources_list()
        assert len(result["resources"]) == 1

    @pytest.mark.asyncio
    async def test_handle_resources_read(self) -> None:
        from jarvis.mcp.server import MCPResource

        server = self._make_server()
        server.register_resource(MCPResource(
            uri="jarvis://test/data", name="Test",
            handler=lambda **kw: "test-content",
        ))
        result = await server.handle_resources_read("jarvis://test/data")
        assert result["contents"][0]["text"] == "test-content"

    @pytest.mark.asyncio
    async def test_handle_resources_read_not_found(self) -> None:
        server = self._make_server()
        result = await server.handle_resources_read("jarvis://missing")
        assert "nicht gefunden" in result["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_resources_template_match(self) -> None:
        from jarvis.mcp.server import MCPResourceTemplate

        server = self._make_server()
        server.register_resource_template(MCPResourceTemplate(
            uri_template="jarvis://entity/{id}",
            name="Entity",
            handler=lambda **kw: f"entity-data-for-{kw.get('uri', '')}",
        ))
        result = await server.handle_resources_read("jarvis://entity/42")
        assert "entity-data-for-jarvis://entity/42" in result["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_prompts_list(self) -> None:
        from jarvis.mcp.server import MCPPrompt

        server = self._make_server()
        server.register_prompt(MCPPrompt(name="summarize", description="Zusammenfassung"))
        result = await server.handle_prompts_list()
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "summarize"

    @pytest.mark.asyncio
    async def test_handle_prompts_get(self) -> None:
        from jarvis.mcp.server import MCPPrompt

        def handler(**kw: str) -> list:
            return [{"role": "user", "content": {"type": "text", "text": f"Fasse zusammen: {kw.get('text', '')}"}}]

        server = self._make_server()
        server.register_prompt(MCPPrompt(
            name="summarize", description="", handler=handler,
        ))
        result = await server.handle_prompts_get("summarize", {"text": "Test"})
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_handle_prompts_get_not_found(self) -> None:
        server = self._make_server()
        result = await server.handle_prompts_get("nonexistent")
        assert "nicht gefunden" in result["description"]

    @pytest.mark.asyncio
    async def test_dispatch_ping(self) -> None:
        server = self._make_server()
        result = await server.dispatch("ping")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_method(self) -> None:
        server = self._make_server()
        result = await server.dispatch("unknown/method")
        assert "error" in result
        assert result["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_process_jsonrpc_request(self) -> None:
        server = self._make_server()
        response = await server.process_jsonrpc_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
        })
        assert response is not None
        assert response["id"] == 1
        assert "result" in response

    @pytest.mark.asyncio
    async def test_process_jsonrpc_notification(self) -> None:
        server = self._make_server()
        response = await server.process_jsonrpc_message({
            "jsonrpc": "2.0",
            "method": "ping",
            # Kein id → Notification
        })
        assert response is None

    @pytest.mark.asyncio
    async def test_handle_http_request_single(self) -> None:
        server = self._make_server()
        result = await server.handle_http_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
        })
        assert isinstance(result, dict)
        assert result.get("id") == 1

    @pytest.mark.asyncio
    async def test_handle_http_request_batch(self) -> None:
        server = self._make_server()
        result = await server.handle_http_request([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_handle_http_auth_failure(self) -> None:
        from jarvis.mcp.server import MCPServerConfig, MCPServerMode

        cfg = MCPServerConfig(mode=MCPServerMode.HTTP, require_auth=True, auth_token="secret123")
        server = self._make_server()
        server._config = cfg
        result = await server.handle_http_request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            auth_token="wrong",
        )
        assert "error" in result
        assert "Unauthorized" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_http_auth_success(self) -> None:
        from jarvis.mcp.server import MCPServerConfig, MCPServerMode

        cfg = MCPServerConfig(mode=MCPServerMode.HTTP, require_auth=True, auth_token="secret123")
        server = self._make_server()
        server._config = cfg
        result = await server.handle_http_request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            auth_token="secret123",
        )
        assert "error" not in result

    def test_stats(self) -> None:
        from jarvis.mcp.server import MCPToolDef, MCPResource, MCPPrompt

        server = self._make_server()
        server.register_tool(MCPToolDef(name="t", description="", input_schema={}, handler=lambda **kw: ""))
        server.register_resource(MCPResource(uri="jarvis://x", name="X"))
        server.register_prompt(MCPPrompt(name="p"))

        stats = server.stats()
        assert stats["tools_registered"] == 1
        assert stats["resources_registered"] == 1
        assert stats["prompts_registered"] == 1
        assert stats["server_info"]["protocol_version"] == "2025-11-25"

    @pytest.mark.asyncio
    async def test_server_start_disabled(self) -> None:
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig

        server = JarvisMCPServer(MCPServerConfig())  # Default: DISABLED
        await server.start()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_server_lifecycle(self) -> None:
        server = self._make_server()
        await server.start()
        assert server.is_running
        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_handle_logging_set_level(self) -> None:
        server = self._make_server()
        result = await server.handle_logging_set_level("debug")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_handle_resources_subscribe(self) -> None:
        server = self._make_server()
        result = await server.handle_resources_subscribe("jarvis://test")
        assert "error" not in result


# ============================================================================
# Resource Provider Tests
# ============================================================================


class TestJarvisResourceProvider:
    def _make_provider(self) -> "JarvisResourceProvider":
        from jarvis.mcp.resources import JarvisResourceProvider
        return JarvisResourceProvider(config=None, memory=None)

    def test_register_all(self) -> None:
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode
        from jarvis.mcp.resources import JarvisResourceProvider

        server = JarvisMCPServer(MCPServerConfig(mode=MCPServerMode.HTTP))
        provider = JarvisResourceProvider()
        count = provider.register_all(server)
        assert count == 8  # 7 Resources + 1 Template
        assert len(server._resources) == 7
        assert len(server._resource_templates) == 1

    def test_read_core_memory_no_manager(self) -> None:
        provider = self._make_provider()
        result = provider._read_core_memory()
        assert "nicht initialisiert" in result

    def test_read_status(self) -> None:
        provider = self._make_provider()
        result = provider._read_status()
        data = json.loads(result)
        assert data["status"] == "running"
        assert "uptime_seconds" in data

    def test_read_capabilities(self) -> None:
        provider = self._make_provider()
        result = provider._read_capabilities()
        data = json.loads(result)
        assert data["agent_name"] == "Jarvis"
        assert "memory_system" in data["capabilities"]

    def test_read_episodes_no_memory(self) -> None:
        provider = self._make_provider()
        result = provider._read_episodes()
        data = json.loads(result)
        assert "error" in data

    def test_read_memory_stats_no_memory(self) -> None:
        provider = self._make_provider()
        result = provider._read_memory_stats()
        data = json.loads(result)
        assert "error" in data

    def test_stats(self) -> None:
        provider = self._make_provider()
        stats = provider.stats()
        assert stats["memory_available"] is False


# ============================================================================
# Prompt Provider Tests
# ============================================================================


class TestJarvisPromptProvider:
    def _make_provider(self) -> "JarvisPromptProvider":
        from jarvis.mcp.prompts import JarvisPromptProvider
        return JarvisPromptProvider()

    def test_register_all(self) -> None:
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode
        from jarvis.mcp.prompts import JarvisPromptProvider

        server = JarvisMCPServer(MCPServerConfig(mode=MCPServerMode.HTTP))
        provider = JarvisPromptProvider()
        count = provider.register_all(server)
        assert count == 8
        assert len(server._prompts) == 8

    def test_prompt_analyze_document(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_analyze_document(content="Test-Text", focus="risks")
        assert len(msgs) == 1
        assert "Risiken" in msgs[0]["content"]["text"]

    def test_prompt_summarize(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_summarize(content="Langer Text", length="short")
        assert "2-3 Sätzen" in msgs[0]["content"]["text"]

    def test_prompt_insurance_advisor(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_insurance_advisor(product="bu", question="Was kostet BU?")
        text = msgs[0]["content"]["text"]
        assert "Berufsunfähigkeit" in text
        assert "Was kostet BU?" in text

    def test_prompt_code_review(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_code_review(code="print('hello')", language="python")
        assert "python" in msgs[0]["content"]["text"]

    def test_prompt_translate(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_translate(content="Hallo Welt", target_language="en")
        assert "en" in msgs[0]["content"]["text"]

    def test_prompt_brainstorm(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_brainstorm(topic="AI", method="swot")
        assert "SWOT" in msgs[0]["content"]["text"]

    def test_prompt_explain_concept(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_explain_concept(concept="MCP", audience="beginner")
        assert "MCP" in msgs[0]["content"]["text"]
        assert "einfach" in msgs[0]["content"]["text"]

    def test_prompt_daily_briefing(self) -> None:
        provider = self._make_provider()
        msgs = provider._prompt_daily_briefing(focus_areas="tasks")
        assert "Aufgaben" in msgs[0]["content"]["text"]

    def test_stats(self) -> None:
        provider = self._make_provider()
        assert provider.stats()["registered_prompts"] == 0
        # Nach register_all wäre es 8


# ============================================================================
# Discovery Tests
# ============================================================================


class TestAgentCard:
    def test_basic_card(self) -> None:
        from jarvis.mcp.discovery import AgentCard

        card = AgentCard(name="Jarvis", version="15.0.0")
        d = card.to_dict()
        assert d["name"] == "Jarvis"
        assert d["protocolVersion"] == "2025-11-25"
        assert "de" in d["languages"]

    def test_card_with_skills(self) -> None:
        from jarvis.mcp.discovery import AgentCard, AgentSkill

        card = AgentCard(
            skills=[
                AgentSkill(id="web", name="Web-Recherche", tags=["web"]),
            ]
        )
        d = card.to_dict()
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "web"

    def test_card_capabilities(self) -> None:
        from jarvis.mcp.discovery import AgentCard

        card = AgentCard(capabilities=["tools", "resources"])
        d = card.to_dict()
        assert d["capabilities"]["tools"] is True
        assert d["capabilities"]["resources"] is True


class TestDiscoveryManager:
    def test_build_card(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager(host="localhost", port=3001)
        card = dm.build_card(
            tool_names=["read_file", "web_search", "search_memory"],
            resource_count=5,
            prompt_count=3,
            server_mode="http",
        )
        assert card.name == "Jarvis"
        assert "tools" in card.capabilities
        assert "resources" in card.capabilities
        assert len(card.skills) > 0

    def test_build_card_no_tools(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager()
        card = dm.build_card(tool_names=[], resource_count=0, prompt_count=0)
        assert len(card.capabilities) == 0

    def test_get_card_dict(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager()
        dm.build_card(tool_names=["read_file"])
        d = dm.get_card()
        assert isinstance(d, dict)
        assert d["name"] == "Jarvis"

    def test_health(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager()
        h = dm.health()
        assert h["status"] == "healthy"
        assert h["agent"] == "jarvis"
        assert h["mcp_protocol"] == "2025-11-25"

    def test_skill_derivation(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager()
        skills = dm._derive_skills([
            "read_file", "write_file",
            "browse_url", "browse_click",
            "web_search",
            "exec_command",
        ])
        skill_ids = [s.id for s in skills]
        assert "file_management" in skill_ids
        assert "browser_automation" in skill_ids
        assert "web_research" in skill_ids
        assert "code_execution" in skill_ids

    def test_stats(self) -> None:
        from jarvis.mcp.discovery import DiscoveryManager

        dm = DiscoveryManager()
        dm.build_card(tool_names=["read_file"])
        stats = dm.stats()
        assert stats["card_built"] is True
        assert stats["skills"] > 0


# ============================================================================
# Bridge Tests
# ============================================================================


class TestAnnotations:
    def test_read_only_annotation(self) -> None:
        from jarvis.mcp.bridge import _build_annotations

        ann = _build_annotations("read_file")
        assert ann["readOnlyHint"] is True
        assert "destructiveHint" not in ann

    def test_destructive_annotation(self) -> None:
        from jarvis.mcp.bridge import _build_annotations

        ann = _build_annotations("write_file")
        assert ann["destructiveHint"] is True
        assert "readOnlyHint" not in ann

    def test_idempotent_annotation(self) -> None:
        from jarvis.mcp.bridge import _build_annotations

        ann = _build_annotations("web_search")
        assert ann["readOnlyHint"] is True
        assert ann["idempotentHint"] is True

    def test_unknown_tool_no_annotations(self) -> None:
        from jarvis.mcp.bridge import _build_annotations

        ann = _build_annotations("custom_tool")
        assert len(ann) == 0

    def test_browse_tools_annotations(self) -> None:
        from jarvis.mcp.bridge import _build_annotations

        # browse_url ist read-only
        assert _build_annotations("browse_url").get("readOnlyHint") is True
        # browse_fill ist destructive
        assert _build_annotations("browse_fill").get("destructiveHint") is True
        # browse_screenshot ist read-only + idempotent
        ann = _build_annotations("browse_screenshot")
        assert ann.get("readOnlyHint") is True
        assert ann.get("idempotentHint") is True


class TestMCPBridge:
    """Tests für die MCPBridge mit Mock-Objekten."""

    def _make_mock_config(self) -> MagicMock:
        """Erstellt einen Mock JarvisConfig."""
        config = MagicMock()
        config.jarvis_home = MagicMock()
        config.jarvis_home.__truediv__ = MagicMock(return_value=MagicMock())
        config.mcp_config_file = MagicMock()
        config.mcp_config_file.exists.return_value = False
        config.owner_name = "test"
        return config

    def _make_mock_client(self) -> MagicMock:
        """Erstellt einen Mock JarvisMCPClient."""
        client = MagicMock()
        client.get_tool_list.return_value = ["read_file", "web_search"]
        client.get_tool_schemas.return_value = {
            "read_file": {
                "name": "read_file",
                "description": "Liest eine Datei",
                "inputSchema": {"type": "object"},
            },
            "web_search": {
                "name": "web_search",
                "description": "Web-Suche",
                "inputSchema": {"type": "object"},
            },
        }
        client._builtin_handlers = {
            "read_file": lambda params: "file content",
            "web_search": lambda params: "search results",
        }
        return client

    def test_bridge_disabled_by_default(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        result = bridge.setup(self._make_mock_client())
        assert result is False
        assert bridge.enabled is False

    def test_bridge_stats_disabled(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        stats = bridge.stats()
        assert stats["enabled"] is False

    def test_get_agent_card_uninitialized(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        card = bridge.get_agent_card()
        assert "error" in card

    def test_get_health_disabled(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        health = bridge.get_health()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_handle_mcp_request_no_server(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        result = await bridge.handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_start_stop_disabled(self) -> None:
        from jarvis.mcp.bridge import MCPBridge

        config = self._make_mock_config()
        bridge = MCPBridge(config)
        await bridge.start()  # Should not crash
        await bridge.stop()   # Should not crash


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCPIntegration:
    """End-to-End-Tests für den vollständigen MCP-Flow."""

    @pytest.mark.asyncio
    async def test_full_initialize_flow(self) -> None:
        """Simuliert den kompletten Client → Server initialize-Flow."""
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode, MCPToolDef

        server = JarvisMCPServer(MCPServerConfig(mode=MCPServerMode.HTTP))
        server.register_tool(MCPToolDef(
            name="greet", description="Begrüßung",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            handler=lambda **kw: f"Hallo {kw.get('name', 'Welt')}!",
        ))

        # 1. Initialize
        init_response = await server.process_jsonrpc_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
        })
        assert init_response["result"]["protocolVersion"] == "2025-11-25"

        # 2. List Tools
        list_response = await server.process_jsonrpc_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        assert len(list_response["result"]["tools"]) == 1

        # 3. Call Tool
        call_response = await server.process_jsonrpc_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "greet", "arguments": {"name": "Jarvis"}},
        })
        assert "Jarvis" in call_response["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_full_resource_flow(self) -> None:
        """Simuliert Resource-Discovery und -Read."""
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode, MCPResource

        server = JarvisMCPServer(MCPServerConfig(mode=MCPServerMode.HTTP))
        server.register_resource(MCPResource(
            uri="jarvis://test/data", name="Test Data",
            mime_type="application/json",
            handler=lambda **kw: '{"key": "value"}',
        ))

        # List
        list_resp = await server.dispatch("resources/list")
        assert len(list_resp["resources"]) == 1

        # Read
        read_resp = await server.dispatch("resources/read", {"uri": "jarvis://test/data"})
        assert '{"key": "value"}' in read_resp["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_full_prompt_flow(self) -> None:
        """Simuliert Prompt-Discovery und -Get."""
        from jarvis.mcp.server import JarvisMCPServer, MCPServerConfig, MCPServerMode
        from jarvis.mcp.prompts import JarvisPromptProvider

        server = JarvisMCPServer(MCPServerConfig(mode=MCPServerMode.HTTP))
        provider = JarvisPromptProvider()
        provider.register_all(server)

        # List
        list_resp = await server.dispatch("prompts/list")
        assert len(list_resp["prompts"]) == 8

        # Get specific prompt
        get_resp = await server.dispatch("prompts/get", {
            "name": "summarize",
            "arguments": {"content": "Test-Text", "length": "short"},
        })
        assert len(get_resp["messages"]) > 0
