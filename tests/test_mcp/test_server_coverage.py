"""Coverage-Tests fuer server.py -- fehlende Pfade abdecken.

Schwerpunkt: JarvisMCPServer dispatch, handle_tools_call (success, timeout, error),
handle_resources_read (template matching, handler errors), handle_prompts_get,
handle_resources_subscribe, handle_logging_set_level, process_jsonrpc_message,
handle_http_request (auth, batch, single), _match_template, _send_progress,
stats, start/stop lifecycle, MCPToolDef/MCPResource/MCPResourceTemplate/MCPPrompt
to_mcp_schema, MCPServerConfig defaults.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.mcp.server import (
    PROTOCOL_VERSION,
    JarvisMCPServer,
    MCPLogEntry,
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPResourceTemplate,
    MCPServerConfig,
    MCPServerMode,
    MCPToolDef,
    ProgressNotification,
    ToolAnnotationKey,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server() -> JarvisMCPServer:
    return JarvisMCPServer()


@pytest.fixture
def configured_server() -> JarvisMCPServer:
    config = MCPServerConfig(
        mode=MCPServerMode.HTTP,
        server_name="test-jarvis",
        server_version="0.1.0",
        expose_tools=True,
        expose_resources=True,
        expose_prompts=True,
        enable_sampling=True,
        enable_logging=True,
    )
    return JarvisMCPServer(config=config)


# ============================================================================
# Dataclass schemas
# ============================================================================


class TestMCPToolDef:
    def test_to_mcp_schema_basic(self) -> None:
        tool = MCPToolDef(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: None,
        )
        schema = tool.to_mcp_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "inputSchema" in schema
        assert "annotations" not in schema

    def test_to_mcp_schema_with_annotations(self) -> None:
        tool = MCPToolDef(
            name="tool_with_annot",
            description="Annotated",
            input_schema={"type": "object"},
            handler=lambda: None,
            annotations={"readOnlyHint": True, "destructiveHint": False},
        )
        schema = tool.to_mcp_schema()
        assert schema["annotations"]["readOnlyHint"] is True


class TestMCPResource:
    def test_to_mcp_schema(self) -> None:
        res = MCPResource(
            uri="jarvis://memory/core",
            name="Core Memory",
            description="Core memory contents",
            mime_type="application/json",
        )
        schema = res.to_mcp_schema()
        assert schema["uri"] == "jarvis://memory/core"
        assert schema["name"] == "Core Memory"
        assert schema["mimeType"] == "application/json"


class TestMCPResourceTemplate:
    def test_to_mcp_schema(self) -> None:
        tmpl = MCPResourceTemplate(
            uri_template="jarvis://memory/entity/{entity_id}",
            name="Entity Memory",
            description="Entity details",
        )
        schema = tmpl.to_mcp_schema()
        assert schema["uriTemplate"] == "jarvis://memory/entity/{entity_id}"


class TestMCPPrompt:
    def test_to_mcp_schema_no_args(self) -> None:
        p = MCPPrompt(name="simple", description="Simple prompt")
        schema = p.to_mcp_schema()
        assert schema["name"] == "simple"
        assert "arguments" not in schema

    def test_to_mcp_schema_with_args(self) -> None:
        p = MCPPrompt(
            name="with_args",
            description="Prompt with args",
            arguments=[
                MCPPromptArgument(name="topic", description="The topic", required=True),
                MCPPromptArgument(name="lang", description="Language"),
            ],
        )
        schema = p.to_mcp_schema()
        assert len(schema["arguments"]) == 2
        assert schema["arguments"][0]["name"] == "topic"
        assert schema["arguments"][0]["required"] is True


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        config = MCPServerConfig()
        assert config.mode == MCPServerMode.DISABLED
        assert config.http_host == "127.0.0.1"
        assert config.http_port == 3001
        assert config.require_auth is False


class TestToolAnnotationKey:
    def test_values(self) -> None:
        assert ToolAnnotationKey.READ_ONLY_HINT == "readOnlyHint"
        assert ToolAnnotationKey.DESTRUCTIVE_HINT == "destructiveHint"


class TestProgressNotification:
    def test_creation(self) -> None:
        pn = ProgressNotification(progress_token="t1", progress=0.5, message="Half done")
        assert pn.progress == 0.5
        assert pn.message == "Half done"


class TestMCPLogEntry:
    def test_defaults(self) -> None:
        entry = MCPLogEntry()
        assert entry.level == "info"
        assert entry.logger == "jarvis"


# ============================================================================
# Registration API
# ============================================================================


class TestRegistration:
    def test_register_tool(self, server: JarvisMCPServer) -> None:
        tool = MCPToolDef(
            name="my_tool",
            description="Desc",
            input_schema={},
            handler=lambda: None,
        )
        server.register_tool(tool)
        assert "my_tool" in server._tools

    def test_register_resource(self, server: JarvisMCPServer) -> None:
        res = MCPResource(uri="jarvis://test", name="Test")
        server.register_resource(res)
        assert "jarvis://test" in server._resources

    def test_register_resource_template(self, server: JarvisMCPServer) -> None:
        tmpl = MCPResourceTemplate(
            uri_template="jarvis://entity/{id}",
            name="Entity",
        )
        server.register_resource_template(tmpl)
        assert "jarvis://entity/{id}" in server._resource_templates

    def test_register_prompt(self, server: JarvisMCPServer) -> None:
        prompt = MCPPrompt(name="my_prompt", description="Desc")
        server.register_prompt(prompt)
        assert "my_prompt" in server._prompts


# ============================================================================
# TRUST-7: register_tool fingerprints the handler source
# ============================================================================


class TestRegisterToolFingerprint:
    """``register_tool`` calls ``FINGERPRINT_LEDGER.register`` on a
    best-effort basis, so the TRUST-7 ledger captures every tool's
    handler source for post-mortem reconstruction.
    """

    def test_register_tool_fingerprints_handler(self, server: JarvisMCPServer) -> None:
        # The hook imports FINGERPRINT_LEDGER lazily inside
        # _fingerprint_tool_handler, so monkey-patching the source
        # module is the right isolation point.
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import FingerprintLedger

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:

            def _my_handler() -> str:
                return "ok"

            tool = MCPToolDef(
                name="probe_tool",
                description="d",
                input_schema={},
                handler=_my_handler,
            )
            server.register_tool(tool)
            history = isolated.history("probe_tool")
            assert len(history) == 1
            fp = history[0]
            assert fp.name == "probe_tool"
            assert fp.kind.value == "tool"
            # source_path should point at this test file (where the handler is defined).
            assert fp.source_path.endswith("test_server_coverage.py")
            assert len(fp.content_hash) == 64
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]

    def test_register_tool_skips_unsourceable_handler(self, server: JarvisMCPServer) -> None:
        # Builtins have no resolvable source file → the fingerprint
        # branch silently no-ops, but registration still succeeds.
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import FingerprintLedger

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:
            tool = MCPToolDef(
                name="builtin_probe",
                description="d",
                input_schema={},
                handler=print,  # builtin
            )
            # Must not raise.
            server.register_tool(tool)
            assert "builtin_probe" in server._tools
            # No fingerprint registered for the builtin.
            assert isolated.history("builtin_probe") == ()
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]


# ============================================================================
# handle_initialize
# ============================================================================


class TestHandleInitialize:
    @pytest.mark.asyncio
    async def test_basic_init(self, server: JarvisMCPServer) -> None:
        result = await server.handle_initialize({})
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert "serverInfo" in result

    @pytest.mark.asyncio
    async def test_init_with_tools(self, configured_server: JarvisMCPServer) -> None:
        configured_server.register_tool(
            MCPToolDef(name="t", description="d", input_schema={}, handler=lambda: None)
        )
        result = await configured_server.handle_initialize({})
        assert "tools" in result["capabilities"]
        assert "sampling" in result["capabilities"]
        assert "logging" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_init_with_resources(self, configured_server: JarvisMCPServer) -> None:
        configured_server.register_resource(MCPResource(uri="jarvis://test", name="Test"))
        result = await configured_server.handle_initialize({})
        assert "resources" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_init_with_prompts(self, configured_server: JarvisMCPServer) -> None:
        configured_server.register_prompt(MCPPrompt(name="p", description="d"))
        result = await configured_server.handle_initialize({})
        assert "prompts" in result["capabilities"]


# ============================================================================
# handle_tools_list / handle_tools_call
# ============================================================================


class TestHandleTools:
    @pytest.mark.asyncio
    async def test_tools_list_empty(self, server: JarvisMCPServer) -> None:
        result = await server.handle_tools_list()
        assert result["tools"] == []

    @pytest.mark.asyncio
    async def test_tools_list_with_tools(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(name="t1", description="d1", input_schema={}, handler=lambda: None)
        )
        result = await server.handle_tools_list()
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "t1"

    @pytest.mark.asyncio
    async def test_tools_call_not_found(self, server: JarvisMCPServer) -> None:
        result = await server.handle_tools_call("nonexistent")
        assert result["isError"] is True
        assert (
            "not_found" in result["content"][0]["text"]
            or "nicht gefunden" in result["content"][0]["text"]
        )

    @pytest.mark.asyncio
    async def test_tools_call_sync_handler(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="sync_tool",
                description="Sync",
                input_schema={},
                handler=lambda: "sync result",
            )
        )
        result = await server.handle_tools_call("sync_tool")
        assert result["isError"] is False
        assert result["content"][0]["text"] == "sync result"

    @pytest.mark.asyncio
    async def test_tools_call_async_handler(self, server: JarvisMCPServer) -> None:
        async def async_handler(msg: str = "default") -> str:
            return f"async: {msg}"

        server.register_tool(
            MCPToolDef(
                name="async_tool",
                description="Async",
                input_schema={},
                handler=async_handler,
            )
        )
        result = await server.handle_tools_call("async_tool", arguments={"msg": "hello"})
        assert result["isError"] is False
        assert "async: hello" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_dict_result(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="dict_tool",
                description="Dict",
                input_schema={},
                handler=lambda: {"key": "value"},
            )
        )
        result = await server.handle_tools_call("dict_tool")
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_tools_call_list_result(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="list_tool",
                description="List",
                input_schema={},
                handler=lambda: [{"type": "text", "text": "item"}],
            )
        )
        result = await server.handle_tools_call("list_tool")
        assert result["isError"] is False
        assert result["content"][0]["text"] == "item"

    @pytest.mark.asyncio
    async def test_tools_call_exception(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="error_tool",
                description="Error",
                input_schema={},
                handler=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        )

        # Use a handler that actually raises
        def bad_handler():
            raise RuntimeError("boom")

        server._tools["error_tool"].handler = bad_handler
        result = await server.handle_tools_call("error_tool")
        assert result["isError"] is True
        assert "Tool-Fehler" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_timeout(self, server: JarvisMCPServer) -> None:
        async def slow_handler():
            await asyncio.sleep(100)

        server.register_tool(
            MCPToolDef(
                name="slow_tool",
                description="Slow",
                input_schema={},
                handler=slow_handler,
            )
        )
        server.HANDLER_TIMEOUT = 0.01  # Very short timeout
        result = await server.handle_tools_call("slow_tool")
        assert result["isError"] is True
        assert "Timeout" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_with_progress_token(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="progress_tool",
                description="With progress",
                input_schema={},
                handler=lambda: "done",
            )
        )
        result = await server.handle_tools_call("progress_tool", progress_token="tok1")
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_tools_call_other_result_type(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="int_tool",
                description="Returns int",
                input_schema={},
                handler=lambda: 42,
            )
        )
        result = await server.handle_tools_call("int_tool")
        assert result["isError"] is False
        assert "42" in result["content"][0]["text"]


# ============================================================================
# handle_resources_list / handle_resources_read
# ============================================================================


class TestHandleResources:
    @pytest.mark.asyncio
    async def test_resources_list(self, server: JarvisMCPServer) -> None:
        server.register_resource(MCPResource(uri="jarvis://test", name="Test"))
        result = await server.handle_resources_list()
        assert len(result["resources"]) == 1

    @pytest.mark.asyncio
    async def test_resources_templates_list(self, server: JarvisMCPServer) -> None:
        server.register_resource_template(
            MCPResourceTemplate(uri_template="jarvis://entity/{id}", name="Entity")
        )
        result = await server.handle_resources_templates_list()
        assert len(result["resourceTemplates"]) == 1

    @pytest.mark.asyncio
    async def test_resources_read_found(self, server: JarvisMCPServer) -> None:
        server.register_resource(
            MCPResource(
                uri="jarvis://test",
                name="Test",
                handler=lambda uri: "test content",
            )
        )
        result = await server.handle_resources_read("jarvis://test")
        assert result["contents"][0]["text"] == "test content"

    @pytest.mark.asyncio
    async def test_resources_read_async_handler(self, server: JarvisMCPServer) -> None:
        async def async_handler(uri: str) -> str:
            return f"async content for {uri}"

        server.register_resource(
            MCPResource(
                uri="jarvis://async",
                name="Async",
                handler=async_handler,
            )
        )
        result = await server.handle_resources_read("jarvis://async")
        assert "async content" in result["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_resources_read_no_handler(self, server: JarvisMCPServer) -> None:
        server.register_resource(
            MCPResource(
                uri="jarvis://no_handler",
                name="No Handler",
            )
        )
        result = await server.handle_resources_read("jarvis://no_handler")
        assert result["contents"][0]["text"] == ""

    @pytest.mark.asyncio
    async def test_resources_read_not_found(self, server: JarvisMCPServer) -> None:
        result = await server.handle_resources_read("jarvis://nonexistent")
        assert (
            "not_found" in result["contents"][0]["text"]
            or "nicht gefunden" in result["contents"][0]["text"]
        )

    @pytest.mark.asyncio
    async def test_resources_read_template_match(self, server: JarvisMCPServer) -> None:
        server.register_resource_template(
            MCPResourceTemplate(
                uri_template="jarvis://entity/{entity_id}",
                name="Entity",
                handler=lambda uri: f"entity data for {uri}",
            )
        )
        result = await server.handle_resources_read("jarvis://entity/123")
        assert "entity data" in result["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_resources_read_handler_exception(self, server: JarvisMCPServer) -> None:
        server.register_resource(
            MCPResource(
                uri="jarvis://error",
                name="Error Resource",
                handler=lambda uri: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        )

        def bad_handler(uri):
            raise RuntimeError("fail")

        server._resources["jarvis://error"].handler = bad_handler
        result = await server.handle_resources_read("jarvis://error")
        assert "Lesefehler" in result["contents"][0]["text"]

    @pytest.mark.asyncio
    async def test_resources_read_timeout(self, server: JarvisMCPServer) -> None:
        async def slow_handler(uri: str):
            await asyncio.sleep(100)

        server.register_resource(
            MCPResource(
                uri="jarvis://slow",
                name="Slow",
                handler=slow_handler,
            )
        )
        server.HANDLER_TIMEOUT = 0.01
        result = await server.handle_resources_read("jarvis://slow")
        assert "Timeout" in result["contents"][0]["text"]


# ============================================================================
# handle_resources_subscribe
# ============================================================================


class TestHandleSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_new(self, server: JarvisMCPServer) -> None:
        result = await server.handle_resources_subscribe("jarvis://test")
        assert result == {}
        assert "jarvis://test" in server._subscribers

    @pytest.mark.asyncio
    async def test_subscribe_limit(self, server: JarvisMCPServer) -> None:
        server._subscribers["jarvis://test"] = [MagicMock()] * server.MAX_SUBSCRIBERS_PER_URI
        result = await server.handle_resources_subscribe("jarvis://test")
        assert "error" in result


# ============================================================================
# notify_subscribers
# ============================================================================


class TestNotifySubscribers:
    @pytest.mark.asyncio
    async def test_notify_sync_callback(self, server: JarvisMCPServer) -> None:
        callback = MagicMock()
        server._subscribers["jarvis://test"] = [callback]
        await server.notify_subscribers("jarvis://test")
        callback.assert_called_once_with("jarvis://test")

    @pytest.mark.asyncio
    async def test_notify_async_callback(self, server: JarvisMCPServer) -> None:
        callback = AsyncMock()
        server._subscribers["jarvis://test"] = [callback]
        await server.notify_subscribers("jarvis://test")
        callback.assert_called_once_with("jarvis://test")

    @pytest.mark.asyncio
    async def test_notify_callback_exception(self, server: JarvisMCPServer) -> None:
        callback = MagicMock(side_effect=RuntimeError("fail"))
        server._subscribers["jarvis://test"] = [callback]
        # Should not raise
        await server.notify_subscribers("jarvis://test")

    @pytest.mark.asyncio
    async def test_notify_no_subscribers(self, server: JarvisMCPServer) -> None:
        await server.notify_subscribers("jarvis://none")  # should not raise


# ============================================================================
# handle_prompts_list / handle_prompts_get
# ============================================================================


class TestHandlePrompts:
    @pytest.mark.asyncio
    async def test_prompts_list(self, server: JarvisMCPServer) -> None:
        server.register_prompt(MCPPrompt(name="p1", description="d1"))
        result = await server.handle_prompts_list()
        assert len(result["prompts"]) == 1

    @pytest.mark.asyncio
    async def test_prompts_get_not_found(self, server: JarvisMCPServer) -> None:
        result = await server.handle_prompts_get("nonexistent")
        assert "not_found" in result["description"] or "nicht gefunden" in result["description"]

    @pytest.mark.asyncio
    async def test_prompts_get_sync_handler(self, server: JarvisMCPServer) -> None:
        server.register_prompt(
            MCPPrompt(
                name="sync_prompt",
                description="Sync prompt",
                handler=lambda topic="default": [
                    {"role": "user", "content": {"type": "text", "text": topic}}
                ],
            )
        )
        result = await server.handle_prompts_get("sync_prompt", arguments={"topic": "AI"})
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_prompts_get_async_handler(self, server: JarvisMCPServer) -> None:
        async def async_prompt(topic: str = "default") -> list:
            return [{"role": "user", "content": {"type": "text", "text": topic}}]

        server.register_prompt(
            MCPPrompt(
                name="async_prompt",
                description="Async prompt",
                handler=async_prompt,
            )
        )
        result = await server.handle_prompts_get("async_prompt", arguments={"topic": "AI"})
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_prompts_get_no_handler(self, server: JarvisMCPServer) -> None:
        server.register_prompt(MCPPrompt(name="no_handler", description="No handler"))
        result = await server.handle_prompts_get("no_handler")
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_prompts_get_exception(self, server: JarvisMCPServer) -> None:
        def bad_handler(**kwargs):
            raise RuntimeError("prompt error")

        server.register_prompt(
            MCPPrompt(
                name="error_prompt",
                description="Error",
                handler=bad_handler,
            )
        )
        result = await server.handle_prompts_get("error_prompt")
        assert "Verarbeitungsfehler" in result["description"]

    @pytest.mark.asyncio
    async def test_prompts_get_timeout(self, server: JarvisMCPServer) -> None:
        async def slow_prompt(**kwargs):
            await asyncio.sleep(100)

        server.register_prompt(
            MCPPrompt(
                name="slow_prompt",
                description="Slow",
                handler=slow_prompt,
            )
        )
        server.HANDLER_TIMEOUT = 0.01
        result = await server.handle_prompts_get("slow_prompt")
        assert "Timeout" in result["description"]

    @pytest.mark.asyncio
    async def test_prompts_get_single_message_wrapped(self, server: JarvisMCPServer) -> None:
        """Handler returns single dict instead of list -> should be wrapped."""
        server.register_prompt(
            MCPPrompt(
                name="single_prompt",
                description="Single",
                handler=lambda: {"role": "user", "content": "single"},
            )
        )
        result = await server.handle_prompts_get("single_prompt")
        assert len(result["messages"]) == 1


# ============================================================================
# handle_logging_set_level
# ============================================================================


class TestHandleLogging:
    @pytest.mark.asyncio
    async def test_set_level(self, server: JarvisMCPServer) -> None:
        result = await server.handle_logging_set_level("debug")
        assert result == {}


# ============================================================================
# dispatch
# ============================================================================


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_method(self, server: JarvisMCPServer) -> None:
        result = await server.dispatch("unknown/method")
        assert "error" in result
        assert result["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_dispatch_initialize(self, server: JarvisMCPServer) -> None:
        result = await server.dispatch("initialize")
        assert "protocolVersion" in result

    @pytest.mark.asyncio
    async def test_dispatch_ping(self, server: JarvisMCPServer) -> None:
        result = await server.dispatch("ping")
        assert result == {}

    @pytest.mark.asyncio
    async def test_dispatch_tools_list(self, server: JarvisMCPServer) -> None:
        result = await server.dispatch("tools/list")
        assert "tools" in result

    @pytest.mark.asyncio
    async def test_dispatch_tools_call(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="dispatch_tool",
                description="D",
                input_schema={},
                handler=lambda: "ok",
            )
        )
        result = await server.dispatch("tools/call", {"name": "dispatch_tool"})
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception(self, server: JarvisMCPServer) -> None:
        # Force an exception in a handler via patching
        with patch.object(server, "handle_tools_list", side_effect=RuntimeError("crash")):
            result = await server.dispatch("tools/list")
            assert "error" in result
            assert result["error"]["code"] == -32603


# ============================================================================
# process_jsonrpc_message
# ============================================================================


class TestProcessJsonrpcMessage:
    @pytest.mark.asyncio
    async def test_request_with_id(self, server: JarvisMCPServer) -> None:
        response = await server.process_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            }
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

    @pytest.mark.asyncio
    async def test_notification_no_id(self, server: JarvisMCPServer) -> None:
        response = await server.process_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "method": "ping",
            }
        )
        assert response is None

    @pytest.mark.asyncio
    async def test_error_response(self, server: JarvisMCPServer) -> None:
        response = await server.process_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "unknown/method",
            }
        )
        assert "error" in response
        assert response["id"] == 2


# ============================================================================
# handle_http_request
# ============================================================================


class TestHandleHttpRequest:
    @pytest.mark.asyncio
    async def test_single_request(self, server: JarvisMCPServer) -> None:
        result = await server.handle_http_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            }
        )
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_batch_request(self, server: JarvisMCPServer) -> None:
        result = await server.handle_http_request(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ]
        )
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_batch_too_large(self, server: JarvisMCPServer) -> None:
        big_batch = [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(100)]
        result = await server.handle_http_request(big_batch)
        assert "error" in result
        assert "Batch zu gross" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_auth_required_success(self) -> None:
        config = MCPServerConfig(require_auth=True, auth_token="secret123")
        server = JarvisMCPServer(config)
        result = await server.handle_http_request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            auth_token="secret123",
        )
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_auth_required_failure(self) -> None:
        config = MCPServerConfig(require_auth=True, auth_token="secret123")
        server = JarvisMCPServer(config)
        result = await server.handle_http_request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            auth_token="wrong",
        )
        assert "error" in result
        assert "Unauthorized" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_auth_no_token(self) -> None:
        config = MCPServerConfig(require_auth=True, auth_token="secret123")
        server = JarvisMCPServer(config)
        result = await server.handle_http_request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_notification_returns_empty(self, server: JarvisMCPServer) -> None:
        result = await server.handle_http_request(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                # no id -> notification
            }
        )
        # Notification returns None from process_jsonrpc_message,
        # http handler wraps it as {"jsonrpc": "2.0", "id": None, "result": {}}
        assert result.get("result") == {} or result == {}

    @pytest.mark.asyncio
    async def test_batch_with_notifications(self, server: JarvisMCPServer) -> None:
        result = await server.handle_http_request(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "method": "ping"},  # notification, no response
            ]
        )
        assert isinstance(result, list)
        assert len(result) == 1  # only the one with id gets a response


# ============================================================================
# _match_template
# ============================================================================


class TestMatchTemplate:
    def test_match_single_param(self, server: JarvisMCPServer) -> None:
        server.register_resource_template(
            MCPResourceTemplate(
                uri_template="jarvis://entity/{id}",
                name="Entity",
                handler=lambda uri: "data",
            )
        )
        result = server._match_template("jarvis://entity/123")
        assert result is not None
        assert result.name == "Entity"

    def test_no_match_different_length(self, server: JarvisMCPServer) -> None:
        server.register_resource_template(
            MCPResourceTemplate(
                uri_template="jarvis://entity/{id}",
                name="Entity",
            )
        )
        result = server._match_template("jarvis://entity/123/extra")
        assert result is None

    def test_no_match_different_prefix(self, server: JarvisMCPServer) -> None:
        server.register_resource_template(
            MCPResourceTemplate(
                uri_template="jarvis://entity/{id}",
                name="Entity",
            )
        )
        result = server._match_template("jarvis://other/123")
        assert result is None

    def test_no_templates(self, server: JarvisMCPServer) -> None:
        result = server._match_template("jarvis://anything")
        assert result is None


# ============================================================================
# _send_progress
# ============================================================================


class TestSendProgress:
    @pytest.mark.asyncio
    async def test_send_progress_no_handler(self, server: JarvisMCPServer) -> None:
        await server._send_progress("no_token", 0.5)  # should not raise

    @pytest.mark.asyncio
    async def test_send_progress_sync_handler(self, server: JarvisMCPServer) -> None:
        handler = MagicMock()
        server._progress_handlers["t1"] = handler
        await server._send_progress("t1", 0.5, "msg")
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_progress_async_handler(self, server: JarvisMCPServer) -> None:
        handler = AsyncMock()
        server._progress_handlers["t2"] = handler
        await server._send_progress("t2", 1.0, "done")
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_progress_handler_exception(self, server: JarvisMCPServer) -> None:
        handler = MagicMock(side_effect=RuntimeError("fail"))
        server._progress_handlers["t3"] = handler
        await server._send_progress("t3", 0.5)  # should not raise


# ============================================================================
# Lifecycle
# ============================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_disabled(self, server: JarvisMCPServer) -> None:
        await server.start()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_start_http(self) -> None:
        config = MCPServerConfig(mode=MCPServerMode.HTTP)
        server = JarvisMCPServer(config)
        await server.start()
        assert server.is_running
        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_is_running_property(self, server: JarvisMCPServer) -> None:
        assert server.is_running is False

    def test_handle_task_exception_cancelled(self, server: JarvisMCPServer) -> None:
        task = MagicMock()
        task.cancelled.return_value = True
        server._handle_task_exception(task)  # should not raise

    def test_handle_task_exception_with_error(self, server: JarvisMCPServer) -> None:
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("test error")
        server._handle_task_exception(task)  # should not raise, just log

    def test_handle_task_exception_no_error(self, server: JarvisMCPServer) -> None:
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        server._handle_task_exception(task)  # should not raise


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    def test_stats_initial(self, server: JarvisMCPServer) -> None:
        stats = server.stats()
        assert stats["mode"] == "disabled"
        assert stats["running"] is False
        assert stats["total_requests"] == 0
        assert stats["tools_registered"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_activity(self, server: JarvisMCPServer) -> None:
        server.register_tool(
            MCPToolDef(
                name="t",
                description="d",
                input_schema={},
                handler=lambda: None,
            )
        )
        server.register_resource(MCPResource(uri="jarvis://r", name="R"))
        await server.handle_tools_list()
        stats = server.stats()
        assert stats["tools_registered"] == 1
        assert stats["resources_registered"] == 1
        assert stats["total_requests"] == 1
