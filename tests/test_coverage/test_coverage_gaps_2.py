"""Zusätzliche Coverage-Tests für model_router (68→85%+) und mcp/client (53→75%+).

Testet die bisher unabgedeckten Pfade:
- OllamaClient: chat(), embed(), embed_batch(), chat_stream(), Fehlerbehandlung
- ModelRouter: get_model_config(), _find_fallback() Randfälle
- JarvisMCPClient: call_tool (MCP-Server-Pfad), connect_all, disconnect_all, _load_server_configs
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jarvis.config import JarvisConfig

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# OllamaClient – chat(), embed(), Fehlerbehandlung
# ============================================================================


class TestOllamaClientChat:
    """Testet OllamaClient.chat() und verwandte Methoden."""

    @pytest.fixture()
    def client(self):
        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        c = OllamaClient(config)
        # Pre-inject mock http client
        mock_http = AsyncMock()
        mock_http.is_closed = False
        c._client = mock_http
        return c, mock_http

    @pytest.mark.asyncio()
    async def test_chat_success(self, client):
        """Erfolgreicher Chat-Call mit Response-Parsing."""
        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hallo!"},
            "eval_count": 42,
            "done": True,
        }
        mock_http.post.return_value = mock_resp

        result = await c.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
        )
        assert result["message"]["content"] == "Hallo!"
        mock_http.post.assert_awaited_once()
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "/api/chat"

    @pytest.mark.asyncio()
    async def test_chat_with_tools(self, client):
        """Chat mit Tool-Schemas sendet tools im Payload."""
        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file"}}],
            },
        }
        mock_http.post.return_value = mock_resp

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        result = await c.chat(
            "qwen3:32b",
            [{"role": "user", "content": "Lies die Datei"}],
            tools=tools,
        )
        payload = mock_http.post.call_args[1]["json"]
        assert "tools" in payload
        assert result["message"]["tool_calls"] is not None

    @pytest.mark.asyncio()
    async def test_chat_with_format_json(self, client):
        """Chat mit format_json=True setzt Format-Parameter."""
        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": '{"key": "value"}'}}
        mock_http.post.return_value = mock_resp

        await c.chat(
            "qwen3:8b",
            [{"role": "user", "content": "JSON bitte"}],
            format_json=True,
        )
        payload = mock_http.post.call_args[1]["json"]
        assert payload["format"] == "json"

    @pytest.mark.asyncio()
    async def test_chat_http_error(self, client):
        """HTTP-Fehler wirft OllamaError mit Status-Code."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_http.post.return_value = mock_resp

        with pytest.raises(OllamaError, match="HTTP 500"):
            await c.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio()
    async def test_chat_timeout(self, client):
        """Timeout wirft OllamaError."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client
        mock_http.post.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(OllamaError, match="Timeout"):
            await c.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio()
    async def test_chat_connect_error(self, client):
        """Connection-Fehler wirft OllamaError."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client
        mock_http.post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(OllamaError, match="nicht erreichbar"):
            await c.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])


class TestOllamaClientEmbed:
    """Testet OllamaClient.embed() und embed_batch()."""

    @pytest.fixture()
    def client(self):
        from jarvis.core.model_router import OllamaClient

        config = JarvisConfig()
        c = OllamaClient(config)
        mock_http = AsyncMock()
        mock_http.is_closed = False
        c._client = mock_http
        return c, mock_http

    @pytest.mark.asyncio()
    async def test_embed_success(self, client):
        """Einzelnes Embedding wird korrekt zurückgegeben."""
        c, mock_http = client

        fake_vec = [0.1] * 768
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [fake_vec]}
        mock_http.post.return_value = mock_resp

        result = await c.embed("nomic-embed-text", "Projektmanagement")
        assert len(result) == 768
        assert result == fake_vec

    @pytest.mark.asyncio()
    async def test_embed_empty_response(self, client):
        """Leere Embedding-Antwort wirft OllamaError."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": []}
        mock_http.post.return_value = mock_resp

        with pytest.raises(OllamaError, match="Keine Embeddings"):
            await c.embed("nomic-embed-text", "test")

    @pytest.mark.asyncio()
    async def test_embed_http_error(self, client):
        """HTTP-Fehler bei Embedding."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_http.post.return_value = mock_resp

        with pytest.raises(OllamaError, match="fehlgeschlagen"):
            await c.embed("nomic-embed-text", "test")

    @pytest.mark.asyncio()
    async def test_embed_timeout(self, client):
        """Embedding-Timeout."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client
        mock_http.post.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(OllamaError, match="Timeout"):
            await c.embed("nomic-embed-text", "test")

    @pytest.mark.asyncio()
    async def test_embed_batch_success(self, client):
        """Batch-Embedding mit mehreren Texten."""
        c, mock_http = client

        vecs = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": vecs}
        mock_http.post.return_value = mock_resp

        result = await c.embed_batch("nomic-embed-text", ["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == vecs[0]

    @pytest.mark.asyncio()
    async def test_embed_batch_http_error(self, client):
        """Batch-Embedding HTTP-Fehler."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_http.post.return_value = mock_resp

        with pytest.raises(OllamaError, match="Batch-Embedding"):
            await c.embed_batch("nomic-embed-text", ["a", "b"])

    @pytest.mark.asyncio()
    async def test_embed_batch_timeout(self, client):
        """Batch-Embedding Timeout."""
        from jarvis.core.model_router import OllamaError

        c, mock_http = client
        mock_http.post.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(OllamaError, match="Timeout"):
            await c.embed_batch("nomic-embed-text", ["a"])


class TestOllamaError:
    """Testet OllamaError-Klasse."""

    def test_error_with_status_code(self):
        from jarvis.core.model_router import OllamaError

        err = OllamaError("Test error", status_code=503)
        assert err.status_code == 503
        assert "Test error" in str(err)

    def test_error_without_status_code(self):
        from jarvis.core.model_router import OllamaError

        err = OllamaError("Connection lost")
        assert err.status_code is None


class TestModelRouterExtended:
    """Erweiterte Tests für ModelRouter – get_model_config und Fallback-Edges."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        return JarvisConfig(jarvis_home=tmp_path)

    @pytest.fixture()
    def router(self, config):
        from jarvis.core.model_router import ModelRouter

        mock_ollama = MagicMock()
        return ModelRouter(config, mock_ollama)

    def test_get_model_config_known(self, router, config):
        """get_model_config gibt Konfiguration für bekanntes Modell."""
        result = router.get_model_config(config.models.planner.name)
        assert "temperature" in result
        assert "top_p" in result
        assert "context_window" in result
        assert result["context_window"] == config.models.planner.context_window

    def test_get_model_config_coder(self, router, config):
        """get_model_config für Coder-Modell."""
        result = router.get_model_config(config.models.coder.name)
        assert result["context_window"] == config.models.coder.context_window

    def test_get_model_config_unknown(self, router):
        """get_model_config gibt Defaults für unbekanntes Modell."""
        result = router.get_model_config("unknown-model:7b")
        assert result["temperature"] == 0.7
        assert result["context_window"] == 32768

    def test_find_fallback_to_any_non_embedding(self, router):
        """Fallback wählt irgendein Nicht-Embedding-Modell."""
        router._available_models = {"nomic-embed-text", "llama3:8b"}
        fallback = router._find_fallback("nonexistent:32b")
        assert fallback == "llama3:8b"

    def test_find_fallback_only_embedding_available(self, router):
        """Kein Fallback wenn nur Embedding-Modelle da sind."""
        router._available_models = {"nomic-embed-text"}
        fallback = router._find_fallback("nonexistent:32b")
        # nomic-embed-text hat "embed" → sollte None zurückgeben
        assert fallback is None

    def test_find_fallback_empty_models(self, router):
        """Kein Fallback bei leerer Modellliste."""
        router._available_models = set()
        fallback = router._find_fallback("qwen3:32b")
        assert fallback is None

    def test_select_model_no_available_models(self, router, config):
        """Kein Fallback wenn _available_models leer (Erststart)."""
        router._available_models = set()
        model = router.select_model("planning")
        # Leeres Set → kein Fallback, gibt gewünschtes Modell zurück
        assert model == config.models.planner.name


class TestMessagesToOllama:
    """Testet die messages_to_ollama Konvertierung."""

    def test_convert_messages(self):
        from jarvis.core.model_router import messages_to_ollama
        from jarvis.models import Message, MessageRole

        messages = [
            Message(role=MessageRole.USER, content="Hallo", channel="cli"),
            Message(role=MessageRole.ASSISTANT, content="Hi!", channel="cli"),
        ]
        result = messages_to_ollama(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hallo"
        assert result[1]["role"] == "assistant"


# ============================================================================
# JarvisMCPClient – Server-Pfade, connect_all, disconnect_all
# ============================================================================


class TestMCPClientServerPaths:
    """Testet MCP-Client Server-bezogene Pfade."""

    @pytest.fixture()
    def mcp(self, tmp_path: Path):
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        config.ensure_directories()
        return JarvisMCPClient(config)

    def test_tool_count_empty(self, mcp):
        """Leerer Client hat 0 Tools."""
        assert mcp.tool_count == 0

    def test_server_count_empty(self, mcp):
        """Leerer Client hat 0 Server."""
        assert mcp.server_count == 0

    @pytest.mark.asyncio()
    async def test_call_tool_server_not_connected(self, mcp):
        """Tool auf nicht verbundenem Server gibt Fehler."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig, MCPToolInfo

        # Registriere Tool das auf einem Server liegt
        mcp._tool_registry["remote_tool"] = MCPToolInfo(
            name="remote_tool",
            server="my_server",
            description="Remote tool",
            input_schema={},
        )

        # Server existiert aber ist disconnected
        mcp._servers["my_server"] = ServerConnection(
            name="my_server",
            config=MCPServerConfig(command="echo", args=[]),
            connected=False,
        )

        result = await mcp.call_tool("remote_tool", {})
        assert result.is_error is True
        assert "nicht verbunden" in result.content

    @pytest.mark.asyncio()
    async def test_call_tool_server_not_found(self, mcp):
        """Tool auf nicht existierendem Server gibt Fehler."""
        from jarvis.models import MCPToolInfo

        mcp._tool_registry["orphan_tool"] = MCPToolInfo(
            name="orphan_tool",
            server="nonexistent_server",
            description="Orphan",
            input_schema={},
        )

        result = await mcp.call_tool("orphan_tool", {})
        assert result.is_error is True
        assert "nicht verbunden" in result.content

    @pytest.mark.asyncio()
    async def test_call_tool_on_connected_server(self, mcp):
        """Tool auf verbundenem Server delegiert an Session."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig, MCPToolInfo

        mock_session = AsyncMock()
        mock_block = MagicMock()
        mock_block.text = "Server-Ergebnis"
        mock_result = MagicMock()
        mock_result.content = [mock_block]
        mock_result.isError = False
        mock_session.call_tool.return_value = mock_result

        mcp._tool_registry["server_tool"] = MCPToolInfo(
            name="server_tool",
            server="live_server",
            description="Tool",
            input_schema={},
        )

        mcp._servers["live_server"] = ServerConnection(
            name="live_server",
            config=MCPServerConfig(command="echo", args=[]),
            session=mock_session,
            connected=True,
        )

        result = await mcp.call_tool("server_tool", {"key": "val"})
        assert result.is_error is False
        assert "Server-Ergebnis" in result.content
        mock_session.call_tool.assert_awaited_once_with("server_tool", arguments={"key": "val"})

    @pytest.mark.asyncio()
    async def test_call_tool_server_exception(self, mcp):
        """Server-Session-Exception wird als Fehler zurückgegeben."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig, MCPToolInfo

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = RuntimeError("Server crashed")

        mcp._tool_registry["crash_tool"] = MCPToolInfo(
            name="crash_tool",
            server="crash_server",
            description="Crashing",
            input_schema={},
        )

        mcp._servers["crash_server"] = ServerConnection(
            name="crash_server",
            config=MCPServerConfig(command="echo", args=[]),
            session=mock_session,
            connected=True,
        )

        result = await mcp.call_tool("crash_tool", {})
        assert result.is_error is True
        assert "Server crashed" in result.content

    @pytest.mark.asyncio()
    async def test_call_tool_server_non_text_block(self, mcp):
        """Server-Response mit Non-Text-Block wird stringifiziert."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig, MCPToolInfo

        class BinaryBlock:
            """Mock block without .text attribute."""

            def __str__(self):
                return "binary-content"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [BinaryBlock()]
        mock_result.isError = False
        mock_session.call_tool.return_value = mock_result

        mcp._tool_registry["bin_tool"] = MCPToolInfo(
            name="bin_tool",
            server="bin_server",
            description="Binary",
            input_schema={},
        )
        mcp._servers["bin_server"] = ServerConnection(
            name="bin_server",
            config=MCPServerConfig(command="echo", args=[]),
            session=mock_session,
            connected=True,
        )

        result = await mcp.call_tool("bin_tool", {})
        assert result.is_error is False


class TestMCPClientConfigLoading:
    """Testet MCP-Server-Config-Loading."""

    @pytest.fixture()
    def mcp_with_config(self, tmp_path: Path):
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        config.ensure_directories()

        # Write MCP config
        mcp_config_path = config.mcp_config_file
        mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_config_path.write_text(
            """
servers:
  file_server:
    command: python
    args: ["-m", "jarvis.mcp.filesystem"]
    enabled: true
    transport: stdio
  disabled_server:
    command: python
    args: ["-m", "jarvis.mcp.shell"]
    enabled: false
    transport: stdio
""",
            encoding="utf-8",
        )

        return JarvisMCPClient(config)

    def test_load_server_configs(self, mcp_with_config):
        """Lädt MCP-Server-Konfiguration aus YAML."""
        configs = mcp_with_config._load_server_configs()
        assert "file_server" in configs
        assert "disabled_server" in configs
        assert configs["file_server"].enabled is True
        assert configs["disabled_server"].enabled is False

    def test_load_server_configs_missing_file(self, tmp_path):
        """Fehlende Config-Datei gibt leeres Dict."""
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path / "nonexistent")
        mcp = JarvisMCPClient(config)
        configs = mcp._load_server_configs()
        assert configs == {}

    def test_load_server_configs_invalid_yaml(self, tmp_path):
        """Ungültige Config gibt leeres Dict."""
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        config.ensure_directories()
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text("not: valid: yaml: [", encoding="utf-8")

        mcp = JarvisMCPClient(config)
        configs = mcp._load_server_configs()
        assert isinstance(configs, dict)

    def test_load_server_configs_no_servers_key(self, tmp_path):
        """Config ohne 'servers' Key gibt leeres Dict."""
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        config.ensure_directories()
        config.mcp_config_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp_config_file.write_text("other_key: value", encoding="utf-8")

        mcp = JarvisMCPClient(config)
        configs = mcp._load_server_configs()
        assert configs == {}

    @pytest.mark.asyncio()
    async def test_connect_all_disabled_servers_skipped(self, mcp_with_config):
        """Disabled Server werden übersprungen bei connect_all."""
        # Patch _connect_server to avoid actual subprocess
        with patch.object(
            mcp_with_config,
            "_connect_server",
            new_callable=AsyncMock,
        ) as mock_connect:
            await mcp_with_config.connect_all()
            # Nur file_server sollte connected werden (disabled_server ist disabled)
            mock_connect.assert_awaited_once()
            call_args = mock_connect.call_args
            assert call_args[0][0] == "file_server"

    @pytest.mark.asyncio()
    async def test_connect_all_server_failure_handled(self, mcp_with_config):
        """Server-Verbindungsfehler werden abgefangen."""
        with patch.object(
            mcp_with_config,
            "_connect_server",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Connection failed"),
        ):
            # Sollte nicht crashen
            await mcp_with_config.connect_all()


class TestMCPClientDisconnect:
    """Testet disconnect_all mit aktiven Servern."""

    @pytest.fixture()
    def mcp_with_servers(self, tmp_path: Path):
        from jarvis.mcp.client import JarvisMCPClient, ServerConnection
        from jarvis.models import MCPServerConfig

        config = JarvisConfig(jarvis_home=tmp_path)
        mcp = JarvisMCPClient(config)

        # Mock-Prozess der noch läuft
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        mcp._servers["running_server"] = ServerConnection(
            name="running_server",
            config=MCPServerConfig(command="echo", args=[]),
            process=mock_proc,
            connected=True,
        )

        return mcp, mock_proc

    @pytest.mark.asyncio()
    async def test_disconnect_terminates_process(self, mcp_with_servers):
        """disconnect_all terminiert laufende Server-Prozesse."""
        mcp, mock_proc = mcp_with_servers

        await mcp.disconnect_all()

        mock_proc.terminate.assert_called_once()
        # Nach disconnect_all wird _servers geleert
        assert len(mcp._servers) == 0

    @pytest.mark.asyncio()
    async def test_disconnect_kills_on_timeout(self, mcp_with_servers):
        """disconnect_all killt Prozess wenn terminate nicht funktioniert."""
        mcp, mock_proc = mcp_with_servers
        mock_proc.wait.side_effect = TimeoutError()

        await mcp.disconnect_all()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio()
    async def test_disconnect_already_stopped_process(self, tmp_path):
        """disconnect_all mit bereits gestopptem Prozess."""
        from jarvis.mcp.client import JarvisMCPClient, ServerConnection
        from jarvis.models import MCPServerConfig

        config = JarvisConfig(jarvis_home=tmp_path)
        mcp = JarvisMCPClient(config)

        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Bereits beendet

        mcp._servers["stopped"] = ServerConnection(
            name="stopped",
            config=MCPServerConfig(command="echo", args=[]),
            process=mock_proc,
            connected=True,
        )

        await mcp.disconnect_all()
        # Servers cleared after disconnect
        assert len(mcp._servers) == 0

    @pytest.mark.asyncio()
    async def test_disconnect_no_process(self, tmp_path):
        """disconnect_all mit Server ohne Prozess."""
        from jarvis.mcp.client import JarvisMCPClient, ServerConnection
        from jarvis.models import MCPServerConfig

        config = JarvisConfig(jarvis_home=tmp_path)
        mcp = JarvisMCPClient(config)

        mcp._servers["no_proc"] = ServerConnection(
            name="no_proc",
            config=MCPServerConfig(command="echo", args=[]),
            process=None,
            connected=True,
        )

        await mcp.disconnect_all()
        assert len(mcp._servers) == 0
        assert mcp.tool_count == 0
