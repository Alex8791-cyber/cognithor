"""Coverage-Tests fuer llm_backend.py -- fehlende Zeilen."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure
from jarvis.core.llm_backend import (
    AnthropicBackend,
    ChatResponse,
    EmbedResponse,
    GeminiBackend,
    LLMBackendError,
    LLMBackendType,
    OllamaBackend,
    OpenAIBackend,
    create_backend,
)


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    cfg = JarvisConfig(jarvis_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


# ============================================================================
# Data classes
# ============================================================================


class TestDataClasses:
    def test_chat_response(self) -> None:
        resp = ChatResponse(content="Hello", model="gpt-4")
        assert resp.content == "Hello"
        assert resp.model == "gpt-4"
        assert resp.tool_calls is None
        assert resp.usage is None
        assert resp.raw is None

    def test_chat_response_with_tools(self) -> None:
        tools = [{"function": {"name": "search", "arguments": {}}}]
        resp = ChatResponse(content="", tool_calls=tools, model="gpt-4")
        assert resp.tool_calls == tools

    def test_embed_response(self) -> None:
        resp = EmbedResponse(embedding=[0.1, 0.2, 0.3], model="embed-v1")
        assert resp.embedding == [0.1, 0.2, 0.3]
        assert resp.model == "embed-v1"

    def test_llm_backend_error(self) -> None:
        err = LLMBackendError("test error", status_code=500)
        assert str(err) == "test error"
        assert err.status_code == 500

    def test_llm_backend_type_enum(self) -> None:
        assert LLMBackendType.OLLAMA == "ollama"
        assert LLMBackendType.OPENAI == "openai"
        assert LLMBackendType.ANTHROPIC == "anthropic"
        assert LLMBackendType.GEMINI == "gemini"
        assert LLMBackendType.LMSTUDIO == "lmstudio"


# ============================================================================
# create_backend factory
# ============================================================================


class TestCreateBackend:
    def test_create_ollama_default(self, config: JarvisConfig) -> None:
        """Default backend type is ollama."""
        backend = create_backend(config)
        assert isinstance(backend, OllamaBackend)
        assert backend.backend_type == LLMBackendType.OLLAMA

    def test_create_openai(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "openai"
        config.openai_api_key = "sk-test"
        config.openai_base_url = "https://api.openai.com/v1"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend.backend_type == LLMBackendType.OPENAI

    def test_create_anthropic(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "anthropic"
        config.anthropic_api_key = "sk-ant-test"
        backend = create_backend(config)
        assert isinstance(backend, AnthropicBackend)
        assert backend.backend_type == LLMBackendType.ANTHROPIC

    def test_create_gemini(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "gemini"
        config.gemini_api_key = "test-key"
        backend = create_backend(config)
        assert isinstance(backend, GeminiBackend)
        assert backend.backend_type == LLMBackendType.GEMINI

    def test_create_lmstudio(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "lmstudio"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)

    def test_create_groq(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "groq"
        config.groq_api_key = "gsk-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)

    def test_create_deepseek(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "deepseek"
        config.deepseek_api_key = "sk-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)


# ============================================================================
# OpenAIBackend
# ============================================================================


class TestOpenAIBackend:
    def _make_backend(self) -> OpenAIBackend:
        return OpenAIBackend(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            timeout=30,
        )

    def test_backend_type(self) -> None:
        backend = self._make_backend()
        assert backend.backend_type == LLMBackendType.OPENAI

    def test_is_reasoning_model(self) -> None:
        backend = self._make_backend()
        assert backend._is_reasoning_model("o1") is True
        assert backend._is_reasoning_model("o1-mini") is True
        assert backend._is_reasoning_model("o3-preview") is True
        assert backend._is_reasoning_model("o4-mini") is True
        assert backend._is_reasoning_model("gpt-5") is True
        assert backend._is_reasoning_model("gpt-5.1-mini") is True
        assert backend._is_reasoning_model("gpt-5.2-pro") is True
        assert backend._is_reasoning_model("gpt-4") is False
        assert backend._is_reasoning_model("gpt-4o") is False
        assert backend._is_reasoning_model("claude-3-opus") is False

    @pytest.mark.asyncio
    async def test_chat(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!", "tool_calls": None},
            }],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello!"
        assert result.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_chat_with_tools(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "web_search", "arguments": '{"q":"test"}'}},
                    ],
                },
            }],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="gpt-4",
                messages=[{"role": "user", "content": "Search"}],
                tools=[{"function": {"name": "web_search"}}],
            )
        assert isinstance(result, ChatResponse)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="OpenAI HTTP 500"):
                await backend.chat(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_chat_reasoning_model_no_temperature(self) -> None:
        """Reasoning models (o1, o3, gpt-5) should not send temperature/top_p."""
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "model": "o3-mini",
            "usage": {},
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="o3-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert isinstance(result, ChatResponse)
        # Check that post was called without temperature in payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert "temperature" not in payload

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.embed("text-embedding-3-small", "Hello")
        assert isinstance(result, EmbedResponse)
        assert result.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.close()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_already_closed(self) -> None:
        backend = self._make_backend()
        backend._client = None
        await backend.close()  # Should not raise


# ============================================================================
# AnthropicBackend
# ============================================================================


class TestAnthropicBackend:
    def _make_backend(self) -> AnthropicBackend:
        return AnthropicBackend(
            api_key="sk-ant-test",
            timeout=30,
            max_tokens=4096,
        )

    def test_backend_type(self) -> None:
        backend = self._make_backend()
        assert backend.backend_type == LLMBackendType.ANTHROPIC

    @pytest.mark.asyncio
    async def test_embed_not_supported(self) -> None:
        backend = self._make_backend()
        with pytest.raises(LLMBackendError, match="Embedding"):
            await backend.embed("model", "text")

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        backend = self._make_backend()
        models = await backend.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_chat(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-opus",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="claude-3-opus",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
            )
        assert isinstance(result, ChatResponse)
        assert "Hello!" in result.content

    def test_convert_tools_to_anthropic(self) -> None:
        tools = [
            {"function": {"name": "search", "description": "Search the web", "parameters": {"type": "object"}}},
            {"name": "read", "description": "Read a file", "inputSchema": {"type": "object"}},
        ]
        converted = AnthropicBackend._convert_tools_to_anthropic(tools)
        assert len(converted) == 2
        assert converted[0]["name"] == "search"
        assert converted[1]["name"] == "read"


# ============================================================================
# GeminiBackend
# ============================================================================


class TestGeminiBackend:
    def _make_backend(self) -> GeminiBackend:
        return GeminiBackend(api_key="test-key", timeout=30)

    def test_backend_type(self) -> None:
        backend = self._make_backend()
        assert backend.backend_type == LLMBackendType.GEMINI

    def test_convert_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        system_text, contents = GeminiBackend._convert_messages(messages)
        assert "helper" in system_text
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_convert_tools_to_gemini(self) -> None:
        tools = [
            {"function": {"name": "search", "description": "Search", "parameters": {"type": "object"}}},
            {"name": "read", "description": "Read file", "inputSchema": {"type": "object"}},
        ]
        declarations = GeminiBackend._convert_tools_to_gemini(tools)
        assert len(declarations) == 2
        assert declarations[0]["name"] == "search"
        assert declarations[1]["name"] == "read"


# ============================================================================
# OllamaBackend
# ============================================================================


class TestOllamaBackend:
    def _make_backend(self) -> OllamaBackend:
        return OllamaBackend(
            base_url="http://localhost:11434",
            timeout=30,
        )

    def test_backend_type(self) -> None:
        backend = self._make_backend()
        assert backend.backend_type == LLMBackendType.OLLAMA

    @pytest.mark.asyncio
    async def test_close_no_client(self) -> None:
        backend = self._make_backend()
        backend._client = None
        await backend.close()  # Should not raise


# ============================================================================
# OllamaBackend -- extended coverage
# ============================================================================


class TestOllamaBackendExtended:
    def _make_backend(self) -> OllamaBackend:
        return OllamaBackend(
            base_url="http://localhost:11434",
            timeout=30,
        )

    @pytest.mark.asyncio
    async def test_chat_basic(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hi there!", "tool_calls": None},
            "model": "llama3",
            "prompt_eval_count": 12,
            "eval_count": 8,
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert isinstance(result, ChatResponse)
        assert result.content == "Hi there!"
        assert result.model == "llama3"
        assert result.tool_calls is None
        assert result.usage["prompt_tokens"] == 12
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 20
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"q": "weather"}}},
                ],
            },
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Search weather"}],
                tools=[{"function": {"name": "web_search"}}],
            )
        assert isinstance(result, ChatResponse)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Ollama HTTP 500"):
                await backend.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_chat_timeout(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("read timed out"))

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Timeout"):
                await backend.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4]],
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.embed("nomic-embed-text", "Hello world")
        assert isinstance(result, EmbedResponse)
        assert result.embedding == [0.1, 0.2, 0.3, 0.4]
        assert result.model == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Ollama Embed HTTP 500"):
                await backend.embed("nomic-embed-text", "Hello")

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is True
        mock_client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "qwen3:32b"},
                {"name": "nomic-embed-text:latest"},
            ],
        }
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.list_models()
        assert result == ["llama3:latest", "qwen3:32b", "nomic-embed-text:latest"]

    @pytest.mark.asyncio
    async def test_list_models_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.list_models()
        assert result == []

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.close()
        mock_client.aclose.assert_awaited_once()
        assert backend._client is None


# ============================================================================
# GeminiBackend -- extended coverage
# ============================================================================


class TestGeminiBackendExtended:
    def _make_backend(self) -> GeminiBackend:
        return GeminiBackend(api_key="test-gemini-key", timeout=30)

    @pytest.mark.asyncio
    async def test_chat(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model",
                },
            }],
            "modelVersion": "gemini-pro",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 6,
                "totalTokenCount": 16,
            },
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="gemini-pro",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello from Gemini!"
        assert result.model == "gemini-pro"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 6
        assert result.usage["total_tokens"] == 16
        # Verify the URL contains generateContent
        call_args = mock_client.post.call_args
        assert "generateContent" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_chat_with_function_call(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "web_search",
                            "args": {"query": "current weather"},
                        },
                    }],
                    "role": "model",
                },
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
                "totalTokenCount": 12,
            },
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="gemini-pro",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=[{"function": {"name": "web_search", "description": "Search", "parameters": {"type": "object"}}}],
            )
        assert isinstance(result, ChatResponse)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"
        assert result.tool_calls[0]["function"]["arguments"] == {"query": "current weather"}
        assert result.content == ""  # No text parts, only function call

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Gemini HTTP 500"):
                await backend.chat(
                    model="gemini-pro",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embedding": {
                "values": [0.5, 0.6, 0.7, 0.8],
            },
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.embed("text-embedding-004", "Hello")
        assert isinstance(result, EmbedResponse)
        assert result.embedding == [0.5, 0.6, 0.7, 0.8]
        assert result.model == "text-embedding-004"
        # Verify the URL contains embedContent
        call_args = mock_client.post.call_args
        assert "embedContent" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_embed_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Gemini Embed HTTP 400"):
                await backend.embed("text-embedding-004", "Hello")

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "models/gemini-pro"},
                {"name": "models/gemini-1.5-flash"},
                {"name": "models/text-embedding-004"},
            ],
        }
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.list_models()
        assert result == ["gemini-pro", "gemini-1.5-flash", "text-embedding-004"]

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.close()
        mock_client.aclose.assert_awaited_once()
        assert backend._client is None

    def test_convert_messages_list_content(self) -> None:
        """Messages with content as list (multi-part) instead of string."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helper."},
                    {"type": "text", "text": "Be concise."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response part"},
                ],
            },
        ]
        system_text, contents = GeminiBackend._convert_messages(messages)
        assert "helper" in system_text
        assert "concise" in system_text
        assert len(contents) == 2
        # User message should have 2 parts
        assert len(contents[0]["parts"]) == 2
        assert contents[0]["parts"][0]["text"] == "Part 1"
        assert contents[0]["parts"][1]["text"] == "Part 2"
        assert contents[0]["role"] == "user"
        # Assistant -> model
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Response part"


# ============================================================================
# AnthropicBackend -- extended coverage
# ============================================================================


class TestAnthropicBackendExtended:
    def _make_backend(self) -> AnthropicBackend:
        return AnthropicBackend(
            api_key="sk-ant-test",
            timeout=30,
            max_tokens=4096,
        )

    @pytest.mark.asyncio
    async def test_chat_with_tool_use(self) -> None:
        """Anthropic response with tool_use content blocks."""
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [
                {"type": "text", "text": "Let me search that for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_01abc",
                    "name": "web_search",
                    "input": {"query": "weather today"},
                },
            ],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 15, "output_tokens": 20},
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.chat(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "What is the weather?"}],
                tools=[{"function": {"name": "web_search", "description": "Search", "parameters": {"type": "object"}}}],
            )
        assert isinstance(result, ChatResponse)
        assert "search that" in result.content
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"
        assert result.tool_calls[0]["function"]["arguments"] == {"query": "weather today"}
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 35

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            with pytest.raises(LLMBackendError, match="Anthropic HTTP 500"):
                await backend.chat(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Anthropic uses POST /messages with empty body -- 200 or 400 means reachable."""
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400  # Validation error = API reachable
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        backend = self._make_backend()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch.object(backend, "_ensure_client", return_value=mock_client):
            result = await backend.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.close()
        mock_client.aclose.assert_awaited_once()
        assert backend._client is None


# ============================================================================
# create_backend -- extended coverage (additional providers)
# ============================================================================


class TestCreateBackendExtended:
    @pytest.fixture()
    def config(self, tmp_path) -> JarvisConfig:
        cfg = JarvisConfig(jarvis_home=tmp_path)
        ensure_directory_structure(cfg)
        return cfg

    def test_create_mistral(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "mistral"
        config.mistral_api_key = "sk-mistral-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "https://api.mistral.ai/v1"
        assert backend._api_key == "sk-mistral-test"

    def test_create_together(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "together"
        config.together_api_key = "sk-together-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "https://api.together.xyz/v1"
        assert backend._api_key == "sk-together-test"

    def test_create_openrouter(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "openrouter"
        config.openrouter_api_key = "sk-or-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "https://openrouter.ai/api/v1"
        assert backend._api_key == "sk-or-test"

    def test_create_xai(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "xai"
        config.xai_api_key = "sk-xai-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "https://api.x.ai/v1"
        assert backend._api_key == "sk-xai-test"

    def test_create_cerebras(self, config: JarvisConfig) -> None:
        config.llm_backend_type = "cerebras"
        config.cerebras_api_key = "sk-cerebras-test"
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "https://api.cerebras.ai/v1"
        assert backend._api_key == "sk-cerebras-test"
