"""Tests für die LLM-Backend-Abstraktion."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

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


# ============================================================================
# OllamaBackend
# ============================================================================


class TestOllamaBackend:
    def test_backend_type(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        assert b.backend_type == LLMBackendType.OLLAMA

    @pytest.mark.asyncio
    async def test_chat_success(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Hallo!", "role": "assistant"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])

        assert isinstance(result, ChatResponse)
        assert result.content == "Hallo!"
        assert result.model == "qwen3:8b"
        assert result.usage["total_tokens"] == 15
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"query": "Wetter"}}}
                ],
            },
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("qwen3:32b", [{"role": "user", "content": "Wetter?"}])
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        with pytest.raises(LLMBackendError, match="HTTP 500"):
            await b.chat("model", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_chat_timeout(self) -> None:
        b = OllamaBackend("http://localhost:11434", timeout=5)
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        b._client = mock_client

        with pytest.raises(LLMBackendError, match="Timeout"):
            await b.chat("model", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_chat_connect_error(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        b._client = mock_client

        with pytest.raises(LLMBackendError, match="nicht erreichbar"):
            await b.chat("model", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_embed_success(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.embed("nomic-embed-text", "Test")
        assert isinstance(result, EmbedResponse)
        assert len(result.embedding) == 3

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        assert await b.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        b._client = mock_client

        assert await b.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "qwen3:8b"}, {"name": "qwen3:32b"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        models = await b.list_models()
        assert "qwen3:8b" in models

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        b = OllamaBackend("http://localhost:11434")
        mock_client = AsyncMock()
        mock_client.is_closed = False
        b._client = mock_client

        await b.close()
        mock_client.aclose.assert_awaited_once()
        assert b._client is None


# ============================================================================
# OpenAIBackend
# ============================================================================


class TestOpenAIBackend:
    def test_backend_type(self) -> None:
        b = OpenAIBackend(api_key="sk-test")
        assert b.backend_type == LLMBackendType.OPENAI

    @pytest.mark.asyncio
    async def test_chat_success(self) -> None:
        b = OpenAIBackend(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Antwort", "role": "assistant"}}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("gpt-4o", [{"role": "user", "content": "Hi"}])

        assert result.content == "Antwort"
        assert result.model == "gpt-4o"
        assert result.usage["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        b = OpenAIBackend(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("gpt-4o", [{"role": "user", "content": "suche"}])
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "search"
        assert result.tool_calls[0]["function"]["arguments"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        b = OpenAIBackend(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.5, 0.6, 0.7]}],
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.embed("text-embedding-3-small", "Test")
        assert len(result.embedding) == 3

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        b = OpenAIBackend(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        models = await b.list_models()
        assert "gpt-4o" in models


# ============================================================================
# AnthropicBackend
# ============================================================================


class TestAnthropicBackend:
    def test_backend_type(self) -> None:
        b = AnthropicBackend(api_key="sk-ant-test")
        assert b.backend_type == LLMBackendType.ANTHROPIC

    @pytest.mark.asyncio
    async def test_chat_success(self) -> None:
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Hallo!"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat(
            "claude-sonnet-4-20250514",
            [
                {"role": "system", "content": "Du bist Jarvis."},
                {"role": "user", "content": "Hi"},
            ],
        )

        assert result.content == "Hallo!"
        assert result.usage["total_tokens"] == 15

        # Verifiziere dass system separat gesendet wurde
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "system" in payload
        assert all(m["role"] != "system" for m in payload["messages"])

    @pytest.mark.asyncio
    async def test_chat_with_tool_use(self) -> None:
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [
                {"type": "text", "text": "Ich suche das für dich."},
                {"type": "tool_use", "name": "web_search", "input": {"query": "Wetter Berlin"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("claude-sonnet-4-20250514", [{"role": "user", "content": "Wetter?"}])

        assert "suche" in result.content
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_embed_not_supported(self) -> None:
        b = AnthropicBackend(api_key="sk-ant-test")
        with pytest.raises(LLMBackendError, match="keine Embedding-API"):
            await b.embed("claude-sonnet-4-20250514", "Test")

    @pytest.mark.asyncio
    async def test_list_models_static(self) -> None:
        b = AnthropicBackend(api_key="sk-ant-test")
        models = await b.list_models()
        assert len(models) >= 3
        assert any("claude" in m for m in models)

    def test_convert_tools_mcp_format(self) -> None:
        tools = [
            {
                "name": "search",
                "description": "Sucht im Web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
        result = AnthropicBackend._convert_tools_to_anthropic(tools)
        assert result[0]["name"] == "search"
        assert "input_schema" in result[0]

    @pytest.mark.asyncio
    async def test_chat_preserves_content_array(self) -> None:
        """Content-Arrays (multimodal) werden nicht zu Strings konvertiert."""
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Ich sehe ein Bild."}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        multimodal_content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
            {"type": "text", "text": "Was siehst du?"},
        ]
        await b.chat(
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": multimodal_content}],
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        user_msg = payload["messages"][0]
        assert isinstance(user_msg["content"], list), "Content-Array darf nicht stringifiziert werden"
        assert user_msg["content"][0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_system_message_extracts_text_from_list(self) -> None:
        """System-Messages mit Content-Arrays werden korrekt extrahiert."""
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        system_content = [
            {"type": "text", "text": "Du bist Jarvis."},
            {"type": "text", "text": "Sei hilfsbereit."},
        ]
        await b.chat(
            "claude-sonnet-4-20250514",
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "Du bist Jarvis." in payload["system"]
        assert "Sei hilfsbereit." in payload["system"]

    @pytest.mark.asyncio
    async def test_plain_string_content_unchanged(self) -> None:
        """Bestehende String-Messages funktionieren weiterhin identisch."""
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Antwort"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        await b.chat(
            "claude-sonnet-4-20250514",
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Einfache Frage"},
            ],
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["system"] == "System prompt"
        assert payload["messages"][0]["content"] == "Einfache Frage"

    @pytest.mark.asyncio
    async def test_mixed_messages(self) -> None:
        """Text + multimodal Messages in derselben Liste."""
        b = AnthropicBackend(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 50, "output_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        messages = [
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Hi! Wie kann ich helfen?"},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                {"type": "text", "text": "Was zeigt dieses Bild?"},
            ]},
        ]
        await b.chat("claude-sonnet-4-20250514", messages)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        api_msgs = payload["messages"]
        assert len(api_msgs) == 3
        assert isinstance(api_msgs[0]["content"], str)
        assert isinstance(api_msgs[2]["content"], list)

    def test_convert_tools_openai_format(self) -> None:
        tools = [
            {
                "function": {
                    "name": "calc",
                    "description": "Berechnet",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = AnthropicBackend._convert_tools_to_anthropic(tools)
        assert result[0]["name"] == "calc"


# ============================================================================
# Factory
# ============================================================================


class TestFactory:
    def test_default_ollama(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "ollama"
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.timeout_seconds = 120
        config.ollama.keep_alive = "30m"

        backend = create_backend(config)
        assert isinstance(backend, OllamaBackend)

    def test_openai(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "openai"
        config.openai_api_key = "sk-test"
        config.openai_base_url = "https://api.openai.com/v1"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)

    def test_anthropic(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "anthropic"
        config.anthropic_api_key = "sk-ant-test"
        config.anthropic_max_tokens = 4096
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, AnthropicBackend)

    def test_unknown_falls_back_to_ollama(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "unknown_provider"
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.timeout_seconds = 120
        config.ollama.keep_alive = "30m"

        backend = create_backend(config)
        assert isinstance(backend, OllamaBackend)

    def test_create_gemini_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "gemini"
        config.gemini_api_key = "AIza-test"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, GeminiBackend)

    def test_create_groq_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "groq"
        config.groq_api_key = "gsk_test"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "groq.com" in backend._base_url

    def test_create_deepseek_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "deepseek"
        config.deepseek_api_key = "sk-ds-test"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "deepseek.com" in backend._base_url

    def test_create_mistral_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "mistral"
        config.mistral_api_key = "mistral-test"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "mistral.ai" in backend._base_url

    def test_create_together_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "together"
        config.together_api_key = "together-test"
        config.ollama.timeout_seconds = 120

        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "together.xyz" in backend._base_url

    def test_create_openrouter_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "openrouter"
        config.openrouter_api_key = "sk-or-test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "openrouter.ai" in backend._base_url

    def test_create_xai_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "xai"
        config.xai_api_key = "xai-test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "x.ai" in backend._base_url

    def test_create_cerebras_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "cerebras"
        config.cerebras_api_key = "csk-test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "cerebras.ai" in backend._base_url

    def test_create_github_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "github"
        config.github_api_key = "ghp_test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "models.inference.ai.azure.com" in backend._base_url

    def test_create_bedrock_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "bedrock"
        config.bedrock_api_key = "bedrock-test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "bedrock-runtime" in backend._base_url

    def test_create_huggingface_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "huggingface"
        config.huggingface_api_key = "hf_test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "huggingface.co" in backend._base_url

    def test_create_moonshot_backend(self) -> None:
        config = MagicMock()
        config.llm_backend_type = "moonshot"
        config.moonshot_api_key = "sk-moon-test"
        config.ollama.timeout_seconds = 120
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert "moonshot.cn" in backend._base_url


# ============================================================================
# GeminiBackend
# ============================================================================


class TestGeminiBackend:
    def test_backend_type(self) -> None:
        b = GeminiBackend(api_key="AIza-test")
        assert b.backend_type == LLMBackendType.GEMINI

    def test_message_conversion(self) -> None:
        """OpenAI-Format Messages werden korrekt zu Gemini-Format konvertiert."""
        messages = [
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Wie geht es dir?"},
        ]
        system_text, contents = GeminiBackend._convert_messages(messages)

        assert system_text == ""
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hallo"}]
        assert contents[1]["role"] == "model"  # assistant → model
        assert contents[1]["parts"] == [{"text": "Hi!"}]
        assert contents[2]["role"] == "user"

    def test_system_message_extraction(self) -> None:
        """System-Messages werden extrahiert und nicht in contents aufgenommen."""
        messages = [
            {"role": "system", "content": "Du bist Jarvis."},
            {"role": "user", "content": "Hi"},
        ]
        system_text, contents = GeminiBackend._convert_messages(messages)

        assert "Du bist Jarvis." in system_text
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_system_message_from_list(self) -> None:
        """System-Messages mit Content-Arrays werden korrekt extrahiert."""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "Du bist Jarvis."},
                {"type": "text", "text": "Sei hilfsbereit."},
            ]},
            {"role": "user", "content": "Hi"},
        ]
        system_text, contents = GeminiBackend._convert_messages(messages)

        assert "Du bist Jarvis." in system_text
        assert "Sei hilfsbereit." in system_text
        assert len(contents) == 1

    @pytest.mark.asyncio
    async def test_chat_response_parsing(self) -> None:
        """Gemini-Antwort wird korrekt zu ChatResponse konvertiert."""
        b = GeminiBackend(api_key="AIza-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hallo Welt!"}],
                    "role": "model",
                },
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("gemini-2.5-pro", [{"role": "user", "content": "Hi"}])

        assert isinstance(result, ChatResponse)
        assert result.content == "Hallo Welt!"
        assert result.model == "gemini-2.5-pro"
        assert result.usage["total_tokens"] == 15
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_embed_response(self) -> None:
        """Gemini Embedding-Antwort wird korrekt geparst."""
        b = GeminiBackend(api_key="AIza-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embedding": {"values": [0.1, 0.2, 0.3, 0.4]},
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.embed("text-embedding-004", "Test")
        assert isinstance(result, EmbedResponse)
        assert len(result.embedding) == 4
        assert result.embedding[0] == 0.1

    def test_tool_call_conversion(self) -> None:
        """OpenAI-Tools werden korrekt zu Gemini functionDeclarations konvertiert."""
        tools = [
            {
                "function": {
                    "name": "web_search",
                    "description": "Sucht im Web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]
        result = GeminiBackend._convert_tools_to_gemini(tools)

        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["description"] == "Sucht im Web"
        assert "properties" in result[0]["parameters"]

    def test_tool_call_conversion_mcp_format(self) -> None:
        """MCP-Format Tools werden ebenfalls konvertiert."""
        tools = [
            {
                "name": "search",
                "description": "Suche",
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]
        result = GeminiBackend._convert_tools_to_gemini(tools)
        assert result[0]["name"] == "search"
        assert "properties" in result[0]["parameters"]

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        """Gemini functionCall Antworten werden korrekt geparst."""
        b = GeminiBackend(api_key="AIza-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "web_search", "args": {"query": "Wetter"}}},
                    ],
                    "role": "model",
                },
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.chat("gemini-2.5-pro", [{"role": "user", "content": "Wetter?"}])
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"
        assert result.tool_calls[0]["function"]["arguments"] == {"query": "Wetter"}

    @pytest.mark.asyncio
    async def test_chat_http_error(self) -> None:
        b = GeminiBackend(api_key="AIza-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        with pytest.raises(LLMBackendError, match="HTTP 400"):
            await b.chat("gemini-2.5-pro", [{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        b = GeminiBackend(api_key="AIza-test")
        mock_client = AsyncMock()
        mock_client.is_closed = False
        b._client = mock_client

        await b.close()
        mock_client.aclose.assert_awaited_once()
        assert b._client is None
