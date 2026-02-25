"""Tests für den UnifiedLLMClient-Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.core.model_router import OllamaClient, OllamaError
from jarvis.core.unified_llm import UnifiedLLMClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama() -> AsyncMock:
    """Mock OllamaClient mit Standardverhalten."""
    client = AsyncMock(spec=OllamaClient)
    client.chat = AsyncMock(
        return_value={
            "message": {
                "role": "assistant",
                "content": "Hallo von Ollama!",
            },
            "model": "qwen3:8b",
            "done": True,
        }
    )
    client.is_available = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=["qwen3:8b", "nomic-embed-text"])
    client.embed = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})
    client.close = AsyncMock()
    return client


@dataclass
class MockChatResponse:
    content: str = ""
    tool_calls: list | None = None
    model: str = ""
    usage: dict | None = None
    raw: dict | None = None


@dataclass
class MockEmbedResponse:
    embedding: list[float] | None = None


class MockBackendType:
    value = "openai"


@pytest.fixture
def mock_backend() -> AsyncMock:
    """Mock LLMBackend mit Standardverhalten."""
    backend = AsyncMock()
    backend.backend_type = MockBackendType()
    backend.chat = AsyncMock(
        return_value=MockChatResponse(
            content="Hallo von OpenAI!",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
    )
    backend.is_available = AsyncMock(return_value=True)
    backend.list_models = AsyncMock(return_value=["gpt-4o", "gpt-4o-mini"])
    backend.embed = AsyncMock(return_value=MockEmbedResponse(embedding=[0.4, 0.5, 0.6]))
    backend.close = AsyncMock()
    return backend


@pytest.fixture
def ollama_client(mock_ollama: AsyncMock) -> UnifiedLLMClient:
    """UnifiedLLMClient im Ollama-Modus (kein Backend)."""
    return UnifiedLLMClient(mock_ollama, backend=None)


@pytest.fixture
def openai_client(mock_ollama: AsyncMock, mock_backend: AsyncMock) -> UnifiedLLMClient:
    """UnifiedLLMClient mit OpenAI-Backend."""
    return UnifiedLLMClient(mock_ollama, backend=mock_backend)


# ============================================================================
# Initialisierung
# ============================================================================


class TestUnifiedLLMInit:
    def test_ollama_mode(self, ollama_client: UnifiedLLMClient) -> None:
        assert ollama_client.backend_type == "ollama"
        assert ollama_client._backend is None

    def test_backend_mode(self, openai_client: UnifiedLLMClient) -> None:
        assert openai_client.backend_type == "openai"
        assert openai_client._backend is not None

    def test_has_embedding_support_ollama(self, ollama_client: UnifiedLLMClient) -> None:
        assert ollama_client.has_embedding_support is True

    def test_has_embedding_support_openai(self, openai_client: UnifiedLLMClient) -> None:
        assert openai_client.has_embedding_support is True


# ============================================================================
# Chat — Ollama-Modus
# ============================================================================


class TestChatOllama:
    @pytest.mark.asyncio
    async def test_chat_delegates_to_ollama(self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock) -> None:
        result = await ollama_client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        mock_ollama.chat.assert_awaited_once()
        assert result["message"]["content"] == "Hallo von Ollama!"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_chat_passes_all_params(self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock) -> None:
        await ollama_client.chat(
            model="qwen3:32b",
            messages=[{"role": "user", "content": "Test"}],
            tools=[{"name": "tool1"}],
            temperature=0.3,
            top_p=0.8,
            stream=False,
            format_json=True,
        )

        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["top_p"] == 0.8
        assert call_kwargs.kwargs["tools"] == [{"name": "tool1"}]
        assert call_kwargs.kwargs["format_json"] is True


# ============================================================================
# Chat — Backend-Modus
# ============================================================================


class TestChatBackend:
    @pytest.mark.asyncio
    async def test_chat_uses_backend(self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock) -> None:
        result = await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        mock_backend.chat.assert_awaited_once()
        # Response sollte im Ollama-Dict-Format sein
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == "Hallo von OpenAI!"
        assert result["model"] == "gpt-4o"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_chat_converts_tool_calls(self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock) -> None:
        mock_backend.chat.return_value = MockChatResponse(
            content="",
            tool_calls=[{"name": "read_file", "arguments": {"path": "/tmp"}}],
            model="gpt-4o",
        )

        result = await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Lies /tmp"}],
        )

        assert result["message"]["tool_calls"] == [
            {"name": "read_file", "arguments": {"path": "/tmp"}}
        ]

    @pytest.mark.asyncio
    async def test_chat_converts_usage(self, openai_client: UnifiedLLMClient) -> None:
        result = await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
        )

        assert result["prompt_eval_count"] == 10
        assert result["eval_count"] == 20

    @pytest.mark.asyncio
    async def test_backend_error_becomes_ollama_error(
        self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock
    ) -> None:
        mock_backend.chat.side_effect = ConnectionError("API down")

        with pytest.raises(OllamaError) as exc_info:
            await openai_client.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
            )

        assert "LLM-Backend-Fehler" in str(exc_info.value)
        assert "API down" in str(exc_info.value)


# ============================================================================
# Embeddings
# ============================================================================


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_embed_ollama_mode(self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock) -> None:
        result = await ollama_client.embed("nomic-embed-text", "Hallo Welt")
        mock_ollama.embed.assert_awaited_once()
        assert result["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_backend_mode(self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock) -> None:
        result = await openai_client.embed("text-embedding-3-small", "Hallo Welt")
        mock_backend.embed.assert_awaited_once()
        assert result["embedding"] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_anthropic_fallback(self, mock_ollama: AsyncMock, mock_backend: AsyncMock) -> None:
        """Anthropic hat kein Embedding → Fallback auf Ollama."""
        mock_backend.embed.side_effect = NotImplementedError("No embedding support")

        client = UnifiedLLMClient(mock_ollama, backend=mock_backend)
        result = await client.embed("nomic-embed-text", "Hallo")

        # Sollte auf Ollama zurückfallen
        mock_ollama.embed.assert_awaited_once()
        assert result["embedding"] == [0.1, 0.2, 0.3]


# ============================================================================
# Meta-Methoden
# ============================================================================


class TestMetaMethods:
    @pytest.mark.asyncio
    async def test_is_available_ollama(self, ollama_client: UnifiedLLMClient) -> None:
        assert await ollama_client.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_backend(self, openai_client: UnifiedLLMClient) -> None:
        assert await openai_client.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_backend_error(self, mock_ollama: AsyncMock, mock_backend: AsyncMock) -> None:
        mock_backend.is_available.side_effect = Exception("Network error")
        client = UnifiedLLMClient(mock_ollama, backend=mock_backend)
        assert await client.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models_ollama(self, ollama_client: UnifiedLLMClient) -> None:
        models = await ollama_client.list_models()
        assert "qwen3:8b" in models

    @pytest.mark.asyncio
    async def test_list_models_backend(self, openai_client: UnifiedLLMClient) -> None:
        models = await openai_client.list_models()
        assert "gpt-4o" in models

    @pytest.mark.asyncio
    async def test_close_closes_both(self, openai_client: UnifiedLLMClient, mock_ollama: AsyncMock, mock_backend: AsyncMock) -> None:
        await openai_client.close()
        mock_backend.close.assert_awaited_once()
        mock_ollama.close.assert_awaited_once()


# ============================================================================
# Factory
# ============================================================================


class TestFactory:
    @pytest.mark.asyncio
    async def test_create_ollama_default(self) -> None:
        """Bei llm_backend_type='ollama' wird kein Backend erstellt."""
        config = MagicMock()
        config.llm_backend_type = "ollama"
        config.ollama = MagicMock()
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.timeout_seconds = 30
        config.ollama.keep_alive = "5m"

        client = UnifiedLLMClient.create(config)

        assert client.backend_type == "ollama"
        assert client._backend is None

    @pytest.mark.asyncio
    async def test_create_fallback_on_error(self) -> None:
        """Wenn Backend-Erstellung fehlschlägt, Fallback auf Ollama."""
        config = MagicMock()
        config.llm_backend_type = "openai"
        config.ollama = MagicMock()
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.timeout_seconds = 30
        config.ollama.keep_alive = "5m"

        with patch("jarvis.core.llm_backend.create_backend", side_effect=ValueError("Bad key")):
            client = UnifiedLLMClient.create(config)

        assert client.backend_type == "ollama"
        assert client._backend is None


# ============================================================================
# Planner-Kompatibilität (End-to-End Simulation)
# ============================================================================


class TestPlannerCompatibility:
    """Simuliert wie der Planner den UnifiedLLMClient nutzt."""

    @pytest.mark.asyncio
    async def test_planner_pattern_ollama(self, ollama_client: UnifiedLLMClient) -> None:
        """Der typische Planner-Code funktioniert im Ollama-Modus."""
        response = await ollama_client.chat(
            model="qwen3:32b",
            messages=[
                {"role": "system", "content": "Du bist Jarvis."},
                {"role": "user", "content": "Zeige mir Dateien in /home"},
            ],
            temperature=0.7,
            top_p=0.9,
        )

        # Planner-Pattern: response.get("message", {}).get("content", "")
        text = response.get("message", {}).get("content", "")
        assert text == "Hallo von Ollama!"

        # Planner-Pattern: tool_calls prüfen
        tool_calls = response.get("message", {}).get("tool_calls", [])
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_planner_pattern_backend(self, openai_client: UnifiedLLMClient) -> None:
        """Der typische Planner-Code funktioniert im Backend-Modus."""
        response = await openai_client.chat(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist Jarvis."},
                {"role": "user", "content": "Zeige mir Dateien in /home"},
            ],
            temperature=0.7,
            top_p=0.9,
        )

        # Exakt dasselbe Pattern wie Planner
        text = response.get("message", {}).get("content", "")
        assert text == "Hallo von OpenAI!"

        tool_calls = response.get("message", {}).get("tool_calls", [])
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_planner_error_pattern(self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock) -> None:
        """Planner fängt OllamaError — muss auch bei Backend-Fehlern funktionieren."""
        mock_backend.chat.side_effect = RuntimeError("Rate limit exceeded")

        # Planner-Pattern: except OllamaError
        try:
            await openai_client.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
            )
            assert False, "Sollte OllamaError werfen"
        except OllamaError as exc:
            assert "Rate limit exceeded" in str(exc)
