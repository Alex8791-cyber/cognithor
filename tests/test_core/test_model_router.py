"""Tests für Model-Router und Ollama-Client.

Testet:
  - Modell-Auswahl nach Aufgabentyp und Komplexität
  - Fallback-Kette wenn Modell nicht verfügbar
  - OllamaClient HTTP-Kommunikation (gemockt)
  - Embedding-Aufrufe
  - Streaming
  - Fehlerbehandlung (Timeout, Verbindungsfehler)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.model_router import (
    ModelRouter,
    OllamaClient,
    messages_to_ollama,
)
from jarvis.models import Message, MessageRole


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path)


@pytest.fixture()
def client(config: JarvisConfig) -> OllamaClient:
    return OllamaClient(config)


@pytest.fixture()
def router(config: JarvisConfig, client: OllamaClient) -> ModelRouter:
    return ModelRouter(config, client)


# ============================================================================
# Modell-Auswahl
# ============================================================================


class TestModelSelection:
    """Testet select_model() für verschiedene Aufgabentypen."""

    def test_planning_uses_planner_model(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("planning", "high")
        assert model == config.models.planner.name

    def test_reflection_uses_planner_model(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("reflection")
        assert model == config.models.planner.name

    def test_code_uses_coder_model(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("code")
        assert model == config.models.coder.name

    def test_simple_uses_executor_model(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("simple_tool_call")
        assert model == config.models.executor.name

    def test_embedding_uses_embedding_model(
        self, router: ModelRouter, config: JarvisConfig
    ) -> None:
        model = router.select_model("embedding")
        assert model == config.models.embedding.name

    def test_general_high_uses_planner(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("general", "high")
        assert model == config.models.planner.name

    def test_general_low_uses_executor(self, router: ModelRouter, config: JarvisConfig) -> None:
        model = router.select_model("general", "low")
        assert model == config.models.executor.name


class TestModelFallback:
    """Testet Fallback wenn Modell nicht verfügbar."""

    def test_fallback_to_available_model(self, router: ModelRouter, config: JarvisConfig) -> None:
        # Simuliere dass nur das executor model verfügbar ist
        router._available_models = {config.models.executor.name}
        model = router.select_model("planning", "high")
        assert model == config.models.executor.name

    def test_no_available_models_returns_requested(self, router: ModelRouter) -> None:
        # Leere Liste → kein Fallback nötig
        router._available_models = set()
        model = router.select_model("planning")
        assert model  # Gibt das angeforderte Modell zurück

    def test_get_model_config_known(self, router: ModelRouter, config: JarvisConfig) -> None:
        cfg = router.get_model_config(config.models.planner.name)
        assert "temperature" in cfg
        assert "context_window" in cfg

    def test_get_model_config_unknown(self, router: ModelRouter) -> None:
        cfg = router.get_model_config("unknown-model")
        assert cfg["temperature"] == 0.7  # Default


# ============================================================================
# OllamaClient (gemockter HTTP)
# ============================================================================


class TestOllamaClient:
    """Tests mit gemocktem HTTP-Client."""

    @pytest.mark.asyncio
    async def test_is_available_success(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            assert await client.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_failure(self, client: OllamaClient) -> None:
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
            mock_ensure.return_value = mock_http
            assert await client.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:32b"},
                {"name": "qwen3:8b"},
                {"name": "nomic-embed-text"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            models = await client.list_models()
        assert len(models) == 3
        assert "qwen3:32b" in models

    @pytest.mark.asyncio
    async def test_chat_success(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hallo!"},
            "eval_count": 10,
        }
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            result = await client.chat(
                model="qwen3:32b",
                messages=[{"role": "user", "content": "Hallo"}],
            )
        assert result["message"]["content"] == "Hallo!"

    @pytest.mark.asyncio
    async def test_embed_success(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            embedding = await client.embed("nomic-embed-text", "test text")
        assert len(embedding) == 4
        assert embedding[0] == 0.1


# ============================================================================
# messages_to_ollama Konvertierung
# ============================================================================


class TestMessagesToOllama:
    def test_basic_conversion(self) -> None:
        msgs = [
            Message(role=MessageRole.SYSTEM, content="Du bist Jarvis."),
            Message(role=MessageRole.USER, content="Hallo"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
        ]
        result = messages_to_ollama(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["content"] == "Hi!"

    def test_tool_message(self) -> None:
        msgs = [
            Message(role=MessageRole.TOOL, content="file content", name="read_file"),
        ]
        result = messages_to_ollama(msgs)
        assert result[0]["role"] == "tool"
