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

from cognithor.config import CognithorConfig
from cognithor.core.model_router import (
    ModelRouter,
    OllamaClient,
    messages_to_ollama,
)
from cognithor.models import Message, MessageRole


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


@pytest.fixture()
def client(config: CognithorConfig) -> OllamaClient:
    return OllamaClient(config)


@pytest.fixture()
def router(config: CognithorConfig, client: OllamaClient) -> ModelRouter:
    return ModelRouter(config, client)


# ============================================================================
# Modell-Auswahl
# ============================================================================


class TestModelSelection:
    """Testet select_model() für verschiedene Aufgabentypen."""

    def test_planning_uses_planner_model(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        model = router.select_model("planning", "high")
        assert model == config.models.planner.name

    def test_reflection_uses_planner_model(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        model = router.select_model("reflection")
        assert model == config.models.planner.name

    def test_code_uses_coder_fast_by_default(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        model = router.select_model("code")
        assert model == config.models.coder_fast.name

    def test_code_high_complexity_uses_coder(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        model = router.select_model("code", "high")
        assert model == config.models.coder.name

    def test_simple_uses_executor_model(self, router: ModelRouter, config: CognithorConfig) -> None:
        model = router.select_model("simple_tool_call")
        assert model == config.models.executor.name

    def test_embedding_uses_embedding_model(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        model = router.select_model("embedding")
        assert model == config.models.embedding.name

    def test_general_high_uses_planner(self, router: ModelRouter, config: CognithorConfig) -> None:
        model = router.select_model("general", "high")
        assert model == config.models.planner.name

    def test_general_low_uses_executor(self, router: ModelRouter, config: CognithorConfig) -> None:
        model = router.select_model("general", "low")
        assert model == config.models.executor.name


class TestModelFallback:
    """Testet Fallback wenn Modell nicht verfügbar."""

    def test_fallback_to_available_model(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        # Simuliere dass nur das executor model verfügbar ist
        router._available_models = {config.models.executor.name}
        model = router.select_model("planning", "high")
        assert model == config.models.executor.name

    def test_no_available_models_returns_requested(self, router: ModelRouter) -> None:
        # Leere Liste → kein Fallback nötig
        router._available_models = set()
        model = router.select_model("planning")
        assert model  # Gibt das angeforderte Modell zurück

    def test_get_model_config_known(self, router: ModelRouter, config: CognithorConfig) -> None:
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
# PII redactor integration in OllamaClient
# ============================================================================


class TestOllamaClientPIIRedactor:
    """Verifies PII redactor plumbs through OllamaClient.chat() + chat_stream()."""

    @pytest.mark.asyncio
    async def test_default_config_no_redaction(self, client: OllamaClient) -> None:
        """With default config (pii_redactor.enabled=False) the original
        message must reach the HTTP payload unchanged."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await client.chat(
                model="qwen3:32b",
                messages=[{"role": "user", "content": "mail me at alice@foo.com"}],
            )
            sent_payload = mock_http.post.call_args.kwargs["json"]
            assert sent_payload["messages"][0]["content"] == "mail me at alice@foo.com"

    @pytest.mark.asyncio
    async def test_enabled_redacts_before_send(self, config: CognithorConfig) -> None:
        """When enabled, PII is redacted from the payload that reaches
        Ollama — the raw email never leaves the process."""
        from cognithor.config import PIIRedactorConfig

        config.security.pii_redactor = PIIRedactorConfig(enabled=True)
        pii_client = OllamaClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        with patch.object(pii_client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await pii_client.chat(
                model="qwen3:32b",
                messages=[
                    {"role": "user", "content": "my email is alice@foo.com ok?"},
                ],
            )
            sent_payload = mock_http.post.call_args.kwargs["json"]
            content = sent_payload["messages"][0]["content"]
            assert "alice@foo.com" not in content
            assert "[REDACTED:email]" in content

    @pytest.mark.asyncio
    async def test_category_filter_respected(self, config: CognithorConfig) -> None:
        """Config's categories list narrows what gets redacted."""
        from cognithor.config import PIIRedactorConfig

        config.security.pii_redactor = PIIRedactorConfig(enabled=True, categories=["email"])
        pii_client = OllamaClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        with patch.object(pii_client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await pii_client.chat(
                model="qwen3:32b",
                messages=[
                    {
                        "role": "user",
                        "content": "email alice@foo.com ssn 123-45-6789",
                    },
                ],
            )
            content = mock_http.post.call_args.kwargs["json"]["messages"][0]["content"]
            # Email redacted, SSN preserved (not in categories list).
            assert "alice@foo.com" not in content
            assert "123-45-6789" in content

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_messages(self, config: CognithorConfig) -> None:
        """The caller's messages list must not be mutated in place."""
        from cognithor.config import PIIRedactorConfig

        config.security.pii_redactor = PIIRedactorConfig(enabled=True)
        pii_client = OllamaClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        caller_messages = [{"role": "user", "content": "alice@foo.com"}]
        with patch.object(pii_client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await pii_client.chat(model="qwen3:32b", messages=caller_messages)
        # Caller's dict must be untouched.
        assert caller_messages[0]["content"] == "alice@foo.com"


# ============================================================================
# Multimodal image attachments (VLM routing)
# ============================================================================


class TestOllamaClientImageAttachments:
    """``chat(images=...)`` must base64-encode files and attach them to the
    last user message, per Ollama's multimodal chat API spec."""

    @pytest.mark.asyncio
    async def test_encodes_path_and_attaches_to_last_user_message(
        self, client: OllamaClient, tmp_path
    ) -> None:
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await client.chat(
                model="qwen3.6:27b",
                messages=[
                    {"role": "system", "content": "you are helpful"},
                    {"role": "user", "content": "what's in this?"},
                ],
                images=[str(img)],
            )
            payload = mock_http.post.call_args.kwargs["json"]
        # The last user message must carry a non-empty base64 images list.
        user_msg = payload["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg.get("images"), list)
        assert len(user_msg["images"]) == 1
        assert user_msg["images"][0].isprintable()  # base64 is printable ASCII

    @pytest.mark.asyncio
    async def test_unreadable_path_is_skipped_not_raised(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"role": "assistant", "content": "ok"}}
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            # Bad path — must not raise; images field simply ends up absent.
            await client.chat(
                model="qwen3.6:27b",
                messages=[{"role": "user", "content": "hi"}],
                images=["/does/not/exist.png"],
            )
            payload = mock_http.post.call_args.kwargs["json"]
        assert payload["messages"][0].get("images", []) == []

    @pytest.mark.asyncio
    async def test_caller_messages_not_mutated(self, client: OllamaClient, tmp_path) -> None:
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG")
        caller = [{"role": "user", "content": "hi"}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"role": "assistant", "content": "ok"}}
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_http
            await client.chat(model="qwen3.6:27b", messages=caller, images=[str(img)])
        # Original dict from caller must be untouched.
        assert "images" not in caller[0]


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
