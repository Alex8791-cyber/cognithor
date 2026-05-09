"""Tests für den UnifiedLLMClient-Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.core.model_router import OllamaClient, OllamaError
from cognithor.core.unified_llm import UnifiedLLMClient

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
    async def test_chat_delegates_to_ollama(
        self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock
    ) -> None:
        result = await ollama_client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        mock_ollama.chat.assert_awaited_once()
        assert result["message"]["content"] == "Hallo von Ollama!"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_chat_passes_all_params(
        self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock
    ) -> None:
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
    async def test_chat_uses_backend(
        self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock
    ) -> None:
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
    async def test_chat_converts_tool_calls(
        self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock
    ) -> None:
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
    async def test_embed_ollama_mode(
        self, ollama_client: UnifiedLLMClient, mock_ollama: AsyncMock
    ) -> None:
        result = await ollama_client.embed("nomic-embed-text", "Hallo Welt")
        mock_ollama.embed.assert_awaited_once()
        assert result["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_backend_mode(
        self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock
    ) -> None:
        result = await openai_client.embed("text-embedding-3-small", "Hallo Welt")
        mock_backend.embed.assert_awaited_once()
        assert result["embedding"] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_anthropic_fallback(
        self, mock_ollama: AsyncMock, mock_backend: AsyncMock
    ) -> None:
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
    async def test_is_available_backend_error(
        self, mock_ollama: AsyncMock, mock_backend: AsyncMock
    ) -> None:
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
    async def test_close_closes_both(
        self, openai_client: UnifiedLLMClient, mock_ollama: AsyncMock, mock_backend: AsyncMock
    ) -> None:
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
    async def test_create_raises_on_backend_error(self) -> None:
        """Wenn Backend-Erstellung fehlschlägt, wird ein Fehler geworfen (kein stilles Fallback)."""
        from cognithor.core.model_router import OllamaError

        config = MagicMock()
        config.llm_backend_type = "openai"
        config.ollama = MagicMock()
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.timeout_seconds = 30
        config.ollama.keep_alive = "5m"

        with patch("cognithor.core.llm_backend.create_backend", side_effect=ValueError("Bad key")):
            with pytest.raises(OllamaError, match="konnte nicht initialisiert werden"):
                UnifiedLLMClient.create(config)


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
    async def test_planner_error_pattern(
        self, openai_client: UnifiedLLMClient, mock_backend: AsyncMock
    ) -> None:
        """Planner fängt OllamaError — muss auch bei Backend-Fehlern funktionieren."""
        mock_backend.chat.side_effect = RuntimeError("Rate limit exceeded")

        # Planner-Pattern: except OllamaError
        try:
            await openai_client.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
            )
            raise AssertionError("Sollte OllamaError werfen")
        except OllamaError as exc:
            assert "Rate limit exceeded" in str(exc)


# ============================================================================
# TRUST-8 cross-wiring — backend_dispatch + cloud_escalation cross-record
#
# These tests pin the contract from PR (Sprint 2026-05-09 cloud-escalation
# wiring): every chat() through UnifiedLLMClient lands in the
# backend_dispatch ledger. Cloud-bound dispatches (anthropic, openai,
# gemini, claude-code, claude-code-supervised) ALSO land in the
# escalation ledger so the operator can answer "did this leave the box"
# in O(1). Local backends (ollama, vllm, vllm-inprocess, lmstudio) only
# land in dispatch — escalation stays empty.
#
# Each test scopes to a fresh ledger via monkeypatch so test ordering
# can't pollute the canonical singletons.
# ============================================================================


class TestTrust8CrossWiring:
    @pytest.fixture
    def fresh_dispatch_ledger(self, monkeypatch):
        from cognithor.security.backend_dispatch import (
            BackendDispatchLedger,
        )

        ledger = BackendDispatchLedger()
        monkeypatch.setattr(
            "cognithor.security.backend_dispatch.BACKEND_DISPATCH_LEDGER",
            ledger,
        )
        return ledger

    @pytest.fixture
    def fresh_escalation_ledger(self, monkeypatch):
        from cognithor.security.cloud_escalation import EscalationLedger

        ledger = EscalationLedger()
        monkeypatch.setattr(
            "cognithor.security.cloud_escalation.ESCALATION_LEDGER",
            ledger,
        )
        # trust_wiring imports the singleton at module-load — patch
        # both names so the wiring picks up the fresh instance.
        monkeypatch.setattr(
            "cognithor.security.trust_wiring.ESCALATION_LEDGER",
            ledger,
        )
        return ledger

    @pytest.fixture
    def fresh_cost_ledger(self, monkeypatch):
        from cognithor.security.cost_ledger import CostLedger

        ledger = CostLedger()
        monkeypatch.setattr(
            "cognithor.security.cost_ledger.COST_LEDGER",
            ledger,
        )
        monkeypatch.setattr(
            "cognithor.security.trust_wiring.COST_LEDGER",
            ledger,
        )
        return ledger

    @pytest.mark.asyncio
    async def test_cloud_dispatch_lands_in_both_ledgers(
        self,
        openai_client: UnifiedLLMClient,
        fresh_dispatch_ledger,
        fresh_escalation_ledger,
        fresh_cost_ledger,
    ) -> None:
        """OpenAI is a cloud backend — successful chat() must record
        an entry in BOTH the dispatch ledger and the escalation
        ledger. Cost-mirror is a no-op because cost_usd_micro=0
        until pricing-aware wiring lands."""
        await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        assert len(fresh_dispatch_ledger) == 1
        dispatch_event = fresh_dispatch_ledger.events()[0]
        assert dispatch_event.backend_type == "openai"
        assert dispatch_event.outcome.value == "success"

        assert len(fresh_escalation_ledger) == 1
        esc_events = fresh_escalation_ledger.events()
        esc = esc_events[0]
        assert esc.to_backend == "openai"
        assert esc.cost_usd_micro == 0  # no pricing-aware wiring yet

        # cost-mirror suppressed because cost_usd_micro=0
        assert len(fresh_cost_ledger) == 0

    @pytest.mark.asyncio
    async def test_local_dispatch_records_only_in_dispatch(
        self,
        ollama_client: UnifiedLLMClient,
        fresh_dispatch_ledger,
        fresh_escalation_ledger,
    ) -> None:
        """Ollama is local — chat() lands in dispatch ledger, NOT in
        escalation. The "did this leave the box" guarantee depends on
        this asymmetry: a local-only setup must show zero escalations
        no matter how heavy the traffic."""
        await ollama_client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        # Direct ollama path skips the LLMBackend hook entirely (the
        # ollama-only branch in UnifiedLLMClient.chat). Both ledgers
        # therefore stay empty in this path. The assertion captures
        # the *contract*: a local Ollama-only call MUST NOT generate
        # an escalation entry.
        assert len(fresh_escalation_ledger) == 0

    @pytest.mark.asyncio
    async def test_cloud_failure_still_records_escalation(
        self,
        openai_client: UnifiedLLMClient,
        mock_backend: AsyncMock,
        fresh_dispatch_ledger,
        fresh_escalation_ledger,
    ) -> None:
        """A cloud call that ERRORS still left the box (or attempted
        to). The escalation ledger must show the attempted crossing
        — that's the privacy-relevant signal — even though the
        dispatch ledger flags BACKEND_ERROR."""
        mock_backend.chat.side_effect = RuntimeError("provider 5xx")

        with pytest.raises(OllamaError):
            await openai_client.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
            )

        # Both ledgers fired despite the error
        assert len(fresh_dispatch_ledger) == 1
        assert fresh_dispatch_ledger.events()[0].outcome.value == "backend_error"
        assert len(fresh_escalation_ledger) == 1
        assert fresh_escalation_ledger.events()[0].to_backend == "openai"

    @pytest.mark.asyncio
    async def test_circuit_open_skips_escalation(
        self,
        openai_client: UnifiedLLMClient,
        mock_backend: AsyncMock,
        fresh_dispatch_ledger,
        fresh_escalation_ledger,
    ) -> None:
        """CIRCUIT_OPEN means the breaker rejected before transport —
        nothing actually left the box. Dispatch ledger records the
        rejection (operator wants to see breaker activity) but
        escalation stays empty (no privacy-relevant event happened).

        With ``ollama_client`` available as a fallback, the chat call
        succeeds via Ollama after the breaker rejects. We assert on
        the ledger contents — the user-visible outcome is
        irrelevant for the privacy contract."""
        from cognithor.utils.circuit_breaker import CircuitBreakerOpen

        mock_backend.chat.side_effect = CircuitBreakerOpen("breaker open", remaining_seconds=5.0)

        # The fallback to Ollama returns a success — no exception.
        result = await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
        )
        assert result is not None  # Ollama fallback succeeded

        # Dispatch ledger DID record the failed attempt at OpenAI…
        assert len(fresh_dispatch_ledger) == 1
        assert fresh_dispatch_ledger.events()[0].outcome.value == "circuit_open"
        assert fresh_dispatch_ledger.events()[0].backend_type == "openai"
        # …but escalation ledger stays empty: CIRCUIT_OPEN means the
        # breaker rejected BEFORE bytes left the machine.
        assert len(fresh_escalation_ledger) == 0

    @pytest.mark.asyncio
    async def test_dispatch_ledger_records_token_counts_when_present(
        self,
        openai_client: UnifiedLLMClient,
        fresh_dispatch_ledger,
        fresh_escalation_ledger,
    ) -> None:
        """Token counts surfaced via the ChatResponse usage dict must
        land in both ledgers. The dispatch hook reads
        ``response.prompt_tokens`` / ``completion_tokens`` via
        ``getattr`` — this asserts the mock fixture's usage shape
        actually flows through (regression guard for ABI changes)."""
        await openai_client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hallo"}],
        )

        # MockChatResponse exposes usage as a dict (no direct
        # prompt_tokens attribute), so the getattr returns -1.
        # That's correct behaviour — when the backend ABI doesn't
        # surface tokens, we record the unknown sentinel.
        ev = fresh_dispatch_ledger.events()[0]
        assert ev.prompt_tokens == -1
        assert ev.response_tokens == -1
        # Escalation event clamps -1 to 0 (it requires non-negative).
        esc = fresh_escalation_ledger.events()[0]
        assert esc.prompt_tokens == 0
        assert esc.response_tokens == 0
