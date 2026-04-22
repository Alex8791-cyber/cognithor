from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from cognithor.core.llm_backend import (
    ChatResponse,
    LLMBadRequestError,
    VLLMNotReadyError,
)
from cognithor.core.unified_llm import BackendStatus, UnifiedLLMClient
from cognithor.utils.circuit_breaker import CircuitState


@pytest.fixture
def mock_vllm_backend() -> AsyncMock:
    mock = AsyncMock()
    mock.backend_type = "vllm"
    return mock


@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    return AsyncMock()


class TestBreakerWiring:
    @pytest.mark.asyncio
    async def test_three_consecutive_failures_open_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
        )
        for _ in range(3):
            with contextlib.suppress(Exception):
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert client.vllm_breaker.state == CircuitState.open
        assert client.backend_status == BackendStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_bad_request_error_is_excluded_from_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = LLMBadRequestError("context too long")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
        )
        for _ in range(5):
            with contextlib.suppress(LLMBadRequestError):
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert client.vllm_breaker.state == CircuitState.closed
        assert client.backend_status == BackendStatus.OK

    @pytest.mark.asyncio
    async def test_half_open_probe_success_closes_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=0.05,
        )
        for _ in range(3):
            with contextlib.suppress(Exception):
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert client.vllm_breaker.state == CircuitState.open

        await asyncio.sleep(0.1)
        mock_vllm_backend.chat.side_effect = None
        mock_vllm_backend.chat.return_value = ChatResponse(content="ok", model="x")

        await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert client.vllm_breaker.state == CircuitState.closed
        assert client.backend_status == BackendStatus.OK


class TestFailFlowDispatch:
    @pytest.mark.asyncio
    async def test_text_request_falls_back_to_ollama_when_vllm_degraded(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        mock_ollama_client.chat = AsyncMock(
            return_value={"message": {"content": "fallback answer"}}
        )

        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=60.0,
        )
        for _ in range(3):
            with contextlib.suppress(Exception):
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])

        result = await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert "fallback answer" in str(result)

    @pytest.mark.asyncio
    async def test_image_request_hard_errors_when_vllm_degraded(
        self, mock_vllm_backend, mock_ollama_client, tmp_path
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG")

        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=60.0,
        )
        for _ in range(3):
            with contextlib.suppress(Exception):
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])

        with pytest.raises(VLLMNotReadyError):
            await client.chat(
                model="x",
                messages=[{"role": "user", "content": "what is this?"}],
                images=[str(img)],
            )
