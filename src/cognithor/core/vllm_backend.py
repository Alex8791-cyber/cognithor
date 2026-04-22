"""vLLM backend — OpenAI-compatible LLMBackend adapter.

vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint.
This class adapts it to Cognithor's LLMBackend ABC with image-payload
conversion for vision models.

See spec: docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendError,
    LLMBackendType,
    VLLMNotReadyError,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


class VLLMBackend(LLMBackend):
    """vLLM OpenAI-compat adapter."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.VLLM

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def is_available(self) -> bool:
        """Ping /health (NOT /v1/health — vLLM exposes /health at server root)."""
        health_url = self._base_url.rsplit("/v1", 1)[0] + "/health"
        client = await self._ensure_client()
        try:
            r = await client.get(health_url)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        client = await self._ensure_client()
        try:
            r = await client.get(f"{self._base_url}/models")
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPStatusError as exc:
            raise LLMBackendError(f"vLLM /models failed: {exc}") from exc
        except httpx.RequestError as exc:
            raise VLLMNotReadyError(f"vLLM not reachable: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # Stubs — implemented in Tasks 14-16
    async def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    async def chat_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def embed(self, *args: Any, **kwargs: Any) -> EmbedResponse:
        raise NotImplementedError
