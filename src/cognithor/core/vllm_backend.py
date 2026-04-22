"""vLLM backend — OpenAI-compatible LLMBackend adapter.

vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint.
This class adapts it to Cognithor's LLMBackend ABC with image-payload
conversion for vision models.

See spec: docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md
"""

from __future__ import annotations

import base64
import json as _json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendError,
    LLMBackendType,
    LLMBadRequestError,
    VLLMNotReadyError,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


def _encode_image_to_data_url(path: str) -> str | None:
    """Read an image file, return OpenAI-vision data-URL string. None if unreadable."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        data = p.read_bytes()
    except OSError:
        return None

    suffix = p.suffix.lower().lstrip(".")
    mime_map = {
        "png": "png",
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "webp": "webp",
        "gif": "gif",
        "bmp": "bmp",
    }
    mime = mime_map.get(suffix, "png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _attach_images_to_last_user(
    messages: list[dict[str, Any]],
    images: list[str],
) -> list[dict[str, Any]]:
    """Return a NEW messages list with images attached to the last user message
    in OpenAI-vision format. Never mutates the caller's list."""
    if not images:
        return list(messages)

    encoded = [e for e in (_encode_image_to_data_url(p) for p in images) if e]
    if not encoded:
        return list(messages)

    new_messages = [dict(m) for m in messages]
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            existing = new_messages[i].get("content")
            text_part = existing if isinstance(existing, str) else ""
            content_list: list[dict[str, Any]] = []
            if text_part:
                content_list.append({"type": "text", "text": text_part})
            for url in encoded:
                content_list.append({"type": "image_url", "image_url": {"url": url}})
            new_messages[i] = {**new_messages[i], "content": content_list}
            break
    else:
        content_list = [{"type": "image_url", "image_url": {"url": u}} for u in encoded]
        new_messages.append({"role": "user", "content": content_list})
    return new_messages


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

    # chat() implemented in Task 14; chat_stream/embed in Tasks 15-16
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Send a chat-completion request to vLLM.

        Raises:
            LLMBadRequestError: on HTTP 400 (excluded from circuit breaker).
            VLLMNotReadyError: on HTTP 5xx or connection failure (counts toward breaker).
            LLMBackendError: on HTTP 4xx (other than 400).
        """
        if images:
            messages = _attach_images_to_last_user(messages, images)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            payload["tools"] = tools
        if format_json:
            payload["response_format"] = {"type": "json_object"}

        client = await self._ensure_client()
        try:
            r = await client.post(f"{self._base_url}/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise VLLMNotReadyError(
                f"vLLM not reachable: {exc}",
                recovery_hint="Check vLLM container is running.",
            ) from exc

        if r.status_code == 400:
            raise LLMBadRequestError(
                f"vLLM rejected the request: {r.text[:200]}",
                status_code=400,
            )
        if r.status_code >= 500:
            raise VLLMNotReadyError(
                f"vLLM returned {r.status_code}: {r.text[:200]}",
                status_code=r.status_code,
                recovery_hint="vLLM may still be loading the model.",
            )
        if r.status_code >= 400:
            raise LLMBackendError(
                f"vLLM returned {r.status_code}: {r.text[:200]}",
                status_code=r.status_code,
            )

        data = r.json()
        first_choice = data.get("choices", [{}])[0]
        content = first_choice.get("message", {}).get("content", "")
        tool_calls = first_choice.get("message", {}).get("tool_calls")
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", model),
            usage=data.get("usage"),
            raw=data,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        images: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from vLLM. Parses OpenAI SSE format."""
        if images:
            messages = _attach_images_to_last_user(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        client = await self._ensure_client()
        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as r:
                if r.status_code >= 500:
                    raise VLLMNotReadyError(f"vLLM streaming returned {r.status_code}")
                if r.status_code == 400:
                    raise LLMBadRequestError(f"vLLM rejected stream request: {r.status_code}")
                if r.status_code >= 400:
                    raise LLMBackendError(f"vLLM streaming returned {r.status_code}")

                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        return
                    try:
                        event = _json.loads(payload_str)
                    except _json.JSONDecodeError:
                        continue
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.RequestError as exc:
            raise VLLMNotReadyError(f"vLLM stream not reachable: {exc}") from exc

    async def embed(self, model: str, text: str) -> EmbedResponse:
        """Send an embedding request to vLLM.

        Raises:
            LLMBadRequestError: on HTTP 400 (excluded from circuit breaker).
            VLLMNotReadyError: on HTTP 5xx or connection failure (counts toward breaker).
            LLMBackendError: on HTTP 4xx (other than 400) or missing data.
        """
        client = await self._ensure_client()
        try:
            r = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": model, "input": text},
            )
        except httpx.RequestError as exc:
            raise VLLMNotReadyError(f"vLLM embed not reachable: {exc}") from exc

        if r.status_code == 400:
            raise LLMBadRequestError(f"vLLM embed rejected: {r.text[:200]}")
        if r.status_code >= 500:
            raise VLLMNotReadyError(f"vLLM embed 5xx: {r.status_code}")
        if r.status_code >= 400:
            raise LLMBackendError(f"vLLM embed: {r.status_code}")

        data = r.json()
        items = data.get("data", [])
        if not items:
            raise LLMBackendError("vLLM embed returned no data")
        return EmbedResponse(
            embedding=items[0].get("embedding", []),
            model=data.get("model", model),
        )
