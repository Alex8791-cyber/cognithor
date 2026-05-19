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
            if isinstance(existing, list):
                # Pre-existing list content (e.g. a prior video attachment in
                # the same turn): extract the text item and preserve all other
                # non-text items so they are not silently dropped.
                text_part = next(
                    (c["text"] for c in existing if c.get("type") == "text"),
                    "",
                )
                preserved_items = [c for c in existing if c.get("type") != "text"]
            else:
                text_part = existing if isinstance(existing, str) else ""
                preserved_items = []
            content_list: list[dict[str, Any]] = list(preserved_items)
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


def _attach_video_to_last_user(
    messages: list[dict[str, Any]],
    video: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a single video to the last user message and build the
    mm_processor_kwargs payload for vLLM's extra_body.

    Args:
        messages: Ollama-shaped chat messages. Not mutated.
        video: ``{"url": str, "sampling": {"fps": float} | {"num_frames": int}}``

    Returns:
        ``(new_messages, extra_body_update)`` where
        - ``new_messages``: a fresh list with the last user message's content
          replaced by a list of content items: ``[video_url, (optional) text]``
        - ``extra_body_update``: ``{"mm_processor_kwargs": {"video": <sampling>}}``
          ready to merge into the outgoing chat-completion body.
    """
    new_messages = [dict(m) for m in messages]

    # Find last user message index (create one if none exists)
    last_idx: int | None = None
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            last_idx = i
            break
    if last_idx is None:
        new_messages.append({"role": "user", "content": ""})
        last_idx = len(new_messages) - 1

    existing = new_messages[last_idx].get("content", "")
    if isinstance(existing, list):
        # Pre-existing list content (e.g. prior image attachment in same turn):
        # extract the text item and preserve all other non-text items.
        text_part = next(
            (c["text"] for c in existing if c.get("type") == "text"),
            "",
        )
        preserved_items = [c for c in existing if c.get("type") != "text"]
    else:
        text_part = existing if isinstance(existing, str) else ""
        preserved_items = []

    content_items: list[dict[str, Any]] = [
        {"type": "video_url", "video_url": {"url": video["url"]}},
        *preserved_items,
    ]
    if text_part:
        content_items.append({"type": "text", "text": text_part})

    new_messages[last_idx] = {**new_messages[last_idx], "content": content_items}

    extra_body = {"mm_processor_kwargs": {"video": video["sampling"]}}
    return new_messages, extra_body


class VLLMBackend(LLMBackend):
    """vLLM OpenAI-compat adapter."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
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
        video: dict[str, Any] | None = None,
        num_ctx: int | None = None,
    ) -> ChatResponse:
        """Send a chat-completion request to vLLM.

        Sprint-23: ``num_ctx`` is forwarded as ``extra_body.num_ctx`` so
        the vLLM engine can apply per-request truncation. The server's
        *physical* context window is fixed at boot time via
        ``--max-model-len`` (Sprint-22 probe verified Qwen3.6:27b at
        128k); any value beyond that is capped server-side.

        Raises:
            LLMBadRequestError: on HTTP 400 (excluded from circuit breaker).
            VLLMNotReadyError: on HTTP 5xx or connection failure (counts toward breaker).
            LLMBackendError: on HTTP 4xx (other than 400).
        """
        if images:
            messages = _attach_images_to_last_user(messages, images)

        extra_body: dict[str, Any] = {}
        if video is not None:
            messages, video_extra = _attach_video_to_last_user(messages, video)
            extra_body.update(video_extra)
        if num_ctx is not None:
            extra_body["num_ctx"] = int(num_ctx)

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
        if extra_body:
            payload["extra_body"] = extra_body

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
        video: dict[str, Any] | None = None,
        num_ctx: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from vLLM. Parses OpenAI SSE format.

        ``video`` is a dict ``{url, sampling}`` per the video-input spec
        (2026-04-23). Exactly zero or one video per turn. Streaming responses
        for video are structurally identical to text responses — vLLM returns
        SSE tokens as it decodes.

        ``num_ctx`` semantics match :meth:`chat`: forwarded as
        ``extra_body.num_ctx`` for vLLM-level per-request truncation.
        """
        if images:
            messages = _attach_images_to_last_user(messages, images)

        extra_body: dict[str, Any] = {}
        if video is not None:
            messages, video_extra = _attach_video_to_last_user(messages, video)
            extra_body.update(video_extra)
        if num_ctx is not None:
            extra_body["num_ctx"] = int(num_ctx)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if extra_body:
            payload["extra_body"] = extra_body
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
