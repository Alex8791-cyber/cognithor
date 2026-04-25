"""OpenAIChatCompletionClient — autogen_ext.models.openai compat wrapper.

Routes calls into Cognithor's existing model router so the 16 supported
providers transparently back AutoGen-shaped client calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _ChatCompletionResponse:
    content: str
    usage: dict[str, int]


async def _dispatch_to_router(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None,
    base_url: str | None,
    extra: dict[str, Any],
) -> _ChatCompletionResponse:
    """Forward to cognithor.core.model_router. Imported lazily to avoid circular deps."""
    # This stays narrow on purpose: the real model_router has many entry points,
    # we use the simple `generate` path which is shared by cognithor.crew agents.
    from cognithor.core import model_router

    text = await model_router.generate(  # type: ignore[attr-defined]
        model=model,
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        **extra,
    )
    return _ChatCompletionResponse(content=str(text), usage={"total_tokens": 0})


class OpenAIChatCompletionClient:
    """OpenAI-shaped chat-completion client backed by Cognithor's model router."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._extra = kwargs

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> _ChatCompletionResponse:
        return await _dispatch_to_router(
            model=self.model,
            messages=messages,
            api_key=self._api_key,
            base_url=self._base_url,
            extra={**self._extra, **kwargs},
        )


__all__ = ["OpenAIChatCompletionClient"]
