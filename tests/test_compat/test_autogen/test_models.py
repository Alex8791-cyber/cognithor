"""OpenAIChatCompletionClient — wraps cognithor.core.model_router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.compat.autogen.models import OpenAIChatCompletionClient


def test_client_stores_model_string() -> None:
    c = OpenAIChatCompletionClient(model="ollama/qwen3:8b")
    assert c.model == "ollama/qwen3:8b"


def test_client_accepts_api_key_kwarg_without_breaking() -> None:
    """AutoGen users pass api_key='...'; we accept and store but never send unless needed."""
    c = OpenAIChatCompletionClient(model="gpt-4", api_key="sk-test")
    assert c.model == "gpt-4"
    assert c._api_key == "sk-test"


def test_client_accepts_base_url_kwarg() -> None:
    c = OpenAIChatCompletionClient(model="ollama/qwen3:8b", base_url="http://localhost:11434")
    assert c._base_url == "http://localhost:11434"


@pytest.mark.asyncio
async def test_client_create_routes_through_cognithor_model_router() -> None:
    """`.create()` dispatches to cognithor.core.model_router.generate."""
    with patch(
        "cognithor.compat.autogen.models._dispatch_to_router", new_callable=AsyncMock
    ) as dispatch:
        dispatch.return_value = MagicMock(content="response", usage={"total_tokens": 5})
        c = OpenAIChatCompletionClient(model="ollama/qwen3:8b")
        result = await c.create(messages=[{"role": "user", "content": "hi"}])
        dispatch.assert_called_once()
        assert result.content == "response"
