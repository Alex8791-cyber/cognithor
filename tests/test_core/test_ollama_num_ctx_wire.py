# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tests for Sprint-23 PR#D — ``num_ctx`` end-to-end into the wire payload.

The Sprint-23 :class:`ContextProfile` system was a *no-op* before this
PR because the ``num_ctx`` value resolved by ``get_model_config()``
never reached :meth:`OllamaClient.chat` / :meth:`chat_stream` — those
methods built an ``options`` dict from the kwargs and dropped any
``num_ctx`` on the floor. These tests pin the contract: when a caller
passes ``num_ctx=...``, the value lands on the wire as
``payload.options.num_ctx``; when omitted, no override is sent so
Ollama uses the model's built-in window (previous behaviour).
"""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.config import CognithorConfig
from cognithor.core.model_router import OllamaClient


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


class _CapturingClient:
    """Pretends to be the httpx client that ``OllamaClient._ensure_client``
    returns. ``post()`` records the JSON payload; ``stream()`` returns an
    async context manager yielding one ``done`` frame.
    """

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured
        self.is_closed = False

    async def post(self, url: str, json: dict[str, Any]) -> Any:
        self._captured["url"] = url
        self._captured["payload"] = json

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "message": {"content": "ok", "role": "assistant"},
                    "done": True,
                    "eval_count": 1,
                }

        return _Resp()

    def stream(self, _method: str, _url: str, json: dict[str, Any]) -> Any:
        captured = self._captured
        captured["payload"] = json

        class _StreamCtx:
            async def __aenter__(self_inner) -> Any:
                class _Resp:
                    status_code = 200

                    async def aread(self_inner_resp) -> bytes:
                        return b""

                    async def aiter_lines(self_inner_resp):
                        import json as _json

                        yield _json.dumps(
                            {
                                "message": {"content": "ok"},
                                "done": True,
                                "eval_count": 1,
                            }
                        )

                return _Resp()

            async def __aexit__(self_inner, *_args: Any) -> None:
                return None

        return _StreamCtx()


# ---------------------------------------------------------------------------
# chat() non-streaming path
# ---------------------------------------------------------------------------


class TestChatNumCtx:
    @pytest.mark.asyncio
    async def test_num_ctx_set_lands_on_wire(self, config: CognithorConfig) -> None:
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        await client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=131072,
        )
        assert captured["payload"]["options"]["num_ctx"] == 131072

    @pytest.mark.asyncio
    async def test_num_ctx_unset_omits_key(self, config: CognithorConfig) -> None:
        # Without an explicit num_ctx, the wire payload must NOT carry
        # the key — otherwise we'd silently override the model default.
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        await client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "num_ctx" not in captured["payload"]["options"]

    @pytest.mark.asyncio
    async def test_num_ctx_coerced_to_int(self, config: CognithorConfig) -> None:
        # Some callers (e.g. config sourced from JSON / YAML) may pass a
        # numeric string. ``int(num_ctx)`` is the cheapest defence —
        # without it, Ollama returns an opaque 400 if it's a string.
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        await client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx="65536",  # type: ignore[arg-type]
        )
        assert captured["payload"]["options"]["num_ctx"] == 65536
        assert isinstance(captured["payload"]["options"]["num_ctx"], int)

    @pytest.mark.asyncio
    async def test_num_ctx_does_not_clobber_options_dict(self, config: CognithorConfig) -> None:
        # Caller supplies an extra options dict alongside num_ctx — both
        # must coexist on the wire.
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        await client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            options={"seed": 42},
            num_ctx=8192,
        )
        opts = captured["payload"]["options"]
        assert opts["num_ctx"] == 8192
        assert opts["seed"] == 42


# ---------------------------------------------------------------------------
# chat_stream() streaming path
# ---------------------------------------------------------------------------


class TestChatStreamNumCtx:
    @pytest.mark.asyncio
    async def test_stream_num_ctx_lands_on_wire(self, config: CognithorConfig) -> None:
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        async for _ in client.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=65536,
        ):
            pass
        assert captured["payload"]["options"]["num_ctx"] == 65536

    @pytest.mark.asyncio
    async def test_stream_num_ctx_unset_omits_key(self, config: CognithorConfig) -> None:
        client = OllamaClient(config)
        captured: dict[str, Any] = {}
        client._client = _CapturingClient(captured)
        async for _ in client.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
        assert "num_ctx" not in captured["payload"]["options"]
