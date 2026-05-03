# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tests for Sprint-23 PR#E — ``num_ctx`` through the LLMBackend abstraction.

PR#D wired ``num_ctx`` only into ``OllamaClient`` (the legacy direct
client). Production code goes through :class:`LLMBackend` —
Ollama / vLLM / OpenAI / Anthropic / Gemini / ClaudeCode all
implement that interface. PR#E pushes ``num_ctx`` through the
abstract ``chat`` / ``chat_stream`` signatures and lets each backend
translate per its own contract.

These tests pin the contract for each backend that *does* honour
``num_ctx`` on the wire:

* **VLLMBackend** (the Sprint-22 primary use case) — forwards as
  ``payload.extra_body.num_ctx``. The vLLM engine applies
  per-request truncation against the boot-time
  ``--max-model-len``.
* **OllamaBackend** — forwards as ``payload.options.num_ctx``
  (Ollama re-loads the KV cache for the requested window).
* **OpenAIBackend** — forwards as ``payload.extra_body.num_ctx`` so
  vLLM's OpenAI-compat server can consume it; real OpenAI ignores
  unknown extras.

Backends with model-intrinsic windows (Anthropic, Gemini, ClaudeCode)
accept the kwarg but only log it — the ``log+ignore`` contract is
documented, not pinned by a fragile log-string assertion.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Tiny shared stub: OpenAI / vLLM / Ollama all use httpx.AsyncClient
# ---------------------------------------------------------------------------


class _CapturingClient:
    """Stub httpx client that records the JSON payload of post()/stream().

    Returns a 200 response with a minimal-but-valid body for whichever
    endpoint is hit. Each backend's chat() reaches into ``message`` /
    ``choices[0].message`` / ``content`` differently — the stub
    response covers all three shapes so the same stub serves Ollama,
    OpenAI, and vLLM.
    """

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured
        self.is_closed = False

    async def post(self, url: str, json: dict[str, Any]) -> Any:
        self._captured["url"] = url
        self._captured["payload"] = json

        class _Resp:
            status_code = 200
            text = ""

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    # OpenAI / vLLM shape
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                                "role": "assistant",
                                "tool_calls": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "model": json.get("model", "stub"),
                    # Ollama shape
                    "message": {"content": "ok", "role": "assistant"},
                    "done": True,
                    "eval_count": 1,
                    "prompt_eval_count": 1,
                }

        return _Resp()

    def stream(self, _method: str, url: str, json: dict[str, Any]) -> Any:
        captured = self._captured
        captured["payload"] = json
        # Decide line shape from the URL: Ollama uses /api/chat
        # (newline-delimited JSON); OpenAI / vLLM use /chat/completions
        # (SSE ``data: …`` frames).
        is_sse = "/chat/completions" in url

        class _StreamCtx:
            async def __aenter__(self_inner) -> Any:
                class _Resp:
                    status_code = 200

                    async def aread(self_inner_resp) -> bytes:
                        return b""

                    async def aiter_lines(self_inner_resp):
                        import json as _json

                        if is_sse:
                            yield "data: " + _json.dumps(
                                {
                                    "choices": [
                                        {
                                            "delta": {"content": ""},
                                            "finish_reason": "stop",
                                        }
                                    ]
                                }
                            )
                            yield "data: [DONE]"
                        else:
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
# VLLMBackend — primary Sprint-22/Sprint-23 target
# ---------------------------------------------------------------------------


class TestVLLMBackendNumCtx:
    @pytest.mark.asyncio
    async def test_chat_forwards_num_ctx_to_extra_body(self) -> None:
        from cognithor.core.vllm_backend import VLLMBackend

        backend = VLLMBackend(base_url="http://localhost:9000")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3.6:27b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=131072,
        )
        # The Sprint-22 side-quest probe pinned 128k for arc_agi3 — this
        # asserts the value reaches vLLM via extra_body, not silently
        # dropped at the Cognithor boundary.
        assert captured["payload"]["extra_body"]["num_ctx"] == 131072

    @pytest.mark.asyncio
    async def test_chat_unset_omits_num_ctx_from_extra_body(self) -> None:
        from cognithor.core.vllm_backend import VLLMBackend

        backend = VLLMBackend(base_url="http://localhost:9000")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3.6:27b",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Without num_ctx, extra_body should not be sent at all (or
        # if sent, must not carry num_ctx).
        eb = captured["payload"].get("extra_body", {})
        assert "num_ctx" not in eb

    @pytest.mark.asyncio
    async def test_chat_stream_forwards_num_ctx(self) -> None:
        from cognithor.core.vllm_backend import VLLMBackend

        backend = VLLMBackend(base_url="http://localhost:9000")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        async for _ in backend.chat_stream(
            model="qwen3.6:27b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=65536,
        ):
            pass
        assert captured["payload"]["extra_body"]["num_ctx"] == 65536

    @pytest.mark.asyncio
    async def test_chat_num_ctx_coexists_with_video_extra(self) -> None:
        # Video uses ``extra_body.mm_processor_kwargs``; num_ctx adds a
        # sibling key. They must coexist on the wire.
        from cognithor.core.vllm_backend import VLLMBackend

        backend = VLLMBackend(base_url="http://localhost:9000")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3.6:27b",
            messages=[{"role": "user", "content": "hi"}],
            video={
                "url": "https://example.com/clip.mp4",
                "sampling": {"fps": 1.0},
            },
            num_ctx=131072,
        )
        eb = captured["payload"]["extra_body"]
        assert eb["num_ctx"] == 131072
        assert "mm_processor_kwargs" in eb


# ---------------------------------------------------------------------------
# OllamaBackend — local default
# ---------------------------------------------------------------------------


class TestOllamaBackendNumCtx:
    @pytest.mark.asyncio
    async def test_chat_forwards_num_ctx_to_options(self) -> None:
        from cognithor.core.llm_backend import OllamaBackend

        backend = OllamaBackend(base_url="http://localhost:11434")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=8192,
        )
        assert captured["payload"]["options"]["num_ctx"] == 8192

    @pytest.mark.asyncio
    async def test_chat_unset_omits_num_ctx(self) -> None:
        from cognithor.core.llm_backend import OllamaBackend

        backend = OllamaBackend(base_url="http://localhost:11434")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "num_ctx" not in captured["payload"]["options"]

    @pytest.mark.asyncio
    async def test_chat_stream_forwards_num_ctx(self) -> None:
        from cognithor.core.llm_backend import OllamaBackend

        backend = OllamaBackend(base_url="http://localhost:11434")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        async for _ in backend.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=32768,
        ):
            pass
        assert captured["payload"]["options"]["num_ctx"] == 32768


# ---------------------------------------------------------------------------
# OpenAIBackend — covers vLLM via OpenAI-compat endpoint
# ---------------------------------------------------------------------------


class TestOpenAIBackendNumCtx:
    @pytest.mark.asyncio
    async def test_chat_forwards_num_ctx_to_extra_body(self) -> None:
        from cognithor.core.llm_backend import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test", base_url="http://localhost:8000/v1")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=131072,
        )
        # vLLM's OpenAI-compat server consumes extra_body. Real OpenAI
        # ignores unknown keys — same wire shape.
        assert captured["payload"]["extra_body"]["num_ctx"] == 131072

    @pytest.mark.asyncio
    async def test_chat_unset_omits_extra_body(self) -> None:
        from cognithor.core.llm_backend import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test", base_url="http://localhost:8000/v1")
        captured: dict[str, Any] = {}
        backend._client = _CapturingClient(captured)
        await backend.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "extra_body" not in captured["payload"]


# ---------------------------------------------------------------------------
# Anthropic / Gemini / ClaudeCode — accept-and-ignore contract
# ---------------------------------------------------------------------------


class TestModelIntrinsicBackendsAcceptNumCtx:
    """The fixed-window backends must *accept* the kwarg without raising.

    We don't pin specific log-message strings (brittle). The contract
    is: kwarg is type-compatible with the abstract signature and a
    typoed call site won't blow up. Any forwarding to the wire is a
    no-op by design (model-intrinsic context window).
    """

    @pytest.mark.asyncio
    async def test_anthropic_accepts_num_ctx_kwarg(self) -> None:
        # We can't easily exercise AnthropicBackend.chat() end-to-end
        # without an API key, but we can verify the signature accepts
        # the kwarg by inspection — keeping the check at the call-site
        # contract level.
        import inspect

        from cognithor.core.llm_backend import AnthropicBackend

        sig = inspect.signature(AnthropicBackend.chat)
        assert "num_ctx" in sig.parameters
        sig_stream = inspect.signature(AnthropicBackend.chat_stream)
        assert "num_ctx" in sig_stream.parameters

    @pytest.mark.asyncio
    async def test_gemini_accepts_num_ctx_kwarg(self) -> None:
        import inspect

        from cognithor.core.llm_backend import GeminiBackend

        sig = inspect.signature(GeminiBackend.chat)
        assert "num_ctx" in sig.parameters
        sig_stream = inspect.signature(GeminiBackend.chat_stream)
        assert "num_ctx" in sig_stream.parameters

    @pytest.mark.asyncio
    async def test_claudecode_accepts_num_ctx_kwarg(self) -> None:
        import inspect

        from cognithor.core.llm_backend import ClaudeCodeBackend

        sig = inspect.signature(ClaudeCodeBackend.chat)
        assert "num_ctx" in sig.parameters
        sig_stream = inspect.signature(ClaudeCodeBackend.chat_stream)
        assert "num_ctx" in sig_stream.parameters

    @pytest.mark.asyncio
    async def test_vllm_inprocess_accepts_num_ctx_kwarg(self) -> None:
        import inspect

        from cognithor.core.vllm_inprocess_backend import VLLMInProcessBackend

        sig = inspect.signature(VLLMInProcessBackend.chat)
        assert "num_ctx" in sig.parameters
        sig_stream = inspect.signature(VLLMInProcessBackend.chat_stream)
        assert "num_ctx" in sig_stream.parameters

    @pytest.mark.asyncio
    async def test_abstract_base_includes_num_ctx(self) -> None:
        # The abstract base must declare the kwarg so any future
        # backend subclass is forced to accept it.
        import inspect

        from cognithor.core.llm_backend import LLMBackend

        sig = inspect.signature(LLMBackend.chat)
        assert "num_ctx" in sig.parameters
        sig_stream = inspect.signature(LLMBackend.chat_stream)
        assert "num_ctx" in sig_stream.parameters
