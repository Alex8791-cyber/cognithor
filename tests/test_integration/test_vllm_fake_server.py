"""End-to-end test: VLLMBackend against a FastAPI impersonating vLLM's OpenAI API.

No GPU, no Docker, no real vLLM. Verifies the full request-response round-trip
on the HTTP layer with real (mocked-server) sockets.
"""

from __future__ import annotations

import asyncio
import json as _json
import socket

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cognithor.core.vllm_backend import VLLMBackend


def _build_fake_vllm_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if body.get("stream"):

            async def gen():
                for chunk in ["Hel", "lo ", "world"]:
                    payload = _json.dumps({"choices": [{"delta": {"content": chunk}}]})
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "choices": [{"message": {"content": "echo: " + body["messages"][-1]["content"]}}],
            "model": body["model"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "fake-model"}]}

    return app


@pytest.fixture
async def fake_server():
    """Start the fake server on an ephemeral port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(
        _build_fake_vllm_app(),
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)
    yield port
    server.should_exit = True
    await task


class TestVLLMBackendEndToEnd:
    @pytest.mark.asyncio
    async def test_full_chat_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            resp = await backend.chat(
                model="fake-model",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert resp.content == "echo: hello"
            assert resp.model == "fake-model"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_full_stream_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            chunks = []
            async for p in backend.chat_stream(
                model="fake-model",
                messages=[{"role": "user", "content": "stream me"}],
            ):
                chunks.append(p)
            assert "".join(chunks) == "Hello world"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_is_available_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            assert await backend.is_available() is True
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_models_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            models = await backend.list_models()
            assert "fake-model" in models
        finally:
            await backend.close()
