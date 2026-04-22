from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.core.llm_backend import (
    LLMBackendType,
    LLMBadRequestError,
)
from cognithor.core.vllm_backend import VLLMBackend, VLLMNotReadyError

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE_URL = "http://localhost:8000/v1"


@pytest.fixture
def backend() -> VLLMBackend:
    return VLLMBackend(base_url=BASE_URL, timeout=5)


class TestVLLMBackendBasics:
    def test_backend_type(self, backend):
        assert backend.backend_type == LLMBackendType.VLLM

    @pytest.mark.asyncio
    async def test_is_available_true_on_200(self, backend, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:8000/health",
            status_code=200,
        )
        assert await backend.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_connection_refused(self, backend):
        assert await backend.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models_from_openai_endpoint(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/models",
            status_code=200,
            json={"data": [{"id": "Qwen/Qwen3.6-27B-FP8"}]},
        )
        models = await backend.list_models()
        assert "Qwen/Qwen3.6-27B-FP8" in models

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, backend):
        await backend.close()
        await backend.close()


class TestVLLMBackendChat:
    @pytest.mark.asyncio
    async def test_chat_sends_openai_payload(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=200,
            json={
                "choices": [{"message": {"content": "Hello!"}}],
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
        resp = await backend.chat(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )
        assert resp.content == "Hello!"
        assert resp.model == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

        request = httpx_mock.get_requests()[0]
        import json as _j

        body = _j.loads(request.content)
        assert body["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert body["temperature"] == 0.7
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_chat_converts_image_paths_to_openai_vision_format(
        self, backend, httpx_mock, tmp_path
    ):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}], "model": "x"},
        )
        await backend.chat(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": "what is this?"}],
            images=[str(img)],
        )

        import json as _j

        body = _j.loads(httpx_mock.get_requests()[0].content)
        last = body["messages"][-1]
        assert isinstance(last["content"], list)
        assert any(c.get("type") == "text" for c in last["content"])
        prefix = "data:image/png;base64,"
        assert any(
            c.get("type") == "image_url" and c["image_url"]["url"].startswith(prefix)
            for c in last["content"]
        )

    @pytest.mark.asyncio
    async def test_chat_raises_bad_request_on_400(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=400,
            json={"error": {"message": "context too long"}},
        )
        with pytest.raises(LLMBadRequestError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])

    @pytest.mark.asyncio
    async def test_chat_raises_not_ready_on_5xx(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=503,
            json={"error": "model loading"},
        )
        with pytest.raises(VLLMNotReadyError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])

    @pytest.mark.asyncio
    async def test_chat_raises_not_ready_on_connection_refused(self, backend):
        with pytest.raises(VLLMNotReadyError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])
