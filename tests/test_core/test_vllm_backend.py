from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.core.llm_backend import (
    LLMBackendType,
)
from cognithor.core.vllm_backend import VLLMBackend

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
