from __future__ import annotations

from cognithor.config import CognithorConfig, VLLMConfig
from cognithor.core.llm_backend import create_backend
from cognithor.core.vllm_backend import VLLMBackend


class TestCreateVLLMBackend:
    def test_returns_vllm_backend_when_config_says_vllm(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, port=8000),
        )
        backend = create_backend(cfg)
        assert isinstance(backend, VLLMBackend)
        assert backend.backend_type.value == "vllm"

    def test_vllm_backend_uses_configured_port(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, port=8042),
        )
        backend = create_backend(cfg)
        assert backend._base_url == "http://localhost:8042/v1"

    def test_vllm_backend_uses_request_timeout(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, request_timeout_seconds=30),
        )
        backend = create_backend(cfg)
        assert backend._timeout == 30
