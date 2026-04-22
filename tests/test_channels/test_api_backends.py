from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cognithor.channels.backends_api import build_backends_app
from cognithor.config import CognithorConfig, VLLMConfig


@pytest.fixture
def client_with_vllm_enabled():
    cfg = CognithorConfig(
        llm_backend_type="ollama",
        vllm=VLLMConfig(enabled=True),
    )
    app = build_backends_app(config=cfg)
    return TestClient(app), cfg


class TestBackendsList:
    def test_lists_all_backends_with_status(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        r = client.get("/api/backends")
        assert r.status_code == 200
        data = r.json()
        assert data["active"] == "ollama"
        names = {b["name"] for b in data["backends"]}
        assert "ollama" in names
        assert "vllm" in names


class TestVLLMStatus:
    def test_status_returns_current_vllm_state(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import (
            DockerInfo,
            HardwareInfo,
            VLLMState,
        )

        with patch("cognithor.core.vllm_orchestrator.VLLMOrchestrator.status") as mock:
            mock.return_value = VLLMState(
                hardware_ok=True,
                hardware_info=HardwareInfo("RTX 5090", 32, (12, 0)),
                docker_ok=True,
                docker_info=DockerInfo(True, "26.0.0", True),
                image_pulled=False,
                container_running=False,
                current_model=None,
            )
            r = client.get("/api/backends/vllm/status")
        assert r.status_code == 200
        data = r.json()
        assert data["hardware_ok"] is True
        assert data["hardware_info"]["gpu_name"] == "RTX 5090"
        assert data["hardware_info"]["vram_gb"] == 32
        assert data["hardware_info"]["compute_capability"] == "12.0"
        assert data["docker_ok"] is True
        assert data["container_running"] is False
