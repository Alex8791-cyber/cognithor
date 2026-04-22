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


class TestVLLMActions:
    def test_check_hardware_delegates_to_orchestrator(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import HardwareInfo

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.check_hardware",
            return_value=HardwareInfo("RTX 5090", 32, (12, 0)),
        ):
            r = client.post("/api/backends/vllm/check-hardware")
        assert r.status_code == 200
        assert r.json()["gpu_name"] == "RTX 5090"
        assert r.json()["vram_gb"] == 32
        assert r.json()["compute_capability"] == "12.0"

    def test_check_hardware_returns_503_on_no_gpu(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.llm_backend import VLLMHardwareError

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.check_hardware",
            side_effect=VLLMHardwareError("No GPU", recovery_hint="Install NVIDIA driver"),
        ):
            r = client.post("/api/backends/vllm/check-hardware")
        assert r.status_code == 503
        body = r.json()
        assert "No GPU" in body["detail"]["message"]
        assert body["detail"]["recovery_hint"] == "Install NVIDIA driver"

    def test_start_container_accepts_model(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import ContainerInfo

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.start_container",
            return_value=ContainerInfo("abc123", 8000, "Qwen/Qwen3.6-27B-FP8"),
        ):
            r = client.post(
                "/api/backends/vllm/start",
                json={"model": "Qwen/Qwen3.6-27B-FP8"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["container_id"] == "abc123"
        assert data["port"] == 8000
        assert data["model"] == "Qwen/Qwen3.6-27B-FP8"

    def test_stop_container(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        with patch("cognithor.core.vllm_orchestrator.VLLMOrchestrator.stop_container") as stop_mock:
            r = client.post("/api/backends/vllm/stop")
        assert r.status_code == 200
        stop_mock.assert_called_once()

    def test_logs_endpoint(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.get_logs",
            return_value=["line1", "line2"],
        ):
            r = client.get("/api/backends/vllm/logs")
        assert r.status_code == 200
        assert r.json()["lines"] == ["line1", "line2"]
