from __future__ import annotations

from cognithor.core.vllm_orchestrator import (
    ContainerInfo,
    DockerInfo,
    HardwareInfo,
    ModelEntry,
    VLLMOrchestrator,
    VLLMState,
)


class TestDataclasses:
    def test_hardware_info_fields(self):
        h = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        assert h.gpu_name == "RTX 5090"
        assert h.vram_gb == 32
        assert h.compute_capability == (12, 0)

    def test_hardware_info_sm_as_string(self):
        h = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        assert h.sm_string == "8.9"

    def test_docker_info_fields(self):
        d = DockerInfo(available=True, version="26.0.0", server_running=True)
        assert d.available is True
        assert d.version == "26.0.0"

    def test_model_entry_from_dict(self):
        m = ModelEntry.from_dict(
            {
                "id": "mmangkad/Qwen3.6-27B-NVFP4",
                "display_name": "Qwen3.6-27B NVFP4",
                "base_model": "Qwen/Qwen3.6-27B",
                "quantization": "NVFP4",
                "vram_gb_min": 14,
                "min_compute_capability": "12.0",
                "min_vllm_version": "pending",
                "capability": "vision",
                "priority": "premium",
                "tested": False,
                "notes": "",
            }
        )
        assert m.id == "mmangkad/Qwen3.6-27B-NVFP4"
        assert m.min_cc_tuple == (12, 0)
        assert m.vram_gb_min == 14
        assert m.priority == "premium"

    def test_vllm_state_initial(self):
        s = VLLMState()
        assert s.hardware_ok is False
        assert s.docker_ok is False
        assert s.container_running is False
        assert s.current_model is None
        assert s.hardware_info is None

    def test_container_info(self):
        c = ContainerInfo(container_id="abc123", port=8000, model="Qwen/Qwen3.6-27B-FP8")
        assert c.container_id == "abc123"
        assert c.port == 8000


class TestOrchestratorInit:
    def test_orchestrator_constructs_with_config(self):
        orch = VLLMOrchestrator(
            docker_image="vllm/vllm-openai:v0.19.1",
            port=8000,
            hf_token="hf_test",
        )
        assert orch.docker_image == "vllm/vllm-openai:v0.19.1"
        assert orch.port == 8000
        assert orch._hf_token == "hf_test"
        assert orch.state.hardware_ok is False
