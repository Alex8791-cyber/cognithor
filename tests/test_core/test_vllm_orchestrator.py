from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cognithor.core.llm_backend import VLLMHardwareError
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


class TestCheckHardware:
    def _mk_orch(self):
        return VLLMOrchestrator()

    def test_detects_rtx_5090(self):
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 5090, 32768, 12.0\n")
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert info.gpu_name == "NVIDIA GeForce RTX 5090"
        assert info.vram_gb == 32
        assert info.compute_capability == (12, 0)

    def test_detects_rtx_4090(self):
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564, 8.9\n")
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert info.gpu_name == "NVIDIA GeForce RTX 4090"
        assert info.vram_gb == 24
        assert info.compute_capability == (8, 9)

    def test_raises_when_nvidia_smi_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(VLLMHardwareError) as exc:
                self._mk_orch().check_hardware()
            assert "nvidia-smi" in str(exc.value).lower()

    def test_raises_when_no_gpu_detected(self):
        mock_result = MagicMock(returncode=0, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VLLMHardwareError):
                self._mk_orch().check_hardware()

    def test_raises_when_nvidia_smi_fails(self):
        mock_result = MagicMock(returncode=9, stdout="", stderr="NVIDIA-SMI has failed")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VLLMHardwareError):
                self._mk_orch().check_hardware()

    def test_picks_first_gpu_when_multiple(self):
        mock_result = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, 32768, 12.0\nNVIDIA GeForce RTX 3060, 12288, 8.6\n",
        )
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert "5090" in info.gpu_name

    def test_state_updated_after_success(self):
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4080, 16380, 8.9\n")
        orch = self._mk_orch()
        with patch("subprocess.run", return_value=mock_result):
            orch.check_hardware()
        assert orch.state.hardware_ok is True
        assert orch.state.hardware_info is not None
        assert orch.state.hardware_info.compute_capability == (8, 9)


class TestCheckDocker:
    def test_docker_running(self):
        mock_stdout = '{"Client":{"Version":"26.0.0"},"Server":{"Version":"26.0.0"}}'
        mock_result = MagicMock(returncode=0, stdout=mock_stdout)
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True
        assert info.server_running is True
        assert info.version == "26.0.0"

    def test_docker_installed_but_server_down(self):
        mock_stdout = '{"Client":{"Version":"26.0.0"}}'
        mock_result = MagicMock(returncode=0, stdout=mock_stdout)
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True
        assert info.server_running is False

    def test_docker_cli_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = VLLMOrchestrator().check_docker()
        assert info.available is False
        assert info.server_running is False

    def test_docker_cmd_fails(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="daemon not running")
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True
        assert info.server_running is False

    def test_state_updated(self):
        mock_stdout = '{"Client":{"Version":"26.0.0"},"Server":{"Version":"26.0.0"}}'
        orch = VLLMOrchestrator()
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=mock_stdout)):
            orch.check_docker()
        assert orch.state.docker_ok is True


class TestPullImage:
    def test_pull_emits_progress_events(self):
        json_lines = [
            '{"status":"Pulling from vllm/vllm-openai","id":"latest"}\n',
            '{"status":"Downloading","progressDetail":{"current":1000000,"total":10000000},"id":"abc123"}\n',
            '{"status":"Download complete","id":"abc123"}\n',
            '{"status":"Status: Downloaded newer image for vllm/vllm-openai:v0.19.1"}\n',
        ]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(json_lines)
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        events: list[dict] = []

        def cb(ev):
            events.append(ev)

        with patch("subprocess.Popen", return_value=mock_proc):
            VLLMOrchestrator().pull_image("vllm/vllm-openai:v0.19.1", progress_callback=cb)

        assert any(e.get("status") == "Downloading" for e in events)
        assert any("current" in (e.get("progressDetail") or {}) for e in events)

    def test_pull_failure_raises(self):
        from cognithor.core.llm_backend import VLLMDockerError

        mock_proc = MagicMock()
        mock_proc.stdout = iter(['{"status":"error"}\n'])
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1

        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(VLLMDockerError):
                VLLMOrchestrator().pull_image("bad/image:tag", progress_callback=None)

    def test_pull_sets_image_pulled_flag(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(['{"status":"Pulling"}\n'])
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        orch = VLLMOrchestrator()
        with patch("subprocess.Popen", return_value=mock_proc):
            orch.pull_image(orch.docker_image, progress_callback=None)
        assert orch.state.image_pulled is True
