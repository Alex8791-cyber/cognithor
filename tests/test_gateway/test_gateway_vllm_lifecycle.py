from __future__ import annotations

from unittest.mock import MagicMock

from cognithor.config import CognithorConfig, VLLMConfig


class TestGatewayVLLMLifecycle:
    def test_shutdown_stops_container_when_toggle_on(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True, auto_stop_on_close=True))
        from cognithor.gateway.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw.on_shutdown_vllm()
        gw._vllm_orchestrator.stop_container.assert_called_once()

    def test_shutdown_leaves_container_running_when_toggle_off(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True, auto_stop_on_close=False))
        from cognithor.gateway.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw.on_shutdown_vllm()
        gw._vllm_orchestrator.stop_container.assert_not_called()

    def test_startup_picks_up_existing_container(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True))
        from cognithor.core.vllm_orchestrator import ContainerInfo
        from cognithor.gateway.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw._vllm_orchestrator.reuse_existing.return_value = ContainerInfo(
            container_id="abc", port=8000, model="Qwen/Qwen2.5-VL-7B-Instruct"
        )
        result = gw.on_startup_vllm()
        assert result is not None
        assert result.model == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_startup_returns_none_when_vllm_disabled(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=False))
        from cognithor.gateway.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = None
        result = gw.on_startup_vllm()
        assert result is None
