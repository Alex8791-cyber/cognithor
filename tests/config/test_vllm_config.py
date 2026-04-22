# tests/config/test_vllm_config.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cognithor.config import CognithorConfig, VLLMConfig


class TestVLLMConfig:
    def test_defaults(self):
        c = VLLMConfig()
        assert c.enabled is False
        assert c.model == ""
        assert c.docker_image == "vllm/vllm-openai:v0.19.1"
        assert c.port == 8000
        assert c.auto_stop_on_close is False
        assert c.skip_hardware_check is False
        assert c.request_timeout_seconds == 60

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VLLMConfig(unknown_field=1)

    def test_cognithor_config_has_vllm_sub_model(self):
        c = CognithorConfig()
        assert hasattr(c, "vllm")
        assert isinstance(c.vllm, VLLMConfig)

    def test_vllm_override(self):
        c = CognithorConfig(vllm={"enabled": True, "port": 8042})
        assert c.vllm.enabled is True
        assert c.vllm.port == 8042
