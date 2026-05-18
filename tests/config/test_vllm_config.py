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
        assert c.docker_image == "vllm/vllm-openai:cu130-nightly"
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


class TestVLLMConfigVideoFields:
    def test_video_defaults(self):
        c = VLLMConfig()
        assert c.video_sampling_mode == "adaptive"
        assert c.video_ffprobe_path == "ffprobe"
        assert c.video_ffprobe_timeout_seconds == 5
        assert c.video_ffprobe_http_timeout_seconds == 30
        assert c.video_max_upload_mb == 500
        assert c.video_quota_gb == 5
        assert c.video_upload_ttl_hours == 24

    def test_video_sampling_mode_literal_rejects_garbage(self):
        with pytest.raises(ValidationError):
            VLLMConfig(video_sampling_mode="totally_bogus")

    def test_video_sampling_mode_accepts_all_four(self):
        for mode in ("adaptive", "fixed_32", "fixed_64", "fps_1"):
            c = VLLMConfig(video_sampling_mode=mode)
            assert c.video_sampling_mode == mode

    def test_timeout_lower_bound(self):
        with pytest.raises(ValidationError):
            VLLMConfig(video_ffprobe_timeout_seconds=0)

    def test_upload_mb_upper_bound(self):
        with pytest.raises(ValidationError):
            VLLMConfig(video_max_upload_mb=999999)


class TestVLLMConfigLauncherFields:
    """Flags that the spike identified as required for 32 GB-class GPUs.
    Defaults match the spike's working RTX 5090 profile."""

    def test_launcher_defaults_match_spike_profile(self):
        c = VLLMConfig()
        assert c.max_model_len == 16384
        assert c.max_num_seqs == 2
        assert c.max_num_batched_tokens == 2048
        assert c.gpu_memory_utilization == 0.94
        # cpu_offload_gb default is 0 (GPU-only): the curated models all fit
        # the detected GPU, and offloaded weights cross PCIe every forward pass.
        assert c.cpu_offload_gb == 0
        assert c.enforce_eager is True

    def test_gpu_util_must_be_in_open_unit_interval(self):
        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=0.0)
        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=1.01)

    def test_larger_gpu_profile(self):
        """User with an A100 loosens the defaults."""
        c = VLLMConfig(
            max_model_len=65536,
            max_num_seqs=8,
            gpu_memory_utilization=0.90,
            cpu_offload_gb=0,
            enforce_eager=False,
        )
        assert c.max_model_len == 65536
        assert c.cpu_offload_gb == 0
        assert c.enforce_eager is False

    def test_speculative_fields_default_off(self):
        """speculative_config / quantization / language_model_only are opt-in:
        a default config emits none of the model-specific launch flags."""
        c = VLLMConfig()
        assert c.speculative_config is None
        assert c.quantization == ""
        assert c.language_model_only is False

    def test_speculative_config_accepts_mtp_dict(self):
        c = VLLMConfig(
            speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3},
            quantization="modelopt",
            language_model_only=True,
        )
        assert c.speculative_config == {"method": "qwen3_5_mtp", "num_speculative_tokens": 3}
        assert c.quantization == "modelopt"
        assert c.language_model_only is True
