"""Sprint-27 VLM-2 — orchestrator launch-flag tests for VLM mode.

Verifies that ``vllm_enabled=True`` extends the docker-run argv
with the spike-doc-recommended flags, and that the launcher stays
identical to the pre-VLM-2 shape when the flag is off (no
regression for the plain text-only Qwen path).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cognithor.config import VLLMConfig
from cognithor.core.vllm_orchestrator import VLLMOrchestrator


def _make_orchestrator(cfg: VLLMConfig) -> VLLMOrchestrator:
    """Construct an orchestrator with a stubbed _hf_token for tests."""

    return VLLMOrchestrator(config=cfg, hf_token="hf_test_token_xx")


def _intercept_docker_run() -> tuple[list[str], MagicMock]:
    """Patch subprocess.run + port + health to capture the docker argv.

    Returns ``(captured_cmd, mock_run)``. The first element is
    populated only after ``_start_container`` is called.
    """

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stdout = "container-id-12345abc\n"
        result.stderr = ""
        return result

    mock_run = MagicMock(side_effect=fake_run)
    return [], mock_run  # type: ignore[return-value]


@pytest.fixture
def captured_argv() -> list[list[str]]:
    """Captures every subprocess.run argv list during a test."""

    return []


@pytest.fixture
def patched_subprocess(captured_argv: list[list[str]]) -> object:
    """Patch subprocess.run to capture argv + always return success."""

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured_argv.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stdout = "container-id-12345abc\n"
        result.stderr = ""
        return result

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        yield mock_run


# ---------------------------------------------------------------------------
# vlm_enabled=False — no regression vs pre-VLM-2 shape
# ---------------------------------------------------------------------------


class TestVLMModeOff:
    def test_no_vlm_flags_when_disabled(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(enabled=True, model="qwen3:32b", vlm_enabled=False)
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("qwen3:32b")
        assert captured_argv, "no docker run captured"
        argv = captured_argv[0]
        # None of the VLM-mode flags must appear.
        for flag in (
            "--quantization",
            "--kv-cache-dtype",
            "--enable-prefix-caching",
            "--num-speculative-tokens",
            "--limit-mm-per-prompt",
            "--served-model-name",
        ):
            assert flag not in argv, f"{flag} should not be present when vlm_enabled=False"


# ---------------------------------------------------------------------------
# vlm_enabled=True — spike-doc flags applied
# ---------------------------------------------------------------------------


class TestVLMModeOn:
    def test_default_fp8_quant_kv_prefix_speculative(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-FP8",
            vlm_enabled=True,
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-FP8")

        argv = captured_argv[0]
        # spike-doc defaults
        assert "--quantization" in argv
        assert argv[argv.index("--quantization") + 1] == "fp8"
        assert "--kv-cache-dtype" in argv
        assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
        assert "--enable-prefix-caching" in argv
        assert "--num-speculative-tokens" in argv
        assert argv[argv.index("--num-speculative-tokens") + 1] == "1"

    def test_limit_mm_per_prompt_json_payload(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-FP8",
            vlm_enabled=True,
            vlm_image_max_per_prompt=8,
            vlm_video_max_per_prompt=2,
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-FP8")

        argv = captured_argv[0]
        idx = argv.index("--limit-mm-per-prompt")
        payload = json.loads(argv[idx + 1])
        assert payload == {"image": 8, "video": 2}

    def test_served_model_name_when_set(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-FP8",
            vlm_enabled=True,
            vlm_served_model_name="qwen3-vl-32b-fp8",
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-FP8")

        argv = captured_argv[0]
        assert "--served-model-name" in argv
        assert argv[argv.index("--served-model-name") + 1] == "qwen3-vl-32b-fp8"

    def test_served_model_name_omitted_when_blank(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-FP8",
            vlm_enabled=True,
            vlm_served_model_name="",
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-FP8")

        argv = captured_argv[0]
        assert "--served-model-name" not in argv

    def test_nvfp4_opt_in(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-NVFP4",
            vlm_enabled=True,
            vlm_quantization="nvfp4",
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-NVFP4")
        argv = captured_argv[0]
        assert argv[argv.index("--quantization") + 1] == "nvfp4"

    def test_speculative_tokens_zero_omits_flag(
        self,
        captured_argv: list[list[str]],
        patched_subprocess: object,
    ) -> None:
        cfg = VLLMConfig(
            enabled=True,
            model="Qwen/Qwen3-VL-32B-Instruct-FP8",
            vlm_enabled=True,
            vlm_num_speculative_tokens=0,
        )
        orch = _make_orchestrator(cfg)
        with (
            patch.object(orch, "_port_available", return_value=True),
            patch.object(
                orch,
                "_wait_for_health",
                return_value=True,
            ),
        ):
            orch.start_container("Qwen/Qwen3-VL-32B-Instruct-FP8")
        argv = captured_argv[0]
        assert "--num-speculative-tokens" not in argv


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestVLMConfigValidation:
    def test_invalid_quant_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            VLLMConfig(
                vlm_enabled=True,
                vlm_quantization="int4",  # type: ignore[arg-type]
            )

    def test_speculative_tokens_negative_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            VLLMConfig(vlm_enabled=True, vlm_num_speculative_tokens=-1)

    def test_image_per_prompt_negative_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            VLLMConfig(vlm_enabled=True, vlm_image_max_per_prompt=-1)
