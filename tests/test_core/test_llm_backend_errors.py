from __future__ import annotations

from cognithor.core.llm_backend import (
    LLMBackendError,
    LLMBadRequestError,
    VLLMDockerError,
    VLLMHardwareError,
    VLLMNotReadyError,
)


class TestErrorHierarchy:
    def test_all_vllm_errors_inherit_from_llm_backend_error(self):
        assert issubclass(LLMBadRequestError, LLMBackendError)
        assert issubclass(VLLMNotReadyError, LLMBackendError)
        assert issubclass(VLLMHardwareError, LLMBackendError)
        assert issubclass(VLLMDockerError, LLMBackendError)

    def test_errors_carry_recovery_hint(self):
        err = VLLMNotReadyError("container down", recovery_hint="Run: docker start vllm")
        assert err.recovery_hint == "Run: docker start vllm"
        assert str(err) == "container down"

    def test_recovery_hint_defaults_to_empty(self):
        err = VLLMDockerError("Docker not found")
        assert err.recovery_hint == ""

    def test_status_code_preserved_from_base(self):
        err = LLMBadRequestError("context too long", status_code=400)
        assert err.status_code == 400


class TestBackendTypeEnum:
    def test_vllm_is_a_backend_type(self):
        from cognithor.core.llm_backend import LLMBackendType

        assert LLMBackendType.VLLM == "vllm"
        assert LLMBackendType.VLLM.value == "vllm"

    def test_vllm_value_matches_config_literal(self):
        """config.CognithorConfig.llm_backend_type accepts "vllm" — keep enum aligned."""
        from cognithor.core.llm_backend import LLMBackendType

        assert "vllm" in {t.value for t in LLMBackendType}
