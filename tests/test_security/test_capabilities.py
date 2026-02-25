"""Tests fuer CapabilityMatrix und PolicyEvaluator."""

import pytest
from jarvis.security.capabilities import (
    CapabilityMatrix,
    PolicyEvaluator,
    SandboxProfile,
    RESTRICTIVE,
    STANDARD,
    PERMISSIVE,
)
from jarvis.models import ToolCapability, ToolCapabilitySpec, PolicyDecision


class TestSandboxProfile:
    def test_restrictive_denies_write(self):
        assert not RESTRICTIVE.is_capability_allowed(ToolCapability.FS_WRITE)

    def test_restrictive_allows_read(self):
        assert RESTRICTIVE.is_capability_allowed(ToolCapability.FS_READ)

    def test_standard_allows_write(self):
        assert STANDARD.is_capability_allowed(ToolCapability.FS_WRITE)

    def test_standard_denies_exec(self):
        assert not STANDARD.is_capability_allowed(ToolCapability.EXEC_PROCESS)

    def test_permissive_allows_all(self):
        for cap in ToolCapability:
            assert PERMISSIVE.is_capability_allowed(cap)

    def test_custom_profile(self):
        profile = SandboxProfile(
            name="custom",
            allowed_capabilities={ToolCapability.FS_READ, ToolCapability.NETWORK_HTTP},
        )
        assert profile.is_capability_allowed(ToolCapability.FS_READ)
        assert profile.is_capability_allowed(ToolCapability.NETWORK_HTTP)
        assert not profile.is_capability_allowed(ToolCapability.FS_WRITE)


class TestCapabilityMatrix:
    def setup_method(self):
        self.matrix = CapabilityMatrix()

    def test_defaults_registered(self):
        assert "read_file" in self.matrix.registered_tools
        assert "exec_command" in self.matrix.registered_tools

    def test_get_spec(self):
        spec = self.matrix.get_spec("read_file")
        assert spec is not None
        assert ToolCapability.FS_READ in spec.capabilities

    def test_get_spec_unknown(self):
        assert self.matrix.get_spec("unknown_tool") is None

    def test_register_custom_tool(self):
        spec = ToolCapabilitySpec(
            tool_name="my_tool",
            capabilities=frozenset({ToolCapability.NETWORK_HTTP}),
        )
        self.matrix.register_tool(spec)
        assert self.matrix.get_spec("my_tool") is not None

    def test_check_allowed_restrictive(self):
        assert self.matrix.check_allowed("read_file", RESTRICTIVE)
        assert not self.matrix.check_allowed("write_file", RESTRICTIVE)
        assert not self.matrix.check_allowed("exec_command", RESTRICTIVE)

    def test_check_allowed_standard(self):
        assert self.matrix.check_allowed("read_file", STANDARD)
        assert self.matrix.check_allowed("write_file", STANDARD)
        assert not self.matrix.check_allowed("exec_command", STANDARD)

    def test_check_allowed_permissive(self):
        assert self.matrix.check_allowed("read_file", PERMISSIVE)
        assert self.matrix.check_allowed("exec_command", PERMISSIVE)

    def test_get_violations(self):
        violations = self.matrix.get_violations("exec_command", RESTRICTIVE)
        assert len(violations) > 0

    def test_no_violations(self):
        violations = self.matrix.get_violations("read_file", STANDARD)
        assert len(violations) == 0

    def test_unknown_tool_violations(self):
        violations = self.matrix.get_violations("unknown_tool", STANDARD)
        assert len(violations) == 1
        assert "Unknown" in violations[0]


class TestPolicyEvaluator:
    def setup_method(self):
        self.evaluator = PolicyEvaluator()

    def test_evaluate_allowed(self):
        decision = self.evaluator.evaluate("read_file", profile_name="standard")
        assert decision.allowed
        assert len(decision.violations) == 0

    def test_evaluate_denied(self):
        decision = self.evaluator.evaluate("exec_command", profile_name="standard")
        assert not decision.allowed
        assert len(decision.violations) > 0

    def test_evaluate_unknown_profile(self):
        decision = self.evaluator.evaluate("read_file", profile_name="nonexistent")
        assert not decision.allowed
        assert "Unknown profile" in decision.violations[0]

    def test_suggested_profile_upgrade(self):
        decision = self.evaluator.evaluate("write_file", profile_name="restrictive")
        assert not decision.allowed
        assert decision.suggested_profile == "standard"

    def test_register_custom_profile(self):
        profile = SandboxProfile(
            name="my_profile",
            allowed_capabilities={ToolCapability.FS_READ},
        )
        self.evaluator.register_profile(profile)
        assert "my_profile" in self.evaluator.available_profiles

    def test_available_profiles(self):
        profiles = self.evaluator.available_profiles
        assert "restrictive" in profiles
        assert "standard" in profiles
        assert "permissive" in profiles

    def test_permissive_allows_everything(self):
        for tool in self.evaluator.matrix.registered_tools:
            decision = self.evaluator.evaluate(tool, profile_name="permissive")
            assert decision.allowed, f"{tool} should be allowed in permissive"

    def test_evaluate_with_params(self):
        decision = self.evaluator.evaluate(
            "read_file",
            params={"path": "/tmp/test.txt"},
            profile_name="standard",
        )
        assert decision.allowed
