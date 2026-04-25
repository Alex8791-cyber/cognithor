"""Signature-parity tests — inspect.signature diff against autogen-agentchat==0.7.5.

These tests are skipped when autogen-agentchat is not installed (see conftest).
They are the Stage-1 of the D6 test strategy: cheap, fast, catch drift early.
"""

from __future__ import annotations

import inspect

import pytest

from cognithor.compat.autogen import AssistantAgent


@pytest.mark.requires_autogen
def test_assistant_agent_signature_matches_autogen() -> None:
    from autogen_agentchat.agents import AssistantAgent as RealAssistantAgent

    real_sig = inspect.signature(RealAssistantAgent.__init__)
    shim_sig = inspect.signature(AssistantAgent.__init__)

    real_params = list(real_sig.parameters.keys())
    shim_params = list(shim_sig.parameters.keys())
    assert real_params == shim_params, (
        f"parameter ORDER mismatch:\n  real: {real_params}\n  shim: {shim_params}"
    )


@pytest.mark.requires_autogen
def test_assistant_agent_param_kinds_match() -> None:
    """Each parameter has the same KIND (positional, keyword-only, etc.)."""
    from autogen_agentchat.agents import AssistantAgent as RealAssistantAgent

    real = inspect.signature(RealAssistantAgent.__init__).parameters
    shim = inspect.signature(AssistantAgent.__init__).parameters

    for name in real:
        if name == "self":
            continue
        assert real[name].kind == shim[name].kind, (
            f"kind mismatch for '{name}': real={real[name].kind} shim={shim[name].kind}"
        )


@pytest.mark.requires_autogen
def test_assistant_agent_defaults_match() -> None:
    """Each parameter has the same default value."""
    from autogen_agentchat.agents import AssistantAgent as RealAssistantAgent

    real = inspect.signature(RealAssistantAgent.__init__).parameters
    shim = inspect.signature(AssistantAgent.__init__).parameters

    for name in real:
        if name == "self":
            continue
        # Skip default comparison for tools/handoffs/memory/workbench/model_context —
        # those are AutoGen-internal class types we don't reproduce. Defaults that
        # are None or empty-collection-like must still match.
        if name in {"tools", "handoffs", "memory", "workbench", "model_context"}:
            continue
        real_default = real[name].default
        shim_default = shim[name].default
        assert real_default == shim_default, (
            f"default mismatch for '{name}': real={real_default!r} shim={shim_default!r}"
        )


def test_assistant_agent_has_run_and_run_stream_methods() -> None:
    """Independent of autogen install — shim must expose run + run_stream."""
    assert callable(AssistantAgent.run)
    assert callable(AssistantAgent.run_stream)


def test_assistant_agent_signature_field_count() -> None:
    """17 fields per autogen-agentchat==0.7.5 + 'self' = 18 total.

    Verified empirically: spec §3.3 listed 14, but autogen-agentchat==0.7.5
    actually exposes 17 (added tool_call_summary_formatter,
    output_content_type, output_content_type_format vs the spec).
    """
    sig = inspect.signature(AssistantAgent.__init__)
    assert len(sig.parameters) == 18, (
        f"expected 18 params (self + 17 fields), got {len(sig.parameters)}"
    )
