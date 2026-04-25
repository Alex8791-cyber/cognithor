"""ReportGenerator — produces final markdown report from analyst + assessor outputs."""

from __future__ import annotations

from insurance_agent_pack.agents.report_generator import build_report_generator


def test_report_generator_role_label() -> None:
    a = build_report_generator(model="ollama/qwen3:8b")
    assert a.role == "report-generator"


def test_report_generator_uses_no_tools() -> None:
    a = build_report_generator(model="ollama/qwen3:8b")
    assert a.tools == []


def test_report_generator_disallows_delegation() -> None:
    a = build_report_generator(model="ollama/qwen3:8b")
    assert a.allow_delegation is False
