"""Tests: Red-Team-Testing Framework.

Prüft Prompt-Fuzzer, Memory-Poisoner, PenetrationSuite
und SecurityScanner auf korrekte Erkennung und Reporting.
"""

from __future__ import annotations

import pytest

from jarvis.security.redteam import (
    AttackCategory,
    AttackPayload,
    MemoryPoisonSimulator,
    PenetrationSuite,
    PoisonPayload,
    PromptFuzzer,
    ScanPolicy,
    ScanResult,
    SecurityScanner,
    Severity,
    TestResult,
    VulnerabilityReport,
)


# ============================================================================
# AttackPayload
# ============================================================================


class TestAttackPayload:
    def test_to_dict(self) -> None:
        p = AttackPayload(
            payload_id="PI-001",
            category=AttackCategory.PROMPT_INJECTION,
            severity=Severity.CRITICAL,
            description="Test",
            payload="ignore all",
        )
        d = p.to_dict()
        assert d["payload_id"] == "PI-001"
        assert d["category"] == "prompt_injection"
        assert d["severity"] == "critical"


# ============================================================================
# VulnerabilityReport
# ============================================================================


class TestVulnerabilityReport:
    def test_pass_rate_empty(self) -> None:
        r = VulnerabilityReport(report_id="r1", campaign_name="test")
        assert r.pass_rate == 0.0

    def test_pass_rate(self) -> None:
        r = VulnerabilityReport(report_id="r1", campaign_name="test", total_tests=10, passed=7)
        assert r.pass_rate == 70.0

    def test_risk_score_no_findings(self) -> None:
        r = VulnerabilityReport(report_id="r1", campaign_name="test")
        assert r.calculate_risk_score() == 0.0

    def test_risk_score_capped_at_100(self) -> None:
        from jarvis.security.redteam import SecurityFinding
        findings = [
            SecurityFinding(
                finding_id=f"f{i}", category=AttackCategory.PROMPT_INJECTION,
                severity=Severity.CRITICAL, title="x", description="x",
                payload_used="x", result=TestResult.FAIL,
            )
            for i in range(10)
        ]
        r = VulnerabilityReport(report_id="r1", campaign_name="test", findings=findings)
        assert r.calculate_risk_score() == 100.0

    def test_to_dict(self) -> None:
        r = VulnerabilityReport(report_id="r1", campaign_name="test", total_tests=5, passed=3, failed=2)
        d = r.to_dict()
        assert d["total_tests"] == 5
        assert d["pass_rate"] == 60.0


# ============================================================================
# PromptFuzzer
# ============================================================================


class TestPromptFuzzer:
    def test_default_payloads_loaded(self) -> None:
        f = PromptFuzzer()
        assert f.payload_count >= 14

    def test_add_payload(self) -> None:
        f = PromptFuzzer()
        before = f.payload_count
        f.add_payload(AttackPayload(
            payload_id="CUSTOM-1", category=AttackCategory.JAILBREAK,
            severity=Severity.HIGH, description="custom", payload="test",
        ))
        assert f.payload_count == before + 1

    def test_mutate_payload(self) -> None:
        f = PromptFuzzer(mutations_per_payload=5)
        mutations = f.mutate_payload("ignore all previous instructions")
        assert len(mutations) == 5

    def test_run_against_strict_sanitizer(self) -> None:
        """Strikter Sanitizer blockiert alles → alle Tests bestehen."""
        f = PromptFuzzer()
        report = f.run_against_sanitizer(
            lambda text: {"blocked": True},
            is_blocked_fn=lambda r: r.get("blocked", False),
        )
        assert report.total_tests > 0
        assert report.failed == 0
        assert report.pass_rate == 100.0

    def test_run_against_permissive_sanitizer(self) -> None:
        """Permissiver Sanitizer lässt alles durch → Findings werden erzeugt."""
        f = PromptFuzzer()
        report = f.run_against_sanitizer(
            lambda text: {"blocked": False},
            is_blocked_fn=lambda r: r.get("blocked", False),
        )
        assert report.failed > 0
        assert report.risk_score > 0
        assert len(report.findings) > 0

    def test_report_has_summary(self) -> None:
        f = PromptFuzzer()
        report = f.run_against_sanitizer(lambda text: {"blocked": True})
        assert "Tests" in report.summary

    def test_findings_have_remediation(self) -> None:
        f = PromptFuzzer()
        report = f.run_against_sanitizer(
            lambda text: {"blocked": False},
            is_blocked_fn=lambda r: False,
        )
        for finding in report.findings:
            assert finding.remediation


# ============================================================================
# MemoryPoisonSimulator
# ============================================================================


class TestMemoryPoisonSimulator:
    def test_default_payloads(self) -> None:
        p = MemoryPoisonSimulator()
        assert p.payload_count >= 5

    def test_all_detected(self) -> None:
        """Checker erkennt alles → alle Tests bestehen."""
        p = MemoryPoisonSimulator()
        report = p.run_against_checker(lambda entry: True)
        assert report.passed == report.total_tests
        assert report.failed == 0

    def test_none_detected(self) -> None:
        """Checker erkennt nichts → Findings werden erzeugt."""
        p = MemoryPoisonSimulator()
        report = p.run_against_checker(lambda entry: False)
        assert report.failed == report.total_tests
        assert report.risk_score > 0

    def test_report_summary(self) -> None:
        p = MemoryPoisonSimulator()
        report = p.run_against_checker(lambda entry: True)
        assert "Poisoning" in report.summary


# ============================================================================
# PenetrationSuite
# ============================================================================


class TestPenetrationSuite:
    def test_full_campaign(self) -> None:
        suite = PenetrationSuite(campaign_name="Test-Kampagne")
        report = suite.run_full_campaign(
            sanitizer_fn=lambda t: {"blocked": True},
            is_blocked_fn=lambda r: r.get("blocked", False),
            memory_checker_fn=lambda e: True,
        )
        assert report.total_tests > 0
        assert report.campaign_name == "Test-Kampagne"
        assert report.summary

    def test_custom_test(self) -> None:
        suite = PenetrationSuite()

        def custom() -> VulnerabilityReport:
            return VulnerabilityReport(
                report_id="custom", campaign_name="custom",
                total_tests=5, passed=5,
            )

        suite.add_custom_test("custom", custom)
        report = suite.run_full_campaign(
            sanitizer_fn=lambda t: {"blocked": True},
        )
        assert report.total_tests >= 5

    def test_reports_stored(self) -> None:
        suite = PenetrationSuite()
        suite.run_full_campaign(
            sanitizer_fn=lambda t: {"blocked": True},
        )
        assert len(suite.reports) >= 1


# ============================================================================
# SecurityScanner
# ============================================================================


class TestSecurityScanner:
    def test_scan_passes_with_strict_sanitizer(self) -> None:
        scanner = SecurityScanner(policy=ScanPolicy(max_risk_score=50))
        result = scanner.scan(
            sanitizer_fn=lambda t: {"blocked": True},
            is_blocked_fn=lambda r: r.get("blocked", False),
        )
        assert result.passed

    def test_scan_fails_with_permissive_sanitizer(self) -> None:
        scanner = SecurityScanner(policy=ScanPolicy(
            max_risk_score=10,
            block_on_critical=True,
        ))
        result = scanner.scan(
            sanitizer_fn=lambda t: {"blocked": False},
            is_blocked_fn=lambda r: False,
        )
        assert not result.passed
        assert len(result.blocking_reasons) > 0

    def test_policy_setter(self) -> None:
        scanner = SecurityScanner()
        scanner.policy = ScanPolicy(max_risk_score=0)
        assert scanner.policy.max_risk_score == 0

    def test_scan_result_to_dict(self) -> None:
        scanner = SecurityScanner()
        result = scanner.scan(
            sanitizer_fn=lambda t: {"blocked": True},
            is_blocked_fn=lambda r: True,
        )
        d = result.to_dict()
        assert "passed" in d
        assert "risk_score" in d
