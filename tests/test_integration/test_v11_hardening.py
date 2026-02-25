"""Tests: Hardening, EU-AI-Act, Multi-Tenant, Memory-Erweiterungen.

83+ Tests für die 3 neuen Module + Memory-Erweiterungen.
"""

from __future__ import annotations

import pytest
from typing import Any

# ============================================================================
# 1. Security Hardening
# ============================================================================

from jarvis.security.hardening import (
    AgentContainer,
    ContainerIsolation,
    CredentialScanner,
    GateDecision,
    GatePolicy,
    IsolationLevel,
    PreCommitHook,
    ScanScheduler,
    SecurityGate,
    WebhookConfig,
    WebhookNotifier,
)


class TestSecurityGate:
    def test_allow_clean(self) -> None:
        gate = SecurityGate()
        r = gate.evaluate(critical_findings=0, high_findings=0, pass_rate=95,
                         stages_run=["adversarial_fuzzing", "dependency_scan"])
        assert r.decision == GateDecision.ALLOW

    def test_block_critical(self) -> None:
        gate = SecurityGate()
        r = gate.evaluate(critical_findings=1)
        assert r.decision == GateDecision.BLOCK

    def test_block_high(self) -> None:
        gate = SecurityGate(GatePolicy(max_high_findings=2))
        r = gate.evaluate(high_findings=5)
        assert r.blocked

    def test_block_pass_rate(self) -> None:
        gate = SecurityGate()
        r = gate.evaluate(pass_rate=50.0, stages_run=["adversarial_fuzzing", "dependency_scan"])
        assert r.blocked

    def test_block_missing_stage(self) -> None:
        gate = SecurityGate()
        r = gate.evaluate(stages_run=[])
        assert r.blocked

    def test_warn_medium(self) -> None:
        gate = SecurityGate()
        r = gate.evaluate(medium_findings=10, stages_run=["adversarial_fuzzing", "dependency_scan"])
        assert r.decision == GateDecision.WARN

    def test_custom_policy(self) -> None:
        policy = GatePolicy(max_critical_findings=5, require_fuzzing=False, block_on_unscanned=False)
        gate = SecurityGate(policy)
        r = gate.evaluate(critical_findings=3, stages_run=[])
        assert r.decision != GateDecision.BLOCK

    def test_history(self) -> None:
        gate = SecurityGate()
        gate.evaluate(stages_run=["adversarial_fuzzing", "dependency_scan"])
        gate.evaluate(critical_findings=1)
        assert gate.gate_count == 2
        assert len(gate.history()) == 2

    def test_stats(self) -> None:
        gate = SecurityGate()
        gate.evaluate(stages_run=["adversarial_fuzzing", "dependency_scan"])
        gate.evaluate(critical_findings=1)
        stats = gate.stats()
        assert stats["total_evaluations"] == 2
        assert stats["blocked"] == 1


class TestContainerIsolation:
    def test_create(self) -> None:
        ci = ContainerIsolation()
        c = ci.create("agent-1", memory_mb=256)
        assert c.agent_id == "agent-1"
        assert c.memory_limit_mb == 256
        assert c.has_own_secrets

    def test_no_network_by_default(self) -> None:
        ci = ContainerIsolation()
        c = ci.create("agent-1")
        assert not c.network_enabled

    def test_docker_args(self) -> None:
        ci = ContainerIsolation()
        c = ci.create("agent-1")
        args = c.to_docker_args()
        assert "--network=none" in args
        assert "--read-only" in args

    def test_destroy(self) -> None:
        ci = ContainerIsolation()
        ci.create("agent-1")
        assert ci.destroy("agent-1")
        assert ci.count == 0

    def test_compose(self) -> None:
        ci = ContainerIsolation()
        ci.create("agent-1")
        ci.create("agent-2")
        compose = ci.generate_compose()
        assert "agent-1:" in compose
        assert "agent-2:" in compose

    def test_stats(self) -> None:
        ci = ContainerIsolation()
        ci.create("a1", isolation=IsolationLevel.CONTAINER)
        stats = ci.stats()
        assert stats["total_containers"] == 1


class TestCredentialScanner:
    def test_no_creds(self) -> None:
        scanner = CredentialScanner()
        assert scanner.scan_text("x = 42") == []

    def test_api_key(self) -> None:
        scanner = CredentialScanner()
        findings = scanner.scan_text('api_key = "sk-abc123456789012345678"')
        assert len(findings) >= 1

    def test_aws_key(self) -> None:
        scanner = CredentialScanner()
        findings = scanner.scan_text("key = AKIAIOSFODNN7EXAMPLE")
        assert any(f["type"] == "AWS-Key" for f in findings)

    def test_scan_files(self) -> None:
        scanner = CredentialScanner()
        results = scanner.scan_files({
            "a.py": "x = 1",
            "b.py": 'password = "secret123"',
        })
        assert "b.py" in results
        assert "a.py" not in results


class TestWebhookNotifier:
    def test_notify(self) -> None:
        notifier = WebhookNotifier()
        notifier.add_webhook(WebhookConfig("https://example.com/hook", ["critical_finding"]))
        sent = notifier.notify("critical_finding", {"details": "test"})
        assert sent == 1

    def test_event_filter(self) -> None:
        notifier = WebhookNotifier()
        notifier.add_webhook(WebhookConfig("https://example.com/hook", ["recall"]))
        sent = notifier.notify("critical_finding", {})
        assert sent == 0

    def test_disabled(self) -> None:
        notifier = WebhookNotifier()
        notifier.add_webhook(WebhookConfig("https://example.com/hook", ["critical_finding"], enabled=False))
        sent = notifier.notify("critical_finding", {})
        assert sent == 0

    def test_remove(self) -> None:
        notifier = WebhookNotifier()
        notifier.add_webhook(WebhookConfig("https://example.com/hook"))
        assert notifier.remove_webhook("https://example.com/hook")
        assert notifier.webhook_count == 0


class TestScanScheduler:
    def test_defaults(self) -> None:
        s = ScanScheduler()
        assert s.count == 3  # 3 Default-Schedules

    def test_add_remove(self) -> None:
        from jarvis.security.hardening import ScheduledScan
        s = ScanScheduler(load_defaults=False)
        s.add(ScheduledScan("custom", "Custom", "0 * * * *", ["prompt_injection"]))
        assert s.count == 1
        assert s.remove("custom")
        assert s.count == 0

    def test_enabled(self) -> None:
        s = ScanScheduler()
        assert len(s.enabled_schedules()) == 3


class TestPreCommitHook:
    def test_bash(self) -> None:
        hook = PreCommitHook.generate_bash()
        assert "#!/bin/bash" in hook

    def test_yaml(self) -> None:
        yaml = PreCommitHook.generate_yaml()
        assert "pre-commit-config" in yaml


# ============================================================================
# 2. EU AI Act Compliance
# ============================================================================

from jarvis.audit.eu_ai_act import (
    ComplianceDocManager,
    DocumentType,
    EUAIActGovernor,
    RiskClassifier,
    RiskLevel,
    SystemCategory,
    TrainingCatalog,
    TrainingTopic,
    TransparencyRegister,
)


class TestRiskClassifier:
    def test_insurance_is_high(self) -> None:
        c = RiskClassifier()
        r = c.classify("BU-Berater", SystemCategory.INSURANCE_ADVISORY)
        assert r.risk_level == RiskLevel.HIGH

    def test_general_purpose_limited(self) -> None:
        c = RiskClassifier()
        r = c.classify("ChatBot", SystemCategory.GENERAL_PURPOSE)
        assert r.risk_level == RiskLevel.LIMITED

    def test_custom_minimal(self) -> None:
        c = RiskClassifier()
        r = c.classify("Todo-App", SystemCategory.CUSTOM)
        assert r.risk_level == RiskLevel.MINIMAL

    def test_obligations(self) -> None:
        c = RiskClassifier()
        r = c.classify("HR-AI", SystemCategory.EMPLOYMENT)
        assert any("Art. 11" in o for o in r.obligations)

    def test_high_risk_systems(self) -> None:
        c = RiskClassifier()
        c.classify("A", SystemCategory.INSURANCE_ADVISORY)
        c.classify("B", SystemCategory.CUSTOM)
        assert len(c.high_risk_systems()) == 1

    def test_stats(self) -> None:
        c = RiskClassifier()
        c.classify("A", SystemCategory.INSURANCE_ADVISORY)
        stats = c.stats()
        assert stats["total_assessments"] == 1


class TestComplianceDocManager:
    def test_create(self) -> None:
        dm = ComplianceDocManager()
        doc = dm.create(DocumentType.TECHNICAL_DOC, "Test", "System-1")
        assert doc.doc_id == "DOC-0001"

    def test_generate_technical(self) -> None:
        dm = ComplianceDocManager()
        doc = dm.generate_technical_doc("Jarvis", {
            "purpose": "Agent-System",
            "version": "10.0",
            "test_count": 3200,
        })
        assert "1_system_description" in doc.content
        assert doc.content["3_development_process"]["testing"] == 3200

    def test_generate_incident(self) -> None:
        dm = ComplianceDocManager()
        doc = dm.generate_incident_report("Jarvis", {
            "id": "INC-001",
            "severity": "critical",
            "description": "Prompt-Injection erkannt",
        })
        assert doc.doc_type == DocumentType.INCIDENT_REPORT

    def test_by_type(self) -> None:
        dm = ComplianceDocManager()
        dm.create(DocumentType.TECHNICAL_DOC, "A", "S1")
        dm.create(DocumentType.TRANSPARENCY, "B", "S1")
        assert len(dm.by_type(DocumentType.TECHNICAL_DOC)) == 1

    def test_stats(self) -> None:
        dm = ComplianceDocManager()
        dm.create(DocumentType.TECHNICAL_DOC, "A", "S1")
        stats = dm.stats()
        assert stats["total_documents"] == 1


class TestTransparencyRegister:
    def test_register(self) -> None:
        tr = TransparencyRegister()
        entry = tr.register("Jarvis", "Versicherungsberatung")
        assert "Art. 52" in entry.disclosure_text

    def test_count(self) -> None:
        tr = TransparencyRegister()
        tr.register("A", "X")
        tr.register("B", "Y")
        assert tr.entry_count == 2


class TestTrainingCatalog:
    def test_defaults(self) -> None:
        tc = TrainingCatalog()
        assert tc.module_count >= 4

    def test_by_topic(self) -> None:
        tc = TrainingCatalog()
        modules = tc.by_topic(TrainingTopic.PROMPT_INJECTION)
        assert len(modules) >= 1

    def test_by_difficulty(self) -> None:
        tc = TrainingCatalog()
        assert len(tc.by_difficulty("beginner")) >= 1

    def test_stats(self) -> None:
        tc = TrainingCatalog()
        stats = tc.stats()
        assert stats["total_hours"] > 0


class TestEUAIActGovernor:
    def test_classify_jarvis(self) -> None:
        gov = EUAIActGovernor()
        assessment = gov.classify_jarvis()
        assert assessment.risk_level == RiskLevel.HIGH

    def test_compliance_status(self) -> None:
        gov = EUAIActGovernor()
        gov.classify_jarvis()
        status = gov.compliance_status()
        assert status["high_risk_systems"] == 1

    def test_stats(self) -> None:
        gov = EUAIActGovernor()
        stats = gov.stats()
        assert "classifier" in stats
        assert "training" in stats


# ============================================================================
# 3. Multi-Tenant, Trust, Emergency
# ============================================================================

from jarvis.core.multitenant import (
    EmergencyAction,
    EmergencyController,
    MultiTenantGovernor,
    Tenant,
    TenantManager,
    TenantPlan,
    TenantStatus,
    TenantUser,
    TrustLevel,
    TrustNegotiator,
    TrustPolicy,
)


class TestTenantManager:
    def test_create(self) -> None:
        tm = TenantManager()
        t = tm.create("Firma A", "admin@firma-a.de", TenantPlan.PROFESSIONAL)
        assert t.plan == TenantPlan.PROFESSIONAL
        assert len(t.users) == 1  # Auto-Admin

    def test_isolation_paths(self) -> None:
        tm = TenantManager()
        t = tm.create("Test", "test@test.de")
        assert t.tenant_id in t.data_path
        assert t.tenant_id in t.db_name

    def test_plan_limits(self) -> None:
        tm = TenantManager()
        t = tm.create("Free", "free@test.de", TenantPlan.FREE)
        assert t.limits["max_agents"] == 1
        assert not t.limits["federation_enabled"]

    def test_enterprise_unlimited(self) -> None:
        tm = TenantManager()
        t = tm.create("Big", "big@corp.de", TenantPlan.ENTERPRISE)
        assert t.limits["max_agents"] == -1
        assert t.can_add_agent(100)

    def test_add_user(self) -> None:
        tm = TenantManager()
        t = tm.create("Test", "admin@test.de", TenantPlan.STARTER)
        user = TenantUser("u2", "Bob", "bob@test.de")
        assert tm.add_user(t.tenant_id, user)
        assert len(t.users) == 2

    def test_add_user_limit(self) -> None:
        tm = TenantManager()
        t = tm.create("Test", "admin@test.de", TenantPlan.FREE)
        user = TenantUser("u2", "Bob", "bob@test.de")
        assert not tm.add_user(t.tenant_id, user)

    def test_suspend(self) -> None:
        tm = TenantManager()
        t = tm.create("Test", "admin@test.de")
        tm.suspend(t.tenant_id)
        assert t.status == TenantStatus.SUSPENDED

    def test_upgrade(self) -> None:
        tm = TenantManager()
        t = tm.create("Test", "admin@test.de")
        tm.upgrade(t.tenant_id, TenantPlan.ENTERPRISE)
        assert t.plan == TenantPlan.ENTERPRISE

    def test_find_by_email(self) -> None:
        tm = TenantManager()
        tm.create("A", "alice@test.de")
        assert len(tm.find_by_email("alice@test.de")) == 1

    def test_stats(self) -> None:
        tm = TenantManager()
        tm.create("A", "a@test.de")
        stats = tm.stats()
        assert stats["total_tenants"] == 1


class TestTrustNegotiator:
    def test_initiate(self) -> None:
        tn = TrustNegotiator()
        r = tn.initiate("local", "remote")
        assert r.trust_level == TrustLevel.BASIC

    def test_verify_escalates(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("local", "remote")
        r = tn.verify("local", "remote")
        assert r.trust_level == TrustLevel.VERIFIED

    def test_violation_demotes(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("local", "remote")
        tn.verify("local", "remote")  # → VERIFIED
        r = tn.report_violation("local", "remote")
        assert r.trust_level == TrustLevel.BASIC

    def test_3_violations_untrusted(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("local", "remote")
        tn.verify("local", "remote")  # → VERIFIED
        tn.verify("local", "remote")  # → TRUSTED
        tn.report_violation("local", "remote")
        tn.report_violation("local", "remote")
        r = tn.report_violation("local", "remote")
        assert r.trust_level == TrustLevel.UNTRUSTED

    def test_can_delegate(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("local", "remote")
        assert not tn.can_delegate("local", "remote")  # BASIC < VERIFIED
        tn.verify("local", "remote")
        assert tn.can_delegate("local", "remote")  # VERIFIED

    def test_can_share_data(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("local", "remote")
        tn.verify("local", "remote")
        assert not tn.can_share_data("local", "remote")  # VERIFIED < TRUSTED
        tn.verify("local", "remote")
        assert tn.can_share_data("local", "remote")  # TRUSTED

    def test_stats(self) -> None:
        tn = TrustNegotiator()
        tn.initiate("a", "b")
        stats = tn.stats()
        assert stats["total_relations"] == 1


class TestEmergencyController:
    def test_lockdown(self) -> None:
        ec = EmergencyController()
        ec.execute(EmergencyAction.LOCKDOWN, "Sicherheitsvorfall")
        assert ec.is_lockdown

    def test_quarantine_agent(self) -> None:
        ec = EmergencyController()
        ec.execute(EmergencyAction.QUARANTINE_AGENT, "Kompromittiert", target="agent-1")
        assert ec.is_quarantined("agent-1")
        assert not ec.is_quarantined("agent-2")

    def test_quarantine_skill(self) -> None:
        ec = EmergencyController()
        ec.execute(EmergencyAction.QUARANTINE_SKILL, "Malware", target="skill-xyz")
        assert ec.is_quarantined("skill-xyz")

    def test_revert(self) -> None:
        ec = EmergencyController()
        event = ec.execute(EmergencyAction.LOCKDOWN, "Test")
        assert ec.is_lockdown
        ec.revert(event.event_id)
        assert not ec.is_lockdown

    def test_revert_quarantine(self) -> None:
        ec = EmergencyController()
        event = ec.execute(EmergencyAction.QUARANTINE_AGENT, "Test", target="a1")
        ec.revert(event.event_id)
        assert not ec.is_quarantined("a1")

    def test_history(self) -> None:
        ec = EmergencyController()
        ec.execute(EmergencyAction.LOCKDOWN, "A")
        ec.execute(EmergencyAction.ROLLBACK, "B")
        assert ec.event_count == 2

    def test_stats(self) -> None:
        ec = EmergencyController()
        ec.execute(EmergencyAction.QUARANTINE_AGENT, "X", target="a1")
        stats = ec.stats()
        assert stats["quarantined_agents"] == 1


class TestMultiTenantGovernor:
    def test_pre_action_ok(self) -> None:
        gov = MultiTenantGovernor()
        t = gov.tenants.create("Test", "a@test.de")
        result = gov.pre_action_check(t.tenant_id, "agent-1")
        assert result["allowed"]

    def test_pre_action_lockdown(self) -> None:
        gov = MultiTenantGovernor()
        t = gov.tenants.create("Test", "a@test.de")
        gov.emergency.execute(EmergencyAction.LOCKDOWN, "Test")
        result = gov.pre_action_check(t.tenant_id, "agent-1")
        assert not result["allowed"]

    def test_pre_action_quarantined(self) -> None:
        gov = MultiTenantGovernor()
        t = gov.tenants.create("Test", "a@test.de")
        gov.emergency.execute(EmergencyAction.QUARANTINE_AGENT, "X", target="agent-1")
        result = gov.pre_action_check(t.tenant_id, "agent-1")
        assert not result["allowed"]

    def test_pre_action_suspended(self) -> None:
        gov = MultiTenantGovernor()
        t = gov.tenants.create("Test", "a@test.de")
        gov.tenants.suspend(t.tenant_id)
        result = gov.pre_action_check(t.tenant_id, "agent-1")
        assert not result["allowed"]

    def test_stats(self) -> None:
        gov = MultiTenantGovernor()
        stats = gov.stats()
        assert "tenants" in stats
        assert "trust" in stats
        assert "emergency" in stats


# ============================================================================
# 4. Memory Erweiterungen
# ============================================================================

from jarvis.memory.hygiene import (
    DuplicateDetector,
    MemoryVersionControl,
)


class TestMemoryVersionControl:
    def test_snapshot(self) -> None:
        mvc = MemoryVersionControl()
        snap = mvc.snapshot([{"content": "Hello"}])
        assert snap.entry_count == 1
        assert snap.content_hash

    def test_diff(self) -> None:
        mvc = MemoryVersionControl()
        s1 = mvc.snapshot([{"content": "A"}])
        s2 = mvc.snapshot([{"content": "A"}, {"content": "B"}])
        d = mvc.diff(s1.snapshot_id, s2.snapshot_id)
        assert d["entries_diff"] == 1
        assert d["hash_changed"] is True

    def test_no_drift(self) -> None:
        mvc = MemoryVersionControl()
        entries = [{"content": "A"}, {"content": "B"}]
        mvc.snapshot(entries)
        mvc.snapshot(entries)
        d = mvc.detect_drift()
        assert not d["drift_detected"]

    def test_drift_detected(self) -> None:
        mvc = MemoryVersionControl()
        mvc.snapshot([{"content": "A"}] * 10)
        mvc.snapshot([{"content": "A"}] * 100)  # 900% Änderung
        d = mvc.detect_drift(max_change_rate=20.0)
        assert d["drift_detected"]

    def test_latest(self) -> None:
        mvc = MemoryVersionControl()
        mvc.snapshot([])
        mvc.snapshot([{"content": "X"}])
        assert mvc.latest().entry_count == 1


class TestDuplicateDetector:
    def test_exact_duplicates(self) -> None:
        entries = [
            {"content": "Die Sonne scheint heute"},
            {"content": "Die Sonne scheint heute"},
            {"content": "Es regnet draußen"},
        ]
        dups = DuplicateDetector.find_duplicates(entries)
        assert len(dups) == 1
        assert dups[0][0] == 0 and dups[0][1] == 1

    def test_near_duplicates(self) -> None:
        entries = [
            {"content": "Die Versicherung bietet BU-Schutz für Kunden"},
            {"content": "Die Versicherung bietet BU-Schutz für alle Kunden"},
            {"content": "Heute ist Montag"},
        ]
        dups = DuplicateDetector.find_duplicates(entries, threshold=0.8)
        assert len(dups) >= 1

    def test_no_duplicates(self) -> None:
        entries = [
            {"content": "Hund"},
            {"content": "Katze"},
        ]
        dups = DuplicateDetector.find_duplicates(entries)
        assert len(dups) == 0

    def test_empty(self) -> None:
        dups = DuplicateDetector.find_duplicates([])
        assert len(dups) == 0
