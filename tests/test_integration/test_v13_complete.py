"""Tests: Impact Assessment, Code-Audit, Endnutzer-Portal.

Vervollständigung der v12-Lücken + neues Endnutzer-Portal.
"""

from __future__ import annotations

import pytest

# ============================================================================
# 1. Impact Assessment & Partizipative Governance
# ============================================================================

from jarvis.audit.impact_assessment import (
    BoardDecision,
    DimensionScore,
    EthicsBoard,
    ImpactAssessor,
    ImpactDimension,
    ImpactLikelihood,
    ImpactSeverity,
    MitigationStatus,
    MitigationTracker,
    StakeholderRegistry,
    StakeholderRole,
    VoteOption,
)


class TestDimensionScore:
    def test_risk_score(self) -> None:
        s = DimensionScore(ImpactDimension.PRIVACY, ImpactSeverity.HIGH, ImpactLikelihood.LIKELY)
        assert s.risk_score == 16  # 4 × 4

    def test_negligible_rare(self) -> None:
        s = DimensionScore(ImpactDimension.ENVIRONMENT, ImpactSeverity.NEGLIGIBLE, ImpactLikelihood.RARE)
        assert s.risk_score == 1

    def test_critical_certain(self) -> None:
        s = DimensionScore(ImpactDimension.SAFETY, ImpactSeverity.CRITICAL, ImpactLikelihood.ALMOST_CERTAIN)
        assert s.risk_score == 25


class TestImpactAssessment:
    def test_create(self) -> None:
        assessor = ImpactAssessor()
        a = assessor.create_assessment("Jarvis", "Beratung", "admin")
        assert a.assessment_id.startswith("IA-")

    def test_risk_level(self) -> None:
        assessor = ImpactAssessor()
        scores = [DimensionScore(ImpactDimension.PRIVACY, ImpactSeverity.HIGH, ImpactLikelihood.LIKELY)]
        a = assessor.create_assessment("Jarvis", "Test", "admin", scores)
        assert a.risk_level == "high"

    def test_low_risk(self) -> None:
        assessor = ImpactAssessor()
        scores = [DimensionScore(ImpactDimension.ENVIRONMENT, ImpactSeverity.LOW, ImpactLikelihood.UNLIKELY)]
        a = assessor.create_assessment("Tool", "Internal", "admin", scores)
        assert a.risk_level == "low"

    def test_high_risk_dimensions(self) -> None:
        assessor = ImpactAssessor()
        scores = [
            DimensionScore(ImpactDimension.PRIVACY, ImpactSeverity.HIGH, ImpactLikelihood.LIKELY),
            DimensionScore(ImpactDimension.ENVIRONMENT, ImpactSeverity.LOW, ImpactLikelihood.RARE),
        ]
        a = assessor.create_assessment("Test", "Test", "admin", scores)
        assert len(a.high_risk_dimensions) == 1

    def test_jarvis_insurance(self) -> None:
        assessor = ImpactAssessor()
        a = assessor.assess_jarvis_insurance()
        assert a.system_name == "Jarvis Versicherungsberater"
        assert len(a.scores) == 5
        assert a.risk_level in ("moderate", "high")


class TestStakeholderRegistry:
    def test_register(self) -> None:
        reg = StakeholderRegistry()
        sh = reg.register("Dr. Müller", StakeholderRole.ETHICS_EXPERT, organization="Uni")
        assert sh.stakeholder_id.startswith("SH-")

    def test_consultation(self) -> None:
        reg = StakeholderRegistry()
        sh = reg.register("Alice", StakeholderRole.AFFECTED_PERSON)
        reg.record_consultation(sh.stakeholder_id, "Kein Einwand")
        assert sh.consulted

    def test_unconsulted(self) -> None:
        reg = StakeholderRegistry()
        reg.register("A", StakeholderRole.OPERATOR)
        reg.register("B", StakeholderRole.REGULATOR)
        assert len(reg.unconsulted()) == 2

    def test_consultation_rate(self) -> None:
        reg = StakeholderRegistry()
        sh = reg.register("A", StakeholderRole.OPERATOR)
        reg.register("B", StakeholderRole.REGULATOR)
        reg.record_consultation(sh.stakeholder_id, "OK")
        assert reg.consultation_rate() == 50.0

    def test_by_role(self) -> None:
        reg = StakeholderRegistry()
        reg.register("A", StakeholderRole.ETHICS_EXPERT)
        reg.register("B", StakeholderRole.ETHICS_EXPERT)
        reg.register("C", StakeholderRole.OPERATOR)
        assert len(reg.by_role(StakeholderRole.ETHICS_EXPERT)) == 2


class TestEthicsBoard:
    def test_create_decision(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Jarvis BU-Beratung freigeben?")
        assert d.decision_id.startswith("BD-")

    def test_voting(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Test?")
        board.cast_vote(d.decision_id, "v1", "Alice", VoteOption.APPROVE)
        board.cast_vote(d.decision_id, "v2", "Bob", VoteOption.APPROVE)
        board.cast_vote(d.decision_id, "v3", "Carol", VoteOption.ABSTAIN)
        assert len(d.votes) == 3

    def test_no_double_vote(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Test?")
        board.cast_vote(d.decision_id, "v1", "Alice", VoteOption.APPROVE)
        assert not board.cast_vote(d.decision_id, "v1", "Alice", VoteOption.REJECT)

    def test_approve(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Approve?")
        board.cast_vote(d.decision_id, "v1", "A", VoteOption.APPROVE)
        board.cast_vote(d.decision_id, "v2", "B", VoteOption.APPROVE)
        board.cast_vote(d.decision_id, "v3", "C", VoteOption.ABSTAIN)
        result = board.finalize(d.decision_id)
        assert result.final_decision == "approved"

    def test_veto(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Veto?")
        board.cast_vote(d.decision_id, "v1", "A", VoteOption.REJECT)
        board.cast_vote(d.decision_id, "v2", "B", VoteOption.REJECT)
        board.cast_vote(d.decision_id, "v3", "C", VoteOption.APPROVE)
        result = board.finalize(d.decision_id)
        assert "rejected" in result.final_decision
        assert result.has_veto

    def test_conditional_approve(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("Conditional?")
        board.cast_vote(d.decision_id, "v1", "A", VoteOption.APPROVE)
        board.cast_vote(d.decision_id, "v2", "B", VoteOption.CONDITIONAL,
                        conditions=["Menschliche Aufsicht einbauen"])
        board.cast_vote(d.decision_id, "v3", "C", VoteOption.APPROVE)
        result = board.finalize(d.decision_id)
        assert result.final_decision == "approved_conditional"
        assert len(result.requires_conditions) == 1

    def test_no_quorum(self) -> None:
        board = EthicsBoard()
        d = board.create_decision("No quorum?")
        board.cast_vote(d.decision_id, "v1", "A", VoteOption.APPROVE)
        result = board.finalize(d.decision_id)
        assert result.final_decision == "deferred_no_quorum"

    def test_stats(self) -> None:
        board = EthicsBoard()
        board.create_decision("Test")
        stats = board.stats()
        assert stats["total_decisions"] == 1


class TestMitigationTracker:
    def test_add(self) -> None:
        tracker = MitigationTracker()
        m = tracker.add("IA-0001", ImpactDimension.PRIVACY, "Datenminimierung einführen")
        assert m.status == MitigationStatus.PLANNED

    def test_update_status(self) -> None:
        tracker = MitigationTracker()
        m = tracker.add("IA-0001", ImpactDimension.PRIVACY, "Test")
        tracker.update_status(m.mitigation_id, MitigationStatus.IMPLEMENTED, 0.8)
        assert m.effectiveness == 0.8

    def test_completion_rate(self) -> None:
        tracker = MitigationTracker()
        m1 = tracker.add("IA-0001", ImpactDimension.PRIVACY, "A")
        m2 = tracker.add("IA-0001", ImpactDimension.SAFETY, "B")
        tracker.update_status(m1.mitigation_id, MitigationStatus.VERIFIED, 0.9)
        assert tracker.completion_rate() == 50.0

    def test_by_assessment(self) -> None:
        tracker = MitigationTracker()
        tracker.add("IA-0001", ImpactDimension.PRIVACY, "A")
        tracker.add("IA-0002", ImpactDimension.SAFETY, "B")
        assert len(tracker.by_assessment("IA-0001")) == 1


class TestImpactAssessor:
    def test_full_workflow(self) -> None:
        assessor = ImpactAssessor()
        # Assessment
        a = assessor.assess_jarvis_insurance()
        # Stakeholder
        sh = assessor.stakeholders.register("Ethik-Prof", StakeholderRole.ETHICS_EXPERT)
        assessor.stakeholders.record_consultation(sh.stakeholder_id, "Zustimmung")
        # Mitigation
        assessor.mitigations.add(a.assessment_id, ImpactDimension.PRIVACY, "DSGVO-Audit")
        # Stats
        stats = assessor.stats()
        assert stats["total_assessments"] == 1
        assert stats["stakeholders"]["total"] == 1


# ============================================================================
# 2. Automatisierte Code-Analyse
# ============================================================================

from jarvis.security.code_audit import (
    CodeAuditor,
    PatternScanner,
    PatternSeverity,
    PermissionAnalyzer,
    RequiredPermission,
)


class TestPatternScanner:
    def test_clean_code(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("x = 1 + 2\nprint(x)")
        assert len(findings) == 0

    def test_eval_detected(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("result = eval(user_input)")
        assert len(findings) >= 1
        assert any(f.pattern.severity == PatternSeverity.CRITICAL for f in findings)

    def test_exec_detected(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("exec(code_string)")
        assert len(findings) >= 1

    def test_subprocess_shell(self) -> None:
        scanner = PatternScanner()
        code = "import subprocess\nsubprocess.run(cmd, shell=True)"
        findings = scanner.scan_code(code)
        assert len(findings) >= 1

    def test_os_system(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("os.system('rm -rf /')")
        assert len(findings) >= 1

    def test_requests_detected(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("response = requests.get('http://evil.com')")
        assert len(findings) >= 1

    def test_base64_decode(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("data = base64.b64decode(payload)")
        assert len(findings) >= 1

    def test_sql_injection(self) -> None:
        scanner = PatternScanner()
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        findings = scanner.scan_code(code)
        assert len(findings) >= 1

    def test_sensitive_paths(self) -> None:
        scanner = PatternScanner()
        findings = scanner.scan_code("open('/etc/passwd').read()")
        assert len(findings) >= 1

    def test_line_numbers(self) -> None:
        scanner = PatternScanner()
        code = "x = 1\ny = 2\nresult = eval('x+y')"
        findings = scanner.scan_code(code, "test.py")
        assert findings[0].line_number == 3
        assert findings[0].file_path == "test.py"

    def test_stats(self) -> None:
        scanner = PatternScanner()
        scanner.scan_code("eval('x')")
        stats = scanner.stats()
        assert stats["total_findings"] >= 1


class TestPermissionAnalyzer:
    def test_no_permissions(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("x = 1 + 2")
        assert len(perms) == 0

    def test_file_read(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("f = open('data.txt')")
        assert RequiredPermission.FILE_READ in perms

    def test_network(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("requests.get('http://api.example.com')")
        assert RequiredPermission.NETWORK in perms

    def test_shell_exec(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("subprocess.run(['ls'])")
        assert RequiredPermission.SHELL_EXEC in perms

    def test_risk_assessment_high(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("subprocess.run(cmd)")
        risk = analyzer.risk_assessment(perms)
        assert risk["risk_level"] == "high"

    def test_risk_assessment_none(self) -> None:
        analyzer = PermissionAnalyzer()
        perms = analyzer.analyze("x = 42")
        risk = analyzer.risk_assessment(perms)
        assert risk["risk_level"] == "none"


class TestCodeAuditor:
    def test_clean_skill(self) -> None:
        auditor = CodeAuditor()
        report = auditor.audit_skill("Calculator", "def add(a, b):\n    return a + b")
        assert report.passed
        assert report.overall_risk == "low"

    def test_dangerous_skill(self) -> None:
        auditor = CodeAuditor()
        code = """
import subprocess
def run_command(cmd):
    result = eval(cmd)
    subprocess.run(cmd, shell=True)
    return result
"""
        report = auditor.audit_skill("DangerousSkill", code)
        assert not report.passed
        assert report.overall_risk in ("critical", "high")
        assert len(report.recommendations) > 0

    def test_network_skill(self) -> None:
        auditor = CodeAuditor()
        code = "import requests\ndef fetch():\n    return requests.get('http://api.example.com')"
        report = auditor.audit_skill("NetworkSkill", code)
        assert "network" in report.permissions or "network" in report.permission_risk

    def test_pass_rate(self) -> None:
        auditor = CodeAuditor()
        auditor.audit_skill("Good", "x = 1")
        auditor.audit_skill("Bad", "eval(input())")
        rate = auditor.pass_rate()
        assert rate == 50.0

    def test_stats(self) -> None:
        auditor = CodeAuditor()
        auditor.audit_skill("Test", "eval('x')")
        stats = auditor.stats()
        assert stats["total_audits"] == 1


# ============================================================================
# 3. Endnutzer-Portal
# ============================================================================

from jarvis.core.user_portal import (
    ConsentManager,
    ConsentPurpose,
    ConsentStatus,
    DecisionViewBuilder,
    NotificationCenter,
    NotificationType,
    UserActivityLog,
    UserPortal,
)


class TestConsentManager:
    def test_request(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        assert c.status == ConsentStatus.PENDING

    def test_grant(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        mgr.grant(c.consent_id)
        assert c.status == ConsentStatus.GRANTED
        assert c.is_valid

    def test_deny(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.MARKETING)
        mgr.deny(c.consent_id)
        assert c.status == ConsentStatus.DENIED

    def test_withdraw(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        mgr.grant(c.consent_id)
        mgr.withdraw(c.consent_id)
        assert c.status == ConsentStatus.WITHDRAWN
        assert not c.is_valid

    def test_has_consent(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.INSURANCE_ADVICE)
        assert not mgr.has_consent("user-1", ConsentPurpose.INSURANCE_ADVICE)
        mgr.grant(c.consent_id)
        assert mgr.has_consent("user-1", ConsentPurpose.INSURANCE_ADVICE)

    def test_can_advise(self) -> None:
        mgr = ConsentManager()
        c1 = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        c2 = mgr.request_consent("user-1", ConsentPurpose.INSURANCE_ADVICE)
        assert not mgr.can_advise("user-1")
        mgr.grant(c1.consent_id)
        mgr.grant(c2.consent_id)
        assert mgr.can_advise("user-1")

    def test_withdraw_all(self) -> None:
        mgr = ConsentManager()
        c1 = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        c2 = mgr.request_consent("user-1", ConsentPurpose.MARKETING)
        mgr.grant(c1.consent_id)
        mgr.grant(c2.consent_id)
        count = mgr.withdraw_all("user-1")
        assert count == 2

    def test_stats(self) -> None:
        mgr = ConsentManager()
        c = mgr.request_consent("user-1", ConsentPurpose.AI_PROCESSING)
        mgr.grant(c.consent_id)
        stats = mgr.stats()
        assert stats["granted"] == 1


class TestDecisionViewBuilder:
    def test_build(self) -> None:
        builder = DecisionViewBuilder()
        view = builder.build(
            "Welche BU passt?", "WWK BU Premium", 0.85,
            why=["Hohe Leistungsquote"],
            alternatives=["R&V Classic"],
        )
        assert view.confidence_label == "Hohe Sicherheit"
        assert len(view.why_this) == 1

    def test_low_confidence(self) -> None:
        builder = DecisionViewBuilder()
        view = builder.build("Test?", "Option A", 0.3)
        assert view.confidence_label == "Vorläufige Einschätzung"

    def test_very_high_confidence(self) -> None:
        builder = DecisionViewBuilder()
        view = builder.build("Test?", "Klar", 0.95)
        assert view.confidence_label == "Sehr hohe Sicherheit"

    def test_ai_disclosure(self) -> None:
        builder = DecisionViewBuilder()
        view = builder.build("Test?", "X", 0.8)
        assert "Art. 52" in view.ai_disclosure


class TestNotificationCenter:
    def test_notify(self) -> None:
        center = NotificationCenter()
        n = center.notify("user-1", NotificationType.CONSENT_REQUEST, "Titel", "Text")
        assert not n.read

    def test_mark_read(self) -> None:
        center = NotificationCenter()
        n = center.notify("user-1", NotificationType.DECISION_READY, "T", "M")
        center.mark_read(n.notification_id)
        assert n.read

    def test_unread(self) -> None:
        center = NotificationCenter()
        center.notify("user-1", NotificationType.CONSENT_REQUEST, "A", "B")
        center.notify("user-1", NotificationType.SECURITY_ALERT, "C", "D")
        assert len(center.unread("user-1")) == 2

    def test_stats(self) -> None:
        center = NotificationCenter()
        center.notify("user-1", NotificationType.DATA_USAGE, "T", "M")
        stats = center.stats()
        assert stats["unread"] == 1


class TestUserActivityLog:
    def test_log(self) -> None:
        log = UserActivityLog()
        a = log.log("user-1", "beratung_gestartet", data_accessed=["Alter", "Beruf"])
        assert a.activity_id.startswith("ACT-")

    def test_history(self) -> None:
        log = UserActivityLog()
        log.log("user-1", "action1")
        log.log("user-1", "action2")
        log.log("user-1", "action3")
        history = log.user_history("user-1", 2)
        assert len(history) == 2

    def test_data_access_report(self) -> None:
        log = UserActivityLog()
        log.log("user-1", "a1", data_accessed=["Alter", "Beruf"])
        log.log("user-1", "a2", data_accessed=["Alter", "Gesundheit"])
        report = log.data_access_report("user-1")
        assert report["Alter"] == 2
        assert report["Beruf"] == 1

    def test_export(self) -> None:
        log = UserActivityLog()
        log.log("user-1", "test", data_accessed=["X"])
        export = log.export_user_data("user-1")
        assert export["user_id"] == "user-1"
        assert len(export["activities"]) == 1

    def test_delete(self) -> None:
        log = UserActivityLog()
        log.log("user-1", "a1")
        log.log("user-1", "a2")
        deleted = log.delete_user_data("user-1")
        assert deleted == 2
        assert log.user_count == 0


class TestUserPortal:
    def test_onboard(self) -> None:
        portal = UserPortal()
        consents = portal.onboard_user("user-1")
        assert len(consents) == len(ConsentPurpose)

    def test_dashboard(self) -> None:
        portal = UserPortal()
        portal.onboard_user("user-1")
        dashboard = portal.user_dashboard("user-1")
        assert "consents" in dashboard
        assert "unread_notifications" in dashboard
        assert dashboard["unread_notifications"] >= 1

    def test_right_to_erasure(self) -> None:
        portal = UserPortal()
        consents = portal.onboard_user("user-1")
        for c in consents:
            portal.consents.grant(c.consent_id)
        portal.activities.log("user-1", "test")
        result = portal.exercise_right_to_erasure("user-1")
        assert result["consents_withdrawn"] == len(ConsentPurpose)
        assert result["activities_deleted"] >= 1

    def test_full_workflow(self) -> None:
        portal = UserPortal()
        # 1. Onboarding
        consents = portal.onboard_user("user-1")
        # 2. Einwilligungen erteilen
        for c in consents:
            if c.purpose in (ConsentPurpose.AI_PROCESSING, ConsentPurpose.INSURANCE_ADVICE):
                portal.consents.grant(c.consent_id)
        assert portal.consents.can_advise("user-1")
        # 3. Beratung
        portal.decisions.build(
            "Welche BU?", "WWK Premium", 0.85,
            why=["Hohe Leistungsquote"],
        )
        portal.activities.log("user-1", "bu_beratung", data_accessed=["Alter", "Beruf"])
        # 4. Dashboard
        dashboard = portal.user_dashboard("user-1")
        assert dashboard["can_advise"]
        assert len(dashboard["decisions"]) == 1

    def test_stats(self) -> None:
        portal = UserPortal()
        portal.onboard_user("user-1")
        stats = portal.stats()
        assert stats["consents"]["total_users"] == 1
