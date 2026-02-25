"""Tests: Marketplace-Governance.

Prüft:
  - ReputationEngine: Trust-Scores, Reviews, Abuse
  - SkillRecallManager: Rückrufe, Blockaden
  - AbuseReporter: Meldungen, Auto-Investigate
  - GovernancePolicy: Automatische Regeln
"""

from __future__ import annotations

import pytest

from jarvis.skills.governance import (
    AbuseReport,
    AbuseReporter,
    GovernancePolicy,
    GovernanceRule,
    RecallNotice,
    RecallReason,
    ReputationEngine,
    ReputationScore,
    SkillRecallManager,
    TrustLevel,
)


# ============================================================================
# Reputation Engine
# ============================================================================


class TestReputationEngine:
    def test_initial_score(self) -> None:
        engine = ReputationEngine()
        score = engine.get_or_create("skill-1")
        assert score.score == 50.0
        assert score.trust_level == TrustLevel.MODERATE

    def test_positive_review(self) -> None:
        engine = ReputationEngine()
        score = engine.add_review("skill-1", positive=True)
        assert score.score > 50.0
        assert score.positive_reviews == 1

    def test_negative_review(self) -> None:
        engine = ReputationEngine()
        score = engine.add_review("skill-1", positive=False)
        assert score.score < 50.0
        assert score.negative_reviews == 1

    def test_abuse_report_penalty(self) -> None:
        engine = ReputationEngine()
        engine.get_or_create("skill-1")
        score = engine.report_abuse("skill-1")
        assert score.score == 40.0  # 50 - 10
        assert score.abuse_reports == 1

    def test_recall_heavy_penalty(self) -> None:
        engine = ReputationEngine()
        engine.get_or_create("skill-1")
        score = engine.apply_recall("skill-1")
        assert score.score == 25.0  # 50 - 25

    def test_security_scan_passed(self) -> None:
        engine = ReputationEngine()
        engine.get_or_create("skill-1")
        score = engine.apply_security_result("skill-1", passed=True)
        assert score.score == 55.0

    def test_security_scan_failed(self) -> None:
        engine = ReputationEngine()
        engine.get_or_create("skill-1")
        score = engine.apply_security_result("skill-1", passed=False)
        assert score.score == 35.0

    def test_score_capped_at_100(self) -> None:
        engine = ReputationEngine()
        score = engine.get_or_create("skill-1")
        score.score = 98.0
        engine.apply_security_result("skill-1", passed=True)
        assert engine.get_score("skill-1").score == 100.0

    def test_score_floor_at_0(self) -> None:
        engine = ReputationEngine()
        score = engine.get_or_create("skill-1")
        score.score = 5.0
        engine.report_abuse("skill-1")
        assert engine.get_score("skill-1").score == 0.0

    def test_trust_levels(self) -> None:
        engine = ReputationEngine()
        s = engine.get_or_create("s1")
        s.score = 90; assert s.trust_level == TrustLevel.VERIFIED
        s.score = 70; assert s.trust_level == TrustLevel.HIGH
        s.score = 50; assert s.trust_level == TrustLevel.MODERATE
        s.score = 30; assert s.trust_level == TrustLevel.LOW
        s.score = 10; assert s.trust_level == TrustLevel.UNTRUSTED

    def test_top_rated(self) -> None:
        engine = ReputationEngine()
        for i in range(5):
            s = engine.get_or_create(f"skill-{i}")
            s.score = (i + 1) * 20
        top = engine.top_rated(3)
        assert len(top) == 3
        assert top[0].score >= top[1].score >= top[2].score

    def test_flagged(self) -> None:
        engine = ReputationEngine()
        s1 = engine.get_or_create("good"); s1.score = 80
        s2 = engine.get_or_create("bad"); s2.score = 20
        flagged = engine.flagged(threshold=30)
        assert len(flagged) == 1
        assert flagged[0].entity_id == "bad"

    def test_stats(self) -> None:
        engine = ReputationEngine()
        engine.get_or_create("s1")
        engine.get_or_create("s2")
        stats = engine.stats()
        assert stats["total_entities"] == 2
        assert "trust_distribution" in stats

    def test_to_dict(self) -> None:
        engine = ReputationEngine()
        score = engine.get_or_create("skill-1")
        d = score.to_dict()
        assert "score" in d
        assert "trust_level" in d


# ============================================================================
# Skill Recall Manager
# ============================================================================


class TestSkillRecallManager:
    def test_issue_recall(self) -> None:
        mgr = SkillRecallManager()
        notice = mgr.issue_recall(
            "skill-bad", "BadSkill",
            RecallReason.MALICIOUS_BEHAVIOR, "Daten exfiltriert",
        )
        assert notice.skill_id == "skill-bad"
        assert mgr.is_recalled("skill-bad")
        assert not mgr.is_installable("skill-bad")

    def test_recall_affects_reputation(self) -> None:
        reputation = ReputationEngine()
        reputation.get_or_create("skill-bad")
        mgr = SkillRecallManager(reputation)
        mgr.issue_recall("skill-bad", "Bad", RecallReason.SECURITY_VULNERABILITY, "CVE found")
        score = reputation.get_score("skill-bad")
        assert score.score == 25.0  # 50 - 25

    def test_lift_recall(self) -> None:
        mgr = SkillRecallManager()
        mgr.issue_recall("skill-1", "Skill", RecallReason.QUALITY_ISSUE, "Buggy")
        assert mgr.is_recalled("skill-1")
        mgr.lift_recall("skill-1")
        assert not mgr.is_recalled("skill-1")
        assert mgr.is_installable("skill-1")

    def test_active_recalls(self) -> None:
        mgr = SkillRecallManager()
        mgr.issue_recall("s1", "S1", RecallReason.DATA_THEFT, "Data theft")
        mgr.issue_recall("s2", "S2", RecallReason.CRYPTO_MINING, "Mining")
        mgr.lift_recall("s1")
        assert len(mgr.active_recalls()) == 1

    def test_all_recalls(self) -> None:
        mgr = SkillRecallManager()
        mgr.issue_recall("s1", "S1", RecallReason.DATA_THEFT, "d")
        assert len(mgr.all_recalls()) == 1

    def test_stats(self) -> None:
        mgr = SkillRecallManager()
        mgr.issue_recall("s1", "S1", RecallReason.MALICIOUS_BEHAVIOR, "d")
        stats = mgr.stats()
        assert stats["total_recalls"] == 1
        assert stats["active_blocks"] == 1

    def test_recall_notice_to_dict(self) -> None:
        mgr = SkillRecallManager()
        notice = mgr.issue_recall("s1", "S1", RecallReason.PRIVACY_VIOLATION, "DSGVO")
        d = notice.to_dict()
        assert d["reason"] == "privacy_violation"
        assert "action_required" in d


# ============================================================================
# Abuse Reporter
# ============================================================================


class TestAbuseReporter:
    def test_submit_report(self) -> None:
        reporter = AbuseReporter()
        report = reporter.submit("skill-1", "user-1", "malware", "Looks suspicious")
        assert report.status == "open"
        assert reporter.report_count == 1

    def test_auto_investigate(self) -> None:
        reporter = AbuseReporter()
        for i in range(3):
            reporter.submit("skill-1", f"user-{i}", "malware", "Bad skill")
        reports = reporter.reports_for_skill("skill-1")
        assert all(r.status == "investigating" for r in reports)

    def test_resolve(self) -> None:
        reporter = AbuseReporter()
        report = reporter.submit("skill-1", "user-1", "spam", "Spam")
        assert reporter.resolve(report.report_id, "dismissed")
        assert reporter.open_reports() == []

    def test_open_reports(self) -> None:
        reporter = AbuseReporter()
        reporter.submit("s1", "u1", "spam", "d")
        reporter.submit("s2", "u2", "malware", "d")
        assert len(reporter.open_reports()) == 2

    def test_stats(self) -> None:
        reporter = AbuseReporter()
        reporter.submit("s1", "u1", "spam", "d")
        stats = reporter.stats()
        assert stats["total_reports"] == 1
        assert stats["open"] == 1

    def test_abuse_affects_reputation(self) -> None:
        rep = ReputationEngine()
        rep.get_or_create("s1")
        reporter = AbuseReporter(rep)
        reporter.submit("s1", "u1", "malware", "d")
        assert rep.get_score("s1").score < 50.0


# ============================================================================
# Governance Policy
# ============================================================================


class TestGovernancePolicy:
    def test_default_rules(self) -> None:
        policy = GovernancePolicy()
        assert policy.rule_count == 4

    def test_evaluate_low_score(self) -> None:
        policy = GovernancePolicy()
        score = ReputationScore(entity_id="s1", entity_type="skill", score=5)
        actions = policy.evaluate(score)
        action_types = [a["action"] for a in actions]
        assert "block" in action_types
        assert "notify" in action_types

    def test_evaluate_high_score(self) -> None:
        policy = GovernancePolicy()
        score = ReputationScore(entity_id="s1", entity_type="skill", score=90)
        actions = policy.evaluate(score)
        assert len(actions) == 0

    def test_evaluate_abuse_threshold(self) -> None:
        policy = GovernancePolicy()
        score = ReputationScore(entity_id="s1", entity_type="skill", score=50, abuse_reports=5)
        actions = policy.evaluate(score)
        assert any(a["action"] == "flag" for a in actions)

    def test_custom_rule(self) -> None:
        policy = GovernancePolicy(rules=[
            GovernanceRule("CUSTOM-1", "Block bei Score < 50", "score < 50", "block"),
        ])
        score = ReputationScore(entity_id="s1", entity_type="skill", score=40)
        actions = policy.evaluate(score)
        assert len(actions) == 1
        assert actions[0]["action"] == "block"

    def test_disabled_rule(self) -> None:
        policy = GovernancePolicy(rules=[
            GovernanceRule("CUSTOM-1", "Disabled", "score < 50", "block", enabled=False),
        ])
        score = ReputationScore(entity_id="s1", entity_type="skill", score=10)
        actions = policy.evaluate(score)
        assert len(actions) == 0

    def test_triggered_actions_history(self) -> None:
        policy = GovernancePolicy()
        score = ReputationScore(entity_id="s1", entity_type="skill", score=5)
        policy.evaluate(score)
        assert len(policy.triggered_actions()) >= 1

    def test_stats(self) -> None:
        policy = GovernancePolicy()
        stats = policy.stats()
        assert stats["total_rules"] == 4
        assert stats["enabled"] == 4
