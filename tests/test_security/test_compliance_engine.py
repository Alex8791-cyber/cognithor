"""Tests for GDPR Compliance Engine."""

from __future__ import annotations

import pytest
from pathlib import Path
from jarvis.security.compliance_engine import ComplianceEngine, ComplianceViolation
from jarvis.security.consent import ConsentManager
from jarvis.security.gdpr import ProcessingBasis, DataPurpose


@pytest.fixture
def consent_mgr(tmp_path):
    return ConsentManager(db_path=str(tmp_path / "consent.db"))


@pytest.fixture
def engine(consent_mgr):
    return ComplianceEngine(consent_manager=consent_mgr)


def test_blocks_without_consent(engine):
    with pytest.raises(ComplianceViolation, match="No consent"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.CONVERSATION,
        )


def test_allows_with_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    # Should not raise
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.CONVERSATION,
    )


def test_legitimate_interest_no_consent_needed(engine):
    # Security monitoring doesn't need consent
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
        purpose=DataPurpose.SECURITY,
    )


def test_privacy_mode_blocks_storage(engine):
    engine.set_privacy_mode(True)
    with pytest.raises(ComplianceViolation, match="[Pp]rivacy mode"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
            purpose=DataPurpose.CONVERSATION,
        )


def test_privacy_mode_allows_security(engine):
    engine.set_privacy_mode(True)
    # Security purpose should still work in privacy mode
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
        purpose=DataPurpose.SECURITY,
    )


def test_osint_requires_explicit_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    with pytest.raises(ComplianceViolation, match="[Oo]sint"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.OSINT,
        )


def test_osint_allowed_with_osint_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.grant_consent("user1", "telegram", "osint")
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.OSINT,
    )


def test_disabled_engine_allows_everything(consent_mgr):
    engine = ComplianceEngine(consent_manager=consent_mgr, enabled=False)
    # Should not raise even without consent
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.CONVERSATION,
    )


def test_fail_closed_when_consent_manager_none():
    """C3: Engine MUST block when consent store is unavailable."""
    engine = ComplianceEngine(consent_manager=None, enabled=True)
    with pytest.raises(ComplianceViolation, match="[Cc]onsent store unavailable"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.CONVERSATION,
        )


def test_system_channel_exempt_from_consent():
    """I1/I2: Cron jobs and sub-agents use legitimate interest, not consent."""
    engine = ComplianceEngine(consent_manager=None, enabled=True)
    # Should NOT raise — system channels are exempt
    for channel in ("cron", "sub_agent", "system", "evolution", "heartbeat"):
        engine.check(
            user_id="system",
            channel=channel,
            legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
            purpose=DataPurpose.SECURITY,
        )


def test_consent_required_false_skips_check():
    """C5: consent_required=False disables consent checks."""
    engine = ComplianceEngine(consent_manager=None, enabled=True, consent_required=False)
    # Should not raise even without consent manager
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.CONVERSATION,
    )
