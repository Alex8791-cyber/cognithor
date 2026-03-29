"""Tests for GDPR Consent Manager."""

from __future__ import annotations

import pytest
from pathlib import Path
from jarvis.security.consent import ConsentManager


@pytest.fixture
def consent_mgr(tmp_path):
    db_path = tmp_path / "consent.db"
    return ConsentManager(db_path=str(db_path))


def test_no_consent_by_default(consent_mgr):
    assert consent_mgr.has_consent("user1", "telegram") is False


def test_grant_consent(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing", context="chat_123")
    assert consent_mgr.has_consent("user1", "telegram") is True


def test_withdraw_consent(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.withdraw_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram") is False


def test_consent_per_channel(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram") is True
    assert consent_mgr.has_consent("user1", "webui") is False


def test_consent_type_specific(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram", "data_processing") is True
    assert consent_mgr.has_consent("user1", "telegram", "cloud_llm") is False


def test_consent_versioning(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing", policy_version="1.0")
    assert consent_mgr.has_consent("user1", "telegram", policy_version="1.0") is True
    assert consent_mgr.has_consent("user1", "telegram", policy_version="2.0") is False


def test_get_user_consents(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.grant_consent("user1", "telegram", "cloud_llm")
    consents = consent_mgr.get_user_consents("user1")
    assert len(consents) == 2


def test_requires_consent_true_when_none(consent_mgr):
    assert consent_mgr.requires_consent("user1", "telegram") is True


def test_requires_consent_false_after_grant(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.requires_consent("user1", "telegram") is False


def test_delete_user(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.delete_user("user1")
    assert consent_mgr.has_consent("user1", "telegram") is False
