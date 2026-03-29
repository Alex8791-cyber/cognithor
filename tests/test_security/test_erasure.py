"""Tests for GDPR Erasure API."""

from __future__ import annotations

import pytest
from jarvis.security.gdpr import GDPRComplianceManager, DataCategory
from jarvis.security.consent import ConsentManager


@pytest.fixture
def consent_mgr(tmp_path):
    return ConsentManager(db_path=str(tmp_path / "consent.db"))


@pytest.fixture
def gdpr_mgr():
    return GDPRComplianceManager()


@pytest.mark.asyncio
async def test_erase_all_deletes_processing_logs(gdpr_mgr):
    gdpr_mgr.log_processing("user1", DataCategory.QUERY, "test")
    gdpr_mgr.log_processing("user1", DataCategory.QUERY, "test2")
    result = await gdpr_mgr.erasure.erase_all("user1")
    assert result["processing_logs"] >= 2


@pytest.mark.asyncio
async def test_erase_all_deletes_consents(gdpr_mgr, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.grant_consent("user1", "webui", "data_processing")
    result = await gdpr_mgr.erasure.erase_all("user1", consent_manager=consent_mgr)
    assert result["consents"] >= 2
    assert consent_mgr.has_consent("user1", "telegram") is False


@pytest.mark.asyncio
async def test_erase_all_calls_handlers(gdpr_mgr):
    deleted = []
    gdpr_mgr.erasure.register_handler(lambda uid: (deleted.append(uid), 1)[1])
    result = await gdpr_mgr.erasure.erase_all("user1")
    assert "user1" in deleted
    assert result["external_handlers"] >= 1
