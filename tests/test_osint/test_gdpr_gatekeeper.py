"""Tests for GDPR Gatekeeper."""
from __future__ import annotations

import pytest
from jarvis.osint.gdpr_gatekeeper import GDPRGatekeeper
from jarvis.osint.models import HIMRequest, GDPRViolationError


def _req(**kw) -> HIMRequest:
    defaults = {
        "target_name": "Test User",
        "requester_justification": "Valid justification for testing",
    }
    defaults.update(kw)
    return HIMRequest(**defaults)


def test_missing_justification_raises():
    gk = GDPRGatekeeper()
    with pytest.raises(GDPRViolationError, match="justification"):
        gk.check(_req(requester_justification="short"))


def test_public_figure_all_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(), github_followers=100)
    assert scope.is_public_figure is True
    assert "social" in scope.allowed_collectors


def test_private_person_social_blocked():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(), github_followers=5)
    assert scope.is_public_figure is False
    assert "social" not in scope.allowed_collectors
    assert "linkedin" not in scope.allowed_collectors


def test_private_person_deep_blocked():
    gk = GDPRGatekeeper()
    with pytest.raises(GDPRViolationError, match="deep"):
        gk.check(_req(depth="deep"), github_followers=5)


def test_project_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(target_type="project"))
    assert "github" in scope.allowed_collectors
    assert "web" in scope.allowed_collectors


def test_org_collectors():
    gk = GDPRGatekeeper()
    scope = gk.check(_req(target_type="org"))
    assert "web" in scope.allowed_collectors
    assert "github" in scope.allowed_collectors
