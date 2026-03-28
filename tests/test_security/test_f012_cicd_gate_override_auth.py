"""Tests fuer F-012: CI/CD Gate Override muss Authentifizierung/Autorisierung pruefen.

Prueft dass:
  - override() nur autorisierte Rollen akzeptiert
  - override() bei unbekannter Rolle PermissionError wirft
  - override() eine nicht-leere Begruendung (mind. 10 Zeichen) verlangt
  - Erfolgreiche Overrides korrekt im Audit-Log landen
  - Abgelehnte Overrides ebenfalls im Audit-Log landen
  - AUTHORIZED_OVERRIDE_ROLES existiert und nicht leer ist
  - Die funktionale Korrektheit (Verdict-Aenderung) erhalten bleibt
"""

from __future__ import annotations

import pytest

from jarvis.security.cicd_gate import GateVerdict, SecurityGate


def _make_gate_with_fail() -> tuple[SecurityGate, str]:
    """Erstellt ein Gate mit einem FAIL-Ergebnis und gibt (gate, gate_id) zurueck."""
    gate = SecurityGate()
    result = gate.evaluate(
        {
            "stages": [
                {
                    "stage": "sast",
                    "result": "failed",
                    "findings": [{"severity": "critical", "title": "test"}],
                }
            ]
        }
    )
    assert result.verdict == GateVerdict.FAIL
    return gate, result.gate_id


class TestAuthorizedOverrideRoles:
    """Prueft dass AUTHORIZED_OVERRIDE_ROLES korrekt definiert ist."""

    def test_roles_exist(self) -> None:
        assert hasattr(SecurityGate, "AUTHORIZED_OVERRIDE_ROLES")

    def test_roles_not_empty(self) -> None:
        assert len(SecurityGate.AUTHORIZED_OVERRIDE_ROLES) > 0

    def test_roles_is_set(self) -> None:
        assert isinstance(SecurityGate.AUTHORIZED_OVERRIDE_ROLES, set)

    def test_known_roles(self) -> None:
        roles = SecurityGate.AUTHORIZED_OVERRIDE_ROLES
        assert "admin" in roles
        assert "security-lead" in roles


class TestOverrideRoleCheck:
    """Prueft dass nur autorisierte Rollen overriden koennen."""

    def test_unauthorized_role_raises(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(PermissionError, match="nicht autorisiert|not authorized"):
            gate.override(gate_id, by="random-user", reason="I want to deploy now please")

    def test_empty_role_raises(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(PermissionError):
            gate.override(gate_id, by="", reason="This is a valid reason")

    def test_admin_role_allowed(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        result = gate.override(gate_id, by="admin", reason="Hotfix fuer Produktion, getestet")
        assert result is not None
        assert result.verdict == GateVerdict.OVERRIDE

    def test_security_lead_role_allowed(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        result = gate.override(
            gate_id, by="security-lead", reason="False positive, manuell verifiziert"
        )
        assert result is not None
        assert result.verdict == GateVerdict.OVERRIDE

    def test_release_manager_role_allowed(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        result = gate.override(
            gate_id, by="release-manager", reason="Kritischer Hotfix, Rollback geplant"
        )
        assert result is not None
        assert result.verdict == GateVerdict.OVERRIDE

    def test_case_sensitive_roles(self) -> None:
        """Rollen sind case-sensitive — 'Admin' != 'admin'."""
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(PermissionError):
            gate.override(gate_id, by="Admin", reason="This is a valid reason text")


class TestOverrideReasonValidation:
    """Prueft dass eine sinnvolle Begruendung verlangt wird."""

    def test_empty_reason_raises(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(ValueError, match="mindestens 10 Zeichen|at least 10 char"):
            gate.override(gate_id, by="admin", reason="")

    def test_short_reason_raises(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(ValueError, match="mindestens 10 Zeichen|at least 10 char"):
            gate.override(gate_id, by="admin", reason="too short")

    def test_whitespace_only_reason_raises(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(ValueError):
            gate.override(gate_id, by="admin", reason="          ")

    def test_valid_reason_accepted(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        result = gate.override(
            gate_id, by="admin", reason="Hotfix fuer kritischen Bug in Produktion"
        )
        assert result is not None


class TestAuditLog:
    """Prueft dass alle Override-Versuche im Audit-Log landen."""

    def test_successful_override_logged(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        gate.override(gate_id, by="admin", reason="Genehmigt nach manueller Pruefung")
        assert len(gate._audit_log) == 1
        entry = gate._audit_log[0]
        assert entry["action"] == "override_approved"
        assert entry["gate_id"] == gate_id
        assert entry["by"] == "admin"
        assert "timestamp" in entry

    def test_denied_role_logged(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(PermissionError):
            gate.override(gate_id, by="hacker", reason="Ich will deployen bitte")
        assert len(gate._audit_log) == 1
        entry = gate._audit_log[0]
        assert entry["action"] == "override_denied"
        assert entry["detail"] == "unauthorized_role"
        assert entry["by"] == "hacker"

    def test_denied_reason_logged(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        with pytest.raises(ValueError):
            gate.override(gate_id, by="admin", reason="short")
        assert len(gate._audit_log) == 1
        entry = gate._audit_log[0]
        assert entry["action"] == "override_denied"
        assert entry["detail"] == "reason_too_short"

    def test_previous_verdict_captured(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        gate.override(gate_id, by="admin", reason="Override nach Security Review Freigabe")
        entry = gate._audit_log[0]
        assert entry["previous_verdict"] == "fail"

    def test_multiple_attempts_all_logged(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        # 1. Denied (bad role)
        with pytest.raises(PermissionError):
            gate.override(gate_id, by="nobody", reason="Versuch eins bitte deployen")
        # 2. Denied (bad reason)
        with pytest.raises(ValueError):
            gate.override(gate_id, by="admin", reason="kurz")
        # 3. Success
        gate.override(gate_id, by="admin", reason="Dritter Versuch, jetzt mit Begruendung")
        assert len(gate._audit_log) == 3
        assert gate._audit_log[0]["action"] == "override_denied"
        assert gate._audit_log[1]["action"] == "override_denied"
        assert gate._audit_log[2]["action"] == "override_approved"


class TestFunctionalCorrectness:
    """Prueft dass die Sicherheitsaenderungen die Funktionalitaet nicht brechen."""

    def test_override_changes_verdict(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        result = gate.override(gate_id, by="admin", reason="Hotfix genehmigt nach Review")
        assert result is not None
        assert result.verdict == GateVerdict.OVERRIDE
        assert result.override_by == "admin"
        assert result.override_reason == "Hotfix genehmigt nach Review"

    def test_unknown_gate_id_returns_none(self) -> None:
        gate, _ = _make_gate_with_fail()
        result = gate.override(
            "nonexistent-id", by="admin", reason="Existiert nicht, aber gueltige Begruendung"
        )
        assert result is None

    def test_override_does_not_affect_other_results(self) -> None:
        gate = SecurityGate()
        r1 = gate.evaluate({"stages": [{"stage": "a", "result": "pass", "findings": []}]})
        r2 = gate.evaluate(
            {
                "stages": [
                    {
                        "stage": "b",
                        "result": "fail",
                        "findings": [{"severity": "critical", "title": "x"}],
                    }
                ]
            }
        )
        gate.override(r2.gate_id, by="admin", reason="Nur r2 overriden, nicht r1")
        assert r1.verdict == GateVerdict.PASS
        assert r2.verdict == GateVerdict.OVERRIDE

    def test_pass_rate_counts_override_as_pass(self) -> None:
        gate, gate_id = _make_gate_with_fail()
        gate.override(gate_id, by="admin", reason="Override fuer pass-rate Berechnung")
        assert gate.pass_rate == 100.0
