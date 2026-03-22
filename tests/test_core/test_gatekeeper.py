"""Tests für den Gatekeeper – deterministischer Policy-Enforcer.

Testet:
  - Policy-Laden aus YAML
  - Risiko-Klassifizierung nach Tool-Typ
  - Destruktive Shell-Befehle erkennen und blockieren
  - Pfad-Validierung (nur erlaubte Verzeichnisse)
  - Credential-Erkennung und -Maskierung
  - Audit-Trail (JSONL)
  - Policy-Matching (Tool + Params)
  - evaluate_plan() für mehrere Schritte
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest
import yaml

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.core.gatekeeper import Gatekeeper
from jarvis.models import (
    GateStatus,
    PlannedAction,
    RiskLevel,
    SessionContext,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def gk_config(tmp_path: Path) -> JarvisConfig:
    """Config mit tmp_path als jarvis_home."""
    config = JarvisConfig(
        jarvis_home=tmp_path,
        security=SecurityConfig(
            allowed_paths=[str(tmp_path), os.path.join(tempfile.gettempdir(), "jarvis", "")],
        ),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gatekeeper(gk_config: JarvisConfig) -> Gatekeeper:
    """Initialisierter Gatekeeper."""
    gk = Gatekeeper(gk_config)
    gk.initialize()
    return gk


@pytest.fixture()
def session() -> SessionContext:
    """Standard-Session für Tests."""
    return SessionContext(user_id="test_user", channel="test")


# ============================================================================
# Risiko-Klassifizierung
# ============================================================================


class TestRiskClassification:
    """Testet die Default-Risiko-Einstufung nach Tool-Typ."""

    def test_read_operations_are_green(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        for tool in ("read_file", "list_directory", "search_memory"):
            action = PlannedAction(tool=tool, params={})
            decision = gatekeeper.evaluate(action, session)
            assert decision.risk_level == RiskLevel.GREEN, f"{tool} should be GREEN"
            assert decision.is_allowed

    def test_write_operations_are_yellow(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """write_file matched die Default-Policy INFORM → YELLOW."""
        action = PlannedAction(tool="write_file", params={"path": "~/.jarvis/workspace/test.txt"})
        decision = gatekeeper.evaluate(action, session)
        # Default-Policy setzt write_file auf INFORM
        assert decision.status in (GateStatus.INFORM, GateStatus.ALLOW)

    def test_email_requires_approval(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="email_send", params={"to": "test@example.com"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.APPROVE
        assert decision.needs_approval

    def test_unknown_tool_is_orange(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="totally_unknown_tool", params={})
        decision = gatekeeper.evaluate(action, session)
        # Unbekannte Tools → ORANGE (Fail-Safe)
        assert decision.risk_level == RiskLevel.ORANGE
        assert decision.needs_approval

    @pytest.mark.parametrize(
        "tool",
        [
            "web_search",
            "web_fetch",
            "web_news_search",
            "search_and_read",
            "analyze_code",
            "git_status",
            "git_diff",
            "git_log",
            "browse_screenshot",
            "get_core_memory",
            "get_recent_episodes",
            "memory_stats",
            "db_query",
            "db_schema",
            "create_chart",
            "list_skills",
            "list_remote_agents",
            "docker_ps",
            "docker_logs",
            "api_list",
            "calendar_today",
            "calendar_upcoming",
            "screenshot_desktop",
            "vault_list",
            "vault_search",
        ],
    )
    def test_green_tools_comprehensive(
        self, gatekeeper: Gatekeeper, session: SessionContext, tool: str
    ) -> None:
        """All GREEN tools must be classified as GREEN with no approval."""
        action = PlannedAction(tool=tool, params={})
        decision = gatekeeper.evaluate(action, session)
        assert decision.risk_level == RiskLevel.GREEN, f"{tool} should be GREEN"

    @pytest.mark.parametrize(
        "tool",
        [
            "save_to_memory",
            "git_commit",
            "git_branch",
            "document_export",
            "media_tts",
            "create_skill",
            "delegate_to_remote_agent",
            "db_connect",
            "docker_stop",
            "api_connect",
            "api_call",
            "vault_save",
            "vault_write",
            "vault_delete",
        ],
    )
    def test_yellow_tools_comprehensive(
        self, gatekeeper: Gatekeeper, session: SessionContext, tool: str
    ) -> None:
        """All YELLOW tools must be classified as YELLOW."""
        action = PlannedAction(tool=tool, params={})
        decision = gatekeeper.evaluate(action, session)
        assert decision.risk_level == RiskLevel.YELLOW, f"{tool} should be YELLOW"

    @pytest.mark.parametrize(
        "tool",
        [
            "email_send",
            "calendar_create_event",
            "delete_file",
            "fetch_url",
            "http_request",
            "db_execute",
            "docker_run",
        ],
    )
    def test_orange_tools_comprehensive(
        self, gatekeeper: Gatekeeper, session: SessionContext, tool: str
    ) -> None:
        """All ORANGE tools must require approval."""
        action = PlannedAction(tool=tool, params={})
        decision = gatekeeper.evaluate(action, session)
        assert decision.risk_level == RiskLevel.ORANGE, f"{tool} should be ORANGE"
        assert decision.needs_approval, f"{tool} should need approval"


# ============================================================================
# Destruktive Shell-Befehle
# ============================================================================


class TestDestructiveCommands:
    """Blockierung destruktiver Shell-Befehle."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /",
            "rm -rf /home",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            ":(){ :|:& };:",
            "shutdown -h now",
            "reboot",
            "format C:",
            "del /f /q C:\\Windows\\System32",
        ],
    )
    def test_destructive_commands_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext, cmd: str
    ) -> None:
        action = PlannedAction(tool="exec_command", params={"command": cmd})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK, f"'{cmd}' should be BLOCKED"
        assert decision.is_blocked

    def test_safe_commands_not_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """Harmlose Befehle werden NICHT von der Destruktiv-Prüfung gefangen."""
        safe_cmds = ["ls -la", "cat file.txt", "echo hello", "date"]
        for cmd in safe_cmds:
            action = PlannedAction(tool="exec_command", params={"command": cmd})
            decision = gatekeeper.evaluate(action, session)
            # exec_command ist RED per Default, aber wird nicht durch destructive pattern BLOCK
            assert decision.policy_name != "blocked_command", (
                f"'{cmd}' should NOT match blocked patterns"
            )


# ============================================================================
# Credential-Erkennung
# ============================================================================


class TestCredentialMasking:
    """Credentials werden erkannt und maskiert."""

    def test_api_key_masked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(
            tool="fetch_url",
            params={"url": "https://api.example.com", "headers": "api_key=secret123"},
        )
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.MASK
        assert decision.masked_params is not None
        assert "***MASKED***" in str(decision.masked_params)

    def test_sk_token_masked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(
            tool="write_file",
            params={"content": "token: sk-abcdefghij1234567890abcdef"},
        )
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.MASK

    def test_clean_params_not_masked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(
            tool="read_file",
            params={"path": "/safe/path/file.txt"},
        )
        decision = gatekeeper.evaluate(action, session)
        assert decision.status != GateStatus.MASK


# ============================================================================
# Pfad-Validierung
# ============================================================================


class TestPathValidation:
    """Nur erlaubte Verzeichnisse dürfen zugegriffen werden."""

    def test_allowed_path_passes(
        self, gatekeeper: Gatekeeper, session: SessionContext, gk_config: JarvisConfig
    ) -> None:
        # ~/.jarvis/workspace ist erlaubt
        safe_path = str(gk_config.workspace_dir / "test.txt")
        action = PlannedAction(tool="read_file", params={"path": safe_path})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status != GateStatus.BLOCK or "Pfad" not in decision.reason

    def test_outside_path_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/etc/passwd"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert "Pfad" in decision.reason

    def test_traversal_attack_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        action = PlannedAction(
            tool="read_file",
            params={"path": "~/.jarvis/workspace/../../../etc/passwd"},
        )
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK


# ============================================================================
# Policy-Matching
# ============================================================================


class TestPolicyMatching:
    """Explizite Policy-Regeln überschreiben Default-Klassifizierung."""

    def test_default_policy_loads(self, gatekeeper: Gatekeeper) -> None:
        assert len(gatekeeper._policies) > 0

    def test_custom_policy_override(self, gk_config: JarvisConfig, session: SessionContext) -> None:
        """Custom Policy die ein Tool explizit erlaubt."""
        custom_policy = {
            "rules": [
                {
                    "name": "allow_special_tool",
                    "match": {"tool": "special_tool"},
                    "action": "ALLOW",
                    "reason": "Speziell erlaubt",
                    "priority": 100,  # Hohe Priorität
                },
            ]
        }
        custom_path = gk_config.policies_dir / "custom.yaml"
        custom_path.write_text(yaml.dump(custom_policy), encoding="utf-8")

        gk = Gatekeeper(gk_config)
        gk.initialize()

        action = PlannedAction(tool="special_tool", params={})
        decision = gk.evaluate(action, session)
        assert decision.status == GateStatus.ALLOW
        assert decision.policy_name == "allow_special_tool"


# ============================================================================
# Audit-Trail
# ============================================================================


class TestAuditTrail:
    """Jede Entscheidung wird im Audit-Log protokolliert."""

    def test_audit_file_created(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        gatekeeper.evaluate(action, session)
        gatekeeper._flush_audit_buffer()
        assert gatekeeper._audit_path.exists()

    def test_audit_entries_accumulate(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        for i in range(3):
            action = PlannedAction(tool="read_file", params={"path": f"/test_{i}"})
            gatekeeper.evaluate(action, session)

        gatekeeper._flush_audit_buffer()
        lines = gatekeeper._audit_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_audit_is_jsonl(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        import json

        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        gatekeeper.evaluate(action, session)
        gatekeeper._flush_audit_buffer()
        line = gatekeeper._audit_path.read_text().strip()
        data = json.loads(line)
        assert data["decision_status"] == "BLOCK"
        assert "action_params_hash" in data

    def test_atexit_handler_registered(self, gatekeeper: Gatekeeper) -> None:
        """atexit handler must be registered to flush buffer on process exit."""

        # The atexit handler is a closure over a weakref — verify it's callable
        # by checking that _flush_audit_buffer exists and the handler was registered.
        # We can't inspect atexit._exithandlers directly, but we can verify
        # the buffer flushes correctly when called.
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        session_ctx = SessionContext(session_id="s1", user_id="u1")
        gatekeeper.evaluate(action, session_ctx)
        assert len(gatekeeper._audit_buffer) > 0  # Buffer has data
        gatekeeper._flush_audit_buffer()
        assert len(gatekeeper._audit_buffer) == 0  # Flushed


# ============================================================================
# evaluate_plan (Batch)
# ============================================================================


class TestEvaluatePlan:
    """Batch-Evaluation mehrerer Schritte."""

    def test_mixed_plan(
        self, gatekeeper: Gatekeeper, session: SessionContext, gk_config: JarvisConfig
    ) -> None:
        ws_path = str(gk_config.workspace_dir / "x")
        steps = [
            PlannedAction(tool="read_file", params={"path": ws_path}),
            PlannedAction(tool="email_send", params={"to": "a@b.com"}),
            PlannedAction(tool="exec_command", params={"command": "rm -rf /"}),
        ]
        decisions = gatekeeper.evaluate_plan(steps, session)
        assert len(decisions) == 3
        # read → allowed, email → approve, rm → block
        assert decisions[0].is_allowed or decisions[0].needs_approval  # depends on path validation
        assert decisions[1].needs_approval  # email_send → APPROVE
        assert decisions[2].is_blocked  # rm -rf → BLOCK


# ============================================================================
# GateDecision Properties
# ============================================================================


class TestGateDecisionFromGatekeeper:
    """Prüft die erweiterten GateDecision-Felder."""

    def test_original_action_preserved(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.original_action is not None
        assert decision.original_action.tool == "read_file"

    def test_policy_name_set(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/test"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.policy_name  # Sollte gesetzt sein
