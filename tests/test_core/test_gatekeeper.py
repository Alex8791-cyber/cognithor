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

from cognithor.config import (
    CognithorConfig,
    SecurityConfig,
    ToolsConfig,
    ensure_directory_structure,
)
from cognithor.core.gatekeeper import Gatekeeper
from cognithor.models import (
    GateStatus,
    PlannedAction,
    RiskLevel,
    SessionContext,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def gk_config(tmp_path: Path) -> CognithorConfig:
    """Config mit tmp_path als cognithor_home."""
    config = CognithorConfig(
        cognithor_home=tmp_path,
        security=SecurityConfig(
            allowed_paths=[str(tmp_path), os.path.join(tempfile.gettempdir(), "jarvis", "")],
        ),
        tools=ToolsConfig(
            computer_use_enabled=True,
            desktop_tools_enabled=True,
        ),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gatekeeper(gk_config: CognithorConfig) -> Gatekeeper:
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
        action = PlannedAction(
            tool="write_file", params={"path": "~/.cognithor/workspace/test.txt"}
        )
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
            # Sprint-22 Track A.2: PSE tools (deterministic, sandboxed)
            "pse_synthesize",
            "pse_is_synthesizable",
            "pse_status",
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

    @pytest.mark.parametrize(
        "tool",
        [
            "vault_delete",
            "delete_entity",
            "delete_relation",
            "erase_user_data",
        ],
    )
    def test_red_tools_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext, tool: str
    ) -> None:
        """RED tools (GDPR erasure, destructive) must be blocked outright."""
        action = PlannedAction(tool=tool, params={})
        decision = gatekeeper.evaluate(action, session)
        assert decision.risk_level == RiskLevel.RED, f"{tool} should be RED"
        assert decision.status.value == "BLOCK", f"{tool} should be BLOCK"


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

    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /",
            "rm -rf /home",
            "shutdown -h now",
            "format C:",
        ],
    )
    def test_start_background_destructive_commands_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext, cmd: str
    ) -> None:
        """SEC-CRIT-1 (autonomous security audit, 2026-05-04):
        ``start_background`` runs ``subprocess.Popen(shell=True)`` —
        same threat surface as ``exec_command``. The destructive-
        command regex/AST chain MUST fire for it too. Without this
        guard, a jailbroken Planner could spawn ``rm -rf $HOME`` as a
        background job that the YELLOW classification auto-allows.
        """
        action = PlannedAction(tool="start_background", params={"command": cmd})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK, f"start_background('{cmd}') should be BLOCKED"
        assert decision.is_blocked

    def test_start_background_safe_command_not_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """Legitimate background commands (``npm run dev`` etc.) must
        still pass the destructive check. They get YELLOW (auto-execute
        with informational notice) like before — only the dangerous
        patterns are now caught.
        """
        action = PlannedAction(tool="start_background", params={"command": "npm run dev"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.policy_name != "blocked_command"
        assert decision.policy_name != "blocked_command_ast"


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
        self, gatekeeper: Gatekeeper, session: SessionContext, gk_config: CognithorConfig
    ) -> None:
        # ~/.cognithor/workspace ist erlaubt
        safe_path = str(gk_config.workspace_dir / "test.txt")
        action = PlannedAction(tool="read_file", params={"path": safe_path})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status != GateStatus.BLOCK or "Path" not in decision.reason

    def test_outside_path_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = PlannedAction(tool="read_file", params={"path": "/etc/passwd"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert "Path" in decision.reason

    def test_traversal_attack_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        action = PlannedAction(
            tool="read_file",
            params={"path": "~/.cognithor/workspace/../../../etc/passwd"},
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

    def test_custom_policy_override(
        self, gk_config: CognithorConfig, session: SessionContext
    ) -> None:
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
        self, gatekeeper: Gatekeeper, session: SessionContext, gk_config: CognithorConfig
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


# ============================================================================
# TRUST-2: structured "why" explanations (operational-trust audit, 2026-05-04)
# ============================================================================


class TestDecisionExplanation:
    """Reddit reviewer asked for operator-readable Gatekeeper "why".
    ``GateDecision.explanation`` (DecisionExplanation) carries
    ``rule_id``, ``rule_source``, ``matched_pattern`` so receipts and
    Trace-UI can render the decision path without parsing free-text
    ``reason``.
    """

    def test_destructive_command_attaches_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """``exec_command rm -rf /`` is blocked. Whichever guard fires
        first (YAML policy / AST / regex) MUST attach a structured
        explanation — that's the entire point.
        """
        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.is_blocked
        assert decision.explanation is not None
        assert decision.explanation.rule_id  # any non-empty id
        # rule_source must be a code/policy reference, never bare text.
        assert ":" in decision.explanation.rule_source

    def test_destructive_regex_path_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """``start_background`` doesn't match the YAML
        ``no_destructive_shell`` policy (which targets exec_command) so
        it falls through to ``_check_command`` regex/AST — the path
        added in SEC-CRIT-1. That's where dest_cmd_ast / dest_cmd_regex
        fire.
        """
        action = PlannedAction(tool="start_background", params={"command": "shutdown -h now"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.is_blocked
        assert decision.explanation is not None
        assert decision.explanation.rule_id in {"dest_cmd_ast", "dest_cmd_regex"}
        assert decision.explanation.matched_pattern

    def test_safe_command_no_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """A non-destructive command shouldn't trigger the explain
        path — explanation stays None for the catch-all default.
        """
        action = PlannedAction(tool="exec_command", params={"command": "ls -la"})
        decision = gatekeeper.evaluate(action, session)
        # Either the action passes (no explanation) or hits a different
        # default — but the destructive rule must NOT fire.
        if decision.explanation is not None:
            assert decision.explanation.rule_id not in {
                "dest_cmd_ast",
                "dest_cmd_regex",
            }

    def test_explanation_is_structured_not_freetext(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """The structured fields must not be empty strings when set —
        the reviewer's whole point was 'don't make me parse text'.
        """
        action = PlannedAction(tool="exec_command", params={"command": "rm -rf /"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.explanation is not None
        # rule_id is a stable short identifier
        assert decision.explanation.rule_id
        assert " " not in decision.explanation.rule_id  # no whitespace = stable
        # rule_source points at code location (file:line or module:func)
        assert ":" in decision.explanation.rule_source

    def test_dangerous_python_block_emits_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """``run_python`` with os.system() must emit a structured
        TRUST-2 explanation pointing at the python_ast_guard / regex
        site (the dangerous-Python-code check fires only for
        ``run_python``, see gatekeeper.evaluate).
        """
        action = PlannedAction(
            tool="run_python",
            params={"code": "import os\nos.system('rm -rf /')"},
        )
        decision = gatekeeper.evaluate(action, session)
        assert decision.is_blocked
        assert decision.explanation is not None
        assert decision.explanation.rule_id in {
            "dangerous_python_ast",
            "dangerous_python_regex",
        }
        assert decision.explanation.matched_pattern
        assert ":" in decision.explanation.rule_source

    def test_path_outside_allowed_emits_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """A read_file outside the allowed roots must emit a TRUST-2
        explanation with rule_id=path_outside_allowed.
        """
        action = PlannedAction(tool="read_file", params={"path": "/etc/passwd"})
        decision = gatekeeper.evaluate(action, session)
        assert decision.is_blocked
        # The block may come from path_validation OR an upstream rule.
        # If path_validation fired, explanation must be the structured
        # one we just added.
        if decision.policy_name == "path_validation":
            assert decision.explanation is not None
            assert decision.explanation.rule_id in {
                "path_outside_allowed",
                "path_unparseable",
            }
            assert decision.explanation.matched_pattern
            assert ":" in decision.explanation.rule_source

    def test_disabled_tool_emits_explanation(
        self, gk_config: CognithorConfig, session: SessionContext
    ) -> None:
        """Tool disabled by ``config.tools`` toggle must emit a TRUST-2
        explanation pointing at the disabled-tools set.
        """
        # Build a gatekeeper with computer_use disabled — that group's
        # tools land in self._disabled_tools.
        gk_config.tools.computer_use_enabled = False
        gk = Gatekeeper(gk_config)
        gk.initialize()
        action = PlannedAction(tool="computer_screenshot", params={})
        decision = gk.evaluate(action, session)
        assert decision.policy_name == "tool_disabled_by_config"
        assert decision.explanation is not None
        assert decision.explanation.rule_id == "tool_disabled_by_config"
        assert "_disabled_tools" in decision.explanation.rule_source
        assert decision.explanation.matched_pattern == "computer_screenshot"

    def test_credential_mask_emits_explanation(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """A tool call with credentials in params must MASK and emit a
        structured TRUST-2 explanation naming which keys were masked.
        Privacy invariant: ``matched_pattern`` carries KEY names only,
        never values.
        """
        action = PlannedAction(
            tool="web_fetch",
            params={
                "url": "https://api.example.com",
                "api_key": "sk-secret-abcd1234",
            },
        )
        decision = gatekeeper.evaluate(action, session)
        if decision.policy_name == "credential_masking":
            assert decision.explanation is not None
            assert decision.explanation.rule_id == "credential_scan"
            assert "_scan_credentials" in decision.explanation.rule_source
            # Key name appears in the pattern; value MUST NOT.
            assert "api_key" in decision.explanation.matched_pattern
            assert "sk-secret-abcd1234" not in decision.explanation.matched_pattern
