"""Coverage-Tests Runde 2: Gatekeeper, Audit, MCP-Connections.

Zielt auf die verbleibenden Lücken in:
- core/gatekeeper.py (73% → 85%+): Param-Matching, Command-Check, Credential-Scan
- security/audit.py (79% → 90%+): Query mit Filtern, log_event, Chain-Verify
- mcp/client.py (76% → 85%+): Server-Connection Mocking
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig
from jarvis.models import (
    AuditEntry,
    GateStatus,
    PlannedAction,
    PolicyParamMatch,
    RiskLevel,
    SessionContext,
)

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# 1. Gatekeeper – _param_matches Vollabdeckung
# ============================================================================


class TestGatekeeperParamMatching:
    """Testet alle 6 Operatoren in _param_matches und Wildcard-Matching."""

    @pytest.fixture()
    def gatekeeper(self, tmp_path: Path):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()
        gk = Gatekeeper(config)
        gk.initialize()
        return gk

    # --- _param_matches: regex ---

    def test_param_regex_match(self, gatekeeper):
        """Regex-Match findet Pattern im Wert."""
        match = PolicyParamMatch(regex=r"^/etc/.*")
        assert gatekeeper._param_matches("/etc/passwd", match) is True

    def test_param_regex_no_match(self, gatekeeper):
        """Regex-Match findet kein Pattern."""
        match = PolicyParamMatch(regex=r"^/etc/.*")
        assert gatekeeper._param_matches("/home/user", match) is False

    def test_param_regex_invalid(self, gatekeeper):
        """Ungültiger Regex gibt False zurück."""
        match = PolicyParamMatch(regex=r"[invalid")
        assert gatekeeper._param_matches("test", match) is False

    # --- _param_matches: startswith ---

    def test_param_startswith_string(self, gatekeeper):
        """startswith mit String."""
        match = PolicyParamMatch(startswith="/etc")
        assert gatekeeper._param_matches("/etc/config", match) is True
        assert gatekeeper._param_matches("/home/user", match) is False

    def test_param_startswith_list(self, gatekeeper):
        """startswith mit Liste (OR)."""
        match = PolicyParamMatch(startswith=["/etc", "/var"])
        assert gatekeeper._param_matches("/etc/config", match) is True
        assert gatekeeper._param_matches("/var/log", match) is True
        assert gatekeeper._param_matches("/home/user", match) is False

    # --- _param_matches: not_startswith ---

    def test_param_not_startswith_blocks(self, gatekeeper):
        """not_startswith blockiert Werte die damit beginnen."""
        match = PolicyParamMatch(not_startswith="/tmp")
        assert gatekeeper._param_matches("/tmp/file", match) is False

    def test_param_not_startswith_allows(self, gatekeeper):
        """not_startswith erlaubt andere Werte."""
        match = PolicyParamMatch(not_startswith="/tmp")
        assert gatekeeper._param_matches("/home/file", match) is True

    def test_param_not_startswith_list(self, gatekeeper):
        """not_startswith mit Liste – blockiert nur wenn ALLE matchen."""
        match = PolicyParamMatch(not_startswith=["/tmp", "/var"])
        # Blockiert nur wenn ALLE Prefixes matchen (alle startswith)
        assert gatekeeper._param_matches("/home/file", match) is True

    # --- _param_matches: contains ---

    def test_param_contains_string(self, gatekeeper):
        """contains findet Substring."""
        match = PolicyParamMatch(contains="password")
        assert gatekeeper._param_matches("my_password_file", match) is True
        assert gatekeeper._param_matches("config.yaml", match) is False

    def test_param_contains_list(self, gatekeeper):
        """contains mit Liste (OR)."""
        match = PolicyParamMatch(contains=["secret", "key"])
        assert gatekeeper._param_matches("api_key", match) is True
        assert gatekeeper._param_matches("secret_data", match) is True
        assert gatekeeper._param_matches("public_info", match) is False

    # --- _param_matches: contains_pattern ---

    def test_param_contains_pattern_match(self, gatekeeper):
        """contains_pattern findet Regex im Wert."""
        match = PolicyParamMatch(contains_pattern=r"\d{4}-\d{2}-\d{2}")
        assert gatekeeper._param_matches("file_2024-01-15.log", match) is True
        assert gatekeeper._param_matches("no_date_here", match) is False

    def test_param_contains_pattern_invalid(self, gatekeeper):
        """Ungültiges contains_pattern gibt False."""
        match = PolicyParamMatch(contains_pattern=r"[broken")
        assert gatekeeper._param_matches("test", match) is False

    # --- _param_matches: equals ---

    def test_param_equals_match(self, gatekeeper):
        """equals prüft exakte Gleichheit."""
        match = PolicyParamMatch(equals="exact_value")
        assert gatekeeper._param_matches("exact_value", match) is True
        assert gatekeeper._param_matches("other_value", match) is False

    # --- _param_matches: Kombinationen ---

    def test_param_combined_and(self, gatekeeper):
        """Mehrere Operatoren sind AND-verknüpft."""
        match = PolicyParamMatch(startswith="/etc", contains="conf")
        assert gatekeeper._param_matches("/etc/config", match) is True
        assert gatekeeper._param_matches("/etc/passwd", match) is False
        assert gatekeeper._param_matches("/var/config", match) is False

    def test_param_no_conditions(self, gatekeeper):
        """Ohne Bedingungen matcht alles."""
        match = PolicyParamMatch()
        assert gatekeeper._param_matches("anything", match) is True

    # --- _any_param_matches (Wildcard *) ---

    def test_any_param_matches_found(self, gatekeeper):
        """Wildcard * findet Match in irgendeinem Parameter."""
        match = PolicyParamMatch(contains="secret")
        params = {"path": "/home/user", "content": "my_secret_data"}
        assert gatekeeper._any_param_matches(params, match) is True

    def test_any_param_matches_not_found(self, gatekeeper):
        """Wildcard * findet nichts."""
        match = PolicyParamMatch(contains="nuclear")
        params = {"path": "/home/user", "content": "normal data"}
        assert gatekeeper._any_param_matches(params, match) is False


class TestGatekeeperCommandCheck:
    """Testet Shell-Befehls-Prüfung gegen destruktive Patterns."""

    @pytest.fixture()
    def gatekeeper(self, tmp_path: Path):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()
        gk = Gatekeeper(config)
        gk.initialize()
        return gk

    def test_destructive_command_blocked(self, gatekeeper):
        """rm -rf wird blockiert."""
        action = PlannedAction(tool="shell", params={"command": "rm -rf /"}, rationale="test")
        result = gatekeeper._check_command("rm -rf /", action)
        assert result is not None
        assert result.status == GateStatus.BLOCK
        assert result.risk_level == RiskLevel.RED

    def test_safe_command_allowed(self, gatekeeper):
        """Harmloser Befehl wird durchgelassen."""
        action = PlannedAction(tool="shell", params={"command": "echo hello"}, rationale="test")
        result = gatekeeper._check_command("echo hello", action)
        assert result is None

    def test_empty_command(self, gatekeeper):
        """Leerer Befehl ist OK."""
        action = PlannedAction(tool="shell", params={}, rationale="test")
        result = gatekeeper._check_command("", action)
        assert result is None

    def test_whitespace_only_command(self, gatekeeper):
        """Nur Whitespace ist OK."""
        action = PlannedAction(tool="shell", params={}, rationale="test")
        result = gatekeeper._check_command("   ", action)
        assert result is None


class TestGatekeeperCredentialScan:
    """Testet Credential-Scanning und Maskierung."""

    @pytest.fixture()
    def gatekeeper(self, tmp_path: Path):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()
        gk = Gatekeeper(config)
        gk.initialize()
        return gk

    def test_no_credentials(self, gatekeeper):
        """Normale Parameter haben keine Credentials."""
        params = {"path": "/home/user", "text": "Hello World"}
        masked, found = gatekeeper._scan_credentials(params)
        assert found is False
        assert masked == params

    def test_empty_params(self, gatekeeper):
        """Leere Parameter sind ok."""
        _masked, found = gatekeeper._scan_credentials({})
        assert found is False


class TestGatekeeperPolicyLoading:
    """Testet Policy-Loading aus YAML."""

    @pytest.fixture()
    def gatekeeper_with_policy(self, tmp_path: Path):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()

        # Custom Policy YAML erstellen
        policy_dir = config.policies_dir
        policy_dir.mkdir(parents=True, exist_ok=True)
        custom_policy = policy_dir / "custom.yaml"
        custom_policy.write_text(
            """
rules:
  - name: block_dangerous_tool
    match:
      tool: dangerous_tool
    action: BLOCK
    reason: "Tool ist verboten"
    priority: 100
  - name: allow_read
    match:
      tool: read_file
      params:
        path:
          startswith: "/home"
    action: ALLOW
    reason: "Home-Verzeichnis ist erlaubt"
    priority: 50
""",
            encoding="utf-8",
        )

        gk = Gatekeeper(config)
        gk.initialize()
        return gk

    def test_custom_policy_loaded(self, gatekeeper_with_policy):
        """Custom Policies werden geladen."""
        assert len(gatekeeper_with_policy._policies) >= 2

    def test_policy_blocks_dangerous_tool(self, gatekeeper_with_policy):
        """Custom Policy blockiert gefährliches Tool."""
        action = PlannedAction(tool="dangerous_tool", params={}, rationale="test")
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decision = gatekeeper_with_policy.evaluate(action, ctx)
        assert decision.status == GateStatus.BLOCK
        assert decision.policy_name == "block_dangerous_tool"

    def test_policy_allows_read(self, gatekeeper_with_policy):
        """Custom Policy erlaubt Lesen im Home."""
        action = PlannedAction(
            tool="read_file",
            params={"path": "/home/alex/doc.md"},
            rationale="test",
        )
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decision = gatekeeper_with_policy.evaluate(action, ctx)
        assert decision.status == GateStatus.ALLOW

    def test_invalid_policy_yaml(self, tmp_path: Path):
        """Ungültige YAML wird übersprungen."""
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()

        policy_dir = config.policies_dir
        policy_dir.mkdir(parents=True, exist_ok=True)
        bad_policy = policy_dir / "custom.yaml"
        bad_policy.write_text("not: a: valid: yaml: list:", encoding="utf-8")

        gk = Gatekeeper(config)
        gk.initialize()  # Sollte nicht crashen


class TestGatekeeperEvaluate:
    """Testet den kompletten evaluate()-Flow."""

    @pytest.fixture()
    def gatekeeper(self, tmp_path: Path):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(jarvis_home=tmp_path / ".jarvis")
        config.ensure_directories()
        gk = Gatekeeper(config)
        gk.initialize()
        return gk

    def test_default_classification(self, gatekeeper):
        """Unknown Tool bekommt Default-Risiko."""
        action = PlannedAction(
            tool="some_unknown_tool",
            params={"data": "test"},
            rationale="test",
        )
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decision = gatekeeper.evaluate(action, ctx)
        # Sollte ALLOW oder ORANGE sein, je nach Default
        assert decision.status in (GateStatus.ALLOW, GateStatus.APPROVE)

    def test_shell_command_check_in_flow(self, gatekeeper):
        """Shell-Commands werden im evaluate()-Flow geprüft."""
        action = PlannedAction(
            tool="exec_command",
            params={"command": "rm -rf /"},
            rationale="Alles löschen",
        )
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decision = gatekeeper.evaluate(action, ctx)
        assert decision.status == GateStatus.BLOCK

    def test_path_validation_blocks(self, gatekeeper):
        """Pfad außerhalb erlaubter Verzeichnisse wird blockiert."""
        action = PlannedAction(
            tool="read_file",
            params={"path": "/etc/shadow"},
            rationale="test",
        )
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decision = gatekeeper.evaluate(action, ctx)
        assert decision.status == GateStatus.BLOCK

    def test_evaluate_plan(self, gatekeeper):
        """evaluate_plan evaluiert mehrere Aktionen."""
        actions = [
            PlannedAction(tool="read_file", params={"path": "/tmp/test"}, rationale="test"),
            PlannedAction(tool="exec_command", params={"command": "echo hi"}, rationale="test"),
        ]
        ctx = SessionContext(session_id="test", channel="cli", user_id="alex")
        decisions = gatekeeper.evaluate_plan(actions, ctx)
        assert len(decisions) == 2


# ============================================================================
# 2. Audit – Query & Events
# ============================================================================


class TestAuditQuery:
    """Testet Audit-Log Query mit verschiedenen Filtern."""

    @pytest.fixture()
    def audit(self, tmp_path: Path):
        from jarvis.security.audit import AuditTrail

        log_dir = tmp_path / "audit_logs"
        return AuditTrail(log_dir)

    @pytest.fixture()
    def populated_audit(self, audit):
        """Audit mit gemischten Einträgen (Tool-Calls + Events)."""
        # Tool-Call Entries
        for i in range(5):
            entry = AuditEntry(
                session_id=f"session-{i % 2}",
                action_tool=f"tool_{i % 3}",
                action_params_hash=f"hash-{i}",
                decision_status=GateStatus.ALLOW if i % 2 == 0 else GateStatus.BLOCK,
                decision_reason=f"Reason {i}",
                risk_level=RiskLevel.GREEN if i % 2 == 0 else RiskLevel.RED,
            )
            audit.record(entry)

        # Event Entries
        audit.record_event(
            session_id="session-0",
            event_type="agent_spawn",
            details={"agent": "planner"},
        )
        audit.record_event(
            session_id="session-1",
            event_type="auth_success",
            details={"user": "alex"},
        )

        return audit

    def test_query_all(self, populated_audit):
        """Query ohne Filter gibt alle Einträge."""
        results = populated_audit.query()
        assert len(results) == 7  # 5 tool-calls + 2 events

    def test_query_by_session(self, populated_audit):
        """Query mit session_id Filter."""
        results = populated_audit.query(session_id="session-0")
        assert len(results) >= 3  # 3 tool-calls + 1 event

    def test_query_by_tool(self, populated_audit):
        """Query mit tool Filter."""
        results = populated_audit.query(tool="tool_0")
        assert all(r.get("action_tool") == "tool_0" for r in results)

    def test_query_by_status(self, populated_audit):
        """Query mit status Filter."""
        results = populated_audit.query(status=GateStatus.BLOCK)
        assert all(r.get("decision_status") == "BLOCK" for r in results)

    def test_query_with_limit(self, populated_audit):
        """Query mit Limit."""
        results = populated_audit.query(limit=2)
        assert len(results) == 2

    def test_query_by_since(self, populated_audit):
        """Query mit since-Filter."""
        past = datetime.now(UTC) - timedelta(hours=1)
        results = populated_audit.query(since=past)
        assert len(results) == 7  # Alle sind recent

        future = datetime.now(UTC) + timedelta(hours=1)
        results = populated_audit.query(since=future)
        assert len(results) == 0  # Keine in der Zukunft

    def test_query_combined_filters(self, populated_audit):
        """Query mit kombinierten Filtern."""
        results = populated_audit.query(
            session_id="session-0",
            status=GateStatus.ALLOW,
        )
        # session-0 hat entries bei i=0,2,4 → status ALLOW bei i=0,2,4
        assert all(r.get("decision_status") == "ALLOW" for r in results)

    def test_query_empty_log(self, audit):
        """Query auf leerem Log gibt leere Liste."""
        results = audit.query()
        assert results == []


class TestAuditLogEvent:
    """Testet das Event-Logging."""

    @pytest.fixture()
    def audit(self, tmp_path: Path):
        from jarvis.security.audit import AuditTrail

        log_dir = tmp_path / "audit_logs"
        return AuditTrail(log_dir)

    def test_log_event_returns_hash(self, audit):
        """log_event gibt einen Hash zurück."""
        h = audit.record_event(
            session_id="test-session",
            event_type="test_event",
            details={"key": "value"},
        )
        assert isinstance(h, str)
        assert len(h) > 0

    def test_log_event_queryable(self, audit):
        """Events sind per Query abrufbar."""
        audit.record_event("s1", "login", {"user": "alex"})
        audit.record_event("s1", "logout", {"user": "alex"})

        results = audit.query(session_id="s1")
        assert len(results) == 2
        event_types = {r.get("event_type") for r in results}
        assert "login" in event_types
        assert "logout" in event_types

    def test_log_event_excluded_by_tool_filter(self, audit):
        """Events werden bei tool-Filter ausgefiltert."""
        audit.record_event("s1", "spawn", {})
        entry = AuditEntry(
            session_id="s1",
            action_tool="read_file",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
        )
        audit.record(entry)

        results = audit.query(tool="read_file")
        assert len(results) == 1
        assert results[0].get("action_tool") == "read_file"


class TestAuditChainVerify:
    """Testet die Hash-Chain-Verifizierung."""

    @pytest.fixture()
    def audit(self, tmp_path: Path):
        from jarvis.security.audit import AuditTrail

        log_dir = tmp_path / "audit_logs"
        return AuditTrail(log_dir)

    def test_verify_empty_chain(self, audit):
        """Leere Chain ist valide."""
        valid, total, _broken = audit.verify_chain()
        assert valid is True
        assert total == 0

    def test_verify_valid_chain(self, audit):
        """Korrekte Chain verifiziert erfolgreich."""
        for i in range(5):
            entry = AuditEntry(
                session_id=f"s-{i}",
                action_tool=f"tool_{i}",
                action_params_hash=f"h-{i}",
                decision_status=GateStatus.ALLOW,
            )
            audit.record(entry)

        valid, total, broken = audit.verify_chain()
        assert valid is True
        assert total == 5
        assert broken == -1

    def test_verify_tampered_chain(self, audit):
        """Manipulierte Chain wird erkannt."""
        for i in range(3):
            entry = AuditEntry(
                session_id="s1",
                action_tool=f"tool_{i}",
                action_params_hash=f"h-{i}",
                decision_status=GateStatus.ALLOW,
            )
            audit.record(entry)

        # Manipuliere eine Zeile
        lines = audit._log_path.read_text().strip().split("\n")
        tampered = json.loads(lines[1])
        tampered["decision_reason"] = "TAMPERED"
        lines[1] = json.dumps(tampered)
        audit._log_path.write_text("\n".join(lines) + "\n")

        valid, _total, broken = audit.verify_chain()
        assert valid is False
        assert broken >= 0

    def test_record_and_verify_with_events(self, audit):
        """Gemischte Entries (Record + Event) verifizieren korrekt."""
        entry = AuditEntry(
            session_id="s1",
            action_tool="tool_1",
            action_params_hash="h1",
            decision_status=GateStatus.ALLOW,
        )
        audit.record(entry)
        audit.record_event("s1", "spawn", {"detail": "test"})
        audit.record(entry)

        valid, total, _broken = audit.verify_chain()
        assert valid is True
        assert total == 3


class TestAuditRecord:
    """Testet das Audit-Recording."""

    @pytest.fixture()
    def audit(self, tmp_path: Path):
        from jarvis.security.audit import AuditTrail

        log_dir = tmp_path / "audit_logs"
        return AuditTrail(log_dir)

    def test_record_returns_hash(self, audit):
        """record() gibt einen Hash-String zurück."""
        entry = AuditEntry(
            session_id="s1",
            action_tool="read_file",
            action_params_hash="abc123",
            decision_status=GateStatus.ALLOW,
        )
        h = audit.record(entry)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_record_increments_count(self, audit):
        """Jeder Record erhöht den Counter."""
        assert audit._entry_count == 0
        for i in range(3):
            entry = AuditEntry(
                session_id="s1",
                action_tool="tool",
                action_params_hash=f"h{i}",
                decision_status=GateStatus.ALLOW,
            )
            audit.record(entry)
        assert audit._entry_count == 3

    def test_record_creates_file(self, audit):
        """Erster Record erstellt die Log-Datei."""
        assert not audit._log_path.exists()
        entry = AuditEntry(
            session_id="s1",
            action_tool="tool",
            action_params_hash="h",
            decision_status=GateStatus.ALLOW,
        )
        audit.record(entry)
        assert audit._log_path.exists()

    def test_record_hash_chain_continuity(self, audit):
        """Hash-Chain ist kontinuierlich."""
        hashes = []
        for i in range(3):
            entry = AuditEntry(
                session_id="s1",
                action_tool="tool",
                action_params_hash=f"h{i}",
                decision_status=GateStatus.ALLOW,
            )
            h = audit.record(entry)
            hashes.append(h)

        # Alle Hashes sind unterschiedlich
        assert len(set(hashes)) == 3

        # Chain verifiziert
        valid, total, _ = audit.verify_chain()
        assert valid is True
        assert total == 3


# ============================================================================
# 3. MCP Client – Connection Management
# ============================================================================


class TestMCPClientConnections:
    """Testet MCP-Server Connection-Management."""

    @pytest.fixture()
    def mcp(self, tmp_path: Path):
        from jarvis.mcp.client import JarvisMCPClient

        config = JarvisConfig(jarvis_home=tmp_path)
        return JarvisMCPClient(config)

    @pytest.mark.asyncio()
    async def test_call_tool_server_not_connected(self, mcp):
        """Tool auf nicht-verbundenem Server gibt Fehler."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig, MCPToolInfo

        # Server registriert aber nicht verbunden
        server_config = MCPServerConfig(command="echo", args=["test"])
        mcp._servers["test-server"] = ServerConnection(
            name="test-server",
            config=server_config,
            connected=False,
        )
        mcp._tool_registry["remote_tool"] = MCPToolInfo(
            name="remote_tool",
            server="test-server",
            description="Test",
            input_schema={},
        )

        result = await mcp.call_tool("remote_tool", {})
        assert result.is_error is True
        assert "nicht verbunden" in result.content

    @pytest.mark.asyncio()
    async def test_multiple_builtin_handlers(self, mcp):
        """Mehrere Builtin-Handler koexistieren."""
        mcp.register_builtin_handler("tool_a", AsyncMock(return_value="A"), "A")
        mcp.register_builtin_handler("tool_b", AsyncMock(return_value="B"), "B")
        mcp.register_builtin_handler("tool_c", AsyncMock(return_value="C"), "C")

        result_a = await mcp.call_tool("tool_a", {})
        result_b = await mcp.call_tool("tool_b", {})
        result_c = await mcp.call_tool("tool_c", {})

        assert "A" in result_a.content
        assert "B" in result_b.content
        assert "C" in result_c.content

    @pytest.mark.asyncio()
    async def test_get_tool_schemas_mixed(self, mcp):
        """Tool-Schemas enthält Builtin + registrierte MCP-Tools."""
        from jarvis.models import MCPToolInfo

        mcp.register_builtin_handler(
            "builtin_1",
            AsyncMock(),
            "Builtin",
            input_schema={"type": "object"},
        )
        mcp._tool_registry["remote_1"] = MCPToolInfo(
            name="remote_1",
            server="srv",
            description="Remote",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        schemas = mcp.get_tool_schemas()
        assert "builtin_1" in schemas
        assert "remote_1" in schemas

    @pytest.mark.asyncio()
    async def test_disconnect_with_active_servers(self, mcp):
        """disconnect_all schließt aktive Server-Verbindungen."""
        from jarvis.mcp.client import ServerConnection
        from jarvis.models import MCPServerConfig

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        server = ServerConnection(
            name="test",
            config=MCPServerConfig(command="test"),
            connected=True,
            process=mock_process,
        )
        mcp._servers["test"] = server

        await mcp.disconnect_all()
        assert server.connected is False
