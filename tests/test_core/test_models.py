"""
Tests für jarvis.models – Das Rückgrat des Systems.

Testet:
  - Erstellung aller Modelle mit Defaults
  - Serialisierung / Deserialisierung (JSON round-trip)
  - Validierung (ungültige Werte werden abgelehnt)
  - Properties (berechnete Felder)
  - Immutability (frozen Modelle sind nicht änderbar)
  - Edge-Cases (leere Listen, Grenzwerte, None-Werte)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.models import (
    ActionPlan,
    AgentResult,
    AuditEntry,
    Chunk,
    CronJob,
    Entity,
    GateDecision,
    GateStatus,
    IncomingMessage,
    MCPServerConfig,
    MCPToolInfo,
    MemorySearchResult,
    MemoryTier,
    Message,
    MessageRole,
    ModelConfig,
    OutgoingMessage,
    PlannedAction,
    PolicyRule,
    ProcedureMetadata,
    Relation,
    RiskLevel,
    SandboxConfig,
    SandboxLevel,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

# ============================================================================
# Enums
# ============================================================================


class TestEnums:
    def test_risk_levels_are_strings(self) -> None:
        assert RiskLevel.GREEN == "green"
        assert RiskLevel.RED == "red"
        assert len(RiskLevel) == 4

    def test_gate_status_values(self) -> None:
        assert GateStatus.ALLOW == "ALLOW"
        assert GateStatus.BLOCK == "BLOCK"
        assert len(GateStatus) == 5

    def test_message_role_values(self) -> None:
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL == "tool"

    def test_memory_tiers(self) -> None:
        assert MemoryTier.CORE == "core"
        assert len(MemoryTier) == 5

    def test_sandbox_levels(self) -> None:
        assert SandboxLevel.PROCESS == "process"
        assert len(SandboxLevel) == 4


# ============================================================================
# Messages
# ============================================================================


class TestMessage:
    def test_create_with_defaults(self) -> None:
        msg = Message(role=MessageRole.USER, content="Hallo")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hallo"
        assert msg.timestamp is not None
        assert msg.name is None
        assert msg.tool_call_id is None

    def test_frozen(self) -> None:
        msg = Message(role=MessageRole.USER, content="Test")
        with pytest.raises(ValidationError):
            msg.content = "Geändert"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        msg = Message(role=MessageRole.ASSISTANT, content="Antwort", name="test_tool")
        data = msg.model_dump_json()
        restored = Message.model_validate_json(data)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.name == msg.name

    def test_tool_message(self) -> None:
        msg = Message(
            role=MessageRole.TOOL,
            content='{"result": "ok"}',
            name="read_file",
            tool_call_id="call_123",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.name == "read_file"


class TestIncomingMessage:
    def test_create_minimal(self) -> None:
        msg = IncomingMessage(channel="cli", user_id="alex", text="Hallo Jarvis")
        assert msg.channel == "cli"
        assert msg.user_id == "alex"
        assert msg.id  # Auto-generiert
        assert len(msg.attachments) == 0

    def test_frozen(self) -> None:
        msg = IncomingMessage(channel="cli", user_id="alex", text="Test")
        with pytest.raises(ValidationError):
            msg.text = "Geändert"  # type: ignore[misc]


class TestOutgoingMessage:
    def test_create_with_metadata(self) -> None:
        msg = OutgoingMessage(
            channel="telegram",
            text="Antwort",
            reply_to="msg_123",
            metadata={"tokens_used": 150},
        )
        assert msg.metadata["tokens_used"] == 150


# ============================================================================
# PGE-Trinität
# ============================================================================


class TestPlannedAction:
    def test_create_minimal(self) -> None:
        action = PlannedAction(tool="read_file")
        assert action.tool == "read_file"
        assert action.params == {}
        assert action.risk_estimate == RiskLevel.GREEN

    def test_create_full(self) -> None:
        action = PlannedAction(
            tool="exec_command",
            params={"command": "ls -la"},
            rationale="Verzeichnis auflisten",
            depends_on=[0, 1],
            risk_estimate=RiskLevel.YELLOW,
            rollback="Kein Rollback nötig",
        )
        assert action.depends_on == [0, 1]
        assert action.risk_estimate == RiskLevel.YELLOW

    def test_frozen(self) -> None:
        action = PlannedAction(tool="test")
        with pytest.raises(ValidationError):
            action.tool = "other"  # type: ignore[misc]


class TestActionPlan:
    def test_direct_response(self) -> None:
        plan = ActionPlan(
            goal="Einfache Frage beantworten",
            direct_response="Das ist die Antwort.",
        )
        assert plan.is_direct_response is True
        assert plan.requires_tools is False

    def test_tool_plan(self) -> None:
        plan = ActionPlan(
            goal="Datei lesen",
            steps=[PlannedAction(tool="read_file", params={"path": "/test.md"})],
            confidence=0.9,
        )
        assert plan.is_direct_response is False
        assert plan.requires_tools is True
        assert plan.confidence == 0.9

    def test_empty_plan(self) -> None:
        plan = ActionPlan(goal="Nichts tun")
        assert plan.is_direct_response is False
        assert plan.requires_tools is False

    def test_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            ActionPlan(goal="Test", confidence=1.5)
        with pytest.raises(ValidationError):
            ActionPlan(goal="Test", confidence=-0.1)

    def test_json_round_trip(self) -> None:
        plan = ActionPlan(
            goal="Test",
            steps=[
                PlannedAction(tool="read_file", params={"path": "/a.md"}),
                PlannedAction(tool="write_file", depends_on=[0]),
            ],
            memory_context=["mem_1", "mem_2"],
            confidence=0.85,
        )
        data = plan.model_dump_json()
        restored = ActionPlan.model_validate_json(data)
        assert len(restored.steps) == 2
        assert restored.steps[1].depends_on == [0]
        assert restored.confidence == 0.85


class TestGateDecision:
    def test_allow(self) -> None:
        d = GateDecision(status=GateStatus.ALLOW, risk_level=RiskLevel.GREEN)
        assert d.is_allowed is True
        assert d.needs_approval is False
        assert d.is_blocked is False

    def test_inform(self) -> None:
        d = GateDecision(status=GateStatus.INFORM, risk_level=RiskLevel.YELLOW)
        assert d.is_allowed is True
        assert d.needs_approval is False

    def test_approve(self) -> None:
        d = GateDecision(
            status=GateStatus.APPROVE,
            risk_level=RiskLevel.ORANGE,
            reason="E-Mail erfordert Bestätigung",
        )
        assert d.is_allowed is False
        assert d.needs_approval is True
        assert d.is_blocked is False

    def test_block(self) -> None:
        d = GateDecision(
            status=GateStatus.BLOCK,
            risk_level=RiskLevel.RED,
            reason="Destruktiver Befehl",
            policy_name="no_destructive_shell",
        )
        assert d.is_allowed is False
        assert d.needs_approval is False
        assert d.is_blocked is True
        assert d.matched_policy == "no_destructive_shell"

    def test_mask(self) -> None:
        d = GateDecision(status=GateStatus.MASK)
        assert d.is_allowed is True  # Ausführen, aber maskiert

    def test_frozen(self) -> None:
        d = GateDecision(status=GateStatus.ALLOW)
        with pytest.raises(ValidationError):
            d.status = GateStatus.BLOCK  # type: ignore[misc]


class TestToolResult:
    def test_success(self) -> None:
        r = ToolResult(tool_name="read_file", content="Dateiinhalt", duration_ms=45)
        assert r.is_error is False
        assert r.duration_ms == 45

    def test_error(self) -> None:
        r = ToolResult(
            tool_name="exec_command",
            is_error=True,
            error_message="Command timed out",
            duration_ms=30000,
        )
        assert r.is_error is True
        assert r.error_message == "Command timed out"


class TestAuditEntry:
    def test_create(self) -> None:
        entry = AuditEntry(
            session_id="sess_123",
            action_tool="read_file",
            action_params_hash="abc123",
            decision_status=GateStatus.ALLOW,
            risk_level=RiskLevel.GREEN,
        )
        assert entry.session_id == "sess_123"
        assert entry.action_tool == "read_file"
        assert entry.user_override is False
        assert entry.id  # Auto-generiert

    def test_frozen(self) -> None:
        entry = AuditEntry(
            session_id="s",
            action_tool="test",
            action_params_hash="h",
            decision_status=GateStatus.ALLOW,
        )
        with pytest.raises(ValidationError):
            entry.user_override = True  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        entry = AuditEntry(
            session_id="sess_abc",
            action_tool="exec_command",
            action_params_hash="deadbeef",
            decision_status=GateStatus.BLOCK,
            decision_reason="Gefährlich",
            risk_level=RiskLevel.RED,
            policy_name="no_destructive_shell",
        )
        data = entry.model_dump_json()
        restored = AuditEntry.model_validate_json(data)
        assert restored.decision_status == GateStatus.BLOCK
        assert restored.action_tool == "exec_command"
        assert restored.policy_name == "no_destructive_shell"


# ============================================================================
# Sandbox & Session
# ============================================================================


class TestSandboxConfig:
    def test_defaults(self) -> None:
        s = SandboxConfig()
        assert s.level == SandboxLevel.PROCESS
        assert s.timeout_seconds == 30
        assert s.network_access is False
        assert len(s.allowed_paths) == 2

    def test_validation(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(timeout_seconds=0)
        with pytest.raises(ValidationError):
            SandboxConfig(max_memory_mb=10)  # Unter Minimum


class TestSessionContext:
    def test_create_and_touch(self) -> None:
        ctx = SessionContext(user_id="alex", channel="cli")
        assert ctx.message_count == 0
        assert ctx.active is True

        old_time = ctx.last_activity
        ctx.touch()
        assert ctx.message_count == 1
        assert ctx.last_activity >= old_time

    def test_multiple_touches(self) -> None:
        ctx = SessionContext()
        for _ in range(5):
            ctx.touch()
        assert ctx.message_count == 5


# ============================================================================
# Memory
# ============================================================================


class TestChunk:
    def test_create(self) -> None:
        c = Chunk(
            text="Jarvis ist ein lokaler KI-Assistent.",
            source_path="memory/knowledge/kunden/soellner.md",
            line_start=1,
            line_end=5,
            memory_tier=MemoryTier.SEMANTIC,
            token_count=12,
        )
        assert c.memory_tier == MemoryTier.SEMANTIC
        assert c.token_count == 12

    def test_frozen(self) -> None:
        c = Chunk(text="Test", source_path="/test.md")
        with pytest.raises(ValidationError):
            c.text = "Geändert"  # type: ignore[misc]


class TestEntity:
    def test_create(self) -> None:
        e = Entity(
            type="person",
            name="Müller, Thomas",
            attributes={"beruf": "Softwareentwickler", "risikoklasse": "1+"},
            source_file="knowledge/kunden/mueller-thomas.md",
        )
        assert e.type == "person"
        assert e.confidence == 1.0


class TestRelation:
    def test_create(self) -> None:
        r = Relation(
            source_entity="entity_1",
            relation_type="hat_police",
            target_entity="entity_2",
            attributes={"seit": "2024-07"},
        )
        assert r.relation_type == "hat_police"


class TestProcedureMetadata:
    def test_empty(self) -> None:
        p = ProcedureMetadata(name="test-procedure")
        assert p.success_rate == 0.0
        assert p.is_reliable is False
        assert p.needs_review is False

    def test_reliable(self) -> None:
        p = ProcedureMetadata(
            name="bu-angebot",
            success_count=9,
            failure_count=1,
            total_uses=10,
        )
        assert p.success_rate == 0.9
        assert p.is_reliable is True

    def test_needs_review(self) -> None:
        p = ProcedureMetadata(
            name="failing-procedure",
            success_count=1,
            failure_count=6,
            total_uses=7,
        )
        assert p.needs_review is True

    def test_not_yet_reliable(self) -> None:
        p = ProcedureMetadata(
            name="new-procedure",
            success_count=5,
            failure_count=0,
            total_uses=5,
        )
        assert p.success_rate == 1.0
        assert p.is_reliable is False  # Braucht 10+ Nutzungen


class TestMemorySearchResult:
    def test_create(self) -> None:
        chunk = Chunk(text="Test", source_path="/test.md")
        result = MemorySearchResult(
            chunk=chunk,
            score=0.85,
            bm25_score=0.7,
            vector_score=0.9,
            recency_factor=0.95,
        )
        assert result.score == 0.85


class TestWorkingMemory:
    def test_empty(self) -> None:
        wm = WorkingMemory()
        assert wm.usage_ratio == 0.0
        assert wm.needs_compaction is False

    def test_needs_compaction(self) -> None:
        wm = WorkingMemory(token_count=27000, max_tokens=32768)
        assert wm.usage_ratio > 0.80
        assert wm.needs_compaction is True

    def test_add_message(self) -> None:
        wm = WorkingMemory()
        msg = Message(role=MessageRole.USER, content="Hallo")
        wm.add_message(msg)
        assert len(wm.chat_history) == 1

    def test_compaction(self) -> None:
        wm = WorkingMemory()
        # System-Message
        wm.add_message(Message(role=MessageRole.SYSTEM, content="Du bist Jarvis"))
        # 10 User/Assistant Messages
        for i in range(10):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            wm.add_message(Message(role=role, content=f"Nachricht {i}"))

        assert len(wm.chat_history) == 11  # 1 System + 10

        removed = wm.clear_for_compaction(keep_last_n=4)

        # System bleibt + letzte 4
        assert len(wm.chat_history) == 5  # 1 System + 4 kept
        assert len(removed) == 6
        # System wurde nicht entfernt
        assert wm.chat_history[0].role == MessageRole.SYSTEM

    def test_compaction_preserves_all_system_messages(self) -> None:
        wm = WorkingMemory()
        wm.add_message(Message(role=MessageRole.SYSTEM, content="Regel 1"))
        wm.add_message(Message(role=MessageRole.SYSTEM, content="Regel 2"))
        for i in range(6):
            wm.add_message(Message(role=MessageRole.USER, content=f"msg {i}"))

        removed = wm.clear_for_compaction(keep_last_n=2)
        system_msgs = [m for m in wm.chat_history if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) == 2
        assert len(removed) == 4

    def test_compaction_too_few_messages(self) -> None:
        wm = WorkingMemory()
        wm.add_message(Message(role=MessageRole.USER, content="A"))
        wm.add_message(Message(role=MessageRole.ASSISTANT, content="B"))
        removed = wm.clear_for_compaction(keep_last_n=4)
        assert len(removed) == 0
        assert len(wm.chat_history) == 2

    def test_zero_max_tokens(self) -> None:
        wm = WorkingMemory(max_tokens=0)
        assert wm.usage_ratio == 1.0
        assert wm.needs_compaction is True


# ============================================================================
# Model-Router, MCP, Policy, Cron
# ============================================================================


class TestModelConfig:
    def test_create(self) -> None:
        m = ModelConfig(name="qwen3:32b", context_window=32768, vram_gb=20.0)
        assert m.speed == "medium"


class TestMCPServerConfig:
    def test_stdio(self) -> None:
        c = MCPServerConfig(command="python", args=["-m", "jarvis.mcp.filesystem"])
        assert c.transport == "stdio"
        assert c.enabled is True

    def test_http(self) -> None:
        c = MCPServerConfig(transport="http", url="http://localhost:3001/mcp")
        assert c.url == "http://localhost:3001/mcp"


class TestMCPToolInfo:
    def test_create(self) -> None:
        t = MCPToolInfo(
            name="read_file",
            server="jarvis-filesystem",
            description="Liest eine Datei",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert t.name == "read_file"


class TestPolicyRule:
    def test_create(self) -> None:
        from jarvis.models import PolicyMatch, PolicyParamMatch

        rule = PolicyRule(
            name="no_destructive_shell",
            match=PolicyMatch(
                tool="exec_command",
                params={"command": PolicyParamMatch(regex="rm -rf")},
            ),
            action=GateStatus.BLOCK,
            reason="Destruktiver Befehl",
            priority=100,
        )
        assert rule.action == GateStatus.BLOCK
        assert rule.priority == 100
        assert rule.match.tool == "exec_command"

    def test_wildcard_match(self) -> None:
        from jarvis.models import PolicyMatch, PolicyParamMatch

        rule = PolicyRule(
            name="credential_mask",
            match=PolicyMatch(
                tool="*",
                params={"*": PolicyParamMatch(contains_pattern="(sk-|token_)")},
            ),
            action=GateStatus.MASK,
        )
        assert rule.match.tool == "*"


class TestCronJob:
    def test_create(self) -> None:
        job = CronJob(
            name="morning_briefing",
            schedule="0 7 * * 1-5",
            prompt="Erstelle mein Briefing",
            channel="telegram",
        )
        assert job.enabled is True


class TestAgentResult:
    def test_success(self) -> None:
        r = AgentResult(response="Hier ist die Antwort.", success=True)
        assert r.success is True
        assert r.error is None

    def test_error(self) -> None:
        r = AgentResult(response="", success=False, error="Ollama nicht erreichbar")
        assert r.success is False


# ============================================================================
# Neue Features – Phase 1 Kompatibilität
# ============================================================================


class TestToolResultSuccess:
    """ToolResult.success Property."""

    def test_success_when_no_error(self) -> None:
        r = ToolResult(tool_name="test", content="OK")
        assert r.success is True
        assert r.is_error is False

    def test_not_success_when_error(self) -> None:
        r = ToolResult(tool_name="test", content="Fehler", is_error=True)
        assert r.success is False

    def test_error_type_field(self) -> None:
        r = ToolResult(tool_name="test", is_error=True, error_type="TimeoutError")
        assert r.error_type == "TimeoutError"


class TestActionPlanHasActions:
    """ActionPlan.has_actions Property."""

    def test_has_actions_with_steps(self) -> None:
        plan = ActionPlan(
            goal="test",
            steps=[PlannedAction(tool="read_file", params={})],
        )
        assert plan.has_actions is True
        assert plan.requires_tools is True

    def test_no_actions_without_steps(self) -> None:
        plan = ActionPlan(goal="test", direct_response="Antwort")
        assert plan.has_actions is False
        assert plan.requires_tools is False


class TestGateDecisionExtended:
    """Erweiterte GateDecision-Felder."""

    def test_original_action(self) -> None:
        action = PlannedAction(tool="test", params={"key": "val"})
        d = GateDecision(
            status=GateStatus.ALLOW,
            original_action=action,
            policy_name="test_policy",
        )
        assert d.original_action is not None
        assert d.original_action.tool == "test"

    def test_masked_params(self) -> None:
        d = GateDecision(
            status=GateStatus.MASK,
            masked_params={"token": "***MASKED***"},
            policy_name="credential_masking",
        )
        assert d.masked_params is not None
        assert "***MASKED***" in d.masked_params["token"]

    def test_policy_name(self) -> None:
        d = GateDecision(status=GateStatus.BLOCK, policy_name="no_destructive_shell")
        assert d.policy_name == "no_destructive_shell"
        assert d.matched_policy == "no_destructive_shell"

    def test_matched_policy_none_when_empty(self) -> None:
        d = GateDecision(status=GateStatus.ALLOW)
        assert d.matched_policy is None  # Empty string → None


class TestSessionContextExtended:
    """Erweiterte SessionContext-Features."""

    def test_max_iterations(self) -> None:
        s = SessionContext(max_iterations=5)
        assert s.max_iterations == 5
        assert s.iteration_count == 0

    def test_iterations_exhausted(self) -> None:
        s = SessionContext(max_iterations=3)
        assert not s.iterations_exhausted
        s.iteration_count = 3
        assert s.iterations_exhausted

    def test_reset_iteration(self) -> None:
        s = SessionContext()
        s.iteration_count = 5
        s.record_block("tool_x")
        s.reset_iteration()
        assert s.iteration_count == 0

    def test_record_block_counting(self) -> None:
        s = SessionContext()
        assert s.record_block("exec_command") == 1
        assert s.record_block("exec_command") == 2
        assert s.record_block("other") == 1
        assert s.record_block("exec_command") == 3


class TestWorkingMemoryExtended:
    """Erweiterte WorkingMemory-Features."""

    def test_core_memory_text(self) -> None:
        wm = WorkingMemory(core_memory_text="Ich bin Jarvis.")
        assert wm.core_memory_text == "Ich bin Jarvis."

    def test_injected_procedures(self) -> None:
        wm = WorkingMemory(injected_procedures=["Schritt 1: ...", "Schritt 2: ..."])
        assert len(wm.injected_procedures) == 2

    def test_clear_for_new_request(self) -> None:
        wm = WorkingMemory(core_memory_text="Kern-Wissen")
        wm.add_message(Message(role=MessageRole.USER, content="Hallo"))
        wm.add_tool_result(ToolResult(tool_name="test", content="OK"))
        wm.injected_procedures = ["proc1"]
        wm.active_plan = ActionPlan(goal="test")

        wm.clear_for_new_request()

        # Chat-History und Core Memory bleiben
        assert len(wm.chat_history) == 1
        assert wm.core_memory_text == "Kern-Wissen"
        # Temporäre Daten gelöscht
        assert len(wm.tool_results) == 0
        assert wm.active_plan is None
        assert len(wm.injected_procedures) == 0
        assert len(wm.injected_memories) == 0


class TestMessageChannel:
    """Message mit optionalem Channel-Feld."""

    def test_message_with_channel(self) -> None:
        m = Message(role=MessageRole.USER, content="Hi", channel="telegram")
        assert m.channel == "telegram"

    def test_message_without_channel(self) -> None:
        m = Message(role=MessageRole.USER, content="Hi")
        assert m.channel is None


class TestOutgoingMessageExtended:
    """OutgoingMessage mit session_id und is_final."""

    def test_session_id(self) -> None:
        m = OutgoingMessage(channel="cli", text="Antwort", session_id="sess-123", is_final=True)
        assert m.session_id == "sess-123"
        assert m.is_final is True

    def test_defaults(self) -> None:
        m = OutgoingMessage(channel="cli", text="Hi")
        assert m.session_id == ""
        assert m.is_final is False
