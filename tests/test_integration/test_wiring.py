"""Integrationstests: Verdrahtung von RuntimeMonitor, AuditLogger, HeartbeatScheduler.

Beweist, dass die Module nicht nur isoliert funktionieren,
sondern auch tatsächlich im Executor/AgentRouter/CronEngine verdrahtet sind.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.audit import AuditCategory, AuditLogger
from jarvis.core.agent_router import AgentProfile, AgentRouter
from jarvis.core.executor import Executor
from jarvis.cron.engine import CronEngine
from jarvis.models import GateDecision, GateStatus, PlannedAction, ToolResult
from jarvis.proactive import (
    ApprovalMode,
    EventType,
    HeartbeatScheduler,
    ProactiveTask,
    TaskStatus,
)
from jarvis.security.monitor import RuntimeMonitor, Verdict


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    return AuditLogger(log_dir=tmp_path / "audit")


@pytest.fixture
def runtime_monitor() -> RuntimeMonitor:
    return RuntimeMonitor(enable_defaults=True)


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    config.jarvis_home = Path("/tmp/jarvis_test")
    return config


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    client = AsyncMock()
    result = MagicMock()
    result.content = "Tool-Ergebnis OK"
    result.is_error = False
    client.call_tool = AsyncMock(return_value=result)
    return client


# ============================================================================
# Executor + RuntimeMonitor Integration
# ============================================================================


class TestExecutorRuntimeMonitorIntegration:
    """Beweist: RuntimeMonitor blockiert Tool-Calls BEVOR sie ausgeführt werden."""

    @pytest.mark.asyncio
    async def test_executor_blocks_system_dir_access(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        """Executor blockiert /etc-Zugriff via RuntimeMonitor."""
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )

        result = await executor._execute_single(
            "file_write", {"path": "/etc/passwd", "content": "hacked"},
        )

        # Blockiert
        assert result.is_error
        assert "SecurityBlock" == result.error_type
        assert "Sicherheitscheck" in result.content

        # MCP-Client wurde NIE aufgerufen (Blockierung VOR Ausführung)
        mock_mcp_client.call_tool.assert_not_called()

        # AuditLogger hat den Block protokolliert
        security_events = audit_logger.query(category=AuditCategory.SECURITY)
        assert len(security_events) >= 1
        assert security_events[0].success is False

    @pytest.mark.asyncio
    async def test_executor_allows_safe_call(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        """Executor erlaubt sichere Tool-Calls und loggt sie."""
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )

        result = await executor._execute_single(
            "memory_search", {"query": "BU-Tarif"},
        )

        assert result.success
        mock_mcp_client.call_tool.assert_called_once()

        # AuditLogger hat den Call protokolliert
        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        assert len(tool_events) >= 1
        assert tool_events[0].tool_name == "memory_search"
        assert tool_events[0].success is True

    @pytest.mark.asyncio
    async def test_executor_logs_agent_name(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        """Executor propagiert Agent-Namen korrekt an AuditLogger."""
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )
        executor.set_agent_context(agent_name="coder")

        await executor._execute_single("exec_command", {"cmd": "ls"})

        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        assert len(tool_events) >= 1
        assert tool_events[0].agent_name == "coder"

        executor.clear_agent_context()

    @pytest.mark.asyncio
    async def test_executor_credential_warning_still_allows(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        """RuntimeMonitor WARNT bei Credentials, blockiert aber nicht."""
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )

        result = await executor._execute_single(
            "send_message",
            {"content": "Mein password ist geheim"},
        )

        # Wird NICHT blockiert (nur Warning)
        assert result.success
        mock_mcp_client.call_tool.assert_called_once()


# ============================================================================
# Executor + AuditLogger: Gatekeeper-Blockierungen
# ============================================================================


class TestExecutorGatekeeperAudit:
    """Beweist: Gatekeeper-Blockierungen werden auditiert."""

    @pytest.mark.asyncio
    async def test_gatekeeper_block_is_audited(
        self, mock_config: Any, mock_mcp_client: Any,
        audit_logger: AuditLogger,
    ) -> None:
        executor = Executor(
            mock_config, mock_mcp_client,
            audit_logger=audit_logger,
        )

        actions = [
            PlannedAction(tool="dangerous_tool", params={"cmd": "rm -rf /"}),
        ]
        decisions = [
            GateDecision(
                action_index=0,
                status=GateStatus.BLOCK,
                reason="Gefährlicher Befehl",
            ),
        ]

        results = await executor.execute(actions, decisions)

        assert len(results) == 1
        assert results[0].is_error
        assert "GatekeeperBlock" == results[0].error_type

        # Audit: Gatekeeper-Block wurde protokolliert
        gate_events = audit_logger.query(category=AuditCategory.GATEKEEPER)
        assert len(gate_events) == 1
        assert gate_events[0].success is False
        assert "BLOCK" in gate_events[0].description


# ============================================================================
# AgentRouter + AuditLogger: Delegation-Logging
# ============================================================================


class TestAgentRouterAuditIntegration:
    """Beweist: AgentRouter loggt Delegationen in den AuditLogger."""

    def test_delegation_is_audited(self, audit_logger: AuditLogger) -> None:
        router = AgentRouter(audit_logger=audit_logger)
        router.initialize()

        # Agenten hinzufügen
        router.add_agent(AgentProfile(
            name="coder", display_name="Coder",
            can_delegate_to=["researcher"],
        ))
        router.add_agent(AgentProfile(
            name="researcher", display_name="Researcher",
        ))

        request = router.create_delegation("coder", "researcher", task="Recherche XYZ")
        assert request is not None

        # Audit: Delegation wurde protokolliert
        delegation_events = audit_logger.query(category=AuditCategory.AGENT_DELEGATION)
        assert len(delegation_events) == 1
        assert "coder" in delegation_events[0].description
        assert "researcher" in delegation_events[0].description

    def test_failed_delegation_not_audited(self, audit_logger: AuditLogger) -> None:
        """Fehlgeschlagene Delegationen (nicht erlaubt) erzeugen kein Audit-Event."""
        router = AgentRouter(audit_logger=audit_logger)
        router.initialize()

        # Agenten hinzufügen, aber KEINE Delegation erlaubt
        router.add_agent(AgentProfile(name="coder", display_name="Coder"))
        router.add_agent(AgentProfile(name="researcher", display_name="Researcher"))

        request = router.create_delegation("coder", "researcher", task="Nicht erlaubt")
        assert request is None

        # Kein Audit-Event (weil Delegation abgelehnt)
        delegation_events = audit_logger.query(category=AuditCategory.AGENT_DELEGATION)
        assert len(delegation_events) == 0


# ============================================================================
# CronEngine + HeartbeatScheduler Integration
# ============================================================================


class TestCronHeartbeatIntegration:
    """Beweist: CronEngine ruft HeartbeatScheduler.tick() auf."""

    @pytest.mark.asyncio
    async def test_heartbeat_triggers_scheduler(self) -> None:
        """CronEngine._execute_heartbeat() ruft HeartbeatScheduler.tick() auf."""
        scheduler = HeartbeatScheduler()
        scheduler.configure(
            EventType.MEMORY_MAINTENANCE,
            enabled=True,
            interval_seconds=0,
        )

        tick_count = 0

        async def maintenance_handler(task: ProactiveTask) -> str:
            nonlocal tick_count
            tick_count += 1
            return "Memory aufgeräumt"

        scheduler.register_handler(EventType.MEMORY_MAINTENANCE, maintenance_handler)

        # CronEngine mit HeartbeatScheduler
        engine = CronEngine(
            heartbeat_config=MagicMock(enabled=True, channel="cli"),
            heartbeat_scheduler=scheduler,
        )
        engine.set_handler(AsyncMock())  # Dummy-Handler für Legacy-Teil

        # Heartbeat ausführen
        await engine._execute_heartbeat()

        # HeartbeatScheduler wurde getriggert
        assert tick_count == 1
        assert scheduler.stats()["total_tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_heartbeat_without_scheduler(self) -> None:
        """CronEngine funktioniert auch ohne HeartbeatScheduler (Legacy)."""
        engine = CronEngine(
            heartbeat_config=MagicMock(enabled=True, channel="cli"),
            heartbeat_scheduler=None,
        )
        handler = AsyncMock()
        engine.set_handler(handler)

        # Soll nicht crashen
        await engine._execute_heartbeat()

        # Legacy-Handler wurde trotzdem aufgerufen
        handler.assert_called_once()


# ============================================================================
# AuditLogger: Credential-Sanitizing in der Kette
# ============================================================================


class TestAuditSanitizingInChain:
    """Beweist: Credentials werden in der gesamten Kette redactiert."""

    @pytest.mark.asyncio
    async def test_credentials_redacted_in_audit(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )

        await executor._execute_single(
            "api_call",
            {"url": "https://example.com", "api_key": "sk-secret-12345"},
        )

        tool_events = audit_logger.query(category=AuditCategory.TOOL_CALL)
        assert len(tool_events) >= 1
        # Credential wurde redaktiert
        assert tool_events[0].parameters.get("api_key") == "***REDACTED***"
        # URL bleibt erhalten
        assert tool_events[0].parameters.get("url") == "https://example.com"


# ============================================================================
# End-to-End: RuntimeMonitor → AuditLogger → Summary
# ============================================================================


class TestEndToEndSecurityAudit:
    """Kompletter Flow: Blockierung → Audit → Zusammenfassung."""

    @pytest.mark.asyncio
    async def test_blocked_call_appears_in_summary(
        self, mock_config: Any, mock_mcp_client: Any,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )
        executor.set_agent_context(agent_name="coder")

        # Erlaubter Call
        await executor._execute_single("memory_search", {"query": "test"})
        # Blockierter Call
        await executor._execute_single("file_write", {"path": "/etc/shadow"})
        # Noch ein erlaubter
        await executor._execute_single("web_search", {"query": "Python"})

        executor.clear_agent_context()

        # Summary prüfen
        summary = audit_logger.summarize(hours=1)
        assert summary.total_entries >= 3
        assert summary.blocked_actions >= 1
        assert summary.by_agent.get("coder", 0) >= 2

    @pytest.mark.asyncio
    async def test_export_includes_blocked_events(
        self, mock_config: Any, mock_mcp_client: Any, tmp_path: Path,
        runtime_monitor: RuntimeMonitor, audit_logger: AuditLogger,
    ) -> None:
        executor = Executor(
            mock_config, mock_mcp_client,
            runtime_monitor=runtime_monitor,
            audit_logger=audit_logger,
        )

        await executor._execute_single("file_read", {"path": "/proc/1/maps"})

        import json

        path = tmp_path / "export.json"
        count = audit_logger.export_json(path, hours=1)
        assert count >= 1

        data = json.loads(path.read_text())
        blocked = [e for e in data["entries"] if not e["success"]]
        assert len(blocked) >= 1


# ============================================================================
# HeartbeatScheduler: Approval-Flow End-to-End
# ============================================================================


class TestHeartbeatApprovalFlow:
    """Testet den kompletten Genehmigungsfluss für proaktive Tasks."""

    @pytest.mark.asyncio
    async def test_ask_mode_waits_then_executes(self) -> None:
        scheduler = HeartbeatScheduler()
        scheduler.configure(
            EventType.EMAIL_TRIAGE,
            enabled=True,
            interval_seconds=99999,  # Hoch → feuert nur einmal (initial)
            approval_mode=ApprovalMode.ASK,
        )

        executed = False

        async def handler(task: ProactiveTask) -> str:
            nonlocal executed
            executed = True
            return "3 neue Mails verarbeitet"

        scheduler.register_handler(EventType.EMAIL_TRIAGE, handler)

        # Tick 1: Task wird erstellt, wartet auf Genehmigung
        processed = await scheduler.tick()
        assert not executed
        awaiting = [t for t in processed if t.status == TaskStatus.AWAITING_APPROVAL]
        assert len(awaiting) == 1

        # Genehmigen
        task_id = awaiting[0].task_id
        scheduler.approve_task(task_id)

        # Tick 2: Genehmigter Task wird ausgeführt
        processed2 = await scheduler.tick()
        assert executed
        completed = [t for t in processed2 if t.status == TaskStatus.COMPLETED]
        assert len(completed) == 1
