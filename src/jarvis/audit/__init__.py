"""Audit Logger: Lückenlose Protokollierung aller Jarvis-Aktionen.

Jede Aktion wird protokolliert:
  - Tool-Calls (Name, Parameter, Ergebnis, Dauer)
  - Datei-Zugriffe (Lesen, Schreiben, Löschen)
  - Netzwerk-Zugriffe (URL, Methode, Statuscode)
  - Agenten-Delegation (Von, An, Aufgabe)
  - Skill-Installationen (Paket, Herkunft, Analyse)
  - Gatekeeper-Entscheidungen (Erlaubt/Blockiert)
  - Memory-Operationen (Indexierung, Suche, Löschung)
  - Security-Events (Blockierungen, Warnungen)

Transparenz:
  - User kann jederzeit das Audit-Log einsehen
  - Zusammenfassungen und Berichte generierbar
  - Export als JSON/CSV für Compliance

DSGVO-Konformität:
  - Personenbezogene Daten werden markiert
  - Löschung nach konfigurierbarer Retention
  - Keine Speicherung von Klartext-Credentials

Bibel-Referenz: §3.5 (Audit & Compliance)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.audit")


# ============================================================================
# Enums
# ============================================================================


class AuditCategory(Enum):
    """Kategorien von Audit-Einträgen."""

    TOOL_CALL = "tool_call"
    FILE_ACCESS = "file_access"
    NETWORK = "network"
    AGENT_DELEGATION = "agent_delegation"
    SKILL_INSTALL = "skill_install"
    GATEKEEPER = "gatekeeper"
    MEMORY_OP = "memory_op"
    SECURITY = "security"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class AuditSeverity(Enum):
    """Schweregrad eines Audit-Eintrags."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Audit Entry
# ============================================================================


@dataclass
class AuditEntry:
    """Ein einzelner Audit-Eintrag.

    Unveränderlich nach Erstellung (Append-Only Log).
    """

    entry_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    category: AuditCategory = AuditCategory.SYSTEM
    severity: AuditSeverity = AuditSeverity.INFO
    action: str = ""  # z.B. "tool_call", "file_write", "gate_block"
    agent_name: str = ""  # Welcher Agent
    tool_name: str = ""  # Welches Tool
    description: str = ""  # Menschenlesbare Beschreibung
    parameters: dict[str, Any] = field(default_factory=dict)
    result: str = ""  # Kurzfassung des Ergebnisses
    success: bool = True
    duration_ms: float = 0.0
    contains_pii: bool = False  # Personenbezogene Daten

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "description": self.description,
            "parameters": self.parameters,
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "contains_pii": self.contains_pii,
        }


# ============================================================================
# Audit Summary
# ============================================================================


@dataclass
class AuditSummary:
    """Zusammenfassung des Audit-Logs für einen Zeitraum."""

    period_start: str
    period_end: str
    total_entries: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_agent: dict[str, int] = field(default_factory=dict)
    tool_usage: dict[str, int] = field(default_factory=dict)
    blocked_actions: int = 0
    warnings: int = 0
    errors: int = 0
    avg_duration_ms: float = 0.0
    pii_entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": f"{self.period_start} → {self.period_end}",
            "total_entries": self.total_entries,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "by_agent": self.by_agent,
            "top_tools": dict(sorted(
                self.tool_usage.items(), key=lambda x: x[1], reverse=True,
            )[:10]),
            "blocked_actions": self.blocked_actions,
            "warnings": self.warnings,
            "errors": self.errors,
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "pii_entries": self.pii_entries,
        }


# ============================================================================
# Audit Logger
# ============================================================================


class AuditLogger:
    """Lückenlose Protokollierung aller Jarvis-Aktionen.

    Usage:
        audit = AuditLogger(log_dir=Path("~/.jarvis/audit"))

        # Tool-Call loggen
        audit.log_tool_call("file_write", {"path": "/tmp/test.txt"}, agent="coder")

        # Gatekeeper-Entscheidung
        audit.log_gatekeeper("BLOCK", "Netzwerkzugriff verweigert", tool="http_fetch")

        # Zusammenfassung
        summary = audit.summarize(hours=24)
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        max_entries: int = 50000,
        retention_days: int = 90,
    ) -> None:
        self._log_dir = log_dir
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._counter = 0
        self._retention_days = retention_days

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging-Methoden ─────────────────────────────────────────

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        *,
        agent_name: str = "",
        result: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> AuditEntry:
        """Protokolliert einen Tool-Call."""
        # Parameter-Sanitizing (keine Credentials loggen)
        safe_params = self._sanitize_params(parameters or {})

        return self._log(
            category=AuditCategory.TOOL_CALL,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            action=f"tool:{tool_name}",
            agent_name=agent_name,
            tool_name=tool_name,
            description=f"Tool '{tool_name}' aufgerufen",
            parameters=safe_params,
            result=result[:500],  # Ergebnis kürzen
            success=success,
            duration_ms=duration_ms,
        )

    def log_file_access(
        self,
        path: str,
        operation: str = "read",
        *,
        agent_name: str = "",
        success: bool = True,
    ) -> AuditEntry:
        """Protokolliert einen Datei-Zugriff."""
        return self._log(
            category=AuditCategory.FILE_ACCESS,
            severity=AuditSeverity.INFO,
            action=f"file:{operation}",
            agent_name=agent_name,
            description=f"Datei {operation}: {path}",
            parameters={"path": path, "operation": operation},
            success=success,
        )

    def log_network(
        self,
        url: str,
        method: str = "GET",
        *,
        agent_name: str = "",
        status_code: int = 0,
        success: bool = True,
    ) -> AuditEntry:
        """Protokolliert einen Netzwerk-Zugriff."""
        return self._log(
            category=AuditCategory.NETWORK,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            action=f"network:{method}",
            agent_name=agent_name,
            description=f"{method} {url}",
            parameters={"url": url, "method": method, "status": status_code},
            success=success,
        )

    def log_agent_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str = "",
    ) -> AuditEntry:
        """Protokolliert eine Agent-zu-Agent Delegation."""
        return self._log(
            category=AuditCategory.AGENT_DELEGATION,
            severity=AuditSeverity.INFO,
            action="delegate",
            agent_name=from_agent,
            description=f"Delegation: {from_agent} → {to_agent}",
            parameters={"from": from_agent, "to": to_agent, "task": task[:200]},
        )

    def log_skill_install(
        self,
        package_id: str,
        *,
        source: str = "",
        success: bool = True,
        analysis_verdict: str = "",
    ) -> AuditEntry:
        """Protokolliert eine Skill-Installation."""
        return self._log(
            category=AuditCategory.SKILL_INSTALL,
            severity=AuditSeverity.WARNING if not success else AuditSeverity.INFO,
            action="skill_install",
            description=f"Skill installiert: {package_id}",
            parameters={
                "package_id": package_id,
                "source": source,
                "analysis": analysis_verdict,
            },
            success=success,
        )

    def log_gatekeeper(
        self,
        decision: str,
        reason: str = "",
        *,
        tool_name: str = "",
        agent_name: str = "",
    ) -> AuditEntry:
        """Protokolliert eine Gatekeeper-Entscheidung."""
        is_block = decision.upper() in ("BLOCK", "DENY")
        return self._log(
            category=AuditCategory.GATEKEEPER,
            severity=AuditSeverity.WARNING if is_block else AuditSeverity.INFO,
            action=f"gate:{decision.lower()}",
            agent_name=agent_name,
            tool_name=tool_name,
            description=f"Gatekeeper: {decision} -- {reason}",
            parameters={"decision": decision, "reason": reason},
            success=not is_block,
        )

    def log_memory_op(
        self,
        operation: str,
        *,
        details: str = "",
        agent_name: str = "",
    ) -> AuditEntry:
        """Protokolliert eine Memory-Operation."""
        return self._log(
            category=AuditCategory.MEMORY_OP,
            severity=AuditSeverity.DEBUG,
            action=f"memory:{operation}",
            agent_name=agent_name,
            description=f"Memory {operation}: {details}",
        )

    def log_security(
        self,
        event_description: str,
        *,
        severity: AuditSeverity = AuditSeverity.WARNING,
        tool_name: str = "",
        agent_name: str = "",
        blocked: bool = False,
    ) -> AuditEntry:
        """Protokolliert ein Security-Event."""
        return self._log(
            category=AuditCategory.SECURITY,
            severity=severity,
            action="security_event",
            tool_name=tool_name,
            agent_name=agent_name,
            description=event_description,
            success=not blocked,
        )

    def log_user_input(
        self,
        channel: str,
        text_preview: str,
        *,
        agent_name: str = "",
    ) -> AuditEntry:
        """Protokolliert eine eingehende Benutzer-Nachricht."""
        return self._log(
            category=AuditCategory.USER_INPUT,
            severity=AuditSeverity.INFO,
            action="user_input",
            agent_name=agent_name,
            description=f"[{channel}] {text_preview[:100]}",
            success=True,
        )

    def log_system(
        self,
        event: str,
        *,
        description: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> AuditEntry:
        """Protokolliert ein System-Event (Start, Stop, Config-Änderung)."""
        return self._log(
            category=AuditCategory.SYSTEM,
            severity=severity,
            action=f"system:{event}",
            description=description or event,
            success=True,
        )

    # ── Abfragen ─────────────────────────────────────────────────

    def query(
        self,
        *,
        category: AuditCategory | None = None,
        severity: AuditSeverity | None = None,
        agent_name: str = "",
        tool_name: str = "",
        success: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Flexible Abfrage des Audit-Logs.

        Alle Filter sind optional und werden kombiniert (AND).
        """
        results: list[AuditEntry] = []

        for entry in reversed(self._entries):
            if category and entry.category != category:
                continue
            if severity and entry.severity != severity:
                continue
            if agent_name and entry.agent_name != agent_name:
                continue
            if tool_name and entry.tool_name != tool_name:
                continue
            if success is not None and entry.success != success:
                continue
            if since:
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    if ts < since:
                        continue
                except (ValueError, TypeError):
                    continue
            if until:
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    if ts > until:
                        continue
                except (ValueError, TypeError):
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def get_blocked_actions(self, limit: int = 50) -> list[AuditEntry]:
        """Alle blockierten Aktionen."""
        return self.query(
            category=AuditCategory.GATEKEEPER,
            success=False,
            limit=limit,
        ) + self.query(
            category=AuditCategory.SECURITY,
            success=False,
            limit=limit,
        )

    # ── Zusammenfassung ──────────────────────────────────────────

    def summarize(self, *, hours: int = 24) -> AuditSummary:
        """Erstellt eine Zusammenfassung des Audit-Logs.

        Args:
            hours: Zeitraum in Stunden (rückwärts ab jetzt).

        Returns:
            AuditSummary mit Statistiken.
        """
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)

        entries = self.query(since=since, limit=50000)

        summary = AuditSummary(
            period_start=since.isoformat(),
            period_end=now.isoformat(),
            total_entries=len(entries),
        )

        cat_counts: dict[str, int] = defaultdict(int)
        sev_counts: dict[str, int] = defaultdict(int)
        agent_counts: dict[str, int] = defaultdict(int)
        tool_counts: dict[str, int] = defaultdict(int)
        total_duration = 0.0
        duration_count = 0

        for entry in entries:
            cat_counts[entry.category.value] += 1
            sev_counts[entry.severity.value] += 1

            if entry.agent_name:
                agent_counts[entry.agent_name] += 1
            if entry.tool_name:
                tool_counts[entry.tool_name] += 1

            if entry.duration_ms > 0:
                total_duration += entry.duration_ms
                duration_count += 1

            if not entry.success and entry.category in (
                AuditCategory.GATEKEEPER, AuditCategory.SECURITY,
            ):
                summary.blocked_actions += 1

            if entry.severity == AuditSeverity.WARNING:
                summary.warnings += 1
            elif entry.severity in (AuditSeverity.ERROR, AuditSeverity.CRITICAL):
                summary.errors += 1

            if entry.contains_pii:
                summary.pii_entries += 1

        summary.by_category = dict(cat_counts)
        summary.by_severity = dict(sev_counts)
        summary.by_agent = dict(agent_counts)
        summary.tool_usage = dict(tool_counts)
        summary.avg_duration_ms = (
            total_duration / duration_count if duration_count > 0 else 0.0
        )

        return summary

    # ── Export ────────────────────────────────────────────────────

    def export_json(self, path: Path, *, hours: int = 24) -> int:
        """Exportiert das Audit-Log als JSON.

        Args:
            path: Ziel-Datei.
            hours: Zeitraum.

        Returns:
            Anzahl exportierter Einträge.
        """
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)
        entries = self.query(since=since, limit=50000)

        data = {
            "export_timestamp": now.isoformat(),
            "period_hours": hours,
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return len(entries)

    def export_csv(self, path: Path, *, hours: int = 24) -> int:
        """Exportiert als CSV für Compliance-Berichte."""
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)
        entries = self.query(since=since, limit=50000)

        lines = [
            "timestamp,category,severity,action,agent,tool,description,success,duration_ms",
        ]
        for e in entries:
            desc = e.description.replace(",", ";").replace("\n", " ")[:100]
            lines.append(
                f"{e.timestamp},{e.category.value},{e.severity.value},"
                f"{e.action},{e.agent_name},{e.tool_name},"
                f'"{desc}",{e.success},{e.duration_ms}'
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return len(entries)

    # ── Retention ────────────────────────────────────────────────

    def cleanup_old_entries(self) -> int:
        """Entfernt Einträge älter als retention_days.

        Returns:
            Anzahl entfernter Einträge.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        before = len(self._entries)

        self._entries = deque(
            (e for e in self._entries
             if self._parse_ts(e.timestamp) > cutoff),
            maxlen=self._entries.maxlen,
        )

        removed = before - len(self._entries)
        if removed:
            logger.info("Audit-Log: %d alte Einträge entfernt (Retention=%dd)",
                        removed, self._retention_days)
        return removed

    def delete_pii_entries(self) -> int:
        """Löscht alle Einträge mit personenbezogenen Daten (DSGVO).

        Returns:
            Anzahl gelöschter Einträge.
        """
        before = len(self._entries)
        self._entries = deque(
            (e for e in self._entries if not e.contains_pii),
            maxlen=self._entries.maxlen,
        )
        return before - len(self._entries)

    # ── Intern ───────────────────────────────────────────────────

    def _log(self, **kwargs: Any) -> AuditEntry:
        """Erstellt und speichert einen Audit-Eintrag."""
        self._counter += 1
        entry = AuditEntry(entry_id=f"audit_{self._counter}", **kwargs)
        self._entries.append(entry)

        # Persistenz (wenn log_dir gesetzt)
        if self._log_dir:
            self._persist_entry(entry)

        return entry

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Schreibt einen Eintrag in die Audit-Datei."""
        try:
            date_str = entry.timestamp[:10]  # YYYY-MM-DD
            log_file = self._log_dir / f"audit_{date_str}.jsonl"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Audit-Persistierung fehlgeschlagen: %s", exc)

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Entfernt Credentials aus Parametern."""
        sensitive_keys = {
            "password", "token", "api_key", "secret",
            "authorization", "credential", "private_key",
        }
        sanitized = {}
        for key, value in params.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:100] + f"...[{len(value)} chars]"
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return datetime.min.replace(tzinfo=timezone.utc)

    # ── Stats ────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "retention_days": self._retention_days,
            "has_persistence": self._log_dir is not None,
        }


# ============================================================================
# Compliance-Framework Re-Exports
# ============================================================================

from jarvis.audit.compliance import (  # noqa: E402
    ComplianceFramework,
    DecisionLog,
    RemediationTracker,
    ReportExporter,
)
from jarvis.audit.ethics import (  # noqa: E402
    BiasDetector,
    BudgetManager,
    CostTracker,
    EconomicGovernor,
    EthicsPolicy,
    FairnessAuditor,
)
from jarvis.audit.ai_act_export import (  # noqa: E402
    ComplianceExporter,
    RiskClassifier as ExportRiskClassifier,
    TransparencyChecker,
)
from jarvis.audit.eu_ai_act import (  # noqa: E402
    ComplianceDocManager,
    EUAIActGovernor,
    RiskClassifier,
    TrainingCatalog,
    TransparencyRegister,
)
# Alias damit beide RiskClassifier erreichbar sind
AIActExportRiskClassifier = ExportRiskClassifier
from jarvis.audit.impact_assessment import (  # noqa: E402
    EthicsBoard,
    ImpactAssessor,
    MitigationTracker,
    StakeholderRegistry,
)

__all__ = [
    "AuditCategory",
    "AuditEntry",
    "AuditLogger",
    "AuditSeverity",
    "AuditSummary",
    "BiasDetector",
    "BudgetManager",
    "ComplianceFramework",
    "CostTracker",
    "DecisionLog",
    "EconomicGovernor",
    "EthicsPolicy",
    "FairnessAuditor",
    "RemediationTracker",
    "ReportExporter",
]
