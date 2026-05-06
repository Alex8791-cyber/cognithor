"""Audit logger: Complete logging of all Cognithor actions.

Every action is logged:
  - Tool calls (name, parameters, result, duration)
  - File accesses (read, write, delete)
  - Network accesses (URL, method, status code)
  - Agent delegation (from, to, task)
  - Skill installations (package, source, analysis)
  - Gatekeeper decisions (allowed/blocked)
  - Memory operations (indexing, search, deletion)
  - Security events (blocks, warnings)

Transparency:
  - User can inspect the audit log at any time
  - Summaries and reports can be generated
  - Export as JSON/CSV for compliance

GDPR compliance:
  - Personal data is marked
  - Deletion after configurable retention
  - No storage of plaintext credentials

Bible reference: §3.5 (Audit & Compliance)
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from cognithor.models import FailureMode

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("cognithor.audit")


# ============================================================================
# Enums
# ============================================================================


class AuditCategory(Enum):
    """Categories of audit entries."""

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
    REFLECTION = "reflection"


class AuditSeverity(Enum):
    """Severity of an audit entry."""

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
    """A single audit entry.

    Immutable after creation (append-only log).

    Hash chain (SEC-HIGH-5, autonomous security audit, 2026-05-04):
    every persisted entry stores ``prev_hash`` = SHA-256 of the
    previous entry's canonical JSON form. ``AuditLogger.validate_chain()``
    walks the JSONL on disk and verifies each link, surfacing any
    tampering (insertion, deletion, mutation). The first entry of a
    chain stores the empty string.
    """

    entry_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    category: AuditCategory = AuditCategory.SYSTEM
    severity: AuditSeverity = AuditSeverity.INFO
    action: str = ""  # e.g. "tool_call", "file_write", "gate_block"
    agent_name: str = ""  # Which agent
    tool_name: str = ""  # Which tool
    description: str = ""  # Human-readable description
    parameters: dict[str, Any] = field(default_factory=dict)
    result: str = ""  # Brief summary of the result
    success: bool = True
    duration_ms: float = 0.0
    contains_pii: bool = False  # Contains personal data
    # TRUST-1 (operational-trust audit, 2026-05-04): correlation key
    # tying every entry from a single Plan→Gate→Execute run together.
    # Empty for entries logged outside a run scope (boot-time, scheduler,
    # GC). ``AuditLogger.run_receipt(session_id)`` aggregates by this.
    session_id: str = ""
    # TRUST-3: structured failure-mode classification (operational-trust
    # audit, 2026-05-04). ``None`` for successful entries; a
    # ``FailureMode`` enum value otherwise. Aggregated by
    # ``AuditLogger.failures_by_mode``.
    failure_mode: FailureMode | None = None
    # SEC-HIGH-5: SHA-256 of the previous entry's canonical JSON.
    # Empty for the first entry written to a freshly-rotated log file.
    prev_hash: str = ""

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
            "session_id": self.session_id,
            "failure_mode": self.failure_mode.value if self.failure_mode else None,
            "prev_hash": self.prev_hash,
        }

    def canonical_hash(self) -> str:
        """SHA-256 of this entry's canonical JSON form.

        Used by the next entry as its ``prev_hash``. ``sort_keys=True``
        + ``ensure_ascii=False`` make the form deterministic across
        runs so the verification on disk matches what was hashed at
        write time.
        """
        import hashlib

        canon = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ============================================================================
# Audit Summary
# ============================================================================


@dataclass
class AuditSummary:
    """Summary of the audit log for a time period."""

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
            "top_tools": dict(
                sorted(
                    self.tool_usage.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
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
    """Complete logging of all Cognithor actions.

    Usage:
        audit = AuditLogger(log_dir=Path("~/.cognithor/audit"))

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
        # SEC-HIGH-5: tracks SHA-256 of the most recently persisted
        # entry per log file (key = file path) so the next entry can
        # link to it via ``prev_hash``. Loaded lazily from disk on
        # first persist to a given file (handles process restarts).
        self._last_hash_per_file: dict[Path, str] = {}
        # Audit-PR2: serialise read-prev-hash + write + cache update
        # so concurrent appenders (multi-channel async handlers) cannot
        # link two entries to the same prev_hash and break the chain.
        # Mirrors HashlineAuditor's threading.Lock pattern.
        self._persist_lock = threading.Lock()

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        # TRUST-10 backfill: record the audit-log schema move from
        # v0 (flat JSONL, no prev_hash) to v1 (hash-chained JSONL,
        # SEC-HIGH-5) into the canonical MIGRATION_LEDGER. The hook
        # is idempotent (the ledger rejects duplicate migration_id);
        # safe to call on every AuditLogger construction.
        self._record_audit_schema_migration()

    @staticmethod
    def _record_audit_schema_migration() -> None:
        """Record the SEC-HIGH-5 audit-log schema migration.

        Idempotent: the canonical MIGRATION_LEDGER rejects duplicate
        migration_id with a chain error, which we silently swallow —
        re-recording the same logical migration after a process
        restart is a no-op, not a failure.

        Best-effort: any exception is logged + swallowed. AuditLogger
        construction MUST NEVER fail because of TRUST-10 backfill.
        """
        from contextlib import suppress

        from cognithor.security.migration_ledger import (
            MIGRATION_LEDGER,
            MigrationChainError,
            MigrationDomain,
            MigrationStatus,
            MigrationStep,
        )

        with suppress(MigrationChainError, ValueError):
            MIGRATION_LEDGER.record(
                MigrationStep(
                    domain=MigrationDomain.AUDIT_LOG,
                    source_version="v0-flat-jsonl",
                    target_version="v1-hashchain-jsonl",
                    status=MigrationStatus.APPLIED,
                    applied_by="system",
                    item_count=-1,  # schema-only, no row migration
                    migration_id="audit_log:v0-flat-jsonl:v1-hashchain-jsonl",
                    notes="SEC-HIGH-5: prev_hash chain on every audit entry",
                )
            )

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
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a tool call."""
        # Parameter sanitizing (do not log credentials)
        safe_params = self._sanitize_params(parameters or {})

        return self._log(
            category=AuditCategory.TOOL_CALL,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            action=f"tool:{tool_name}",
            agent_name=agent_name,
            tool_name=tool_name,
            description=f"Tool '{tool_name}' called",
            parameters=safe_params,
            result=result[:500],  # Truncate result
            success=success,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    def log_file_access(
        self,
        path: str,
        operation: str = "read",
        *,
        agent_name: str = "",
        success: bool = True,
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a file access."""
        return self._log(
            category=AuditCategory.FILE_ACCESS,
            severity=AuditSeverity.INFO,
            action=f"file:{operation}",
            agent_name=agent_name,
            description=f"File {operation}: {path}",
            parameters={"path": path, "operation": operation},
            success=success,
            session_id=session_id,
        )

    def log_network(
        self,
        url: str,
        method: str = "GET",
        *,
        agent_name: str = "",
        status_code: int = 0,
        success: bool = True,
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a network access."""
        return self._log(
            category=AuditCategory.NETWORK,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            action=f"network:{method}",
            agent_name=agent_name,
            description=f"{method} {url}",
            parameters={"url": url, "method": method, "status": status_code},
            success=success,
            session_id=session_id,
        )

    def log_agent_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str = "",
        *,
        session_id: str = "",
    ) -> AuditEntry:
        """Logs an agent-to-agent delegation."""
        return self._log(
            category=AuditCategory.AGENT_DELEGATION,
            severity=AuditSeverity.INFO,
            action="delegate",
            agent_name=from_agent,
            description=f"Delegation: {from_agent} → {to_agent}",
            parameters={"from": from_agent, "to": to_agent, "task": task[:200]},
            session_id=session_id,
        )

    def log_skill_install(
        self,
        package_id: str,
        *,
        source: str = "",
        success: bool = True,
        analysis_verdict: str = "",
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a skill installation."""
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
            session_id=session_id,
        )

    def log_gatekeeper(
        self,
        decision: str,
        reason: str = "",
        *,
        tool_name: str = "",
        agent_name: str = "",
        session_id: str = "",
        failure_mode: FailureMode | None = None,
    ) -> AuditEntry:
        """Logs a gatekeeper decision.

        ``failure_mode``: optional TRUST-3 explicit classification. The
        gatekeeper passes a structured value (e.g.
        ``FailureMode.PERMISSION_SCOPE_DENIED``) so the receipt
        aggregator no longer has to infer from the free-text reason.
        Ignored for non-block decisions (success path).
        """
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
            session_id=session_id,
            failure_mode=failure_mode if is_block else None,
        )

    def log_memory_op(
        self,
        operation: str,
        *,
        details: str = "",
        agent_name: str = "",
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a memory operation."""
        return self._log(
            category=AuditCategory.MEMORY_OP,
            severity=AuditSeverity.DEBUG,
            action=f"memory:{operation}",
            agent_name=agent_name,
            description=f"Memory {operation}: {details}",
            session_id=session_id,
        )

    def log_security(
        self,
        event_description: str,
        *,
        severity: AuditSeverity = AuditSeverity.WARNING,
        tool_name: str = "",
        agent_name: str = "",
        blocked: bool = False,
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a security event."""
        return self._log(
            category=AuditCategory.SECURITY,
            severity=severity,
            action="security_event",
            tool_name=tool_name,
            agent_name=agent_name,
            description=event_description,
            success=not blocked,
            session_id=session_id,
        )

    def log_user_input(
        self,
        channel: str,
        text_preview: str,
        *,
        agent_name: str = "",
        session_id: str = "",
    ) -> AuditEntry:
        """Logs an incoming user message."""
        return self._log(
            category=AuditCategory.USER_INPUT,
            severity=AuditSeverity.INFO,
            action="user_input",
            agent_name=agent_name,
            description=f"[{channel}] {text_preview[:100]}",
            success=True,
            session_id=session_id,
        )

    def log_system(
        self,
        event: str,
        *,
        description: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
        session_id: str = "",
    ) -> AuditEntry:
        """Logs a system event (start, stop, config change)."""
        return self._log(
            category=AuditCategory.SYSTEM,
            severity=severity,
            action=f"system:{event}",
            description=description or event,
            success=True,
            session_id=session_id,
        )

    def log_reflection_event(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        session_id: str = "",
        agent_name: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> AuditEntry:
        """Log an autonomous Reflector event (memory writes, learning outcomes).

        The full payload (including the caller-supplied ``payload_sha256``)
        lands in ``AuditEntry.parameters`` as a structured dict — not
        smuggled into the free-form description. ``action`` is the event
        ID (e.g. ``"causal_sequence_recorded"``,
        ``"causal_skipped_empty_sequence"``).
        """
        return self._log(
            category=AuditCategory.REFLECTION,
            severity=severity,
            action=action,
            agent_name=agent_name,
            description=f"Reflection event: {action}",
            parameters=payload,
            session_id=session_id,
        )

    # ── Queries ─────────────────────────────────────────────────

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
        """Flexible query of the audit log.

        All filters are optional and combined (AND).
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
        """All blocked actions."""
        return self.query(
            category=AuditCategory.GATEKEEPER,
            success=False,
            limit=limit,
        ) + self.query(
            category=AuditCategory.SECURITY,
            success=False,
            limit=limit,
        )

    # ── Summary ──────────────────────────────────────────

    def summarize(self, *, hours: int = 24) -> AuditSummary:
        """Creates a summary of the audit log.

        Args:
            hours: Time period in hours (backwards from now).

        Returns:
            AuditSummary with statistics.
        """
        now = datetime.now(UTC)
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
                AuditCategory.GATEKEEPER,
                AuditCategory.SECURITY,
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
        summary.avg_duration_ms = total_duration / duration_count if duration_count > 0 else 0.0

        return summary

    # ── GDPR Art. 15 Export ─────────────────────────────────────

    def get_entries_for_export(
        self,
        *,
        channel: str = "",
        hours: int = 0,
        max_entries: int = 10000,
    ) -> list[dict[str, Any]]:
        """Export audit entries for GDPR Art. 15 data subject access.

        Args:
            channel: Filter by channel name (empty = all).
            hours: Only entries from last N hours (0 = all).
            max_entries: Maximum entries to return.

        Returns:
            List of entry dicts (sanitized, no internal IDs).
        """
        cutoff = None
        if hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=hours)

        results: list[dict[str, Any]] = []
        for entry in self._entries:
            if len(results) >= max_entries:
                break
            if cutoff:
                ts = self._parse_ts(entry.timestamp)
                if ts and ts < cutoff:
                    continue
            if channel:
                desc = entry.description.lower()
                action = entry.action.lower()
                if channel.lower() not in desc and channel.lower() not in action:
                    continue
            d = entry.to_dict()
            d.pop("entry_id", None)
            results.append(d)
        return results

    # ── Export ────────────────────────────────────────────────────

    def export_json(self, path: Path, *, hours: int = 24) -> int:
        """Exports the audit log as JSON.

        Args:
            path: Target file.
            hours: Time period.

        Returns:
            Number of exported entries.
        """
        now = datetime.now(UTC)
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
        """Exports as CSV for compliance reports."""
        now = datetime.now(UTC)
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
        """Removes entries older than retention_days.

        Returns:
            Number of removed entries.
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
        # PASS-3: hold ``_persist_lock`` while replacing ``self._entries``
        # so a concurrent ``_log()`` cannot append to the about-to-be-
        # discarded deque (silently dropping the new entry).
        with self._persist_lock:
            before = len(self._entries)
            self._entries = deque(
                (e for e in self._entries if self._parse_ts(e.timestamp) > cutoff),
                maxlen=self._entries.maxlen,
            )
            removed = before - len(self._entries)

        if removed:
            logger.info(
                "Audit log: %d old entries removed (retention=%dd)",
                removed,
                self._retention_days,
            )
        return removed

    def delete_pii_entries(self) -> int:
        """Deletes all entries with personal data (GDPR).

        Returns:
            Number of deleted entries.
        """
        # PASS-3: same race fix as cleanup_old_entries above.
        with self._persist_lock:
            before = len(self._entries)
            self._entries = deque(
                (e for e in self._entries if not e.contains_pii),
                maxlen=self._entries.maxlen,
            )
            return before - len(self._entries)

    # ── Internal ───────────────────────────────────────────────────

    def _log(self, **kwargs: Any) -> AuditEntry:
        """Creates and stores an audit entry."""
        # PASS-3: counter increment + entry construction + deque append
        # must be atomic. Without the lock, two concurrent gateway
        # coroutines (multi-channel topology) could both read counter=N,
        # both create ``audit_{N+1}``, and produce duplicate entry_ids
        # that corrupt receipt sorting. The persistence call below is
        # already lock-internal (acquires the same lock again — re-entrant
        # via RLock semantics in the Python lock model would be wrong;
        # we use a plain Lock here so we release before calling persist).
        with self._persist_lock:
            self._counter += 1
            entry = AuditEntry(entry_id=f"audit_{self._counter}", **kwargs)
            self._entries.append(entry)

        # Persistence (if log_dir is set) — re-acquires _persist_lock
        # internally; that's why the construction-block above releases
        # before calling here.
        if self._log_dir:
            self._persist_entry(entry)

        return entry

    def _last_hash_for_file(self, log_file: Path) -> str:
        """Return the canonical SHA-256 of the last entry already in
        ``log_file``, or "" if the file is empty / missing.

        Used to compute ``prev_hash`` for the next entry. Caches per
        path so a busy logger doesn't re-read the file on every write.
        """
        if log_file in self._last_hash_per_file:
            return self._last_hash_per_file[log_file]
        if not log_file.exists():
            self._last_hash_per_file[log_file] = ""
            return ""
        try:
            # Walk the file to find the last non-blank line.
            last_line = ""
            with log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if not last_line:
                self._last_hash_per_file[log_file] = ""
                return ""
            data = json.loads(last_line)
            # Reconstruct a minimal AuditEntry to compute the hash; we
            # only need the canonical JSON form, so a generic dict is
            # fine.
            import hashlib

            canon = json.dumps(data, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
            self._last_hash_per_file[log_file] = h
            return h
        except Exception as exc:
            logger.warning("Audit chain head read failed for %s: %s", log_file, exc)
            self._last_hash_per_file[log_file] = ""
            return ""

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Writes an entry to the audit file with hash-chain link.

        SEC-HIGH-5: ``entry.prev_hash`` is set to the SHA-256 of the
        previous entry already on disk for this date's log file before
        writing, then this entry's own hash is cached for the next
        write. Tampering after-the-fact (insertion / deletion / edit)
        can be surfaced by ``validate_chain``.

        Audit-PR2: the read-prev / write / cache-update sequence is
        guarded by ``self._persist_lock`` because the gateway feeds
        the same singleton from multiple async coroutines (each
        running on a thread pool). Without the lock, two concurrent
        appenders read the same prev_hash and either link both
        entries to the same predecessor (chain breaks at validation)
        or stomp the cache so the *next* entry chains off the wrong
        head.

        The on-disk JSON is written with ``sort_keys=True`` so the
        bytes match what ``canonical_hash()`` hashes — the cache hit
        on the next call and a freshly-recomputed hash from disk
        (after process restart) yield the same digest regardless of
        the dataclass's insertion order.
        """
        if self._log_dir is None:
            return
        try:
            date_str = entry.timestamp[:10]  # YYYY-MM-DD
            log_file = self._log_dir / f"audit_{date_str}.jsonl"
            with self._persist_lock:
                entry.prev_hash = self._last_hash_for_file(log_file)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            entry.to_dict(),
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n",
                    )
                # Cache this entry's hash for the next persist.
                self._last_hash_per_file[log_file] = entry.canonical_hash()
        except Exception as exc:
            logger.error("Audit persistence failed: %s", exc)

    def validate_chain(self, log_file: Path) -> tuple[bool, list[str]]:
        """Walk an audit JSONL and verify the SHA-256 hash chain.

        Returns ``(ok, errors)``. ``ok=True`` means every line's
        ``prev_hash`` matches the canonical hash of the line above it.
        Errors carry human-readable descriptions of which entry broke
        the chain (insertion, deletion, or mutation).

        SEC-HIGH-5: invoke at startup or on demand to detect post-hoc
        tampering of the audit log on disk.
        """
        import hashlib

        if not log_file.exists():
            return True, []

        errors: list[str] = []
        expected_prev = ""
        line_no = 0
        try:
            with log_file.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    line_no += 1
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"line {line_no}: invalid JSON ({exc})")
                        return False, errors
                    actual_prev = data.get("prev_hash", "")
                    if actual_prev != expected_prev:
                        errors.append(
                            f"line {line_no} (entry_id={data.get('entry_id', '?')}): "
                            f"prev_hash mismatch — expected {expected_prev[:12] or '(empty)'} "
                            f"but found {actual_prev[:12] or '(empty)'}"
                        )
                    canon = json.dumps(data, sort_keys=True, ensure_ascii=False)
                    expected_prev = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        except OSError as exc:
            errors.append(f"read error: {exc}")
            return False, errors

        return len(errors) == 0, errors

    # ── Failure-Mode classification + aggregator (TRUST-3) ─────────

    @staticmethod
    def classify_failure(entry: AuditEntry) -> FailureMode | None:
        """Map an audit entry to a structured ``FailureMode``.

        TRUST-3 (operational-trust audit, 2026-05-04). Reviewer asked
        for "failure-mode classification" so an operator can answer
        *what kind of thing went wrong* without parsing free-text.

        Returns ``None`` for successful entries. For failures, walks
        a deterministic decision-tree over the existing fields
        (``category``, ``action``, ``description``) and falls back to
        ``FailureMode.UNKNOWN`` when nothing matches — that's the
        signal that a new enum value is needed.

        Pure function. Callers can pre-set ``entry.failure_mode``
        explicitly (more reliable); when None this classifier infers.
        """
        if entry.success:
            return None
        if entry.failure_mode is not None:
            return entry.failure_mode

        action = entry.action or ""
        description = (entry.description or "").lower()

        # Category-driven routing first — most reliable signal.
        if entry.category == AuditCategory.GATEKEEPER:
            # TRUST-5..6 hooks: distinguish scope / budget blocks
            # from generic gatekeeper blocks before falling through.
            if (
                "permission_scope" in description
                or "scope=" in description
                or "scope " in description
            ):
                return FailureMode.PERMISSION_SCOPE_DENIED
            if (
                "budget" in description
                or "cost limit" in description
                or "budget_exceeded" in description
            ):
                return FailureMode.BUDGET_EXCEEDED
            if "approve" in description or "approval" in description:
                return FailureMode.GATEKEEPER_APPROVAL_DENIED
            return FailureMode.GATEKEEPER_BLOCK

        if entry.category == AuditCategory.SECURITY:
            # TRUST-7..10 surfaces — match before the catch-all so a
            # SECURITY-categorised entry with the right keyword gets
            # the structured failure mode.
            if "fingerprint" in description and (
                "drift" in description or "diverg" in description or "mismatch" in description
            ):
                return FailureMode.FINGERPRINT_DRIFT
            if "escalation" in description and ("reject" in description or "denied" in description):
                return FailureMode.CLOUD_ESCALATION_REJECTED
            if "provenance" in description and ("expired" in description or "stale" in description):
                return FailureMode.PROVENANCE_EXPIRED
            if "migration" in description and ("chain" in description or "mismatch" in description):
                return FailureMode.MIGRATION_CHAIN_ERROR
            if "sandbox" in description:
                return FailureMode.SANDBOX_REFUSED
            if "auth" in description or "credential" in description:
                return FailureMode.AUTH_ERROR
            return FailureMode.GATEKEEPER_BLOCK

        if entry.category == AuditCategory.NETWORK:
            return FailureMode.NETWORK_ERROR

        if entry.category == AuditCategory.TOOL_CALL:
            if "timeout" in description or action.endswith("timeout"):
                return FailureMode.TOOL_TIMEOUT
            if "not found" in description or "no such tool" in description:
                return FailureMode.TOOL_NOT_FOUND
            if "invalid" in description and ("param" in description or "argument" in description):
                return FailureMode.TOOL_INVALID_PARAMS
            if "sandbox" in description or "refused" in description:
                return FailureMode.SANDBOX_REFUSED
            if "quota" in description or "rate limit" in description:
                return FailureMode.QUOTA_EXCEEDED
            return FailureMode.TOOL_INTERNAL_ERROR

        return FailureMode.UNKNOWN

    def failures_by_mode(
        self,
        *,
        hours: int = 24,
    ) -> dict[str, int]:
        """Aggregate failure counts by ``FailureMode`` for the last
        *hours* hours.

        Returns a dict keyed by the FailureMode enum value (string),
        sorted descending by count for stable display order. Empty
        dict means "no failures in the window" — the operator can
        treat this as a success signal.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        counts: dict[str, int] = {}
        for entry in self._entries:
            try:
                ts = datetime.fromisoformat(entry.timestamp)
            except (ValueError, TypeError):
                continue
            if ts < cutoff:
                continue
            mode = self.classify_failure(entry)
            if mode is None:
                continue
            key = mode.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    # ── Run-Receipt (TRUST-1) ───────────────────────────────────────

    # Schema version of the receipt format. Bump on breaking changes
    # so consumers can detect old/new bundles.
    RECEIPT_SCHEMA_VERSION = 1

    def run_receipt(
        self,
        session_id: str,
        *,
        signing_key: str | None = None,
        include_trust: bool = False,
    ) -> dict[str, Any]:
        """Aggregate every audit entry tagged with *session_id* into a
        single receipt bundle, suitable for post-mortem reconstruction
        of "what did the agent do during run X".

        TRUST-1 (operational-trust audit, 2026-05-04). Reviewer asked:
        "If something goes wrong, can an operator reconstruct exactly
        what the agent knew, what it decided, which tool it called,
        why it was allowed, what changed, and how to roll it back?"

        The bundle contains:

        * ``schema_version`` — bump on breaking changes
        * ``session_id``
        * ``period_start`` / ``period_end`` — first + last entry timestamps
        * ``entry_count``
        * ``aggregate`` — counts by category, severity, tool; total
          duration_ms; success / failure counts; PII-flagged count
        * ``entries`` — every full ``AuditEntry.to_dict()`` for the run,
          ordered by entry_id
        * ``hash_chain_head`` / ``hash_chain_tail`` — first + last
          ``prev_hash`` values (lets a verifier locate the run inside
          the JSONL chain)
        * ``signature`` — HMAC-SHA-256 of the canonical bundle (without
          the signature field). Empty when *signing_key* is None.

        Pass ``include_trust=True`` to fold the TRUST-5..10 ledger
        bundle (permission scopes, cost, fingerprints, escalations,
        provenance, migrations) into a top-level ``"trust"`` key.
        Default is False so existing consumers see no shape change.
        The signature, when requested, covers the merged bundle —
        operators get tamper-detection across the audit entries AND
        the trust state at run time.

        The receipt is purely an aggregation over already-persisted
        data — no side effects, no mutation. Safe to call on a live
        logger or post-mortem on disk-only entries via a logger
        instantiated against the persisted ``log_dir``.
        """
        # 1. Pull matching entries from the in-memory ring buffer.
        entries = [e for e in self._entries if e.session_id == session_id]

        # 2. If we have a log_dir but no in-memory hits (e.g. restarted
        # process), scan today's JSONL too. We do NOT walk every file
        # by default — operators can replay specific files via
        # ``run_receipt_from_file`` for cross-day audits.
        from_disk: list[dict[str, Any]] = []
        if not entries and self._log_dir is not None:
            for jsonl in sorted(self._log_dir.glob("audit_*.jsonl")):
                try:
                    for raw_line in jsonl.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("session_id", "") == session_id:
                            from_disk.append(data)
                except (OSError, json.JSONDecodeError):
                    continue

        # 3. Build the entry list — prefer in-memory (richer enums,
        # known-good shape); fall back to disk dicts.
        if entries:
            entry_dicts = [e.to_dict() for e in entries]
        else:
            entry_dicts = from_disk

        # 4. Aggregate (delegate to the static builder so the REST
        # endpoint and other callers can produce the same receipt
        # shape from already-loaded dicts without going through the
        # in-memory ring buffer).
        return self.build_receipt_from_entries(
            session_id,
            entry_dicts,
            signing_key=signing_key,
            include_trust=include_trust,
        )

    @classmethod
    def build_receipt_from_entries(
        cls,
        session_id: str,
        entry_dicts: list[dict[str, Any]],
        *,
        signing_key: str | None = None,
        include_trust: bool = False,
    ) -> dict[str, Any]:
        """Build a TRUST-1 receipt from already-loaded entry dicts.

        Lets non-AuditLogger callers (REST endpoints, post-mortem
        tools) produce the same receipt shape without going through
        the in-memory ring buffer or the ``audit_<date>.jsonl`` glob.
        ``entry_dicts`` is the list of ``AuditEntry.to_dict()`` (or
        equivalent JSONL-parsed) dicts already filtered by
        ``session_id``.

        Same return shape and signing semantics as
        :meth:`run_receipt`. ``include_trust=True`` folds the
        TRUST-5..10 bundle under ``"trust"``.
        """
        import hashlib
        import hmac

        if not entry_dicts:
            bundle: dict[str, Any] = {
                "schema_version": cls.RECEIPT_SCHEMA_VERSION,
                "session_id": session_id,
                "period_start": "",
                "period_end": "",
                "entry_count": 0,
                "aggregate": {
                    "by_category": {},
                    "by_severity": {},
                    "by_tool": {},
                    "total_duration_ms": 0.0,
                    "success_count": 0,
                    "failure_count": 0,
                    "pii_count": 0,
                },
                "entries": [],
                "hash_chain_head": "",
                "hash_chain_tail": "",
                "signature": "",
            }
        else:
            by_category: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            by_tool: dict[str, int] = {}
            total_duration = 0.0
            success_count = 0
            failure_count = 0
            pii_count = 0
            for d in entry_dicts:
                by_category[d.get("category", "unknown")] = (
                    by_category.get(d.get("category", "unknown"), 0) + 1
                )
                by_severity[d.get("severity", "unknown")] = (
                    by_severity.get(d.get("severity", "unknown"), 0) + 1
                )
                tool = d.get("tool_name", "")
                if tool:
                    by_tool[tool] = by_tool.get(tool, 0) + 1
                total_duration += float(d.get("duration_ms", 0.0))
                if d.get("success", True):
                    success_count += 1
                else:
                    failure_count += 1
                if d.get("contains_pii", False):
                    pii_count += 1

            def _entry_sort_key(d: dict[str, Any]) -> tuple[int, str]:
                eid = str(d.get("entry_id", ""))
                try:
                    return (int(eid.rsplit("_", 1)[-1]), eid)
                except (ValueError, IndexError):
                    return (0, eid)

            ordered = sorted(entry_dicts, key=_entry_sort_key)

            bundle = {
                "schema_version": cls.RECEIPT_SCHEMA_VERSION,
                "session_id": session_id,
                "period_start": ordered[0].get("timestamp", ""),
                "period_end": ordered[-1].get("timestamp", ""),
                "entry_count": len(ordered),
                "aggregate": {
                    "by_category": by_category,
                    "by_severity": by_severity,
                    "by_tool": by_tool,
                    "total_duration_ms": round(total_duration, 2),
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "pii_count": pii_count,
                },
                "entries": ordered,
                "hash_chain_head": ordered[0].get("prev_hash", ""),
                "hash_chain_tail": ordered[-1].get("prev_hash", ""),
                "signature": "",
            }

        if include_trust:
            # Deep-PR2 (DEEP-2): receipt-bundle assembly is purely
            # advisory aggregation — a fault inside any of the six
            # ledger snapshots (cost / fingerprint / scope / etc.)
            # used to propagate as an unhandled exception out of
            # ``build_receipt_from_entries`` and surfaced as a 500
            # at the REST endpoint with no partial receipt. The TRUST
            # convention established in `mcp/video_tools.py` is
            # best-effort emission. Mirror it: any failure produces
            # a structured error stub so the rest of the receipt
            # still ships.
            try:
                from cognithor.security.trust_bundle import build_trust_bundle

                bundle["trust"] = build_trust_bundle(session_id)
            except Exception as exc:
                logger.warning(
                    "trust_bundle_build_failed session_id=%s error=%s type=%s",
                    session_id,
                    str(exc),
                    type(exc).__name__,
                )
                bundle["trust"] = {
                    "error": "trust bundle unavailable",
                    "reason": str(exc)[:200],
                    "error_type": type(exc).__name__,
                }

        if signing_key:
            canonical = json.dumps(
                {k: v for k, v in bundle.items() if k != "signature"},
                sort_keys=True,
                ensure_ascii=False,
            )
            bundle["signature"] = hmac.new(
                signing_key.encode("utf-8"),
                canonical.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        return bundle

    @staticmethod
    def verify_receipt_signature(
        receipt: dict[str, Any],
        signing_key: str,
    ) -> bool:
        """Verify a signed receipt bundle against the same signing
        key. Returns ``True`` only if the recomputed HMAC matches
        the receipt's ``signature`` field.

        Use case: the operator persists a receipt as evidence and
        later wants to confirm it hasn't been edited.
        """
        import hashlib
        import hmac

        sig = receipt.get("signature", "")
        if not sig:
            return False
        canonical = json.dumps(
            {k: v for k, v in receipt.items() if k != "signature"},
            sort_keys=True,
            ensure_ascii=False,
        )
        expected = hmac.new(
            signing_key.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        # ``compare_digest`` is constant-time to defend against timing
        # oracles (the receipt could come from an untrusted source).
        return hmac.compare_digest(expected, sig)

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Removes credentials from parameters."""
        sensitive_keys = {
            "password",
            "token",
            "api_key",
            "secret",
            "authorization",
            "credential",
            "private_key",
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
            return datetime.min.replace(tzinfo=UTC)

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

from cognithor.audit.ai_act_export import (
    ComplianceExporter,
    TransparencyChecker,
)
from cognithor.audit.ai_act_export import (
    RiskClassifier as ExportRiskClassifier,
)
from cognithor.audit.compliance import (
    ComplianceFramework,
    DecisionLog,
    RemediationTracker,
    ReportExporter,
)
from cognithor.audit.ethics import (
    BiasDetector,
    BudgetManager,
    CostTracker,
    EconomicGovernor,
    EthicsPolicy,
    FairnessAuditor,
)
from cognithor.audit.eu_ai_act import (
    ComplianceDocManager,
    EUAIActGovernor,
    RiskClassifier,
    TrainingCatalog,
    TransparencyRegister,
)

# Alias so both RiskClassifier variants are accessible
AIActExportRiskClassifier = ExportRiskClassifier
from cognithor.audit.impact_assessment import (
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
