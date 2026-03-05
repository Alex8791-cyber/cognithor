"""GDPR Compliance Toolkit.

Provides data processing logs, model usage records, retention policies,
right-to-erasure support, and audit export (PDF + JSON) to fulfil
DSGVO/GDPR obligations (Art. 5, 6, 13-17, 25, 30, 32, 35).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from jarvis.utils.logging import get_logger

_log = get_logger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────


class ProcessingBasis(StrEnum):
    """Legal basis for data processing (Art. 6 GDPR)."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_INTEREST = "public_interest"
    LEGITIMATE_INTEREST = "legitimate_interest"


class DataCategory(StrEnum):
    """Categories of personal data processed."""

    QUERY = "query"
    MEMORY = "memory"
    FILE_CONTENT = "file_content"
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    TELEMETRY = "telemetry"
    VOICE = "voice"
    CREDENTIAL = "credential"


class ErasureStatus(StrEnum):
    """Status of a right-to-erasure request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    DENIED = "denied"


class RetentionAction(StrEnum):
    """Action when retention period expires."""

    DELETE = "delete"
    ANONYMIZE = "anonymize"
    ARCHIVE = "archive"


# ── Data Processing Log ──────────────────────────────────────────────────


@dataclass
class DataProcessingRecord:
    """Single record of personal data processing (Art. 30 GDPR)."""

    record_id: str = ""
    user_id: str = ""
    timestamp: str = ""
    category: DataCategory = DataCategory.QUERY
    purpose: str = ""
    legal_basis: ProcessingBasis = ProcessingBasis.LEGITIMATE_INTEREST
    tool_name: str = ""
    data_summary: str = ""
    data_hash: str = ""
    retention_days: int = 90
    third_party: str = ""
    country: str = "DE"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "purpose": self.purpose,
            "legal_basis": self.legal_basis.value,
            "tool_name": self.tool_name,
            "data_summary": self.data_summary,
            "data_hash": self.data_hash,
            "retention_days": self.retention_days,
            "third_party": self.third_party,
            "country": self.country,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataProcessingRecord:
        return cls(
            record_id=d.get("record_id", ""),
            user_id=d.get("user_id", ""),
            timestamp=d.get("timestamp", ""),
            category=DataCategory(d["category"]) if "category" in d else DataCategory.QUERY,
            purpose=d.get("purpose", ""),
            legal_basis=ProcessingBasis(d["legal_basis"]) if "legal_basis" in d else ProcessingBasis.LEGITIMATE_INTEREST,
            tool_name=d.get("tool_name", ""),
            data_summary=d.get("data_summary", ""),
            data_hash=d.get("data_hash", ""),
            retention_days=d.get("retention_days", 90),
            third_party=d.get("third_party", ""),
            country=d.get("country", "DE"),
        )


class DataProcessingLog:
    """Append-only log of all personal data processing activities.

    Fulfils Art. 30 GDPR — Records of processing activities.
    """

    def __init__(self) -> None:
        self._records: list[DataProcessingRecord] = []
        self._counter = 0

    @property
    def records(self) -> list[DataProcessingRecord]:
        return list(self._records)

    def record(
        self,
        user_id: str,
        category: DataCategory,
        purpose: str,
        *,
        tool_name: str = "",
        data_summary: str = "",
        raw_data: str = "",
        legal_basis: ProcessingBasis = ProcessingBasis.LEGITIMATE_INTEREST,
        retention_days: int = 90,
        third_party: str = "",
        country: str = "DE",
    ) -> DataProcessingRecord:
        """Record a data processing activity."""
        self._counter += 1
        data_hash = ""
        if raw_data:
            data_hash = hashlib.sha256(raw_data.encode()).hexdigest()[:16]

        rec = DataProcessingRecord(
            record_id=f"dpr-{self._counter:06d}",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            purpose=purpose,
            legal_basis=legal_basis,
            tool_name=tool_name,
            data_summary=data_summary,
            data_hash=data_hash,
            retention_days=retention_days,
            third_party=third_party,
            country=country,
        )
        self._records.append(rec)
        return rec

    def query(
        self,
        *,
        user_id: str = "",
        category: DataCategory | None = None,
        tool_name: str = "",
    ) -> list[DataProcessingRecord]:
        """Query records by user, category, or tool."""
        results = self._records
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if category is not None:
            results = [r for r in results if r.category == category]
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        return results

    def user_report(self, user_id: str) -> dict:
        """Generate a DSGVO data subject access report (Art. 15)."""
        user_records = self.query(user_id=user_id)
        categories = sorted({r.category.value for r in user_records})
        tools = sorted({r.tool_name for r in user_records if r.tool_name})
        purposes = sorted({r.purpose for r in user_records if r.purpose})
        third_parties = sorted({r.third_party for r in user_records if r.third_party})

        return {
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(user_records),
            "categories": categories,
            "tools_used": tools,
            "purposes": purposes,
            "third_parties": third_parties,
            "records": [r.to_dict() for r in user_records],
        }

    def delete_user_records(self, user_id: str) -> int:
        """Delete all records for a user (Art. 17). Returns count deleted."""
        before = len(self._records)
        self._records = [r for r in self._records if r.user_id != user_id]
        deleted = before - len(self._records)
        _log.info("gdpr.records_deleted", user_id=user_id, count=deleted)
        return deleted


# ── Model Usage Records ──────────────────────────────────────────────────


@dataclass
class ModelUsageRecord:
    """Record of a single LLM invocation."""

    record_id: str = ""
    timestamp: str = ""
    user_id: str = ""
    model_name: str = ""
    provider: str = "ollama"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    purpose: str = ""
    contains_pii: bool = False
    input_hash: str = ""
    success: bool = True

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "purpose": self.purpose,
            "contains_pii": self.contains_pii,
            "input_hash": self.input_hash,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelUsageRecord:
        return cls(
            record_id=d.get("record_id", ""),
            timestamp=d.get("timestamp", ""),
            user_id=d.get("user_id", ""),
            model_name=d.get("model_name", ""),
            provider=d.get("provider", "ollama"),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            latency_ms=d.get("latency_ms", 0.0),
            purpose=d.get("purpose", ""),
            contains_pii=d.get("contains_pii", False),
            input_hash=d.get("input_hash", ""),
            success=d.get("success", True),
        )


class ModelUsageLog:
    """Tracks all LLM invocations for compliance and cost auditing."""

    def __init__(self) -> None:
        self._records: list[ModelUsageRecord] = []
        self._counter = 0

    @property
    def records(self) -> list[ModelUsageRecord]:
        return list(self._records)

    def record(
        self,
        user_id: str,
        model_name: str,
        *,
        provider: str = "ollama",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        purpose: str = "",
        contains_pii: bool = False,
        raw_input: str = "",
        success: bool = True,
    ) -> ModelUsageRecord:
        """Record an LLM invocation."""
        self._counter += 1
        input_hash = ""
        if raw_input:
            input_hash = hashlib.sha256(raw_input.encode()).hexdigest()[:16]

        rec = ModelUsageRecord(
            record_id=f"mur-{self._counter:06d}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            model_name=model_name,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            purpose=purpose,
            contains_pii=contains_pii,
            input_hash=input_hash,
            success=success,
        )
        self._records.append(rec)
        return rec

    def query(
        self,
        *,
        user_id: str = "",
        model_name: str = "",
        contains_pii: bool | None = None,
    ) -> list[ModelUsageRecord]:
        """Query model usage records."""
        results = self._records
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if contains_pii is not None:
            results = [r for r in results if r.contains_pii == contains_pii]
        return results

    def usage_summary(self) -> dict:
        """Aggregate usage statistics per model."""
        by_model: dict[str, dict[str, Any]] = {}
        for r in self._records:
            m = by_model.setdefault(r.model_name, {
                "calls": 0, "total_tokens": 0, "total_latency_ms": 0.0,
                "pii_calls": 0, "errors": 0,
            })
            m["calls"] += 1
            m["total_tokens"] += r.total_tokens
            m["total_latency_ms"] += r.latency_ms
            if r.contains_pii:
                m["pii_calls"] += 1
            if not r.success:
                m["errors"] += 1
        return by_model

    def delete_user_records(self, user_id: str) -> int:
        """Delete all records for a user. Returns count deleted."""
        before = len(self._records)
        self._records = [r for r in self._records if r.user_id != user_id]
        return before - len(self._records)


# ── Retention Policies ───────────────────────────────────────────────────


@dataclass
class RetentionPolicy:
    """Configurable data retention policy."""

    name: str = ""
    category: DataCategory = DataCategory.QUERY
    retention_days: int = 90
    action: RetentionAction = RetentionAction.DELETE
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "retention_days": self.retention_days,
            "action": self.action.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RetentionPolicy:
        return cls(
            name=d.get("name", ""),
            category=DataCategory(d["category"]) if "category" in d else DataCategory.QUERY,
            retention_days=d.get("retention_days", 90),
            action=RetentionAction(d["action"]) if "action" in d else RetentionAction.DELETE,
            description=d.get("description", ""),
        )


# Default retention policies per DSGVO Art. 5(1)(e)
DEFAULT_RETENTION_POLICIES: list[RetentionPolicy] = [
    RetentionPolicy(
        name="query_retention",
        category=DataCategory.QUERY,
        retention_days=90,
        action=RetentionAction.DELETE,
        description="User queries deleted after 90 days",
    ),
    RetentionPolicy(
        name="conversation_retention",
        category=DataCategory.CONVERSATION,
        retention_days=180,
        action=RetentionAction.ANONYMIZE,
        description="Conversations anonymized after 180 days",
    ),
    RetentionPolicy(
        name="telemetry_retention",
        category=DataCategory.TELEMETRY,
        retention_days=365,
        action=RetentionAction.DELETE,
        description="Telemetry data deleted after 1 year",
    ),
    RetentionPolicy(
        name="voice_retention",
        category=DataCategory.VOICE,
        retention_days=30,
        action=RetentionAction.DELETE,
        description="Voice data deleted after 30 days",
    ),
    RetentionPolicy(
        name="credential_retention",
        category=DataCategory.CREDENTIAL,
        retention_days=0,
        action=RetentionAction.DELETE,
        description="Credentials never retained in processing logs",
    ),
    RetentionPolicy(
        name="memory_retention",
        category=DataCategory.MEMORY,
        retention_days=365,
        action=RetentionAction.ARCHIVE,
        description="Memory entries archived after 1 year",
    ),
]


class RetentionEnforcer:
    """Enforces retention policies by identifying and marking expired records."""

    def __init__(
        self,
        policies: list[RetentionPolicy] | None = None,
    ) -> None:
        effective = DEFAULT_RETENTION_POLICIES if policies is None else policies
        self._policies = {p.category: p for p in effective}

    @property
    def policies(self) -> dict[DataCategory, RetentionPolicy]:
        return dict(self._policies)

    def add_policy(self, policy: RetentionPolicy) -> None:
        self._policies[policy.category] = policy

    def find_expired(
        self,
        records: list[DataProcessingRecord],
        *,
        now: datetime | None = None,
    ) -> list[tuple[DataProcessingRecord, RetentionAction]]:
        """Find records that have exceeded their retention period."""
        now = now or datetime.now(timezone.utc)
        expired: list[tuple[DataProcessingRecord, RetentionAction]] = []

        for rec in records:
            policy = self._policies.get(rec.category)
            if policy is None:
                continue

            retention = policy.retention_days
            if retention <= 0:
                # Immediate deletion policy
                expired.append((rec, policy.action))
                continue

            try:
                created = datetime.fromisoformat(rec.timestamp)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if now - created > timedelta(days=retention):
                expired.append((rec, policy.action))

        return expired

    def enforce(
        self,
        log: DataProcessingLog,
        *,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Enforce retention policies on a processing log.

        Returns counts per action taken: ``{"delete": N, "anonymize": M, ...}``.
        """
        expired = self.find_expired(log.records, now=now)
        counts: dict[str, int] = {}

        delete_ids: set[str] = set()
        for rec, action in expired:
            counts[action.value] = counts.get(action.value, 0) + 1
            if action == RetentionAction.DELETE:
                delete_ids.add(rec.record_id)

        if delete_ids:
            log._records = [
                r for r in log._records if r.record_id not in delete_ids
            ]

        _log.info(
            "gdpr.retention_enforced",
            expired_count=len(expired),
            actions=counts,
        )
        return counts


# ── Right to Erasure ─────────────────────────────────────────────────────


@dataclass
class ErasureRequest:
    """Tracks a right-to-erasure request (Art. 17 GDPR)."""

    request_id: str = ""
    user_id: str = ""
    requested_at: str = ""
    status: ErasureStatus = ErasureStatus.PENDING
    completed_at: str = ""
    records_deleted: int = 0
    model_records_deleted: int = 0
    erasure_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "requested_at": self.requested_at,
            "status": self.status.value,
            "completed_at": self.completed_at,
            "records_deleted": self.records_deleted,
            "model_records_deleted": self.model_records_deleted,
            "erasure_log": list(self.erasure_log),
        }


class ErasureManager:
    """Handles right-to-erasure requests across all data stores."""

    def __init__(self) -> None:
        self._requests: list[ErasureRequest] = []
        self._counter = 0
        self._erasure_handlers: list[Callable[[str], int]] = []

    @property
    def requests(self) -> list[ErasureRequest]:
        return list(self._requests)

    def register_handler(self, handler: Callable[[str], int]) -> None:
        """Register an additional erasure handler (e.g., for memory store)."""
        self._erasure_handlers.append(handler)

    def request_erasure(
        self,
        user_id: str,
        processing_log: DataProcessingLog,
        usage_log: ModelUsageLog,
    ) -> ErasureRequest:
        """Execute a right-to-erasure request.

        Deletes all user data from processing log, model usage log,
        and any registered external handlers.
        """
        self._counter += 1
        req = ErasureRequest(
            request_id=f"era-{self._counter:06d}",
            user_id=user_id,
            requested_at=datetime.now(timezone.utc).isoformat(),
            status=ErasureStatus.IN_PROGRESS,
        )

        # Delete from processing log
        dp_deleted = processing_log.delete_user_records(user_id)
        req.records_deleted = dp_deleted
        req.erasure_log.append(f"Processing log: {dp_deleted} records deleted")

        # Delete from model usage log
        mu_deleted = usage_log.delete_user_records(user_id)
        req.model_records_deleted = mu_deleted
        req.erasure_log.append(f"Model usage log: {mu_deleted} records deleted")

        # Run external handlers
        for handler in self._erasure_handlers:
            try:
                count = handler(user_id)
                req.erasure_log.append(f"External handler: {count} records deleted")
            except Exception as e:
                req.erasure_log.append(f"External handler failed: {e}")
                req.status = ErasureStatus.PARTIALLY_COMPLETED

        if req.status != ErasureStatus.PARTIALLY_COMPLETED:
            req.status = ErasureStatus.COMPLETED
        req.completed_at = datetime.now(timezone.utc).isoformat()

        self._requests.append(req)
        _log.info(
            "gdpr.erasure_completed",
            request_id=req.request_id,
            user_id=user_id,
            status=req.status.value,
            total_deleted=req.records_deleted + req.model_records_deleted,
        )
        return req


# ── Audit Export ─────────────────────────────────────────────────────────


class AuditExporter:
    """Exports compliance audit trails in multiple formats."""

    def __init__(
        self,
        processing_log: DataProcessingLog,
        usage_log: ModelUsageLog,
        erasure_manager: ErasureManager | None = None,
        retention_enforcer: RetentionEnforcer | None = None,
    ) -> None:
        self._processing_log = processing_log
        self._usage_log = usage_log
        self._erasure_manager = erasure_manager
        self._retention_enforcer = retention_enforcer

    def to_json(self, *, user_id: str = "") -> str:
        """Export audit trail as JSON."""
        data = self._build_report(user_id=user_id)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save_json(self, path: str | Path, *, user_id: str = "") -> None:
        """Save audit trail as JSON file."""
        Path(path).write_text(self.to_json(user_id=user_id), encoding="utf-8")

    def to_markdown(self, *, user_id: str = "") -> str:
        """Export audit trail as Markdown."""
        data = self._build_report(user_id=user_id)
        lines: list[str] = []

        lines.append(f"# GDPR Compliance Audit Report")
        lines.append(f"")
        lines.append(f"**Generated:** {data['generated_at']}")
        if user_id:
            lines.append(f"**User:** {user_id}")
        lines.append(f"")

        # Summary
        summary = data["summary"]
        lines.append("## Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Processing Records | {summary['total_processing_records']} |")
        lines.append(f"| Total Model Usage Records | {summary['total_model_usage_records']} |")
        lines.append(f"| Data Categories | {', '.join(summary['categories'])} |")
        lines.append(f"| Models Used | {', '.join(summary['models_used'])} |")
        lines.append(f"| PII Invocations | {summary['pii_invocations']} |")
        lines.append(f"")

        # Retention policies
        if data.get("retention_policies"):
            lines.append("## Retention Policies")
            lines.append(f"")
            lines.append(f"| Category | Retention | Action | Description |")
            lines.append(f"|----------|-----------|--------|-------------|")
            for p in data["retention_policies"]:
                lines.append(
                    f"| {p['category']} | {p['retention_days']}d | "
                    f"{p['action']} | {p['description']} |"
                )
            lines.append(f"")

        # Erasure requests
        if data.get("erasure_requests"):
            lines.append("## Erasure Requests")
            lines.append(f"")
            for er in data["erasure_requests"]:
                lines.append(f"- **{er['request_id']}** ({er['status']}): "
                             f"User {er['user_id']}, "
                             f"{er['records_deleted']} processing + "
                             f"{er['model_records_deleted']} model records deleted")
            lines.append(f"")

        # Processing records
        if data.get("processing_records"):
            lines.append("## Processing Records")
            lines.append(f"")
            lines.append(f"| ID | User | Category | Tool | Purpose | Legal Basis |")
            lines.append(f"|----|------|----------|------|---------|-------------|")
            for r in data["processing_records"][:100]:
                lines.append(
                    f"| {r['record_id']} | {r['user_id']} | {r['category']} | "
                    f"{r['tool_name']} | {r['purpose']} | {r['legal_basis']} |"
                )
            if len(data["processing_records"]) > 100:
                lines.append(f"| ... | {len(data['processing_records']) - 100} more records | | | | |")
            lines.append(f"")

        return "\n".join(lines)

    def _build_report(self, *, user_id: str = "") -> dict:
        """Build the complete audit report data structure."""
        if user_id:
            proc_records = self._processing_log.query(user_id=user_id)
            usage_records = self._usage_log.query(user_id=user_id)
        else:
            proc_records = self._processing_log.records
            usage_records = self._usage_log.records

        categories = sorted({r.category.value for r in proc_records})
        models_used = sorted({r.model_name for r in usage_records})
        pii_count = sum(1 for r in usage_records if r.contains_pii)

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id or "all",
            "summary": {
                "total_processing_records": len(proc_records),
                "total_model_usage_records": len(usage_records),
                "categories": categories,
                "models_used": models_used,
                "pii_invocations": pii_count,
            },
            "processing_records": [r.to_dict() for r in proc_records],
            "model_usage_records": [r.to_dict() for r in usage_records],
        }

        if self._retention_enforcer:
            report["retention_policies"] = [
                p.to_dict() for p in self._retention_enforcer.policies.values()
            ]

        if self._erasure_manager:
            requests = self._erasure_manager.requests
            if user_id:
                requests = [r for r in requests if r.user_id == user_id]
            report["erasure_requests"] = [r.to_dict() for r in requests]

        return report


# ── GDPR Compliance Manager ─────────────────────────────────────────────


class GDPRComplianceManager:
    """Orchestrates all GDPR compliance functions.

    Provides a unified interface for data processing logging,
    model usage tracking, retention enforcement, erasure, and export.
    """

    def __init__(
        self,
        retention_policies: list[RetentionPolicy] | None = None,
    ) -> None:
        self.processing_log = DataProcessingLog()
        self.usage_log = ModelUsageLog()
        self.retention = RetentionEnforcer(retention_policies)
        self.erasure = ErasureManager()
        self.exporter = AuditExporter(
            self.processing_log,
            self.usage_log,
            self.erasure,
            self.retention,
        )

    def log_processing(
        self,
        user_id: str,
        category: DataCategory,
        purpose: str,
        **kwargs: Any,
    ) -> DataProcessingRecord:
        """Convenience: log a data processing activity."""
        return self.processing_log.record(
            user_id, category, purpose, **kwargs,
        )

    def log_model_usage(
        self,
        user_id: str,
        model_name: str,
        **kwargs: Any,
    ) -> ModelUsageRecord:
        """Convenience: log an LLM invocation."""
        return self.usage_log.record(user_id, model_name, **kwargs)

    def enforce_retention(self, *, now: datetime | None = None) -> dict[str, int]:
        """Run retention enforcement."""
        return self.retention.enforce(self.processing_log, now=now)

    def erase_user(self, user_id: str) -> ErasureRequest:
        """Execute right-to-erasure for a user."""
        return self.erasure.request_erasure(
            user_id, self.processing_log, self.usage_log,
        )

    def user_report(self, user_id: str) -> dict:
        """Generate a data subject access report (Art. 15)."""
        return self.processing_log.user_report(user_id)

    def compliance_summary(self) -> dict:
        """Overall GDPR compliance status."""
        proc_count = len(self.processing_log.records)
        usage_count = len(self.usage_log.records)
        pii_count = sum(1 for r in self.usage_log.records if r.contains_pii)
        policy_count = len(self.retention.policies)
        erasure_count = len(self.erasure.requests)
        completed = sum(
            1 for r in self.erasure.requests
            if r.status == ErasureStatus.COMPLETED
        )

        return {
            "processing_records": proc_count,
            "model_usage_records": usage_count,
            "pii_invocations": pii_count,
            "retention_policies": policy_count,
            "erasure_requests_total": erasure_count,
            "erasure_requests_completed": completed,
            "has_retention_policies": policy_count > 0,
            "has_processing_log": proc_count > 0,
            "has_model_usage_log": usage_count > 0,
        }
