"""Jarvis · Security-Hardening & CI/CD-Integration.

Durchgängige CI/CD-Security-Gates und Container-Isolation:

  - SecurityGate:          Pre-Deploy-Gate mit konfigurierbaren Schwellwerten
  - PreCommitHook:         Git-Pre-Commit-Integration für Security-Checks
  - ContainerIsolation:    Per-Agent-Container mit eigenen Secrets
  - WebhookNotifier:       Benachrichtigungen bei Security-Events
  - SecurityScanScheduler: Zeitgesteuerte, wiederkehrende Scans
  - HardeningConfig:       Zentrale Härtungskonfiguration

Architektur-Bibel: §11.8 (DevSecOps), §14.5 (Container-Security)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================================
# Security Gate (CI/CD)
# ============================================================================


class GateDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class GatePolicy:
    """Konfigurierbare Schwellwerte für das Security-Gate."""

    max_critical_findings: int = 0       # 0 = kein Critical erlaubt
    max_high_findings: int = 3
    max_total_findings: int = 20
    min_pass_rate: float = 80.0          # Pipeline-Pass-Rate
    require_dependency_scan: bool = True
    require_fuzzing: bool = True
    require_inversion_check: bool = True
    block_on_unscanned: bool = True      # Block wenn Scan fehlt

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_critical": self.max_critical_findings,
            "max_high": self.max_high_findings,
            "max_total": self.max_total_findings,
            "min_pass_rate": self.min_pass_rate,
            "require_dep_scan": self.require_dependency_scan,
            "require_fuzzing": self.require_fuzzing,
            "block_unscanned": self.block_on_unscanned,
        }


@dataclass
class GateResult:
    """Ergebnis einer Security-Gate-Prüfung."""

    decision: GateDecision
    reasons: list[str] = field(default_factory=list)
    findings_summary: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""
    pipeline_run_id: str = ""
    policy_used: str = "default"

    @property
    def blocked(self) -> bool:
        return self.decision == GateDecision.BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "blocked": self.blocked,
            "reasons": self.reasons,
            "findings": self.findings_summary,
            "timestamp": self.timestamp,
            "run_id": self.pipeline_run_id,
        }


class SecurityGate:
    """Pre-Deploy-Gate: Blockiert Deployments bei Sicherheitsrisiken."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self._policy = policy or GatePolicy()
        self._history: list[GateResult] = []

    @property
    def policy(self) -> GatePolicy:
        return self._policy

    def evaluate(
        self,
        *,
        critical_findings: int = 0,
        high_findings: int = 0,
        medium_findings: int = 0,
        low_findings: int = 0,
        pass_rate: float = 100.0,
        stages_run: list[str] | None = None,
        pipeline_run_id: str = "",
    ) -> GateResult:
        """Prüft ob ein Deployment erlaubt ist."""
        total = critical_findings + high_findings + medium_findings + low_findings
        reasons: list[str] = []
        decision = GateDecision.ALLOW

        # Critical Findings
        if critical_findings > self._policy.max_critical_findings:
            decision = GateDecision.BLOCK
            reasons.append(f"{critical_findings} kritische Findings (max: {self._policy.max_critical_findings})")

        # High Findings
        if high_findings > self._policy.max_high_findings:
            if decision != GateDecision.BLOCK:
                decision = GateDecision.BLOCK
            reasons.append(f"{high_findings} High-Findings (max: {self._policy.max_high_findings})")

        # Total
        if total > self._policy.max_total_findings:
            decision = GateDecision.BLOCK
            reasons.append(f"{total} Gesamt-Findings (max: {self._policy.max_total_findings})")

        # Pass-Rate
        if pass_rate < self._policy.min_pass_rate:
            decision = GateDecision.BLOCK
            reasons.append(f"Pass-Rate {pass_rate:.1f}% < Minimum {self._policy.min_pass_rate}%")

        # Pflicht-Stages
        stages = set(stages_run or [])
        if self._policy.require_fuzzing and "adversarial_fuzzing" not in stages:
            if self._policy.block_on_unscanned:
                decision = GateDecision.BLOCK
            else:
                if decision == GateDecision.ALLOW:
                    decision = GateDecision.WARN
            reasons.append("Adversarial-Fuzzing nicht durchgeführt")

        if self._policy.require_dependency_scan and "dependency_scan" not in stages:
            if self._policy.block_on_unscanned:
                decision = GateDecision.BLOCK
            reasons.append("Dependency-Scan nicht durchgeführt")

        # Warnung bei medium Findings
        if medium_findings > 5 and decision == GateDecision.ALLOW:
            decision = GateDecision.WARN
            reasons.append(f"{medium_findings} Medium-Findings (Warnung)")

        if not reasons:
            reasons.append("Alle Prüfungen bestanden")

        result = GateResult(
            decision=decision,
            reasons=reasons,
            findings_summary={
                "critical": critical_findings,
                "high": high_findings,
                "medium": medium_findings,
                "low": low_findings,
                "total": total,
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            pipeline_run_id=pipeline_run_id,
        )
        self._history.append(result)
        return result

    def history(self, limit: int = 20) -> list[GateResult]:
        return list(reversed(self._history[-limit:]))

    @property
    def gate_count(self) -> int:
        return len(self._history)

    def stats(self) -> dict[str, Any]:
        h = self._history
        return {
            "total_evaluations": len(h),
            "allowed": sum(1 for r in h if r.decision == GateDecision.ALLOW),
            "blocked": sum(1 for r in h if r.decision == GateDecision.BLOCK),
            "warned": sum(1 for r in h if r.decision == GateDecision.WARN),
            "block_rate": (
                sum(1 for r in h if r.blocked) / len(h) * 100 if h else 0
            ),
        }


# ============================================================================
# Pre-Commit Hook Generator
# ============================================================================


class PreCommitHook:
    """Generiert Git-Pre-Commit-Hooks für Security-Checks."""

    @staticmethod
    def generate_bash() -> str:
        return r"""#!/bin/bash
# Jarvis Security Pre-Commit Hook
# Installieren: cp pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "🔬 Jarvis Security-Check vor Commit..."

# 1. Credential-Scan
if grep -rn "api_key\|password\|secret.*=.*['\"]" --include="*.py" src/; then
    echo "❌ BLOCKED: Mögliche Credentials im Code gefunden!"
    exit 1
fi

# 2. Dependency-Check
python -m jarvis.security.mlops_pipeline --stages dependency_scan --trigger pre-commit
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Verwundbare Dependencies gefunden!"
    exit 1
fi

# 3. Prompt-Injection Patterns
if grep -rn "ignore.*previous\|system.*prompt\|override.*instructions" --include="*.py" src/; then
    echo "⚠️ WARNUNG: Prompt-Injection-Muster im Code!"
fi

echo "✅ Security-Check bestanden"
exit 0
"""

    @staticmethod
    def generate_yaml() -> str:
        """Pre-Commit YAML Konfiguration."""
        return """# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: jarvis-security
        name: Jarvis Security Gate
        entry: python -m jarvis.security.mlops_pipeline --trigger pre-commit
        language: python
        types: [python]
        stages: [commit]
      - id: jarvis-credential-scan
        name: Credential Scanner
        entry: python -c "from jarvis.security.hardening import CredentialScanner; CredentialScanner().scan_staged()"
        language: python
        types: [python]
        stages: [commit]
"""


# ============================================================================
# Container Isolation
# ============================================================================


class IsolationLevel(Enum):
    NONE = "none"
    PROCESS = "process"           # Eigener Prozess
    NAMESPACE = "namespace"       # Linux-Namespaces
    CONTAINER = "container"       # Docker/Podman
    VM = "vm"                     # Volle VM-Isolation


@dataclass
class AgentContainer:
    """Container-Konfiguration für einen isolierten Agenten."""

    agent_id: str
    isolation_level: IsolationLevel = IsolationLevel.NAMESPACE
    memory_limit_mb: int = 512
    cpu_limit_cores: float = 1.0
    network_enabled: bool = False  # Default: kein Netzwerk
    allowed_domains: list[str] = field(default_factory=list)
    filesystem_readonly: bool = True
    has_own_secrets: bool = True
    secrets_path: str = ""
    pid_namespace: bool = True
    tmpfs_size_mb: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "isolation": self.isolation_level.value,
            "memory_mb": self.memory_limit_mb,
            "cpu_cores": self.cpu_limit_cores,
            "network": self.network_enabled,
            "allowed_domains": self.allowed_domains,
            "readonly_fs": self.filesystem_readonly,
            "own_secrets": self.has_own_secrets,
        }

    def to_docker_args(self) -> list[str]:
        """Generiert Docker-Run-Argumente."""
        args = [
            f"--memory={self.memory_limit_mb}m",
            f"--cpus={self.cpu_limit_cores}",
            "--read-only" if self.filesystem_readonly else "",
            f"--tmpfs=/tmp:size={self.tmpfs_size_mb}m",
            "--pids-limit=100",
            "--security-opt=no-new-privileges",
        ]
        if not self.network_enabled:
            args.append("--network=none")
        if not self.pid_namespace:
            args.append("--pid=host")
        return [a for a in args if a]


class ContainerIsolation:
    """Verwaltet Container-Isolation pro Agent."""

    def __init__(self) -> None:
        self._containers: dict[str, AgentContainer] = {}

    def create(
        self,
        agent_id: str,
        *,
        isolation: IsolationLevel = IsolationLevel.NAMESPACE,
        memory_mb: int = 512,
        network: bool = False,
        allowed_domains: list[str] | None = None,
    ) -> AgentContainer:
        container = AgentContainer(
            agent_id=agent_id,
            isolation_level=isolation,
            memory_limit_mb=memory_mb,
            network_enabled=network,
            allowed_domains=allowed_domains or [],
            secrets_path=f"/run/secrets/{agent_id}",
        )
        self._containers[agent_id] = container
        return container

    def get(self, agent_id: str) -> AgentContainer | None:
        return self._containers.get(agent_id)

    def destroy(self, agent_id: str) -> bool:
        if agent_id in self._containers:
            del self._containers[agent_id]
            return True
        return False

    def all_containers(self) -> list[AgentContainer]:
        return list(self._containers.values())

    @property
    def count(self) -> int:
        return len(self._containers)

    def generate_compose(self) -> str:
        """Generiert Docker-Compose für alle Agenten."""
        services: list[str] = []
        for c in self._containers.values():
            svc = f"""  {c.agent_id}:
    image: jarvis-agent:latest
    mem_limit: {c.memory_limit_mb}m
    cpus: {c.cpu_limit_cores}
    read_only: {str(c.filesystem_readonly).lower()}
    security_opt:
      - no-new-privileges
    tmpfs:
      - /tmp:size={c.tmpfs_size_mb}m
    environment:
      - AGENT_ID={c.agent_id}
      - SECRETS_PATH={c.secrets_path}"""
            if not c.network_enabled:
                svc += "\n    network_mode: none"
            services.append(svc)

        return "version: '3.8'\nservices:\n" + "\n".join(services)

    def stats(self) -> dict[str, Any]:
        containers = list(self._containers.values())
        return {
            "total_containers": len(containers),
            "by_isolation": {
                level.value: sum(1 for c in containers if c.isolation_level == level)
                for level in IsolationLevel
                if any(c.isolation_level == level for c in containers)
            },
            "network_enabled": sum(1 for c in containers if c.network_enabled),
            "total_memory_mb": sum(c.memory_limit_mb for c in containers),
        }


# ============================================================================
# Credential Scanner
# ============================================================================


class CredentialScanner:
    """Scannt Code auf versehentlich eingebettete Credentials."""

    PATTERNS = [
        (r"api[_-]?key\s*=\s*['\"][^'\"]{8,}", "API-Key"),
        (r"password\s*=\s*['\"][^'\"]{4,}", "Passwort"),
        (r"secret\s*=\s*['\"][^'\"]{8,}", "Secret"),
        (r"token\s*=\s*['\"][^'\"]{8,}", "Token"),
        (r"-----BEGIN.*PRIVATE KEY-----", "Private-Key"),
        (r"sk-[a-zA-Z0-9]{20,}", "OpenAI-Key"),
        (r"anthropic_key\s*=", "Anthropic-Key"),
        (r"AKIA[0-9A-Z]{16}", "AWS-Key"),
    ]

    def scan_text(self, text: str) -> list[dict[str, str]]:
        """Scannt einen Text auf Credential-Patterns."""
        import re
        findings = []
        for pattern, cred_type in self.PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"type": cred_type, "pattern": pattern[:40]})
        return findings

    def scan_files(self, file_contents: dict[str, str]) -> dict[str, list[dict[str, str]]]:
        """Scannt mehrere Dateien."""
        results = {}
        for path, content in file_contents.items():
            findings = self.scan_text(content)
            if findings:
                results[path] = findings
        return results


# ============================================================================
# Webhook Notifier
# ============================================================================


@dataclass
class WebhookConfig:
    """Webhook-Konfiguration für Security-Events."""

    url: str
    events: list[str] = field(default_factory=lambda: ["gate_blocked", "critical_finding", "recall"])
    enabled: bool = True
    secret: str = ""
    timeout_seconds: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url[:50] + "..." if len(self.url) > 50 else self.url,
            "events": self.events,
            "enabled": self.enabled,
        }


class WebhookNotifier:
    """Sendet Benachrichtigungen bei Security-Events."""

    def __init__(self) -> None:
        self._webhooks: list[WebhookConfig] = []
        self._sent: list[dict[str, Any]] = []

    def add_webhook(self, config: WebhookConfig) -> None:
        self._webhooks.append(config)

    def remove_webhook(self, url: str) -> bool:
        before = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w.url != url]
        return len(self._webhooks) < before

    def notify(self, event: str, payload: dict[str, Any]) -> int:
        """Benachrichtigt alle registrierten Webhooks.

        Returns:
            Anzahl gesendeter Nachrichten.
        """
        sent = 0
        for webhook in self._webhooks:
            if not webhook.enabled:
                continue
            if event not in webhook.events:
                continue
            # In Produktion: HTTP POST an webhook.url
            notification = {
                "event": event,
                "url": webhook.url,
                "payload_keys": list(payload.keys()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._sent.append(notification)
            sent += 1
        return sent

    @property
    def webhook_count(self) -> int:
        return len(self._webhooks)

    @property
    def sent_count(self) -> int:
        return len(self._sent)

    def stats(self) -> dict[str, Any]:
        return {
            "webhooks": len(self._webhooks),
            "enabled": sum(1 for w in self._webhooks if w.enabled),
            "total_sent": len(self._sent),
        }


# ============================================================================
# Security Scan Scheduler
# ============================================================================


@dataclass
class ScheduledScan:
    """Geplanter Security-Scan."""

    scan_id: str
    name: str
    cron: str  # z.B. "0 3 * * 1" (Montags 3 Uhr)
    stages: list[str] = field(default_factory=list)
    enabled: bool = True
    last_run: str = ""
    next_run: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "name": self.name,
            "cron": self.cron,
            "stages": self.stages,
            "enabled": self.enabled,
            "last_run": self.last_run,
        }


class ScanScheduler:
    """Verwaltet zeitgesteuerte Security-Scans."""

    DEFAULT_SCHEDULES = [
        ScheduledScan("daily-quick", "Täglicher Quick-Scan", "0 2 * * *",
                       ["prompt_injection", "model_inversion"]),
        ScheduledScan("weekly-full", "Wöchentlicher Full-Scan", "0 3 * * 1",
                       ["adversarial_fuzzing", "prompt_injection", "model_inversion",
                        "dependency_scan", "memory_poisoning"]),
        ScheduledScan("monthly-pentest", "Monatlicher Penetration-Test", "0 4 1 * *",
                       ["adversarial_fuzzing", "prompt_injection", "model_inversion",
                        "data_exfiltration", "privilege_escalation", "denial_of_service",
                        "dependency_scan"]),
    ]

    def __init__(self, load_defaults: bool = True) -> None:
        self._schedules: dict[str, ScheduledScan] = {}
        if load_defaults:
            for s in self.DEFAULT_SCHEDULES:
                self._schedules[s.scan_id] = s

    def add(self, scan: ScheduledScan) -> None:
        self._schedules[scan.scan_id] = scan

    def remove(self, scan_id: str) -> bool:
        if scan_id in self._schedules:
            del self._schedules[scan_id]
            return True
        return False

    def get(self, scan_id: str) -> ScheduledScan | None:
        return self._schedules.get(scan_id)

    def all_schedules(self) -> list[ScheduledScan]:
        return list(self._schedules.values())

    def enabled_schedules(self) -> list[ScheduledScan]:
        return [s for s in self._schedules.values() if s.enabled]

    @property
    def count(self) -> int:
        return len(self._schedules)

    def stats(self) -> dict[str, Any]:
        schedules = list(self._schedules.values())
        return {
            "total": len(schedules),
            "enabled": sum(1 for s in schedules if s.enabled),
            "schedules": [s.to_dict() for s in schedules],
        }
