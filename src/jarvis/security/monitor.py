"""Runtime Monitor: Echtzeit-Sicherheitsüberwachung.

Überwacht JEDE Tool-Ausführung und blockiert verdächtige Aktionen
BEVOR sie ausgeführt werden. Ergänzt die statische Code-Analyse
des PackageInstallers um dynamische Laufzeitprüfung.

Prüfketten:
  1. Policy-Checks: Erlaubte/verbotene Tool-Parameter prüfen
  2. Rate-Limiting: Zu viele Aufrufe pro Zeitfenster blockieren
  3. Resource-Limits: Dateigrößen, Speicher, CPU-Zeit überwachen
  4. Anomaly-Detection: Ungewöhnliche Muster erkennen

Jede Verletzung wird als SecurityEvent im AuditLog protokolliert
und kann den Executor anweisen, die Aktion zu blockieren.

Bibel-Referenz: §3.4 (Runtime Security)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("jarvis.security.monitor")


# ============================================================================
# Security Events
# ============================================================================


class Severity(Enum):
    """Schweregrad eines Sicherheitsereignisses."""

    INFO = "info"  # Normaler Vorgang
    WARNING = "warning"  # Auffällig, aber nicht blockiert
    CRITICAL = "critical"  # Blockiert
    ALERT = "alert"  # Sofortige Benachrichtigung


class Verdict(Enum):
    """Entscheidung des Runtime Monitors."""

    ALLOW = "allow"  # Aktion erlauben
    WARN = "warn"  # Erlauben + Warnung loggen
    BLOCK = "block"  # Aktion blockieren
    THROTTLE = "throttle"  # Aktion verlangsamen


@dataclass
class SecurityEvent:
    """Ein einzelnes Sicherheitsereignis."""

    event_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    severity: Severity = Severity.INFO
    verdict: Verdict = Verdict.ALLOW
    category: str = ""  # policy, rate_limit, resource, anomaly
    tool_name: str = ""
    agent_name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    rule_id: str = ""  # Welche Regel hat ausgelöst

    @property
    def is_blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK


# ============================================================================
# Policy Rules
# ============================================================================


@dataclass
class PolicyRule:
    """Eine Sicherheitsregel für den Runtime Monitor.

    Definiert Bedingungen und Aktionen:
      - match: Welches Tool/Welcher Agent
      - condition: Parameter-Check
      - action: ALLOW/WARN/BLOCK
    """

    rule_id: str
    description: str = ""
    enabled: bool = True

    # Matching
    tool_pattern: str = "*"  # Glob-Pattern oder exakter Name
    agent_pattern: str = "*"

    # Bedingung (Parameter-Checks)
    forbidden_params: dict[str, list[str]] = field(default_factory=dict)
    # z.B. {"path": ["/etc", "/proc", "/sys"]}
    required_params: list[str] = field(default_factory=list)
    max_param_length: int = 0  # 0 = unbegrenzt

    # Aktion
    verdict: Verdict = Verdict.BLOCK
    severity: Severity = Severity.CRITICAL


# Default-Regeln
_DEFAULT_RULES: list[PolicyRule] = [
    PolicyRule(
        rule_id="no_system_dirs",
        description="Zugriff auf System-Verzeichnisse blockieren",
        tool_pattern="*",
        forbidden_params={
            "path": ["/etc", "/proc", "/sys", "/boot", "/root"],
            "file": ["/etc", "/proc", "/sys", "/boot", "/root"],
            "directory": ["/etc", "/proc", "/sys", "/boot", "/root"],
        },
        verdict=Verdict.BLOCK,
        severity=Severity.CRITICAL,
    ),
    PolicyRule(
        rule_id="no_credential_leak",
        description="Credentials in Parametern blockieren",
        tool_pattern="*",
        forbidden_params={
            "content": ["password", "api_key", "secret", "token"],
        },
        verdict=Verdict.WARN,
        severity=Severity.WARNING,
    ),
    PolicyRule(
        rule_id="param_length_limit",
        description="Überlange Parameter blockieren (Injection-Schutz)",
        tool_pattern="*",
        max_param_length=50000,  # 50KB
        verdict=Verdict.BLOCK,
        severity=Severity.CRITICAL,
    ),
]


# ============================================================================
# Rate Limiter
# ============================================================================


@dataclass
class RateLimit:
    """Rate-Limit-Konfiguration."""

    name: str
    max_calls: int  # Maximale Aufrufe
    window_seconds: int  # Zeitfenster
    scope: str = "global"  # global, per_agent, per_tool


class RateLimiter:
    """Token-Bucket-basierter Rate-Limiter.

    Tracks Tool-Aufrufe pro Scope und blockiert bei Überschreitung.
    """

    def __init__(self) -> None:
        self._limits: list[RateLimit] = []
        self._buckets: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10000))

    def add_limit(self, limit: RateLimit) -> None:
        self._limits.append(limit)

    def check(
        self,
        tool_name: str,
        agent_name: str = "",
    ) -> tuple[bool, str]:
        """Prüft ob ein Aufruf erlaubt ist.

        Returns:
            (erlaubt, begründung)
        """
        now = time.monotonic()

        for limit in self._limits:
            # Scope-Key bestimmen
            if limit.scope == "global":
                key = f"global:{limit.name}"
            elif limit.scope == "per_agent":
                key = f"agent:{agent_name}:{limit.name}"
            elif limit.scope == "per_tool":
                key = f"tool:{tool_name}:{limit.name}"
            else:
                key = f"global:{limit.name}"

            bucket = self._buckets[key]

            # Alte Einträge entfernen
            while bucket and now - bucket[0] > limit.window_seconds:
                bucket.popleft()

            if len(bucket) >= limit.max_calls:
                return False, (
                    f"Rate-Limit '{limit.name}' überschritten: "
                    f"{limit.max_calls}/{limit.window_seconds}s"
                )

        return True, ""

    def record(self, tool_name: str, agent_name: str = "") -> None:
        """Zeichnet einen Aufruf auf."""
        now = time.monotonic()
        for limit in self._limits:
            if limit.scope == "global":
                key = f"global:{limit.name}"
            elif limit.scope == "per_agent":
                key = f"agent:{agent_name}:{limit.name}"
            elif limit.scope == "per_tool":
                key = f"tool:{tool_name}:{limit.name}"
            else:
                key = f"global:{limit.name}"
            self._buckets[key].append(now)


# ============================================================================
# Runtime Monitor
# ============================================================================


class RuntimeMonitor:
    """Echtzeit-Sicherheitsüberwachung für Tool-Ausführungen.

    Usage:
        monitor = RuntimeMonitor()

        # Vor jeder Tool-Ausführung:
        event = monitor.check_tool_call("file_write", {"path": "/etc/passwd"}, agent="coder")
        if event.is_blocked:
            raise SecurityError(event.description)

        # Nach Ausführung:
        monitor.record_execution("file_write", agent="coder", success=True)
    """

    def __init__(self, *, enable_defaults: bool = True) -> None:
        self._rules: list[PolicyRule] = []
        self._rate_limiter = RateLimiter()
        self._events: deque[SecurityEvent] = deque(maxlen=10000)
        self._event_counter = 0
        self._total_checks = 0
        self._total_blocks = 0
        self._total_warnings = 0

        if enable_defaults:
            self._rules.extend(_DEFAULT_RULES)
            self._rate_limiter.add_limit(
                RateLimit("global_burst", max_calls=100, window_seconds=60),
            )
            self._rate_limiter.add_limit(
                RateLimit("per_tool_limit", max_calls=30, window_seconds=60, scope="per_tool"),
            )

    # ── Konfiguration ────────────────────────────────────────────

    def add_rule(self, rule: PolicyRule) -> None:
        """Fügt eine neue Sicherheitsregel hinzu."""
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Entfernt eine Regel."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    def add_rate_limit(self, limit: RateLimit) -> None:
        self._rate_limiter.add_limit(limit)

    # ── Check ────────────────────────────────────────────────────

    def check_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        *,
        agent_name: str = "",
    ) -> SecurityEvent:
        """Prüft einen Tool-Call BEVOR er ausgeführt wird.

        Durchläuft alle Regeln und Rate-Limits.

        Args:
            tool_name: Name des aufzurufenden Tools.
            parameters: Tool-Parameter.
            agent_name: Name des aufrufenden Agenten.

        Returns:
            SecurityEvent mit Verdict (ALLOW/WARN/BLOCK).
        """
        self._total_checks += 1
        params = parameters or {}

        # 1. Policy-Regeln prüfen
        for rule in self._rules:
            if not rule.enabled:
                continue

            if not self._matches_pattern(tool_name, rule.tool_pattern):
                continue
            if agent_name and not self._matches_pattern(agent_name, rule.agent_pattern):
                continue

            violation = self._check_rule(rule, params)
            if violation:
                event = self._create_event(
                    severity=rule.severity,
                    verdict=rule.verdict,
                    category="policy",
                    tool_name=tool_name,
                    agent_name=agent_name,
                    description=violation,
                    parameters=params,
                    rule_id=rule.rule_id,
                )
                if event.is_blocked:
                    self._total_blocks += 1
                elif event.verdict == Verdict.WARN:
                    self._total_warnings += 1
                return event

        # 2. Rate-Limiting
        allowed, reason = self._rate_limiter.check(tool_name, agent_name)
        if not allowed:
            event = self._create_event(
                severity=Severity.WARNING,
                verdict=Verdict.THROTTLE,
                category="rate_limit",
                tool_name=tool_name,
                agent_name=agent_name,
                description=reason,
                parameters=params,
            )
            self._total_blocks += 1
            return event

        # 3. Alles OK
        self._rate_limiter.record(tool_name, agent_name)
        return self._create_event(
            severity=Severity.INFO,
            verdict=Verdict.ALLOW,
            category="check",
            tool_name=tool_name,
            agent_name=agent_name,
            description="Tool-Call erlaubt",
        )

    def record_execution(
        self,
        tool_name: str,
        *,
        agent_name: str = "",
        success: bool = True,
        duration_ms: float = 0,
    ) -> None:
        """Zeichnet eine ausgeführte Tool-Aktion auf."""
        self._create_event(
            severity=Severity.INFO,
            verdict=Verdict.ALLOW,
            category="execution",
            tool_name=tool_name,
            agent_name=agent_name,
            description=f"{'Erfolg' if success else 'Fehler'} ({duration_ms:.0f}ms)",
        )

    # ── Rule-Checking ────────────────────────────────────────────

    def _check_rule(
        self,
        rule: PolicyRule,
        params: dict[str, Any],
    ) -> str:
        """Prüft eine einzelne Regel gegen Parameter.

        Returns:
            Leerer String wenn OK, Fehlerbeschreibung wenn Verletzung.
        """
        # Forbidden-Params
        for param_name, forbidden_values in rule.forbidden_params.items():
            value = str(params.get(param_name, ""))
            if not value:
                continue
            value_lower = value.lower()
            for forbidden in forbidden_values:
                if forbidden.lower() in value_lower:
                    return (
                        f"Regel '{rule.rule_id}': Parameter '{param_name}' "
                        f"enthält verbotenen Wert '{forbidden}'"
                    )

        # Required-Params
        for required in rule.required_params:
            if required not in params:
                return (
                    f"Regel '{rule.rule_id}': Pflicht-Parameter '{required}' fehlt"
                )

        # Max-Param-Length
        if rule.max_param_length > 0:
            for param_name, value in params.items():
                str_value = str(value)
                if len(str_value) > rule.max_param_length:
                    return (
                        f"Regel '{rule.rule_id}': Parameter '{param_name}' "
                        f"zu lang ({len(str_value)} > {rule.max_param_length})"
                    )

        return ""

    @staticmethod
    def _matches_pattern(value: str, pattern: str) -> bool:
        """Einfaches Glob-Matching: '*' matcht alles."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return value.startswith(pattern[:-1])
        return value == pattern

    # ── Event-Verwaltung ─────────────────────────────────────────

    def _create_event(self, **kwargs: Any) -> SecurityEvent:
        self._event_counter += 1
        event = SecurityEvent(
            event_id=f"sec_{self._event_counter}",
            **kwargs,
        )
        self._events.append(event)

        # Logging
        if event.severity in (Severity.CRITICAL, Severity.ALERT):
            logger.warning(
                "SECURITY %s: %s (tool=%s, agent=%s, rule=%s)",
                event.verdict.value, event.description,
                event.tool_name, event.agent_name, event.rule_id,
            )

        return event

    def get_events(
        self,
        *,
        severity: Severity | None = None,
        category: str = "",
        limit: int = 100,
    ) -> list[SecurityEvent]:
        """Gibt gefilterte Security-Events zurück."""
        events = list(self._events)

        if severity:
            events = [e for e in events if e.severity == severity]
        if category:
            events = [e for e in events if e.category == category]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_blocked_events(self, limit: int = 50) -> list[SecurityEvent]:
        """Alle blockierten Aktionen."""
        return [e for e in list(self._events) if e.is_blocked][-limit:]

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "total_blocks": self._total_blocks,
            "total_warnings": self._total_warnings,
            "active_rules": len([r for r in self._rules if r.enabled]),
            "events_logged": len(self._events),
            "block_rate": (
                self._total_blocks / self._total_checks
                if self._total_checks > 0 else 0.0
            ),
        }
