"""Jarvis · Red-Team-Testing Framework (Deprecation-Shim).

Dieses Modul existiert nur noch für Rückwärtskompatibilität.
Alle Klassen und Funktionen leben jetzt in ``jarvis.security.red_team``.

Beim Import wird eine ``DeprecationWarning`` ausgegeben.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "jarvis.security.redteam ist veraltet -- bitte jarvis.security.red_team verwenden.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module
from jarvis.security.red_team import (  # noqa: E402, F401
    AttackCategory,
    AttackPayload,
    AttackPlaybook,
    AttackResult,
    AttackSeverity,
    AttackVector,
    CICDGenerator,
    CICDPlatform,
    JailbreakSimulator,
    MemoryPoisonSimulator,
    PenetrationSuite,
    PoisonPayload,
    PromptFuzzer,
    PromptInjectionTester,
    RedTeamFramework,
    RedTeamReport,
    RedTeamRunner,
    ScanPolicy,
    ScanResult,
    SecurityFinding,
    SecurityScanner,
    Severity,
    TestResult,
    VulnerabilityReport,
)

__all__ = [
    "AttackCategory",
    "AttackPayload",
    "AttackPlaybook",
    "AttackResult",
    "AttackSeverity",
    "AttackVector",
    "CICDGenerator",
    "CICDPlatform",
    "JailbreakSimulator",
    "MemoryPoisonSimulator",
    "PenetrationSuite",
    "PoisonPayload",
    "PromptFuzzer",
    "PromptInjectionTester",
    "RedTeamFramework",
    "RedTeamReport",
    "RedTeamRunner",
    "ScanPolicy",
    "ScanResult",
    "SecurityFinding",
    "SecurityScanner",
    "Severity",
    "TestResult",
    "VulnerabilityReport",
]
