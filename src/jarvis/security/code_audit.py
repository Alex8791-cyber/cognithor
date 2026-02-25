"""Jarvis · Automatisierte Code-Analyse für Skills.

Statische Analyse zur Erkennung von Sicherheitsrisiken in Skills:

  - CodePattern:           Erkennbare Muster (gefährliche Aufrufe, Exfiltration...)
  - PatternScanner:        AST-basierte Mustererkennung
  - PermissionAnalyzer:    Analysiert benötigte Berechtigungen
  - DependencyAuditor:     Prüft Abhängigkeiten auf bekannte Schwachstellen
  - SkillSecurityReport:   Zusammenfassung aller Findings
  - CodeAuditor:           Hauptklasse

Architektur-Bibel: §15.3 (Supply-Chain-Security), §14.6 (Code-Audit)
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Code Patterns
# ============================================================================


class PatternSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PatternCategory(Enum):
    DANGEROUS_CALL = "dangerous_call"
    DATA_EXFILTRATION = "data_exfiltration"
    CREDENTIAL_ACCESS = "credential_access"
    FILE_SYSTEM = "file_system"
    NETWORK_ACCESS = "network_access"
    CODE_EXECUTION = "code_execution"
    OBFUSCATION = "obfuscation"
    PERMISSION_ABUSE = "permission_abuse"
    INJECTION_RISK = "injection_risk"


@dataclass
class CodePattern:
    """Ein erkennbares Sicherheitsmuster im Code."""

    pattern_id: str
    name: str
    category: PatternCategory
    severity: PatternSeverity
    description: str
    detector: str = "regex"      # regex, ast, heuristic
    pattern: str = ""            # Regex oder AST-Node-Type
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pattern_id,
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
        }


@dataclass
class CodeFinding:
    """Ein konkreter Fund in einer Code-Datei."""

    finding_id: str
    pattern: CodePattern
    file_path: str
    line_number: int
    code_snippet: str
    confidence: float = 1.0   # 0-1
    false_positive: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "pattern": self.pattern.name,
            "severity": self.pattern.severity.value,
            "file": self.file_path,
            "line": self.line_number,
            "confidence": self.confidence,
        }


# ============================================================================
# Pattern Scanner
# ============================================================================


class PatternScanner:
    """AST- und Regex-basierte Mustererkennung in Python-Code."""

    BUILT_IN_PATTERNS = [
        # Kritisch: Code-Ausführung
        CodePattern("CP-001", "eval() Aufruf", PatternCategory.CODE_EXECUTION,
                    PatternSeverity.CRITICAL, "eval() erlaubt beliebige Code-Ausführung",
                    "ast", "eval", "Ersetze eval() durch ast.literal_eval() oder sichere Alternative"),
        CodePattern("CP-002", "exec() Aufruf", PatternCategory.CODE_EXECUTION,
                    PatternSeverity.CRITICAL, "exec() erlaubt beliebige Code-Ausführung",
                    "ast", "exec", "Entferne exec() und nutze sichere Alternativen"),
        CodePattern("CP-003", "subprocess Shell=True", PatternCategory.CODE_EXECUTION,
                    PatternSeverity.CRITICAL, "Shell-Injection möglich bei shell=True",
                    "regex", r"subprocess\.\w+\(.*shell\s*=\s*True", "Verwende shell=False + Argument-Liste"),
        CodePattern("CP-004", "os.system() Aufruf", PatternCategory.CODE_EXECUTION,
                    PatternSeverity.HIGH, "Direkte Shell-Ausführung",
                    "regex", r"os\.system\(", "Verwende subprocess.run() mit shell=False"),

        # Hoch: Netzwerk/Exfiltration
        CodePattern("CP-005", "HTTP-Request an externe URL", PatternCategory.NETWORK_ACCESS,
                    PatternSeverity.HIGH, "Skill macht HTTP-Requests nach außen",
                    "regex", r"requests\.\w+\(|urllib\.request|httpx\.\w+\(", "Netzwerkzugriff muss genehmigt werden"),
        CodePattern("CP-006", "Socket-Nutzung", PatternCategory.NETWORK_ACCESS,
                    PatternSeverity.HIGH, "Direkter Socket-Zugriff",
                    "regex", r"socket\.socket\(|socket\.connect", "Verwende genehmigte HTTP-Bibliotheken"),
        CodePattern("CP-007", "DNS-Exfiltration", PatternCategory.DATA_EXFILTRATION,
                    PatternSeverity.CRITICAL, "DNS-Anfragen können zur Datenexfiltration genutzt werden",
                    "regex", r"dns\.resolver|socket\.getaddrinfo.*encode", "DNS-Zugriff einschränken"),

        # Mittel: Dateisystem
        CodePattern("CP-008", "Sensible Pfade lesen", PatternCategory.FILE_SYSTEM,
                    PatternSeverity.MEDIUM, "Zugriff auf sensible Dateipfade",
                    "regex", r"/etc/passwd|/etc/shadow|\.ssh/|\.env|\.aws/credentials",
                    "Dateizugriff auf Sandbox beschränken"),
        CodePattern("CP-009", "Temporäre Datei mit Secrets", PatternCategory.CREDENTIAL_ACCESS,
                    PatternSeverity.MEDIUM, "Credentials in temporäre Dateien geschrieben",
                    "regex", r"(tempfile|/tmp).*(?:key|secret|password|token)",
                    "Secrets nur im Vault speichern"),

        # Obfuskation
        CodePattern("CP-010", "Base64-Kodierung", PatternCategory.OBFUSCATION,
                    PatternSeverity.LOW, "Base64-Kodierung kann bösartigen Code verstecken",
                    "regex", r"base64\.\w*decode\(", "Prüfe was dekodiert wird"),
        CodePattern("CP-011", "Compile/Marshal", PatternCategory.OBFUSCATION,
                    PatternSeverity.HIGH, "Dynamische Code-Kompilierung",
                    "regex", r"compile\(|marshal\.loads\(|pickle\.loads\(",
                    "Keine dynamische Code-Kompilierung erlaubt"),

        # Injection
        CodePattern("CP-012", "SQL-Injection-Risiko", PatternCategory.INJECTION_RISK,
                    PatternSeverity.HIGH, "String-Formatierung in SQL-Queries",
                    "regex", r"(execute|cursor)\(.*(%s|\.format\(|f['\"])",
                    "Verwende parametrisierte Queries"),
    ]

    def __init__(self, load_defaults: bool = True) -> None:
        self._patterns: list[CodePattern] = []
        self._findings: list[CodeFinding] = []
        self._counter = 0
        if load_defaults:
            self._patterns = list(self.BUILT_IN_PATTERNS)

    def add_pattern(self, pattern: CodePattern) -> None:
        self._patterns.append(pattern)

    def scan_code(self, code: str, file_path: str = "<stdin>") -> list[CodeFinding]:
        """Scannt Python-Code auf alle registrierten Muster."""
        findings: list[CodeFinding] = []

        for pattern in self._patterns:
            if pattern.detector == "regex" and pattern.pattern:
                findings.extend(self._scan_regex(code, pattern, file_path))
            elif pattern.detector == "ast":
                findings.extend(self._scan_ast(code, pattern, file_path))

        self._findings.extend(findings)
        return findings

    def _scan_regex(self, code: str, pattern: CodePattern, file_path: str) -> list[CodeFinding]:
        findings = []
        for i, line in enumerate(code.split("\n"), 1):
            if re.search(pattern.pattern, line, re.IGNORECASE):
                self._counter += 1
                findings.append(CodeFinding(
                    finding_id=f"CF-{self._counter:04d}",
                    pattern=pattern,
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()[:120],
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ))
        return findings

    def _scan_ast(self, code: str, pattern: CodePattern, file_path: str) -> list[CodeFinding]:
        findings = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == pattern.pattern:
                    self._counter += 1
                    findings.append(CodeFinding(
                        finding_id=f"CF-{self._counter:04d}",
                        pattern=pattern,
                        file_path=file_path,
                        line_number=getattr(node, "lineno", 0),
                        code_snippet=f"{func_name}(...)",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ))
        return findings

    @property
    def finding_count(self) -> int:
        return len(self._findings)

    def findings_by_severity(self, severity: PatternSeverity) -> list[CodeFinding]:
        return [f for f in self._findings if f.pattern.severity == severity]

    def stats(self) -> dict[str, Any]:
        return {
            "patterns_loaded": len(self._patterns),
            "total_findings": len(self._findings),
            "by_severity": {
                s.value: sum(1 for f in self._findings if f.pattern.severity == s)
                for s in PatternSeverity
                if any(f.pattern.severity == s for f in self._findings)
            },
            "by_category": {
                c.value: sum(1 for f in self._findings if f.pattern.category == c)
                for c in PatternCategory
                if any(f.pattern.category == c for f in self._findings)
            },
        }


# ============================================================================
# Permission Analyzer
# ============================================================================


class RequiredPermission(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK = "network"
    SHELL_EXEC = "shell_exec"
    DATABASE = "database"
    SECRETS = "secrets"
    SYSTEM_INFO = "system_info"
    EXTERNAL_API = "external_api"


class PermissionAnalyzer:
    """Analysiert welche Berechtigungen ein Skill benötigt."""

    PERMISSION_INDICATORS: dict[RequiredPermission, list[str]] = {
        RequiredPermission.FILE_READ: ["open(", "Path(", "os.path", "read()", "readlines()"],
        RequiredPermission.FILE_WRITE: ["write(", "writelines(", "os.mkdir", "shutil.copy"],
        RequiredPermission.NETWORK: ["requests.", "httpx.", "urllib.", "socket."],
        RequiredPermission.SHELL_EXEC: ["subprocess.", "os.system(", "os.popen("],
        RequiredPermission.DATABASE: ["sqlite3.", "psycopg", "pymongo.", "sqlalchemy."],
        RequiredPermission.SECRETS: ["os.environ", "dotenv", "keyring.", "vault."],
        RequiredPermission.SYSTEM_INFO: ["platform.", "psutil.", "os.uname", "sys.platform"],
        RequiredPermission.EXTERNAL_API: ["api_key", "API_KEY", "authorization:", "Bearer "],
    }

    def analyze(self, code: str) -> dict[RequiredPermission, list[str]]:
        """Analysiert benötigte Berechtigungen."""
        required: dict[RequiredPermission, list[str]] = {}
        for perm, indicators in self.PERMISSION_INDICATORS.items():
            evidence = []
            for indicator in indicators:
                if indicator in code:
                    evidence.append(indicator)
            if evidence:
                required[perm] = evidence
        return required

    def risk_assessment(self, permissions: dict[RequiredPermission, list[str]]) -> dict[str, Any]:
        """Bewertet das Risiko basierend auf benötigten Berechtigungen."""
        high_risk = {RequiredPermission.SHELL_EXEC, RequiredPermission.NETWORK, RequiredPermission.SECRETS}
        medium_risk = {RequiredPermission.DATABASE, RequiredPermission.EXTERNAL_API}

        risk_perms = set(permissions.keys())
        critical = risk_perms & high_risk
        medium = risk_perms & medium_risk

        if critical:
            level = "high"
        elif medium:
            level = "medium"
        elif permissions:
            level = "low"
        else:
            level = "none"

        return {
            "risk_level": level,
            "total_permissions": len(permissions),
            "high_risk_permissions": [p.value for p in critical],
            "medium_risk_permissions": [p.value for p in medium],
            "all_permissions": [p.value for p in permissions.keys()],
        }


# ============================================================================
# Skill Security Report
# ============================================================================


@dataclass
class SkillSecurityReport:
    """Zusammenfassender Sicherheitsbericht für einen Skill."""

    report_id: str
    skill_name: str
    file_path: str
    code_lines: int
    scan_timestamp: str
    findings: list[CodeFinding] = field(default_factory=list)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    permission_risk: str = "none"
    overall_risk: str = "low"
    passed: bool = True
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "skill": self.skill_name,
            "lines": self.code_lines,
            "findings": len(self.findings),
            "critical": sum(1 for f in self.findings if f.pattern.severity == PatternSeverity.CRITICAL),
            "high": sum(1 for f in self.findings if f.pattern.severity == PatternSeverity.HIGH),
            "permission_risk": self.permission_risk,
            "overall_risk": self.overall_risk,
            "passed": self.passed,
        }


# ============================================================================
# Code Auditor (Hauptklasse)
# ============================================================================


class CodeAuditor:
    """Hauptklasse: Automatisierte Code-Analyse für Skills."""

    def __init__(self) -> None:
        self._scanner = PatternScanner()
        self._perm_analyzer = PermissionAnalyzer()
        self._reports: list[SkillSecurityReport] = []
        self._counter = 0

    @property
    def scanner(self) -> PatternScanner:
        return self._scanner

    @property
    def perm_analyzer(self) -> PermissionAnalyzer:
        return self._perm_analyzer

    def audit_skill(self, skill_name: str, code: str, file_path: str = "") -> SkillSecurityReport:
        """Vollständiger Sicherheits-Audit eines Skills."""
        self._counter += 1

        # 1. Pattern-Scan
        findings = self._scanner.scan_code(code, file_path or f"{skill_name}.py")

        # 2. Berechtigungs-Analyse
        perms = self._perm_analyzer.analyze(code)
        perm_risk = self._perm_analyzer.risk_assessment(perms)

        # 3. Gesamt-Risiko
        critical_count = sum(1 for f in findings if f.pattern.severity == PatternSeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.pattern.severity == PatternSeverity.HIGH)

        if critical_count > 0:
            overall = "critical"
        elif high_count > 0 or perm_risk["risk_level"] == "high":
            overall = "high"
        elif findings or perm_risk["risk_level"] == "medium":
            overall = "medium"
        else:
            overall = "low"

        passed = critical_count == 0 and high_count <= 2

        # 4. Empfehlungen
        recommendations = []
        if critical_count > 0:
            recommendations.append(f"❌ {critical_count} kritische Findings beheben (Pflicht)")
        if high_count > 0:
            recommendations.append(f"⚠️ {high_count} High-Findings prüfen")
        for f in findings:
            if f.pattern.recommendation and f.pattern.recommendation not in recommendations:
                recommendations.append(f.pattern.recommendation)

        report = SkillSecurityReport(
            report_id=f"SAR-{self._counter:04d}",
            skill_name=skill_name,
            file_path=file_path,
            code_lines=len(code.split("\n")),
            scan_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            findings=findings,
            permissions={p.value: ev for p, ev in perms.items()},
            permission_risk=perm_risk["risk_level"],
            overall_risk=overall,
            passed=passed,
            recommendations=recommendations,
        )
        self._reports.append(report)
        return report

    def all_reports(self) -> list[SkillSecurityReport]:
        return list(self._reports)

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def pass_rate(self) -> float:
        if not self._reports:
            return 0.0
        return round(sum(1 for r in self._reports if r.passed) / len(self._reports) * 100, 1)

    def stats(self) -> dict[str, Any]:
        reports = self._reports
        return {
            "total_audits": len(reports),
            "pass_rate": self.pass_rate(),
            "total_findings": sum(len(r.findings) for r in reports),
            "scanner": self._scanner.stats(),
        }
