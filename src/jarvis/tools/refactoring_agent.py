"""Autonomer Refactoring-Agent: Orchestriert Code- und Architektur-Analyse.

Kein automatisches Refactoring — nur Report (Mensch entscheidet).
Standalone ausfuehrbar: python -m jarvis.tools.refactoring_agent src/jarvis/
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.models import CodeSmell, ArchitectureFinding, RefactoringReport
from jarvis.tools.code_analyzer import CodeSmellDetector
from jarvis.tools.architecture_analyzer import ArchitectureAnalyzer
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class RefactoringAgent:
    """Orchestriert CodeSmellDetector + ArchitectureAnalyzer."""

    def __init__(
        self,
        code_detector: CodeSmellDetector | None = None,
        arch_analyzer: ArchitectureAnalyzer | None = None,
    ) -> None:
        self._code_detector = code_detector or CodeSmellDetector()
        self._arch_analyzer = arch_analyzer or ArchitectureAnalyzer()

    def run_analysis(self, src_dir: str | Path) -> RefactoringReport:
        """Fuehrt eine vollstaendige Analyse durch.

        Args:
            src_dir: Quellverzeichnis (z.B. src/jarvis/).

        Returns:
            RefactoringReport mit allen Findings.
        """
        src_dir = Path(src_dir)
        log.info("refactoring_analysis_start", src_dir=str(src_dir))

        # Code-Smells
        code_smells = self._code_detector.analyze_directory(src_dir, recursive=True)

        # Architektur-Analyse
        module_count = self._arch_analyzer.build_import_graph(src_dir)
        circular = self._arch_analyzer.detect_circular_imports()
        layer_violations = self._arch_analyzer.detect_layer_violations()
        arch_findings = circular + layer_violations

        # Datei-Zaehlung
        total_files = sum(1 for _ in src_dir.rglob("*.py") if "__pycache__" not in str(_))

        # Summary erstellen
        summary_parts: list[str] = []
        summary_parts.append(f"Analysiert: {total_files} Dateien, {module_count} Module")

        if code_smells:
            by_type: dict[str, int] = {}
            for smell in code_smells:
                by_type[smell.smell_type] = by_type.get(smell.smell_type, 0) + 1
            smell_summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
            summary_parts.append(f"Code-Smells: {len(code_smells)} ({smell_summary})")
        else:
            summary_parts.append("Code-Smells: Keine gefunden")

        if arch_findings:
            summary_parts.append(f"Architektur-Findings: {len(arch_findings)}")
        else:
            summary_parts.append("Architektur: Keine Probleme gefunden")

        summary = ". ".join(summary_parts)

        report = RefactoringReport(
            code_smells=code_smells,
            architecture_findings=arch_findings,
            summary=summary,
            total_files_analyzed=total_files,
        )

        log.info(
            "refactoring_analysis_complete",
            files=total_files,
            smells=len(code_smells),
            arch_findings=len(arch_findings),
        )

        return report

    def get_priority_issues(self, report: RefactoringReport, n: int = 10) -> list[dict[str, Any]]:
        """Top-N Issues nach Severity sortiert.

        Severity-Ranking: error > warning > info
        """
        severity_order = {"error": 0, "warning": 1, "info": 2}

        issues: list[dict[str, Any]] = []

        for smell in report.code_smells:
            issues.append({
                "type": "code_smell",
                "subtype": smell.smell_type,
                "severity": smell.severity,
                "file": smell.file_path,
                "line": smell.line,
                "message": smell.message,
                "suggestion": smell.suggestion,
                "sort_key": severity_order.get(smell.severity, 9),
            })

        for finding in report.architecture_findings:
            issues.append({
                "type": "architecture",
                "subtype": finding.finding_type,
                "severity": finding.severity,
                "file": "",
                "line": 0,
                "message": finding.message,
                "suggestion": "",
                "sort_key": severity_order.get(finding.severity, 9),
            })

        issues.sort(key=lambda x: x["sort_key"])
        return issues[:n]


# Standalone-Ausfuehrung
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m jarvis.tools.refactoring_agent <src_dir>")
        sys.exit(1)

    src = sys.argv[1]
    agent = RefactoringAgent()
    report = agent.run_analysis(src)

    print(f"\n=== Refactoring Report ===")
    print(f"Summary: {report.summary}")
    print(f"\nTop Issues:")
    for issue in agent.get_priority_issues(report):
        print(f"  [{issue['severity']}] {issue['subtype']}: {issue['message']}")
        if issue['suggestion']:
            print(f"    → {issue['suggestion']}")
