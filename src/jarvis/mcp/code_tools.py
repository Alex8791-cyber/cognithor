"""Code-Tools für Jarvis -- Python-REPL und Code-Analyse als MCP-Tools.

Zwei Tools:
  - run_python: Führt Python-Code in der Sandbox aus (temp-Datei + SandboxExecutor)
  - analyze_code: Kombiniert CodeSmellDetector + CodeAuditor für strukturierte Analyse

Factory: register_code_tools(mcp_client, config) → CodeTools

Bibel-Referenz: §5.3 (MCP-Tools), §4.3 (Sandbox)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.sandbox import (
    NetworkPolicy,
    SandboxConfig,
    SandboxExecutor,
    SandboxLevel,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# Maximale Code-Groesse (1 MB)
MAX_CODE_SIZE = 1_048_576

__all__ = [
    "CodeTools",
    "register_code_tools",
]


class CodeTools:
    """Python-REPL und Code-Analyse mit Sandbox-Isolation.

    Security-Architektur:
      - run_python: Code wird als temp-Datei geschrieben und via SandboxExecutor ausgeführt
      - analyze_code: Read-only, risikolos (AST-basiert)
      - Workspace-Confinement: Alle temp-Dateien im Workspace
    """

    def __init__(self, config: "JarvisConfig") -> None:
        self._config = config

        # Sandbox-Konfiguration (gleich wie ShellTools)
        sandbox_config = SandboxConfig(
            workspace_dir=config.workspace_dir,
            default_timeout=60,
        )

        sandbox_level = getattr(config, "sandbox_level", "bwrap")
        if sandbox_level in ("bwrap", "firejail", "bare"):
            sandbox_config.preferred_level = SandboxLevel(sandbox_level)

        sandbox_network = getattr(config, "sandbox_network", "allow")
        if sandbox_network in ("allow", "block"):
            sandbox_config.network = NetworkPolicy(sandbox_network)

        self._sandbox = SandboxExecutor(sandbox_config)
        self._workspace = config.workspace_dir

        log.info(
            "code_tools_init",
            sandbox_level=self._sandbox.level.value,
            workspace=str(self._workspace),
        )

    async def run_python(
        self,
        code: str,
        timeout: int = 60,
        working_dir: str | None = None,
    ) -> str:
        """Führt Python-Code in der Sandbox aus.

        Schreibt den Code in eine temporäre .py-Datei im Workspace,
        führt sie über den SandboxExecutor aus und bereinigt anschließend.

        Args:
            code: Python-Code als String.
            timeout: Timeout in Sekunden (Default: 60).
            working_dir: Arbeitsverzeichnis (Default: Workspace).

        Returns:
            Kombinierter stdout + stderr Output.
        """
        if not code.strip():
            return "Kein Code angegeben."

        code_size = len(code.encode("utf-8"))
        if code_size > MAX_CODE_SIZE:
            return f"Code zu gross ({code_size / 1_048_576:.1f} MB, max 1 MB)"

        cwd = working_dir or str(self._workspace)

        # Working-Directory validieren -- muss unter Workspace liegen
        cwd_path = Path(cwd).expanduser().resolve()
        workspace_root = self._workspace.expanduser().resolve()
        try:
            cwd_path.relative_to(workspace_root)
        except ValueError:
            return (
                f"Zugriff verweigert: Arbeitsverzeichnis '{cwd}' liegt ausserhalb "
                f"des Workspace ({workspace_root})"
            )
        cwd_path.mkdir(parents=True, exist_ok=True)

        # Temp-Datei erstellen
        temp_name = f"_jarvis_run_{uuid.uuid4().hex[:12]}.py"
        temp_path = cwd_path / temp_name

        try:
            # Code in temp-Datei schreiben
            temp_path.write_text(code, encoding="utf-8")

            log.info(
                "run_python_start",
                temp_file=str(temp_path),
                code_lines=code.count("\n") + 1,
                timeout=timeout,
            )

            # Via Sandbox ausführen
            command = f"python {temp_name}"
            result = await self._sandbox.execute(
                command,
                working_dir=str(cwd_path),
                timeout=timeout,
            )

            log.info(
                "run_python_done",
                exit_code=result.exit_code,
                sandbox=result.sandbox_level,
                stdout_len=len(result.stdout),
                stderr_len=len(result.stderr),
                timed_out=result.timed_out,
            )

            return result.output

        finally:
            # Temp-Datei bereinigen
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    async def analyze_code(
        self,
        code: str = "",
        file_path: str = "",
    ) -> str:
        """Analysiert Python-Code auf Smells und Sicherheitsprobleme.

        Kombiniert CodeSmellDetector (Qualität) und CodeAuditor (Sicherheit)
        für einen strukturierten Bericht.

        Args:
            code: Python-Code als String (optional wenn file_path gegeben).
            file_path: Pfad zu einer Python-Datei (optional wenn code gegeben).

        Returns:
            Strukturierter Analyse-Bericht als Text.
        """
        if not code and not file_path:
            return "Fehler: Entweder 'code' oder 'file_path' muss angegeben werden."

        # Code aus Datei laden wenn nötig
        source = code
        source_name = "<inline>"
        if file_path and not code:
            try:
                path = Path(file_path)
                if not path.exists():
                    return f"Fehler: Datei '{file_path}' nicht gefunden."
                if not path.suffix == ".py":
                    return f"Fehler: Nur Python-Dateien (.py) werden unterstützt."
                source = path.read_text(encoding="utf-8")
                source_name = str(path)
            except (OSError, UnicodeDecodeError) as e:
                return f"Fehler beim Lesen der Datei: {e}"

        # 1. Code-Smell-Analyse
        from jarvis.tools.code_analyzer import CodeSmellDetector

        smell_detector = CodeSmellDetector()
        smells = []

        # Für Datei-Analyse nutze analyze_file, sonst parse den Code direkt
        if file_path and not code:
            smells = smell_detector.analyze_file(file_path)
        else:
            # Inline-Code: Manuell parsen
            import ast
            try:
                ast.parse(source)
            except SyntaxError as e:
                smells_text = f"Syntaxfehler in Zeile {e.lineno}: {e.msg}"
            else:
                # analyze_file braucht eine echte Datei -- temp-Datei erstellen
                tmp = self._workspace / f"_jarvis_analyze_{uuid.uuid4().hex[:8]}.py"
                try:
                    tmp.write_text(source, encoding="utf-8")
                    smells = smell_detector.analyze_file(str(tmp))
                finally:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except OSError:
                        pass

        # 2. Security-Analyse
        from jarvis.security.code_audit import CodeAuditor

        auditor = CodeAuditor()
        security_report = auditor.audit_skill(
            skill_name=source_name,
            code=source,
            file_path=source_name,
        )

        # 3. Bericht zusammenstellen
        lines = len(source.split("\n"))
        parts: list[str] = [
            f"## Code-Analyse: {source_name}",
            f"Zeilen: {lines}",
            "",
        ]

        # Smells
        if smells:
            parts.append(f"### Code-Smells ({len(smells)} gefunden)")
            for smell in smells:
                parts.append(
                    f"- [{smell.severity.upper()}] Zeile {smell.line}: "
                    f"{smell.smell_type} -- {smell.message}"
                )
                if smell.suggestion:
                    parts.append(f"  Vorschlag: {smell.suggestion}")
        else:
            parts.append("### Code-Smells: Keine gefunden")

        parts.append("")

        # Security
        findings = security_report.findings
        if findings:
            parts.append(f"### Sicherheit ({len(findings)} Findings)")
            for f in findings:
                parts.append(
                    f"- [{f.pattern.severity.value.upper()}] Zeile {f.line_number}: "
                    f"{f.pattern.name} -- {f.code_snippet}"
                )
                if f.pattern.recommendation:
                    parts.append(f"  Empfehlung: {f.pattern.recommendation}")
        else:
            parts.append("### Sicherheit: Keine Findings")

        parts.append("")

        # Gesamtbewertung
        parts.append(f"### Gesamt-Risiko: {security_report.overall_risk.upper()}")
        parts.append(f"Audit bestanden: {'Ja' if security_report.passed else 'Nein'}")

        if security_report.recommendations:
            parts.append("")
            parts.append("### Empfehlungen")
            for rec in security_report.recommendations:
                parts.append(f"- {rec}")

        return "\n".join(parts)


def register_code_tools(
    mcp_client: Any,
    config: "JarvisConfig",
) -> CodeTools:
    """Registriert Code-Tools beim MCP-Client.

    Returns:
        CodeTools-Instanz.
    """
    tools = CodeTools(config)

    mcp_client.register_builtin_handler(
        "run_python",
        tools.run_python,
        description=(
            "Führt Python-Code in einer Sandbox aus. "
            "Der Code wird als temporäre Datei geschrieben und via Python-Interpreter ausgeführt. "
            "Arbeitsverzeichnis: ~/.jarvis/workspace/. "
            "Für autonomes Coding: Schreibe, teste und iteriere Code."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python-Code der ausgeführt werden soll",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in Sekunden (Default: 60)",
                    "default": 60,
                },
                "working_dir": {
                    "type": "string",
                    "description": "Arbeitsverzeichnis (Default: ~/.jarvis/workspace/)",
                    "default": None,
                },
            },
            "required": ["code"],
        },
    )

    mcp_client.register_builtin_handler(
        "analyze_code",
        tools.analyze_code,
        description=(
            "Analysiert Python-Code auf Code-Smells und Sicherheitsprobleme. "
            "Kombiniert AST-basierte Qualitätsanalyse mit Security-Pattern-Scanning. "
            "Read-only, kein Risiko. Akzeptiert Code als String oder Dateipfad."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python-Code als String (optional wenn file_path gegeben)",
                    "default": "",
                },
                "file_path": {
                    "type": "string",
                    "description": "Pfad zu einer Python-Datei (optional wenn code gegeben)",
                    "default": "",
                },
            },
        },
    )

    log.info(
        "code_tools_registered",
        tools=["run_python", "analyze_code"],
        sandbox_level=tools._sandbox.level.value,
    )
    return tools
