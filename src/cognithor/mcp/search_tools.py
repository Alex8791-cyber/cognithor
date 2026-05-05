"""Such-Tools fuer Cognithor -- Dateisuche und Inhaltssuche als MCP-Tools.

Drei Tools:
  - search_files: Dateien nach Name/Glob-Pattern finden
  - find_in_files: Dateiinhalte durchsuchen (Text/Regex)
  - find_and_replace: Suchen und Ersetzen in Dateien (mit dry_run Default)

Factory: register_search_tools(mcp_client, config) -> None

Bibel-Referenz: §5.3 (MCP-Tools)
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)


# Audit-PR9 (audit-HIGH-4): ReDoS guard for user-supplied regex.
# Python's `re` has no per-call timeout — the only realistic
# protection is to reject patterns that look catastrophic before
# we hand them to `re.compile`. The heuristic catches the
# canonical "nested quantifier" shape — ``(a+)+`` style — by
# looking for ``<quantifier><close-paren><optional-ws><quantifier>``
# anywhere in the source, plus consecutive quantifiers like
# ``a++`` or ``a*?+``. Pattern length is also capped (no
# legitimate workspace search needs more than 256 chars).
_REDOS_NESTED_QUANT_RE = re.compile(
    r"""
    [+*?]\s*[+*?]              # consecutive quantifiers e.g. ``++`` ``*?+``
    |
    [+*?]\)\s*[+*?]            # ``+)+`` style — quantifier, close-paren, quantifier
    |
    [+*?]\}\s*[+*?]            # same with explicit ``{n,m}`` close-brace
    """,
    re.VERBOSE,
)
_REDOS_PATTERN_MAX_LEN = 256


def _validate_user_regex(pattern: str) -> str | None:
    """Return an error message when ``pattern`` is unsafe, else ``None``."""

    if len(pattern) > _REDOS_PATTERN_MAX_LEN:
        return (
            f"regex pattern exceeds {_REDOS_PATTERN_MAX_LEN} chars — "
            "split into a smaller search or use a literal query"
        )
    if _REDOS_NESTED_QUANT_RE.search(pattern):
        return (
            "regex pattern looks catastrophic (nested quantifiers like "
            "(a+)+ can take exponential time on adversarial input). "
            "Refactor to a non-nested form."
        )
    return None


# Ausgeschlossene Verzeichnisse
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
    }
)

# Ausgeschlossene Suffixe (egg-info Verzeichnisse)
_EXCLUDED_SUFFIX = ".egg-info"

# Maximale Dateigroesse fuer Inhaltssuche (10 MB)
_MAX_FILE_SIZE = 10 * 1024 * 1024

# Bytes fuer Binaer-Erkennung
_BINARY_CHECK_BYTES = 8192

__all__ = [
    "SearchTools",
    "SearchToolsError",
    "register_search_tools",
]


class SearchToolsError(Exception):
    """Fehler bei Such-Operationen."""


class SearchTools:
    """Dateisuche und Inhaltssuche mit Workspace-Validierung. [B§5.3]

    Alle Pfade werden gegen config.workspace_dir validiert.
    Binary-Dateien werden automatisch uebersprungen.
    """

    def __init__(self, config: CognithorConfig) -> None:
        self._config = config
        self._workspace = config.workspace_dir

        log.info(
            "search_tools_init",
            workspace=str(self._workspace),
        )

    def _validate_path(self, path_str: str) -> Path:
        """Validiert und normalisiert einen Pfad gegen den Workspace.

        Raises:
            SearchToolsError: Wenn Pfad ausserhalb des Workspace liegt.
        """
        try:
            path = Path(path_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise SearchToolsError(t("tools.search_invalid_path", path=path_str)) from exc

        workspace_root = self._workspace.expanduser().resolve()
        try:
            path.relative_to(workspace_root)
        except ValueError as exc:
            raise SearchToolsError(
                t("tools.search_access_denied", path=path_str, root=workspace_root)
            ) from exc

        return path

    def _get_search_root(self, path: str | None) -> Path:
        """Ermittelt den Such-Pfad (Default: workspace_dir)."""
        if path:
            return self._validate_path(path)
        return self._workspace.expanduser().resolve()

    @staticmethod
    def _is_excluded_dir(name: str) -> bool:
        """Prueft ob ein Verzeichnisname ausgeschlossen ist."""
        return name in EXCLUDED_DIRS or name.endswith(_EXCLUDED_SUFFIX)

    @staticmethod
    def _is_binary(file_path: Path) -> bool:
        """Prueft ob eine Datei binaer ist (Null-Byte in den ersten 8 KB).

        Returns:
            True wenn binaer, False wenn Text.
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(_BINARY_CHECK_BYTES)
                return b"\x00" in chunk
        except (OSError, PermissionError):
            return True  # Im Zweifel als binaer behandeln

    @staticmethod
    def _read_file_text(file_path: Path) -> str | None:
        """Liest Dateiinhalt als Text mit Encoding-Fallback.

        Returns:
            Dateiinhalt oder None bei Fehler.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="latin-1")
            except (OSError, UnicodeDecodeError):
                return None
        except OSError:
            return None

    def _walk_files(
        self,
        root: Path,
        file_glob: str | None = None,
    ) -> list[Path]:
        """Traversiert Verzeichnisbaum mit os.scandir (schnell).

        Ueberspringt ausgeschlossene Verzeichnisse.

        Args:
            root: Startverzeichnis.
            file_glob: Optionaler Glob-Filter fuer Dateinamen.

        Returns:
            Liste von Dateipfaden.
        """
        results: list[Path] = []

        def _recurse(directory: Path) -> None:
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            if not self._is_excluded_dir(entry.name):
                                _recurse(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            file_path = Path(entry.path)
                            if file_glob:
                                if file_path.match(file_glob):
                                    results.append(file_path)
                            else:
                                results.append(file_path)
            except (PermissionError, OSError):
                pass  # Skip unreadable directories

        _recurse(root)
        return results

    async def search_files(
        self,
        pattern: str,
        path: str = "",
        max_results: int = 100,
    ) -> str:
        """Findet Dateien nach Name/Glob-Pattern.

        Args:
            pattern: Glob-Pattern (z.B. "*.py", "**/*.json").
            path: Suchverzeichnis (Default: Workspace).
            max_results: Maximale Ergebnisanzahl (Default: 100).

        Returns:
            Liste der gefundenen Dateien (relativ zum Workspace).
        """
        if not pattern or not pattern.strip():
            return t("tools.search_pattern_required")

        root = self._get_search_root(path or None)
        self._workspace.expanduser().resolve()
        max_results = max(1, min(max_results, 10000))

        if not root.exists():
            return t("tools.search_dir_not_found", root=root)

        if not root.is_dir():
            return t("tools.search_not_a_dir", root=root)

        # Verwende pathlib.glob fuer rekursive Suche
        matches: list[str] = []
        try:
            for match_path in root.glob(pattern.strip()):
                # Ueberspringe ausgeschlossene Verzeichnisse
                parts = match_path.relative_to(root).parts
                if any(self._is_excluded_dir(part) for part in parts):
                    continue

                if match_path.is_file():
                    # Always return the full absolute path so the LLM
                    # can use it directly in follow-up operations
                    # (read, edit, delete).
                    matches.append(str(match_path.resolve()))

                if len(matches) >= max_results:
                    break
        except (OSError, ValueError) as exc:
            return t("tools.search_error", exc=exc)

        if not matches:
            return t("tools.search_no_files", pattern=pattern)

        total_note = ""
        if len(matches) >= max_results:
            total_note = t("tools.search_result_limited", count=max_results)

        header = t("tools.search_found_files", count=len(matches), pattern=pattern) + "\n"
        file_list = "\n".join(f"  {m}" for m in sorted(matches))

        return header + file_list + total_note

    async def find_in_files(
        self,
        query: str,
        path: str = "",
        glob: str = "",
        max_results: int = 50,
        context_lines: int = 2,
        regex: bool = False,
    ) -> str:
        """Durchsucht Dateiinhalte nach Text oder Regex.

        Args:
            query: Suchbegriff oder Regex-Pattern.
            path: Suchverzeichnis (Default: Workspace).
            glob: Datei-Filter (z.B. "*.py").
            max_results: Maximale Trefferanzahl (Default: 50).
            context_lines: Kontextzeilen vor/nach Treffer (Default: 2).
            regex: Wenn True, als Regex interpretieren.

        Returns:
            Treffer mit Datei:Zeile und Kontext.
        """
        if not query or not query.strip():
            return t("tools.search_query_required")

        root = self._get_search_root(path or None)
        workspace_root = self._workspace.expanduser().resolve()
        max_results = max(1, min(max_results, 1000))
        context_lines = max(0, min(context_lines, 10))

        if not root.exists():
            return t("tools.search_dir_not_found", root=root)

        # Audit-PR9: ReDoS guard. Reject user-supplied regex with
        # catastrophic shape BEFORE compile so we don't spend time
        # in `re.search` on adversarial input.
        if regex:
            redos_err = _validate_user_regex(query)
            if redos_err is not None:
                return f"Error: {redos_err}"
        # Regex kompilieren oder escaped Pattern erstellen
        try:
            search_pattern = re.compile(query) if regex else re.compile(re.escape(query))
        except re.error as exc:
            return t("tools.search_invalid_regex", exc=exc)

        # Dateien sammeln
        file_glob = glob.strip() if glob else None
        files = self._walk_files(root, file_glob)

        results: list[str] = []
        files_with_matches = 0

        for file_path in sorted(files):
            # Groessenlimit pruefen
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_SIZE:
                continue

            # Binaerdateien ueberspringen
            if self._is_binary(file_path):
                continue

            content = self._read_file_text(file_path)
            if content is None:
                continue

            lines = content.split("\n")
            file_matches: list[str] = []

            for line_num, line in enumerate(lines):
                if search_pattern.search(line):
                    # Relativer Pfad
                    try:
                        rel_path = file_path.relative_to(workspace_root)
                    except ValueError:
                        rel_path = file_path

                    # Kontextzeilen sammeln
                    context_parts: list[str] = []

                    # Kontext davor
                    start = max(0, line_num - context_lines)
                    for ctx_num in range(start, line_num):
                        context_parts.append(f"  {ctx_num + 1:6d}  {lines[ctx_num]}")

                    # Trefferzeile (markiert)
                    context_parts.append(f"> {line_num + 1:6d}  {line}")

                    # Kontext danach
                    end = min(len(lines), line_num + context_lines + 1)
                    for ctx_num in range(line_num + 1, end):
                        context_parts.append(f"  {ctx_num + 1:6d}  {lines[ctx_num]}")

                    file_matches.append("\n".join(context_parts))

                    if len(results) + len(file_matches) >= max_results:
                        break

            if file_matches:
                files_with_matches += 1
                try:
                    rel_path = file_path.relative_to(workspace_root)
                except ValueError:
                    rel_path = file_path

                results.append(f"\n--- {rel_path} ---")
                results.extend(file_matches)

            if len(results) >= max_results + files_with_matches:
                break

        if not results:
            return t("tools.search_no_matches", query=query)

        total_note = ""
        if len(results) >= max_results:
            total_note = t("tools.search_matches_limited", count=max_results)

        header = t("tools.search_matches_header", query=query, count=files_with_matches) + "\n"
        return header + "\n".join(results) + total_note

    async def find_and_replace(
        self,
        query: str,
        replacement: str,
        path: str = "",
        glob: str = "",
        dry_run: bool = True,
        regex: bool = False,
    ) -> str:
        """Sucht und ersetzt in Dateien.

        Args:
            query: Suchbegriff oder Regex-Pattern.
            replacement: Ersetzungstext.
            path: Suchverzeichnis (Default: Workspace).
            glob: Datei-Filter (z.B. "*.py").
            dry_run: Wenn True, nur anzeigen was geaendert wuerde (Default: True!).
            regex: Wenn True, als Regex interpretieren.

        Returns:
            Liste der Aenderungen (oder Vorschau im dry_run-Modus).
        """
        if not query or not query.strip():
            return t("tools.search_query_required")

        root = self._get_search_root(path or None)
        workspace_root = self._workspace.expanduser().resolve()

        if not root.exists():
            return t("tools.search_dir_not_found", root=root)

        # Audit-PR9: ReDoS guard (same as find_in_files). Reject
        # catastrophic-shape user regex before compile.
        if regex:
            redos_err = _validate_user_regex(query)
            if redos_err is not None:
                return f"Error: {redos_err}"
        # Pattern kompilieren
        try:
            search_pattern = re.compile(query) if regex else re.compile(re.escape(query))
        except re.error as exc:
            return t("tools.search_invalid_regex", exc=exc)

        # Dateien sammeln
        file_glob = glob.strip() if glob else None
        files = self._walk_files(root, file_glob)

        changes: list[str] = []
        files_changed = 0
        total_replacements = 0

        for file_path in sorted(files):
            # Groessenlimit pruefen
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_SIZE:
                continue

            # Binaerdateien ueberspringen
            if self._is_binary(file_path):
                continue

            content = self._read_file_text(file_path)
            if content is None:
                continue

            # Treffer zaehlen
            match_count = len(search_pattern.findall(content))
            if match_count == 0:
                continue

            try:
                rel_path = file_path.relative_to(workspace_root)
            except ValueError:
                rel_path = file_path

            if dry_run:
                # Vorschau: Zeige betroffene Zeilen
                lines = content.split("\n")
                preview_lines: list[str] = []
                for line_num, line in enumerate(lines):
                    if search_pattern.search(line):
                        new_line = search_pattern.sub(replacement, line)
                        preview_lines.append(f"  Zeile {line_num + 1}:")
                        preview_lines.append(f"    - {line.strip()}")
                        preview_lines.append(f"    + {new_line.strip()}")

                changes.append(f"\n{rel_path} ({match_count} Ersetzung(en)):")
                changes.extend(preview_lines)
            else:
                # Tatsaechliche Ersetzung
                new_content = search_pattern.sub(replacement, content)

                if new_content != content:
                    # Backup erstellen (.bak)
                    bak_path = file_path.with_suffix(file_path.suffix + ".bak")
                    with contextlib.suppress(OSError):
                        shutil.copy2(str(file_path), str(bak_path))

                    # Datei schreiben
                    try:
                        file_path.write_text(new_content, encoding="utf-8")
                        files_changed += 1
                        total_replacements += match_count
                        changes.append(f"  {rel_path}: {match_count} Ersetzung(en)")
                    except OSError as exc:
                        changes.append(f"  {rel_path}: FEHLER beim Schreiben: {exc}")

        if not changes:
            return t("tools.search_no_matches", query=query)

        if dry_run:
            affected = len([c for c in changes if c.startswith("\n")])
            header = t(
                "tools.search_dry_run_header",
                query=query,
                replacement=replacement,
                count=affected,
            )
        else:
            header = t(
                "tools.search_replace_done_header",
                query=query,
                replacement=replacement,
                files=files_changed,
                total=total_replacements,
            )

        log.info(
            "find_and_replace",
            query=query[:100],
            replacement=replacement[:50],
            dry_run=dry_run,
            files_changed=files_changed if not dry_run else 0,
            total_replacements=total_replacements if not dry_run else 0,
        )

        return header + "\n".join(changes)


def register_search_tools(
    mcp_client: Any,
    config: CognithorConfig,
) -> None:
    """Registriert Such-Tools beim MCP-Client."""
    tools = SearchTools(config)

    mcp_client.register_builtin_handler(
        "search_files",
        tools.search_files,
        description=(
            "Findet Dateien nach Name oder Glob-Pattern (z.B. '*.py', '**/*.json'). "
            "Rekursive Suche mit automatischem Ausschluss von .git, node_modules, etc."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob-Pattern (z.B. '*.py', '**/*.json', 'config*')",
                },
                "path": {
                    "type": "string",
                    "description": "Suchverzeichnis (Default: Workspace)",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximale Ergebnisanzahl (Default: 100)",
                    "default": 100,
                },
            },
            "required": ["pattern"],
        },
    )

    mcp_client.register_builtin_handler(
        "find_in_files",
        tools.find_in_files,
        description=(
            "Durchsucht Dateiinhalte nach Text oder Regex. "
            "Zeigt Treffer mit Kontext (Zeilen davor/danach). "
            "Ueberspringt automatisch Binaerdateien und grosse Dateien."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff oder Regex-Pattern",
                },
                "path": {
                    "type": "string",
                    "description": "Suchverzeichnis (Default: Workspace)",
                    "default": "",
                },
                "glob": {
                    "type": "string",
                    "description": "Datei-Filter (z.B. '*.py', '*.ts')",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximale Trefferanzahl (Default: 50)",
                    "default": 50,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Kontextzeilen vor/nach Treffer (Default: 2)",
                    "default": 2,
                },
                "regex": {
                    "type": "boolean",
                    "description": "Als Regex interpretieren (Default: false)",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "find_and_replace",
        tools.find_and_replace,
        description=(
            "Sucht und ersetzt Text in Dateien. "
            "WICHTIG: Standard ist dry_run=true (nur Vorschau). "
            "Erstellt .bak-Backups vor Aenderungen. "
            "Unterstuetzt Text und Regex."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff oder Regex-Pattern",
                },
                "replacement": {
                    "type": "string",
                    "description": "Ersetzungstext",
                },
                "path": {
                    "type": "string",
                    "description": "Suchverzeichnis (Default: Workspace)",
                    "default": "",
                },
                "glob": {
                    "type": "string",
                    "description": "Datei-Filter (z.B. '*.py')",
                    "default": "",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Nur Vorschau, keine Aenderungen (Default: true!)",
                    "default": True,
                },
                "regex": {
                    "type": "boolean",
                    "description": "Als Regex interpretieren (Default: false)",
                    "default": False,
                },
            },
            "required": ["query", "replacement"],
        },
    )

    log.info(
        "search_tools_registered",
        tools=["search_files", "find_in_files", "find_and_replace"],
    )
