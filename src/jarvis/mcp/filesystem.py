"""Dateisystem-Tools für Jarvis.

Implementiert als eingebaute Handler (Phase 1) und als FastMCP-Server (später).
Alle Operationen prüfen Pfade gegen die Sandbox-Konfiguration.

Tools:
  - read_file: Datei lesen (mit optionalem Zeilenbereich)
  - write_file: Datei erstellen/überschreiben (atomar)
  - edit_file: String ersetzen (str_replace-Logik)
  - list_directory: Verzeichnisbaum auflisten

Bibel-Referenz: §5.3 (jarvis-fs Server)
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# Ausgeschlossene Verzeichnisse beim Baum-Listing
EXCLUDED_DIRECTORIES = frozenset({
    "node_modules", "__pycache__", ".git", "venv", ".venv",
})

# Maximale Eintraege im Verzeichnisbaum
MAX_TREE_ENTRIES = 200

__all__ = [
    "FileSystemTools",
    "FileSystemError",
    "register_fs_tools",
]


class FileSystemError(Exception):
    """Fehler bei Dateisystem-Operationen."""


class FileSystemTools:
    """Dateisystem-Operationen mit Sandbox-Validierung. [B§5.3]"""

    def __init__(self, config: JarvisConfig) -> None:
        """Initialisiert FileSystemTools mit Sandbox-Pfaden aus der Konfiguration."""
        self._config = config
        self._allowed_roots: list[Path] = [
            Path(p).expanduser().resolve() for p in config.security.allowed_paths
        ]

    def _validate_path(self, path_str: str) -> Path:
        """Validiert und normalisiert einen Dateipfad.

        Verhindert:
          - Path Traversal (../../etc/passwd)
          - Symlink-Escapes
          - Zugriff außerhalb erlaubter Verzeichnisse

        Raises:
            FileSystemError: Wenn Pfad ungültig oder nicht erlaubt.
        """
        try:
            path = Path(path_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise FileSystemError(f"Ungültiger Pfad: {path_str}") from exc

        # Prüfe ob Pfad in einem erlaubten Verzeichnis liegt
        for root in self._allowed_roots:
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue

        raise FileSystemError(
            f"Zugriff verweigert: {path_str} liegt außerhalb erlaubter Verzeichnisse "
            f"({', '.join(str(r) for r in self._allowed_roots)})"
        )

    def read_file(
        self,
        path: str,
        line_start: int = 0,
        line_end: int = -1,
    ) -> str:
        """Liest eine Datei. Optional nur bestimmte Zeilen.

        Args:
            path: Dateipfad (wird gegen Sandbox geprüft)
            line_start: Erste Zeile (0-basiert, inklusive)
            line_end: Letzte Zeile (-1 = bis Ende, inklusive)

        Returns:
            Dateiinhalt als String.
        """
        validated = self._validate_path(path)

        if not validated.exists():
            raise FileSystemError(f"Datei nicht gefunden: {path}")

        if not validated.is_file():
            raise FileSystemError(f"Kein reguläres File: {path}")

        # Größenbeschränkung (maximal 1MB lesen)
        size = validated.stat().st_size
        if size > 1_048_576:
            raise FileSystemError(
                f"Datei zu groß ({size:,} Bytes). Maximum: 1 MB. "
                f"Verwende line_start/line_end um Abschnitte zu lesen."
            )

        try:
            content = validated.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback für nicht-UTF-8 Dateien
            try:
                content = validated.read_text(encoding="latin-1")
            except Exception as exc:
                raise FileSystemError(f"Datei nicht lesbar: {exc}") from exc

        # Zeilenbereich filtern
        if line_start > 0 or line_end >= 0:
            lines = content.split("\n")
            end = line_end + 1 if line_end >= 0 else len(lines)
            selected = lines[line_start:end]
            # Zeilennummern hinzufügen für Kontext
            numbered = [f"{line_start + i + 1:4d} │ {line}" for i, line in enumerate(selected)]
            return "\n".join(numbered)

        return content

    def write_file(self, path: str, content: str) -> str:
        """Erstellt oder überschreibt eine Datei atomar.

        Schreibt zuerst in eine temporäre Datei, dann rename.
        So wird bei einem Absturz nie eine halb-geschriebene Datei erzeugt.

        Args:
            path: Dateipfad
            content: Zu schreibender Inhalt

        Returns:
            Bestätigungsnachricht.
        """
        validated = self._validate_path(path)

        # Elternverzeichnis erstellen wenn nötig
        validated.parent.mkdir(parents=True, exist_ok=True)

        # Atomares Schreiben via temporäre Datei
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(validated.parent),
                prefix=".jarvis_write_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                # Atomares Rename
                os.replace(tmp_path, str(validated))
            except Exception:
                # Temporäre Datei aufräumen bei Fehler
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except OSError as exc:
            raise FileSystemError(f"Schreibfehler: {exc}") from exc

        size = validated.stat().st_size
        log.info("file_written", path=str(validated), size=size)
        return f"Datei geschrieben: {path} ({size:,} Bytes)"

    # Maximale Groesse fuer edit-Parameter (500 KB)
    MAX_EDIT_SIZE = 512_000

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Ersetzt einen eindeutigen String in einer Datei.

        Der old_text muss EXAKT EINMAL in der Datei vorkommen.
        Verhindert versehentliche Mehrfach-Ersetzungen.

        Args:
            path: Dateipfad
            old_text: Zu ersetzender Text (muss eindeutig sein)
            new_text: Neuer Text

        Returns:
            Bestätigungsnachricht mit Zeilenänderungen.
        """
        if len(old_text) > self.MAX_EDIT_SIZE:
            raise FileSystemError(
                f"old_text zu gross ({len(old_text):,} Bytes, max {self.MAX_EDIT_SIZE:,})"
            )
        if len(new_text) > self.MAX_EDIT_SIZE:
            raise FileSystemError(
                f"new_text zu gross ({len(new_text):,} Bytes, max {self.MAX_EDIT_SIZE:,})"
            )

        validated = self._validate_path(path)

        if not validated.exists():
            raise FileSystemError(f"Datei nicht gefunden: {path}")

        content = validated.read_text(encoding="utf-8")

        # Eindeutigkeits-Prüfung
        count = content.count(old_text)
        if count == 0:
            raise FileSystemError(f"Text nicht gefunden in {path}. Gesucht: '{old_text[:100]}...'")
        if count > 1:
            raise FileSystemError(
                f"Text kommt {count}x vor in {path} – muss eindeutig sein. "
                f"Verwende einen längeren Ausschnitt als old_text."
            )

        # Ersetzen
        new_content = content.replace(old_text, new_text, 1)

        # Atomar schreiben (validated path to avoid double-validation)
        self.write_file(str(validated), new_content)

        # Zeilenänderungen berechnen
        old_lines = old_text.count("\n")
        new_lines = new_text.count("\n")
        diff = new_lines - old_lines

        return (
            f"Datei bearbeitet: {path}\n"
            f"Ersetzt: {len(old_text)} → {len(new_text)} Zeichen\n"
            f"Zeilen: {'+' if diff >= 0 else ''}{diff}"
        )

    def list_directory(self, path: str, depth: int = 2) -> str:
        """Listet Verzeichnisinhalt als Baumstruktur.

        Args:
            path: Verzeichnispfad
            depth: Maximale Tiefe (Default: 2)

        Returns:
            Baumstruktur als Text.
        """
        validated = self._validate_path(path)

        if not validated.exists():
            raise FileSystemError(f"Verzeichnis nicht gefunden: {path}")

        if not validated.is_dir():
            raise FileSystemError(f"Kein Verzeichnis: {path}")

        lines: list[str] = [f"{validated.name}/"]
        self._tree(validated, lines, prefix="", depth=depth, max_depth=depth)

        if len(lines) > MAX_TREE_ENTRIES:
            total = len(lines)
            lines = lines[:MAX_TREE_ENTRIES]
            lines.append(f"... ({total - MAX_TREE_ENTRIES} weitere Eintraege)")

        return "\n".join(lines)

    def _tree(
        self,
        directory: Path,
        lines: list[str],
        prefix: str,
        depth: int,
        max_depth: int,
    ) -> None:
        """Rekursive Baum-Generierung."""
        if depth <= 0:
            return

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}└── [Zugriff verweigert]")
            return

        # Versteckte Dateien und bekannte Build-Verzeichnisse ausfiltern
        entries = [
            e
            for e in entries
            if not e.name.startswith(".")
            and e.name not in EXCLUDED_DIRECTORIES
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._tree(entry, lines, child_prefix, depth - 1, max_depth)
            else:
                size = entry.stat().st_size
                size_str = self._format_size(size)
                lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

    @staticmethod
    def _format_size(size: int) -> str:
        """Formatiert Dateigröße in menschenlesbares Format."""
        if size < 1024:
            return f"{size} B"
        if size < 1_048_576:
            return f"{size / 1024:.1f} KB"
        return f"{size / 1_048_576:.1f} MB"


def register_fs_tools(
    mcp_client: Any,
    config: JarvisConfig,
) -> FileSystemTools:
    """Registriert Dateisystem-Tools beim MCP-Client.

    Returns:
        FileSystemTools-Instanz für direkten Zugriff.
    """
    fs = FileSystemTools(config)

    mcp_client.register_builtin_handler(
        "read_file",
        fs.read_file,
        description="Liest eine Datei. Pfad wird gegen Sandbox-Policy geprüft.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Dateipfad"},
                "line_start": {
                    "type": "integer",
                    "description": "Erste Zeile (0-basiert)",
                    "default": 0,
                },
                "line_end": {
                    "type": "integer",
                    "description": "Letzte Zeile (-1 = Ende)",
                    "default": -1,
                },
            },
            "required": ["path"],
        },
    )

    mcp_client.register_builtin_handler(
        "write_file",
        fs.write_file,
        description="Erstellt oder überschreibt eine Datei (atomar).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Dateipfad"},
                "content": {"type": "string", "description": "Dateiinhalt"},
            },
            "required": ["path", "content"],
        },
    )

    mcp_client.register_builtin_handler(
        "edit_file",
        fs.edit_file,
        description="Ersetzt einen eindeutigen String in einer Datei (str_replace).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Dateipfad"},
                "old_text": {
                    "type": "string",
                    "description": "Zu ersetzender Text (muss eindeutig sein)",
                },
                "new_text": {"type": "string", "description": "Neuer Text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    )

    mcp_client.register_builtin_handler(
        "list_directory",
        fs.list_directory,
        description="Listet Verzeichnisinhalt als Baumstruktur.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Verzeichnispfad"},
                "depth": {"type": "integer", "description": "Maximale Tiefe", "default": 2},
            },
            "required": ["path"],
        },
    )

    log.info(
        "fs_tools_registered", tools=["read_file", "write_file", "edit_file", "list_directory"]
    )
    return fs
