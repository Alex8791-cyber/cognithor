"""Tests für FileSystemTools – Sandbox-gesicherte Datei-Operationen.

Testet:
  - read_file: Lesen, Zeilenbereich, Größenlimit, Encoding-Fallback
  - write_file: Atomares Schreiben, Verzeichnis-Erstellung
  - edit_file: Eindeutiger Ersatz, Fehler bei 0 oder >1 Treffern
  - list_directory: Baumstruktur, versteckte Dateien, Tiefenlimit
  - Pfad-Validierung: Sandbox-Grenzen, Traversal-Angriffe
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.mcp.filesystem import FileSystemError, FileSystemTools

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """Ein temporäres Sandbox-Verzeichnis."""
    sb = tmp_path / "sandbox"
    sb.mkdir()
    return sb


@pytest.fixture()
def config(tmp_path: Path, sandbox: Path) -> JarvisConfig:
    """Config deren allowed_paths auf die Sandbox zeigt."""
    cfg = JarvisConfig(
        jarvis_home=tmp_path / ".jarvis",
        security=SecurityConfig(
            allowed_paths=[str(sandbox), str(tmp_path / ".jarvis")],
        ),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def fs(config: JarvisConfig) -> FileSystemTools:
    return FileSystemTools(config)


# =============================================================================
# Pfad-Validierung
# =============================================================================


class TestPathValidation:
    def test_allowed_path_accepted(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Pfade innerhalb der Sandbox werden akzeptiert."""
        (sandbox / "test.txt").write_text("hello")
        path = fs._validate_path(str(sandbox / "test.txt"))
        assert path == (sandbox / "test.txt").resolve()

    def test_outside_path_rejected(self, fs: FileSystemTools) -> None:
        """Pfade außerhalb aller allowed_paths werden blockiert."""
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs._validate_path("/etc/passwd")

    def test_traversal_attack_rejected(self, fs: FileSystemTools, sandbox: Path) -> None:
        """../../-Angriffe werden durch resolve() erkannt."""
        evil_path = str(sandbox / ".." / ".." / "etc" / "passwd")
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs._validate_path(evil_path)

    def test_relative_path_resolved(self, fs: FileSystemTools) -> None:
        """Relative Pfade werden aufgelöst und geprüft."""
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs._validate_path("../../../etc/shadow")

    def test_home_expansion(self, fs: FileSystemTools) -> None:
        """~ wird expandiert und dann gegen Sandbox geprüft."""
        # ~ expandiert zu /root oder /home/user — beides nicht in Sandbox
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs._validate_path("~/geheim.txt")


# =============================================================================
# read_file
# =============================================================================


class TestReadFile:
    def test_read_existing_file(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Vorhandene Datei wird korrekt gelesen."""
        f = sandbox / "hello.txt"
        f.write_text("Hallo Welt", encoding="utf-8")
        content = fs.read_file(str(f))
        assert content == "Hallo Welt"

    def test_read_nonexistent_file(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Nicht vorhandene Datei erzeugt Fehler."""
        with pytest.raises(FileSystemError, match="nicht gefunden"):
            fs.read_file(str(sandbox / "gibts_nicht.txt"))

    def test_read_directory_fails(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Verzeichnis lesen erzeugt Fehler."""
        with pytest.raises(FileSystemError, match="Kein reguläres File"):
            fs.read_file(str(sandbox))

    def test_read_outside_sandbox(self, fs: FileSystemTools) -> None:
        """Lesen außerhalb der Sandbox wird blockiert."""
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs.read_file("/etc/hostname")

    def test_read_line_range(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Zeilenbereich wird korrekt gefiltert."""
        f = sandbox / "lines.txt"
        f.write_text("Zeile 1\nZeile 2\nZeile 3\nZeile 4\nZeile 5", encoding="utf-8")
        result = fs.read_file(str(f), line_start=1, line_end=2)
        assert "Zeile 2" in result
        assert "Zeile 3" in result
        assert "Zeile 1" not in result
        assert "Zeile 4" not in result

    def test_read_line_range_with_numbers(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Zeilenbereich enthält Zeilennummern."""
        f = sandbox / "numbered.txt"
        f.write_text("AAA\nBBB\nCCC", encoding="utf-8")
        result = fs.read_file(str(f), line_start=1, line_end=1)
        assert "│" in result  # Zeilennummer-Separator
        assert "BBB" in result

    def test_read_file_too_large(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Dateien >1MB erzeugen Fehler."""
        f = sandbox / "big.txt"
        f.write_bytes(b"x" * (1_048_577))  # 1MB + 1 Byte
        with pytest.raises(FileSystemError, match="zu groß"):
            fs.read_file(str(f))

    def test_read_latin1_fallback(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Nicht-UTF-8-Dateien werden via latin-1 gelesen."""
        f = sandbox / "latin.txt"
        f.write_bytes("Ärger mit Ü-Zeichen".encode("latin-1"))
        content = fs.read_file(str(f))
        assert "rger" in content  # Mindestens teilweise lesbar

    def test_read_utf8_umlauts(self, fs: FileSystemTools, sandbox: Path) -> None:
        """UTF-8-Umlaute werden korrekt gelesen."""
        f = sandbox / "umlaute.txt"
        f.write_text("Ärzte empfehlen Öl und Übung", encoding="utf-8")
        content = fs.read_file(str(f))
        assert "Ärzte" in content
        assert "Übung" in content


# =============================================================================
# write_file
# =============================================================================


class TestWriteFile:
    def test_write_new_file(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Neue Datei wird erstellt."""
        result = fs.write_file(str(sandbox / "new.txt"), "Inhalt")
        assert "geschrieben" in result
        assert (sandbox / "new.txt").read_text() == "Inhalt"

    def test_write_overwrites_existing(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Bestehende Datei wird überschrieben."""
        f = sandbox / "existing.txt"
        f.write_text("alt")
        fs.write_file(str(f), "neu")
        assert f.read_text() == "neu"

    def test_write_creates_parent_directories(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehlende Elternverzeichnisse werden erstellt."""
        deep_path = sandbox / "a" / "b" / "c" / "deep.txt"
        fs.write_file(str(deep_path), "tief")
        assert deep_path.read_text() == "tief"

    def test_write_outside_sandbox_blocked(self, fs: FileSystemTools) -> None:
        """Schreiben außerhalb der Sandbox wird blockiert."""
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs.write_file("/tmp/evil.txt", "böse")

    def test_write_returns_size(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Rückgabe enthält Dateigröße."""
        result = fs.write_file(str(sandbox / "sized.txt"), "1234567890")
        assert "10" in result  # 10 Bytes

    def test_write_utf8(self, fs: FileSystemTools, sandbox: Path) -> None:
        """UTF-8-Zeichen werden korrekt geschrieben."""
        f = sandbox / "utf8.txt"
        fs.write_file(str(f), "Ärger mit Ü → gelöst")
        content = f.read_text(encoding="utf-8")
        assert "Ärger" in content
        assert "gelöst" in content


# =============================================================================
# edit_file
# =============================================================================


class TestEditFile:
    def test_edit_unique_text(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Eindeutiger Text wird korrekt ersetzt."""
        f = sandbox / "edit.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = fs.edit_file(str(f), "World", "Jarvis")
        assert "bearbeitet" in result
        assert f.read_text() == "Hello Jarvis"

    def test_edit_text_not_found(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehler wenn zu ersetzender Text nicht vorhanden."""
        f = sandbox / "edit2.txt"
        f.write_text("Hello World", encoding="utf-8")
        with pytest.raises(FileSystemError, match="nicht gefunden"):
            fs.edit_file(str(f), "gibts nicht", "egal")

    def test_edit_text_not_unique(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehler wenn Text mehrfach vorkommt."""
        f = sandbox / "edit3.txt"
        f.write_text("abc abc abc", encoding="utf-8")
        with pytest.raises(FileSystemError, match="3x vor"):
            fs.edit_file(str(f), "abc", "xyz")

    def test_edit_nonexistent_file(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehler bei nicht vorhandener Datei."""
        with pytest.raises(FileSystemError, match="nicht gefunden"):
            fs.edit_file(str(sandbox / "nope.txt"), "a", "b")

    def test_edit_multiline(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Mehrzeilige Ersetzung funktioniert."""
        f = sandbox / "multi.txt"
        f.write_text("Zeile 1\nZeile 2\nZeile 3", encoding="utf-8")
        fs.edit_file(str(f), "Zeile 2", "Neue Zeile A\nNeue Zeile B")
        content = f.read_text()
        assert "Neue Zeile A\nNeue Zeile B" in content
        assert "Zeile 2" not in content

    def test_edit_reports_line_diff(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Rückgabe zeigt Zeilenänderung."""
        f = sandbox / "diff.txt"
        f.write_text("A\nB\nC", encoding="utf-8")
        result = fs.edit_file(str(f), "B", "X\nY\nZ")
        assert "+2" in result  # 1 Zeile → 3 Zeilen = +2


# =============================================================================
# list_directory
# =============================================================================


class TestListDirectory:
    def test_list_simple_dir(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Verzeichnisinhalt wird aufgelistet."""
        (sandbox / "a.txt").write_text("a")
        (sandbox / "b.txt").write_text("b")
        (sandbox / "sub").mkdir()
        result = fs.list_directory(str(sandbox))
        assert "a.txt" in result
        assert "b.txt" in result
        assert "sub/" in result

    def test_list_nonexistent_dir(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehler bei nicht vorhandenem Verzeichnis."""
        with pytest.raises(FileSystemError, match="nicht gefunden"):
            fs.list_directory(str(sandbox / "nope"))

    def test_list_file_not_dir(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Fehler wenn Pfad eine Datei statt Verzeichnis ist."""
        f = sandbox / "file.txt"
        f.write_text("x")
        with pytest.raises(FileSystemError, match="Kein Verzeichnis"):
            fs.list_directory(str(f))

    def test_list_hidden_files_filtered(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Versteckte Dateien und __pycache__ werden ausgeblendet."""
        (sandbox / ".hidden").write_text("x")
        (sandbox / "__pycache__").mkdir()
        (sandbox / "visible.txt").write_text("x")
        result = fs.list_directory(str(sandbox))
        assert ".hidden" not in result
        assert "__pycache__" not in result
        assert "visible.txt" in result

    def test_list_depth_limit(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Tiefenlimit wird respektiert."""
        deep = sandbox / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("x")
        result = fs.list_directory(str(sandbox), depth=1)
        assert "deep.txt" not in result  # Zu tief

    def test_list_shows_file_sizes(self, fs: FileSystemTools, sandbox: Path) -> None:
        """Dateigrößen werden angezeigt."""
        (sandbox / "small.txt").write_text("x")
        result = fs.list_directory(str(sandbox))
        assert "B" in result  # Größenangabe enthalten

    def test_list_outside_sandbox_blocked(self, fs: FileSystemTools) -> None:
        """Verzeichnislisting außerhalb Sandbox wird blockiert."""
        with pytest.raises(FileSystemError, match="Zugriff verweigert"):
            fs.list_directory("/etc")


# =============================================================================
# register_fs_tools
# =============================================================================


class TestRegisterFsTools:
    def test_registers_all_tools(self, config: JarvisConfig) -> None:
        """Alle 4 Tools werden beim MCP-Client registriert."""
        from jarvis.mcp.client import JarvisMCPClient
        from jarvis.mcp.filesystem import register_fs_tools

        client = JarvisMCPClient(config)
        fs = register_fs_tools(client, config)

        assert isinstance(fs, FileSystemTools)
        tools = client.get_tool_list()
        assert "read_file" in tools
        assert "write_file" in tools
        assert "edit_file" in tools
        assert "list_directory" in tools
        assert len(tools) == 4

    def test_schemas_contain_descriptions(self, config: JarvisConfig) -> None:
        """Tool-Schemas enthalten Beschreibungen."""
        from jarvis.mcp.client import JarvisMCPClient
        from jarvis.mcp.filesystem import register_fs_tools

        client = JarvisMCPClient(config)
        register_fs_tools(client, config)

        schemas = client.get_tool_schemas()
        for name, schema in schemas.items():
            assert "description" in schema, f"Schema für {name} hat keine Beschreibung"
            assert schema["description"], f"Beschreibung für {name} ist leer"
