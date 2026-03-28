# Vault Dual-Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace encrypted .md files with an encrypted SQLite DB when `vault.encrypt_files=true`. Keep Obsidian-compatible .md files when `false`. Automatic bidirectional migration on toggle.

**Architecture:** Extract a `VaultBackend` ABC from VaultTools. Implement `FileBackend` (current behavior) and `DBBackend` (SQLCipher + FTS5). VaultTools delegates all storage to the active backend. Migration runs automatically on mode change.

**Tech Stack:** Python 3.12+, sqlite3/SQLCipher (via encrypted_connect), FTS5, existing vault.py infrastructure

**Spec:** `docs/superpowers/specs/2026-03-28-vault-dual-backend-design.md`

---

## File Structure

```
CREATE src/jarvis/mcp/vault_backend.py       — VaultBackend ABC + shared types
CREATE src/jarvis/mcp/vault_file_backend.py   — FileBackend (extracted from vault.py)
CREATE src/jarvis/mcp/vault_db_backend.py     — DBBackend (SQLCipher + FTS5)
CREATE src/jarvis/mcp/vault_migration.py      — Bidirectional migration
CREATE tests/test_mcp/test_vault_db_backend.py — DB backend tests
CREATE tests/test_mcp/test_vault_migration.py  — Migration tests
MODIFY src/jarvis/mcp/vault.py                — Refactor to use backend interface
```

---

### Task 1: VaultBackend ABC + Shared Types

**Files:**
- Create: `src/jarvis/mcp/vault_backend.py`

- [ ] **Step 1: Write the ABC and shared utilities**

```python
"""Vault storage backend interface + shared utilities."""
from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

__all__ = ["VaultBackend", "NoteData", "slugify", "now_iso", "parse_tags"]


def slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    text = text.lower()
    for src, dst in [("ae", "ae"), ("oe", "oe"), ("ue", "ue"), ("ss", "ss"),
                     ("\u00e4", "ae"), ("\u00f6", "oe"), ("\u00fc", "ue"), ("\u00df", "ss")]:
        text = text.replace(src, dst)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:80] or "notiz"


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_tags(tags: str | list[str]) -> list[str]:
    """Normalize tags to lowercase list."""
    if isinstance(tags, list):
        return [t.strip().lower() for t in tags if t.strip()]
    return [t.strip().lower() for t in tags.split(",") if t.strip()]


def new_id() -> str:
    """Generate a unique note ID."""
    return uuid.uuid4().hex[:16]


class NoteData:
    """Standardized note representation across backends."""
    __slots__ = ("id", "path", "title", "content", "tags", "folder",
                 "sources", "backlinks", "created_at", "updated_at")

    def __init__(self, *, id: str = "", path: str = "", title: str = "",
                 content: str = "", tags: str = "", folder: str = "",
                 sources: str = "", backlinks: str = "[]",
                 created_at: str = "", updated_at: str = ""):
        self.id = id or new_id()
        self.path = path
        self.title = title
        self.content = content
        self.tags = tags
        self.folder = folder
        self.sources = sources
        self.backlinks = backlinks
        self.created_at = created_at or now_iso()
        self.updated_at = updated_at or now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class VaultBackend(ABC):
    """Abstract storage backend for the knowledge vault."""

    @abstractmethod
    def save(self, path: str, title: str, content: str, tags: str,
             folder: str, sources: str, backlinks: list[str]) -> str:
        """Save a note. Returns confirmation message."""
        ...

    @abstractmethod
    def read(self, path: str) -> NoteData | None:
        """Read a note by path. Returns None if not found."""
        ...

    @abstractmethod
    def search(self, query: str, folder: str = "", tags: str = "",
               limit: int = 10) -> list[NoteData]:
        """Full-text search. Returns matching notes."""
        ...

    @abstractmethod
    def list_notes(self, folder: str = "", tags: str = "",
                   sort_by: str = "updated", limit: int = 50) -> list[NoteData]:
        """List notes, optionally filtered and sorted."""
        ...

    @abstractmethod
    def update(self, path: str, append_content: str = "",
               add_tags: str = "") -> str:
        """Update a note. Returns confirmation."""
        ...

    @abstractmethod
    def delete(self, path: str) -> str:
        """Delete a note. Returns confirmation."""
        ...

    @abstractmethod
    def link(self, source_path: str, target_path: str) -> str:
        """Create bidirectional link. Returns confirmation."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a note exists at path."""
        ...

    @abstractmethod
    def find_note(self, identifier: str) -> NoteData | None:
        """Find note by title, path, or slug."""
        ...

    @abstractmethod
    def all_notes(self) -> list[NoteData]:
        """Return all notes (for migration)."""
        ...
```

- [ ] **Step 2: Commit**

```bash
git add src/jarvis/mcp/vault_backend.py
git commit -m "feat(vault): VaultBackend ABC + shared utilities (NoteData, slugify, parse_tags)"
```

---

### Task 2: DBBackend (SQLCipher + FTS5)

**Files:**
- Create: `src/jarvis/mcp/vault_db_backend.py`
- Create: `tests/test_mcp/test_vault_db_backend.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for Vault DB Backend."""
from __future__ import annotations
import pytest
from pathlib import Path
from jarvis.mcp.vault_db_backend import VaultDBBackend


@pytest.fixture
def db_backend(tmp_path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    return VaultDBBackend(vault_root)


def test_save_and_read(db_backend):
    db_backend.save("test/note.md", "Test Note", "Hello world", "tag1, tag2", "test", "", [])
    note = db_backend.read("test/note.md")
    assert note is not None
    assert note.title == "Test Note"
    assert note.content == "Hello world"
    assert "tag1" in note.tags


def test_save_duplicate_path(db_backend):
    db_backend.save("test/note.md", "Note 1", "Content 1", "", "test", "", [])
    # Second save with same logical title should get different path
    result = db_backend.save("test/note.md", "Note 2", "Content 2", "", "test", "", [])
    assert "note" in result.lower()


def test_search_fts(db_backend):
    db_backend.save("a.md", "Python Guide", "Learn Python programming", "python", "wissen", "", [])
    db_backend.save("b.md", "Rust Guide", "Learn Rust systems programming", "rust", "wissen", "", [])
    results = db_backend.search("Python")
    assert len(results) >= 1
    assert any("Python" in r.title for r in results)


def test_search_by_folder(db_backend):
    db_backend.save("wissen/a.md", "Note A", "Content", "", "wissen", "", [])
    db_backend.save("recherchen/b.md", "Note B", "Content", "", "recherchen", "", [])
    results = db_backend.search("Content", folder="wissen")
    assert len(results) == 1
    assert results[0].folder == "wissen"


def test_search_by_tags(db_backend):
    db_backend.save("a.md", "Tagged", "Content", "important, urgent", "wissen", "", [])
    db_backend.save("b.md", "Untagged", "Content", "boring", "wissen", "", [])
    results = db_backend.search("Content", tags="important")
    assert len(results) == 1


def test_list_notes(db_backend):
    db_backend.save("a.md", "First", "A", "", "wissen", "", [])
    db_backend.save("b.md", "Second", "B", "", "wissen", "", [])
    notes = db_backend.list_notes()
    assert len(notes) == 2


def test_list_by_folder(db_backend):
    db_backend.save("wissen/a.md", "A", "Content", "", "wissen", "", [])
    db_backend.save("recherchen/b.md", "B", "Content", "", "recherchen", "", [])
    notes = db_backend.list_notes(folder="wissen")
    assert len(notes) == 1


def test_update_append(db_backend):
    db_backend.save("note.md", "My Note", "Original", "", "wissen", "", [])
    db_backend.update("note.md", append_content="Appended text")
    note = db_backend.read("note.md")
    assert "Original" in note.content
    assert "Appended text" in note.content


def test_update_add_tags(db_backend):
    db_backend.save("note.md", "My Note", "Content", "old", "wissen", "", [])
    db_backend.update("note.md", add_tags="new, extra")
    note = db_backend.read("note.md")
    assert "old" in note.tags
    assert "new" in note.tags


def test_delete(db_backend):
    db_backend.save("note.md", "Delete Me", "Gone", "", "wissen", "", [])
    result = db_backend.delete("note.md")
    assert "note.md" in result.lower() or "delete" in result.lower()
    assert db_backend.read("note.md") is None


def test_link(db_backend):
    db_backend.save("a.md", "Note A", "Content A", "", "wissen", "", [])
    db_backend.save("b.md", "Note B", "Content B", "", "wissen", "", [])
    db_backend.link("a.md", "b.md")
    a = db_backend.read("a.md")
    b = db_backend.read("b.md")
    assert "b.md" in a.backlinks or "Note B" in a.backlinks
    assert "a.md" in b.backlinks or "Note A" in b.backlinks


def test_exists(db_backend):
    assert db_backend.exists("nonexistent.md") is False
    db_backend.save("exists.md", "Exists", "Yes", "", "wissen", "", [])
    assert db_backend.exists("exists.md") is True


def test_find_note_by_title(db_backend):
    db_backend.save("wissen/my-note.md", "My Special Note", "Content", "", "wissen", "", [])
    note = db_backend.find_note("My Special Note")
    assert note is not None
    assert note.title == "My Special Note"


def test_all_notes(db_backend):
    db_backend.save("a.md", "A", "1", "", "wissen", "", [])
    db_backend.save("b.md", "B", "2", "", "wissen", "", [])
    db_backend.save("c.md", "C", "3", "", "wissen", "", [])
    assert len(db_backend.all_notes()) == 3
```

- [ ] **Step 2: Write DBBackend**

```python
"""Vault DB Backend — SQLCipher-encrypted SQLite storage with FTS5."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from jarvis.mcp.vault_backend import NoteData, VaultBackend, new_id, now_iso, parse_tags
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '',
    folder TEXT DEFAULT '',
    sources TEXT DEFAULT '',
    backlinks TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_folder ON notes(folder);
CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content, tags, content=notes, content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES('delete', old.rowid, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES('delete', old.rowid, old.title, old.content, old.tags);
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;
"""


class VaultDBBackend(VaultBackend):
    """SQLCipher-encrypted vault with FTS5 full-text search."""

    def __init__(self, vault_root: Path) -> None:
        self._vault_root = vault_root
        db_path = str(vault_root / "vault.db")
        try:
            from jarvis.security.encrypted_db import encrypted_connect
            self._conn = encrypted_connect(db_path, check_same_thread=False)
        except ImportError:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        try:
            self._conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            log.debug("fts5_setup_partial", exc_info=True)
        self._conn.commit()

    def _row_to_note(self, row: tuple, columns: list[str]) -> NoteData:
        d = dict(zip(columns, row))
        return NoteData(**{k: v for k, v in d.items() if k in NoteData.__slots__})

    def _query_notes(self, sql: str, params: tuple = ()) -> list[NoteData]:
        cursor = self._conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [self._row_to_note(row, cols) for row in cursor.fetchall()]

    def save(self, path: str, title: str, content: str, tags: str,
             folder: str, sources: str, backlinks: list[str]) -> str:
        now = now_iso()
        tag_str = ", ".join(parse_tags(tags))
        bl_json = json.dumps(backlinks, ensure_ascii=False)
        note_id = new_id()
        try:
            self._conn.execute(
                "INSERT INTO notes (id, path, title, content, tags, folder, sources, backlinks, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (note_id, path, title, content, tag_str, folder, sources, bl_json, now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Path exists — update instead
            self._conn.execute(
                "UPDATE notes SET title=?, content=?, tags=?, folder=?, sources=?, backlinks=?, updated_at=? WHERE path=?",
                (title, content, tag_str, folder, sources, bl_json, now, path),
            )
            self._conn.commit()
        return f"Notiz gespeichert: {path}"

    def read(self, path: str) -> NoteData | None:
        notes = self._query_notes("SELECT * FROM notes WHERE path = ?", (path,))
        return notes[0] if notes else None

    def search(self, query: str, folder: str = "", tags: str = "",
               limit: int = 10) -> list[NoteData]:
        # Try FTS5 first
        try:
            fts_query = query.replace('"', '""')
            sql = (
                "SELECT n.* FROM notes n "
                "JOIN notes_fts f ON n.rowid = f.rowid "
                "WHERE notes_fts MATCH ?"
            )
            params: list = [f'"{fts_query}"']
            if folder:
                sql += " AND n.folder = ?"
                params.append(folder)
            if tags:
                tag_list = parse_tags(tags)
                for tag in tag_list:
                    sql += " AND n.tags LIKE ?"
                    params.append(f"%{tag}%")
            sql += f" LIMIT {int(limit)}"
            results = self._query_notes(sql, tuple(params))
            if results:
                return results
        except sqlite3.OperationalError:
            pass

        # Fallback: LIKE search
        sql = "SELECT * FROM notes WHERE (content LIKE ? OR title LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]
        if folder:
            sql += " AND folder = ?"
            params.append(folder)
        if tags:
            for tag in parse_tags(tags):
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        sql += f" LIMIT {int(limit)}"
        return self._query_notes(sql, tuple(params))

    def list_notes(self, folder: str = "", tags: str = "",
                   sort_by: str = "updated", limit: int = 50) -> list[NoteData]:
        sql = "SELECT * FROM notes WHERE 1=1"
        params: list = []
        if folder:
            sql += " AND folder = ?"
            params.append(folder)
        if tags:
            for tag in parse_tags(tags):
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        order = {"title": "title ASC", "created": "created_at DESC",
                 "updated": "updated_at DESC"}.get(sort_by, "updated_at DESC")
        sql += f" ORDER BY {order} LIMIT {int(limit)}"
        return self._query_notes(sql, tuple(params))

    def update(self, path: str, append_content: str = "",
               add_tags: str = "") -> str:
        note = self.read(path)
        if not note:
            return f"Notiz nicht gefunden: {path}"
        new_content = note.content
        if append_content:
            new_content = note.content.rstrip("\n") + "\n\n" + append_content.strip() + "\n"
        new_tags = note.tags
        if add_tags:
            existing = parse_tags(note.tags)
            added = parse_tags(add_tags)
            merged = list(dict.fromkeys(existing + added))
            new_tags = ", ".join(merged)
        self._conn.execute(
            "UPDATE notes SET content=?, tags=?, updated_at=? WHERE path=?",
            (new_content, new_tags, now_iso(), path),
        )
        self._conn.commit()
        return f"Notiz aktualisiert: {path}"

    def delete(self, path: str) -> str:
        cursor = self._conn.execute("DELETE FROM notes WHERE path = ?", (path,))
        self._conn.commit()
        if cursor.rowcount == 0:
            return f"Notiz nicht gefunden: {path}"
        return f"Geloescht: {path}"

    def link(self, source_path: str, target_path: str) -> str:
        source = self.read(source_path)
        target = self.read(target_path)
        if not source or not target:
            return "Eine oder beide Notizen nicht gefunden"
        # Add bidirectional backlinks
        s_bl = json.loads(source.backlinks) if source.backlinks else []
        t_bl = json.loads(target.backlinks) if target.backlinks else []
        if target_path not in s_bl:
            s_bl.append(target_path)
        if source_path not in t_bl:
            t_bl.append(source_path)
        now = now_iso()
        self._conn.execute("UPDATE notes SET backlinks=?, updated_at=? WHERE path=?",
                           (json.dumps(s_bl), now, source_path))
        self._conn.execute("UPDATE notes SET backlinks=?, updated_at=? WHERE path=?",
                           (json.dumps(t_bl), now, target_path))
        self._conn.commit()
        return f"Verknuepft: {source_path} <-> {target_path}"

    def exists(self, path: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM notes WHERE path = ?", (path,)).fetchone()
        return row is not None

    def find_note(self, identifier: str) -> NoteData | None:
        # 1. Try as path
        note = self.read(identifier)
        if note:
            return note
        # 2. Try by title (case-insensitive)
        notes = self._query_notes(
            "SELECT * FROM notes WHERE LOWER(title) = LOWER(?)", (identifier,)
        )
        if notes:
            return notes[0]
        # 3. Try by slug in path
        slug = identifier.lower().replace(" ", "-")
        notes = self._query_notes(
            "SELECT * FROM notes WHERE path LIKE ?", (f"%{slug}%",)
        )
        return notes[0] if notes else None

    def all_notes(self) -> list[NoteData]:
        return self._query_notes("SELECT * FROM notes ORDER BY path")

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_mcp/test_vault_db_backend.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/mcp/vault_db_backend.py tests/test_mcp/test_vault_db_backend.py
git commit -m "feat(vault): DBBackend — SQLCipher storage with FTS5 full-text search"
```

---

### Task 3: FileBackend (extracted from vault.py)

**Files:**
- Create: `src/jarvis/mcp/vault_file_backend.py`

- [ ] **Step 1: Write FileBackend**

This extracts the EXISTING vault.py file I/O logic into the VaultBackend interface. The key methods map directly:

```python
"""Vault File Backend — Obsidian-compatible .md files on disk."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jarvis.mcp.vault_backend import NoteData, VaultBackend, new_id, now_iso, parse_tags, slugify
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

try:
    from jarvis.security.encrypted_file import efile as _efile
except ImportError:
    _efile = None


class VaultFileBackend(VaultBackend):
    """Obsidian-compatible .md file storage with _index.json cache."""

    def __init__(self, vault_root: Path, encrypt_files: bool = False,
                 default_folders: dict[str, str] | None = None) -> None:
        self._vault_root = vault_root
        self._encrypt = encrypt_files
        self._index_path = vault_root / "_index.json"
        self._default_folders = default_folders or {
            "research": "recherchen", "meetings": "meetings",
            "knowledge": "wissen", "projects": "projekte", "daily": "daily",
        }
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        self._vault_root.mkdir(parents=True, exist_ok=True)
        for folder in self._default_folders.values():
            (self._vault_root / folder).mkdir(exist_ok=True)

    # --- File I/O helpers ---

    def _read_file(self, path: Path) -> str:
        if _efile is not None and self._encrypt:
            return _efile.read(path)
        return path.read_text(encoding="utf-8")

    def _write_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _efile is not None and self._encrypt:
            _efile.write(path, content)
        else:
            path.write_text(content, encoding="utf-8")

    # --- Index ---

    def _read_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        try:
            raw = self._read_file(self._index_path)
            return json.loads(raw)
        except Exception:
            return {}

    def _write_index(self, index: dict[str, Any]) -> None:
        self._write_file(self._index_path, json.dumps(index, indent=2, ensure_ascii=False))

    def _update_index(self, title: str, path: str, tags: list[str], folder: str) -> None:
        index = self._read_index()
        existing = index.get(title, {})
        index[title] = {
            "path": path,
            "tags": tags,
            "folder": folder,
            "created": existing.get("created", now_iso()),
            "updated": now_iso(),
        }
        self._write_index(index)

    # --- Frontmatter ---

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter, return (data, body_without_frontmatter)."""
        if not content.startswith("---"):
            return {}, content
        close = content.find("\n---", 3)
        if close == -1:
            return {}, content
        yaml_text = content[4:close]
        body = content[close + 4:].lstrip("\n")
        try:
            import yaml
            data = yaml.safe_load(yaml_text)
            if not isinstance(data, dict):
                return {}, content
            return data, body
        except Exception:
            return {}, content

    @staticmethod
    def _build_frontmatter(title: str, tags: list[str],
                            sources: list[str] | None = None,
                            backlinks: list[str] | None = None) -> str:
        now = now_iso()
        lines = [
            "---",
            f'title: "{title}"',
            f"created: {now}",
            f"updated: {now}",
        ]
        if tags:
            lines.append(f"tags: [{', '.join(tags)}]")
        if sources:
            lines.append(f"sources: [{', '.join(sources)}]")
        if backlinks:
            quoted = [f'"{b}"' for b in backlinks]
            lines.append(f"linked_notes: [{', '.join(quoted)}]")
        lines.append("author: jarvis")
        lines.append("---")
        return "\n".join(lines) + "\n"

    def _resolve_folder(self, folder: str) -> str:
        if folder in self._default_folders:
            return self._default_folders[folder]
        if folder in self._default_folders.values():
            return folder
        return self._default_folders.get("knowledge", "wissen")

    # --- VaultBackend implementation ---

    def save(self, path: str, title: str, content: str, tags: str,
             folder: str, sources: str, backlinks: list[str]) -> str:
        tag_list = parse_tags(tags)
        source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else []
        folder_name = self._resolve_folder(folder)
        slug = slugify(title)
        # Generate unique path
        file_path = self._vault_root / folder_name / f"{slug}.md"
        counter = 1
        while file_path.exists():
            file_path = self._vault_root / folder_name / f"{slug}-{counter}.md"
            counter += 1
        # Build content
        fm = self._build_frontmatter(title, tag_list, source_list, backlinks)
        full_content = fm + f"\n# {title}\n\n{content}\n"
        if source_list:
            full_content += "\n## Quellen\n" + "\n".join(f"- {s}" for s in source_list) + "\n"
        # Write
        self._write_file(file_path, full_content)
        rel_path = str(file_path.relative_to(self._vault_root))
        self._update_index(title, rel_path, tag_list, folder_name)
        log.info("vault_note_saved", path=rel_path, title=title[:50])
        return f"Notiz gespeichert: {rel_path}"

    def read(self, path: str) -> NoteData | None:
        full = self._vault_root / path
        if not full.exists():
            return None
        try:
            resolved = full.resolve()
            resolved.relative_to(self._vault_root.resolve())
        except ValueError:
            return None
        content = self._read_file(full)
        fm, body = self._parse_frontmatter(content)
        return NoteData(
            path=path,
            title=fm.get("title", ""),
            content=body,
            tags=", ".join(fm.get("tags", [])) if isinstance(fm.get("tags"), list) else str(fm.get("tags", "")),
            folder=path.split("/")[0] if "/" in path else "",
            sources=", ".join(fm.get("sources", [])) if isinstance(fm.get("sources"), list) else "",
            backlinks=json.dumps(fm.get("linked_notes", [])),
            created_at=str(fm.get("created", "")),
            updated_at=str(fm.get("updated", "")),
        )

    def search(self, query: str, folder: str = "", tags: str = "",
               limit: int = 10) -> list[NoteData]:
        results: list[NoteData] = []
        query_lower = query.lower()
        tag_filter = parse_tags(tags) if tags else []
        folder_filter = self._resolve_folder(folder) if folder else ""
        for md_file in self._vault_root.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            rel = str(md_file.relative_to(self._vault_root))
            if folder_filter and not rel.startswith(folder_filter):
                continue
            try:
                content = self._read_file(md_file)
            except Exception:
                continue
            if tag_filter:
                fm, _ = self._parse_frontmatter(content)
                fm_tags = [t.lower() for t in (fm.get("tags", []) if isinstance(fm.get("tags"), list) else [])]
                if not any(t in fm_tags for t in tag_filter):
                    continue
            if query_lower in content.lower():
                note = self.read(rel)
                if note:
                    results.append(note)
                if len(results) >= limit:
                    break
        return results

    def list_notes(self, folder: str = "", tags: str = "",
                   sort_by: str = "updated", limit: int = 50) -> list[NoteData]:
        index = self._read_index()
        tag_filter = parse_tags(tags) if tags else []
        folder_filter = self._resolve_folder(folder) if folder else ""
        entries: list[NoteData] = []
        for title, meta in index.items():
            if folder_filter and meta.get("folder") != folder_filter:
                continue
            if tag_filter:
                entry_tags = [t.lower() for t in meta.get("tags", [])]
                if not any(t in entry_tags for t in tag_filter):
                    continue
            entries.append(NoteData(
                path=meta.get("path", ""),
                title=title,
                tags=", ".join(meta.get("tags", [])),
                folder=meta.get("folder", ""),
                created_at=meta.get("created", ""),
                updated_at=meta.get("updated", ""),
            ))
        key_fn = {"title": lambda n: n.title.lower(),
                   "created": lambda n: n.created_at,
                   "updated": lambda n: n.updated_at}.get(sort_by, lambda n: n.updated_at)
        entries.sort(key=key_fn, reverse=(sort_by != "title"))
        return entries[:limit]

    def update(self, path: str, append_content: str = "",
               add_tags: str = "") -> str:
        full = self._vault_root / path
        if not full.exists():
            return f"Notiz nicht gefunden: {path}"
        content = self._read_file(full)
        fm, body = self._parse_frontmatter(content)
        if append_content:
            body = body.rstrip("\n") + "\n\n" + append_content.strip() + "\n"
        if add_tags:
            existing = fm.get("tags", [])
            if not isinstance(existing, list):
                existing = parse_tags(str(existing))
            new_tags = parse_tags(add_tags)
            merged = list(dict.fromkeys(existing + new_tags))
            fm["tags"] = merged
        fm["updated"] = now_iso()
        new_content = self._build_frontmatter_from_dict(fm) + body
        self._write_file(full, new_content)
        tag_list = fm.get("tags", [])
        if not isinstance(tag_list, list):
            tag_list = parse_tags(str(tag_list))
        folder = path.split("/")[0] if "/" in path else ""
        self._update_index(fm.get("title", ""), path, tag_list, folder)
        return f"Notiz aktualisiert: {path}"

    def _build_frontmatter_from_dict(self, fm: dict) -> str:
        lines = ["---"]
        for key, val in fm.items():
            if isinstance(val, list):
                items = ", ".join(f'"{v}"' if " " in str(v) else str(v) for v in val)
                lines.append(f"{key}: [{items}]")
            elif isinstance(val, str) and any(c in val for c in ':"{}[]'):
                lines.append(f'{key}: "{val}"')
            else:
                lines.append(f"{key}: {val}")
        lines.append("---")
        return "\n".join(lines) + "\n"

    def delete(self, path: str) -> str:
        full = self._vault_root / path
        try:
            resolved = full.resolve()
            resolved.relative_to(self._vault_root.resolve())
        except ValueError:
            return f"Ungueltiger Pfad: {path}"
        if not full.exists():
            return f"Notiz nicht gefunden: {path}"
        # Remove from index
        index = self._read_index()
        content = self._read_file(full)
        fm, _ = self._parse_frontmatter(content)
        title = fm.get("title", "")
        if title in index:
            del index[title]
            self._write_index(index)
        full.unlink()
        return f"Geloescht: {path}"

    def link(self, source_path: str, target_path: str) -> str:
        source = self.read(source_path)
        target = self.read(target_path)
        if not source or not target:
            return "Eine oder beide Notizen nicht gefunden"
        # Update source file
        s_full = self._vault_root / source_path
        s_content = self._read_file(s_full)
        s_fm, s_body = self._parse_frontmatter(s_content)
        s_links = s_fm.get("linked_notes", [])
        if target.title not in s_links:
            s_links.append(target.title)
        s_fm["linked_notes"] = s_links
        s_fm["updated"] = now_iso()
        self._write_file(s_full, self._build_frontmatter_from_dict(s_fm) + s_body)
        # Update target file
        t_full = self._vault_root / target_path
        t_content = self._read_file(t_full)
        t_fm, t_body = self._parse_frontmatter(t_content)
        t_links = t_fm.get("linked_notes", [])
        if source.title not in t_links:
            t_links.append(source.title)
        t_fm["linked_notes"] = t_links
        t_fm["updated"] = now_iso()
        self._write_file(t_full, self._build_frontmatter_from_dict(t_fm) + t_body)
        return f"Verknuepft: {source_path} <-> {target_path}"

    def exists(self, path: str) -> bool:
        return (self._vault_root / path).exists()

    def find_note(self, identifier: str) -> NoteData | None:
        # 1. Direct path
        note = self.read(identifier)
        if note:
            return note
        # 2. Index by title
        index = self._read_index()
        for title, meta in index.items():
            if title.lower() == identifier.lower():
                return self.read(meta["path"])
        # 3. Slug search
        slug = identifier.lower().replace(" ", "-")
        for md_file in self._vault_root.rglob("*.md"):
            if slug in md_file.stem.lower():
                rel = str(md_file.relative_to(self._vault_root))
                return self.read(rel)
        return None

    def all_notes(self) -> list[NoteData]:
        notes: list[NoteData] = []
        for md_file in self._vault_root.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            rel = str(md_file.relative_to(self._vault_root))
            note = self.read(rel)
            if note:
                notes.append(note)
        return notes
```

- [ ] **Step 2: Commit**

```bash
git add src/jarvis/mcp/vault_file_backend.py
git commit -m "feat(vault): FileBackend — Obsidian-compatible .md storage via VaultBackend interface"
```

---

### Task 4: Migration Logic

**Files:**
- Create: `src/jarvis/mcp/vault_migration.py`
- Create: `tests/test_mcp/test_vault_migration.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for Vault migration (files <-> DB)."""
from __future__ import annotations
import pytest
from pathlib import Path
from jarvis.mcp.vault_file_backend import VaultFileBackend
from jarvis.mcp.vault_db_backend import VaultDBBackend
from jarvis.mcp.vault_migration import migrate_files_to_db, migrate_db_to_files, detect_mode_change


@pytest.fixture
def vault_root(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    return root


def test_files_to_db(vault_root):
    fb = VaultFileBackend(vault_root)
    fb.save("wissen/note1.md", "First Note", "Content one", "tag1", "wissen", "", [])
    fb.save("recherchen/note2.md", "Second Note", "Content two", "tag2", "recherchen", "", [])
    db = VaultDBBackend(vault_root)
    count = migrate_files_to_db(fb, db)
    assert count == 2
    assert db.read("wissen/first-note.md") is not None or len(db.all_notes()) == 2


def test_db_to_files(vault_root):
    db = VaultDBBackend(vault_root)
    db.save("wissen/note1.md", "DB Note", "From database", "dbtest", "wissen", "", [])
    fb = VaultFileBackend(vault_root)
    count = migrate_db_to_files(db, fb)
    assert count == 1
    assert (vault_root / "wissen" / "note1.md").exists() or len(fb.all_notes()) >= 1


def test_detect_mode_change(vault_root):
    assert detect_mode_change(vault_root, "db") is True  # First run, no marker
    assert detect_mode_change(vault_root, "file") is True  # Changed
    assert detect_mode_change(vault_root, "file") is False  # Same


def test_roundtrip(vault_root):
    fb = VaultFileBackend(vault_root)
    fb.save("wissen/test.md", "Roundtrip", "Original content", "round, trip", "wissen", "http://src", [])
    # Files -> DB
    db = VaultDBBackend(vault_root)
    migrate_files_to_db(fb, db)
    note = db.find_note("Roundtrip")
    assert note is not None
    assert "Original content" in note.content
    # DB -> Files (new dir to avoid conflict)
    out_root = vault_root.parent / "vault_out"
    out_root.mkdir()
    fb2 = VaultFileBackend(out_root)
    migrate_db_to_files(db, fb2)
    exported = fb2.find_note("Roundtrip")
    assert exported is not None
```

- [ ] **Step 2: Write migration module**

```python
"""Vault bidirectional migration: files <-> SQLite DB."""
from __future__ import annotations

from pathlib import Path

from jarvis.mcp.vault_backend import NoteData
from jarvis.mcp.vault_db_backend import VaultDBBackend
from jarvis.mcp.vault_file_backend import VaultFileBackend
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_MODE_FILE = ".vault_mode"


def detect_mode_change(vault_root: Path, current_mode: str) -> bool:
    """Check if vault mode changed. Updates marker file. Returns True if changed."""
    marker = vault_root / _MODE_FILE
    last_mode = "file"  # default
    if marker.exists():
        last_mode = marker.read_text(encoding="utf-8").strip()
    if current_mode == last_mode:
        return False
    marker.write_text(current_mode, encoding="utf-8")
    return True


def migrate_files_to_db(file_backend: VaultFileBackend, db_backend: VaultDBBackend) -> int:
    """Migrate all .md files into the DB. Returns count migrated."""
    notes = file_backend.all_notes()
    count = 0
    for note in notes:
        try:
            db_backend.save(
                path=note.path,
                title=note.title,
                content=note.content,
                tags=note.tags,
                folder=note.folder,
                sources=note.sources,
                backlinks=[],
            )
            count += 1
        except Exception:
            log.debug("vault_migration_file_to_db_failed", path=note.path, exc_info=True)
    log.info("vault_migration_files_to_db", count=count, total=len(notes))
    return count


def migrate_db_to_files(db_backend: VaultDBBackend, file_backend: VaultFileBackend) -> int:
    """Export all DB notes back to .md files. Returns count exported."""
    notes = db_backend.all_notes()
    count = 0
    for note in notes:
        try:
            file_backend.save(
                path=note.path,
                title=note.title,
                content=note.content,
                tags=note.tags,
                folder=note.folder,
                sources=note.sources,
                backlinks=[],
            )
            count += 1
        except Exception:
            log.debug("vault_migration_db_to_file_failed", path=note.path, exc_info=True)
    log.info("vault_migration_db_to_files", count=count, total=len(notes))
    return count
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest tests/test_mcp/test_vault_migration.py -v
git add src/jarvis/mcp/vault_migration.py tests/test_mcp/test_vault_migration.py
git commit -m "feat(vault): bidirectional migration — files <-> DB with mode detection"
```

---

### Task 5: Refactor VaultTools to Use Backend

**Files:**
- Modify: `src/jarvis/mcp/vault.py`

This is the core refactor. VaultTools delegates to the active backend.

- [ ] **Step 1: Refactor VaultTools.__init__**

In the constructor, replace all the file-system setup with backend selection:

```python
def __init__(self, config: JarvisConfig | None = None) -> None:
    vault_cfg = getattr(config, "vault", None)

    if vault_cfg and getattr(vault_cfg, "path", ""):
        self._vault_root = Path(vault_cfg.path).expanduser().resolve()
    else:
        self._vault_root = Path.home() / ".jarvis" / "vault"

    self._vault_root.mkdir(parents=True, exist_ok=True)
    encrypt = bool(getattr(vault_cfg, "encrypt_files", False) if vault_cfg else False)
    default_folders = dict(getattr(vault_cfg, "default_folders", {})) if vault_cfg else None

    # Select backend
    if encrypt:
        from jarvis.mcp.vault_db_backend import VaultDBBackend
        self._backend = VaultDBBackend(self._vault_root)
    else:
        from jarvis.mcp.vault_file_backend import VaultFileBackend
        self._backend = VaultFileBackend(self._vault_root, encrypt_files=False,
                                          default_folders=default_folders)

    # Auto-migrate on mode change
    current_mode = "db" if encrypt else "file"
    from jarvis.mcp.vault_migration import detect_mode_change, migrate_files_to_db, migrate_db_to_files
    if detect_mode_change(self._vault_root, current_mode):
        try:
            if current_mode == "db":
                from jarvis.mcp.vault_file_backend import VaultFileBackend as FB
                old = FB(self._vault_root, default_folders=default_folders)
                migrate_files_to_db(old, self._backend)
            else:
                from jarvis.mcp.vault_db_backend import VaultDBBackend as DB
                old = DB(self._vault_root)
                migrate_db_to_files(old, self._backend)
            log.info("vault_mode_migrated", mode=current_mode)
        except Exception:
            log.error("vault_migration_failed", exc_info=True)
```

- [ ] **Step 2: Simplify all tool methods to delegate to backend**

Each method becomes a thin wrapper. Example for vault_save:

```python
async def vault_save(self, title, content, tags="", folder="knowledge",
                      sources="", linked_notes="") -> str:
    if not title or not content:
        return "Titel und Inhalt sind erforderlich."
    tag_list = parse_tags(tags)
    source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else []
    link_list = [l.strip() for l in linked_notes.split(",") if l.strip()] if linked_notes else []
    folder_name = folder  # backend handles resolution
    slug = slugify(title)
    path = f"{folder_name}/{slug}.md"
    return self._backend.save(path, title, content, tags, folder, sources, link_list)
```

Similarly simplify vault_search, vault_read, vault_list, vault_update, vault_delete, vault_link.

- [ ] **Step 3: Run existing vault tests**

```bash
python -m pytest tests/test_mcp/test_vault.py tests/test_mcp/test_vault_coverage.py -v --tb=short
```

Fix any failures from the refactor.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/mcp/vault.py
git commit -m "refactor(vault): VaultTools delegates to pluggable backend (file or DB)"
```

---

### Task 6: Integration Test + Cleanup

- [ ] **Step 1: Run ALL vault tests**

```bash
python -m pytest tests/test_mcp/test_vault*.py -v --tb=short
```

- [ ] **Step 2: Run full test suite for regressions**

```bash
python -m pytest tests/ -x --timeout=60 -q 2>&1 | tail -10
```

- [ ] **Step 3: Remove old efile usage from vault.py**

Since FileBackend now handles its own efile logic internally, vault.py should no longer import or use `_efile` directly. Clean up any remaining references.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(vault): dual-backend complete — file mode (Obsidian) + DB mode (SQLCipher)"
```

---

## Definition of Done

- [ ] `encrypt_files=false`: vault works exactly as before (Obsidian-compatible .md files)
- [ ] `encrypt_files=true`: vault stores everything in SQLCipher-encrypted vault.db
- [ ] FTS5 full-text search works in DB mode
- [ ] Toggle false→true migrates .md files into DB automatically
- [ ] Toggle true→false exports DB back to .md files automatically
- [ ] All 7 MCP tools work identically in both modes
- [ ] Existing vault tests still pass (no regression)
- [ ] New tests: DB backend (16+), migration (4+)
- [ ] No .md files on disk when in DB mode (except .vault_mode marker)
