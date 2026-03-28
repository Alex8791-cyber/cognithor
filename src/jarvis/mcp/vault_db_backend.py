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
