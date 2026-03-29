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
    fb.save(
        "wissen/test.md", "Roundtrip", "Original content", "round, trip", "wissen", "http://src", []
    )
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
