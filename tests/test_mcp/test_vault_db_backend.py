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
