"""Tests for encrypted file I/O."""

from __future__ import annotations

import os
import pytest
from jarvis.security.encrypted_file import EncryptedFileIO, _MAGIC_HEADER


@pytest.fixture
def eio(tmp_path, monkeypatch):
    """EncryptedFileIO with a test key."""
    monkeypatch.setenv("JARVIS_DB_KEY", "test_key_for_encryption_1234567890abcdef")
    io = EncryptedFileIO()
    io._initialized = False  # Force re-init with new env
    io._fernet = None
    return io


@pytest.fixture
def eio_no_key(tmp_path, monkeypatch):
    """EncryptedFileIO without a key."""
    monkeypatch.delenv("JARVIS_DB_KEY", raising=False)
    io = EncryptedFileIO()
    io._initialized = False
    io._fernet = None
    # Force no key by breaking keyring + credential store
    io._get_key = lambda: ""
    io._initialized = False
    return io


def test_write_read_encrypted(eio, tmp_path):
    path = tmp_path / "note.md"
    eio.write(path, "# Secret Research\nSensitive data here.")

    # File should have magic header
    with open(path, "rb") as f:
        assert f.read(len(_MAGIC_HEADER)) == _MAGIC_HEADER

    # Read should decrypt
    content = eio.read(path)
    assert content == "# Secret Research\nSensitive data here."


def test_encrypted_file_unreadable_as_text(eio, tmp_path):
    path = tmp_path / "note.md"
    eio.write(path, "Secret content")

    # Raw file content should NOT contain the plaintext
    with open(path, "rb") as f:
        raw = f.read()
    assert b"Secret content" not in raw


def test_read_plaintext_fallback(eio, tmp_path):
    path = tmp_path / "plain.md"
    with open(path, "w") as f:
        f.write("# Plain Markdown\nNo encryption.")

    # Should read plaintext normally
    content = eio.read(path)
    assert content == "# Plain Markdown\nNo encryption."


def test_is_encrypted(eio, tmp_path):
    enc_path = tmp_path / "encrypted.md"
    plain_path = tmp_path / "plain.md"

    eio.write(enc_path, "encrypted content")
    with open(plain_path, "w") as f:
        f.write("plain content")

    assert eio.is_encrypted(enc_path) is True
    assert eio.is_encrypted(plain_path) is False


def test_migrate_plaintext_to_encrypted(eio, tmp_path):
    path = tmp_path / "migrate_me.md"
    with open(path, "w") as f:
        f.write("Original plaintext content")

    assert eio.is_encrypted(path) is False
    result = eio.migrate(path)
    assert result is True
    assert eio.is_encrypted(path) is True

    # Content should still be readable
    content = eio.read(path)
    assert content == "Original plaintext content"


def test_migrate_already_encrypted_noop(eio, tmp_path):
    path = tmp_path / "already.md"
    eio.write(path, "Already encrypted")

    result = eio.migrate(path)
    assert result is False  # Already encrypted


def test_migrate_directory(eio, tmp_path):
    subdir = tmp_path / "vault" / "wissen"
    subdir.mkdir(parents=True)

    for i in range(5):
        with open(subdir / f"note_{i}.md", "w") as f:
            f.write(f"Content {i}")

    count = eio.migrate_directory(subdir, "*.md")
    assert count == 5

    # All should now be encrypted
    for i in range(5):
        assert eio.is_encrypted(subdir / f"note_{i}.md")
        assert eio.read(subdir / f"note_{i}.md") == f"Content {i}"


def test_file_not_found_raises(eio, tmp_path):
    with pytest.raises(FileNotFoundError):
        eio.read(tmp_path / "nonexistent.md")


def test_write_without_key_writes_plaintext(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_DB_KEY", raising=False)
    io = EncryptedFileIO()
    io._initialized = True
    io._fernet = None  # No key

    path = tmp_path / "unencrypted.md"
    io.write(path, "Plaintext content")

    with open(path, "r") as f:
        assert f.read() == "Plaintext content"
