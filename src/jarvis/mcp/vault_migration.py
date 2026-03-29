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
