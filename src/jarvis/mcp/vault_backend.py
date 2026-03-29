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
    for src, dst in [
        ("ae", "ae"),
        ("oe", "oe"),
        ("ue", "ue"),
        ("ss", "ss"),
        ("\u00e4", "ae"),
        ("\u00f6", "oe"),
        ("\u00fc", "ue"),
        ("\u00df", "ss"),
    ]:
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

    __slots__ = (
        "id",
        "path",
        "title",
        "content",
        "tags",
        "folder",
        "sources",
        "backlinks",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        *,
        id: str = "",
        path: str = "",
        title: str = "",
        content: str = "",
        tags: str = "",
        folder: str = "",
        sources: str = "",
        backlinks: str = "[]",
        created_at: str = "",
        updated_at: str = "",
    ):
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
    def save(
        self,
        path: str,
        title: str,
        content: str,
        tags: str,
        folder: str,
        sources: str,
        backlinks: list[str],
    ) -> str:
        """Save a note. Returns confirmation message."""
        ...

    @abstractmethod
    def read(self, path: str) -> NoteData | None:
        """Read a note by path. Returns None if not found."""
        ...

    @abstractmethod
    def search(
        self, query: str, folder: str = "", tags: str = "", limit: int = 10
    ) -> list[NoteData]:
        """Full-text search. Returns matching notes."""
        ...

    @abstractmethod
    def list_notes(
        self, folder: str = "", tags: str = "", sort_by: str = "updated", limit: int = 50
    ) -> list[NoteData]:
        """List notes, optionally filtered and sorted."""
        ...

    @abstractmethod
    def update(self, path: str, append_content: str = "", add_tags: str = "") -> str:
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
