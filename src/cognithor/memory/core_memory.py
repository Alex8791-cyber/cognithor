"""Core Memory · Tier 1 -- Identity, rules, preferences. [B§4.2]

ALWAYS loaded. In every session. Completely.
Changes only by user or explicit command.
No recency decay.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from cognithor.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from cognithor.security.encrypted_file import efile as _efile
except ImportError:  # encryption module not available
    _efile = None  # type: ignore[assignment]


class CoreMemory:
    """Manage the CORE.md file -- Jarvis' identity.

    Source of Truth: ~/.cognithor/memory/CORE.md
    """

    def __init__(self, core_file: str | Path) -> None:
        """Initialize CoreMemory with the path to CORE.md."""
        self._path = Path(core_file)
        self._content: str = ""
        self._sections: dict[str, str] = {}

    @property
    def path(self) -> Path:
        """Return the path to CORE.md."""
        return self._path

    @property
    def content(self) -> str:
        """Complete CORE.md content."""
        return self._content

    @property
    def sections(self) -> dict[str, str]:
        """Parsed sections as {header: content}."""
        return dict(self._sections)

    def load(self) -> str:
        """Load CORE.md from disk. Create default if not found.

        Returns:
            Complete content as string.
        """
        if not self._path.exists():
            self._content = ""
            self._sections = {}
            return ""

        if _efile is not None:
            self._content = _efile.read(self._path)
            # Opportunistic migration: if the file was plaintext and encryption
            # is available, silently encrypt it in-place on first read.
            if _efile.is_available and not _efile.is_encrypted(self._path):
                with contextlib.suppress(Exception):
                    _efile.migrate(self._path)
        else:
            self._content = self._path.read_text(encoding="utf-8")
        self._sections = self._parse_sections(self._content)
        return self._content

    def get_section(self, name: str) -> str:
        """Return the content of a section.

        Args:
            name: Section name (case-insensitive, without '#').

        Returns:
            Section content or empty string.
        """
        name_lower = name.lower().strip()
        for key, value in self._sections.items():
            if key.lower().strip() == name_lower:
                return value
        return ""

    # CORE.md should never exceed this size. A healthy CORE.md is 1-50 KB.
    # If it grows beyond this, something is appending in a loop.
    _MAX_CORE_BYTES = 1_000_000  # 1 MB

    def save(
        self,
        content: str | None = None,
        *,
        provenance_source_type: str | None = None,
        provenance_source_id: str | None = None,
        provenance_notes: str = "",
    ) -> None:
        """Save CORE.md to disk.

        Args:
            content: New content. If None, current content is saved.
            provenance_source_type: Optional TRUST-9 ``SourceType``
                value. When set together with ``provenance_source_id``,
                a tag is written to the canonical
                ``PROVENANCE_LEDGER`` keyed by
                ``"core_memory:" + path.name``.
            provenance_source_id: Stable upstream id (audit-log entry
                id, message id, …). Required when
                ``provenance_source_type`` is set.
            provenance_notes: Short audit-log breadcrumb. MUST NOT
                contain prompt / response content.
        """
        if content is not None:
            self._content = content
            self._sections = self._parse_sections(content)

        # Guard: prevent runaway growth that killed 24 GB of RAM
        size = len(self._content.encode("utf-8"))
        if size > self._MAX_CORE_BYTES:
            logger.error(
                "core_memory_too_large",
                size_bytes=size,
                max_bytes=self._MAX_CORE_BYTES,
                hint="CORE.md growth blocked — possible append loop",
            )
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        if _efile is not None:
            _efile.write(self._path, self._content)
        else:
            self._path.write_text(self._content, encoding="utf-8")

        if provenance_source_type and provenance_source_id:
            self._tag_provenance(
                item_id=f"core_memory:{self._path.name}",
                source_type_value=provenance_source_type,
                source_id=provenance_source_id,
                notes=provenance_notes,
            )

    @staticmethod
    def _tag_provenance(
        *,
        item_id: str,
        source_type_value: str,
        source_id: str,
        notes: str,
    ) -> None:
        """Best-effort TRUST-9 provenance tag for a CORE.md write.

        Failures are silently swallowed — saving CORE.md MUST NEVER
        fail because of provenance tagging. Unknown
        ``source_type_value`` is coerced to
        :data:`SourceType.UNKNOWN`.
        """
        from cognithor.memory.provenance import (
            PROVENANCE_LEDGER,
            ProvenanceTag,
            SourceType,
        )

        try:
            try:
                source_type = SourceType(source_type_value)
            except ValueError:
                source_type = SourceType.UNKNOWN
            PROVENANCE_LEDGER.tag(
                item_id,
                ProvenanceTag(
                    source_type=source_type,
                    source_id=source_id,
                    notes=notes,
                ),
            )
        except ValueError:
            pass

    def create_default(self) -> str:
        """Create a default CORE.md and return its content."""
        default = (
            "# Identität\n"
            "Ich bin Jarvis, ein lokaler AI-Assistent.\n\n"
            "# Regeln\n"
            "- Kundendaten NIEMALS in Logs schreiben\n"
            "- E-Mails IMMER zur Bestätigung vorlegen\n\n"
            "# Präferenzen\n"
            "- Codesprache: Python\n"
            "- Kommunikation: Direkt, keine Floskeln\n"
            "- Zeitzone: Europe/Berlin\n"
        )
        self.save(default)
        return default

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str]:
        """Parse Markdown into sections based on H1/H2 headers.

        Returns:
            Dict of {header_name: content_text}.
        """
        sections: dict[str, str] = {}
        current_header: str | None = None
        current_lines: list[str] = []

        for line in text.split("\n"):
            match = re.match(r"^(#{1,2})\s+(.+)$", line)
            if match:
                # Save previous section
                if current_header is not None:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Last section
        if current_header is not None:
            sections[current_header] = "\n".join(current_lines).strip()

        return sections
