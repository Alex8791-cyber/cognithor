"""Knowledge Vault — Obsidian-kompatibles Markdown-Vault fuer persistente Notizen.

Ermoeglicht dem Agenten Wissensartikel, Recherche-Ergebnisse, Meeting-Notizen
und Projektnotizen in einem strukturierten Markdown-Vault zu speichern.

Tools:
  - vault_save: Notiz erstellen mit Frontmatter, Tags, [[Backlinks]]
  - vault_search: Volltextsuche mit Ordner/Tag/Datum-Filter
  - vault_list: Notizen auflisten, gefiltert und sortiert
  - vault_read: Einzelne Notiz lesen (per Titel, Pfad oder Slug)
  - vault_update: An Notiz anhaengen, Tags ergaenzen, Timestamp aktualisieren
  - vault_link: Verknuepfung zwischen Notizen erstellen
  - vault_delete: Notiz loeschen (GDPR erasure)

Format: Obsidian-kompatibles Markdown mit YAML-Frontmatter.
Storage: Delegates to pluggable backend (FileBackend or DBBackend).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jarvis.mcp.vault_backend import parse_tags as _parse_tags
from jarvis.mcp.vault_backend import slugify as _ext_slugify
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

__all__ = [
    "VaultTools",
    "register_vault_tools",
]


# ── Legacy module-level helpers (kept for backward compatibility) ─────────


def _slugify(text: str) -> str:
    """Wandelt einen Titel in einen Dateinamen-sicheren Slug um."""
    slug = text.lower().strip()
    slug = re.sub(r"[äÄ]", "ae", slug)
    slug = re.sub(r"[öÖ]", "oe", slug)
    slug = re.sub(r"[üÜ]", "ue", slug)
    slug = re.sub(r"[ß]", "ss", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80] or "notiz"


def _now_iso() -> str:
    """Aktuelle UTC-Zeit als ISO-String."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


class VaultTools:
    """Knowledge Vault: Obsidian-kompatibles Markdown-Notizen-System.

    Delegates all storage operations to a pluggable backend:
    - VaultFileBackend (default): Obsidian-compatible .md files
    - VaultDBBackend: SQLCipher-encrypted SQLite with FTS5

    Backend is selected based on config.vault.encrypt_files.
    Auto-migration runs on mode change.
    """

    def __init__(self, config: JarvisConfig | None = None) -> None:
        vault_cfg = getattr(config, "vault", None)

        if vault_cfg and getattr(vault_cfg, "path", ""):
            self._vault_root = Path(vault_cfg.path).expanduser().resolve()
        else:
            self._vault_root = Path.home() / ".jarvis" / "vault"

        self._vault_root.mkdir(parents=True, exist_ok=True)

        _raw_encrypt = getattr(vault_cfg, "encrypt_files", False) if vault_cfg else False
        # Guard against MagicMock or other non-bool truthy values: only True/1/"true" count
        if isinstance(_raw_encrypt, bool):
            encrypt = _raw_encrypt
        elif isinstance(_raw_encrypt, (int, float)):
            encrypt = bool(_raw_encrypt)
        elif isinstance(_raw_encrypt, str):
            encrypt = _raw_encrypt.lower() in ("true", "1", "yes")
        else:
            encrypt = False
        default_folders = dict(getattr(vault_cfg, "default_folders", {})) if vault_cfg else None

        # Select backend
        if encrypt:
            from jarvis.mcp.vault_db_backend import VaultDBBackend

            self._backend = VaultDBBackend(self._vault_root)
        else:
            from jarvis.mcp.vault_file_backend import VaultFileBackend

            self._backend = VaultFileBackend(
                self._vault_root,
                encrypt_files=False,
                default_folders=default_folders,
            )

        # Auto-migrate on mode change
        current_mode = "db" if encrypt else "file"
        from jarvis.mcp.vault_migration import (
            detect_mode_change,
            migrate_db_to_files,
            migrate_files_to_db,
        )

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

    # ── Backward-compatible property accessors ───────────────────────────
    # These delegate to the backend for tests that access internal state.

    @property
    def _default_folders(self) -> dict[str, str]:
        return getattr(
            self._backend,
            "_default_folders",
            {
                "research": "recherchen",
                "meetings": "meetings",
                "knowledge": "wissen",
                "projects": "projekte",
                "daily": "daily",
            },
        )

    @property
    def _index_path(self) -> Path:
        return getattr(self._backend, "_index_path", self._vault_root / "_index.json")

    @property
    def _encrypt_files(self) -> bool:
        return getattr(self._backend, "_encrypt", False)

    # ── Backward-compatible private methods (delegate to backend) ────────

    def _validate_vault_path(self, path: Path) -> Path | None:
        """Validiert, dass ein Pfad innerhalb des Vault-Roots liegt."""
        try:
            resolved = path.resolve()
            resolved.relative_to(self._vault_root.resolve())
            return resolved
        except (ValueError, OSError):
            log.warning(
                "vault_path_traversal_blocked",
                attempted_path=str(path),
                vault_root=str(self._vault_root),
            )
            return None

    def _ensure_structure(self) -> None:
        """Erstellt Vault-Verzeichnisstruktur falls nicht vorhanden."""
        if hasattr(self._backend, "_ensure_structure"):
            self._backend._ensure_structure()

    def _read_index(self) -> dict[str, Any]:
        """Liest den _index.json."""
        if hasattr(self._backend, "_read_index"):
            return self._backend._read_index()
        return {}

    def _write_index(self, index: dict[str, Any]) -> None:
        """Schreibt den _index.json."""
        if hasattr(self._backend, "_write_index"):
            self._backend._write_index(index)

    def _update_index(self, title: str, path: str, tags: list[str], folder: str) -> None:
        """Aktualisiert einen Eintrag im Index."""
        if hasattr(self._backend, "_update_index"):
            self._backend._update_index(title, path, tags, folder)

    def _build_frontmatter(
        self,
        title: str,
        tags: list[str],
        sources: list[str] | None = None,
        linked_notes: list[str] | None = None,
    ) -> str:
        """Generiert Obsidian-kompatibles YAML-Frontmatter."""
        if hasattr(self._backend, "_build_frontmatter"):
            return self._backend._build_frontmatter(title, tags, sources, linked_notes)
        # Fallback
        now = _now_iso()
        lines = [
            "---",
            f'title: "{title}"',
            f"created: {now}",
            f"updated: {now}",
            f"tags: [{', '.join(tags)}]",
        ]
        if sources:
            lines.append(f"sources: [{', '.join(sources)}]")
        if linked_notes:
            escaped = [f'"{n}"' for n in linked_notes]
            lines.append(f"linked_notes: [{', '.join(escaped)}]")
        lines.append("author: jarvis")
        lines.append("---")
        return "\n".join(lines)

    def _resolve_folder(self, folder: str) -> str:
        """Loest einen logischen Ordnernamen zu einem Pfad auf."""
        if hasattr(self._backend, "_resolve_folder"):
            return self._backend._resolve_folder(folder)
        folders = self._default_folders
        if folder in folders:
            return folders[folder]
        if folder in folders.values():
            return folder
        return folders.get("knowledge", "wissen")

    def _find_note(self, identifier: str) -> Path | None:
        """Findet eine Notiz per Titel, Pfad oder Slug.

        Returns a Path for backward compatibility with existing tests.
        Delegates to backend.find_note() internally.
        """
        note = self._backend.find_note(identifier)
        if note is None:
            return None
        # Convert NoteData path back to absolute Path
        return self._vault_root / note.path

    # ── Frontmatter-Parsing (backward compatibility) ─────────────────────

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], int, int]:
        """Parst YAML-Frontmatter aus Markdown-Inhalt.

        Returns:
            (frontmatter_dict, start_pos, end_pos) wobei start/end die
            Positionen des gesamten Frontmatter-Blocks inkl. --- Delimiter sind.
            Bei fehlendem Frontmatter: ({}, -1, -1).
        """
        if not content.startswith("---"):
            return {}, -1, -1
        close = content.find("\n---", 3)
        if close < 0:
            return {}, -1, -1
        yaml_text = content[4:close]
        try:
            data = yaml.safe_load(yaml_text)
            if not isinstance(data, dict):
                return {}, -1, -1
            return data, 0, close + 4
        except yaml.YAMLError:
            log.debug("vault_frontmatter_parse_error", content_start=content[:80])
            return {}, -1, -1

    @staticmethod
    def _serialize_frontmatter(data: dict[str, Any]) -> str:
        """Serialisiert ein Dict als YAML-Frontmatter-Block."""
        lines = ["---"]
        for key, value in data.items():
            if isinstance(value, list):
                items = []
                for item in value:
                    s = str(item)
                    if "," in s or '"' in s or "'" in s:
                        items.append(f'"{s}"')
                    else:
                        items.append(s)
                lines.append(f"{key}: [{', '.join(items)}]")
            elif isinstance(value, str) and ('"' in value or ":" in value or "," in value):
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {value}")
        lines.append("---")
        return "\n".join(lines)

    def _extract_frontmatter_tags(self, content: str) -> list[str]:
        """Extrahiert Tags aus dem YAML-Frontmatter."""
        data, _, _ = self._parse_frontmatter(content)
        tags = data.get("tags", [])
        if isinstance(tags, list):
            return [str(t).strip().lower() for t in tags if str(t).strip()]
        if isinstance(tags, str):
            return [t.strip().lower() for t in tags.split(",") if t.strip()]
        return []

    def _extract_frontmatter_field(self, content: str, field: str) -> str:
        """Extrahiert ein einzelnes Feld aus dem Frontmatter."""
        data, _, _ = self._parse_frontmatter(content)
        value = data.get(field, "")
        if value is None:
            return ""
        return str(value).strip()

    def _replace_frontmatter_field(self, content: str, field: str, value: Any) -> str:
        """Ersetzt oder fuegt ein Feld im YAML-Frontmatter hinzu."""
        data, start, end = self._parse_frontmatter(content)
        if start < 0:
            return content
        if isinstance(value, str):
            try:
                parsed = yaml.safe_load(value)
                if isinstance(parsed, list | dict):
                    value = parsed
            except yaml.YAMLError:
                pass
        data[field] = value
        new_fm = self._serialize_frontmatter(data)
        body = content[end:]
        return new_fm + body

    def _add_linked_note(self, content: str, note_title: str) -> str:
        """Fuegt eine Notiz zur linked_notes-Liste im Frontmatter hinzu."""
        data, start, _ = self._parse_frontmatter(content)
        if start < 0:
            return content
        existing = data.get("linked_notes", [])
        if not isinstance(existing, list):
            existing = []
        existing = [str(n).strip().strip('"') for n in existing if str(n).strip()]
        if note_title not in existing:
            existing.append(note_title)
        escaped = [f'"{n}"' for n in existing]
        new_val = f"[{', '.join(escaped)}]"
        return self._replace_frontmatter_field(content, "linked_notes", new_val)

    def _extract_snippet(self, content: str, query: str, context_chars: int = 100) -> str:
        """Extrahiert einen kurzen Kontext-Snippet um den Suchbegriff."""
        _, _, fm_end = self._parse_frontmatter(content)
        body = content[fm_end:] if fm_end > 0 else content
        idx = body.lower().find(query)
        if idx < 0:
            return ""
        start = max(0, idx - context_chars)
        end = min(len(body), idx + len(query) + context_chars)
        snippet = body[start:end].strip()
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet[:250]

    # ── Tool: vault_save ─────────────────────────────────────────────────

    async def vault_save(
        self,
        title: str,
        content: str,
        tags: str = "",
        folder: str = "knowledge",
        sources: str = "",
        linked_notes: str = "",
    ) -> str:
        """Erstellt eine neue Notiz im Vault."""
        if not title.strip():
            return "Fehler: Kein Titel angegeben."
        if not content.strip():
            return "Fehler: Kein Inhalt angegeben."

        link_list = (
            [n.strip() for n in linked_notes.split(",") if n.strip()] if linked_notes else []
        )

        return self._backend.save(
            path=f"{folder}/{_ext_slugify(title)}.md",
            title=title,
            content=content,
            tags=tags,
            folder=folder,
            sources=sources,
            backlinks=link_list,
        )

    # ── Tool: vault_search ───────────────────────────────────────────────

    async def vault_search(
        self,
        query: str,
        folder: str = "",
        tags: str = "",
        limit: int = 10,
    ) -> str:
        """Durchsucht das Vault nach Notizen."""
        if not query.strip():
            return "Fehler: Kein Suchbegriff angegeben."

        results = self._backend.search(query, folder=folder, tags=tags, limit=int(limit))

        if not results:
            return f"Keine Notizen gefunden für: {query}"

        lines = [f"Vault-Suche: {query} ({len(results)} Treffer)\n"]
        for i, note in enumerate(results, 1):
            snippet = note.content[:150].replace("\n", " ") if note.content else ""
            lines.append(f"[{i}] {note.title}")
            lines.append(f"    Pfad: {note.path}")
            if snippet:
                lines.append(f"    ...{snippet}...")
            lines.append("")
        return "\n".join(lines)

    # ── Tool: vault_list ─────────────────────────────────────────────────

    async def vault_list(
        self,
        folder: str = "",
        tags: str = "",
        sort_by: str = "updated",
        limit: int = 20,
    ) -> str:
        """Listet Notizen im Vault auf."""
        notes = self._backend.list_notes(
            folder=folder,
            tags=tags,
            sort_by=sort_by,
            limit=int(limit),
        )

        if not notes:
            return "Keine Notizen im Vault gefunden."

        lines = [f"Vault-Inhalt ({len(notes)} Notizen):\n"]
        for i, n in enumerate(notes, 1):
            lines.append(f"[{i}] {n.title}")
            lines.append(f"    Pfad: {n.path}")
            if n.tags:
                lines.append(f"    Tags: {n.tags}")
            lines.append(f"    Aktualisiert: {n.updated_at or '?'}")
            lines.append("")
        return "\n".join(lines)

    # ── Tool: vault_read ─────────────────────────────────────────────────

    async def vault_read(self, identifier: str) -> str:
        """Liest eine einzelne Notiz aus dem Vault."""
        if not identifier.strip():
            return "Fehler: Kein Identifier angegeben."

        note = self._backend.find_note(identifier)
        if not note:
            return f"Notiz nicht gefunden: {identifier}"

        # For FileBackend: read raw file content to preserve frontmatter for tests
        if hasattr(self._backend, "_read_file"):
            full = self._vault_root / note.path
            if full.exists():
                return self._backend._read_file(full)

        # For DBBackend: reconstruct display content
        return f"# {note.title}\n\nPfad: {note.path}\nTags: {note.tags}\n\n{note.content}"

    # ── Tool: vault_update ───────────────────────────────────────────────

    async def vault_update(
        self,
        identifier: str,
        append_content: str = "",
        add_tags: str = "",
    ) -> str:
        """Aktualisiert eine bestehende Notiz."""
        if not identifier.strip():
            return "Fehler: Kein Identifier angegeben."

        if not append_content.strip() and not add_tags.strip():
            return "Fehler: Weder Inhalt noch Tags zum Aktualisieren angegeben."

        note = self._backend.find_note(identifier)
        if note is None:
            return f"Notiz nicht gefunden: {identifier}"

        return self._backend.update(note.path, append_content=append_content, add_tags=add_tags)

    # ── Tool: vault_link ─────────────────────────────────────────────────

    async def vault_link(
        self,
        source_note: str,
        target_note: str,
    ) -> str:
        """Erstellt eine bidirektionale Verknuepfung zwischen zwei Notizen."""
        src = self._backend.find_note(source_note)
        tgt = self._backend.find_note(target_note)

        if not src:
            return f"Quell-Notiz nicht gefunden: {source_note}"
        if not tgt:
            return f"Ziel-Notiz nicht gefunden: {target_note}"

        return self._backend.link(src.path, tgt.path)

    # ── Tool: vault_delete ──────────────────────────────────────────────

    async def vault_delete(self, path: str) -> str:
        """Loescht eine Notiz aus dem Vault (GDPR erasure)."""
        if not path.strip():
            return "Fehler: Kein Pfad angegeben."

        # Path validation
        try:
            resolved = (self._vault_root / path).resolve()
            resolved.relative_to(self._vault_root.resolve())
        except (ValueError, OSError):
            return f"Ungültiger Pfad (Path-Traversal blockiert): {path}"

        return self._backend.delete(path)


# ── MCP client registration ─────────────────────────────────────────────


def register_vault_tools(
    mcp_client: Any,
    config: Any | None = None,
) -> VaultTools:
    """Registriert Vault-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: JarvisConfig (optional).

    Returns:
        VaultTools-Instanz.
    """
    vault = VaultTools(config=config)

    mcp_client.register_builtin_handler(
        "vault_save",
        vault.vault_save,
        description=(
            "Speichert eine Notiz im Knowledge Vault (Obsidian-kompatibel). "
            "Erstellt Markdown mit YAML-Frontmatter, Tags und [[Backlinks]]. "
            "Ideal für Recherche-Ergebnisse, Meeting-Notizen, Wissensartikel."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titel der Notiz",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown-Inhalt der Notiz",
                },
                "tags": {
                    "type": "string",
                    "description": "Kommagetrennte Tags (z.B. 'finanzen, tesla')",
                    "default": "",
                },
                "folder": {
                    "type": "string",
                    "description": "Ordner: research, meetings, knowledge, projects, daily",
                    "default": "knowledge",
                    "enum": ["research", "meetings", "knowledge", "projects", "daily"],
                },
                "sources": {
                    "type": "string",
                    "description": "Kommagetrennte Quell-URLs",
                    "default": "",
                },
                "linked_notes": {
                    "type": "string",
                    "description": "Kommagetrennte Titel verknüpfter Notizen",
                    "default": "",
                },
            },
            "required": ["title", "content"],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_search",
        vault.vault_search,
        description=(
            "Durchsucht das Knowledge Vault nach Notizen. "
            "Volltextsuche in Titel und Inhalt, filterbar nach Ordner und Tags."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff",
                },
                "folder": {
                    "type": "string",
                    "description": "Nur in diesem Ordner suchen (optional)",
                    "default": "",
                },
                "tags": {
                    "type": "string",
                    "description": "Nur Notizen mit diesen Tags (kommagetrennt, optional)",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximale Anzahl Ergebnisse",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_list",
        vault.vault_list,
        description=(
            "Listet Notizen im Knowledge Vault auf. "
            "Filterbar nach Ordner und Tags, sortierbar nach Datum oder Titel."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Nur Notizen aus diesem Ordner (optional)",
                    "default": "",
                },
                "tags": {
                    "type": "string",
                    "description": "Nur Notizen mit diesen Tags (optional)",
                    "default": "",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sortierung: updated, created, title",
                    "default": "updated",
                    "enum": ["updated", "created", "title"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximale Anzahl",
                    "default": 20,
                },
            },
            "required": [],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_read",
        vault.vault_read,
        description=(
            "Liest eine einzelne Notiz aus dem Vault. Akzeptiert Titel, relativen Pfad oder Slug."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Titel, Pfad oder Slug der Notiz",
                },
            },
            "required": ["identifier"],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_update",
        vault.vault_update,
        description=(
            "Aktualisiert eine bestehende Notiz im Vault. "
            "Kann Text anhängen und/oder Tags ergänzen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Titel, Pfad oder Slug der Notiz",
                },
                "append_content": {
                    "type": "string",
                    "description": "Text der angehängt wird",
                    "default": "",
                },
                "add_tags": {
                    "type": "string",
                    "description": "Neue Tags (kommagetrennt)",
                    "default": "",
                },
            },
            "required": ["identifier"],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_link",
        vault.vault_link,
        description=(
            "Erstellt eine bidirektionale [[Backlink]]-Verknüpfung zwischen zwei Notizen im Vault."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_note": {
                    "type": "string",
                    "description": "Titel/Pfad/Slug der Quell-Notiz",
                },
                "target_note": {
                    "type": "string",
                    "description": "Titel/Pfad/Slug der Ziel-Notiz",
                },
            },
            "required": ["source_note", "target_note"],
        },
    )

    mcp_client.register_builtin_handler(
        "vault_delete",
        vault.vault_delete,
        description="Delete a vault note by path (GDPR erasure)",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the note to delete (relative to vault root)",
                },
            },
            "required": ["path"],
        },
    )

    log.info(
        "vault_tools_registered",
        tools=[
            "vault_save",
            "vault_search",
            "vault_list",
            "vault_read",
            "vault_update",
            "vault_link",
            "vault_delete",
        ],
    )
    return vault
