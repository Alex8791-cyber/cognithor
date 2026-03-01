"""Knowledge Vault — Obsidian-kompatibles Markdown-Vault für persistente Notizen.

Ermöglicht dem Agenten Wissensartikel, Recherche-Ergebnisse, Meeting-Notizen
und Projektnotizen in einem strukturierten Markdown-Vault zu speichern.

Tools:
  - vault_save: Notiz erstellen mit Frontmatter, Tags, [[Backlinks]]
  - vault_search: Volltextsuche mit Ordner/Tag/Datum-Filter
  - vault_list: Notizen auflisten, gefiltert und sortiert
  - vault_read: Einzelne Notiz lesen (per Titel, Pfad oder Slug)
  - vault_update: An Notiz anhängen, Tags ergänzen, Timestamp aktualisieren
  - vault_link: Verknüpfung zwischen Notizen erstellen

Format: Obsidian-kompatibles Markdown mit YAML-Frontmatter.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

__all__ = [
    "VaultTools",
    "register_vault_tools",
]


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_tags(tags: str | list[str]) -> list[str]:
    """Normalisiert Tags zu einer einheitlichen Liste."""
    if isinstance(tags, list):
        return [t.strip().lower() for t in tags if t.strip()]
    return [t.strip().lower() for t in tags.split(",") if t.strip()]


class VaultTools:
    """Knowledge Vault: Obsidian-kompatibles Markdown-Notizen-System.

    Verzeichnisstruktur:
        ~/.jarvis/vault/
        ├── recherchen/     # Web-Recherche-Ergebnisse
        ├── meetings/       # Meeting-Notizen
        ├── wissen/         # Wissensartikel
        ├── projekte/       # Projektnotizen
        ├── daily/          # Tagesnotizen
        └── _index.json     # Schnell-Lookup (Titel → Pfad, Tags, Datum)
    """

    def __init__(self, config: JarvisConfig | None = None) -> None:
        vault_cfg = getattr(config, "vault", None)

        if vault_cfg and getattr(vault_cfg, "path", ""):
            self._vault_root = Path(vault_cfg.path).expanduser().resolve()
        else:
            self._vault_root = Path.home() / ".jarvis" / "vault"

        if vault_cfg:
            self._default_folders = dict(getattr(vault_cfg, "default_folders", {}))
        else:
            self._default_folders = {
                "research": "recherchen",
                "meetings": "meetings",
                "knowledge": "wissen",
                "projects": "projekte",
                "daily": "daily",
            }

        self._index_path = self._vault_root / "_index.json"
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Erstellt Vault-Verzeichnisstruktur falls nicht vorhanden."""
        self._vault_root.mkdir(parents=True, exist_ok=True)
        for folder_name in self._default_folders.values():
            (self._vault_root / folder_name).mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index({})
        log.info("vault_structure_ensured", path=str(self._vault_root))

    # ── Index-Verwaltung ─────────────────────────────────────────────────

    def _read_index(self) -> dict[str, Any]:
        """Liest den _index.json."""
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_index(self, index: dict[str, Any]) -> None:
        """Schreibt den _index.json."""
        self._index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _update_index(
        self,
        title: str,
        path: str,
        tags: list[str],
        folder: str,
    ) -> None:
        """Aktualisiert einen Eintrag im Index."""
        index = self._read_index()
        index[title] = {
            "path": path,
            "tags": tags,
            "folder": folder,
            "created": index.get(title, {}).get("created", _now_iso()),
            "updated": _now_iso(),
        }
        self._write_index(index)

    # ── Frontmatter-Generierung ──────────────────────────────────────────

    def _build_frontmatter(
        self,
        title: str,
        tags: list[str],
        sources: list[str] | None = None,
        linked_notes: list[str] | None = None,
    ) -> str:
        """Generiert Obsidian-kompatibles YAML-Frontmatter."""
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
        """Löst einen logischen Ordnernamen zu einem Pfad auf."""
        # Direkt-Mapping: logischer Name → Verzeichnisname
        if folder in self._default_folders:
            return self._default_folders[folder]
        # Prüfe ob es ein direkter Verzeichnisname ist
        if folder in self._default_folders.values():
            return folder
        # Fallback: wissen
        return self._default_folders.get("knowledge", "wissen")

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
        """Erstellt eine neue Notiz im Vault.

        Args:
            title: Titel der Notiz.
            content: Markdown-Inhalt.
            tags: Kommagetrennte Tags (z.B. 'finanzen, tesla').
            folder: Ordner (research/meetings/knowledge/projects/daily).
            sources: Kommagetrennte Quell-URLs.
            linked_notes: Kommagetrennte Titel verknüpfter Notizen.
        """
        if not title.strip():
            return "Fehler: Kein Titel angegeben."
        if not content.strip():
            return "Fehler: Kein Inhalt angegeben."

        tag_list = _parse_tags(tags)
        source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else []
        link_list = [n.strip() for n in linked_notes.split(",") if n.strip()] if linked_notes else []

        folder_name = self._resolve_folder(folder)
        slug = _slugify(title)
        target_dir = self._vault_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{slug}.md"

        # Duplikat-Vermeidung
        if file_path.exists():
            counter = 1
            while file_path.exists():
                file_path = target_dir / f"{slug}-{counter}.md"
                counter += 1

        # Frontmatter + Inhalt zusammenbauen
        frontmatter = self._build_frontmatter(title, tag_list, source_list, link_list)
        body_parts = [frontmatter, "", f"# {title}", "", content]

        # Quellen-Abschnitt
        if source_list:
            body_parts.extend(["", "## Quellen"])
            for src in source_list:
                body_parts.append(f"- [{src}]({src})")

        # Verknüpfte Notizen
        if link_list:
            body_parts.extend(["", "## Verknüpfte Notizen"])
            for link in link_list:
                body_parts.append(f"- [[{link}]]")

        file_path.write_text("\n".join(body_parts) + "\n", encoding="utf-8")

        # Index aktualisieren
        rel_path = str(file_path.relative_to(self._vault_root))
        self._update_index(title, rel_path, tag_list, folder_name)

        log.info("vault_note_saved", title=title, path=rel_path)
        return f"Notiz gespeichert: {rel_path}"

    # ── Tool: vault_search ───────────────────────────────────────────────

    async def vault_search(
        self,
        query: str,
        folder: str = "",
        tags: str = "",
        limit: int = 10,
    ) -> str:
        """Durchsucht das Vault nach Notizen.

        Args:
            query: Suchbegriff (durchsucht Titel und Inhalt).
            folder: Optional: Nur in diesem Ordner suchen.
            tags: Optional: Nur Notizen mit diesen Tags (kommagetrennt).
            limit: Maximale Anzahl Ergebnisse.
        """
        if not query.strip():
            return "Fehler: Kein Suchbegriff angegeben."

        query_lower = query.lower()
        tag_filter = _parse_tags(tags) if tags else []
        folder_filter = self._resolve_folder(folder) if folder else ""

        results: list[dict[str, Any]] = []

        for md_file in self._vault_root.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue

            # Ordner-Filter
            if folder_filter:
                try:
                    rel = md_file.relative_to(self._vault_root)
                    if rel.parts[0] != folder_filter:
                        continue
                except (ValueError, IndexError):
                    continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # Tag-Filter (aus Frontmatter)
            if tag_filter:
                fm_tags = self._extract_frontmatter_tags(content)
                if not any(t in fm_tags for t in tag_filter):
                    continue

            # Volltextsuche
            if query_lower in content.lower():
                fm_title = self._extract_frontmatter_field(content, "title") or md_file.stem
                rel_path = str(md_file.relative_to(self._vault_root))
                # Kontext-Snippet extrahieren
                snippet = self._extract_snippet(content, query_lower)
                results.append({
                    "title": fm_title,
                    "path": rel_path,
                    "snippet": snippet,
                })

            if len(results) >= limit:
                break

        if not results:
            return f"Keine Notizen gefunden für: {query}"

        lines = [f"Vault-Suche: {query} ({len(results)} Treffer)\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    Pfad: {r['path']}")
            if r["snippet"]:
                lines.append(f"    ...{r['snippet']}...")
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
        """Listet Notizen im Vault auf.

        Args:
            folder: Optional: Nur Notizen aus diesem Ordner.
            tags: Optional: Nur Notizen mit diesen Tags.
            sort_by: Sortierung: 'updated', 'created', 'title'.
            limit: Maximale Anzahl.
        """
        index = self._read_index()
        tag_filter = _parse_tags(tags) if tags else []
        folder_filter = self._resolve_folder(folder) if folder else ""

        entries: list[dict[str, Any]] = []
        for title, meta in index.items():
            if folder_filter and meta.get("folder", "") != folder_filter:
                continue
            if tag_filter and not any(t in meta.get("tags", []) for t in tag_filter):
                continue
            entries.append({"title": title, **meta})

        # Sortierung
        if sort_by == "title":
            entries.sort(key=lambda e: e.get("title", "").lower())
        elif sort_by == "created":
            entries.sort(key=lambda e: e.get("created", ""), reverse=True)
        else:
            entries.sort(key=lambda e: e.get("updated", ""), reverse=True)

        entries = entries[:limit]

        if not entries:
            return "Keine Notizen im Vault gefunden."

        lines = [f"Vault-Inhalt ({len(entries)} Notizen):\n"]
        for i, e in enumerate(entries, 1):
            tags_str = ", ".join(e.get("tags", []))
            lines.append(f"[{i}] {e['title']}")
            lines.append(f"    Pfad: {e.get('path', '?')}")
            if tags_str:
                lines.append(f"    Tags: {tags_str}")
            lines.append(f"    Aktualisiert: {e.get('updated', '?')}")
            lines.append("")
        return "\n".join(lines)

    # ── Tool: vault_read ─────────────────────────────────────────────────

    async def vault_read(self, identifier: str) -> str:
        """Liest eine einzelne Notiz aus dem Vault.

        Args:
            identifier: Titel, Pfad (relativ zum Vault) oder Slug der Notiz.
        """
        if not identifier.strip():
            return "Fehler: Kein Identifier angegeben."

        # 1. Direkt als Pfad versuchen
        direct_path = self._vault_root / identifier
        if direct_path.exists() and direct_path.is_file():
            return direct_path.read_text(encoding="utf-8")

        # 2. Im Index nach Titel suchen
        index = self._read_index()
        for title, meta in index.items():
            if title.lower() == identifier.lower():
                note_path = self._vault_root / meta["path"]
                if note_path.exists():
                    return note_path.read_text(encoding="utf-8")

        # 3. Nach Slug suchen
        slug = _slugify(identifier)
        for md_file in self._vault_root.rglob("*.md"):
            if md_file.stem == slug:
                return md_file.read_text(encoding="utf-8")

        return f"Notiz nicht gefunden: {identifier}"

    # ── Tool: vault_update ───────────────────────────────────────────────

    async def vault_update(
        self,
        identifier: str,
        append_content: str = "",
        add_tags: str = "",
    ) -> str:
        """Aktualisiert eine bestehende Notiz.

        Args:
            identifier: Titel, Pfad oder Slug der Notiz.
            append_content: Text der an die Notiz angehängt wird.
            add_tags: Neue Tags (kommagetrennt) die ergänzt werden.
        """
        if not identifier.strip():
            return "Fehler: Kein Identifier angegeben."

        if not append_content.strip() and not add_tags.strip():
            return "Fehler: Weder Inhalt noch Tags zum Aktualisieren angegeben."

        # Notiz finden
        note_path = self._find_note(identifier)
        if note_path is None:
            return f"Notiz nicht gefunden: {identifier}"

        content = note_path.read_text(encoding="utf-8")

        # Tags ergänzen
        if add_tags.strip():
            new_tags = _parse_tags(add_tags)
            existing_tags = self._extract_frontmatter_tags(content)
            all_tags = list(dict.fromkeys(existing_tags + new_tags))  # dedupliziert, Reihenfolge erhalten
            content = self._replace_frontmatter_field(content, "tags", f"[{', '.join(all_tags)}]")

            # Index aktualisieren
            title = self._extract_frontmatter_field(content, "title") or note_path.stem
            rel_path = str(note_path.relative_to(self._vault_root))
            folder = note_path.relative_to(self._vault_root).parts[0] if len(note_path.relative_to(self._vault_root).parts) > 1 else ""
            self._update_index(title, rel_path, all_tags, folder)

        # Updated-Timestamp aktualisieren
        content = self._replace_frontmatter_field(content, "updated", _now_iso())

        # Inhalt anhängen
        if append_content.strip():
            content = content.rstrip("\n") + "\n\n" + append_content.strip() + "\n"

        note_path.write_text(content, encoding="utf-8")

        log.info("vault_note_updated", path=str(note_path))
        return f"Notiz aktualisiert: {note_path.relative_to(self._vault_root)}"

    # ── Tool: vault_link ─────────────────────────────────────────────────

    async def vault_link(
        self,
        source_note: str,
        target_note: str,
    ) -> str:
        """Erstellt eine bidirektionale Verknüpfung zwischen zwei Notizen.

        Args:
            source_note: Titel/Pfad/Slug der Quell-Notiz.
            target_note: Titel/Pfad/Slug der Ziel-Notiz.
        """
        source_path = self._find_note(source_note)
        target_path = self._find_note(target_note)

        if source_path is None:
            return f"Quell-Notiz nicht gefunden: {source_note}"
        if target_path is None:
            return f"Ziel-Notiz nicht gefunden: {target_note}"

        source_title = self._extract_frontmatter_field(
            source_path.read_text(encoding="utf-8"), "title",
        ) or source_path.stem
        target_title = self._extract_frontmatter_field(
            target_path.read_text(encoding="utf-8"), "title",
        ) or target_path.stem

        # [[Backlink]] in Quell-Notiz einfügen
        source_content = source_path.read_text(encoding="utf-8")
        backlink_marker = f"[[{target_title}]]"
        if backlink_marker not in source_content:
            # linked_notes im Frontmatter aktualisieren
            source_content = self._add_linked_note(source_content, target_title)
            source_content = self._replace_frontmatter_field(source_content, "updated", _now_iso())
            source_path.write_text(source_content, encoding="utf-8")

        # [[Backlink]] in Ziel-Notiz einfügen
        target_content = target_path.read_text(encoding="utf-8")
        backlink_marker = f"[[{source_title}]]"
        if backlink_marker not in target_content:
            target_content = self._add_linked_note(target_content, source_title)
            target_content = self._replace_frontmatter_field(target_content, "updated", _now_iso())
            target_path.write_text(target_content, encoding="utf-8")

        log.info("vault_notes_linked", source=source_title, target=target_title)
        return f"Verknüpfung erstellt: [[{source_title}]] ↔ [[{target_title}]]"

    # ── Hilfsmethoden ────────────────────────────────────────────────────

    def _find_note(self, identifier: str) -> Path | None:
        """Findet eine Notiz per Titel, Pfad oder Slug."""
        # 1. Direkt als relativer Pfad
        direct = self._vault_root / identifier
        if direct.exists() and direct.is_file():
            return direct

        # 2. Index-Lookup per Titel
        index = self._read_index()
        for title, meta in index.items():
            if title.lower() == identifier.lower():
                path = self._vault_root / meta["path"]
                if path.exists():
                    return path

        # 3. Slug-Suche
        slug = _slugify(identifier)
        for md_file in self._vault_root.rglob("*.md"):
            if md_file.stem == slug:
                return md_file

        return None

    def _extract_frontmatter_tags(self, content: str) -> list[str]:
        """Extrahiert Tags aus dem YAML-Frontmatter."""
        match = re.search(r"^tags:\s*\[([^\]]*)\]", content, re.MULTILINE)
        if match:
            return [t.strip().lower() for t in match.group(1).split(",") if t.strip()]
        return []

    def _extract_frontmatter_field(self, content: str, field: str) -> str:
        """Extrahiert ein einzelnes Feld aus dem Frontmatter."""
        match = re.search(rf'^{field}:\s*"?([^"\n]*)"?', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def _replace_frontmatter_field(self, content: str, field: str, value: str) -> str:
        """Ersetzt ein Feld im YAML-Frontmatter."""
        pattern = rf'^({field}:\s*).*$'
        replacement = rf'\g<1>{value}'
        new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
        if count == 0:
            # Feld existiert nicht → vor dem schließenden --- einfügen
            new_content = content.replace("\n---\n", f"\n{field}: {value}\n---\n", 1)
        return new_content

    def _add_linked_note(self, content: str, note_title: str) -> str:
        """Fügt eine Notiz zur linked_notes-Liste im Frontmatter hinzu."""
        match = re.search(r'^linked_notes:\s*\[([^\]]*)\]', content, re.MULTILINE)
        if match:
            existing = [n.strip().strip('"') for n in match.group(1).split(",") if n.strip()]
            if note_title not in existing:
                existing.append(note_title)
            escaped = [f'"{n}"' for n in existing]
            new_val = f"[{', '.join(escaped)}]"
            content = self._replace_frontmatter_field(content, "linked_notes", new_val)
        else:
            # Feld hinzufügen
            content = self._replace_frontmatter_field(
                content, "linked_notes", f'["{note_title}"]',
            )
        return content

    def _extract_snippet(self, content: str, query: str, context_chars: int = 100) -> str:
        """Extrahiert einen kurzen Kontext-Snippet um den Suchbegriff."""
        # Frontmatter überspringen
        body = content
        fm_end = content.find("---", 3)
        if fm_end > 0:
            body = content[fm_end + 3:]

        idx = body.lower().find(query)
        if idx < 0:
            return ""

        start = max(0, idx - context_chars)
        end = min(len(body), idx + len(query) + context_chars)
        snippet = body[start:end].strip()
        # Zeilenumbrüche entfernen
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet[:250]


# ── MCP-Client-Registrierung ─────────────────────────────────────────────


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
            "Liest eine einzelne Notiz aus dem Vault. "
            "Akzeptiert Titel, relativen Pfad oder Slug."
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
            "Erstellt eine bidirektionale [[Backlink]]-Verknüpfung "
            "zwischen zwei Notizen im Vault."
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

    log.info(
        "vault_tools_registered",
        tools=["vault_save", "vault_search", "vault_list", "vault_read", "vault_update", "vault_link"],
    )
    return vault
