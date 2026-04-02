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

    def __init__(
        self,
        vault_root: Path,
        encrypt_files: bool = False,
        default_folders: dict[str, str] | None = None,
    ) -> None:
        self._vault_root = vault_root
        self._encrypt = encrypt_files
        self._index_path = vault_root / "_index.json"
        self._default_folders = default_folders or {
            "research": "recherchen",
            "meetings": "meetings",
            "knowledge": "wissen",
            "projects": "projekte",
            "daily": "daily",
        }
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        self._vault_root.mkdir(parents=True, exist_ok=True)
        for folder in self._default_folders.values():
            (self._vault_root / folder).mkdir(exist_ok=True)
        if not self._index_path.exists():
            self._write_index({})

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
        body = content[close + 4 :].lstrip("\n")
        try:
            import yaml

            data = yaml.safe_load(yaml_text)
            if not isinstance(data, dict):
                return {}, content
            return data, body
        except Exception:
            return {}, content

    @staticmethod
    def _build_frontmatter(
        title: str,
        tags: list[str],
        sources: list[str] | None = None,
        backlinks: list[str] | None = None,
    ) -> str:
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
            full_content += (
                "\n## Quellen\n" + "\n".join(f"- [{s}]({s})" for s in source_list) + "\n"
            )
        if backlinks:
            full_content += (
                "\n## Verknüpfte Notizen\n" + "\n".join(f"- [[{b}]]" for b in backlinks) + "\n"
            )
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
            tags=", ".join(fm.get("tags", []))
            if isinstance(fm.get("tags"), list)
            else str(fm.get("tags", "")),
            folder=path.split("/")[0] if "/" in path else "",
            sources=", ".join(fm.get("sources", [])) if isinstance(fm.get("sources"), list) else "",
            backlinks=json.dumps(fm.get("linked_notes", [])),
            created_at=str(fm.get("created", "")),
            updated_at=str(fm.get("updated", "")),
        )

    def search(
        self, query: str, folder: str = "", tags: str = "", limit: int = 10
    ) -> list[NoteData]:
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
                fm_tags = [
                    t.lower()
                    for t in (fm.get("tags", []) if isinstance(fm.get("tags"), list) else [])
                ]
                if not any(t in fm_tags for t in tag_filter):
                    continue
            if query_lower in content.lower():
                note = self.read(rel)
                if note:
                    results.append(note)
                if len(results) >= limit:
                    break
        return results

    def list_notes(
        self, folder: str = "", tags: str = "", sort_by: str = "updated", limit: int = 50
    ) -> list[NoteData]:
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
            entries.append(
                NoteData(
                    path=meta.get("path", ""),
                    title=title,
                    tags=", ".join(meta.get("tags", [])),
                    folder=meta.get("folder", ""),
                    created_at=meta.get("created", ""),
                    updated_at=meta.get("updated", ""),
                )
            )
        key_fn = {
            "title": lambda n: n.title.lower(),
            "created": lambda n: n.created_at,
            "updated": lambda n: n.updated_at,
        }.get(sort_by, lambda n: n.updated_at)
        entries.sort(key=key_fn, reverse=(sort_by != "title"))
        return entries[:limit]

    def update(self, path: str, append_content: str = "", add_tags: str = "") -> str:
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
        return f"Verknüpfung erstellt: [[{source.title}]] <-> [[{target.title}]]"

    def exists(self, path: str) -> bool:
        return (self._vault_root / path).exists()

    def find_note(self, identifier: str) -> NoteData | None:
        # Normalize path separators
        normalized = identifier.replace("\\", "/")

        # 1. Direct path (try both original and normalized)
        for path_variant in (identifier, normalized):
            note = self.read(path_variant)
            if note:
                return note

        # 2. Try with .md extension if not present
        if not normalized.endswith(".md"):
            note = self.read(normalized + ".md")
            if note:
                return note

        # 3. Index by title (case-insensitive)
        id_lower = identifier.lower().strip()
        index = self._read_index()
        for title, meta in index.items():
            if title.lower() == id_lower:
                return self.read(meta["path"])

        # 4. Index by partial title match
        for title, meta in index.items():
            if id_lower in title.lower() or title.lower() in id_lower:
                return self.read(meta["path"])

        # 5. Slug search across filesystem
        slug = id_lower.replace(" ", "-")
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
