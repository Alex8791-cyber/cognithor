"""Markdown document parser for Hierarchical Document Reasoning."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.memory.hierarchical.models import RawSection
from cognithor.memory.hierarchical.parsers.base import DocumentParser

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_SETEXT_H1_RE = re.compile(r"^={3,}\s*$")
_SETEXT_H2_RE = re.compile(r"^-{3,}\s*$")


class MarkdownParser(DocumentParser):
    """Parse Markdown documents into flat sections."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".md", ".markdown"})

    def parse(self, content: str | bytes, source_path: Path) -> list[RawSection]:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        if not content.strip():
            return []

        lines = content.splitlines()
        sections: list[dict[str, object]] = []
        current_content_lines: list[str] = []
        in_fence = False
        fence_marker: str = ""

        def _flush(new_level: int, new_title: str) -> None:
            body = "\n".join(current_content_lines).strip()
            if sections:
                # attach accumulated content to the last section
                last = sections[-1]
                existing = str(last["content"])
                combined = (existing + "\n" + body).strip() if existing else body
                last["content"] = combined
            elif body:
                # content before first heading
                sections.append({"level": 0, "title": "Document", "content": body})
            current_content_lines.clear()
            sections.append({"level": new_level, "title": new_title, "content": ""})

        i = 0
        while i < len(lines):
            line = lines[i]

            # Toggle fenced code blocks
            fence_match = _FENCE_RE.match(line)
            if fence_match:
                if not in_fence:
                    in_fence = True
                    fence_marker = fence_match.group(1)[0]
                    current_content_lines.append(line)
                    i += 1
                    continue
                elif line.strip().startswith(fence_marker) and len(line.strip().rstrip()) >= len(
                    fence_marker
                ):
                    in_fence = False
                    current_content_lines.append(line)
                    i += 1
                    continue

            if in_fence:
                current_content_lines.append(line)
                i += 1
                continue

            # ATX headings
            atx = _ATX_RE.match(line)
            if atx:
                level = len(atx.group(1))
                title = atx.group(2).rstrip("#").strip()
                _flush(level, title)
                i += 1
                continue

            # Setext headings (look-ahead: next line is === or ---)
            if i + 1 < len(lines) and line.strip():
                next_line = lines[i + 1]
                if _SETEXT_H1_RE.match(next_line):
                    _flush(1, line.strip())
                    i += 2
                    continue
                if _SETEXT_H2_RE.match(next_line):
                    _flush(2, line.strip())
                    i += 2
                    continue

            current_content_lines.append(line)
            i += 1

        # Flush remaining content
        body = "\n".join(current_content_lines).strip()
        if sections:
            last = sections[-1]
            existing = str(last["content"])
            combined = (existing + "\n" + body).strip() if existing else body
            last["content"] = combined
        elif body:
            sections.append({"level": 0, "title": "Document", "content": body})

        # No headings found at all
        if not sections:
            return [
                RawSection(
                    level=0,
                    title="Document",
                    content=content.strip(),
                    position=0,
                )
            ]

        return [
            RawSection(
                level=int(s["level"]),  # type: ignore[call-overload]
                title=str(s["title"]),
                content=str(s["content"]),
                position=idx,
            )
            for idx, s in enumerate(sections)
        ]
