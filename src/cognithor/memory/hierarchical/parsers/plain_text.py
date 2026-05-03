"""Plain-text document parser with German legal heuristics."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.memory.hierarchical.models import RawSection
from cognithor.memory.hierarchical.parsers.base import DocumentParser

# Priority-ordered heading patterns
_PARAGRAPH_RE = re.compile(r"^(\u00a7|Par\.|par\.)\s*(\d+)", re.UNICODE)
_ARTICLE_RE = re.compile(r"^Art\.\s*(\d+)", re.IGNORECASE)
_ABS_RE = re.compile(r"^Abs\.\s*(\d+)", re.IGNORECASE)
_NUMBERED_L1_RE = re.compile(r"^(\d+)\.\s+\S")
_NUMBERED_L2_RE = re.compile(r"^(\d+)\.(\d+)\s+\S")
_NUMBERED_L3_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\s+\S")


def _is_all_caps(line: str) -> bool:
    """Return True if *line* is an ALL-CAPS heading (3+ words, <=80 chars)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    words = stripped.split()
    if len(words) < 3:
        return False
    alpha_text = re.sub(r"[^a-zA-Z\u00c0-\u024f]", "", stripped)
    if not alpha_text:
        return False
    return alpha_text == alpha_text.upper()


class PlainTextParser(DocumentParser):
    """Parse plain-text documents using German legal heuristics."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".txt", ".text"})

    def parse(self, content: str | bytes, source_path: Path) -> list[RawSection]:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        if not content.strip():
            return []

        lines = content.splitlines()
        sections: list[dict[str, object]] = []
        current_lines: list[str] = []

        def _flush(level: int, title: str) -> None:
            body = "\n".join(current_lines).strip()
            if sections:
                last = sections[-1]
                existing = str(last["content"])
                combined = (existing + "\n" + body).strip() if existing else body
                last["content"] = combined
            elif body:
                sections.append({"level": 0, "title": "Document", "content": body})
            current_lines.clear()
            sections.append({"level": level, "title": title, "content": ""})

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 1. German paragraph markers: section sign or Art./Abs.
            m = _PARAGRAPH_RE.match(stripped)
            if m:
                _flush(1, stripped.split("\n")[0][:80])
                continue

            m = _ARTICLE_RE.match(stripped)
            if m:
                _flush(1, stripped[:80])
                continue

            m = _ABS_RE.match(stripped)
            if m:
                _flush(2, stripped[:80])
                continue

            # 2. Numbered patterns (check deeper levels first)
            m = _NUMBERED_L3_RE.match(stripped)
            if m:
                _flush(3, stripped[:80])
                continue

            m = _NUMBERED_L2_RE.match(stripped)
            if m:
                _flush(2, stripped[:80])
                continue

            m = _NUMBERED_L1_RE.match(stripped)
            if m:
                _flush(1, stripped[:80])
                continue

            # 3. ALL-CAPS lines
            if _is_all_caps(stripped):
                _flush(1, stripped)
                continue

            # 4. Isolated short lines (blank before and after, <60 chars)
            #    Only applies when the document has more than one non-blank line.
            if stripped and len(stripped) < 60 and len(lines) > 1:
                prev_blank = i == 0 or not lines[i - 1].strip()
                next_blank = (
                    i == len(lines) - 1 or not lines[i + 1].strip() if i + 1 < len(lines) else True
                )
                # Must have content elsewhere (not be the only line)
                has_other_content = any(ln.strip() for j, ln in enumerate(lines) if j != i)
                if prev_blank and next_blank and has_other_content:
                    _flush(2, stripped)
                    continue

            current_lines.append(line)

        # Flush remaining
        body = "\n".join(current_lines).strip()
        if sections:
            last = sections[-1]
            existing = str(last["content"])
            combined = (existing + "\n" + body).strip() if existing else body
            last["content"] = combined
        elif body:
            # No headings found — fallback: split by double-newlines
            paragraphs = re.split(r"\n\s*\n", content.strip())
            if len(paragraphs) <= 1:
                return [RawSection(level=0, title="Document", content=content.strip(), position=0)]
            return [
                RawSection(
                    level=1,
                    title=p.strip().splitlines()[0][:60] if p.strip() else "Paragraph",
                    content=p.strip(),
                    position=idx,
                )
                for idx, p in enumerate(paragraphs)
                if p.strip()
            ]

        return [
            RawSection(
                level=int(cast("int", s["level"])),
                title=str(s["title"]),
                content=str(s["content"]),
                position=idx,
            )
            for idx, s in enumerate(sections)
        ]
