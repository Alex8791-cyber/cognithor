"""HTML document parser for Hierarchical Document Reasoning."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.memory.hierarchical.models import RawSection
from cognithor.memory.hierarchical.parsers.base import DocumentParser

_HEADING_RE = re.compile(r"^h([1-6])$", re.IGNORECASE)
_REMOVE_TAGS = {"nav", "footer", "aside", "header", "script", "style"}
_REMOVE_CLASS_PATTERNS = {"nav", "menu", "footer", "sidebar"}


class HtmlParser(DocumentParser):
    """Parse HTML documents using BeautifulSoup."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    def parse(self, content: str | bytes, source_path: Path) -> list[RawSection]:
        from bs4 import BeautifulSoup

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        if not content.strip():
            return []

        soup = BeautifulSoup(content, "html.parser")

        # Remove unwanted elements
        self._strip_elements(soup)

        # Walk through top-level body content
        body = soup.body or soup
        sections: list[RawSection] = []
        current_content_parts: list[str] = []
        position = 0

        def _flush(level: int, title: str) -> None:
            nonlocal position
            body_text = "\n".join(current_content_parts).strip()
            if sections:
                last = sections[-1]
                combined = (last.content + "\n" + body_text).strip() if last.content else body_text
                sections[-1] = RawSection(
                    level=last.level,
                    title=last.title,
                    content=combined,
                    position=last.position,
                )
            elif body_text:
                sections.append(
                    RawSection(level=0, title="Document", content=body_text, position=position)
                )
                position += 1
            current_content_parts.clear()
            sections.append(RawSection(level=level, title=title, content="", position=position))
            position += 1

        for element in body.descendants:
            if element.name is None:
                continue

            # Check if it's a heading
            m = _HEADING_RE.match(element.name)
            if m:
                level = int(m.group(1))
                title = element.get_text(strip=True)
                if title:
                    _flush(level, title)
                continue

            # Collect text from non-heading block elements
            if element.name in {
                "p",
                "li",
                "td",
                "th",
                "dd",
                "dt",
                "blockquote",
                "pre",
                "div",
                "section",
                "article",
            }:
                # Only collect if this element has no heading descendants
                has_heading_child = any(
                    _HEADING_RE.match(child.name) for child in element.descendants if child.name
                )
                if not has_heading_child:
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        current_content_parts.append(text)

        # Flush remaining
        body_text = "\n".join(current_content_parts).strip()
        if sections and body_text:
            last = sections[-1]
            combined = (last.content + "\n" + body_text).strip() if last.content else body_text
            sections[-1] = RawSection(
                level=last.level,
                title=last.title,
                content=combined,
                position=last.position,
            )
        elif body_text:
            sections.append(RawSection(level=0, title="Document", content=body_text, position=0))

        # Deduplicate content lines within each section
        result: list[RawSection] = []
        for s in sections:
            lines = s.content.split("\n")
            seen: set[str] = set()
            deduped: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped not in seen:
                    seen.add(stripped)
                    deduped.append(line)
            result.append(
                RawSection(
                    level=s.level,
                    title=s.title,
                    content="\n".join(deduped).strip(),
                    position=s.position,
                )
            )

        return result

    @staticmethod
    def _strip_elements(soup: object) -> None:
        """Remove unwanted elements from the soup in-place."""
        from bs4 import Tag

        # Remove by tag name
        for tag_name in _REMOVE_TAGS:
            for el in list(getattr(soup, "find_all", lambda *a: [])(tag_name)):
                el.decompose()

        # Remove by class name patterns
        for el in list(getattr(soup, "find_all", lambda *a: [])(True)):
            if not isinstance(el, Tag):
                continue
            try:
                classes = el.get("class", [])
            except (AttributeError, TypeError):
                continue
            if classes is None:
                continue
            if isinstance(classes, str):
                classes = [classes]
            for cls in classes:
                if any(pat in cls.lower() for pat in _REMOVE_CLASS_PATTERNS):
                    el.decompose()
                    break
