"""DOCX document parser for Hierarchical Document Reasoning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.memory.hierarchical.models import ParserError, RawSection
from cognithor.memory.hierarchical.parsers.base import DocumentParser


class DocxParser(DocumentParser):
    """Parse DOCX documents using python-docx."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    def parse(self, content: str | bytes, source_path: Path) -> list[RawSection]:
        try:
            import docx
        except ImportError as exc:
            raise ParserError("python-docx not installed — run: pip install python-docx") from exc

        if isinstance(content, str):
            content = content.encode("utf-8")

        import io

        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as exc:
            raise ParserError(f"Failed to open DOCX: {exc}") from exc

        sections: list[RawSection] = []
        current_content_parts: list[str] = []
        position = 0

        def _flush(level: int, title: str) -> None:
            nonlocal position
            body = "\n".join(current_content_parts).strip()
            if sections:
                last = sections[-1]
                combined = (last.content + "\n" + body).strip() if last.content else body
                sections[-1] = RawSection(
                    level=last.level,
                    title=last.title,
                    content=combined,
                    position=last.position,
                    page=last.page,
                )
            elif body:
                sections.append(
                    RawSection(level=0, title="Document", content=body, position=position)
                )
                position += 1
            current_content_parts.clear()
            sections.append(RawSection(level=level, title=title, content="", position=position))
            position += 1

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "tbl":
                # Table: extract all cell text
                table_text = self._extract_table_text(element)
                if table_text:
                    current_content_parts.append(table_text)
                continue

            if tag != "p":
                continue

            # It's a paragraph element — find the matching Paragraph object
            para = self._find_paragraph(doc, element)
            if para is None:
                continue

            text = para.text.strip()

            # Check for heading style
            heading_level = self._get_heading_level(para)
            if heading_level is not None:
                _flush(heading_level, text or "Untitled")
                continue

            # Fallback: bold + large font → heading level 1
            if text and self._is_bold_large(para):
                _flush(1, text)
                continue

            if text:
                current_content_parts.append(text)

        # Flush remaining content
        body = "\n".join(current_content_parts).strip()
        if sections and body:
            last = sections[-1]
            combined = (last.content + "\n" + body).strip() if last.content else body
            sections[-1] = RawSection(
                level=last.level,
                title=last.title,
                content=combined,
                position=last.position,
                page=last.page,
            )
        elif body:
            sections.append(RawSection(level=0, title="Document", content=body, position=0))

        return sections

    @staticmethod
    def _get_heading_level(para: Any) -> int | None:
        """Extract heading level from paragraph style name, e.g. 'Heading 2' -> 2."""
        style_name = getattr(para.style, "name", "") or ""
        if style_name.startswith("Heading"):
            rest = style_name[len("Heading") :].strip()
            if rest.isdigit():
                return int(rest)
            # "Heading" alone → level 1
            return 1
        return None

    @staticmethod
    def _is_bold_large(para: Any) -> bool:
        """Check if paragraph is bold with font size >= 14pt."""
        for run in para.runs:
            # python-docx stores size in EMU; Pt(14) = 177800
            if run.bold and run.font.size is not None and run.font.size >= 177800:
                return True
        return False

    @staticmethod
    def _extract_table_text(tbl_element: Any) -> str:
        """Extract all text from a table XML element."""

        _ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        cells: list[str] = []
        for tc in tbl_element.iter(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"
        ):
            parts: list[str] = []
            for p in tc.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
                if p.text:
                    parts.append(p.text)
            if parts:
                cells.append(" ".join(parts))
        return "\n".join(cells)

    @staticmethod
    def _find_paragraph(doc: Any, element: Any) -> Any:
        """Find the python-docx Paragraph object matching an XML element."""
        for para in doc.paragraphs:
            if para._element is element:
                return para
        return None
