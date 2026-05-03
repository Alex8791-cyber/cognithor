"""PDF document parser for Hierarchical Document Reasoning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.memory.hierarchical.models import ParserError, RawSection
from cognithor.memory.hierarchical.parsers.base import DocumentParser


class PDFParser(DocumentParser):
    """Parse PDF documents using pymupdf (fitz)."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def parse(self, content: str | bytes, source_path: Path) -> list[RawSection]:
        try:
            import fitz
        except ImportError as exc:
            raise ParserError("pymupdf not installed — run: pip install pymupdf") from exc

        if isinstance(content, str):
            content = content.encode("utf-8")

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ParserError(f"Failed to open PDF: {exc}") from exc

        # Strategy 1: use TOC if available
        toc: list[Any] = doc.get_toc()
        if toc:
            return self._parse_with_toc(doc, toc)

        # Strategy 2: font-size heuristic
        return self._parse_with_font_heuristic(doc)

    # --------------------------------------------------------------------- #
    # Strategy 1: TOC-based
    # --------------------------------------------------------------------- #
    def _parse_with_toc(self, doc: Any, toc: list[Any]) -> list[RawSection]:
        """Use the document's table of contents to derive sections."""
        sections: list[RawSection] = []
        page_count = len(doc)

        for idx, entry in enumerate(toc):
            level = int(entry[0])
            title = str(entry[1]).strip()
            page_num = int(entry[2]) - 1  # 0-indexed

            # Determine content range: from this TOC entry to the next
            if idx + 1 < len(toc):
                end_page = int(toc[idx + 1][2]) - 1
            else:
                end_page = page_count - 1

            content_parts: list[str] = []
            for p in range(max(0, page_num), min(end_page + 1, page_count)):
                page = doc[p]
                text = page.get_text("text")
                if text:
                    content_parts.append(text.strip())

            sections.append(
                RawSection(
                    level=level,
                    title=title,
                    content="\n".join(content_parts),
                    position=idx,
                    page=page_num + 1,  # 1-indexed for display
                )
            )

        return sections

    # --------------------------------------------------------------------- #
    # Strategy 2: Font-size heuristic
    # --------------------------------------------------------------------- #
    def _parse_with_font_heuristic(self, doc: Any) -> list[RawSection]:
        """Detect headings by font size — top 20% sizes are headings."""
        all_spans: list[dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            blocks = page.get_text("dict", flags=0).get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            all_spans.append(
                                {
                                    "text": text,
                                    "size": span.get("size", 12.0),
                                    "page": page_idx + 1,
                                }
                            )

        if not all_spans:
            return []

        # Find the size threshold for headings (top 20%)
        sizes = sorted({s["size"] for s in all_spans}, reverse=True)
        threshold_idx = max(1, len(sizes) // 5)
        heading_sizes = set(sizes[:threshold_idx])

        sections: list[RawSection] = []
        current_title = ""
        current_content_parts: list[str] = []
        current_page: int | None = None
        position = 0

        for span in all_spans:
            if span["size"] in heading_sizes:
                # Flush previous section
                if current_title or current_content_parts:
                    sections.append(
                        RawSection(
                            level=1 if current_title else 0,
                            title=current_title or "Document",
                            content="\n".join(current_content_parts).strip(),
                            position=position,
                            page=current_page,
                        )
                    )
                    position += 1
                current_title = span["text"]
                current_content_parts = []
                current_page = span["page"]
            else:
                current_content_parts.append(span["text"])
                if current_page is None:
                    current_page = span["page"]

        # Flush last section
        if current_title or current_content_parts:
            sections.append(
                RawSection(
                    level=1 if current_title else 0,
                    title=current_title or "Document",
                    content="\n".join(current_content_parts).strip(),
                    position=position,
                    page=current_page,
                )
            )

        return sections
