"""Parser factory for Hierarchical Document Reasoning."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from cognithor.memory.hierarchical.models import ParserError

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.memory.hierarchical.parsers.base import DocumentParser

_EXTENSION_MAP: dict[str, str] = {
    ".md": "MarkdownParser",
    ".markdown": "MarkdownParser",
    ".pdf": "PDFParser",
    ".docx": "DocxParser",
    ".html": "HtmlParser",
    ".htm": "HtmlParser",
    ".txt": "PlainTextParser",
    ".text": "PlainTextParser",
    "": "PlainTextParser",
}

_MODULE_MAP: dict[str, str] = {
    "MarkdownParser": "cognithor.memory.hierarchical.parsers.markdown",
    "PDFParser": "cognithor.memory.hierarchical.parsers.pdf",
    "DocxParser": "cognithor.memory.hierarchical.parsers.docx",
    "HtmlParser": "cognithor.memory.hierarchical.parsers.html",
    "PlainTextParser": "cognithor.memory.hierarchical.parsers.plain_text",
}


def get_parser(source_path: Path) -> DocumentParser:
    """Return the appropriate parser for *source_path* based on its extension.

    Raises ``ParserError`` when the file type is not supported or the
    parser module has not been implemented yet.
    """
    suffix = source_path.suffix.lower()
    class_name = _EXTENSION_MAP.get(suffix)
    if class_name is None:
        raise ParserError(f"Unsupported file type: {suffix}")

    module_path = _MODULE_MAP[class_name]
    try:
        import importlib

        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ParserError(
            f"Parser {class_name} not yet available — install its module ({module_path}): {exc}"
        ) from exc

    cls = getattr(mod, class_name)
    return cast("DocumentParser", cls())
