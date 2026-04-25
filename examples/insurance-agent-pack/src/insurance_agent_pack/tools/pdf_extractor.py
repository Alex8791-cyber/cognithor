"""PDF extractor — read text from a PDF file using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def pdf_extract_text(path: str, *, max_chars: int | None = None) -> str:
    """Return the text content of a PDF file.

    Args:
        path: filesystem path to the PDF.
        max_chars: optional truncation; useful for testing or downstream
            tokens-budget enforcement.

    Raises:
        FileNotFoundError: when the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(p)
    try:
        chunks: list[str] = []
        for page in doc:
            chunks.append(page.get_text())
        text = "\n".join(chunks)
    finally:
        doc.close()

    if max_chars is not None:
        return text[:max_chars]
    return text
