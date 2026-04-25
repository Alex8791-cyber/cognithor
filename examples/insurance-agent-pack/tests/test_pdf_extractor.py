"""PDF extractor — reads a synthetic PDF, returns extracted text."""

from __future__ import annotations

from pathlib import Path

import pytest
from insurance_agent_pack.tools.pdf_extractor import pdf_extract_text

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_policy.pdf"


def test_extract_returns_string() -> None:
    text = pdf_extract_text(str(FIXTURE))
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_handles_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        pdf_extract_text("/nonexistent/path/missing.pdf")


def test_extract_truncates_at_limit() -> None:
    text = pdf_extract_text(str(FIXTURE), max_chars=50)
    assert len(text) <= 50
