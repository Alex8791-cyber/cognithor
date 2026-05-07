"""Offline structural checks on the golden set.

The full LLM-judged regression test requires API keys + a running
Cognithor instance and lives in ``test_golden_judged.py`` (skipped
when prereqs are missing).

These offline tests verify the corpus itself is well-formed so a typo
doesn't silently disable a regression check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CORPUS_PATH = Path(__file__).parent / "golden_set.yaml"


def _load() -> list[dict[str, Any]]:
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data.get("prompts", []))


CORPUS = _load()


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_entry_has_required_fields(entry: dict[str, Any]) -> None:
    """Every prompt must have id, language, prompt, task_type, rubric."""
    for key in ("id", "language", "prompt", "task_type", "rubric"):
        assert key in entry, f"{entry.get('id', '?')} missing {key!r}"
    assert entry["language"] in {"de", "en"}, entry["id"]
    assert entry["prompt"].strip(), f"{entry['id']} has empty prompt"


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_rubric_is_actionable(entry: dict[str, Any]) -> None:
    rubric = entry["rubric"]
    has_assertion = (
        rubric.get("must_include")
        or rubric.get("must_not_hallucinate")
        or rubric.get("json_schema")
    )
    assert has_assertion, (
        f"{entry['id']} has a max_words limit but no positive assertion. "
        "Add must_include, must_not_hallucinate, or json_schema."
    )


def test_no_duplicate_ids() -> None:
    ids = [e["id"] for e in CORPUS]
    dupes = {x for x in ids if ids.count(x) > 1}
    assert not dupes, f"Duplicate prompt IDs: {dupes}"


def test_corpus_has_minimum_breadth() -> None:
    """Sentinel: ensure the corpus covers more than one task type."""
    types = {e["task_type"] for e in CORPUS}
    assert len(types) >= 8, (
        f"Golden set covers only {types}. Need ≥8 task types for the "
        "regression suite to catch personality/refusal/code drift."
    )


def test_languages_balanced() -> None:
    """At least 5 prompts per supported language."""
    by_lang: dict[str, int] = {}
    for entry in CORPUS:
        by_lang[entry["language"]] = by_lang.get(entry["language"], 0) + 1
    assert by_lang.get("de", 0) >= 5, by_lang
    assert by_lang.get("en", 0) >= 5, by_lang


def test_refusal_prompts_present() -> None:
    """Refusal-class prompts protect against jailbreak/credential-dump regressions."""
    refusal = [e for e in CORPUS if e["task_type"] in ("refusal", "tool_selection")]
    assert len(refusal) >= 3, (
        f"Only {len(refusal)} refusal/tool-selection prompts. "
        "Add more — these guard the high-risk regression surface."
    )
