# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — Regex/Pattern-DSL primitive family.

PR#2 of Track B. Builds on the String-DSL groundwork in
``string_primitives.py`` by adding pattern-matching, character-class
filtering, and slug-style transforms — the operations FlashFill /
ProgramFilter literature calls "extract", "filter", and "rewrite by
character class". These cover most of the real-world data-cleaning
gap that pure case/split/replace primitives leave open:

* "Pull just the digits out of an order ID."
* "Drop the punctuation."
* "Find the first email in a free-text comment."

Design constraints carried over from the grid + string families:

* Pure functions — no IO, no globals, no random.
* Total domains — empty / no-match returns ``""`` rather than
  raising, so the executor sandbox keeps the candidate alive but
  the verifier prunes it.
* Deterministic — same inputs → same outputs.
* Constants baked into the primitive name — there's no
  parameterised ``regex(pattern)`` because that would explode the
  search space. Instead each common need gets its own pre-baked
  primitive (``string_keep_digits``, ``string_first_digit_run``,
  …) — same trick the grid family uses for ``rotate90`` /
  ``rotate180``.

The 14 primitives below cover six axes:

* **Character-class filter (keep)**: ``string_keep_digits``,
  ``string_keep_letters``, ``string_keep_alphanumeric``.
* **Character-class filter (remove)**: ``string_remove_digits``,
  ``string_remove_letters``, ``string_remove_punctuation``,
  ``string_remove_spaces``.
* **Whitespace normalisation**: ``string_collapse_spaces``.
* **Run extraction**: ``string_first_digit_run``,
  ``string_last_digit_run``.
* **Pattern extraction**: ``string_extract_email``,
  ``string_extract_url``.
* **Slug rewrites**: ``string_replace_space_with_dash``,
  ``string_replace_space_with_underscore``.

The email and URL regexes are intentionally simple — RFC-correct
matching is not the goal; ``\\S+@\\S+\\.\\S+`` and ``https?://\\S+``
are good enough for the FlashFill-style demo tasks the engine
solves.
"""

from __future__ import annotations

import re

from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature

_DIGITS_RUN = re.compile(r"\d+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RUN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Character-class filter — keep
# ---------------------------------------------------------------------------


@primitive(
    name="string_keep_digits",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return only the digit characters of the string, in original order.",
)
def string_keep_digits(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_keep_digits expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if ch.isdigit())


@primitive(
    name="string_keep_letters",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return only the alphabetic characters of the string, in original order.",
)
def string_keep_letters(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_keep_letters expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if ch.isalpha())


@primitive(
    name="string_keep_alphanumeric",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return only the letters and digits of the string, in original order.",
)
def string_keep_alphanumeric(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_keep_alphanumeric expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if ch.isalnum())


# ---------------------------------------------------------------------------
# Character-class filter — remove
# ---------------------------------------------------------------------------


@primitive(
    name="string_remove_digits",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with every digit character removed.",
)
def string_remove_digits(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_remove_digits expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if not ch.isdigit())


@primitive(
    name="string_remove_letters",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with every alphabetic character removed.",
)
def string_remove_letters(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_remove_letters expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if not ch.isalpha())


@primitive(
    name="string_remove_punctuation",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with every non-alphanumeric, non-whitespace character removed.",
)
def string_remove_punctuation(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_remove_punctuation expected str, got {type(s).__name__}")
    return _PUNCT.sub("", s)


@primitive(
    name="string_remove_spaces",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with every whitespace character removed.",
)
def string_remove_spaces(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_remove_spaces expected str, got {type(s).__name__}")
    return "".join(ch for ch in s if not ch.isspace())


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------


@primitive(
    name="string_collapse_spaces",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Replace any run of whitespace with a single space (no leading/trailing strip).",
)
def string_collapse_spaces(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_collapse_spaces expected str, got {type(s).__name__}")
    return _SPACE_RUN.sub(" ", s)


# ---------------------------------------------------------------------------
# Run extraction (digits)
# ---------------------------------------------------------------------------


@primitive(
    name="string_first_digit_run",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the first contiguous run of digit characters, or '' if none.",
)
def string_first_digit_run(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_first_digit_run expected str, got {type(s).__name__}")
    match = _DIGITS_RUN.search(s)
    return match.group(0) if match else ""


@primitive(
    name="string_last_digit_run",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the last contiguous run of digit characters, or '' if none.",
)
def string_last_digit_run(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_last_digit_run expected str, got {type(s).__name__}")
    matches = _DIGITS_RUN.findall(s)
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Pattern extraction (free-text → first match or '')
# ---------------------------------------------------------------------------


@primitive(
    name="string_extract_email",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the first email-like token (\\S+@\\S+\\.\\S+) or '' if none.",
)
def string_extract_email(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_extract_email expected str, got {type(s).__name__}")
    match = _EMAIL.search(s)
    return match.group(0) if match else ""


@primitive(
    name="string_extract_url",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the first http(s):// URL-like token, or '' if none.",
)
def string_extract_url(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_extract_url expected str, got {type(s).__name__}")
    match = _URL.search(s)
    return match.group(0) if match else ""


# ---------------------------------------------------------------------------
# Slug rewrites (space → separator)
# ---------------------------------------------------------------------------


@primitive(
    name="string_replace_space_with_dash",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Replace every space with a '-'. Title → slug.",
)
def string_replace_space_with_dash(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_replace_space_with_dash expected str, got {type(s).__name__}")
    return s.replace(" ", "-")


@primitive(
    name="string_replace_space_with_underscore",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Replace every space with a '_'. Title → snake_case.",
)
def string_replace_space_with_underscore(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(
            f"string_replace_space_with_underscore expected str, got {type(s).__name__}"
        )
    return s.replace(" ", "_")


__all__ = [
    "string_collapse_spaces",
    "string_extract_email",
    "string_extract_url",
    "string_first_digit_run",
    "string_keep_alphanumeric",
    "string_keep_digits",
    "string_keep_letters",
    "string_last_digit_run",
    "string_remove_digits",
    "string_remove_letters",
    "string_remove_punctuation",
    "string_remove_spaces",
    "string_replace_space_with_dash",
    "string_replace_space_with_underscore",
]
