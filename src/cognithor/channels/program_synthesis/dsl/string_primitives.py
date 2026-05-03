# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — String-DSL primitive family.

Phase-1 of Cognithor's open-world expansion. The grid-DSL family in
``primitives.py`` is now joined by a ``String`` family covering the
core FlashFill-style operations: case manipulation, splitting,
joining, extraction, replacement.

Design constraints carried over from the grid family:

* Pure functions — no IO, no globals, no random.
* Total domains where possible — primitives that can fail (e.g. ``int``
  parsing on a non-numeric string) raise on bad input so the executor's
  sandbox catches the error and the search engine prunes the candidate.
* Deterministic — same inputs → same outputs.
* JSON-serialisable signatures — the cache key only needs the type
  string, not a Python type object.

Every primitive is registered in the singleton ``REGISTRY`` via the
``@primitive`` decorator. Type filtering at search time keeps grid
candidates and string candidates from cross-polluting (a ``Grid``
input never matches a ``String``-input primitive's type signature).

The 14 primitives below cover the seven axes of common string work:

* **Identity**:   ``string_identity``    — passthrough (debug + leaf)
* **Case**:       ``string_lower``, ``string_upper``, ``string_capitalize``
* **Trim**:       ``string_strip``
* **Split/Join**: ``string_split_space``, ``string_split_comma``,
                  ``string_join_space``, ``string_join_comma``
* **Replace**:    ``string_replace_dash_with_space``,
                  ``string_replace_underscore_with_space``
* **Slice**:      ``string_first_word``, ``string_last_word``
* **Concat**:     ``string_reverse``

Constants like the split-separator and replacement chars are *baked
into the primitive name* rather than passed as args so the search
engine can enumerate concrete programs without an unbounded const
search space (mirrors how the grid family pre-bakes ``rotate90`` /
``rotate180`` instead of a parameterised ``rotate(deg)``).
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature

# ---------------------------------------------------------------------------
# Identity (also used as the bottom-up bank seed for String tasks)
# ---------------------------------------------------------------------------


@primitive(
    name="string_identity",
    signature=Signature(("String",), "String"),
    cost=0.0,
    description="Return the string unchanged. Useful as a no-op leaf.",
)
def string_identity(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_identity expected str, got {type(s).__name__}")
    return s


# ---------------------------------------------------------------------------
# Case manipulation
# ---------------------------------------------------------------------------


@primitive(
    name="string_lower",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string in all-lowercase.",
)
def string_lower(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_lower expected str, got {type(s).__name__}")
    return s.lower()


@primitive(
    name="string_upper",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string in all-uppercase.",
)
def string_upper(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_upper expected str, got {type(s).__name__}")
    return s.upper()


@primitive(
    name="string_capitalize",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with the first char uppercased and rest lowercased.",
)
def string_capitalize(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_capitalize expected str, got {type(s).__name__}")
    return s.capitalize()


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


@primitive(
    name="string_strip",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with leading/trailing whitespace removed.",
)
def string_strip(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_strip expected str, got {type(s).__name__}")
    return s.strip()


# ---------------------------------------------------------------------------
# Split / join
# ---------------------------------------------------------------------------


@primitive(
    name="string_split_space",
    signature=Signature(("String",), "StringList"),
    cost=1.0,
    description="Split on any whitespace run; collapse consecutive separators.",
)
def string_split_space(s: str) -> list[str]:
    if not isinstance(s, str):
        raise TypeError(f"string_split_space expected str, got {type(s).__name__}")
    return s.split()


@primitive(
    name="string_split_comma",
    signature=Signature(("String",), "StringList"),
    cost=1.0,
    description="Split on commas. Empty fields kept (use string_strip on parts to clean).",
)
def string_split_comma(s: str) -> list[str]:
    if not isinstance(s, str):
        raise TypeError(f"string_split_comma expected str, got {type(s).__name__}")
    return s.split(",")


@primitive(
    name="string_join_space",
    signature=Signature(("StringList",), "String"),
    cost=1.0,
    description="Join a list of strings with single spaces.",
)
def string_join_space(parts: list[str]) -> str:
    if not isinstance(parts, list):
        raise TypeError(f"string_join_space expected list, got {type(parts).__name__}")
    return " ".join(str(p) for p in parts)


@primitive(
    name="string_join_comma",
    signature=Signature(("StringList",), "String"),
    cost=1.0,
    description="Join a list of strings with comma+space.",
)
def string_join_comma(parts: list[str]) -> str:
    if not isinstance(parts, list):
        raise TypeError(f"string_join_comma expected list, got {type(parts).__name__}")
    return ", ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Replace (constant separators baked into the primitive name)
# ---------------------------------------------------------------------------


@primitive(
    name="string_replace_dash_with_space",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Replace every '-' with a space. Common in slug → title transforms.",
)
def string_replace_dash_with_space(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_replace_dash_with_space expected str, got {type(s).__name__}")
    return s.replace("-", " ")


@primitive(
    name="string_replace_underscore_with_space",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Replace every '_' with a space. snake_case → words.",
)
def string_replace_underscore_with_space(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(
            f"string_replace_underscore_with_space expected str, got {type(s).__name__}"
        )
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# Slice (first / last word) — operates on the un-split string for direct use
# ---------------------------------------------------------------------------


@primitive(
    name="string_first_word",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the first whitespace-delimited word, or '' for an empty input.",
)
def string_first_word(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_first_word expected str, got {type(s).__name__}")
    parts = s.split()
    return parts[0] if parts else ""


@primitive(
    name="string_last_word",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the last whitespace-delimited word, or '' for an empty input.",
)
def string_last_word(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_last_word expected str, got {type(s).__name__}")
    parts = s.split()
    return parts[-1] if parts else ""


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------


@primitive(
    name="string_reverse",
    signature=Signature(("String",), "String"),
    cost=1.0,
    description="Return the string with characters in reverse order.",
)
def string_reverse(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"string_reverse expected str, got {type(s).__name__}")
    return s[::-1]


__all__ = [
    "string_capitalize",
    "string_first_word",
    "string_identity",
    "string_join_comma",
    "string_join_space",
    "string_last_word",
    "string_lower",
    "string_replace_dash_with_space",
    "string_replace_underscore_with_space",
    "string_reverse",
    "string_split_comma",
    "string_split_space",
    "string_strip",
    "string_upper",
]
