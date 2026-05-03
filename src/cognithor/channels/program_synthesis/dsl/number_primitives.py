# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — Number/Int-DSL primitive family.

PR#3 of Track B. Brings ``Int`` to first-class status alongside
``Grid`` and ``String``: integer arithmetic primitives plus the two
conversions (``int_to_string`` / ``string_to_int``) that bridge the
two text-shaped families. Combined with PR#1 + PR#2 the engine can
now solve demo specs like:

    examples = ((3, 9), (4, 16))         # int → int (square)
    examples = (("12", 13), ("99", 100)) # str → int (parse + +1)
    examples = ((5, "5"), (42, "42"))    # int → str (stringify)

Design constraints carry over:

* Pure functions — no IO, no globals, no random.
* Total domains where reasonable. Conversions that *could* fail
  (``string_to_int`` on a non-numeric string) raise so the executor
  sandbox prunes the candidate. Other arithmetic over Python ints
  is total and never raises.
* Constants baked into the primitive name — ``int_double`` /
  ``int_triple`` / ``int_increment`` instead of a parameterised
  ``int_mul(k)`` — keeps the search space bounded the same way the
  grid family does for ``rotate90`` / ``rotate180``.

The 12 primitives below cover six axes:

* **Identity**:       ``int_identity``
* **Shift**:          ``int_increment``, ``int_decrement``
* **Multiplicative**: ``int_double``, ``int_triple``, ``int_half``
* **Sign**:           ``int_negate``, ``int_abs``
* **Power**:          ``int_square``
* **Bridges**:        ``int_to_string``, ``string_to_int``,
                      ``string_length``

``string_length`` lives in this family — not in ``string_primitives``
— because it is the natural "string-shaped → Int-shaped" companion
to ``string_to_int`` and only makes sense once ``Int`` is registered
as an allowed type.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@primitive(
    name="int_identity",
    signature=Signature(("Int",), "Int"),
    cost=0.0,
    description="Return the integer unchanged. Useful as a no-op leaf.",
)
def int_identity(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_identity expected int, got {type(n).__name__}")
    return n


# ---------------------------------------------------------------------------
# Additive shift
# ---------------------------------------------------------------------------


@primitive(
    name="int_increment",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return n + 1.",
)
def int_increment(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_increment expected int, got {type(n).__name__}")
    return n + 1


@primitive(
    name="int_decrement",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return n - 1.",
)
def int_decrement(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_decrement expected int, got {type(n).__name__}")
    return n - 1


# ---------------------------------------------------------------------------
# Multiplicative
# ---------------------------------------------------------------------------


@primitive(
    name="int_double",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return 2 * n.",
)
def int_double(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_double expected int, got {type(n).__name__}")
    return 2 * n


@primitive(
    name="int_triple",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return 3 * n.",
)
def int_triple(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_triple expected int, got {type(n).__name__}")
    return 3 * n


@primitive(
    name="int_half",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return n // 2 (integer division, rounds toward negative infinity).",
)
def int_half(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_half expected int, got {type(n).__name__}")
    return n // 2


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


@primitive(
    name="int_negate",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return -n.",
)
def int_negate(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_negate expected int, got {type(n).__name__}")
    return -n


@primitive(
    name="int_abs",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return |n|.",
)
def int_abs(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_abs expected int, got {type(n).__name__}")
    return abs(n)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


@primitive(
    name="int_square",
    signature=Signature(("Int",), "Int"),
    cost=1.0,
    description="Return n * n.",
)
def int_square(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_square expected int, got {type(n).__name__}")
    return n * n


# ---------------------------------------------------------------------------
# Bridges between Int and String
# ---------------------------------------------------------------------------


@primitive(
    name="int_to_string",
    signature=Signature(("Int",), "String"),
    cost=1.0,
    description="Return the base-10 decimal string representation of the integer.",
)
def int_to_string(n: int) -> str:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"int_to_string expected int, got {type(n).__name__}")
    return str(n)


@primitive(
    name="string_to_int",
    signature=Signature(("String",), "Int"),
    cost=1.0,
    description=(
        "Parse the string as a base-10 integer. "
        "Raises ``ValueError`` on non-numeric input so the executor sandbox "
        "prunes the candidate."
    ),
)
def string_to_int(s: str) -> int:
    if not isinstance(s, str):
        raise TypeError(f"string_to_int expected str, got {type(s).__name__}")
    return int(s)


@primitive(
    name="string_length",
    signature=Signature(("String",), "Int"),
    cost=1.0,
    description="Return the number of characters in the string.",
)
def string_length(s: str) -> int:
    if not isinstance(s, str):
        raise TypeError(f"string_length expected str, got {type(s).__name__}")
    return len(s)


__all__ = [
    "int_abs",
    "int_decrement",
    "int_double",
    "int_half",
    "int_identity",
    "int_increment",
    "int_negate",
    "int_square",
    "int_to_string",
    "int_triple",
    "string_length",
    "string_to_int",
]
