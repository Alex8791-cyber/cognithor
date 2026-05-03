# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — List/Sequence-DSL primitive family.

PR#4 of Track B and the final piece of the Open-World DSL push.
Adds reductions, reorderings, and bridges over both ``StringList``
(introduced in PR#1) and the new ``IntList`` type, so the engine
can solve demo specs like:

    examples = ((["c", "a", "b"], ["a", "b", "c"]),  # str-list sort
                (["x", "y"],      ["x", "y"]))

    examples = (([1, 2, 3], 6),                     # int-list sum
                ([10, 20], 30))

    examples = (([5, 1, 9], 9),                     # int-list max
                ([3, 7, 4], 7))

Design constraints carry over:

* Pure functions — no IO, no globals, no random.
* Total domains where reasonable. Reductions over an empty list
  raise (``max(empty)`` / ``min(empty)`` are undefined) so the
  executor sandbox prunes the candidate.
* JSON-serialisable signature strings — ``IntList`` and
  ``StringList`` are both registered in :data:`ALLOWED_TYPES`.

The 14 primitives below cover four axes:

* **String-list reorder**:    ``string_list_reverse``,
                              ``string_list_unique``,
                              ``string_list_sort``
* **String-list reduce**:     ``string_list_first``,
                              ``string_list_last``,
                              ``string_list_length``,
                              ``string_list_count_nonempty``
* **Int-list reorder**:       ``int_list_reverse``,
                              ``int_list_sort``
* **Int-list reduce**:        ``int_list_first``,
                              ``int_list_sum``,
                              ``int_list_max``,
                              ``int_list_min``,
                              ``int_list_length``

Reductions to ``Int`` (length / sum / max / min) are the bridges
that connect this family back to the Number-DSL family from PR#3 —
e.g. ``int_increment(int_list_max(...))`` is now a reachable
program.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.registry import primitive
from cognithor.channels.program_synthesis.dsl.signatures import Signature

# ---------------------------------------------------------------------------
# StringList — reorder
# ---------------------------------------------------------------------------


@primitive(
    name="string_list_reverse",
    signature=Signature(("StringList",), "StringList"),
    cost=1.0,
    description="Return the list with elements in reverse order.",
)
def string_list_reverse(parts: list[str]) -> list[str]:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_reverse expected list, got {type(parts).__name__}")
    return list(reversed(parts))


@primitive(
    name="string_list_unique",
    signature=Signature(("StringList",), "StringList"),
    cost=1.0,
    description="Deduplicate the list, preserving first-seen order.",
)
def string_list_unique(parts: list[str]) -> list[str]:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_unique expected list, got {type(parts).__name__}")
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


@primitive(
    name="string_list_sort",
    signature=Signature(("StringList",), "StringList"),
    cost=1.0,
    description="Return the list sorted lexicographically (Python default str ordering).",
)
def string_list_sort(parts: list[str]) -> list[str]:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_sort expected list, got {type(parts).__name__}")
    return sorted(parts)


# ---------------------------------------------------------------------------
# StringList — reduce
# ---------------------------------------------------------------------------


@primitive(
    name="string_list_first",
    signature=Signature(("StringList",), "String"),
    cost=1.0,
    description="Return the first element, or '' for an empty list.",
)
def string_list_first(parts: list[str]) -> str:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_first expected list, got {type(parts).__name__}")
    return parts[0] if parts else ""


@primitive(
    name="string_list_last",
    signature=Signature(("StringList",), "String"),
    cost=1.0,
    description="Return the last element, or '' for an empty list.",
)
def string_list_last(parts: list[str]) -> str:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_last expected list, got {type(parts).__name__}")
    return parts[-1] if parts else ""


@primitive(
    name="string_list_length",
    signature=Signature(("StringList",), "Int"),
    cost=1.0,
    description="Return the number of elements in the list.",
)
def string_list_length(parts: list[str]) -> int:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_length expected list, got {type(parts).__name__}")
    return len(parts)


@primitive(
    name="string_list_count_nonempty",
    signature=Signature(("StringList",), "Int"),
    cost=1.0,
    description="Return the number of elements that are not the empty string.",
)
def string_list_count_nonempty(parts: list[str]) -> int:
    if not isinstance(parts, list):
        raise TypeError(f"string_list_count_nonempty expected list, got {type(parts).__name__}")
    return sum(1 for p in parts if p)


# ---------------------------------------------------------------------------
# IntList — reorder
# ---------------------------------------------------------------------------


@primitive(
    name="int_list_reverse",
    signature=Signature(("IntList",), "IntList"),
    cost=1.0,
    description="Return the list of ints in reverse order.",
)
def int_list_reverse(values: list[int]) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"int_list_reverse expected list, got {type(values).__name__}")
    return list(reversed(values))


@primitive(
    name="int_list_sort",
    signature=Signature(("IntList",), "IntList"),
    cost=1.0,
    description="Return the list of ints sorted ascending.",
)
def int_list_sort(values: list[int]) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"int_list_sort expected list, got {type(values).__name__}")
    return sorted(values)


# ---------------------------------------------------------------------------
# IntList — reduce
# ---------------------------------------------------------------------------


@primitive(
    name="int_list_first",
    signature=Signature(("IntList",), "Int"),
    cost=1.0,
    description="Return the first integer; raises on empty list.",
)
def int_list_first(values: list[int]) -> int:
    if not isinstance(values, list):
        raise TypeError(f"int_list_first expected list, got {type(values).__name__}")
    if not values:
        raise ValueError("int_list_first: empty list")
    return int(values[0])


@primitive(
    name="int_list_sum",
    signature=Signature(("IntList",), "Int"),
    cost=1.0,
    description="Return the sum of the integers; 0 for an empty list.",
)
def int_list_sum(values: list[int]) -> int:
    if not isinstance(values, list):
        raise TypeError(f"int_list_sum expected list, got {type(values).__name__}")
    return sum(int(v) for v in values)


@primitive(
    name="int_list_max",
    signature=Signature(("IntList",), "Int"),
    cost=1.0,
    description="Return the maximum integer; raises on empty list.",
)
def int_list_max(values: list[int]) -> int:
    if not isinstance(values, list):
        raise TypeError(f"int_list_max expected list, got {type(values).__name__}")
    if not values:
        raise ValueError("int_list_max: empty list")
    return int(max(values))


@primitive(
    name="int_list_min",
    signature=Signature(("IntList",), "Int"),
    cost=1.0,
    description="Return the minimum integer; raises on empty list.",
)
def int_list_min(values: list[int]) -> int:
    if not isinstance(values, list):
        raise TypeError(f"int_list_min expected list, got {type(values).__name__}")
    if not values:
        raise ValueError("int_list_min: empty list")
    return int(min(values))


@primitive(
    name="int_list_length",
    signature=Signature(("IntList",), "Int"),
    cost=1.0,
    description="Return the number of elements in the int list.",
)
def int_list_length(values: list[int]) -> int:
    if not isinstance(values, list):
        raise TypeError(f"int_list_length expected list, got {type(values).__name__}")
    return len(values)


__all__ = [
    "int_list_first",
    "int_list_length",
    "int_list_max",
    "int_list_min",
    "int_list_reverse",
    "int_list_sort",
    "int_list_sum",
    "string_list_count_nonempty",
    "string_list_first",
    "string_list_last",
    "string_list_length",
    "string_list_reverse",
    "string_list_sort",
    "string_list_unique",
]
