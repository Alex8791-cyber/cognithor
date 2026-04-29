# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Predicate type + closed-set constructors (spec §6.4 + §7.5).

Phase 1.5 introduces Higher-Order primitives (``map_objects``,
``filter_objects``, ``branch``) that take *predicates* and *lambdas*
as arguments. Per spec §6.4, free Python lambdas would unravel the
sandbox guarantee — instead we ship a **closed, enumerable set of
predicate constructors**. The search engine treats them like primitives:
finite, typed, cost-bounded.

Ten leaf constructors + three combinators::

    color_eq(Color)         color_in(tuple[Color, ...])
    size_eq(Int)            size_gt(Int)            size_lt(Int)
    is_rectangle()          is_square()
    is_largest_in(ObjectSet)  is_smallest_in(ObjectSet)
    touches_border(grid_shape)
    not(Predicate)          and(Predicate, Predicate)   or(Predicate, Predicate)

The :func:`evaluate_predicate` function takes a Predicate plus a context
(the Object under test plus the host grid + ObjectSet for primitives
that need them) and returns ``bool``. It's pure — same inputs always
produce the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet

# ---------------------------------------------------------------------------
# Predicate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """A geschlossener (closed-set) Predicate over an Object.

    ``constructor`` is one of the names in :data:`PREDICATE_CONSTRUCTORS`.
    ``args`` is a tuple of primitive values (int, str, tuple of int,
    Predicate, ObjectSet) — never a free Python callable.

    Phase 1.5 only allows Object as the predicate domain; future
    phases may extend to Mask / Cell.
    """

    constructor: str
    args: tuple[Any, ...] = ()
    domain: str = "Object"
    output_type: str = "Bool"

    def __post_init__(self) -> None:
        if self.constructor not in PREDICATE_CONSTRUCTORS:
            raise ValueError(
                f"Unknown predicate constructor {self.constructor!r}; "
                f"allowed: {sorted(PREDICATE_CONSTRUCTORS)}"
            )

    def to_source(self) -> str:
        if not self.args:
            return f"{self.constructor}()"
        rendered: list[str] = []
        for a in self.args:
            if isinstance(a, Predicate):
                rendered.append(a.to_source())
            elif isinstance(a, tuple):
                inner = ", ".join(repr(x) for x in a)
                rendered.append(f"({inner})")
            else:
                rendered.append(repr(a))
        return f"{self.constructor}({', '.join(rendered)})"


# ---------------------------------------------------------------------------
# Constructor registry — closed set, enumeration-ready.
# ---------------------------------------------------------------------------


# Each entry: name → expected arg arity (None = variable).
PREDICATE_CONSTRUCTORS: dict[str, int] = {
    # Leaf constructors.
    "color_eq": 1,
    "color_in": 1,  # single tuple arg
    "size_eq": 1,
    "size_gt": 1,
    "size_lt": 1,
    "is_rectangle": 0,
    "is_square": 0,
    "is_largest_in": 1,
    "is_smallest_in": 1,
    "touches_border": 0,
    # Combinators.
    "not": 1,
    "and": 2,
    "or": 2,
}


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateContext:
    """Side data the evaluator needs that the Object alone doesn't carry.

    ``grid_shape`` is required for ``touches_border``. ``object_set`` is
    required for ``is_largest_in`` / ``is_smallest_in`` — the predicate
    asks "is *this* object the largest in *that* set?".

    Either field may be ``None`` if the predicate doesn't need it; the
    evaluator surfaces ValueError for predicates that require missing
    context so the search engine catches the bug instead of silently
    returning False.
    """

    grid_shape: tuple[int, int] | None = None
    object_set: ObjectSet | None = None


# ---------------------------------------------------------------------------
# Pure evaluation
# ---------------------------------------------------------------------------


def evaluate_predicate(
    pred: Predicate,
    obj: Object,
    context: PredicateContext | None = None,
) -> bool:
    """Compute ``pred`` against ``obj`` (with optional ``context``).

    Strict typing: passing the wrong arg shape (e.g. an int where a
    Predicate is expected for ``not``) raises TypeError. The search
    engine constructs predicates programmatically so this surfaces
    bugs early.
    """
    ctx = context if context is not None else PredicateContext()
    name = pred.constructor

    if name == "color_eq":
        (target,) = pred.args
        return int(obj.color) == int(target)

    if name == "color_in":
        (colors,) = pred.args
        if not isinstance(colors, tuple):
            raise TypeError("color_in: arg must be tuple[Color, ...]")
        return int(obj.color) in {int(c) for c in colors}

    if name == "size_eq":
        (n,) = pred.args
        return obj.size == int(n)
    if name == "size_gt":
        (n,) = pred.args
        return obj.size > int(n)
    if name == "size_lt":
        (n,) = pred.args
        return obj.size < int(n)

    if name == "is_rectangle":
        return obj.is_rectangle()
    if name == "is_square":
        return obj.is_square()

    if name == "is_largest_in":
        (target_set,) = pred.args
        return _is_extreme(obj, target_set, want_max=True)
    if name == "is_smallest_in":
        (target_set,) = pred.args
        return _is_extreme(obj, target_set, want_max=False)

    if name == "touches_border":
        if ctx.grid_shape is None:
            raise ValueError("touches_border requires PredicateContext.grid_shape")
        h, w = ctx.grid_shape
        if not obj.cells:
            return False
        rs = [r for r, _ in obj.cells]
        cs = [c for _, c in obj.cells]
        return min(rs) == 0 or max(rs) == h - 1 or min(cs) == 0 or max(cs) == w - 1

    # Combinators.
    if name == "not":
        (inner,) = pred.args
        if not isinstance(inner, Predicate):
            raise TypeError("not: arg must be a Predicate")
        return not evaluate_predicate(inner, obj, ctx)
    if name == "and":
        a, b = pred.args
        if not (isinstance(a, Predicate) and isinstance(b, Predicate)):
            raise TypeError("and: both args must be Predicate")
        return evaluate_predicate(a, obj, ctx) and evaluate_predicate(b, obj, ctx)
    if name == "or":
        a, b = pred.args
        if not (isinstance(a, Predicate) and isinstance(b, Predicate)):
            raise TypeError("or: both args must be Predicate")
        return evaluate_predicate(a, obj, ctx) or evaluate_predicate(b, obj, ctx)

    raise ValueError(f"evaluate_predicate: unhandled constructor {name!r}")


def _is_extreme(obj: Object, target_set: Any, *, want_max: bool) -> bool:
    if not isinstance(target_set, ObjectSet):
        raise TypeError("is_(largest|smallest)_in: arg must be ObjectSet")
    if target_set.is_empty():
        return False
    sizes = [o.size for o in target_set.objects]
    extreme = max(sizes) if want_max else min(sizes)
    if obj.size != extreme:
        return False
    # Tie-break: discovery-order — only the FIRST object with the
    # extreme size satisfies the predicate. Keeps the result
    # deterministic for cache fingerprinting.
    for candidate in target_set.objects:
        if candidate.size == extreme:
            return candidate is obj or candidate == obj
    return False


# ---------------------------------------------------------------------------
# Convenience builders — short helpers for common cases.
# ---------------------------------------------------------------------------


def color_eq(color: int) -> Predicate:
    return Predicate(constructor="color_eq", args=(color,))


def color_in(colors: tuple[int, ...]) -> Predicate:
    return Predicate(constructor="color_in", args=(colors,))


def size_eq(n: int) -> Predicate:
    return Predicate(constructor="size_eq", args=(n,))


def size_gt(n: int) -> Predicate:
    return Predicate(constructor="size_gt", args=(n,))


def size_lt(n: int) -> Predicate:
    return Predicate(constructor="size_lt", args=(n,))


def is_rectangle() -> Predicate:
    return Predicate(constructor="is_rectangle")


def is_square() -> Predicate:
    return Predicate(constructor="is_square")


def is_largest_in(s: ObjectSet) -> Predicate:
    return Predicate(constructor="is_largest_in", args=(s,))


def is_smallest_in(s: ObjectSet) -> Predicate:
    return Predicate(constructor="is_smallest_in", args=(s,))


def touches_border() -> Predicate:
    return Predicate(constructor="touches_border")


def pred_not(p: Predicate) -> Predicate:
    return Predicate(constructor="not", args=(p,))


def pred_and(a: Predicate, b: Predicate) -> Predicate:
    return Predicate(constructor="and", args=(a, b))


def pred_or(a: Predicate, b: Predicate) -> Predicate:
    return Predicate(constructor="or", args=(a, b))


__all__ = [
    "PREDICATE_CONSTRUCTORS",
    "Predicate",
    "PredicateContext",
    "color_eq",
    "color_in",
    "evaluate_predicate",
    "is_largest_in",
    "is_rectangle",
    "is_smallest_in",
    "is_square",
    "pred_and",
    "pred_not",
    "pred_or",
    "size_eq",
    "size_gt",
    "size_lt",
    "touches_border",
]


# Suppress unused-import lint — np is used elsewhere when this module
# extends. Reserved for future numerics-aware predicates.
_ = np
