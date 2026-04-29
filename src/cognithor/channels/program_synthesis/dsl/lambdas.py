# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Lambda type + closed-set constructors (spec §6.4 + §7.5).

Higher-order primitives (``map_objects``, ``branch``) take Lambda
arguments. Like Predicates, the spec mandates a *closed* enumerable
set — free Python lambdas would unravel the sandbox guarantee.

Phase 1.5 ships three Lambda constructors over Object → Object:

* ``identity_lambda()`` — passes the object through unchanged.
* ``recolor_lambda(new_color)`` — repaints every cell with new_color.
* ``shift_lambda(dy, dx)`` — translates every cell by (dy, dx).

The :func:`evaluate_lambda` function takes a Lambda + an Object and
returns the transformed Object. It's pure — same inputs, same output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognithor.channels.program_synthesis.dsl.types_grid import Object


@dataclass(frozen=True)
class Lambda:
    """A geschlossener (closed-set) 1-arg function over Object.

    ``constructor`` is one of the names in :data:`LAMBDA_CONSTRUCTORS`.
    ``args`` is a tuple of primitive values — never a free Python
    callable.

    Phase 1.5 only allows Object → Object lambdas; future phases may
    extend to Object → T or T → T.
    """

    constructor: str
    args: tuple[Any, ...] = ()
    variable_type: str = "Object"
    output_type: str = "Object"

    def __post_init__(self) -> None:
        if self.constructor not in LAMBDA_CONSTRUCTORS:
            raise ValueError(
                f"Unknown lambda constructor {self.constructor!r}; "
                f"allowed: {sorted(LAMBDA_CONSTRUCTORS)}"
            )

    def to_source(self) -> str:
        if not self.args:
            return f"{self.constructor}()"
        rendered = ", ".join(repr(a) for a in self.args)
        return f"{self.constructor}({rendered})"


# Constructor registry — closed set, enumeration-ready.
LAMBDA_CONSTRUCTORS: dict[str, int] = {
    "identity_lambda": 0,
    "recolor_lambda": 1,
    "shift_lambda": 2,
}


# ---------------------------------------------------------------------------
# Pure evaluation
# ---------------------------------------------------------------------------


def evaluate_lambda(fn: Lambda, obj: Object) -> Object:
    """Apply ``fn`` to ``obj`` and return the transformed Object.

    Strict typing: invalid arg shapes raise TypeError. The output is a
    fresh Object — input is never mutated (Object is frozen anyway,
    but the contract is explicit).
    """
    name = fn.constructor

    if name == "identity_lambda":
        return obj

    if name == "recolor_lambda":
        (new_color,) = fn.args
        if not isinstance(new_color, int) or isinstance(new_color, bool):
            raise TypeError("recolor_lambda: arg must be int")
        if not 0 <= new_color <= 9:
            raise TypeError(f"recolor_lambda: color {new_color} out of ARC range 0..9")
        return Object(color=new_color, cells=obj.cells)

    if name == "shift_lambda":
        dy, dx = fn.args
        if not (isinstance(dy, int) and isinstance(dx, int)) or (
            isinstance(dy, bool) or isinstance(dx, bool)
        ):
            raise TypeError("shift_lambda: dy and dx must both be int")
        new_cells = tuple((r + dy, c + dx) for r, c in obj.cells)
        return Object(color=obj.color, cells=new_cells)

    raise ValueError(f"evaluate_lambda: unhandled constructor {name!r}")


# ---------------------------------------------------------------------------
# Convenience builders.
# ---------------------------------------------------------------------------


def identity_lambda() -> Lambda:
    return Lambda(constructor="identity_lambda")


def recolor_lambda(new_color: int) -> Lambda:
    return Lambda(constructor="recolor_lambda", args=(new_color,))


def shift_lambda(dy: int, dx: int) -> Lambda:
    return Lambda(constructor="shift_lambda", args=(dy, dx))


__all__ = [
    "LAMBDA_CONSTRUCTORS",
    "Lambda",
    "evaluate_lambda",
    "identity_lambda",
    "recolor_lambda",
    "shift_lambda",
]
