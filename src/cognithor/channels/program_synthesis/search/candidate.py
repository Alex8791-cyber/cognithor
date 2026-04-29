# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Program-tree representation used by the enumerative search (spec §6.2).

A program is a typed tree of DSL primitive applications, leafed by either
``InputRef`` (the task's input grid) or ``Const`` (a literal value such
as a color index).

Every node is frozen / immutable. Equality is structural so the
observational-equivalence pruner can use programs directly as keys.
``stable_hash`` returns a SHA-256 over the canonical source form so the
hash is portable across Python versions and processes (the sandbox
worker re-hashes for cache validation).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class InputRef:
    """A reference to the task's input grid.

    Always the leaf of a Grid-typed sub-tree. ``output_type`` is fixed at
    ``"Grid"`` because the input is always the raw grid handed to the
    program; future generalisations (multi-input tasks) would extend
    this rather than mutate it.
    """

    output_type: str = "Grid"

    def to_source(self) -> str:
        return "input"

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Const:
    """A literal value (color index, integer parameter, enum tag, ...)."""

    value: int | str
    output_type: str

    def to_source(self) -> str:
        # Ints render as their decimal value, strings keep their quotes
        # so the source round-trips through eval() (used only by the
        # `cognithor pse explain` formatter, never by the search engine).
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return str(self.value)
        return repr(self.value)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Program:
    """A non-leaf node: a primitive applied to an ordered tuple of children.

    ``children`` are themselves :class:`ProgramNode` (one of Program,
    InputRef, or Const). The structural ordering is part of the
    program's identity — search-side equivalence relies on it.
    """

    primitive: str
    children: tuple[ProgramNode, ...]
    output_type: str

    def to_source(self) -> str:
        if not self.children:
            return f"{self.primitive}()"
        args = ", ".join(c.to_source() for c in self.children)
        return f"{self.primitive}({args})"

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def stable_hash(self) -> str:
        """SHA-256 over the canonical source form.

        Stable across processes and Python versions. Cache keys for
        synthesized programs use this as their suffix.
        """
        return "sha256:" + hashlib.sha256(self.to_source().encode("utf-8")).hexdigest()


# Union of all node kinds. For runtime isinstance checks consumers
# should use the explicit tuple ``(Program, InputRef, Const)``; this
# alias is annotation-only.
type ProgramNode = Program | InputRef | Const
