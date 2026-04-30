# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.5.1 — Local-Edit Repair (plan task 9 slice).

The Refiner's first repair stage is deterministic local edits: small
1-step mutations of the candidate program tree the caller can re-feed
into the verifier in priority order. Local-Edit is cheap (no LLM
call, no symbolic-repair search), so it runs *before* any of the
mode-selected stages from :mod:`refiner.mode_controller`.

Phase-1 already locks the program-tree shape (frozen
``Program``/``InputRef``/``Const`` from
:mod:`channels.program_synthesis.search.candidate`). This module
walks the tree and emits a *generator* of plausible 1-step edits —
the caller (verifier) decides which one to evaluate first based on
its own scoring signal.

Three mutation classes:

* **Primitive substitution** — replace a Program node's primitive
  with another registered primitive of identical
  ``(arity, output_type)`` signature. This catches off-by-one DSL
  picks (``rotate90`` instead of ``rotate270``).
* **Child swap** — for binary primitives, swap the two children's
  positions. Catches argument-order bugs.
* **Color-literal mutation** — replace any ``Const(int, "Color")``
  leaf with each of the 10 ARC colors (excluding the original).
  Catches off-by-one color picks (``recolor(input, 1, 2)`` vs
  ``recolor(input, 1, 5)``).

Each mutation produces a *new* tree (the inputs are frozen, never
mutated). The generator is lazy — the verifier can stop early.

The module is :class:`Phase2Config`-overridable for the color set
(via the existing :data:`HIGH_IMPACT_PRIMITIVES` / similar Phase-1
constants); currently the color set is hardcoded to ARC's 0-9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    Program,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cognithor.channels.program_synthesis.dsl.registry import (
        PrimitiveRegistry,
    )
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


_ARC_COLORS = tuple(range(10))


class LocalEditMutator:
    """Yields 1-step deterministic edits of a candidate program tree.

    Caller supplies the live :class:`PrimitiveRegistry` so the
    primitive-substitution mutator can find compatible alternatives
    by ``(arity, output_type)``. Without a registry the mutator
    falls back to child-swap + color-literal changes only.
    """

    def __init__(self, registry: PrimitiveRegistry | None = None) -> None:
        self._registry = registry

    def mutate(self, program: ProgramNode) -> Iterator[ProgramNode]:
        """Yield each plausible 1-step edit of *program*.

        Order:

        1. Primitive substitutions at the root, then deeper.
        2. Child swaps at the root, then deeper.
        3. Color-literal changes (every ``Const`` leaf, every
           non-original color).

        The original program is never yielded. A program with no
        eligible edits yields nothing.
        """
        yield from self._primitive_substitutions(program)
        yield from self._child_swaps(program)
        yield from self._color_literal_changes(program)

    # -- primitive substitution ---------------------------------------

    def _primitive_substitutions(self, program: ProgramNode) -> Iterator[ProgramNode]:
        if self._registry is None:
            return
        if isinstance(program, Program):
            target_arity = len(program.children)
            target_output = program.output_type
            for replacement in self._registry.all_primitives():
                if replacement.name == program.primitive:
                    continue
                if replacement.signature.arity != target_arity:
                    continue
                if replacement.signature.output != target_output:
                    continue
                yield Program(
                    primitive=replacement.name,
                    children=program.children,
                    output_type=program.output_type,
                )
            for i, child in enumerate(program.children):
                for mutated_child in self._primitive_substitutions(child):
                    new_children = (
                        *program.children[:i],
                        mutated_child,
                        *program.children[i + 1 :],
                    )
                    yield Program(
                        primitive=program.primitive,
                        children=new_children,
                        output_type=program.output_type,
                    )

    # -- child swap ---------------------------------------------------

    def _child_swaps(self, program: ProgramNode) -> Iterator[ProgramNode]:
        if isinstance(program, Program):
            if len(program.children) == 2:
                yield Program(
                    primitive=program.primitive,
                    children=(program.children[1], program.children[0]),
                    output_type=program.output_type,
                )
            for i, child in enumerate(program.children):
                for mutated_child in self._child_swaps(child):
                    new_children = (
                        *program.children[:i],
                        mutated_child,
                        *program.children[i + 1 :],
                    )
                    yield Program(
                        primitive=program.primitive,
                        children=new_children,
                        output_type=program.output_type,
                    )

    # -- color literal change ----------------------------------------

    def _color_literal_changes(self, program: ProgramNode) -> Iterator[ProgramNode]:
        if isinstance(program, Const) and program.output_type == "Color":
            current = program.value
            for c in _ARC_COLORS:
                if c == current:
                    continue
                yield Const(value=c, output_type="Color")
            return
        if isinstance(program, Program):
            for i, child in enumerate(program.children):
                for mutated_child in self._color_literal_changes(child):
                    new_children = (
                        *program.children[:i],
                        mutated_child,
                        *program.children[i + 1 :],
                    )
                    yield Program(
                        primitive=program.primitive,
                        children=new_children,
                        output_type=program.output_type,
                    )


__all__ = ["LocalEditMutator"]
