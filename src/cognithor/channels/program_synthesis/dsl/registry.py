# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""DSL primitive registry (spec §7.1).

A primitive is a pure function ``Tuple[T1, ...] -> T`` together with a
static :class:`Signature`, a numeric ``cost`` (Occam-prior), a docstring,
and at least one input/output example.

The registry is process-local (no globals leak across tests) but the
canonical instance is exposed as :data:`REGISTRY` for the rest of the
package; tests build fresh registries via :class:`PrimitiveRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.core.exceptions import (
    DSLError,
    UnknownPrimitiveError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cognithor.channels.program_synthesis.dsl.signatures import Signature


@dataclass(frozen=True)
class PrimitiveSpec:
    """Static description of one DSL primitive."""

    name: str
    signature: Signature
    cost: float
    fn: Callable[..., Any]
    description: str = ""
    examples: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise DSLError(f"Invalid primitive name: {self.name!r}")
        if self.cost < 0:
            raise DSLError(f"Primitive cost must be >= 0, got {self.cost}")


class PrimitiveRegistry:
    """Append-only registry of DSL primitives.

    A primitive may be registered exactly once per registry instance.
    Re-registration raises :class:`DSLError`.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, PrimitiveSpec] = {}
        self._by_arity: dict[int, list[PrimitiveSpec]] = {}

    # -- Registration ---------------------------------------------------

    def register(self, spec: PrimitiveSpec) -> PrimitiveSpec:
        if spec.name in self._by_name:
            raise DSLError(f"Primitive already registered: {spec.name!r}")
        self._by_name[spec.name] = spec
        self._by_arity.setdefault(spec.signature.arity, []).append(spec)
        return spec

    # -- Lookup ---------------------------------------------------------

    def get(self, name: str) -> PrimitiveSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise UnknownPrimitiveError(name) from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    # -- Iteration ------------------------------------------------------

    def all_primitives(self) -> tuple[PrimitiveSpec, ...]:
        return tuple(self._by_name.values())

    def primitives_with_arity(self, arity: int) -> tuple[PrimitiveSpec, ...]:
        return tuple(self._by_arity.get(arity, ()))

    def primitives_by_output_type(self, output_type: str) -> tuple[PrimitiveSpec, ...]:
        return tuple(
            spec for spec in self._by_name.values() if spec.signature.output == output_type
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())


# Process-local singleton used by the rest of the package. Tests should
# construct their own :class:`PrimitiveRegistry` instances rather than
# mutating this one.
REGISTRY: PrimitiveRegistry = PrimitiveRegistry()


def primitive(
    *,
    name: str,
    signature: Signature,
    cost: float,
    description: str = "",
    examples: tuple[tuple[str, str], ...] = (),
    registry: PrimitiveRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers ``fn`` as a primitive in ``registry``.

    Used as::

        @primitive(name="rotate90", signature=Signature(("Grid",), "Grid"), cost=1.0)
        def rotate90(grid):
            return np.rot90(grid, k=-1).copy()
    """

    target = registry if registry is not None else REGISTRY

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = PrimitiveSpec(
            name=name,
            signature=signature,
            cost=cost,
            fn=fn,
            description=description,
            examples=examples,
        )
        target.register(spec)
        return fn

    return decorator
