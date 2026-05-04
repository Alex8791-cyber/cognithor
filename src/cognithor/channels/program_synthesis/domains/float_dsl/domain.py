"""``FloatDomain`` — wires Float catalog + verifier into the
Sprint-26.1 ``DomainRegistry``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.float_dsl.catalog import (
    FloatCatalog,
    build_float_catalog,
)
from cognithor.channels.program_synthesis.domains.float_dsl.types import (
    FLOAT_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.float_dsl.verifier import (
    FloatVerifier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


FLOAT_DOMAIN_METADATA = DomainMetadata(
    name="float",
    display_name="Float-Precision",
    description=(
        "Synthesises floating-point-aware programs that handle "
        "epsilon, NaN, Inf, denormals, and accumulator drift correctly."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.PROPERTY,
        }
    ),
    type_tags=FLOAT_TYPE_TAGS,
    benchmark_name="float-precision-100",
    benchmark_target=0.70,
    few_shot_bank_path="prompts/pse/float/examples.jsonl",
)


class FloatDomain:
    """:class:`Domain` implementation for Float-precision synthesis."""

    def __init__(self) -> None:
        self._catalog: FloatCatalog = build_float_catalog()
        self._verifier = FloatVerifier(self._catalog)

    @property
    def metadata(self) -> DomainMetadata:
        return FLOAT_DOMAIN_METADATA

    def primitives(self) -> FloatCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_float_domain(registry: DomainRegistry) -> None:
    """Register :class:`FloatDomain` with ``registry``."""
    registry.register(FLOAT_DOMAIN_METADATA, lambda: FloatDomain())
