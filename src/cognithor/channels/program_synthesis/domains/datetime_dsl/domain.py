"""``DatetimeDomain`` — wires Datetime catalog + verifier into the
Sprint-26.1 ``DomainRegistry``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.catalog import (
    DatetimeCatalog,
    build_datetime_catalog,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.types import (
    DATETIME_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.verifier import (
    DatetimeVerifier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


DATETIME_DOMAIN_METADATA = DomainMetadata(
    name="datetime",
    display_name="Datetime",
    description=(
        "Synthesises tz-aware datetime transformation pipelines. "
        "Verifier runs the pipeline deterministically and respects "
        "DST + leap-day calendar arithmetic."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.PROPERTY,
            DomainCapability.BRIDGE,
        }
    ),
    type_tags=DATETIME_TYPE_TAGS,
    benchmark_name="tempeval-3",
    benchmark_target=0.80,
    few_shot_bank_path="prompts/pse/datetime/examples.jsonl",
)


class DatetimeDomain:
    """:class:`Domain` implementation for Datetime synthesis."""

    def __init__(self) -> None:
        self._catalog: DatetimeCatalog = build_datetime_catalog()
        self._verifier = DatetimeVerifier(self._catalog)

    @property
    def metadata(self) -> DomainMetadata:
        return DATETIME_DOMAIN_METADATA

    def primitives(self) -> DatetimeCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_datetime_domain(registry: DomainRegistry) -> None:
    """Register :class:`DatetimeDomain` with ``registry``."""
    registry.register(DATETIME_DOMAIN_METADATA, lambda: DatetimeDomain())
