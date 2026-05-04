"""``JsonDomain`` — wires JSON catalog + verifier into the
Sprint-26.1 ``DomainRegistry``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.json_dsl.catalog import (
    JsonCatalog,
    build_json_catalog,
)
from cognithor.channels.program_synthesis.domains.json_dsl.types import (
    JSON_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.json_dsl.verifier import (
    JsonVerifier,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


JSON_DOMAIN_METADATA = DomainMetadata(
    name="json",
    display_name="JSON",
    description=(
        "Synthesises JSON transformation pipelines from "
        "(input_json, output_json) examples. Verifier interprets the "
        "pipeline spec deterministically — no eval / exec."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.PROPERTY,
            DomainCapability.BRIDGE,
        }
    ),
    type_tags=JSON_TYPE_TAGS,
    benchmark_name="jq-cookbook",
    benchmark_target=0.65,
    few_shot_bank_path="prompts/pse/json/examples.jsonl",
)


class JsonDomain:
    """:class:`Domain` implementation for JSON synthesis."""

    def __init__(self) -> None:
        self._catalog: JsonCatalog = build_json_catalog()
        self._verifier = JsonVerifier(self._catalog)

    @property
    def metadata(self) -> DomainMetadata:
        return JSON_DOMAIN_METADATA

    def primitives(self) -> JsonCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_json_domain(registry: DomainRegistry) -> None:
    """Register :class:`JsonDomain` with ``registry``."""
    registry.register(JSON_DOMAIN_METADATA, lambda: JsonDomain())
