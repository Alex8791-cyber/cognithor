"""``BytesDomain`` — wires BinaryData catalog + verifier into the
Sprint-26.1 ``DomainRegistry``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.catalog import (
    BytesCatalog,
    build_bytes_catalog,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.types import (
    BYTES_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.verifier import (
    BytesVerifier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


BYTES_DOMAIN_METADATA = DomainMetadata(
    name="bytes",
    display_name="BinaryData",
    description=(
        "Synthesises byte-level read/write/encode/hash/bitfield "
        "pipelines. Endianness is always explicit. Encode/decode "
        "pairs satisfy the roundtrip property "
        "decode(encode(x)) == x."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.PROPERTY,
            DomainCapability.BRIDGE,
        }
    ),
    type_tags=BYTES_TYPE_TAGS,
    benchmark_name="binary-format-200",
    benchmark_target=0.70,
    few_shot_bank_path="prompts/pse/bytes/examples.jsonl",
)


class BytesDomain:
    """:class:`Domain` implementation for BinaryData synthesis."""

    def __init__(self) -> None:
        self._catalog: BytesCatalog = build_bytes_catalog()
        self._verifier = BytesVerifier(self._catalog)

    @property
    def metadata(self) -> DomainMetadata:
        return BYTES_DOMAIN_METADATA

    def primitives(self) -> BytesCatalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_bytes_domain(registry: DomainRegistry) -> None:
    """Register :class:`BytesDomain` with ``registry``."""
    registry.register(BYTES_DOMAIN_METADATA, lambda: BytesDomain())
