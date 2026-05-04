"""``ImageV2Domain`` — wires Image V2 catalog + verifier into the
Sprint-26.1 ``DomainRegistry``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.image_v2.catalog import (
    ImageV2Catalog,
    build_image_v2_catalog,
)
from cognithor.channels.program_synthesis.domains.image_v2.types import (
    IMAGE_V2_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.image_v2.verifier import (
    ImageV2Verifier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


IMAGE_V2_DOMAIN_METADATA = DomainMetadata(
    name="image_v2",
    display_name="Image (Sprint-26.4)",
    description=(
        "Sprint-26.4 ARC-AGI primitive expansion — symmetry, "
        "anchors, conditional fill, pattern tiling, self-tile-by-mask, "
        "connected components."
    ),
    capabilities=frozenset(
        {
            DomainCapability.SYNTHESISE,
            DomainCapability.PROPERTY,
        }
    ),
    type_tags=IMAGE_V2_TYPE_TAGS,
    benchmark_name="arc-agi-1-training",
    benchmark_target=0.15,
    few_shot_bank_path="prompts/pse/image/examples.jsonl",
)


class ImageV2Domain:
    def __init__(self) -> None:
        self._catalog: ImageV2Catalog = build_image_v2_catalog()
        self._verifier = ImageV2Verifier(self._catalog)

    @property
    def metadata(self) -> DomainMetadata:
        return IMAGE_V2_DOMAIN_METADATA

    def primitives(self) -> ImageV2Catalog:
        return self._catalog

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        return self._verifier.verify(program, examples)


def register_image_v2_domain(registry: DomainRegistry) -> None:
    """Register :class:`ImageV2Domain` with ``registry``."""
    registry.register(IMAGE_V2_DOMAIN_METADATA, lambda: ImageV2Domain())
