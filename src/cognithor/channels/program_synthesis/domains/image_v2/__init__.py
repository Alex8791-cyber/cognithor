"""Image / Pixel-Grid synthesis domain (Sprint-26.4).

Sprint-26.4 ships 12+ new pixel-grid primitives targeting ARC-AGI
training tasks the Sprint-10 catalog couldn't solve: symmetry
completion, anchor-based alignment, conditional fill, pattern tiling,
self-tile-by-mask.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.image_v2.catalog import (
    IMAGE_V2_PRIMITIVE_NAMES,
    ImageV2Catalog,
    ImageV2Primitive,
    build_image_v2_catalog,
)
from cognithor.channels.program_synthesis.domains.image_v2.domain import (
    ImageV2Domain,
    register_image_v2_domain,
)
from cognithor.channels.program_synthesis.domains.image_v2.types import (
    IMAGE_V2_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.image_v2.verifier import (
    ImageV2Verifier,
    ImageV2VerifierError,
)

__all__ = [
    "IMAGE_V2_PRIMITIVE_NAMES",
    "IMAGE_V2_TYPE_TAGS",
    "ImageV2Catalog",
    "ImageV2Domain",
    "ImageV2Primitive",
    "ImageV2Verifier",
    "ImageV2VerifierError",
    "build_image_v2_catalog",
    "register_image_v2_domain",
]
