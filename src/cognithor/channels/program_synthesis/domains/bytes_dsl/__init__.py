"""BinaryData synthesis domain (Sprint-26.4).

Module name ``bytes_dsl`` so it doesn't shadow the stdlib ``bytes``
type or builtins.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.bytes_dsl.catalog import (
    BYTES_PRIMITIVE_NAMES,
    BytesCatalog,
    BytesPrimitive,
    build_bytes_catalog,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.domain import (
    BytesDomain,
    register_bytes_domain,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.types import (
    BYTES_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl.verifier import (
    BytesVerifier,
    BytesVerifierError,
)

__all__ = [
    "BYTES_PRIMITIVE_NAMES",
    "BYTES_TYPE_TAGS",
    "BytesCatalog",
    "BytesDomain",
    "BytesPrimitive",
    "BytesVerifier",
    "BytesVerifierError",
    "build_bytes_catalog",
    "register_bytes_domain",
]
