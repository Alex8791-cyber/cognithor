"""Float-precision synthesis domain (Sprint-26.3).

15+ primitives covering precision-aware floating-point operations:
Kahan summation, NaN handling, denormal detection, safe division,
relative error.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.float_dsl.catalog import (
    FLOAT_PRIMITIVE_NAMES,
    FloatCatalog,
    FloatPrimitive,
    build_float_catalog,
)
from cognithor.channels.program_synthesis.domains.float_dsl.domain import (
    FloatDomain,
    register_float_domain,
)
from cognithor.channels.program_synthesis.domains.float_dsl.types import (
    FLOAT_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.float_dsl.verifier import (
    FloatVerifier,
    FloatVerifierError,
)

__all__ = [
    "FLOAT_PRIMITIVE_NAMES",
    "FLOAT_TYPE_TAGS",
    "FloatCatalog",
    "FloatDomain",
    "FloatPrimitive",
    "FloatVerifier",
    "FloatVerifierError",
    "build_float_catalog",
    "register_float_domain",
]
