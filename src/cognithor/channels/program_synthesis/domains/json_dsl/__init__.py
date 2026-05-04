"""JSON synthesis domain (Sprint-26.2).

Module name ``json_dsl`` (not ``json``) so it doesn't shadow the
stdlib ``json`` module.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.json_dsl.catalog import (
    JSON_PRIMITIVE_NAMES,
    JsonCatalog,
    JsonPrimitive,
    build_json_catalog,
)
from cognithor.channels.program_synthesis.domains.json_dsl.domain import (
    JsonDomain,
    register_json_domain,
)
from cognithor.channels.program_synthesis.domains.json_dsl.types import (
    JSON_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.json_dsl.verifier import (
    JsonVerifier,
    JsonVerifierError,
)

__all__ = [
    "JSON_PRIMITIVE_NAMES",
    "JSON_TYPE_TAGS",
    "JsonCatalog",
    "JsonDomain",
    "JsonPrimitive",
    "JsonVerifier",
    "JsonVerifierError",
    "build_json_catalog",
    "register_json_domain",
]
