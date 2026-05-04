"""Datetime synthesis domain (Sprint-26.3).

Module name ``datetime_dsl`` so it doesn't shadow the stdlib
``datetime`` module.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.datetime_dsl.catalog import (
    DATETIME_PRIMITIVE_NAMES,
    DatetimeCatalog,
    DatetimePrimitive,
    build_datetime_catalog,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.domain import (
    DatetimeDomain,
    register_datetime_domain,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.types import (
    DATETIME_TYPE_TAGS,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl.verifier import (
    DatetimeVerifier,
    DatetimeVerifierError,
)

__all__ = [
    "DATETIME_PRIMITIVE_NAMES",
    "DATETIME_TYPE_TAGS",
    "DatetimeCatalog",
    "DatetimeDomain",
    "DatetimePrimitive",
    "DatetimeVerifier",
    "DatetimeVerifierError",
    "build_datetime_catalog",
    "register_datetime_domain",
]
