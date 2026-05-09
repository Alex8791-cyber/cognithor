"""Central registration of all Sprint-26 domains.

Sprint-26.1-26.4 ship the seven domain modules (sql, json_dsl,
datetime_dsl, ast_dsl, float_dsl, bytes_dsl, image_v2). Each module
provides a ``register_<name>_domain(registry)`` helper that adds its
metadata + factory to the canonical :data:`DOMAIN_REGISTRY`. This
wrapper calls all seven in order so the gateway boot path only has to
say::

    from cognithor.channels.program_synthesis.domains import (
        DOMAIN_REGISTRY,
        register_all_sprint26_domains,
    )

    register_all_sprint26_domains(DOMAIN_REGISTRY)

Calling the wrapper twice on the same registry raises
``DomainAlreadyRegisteredError`` from the underlying registry — the
wrapper does NOT swallow that, because a double-init usually points
at a real wiring bug.

For idempotent gateway-boot wiring (which may run more than once in
the same Python process — test fixtures, hot-reload), use
:func:`register_missing_sprint26_domains` instead. It registers only
the Sprint-26 domains not yet present in the registry and returns the
list of names it actually registered, so the caller can log
appropriately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from cognithor.channels.program_synthesis.domains.ast_dsl import (
    register_ast_domain,
)
from cognithor.channels.program_synthesis.domains.bytes_dsl import (
    register_bytes_domain,
)
from cognithor.channels.program_synthesis.domains.datetime_dsl import (
    register_datetime_domain,
)
from cognithor.channels.program_synthesis.domains.float_dsl import (
    register_float_domain,
)
from cognithor.channels.program_synthesis.domains.image_v2 import (
    register_image_v2_domain,
)
from cognithor.channels.program_synthesis.domains.json_dsl import (
    register_json_domain,
)
from cognithor.channels.program_synthesis.domains.sql import (
    register_sql_domain,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.domains.registry import (
        DomainRegistry,
    )


# Stable ordering — matches Owner-Decision D3 (Sprint-26 priority
# sequence): SQL → JSON → Datetime → AST → BinaryData → Float →
# Image-Boost. Tests assert on this order so any future re-shuffle
# is intentional.
SPRINT26_DOMAIN_NAMES: tuple[str, ...] = (
    "sql",
    "json",
    "datetime",
    "ast",
    "bytes",
    "float",
    "image_v2",
)


# Name → registrar function mapping in the SPRINT26_DOMAIN_NAMES order.
# Used by both ``register_all_sprint26_domains`` (registers all) and
# ``register_missing_sprint26_domains`` (registers only the gaps).
_SPRINT26_REGISTRARS: tuple[tuple[str, Callable[[DomainRegistry], None]], ...] = (
    ("sql", register_sql_domain),
    ("json", register_json_domain),
    ("datetime", register_datetime_domain),
    ("ast", register_ast_domain),
    ("bytes", register_bytes_domain),
    ("float", register_float_domain),
    ("image_v2", register_image_v2_domain),
)


def register_all_sprint26_domains(registry: DomainRegistry) -> None:
    """Register every Sprint-26 domain into ``registry``.

    Calls the seven ``register_<name>_domain`` helpers in
    Owner-Decision-D3 order. Each underlying registration uses the
    lazy plugin-loader pattern from Sprint-26.1, so the actual domain
    instances aren't built until the synthesis pipeline asks for them.

    Raises ``DomainAlreadyRegisteredError`` when invoked twice with
    the same name — that's a wiring bug worth surfacing rather than
    silently no-oping. Gateway-boot wiring that may run more than once
    in the same process (test fixtures, hot-reload) should use
    :func:`register_missing_sprint26_domains` instead.
    """
    for _name, register_fn in _SPRINT26_REGISTRARS:
        register_fn(registry)


def register_missing_sprint26_domains(registry: DomainRegistry) -> list[str]:
    """Register only the Sprint-26 domains not yet present in ``registry``.

    Idempotent: calling twice on a fully-populated registry returns
    ``[]`` and mutates nothing. Calling on a partially-populated
    registry fills only the gaps and returns the names that were
    actually registered, in Owner-D3 order.

    The intended caller is the gateway boot path, which runs once per
    Python process under normal operation but may run multiple times
    in test fixtures and hot-reload paths. Compared to
    ``register_all_sprint26_domains``, this function never raises
    ``DomainAlreadyRegisteredError`` because it pre-checks before
    each registration.

    Foreign domain names (anything outside ``SPRINT26_DOMAIN_NAMES``)
    are ignored — this function only manages the Sprint-26 catalog.

    Returns:
        The Sprint-26 names that were actually registered by THIS call.
    """
    newly_registered: list[str] = []
    for name, register_fn in _SPRINT26_REGISTRARS:
        if name in registry:
            continue
        register_fn(registry)
        newly_registered.append(name)
    return newly_registered
