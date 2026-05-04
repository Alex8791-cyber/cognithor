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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def register_all_sprint26_domains(registry: DomainRegistry) -> None:
    """Register every Sprint-26 domain into ``registry``.

    Calls the seven ``register_<name>_domain`` helpers in
    Owner-Decision-D3 order. Each underlying registration uses the
    lazy plugin-loader pattern from Sprint-26.1, so the actual domain
    instances aren't built until the synthesis pipeline asks for them.

    Raises ``DomainAlreadyRegisteredError`` when invoked twice with
    the same name — that's a wiring bug worth surfacing rather than
    silently no-oping.
    """
    register_sql_domain(registry)
    register_json_domain(registry)
    register_datetime_domain(registry)
    register_ast_domain(registry)
    register_bytes_domain(registry)
    register_float_domain(registry)
    register_image_v2_domain(registry)
