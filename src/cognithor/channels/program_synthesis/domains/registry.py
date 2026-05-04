"""``DomainRegistry`` — process-local registry of Sprint-26 domains.

Domains are registered via :meth:`DomainRegistry.register` (or via the
plugin-loader entry point ``cognithor.pse.domains``). The registry is
intentionally process-local; tests construct a fresh instance to keep
state out of globals. The ``DOMAIN_REGISTRY`` module attribute holds
the canonical instance the synthesis pipeline uses.

The registry only stores **metadata + factory** — instantiating a
domain may be expensive (loading few-shot banks, opening duckdb
connections), so we lazily build domain objects on first ``.get(name)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cognithor.channels.program_synthesis.domains.base import (
        Domain,
        DomainMetadata,
    )

log = get_logger(__name__)


class DomainAlreadyRegisteredError(Exception):
    """Raised when ``register`` is called twice for the same domain."""


class UnknownDomainError(KeyError):
    """Raised when ``get`` / ``metadata`` is asked for a missing domain."""


class DomainRegistry:
    """Append-only registry of Sprint-26 domains.

    Plugin-loader pattern: an external module (or a Sprint-26 sub-PR)
    can populate the registry by importing
    :data:`DOMAIN_REGISTRY` and calling
    :meth:`register`. The synthesis pipeline reads via :meth:`get`.

    Two-step instantiation lets us register cheap *metadata* at
    import-time (so ``list_metadata()`` works without paying the cost
    of building every verifier) and only construct the full
    :class:`Domain` on demand.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Domain]] = {}
        self._metadata: dict[str, DomainMetadata] = {}
        self._instances: dict[str, Domain] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        metadata: DomainMetadata,
        factory: Callable[[], Domain],
    ) -> None:
        """Register a new domain.

        ``metadata`` is the static description; ``factory`` is a
        zero-arg callable that returns a fully-constructed
        :class:`Domain` instance. The factory is called at most once
        per registry instance.

        Re-registration of the same name raises
        :class:`DomainAlreadyRegisteredError` to avoid silent overrides
        — callers that want to swap a domain (e.g. test fixtures) must
        construct a fresh registry.
        """
        if metadata.name in self._metadata:
            msg = f"Domain {metadata.name!r} is already registered"
            raise DomainAlreadyRegisteredError(msg)
        self._metadata[metadata.name] = metadata
        self._factories[metadata.name] = factory
        log.debug(
            "domain_registered",
            name=metadata.name,
            capabilities=sorted(c.value for c in metadata.capabilities),
            benchmark=metadata.benchmark_name,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Domain:
        """Return the (lazily-built) :class:`Domain` instance.

        Raises :class:`UnknownDomainError` when ``name`` was never
        registered.
        """
        if name not in self._factories:
            available = sorted(self._factories)
            msg = f"Domain {name!r} is not registered. Available: {available}"
            raise UnknownDomainError(msg)
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
            log.debug("domain_instantiated", name=name)
        return self._instances[name]

    def metadata(self, name: str) -> DomainMetadata:
        """Return the metadata block for ``name`` without instantiating."""
        if name not in self._metadata:
            available = sorted(self._metadata)
            msg = f"Domain {name!r} is not registered. Available: {available}"
            raise UnknownDomainError(msg)
        return self._metadata[name]

    # ------------------------------------------------------------------
    # Iteration + introspection
    # ------------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def names(self) -> list[str]:
        """Return the registered domain names sorted alphabetically."""
        return sorted(self._metadata)

    def list_metadata(self) -> list[DomainMetadata]:
        """Return all registered metadata blocks (sorted by name)."""
        return [self._metadata[n] for n in self.names()]

    def filter_by_capability(
        self,
        *capabilities: str,
    ) -> Iterable[DomainMetadata]:
        """Yield metadata of domains that hold *every* capability."""
        wanted = frozenset(capabilities)
        for meta in self.list_metadata():
            if wanted <= {c.value for c in meta.capabilities}:
                yield meta


# Canonical, process-local registry.  The Sprint-26 sub-PRs (SQL,
# JSON, Datetime, …) populate this on import via their own module's
# top-level ``DOMAIN_REGISTRY.register(...)`` call.
DOMAIN_REGISTRY: DomainRegistry = DomainRegistry()
