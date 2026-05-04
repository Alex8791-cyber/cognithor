"""Bridge registry + ``BridgeOperator`` (Sprint-26.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class BridgeNotWhitelistedError(Exception):
    """Raised when a bridge pair is outside the Sprint-26 whitelist."""


@dataclass(frozen=True)
class BridgeOperator:
    """One typed transformer between two domains.

    ``from_type`` and ``to_type`` are canonical type-tag strings —
    e.g. ``"json"`` (the source domain's value type) and
    ``"sql_literal"`` (the destination's wrapped form). The exact
    pair must appear in :data:`SPRINT26_BRIDGE_WHITELIST` or
    :class:`BridgeRegistry.register` raises.
    """

    from_type: str
    to_type: str
    fn: Callable[[Any], Any]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.from_type or not self.to_type:
            msg = "Bridge from_type and to_type must be non-empty"
            raise ValueError(msg)
        if self.from_type == self.to_type:
            msg = "Bridge from_type and to_type must differ"
            raise ValueError(msg)


class BridgeRegistry:
    """Append-only registry of cross-domain bridge operators.

    Lookup is keyed by the ``(from_type, to_type)`` tuple. The
    Sprint-26 whitelist is enforced at register-time so a bug-fix to
    a domain catalog can't quietly add an unsanctioned bridge pair.
    """

    def __init__(self, whitelist: frozenset[tuple[str, str]] | None = None) -> None:
        self._operators: dict[tuple[str, str], BridgeOperator] = {}
        # Whitelist is resolved lazily via :meth:`_active_whitelist` so
        # this module stays importable before the whitelist module
        # has finished evaluating its top level (avoids a circular
        # import between registry.py and whitelist.py).
        self._explicit_whitelist = whitelist

    def _active_whitelist(self) -> frozenset[tuple[str, str]]:
        if self._explicit_whitelist is not None:
            return self._explicit_whitelist
        from cognithor.channels.program_synthesis.domains.bridges.whitelist import (
            SPRINT26_BRIDGE_WHITELIST,
        )

        return SPRINT26_BRIDGE_WHITELIST

    def register(self, operator: BridgeOperator) -> None:
        key = (operator.from_type, operator.to_type)
        if key not in self._active_whitelist():
            msg = (
                f"Bridge pair {key!r} is not in the Sprint-26 "
                f"whitelist (Owner-Decision D5). Add it to "
                f"SPRINT26_BRIDGE_WHITELIST first if intentional."
            )
            raise BridgeNotWhitelistedError(msg)
        if key in self._operators:
            msg = f"Bridge {key!r} is already registered"
            raise ValueError(msg)
        self._operators[key] = operator

    def get(self, from_type: str, to_type: str) -> BridgeOperator:
        key = (from_type, to_type)
        if key not in self._operators:
            available = sorted(self._operators)
            msg = f"Bridge {key!r} not registered. Available: {available}"
            raise KeyError(msg)
        return self._operators[key]

    def has(self, from_type: str, to_type: str) -> bool:
        return (from_type, to_type) in self._operators

    def names(self) -> list[tuple[str, str]]:
        return sorted(self._operators)

    def __len__(self) -> int:
        return len(self._operators)

    def __contains__(self, key: object) -> bool:
        return key in self._operators


# Canonical, process-local registry used by the synthesis pipeline.
BRIDGE_REGISTRY: BridgeRegistry = BridgeRegistry()
