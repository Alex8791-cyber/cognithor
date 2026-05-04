"""``PermissionScope`` foundation (TRUST-5, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: today's gatekeeper applies one global GREEN/
YELLOW/ORANGE/RED set per tool. Channels, users, and workflows have
no separate permission boundaries — all callers see the same tool
whitelist. A Telegram message and a Slack DM hit the same risk
classifier; an unattended cron run inherits the same approvals as an
interactive chat session.

This module ships the **foundation**: a frozen ``PermissionScope``
data model + an in-memory ``ScopeRegistry`` + a ``ScopeViolation``
exception. Wiring scopes into the gatekeeper, channel base class, and
DB persistence is **deliberately deferred** to a follow-up — a single
mergeable commit shouldn't touch ten files at once.

Contract:

* A scope is keyed by ``(axis, identity)`` — e.g.
  ``("channel", "telegram")`` or ``("user", "alex@cognithor.ai")``.
* Each scope declares ``tool_allowlist``, ``tool_denylist``, and a
  ``max_risk`` ceiling.
* ``ScopeRegistry.evaluate(scope_keys, tool_name, tool_risk)`` returns
  the most-restrictive verdict across the matching scopes — denylist
  beats allowlist beats max-risk.
* No DB, no IO. Persistence is the next PR's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from cognithor.models import RiskLevel


class ScopeAxis(StrEnum):
    """Axes a scope can be keyed on.

    Sprint-26-Lite picks four axes that map to existing concepts in
    the codebase. New axes can be added by extending this enum + the
    corresponding identity field on ``PermissionScope``.
    """

    CHANNEL = "channel"
    USER = "user"
    WORKFLOW = "workflow"
    PACK = "pack"


# Total ordering of risk levels — denser ints sort lower-to-higher
# severity. Used by ``max_risk`` ceiling check.
_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.GREEN: 0,
    RiskLevel.YELLOW: 1,
    RiskLevel.ORANGE: 2,
    RiskLevel.RED: 3,
}


def _risk_value(level: RiskLevel) -> int:
    return _RISK_ORDER.get(level, len(_RISK_ORDER))


class ScopeViolation(Exception):
    """Raised when a tool call violates a permission scope.

    The message names the (axis, identity, tool, reason) tuple so the
    audit log + Trace-UI can render a structured ``DecisionExplanation``
    (TRUST-2 hook) without re-parsing.
    """

    def __init__(
        self,
        *,
        axis: str,
        identity: str,
        tool: str,
        reason: str,
    ) -> None:
        super().__init__(f"Scope violation: {axis}={identity!r} blocks {tool!r} ({reason})")
        self.axis = axis
        self.identity = identity
        self.tool = tool
        self.reason = reason


@dataclass(frozen=True)
class PermissionScope:
    """A typed permission slice for one (axis, identity) pair.

    Frozen so the registry can hash it for de-duplication. The fields
    are intentionally minimal — extending the surface (rate-limits,
    cost ceilings, time-of-day windows) is the follow-up PR's job.

    Semantics:

    * ``tool_allowlist`` empty ⇒ all tools allowed (subject to
      denylist + max_risk). Non-empty ⇒ only listed tools allowed.
    * ``tool_denylist`` always wins over allowlist.
    * ``max_risk`` ceiling — any tool classified above this level is
      denied regardless of allowlist membership.
    """

    axis: ScopeAxis
    identity: str
    tool_allowlist: frozenset[str] = field(default_factory=frozenset)
    tool_denylist: frozenset[str] = field(default_factory=frozenset)
    max_risk: RiskLevel = RiskLevel.RED

    def __post_init__(self) -> None:
        if not self.identity:
            msg = "PermissionScope.identity must be a non-empty string"
            raise ValueError(msg)
        overlap = self.tool_allowlist & self.tool_denylist
        if overlap:
            msg = (
                f"Scope ({self.axis}, {self.identity!r}) has tools in both "
                f"allowlist and denylist: {sorted(overlap)}"
            )
            raise ValueError(msg)

    @property
    def key(self) -> tuple[str, str]:
        """Stable hash key for the registry."""
        return (self.axis.value, self.identity)


@dataclass(frozen=True)
class ScopeVerdict:
    """Outcome of a scope evaluation.

    Frozen so the audit log can embed it directly. ``allowed`` is the
    binary verdict; ``reasons`` lists every contributing rule
    (deterministic ordering for stable display in Trace-UI).
    """

    allowed: bool
    reasons: tuple[str, ...] = ()

    @property
    def denied(self) -> bool:
        return not self.allowed


class ScopeRegistry:
    """In-memory registry of permission scopes.

    Process-local on purpose — persistence (writing to a DB, reloading
    on boot) is the follow-up PR's job. Keeping this layer storage-free
    makes it cheap to test and lets callers pin a specific scope-set
    per request without polluting global state.
    """

    def __init__(self) -> None:
        self._scopes: dict[tuple[str, str], PermissionScope] = {}

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_data: object) -> ScopeRegistry:
        """Build a registry from a serialised scope list.

        ``config_data`` is the deserialised value of the
        ``security.scopes`` config key. Accepted shapes:

        * ``None`` / empty → empty registry (production fallback)
        * ``list[dict]`` — each dict has ``axis``, ``identity``,
          optional ``tool_allowlist``, ``tool_denylist``, ``max_risk``
        * ``dict`` containing a ``scopes`` key with the same list

        Malformed entries log a warning and are skipped — a single
        bad row in YAML must not take down the whole gateway boot.
        """
        from cognithor.utils.logging import get_logger

        logger = get_logger(__name__)

        if config_data is None:
            return cls()

        raw_list: object
        if isinstance(config_data, dict):
            raw_list = config_data.get("scopes", [])
        else:
            raw_list = config_data
        if not isinstance(raw_list, list):
            logger.warning(
                "scope_config_invalid_shape",
                got=type(raw_list).__name__,
            )
            return cls()

        registry = cls()
        for index, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                logger.warning(
                    "scope_config_skip_non_dict",
                    index=index,
                    type=type(raw).__name__,
                )
                continue
            try:
                scope = cls._scope_from_dict(raw)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "scope_config_skip_malformed",
                    index=index,
                    error=str(exc),
                )
                continue
            registry.register(scope)
        return registry

    @staticmethod
    def _scope_from_dict(raw: dict[str, object]) -> PermissionScope:
        from cognithor.models import RiskLevel as _RiskLevel

        axis_raw = raw.get("axis")
        if not isinstance(axis_raw, str):
            msg = "axis must be a string"
            raise ValueError(msg)
        axis = ScopeAxis(axis_raw)

        identity = raw.get("identity")
        if not isinstance(identity, str) or not identity:
            msg = "identity must be a non-empty string"
            raise ValueError(msg)

        allowlist_raw = raw.get("tool_allowlist", [])
        denylist_raw = raw.get("tool_denylist", [])
        if not isinstance(allowlist_raw, list) or not isinstance(denylist_raw, list):
            msg = "tool_allowlist / tool_denylist must be lists"
            raise ValueError(msg)

        max_risk_raw = raw.get("max_risk", "red")
        if not isinstance(max_risk_raw, str):
            msg = "max_risk must be a string"
            raise ValueError(msg)
        max_risk = _RiskLevel(max_risk_raw.lower())

        return PermissionScope(
            axis=axis,
            identity=identity,
            tool_allowlist=frozenset(str(t) for t in allowlist_raw),
            tool_denylist=frozenset(str(t) for t in denylist_raw),
            max_risk=max_risk,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, scope: PermissionScope) -> None:
        """Add or replace a scope. Existing key is overwritten silently
        — scopes are config-driven and re-config should not raise.
        """
        self._scopes[scope.key] = scope

    def remove(self, axis: ScopeAxis, identity: str) -> bool:
        """Remove a scope by key. Returns True if a scope was deleted."""
        key = (axis.value, identity)
        return self._scopes.pop(key, None) is not None

    def clear(self) -> None:
        self._scopes.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, axis: ScopeAxis, identity: str) -> PermissionScope | None:
        return self._scopes.get((axis.value, identity))

    def list_scopes(self) -> list[PermissionScope]:
        """Return all scopes sorted by (axis, identity) for stable display."""
        return [self._scopes[k] for k in sorted(self._scopes)]

    def __len__(self) -> int:
        return len(self._scopes)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scope_keys: list[tuple[ScopeAxis, str]],
        tool_name: str,
        tool_risk: RiskLevel,
    ) -> ScopeVerdict:
        """Return the most-restrictive verdict for the given tool call.

        ``scope_keys`` lists every (axis, identity) the caller falls
        under — usually ``[(CHANNEL, "telegram"), (USER, "alex"),
        (WORKFLOW, "morning_brief")]``. Missing scopes are skipped.

        Decision precedence (most-restrictive wins):

        1. Any matching scope's denylist contains ``tool_name`` ⇒ deny
        2. Any matching scope's max_risk lower than ``tool_risk`` ⇒ deny
        3. Any matching scope has a non-empty allowlist that does NOT
           contain ``tool_name`` ⇒ deny
        4. Otherwise ⇒ allow
        """
        reasons: list[str] = []
        for axis, identity in scope_keys:
            scope = self._scopes.get((axis.value, identity))
            if scope is None:
                continue

            if tool_name in scope.tool_denylist:
                return ScopeVerdict(
                    allowed=False,
                    reasons=(f"{axis.value}={identity!r}: tool {tool_name!r} in denylist",),
                )

            if _risk_value(tool_risk) > _risk_value(scope.max_risk):
                return ScopeVerdict(
                    allowed=False,
                    reasons=(
                        f"{axis.value}={identity!r}: tool risk "
                        f"{tool_risk.value} exceeds max_risk "
                        f"{scope.max_risk.value}",
                    ),
                )

            if scope.tool_allowlist and tool_name not in scope.tool_allowlist:
                return ScopeVerdict(
                    allowed=False,
                    reasons=(f"{axis.value}={identity!r}: tool {tool_name!r} not in allowlist",),
                )

            reasons.append(f"{axis.value}={identity!r}: ok")

        return ScopeVerdict(allowed=True, reasons=tuple(reasons))

    def assert_allowed(
        self,
        scope_keys: list[tuple[ScopeAxis, str]],
        tool_name: str,
        tool_risk: RiskLevel,
    ) -> None:
        """Raise :class:`ScopeViolation` when the call is denied.

        Convenience wrapper for the gatekeeper: the audit log already
        knows how to format a ``ScopeViolation``, so callers that just
        want a binary fail-fast can use this instead of branching on
        :meth:`evaluate`.
        """
        verdict = self.evaluate(scope_keys, tool_name, tool_risk)
        if verdict.allowed:
            return
        if not verdict.reasons:
            reason = "no matching scope verdict"
            axis = "?"
            identity = "?"
        else:
            head = verdict.reasons[0]
            # head shape: "<axis>=<repr(identity)>: <reason>"
            axis_part, _, rest = head.partition("=")
            ident_part, _, reason = rest.partition(": ")
            axis = axis_part
            identity = ident_part.strip("'\"")
        raise ScopeViolation(axis=axis, identity=identity, tool=tool_name, reason=reason)


# Process-local default registry. The gateway boot path can populate
# it from config / DB; tests construct fresh registries to keep state
# out of globals.
SCOPE_REGISTRY: ScopeRegistry = ScopeRegistry()


def _record_scope_registry_migration() -> None:
    """TRUST-10 self-audit: announce the scope-registry schema."""
    from contextlib import suppress

    from cognithor.security.migration_ledger import (
        MIGRATION_LEDGER,
        MigrationChainError,
        MigrationDomain,
        MigrationStatus,
        MigrationStep,
    )

    with suppress(MigrationChainError, ValueError):
        MIGRATION_LEDGER.record(
            MigrationStep(
                domain=MigrationDomain.SCOPE_REGISTRY,
                source_version="v0-no-registry",
                target_version="v1-axis-identity-registry",
                status=MigrationStatus.APPLIED,
                applied_by="system",
                item_count=-1,
                migration_id="scope_registry:v0-no-registry:v1-axis-identity-registry",
                notes=(
                    "TRUST-5 ScopeRegistry schema active "
                    "(ScopeAxis enum, allow/deny/max_risk evaluation)"
                ),
            )
        )


_record_scope_registry_migration()
