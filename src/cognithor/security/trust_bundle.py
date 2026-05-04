"""TRUST-1 receipt bundle composer (operational-trust audit, 2026-05-04).

The six TRUST-5..10 foundations each ship their own in-memory ledger:

* :data:`cognithor.security.permission_scope.SCOPE_REGISTRY` (TRUST-5)
* :data:`cognithor.security.cost_ledger.COST_LEDGER` (TRUST-6)
* :data:`cognithor.security.fingerprint.FINGERPRINT_LEDGER` (TRUST-7)
* :data:`cognithor.security.cloud_escalation.ESCALATION_LEDGER` (TRUST-8)
* :data:`cognithor.memory.provenance.PROVENANCE_LEDGER` (TRUST-9)
* :data:`cognithor.security.migration_ledger.MIGRATION_LEDGER` (TRUST-10)

The TRUST-1 :meth:`~cognithor.audit.AuditLogger.run_receipt` API stops
at audit-entry aggregation. To answer the reviewer-feedback question
**"can an operator reconstruct exactly what the agent knew, what it
decided, which tool it called, why it was allowed, what changed, and
how to roll it back?"** we need to fold the six ledger snapshots into
the same receipt.

This module is the glue. ``build_trust_bundle(run_id, *, ledgers=None)``
returns a single JSON-serialisable dict that the receipt API can
embed under a top-level ``"trust"`` key. Filterable views per-run for
the ledgers that carry a ``run_id`` axis (cost + escalation); full
snapshots for the rest, since they describe *the world the agent
ran in* (fingerprints, provenance, migrations) or are session-scoped
already (permission scopes).

The bundle is *additive* on TRUST-1 — existing consumers see no
change to ``run_receipt()`` until they opt in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognithor.memory.provenance import PROVENANCE_LEDGER
from cognithor.security.cloud_escalation import ESCALATION_LEDGER
from cognithor.security.cost_ledger import COST_LEDGER
from cognithor.security.fingerprint import FINGERPRINT_LEDGER
from cognithor.security.migration_ledger import MIGRATION_LEDGER
from cognithor.security.permission_scope import SCOPE_REGISTRY

if TYPE_CHECKING:
    from cognithor.memory.provenance import ProvenanceLedger
    from cognithor.security.cloud_escalation import EscalationLedger
    from cognithor.security.cost_ledger import CostLedger
    from cognithor.security.fingerprint import FingerprintLedger
    from cognithor.security.migration_ledger import MigrationLedger
    from cognithor.security.permission_scope import ScopeRegistry


# Schema version of the trust-bundle. Bump on breaking changes
# (renamed keys, removed sections, new required fields). Additive
# additions don't bump.
TRUST_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrustLedgers:
    """Override-bag so callers can inject test ledgers.

    Production code passes ``None`` to :func:`build_trust_bundle` and
    gets the canonical process-local instances. Tests construct fresh
    ledgers and pass them via :func:`build_trust_bundle(...,
    ledgers=TrustLedgers(...))` to keep state isolated.
    """

    permission_scope: ScopeRegistry = SCOPE_REGISTRY
    cost: CostLedger = COST_LEDGER
    fingerprint: FingerprintLedger = FINGERPRINT_LEDGER
    escalation: EscalationLedger = ESCALATION_LEDGER
    provenance: ProvenanceLedger = PROVENANCE_LEDGER
    migration: MigrationLedger = MIGRATION_LEDGER


def _scope_snapshot(registry: ScopeRegistry) -> list[dict[str, object]]:
    """Build a JSON-serialisable scope snapshot.

    ``ScopeRegistry`` does not expose a built-in snapshot method —
    this helper formats the active scopes into the same shape as the
    other foundations (sorted by axis + identity, sets coerced to
    sorted lists).
    """
    out: list[dict[str, object]] = []
    for scope in registry.list_scopes():
        out.append(
            {
                "axis": scope.axis.value,
                "identity": scope.identity,
                "tool_allowlist": sorted(scope.tool_allowlist),
                "tool_denylist": sorted(scope.tool_denylist),
                "max_risk": scope.max_risk.value,
            }
        )
    return out


def build_trust_bundle(
    run_id: str,
    *,
    ledgers: TrustLedgers | None = None,
) -> dict[str, object]:
    """Compose the cross-ledger trust bundle for ``run_id``.

    Parameters
    ----------
    run_id:
        TRUST-1 run / session id. Used to filter the ledgers that
        carry a ``run_id`` axis (:class:`CostLedger`,
        :class:`EscalationLedger`). When empty, those sections come
        out empty too — they would otherwise dump every cost ever,
        which the receipt doesn't want.
    ledgers:
        Inject test instances. ``None`` ⇒ canonical process-local
        ledgers.

    Returns
    -------
    Dict shape::

        {
          "schema_version": 1,
          "run_id": "<run_id>",
          "permission_scopes": [{...}, ...],
          "cost": {
            "summary": {
              "entry_count": int,
              "total_cost_usd_micro": int,
              "total_cost_usd": float,
              "by_kind": {<kind>: micro_usd, ...},
              "by_tool": {...},
              "by_backend": {...},
            },
            "entries": [{...}, ...]
          },
          "fingerprints": {
            "all": [{...}, ...],
            "divergent_names": [str, ...]
          },
          "escalations": {
            "summary": {...},
            "entries": [{...}, ...]
          },
          "provenance": {<item_id>: [{...}, ...]},
          "migrations": {"head_version": {...}, "steps": [{...}, ...]}
        }
    """
    actual_ledgers = ledgers if ledgers is not None else TrustLedgers()

    # --- Permission scopes (full snapshot — registry is process-scoped).
    scopes = _scope_snapshot(actual_ledgers.permission_scope)

    # --- Cost (run-scoped where possible).
    cost_entries = actual_ledgers.cost.by_run(run_id) if run_id else ()
    cost_summary = actual_ledgers.cost.summarise(cost_entries)
    cost_block = {
        "summary": {
            "entry_count": cost_summary.entry_count,
            "total_cost_usd_micro": cost_summary.total_cost_usd_micro,
            "total_cost_usd": round(cost_summary.total_cost_usd, 6),
            "by_kind": {kind.value: amount for kind, amount in cost_summary.by_kind.items()},
            "by_tool": dict(cost_summary.by_tool),
            "by_backend": dict(cost_summary.by_backend),
        },
        "entries": (
            [entry for entry in actual_ledgers.cost.snapshot() if entry.get("run_id") == run_id]
            if run_id
            else []
        ),
    }

    # --- Fingerprints (full snapshot — describes the boot-time world).
    fingerprints_block = {
        "all": actual_ledgers.fingerprint.snapshot(),
        "divergent_names": actual_ledgers.fingerprint.divergent_names(),
    }

    # --- Cloud-escalations (run-scoped).
    escalation_events = actual_ledgers.escalation.by_run(run_id) if run_id else ()
    escalation_summary = actual_ledgers.escalation.summarise(escalation_events)
    escalation_block = {
        "summary": {
            "event_count": escalation_summary.event_count,
            "total_prompt_tokens": escalation_summary.total_prompt_tokens,
            "total_response_tokens": escalation_summary.total_response_tokens,
            "total_tokens": escalation_summary.total_tokens,
            "total_cost_usd_micro": escalation_summary.total_cost_usd_micro,
            "total_cost_usd": round(escalation_summary.total_cost_usd, 6),
            "by_reason": {
                reason.value: count for reason, count in escalation_summary.by_reason.items()
            },
            "by_destination": dict(escalation_summary.by_destination),
        },
        "entries": (
            [
                entry
                for entry in actual_ledgers.escalation.snapshot()
                if entry.get("run_id") == run_id
            ]
            if run_id
            else []
        ),
    }

    # --- Provenance (full snapshot — memory items don't carry run_id).
    provenance_block = actual_ledgers.provenance.snapshot()

    # --- Migrations (full snapshot — describes the schema chain).
    migrations_block = actual_ledgers.migration.snapshot()

    return {
        "schema_version": TRUST_BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "permission_scopes": scopes,
        "cost": cost_block,
        "fingerprints": fingerprints_block,
        "escalations": escalation_block,
        "provenance": provenance_block,
        "migrations": migrations_block,
    }


__all__ = [
    "TRUST_BUNDLE_SCHEMA_VERSION",
    "TrustLedgers",
    "build_trust_bundle",
]
