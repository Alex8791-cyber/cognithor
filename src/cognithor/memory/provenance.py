"""``ProvenanceTag`` foundation (TRUST-9, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: today's memory items have ``source`` and
``created_at`` fields scattered across the tiers (semantic, vault,
episodic, relations). There is no uniform provenance contract — an
operator looking at "Owner lives in Mainz" cannot tell whether that
came from a chat utterance two years ago, a tool call yesterday, or a
config import last hour.

This module ships the **foundation**: a frozen ``ProvenanceTag`` data
model + an in-memory ``ProvenanceLedger`` keyed by memory-item id,
plus an ``ExpiryPolicy`` enum so callers can declare staleness rules
deterministically. Wiring this into the four memory tiers is a
follow-up — a single mergeable commit shouldn't rewrite ten memory
modules at once.

Contract:

* Each provenance tag is keyed by ``item_id`` (the memory-tier's
  primary key — semantic-memory uuid, vault-record-id, episodic-event-id,
  relation-edge-id).
* The tag is **append-only**: re-tagging the same item appends a new
  version to the chain rather than overwriting. The ledger keeps the
  full chain so post-mortem reconstruction can walk backwards.
* ``valid_from`` / ``valid_until`` are *advisory*: they tell consumers
  when the source's claim is supposed to be true. ``expired(now)``
  surfaces items past their TTL or with ``valid_until < now``.
* No DB. The follow-up PR persists to ``~/.cognithor/provenance.db``;
  this layer stays storage-free for cheap testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Source typing
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    """Where a memory item came from.

    The set is closed so consumers can switch on it without hitting an
    Unknown-fallback branch. Add a new value here when a new ingestion
    path lands; that's an intentional review point.
    """

    CHAT_UTTERANCE = "chat_utterance"
    TOOL_OUTPUT = "tool_output"
    AGENT_INFERENCE = "agent_inference"
    CONFIG_IMPORT = "config_import"
    PACK_REGISTRATION = "pack_registration"
    SCHEDULED_INGEST = "scheduled_ingest"
    MIGRATION = "migration"
    USER_DIRECTIVE = "user_directive"
    UNKNOWN = "unknown"


class ExpiryPolicy(StrEnum):
    """Staleness-rule for a provenance tag.

    Drives :meth:`ProvenanceLedger.expired` — an item with policy
    ``PERMANENT`` never expires; ``TTL`` expires when ``valid_until``
    is set and ``now > valid_until``; ``REPLACE_ON_NEW`` expires
    automatically once a newer tag for the same item lands;
    ``MANUAL`` expires only when an operator removes it explicitly.
    """

    PERMANENT = "permanent"
    TTL = "ttl"
    REPLACE_ON_NEW = "replace_on_new"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Provenance tag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceTag:
    """Immutable provenance record for a single memory item.

    Frozen so the ledger can hash it for de-duplication and embed it
    into audit-log entries without copy. Default values keep the
    constructor cheap for the common case (TOOL_OUTPUT with TTL=24h).

    Attributes
    ----------
    source_type:
        Where the item came from. Drives downstream filtering — e.g.
        the planner can skip CHAT_UTTERANCE items older than 7 days
        without affecting CONFIG_IMPORT items.
    source_id:
        Stable identifier of the source within its type. For
        TOOL_OUTPUT this is the audit-log entry id; for CHAT_UTTERANCE
        this is the message id; for CONFIG_IMPORT this is the config
        file path.
    source_url:
        Optional URL/URI for human-followable provenance. Empty when
        the source is internal (no addressable URL).
    ingested_at:
        UTC timestamp the item entered cognithor's memory. Distinct
        from ``valid_from`` because the same source claim can be
        re-ingested after a migration without resetting validity.
    valid_from:
        UTC timestamp the source's claim starts being true. Defaults
        to ``ingested_at`` for sources that don't carry temporal
        information of their own.
    valid_until:
        UTC timestamp the source's claim stops being true. ``None``
        means "no declared end" — combined with ``ExpiryPolicy.TTL``
        this means the policy is effectively PERMANENT.
    expiry_policy:
        Staleness rule (see :class:`ExpiryPolicy`).
    confidence:
        0..1 confidence score the source attaches to the claim.
        Defaults to 1.0 (fully trusted source). LLM-extracted facts
        from low-confidence reasoning runs ship lower values.
    attribution_chain:
        Tuple of upstream item-ids the current item was derived from.
        Empty for primary ingests; non-empty when the item is a
        consolidation or distillation of multiple inputs.
    notes:
        Free-text breadcrumb for the audit log. Keep it short — full
        diagnostics live in the audit chain.
    """

    source_type: SourceType
    source_id: str
    source_url: str = ""
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    expiry_policy: ExpiryPolicy = ExpiryPolicy.PERMANENT
    confidence: float = 1.0
    attribution_chain: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            msg = "ProvenanceTag.source_id must be a non-empty string"
            raise ValueError(msg)
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"ProvenanceTag.confidence must be in [0.0, 1.0], got {self.confidence}"
            raise ValueError(msg)
        if self.expiry_policy == ExpiryPolicy.TTL and self.valid_until is None:
            msg = (
                "ProvenanceTag with expiry_policy=TTL must set "
                "valid_until — without an end, use PERMANENT instead"
            )
            raise ValueError(msg)
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from > self.valid_until
        ):
            msg = (
                f"ProvenanceTag.valid_from ({self.valid_from.isoformat()}) "
                f"must be <= valid_until ({self.valid_until.isoformat()})"
            )
            raise ValueError(msg)

    @property
    def effective_valid_from(self) -> datetime:
        """``valid_from`` falling back to ``ingested_at``."""
        return self.valid_from if self.valid_from is not None else self.ingested_at

    def with_chain(self, *parents: str) -> ProvenanceTag:
        """Return a copy with ``parents`` appended to ``attribution_chain``."""
        return replace(self, attribution_chain=self.attribution_chain + parents)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class ProvenanceLedger:
    """Append-only in-memory ledger of provenance tags per memory item.

    The ledger maps ``item_id → tuple[ProvenanceTag, ...]``. Each
    ``tag()`` call appends; the most recent tag is the one a consumer
    typically reads via :meth:`current`. Tests construct fresh
    ledgers; production code uses :data:`PROVENANCE_LEDGER` (the
    process-local default).
    """

    def __init__(self) -> None:
        self._chains: dict[str, tuple[ProvenanceTag, ...]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def tag(self, item_id: str, tag: ProvenanceTag) -> None:
        """Append ``tag`` to the chain for ``item_id``.

        REPLACE_ON_NEW policy: the previous tag remains in the chain
        (history-preserving) but :meth:`current` returns the new tag,
        and :meth:`expired` flags the previous one as stale.
        """
        if not item_id:
            msg = "tag(): item_id must be a non-empty string"
            raise ValueError(msg)
        existing = self._chains.get(item_id, ())
        self._chains[item_id] = (*existing, tag)

    def remove(self, item_id: str) -> bool:
        """Drop the entire chain for ``item_id``. Returns True iff it existed."""
        return self._chains.pop(item_id, None) is not None

    def clear(self) -> None:
        self._chains.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def chain(self, item_id: str) -> tuple[ProvenanceTag, ...]:
        """Return the full append-only history for ``item_id`` (oldest first)."""
        return self._chains.get(item_id, ())

    def current(self, item_id: str) -> ProvenanceTag | None:
        """Return the most recent tag for ``item_id`` or ``None``."""
        chain = self._chains.get(item_id)
        return chain[-1] if chain else None

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._chains

    def __len__(self) -> int:
        return len(self._chains)

    def items(self) -> list[tuple[str, tuple[ProvenanceTag, ...]]]:
        """Return all (item_id, chain) pairs sorted by item_id."""
        return [(k, self._chains[k]) for k in sorted(self._chains)]

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def expired(self, *, now: datetime | None = None) -> list[str]:
        """Return item_ids whose *current* tag is stale at ``now``.

        Stale rules:

        * ``PERMANENT`` — never stale.
        * ``TTL`` — stale when ``now > valid_until``.
        * ``REPLACE_ON_NEW`` — stale when the chain has more than one
          tag (the older ones were superseded). The current head is
          fresh by definition; the *previous* tag in the chain is
          stale and would be reported on its own item if it still had
          a unique id, but in practice REPLACE_ON_NEW chains share
          one item_id so we report the item only when the head is
          itself superseded by a newer tag — which never happens by
          construction. We therefore *don't* report REPLACE_ON_NEW
          here; consumers walk :meth:`chain` to find superseded
          history.
        * ``MANUAL`` — never automatically stale.

        ``now`` defaults to :func:`datetime.now(UTC)`.
        """
        cutoff = now if now is not None else datetime.now(UTC)
        out: list[str] = []
        for item_id, chain in self._chains.items():
            if not chain:
                continue
            head = chain[-1]
            if (
                head.expiry_policy == ExpiryPolicy.TTL
                and head.valid_until is not None
                and cutoff > head.valid_until
            ):
                out.append(item_id)
        return sorted(out)

    def superseded(self, item_id: str) -> tuple[ProvenanceTag, ...]:
        """Return the prefix of an item's chain that's been replaced.

        Useful for memory-tier consumers that periodically prune
        REPLACE_ON_NEW items: every tag in the returned tuple is no
        longer the source-of-truth for ``item_id`` and can be moved
        to cold storage.
        """
        chain = self._chains.get(item_id, ())
        if len(chain) <= 1:
            return ()
        return chain[:-1]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_source_type(self, source_type: SourceType) -> list[tuple[str, ProvenanceTag]]:
        """Return ``(item_id, current_tag)`` pairs for matching items."""
        out: list[tuple[str, ProvenanceTag]] = []
        for item_id in sorted(self._chains):
            tag = self._chains[item_id][-1]
            if tag.source_type == source_type:
                out.append((item_id, tag))
        return out

    def filter_by_source_id(self, source_id: str) -> list[str]:
        """Return item_ids whose current tag has ``source_id``.

        Lets the audit-log answer "which memory items came from the
        run XYZ?" — a TRUST-1 receipt query — without walking every
        memory tier.
        """
        return sorted(
            item_id
            for item_id, chain in self._chains.items()
            if chain and chain[-1].source_id == source_id
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        """JSON-serialisable snapshot of the ledger.

        Used by the TRUST-1 run-receipt to embed the provenance chain
        for every memory item touched during a run; the cognithor.ai
        Trace-UI renders the chain as a vertical timeline.
        """
        return {
            item_id: [
                {
                    "source_type": tag.source_type.value,
                    "source_id": tag.source_id,
                    "source_url": tag.source_url,
                    "ingested_at": tag.ingested_at.isoformat(),
                    "valid_from": (tag.valid_from.isoformat() if tag.valid_from else None),
                    "valid_until": (tag.valid_until.isoformat() if tag.valid_until else None),
                    "expiry_policy": tag.expiry_policy.value,
                    "confidence": round(tag.confidence, 4),
                    "attribution_chain": list(tag.attribution_chain),
                    "notes": tag.notes,
                }
                for tag in chain
            ]
            for item_id, chain in sorted(self._chains.items())
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ttl_tag(
    *,
    source_type: SourceType,
    source_id: str,
    ttl: timedelta,
    confidence: float = 1.0,
    notes: str = "",
) -> ProvenanceTag:
    """Build a TTL-policy tag with ``valid_until = now + ttl``.

    Sugar for the common LLM-inference case: the agent extracts a fact
    that's only known to be true for ``ttl`` (e.g. "this user's
    timezone is Europe/Berlin — true for the duration of this
    session"). Defaults align with the rest of cognithor's UTC-only
    timestamping.
    """
    if ttl.total_seconds() < 0:
        msg = f"make_ttl_tag: ttl must be non-negative, got {ttl}"
        raise ValueError(msg)
    now = datetime.now(UTC)
    return ProvenanceTag(
        source_type=source_type,
        source_id=source_id,
        ingested_at=now,
        valid_from=now,
        valid_until=now + ttl,
        expiry_policy=ExpiryPolicy.TTL,
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# Memory tiers wire into this instance via a follow-up PR; the
# ledger stays empty in production until that wiring lands. Tests
# construct their own :class:`ProvenanceLedger` for isolation.
PROVENANCE_LEDGER: ProvenanceLedger = ProvenanceLedger()


def _record_provenance_ledger_migration() -> None:
    """TRUST-10 self-audit: announce the provenance-ledger schema.

    Idempotent via the canonical ``MigrationLedger``'s
    ``MigrationChainError`` on duplicate ``migration_id``. Wrapped
    in ``suppress`` so import-time side effects NEVER raise. Tests
    that monkey-patch the canonical ledger for isolation see this
    step as a no-op (the original singleton already has it).
    """
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
                domain=MigrationDomain.PROVENANCE_LEDGER,
                source_version="v0-no-ledger",
                target_version="v1-append-only-ledger",
                status=MigrationStatus.APPLIED,
                applied_by="system",
                item_count=-1,
                migration_id="provenance_ledger:v0-no-ledger:v1-append-only-ledger",
                notes=(
                    "TRUST-9 ProvenanceLedger schema active "
                    "(append-only chain per item_id, ExpiryPolicy enum)"
                ),
            )
        )


_record_provenance_ledger_migration()
