"""``MigrationStep`` foundation (TRUST-10, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: cognithor's persistence layers evolve. Memory
schemas grow new columns, the audit-log JSONL adds a hash-chain
field, pack manifests bump their version. Today those migrations
happen ad hoc — a one-shot script, a release-note line, a code path
that quietly mutates rows on first read. An operator who needs to
explain "why is this row's shape different from what the schema says?"
has no answer: the migration trail is implicit.

This module ships the **foundation**: a frozen ``MigrationStep``
dataclass + an append-only, per-domain ``MigrationLedger`` that
enforces chain integrity (each step's ``source_version`` must equal
the previous applied step's ``target_version`` for the same domain).
Wiring this into the persistence-bootstrap path (one ``record()``
call per actually-applied migration) is a deliberate follow-up; this
layer stays storage-free for cheap testing.

Contract:

* Migrations are scoped to a :class:`MigrationDomain` (memory tier,
  audit log, pack-manifest, config-schema). Cross-domain migrations
  decompose into multiple steps under the relevant domains.
* Each step carries a SHA-256 ``checksum_before`` / ``checksum_after``
  so an operator can verify the migration touched what it claimed
  to touch.
* The ledger refuses to record a step whose ``source_version`` does
  not match the current head version of that domain — the chain is
  the integrity contract.
* Status transitions are explicit: ``PENDING`` → ``APPLIED`` is the
  happy path; ``APPLIED`` → ``ROLLED_BACK`` records a reversal as a
  fresh step (history is append-only, never mutated). ``FAILED`` is
  terminal — a failed step does not advance the head version, so
  the next migration can retry from the same starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain + status
# ---------------------------------------------------------------------------


class MigrationDomain(StrEnum):
    """Persistence layer being migrated.

    Closed set so consumers can switch on it without an Unknown
    fallback. Each domain has an independent version chain.
    """

    MEMORY_SEMANTIC = "memory_semantic"
    MEMORY_VAULT = "memory_vault"
    MEMORY_EPISODIC = "memory_episodic"
    MEMORY_RELATIONS = "memory_relations"
    AUDIT_LOG = "audit_log"
    PACK_MANIFEST = "pack_manifest"
    CONFIG_SCHEMA = "config_schema"
    PROVENANCE_LEDGER = "provenance_ledger"
    FINGERPRINT_LEDGER = "fingerprint_ledger"
    COST_LEDGER = "cost_ledger"
    ESCALATION_LEDGER = "escalation_ledger"
    SCOPE_REGISTRY = "scope_registry"


class MigrationStatus(StrEnum):
    """Outcome of a recorded migration step.

    ``PENDING`` — declared but not yet applied (planning).
    ``APPLIED`` — successfully advanced the chain. Only APPLIED
    steps move the head version.
    ``FAILED`` — attempt rolled back at the persistence layer; the
    head version stays at ``source_version``. Recorded so an
    operator can see "we tried, it didn't take".
    ``ROLLED_BACK`` — a previously APPLIED step was explicitly
    reversed. The reversal is a separate step recorded after the
    fact, so the chain remains append-only.
    """

    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationStep:
    """Single migration record.

    Frozen so the audit log can hash + embed it without copy.

    Attributes
    ----------
    domain:
        Which persistence layer this migration touches.
    source_version:
        The head version expected to exist *before* the migration
        ran. The ledger refuses to record an APPLIED step if the
        domain's current head doesn't match. Free-form string —
        ``"v3"``, ``"2026.04.16"``, ``"sha:abc123"``.
    target_version:
        The version the domain reaches if the migration applies
        cleanly. Same shape as ``source_version``.
    status:
        :class:`MigrationStatus`.
    applied_at:
        UTC timestamp the migration ran (or was declared).
    applied_by:
        Free-form principal — ``"system"`` for automatic on-boot
        migrations, ``"alex@cognithor.ai"`` for owner-initiated
        migrations. Empty when not known.
    item_count:
        Number of rows / records touched by the migration. ``-1``
        for non-data-bearing migrations (schema-only). Validated
        non-negative or exactly -1.
    checksum_before:
        Lowercase-hex SHA-256 of the canonical pre-migration state.
        Empty when the migration is not data-bearing or the layer
        does not provide a digest.
    checksum_after:
        Lowercase-hex SHA-256 of the canonical post-migration
        state. Same emptiness rules as ``checksum_before``.
    rollback_of:
        The migration_id of the step this one rolls back. Empty
        for forward migrations. The ledger validates that the
        referenced step exists and is APPLIED in the same domain.
    migration_id:
        Stable identifier — typically ``"<domain>:<source>:<target>"``.
        Free-form; the ledger uses it for ``rollback_of`` lookups.
    notes:
        Short free-text breadcrumb.
    """

    domain: MigrationDomain
    source_version: str
    target_version: str
    status: MigrationStatus
    applied_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    applied_by: str = ""
    item_count: int = -1
    checksum_before: str = ""
    checksum_after: str = ""
    rollback_of: str = ""
    migration_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_version:
            msg = "MigrationStep.source_version must be a non-empty string"
            raise ValueError(msg)
        if not self.target_version:
            msg = "MigrationStep.target_version must be a non-empty string"
            raise ValueError(msg)
        if self.source_version == self.target_version and self.status != MigrationStatus.FAILED:
            msg = (
                "MigrationStep.source_version must differ from target_version "
                "(unless status=FAILED for an attempted no-op)"
            )
            raise ValueError(msg)
        if self.item_count < -1:
            msg = (
                "MigrationStep.item_count must be -1 (unknown / schema-only) "
                f"or a non-negative count, got {self.item_count}"
            )
            raise ValueError(msg)
        for label, value in (
            ("checksum_before", self.checksum_before),
            ("checksum_after", self.checksum_after),
        ):
            if value and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                msg = (
                    f"MigrationStep.{label} must be empty or 64 lowercase-hex "
                    f"chars (SHA-256), got {value!r}"
                )
                raise ValueError(msg)
        if self.rollback_of and self.status != MigrationStatus.ROLLED_BACK:
            msg = (
                "MigrationStep.rollback_of can only be set on a step with "
                f"status=ROLLED_BACK; got status={self.status}"
            )
            raise ValueError(msg)

    @property
    def is_data_bearing(self) -> bool:
        return self.item_count >= 0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class MigrationChainError(RuntimeError):
    """Raised when a record() call would violate the per-domain chain."""


class MigrationLedger:
    """Append-only ledger of migration steps with per-domain chain integrity.

    The ledger maintains:

    * ``_steps`` — insertion-ordered list of every step (incl.
      PENDING, FAILED, ROLLED_BACK).
    * ``_head_version`` — the current advertised version per domain
      (only ``APPLIED`` and ``ROLLED_BACK`` steps move this).
    * ``_by_id`` — index for ``rollback_of`` lookups.
    """

    def __init__(self) -> None:
        self._steps: list[MigrationStep] = []
        self._head_version: dict[MigrationDomain, str] = {}
        self._by_id: dict[str, MigrationStep] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, step: MigrationStep) -> None:
        """Append ``step``, validating the per-domain chain.

        Validation rules:

        * If ``step.status`` is ``APPLIED`` or ``ROLLED_BACK`` and
          the domain has a head version, ``step.source_version`` must
          equal that head. ``PENDING`` and ``FAILED`` steps are
          recorded without head-check (they don't advance the chain).
        * If ``step.rollback_of`` is set, the referenced step must
          exist, be in the same domain, and have ``status=APPLIED``.
        * If ``step.migration_id`` is set, it must be unique
          across the ledger.
        """
        head = self._head_version.get(step.domain)
        head_moving = {MigrationStatus.APPLIED, MigrationStatus.ROLLED_BACK}
        if step.status in head_moving and head is not None and step.source_version != head:
            msg = (
                f"chain mismatch on {step.domain.value}: head is "
                f"{head!r}, step.source_version is {step.source_version!r}"
            )
            raise MigrationChainError(msg)
        if step.rollback_of:
            target = self._by_id.get(step.rollback_of)
            if target is None:
                msg = f"rollback_of refers to unknown migration_id {step.rollback_of!r}"
                raise MigrationChainError(msg)
            if target.domain != step.domain:
                msg = (
                    f"rollback_of refers to a step in domain "
                    f"{target.domain.value}, but this step is in "
                    f"{step.domain.value}"
                )
                raise MigrationChainError(msg)
            if target.status != MigrationStatus.APPLIED:
                msg = (
                    f"rollback_of {step.rollback_of!r} has status "
                    f"{target.status.value}, only APPLIED steps can be "
                    "rolled back"
                )
                raise MigrationChainError(msg)
        if step.migration_id and step.migration_id in self._by_id:
            msg = f"duplicate migration_id {step.migration_id!r}"
            raise MigrationChainError(msg)

        self._steps.append(step)
        if step.migration_id:
            self._by_id[step.migration_id] = step
        if step.status in {MigrationStatus.APPLIED, MigrationStatus.ROLLED_BACK}:
            self._head_version[step.domain] = step.target_version

    def clear(self) -> None:
        self._steps.clear()
        self._head_version.clear()
        self._by_id.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._steps)

    def steps(self) -> tuple[MigrationStep, ...]:
        """All steps in insertion order."""
        return tuple(self._steps)

    def for_domain(self, domain: MigrationDomain) -> tuple[MigrationStep, ...]:
        """Steps for a single domain in insertion order."""
        return tuple(s for s in self._steps if s.domain == domain)

    def head_version(self, domain: MigrationDomain) -> str | None:
        """Current head version for ``domain`` (``None`` if no APPLIED step)."""
        return self._head_version.get(domain)

    def get(self, migration_id: str) -> MigrationStep | None:
        return self._by_id.get(migration_id)

    def applied_only(self, domain: MigrationDomain) -> tuple[MigrationStep, ...]:
        """APPLIED + ROLLED_BACK steps for ``domain`` (the head-moving subset)."""
        moving = {MigrationStatus.APPLIED, MigrationStatus.ROLLED_BACK}
        return tuple(s for s in self._steps if s.domain == domain and s.status in moving)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """JSON-serialisable representation.

        Shape::

            {
              "head_version": {"<domain>": "<version>", ...},
              "steps": [<step-dict>, ...]
            }

        Embedded in TRUST-1 run receipts on first-run-after-migration
        so the operator can confirm which schema the run executed
        against.
        """
        return {
            "head_version": {
                domain.value: version
                for domain, version in sorted(
                    self._head_version.items(), key=lambda kv: kv[0].value
                )
            },
            "steps": [
                {
                    "domain": s.domain.value,
                    "source_version": s.source_version,
                    "target_version": s.target_version,
                    "status": s.status.value,
                    "applied_at": s.applied_at.isoformat(),
                    "applied_by": s.applied_by,
                    "item_count": s.item_count,
                    "checksum_before": s.checksum_before,
                    "checksum_after": s.checksum_after,
                    "rollback_of": s.rollback_of,
                    "migration_id": s.migration_id,
                    "notes": s.notes,
                }
                for s in self._steps
            ],
        }


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# Persistence-bootstrap wires into this instance via a follow-up PR.
MIGRATION_LEDGER: MigrationLedger = MigrationLedger()
