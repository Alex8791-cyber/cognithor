# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE audit trail — Hashline-style append-only entries (spec §13.2).

Every synthesis request lands as one entry::

    {
      "ts": "2026-04-30T01:13:42.123Z",
      "actor": "planner@cognithor",
      "capability": "pse:synthesize",
      "spec_hash": "sha256:abc...",
      "budget": {"max_depth": 4, "wall_clock_seconds": 30.0},
      "result_status": "success",
      "program_hash": "sha256:def...",
      "duration_ms": 4321,
      "candidates_explored": 8742
    }

The Hashline guarantee (manipulationssicher via SHA-256 chain) is the
job of the host Hashline subsystem; this module produces the entries
in the canonical shape and writes them to a sink (memory, file, or the
external Hashline writer).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _utc_now_iso() -> str:
    """Spec-canonical timestamp: UTC ISO with millisecond precision + ``Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{datetime.now(UTC).microsecond // 1000:03d}Z"
    )


@dataclass(frozen=True)
class AuditEntry:
    """One audit record. Frozen so a writer can't mutate after emit."""

    ts: str
    actor: str
    capability: str
    spec_hash: str
    budget: dict[str, Any]
    result_status: str
    program_hash: str | None
    duration_ms: int
    candidates_explored: int
    extra: tuple[tuple[str, Any], ...] = ()

    def to_json(self) -> str:
        """Canonical JSON serialization for the Hashline body.

        Keys sorted, separators tight, NaN/Inf rejected — gives a
        deterministic byte-string the chain hash can be computed over.
        """
        payload = {
            "ts": self.ts,
            "actor": self.actor,
            "capability": self.capability,
            "spec_hash": self.spec_hash,
            "budget": self.budget,
            "result_status": self.result_status,
            "program_hash": self.program_hash,
            "duration_ms": self.duration_ms,
            "candidates_explored": self.candidates_explored,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class AuditTrail:
    """Append-only sink for :class:`AuditEntry`.

    Phase-1 implementation keeps entries in memory (tests + dev mode);
    production wires a callback to the Hashline writer via
    ``on_emit``. Each call hashes the JSON body and chains it with
    the previous entry's hash, mimicking the Hashline-Guard invariant
    (spec §13.2 calls out the manipulationssichere Hashkette).
    """

    GENESIS_HASH = "sha256:" + ("0" * 64)

    def __init__(self, on_emit: Callable[[AuditEntry, str], None] | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._chain_hash: str = self.GENESIS_HASH
        self._on_emit = on_emit

    def emit(self, entry: AuditEntry) -> str:
        """Record *entry*; return the new chain-hash after this entry."""
        body = entry.to_json()
        chained = (self._chain_hash + body).encode("utf-8")
        new_hash = "sha256:" + hashlib.sha256(chained).hexdigest()
        self._chain_hash = new_hash
        self._entries.append(entry)
        if self._on_emit is not None:
            self._on_emit(entry, new_hash)
        return new_hash

    def latest_hash(self) -> str:
        return self._chain_hash

    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        """Clear entries + reset chain to genesis. Tests only."""
        self._entries.clear()
        self._chain_hash = self.GENESIS_HASH

    def verify(self) -> bool:
        """Re-walk the chain; True iff every entry's hash matches.

        Catches tampering: if an entry's body or order were modified
        between emit and verify, the chain hashes diverge.
        """
        running = self.GENESIS_HASH
        for entry in self._entries:
            chained = (running + entry.to_json()).encode("utf-8")
            running = "sha256:" + hashlib.sha256(chained).hexdigest()
        return running == self._chain_hash


def audit_entry_for(
    *,
    actor: str,
    capability: str,
    spec_hash: str,
    budget: dict[str, Any],
    result_status: str,
    program_hash: str | None,
    duration_ms: int,
    candidates_explored: int,
    extra: dict[str, Any] | None = None,
) -> AuditEntry:
    """Convenience constructor that fills the timestamp + sorted-extras."""
    return AuditEntry(
        ts=_utc_now_iso(),
        actor=actor,
        capability=capability,
        spec_hash=spec_hash,
        budget=dict(budget),
        result_status=result_status,
        program_hash=program_hash,
        duration_ms=duration_ms,
        candidates_explored=candidates_explored,
        extra=tuple(sorted(extra.items())) if extra else (),
    )


__all__ = [
    "AuditEntry",
    "AuditTrail",
    "audit_entry_for",
]


_ = field  # keep dataclass.field import valid for future fields
