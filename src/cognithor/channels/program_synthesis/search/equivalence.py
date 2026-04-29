# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Observational-equivalence pruner (spec §9).

Two programs are *observationally equivalent* with respect to a TaskSpec
if they produce identical outputs on every demo input. The enumerator
keeps only one representative per equivalence class — the cheapest by
DSL-cost — and discards the rest.

Phase 1 implementation is a deterministic SHA-256 over the
concatenated, canonical-byte representation of each output. Sentinel
hashes encode error states so a candidate that crashes on the same
input the same way as another candidate is also pruned.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet
from cognithor.channels.program_synthesis.search.executor import (
    ExecutionResult,
    InProcessExecutor,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import ProgramNode
    from cognithor.channels.program_synthesis.search.executor import Executor


def _value_bytes(value: Any) -> bytes:
    """Canonical byte representation of one execution result.

    Order is part of the byte stream (e.g. ``ObjectSet`` order matters
    for fingerprinting — two identical sets in different orders are
    *not* equivalent because downstream primitives such as
    ``largest_object`` are order-sensitive on ties).
    """
    if isinstance(value, np.ndarray):
        # Type tag protects against shape collisions across dtypes.
        return f"ndarray:{value.dtype.str}:{value.shape}:".encode() + value.tobytes()
    if isinstance(value, np.bool_):
        return b"npbool:" + (b"1" if value else b"0")
    if isinstance(value, ObjectSet):
        parts = [b"objectset:" + str(len(value)).encode("ascii")]
        for o in value:
            parts.append(_value_bytes(o))
        return b"|".join(parts)
    if isinstance(value, Object):
        return f"object:{value.color}:{value.cells}".encode()
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, int):
        return f"int:{value}".encode("ascii")
    if isinstance(value, str):
        return f"str:{value}".encode()
    # Fall-through for any future type — repr() keeps the fingerprint
    # well-defined even if it's not byte-perfect across processes.
    return f"repr:{type(value).__name__}:{value!r}".encode()


def _result_bytes(r: ExecutionResult) -> bytes:
    if r.ok:
        return b"ok:" + _value_bytes(r.value)
    # Error fingerprints share the same bucket so candidates that fail
    # identically are pruned as duplicates.
    return f"err:{r.error or 'unknown'}".encode("ascii")


class ObservationalEquivalencePruner:
    """Bucket programs by their (input-tuple → output-tuple) fingerprint.

    The pruner is keyed by *output type* — programs with different
    output types are never compared (they live in disjoint sub-banks
    of the enumerator).
    """

    # Sentinel returned when more than half the inputs crash. Spec §9.1
    # treats such candidates as untrustworthy regardless of partial
    # matches — they're dropped before they pollute the bank.
    UNRELIABLE_SENTINEL: str | None = None

    def __init__(
        self,
        executor: Executor | None = None,
        unreliable_threshold: float = 0.5,
    ) -> None:
        if not 0.0 < unreliable_threshold <= 1.0:
            raise ValueError(
                f"unreliable_threshold must be in (0, 1]; got {unreliable_threshold!r}"
            )
        self._executor = executor if executor is not None else InProcessExecutor()
        self._threshold = unreliable_threshold
        # type_tag -> set of fingerprints already seen
        self._seen: dict[str, set[str]] = {}

    # -- Public API --------------------------------------------------

    def fingerprint(self, program: ProgramNode, demo_inputs: tuple[Any, ...]) -> str | None:
        """Return a SHA-256 hex digest, or ``None`` if the program is unreliable.

        ``None`` means "drop this candidate entirely" — caller should
        not register it.
        """
        if not demo_inputs:
            # Empty demo set degenerates to a constant program; we still
            # need a fingerprint, but it's unique by source rather than
            # by behaviour. Use the tree's stable hash as the fallback.
            from cognithor.channels.program_synthesis.search.candidate import (
                Program as _Program,
            )

            if isinstance(program, _Program):
                return program.stable_hash().split(":", 1)[1]
            return hashlib.sha256(repr(program).encode()).hexdigest()

        results: list[ExecutionResult] = []
        crash_count = 0
        for inp in demo_inputs:
            r = self._executor.execute(program, inp)
            if not r.ok:
                crash_count += 1
            results.append(r)

        if crash_count / len(demo_inputs) > self._threshold:
            return None

        digest = hashlib.sha256()
        for r in results:
            digest.update(_result_bytes(r))
            digest.update(b"||")
        return digest.hexdigest()

    def is_duplicate(self, fingerprint: str, type_tag: str) -> bool:
        """True iff a program with this fingerprint already exists for *type_tag*."""
        return fingerprint in self._seen.get(type_tag, set())

    def register(self, fingerprint: str, type_tag: str) -> None:
        """Mark *fingerprint* under *type_tag* as seen.

        Idempotent — calling twice with the same args is a no-op.
        """
        self._seen.setdefault(type_tag, set()).add(fingerprint)

    def reset(self) -> None:
        """Clear the seen map. Used between independent search runs."""
        self._seen.clear()

    # -- Convenience -------------------------------------------------

    def admit(
        self,
        program: ProgramNode,
        type_tag: str,
        demo_inputs: tuple[Any, ...],
    ) -> bool:
        """Return True iff the program is new and reliable; register it.

        Combines fingerprint → is_duplicate → register into one call.
        Returns False for unreliable programs (over-threshold crashes)
        and for already-seen fingerprints.
        """
        fp = self.fingerprint(program, demo_inputs)
        if fp is None:
            return False
        if self.is_duplicate(fp, type_tag):
            return False
        self.register(fp, type_tag)
        return True
