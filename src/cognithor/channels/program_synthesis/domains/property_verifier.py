"""``PropertyVerifier`` — hypothesis-based property tests per domain.

Sprint-26 §26.1 deliverable. The verifier sits between
example-equality and the LLM-prior step: a synthesised program that
matches all examples but violates a domain property (e.g. SQL query
not idempotent, datetime fn not tz-roundtrip-stable) is rejected
*before* we spend tokens on a refinement pass.

Each domain ships a list of :class:`PropertyTest` callables. The
verifier runs them in declaration order and returns the first failure
(``ok=False``, attached metadata for the audit log) or a pass record.

Hypothesis is imported lazily so this module stays cheap to load when
no property suite has been registered yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PropertyResult:
    """Outcome of a single property-test run.

    Frozen so it can be embedded in audit-log entries without further
    copy. ``examples`` lists the (counter-)examples hypothesis surfaced
    when ``ok`` is ``False``.
    """

    name: str
    ok: bool
    examples: tuple[dict[str, Any], ...] = ()
    error_message: str = ""
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.ok


# A property test is a zero-arg callable that returns a
# :class:`PropertyResult`.  Domains register them as bound functions
# closing over the synthesised program.
PropertyTest = Callable[[], PropertyResult]


@dataclass
class PropertyVerifier:
    """Run a list of :class:`PropertyTest` callables in order.

    The verifier is *not* hypothesis-aware itself; individual tests
    use ``hypothesis`` (or any other generator) internally and return
    a :class:`PropertyResult`. Keeping the verifier protocol-free
    means tests for the verifier don't drag in hypothesis as a hard
    runtime dep.

    Attributes
    ----------
    tests:
        Ordered list of property tests. Verification stops at the
        first failure (``fail_fast`` defaults to True) or runs them
        all (when False) which is useful for full audit reports.
    fail_fast:
        Stop at the first failing property — synthesis-pipeline
        default. Tests can flip to False for diagnostic runs.
    """

    tests: list[PropertyTest] = field(default_factory=list)
    fail_fast: bool = True

    def add(self, test: PropertyTest) -> None:
        """Append a property test to the suite."""
        self.tests.append(test)

    def run(self) -> tuple[PropertyResult, ...]:
        """Execute every registered property test and return results.

        Always returns at least one record. With ``fail_fast=True`` the
        tuple has length 1..N depending on where the first failure
        occurs; with ``fail_fast=False`` length always equals
        ``len(tests)``.
        """
        if not self.tests:
            return ()

        results: list[PropertyResult] = []
        for test in self.tests:
            try:
                outcome = test()
            except Exception as exc:
                outcome = PropertyResult(
                    name=getattr(test, "__name__", "<anonymous>"),
                    ok=False,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            results.append(outcome)
            if self.fail_fast and not outcome.ok:
                break
        return tuple(results)

    def all_pass(self) -> bool:
        """Convenience: True iff every registered test passes.

        Always runs in non-fail-fast mode internally so the answer is
        meaningful even when callers configured ``fail_fast=True`` for
        the synthesis-pipeline path.
        """
        # Snapshot + flip flag so we don't mutate state.
        original = self.fail_fast
        self.fail_fast = False
        try:
            results = self.run()
        finally:
            self.fail_fast = original
        return all(r.ok for r in results)
