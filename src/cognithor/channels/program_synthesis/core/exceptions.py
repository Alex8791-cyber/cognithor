# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE exception hierarchy (spec §6.3).

All engine errors derive from :class:`PSEError`. Sandbox failures are
hard-fail and never auto-retried; their subclasses always escalate to a
``SandboxError`` branch so callers can short-circuit on a single
``except SandboxError`` clause.
"""

from __future__ import annotations


class PSEError(Exception):
    """Base class for all Program Synthesis Engine errors."""


class DSLError(PSEError):
    """Errors in the DSL layer (unknown primitive, type mismatch)."""


class TypeMismatchError(DSLError):
    """A primitive was applied to argument(s) of the wrong type."""


class UnknownPrimitiveError(DSLError):
    """The referenced primitive is not registered in the DSL registry."""


class SearchError(PSEError):
    """Errors raised by the enumerative search engine."""


class BudgetExceededError(SearchError):
    """The compute budget (depth / candidates / wall clock) was exhausted.

    Note: Search returns a :class:`SynthesisResult` with status
    ``BUDGET_EXCEEDED`` for the *normal* exhaustion path. This exception
    is reserved for hard-failure cases (e.g. budget invariants violated
    mid-search).
    """


class NoSolutionError(SearchError):
    """The search space contained no program matching the spec."""


class SandboxError(PSEError):
    """Errors in the sandbox layer. Hard-fail, never auto-retry."""


class SandboxViolationError(SandboxError):
    """Code attempted an action forbidden by sandbox policy.

    Always logged at high severity (Hashline Guard).
    """


class SandboxTimeoutError(SandboxError):
    """A candidate exceeded its wall-clock or per-candidate timeout."""


class SandboxOOMError(SandboxError):
    """A candidate exceeded its memory limit."""


class VerificationError(PSEError):
    """A verifier stage produced an unexpected internal failure."""
