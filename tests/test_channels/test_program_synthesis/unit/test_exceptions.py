# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE exception hierarchy tests."""

from __future__ import annotations

from cognithor.channels.program_synthesis.core.exceptions import (
    BudgetExceededError,
    DSLError,
    NoSolutionError,
    PSEError,
    SandboxError,
    SandboxOOMError,
    SandboxTimeoutError,
    SandboxViolationError,
    SearchError,
    TypeMismatchError,
    UnknownPrimitiveError,
    VerificationError,
)


class TestExceptionHierarchy:
    def test_pse_error_is_root(self) -> None:
        for cls in (
            DSLError,
            SearchError,
            SandboxError,
            VerificationError,
            TypeMismatchError,
            UnknownPrimitiveError,
            BudgetExceededError,
            NoSolutionError,
            SandboxViolationError,
            SandboxTimeoutError,
            SandboxOOMError,
        ):
            assert issubclass(cls, PSEError), cls.__name__

    def test_dsl_error_subclasses(self) -> None:
        assert issubclass(TypeMismatchError, DSLError)
        assert issubclass(UnknownPrimitiveError, DSLError)

    def test_search_error_subclasses(self) -> None:
        assert issubclass(BudgetExceededError, SearchError)
        assert issubclass(NoSolutionError, SearchError)

    def test_sandbox_error_subclasses(self) -> None:
        assert issubclass(SandboxViolationError, SandboxError)
        assert issubclass(SandboxTimeoutError, SandboxError)
        assert issubclass(SandboxOOMError, SandboxError)

    def test_single_except_catches_all_sandbox_failures(self) -> None:
        # A single `except SandboxError` clause must catch every sandbox
        # failure type — protects callers from depending on subclass order.
        for exc in (
            SandboxViolationError("v"),
            SandboxTimeoutError("t"),
            SandboxOOMError("o"),
        ):
            try:
                raise exc
            except SandboxError as caught:
                assert caught is exc
